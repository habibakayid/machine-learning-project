"""Export selection evidence, final metrics, figures, and demo fixtures."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

from optimize import OUT, ROOT, FILES, CLASSES, BASELINES, Study, scores
from train_models import Experiment, combine, predict, predict_image

NAMES = {"Logistic Regression":"LR","KNN Classifier":"KNN","K-Means":"K-Means"}


def tables(predictions):
    metrics, classes, cells = [], [], []
    keys = ["scope","family","candidate","source","evaluation"]
    for key, rows in predictions.groupby(keys):
        info=dict(zip(keys,key))
        assert rows.path.is_unique
        pooled,report=scores(rows.label.to_numpy(),rows.predicted.to_numpy())
        values=pooled
        std=0.
        if info["evaluation"] in ("cv","group_excluded_refitted_3fold"):
            folds=pd.DataFrame([scores(f.label.to_numpy(),f.predicted.to_numpy())[0]
                                for _,f in rows.groupby("fold")])
            values=folds.mean().to_dict()
            std=folds.macro_f1.std(ddof=0)
        metrics.append({**info,**values,"cv_f1_std":std,"pooled_accuracy":pooled["accuracy"],
                        "correct":int(rows.label.eq(rows.predicted).sum()),"n":len(rows)})
        classes.extend({**info,"label":label,**report[label]} for label in CLASSES)
        matrix=confusion_matrix(rows.label,rows.predicted,labels=CLASSES)
        assert matrix.sum()==len(rows) and matrix.trace()==rows.label.eq(rows.predicted).sum()
        for i,actual in enumerate(CLASSES):
            for j,predicted in enumerate(CLASSES):
                cells.append({**info,"actual":actual,"predicted":predicted,
                              "count":int(matrix[i,j]),
                              "row_percent":100*matrix[i,j]/max(1,matrix[i].sum())})
    return pd.DataFrame(metrics),pd.DataFrame(classes),pd.DataFrame(cells)


def matrix_figures(counts):
    folder=OUT/"figures"
    folder.mkdir(exist_ok=True)
    for (scope,family,source),frame in counts.groupby(["scope","family","source"]):
        fig,axes=plt.subplots(3,2,figsize=(9,12),layout="constrained")
        for i,split in enumerate(("train","cv","validation")):
            rows=frame[frame.evaluation.eq(split)]
            for j,(value,fmt,unit) in enumerate((("count","d","counts"),("row_percent",".1f","% of true class"))):
                matrix=rows.pivot(index="actual",columns="predicted",values=value).loc[CLASSES,CLASSES]
                options={"vmin":0,"vmax":100} if j else {}
                ConfusionMatrixDisplay(matrix.to_numpy(),display_labels=[c.title() for c in CLASSES]).plot(
                    ax=axes[i,j],cmap="Greens",colorbar=False,values_format=fmt,
                    xticks_rotation=25,im_kw=options)
                axes[i,j].set_title(f"{split.title()}{', pooled' if split=='cv' else ''} · {unit}")
        fig.suptitle(f"{scope.title()} · {family} · {source.title()}")
        fig.savefig(folder/f"confusion_{scope}_{NAMES[family].lower()}_{source}.png",dpi=170)
        plt.close(fig)


def comparison_figures(metrics,classes):
    folder=OUT/"figures"
    for scope in ("primary","supplemental"):
        rows=metrics[metrics.scope.eq(scope)&metrics.evaluation.isin(["train","cv","validation"])]
        fig,axes=plt.subplots(1,rows.source.nunique(),squeeze=False,figsize=(6*rows.source.nunique(),4.5),sharey=True)
        for ax,(source,part) in zip(axes[0],rows.groupby("source"),strict=True):
            chart=part.pivot(index="family",columns="evaluation",values="accuracy")[["train","cv","validation"]]
            chart.rename(index=NAMES).mul(100).plot.bar(ax=ax,color=["#98b0a1","#356a5c","#bc7654"],rot=0)
            ax.set(title=source.title(),ylabel="Accuracy (%)",xlabel="",ylim=(0,105))
            ax.legend(fontsize=8)
        fig.suptitle(f"{scope.title()} training, CV and validation")
        fig.tight_layout()
        fig.savefig(folder/f"performance_{scope}.png",dpi=180)
        plt.close(fig)
    validation=classes[classes.evaluation.eq("validation")]
    combos=list(validation.groupby(["scope","source"]))
    fig,axes=plt.subplots(1,len(combos),figsize=(6*len(combos),4.5),squeeze=False,sharey=True)
    for ax,((scope,source),part) in zip(axes[0],combos,strict=True):
        chart=part.pivot(index="label",columns="family",values="f1-score").loc[list(CLASSES)]
        chart.rename(columns=NAMES).mul(100).plot.bar(ax=ax,rot=25,color=["#527f65","#b78151","#67809b"])
        ax.set(title=f"{scope.title()} · {source.title()}",ylabel="Validation F1 (%)",xlabel="",ylim=(0,100))
    fig.tight_layout()
    fig.savefig(folder/"per_class_f1.png",dpi=180)
    plt.close(fig)
    for name,splits in (("cross_validation",["cv"]),("train_vs_validation",["train","validation"])):
        data=metrics[metrics.evaluation.isin(splits)].copy()
        data["dataset"]=data.scope+" / "+data.source+" / "+data.evaluation
        chart=data.pivot(index="dataset",columns="family",values="macro_f1").rename(columns=NAMES)
        ax=chart.mul(100).plot.barh(figsize=(10, max(4,len(chart)*.65)),color=["#527f65","#b78151","#67809b"])
        ax.set(xlabel="Macro F1 (%)",ylabel="",xlim=(0,105),title=name.replace("_"," ").title())
        ax.figure.tight_layout()
        ax.figure.savefig(folder/f"{name}.png",dpi=180)
        plt.close(ax.figure)


def report_table(metrics,scope):
    rows=["| Model | Source | Train accuracy / F1 | CV accuracy / F1 | Validation accuracy / F1 |",
          "|---|---|---:|---:|---:|"]
    for (family,source),part in metrics[metrics.scope.eq(scope)].groupby(["family","source"]):
        cells=[]
        for split in ("train","cv","validation"):
            r=part[part.evaluation.eq(split)].iloc[0]
            cells.append(f"{r.accuracy:.2%} / {r.macro_f1:.2%}")
        rows.append(f"| {family} | {source} | {' | '.join(cells)} |")
    return "\n".join(rows)


def main():
    experiments,selected,frames,manifest,fixtures=[],[],[],[],[]
    candidate_frames=[]
    diagnostics=[]
    for scope in BASELINES:
        for family in FILES:
            folder=OUT/scope/family.replace(" ","_").lower()
            table=pd.read_csv(folder/"experiments.csv")
            chosen=pd.read_csv(folder/"selected.csv").iloc[0]
            experiments.append(table)
            selected.append(chosen.to_dict())
            saved=pd.read_csv(folder/"predictions.csv.gz")
            candidate_frames.append(saved)
            rows=saved[saved.candidate.eq(chosen.candidate)].copy()
            assert rows.groupby(["source","evaluation"]).path.nunique().min()>0
            study=Study(scope,family)
            bundle=joblib.load(folder/"selected.joblib")
            exp=Experiment(**bundle["experiment"])
            x,_=study.data(exp)
            for split,idx in (("train",study.fit),("validation",study.val)):
                part=rows[rows.evaluation.eq(split)].set_index("path")
                direct=predict(bundle,x[idx])
                assert np.array_equal(direct,part.loc[study.dev.path.iloc[idx],"predicted"])
            baseline=saved[saved.candidate.eq("baseline")]
            old=joblib.load(BASELINES[scope][family])
            base_exp=Experiment(**old["experiment"])
            base_x,_=study.data(base_exp)
            stored=baseline[baseline.evaluation.eq("validation")].set_index("path")
            assert np.array_equal(predict(old,base_x[study.val]),
                                  stored.loc[study.dev.path.iloc[study.val],"predicted"])
            if chosen.candidate!="baseline":
                assert chosen.eligible
            frames.append(rows)
            model_path=folder/"selected.joblib"
            transform_steps=[name for name,_ in bundle["pipeline"].steps]
            manifest.append({"scope":scope,"family":family,
                "path":model_path.relative_to(ROOT).as_posix(),
                "sha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "candidate":chosen.candidate,"promoted":chosen.candidate!="baseline",
                "fit_sources":["kaggle"] if scope=="primary" else ["kaggle","field"],
                "fit_counts":study.dev.iloc[study.fit].groupby("source").size().to_dict(),
                "class_order":list(CLASSES),
                "development_manifest_sha256":hashlib.sha256(
                    (OUT.parent/"generalization/development_split.csv").read_bytes()).hexdigest(),
                "steps":transform_steps,"experiment":bundle["experiment"],
                "input_features":bundle["pipeline"].n_features_in_,
                "estimator_features":bundle["pipeline"].steps[-1][1].n_features_in_,
                "estimator_parameters":bundle["pipeline"].steps[-1][1].get_params(deep=False),
                "mapping":None if bundle["mapping"] is None else bundle["mapping"].tolist()})
            validation=rows[rows.source.eq("kaggle")&rows.evaluation.eq("validation")]
            correct=validation[validation.label.eq(validation.predicted)].groupby("label").head(1)
            assert set(correct.label)==set(CLASSES)
            sample=study.dev[study.dev.path.isin(correct.path)]
            for item in sample.itertuples():
                with Image.open(ROOT/item.path) as img:
                    prediction,probs,cluster=predict_image(bundle,img)
                i=study.dev.index.get_loc(item.Index)
                assert prediction==predict(bundle,x[i:i+1])[0]
                if probs:
                    assert set(probs)==set(CLASSES) and np.isclose(sum(probs.values()),1)
                    assert np.allclose(list(probs.values()),bundle["pipeline"].predict_proba(x[i:i+1])[0])
                fixtures.append({"scope":scope,"family":family,"actual":item.label,"path":item.path,
                    "predicted":prediction,"probabilities":probs,
                    "purpose":"Correctly classified validation fixture for output rendering; not an accuracy sample",
                    "cluster_display":None if cluster is None else cluster+1})
            if family=="KNN Classifier":
                diagnostic=pd.read_csv(folder/"training_diagnostics.csv")
                diagnostics.append(diagnostic.assign(candidate=chosen.candidate,scope=scope,family=family))
            if family=="K-Means":
                estimator=bundle["pipeline"].steps[-1][1]
                idx=study.fit
                cluster=estimator.predict(bundle["pipeline"][:-1].transform(x[idx]))
                from kmeans_study import cluster_labels
                assert np.array_equal(cluster_labels(cluster,study.y[idx],estimator.n_clusters),bundle["mapping"])
                cluster_table=pd.read_csv(folder/"cluster_metrics.csv")
                cluster_table[cluster_table.candidate.eq(chosen.candidate)].assign(scope=scope).to_csv(
                    folder/"selected_cluster_metrics.csv",index=False)
            print(f"Verified {scope} / {family} and five direct image predictions.",flush=True)
    all_experiments=pd.concat(experiments,ignore_index=True)
    all_experiments.to_csv(OUT/"ALL_EXPERIMENTS.csv",index=False)
    _,candidate_classes,candidate_counts=tables(pd.concat(candidate_frames,ignore_index=True))
    candidate_classes.to_csv(OUT/"ALL_PER_CLASS_METRICS.csv",index=False)
    candidate_counts.to_csv(OUT/"ALL_CANDIDATE_CONFUSIONS.csv",index=False)
    pd.DataFrame(selected).to_csv(OUT/"SELECTION_RESULTS.csv",index=False)
    output=pd.concat([*frames,*diagnostics],ignore_index=True)
    metrics,classes,cells=tables(output)
    base=metrics[metrics.evaluation.eq("train")].set_index(["scope","family","source"]).accuracy
    val=metrics[metrics.evaluation.eq("validation")].set_index(["scope","family","source"]).accuracy
    metrics=metrics.join((base-val).rename("resubstitution_validation_gap"),on=["scope","family","source"])
    metrics.to_csv(OUT/"FINAL_MODEL_METRICS.csv",index=False)
    classes.to_csv(OUT/"FINAL_PER_CLASS_METRICS.csv",index=False)
    cells.to_csv(OUT/"FINAL_CONFUSION_MATRICES.csv",index=False)
    changes=[]
    for row in selected:
        keys=["source","actual","predicted"]
        pool=candidate_counts[candidate_counts.scope.eq(row["scope"]) &
                              candidate_counts.family.eq(row["family"]) &
                              candidate_counts.evaluation.eq("validation")]
        baseline=pool[pool.candidate.eq("baseline")][[*keys,"count"]].rename(columns={"count":"baseline_count"})
        final=pool[pool.candidate.eq(row["candidate"])][[*keys,"count"]].rename(columns={"count":"final_count"})
        change=baseline.merge(final,on=keys,validate="one_to_one")
        change["error_reduction"]=change.baseline_count-change.final_count
        changes.append(change.assign(scope=row["scope"],family=row["family"],candidate=row["candidate"]))
    pd.concat(changes,ignore_index=True).to_csv(OUT/"ERROR_CHANGES.csv",index=False)
    matrix_figures(cells[cells.evaluation.isin(["train","cv","validation"])])
    comparison_figures(metrics,classes)
    (OUT/"MODEL_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    (OUT/"DEMO_FIXTURES.json").write_text(json.dumps(fixtures,indent=2)+"\n",encoding="utf-8")
    snapshot=json.loads((OUT/"baseline_snapshot.json").read_text())
    for item in snapshot:
        assert hashlib.sha256((ROOT/item["path"]).read_bytes()).hexdigest()==item["sha256"]
    pd.read_csv(OUT/"DATA_SPLITS.csv").to_csv(OUT/"FINAL_DATA_SPLITS.csv",index=False)
    report=["# Final optimization results", "## Primary Kaggle only",report_table(metrics,"primary"),
            "## Supplemental mixed-source experiment",report_table(metrics,"supplemental")]
    (OUT/"METRICS_TABLES.md").write_text("\n\n".join(report)+"\n",encoding="utf-8")
    print(f"Verified {len(snapshot)} original model hashes. {len(all_experiments)} candidates.",flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=4):
        main()
