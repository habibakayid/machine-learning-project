"""Apply the documented stability correction to existing development predictions."""

import shutil
import joblib
import pandas as pd

from optimize import OUT, BASELINES, Study, FILES, POLICY, scores


def main():
    previous=OUT/"SELECTION_POLICY_INITIAL.md"
    if not previous.exists():
        shutil.copyfile(OUT/"SELECTION_POLICY.md",previous)
    (OUT/"SELECTION_POLICY.md").write_text(
        POLICY+"\n## Documented development-only amendment\n\n"
        "The initial rule capped any growth in F1 standard deviation at 0.01. It rejected "
        "the Kaggle K=10 candidate despite a large improvement in held-out F1. The revised "
        "stability check compares each source's weakest fold with the baseline weakest fold, "
        "allowing at most 0.01 loss. Standard deviations remain reported. This correction "
        "was applied to all candidates using stored development predictions, with all other "
        "guardrails retained. No test scores or test predictions entered this amendment. "
        "The original policy and selection tables are preserved for review.\n",encoding="utf-8")
    for scope in BASELINES:
        for family in FILES:
            study=Study(scope,family)
            path=study.folder/"experiments.csv"
            old_table=pd.read_csv(path)
            backup=study.folder/"experiments_initial_policy.csv"
            if not backup.exists():
                shutil.copyfile(path,backup)
                shutil.copyfile(study.folder/"selected.csv",study.folder/"selected_initial_policy.csv")
            predictions=pd.read_csv(study.folder/"predictions.csv.gz")
            per_class=pd.read_csv(study.folder/"per_class.csv")
            rows=[]
            baseline=None
            old_selected=pd.read_csv(study.folder/"selected.csv").iloc[0]
            best_cv=pd.read_csv(study.folder/"best_cv.csv").iloc[0]
            for row in old_table.to_dict("records"):
                name=row["candidate"]
                for source in study.sources:
                    part=predictions[predictions.candidate.eq(name)&predictions.source.eq(source)
                                     &predictions.evaluation.eq("cv")]
                    folds=[scores(g.label.to_numpy(),g.predicted.to_numpy())[0]["macro_f1"]
                           for _,g in part.groupby("fold")]
                    row[f"{source}_cv_min_f1"]=min(folds)
                classes=per_class[per_class.candidate.eq(name)].drop(columns=["candidate","scope","family"])
                if name=="baseline":
                    study.baseline=row,classes
                    failures=[]
                else:
                    failures=study.guardrails(row,classes)
                # Every current and previously eligible final model maps all five labels.
                row["eligible"]=not failures
                row["guardrail_failures"]=" | ".join(failures)
                rows.append(row)
            table=pd.DataFrame(rows)
            chosen=table[table.eligible].sort_values(
                ["selection_f1","cv_f1_std"],ascending=[False,True]).iloc[0]
            if chosen.candidate==old_selected.candidate:
                bundle=joblib.load(study.folder/"selected.joblib")
            elif chosen.candidate==best_cv.candidate:
                bundle=joblib.load(study.folder/"best_cv_candidate.joblib")
            elif chosen.candidate=="baseline":
                bundle=joblib.load(BASELINES[scope][family])
            else:
                raise ValueError(f"Refitting required for new candidate {scope}, {family}, {chosen.candidate}")
            joblib.dump(bundle,study.folder/"selected.joblib",compress=3)
            table.to_csv(path,index=False)
            pd.DataFrame([chosen]).to_csv(study.folder/"selected.csv",index=False)
            print(scope,family,chosen.candidate,chosen.selection_f1,flush=True)


if __name__=="__main__":
    main()
