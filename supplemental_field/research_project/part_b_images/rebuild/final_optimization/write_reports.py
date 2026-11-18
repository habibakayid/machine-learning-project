"""Write source reports from the verified optimization results."""

import json
from pathlib import Path
import pandas as pd

OUT=Path(__file__).resolve().parent


def main():
    metrics=pd.read_csv(OUT/"FINAL_MODEL_METRICS.csv")
    selected=pd.read_csv(OUT/"SELECTION_RESULTS.csv")
    experiments=pd.read_csv(OUT/"ALL_EXPERIMENTS.csv")
    manifest=json.loads((OUT/"MODEL_MANIFEST.json").read_text())
    audit=json.loads((OUT/"DATA_AUDIT.json").read_text())
    classes=pd.read_csv(OUT/"FINAL_PER_CLASS_METRICS.csv")
    counts=pd.read_csv(OUT/"FINAL_CONFUSION_MATRICES.csv")
    leaders=experiments.loc[experiments.groupby(["scope","family","stage"]).selection_f1.idxmax()]
    leaders.to_csv(OUT/"EXPERIMENT_STAGE_SUMMARY.csv",index=False)

    def candidate(scope,family,name):
        return experiments[experiments.scope.eq(scope)&experiments.family.eq(family)
                           &experiments.candidate.eq(name)].iloc[0]

    primary_lr=candidate("primary","Logistic Regression","C=3")
    lr_no_hog=candidate("primary","Logistic Regression","No HOG")
    lr_compact=candidate("primary","Logistic Regression","Compact foreground")
    primary_knn=selected[selected.scope.eq("primary")&selected.family.eq("KNN Classifier")].iloc[0]
    knn_pca=leaders[leaders.scope.eq("primary")&leaders.family.eq("KNN Classifier")
                    &leaders.stage.eq("PCA128")].iloc[0]
    balanced=candidate("supplemental","Logistic Regression","balanced,C=5")
    equal=candidate("supplemental","Logistic Regression","C=5")

    def put(name, paragraphs):
        (OUT/name).write_text("\n\n".join(paragraphs)+"\n",encoding="utf-8")

    selection=[
        "# Final model selection",
        "The initial policy was recorded before fitting. The [selection policy](SELECTION_POLICY.md) "
        "documents one development-only correction to the stability rule, with the initial policy and "
        "decisions preserved. The primary suite fits "
        "Kaggle only and is the application default. Supplemental training uses iNaturalist as a separately "
        "identified generalization experiment. Numerical target ranges do not enter selection.",
        "| Suite | Algorithm | Selected candidate | CV selection macro F1 | Promoted over baseline |\n"
        "|---|---|---|---:|---|",
    ]
    for row in selected.itertuples():
        selection[-1]+=f"\n| {row.scope} | {row.family} | {row.candidate} | {row.selection_f1:.2%} "
        selection[-1]+=f"| {'Yes' if row.candidate!='baseline' else 'No'} |"
    selection.append(
        "Primary LR retains degree-2 polynomial Nyström, gamma=0.18 and 1,300 landmarks, "
        "with C reduced to 3. Supplemental LR uses C=5 with equal per-image sample weights. "
        "Both retain all five feature groups. Primary KNN uses a 500-landmark degree-2 map "
        "and a four-dimensional supervised LDA projection fitted inside every CV fold, "
        "31 Euclidean neighbors and uniform voting. Supplemental KNN keeps the same "
        "field-balanced fitting representation as its baseline, with 31 Euclidean neighbors "
        "and distance voting. No PCA model beat the selected LDA configurations under the "
        "promotion checks. Local visual words with PCA20 use 10 primary clusters and "
        "15 supplemental clusters; each cluster is mapped from fitting labels only."
    )
    for model in manifest:
        exp=model["experiment"]
        path=Path(model["path"]).relative_to("part_b_images/rebuild/final_optimization").as_posix()
        selection.append(
            f"## {model['scope'].title()} / {model['family']}\n\n"
            f"Bundle [selected.joblib]({path}). Fitting counts {model['fit_counts']}. "
            f"Preparation {exp['mode']}, groups {exp['groups']}, weights {exp['weights']}, "
            f"scaling {exp['scale']}. Steps {model['steps']}. Input width "
            f"{model['input_features']}, estimator width {model['estimator_features']}. "
            f"Estimator settings {model['estimator_parameters']}."
        )
    selection.append(
        "Earlier bundles remain in their original folders. Unconstrained CV leaders are preserved "
        "as best_cv_candidate.joblib inside each study folder. The selected.joblib implements all promotion "
        "checks. The [experiment CSV](ALL_EXPERIMENTS.csv) records rejection reasons. "
        "The [model manifest](MODEL_MANIFEST.json) pins hashes and fitting sources."
    )
    put("FINAL_MODEL_SELECTION.md",selection)

    over=[
        "# Overfitting analysis",
        "| Suite | Model | Source | Train accuracy | Validation accuracy | Gap |\n"
        "|---|---|---|---:|---:|---:|",
    ]
    for (scope,family,source),part in metrics.groupby(["scope","family","source"]):
        train=part[part.evaluation.eq("train")].iloc[0]
        val=part[part.evaluation.eq("validation")].iloc[0]
        over[-1]+=f"\n| {scope} | {family} | {source} | {train.accuracy:.2%} | "
        over[-1]+=f"{val.accuracy:.2%} | {100*(train.accuracy-val.accuracy):.2f} pp |"
    over.extend([
        "## KNN diagnostics",
        "Resubstitution includes each query as a possible zero-distance neighbor. The leave-one-out and "
        "same-group-excluded neighbor diagnostics freeze the learned representation. They omit the stated "
        "neighbors, but the scaler, vocabulary, Nyström map and LDA have seen those training images. "
        "They are descriptive fitting diagnostics.",
        "The group_excluded_refitted_3fold row excludes the held-out groups from every transformation "
        "and the neighbor pool. It is computed from the fold-refitted training predictions and equals "
        "grouped CV. It fits about two thirds of groups per fold. Exact end-to-end leave-one-image-out "
        "or leave-one-group-out refitting was not executed. Frozen-representation LOO is explicitly "
        "distinguished from those independent-refit estimates.",
        "| Suite | Source | Evaluation | Accuracy | Macro F1 |\n|---|---|---|---:|---:|",
    ])
    for row in metrics[metrics.family.eq("KNN Classifier")].itertuples():
        over[-1]+=f"\n| {row.scope} | {row.source} | {row.evaluation} | {row.accuracy:.2%} | {row.macro_f1:.2%} |"
    over.append(
        "LR gaps are assessed together with CV, validation and class scores. Lower training accuracy alone "
        "does not select a model. K-Means uses labels only to name clusters after fitting. Low training and "
        "validation scores together indicate weak alignment of image clusters and flower labels. "
        "Silhouette uses Euclidean coordinates and raw clusters in the fitted space. ARI and NMI use raw "
        "cluster IDs and the correct partition labels. Large partitions use a seeded 800-row silhouette sample."
    )
    put("OVERFITTING_ANALYSIS.md",over)

    put("DATA_INTEGRITY_AUDIT.md",[
        "# Data integrity audit",
        f"The recheck decoded {audit['scanned']} images with {len(audit['unreadable'])} failures. "
        f"Kaggle retains {audit['retained_kaggle']} of {audit['raw_count']} photos in "
        f"{audit['kaggle_groups']} groups. All five classes appear in every required split and fold.",
        f"Detected related groups crossing rebuilt splits: {audit['rebuilt_related_groups_crossing_splits']}. "
        f"Groups crossing CV folds: {audit['related_groups_crossing_cv_folds']}. "
        f"Matches across current partitions and external cohorts: {len(audit['active_cross_partition_matches'])}. "
        "Observer separation was checked. File hashes, decoded-pixel hashes, available Flickr IDs, "
        "DCT hashes and thumbnails detect many copies. Crops, edits or observer aliases can evade the checks.",
        f"The legacy split had {audit['legacy_related_groups_crossing_splits']} related groups crossing "
        "partitions. The rebuilt internal test reuses legacy development photos. All published field "
        "cohorts have been accessed. A common unseen test for the final bundles is unavailable. "
        "The runner loads only fixed development manifests and makes no historical test predictions.",
        "Primary training uses 3,017 Kaggle photos with 645 validation photos. Supplemental training adds "
        "450 field photos and a separate 112 field validation photos. Field features and labels never enter "
        "the primary fit. All fold-fitted transformations use fitting rows only. LDA uses fitting labels "
        "intentionally, while K-Means centroids and vocabularies receive no flower labels.",
        "The course PDF specifies one dataset per part. No documented instructor permission for combined "
        "primary training was found. Page 6 prohibits AI-written submitted code. This AI-assisted workspace "
        "needs a permitted, student-authored submission route confirmed by the TA.",
        "[Class/fold counts](FINAL_DATA_SPLITS.csv) · [Overlap checks](DATA_OVERLAP.csv) · "
        "[Audit summary](DATA_AUDIT.json) · [Earlier test cohorts](../final_review/TEST_COHORTS.md)",
    ])

    report=[
        "# Final optimization and stabilization",
        f"Executed {len(experiments)} controlled candidate configurations across two separately fitted suites. "
        "Earlier models and results were preserved, and their hashes verified after fitting. Existing code, "
        "manifests, guidelines, studies and predictions were reviewed first.",
        (OUT/"METRICS_TABLES.md").read_text().replace("# Final optimization results\n\n",""),
        "CV metrics are three-fold means. CV class metrics and confusion matrices pool one held-out "
        "prediction per original training image. Correct counts use that pooled denominator. "
        "The CSV includes pooled accuracy alongside mean fold accuracy.",
        "## Model changes and evidence",
        "LR stages keep gamma at 0.18 while varying C and landmarks. Matched ablations check HOG, "
        "LBP, edges, local descriptors and foreground cues. Supplemental weighting uses the same geometry. "
        "KNN varies neighborhoods in its main sweep, then compares fixed-neighborhood feature and PCA "
        "alternatives. K-Means reuses local features for the limited K=5/10/15 comparison. "
        "[Exact selections](FINAL_MODEL_SELECTION.md) · [Gaps and diagnostics](OVERFITTING_ANALYSIS.md)",
        f"For primary LR, removing HOG raised CV macro F1 to {lr_no_hog.selection_f1:.2%}, "
        f"but validation F1 fell to {lr_no_hog.kaggle_validation_macro_f1:.2%} and failed the "
        f"promotion check. The selected full-feature model gives {primary_lr.selection_f1:.2%} CV "
        f"and {primary_lr.kaggle_validation_macro_f1:.2%} validation F1. Compact foreground "
        f"features give {lr_compact.selection_f1:.2%} CV F1. Smaller landmark and simpler feature "
        "models remain in the comparison CSV; they did not beat the full selected model under the checks.",
        f"For primary KNN, at the same chosen neighborhood, PCA128 gives "
        f"{knn_pca.selection_f1:.2%} CV F1 versus {primary_knn.selection_f1:.2%} for the selected "
        "Nyström/LDA representation. Removing HOG or local descriptors also lowered CV F1. "
        "The selected LDA representation uses labels strictly from the fitting fold.",
        f"For supplemental LR, source balancing at C=5 gives {balanced.selection_f1:.2%} joint CV F1 "
        f"versus {equal.selection_f1:.2%} with equal per-image weights. Its field validation accuracy "
        f"is {balanced.field_validation_accuracy:.2%}, but its field gap is {100*balanced.field_gap:.2f} pp "
        f"and its Kaggle validation F1 is lower. Equal per-image weights remain selected. "
        "[Stage leaders](EXPERIMENT_STAGE_SUMMARY.csv) retain the best CV result in every stage, "
        "including rejected candidates and their rejection reasons.",
        "## Weak classes and largest validation errors",
        "| Suite | Model | Source | Lowest validation F1 | F1 | Largest error | Count |\n"
        "|---|---|---|---|---:|---|---:|",
    ]
    for (scope,family,source),part in classes[classes.evaluation.eq("validation")].groupby(
        ["scope","family","source"]):
        weak=part.loc[part["f1-score"].idxmin()]
        errors=counts[counts.scope.eq(scope)&counts.family.eq(family)&counts.source.eq(source)
                      &counts.evaluation.eq("validation")&counts.actual.ne(counts.predicted)]
        err=errors.loc[errors["count"].idxmax()]
        report[-1]+=f"\n| {scope} | {family} | {source} | {weak.label} | {weak['f1-score']:.2%} "
        report[-1]+=f"| {err.actual} → {err.predicted} | {err['count']} |"
    report.extend([
        "## K-Means clustering diagnostics",
        "Silhouette, ARI and NMI describe raw cluster assignments, before mapping to flower names. "
        "CV rows below average the three separately fitted folds. Partitions above "
        "800 rows use a seeded 800-row silhouette sample; other partitions use every row.",
        "| Suite | Source | Evaluation | Clusters | Silhouette | ARI | NMI |\n"
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for scope in ("primary","supplemental"):
        cluster=pd.read_csv(OUT/scope/"k-means"/"selected_cluster_metrics.csv")
        cluster["evaluation"]=cluster.evaluation.str.replace(r"cv_\d+","cv",regex=True)
        for (source,evaluation),part in cluster.groupby(["source","evaluation"]):
            row=part[["clusters","silhouette","ari","nmi"]].mean()
            report[-1]+=f"\n| {scope} | {source} | {evaluation} | {row.clusters:.0f} "
            report[-1]+=f"| {row.silhouette:.4f} | {row.ari:.4f} | {row.nmi:.4f} |"
    report.extend([
        "Image similarity does not necessarily follow flower taxonomy. Background, lighting "
        "and color can dominate Euclidean clusters. Additional clusters allow several appearances "
        "per flower class, but weak silhouette and low field ARI show that this remains a limited "
        "classifier. Labels name clusters only after centroid fitting; they do not guide K-Means.",
        "## Error changes against each suite's baseline",
        "The [error-change table](ERROR_CHANGES.csv) records every validation count before and after selection. "
        "Positive error_reduction means fewer mistakes for that true/predicted pair. Diagonal changes "
        "describe correct predictions and should be interpreted separately.",
        "| Suite | Model | Source | Largest corrected confusion | Fewer errors |\n|---|---|---|---|---:|",
    ])
    errors=pd.read_csv(OUT/"ERROR_CHANGES.csv")
    for (scope,family,source),part in errors[errors.actual.ne(errors.predicted)].groupby(["scope","family","source"]):
        gain=part.loc[part.error_reduction.idxmax()]
        report[-1]+=f"\n| {scope} | {family} | {source} | {gain.actual} → {gain.predicted} | {gain.error_reduction} |"
    report.extend([
        "Similar colors, petal structure and framing are plausible causes of confusion. Earlier contact "
        "sheets showed wild roses among foliage and slender yellow field tulips. These are qualitative "
        "findings, without segmentation or species ground truth for Kaggle. "
        "[Earlier framing audit](../generalization/framing_summary.csv)",
        "## Run and verification",
        "From the folder containing gui and part_b_images, run the command below. The Training data "
        "selector defaults to Kaggle only. The supplemental selector labels mixed-source training.",
        "    .\\.venv\\Scripts\\python.exe -m streamlit run gui/image_app.py",
        "Saved bundles were checked against stored predictions and five direct image predictions each. "
        "[Browser checks](GUI_VERIFICATION.md) record automated image-upload verification.",
        "## Submission status and limitations",
        "The primary fitting protocol now uses one dataset. The course restriction on AI-written code "
        "needs resolution with the TA. This workspace has empty Part A and Part C folders and no Git "
        "history. Part B technical readiness does not establish whole-course submission eligibility.",
        "CV and validation images have informed repeated development decisions. They are development "
        "estimates. No new common external test was fabricated, and historical tests remain cohort-specific. "
        "Frozen-representation LOO and group-excluded neighbor scores are descriptive only. "
        "Exact end-to-end leave-one-out refitting was not run. No presentation PDF was created.",
        "## Source artifacts",
        "[All experiments](ALL_EXPERIMENTS.csv) · [Final metrics](FINAL_MODEL_METRICS.csv) · "
        "[All candidate per-class metrics](ALL_PER_CLASS_METRICS.csv) · "
        "[Final per-class metrics](FINAL_PER_CLASS_METRICS.csv) · "
        "[All candidate confusion counts](ALL_CANDIDATE_CONFUSIONS.csv) · "
        "[Final confusion counts](FINAL_CONFUSION_MATRICES.csv) · "
        "[Model manifest](MODEL_MANIFEST.json) · [Figures](figures/)",
    ])
    put("FINAL_REPORT.md",report)
    print("Wrote final reports.",flush=True)


if __name__=="__main__":
    main()
