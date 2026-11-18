# Final project audit

**Verified scope** is the Part B flower-classification demo and its saved evidence. The workspace's Part A and Part C folders contain no files, and this copy has no Git metadata.

## Verified development results

| Deployed model | Kaggle CV accuracy | Kaggle validation accuracy | Field CV accuracy | Field validation accuracy |
|---|---:|---:|---:|---:|
| Field-trained Nyström Logistic Regression | 76.67% | 75.66% | 61.11% | 54.46% |
| Nyström + LDA Manhattan KNN | 72.19% | 71.32% | 63.33% | 62.50% |
| Local visual words K=15 K-Means | 49.49% | 51.78% | 28.22% | 32.14% |

The audit loaded ten bundles and ran inference on the fixed development photos. The archived development predictions matched the reloaded models where a prior prediction table was available. CV uses saved out-of-fold predictions whose paths and groups match the fixed assignments. No training script or historical test evaluation was run. [Six development tables](DEVELOPMENT_TABLES.md) · [Exact model choices](FINAL_MODEL_SELECTION.md) · [Model hashes and feature dimensions](MODEL_MANIFEST.json)

## Data integrity

| Partition | Daisy | Dandelion | Rose | Sunflower | Tulip | Total |
|---|---:|---:|---:|---:|---:|---:|
| Kaggle raw | 764 | 1,052 | 784 | 733 | 984 | 4,317 |
| Rebuilt Kaggle train | 536 | 737 | 546 | 512 | 686 | 3,017 |
| Rebuilt Kaggle validation | 114 | 157 | 117 | 110 | 147 | 645 |
| Rebuilt Kaggle internal test | 114 | 158 | 117 | 109 | 147 | 645 |
| Field train | 96 | 96 | 96 | 66 | 96 | 450 |
| Field validation | 24 | 24 | 24 | 16 | 24 | 112 |

All 5,610 audited images decoded. The Kaggle cleaning retained 4,307 images in 4,284 groups after removing two redundant copies and eight records with conflicting exact labels. The grouped stratified split preserves all five classes in each partition and held-out fold. The field train/validation split and CV also keep each observer together. A combined audit checked file hashes, decoded-pixel hashes, available Flickr photo IDs, and perceptual similarity on original and mirrored thumbnails. It found zero detected related groups crossing rebuilt partitions, CV folds, or the published field cohorts. The observer manifests also had zero cross-cohort overlaps. These checks cover recorded identifiers and detectable image similarities. They cannot rule out every crop, edit, or attribution alias.

The legacy Kaggle split had eight detected related groups crossing its boundaries. Its 647-image test and the rebuilt 645-image test share 99 paths. The other 546 rebuilt test images were in legacy training or validation. Treat both as previously accessed internal benchmarks. [All class and fold counts](DATA_SPLITS.csv) · [Pairwise overlaps](DATA_OVERLAP.csv) · [Audit summary](DATA_AUDIT.json)

## Generalization and remaining limitations

- **Logistic Regression:** Kaggle and field gaps for the deployed model are 22.95 and 42.87 percentage points. C=3 reduced the Kaggle gap to 14.12 points but increased the field gap to 44.65 points. Compact reduced both gaps and scored lower on the joint CV criterion.
- **KNN:** The 100% training accuracy includes the query as a zero-distance neighbor. Grouped CV and validation are the generalization checks. The study's same-group-excluded training diagnostic was 77.69% on Kaggle and 89.56% on field photos, with the transformations already fitted on training data.
- **K-Means:** Low field training accuracy of 27.33% and field validation accuracy of 32.14% indicate weak flower-label alignment. On field validation, silhouette is 0.001, ARI is 0.049, and NMI is 0.204. Color, background and framing can group images by appearance. A 15-to-5 majority mapping improves five-class predictions even when raw clusters overlap.
- **Test comparison:** The 191-photo cohort compares deployed Logistic Regression at 119/191 and deployed KNN at 121/191. The deployed K-Means has no independent test result. Earlier test results for other variants and other cohorts remain separate. [Cohort tables](TEST_COHORTS.md)

## Demo and course requirements

The no-flags Streamlit command loads the selected bundles. Chromium exercised all 15 model/class upload combinations against the saved inference fixtures, including the preview, predicted label, five probability bars or cluster ID, invalid JPEG recovery, and a narrow viewport. These are prediction-agreement checks, and some examples are genuinely misclassified. The inference path opens images, extracts features, and calls already-fitted pipelines. [Demo checklist](FINAL_DEMO_CHECKLIST.md)

The original *Machine Learning Project Guidelines* PDF was read from the Downloads folder. Pages 2–5 require the three parts, a GUI for numerical/image/text input, a team submission and cover sheet, and test results with per-class metrics. Part B satisfies the five-class and majority-mapping structure, but there is no common unseen test for all deployed bundles. The course also describes one dataset per part, so the supplemental field-training setup needs confirmation from the TA. Page 6 expressly prohibits using AI to write submitted code. This workspace contains AI-written project changes, so the demo audit does not establish that the code qualifies for submission under that rule. Before submission, clarify the permitted scope with the TA and prepare student-authored work as required.

No presentation PDF was generated. [Structure and preserved history](FINAL_PROJECT_STRUCTURE.md)
