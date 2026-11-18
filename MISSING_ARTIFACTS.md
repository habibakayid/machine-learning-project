# Missing artifacts and deliberate exclusions

## Required dataset download

- All 4,317 original Kaggle images are available in the original development project but deliberately absent from this distributable because the Kaggle data card's redistribution license is Unknown. This is a distribution exclusion, not a claim that the source data was lost. A tested importer checks the archived hashes and restores data from a recipient's own Kaggle ZIP or local copy.
- Original resized 64/96-pixel image arrays, photo contact sheets and Kaggle GUI screenshots are likewise omitted. Their original files were preserved in the development project. Numeric feature arrays, manifests, models, metrics and quantitative result plots remain included.
- Download access/terms can change. The package cannot supply credentials or guarantee that a future changed upstream dataset will match the frozen 4,317-image inventory. Mismatches are reported and never silently accepted.

## Experiment evidence that does not exist

- No genuinely new untouched common external test for the three finalized primary models. Historical internal and field cohorts were previously accessed, and are identified as such.
- No newly computed learning curves for the final C=3 LR / 31-neighbor KNN / 10-cluster primary K-Means. Existing historical curves are included and labeled historical; generating new ones would violate the no-training packaging instruction.
- No exact end-to-end leave-one-image-out or leave-one-group-out KNN refitting. Archived frozen-representation neighbor exclusions are descriptive; independently refitted three-fold group CV is available.
- No claim that complete retraining was executed or that its outputs are bitwise identical. Packaging checked the selected recipe cloning and frozen inputs only, then verified saved inference/evaluation.
- No separate immutable environment or training-console log for every early 64/96-pixel development iteration. The current pinned environment, actual surviving scripts, numerical feature arrays and saved results are retained. Some earliest pixel-array workflows may require examining those historical assumptions.
- No Git history was present. Part A and Part C were empty and are outside this Part B handover.
- No documented instructor approval allowing mixed-source primary training, and no resolution of the original course restriction on AI-written submitted code. These affect academic submission eligibility, not saved-model execution.
- No oral-examination or final presentation PDF has been generated.

## Distribution inventory

`verification/EXCLUDED_FROM_ARCHIVE.json` lists every excluded source file and its reason, including reproducible virtual environments, bytecode, feature caches and pixel arrays. `verification/COPY_MANIFEST.json` maps included files to their original locations and SHA256 values. Every surviving original joblib model and all numeric experiment results are preserved in the historical snapshot. The only intentional modification to copied application code is a missing-dataset guard for its example button; it does not change preprocessing or model inference.
