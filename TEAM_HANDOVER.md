# Team handover

Release: FULL_PART_B_TEAM_RELEASE, 2026-11-18. This package separates the primary Kaggle-only deliverable from historical mixed-source research. No original project file was modified. Do not commit credentials, overwrite frozen model bundles, or treat historical test accuracy as a new independent result.

## First run

1. Install Python 3.11 x64 and run setup_windows.bat.
2. Obtain the Kaggle collection and run tools/import_dataset.py as documented in README.md. It verifies all hashes and restores GUI examples.
3. Run tools/verify_package.py --check-dataset and app/tests/verify_release.py.
4. Run run_gui.bat and try every algorithm on examples and your own photos.
5. Read MODEL_RESULTS.md and MISSING_ARTIFACTS.md before presenting results.

## Responsibilities and reports

Use app/BUG_REPORT_TEMPLATE.md. Report the exact release, algorithm, image, expected identification, actual output and reproducible steps. Keep new team photos in a separate review directory. Separate application failures from ordinary classification errors. Do not add new photos to fitting or tune against test cohorts as part of team testing.

Only designated future experiments should execute tools/train_primary.py or the original research search, in new run directories. This release does not automatically promote a newly trained bundle. Archive the complete run settings and comparisons before any later promotion. Re-run both direct inference and browser verification after any separately authorized release change.

## Supporting evidence

- verification/COPY_MANIFEST.json: original-to-package provenance and file hashes.
- verification/PACKAGE_CHECK.json: executed package structure, model and split checks.
- verification/PORTABILITY_CHECK.json: relocation, dataset import, GUI and saved evaluation checks.
- app/evidence/RELEASE_VERIFICATION.json and BROWSER_VERIFICATION.json: all 15 functional cases and invalid-upload handling.
- primary/results/: primary development metrics and per-class/confusion evidence.
- supplemental_field/research_project/part_b_images/rebuild/final_optimization/: all 124 experiments, rejected candidates, policy history and original final audit.

Verification on the preparation machine does not guarantee every team member's hardware, browser or network. Dependency installation needs internet; installed inference runs locally. No new independent accuracy claim is made. Academic submission still needs the students to resolve the course's AI-assisted-code policy with the teaching assistant.
