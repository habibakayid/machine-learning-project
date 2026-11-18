# Frozen primary inference application

Use the root README.md, setup_windows.bat and run_gui.bat. This directory contains the exact three selected Kaggle-only bundles and their original preprocessing/inference modules. No field model is selectable. The only application change from TEAM_TESTING_RELEASE disables the example button until dataset import restores its image; upload inference is unchanged.

After importing the dataset, run the root environment's Python against `app/tests/verify_release.py`. It checks 15 fixtures against original direct-inference expectations. Images are intentionally not distributed because the Kaggle collection's license is Unknown. See the root MISSING_ARTIFACTS.md. MODEL_MANIFEST.json contains exact model hashes/configurations; the root MODEL_RESULTS.md contains their development metrics.
