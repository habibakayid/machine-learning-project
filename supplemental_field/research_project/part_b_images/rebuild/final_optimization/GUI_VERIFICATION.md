# Verified Streamlit integration

Tested the running application at http://127.0.0.1:8508/ using Playwright and Chromium. The Browser plugin was unavailable. The original two-column layout and three algorithm choices remain, with an explicit training-source selector that defaults to Kaggle only.

The [browser result](GUI_TEST_RESULTS.json) records 30 successful uploads: three algorithms times five actual classes, for each of the primary and supplemental suites. The fixtures deliberately include one correctly classified validation example for each output label, to exercise all five rendering paths. They are functional checks, not a new performance sample. Dataset-level accuracy comes from the full development predictions.

For every fixture, the browser prediction matched direct inference from the saved bundle. The five supervised probability values matched to the UI's one-decimal percentage rounding. K-Means displayed the saved cluster mapping and no probability bars. The [fixtures](DEMO_FIXTURES.json) retain image paths, direct predictions, probabilities and cluster numbers.

The following checks passed:

- Default Kaggle-only training selection and both source selectors.
- All three algorithms and all five output labels in each suite.
- Corrupted JPEG error handling and a GIF disguised as a JPEG being rejected.
- Recovery with a valid image after rejected uploads.
- Desktop 1280 × 940 and mobile 390 × 844; no horizontal mobile overflow.
- No page errors, console errors or Streamlit exceptions in the recorded cases.

Screenshots: [primary LR](gui_screenshots/optimization-primary-Logistic_Regression.png), [primary KNN](gui_screenshots/optimization-primary-KNN_Classifier.png), [primary K-Means](gui_screenshots/optimization-primary-K-Means.png), [supplemental LR](gui_screenshots/optimization-supplemental-Logistic_Regression.png), [supplemental KNN](gui_screenshots/optimization-supplemental-KNN_Classifier.png), [supplemental K-Means](gui_screenshots/optimization-supplemental-K-Means.png), [mobile](gui_screenshots/optimization-mobile.png).

The model verifier also compared every selected bundle's training and validation predictions to the saved experiment predictions. It checked 30 direct image predictions, recomputed the K-Means mapping from fitting labels, and verified the 11 preserved original model hashes. Streamlit calls the shared `predict_image` preprocessing function and contains no model fitting.

## Repeat the checks

Run from the root containing gui and part_b_images. Keep the server running in a separate shell:

    .\.venv\Scripts\python.exe -m streamlit run gui/image_app.py --server.port 8508 --server.headless true

The browser check uses Node.js and Playwright. In this machine's existing environment:

    $env:NODE_PATH = 'C:/Users/Dell/AppData/Local/npm-cache/_npx/e41f203b7505f1fb/node_modules'
    $env:FLOWERS_CHROMIUM = (Get-ChildItem -LiteralPath "$env:LOCALAPPDATA/ms-playwright/chromium_headless_shell-1248" -Recurse -Filter 'chrome-headless-shell.exe' | Select-Object -First 1).FullName
    $env:FLOWERS_QA_FOLDER = (Resolve-Path 'part_b_images/rebuild/final_optimization/gui_screenshots').Path
    node part_b_images/rebuild/final_optimization/verify_gui.cjs
    .\.venv\Scripts\python.exe part_b_images/rebuild/final_optimization/verify_artifacts.py

On another machine, point NODE_PATH and FLOWERS_CHROMIUM at an installed Playwright package and Chromium executable. These are browser-test dependencies, not application dependencies. The artifact verification saves VERIFICATION_RESULTS.json after independently checking exported confusion counts, per-class metrics, model hashes, output coverage and local documentation links.
