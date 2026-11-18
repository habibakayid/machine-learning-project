# Application verification

First import Kaggle data with the root tools/import_dataset.py, which restores the eight fixture images. Run all commands below from the full release root.

```powershell
.\.venv\Scripts\python.exe app\tests\verify_release.py
.\.venv\Scripts\python.exe -m streamlit run app\app.py --server.port 8510 --server.headless true
```

For optional browser testing, in another shell install Node.js and Playwright:

```powershell
npm install --prefix .qa playwright@1.65.1
$env:NODE_PATH = (Resolve-Path '.qa/node_modules').Path
& '.qa/node_modules/.bin/playwright.cmd' install chromium
$env:FLOWERS_QA_FOLDER = (Resolve-Path 'verification').Path
node app\tests\browser_check.cjs
```

FLOWERS_CHROMIUM may optionally point at an installed Chromium executable. The test reads resources relative to its own directory. It verifies all 15 model/class combinations, prediction/probability/cluster parity, invalid uploads, recovery and a mobile example. It writes app/evidence/BROWSER_VERIFICATION.json. These are functional fixtures selected from already-accessed validation data, not an unseen accuracy sample. Browser plugin was not available during packaging; ordinary Playwright was used. Photo screenshots generated in a local test are not distributed with this release.
