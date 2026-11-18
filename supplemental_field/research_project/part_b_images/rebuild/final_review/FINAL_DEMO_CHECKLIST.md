# Final Part B demo checklist

1. In VS Code, open the folder that contains `gui`, `models` and `part_b_images`. Open PowerShell there and run the command below with the existing environment.

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run gui/image_app.py
   ```

2. Open the local URL printed by Streamlit. Point out the five flower labels and the explanation under the model selector.
3. Select **Logistic Regression**, upload a local image, and show the preview, prediction, and five probability estimates.
4. Select **KNN Classifier** using the same image. Explain the 15-neighbor vote, Manhattan distance, and how the projected features are fitted on training data.
5. Select **K-Means** and show the cluster ID and mapped label. Explain that 15 clusters map back to five flower names using training labels. Use the archived K=5 trial for comparison.
6. To reproduce the audit's examples, use the five development image paths in [Demo fixtures](DEMO_FIXTURES.json). Review the expected predictions beforehand. Show a classification error as well as a success.
7. Demonstrate invalid-file handling with a separate disposable invalid JPEG. Upload a valid image afterward to show recovery.
8. Present [development results](DEVELOPMENT_TABLES.md), then [the paired 191-photo test](TEST_COHORTS.md). Explain why the final K-Means test column is unavailable. Use the [validation confusion matrices](CONFUSION_MATRIX_AUDIT.md) to discuss the leading errors.

The no-flags command uses all three selected bundles. Pass `-- --archived-rebuild` for the earlier rebuilt suite, `-- --archived-knn` for the compact field KNN, or `-- --archived-kmeans` for the compact K=5 K-Means. If the app reports a missing model, restore the exact bundle from [Final model selection](FINAL_MODEL_SELECTION.md).

Before a course submission, check the outstanding full-project and academic-integrity requirements in the [audit](FINAL_AUDIT.md).
