# Part B · Five-class flower classification

## 1. Dataset and audit

| Flower | Raw | Train | Validation | Internal test |
|---|---:|---:|---:|---:|
| Daisy | 764 | 536 | 114 | 114 |
| Dandelion | 1,052 | 737 | 157 | 158 |
| Rose | 784 | 546 | 117 | 117 |
| Sunflower | 733 | 512 | 110 | 109 |
| Tulip | 984 | 686 | 147 | 147 |
| **Total** | **4,317** | **3,017** | **645** | **645** |

All source images decoded. The pipeline excluded two same-label exact copies and eight exact copies across rose and tulip bearing conflicting labels. It retained 4,307 images in 4,284 related-photo groups. The audit records SHA-256 file and decoded-pixel hashes, likely shared Flickr photo IDs, and perceptual matches using a DCT hash and thumbnail-color distance on original and mirrored images. Detected near-copies stay together. The heuristic can miss crops or other substantial edits. The typical source resolution was 320 × 240 pixels. [Dataset source](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) · [Image inventory](results/raw_inventory.csv) · [Related photos](results/related_photos.csv) · [Sizes and distribution](figures/dataset.png)

Eight of the detected related-photo groups spanned the historical split. Zero span the rebuilt split. The reproducible split uses 20 shuffled, stratified group folds with seed 42, assigning 14 folds to training, three to validation, and three to internal test. Every class is present in every partition.

## 2. Legacy reproduction and preprocessing

The saved historical feature arrays reproduced **59.60% validation accuracy** and **58.90% macro F1** for Logistic Regression with HOG + RGB + spatial HSV. The archived models reproduced **57.34%**, **46.68%**, and **27.36%** historical test accuracy for Logistic Regression, KNN, and K-Means, respectively. The reported **60.84%** augmented run had no matching training script in this copy. A rebuild of the same unaugmented feature recipe scored **58.07% CV** and **56.90% validation** after grouping related images. [Legacy reproduction](results/legacy_reproduction.csv) · [Historical test summary](../results/final_test_summary.csv)

The legacy 64 × 64 white padding could contribute to the color histograms. Its marginal RGB and HSV histograms discarded which colors and saturation levels occurred together. Fine-grained HOG bins depended on a flower's position and orientation. The scripts checked that cached label sequences agreed, which cannot catch reordered photos within one class. The old GUI also carried its own preprocessing code alongside the training scripts.

The rebuilt extractor corrects EXIF orientation, converts to RGB, resamples with Lanczos and scales pixels to `[0, 1]`. It compares 128 × 128 center cropping with aspect-preserving resizing and reflected padding. Color and texture histograms exclude the padding. The all-feature, row-normalized crop configuration scored **72.99% CV** versus **72.89%** for padding. The saved models and GUI use the same extractor.

## 3. Features

| Feature group | Dimensions | What it describes |
|---|---:|---|
| Joint color | 2,399 | Hue/saturation, saturation/value, Lab chroma, hue/lightness, and spatial position |
| HOG | 1,764 | Local gradient directions on the 128-pixel image with 16-pixel cells |
| LBP | 216 | Texture patterns at three radii |
| Edges | 432 | Gradient direction, direction relative to the image center, and edge strength at three scales |
| Local visual words | 1,320 | Pooled differences between image patches and 24 patch centers |

The color histograms compare the full image, two levels of center weighting, the periphery, a center-weighted view that downweights green, and four hue/saturation quadrants. The square root reduces the influence of dominant bins. The local descriptor samples 98 patches at two scales. Each patch combines 36 HOG values and 19 coarse color values. MiniBatchKMeans learns 24 centers from training patches. Each photo contributes a normalized sum of patch residuals per center. The representation adds 24 × 55 pooled dimensions. [Spatial pooling reference](https://mplab.ucsd.edu/~marni/Igert/Lazebnik_06.pdf)

## 4. Experiments and selection

| Logistic Regression experiment | CV accuracy | Validation accuracy |
|---|---:|---:|
| Legacy HOG + RGB + HSV | 58.07% | 56.90% |
| Joint color alone | 67.02% | 64.65% |
| HOG alone | 47.37% | 44.03% |
| Color + HOG | 70.33% | 69.92% |
| Color + HOG + LBP | 71.59% | 71.78% |
| All global features, crop | 72.99% | 73.49% |
| Standardized global groups | 74.48% | 72.56% |
| Local visual words alone | 66.32% | 66.51% |
| Global + local descriptors | 76.20% | 75.50% |
| Global + local, flip | 76.17% | 77.52% |
| Global + local, quadratic map | **76.90%** | 76.43% |

Color produced the largest gain. HOG, LBP and the multiscale edge features added complementary information. Standardization helped the global representation. For that representation, 160-component PCA scored 70.57% CV and a larger logistic `C` raised the training score while lowering CV accuracy. With local features, a quadratic similarity map led in CV macro F1.

The augmentation experiments compared original training photos with horizontal flips and with a flip plus a separate 9° rotation, 7% brighter exposure and 8% lower contrast. Each fold was split before augmented rows were added. The flip trial had the highest validation accuracy, while its mean CV macro F1 was lower than the quadratic model's. The final winner uses original training rows.

Every model used the same three stratified group CV folds inside the 3,017-image training partition. Training accuracy uses original training images. The mean fold macro F1 selected the configuration within each model family, with mean CV accuracy as the tiebreaker. The final K-Means configuration was selected among five-cluster one-to-one mappings so every flower label is available. Validation, internal test, and external test were reserved from fitting. [Experiment history](results/experiments.csv) · [Fold assignments](results/cv_folds.csv)

## 5. Saved models and hyperparameters

| Family | Fitted features and settings | Estimator inputs |
|---|---|---:|
| Logistic Regression | Center crop, standardized global groups, 24 visual words, weights 2.0 / 0.55 / 0.7 / 0.7 / 1.25, degree-2 Nyström map, gamma 0.18, 1,300 landmarks, `C=18` | 1,300 |
| KNN | Same global and local groups and weights, per-image L2 normalization, PCA 128, 13 Euclidean neighbors, distance-weighted votes | 128 |
| K-Means | Center crop, four global groups with per-image L2 normalization, weights 2.0 / 0.65 / 0.85 / 0.85, PCA 40, k-means++ initialization, 15 starts, five clusters | 40 |

The Logistic Regression and KNN representations each have 6,131 dimensions before their learned similarity map or PCA. K-Means uses 4,811 before PCA. By approximating degree-2 similarities between training images, the Nyström map lets Logistic Regression learn interactions between the handcrafted features. K-Means groups images by feature distance. The label assignment maximizes training label matches while using each of the five labels once. A cluster can still contain a mix of species because the clustering objective uses feature geometry. [Nyström transformer](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html) · [Selected configurations](results/selected_models.csv)

## 6. Evaluation protocol

The 645-image internal test is held out in the rebuilt pipeline. Its members include 452 images from the historical training list, 94 from the historical validation list, and 99 from the previously accessed historical test list. It estimates performance within the same collection and is historically reused.

The independent field test used 234 photos from 2025 iNaturalist observations. The class mapping uses *Bellis perennis*, *Taraxacum*, *Rosa*, *Helianthus annuus*, and *Tulipa*. Research Grade observations carry the flowering annotation, and at most one photo per observer was sampled within each class. The first eligible openly licensed photo of each selected observation was used. The cap was 50 per class. Tulip had 34 eligible observers, yielding 50 photos for each of the other four classes and 34 for tulip. The overlap audit compared the external photos with all raw Kaggle photos and found zero matches under the audit method. The manifest was fixed before scoring the CV-selected winner. It records observation links, dates, taxa, attribution, licenses and file hashes. [iNaturalist annotation filters](https://help.inaturalist.org/en/support/solutions/articles/151000170738-how-to-use-inaturalist-search-urls) · [External manifest](results/external_manifest.csv)

## 7. Results

| Model | Train acc. | CV acc. | Validation acc. | Internal test acc. | External test acc. | CV macro F1 | Validation macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | 99.30% | **76.90%** | **76.43%** | **74.73%** | **46.58%** | **76.66%** | **76.21%** |
| KNN | 100.00% | 67.15% | 65.89% | 70.39% | — | 66.96% | 65.86% |
| K-Means | 38.88% | 39.08% | 36.59% | 39.07% | — | 38.85% | 36.39% |

Logistic Regression got 482 of 645 internal test images correct, with **74.58% macro F1**. It got 109 of 234 independent field images correct, with **42.37% macro F1**, **49.08% macro precision**, and **44.35% macro recall**. The independent test evaluated the CV-selected Logistic Regression model once. The figures show counts and percentages normalized by the true class. [Machine-readable comparison](results/model_comparison.csv) · [Comparison chart](figures/model_comparison.png) · [All per-class metrics](results/per_class_metrics.csv) · [All confusion matrices](results/confusion_matrices.csv)

| Flower | Internal precision | Internal recall | Internal F1 | External precision | External recall | External F1 | External support |
|---|---:|---:|---:|---:|---:|---:|---:|
| Daisy | 66.95% | 69.30% | 68.10% | 49.02% | 50.00% | 49.50% | 50 |
| Dandelion | 77.12% | 74.68% | 75.88% | 36.45% | 78.00% | 49.68% | 50 |
| Rose | 73.39% | 68.38% | 70.80% | 64.71% | 22.00% | 32.84% | 50 |
| Sunflower | 79.82% | 83.49% | 81.61% | 66.67% | 60.00% | 63.16% | 50 |
| Tulip | 75.50% | 77.55% | 76.51% | 28.57% | 11.76% | 16.67% | 34 |

## 8. Confusions and generalization

On internal test, the leading two-way confusions were **daisy ↔ dandelion** (35 errors) and **rose ↔ tulip** (33). Nineteen dandelions went to daisy and 16 daisies went to dandelion. Seventeen roses went to tulip and 16 tulips went to rose. Sunflower had the highest internal recall at 83.49%. [Internal-test confusion matrix](figures/logistic_regression_internal_test.png)

On the field set, Logistic Regression predicted dandelion for 107 photos. This included 19 of 34 tulips and 17 of 50 sunflowers. Rose recall fell from 68.38% to 22.00%, and tulip recall fell from 77.55% to 11.76%. The iNaturalist set uses narrower or different taxonomic coverage within the same labels and includes field photos at varying distances. Its results describe transfer to that sample. The field test does not establish how much each dataset difference contributed. [External-test confusion matrix](figures/logistic_regression_external_test.png)

The selected Logistic Regression's 22.87-point train-to-validation gap shows overfitting. Its validation accuracy rose from 69.77% with 754 original training examples to 76.43% with 3,017 in the learning curve. The smallest subset uses 754 Nyström landmarks and later subsets use 1,300. KNN includes each training image as its own zero-distance neighbor when reporting training accuracy, which explains the 100% in that column. K-Means had a training silhouette score of **0.114** on a seeded 1,300-image sample, consistent with weak separation into five clusters. [Learning curve](figures/learning_curve.png) · [Learning curve data](results/learning_curve.csv)

## 9. Conclusion and demonstration

The best five-class system improved cross-validation accuracy by **18.83 percentage points** over the legacy recipe on the same rebuilt split. Joint color, shape, texture, and local patch features fed into the quadratic map produced the strongest CV result. The project fell short of **80% independent test accuracy**, and the field test exposed a substantial generalization gap.

The full training run reproduced the selected settings. The saved models were exercised in Streamlit with a local example, a real JPEG upload, and an invalid file on desktop and mobile Chromium. The demo displays the selected model, image, predicted flower, and either five estimated class probabilities or the K-Means cluster. [Run instructions](../../README.md) · [Streamlit app](../../gui/image_app.py)
