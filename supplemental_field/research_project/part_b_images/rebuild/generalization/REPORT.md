# Part B generalization follow-up

The study reuses the existing Kaggle split, the three classifier families, and the original feature and model pipelines. The earlier bundles and Streamlit workflow remain available. All candidates use handcrafted features.

## Framing and appearance

The audit used a seeded sample of 100 Kaggle development photos per class and all 234 photos from the published 2025 field test. The 2025 sample was used to describe the domain shift. Its photos and predictions were excluded from fitting and model selection.

| Source and class | Median area removed by square crop | Median foreground cue removed | Median green pixel fraction |
|---|---:|---:|---:|
| Kaggle rose | 33.33% | 22.53% | 8.77% |
| 2025 iNaturalist rose | 25.00% | 20.15% | 55.42% |
| Kaggle tulip | 33.40% | 26.80% | 22.99% |
| 2025 iNaturalist tulip | 25.00% | 22.57% | 26.41% |

The foreground cue emphasizes colors that differ from the photo border and downweights green. The discarded cue scores are proxies for lost visual information. The seeded sheets show petals clipped in some off-center roses and a tall field tulip clipped at the flower tip and stem. Other crops remove background and make small flowers occupy more of the square. [Framing audit](framing_summary.csv) · [Rose sheet](rose_framing.png) · [Tulip sheet](tulip_framing.png)

The sampled Kaggle roses include layered garden roses and bouquets. The 2025 iNaturalist sample includes smaller wild flowers against foliage, sometimes photographed at shrub scale. It includes 16 *Rosa rugosa* and 13 *Rosa multiflora* photos out of 50 roses. For tulip, 21 of 34 observations were *Tulipa sylvestris* or its subspecies. The sheet includes slender yellow wild tulips, while many sampled Kaggle tulips are broad, brightly colored garden varieties. Kaggle supplies folder labels at the class level, so the audit does not assign species to its photos. The iNaturalist flowering annotation applies to the observation, which can contain multiple photos. [Taxon counts](older_field_taxa.csv) · [iNaturalist annotations](https://help.inaturalist.org/en/support/solutions/articles/151000169941-what-are-tags-observation-fields-and-annotations-)

## Development protocol

- Kaggle uses its original 3,017 training rows, three fixed grouped CV folds, and 645 validation rows. Its historical internal test score remains in the [previous report](../REPORT.md).
- The 562 additional field photos come from 2022–2024 iNaturalist observations. The sampler retains at most one photo per observer within each class, and groups all photos by observer across classes when splitting. It stratifies the groups into 112 validation photos and 450 training photos with three CV folds.
- The 150 fresh test photos come from 2026 iNaturalist observations. Support is 38 daisy, 38 dandelion, 38 rose, 25 sunflower, and 11 tulip. The two field manifests have disjoint observers from each other and from the 2025 test. The overlap audit used file hashes, decoded-pixel hashes, Flickr IDs where available, and perceptual checks against the raw Kaggle collection and both field tests. The audit found zero overlaps under those checks.
- Model selection averages Kaggle and field CV macro F1 with equal weight. Logistic Regression candidates vary cropping, feature size, regularization, and exposure to field examples. A weighted fit gives the two sources equal total training weight. The class-balanced variant also gives each class equal weight within each source. KNN and K-Means use the compact features and the same development partitions.

The plain representation uses 369 dimensions from joint color histograms, pooled gradient summaries, and LBP. Adding foreground-weighted summaries gives 738 dimensions. Both use linear Logistic Regression with L2 regularization. The original Nyström Logistic Regression uses 6,131 pooled input dimensions and maps them onto 1,300 similarity features. Train and validation columns use the fitted model's predictions. The search table reports mean fold metrics, and the detailed development tables pool the out-of-fold predictions across folds. [Development split](development_split.csv) · [Candidate results](experiments.csv) · [Supplemental manifest](field_manifest.csv) · [Fresh test manifest](fresh_manifest.csv) · [Overlap exclusions](overlap_exclusions.csv)

## Development results

Each cell is **accuracy / macro F1**, in percent.

| Model | Kaggle train | Kaggle CV | Kaggle validation | Field train | Field CV | Field validation |
|---|---:|---:|---:|---:|---:|---:|
| Existing Nyström LR | 99.30 / 99.28 | 76.90 / 76.66 | 76.43 / 76.21 | — | 46.44 / 43.75 | 45.54 / 43.50 |
| Nyström LR + field training | 98.61 / 98.61 | 76.67 / 76.51 | 75.66 / 75.48 | 97.33 / 97.31 | 61.11 / 60.52 | 54.46 / 54.92 |
| Compact LR + balanced field training | 76.96 / 76.86 | 69.31 / 69.21 | 68.84 / 68.36 | 90.67 / 90.83 | 63.33 / 62.68 | 60.71 / 61.43 |
| Compact KNN + field training | 71.73 / 71.59 | 64.57 / 64.37 | 64.50 / 64.28 | 54.00 / 53.20 | 43.78 / 41.86 | 49.11 / 47.62 |
| Compact K-Means + field training | 39.64 / 38.10 | 37.22 / 35.44 | 37.05 / 35.57 | 25.11 / 19.35 | 24.44 / 20.18 | 22.32 / 18.08 |

**Cropping.** With the original Nyström settings, the full frame raised field CV accuracy from 46.44% to 48.44%. Field CV rose recall went from 27.08% to 32.29% and tulip recall stayed at 13.54%. Field validation accuracy fell from 45.54% to 43.75%. In the balanced compact model, cropping improved field CV from 58.00% to 63.33% and rose/tulip recall from 58.33%/39.58% to 64.58%/45.83%. Full framing preserves the edge details visible in the sheets, but the CV results depend on the representation and training data.

**Regularization.** Reducing `C` from 18 to 1 in the same Kaggle-only Nyström pipeline cut the Kaggle train–validation gap from 22.87 to 15.47 percentage points while CV accuracy stayed near 77%. Field CV accuracy fell to 43.78%. The selected compact model further reduced the Kaggle gap to 8.13 points and the field gap to 29.95 points. The overall selected Nyström model still has gaps of 22.95 points on Kaggle and 42.87 points on field validation.

**Foreground cues and new examples.** Adding the foreground summaries to the full-frame compact Kaggle-only model improved Kaggle CV from 70.83% to 72.82% and field CV from 37.78% to 41.33%. At the original Nyström settings, adding field training lifted field CV by 14.67 points at a cost of 0.23 points on Kaggle CV. This was the strongest joint result by the preselected metric, with 68.52% mean of the two CV macro F1 scores. The best compact model reached 65.95% on that metric, with a higher field-only CV score.

**Learning curves.** The compact model's field validation accuracy increased from 49.11% at 867 fitting photos to 60.71% at 3,467. The mixed Nyström model improved from 45.54% to 54.46% over those sizes, with a peak of 55.36% at 1,734. The remaining gaps differ by representation and source. [Learning curve plot](learning_curves.png) · [Learning curve data](learning_curves.csv) · [Development per-class metrics](development_per_class.csv) · [Development confusion counts](development_confusions.csv)

## Fresh independent test

The configurations were fixed using development CV before scoring this set. The archived 645-image Kaggle test and the published 234-image 2025 field test were not rescored to select the follow-up models.

| Model | Correct / 150 | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Existing Nyström LR | 79 | 52.67% | 51.01% |
| **Nyström LR + field training** | **112** | **74.67%** | **71.81%** |
| Compact LR + balanced field training | 107 | 71.33% | 69.16% |
| Compact KNN + field training | 78 | 52.00% | 50.58% |
| Compact K-Means + field training | 41 | 27.33% | 18.06% |

| Flower | Support | Old LR recall | New Nyström precision | New Nyström recall | New Nyström F1 | Compact precision | Compact recall | Compact F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Daisy | 38 | 50.00% | 73.17% | 78.95% | 75.95% | 69.05% | 76.32% | 72.50% |
| Dandelion | 38 | 76.32% | 73.81% | 81.58% | 77.50% | 65.85% | 71.05% | 68.35% |
| Rose | 38 | 18.42% | 90.00% | 71.05% | 79.41% | 90.00% | 71.05% | 79.41% |
| Sunflower | 25 | 76.00% | 77.27% | 68.00% | 72.34% | 85.00% | 68.00% | 75.56% |
| Tulip | 11 | 45.45% | 46.67% | 63.64% | 53.85% | 41.18% | 63.64% | 50.00% |

Field training improved the selected model by **22.00 percentage points** over the original on the same 150 images. Rose recall rose from 7/38 to 27/38. Both updated LR configurations got 7/11 tulips. Sunflower recall fell from 19/25 to 17/25. The compact model beat the original on this sample and finished five correct predictions behind the updated Nyström model. The top accuracy fell short of 80%. The test is specific to the sampled iNaturalist period, taxa, and photo filters. Eleven tulips give little precision for that class, and the result does not establish performance on other field collections. [Fresh per-class metrics for all models](fresh_test_per_class.csv) · [Fresh confusion counts and row percentages](fresh_test_confusions.csv) · [Per-image predictions](fresh_test_predictions.csv)

The original model files remain in the [prior model directory](../models/). The [run instructions](../../../README.md) show how to launch the new three-model set in Streamlit. The compact Logistic Regression benchmark is saved as `models/compact_logistic.joblib`.
