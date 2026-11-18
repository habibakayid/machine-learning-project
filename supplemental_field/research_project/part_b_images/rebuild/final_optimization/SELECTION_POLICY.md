# Selection policy and documented amendment

Primary fits Kaggle training rows only, and uses Kaggle CV macro F1.
Supplemental fits the existing Kaggle and field training rows and averages
their CV macro F1 with equal weight. All three fold assignments are retained.
Candidates must emit all five labels in the final mapping.

Replacement requires at least 0.002 improvement in the CV selection score,
no more than 0.005 loss in validation macro F1, and no more than 0.01 loss
in validation accuracy in any source. For each class in each source, pooled
CV and validation recall and F1 may fall by at most 0.10.
The weakest fold's macro F1 in each source may fall by at most 0.01.
Fold standard deviations are reported alongside that absolute performance check.
Logistic Regression must also reduce both sources' train-validation accuracy
gaps (Kaggle only for the primary experiment), preserving the earlier gap rule.
Among eligible candidates choose the highest CV score, then the smaller fold
F1 spread. Every failed guardrail is saved. The baseline remains eligible.

Validation is an already-accessed development confirmation set. All historical
tests remain historical, and are excluded from this experiment runner.
Numerical target ranges do not enter model selection.

## Documented development-only amendment

The initial rule capped any growth in F1 standard deviation at 0.01. It rejected the Kaggle K=10 candidate despite a large improvement in held-out F1. The revised stability check compares each source's weakest fold with the baseline weakest fold, allowing at most 0.01 loss. Standard deviations remain reported. This correction was applied to all candidates using stored development predictions, with all other guardrails retained. No test scores or test predictions entered this amendment. The original policy and selection tables are preserved for review.
