# Selection policy fixed before experiments

Primary fits Kaggle training rows only, and uses Kaggle CV macro F1.
Supplemental fits the existing Kaggle and field training rows and averages
their CV macro F1 with equal weight. All three fold assignments are retained.
Candidates must emit all five labels in the final mapping.

Replacement requires at least 0.002 improvement in the CV selection score,
no more than 0.005 loss in validation macro F1, and no more than 0.01 loss
in validation accuracy in any source. For each class in each source, pooled
CV and validation recall and F1 may fall by at most 0.10.
The mean fold F1 standard deviation may grow by at most 0.01.
Logistic Regression must also reduce both sources' train-validation accuracy
gaps (Kaggle only for the primary experiment), preserving the earlier gap rule.
Among eligible candidates choose the highest CV score, then the smaller fold
F1 spread. Every failed guardrail is saved. The baseline remains eligible.

Validation is an already-accessed development confirmation set. All historical
tests remain historical, and are excluded from this experiment runner.
Numerical target ranges do not enter model selection.
