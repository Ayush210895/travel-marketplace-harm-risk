# Development Model Results

## Evaluation Setup

- Mature training rows: 74,571
- Future validation rows: 17,908
- Harmful validation bookings: 452
- Manual-review capacity: 896 bookings, or 5% of validation traffic
- XGBoost search: 12 randomized configurations across three chronological, label-mature folds
- At this development stage, the final test period remained untouched

## Validation Results

| Model | PR-AUC | ROC-AUC | Brier | Precision at 5% | Recall at 5% | Harm value captured at 5% |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.0620 | 0.6290 | 0.0243 | 9.15% | 18.14% | 24.07% |
| Balanced logistic regression | 0.0577 | 0.6244 | 0.2507 | 9.26% | 18.36% | 24.02% |
| Tuned XGBoost | 0.0714 | 0.6330 | 0.0242 | 8.37% | 16.59% | 25.90% |

At the same review capacity, unweighted logistic regression catches 82 harmful bookings and tuned XGBoost catches 75. XGBoost nevertheless captures a larger percentage of total harmful value, which means the harmful bookings it ranks highest are more expensive on average.

## Interpretation

There is no automatic winner yet:

- Choose logistic regression if the operating objective treats each harmful booking similarly and prioritizes incident recall.
- Choose XGBoost if prevented financial loss is the primary objective.
- Consider value-aware ranking if review priority should combine harm probability with expected loss.

Balanced class weights did not provide a meaningful recall improvement and substantially worsened calibration. The unweighted logistic model is therefore the primary linear baseline.

The tuned XGBoost configuration selected no positive-class weighting. Its strongest settings used a low learning rate, 591 trees, depth 5, strong L2 regularization, and row and feature subsampling. The winning configuration's mean rolling-fold recall at 5% was 18.85%, compared with 16.59% on the later validation period. This gap supports continued temporal monitoring and cautions against selecting a model from cross-validation alone.

## Value-Aware Ranking Decision

The selected business objective is harmful financial value prevented. The queue score is therefore:

`predicted harm probability * estimated booking value`

| Model | Precision at 5% | Recall at 5% | Harm value captured at 5% |
|---|---:|---:|---:|
| Logistic expected-loss ranking | 7.14% | 14.16% | 28.00% |
| XGBoost expected-loss ranking | 7.25% | 14.38% | 29.44% |

Tuned XGBoost with expected-loss ranking is the current operating candidate. Actual realized harm amount is used only for retrospective evaluation, never as a prediction-time feature or queue input.

## Imbalance Experiment

SMOTENC was tested only on mature training data. Validation retained the natural 2.52% harm rate.

| SMOTENC sampling ratio | Precision at 5% | Recall at 5% | Harm value captured at 5% |
|---:|---:|---:|---:|
| 0.10 | 5.80% | 11.50% | 20.95% |
| 0.25 | 4.91% | 9.73% | 19.78% |
| 0.50 | 4.58% | 9.07% | 19.24% |

SMOTENC was rejected because every tested level reduced the selected business metric. This demonstrates that imbalance treatment was validated empirically rather than added automatically.

## Current Boundary

The business ranking objective, imbalance strategy, model choice, and operating policy were locked before opening the final test period.

## Final Test

- Mature training rows at the test cutoff: 88,479
- Final test rows: 17,909
- Harmful test bookings: 480
- Bookings flagged and ranked for manual review at 5% capacity: 896 of 17,909

| Locked ranking policy | Precision at 5% | Recall at 5% | Harm value captured at 5% |
|---|---:|---:|---:|
| XGBoost probability | 11.05% | 20.63% | 28.93% |
| XGBoost expected loss | 9.71% | 18.13% | **37.19%** |

The selected expected-loss policy flags and ranks the top 5% of bookings for manual review. Within those 896 flagged bookings, it catches 87 harmful bookings and captures 37.19% of realized harmful value. Probability-only ranking catches more incidents, 99, but captures substantially less harmful value. This confirms the expected trade-off established before the test was opened.
