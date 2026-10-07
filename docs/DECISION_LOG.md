# Project Decision Log

## 1. Prediction Target

**Decision:** Predict whether a mature historical booking was confirmed as harmful.

- `is_harmful = 1` for confirmed fraud, abuse, chargeback-related harm, or another defined harmful outcome.
- `is_harmful = 0` when the outcome window completed without confirmed harm.
- Unresolved or immature outcomes are excluded from supervised training and evaluation.
- Harm type, amount, and severity remain evaluation fields rather than prediction-time inputs.

**Reason:** The model must rank bookings before fulfillment, while learning from historical outcomes that became known later.

## 2. Dataset Design

**Decision:** Use a hybrid travel dataset.

- Preserve the real booking context from the public Hotel Booking Demand dataset.
- Add clearly identified synthetic traveler, property, device, payment, and risk fields.
- Preserve the original cancellation outcome separately from the simulated harmful-booking target.

**Reason:** A real travel backbone is more relevant than a generic payment dataset, while simulation supplies the identity and harm fields that public travel datasets do not provide.

## 3. Prediction-Time Boundary

**Decision:** Score each booking immediately after creation and exclude information created or changed later.

Excluded fields include cancellation, assigned room, later booking changes, waiting-list duration, final reservation status, harm details, and outcome-maturity time.

**Reason:** Using later information would make historical evaluation look stronger than the model could perform at the real decision time.

## 4. Missing Identity and Device Signals

**Decision:** Retain bookings with missing `email_age_days` or `device_trust_score`, create a binary missing indicator for each field, and fill the original numerical fields with the training-set median.

**Reason:** Missing identity information can itself be associated with suspicious behavior. Deleting those rows could remove important risk cases and bias the sample toward well-documented bookings. Zero is not a neutral replacement: zero email age represents a genuinely new email, while zero device trust represents the lowest possible trust. The median provides a stable central value, while the separate indicator preserves the information that the original value was unavailable. Medians must be learned from training data only and then applied unchanged to validation, test, and production data.

## 5. Validation Structure

**Decision:** Preserve booking-time order when creating training, validation, and final test periods.

- Training contains earlier bookings.
- Validation contains the following period and supports model and threshold selection.
- The final test set contains the latest period and remains untouched until the approach is selected.
- Returning travelers, devices, payment instruments, and properties may naturally appear in later periods because that reflects production traffic.
- Entity-history features must use earlier activity only.
- Model performance will be reported separately for previously seen and first-seen entities to expose cold-start behavior.

**Reason:** A random split would mix market conditions from different dates and could allow repeated entities and later behavior to make evaluation unrealistically easy. A chronological design tests whether the model trained on the past can rank future bookings.

## 6. Label Maturity

**Decision:** At each simulated training date, exclude earlier bookings whose outcomes had not yet matured.

The rows are not permanently discarded. When their outcomes become available, they can enter a later retraining dataset.

**Reason:** Knowing an outcome today does not mean it would have been available when the historical model was trained. Enforcing outcome maturity reproduces the information the team actually possessed at each cutoff and prevents label-availability leakage.

## 7. Entity Identifiers

**Decision:** Do not feed raw traveler, device, payment-instrument, or property identifiers into the model. Retain them for joins, point-in-time history construction, validation, and monitoring.

The model will use derived behavior such as:

- Prior bookings, mature harmful outcomes, chargebacks, and recent activity for a traveler.
- Recent bookings, distinct travelers, mature harmful outcomes, and account sharing for a device.
- Recent attempts, distinct travelers or devices, and mature chargebacks for a payment instrument.
- Prior booking volume and mature harmful-outcome behavior for a property.
- New-entity indicators and default history values when an entity has not appeared before.

**Reason:** Raw identifiers encourage memorization and do not generalize to a new traveler or device. Point-in-time behavioral summaries preserve the useful repeat-pattern signal while giving the model features it can apply to both known and unseen entities.

## 8. Sparse Entity Risk Rates

**Decision:** Use a weighted, smoothed harm index rather than a raw entity harm rate.

The implementation combines each entity's mature harmful outcomes with the marketplace-wide mature harm rate using a 20-booking prior. It also retains the entity's history count as a separate feature.

For example, with a 2.5% marketplace rate, an entity with one harmful outcome from one mature booking receives a smoothed rate of about 7.1%, not a raw rate of 100%. An entity with 30 harmful outcomes across 1,000 mature bookings remains close to its observed rate of 3%.

**Reason:** A raw percentage based on one or two observations is unstable. Smoothing allows small histories to move away from the marketplace baseline without letting them dominate as if they carried the same evidence as hundreds of observations.

## 9. Agent and Company Identifiers

**Decision:** Do not one-hot encode or otherwise feed raw agent and company IDs into the model.

The model will instead use:

- `has_agent` and `has_company` indicators.
- Prior booking volume for a known agent or company.
- Mature outcome counts and smoothed historical harm rates.
- Prior chargeback counts and new-entity indicators.

Missing agent or company values mean that no such entity is attached to the booking. They are not pooled into one shared entity when history is calculated.

**Reason:** Raw IDs could allow the model to memorize entities that happened to be risky in the training period. Historical behavior retains useful risk information, works for new entities, and separates evidence strength from the estimated harm rate.

## 10. Categorical Cleaning and Country

**Decision:** Retain country as a model feature after controlled cleaning and rare-category grouping.

- Standardize whitespace, missing tokens, and confirmed country-code aliases before counting categories.
- Keep missing or undefined values as `Unknown` because missingness can carry information.
- Group categories observed fewer than 50 times in the mature training data into `Other`.
- Map categories first seen in validation, test, or production to `Other`.
- One-hot encode the cleaned nominal categories, with the encoder fitted on training data only.
- Keep `traveler_country_matches_booking` as a separate behavioral feature.
- Evaluate performance across major country groups to detect geographic instability or unfair concentration of false positives.

**Reason:** Country can provide useful travel context, but unclean codes and very sparse categories create unstable estimates. Cleaning before grouping prevents equivalent values from being counted separately, while the `Unknown` and `Other` categories handle different situations: unavailable information versus a valid but uncommon value.

## 11. Extreme Average Daily Rate

**Decision:** Retain bookings with an invalid or extreme average daily rate, create `adr_negative` and `adr_extreme` indicators, replace negative ADR with zero, and cap high ADR at the 99.9th percentile learned from the mature training data.

The mature development-training period produces an initial cap of approximately `$296.14`. The fitted cap is applied unchanged to validation, test, and production records. The capped and uncapped approaches will be compared during model validation.

**Reason:** An extreme value may represent a valid expensive booking, a currency problem, or a data-entry error. A negative price is not a valid exposure amount but remains a useful data-quality signal. Removing either row would discard the complete booking. Capping limits numerical influence, while the indicators preserve the fact that the original values were unusual.

## 12. Zero and Missing Guest Counts

**Decision:** Retain bookings with zero reported guests and add the following features:

- `total_guests`, calculated from adults, children, and babies.
- `invalid_guest_count`, equal to one when total guests is zero.
- `children_missing`, equal to one when the original child count is unavailable.

Missing child counts are treated as zero only when deriving `total_guests`; the missing indicator preserves that assumption. The original booking is not automatically changed to one adult.

**Reason:** The dataset has no independent occupancy field that can reliably reconstruct the intended guest count. Removing the complete booking could discard a useful risk or data-quality signal, while imputing one adult would invent information. In production, the team would also verify whether the booking service prevents zero-guest records from reaching the scoring population.

## 13. Candidate Models and Model-Specific Missing Values

**Decision:** Develop both logistic regression and XGBoost using the same chronological training, validation, and final test periods.

- Logistic regression is the interpretable baseline. Numerical missing values are filled using training-set statistics, missing indicators are retained, and numerical features are standardized. Both unweighted and balanced-class versions are evaluated rather than assuming weighting will improve the operational ranking.
- XGBoost is the nonlinear candidate. Numerical missing values remain missing where the library can handle them directly, and numerical scaling is not applied.
- Both models use the same approved prediction-time features, cleaned categories, target definition, and evaluation population.
- `traveler_days_since_last_booking` remains null for a new traveler in the feature table. Logistic regression receives a training-median placeholder plus `is_new_traveler`; XGBoost receives the missing value plus the same indicator.

**Reason:** Logistic regression provides a clear benchmark and easily explained coefficients. XGBoost can learn nonlinear effects and interactions among booking, identity, payment, and history signals. Separate preprocessing respects the requirements of each algorithm while preserving a fair business-level comparison.

**Observed development result:** Balanced class weights produced almost no improvement in recall within the top 5%, but substantially worsened probability calibration. The unweighted model is therefore the primary logistic baseline; the weighted result remains documented as an imbalance experiment.

## 14. Calendar and Time Features

**Decision:** Convert booking hour, booking day of week, booking month, arrival day of week, and arrival month into sine/cosine feature pairs. Also retain weekend indicators and the original `lead_time`.

After deriving these features, raw booking timestamps and overlapping arrival year, month, week-number, and day fields are excluded from the model matrix. Original timestamps remain available for chronological splitting, audit, and drift monitoring.

**Reason:** Calendar variables repeat in cycles. December is close to January, Sunday is close to Monday, and 11 PM is close to midnight. Plain integer encoding would incorrectly treat these values as far apart, while one-hot encoding would not express their circular relationship. Removing overlapping raw date fields limits redundancy and discourages memorization of a particular historical period.

## 15. Stay Duration and Booking-Value Exposure

**Decision:** Create:

- `total_nights = stays_in_week_nights + stays_in_weekend_nights`
- `estimated_booking_value = capped_adr * max(total_nights, 1)`
- `log_estimated_booking_value = log(1 + estimated_booking_value)`

The ADR cap must be fitted and applied before estimated booking value is calculated. A zero-night record receives one night only in the exposure calculation; its original `total_nights = 0` remains available to the model.

**Reason:** Nightly rate alone does not measure the amount exposed in a booking. A seven-night booking at the same rate carries more financial value than a one-night booking. Retaining total nights separately prevents the minimum-one-night calculation from hiding an unusual zero-night record. The log version reduces right skew for logistic regression, while the raw value remains available to XGBoost and business-facing explanations.

## 16. XGBoost Hyperparameter Search

**Decision:** Tune XGBoost with randomized search over chronological, rolling validation folds that enforce label maturity at every fold cutoff.

The search covers learning rate, tree depth, number of trees, minimum child weight, row and feature subsampling, split penalty, L1 regularization, L2 regularization, and positive-class weight. Recall within the top 5% review capacity is the primary search score. Average precision, precision at 5%, calibration, and harmful value captured are supporting measures.

Validation and final test data retain their natural class distribution. The final test period remains untouched until model choice, hyperparameters, and the operating threshold are fixed.

**Reason:** A full grid becomes expensive as the number of parameters grows and spends equal effort on combinations that are unlikely to work. Randomized search explores a wider range within a fixed compute budget. Chronological folds test whether the configuration generalizes forward in time, while label-maturity checks prevent outcomes from entering training before they would have been known.

## 17. Value-Aware Review Priority

**Decision:** Optimize the review queue for expected financial harm prevented. Rank bookings by:

`predicted harm probability * estimated booking value`

The estimated booking value is available at scoring time. Actual `harm_amount` remains an outcome-only field used exclusively for retrospective evaluation. The current candidate policy uses tuned XGBoost and sends the top 5% of expected-loss scores to manual review.

**Observed development result:** XGBoost probability ranking captured 25.90% of harmful value, while expected-loss ranking captured 29.44% at the same review capacity. Expected-loss ranking reduced harmful-booking recall from 16.59% to 14.38%, which is consistent with prioritizing fewer but more expensive incidents.

**Reason:** The business objective values prevented financial loss more than the raw number of harmful incidents. Multiplying probability by prediction-time exposure aligns queue priority with that objective without leaking the later realized loss amount.

## 18. SMOTENC Experiment

**Decision:** Do not use SMOTENC in the current operating candidate.

SMOTENC was applied only to mature training data, with categorical variables preserved during synthetic sampling. Sampling strategies of 0.10, 0.25, and 0.50 were evaluated on the unchanged future validation period.

| Minority-to-majority ratio | Harm value captured at 5% |
|---:|---:|
| 0.10 | 20.95% |
| 0.25 | 19.78% |
| 0.50 | 19.24% |

All three versions performed below unweighted logistic expected-loss ranking at 28.00% and tuned XGBoost expected-loss ranking at 29.44%.

**Reason:** Class imbalance does not automatically require synthetic oversampling. In this dataset, synthetic neighbor combinations weakened the ranking of the highest-value harmful bookings. The natural-distribution XGBoost model also selected `scale_pos_weight = 1`, providing separate evidence that heavier positive weighting was not beneficial.

## 19. Probability Calibration

**Decision:** Retain the original XGBoost probabilities rather than applying Platt or isotonic calibration.

The future validation period was divided chronologically: calibration methods were fitted on the first half and evaluated on the second half. The uncalibrated model achieved a Brier score of 0.02329, log loss of 0.10983, expected calibration error of 0.00522, and 29.46% harmful-value capture at the 5% review limit.

Isotonic calibration improved Brier score by only 0.00003 while worsening log loss, calibration error, and harmful-value capture. Platt calibration also reduced harmful-value capture.

**Reason:** A calibration method must produce a meaningful and consistent improvement, not win through a negligible change in one metric. The raw probabilities were already well calibrated and supported the stronger expected-loss ranking.

## 20. Final Test Result

**Decision:** Accept tuned, uncalibrated XGBoost with expected-loss ranking as the final evaluated candidate.

The model was retrained on 88,479 bookings whose outcomes had matured by the test cutoff. The locked policy was then evaluated once on 17,909 untouched future bookings.

- Review capacity: 896 bookings, or 5% of traffic.
- Harmful bookings caught: 87 of 480.
- Precision at 5%: 9.71%.
- Recall at 5%: 18.13%.
- Harmful value captured at 5%: 37.19%.

For comparison, probability-only ranking caught 99 harmful bookings but captured 28.93% of harmful value. The result confirms that expected-loss ranking sacrifices some incident count to prioritize more expensive harm, matching the selected business objective.

## 21. Batch Scoring and Monitoring

**Decision:** Deploy the candidate as a batch ranking workflow that scores every booking, calculates expected loss, and sends exactly the top 5% to manual review.

Monitoring covers score distributions, PSI drift, review volume, mature-outcome precision and recall, harmful-value capture, false-positive rate, new-entity performance, and country groups with sufficient volume.

The production-style test split produced 5.00% review volume, 39.63% harmful-value capture in the later batch, and expected-loss score PSI of 0.070 versus the preceding batch. Spain's false-positive rate was 9.78% versus 4.65% overall, which is retained as an investigation signal rather than treated as proof of bias.

**Reason:** A model is useful only when its scores become a controlled queue and its behavior remains observable after deployment. Monitoring both aggregate business value and segment-level errors catches failures that one overall metric would hide.
