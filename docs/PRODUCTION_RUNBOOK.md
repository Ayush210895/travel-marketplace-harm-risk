# Production Runbook

## Scoring Contract

Score each booking after creation and before fulfillment. Upstream systems must supply prediction-time booking fields and point-in-time entity-history features. Raw entity IDs are used only to build those histories and are not model inputs.

The batch scorer produces:

- `harm_probability`
- `estimated_booking_value`
- `expected_loss_score`
- `review_rank`
- `send_to_review`

The highest 5% of expected-loss scores enter manual review.

## Outcome Maturity

Do not calculate supervised performance until the booking's outcome window has completed. A booking can enter later retraining data only when `outcome_matured_at` is no later than the training cutoff.

## Monitoring Cadence

### Every Scoring Batch

- Row count and schema validation
- Missing rates for email age and device trust
- Unknown and Other category rates
- Mean, median, p95, and p99 expected-loss score
- Exact review volume and review rate
- New traveler, device, and payment shares

### After Outcomes Mature

- Precision and recall at 5% review capacity
- Harmful value captured at 5%
- False-positive rate
- Performance for new versus known entities
- Country-group review, recall, and false-positive rates for groups with sufficient volume

## Drift Guidance

Expected-loss score PSI is a screening signal:

- Below 0.10: low observed drift
- 0.10 to 0.25: investigate feature and traffic changes
- Above 0.25: material shift requiring model and data review

These are operating guidelines, not universal statistical laws. Review score drift together with input drift, outcome performance, campaign changes, seasonality, and marketplace events.

The production-style demonstration produced PSI 0.070 between consecutive test batches, indicating low score drift.

## Segment Review

A higher country review rate is not automatically unfair because underlying harm rates and booking values differ. Investigate when a sufficiently large segment has:

- A materially higher false-positive rate than the overall population
- Worsening recall or value capture over time
- Large changes in review rate without a corresponding change in confirmed harm
- Poor calibration among similarly scored bookings

The demonstration surfaced Spain with a 9.78% false-positive rate versus 4.65% overall. This is an investigation signal, not proof of bias. Review booking-value mix, payment behavior, sample stability, and performance with country removed before changing policy.

## Retraining

Evaluate retraining monthly or when monitoring shows a sustained material change. Retraining must:

1. Use only mature outcomes at the new cutoff.
2. Refit every preprocessing statistic on the new training period.
3. Preserve a later validation period and untouched final period.
4. Compare the challenger with the current model at the same review capacity.
5. Require improvement in harmful-value capture without unacceptable segment harm.
6. Version the model, preprocessing, data cutoff, parameters, and metrics together.

## Failure Handling

- Schema failure: reject the batch and alert; do not silently drop columns.
- Missing history service: use explicit new-entity defaults and record degraded mode.
- Scoring failure: retain the last successful queue and retry; do not send an incomplete ranking.
- Review-capacity change: rerank the complete batch and apply the new approved percentage.
- Severe drift or data corruption: fall back to the previous validated model or approved rules.

## Rollback Evidence

Keep the prior model artifact, feature contract, training cutoff, monitoring baseline, and deployment identifier. A rollback is complete only when scoring output, queue volume, and monitoring telemetry are verified with the restored version.
