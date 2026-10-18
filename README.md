# Travel Marketplace Harm-Risk Prioritization

[![Quality gate](https://github.com/Ayush210895/travel-marketplace-harm-risk/actions/workflows/quality-gate.yml/badge.svg)](https://github.com/Ayush210895/travel-marketplace-harm-risk/actions/workflows/quality-gate.yml)

An end-to-end applied machine-learning project that scores hotel bookings before fulfillment and ranks the highest expected financial harm for manual review.

## Final Result

The locked policy uses tuned XGBoost and ranks bookings by:

```text
predicted harm probability * estimated booking value
```

On the untouched future test period, the model flagged and ranked the top 5% of bookings for manual review:

- Captured **37.19% of total harmful value**.
- Flagged **896 of 17,909 bookings** for manual review, matching the 5% review capacity.
- Caught 87 of 480 harmful bookings.
- Achieved 9.71% precision and 18.13% recall.
- Outperformed probability-only ranking on harmful value captured: 37.19% versus 28.93%.

## Business Contract

- One row is one booking.
- Scoring occurs immediately after booking creation and before fulfillment.
- The model estimates whether the booking will later be confirmed as harmful.
- Manual-review capacity is fixed at 5% of booking volume.
- The queue prioritizes expected financial loss rather than incident count alone.
- Only mature historical outcomes can enter training or evaluation.

## Data

The travel backbone is the public Hotel Booking Demand dataset from Antonio, Almeida, and Nunes, published in *Data in Brief* (2019): <https://doi.org/10.1016/j.dib.2018.11.126>.

The public dataset contains genuine hotel-booking context but no fraud label, customer identity, device, payment instrument, or harmful outcome. Those fields are generated deterministically for this project and are explicitly synthetic. The original cancellation outcome remains separate from the synthetic `is_harmful` target.

Raw and generated data are excluded from Git.

## Modeling Design

- Chronological train, validation, and final test periods.
- Label-maturity checks at every training cutoff.
- Point-in-time traveler, device, payment, property, agent, and company history.
- Raw entity IDs excluded from the model.
- Smoothed historical harm rates for sparse entities.
- Train-fitted missing-value handling, rare-category grouping, and outlier treatment.
- Cyclical booking and arrival calendar features.
- Logistic regression baseline and tuned XGBoost candidate.
- Randomized XGBoost search over three rolling, label-mature folds.
- SMOTENC tested and rejected because it reduced future harmful-value capture.
- Platt and isotonic calibration tested and rejected because neither produced a consistent improvement.

## Leakage Boundary

The model excludes fields created or changed after booking time, including cancellation outcome, assigned room, later booking changes, waiting-list duration, final reservation status, harm type, severity, amount, and outcome-maturity timestamp.

Outcome fields are retained only for retrospective training and evaluation.

## Repository Structure

```text
src/marketplace_risk/
  build_hybrid_dataset.py   Build deterministic hybrid travel-risk data
  historical_features.py   Point-in-time entity history
  preprocessing.py         Train-fitted cleaning and feature engineering
  split.py                 Chronological splits and maturity eligibility
  modeling.py              Baselines and XGBoost tuning
  smote_experiment.py       SMOTENC comparison
  calibration.py           Platt and isotonic evaluation
  final_evaluation.py       Locked final test
  scoring.py                Batch scoring and top-5% review queue
  monitoring.py             Performance, drift, and segment checks
tests/                      Leakage, preprocessing, scoring, and metric tests
docs/                       Audit, decisions, results, and production runbook
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Commands below use `PYTHONPATH=src` for portability across editable-install environments.

## Reproduce the Project

Download the public source data:

```bash
mkdir -p data/raw data/processed artifacts/models
curl -L \
  https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv \
  -o data/raw/hotels.csv
```

Build the hybrid dataset and chronological split:

```bash
PYTHONPATH=src .venv/bin/python -m marketplace_risk.build_hybrid_dataset \
  --source data/raw/hotels.csv \
  --output data/processed/hybrid_bookings.csv

PYTHONPATH=src .venv/bin/python -m marketplace_risk.split \
  --data data/processed/hybrid_bookings.csv \
  --assignments data/processed/split_assignments.csv \
  --manifest artifacts/split_manifest.json
```

Train and tune models:

```bash
PYTHONPATH=src .venv/bin/python -m marketplace_risk.modeling \
  --data data/processed/hybrid_bookings.csv \
  --manifest artifacts/split_manifest.json \
  --output-dir artifacts/models \
  --search-iterations 12
```

Run the test suite:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

## Batch Scoring

The input batch must contain prediction-time booking fields and point-in-time history features. The scorer creates probabilities, booking-value exposure, expected-loss scores, ranks, and a 5% review flag.

```bash
PYTHONPATH=src .venv/bin/python -m marketplace_risk.scoring \
  --model artifacts/models/final_test_metrics.joblib \
  --input path/to/booking_batch.csv \
  --output artifacts/scored/booking_scores.csv
```

## Monitoring

Run performance monitoring only after outcomes have matured:

```bash
PYTHONPATH=src .venv/bin/python -m marketplace_risk.monitoring \
  --current artifacts/scored/current_mature_scores.csv \
  --reference artifacts/scored/reference_mature_scores.csv \
  --output artifacts/monitoring/report.json
```

Monitoring covers review rate, harm rate, precision, recall, false-positive rate, harmful-value capture, expected-loss score drift, country groups, and new-entity segments.

## Documentation

- [Data audit](docs/DATA_AUDIT.md)
- [Decision log](docs/DECISION_LOG.md)
- [Model results](docs/MODEL_RESULTS.md)
- [Production runbook](docs/PRODUCTION_RUNBOOK.md)

## Important Boundary

This is a portfolio implementation built on public booking data with synthetic risk fields. Its results demonstrate methodology and engineering decisions; they are not claims about Expedia or any other travel company's customers, losses, or production systems.
