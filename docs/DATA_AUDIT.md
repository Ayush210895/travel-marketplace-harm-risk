# Hybrid Booking Dataset Audit

## Build Summary

- Rows: 119,390
- Columns: 55
- Booking-time range: June 24, 2013 through August 31, 2017
- Harmful bookings: 3,065
- Harm rate: 2.57%
- Unique travelers: 106,679
- Unique properties: 349
- Unique devices: 72,746
- Unique payment instruments: 69,373

## Outcome Distribution

| Harm type | Records |
|---|---:|
| None | 116,325 |
| Payment fraud | 991 |
| Chargeback | 909 |
| Account abuse | 670 |
| Policy abuse | 495 |

| Severity | Records |
|---|---:|
| None | 116,325 |
| Low | 1,871 |
| Medium | 1,070 |
| High | 124 |

## Missing Values

| Column | Missing | Interpretation |
|---|---:|---|
| `children` | 4 | Source-data omission |
| `country` | 488 | Source-data omission |
| `agent` | 16,340 | Source uses null when no travel agent applies |
| `company` | 112,593 | Source uses null when no company applies |
| `email_age_days` | 4,160 | Simulated unavailable identity signal |
| `device_trust_score` | 6,598 | Simulated unavailable device signal |

## Point-in-Time Controls

- Rows are ordered by reconstructed booking timestamp.
- Shared-device account counts use only the trailing 30 days at scoring time.
- Prior harmful-booking and chargeback counts include only outcomes that matured before the current booking.
- Booking identifiers and entity identifiers are retained for history, splitting, and monitoring but excluded from the first model.
- Cancellation, assigned room, booking changes, waiting-list duration, reservation status, harm details, and outcome-maturity time are excluded from model features.

## Reproducibility

- Random seed: 42
- Requested harm rate: 2.50%
- Realized harm rate: 2.57%
- Builder: `src/marketplace_risk/build_hybrid_dataset.py`
- Contract tests: `tests/test_hybrid_dataset.py`
