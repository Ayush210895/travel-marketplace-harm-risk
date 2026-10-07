"""Fit the locked operating candidate and evaluate the untouched final period."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from marketplace_risk.modeling import build_xgboost_pipeline, capacity_metrics
from marketplace_risk.schema import TARGET_COLUMN, model_feature_columns


def run_final_evaluation(
    data_path: Path,
    manifest_path: Path,
    development_metrics_path: Path,
    output_path: Path,
) -> dict[str, object]:
    data = pd.read_csv(
        data_path, parse_dates=["booking_timestamp", "outcome_matured_at"]
    ).sort_values("booking_timestamp", kind="stable").reset_index(drop=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    development = json.loads(development_metrics_path.read_text(encoding="utf-8"))
    test_start = pd.Timestamp(manifest["test_start"])

    train = data[
        (data["booking_timestamp"] < test_start)
        & (data["outcome_matured_at"] <= test_start)
    ].copy()
    test = data[data["booking_timestamp"] >= test_start].copy()
    columns = model_feature_columns(data.columns.tolist())

    model = build_xgboost_pipeline(train[columns]).set_params(
        **development["xgboost_best_parameters"]
    )
    model.fit(train[columns], train[TARGET_COLUMN].to_numpy())
    probability = model.predict_proba(test[columns])[:, 1]
    engineered = model.named_steps["features"].transform(test[columns])
    expected_loss_score = (
        probability * engineered["estimated_booking_value"].to_numpy()
    )
    probability_metrics = capacity_metrics(
        test[TARGET_COLUMN].to_numpy(),
        probability,
        test["harm_amount"].to_numpy(),
    )
    expected_loss_metrics = capacity_metrics(
        test[TARGET_COLUMN].to_numpy(),
        expected_loss_score,
        test["harm_amount"].to_numpy(),
        probability_scores=False,
    )
    results: dict[str, object] = {
        "policy_locked_before_test": True,
        "model": "xgboost",
        "calibration": "uncalibrated",
        "ranking_score": "predicted_harm_probability * estimated_booking_value",
        "review_fraction": 0.05,
        "training_rows": len(train),
        "test_rows": len(test),
        "test_start": test_start.isoformat(),
        "probability_ranking": probability_metrics,
        "expected_loss_ranking": expected_loss_metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    joblib.dump(model, output_path.with_suffix(".joblib"))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_final_evaluation(
        args.data,
        args.manifest,
        args.development_metrics,
        args.output,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
