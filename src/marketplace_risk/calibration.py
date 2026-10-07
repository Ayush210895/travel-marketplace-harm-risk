"""Evaluate probability calibration before opening the final test period."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from marketplace_risk.modeling import capacity_metrics
from marketplace_risk.schema import model_feature_columns


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def expected_calibration_error(
    labels: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> float:
    """Calculate equal-frequency expected calibration error."""
    frame = pd.DataFrame({"label": labels, "probability": probability})
    frame["bin"] = pd.qcut(
        frame["probability"], q=bins, labels=False, duplicates="drop"
    )
    grouped = frame.groupby("bin", observed=True).agg(
        rows=("label", "size"),
        observed=("label", "mean"),
        predicted=("probability", "mean"),
    )
    return float(
        (
            grouped["rows"]
            / len(frame)
            * (grouped["observed"] - grouped["predicted"]).abs()
        ).sum()
    )


def calibration_metrics(
    labels: np.ndarray,
    probability: np.ndarray,
    estimated_value: np.ndarray,
    harm_amount: np.ndarray,
) -> dict[str, float | int]:
    probability_metrics = {
        "brier_score": float(brier_score_loss(labels, probability)),
        "log_loss": float(log_loss(labels, probability)),
        "expected_calibration_error": expected_calibration_error(
            labels, probability
        ),
    }
    ranking = capacity_metrics(
        labels,
        probability * estimated_value,
        harm_amount,
        probability_scores=False,
    )
    return probability_metrics | {
        "precision_at_5pct": ranking["precision_at_5pct"],
        "recall_at_5pct": ranking["recall_at_5pct"],
        "harm_value_capture_at_5pct": ranking["harm_value_capture_at_5pct"],
    }


def run_calibration(
    data_path: Path,
    manifest_path: Path,
    model_path: Path,
    output_path: Path,
) -> dict[str, object]:
    data = pd.read_csv(
        data_path, parse_dates=["booking_timestamp", "outcome_matured_at"]
    ).sort_values("booking_timestamp", kind="stable").reset_index(drop=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_start = pd.Timestamp(manifest["validation_start"])
    test_start = pd.Timestamp(manifest["test_start"])
    validation = data[
        (data["booking_timestamp"] >= validation_start)
        & (data["booking_timestamp"] < test_start)
    ].copy()
    split_index = len(validation) // 2
    calibration = validation.iloc[:split_index].copy()
    evaluation = validation.iloc[split_index:].copy()
    columns = model_feature_columns(data.columns.tolist())
    model = joblib.load(model_path)

    calibration_probability = model.predict_proba(calibration[columns])[:, 1]
    evaluation_probability = model.predict_proba(evaluation[columns])[:, 1]
    calibration_labels = calibration["is_harmful"].to_numpy()
    evaluation_labels = evaluation["is_harmful"].to_numpy()

    platt = LogisticRegression().fit(
        _logit(calibration_probability).reshape(-1, 1), calibration_labels
    )
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(
        calibration_probability, calibration_labels
    )
    probabilities = {
        "uncalibrated": evaluation_probability,
        "platt": platt.predict_proba(
            _logit(evaluation_probability).reshape(-1, 1)
        )[:, 1],
        "isotonic": isotonic.predict(evaluation_probability),
    }
    engineered = model.named_steps["features"].transform(evaluation[columns])
    estimated_value = engineered["estimated_booking_value"].to_numpy()
    results: dict[str, object] = {
        "calibration_rows": len(calibration),
        "evaluation_rows": len(evaluation),
        "evaluation_harm_rate": float(evaluation_labels.mean()),
        "methods": {},
    }
    for name, probability in probabilities.items():
        results["methods"][name] = calibration_metrics(
            evaluation_labels,
            probability,
            estimated_value,
            evaluation["harm_amount"].to_numpy(),
        )

    baseline = results["methods"]["uncalibrated"]
    eligible_methods = [
        name
        for name in ("platt", "isotonic")
        if results["methods"][name]["brier_score"] <= baseline["brier_score"] * 0.995
        and results["methods"][name]["log_loss"] <= baseline["log_loss"]
        and results["methods"][name]["expected_calibration_error"]
        <= baseline["expected_calibration_error"]
        and results["methods"][name]["harm_value_capture_at_5pct"]
        >= baseline["harm_value_capture_at_5pct"] * 0.99
    ]
    selected_method = (
        min(
            eligible_methods,
            key=lambda name: results["methods"][name]["brier_score"],
        )
        if eligible_methods
        else "uncalibrated"
    )
    results["selected_method"] = selected_method

    full_probability = model.predict_proba(validation[columns])[:, 1]
    full_labels = validation["is_harmful"].to_numpy()
    if selected_method == "platt":
        final_calibrator = LogisticRegression().fit(
            _logit(full_probability).reshape(-1, 1), full_labels
        )
    elif selected_method == "isotonic":
        final_calibrator = IsotonicRegression(out_of_bounds="clip").fit(
            full_probability, full_labels
        )
    else:
        final_calibrator = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    joblib.dump(
        {"method": selected_method, "calibrator": final_calibrator},
        output_path.with_suffix(".joblib"),
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_calibration(
        args.data, args.manifest, args.model, args.output
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
