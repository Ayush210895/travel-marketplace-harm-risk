"""Monitor review-queue performance, drift, and country-level errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    reference_count = np.histogram(reference, bins=edges)[0]
    current_count = np.histogram(current, bins=edges)[0]
    reference_share = np.clip(reference_count / len(reference), 1e-6, None)
    current_share = np.clip(current_count / len(current), 1e-6, None)
    return float(
        ((current_share - reference_share) * np.log(current_share / reference_share)).sum()
    )


def _performance(frame: pd.DataFrame) -> dict[str, float | int]:
    reviewed = frame["send_to_review"].eq(1)
    harmful = frame["is_harmful"].eq(1)
    caught = int((reviewed & harmful).sum())
    safe = ~harmful
    total_harm = float(frame.loc[harmful, "harm_amount"].sum())
    captured_harm = float(frame.loc[reviewed & harmful, "harm_amount"].sum())
    return {
        "rows": len(frame),
        "review_rate": float(reviewed.mean()),
        "harm_rate": float(harmful.mean()),
        "precision": caught / int(reviewed.sum()) if reviewed.any() else 0.0,
        "recall": caught / int(harmful.sum()) if harmful.any() else 0.0,
        "false_positive_rate": (
            int((reviewed & safe).sum()) / int(safe.sum()) if safe.any() else 0.0
        ),
        "harm_value_capture": captured_harm / total_harm if total_harm else 0.0,
    }


def monitoring_report(
    current: pd.DataFrame,
    reference: pd.DataFrame | None = None,
    minimum_segment_rows: int = 100,
) -> dict[str, object]:
    required = {
        "expected_loss_score",
        "send_to_review",
        "is_harmful",
        "harm_amount",
    }
    missing = required - set(current.columns)
    if missing:
        raise ValueError(f"Missing monitoring columns: {sorted(missing)}")

    report: dict[str, object] = {
        "overall": _performance(current),
        "score_distribution": {
            "mean": float(current["expected_loss_score"].mean()),
            "median": float(current["expected_loss_score"].median()),
            "p95": float(current["expected_loss_score"].quantile(0.95)),
            "p99": float(current["expected_loss_score"].quantile(0.99)),
        },
    }
    if reference is not None:
        report["expected_loss_score_psi"] = population_stability_index(
            reference["expected_loss_score"].to_numpy(),
            current["expected_loss_score"].to_numpy(),
        )

    if "country_group" in current:
        country_rows = []
        for country, frame in current.groupby("country_group", dropna=False):
            if len(frame) >= minimum_segment_rows:
                country_rows.append({"country_group": str(country)} | _performance(frame))
        report["country_segments"] = sorted(
            country_rows, key=lambda row: row["rows"], reverse=True
        )

    for column in ("is_new_traveler", "is_new_device", "is_new_payment"):
        if column in current:
            report[f"segments_by_{column}"] = [
                {column: int(value)} | _performance(frame)
                for value, frame in current.groupby(column)
            ]
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-segment-rows", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    current = pd.read_csv(args.current)
    reference = pd.read_csv(args.reference) if args.reference else None
    report = monitoring_report(current, reference, args.minimum_segment_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
