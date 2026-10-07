"""Point-in-time development splits for booking-risk modeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def create_split_assignments(
    data: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Create chronological splits and point-in-time label eligibility flags."""
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("Train and validation fractions must be positive.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave a test period.")

    required = {"booking_id", "booking_timestamp", "outcome_matured_at", "is_harmful"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    ordered = data.sort_values("booking_timestamp", kind="stable").reset_index(drop=True)
    row_count = len(ordered)
    validation_start_index = int(row_count * train_fraction)
    test_start_index = int(row_count * (train_fraction + validation_fraction))
    validation_start = ordered.loc[validation_start_index, "booking_timestamp"]
    test_start = ordered.loc[test_start_index, "booking_timestamp"]

    split = pd.Series("test", index=ordered.index, dtype="object")
    split.loc[: validation_start_index - 1] = "initial_train"
    split.loc[validation_start_index : test_start_index - 1] = "validation"

    assignments = ordered[["booking_id", "booking_timestamp", "outcome_matured_at"]].copy()
    assignments["development_split"] = split
    assignments["eligible_at_validation_train_time"] = (
        (ordered["booking_timestamp"] < validation_start)
        & (ordered["outcome_matured_at"] <= validation_start)
    )
    assignments["eligible_at_test_train_time"] = (
        (ordered["booking_timestamp"] < test_start)
        & (ordered["outcome_matured_at"] <= test_start)
    )

    split_summary = (
        ordered.assign(development_split=split)
        .groupby("development_split", sort=False)
        .agg(
            rows=("is_harmful", "size"),
            harm_rate=("is_harmful", "mean"),
            start=("booking_timestamp", "min"),
            end=("booking_timestamp", "max"),
        )
    )

    manifest: dict[str, object] = {
        "rows": row_count,
        "train_fraction": train_fraction,
        "validation_fraction": validation_fraction,
        "test_fraction": 1 - train_fraction - validation_fraction,
        "validation_start": validation_start.isoformat(),
        "test_start": test_start.isoformat(),
        "initial_train_rows": int((split == "initial_train").sum()),
        "validation_rows": int((split == "validation").sum()),
        "test_rows": int((split == "test").sum()),
        "eligible_training_rows_at_validation": int(
            assignments["eligible_at_validation_train_time"].sum()
        ),
        "excluded_immature_rows_at_validation": int(
            ((split == "initial_train") & ~assignments["eligible_at_validation_train_time"]).sum()
        ),
        "eligible_training_rows_at_test": int(
            assignments["eligible_at_test_train_time"].sum()
        ),
        "excluded_immature_rows_at_test": int(
            (
                (ordered["booking_timestamp"] < test_start)
                & ~assignments["eligible_at_test_train_time"]
            ).sum()
        ),
        "split_harm_rates": {
            name: float(row["harm_rate"]) for name, row in split_summary.iterrows()
        },
    }
    return assignments, manifest


def write_split_artifacts(
    data_path: Path,
    assignments_path: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    data = pd.read_csv(
        data_path, parse_dates=["booking_timestamp", "outcome_matured_at"]
    )
    assignments, manifest = create_split_assignments(data)
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(assignments_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return assignments, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, manifest = write_split_artifacts(args.data, args.assignments, args.manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
