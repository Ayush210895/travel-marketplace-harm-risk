"""Score booking batches and create a capacity-constrained review queue."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from marketplace_risk.schema import model_feature_columns


PASSTHROUGH_COLUMNS = [
    "booking_id",
    "booking_timestamp",
    "country",
    "is_harmful",
    "harm_amount",
    "outcome_matured_at",
    "is_new_traveler",
    "is_new_device",
    "is_new_payment",
]


def rank_review_queue(
    probability: np.ndarray,
    estimated_booking_value: np.ndarray,
    review_fraction: float = 0.05,
) -> pd.DataFrame:
    if len(probability) != len(estimated_booking_value) or len(probability) == 0:
        raise ValueError("Probability and value arrays must have equal non-zero length")
    if not 0 < review_fraction <= 1:
        raise ValueError("review_fraction must be in (0, 1]")
    expected_loss = np.asarray(probability) * np.asarray(estimated_booking_value)
    order = np.argsort(-expected_loss, kind="stable")
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(1, len(order) + 1)
    review_count = max(1, math.ceil(len(order) * review_fraction))
    return pd.DataFrame(
        {
            "harm_probability": probability,
            "estimated_booking_value": estimated_booking_value,
            "expected_loss_score": expected_loss,
            "review_rank": rank,
            "send_to_review": (rank <= review_count).astype("int8"),
        }
    )


def score_bookings(
    model_path: Path,
    input_path: Path,
    output_path: Path,
    review_fraction: float = 0.05,
) -> pd.DataFrame:
    model = joblib.load(model_path)
    parse_dates = [
        column
        for column in ("booking_timestamp", "outcome_matured_at")
        if column in pd.read_csv(input_path, nrows=0).columns
    ]
    data = pd.read_csv(input_path, parse_dates=parse_dates)
    feature_columns = model_feature_columns(data.columns.tolist())
    features = data[feature_columns]
    probability = model.predict_proba(features)[:, 1]
    engineered = model.named_steps["features"].transform(features)
    queue = rank_review_queue(
        probability,
        engineered["estimated_booking_value"].to_numpy(),
        review_fraction,
    )
    output = data[
        [column for column in PASSTHROUGH_COLUMNS if column in data]
    ].reset_index(drop=True)
    output["country_group"] = engineered["country"].to_numpy()
    output = pd.concat([output, queue], axis=1)
    output = output.sort_values("review_rank", kind="stable").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-fraction", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = score_bookings(
        args.model, args.input, args.output, args.review_fraction
    )
    print(f"Scored {len(output):,} bookings")
    print(f"Queued {int(output['send_to_review'].sum()):,} bookings for review")


if __name__ == "__main__":
    main()
