"""Leakage-safe point-in-time entity history features."""

from __future__ import annotations

from collections import Counter, defaultdict
import heapq

import numpy as np
import pandas as pd


CORE_ENTITY_COLUMNS = {
    "traveler": "traveler_id",
    "device": "device_id",
    "payment": "payment_instrument_id",
    "property": "property_id",
}

OPTIONAL_ENTITY_COLUMNS = {
    "agent": "agent",
    "company": "company",
}


def _smoothed_rate(
    positive_count: int,
    mature_count: int,
    global_rate: float,
    prior_strength: float,
) -> float:
    return (positive_count + prior_strength * global_rate) / (
        mature_count + prior_strength
    )


def add_point_in_time_entity_features(
    data: pd.DataFrame,
    prior_strength: float = 20.0,
    initial_harm_rate: float = 0.025,
) -> pd.DataFrame:
    """Create entity history using only activity known before each booking.

    An outcome contributes to history only after ``outcome_matured_at``. Raw
    booking counts are available immediately after a booking is created.
    """
    required = {
        "booking_timestamp",
        "outcome_matured_at",
        "is_harmful",
        "harm_type",
        *CORE_ENTITY_COLUMNS.values(),
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive")

    entity_columns = CORE_ENTITY_COLUMNS | {
        name: column
        for name, column in OPTIONAL_ENTITY_COLUMNS.items()
        if column in data.columns
    }

    result = data.sort_values("booking_timestamp", kind="stable").reset_index(drop=True).copy()
    row_count = len(result)

    booking_counts = {
        name: defaultdict(int) for name in entity_columns
    }
    mature_counts = {
        name: defaultdict(int) for name in entity_columns
    }
    harm_counts = {
        name: defaultdict(int) for name in entity_columns
    }
    chargeback_counts = {
        name: defaultdict(int) for name in entity_columns
    }
    last_booking_time: dict[str, pd.Timestamp] = {}
    payment_travelers: defaultdict[str, Counter[str]] = defaultdict(Counter)
    pending: list[tuple[int, int, tuple[str | None, ...], int, int]] = []
    global_mature = 0
    global_harm = 0

    feature_values: dict[str, np.ndarray] = {}
    for entity_name in entity_columns:
        feature_values[f"{entity_name}_prior_bookings"] = np.zeros(
            row_count, dtype=np.int32
        )
        feature_values[f"{entity_name}_mature_outcomes"] = np.zeros(
            row_count, dtype=np.int32
        )
        feature_values[f"{entity_name}_smoothed_harm_rate"] = np.zeros(
            row_count, dtype=float
        )
        feature_values[f"{entity_name}_prior_chargebacks"] = np.zeros(
            row_count, dtype=np.int16
        )
        feature_values[f"is_new_{entity_name}"] = np.zeros(row_count, dtype=np.int8)

    traveler_days_since_last = np.full(row_count, np.nan)
    payment_distinct_travelers = np.zeros(row_count, dtype=np.int16)

    for index, row in result.iterrows():
        booking_time = row["booking_timestamp"]
        booking_nanoseconds = int(booking_time.value)

        while pending and pending[0][0] <= booking_nanoseconds:
            _, _, matured_entities, matured_harm, matured_chargeback = heapq.heappop(
                pending
            )
            global_mature += 1
            global_harm += matured_harm
            for entity_name, entity_id in zip(
                entity_columns, matured_entities, strict=True
            ):
                if entity_id is None:
                    continue
                mature_counts[entity_name][entity_id] += 1
                harm_counts[entity_name][entity_id] += matured_harm
                chargeback_counts[entity_name][entity_id] += matured_chargeback

        global_rate = (
            global_harm + prior_strength * initial_harm_rate
        ) / (
            global_mature + prior_strength
        )
        entities = tuple(
            None if pd.isna(row[column]) else str(row[column])
            for column in entity_columns.values()
        )

        for entity_name, entity_id in zip(entity_columns, entities, strict=True):
            if entity_id is None:
                feature_values[f"{entity_name}_smoothed_harm_rate"][index] = global_rate
                continue
            prior_bookings = booking_counts[entity_name][entity_id]
            mature_outcomes = mature_counts[entity_name][entity_id]
            feature_values[f"{entity_name}_prior_bookings"][index] = prior_bookings
            feature_values[f"{entity_name}_mature_outcomes"][index] = mature_outcomes
            feature_values[f"{entity_name}_prior_chargebacks"][index] = min(
                chargeback_counts[entity_name][entity_id], 32767
            )
            feature_values[f"{entity_name}_smoothed_harm_rate"][index] = _smoothed_rate(
                harm_counts[entity_name][entity_id],
                mature_outcomes,
                global_rate,
                prior_strength,
            )
            feature_values[f"is_new_{entity_name}"][index] = int(prior_bookings == 0)

        traveler_id = entities[0]
        assert traveler_id is not None
        if traveler_id in last_booking_time:
            traveler_days_since_last[index] = (
                booking_time - last_booking_time[traveler_id]
            ).total_seconds() / 86400.0

        payment_id = entities[2]
        assert payment_id is not None
        payment_distinct_travelers[index] = min(
            len(payment_travelers[payment_id]) + int(traveler_id not in payment_travelers[payment_id]),
            32767,
        )

        for entity_name, entity_id in zip(entity_columns, entities, strict=True):
            if entity_id is None:
                continue
            booking_counts[entity_name][entity_id] += 1
        last_booking_time[traveler_id] = booking_time
        payment_travelers[payment_id][traveler_id] += 1

        heapq.heappush(
            pending,
            (
                int(row["outcome_matured_at"].value),
                index,
                entities,
                int(row["is_harmful"]),
                int(row["harm_type"] == "chargeback"),
            ),
        )

    for feature_name, values in feature_values.items():
        result[feature_name] = values
    result["traveler_days_since_last_booking"] = traveler_days_since_last
    result["payment_distinct_travelers"] = payment_distinct_travelers
    for entity_name in OPTIONAL_ENTITY_COLUMNS:
        if entity_name in entity_columns:
            result[f"has_{entity_name}"] = result[
                entity_columns[entity_name]
            ].notna().astype(np.int8)
    return result
