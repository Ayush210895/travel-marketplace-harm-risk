"""Build a travel-specific hybrid booking-risk dataset.

The booking context comes from the public Hotel Booking Demand dataset. Identity,
device, payment, entity, and harmful-outcome fields are simulated. The builder is
deterministic so experiments and tests remain reproducible.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import heapq
from pathlib import Path

import numpy as np
import pandas as pd

from marketplace_risk.historical_features import add_point_in_time_entity_features


MONTH_NUMBER = {
    month: number
    for number, month in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def _calibrate_intercept(score: np.ndarray, target_rate: float) -> float:
    """Find an intercept whose mean event probability matches target_rate."""
    low, high = -20.0, 5.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if _sigmoid(score + midpoint).mean() > target_rate:
            high = midpoint
        else:
            low = midpoint
    return (low + high) / 2.0


def _build_booking_time(data: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    arrival_date = pd.to_datetime(
        {
            "year": data["arrival_date_year"],
            "month": data["arrival_date_month"].map(MONTH_NUMBER),
            "day": data["arrival_date_day_of_month"],
        }
    )
    booking_date = arrival_date - pd.to_timedelta(data["lead_time"], unit="D")
    seconds = rng.integers(0, 24 * 60 * 60, size=len(data))
    return booking_date + pd.to_timedelta(seconds, unit="s")


def _assign_travelers(
    repeated_guest: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    traveler_ids: list[str] = []
    existing: list[str] = []
    next_id = 1

    for is_repeated in repeated_guest:
        reuse_probability = 0.92 if is_repeated else 0.08
        if existing and rng.random() < reuse_probability:
            traveler_id = existing[int(rng.integers(0, len(existing)))]
        else:
            traveler_id = f"T{next_id:07d}"
            next_id += 1
            existing.append(traveler_id)
        traveler_ids.append(traveler_id)
    return np.asarray(traveler_ids)


def _entity_lookup(
    entity_ids: np.ndarray, rng: np.random.Generator, distribution: str
) -> dict[str, float]:
    unique_ids = np.unique(entity_ids)
    if distribution == "traveler":
        values = rng.beta(1.4, 18.0, size=len(unique_ids))
    elif distribution == "property":
        values = rng.beta(1.2, 35.0, size=len(unique_ids))
    elif distribution == "device":
        values = rng.beta(1.1, 24.0, size=len(unique_ids))
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    return dict(zip(unique_ids, values, strict=True))


def _point_in_time_device_accounts(
    booking_times: pd.Series,
    device_ids: np.ndarray,
    traveler_ids: np.ndarray,
    window_days: int = 30,
) -> np.ndarray:
    """Count distinct accounts seen on a device in the trailing time window."""
    result = np.ones(len(device_ids), dtype=np.int16)
    windows: defaultdict[str, deque[tuple[pd.Timestamp, str]]] = defaultdict(deque)
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    window = pd.Timedelta(days=window_days)

    for index, (booking_time, device_id, traveler_id) in enumerate(
        zip(booking_times, device_ids, traveler_ids, strict=True)
    ):
        cutoff = booking_time - window
        device_window = windows[device_id]
        device_counts = counts[device_id]
        while device_window and device_window[0][0] < cutoff:
            _, expired_traveler = device_window.popleft()
            device_counts[expired_traveler] -= 1
            if device_counts[expired_traveler] == 0:
                del device_counts[expired_traveler]

        device_window.append((booking_time, traveler_id))
        device_counts[traveler_id] += 1
        result[index] = min(len(device_counts), 32767)

    return result


def _point_in_time_outcome_history(
    booking_times: pd.Series,
    outcome_times: pd.Series,
    traveler_ids: np.ndarray,
    labels: np.ndarray,
    chargebacks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Count only outcomes that matured before the current booking timestamp."""
    prior_harm = np.zeros(len(labels), dtype=np.int16)
    prior_chargeback = np.zeros(len(labels), dtype=np.int16)
    harm_counts: defaultdict[str, int] = defaultdict(int)
    chargeback_counts: defaultdict[str, int] = defaultdict(int)
    pending_outcomes: list[tuple[int, int, str, int, int]] = []

    for index, (booking_time, outcome_time, traveler_id) in enumerate(
        zip(booking_times, outcome_times, traveler_ids, strict=True)
    ):
        booking_nanoseconds = int(booking_time.value)
        while pending_outcomes and pending_outcomes[0][0] <= booking_nanoseconds:
            _, _, matured_traveler, matured_harm, matured_chargeback = heapq.heappop(
                pending_outcomes
            )
            harm_counts[matured_traveler] += matured_harm
            chargeback_counts[matured_traveler] += matured_chargeback

        prior_harm[index] = min(harm_counts[traveler_id], 32767)
        prior_chargeback[index] = min(chargeback_counts[traveler_id], 32767)
        heapq.heappush(
            pending_outcomes,
            (
                int(outcome_time.value),
                index,
                traveler_id,
                int(labels[index]),
                int(chargebacks[index]),
            ),
        )

    return prior_harm, prior_chargeback


def build_hybrid_dataset(
    source_path: Path,
    output_path: Path,
    seed: int = 42,
    target_harm_rate: float = 0.025,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = pd.read_csv(source_path)
    data["booking_timestamp"] = _build_booking_time(data, rng)
    data = data.sort_values("booking_timestamp", kind="stable").reset_index(drop=True)

    row_count = len(data)
    data["booking_id"] = [f"B{number:08d}" for number in range(1, row_count + 1)]
    data["traveler_id"] = _assign_travelers(
        data["is_repeated_guest"].to_numpy(), rng
    )

    hotel_property_counts = {"City Hotel": 260, "Resort Hotel": 90}
    property_numbers = np.empty(row_count, dtype=int)
    for hotel_name, property_count in hotel_property_counts.items():
        mask = data["hotel"].eq(hotel_name).to_numpy()
        popularity = rng.zipf(1.8, size=mask.sum())
        property_numbers[mask] = ((popularity - 1) % property_count) + 1
    hotel_prefix = data["hotel"].map({"City Hotel": "C", "Resort Hotel": "R"})
    data["property_id"] = [
        f"P{prefix}{number:04d}"
        for prefix, number in zip(hotel_prefix, property_numbers, strict=True)
    ]

    traveler_codes, unique_travelers = pd.factorize(data["traveler_id"], sort=False)
    traveler_count = len(unique_travelers)
    primary_device = rng.integers(1, max(2, int(traveler_count * 0.82)), traveler_count)
    primary_payment = rng.integers(1, max(2, int(traveler_count * 0.88)), traveler_count)

    is_new_device = rng.binomial(
        1,
        np.clip(
            0.08
            + 0.20 * (data["is_repeated_guest"].to_numpy() == 0)
            + 0.08 * (data["lead_time"].to_numpy() <= 2),
            0,
            0.70,
        ),
    )
    device_numbers = primary_device[traveler_codes].copy()
    new_device_mask = is_new_device.astype(bool)
    device_numbers[new_device_mask] = rng.integers(
        max(2, int(traveler_count * 0.82)),
        max(3, int(traveler_count * 1.12)),
        new_device_mask.sum(),
    )
    data["device_id"] = [f"D{number:07d}" for number in device_numbers]
    data["is_new_device"] = is_new_device

    payment_numbers = primary_payment[traveler_codes].copy()
    changed_payment = rng.random(row_count) < (0.06 + 0.12 * is_new_device)
    payment_numbers[changed_payment] = rng.integers(
        max(2, int(traveler_count * 0.88)),
        max(3, int(traveler_count * 1.08)),
        changed_payment.sum(),
    )
    data["payment_instrument_id"] = [f"PM{number:07d}" for number in payment_numbers]

    traveler_age_base = rng.gamma(2.2, 310.0, traveler_count)
    account_age = traveler_age_base[traveler_codes] + rng.normal(0, 18, row_count)
    account_age -= 150 * is_new_device
    data["account_age_days"] = np.clip(account_age, 0, 3650).round().astype(int)

    email_age = data["account_age_days"].to_numpy() * rng.uniform(0.55, 1.8, row_count)
    email_age += rng.normal(0, 45, row_count)
    data["email_age_days"] = np.clip(email_age, 0, 5000).round()
    email_missing = rng.random(row_count) < 0.035
    data.loc[email_missing, "email_age_days"] = np.nan

    country_mismatch_probability = np.clip(
        0.035 + 0.12 * is_new_device + 0.05 * (data["lead_time"].to_numpy() <= 1),
        0,
        0.45,
    )
    data["traveler_country_matches_booking"] = 1 - rng.binomial(
        1, country_mismatch_probability
    )

    payment_methods = np.array(
        ["credit_card", "debit_card", "digital_wallet", "bank_transfer", "pay_later"]
    )
    payment_probabilities = np.array([0.54, 0.20, 0.14, 0.04, 0.08])
    data["payment_method"] = rng.choice(
        payment_methods, size=row_count, p=payment_probabilities
    )
    attempts_lambda = (
        0.15
        + 0.55 * is_new_device
        + 0.75 * (data["traveler_country_matches_booking"].to_numpy() == 0)
    )
    data["payment_attempts_24h"] = 1 + rng.poisson(attempts_lambda)

    data["is_mobile"] = rng.binomial(
        1,
        np.clip(
            0.58
            + 0.08 * data["market_segment"].eq("Online TA").to_numpy()
            - 0.10 * data["distribution_channel"].eq("Corporate").to_numpy(),
            0.15,
            0.90,
        ),
    )

    traveler_booking_pressure = rng.negative_binomial(1, 0.72, row_count)
    data["bookings_24h"] = np.clip(
        1
        + traveler_booking_pressure
        + (data["payment_attempts_24h"].to_numpy() >= 3).astype(int),
        1,
        20,
    )

    device_account_count = _point_in_time_device_accounts(
        data["booking_timestamp"],
        data["device_id"].to_numpy(),
        data["traveler_id"].to_numpy(),
    )
    data["accounts_on_device_30d"] = np.clip(device_account_count, 1, 12)

    traveler_risk_lookup = _entity_lookup(
        data["traveler_id"].to_numpy(), rng, "traveler"
    )
    property_risk_lookup = _entity_lookup(
        data["property_id"].to_numpy(), rng, "property"
    )
    device_risk_lookup = _entity_lookup(data["device_id"].to_numpy(), rng, "device")
    traveler_latent = data["traveler_id"].map(traveler_risk_lookup).to_numpy()
    property_latent = data["property_id"].map(property_risk_lookup).to_numpy()
    device_latent = data["device_id"].map(device_risk_lookup).to_numpy()

    device_trust = 0.92 - 1.9 * device_latent - 0.17 * is_new_device
    device_trust -= 0.12 * (data["accounts_on_device_30d"].to_numpy() >= 3)
    device_trust += rng.normal(0, 0.08, row_count)
    data["device_trust_score"] = np.clip(device_trust, 0, 1).round(4)
    trust_missing = rng.random(row_count) < 0.055
    data.loc[trust_missing, "device_trust_score"] = np.nan

    total_nights = (
        data["stays_in_week_nights"].to_numpy()
        + data["stays_in_weekend_nights"].to_numpy()
    )
    booking_amount = np.maximum(data["adr"].to_numpy(), 0) * np.maximum(total_nights, 1)
    amount_scaled = np.log1p(booking_amount) / np.log(2000)
    low_trust = data["device_trust_score"].fillna(0.50).to_numpy() < 0.35
    mismatch = data["traveler_country_matches_booking"].to_numpy() == 0
    high_velocity = data["bookings_24h"].to_numpy() >= 4
    many_attempts = data["payment_attempts_24h"].to_numpy() >= 3
    shared_device = data["accounts_on_device_30d"].to_numpy() >= 3
    new_account = data["account_age_days"].to_numpy() <= 21
    last_minute = data["lead_time"].to_numpy() <= 2

    risk_score = (
        5.0 * traveler_latent
        + 6.0 * device_latent
        + 3.0 * property_latent
        + 0.7 * amount_scaled
        + 0.9 * mismatch
        + 0.8 * many_attempts
        + 0.9 * high_velocity
        + 0.9 * shared_device
        + 0.6 * low_trust
        + 0.4 * last_minute
        + 0.3 * data["is_canceled"].to_numpy()
        + 1.5 * (new_account & is_new_device.astype(bool) & mismatch)
        + 1.2 * (low_trust & shared_device)
        + 0.9 * (many_attempts & (booking_amount >= 600))
        + rng.normal(0, 0.55, row_count)
    )
    intercept = _calibrate_intercept(risk_score, target_harm_rate)
    harm_probability = _sigmoid(risk_score + intercept)
    is_harmful = rng.binomial(1, harm_probability)
    data["is_harmful"] = is_harmful

    harm_indices = np.flatnonzero(is_harmful)
    harm_type = np.full(row_count, "none", dtype=object)
    type_probabilities = np.column_stack(
        [
            0.30 + 0.30 * many_attempts[harm_indices],
            0.25 + 0.25 * shared_device[harm_indices],
            0.25 + 0.20 * data["is_canceled"].to_numpy()[harm_indices],
            np.full(len(harm_indices), 0.20),
        ]
    )
    type_probabilities /= type_probabilities.sum(axis=1, keepdims=True)
    type_names = np.array(
        ["payment_fraud", "account_abuse", "chargeback", "policy_abuse"]
    )
    random_draw = rng.random(len(harm_indices))
    cumulative = type_probabilities.cumsum(axis=1)
    selected_type = (random_draw[:, None] > cumulative).sum(axis=1)
    harm_type[harm_indices] = type_names[selected_type]
    data["harm_type"] = harm_type

    chargeback = (harm_type == "chargeback") | (
        (is_harmful == 1) & (rng.random(row_count) < 0.18)
    )
    recovery_fraction = rng.beta(2.2, 2.6, row_count)
    data["harm_amount"] = np.where(
        is_harmful == 1,
        np.maximum(20, booking_amount * recovery_fraction),
        0,
    ).round(2)
    severity = np.full(row_count, "none", dtype=object)
    severity[(is_harmful == 1) & (data["harm_amount"].to_numpy() < 150)] = "low"
    severity[
        (is_harmful == 1)
        & (data["harm_amount"].to_numpy() >= 150)
        & (data["harm_amount"].to_numpy() < 600)
    ] = "medium"
    severity[(is_harmful == 1) & (data["harm_amount"].to_numpy() >= 600)] = "high"
    data["severity"] = severity

    maturity_days = np.where(
        is_harmful == 1,
        rng.integers(14, 121, row_count),
        rng.integers(30, 91, row_count),
    )
    data["outcome_matured_at"] = data["booking_timestamp"] + pd.to_timedelta(
        maturity_days, unit="D"
    )
    prior_harm, prior_chargeback = _point_in_time_outcome_history(
        data["booking_timestamp"],
        data["outcome_matured_at"],
        data["traveler_id"].to_numpy(),
        is_harmful,
        chargeback.astype(int),
    )
    data["prior_harmful_bookings"] = prior_harm
    data["prior_chargebacks"] = prior_chargeback

    data = add_point_in_time_entity_features(data, prior_strength=20.0)

    ordered_front = [
        "booking_id",
        "booking_timestamp",
        "traveler_id",
        "property_id",
        "device_id",
        "payment_instrument_id",
    ]
    remaining = [column for column in data.columns if column not in ordered_front]
    data = data[ordered_front + remaining]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-harm-rate", type=float, default=0.025)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = build_hybrid_dataset(
        source_path=args.source,
        output_path=args.output,
        seed=args.seed,
        target_harm_rate=args.target_harm_rate,
    )
    print(f"Wrote {len(data):,} bookings to {args.output}")
    print(f"Harm rate: {data['is_harmful'].mean():.4%}")


if __name__ == "__main__":
    main()
