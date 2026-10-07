"""Training-fitted preprocessing for booking-risk model features."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


NOMINAL_COLUMNS = [
    "hotel",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "payment_method",
]

UNKNOWN_TOKENS = {"", "undefined", "unknown", "null", "none", "n/a", "na"}

# The source mostly uses ISO alpha-3 codes but contains these legacy/mixed forms.
COUNTRY_ALIASES = {
    "CN": "CHN",
    "TMP": "TLS",
}

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


def _clean_value(value: object, column: str) -> str:
    if pd.isna(value):
        return "Unknown"
    cleaned = str(value).strip()
    if cleaned.casefold() in UNKNOWN_TOKENS:
        return "Unknown"
    if column == "country":
        cleaned = cleaned.upper()
        return COUNTRY_ALIASES.get(cleaned, cleaned)
    return cleaned


class RareCategoryCleaner(TransformerMixin, BaseEstimator):
    """Clean nominal values and group sparse categories using training counts."""

    def __init__(
        self,
        columns: list[str] | None = None,
        min_frequency: int = 50,
    ) -> None:
        self.columns = columns
        self.min_frequency = min_frequency

    def fit(self, data: pd.DataFrame, y: object = None) -> "RareCategoryCleaner":
        if self.min_frequency <= 0:
            raise ValueError("min_frequency must be positive")
        columns = self.columns if self.columns is not None else NOMINAL_COLUMNS
        missing = set(columns) - set(data.columns)
        if missing:
            raise ValueError(f"Missing categorical columns: {sorted(missing)}")

        self.columns_ = list(columns)
        self.retained_categories_: dict[str, set[str]] = {}
        for column in self.columns_:
            cleaned = data[column].map(lambda value: _clean_value(value, column))
            counts = Counter(cleaned)
            retained = {
                value
                for value, count in counts.items()
                if count >= self.min_frequency
            }
            retained.add("Unknown")
            self.retained_categories_[column] = retained
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "retained_categories_"):
            raise ValueError("RareCategoryCleaner must be fitted before transform")
        missing = set(self.columns_) - set(data.columns)
        if missing:
            raise ValueError(f"Missing categorical columns: {sorted(missing)}")

        result = data.copy()
        for column in self.columns_:
            retained = self.retained_categories_[column]
            cleaned = result[column].map(lambda value: _clean_value(value, column))
            result[column] = cleaned.where(cleaned.isin(retained), "Other")
        return result


class AdrOutlierCapper(TransformerMixin, BaseEstimator):
    """Cap extreme ADR values using a threshold learned from training data."""

    def __init__(self, quantile: float = 0.999) -> None:
        self.quantile = quantile

    def fit(self, data: pd.DataFrame, y: object = None) -> "AdrOutlierCapper":
        if not 0 < self.quantile < 1:
            raise ValueError("quantile must be between zero and one")
        if "adr" not in data.columns:
            raise ValueError("Missing required column: adr")
        self.adr_cap_ = float(data["adr"].quantile(self.quantile))
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "adr_cap_"):
            raise ValueError("AdrOutlierCapper must be fitted before transform")
        if "adr" not in data.columns:
            raise ValueError("Missing required column: adr")

        result = data.copy()
        result["adr_negative"] = result["adr"].lt(0).astype("int8")
        result["adr_extreme"] = result["adr"].gt(self.adr_cap_).astype("int8")
        result["adr"] = result["adr"].clip(lower=0, upper=self.adr_cap_)
        return result


class GuestCountFeatures(TransformerMixin, BaseEstimator):
    """Create party-size and data-quality features available at booking time."""

    REQUIRED_COLUMNS = ["adults", "children", "babies"]

    def fit(self, data: pd.DataFrame, y: object = None) -> "GuestCountFeatures":
        missing = set(self.REQUIRED_COLUMNS) - set(data.columns)
        if missing:
            raise ValueError(f"Missing guest-count columns: {sorted(missing)}")
        self.is_fitted_ = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.REQUIRED_COLUMNS) - set(data.columns)
        if missing:
            raise ValueError(f"Missing guest-count columns: {sorted(missing)}")

        result = data.copy()
        result["children_missing"] = result["children"].isna().astype("int8")
        result["total_guests"] = result[self.REQUIRED_COLUMNS].fillna(0).sum(axis=1)
        result["invalid_guest_count"] = result["total_guests"].le(0).astype("int8")
        return result


class TemporalFeatures(TransformerMixin, BaseEstimator):
    """Create cyclical booking and arrival features known at booking time."""

    REQUIRED_COLUMNS = [
        "booking_timestamp",
        "arrival_date_year",
        "arrival_date_month",
        "arrival_date_day_of_month",
    ]
    RAW_COLUMNS_TO_DROP = [
        "booking_timestamp",
        "arrival_date_year",
        "arrival_date_month",
        "arrival_date_week_number",
        "arrival_date_day_of_month",
    ]

    def fit(self, data: pd.DataFrame, y: object = None) -> "TemporalFeatures":
        missing = set(self.REQUIRED_COLUMNS) - set(data.columns)
        if missing:
            raise ValueError(f"Missing temporal columns: {sorted(missing)}")
        self.is_fitted_ = True
        return self

    @staticmethod
    def _cyclical(values: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
        angle = 2 * np.pi * values / period
        return np.sin(angle), np.cos(angle)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.REQUIRED_COLUMNS) - set(data.columns)
        if missing:
            raise ValueError(f"Missing temporal columns: {sorted(missing)}")

        result = data.copy()
        booking_time = pd.to_datetime(result["booking_timestamp"])
        arrival_month = result["arrival_date_month"].map(MONTH_NUMBER)
        if arrival_month.isna().any():
            invalid = sorted(result.loc[arrival_month.isna(), "arrival_date_month"].unique())
            raise ValueError(f"Unknown arrival months: {invalid}")
        arrival_date = pd.to_datetime(
            {
                "year": result["arrival_date_year"],
                "month": arrival_month,
                "day": result["arrival_date_day_of_month"],
            }
        )

        result["booking_hour_sin"], result["booking_hour_cos"] = self._cyclical(
            booking_time.dt.hour, 24
        )
        result["booking_day_of_week_sin"], result["booking_day_of_week_cos"] = (
            self._cyclical(booking_time.dt.dayofweek, 7)
        )
        result["booking_month_sin"], result["booking_month_cos"] = self._cyclical(
            booking_time.dt.month - 1, 12
        )
        result["arrival_day_of_week_sin"], result["arrival_day_of_week_cos"] = (
            self._cyclical(arrival_date.dt.dayofweek, 7)
        )
        result["arrival_month_sin"], result["arrival_month_cos"] = self._cyclical(
            arrival_month - 1, 12
        )
        result["is_weekend_booking"] = booking_time.dt.dayofweek.ge(5).astype("int8")
        result["is_weekend_arrival"] = arrival_date.dt.dayofweek.ge(5).astype("int8")

        return result.drop(
            columns=[column for column in self.RAW_COLUMNS_TO_DROP if column in result]
        )


class BookingValueFeatures(TransformerMixin, BaseEstimator):
    """Create stay duration and approximate booking-value exposure."""

    REQUIRED_COLUMNS = ["adr", "stays_in_week_nights", "stays_in_weekend_nights"]

    def fit(self, data: pd.DataFrame, y: object = None) -> "BookingValueFeatures":
        missing = set(self.REQUIRED_COLUMNS) - set(data.columns)
        if missing:
            raise ValueError(f"Missing booking-value columns: {sorted(missing)}")
        self.is_fitted_ = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.REQUIRED_COLUMNS) - set(data.columns)
        if missing:
            raise ValueError(f"Missing booking-value columns: {sorted(missing)}")

        result = data.copy()
        result["total_nights"] = (
            result["stays_in_week_nights"] + result["stays_in_weekend_nights"]
        )
        result["estimated_booking_value"] = (
            result["adr"] * result["total_nights"].clip(lower=1)
        )
        result["log_estimated_booking_value"] = np.log1p(
            result["estimated_booking_value"]
        )
        return result


class MissingSignalFeatures(TransformerMixin, BaseEstimator):
    """Preserve prediction-time missingness before model-specific imputation."""

    COLUMNS = ["email_age_days", "device_trust_score"]

    def fit(self, data: pd.DataFrame, y: object = None) -> "MissingSignalFeatures":
        missing = set(self.COLUMNS) - set(data.columns)
        if missing:
            raise ValueError(f"Missing signal columns: {sorted(missing)}")
        self.is_fitted_ = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.COLUMNS) - set(data.columns)
        if missing:
            raise ValueError(f"Missing signal columns: {sorted(missing)}")
        result = data.copy()
        for column in self.COLUMNS:
            result[f"{column.removesuffix('_days').removesuffix('_score')}_missing"] = (
                result[column].isna().astype("int8")
            )
        return result
