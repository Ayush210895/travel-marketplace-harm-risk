"""Dataset contracts shared by data preparation and modeling code."""

TARGET_COLUMN = "is_harmful"

IDENTIFIER_COLUMNS = [
    "booking_id",
    "traveler_id",
    "property_id",
    "device_id",
    "payment_instrument_id",
    "agent",
    "company",
]

POST_BOOKING_COLUMNS = [
    "is_canceled",
    "assigned_room_type",
    "booking_changes",
    "days_in_waiting_list",
    "reservation_status",
    "reservation_status_date",
    "harm_type",
    "harm_amount",
    "severity",
    "outcome_matured_at",
]

SYNTHETIC_PREDICTION_COLUMNS = [
    "booking_id",
    "traveler_id",
    "property_id",
    "device_id",
    "payment_instrument_id",
    "booking_timestamp",
    "account_age_days",
    "email_age_days",
    "device_trust_score",
    "is_new_device",
    "traveler_country_matches_booking",
    "payment_method",
    "payment_attempts_24h",
    "bookings_24h",
    "accounts_on_device_30d",
    "prior_chargebacks",
    "prior_harmful_bookings",
    "is_mobile",
]

SYNTHETIC_OUTCOME_COLUMNS = [
    TARGET_COLUMN,
    "harm_type",
    "harm_amount",
    "severity",
    "outcome_matured_at",
]


def model_feature_columns(columns: list[str]) -> list[str]:
    """Return prediction-time columns, excluding identifiers and outcomes."""
    excluded = set(IDENTIFIER_COLUMNS + POST_BOOKING_COLUMNS + [TARGET_COLUMN])
    return [column for column in columns if column not in excluded]
