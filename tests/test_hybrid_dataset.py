from pathlib import Path
import tempfile
import unittest

import pandas as pd

from marketplace_risk.build_hybrid_dataset import (
    _point_in_time_device_accounts,
    _point_in_time_outcome_history,
    build_hybrid_dataset,
)
from marketplace_risk.schema import POST_BOOKING_COLUMNS, model_feature_columns


class HybridDatasetTests(unittest.TestCase):
    def test_builder_preserves_rows_and_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            output = Path(directory) / "hybrid.csv"
            sample = pd.DataFrame(
                {
                    "hotel": ["City Hotel", "Resort Hotel", "City Hotel", "Resort Hotel"],
                    "is_canceled": [0, 1, 0, 1],
                    "lead_time": [14, 1, 90, 2],
                    "arrival_date_year": [2016] * 4,
                    "arrival_date_month": ["January"] * 4,
                    "arrival_date_week_number": [1] * 4,
                    "arrival_date_day_of_month": [20, 21, 22, 23],
                    "stays_in_weekend_nights": [1, 0, 2, 1],
                    "stays_in_week_nights": [2, 1, 3, 2],
                    "adults": [2, 1, 2, 2],
                    "children": [0.0, 0.0, 1.0, 0.0],
                    "babies": [0, 0, 0, 0],
                    "meal": ["BB"] * 4,
                    "country": ["PRT", "GBR", "FRA", "ESP"],
                    "market_segment": ["Online TA", "Direct", "Corporate", "Online TA"],
                    "distribution_channel": ["TA/TO", "Direct", "Corporate", "TA/TO"],
                    "is_repeated_guest": [0, 0, 1, 0],
                    "previous_cancellations": [0, 1, 0, 0],
                    "previous_bookings_not_canceled": [0, 0, 2, 0],
                    "reserved_room_type": ["A", "A", "D", "E"],
                    "assigned_room_type": ["A", "D", "D", "E"],
                    "booking_changes": [0, 1, 0, 2],
                    "deposit_type": ["No Deposit"] * 4,
                    "agent": [9.0, None, 14.0, 9.0],
                    "company": [None] * 4,
                    "days_in_waiting_list": [0] * 4,
                    "customer_type": ["Transient"] * 4,
                    "adr": [100.0, 600.0, 120.0, 450.0],
                    "required_car_parking_spaces": [0, 0, 1, 0],
                    "total_of_special_requests": [1, 0, 2, 0],
                    "reservation_status": ["Check-Out", "Canceled", "Check-Out", "Canceled"],
                    "reservation_status_date": ["2016-01-20"] * 4,
                }
            )
            sample.to_csv(source, index=False)

            result = build_hybrid_dataset(source, output, seed=7, target_harm_rate=0.10)

            self.assertEqual(len(result), len(sample))
            self.assertTrue(result["booking_id"].is_unique)
            self.assertTrue(result["booking_timestamp"].is_monotonic_increasing)
            self.assertTrue(result["is_harmful"].isin([0, 1]).all())
            self.assertTrue(output.exists())

    def test_model_features_exclude_post_booking_information(self) -> None:
        columns = [
            "hotel",
            "lead_time",
            "agent",
            "company",
            "is_harmful",
            *POST_BOOKING_COLUMNS,
        ]
        features = model_feature_columns(columns)

        self.assertEqual(features, ["hotel", "lead_time"])

    def test_device_account_count_uses_only_prior_30_days(self) -> None:
        times = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-10", "2026-02-15"]))
        counts = _point_in_time_device_accounts(
            times,
            device_ids=pd.Series(["D1", "D1", "D1"]).to_numpy(),
            traveler_ids=pd.Series(["T1", "T2", "T3"]).to_numpy(),
        )

        self.assertEqual(counts.tolist(), [1, 2, 1])

    def test_prior_harm_waits_for_outcome_maturity(self) -> None:
        booking_times = pd.Series(
            pd.to_datetime(["2026-01-01", "2026-01-15", "2026-02-15"])
        )
        outcome_times = pd.Series(
            pd.to_datetime(["2026-02-01", "2026-02-20", "2026-03-15"])
        )
        prior_harm, prior_chargebacks = _point_in_time_outcome_history(
            booking_times,
            outcome_times,
            traveler_ids=pd.Series(["T1", "T1", "T1"]).to_numpy(),
            labels=pd.Series([1, 0, 0]).to_numpy(),
            chargebacks=pd.Series([1, 0, 0]).to_numpy(),
        )

        self.assertEqual(prior_harm.tolist(), [0, 0, 1])
        self.assertEqual(prior_chargebacks.tolist(), [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
