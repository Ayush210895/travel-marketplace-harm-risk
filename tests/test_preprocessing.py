import unittest

import numpy as np
import pandas as pd

from marketplace_risk.preprocessing import (
    AdrOutlierCapper,
    BookingValueFeatures,
    GuestCountFeatures,
    RareCategoryCleaner,
    TemporalFeatures,
)


class RareCategoryCleanerTests(unittest.TestCase):
    def test_cleaning_precedes_frequency_grouping(self) -> None:
        train = pd.DataFrame(
            {
                "country": ["CN", "CHN", "CHN", None, "IND"],
                "meal": ["BB", "BB", "Undefined", "HB", "SC"],
            }
        )
        cleaner = RareCategoryCleaner(
            columns=["country", "meal"], min_frequency=2
        ).fit(train)

        transformed = cleaner.transform(train)

        self.assertEqual(transformed["country"].tolist()[:3], ["CHN"] * 3)
        self.assertEqual(transformed.loc[3, "country"], "Unknown")
        self.assertEqual(transformed.loc[4, "country"], "Other")
        self.assertEqual(transformed.loc[2, "meal"], "Unknown")

    def test_unseen_category_maps_to_other(self) -> None:
        train = pd.DataFrame({"country": ["IND", "IND", None]})
        validation = pd.DataFrame({"country": ["IND", "BRA", None]})
        cleaner = RareCategoryCleaner(columns=["country"], min_frequency=2).fit(train)

        transformed = cleaner.transform(validation)

        self.assertEqual(
            transformed["country"].tolist(), ["IND", "Other", "Unknown"]
        )


class AdrOutlierCapperTests(unittest.TestCase):
    def test_validation_uses_training_cap_and_preserves_outlier_signal(self) -> None:
        train = pd.DataFrame({"adr": [50.0, 100.0, 150.0, 200.0]})
        validation = pd.DataFrame({"adr": [-6.0, 80.0, 5000.0]})
        capper = AdrOutlierCapper(quantile=0.75).fit(train)

        transformed = capper.transform(validation)

        self.assertEqual(capper.adr_cap_, 162.5)
        self.assertEqual(transformed["adr"].tolist(), [0.0, 80.0, 162.5])
        self.assertEqual(transformed["adr_extreme"].tolist(), [0, 0, 1])
        self.assertEqual(transformed["adr_negative"].tolist(), [1, 0, 0])


class GuestCountFeatureTests(unittest.TestCase):
    def test_invalid_count_and_missing_children_remain_visible(self) -> None:
        data = pd.DataFrame(
            {
                "adults": [0, 2, 1],
                "children": [0.0, None, 2.0],
                "babies": [0, 0, 1],
            }
        )

        transformed = GuestCountFeatures().fit(data).transform(data)

        self.assertEqual(transformed["total_guests"].tolist(), [0.0, 2.0, 4.0])
        self.assertEqual(transformed["invalid_guest_count"].tolist(), [1, 0, 0])
        self.assertEqual(transformed["children_missing"].tolist(), [0, 1, 0])


class TemporalFeatureTests(unittest.TestCase):
    def test_raw_dates_become_cyclical_features(self) -> None:
        data = pd.DataFrame(
            {
                "booking_timestamp": pd.to_datetime(
                    ["2026-01-05 00:00:00", "2026-07-04 12:00:00"]
                ),
                "arrival_date_year": [2026, 2026],
                "arrival_date_month": ["January", "July"],
                "arrival_date_week_number": [2, 27],
                "arrival_date_day_of_month": [10, 10],
                "lead_time": [5, 6],
            }
        )

        transformed = TemporalFeatures().fit(data).transform(data)

        self.assertNotIn("booking_timestamp", transformed)
        self.assertNotIn("arrival_date_month", transformed)
        self.assertIn("lead_time", transformed)
        self.assertAlmostEqual(transformed.loc[0, "booking_month_sin"], 0.0)
        self.assertAlmostEqual(transformed.loc[0, "booking_month_cos"], 1.0)
        self.assertAlmostEqual(transformed.loc[1, "booking_hour_sin"], 0.0)
        self.assertAlmostEqual(transformed.loc[1, "booking_hour_cos"], -1.0)


class BookingValueFeatureTests(unittest.TestCase):
    def test_total_nights_and_estimated_value(self) -> None:
        data = pd.DataFrame(
            {
                "adr": [200.0, 150.0],
                "stays_in_week_nights": [0, 2],
                "stays_in_weekend_nights": [0, 1],
            }
        )

        transformed = BookingValueFeatures().fit(data).transform(data)

        self.assertEqual(transformed["total_nights"].tolist(), [0, 3])
        self.assertEqual(
            transformed["estimated_booking_value"].tolist(), [200.0, 450.0]
        )
        self.assertAlmostEqual(
            transformed.loc[0, "log_estimated_booking_value"], np.log1p(200.0)
        )


if __name__ == "__main__":
    unittest.main()
