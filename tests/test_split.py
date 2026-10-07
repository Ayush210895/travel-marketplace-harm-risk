import unittest

import pandas as pd

from marketplace_risk.split import create_split_assignments


class PointInTimeSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        booking_time = pd.date_range("2026-01-01", periods=20, freq="D")
        self.data = pd.DataFrame(
            {
                "booking_id": [f"B{i:02d}" for i in range(20)],
                "booking_timestamp": booking_time,
                "outcome_matured_at": booking_time + pd.to_timedelta(
                    [2] * 12 + [10] * 8, unit="D"
                ),
                "is_harmful": [0, 1] * 10,
            }
        )

    def test_chronological_periods_do_not_overlap(self) -> None:
        assignments, manifest = create_split_assignments(self.data)

        self.assertEqual(manifest["initial_train_rows"], 14)
        self.assertEqual(manifest["validation_rows"], 3)
        self.assertEqual(manifest["test_rows"], 3)
        self.assertEqual(
            assignments["development_split"].tolist(),
            ["initial_train"] * 14 + ["validation"] * 3 + ["test"] * 3,
        )

    def test_training_eligibility_requires_mature_label(self) -> None:
        assignments, manifest = create_split_assignments(self.data)
        validation_start = pd.Timestamp(manifest["validation_start"])
        eligible = assignments["eligible_at_validation_train_time"]

        self.assertTrue(
            (
                assignments.loc[eligible, "outcome_matured_at"] <= validation_start
            ).all()
        )
        self.assertFalse(assignments.loc[13, "eligible_at_validation_train_time"])


if __name__ == "__main__":
    unittest.main()
