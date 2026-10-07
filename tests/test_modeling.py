import unittest

import numpy as np
import pandas as pd

from marketplace_risk.modeling import capacity_metrics, rolling_mature_splits


class CapacityMetricTests(unittest.TestCase):
    def test_top_five_percent_uses_ranked_scores(self) -> None:
        labels = np.zeros(100, dtype=int)
        labels[[10, 20, 30, 40]] = 1
        scores = np.linspace(0, 0.80, 100)
        scores[[10, 20, 30]] = [0.99, 0.98, 0.97]

        metrics = capacity_metrics(labels, scores)

        self.assertEqual(metrics["review_count"], 5)
        self.assertAlmostEqual(metrics["precision_at_5pct"], 3 / 5)
        self.assertAlmostEqual(metrics["recall_at_5pct"], 3 / 4)

    def test_expected_loss_scores_do_not_require_probability_range(self) -> None:
        labels = np.array([0, 1, 0, 1])
        expected_loss = np.array([5.0, 80.0, 10.0, 100.0])

        metrics = capacity_metrics(
            labels,
            expected_loss,
            review_fraction=0.25,
            probability_scores=False,
        )

        self.assertNotIn("brier_score", metrics)
        self.assertEqual(metrics["recall_at_5pct"], 0.5)


class RollingSplitTests(unittest.TestCase):
    def test_training_labels_are_mature_at_each_fold_cutoff(self) -> None:
        booking_time = pd.date_range("2026-01-01", periods=20, freq="D")
        data = pd.DataFrame(
            {
                "booking_timestamp": booking_time,
                "outcome_matured_at": booking_time + pd.to_timedelta(3, unit="D"),
            }
        )

        splits = rolling_mature_splits(data, validation_fractions=(0.50, 0.75))

        for train_indices, validation_indices in splits:
            cutoff = data.loc[validation_indices[0], "booking_timestamp"]
            self.assertTrue(
                (data.loc[train_indices, "outcome_matured_at"] <= cutoff).all()
            )
            self.assertLess(train_indices.max(), validation_indices.min())


if __name__ == "__main__":
    unittest.main()
