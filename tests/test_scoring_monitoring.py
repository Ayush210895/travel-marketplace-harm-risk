import unittest

import numpy as np
import pandas as pd

from marketplace_risk.monitoring import monitoring_report, population_stability_index
from marketplace_risk.scoring import rank_review_queue


class ReviewQueueTests(unittest.TestCase):
    def test_expected_loss_ranking_and_capacity(self) -> None:
        queue = rank_review_queue(
            probability=np.array([0.9, 0.2, 0.6, 0.1]),
            estimated_booking_value=np.array([10.0, 100.0, 20.0, 5.0]),
            review_fraction=0.25,
        )

        self.assertEqual(int(queue["send_to_review"].sum()), 1)
        self.assertEqual(int(queue.loc[1, "review_rank"]), 1)


class MonitoringTests(unittest.TestCase):
    def test_identical_populations_have_zero_psi(self) -> None:
        values = np.arange(100, dtype=float)
        self.assertAlmostEqual(population_stability_index(values, values), 0.0)

    def test_report_calculates_review_performance(self) -> None:
        frame = pd.DataFrame(
            {
                "expected_loss_score": [4.0, 3.0, 2.0, 1.0],
                "send_to_review": [1, 1, 0, 0],
                "is_harmful": [1, 0, 1, 0],
                "harm_amount": [100.0, 0.0, 50.0, 0.0],
            }
        )

        report = monitoring_report(frame)

        self.assertEqual(report["overall"]["precision"], 0.5)
        self.assertEqual(report["overall"]["recall"], 0.5)
        self.assertAlmostEqual(report["overall"]["harm_value_capture"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
