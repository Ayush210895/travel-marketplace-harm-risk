import unittest

import numpy as np

from marketplace_risk.calibration import expected_calibration_error


class CalibrationMetricTests(unittest.TestCase):
    def test_perfect_group_calibration_has_zero_error(self) -> None:
        labels = np.array([0, 0, 1, 1])
        probability = np.array([0.0, 0.0, 1.0, 1.0])

        observed = expected_calibration_error(labels, probability, bins=2)

        self.assertAlmostEqual(observed, 0.0)


if __name__ == "__main__":
    unittest.main()
