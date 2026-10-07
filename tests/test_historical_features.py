import unittest

import pandas as pd

from marketplace_risk.historical_features import add_point_in_time_entity_features


class HistoricalFeatureTests(unittest.TestCase):
    def test_outcome_enters_rate_only_after_maturity(self) -> None:
        data = pd.DataFrame(
            {
                "booking_timestamp": pd.to_datetime(
                    ["2026-01-01", "2026-01-10", "2026-02-10"]
                ),
                "outcome_matured_at": pd.to_datetime(
                    ["2026-02-01", "2026-02-20", "2026-03-10"]
                ),
                "traveler_id": ["T1", "T1", "T1"],
                "device_id": ["D1", "D1", "D1"],
                "payment_instrument_id": ["P1", "P1", "P1"],
                "property_id": ["H1", "H1", "H1"],
                "is_harmful": [1, 0, 0],
                "harm_type": ["chargeback", "none", "none"],
            }
        )

        result = add_point_in_time_entity_features(data, prior_strength=20.0)

        self.assertEqual(result["traveler_prior_bookings"].tolist(), [0, 1, 2])
        self.assertEqual(result["traveler_mature_outcomes"].tolist(), [0, 0, 1])
        self.assertEqual(result["traveler_prior_chargebacks"].tolist(), [0, 0, 1])
        self.assertEqual(result["is_new_traveler"].tolist(), [1, 0, 0])

    def test_smoothing_reduces_small_sample_extremes(self) -> None:
        data = pd.DataFrame(
            {
                "booking_timestamp": pd.to_datetime(["2026-01-01", "2026-03-01"]),
                "outcome_matured_at": pd.to_datetime(["2026-02-01", "2026-04-01"]),
                "traveler_id": ["T1", "T1"],
                "device_id": ["D1", "D1"],
                "payment_instrument_id": ["P1", "P1"],
                "property_id": ["H1", "H1"],
                "is_harmful": [1, 0],
                "harm_type": ["chargeback", "none"],
            }
        )

        result = add_point_in_time_entity_features(
            data, prior_strength=20.0, initial_harm_rate=0.025
        )

        observed = result.loc[1, "traveler_smoothed_harm_rate"]
        self.assertGreater(observed, 0.025)
        self.assertLess(observed, 1.0)

    def test_missing_agent_and_company_are_not_shared_entities(self) -> None:
        data = pd.DataFrame(
            {
                "booking_timestamp": pd.to_datetime(
                    ["2026-01-01", "2026-03-01", "2026-04-01"]
                ),
                "outcome_matured_at": pd.to_datetime(
                    ["2026-02-01", "2026-03-15", "2026-05-01"]
                ),
                "traveler_id": ["T1", "T2", "T3"],
                "device_id": ["D1", "D2", "D3"],
                "payment_instrument_id": ["P1", "P2", "P3"],
                "property_id": ["H1", "H2", "H3"],
                "agent": [None, None, 9.0],
                "company": [None, None, 40.0],
                "is_harmful": [1, 0, 0],
                "harm_type": ["chargeback", "none", "none"],
            }
        )

        result = add_point_in_time_entity_features(data, prior_strength=20.0)

        self.assertEqual(result["has_agent"].tolist(), [0, 0, 1])
        self.assertEqual(result["has_company"].tolist(), [0, 0, 1])
        self.assertEqual(result["agent_prior_bookings"].tolist(), [0, 0, 0])
        self.assertEqual(result["company_prior_bookings"].tolist(), [0, 0, 0])
        self.assertEqual(result["is_new_agent"].tolist(), [0, 0, 1])
        self.assertEqual(result["is_new_company"].tolist(), [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
