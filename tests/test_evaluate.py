import unittest
from datetime import date

from evaluate import FEE_SCENARIOS, run_fee_sensitivity, run_strategy_variants, strategy_variants


def signal(asof, target, tcn, utility, ma20, weight):
    return {
        "model": "tiny_tcn", "asof_date": asof, "target_date": target,
        "tcn_holds": tcn, "utility_holds": utility, "ma20": ma20,
        "strategy_weight": weight, "fit_asof": asof, "max_training_label_date": asof,
    }


class EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"date": date(2025, 1, 2), "open": 100.0, "close": 102.0},
            {"date": date(2025, 1, 3), "open": 103.0, "close": 101.0},
            {"date": date(2025, 1, 6), "open": 100.0, "close": 105.0},
        ]
        self.signals = [
            signal("2025-01-02", "2025-01-03", True, False, 100.0, 0.5),
            signal("2025-01-03", "2025-01-06", False, False, 102.0, 0.0),
        ]

    def test_strategy_variant_weights(self):
        variants = strategy_variants(self.rows, self.signals)
        self.assertEqual(list(variants), ["buy_hold", "ma20", "tcn_only", "utility_only",
                                          "dual_consensus", "dual_ma20", "full"])
        self.assertEqual([rec["strategy_weight"] for rec in variants["buy_hold"]], [1.0, 1.0])
        self.assertEqual([rec["strategy_weight"] for rec in variants["ma20"]], [1.0, 0.0])
        self.assertEqual([rec["strategy_weight"] for rec in variants["tcn_only"]], [1.0, 0.0])
        self.assertEqual([rec["strategy_weight"] for rec in variants["utility_only"]], [0.0, 0.0])
        self.assertEqual([rec["strategy_weight"] for rec in variants["dual_consensus"]], [0.0, 0.0])
        self.assertEqual([rec["strategy_weight"] for rec in variants["dual_ma20"]], [0.5, 0.0])
        self.assertEqual([rec["strategy_weight"] for rec in variants["full"]], [0.5, 0.0])

    def test_all_strategy_variants_pass_ledger_audit(self):
        metrics = run_strategy_variants(self.rows, self.signals)
        self.assertEqual(len(metrics), 7)
        benchmark = next(rec for rec in metrics if rec["name"] == "buy_hold")
        self.assertEqual(benchmark["excess_return_vs_buy_hold"], 0.0)
        for rec in metrics:
            self.assertEqual(rec["trading_days"], 2)
            self.assertGreater(rec["final_equity"], 0.0)

    def test_higher_fees_do_not_improve_fixed_strategy(self):
        metrics = run_fee_sensitivity(self.rows, self.signals)
        self.assertEqual([rec["fee_rate_each_side"] for rec in metrics], list(FEE_SCENARIOS))
        self.assertEqual(metrics[0]["fees_paid"], 0.0)
        self.assertGreater(metrics[1]["final_equity"], metrics[2]["final_equity"])
        self.assertGreater(metrics[0]["final_equity"], metrics[1]["final_equity"])
        self.assertLess(metrics[1]["fees_paid"], metrics[2]["fees_paid"])


if __name__ == "__main__":
    unittest.main()
