import unittest
from datetime import date

from account import LongOnlyT1Account, simulate, verify_consensus_ledger
from features import FEE_RATE


D1 = date(2025, 1, 6)
D2 = date(2025, 1, 7)


def decision(target_date, asof_date, weight):
    return {
        "target_date": str(target_date),
        "asof_date": str(asof_date),
        "strategy_weight": weight,
        "fit_asof": str(asof_date),
        "max_training_label_date": str(asof_date),
    }


class LongOnlyT1AccountTest(unittest.TestCase):
    def test_same_day_purchase_cannot_be_sold(self):
        account = LongOnlyT1Account(100.0, [D1])
        account.buy(D1, 10.0, 2.0)

        with self.assertRaisesRegex(ValueError, r"T\+1"):
            account.sell(D1, 10.0, 1.0)

        self.assertEqual(account.units, 2.0)
        self.assertEqual(account.sellable(D1), 0.0)
        self.assertEqual(account.cash, 80.0)

    def test_old_lots_are_sellable_but_new_lot_stays_frozen(self):
        account = LongOnlyT1Account(1000.0, [D1, D2])
        account.buy(D1, 10.0, 5.0)
        account.buy(D2, 12.0, 2.0)

        self.assertEqual(account.sellable(D2), 5.0)
        account.sell(D2, 13.0, 5.0)
        self.assertEqual(account.units, 2.0)
        self.assertEqual(account.cash, 991.0)

        with self.assertRaisesRegex(ValueError, r"T\+1"):
            account.sell(D2, 13.0, 0.1)
        self.assertEqual(account.units, 2.0)
        self.assertEqual(len(account.lots), 1)
        self.assertEqual(account.lots[0].bought_on, D2)

    def test_cannot_sell_more_than_total_holdings(self):
        account = LongOnlyT1Account(100.0, [D1, D2])
        account.buy(D1, 10.0, 2.0)

        with self.assertRaisesRegex(ValueError, "禁止做空"):
            account.sell(D2, 10.0, 2.1)

        self.assertEqual(account.units, 2.0)
        self.assertEqual(account.cash, 80.0)

    def test_cannot_overdraw_cash_including_fees(self):
        account = LongOnlyT1Account(100.0, [D1], fee_rate=0.01)

        with self.assertRaisesRegex(ValueError, "现金不足"):
            account.buy(D1, 10.0, 10.0)

        self.assertEqual(account.cash, 100.0)
        self.assertEqual(account.units, 0.0)
        self.assertEqual(account.lots, [])


class SimulationAndSignalClockTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"date": D1, "open": 10.0, "close": 11.0},
            {"date": D2, "open": 12.0, "close": 13.0},
        ]

    def test_manual_simulation_fees_consensus_and_no_rebalance(self):
        capital = 1000.0
        decisions = [
            decision(D1, date(2025, 1, 3), 0.5),
            decision(D2, D1, 0.5),
        ]

        run = simulate(self.rows, decisions, capital=capital)
        verify_consensus_ledger(self.rows, decisions, run)

        quantity = (capital * 0.5 / 10.0) / (1.0 + 0.5 * FEE_RATE)
        cash_after_buy = capital - quantity * 10.0 * (1.0 + FEE_RATE)
        expected_buy_fee = quantity * 10.0 * FEE_RATE
        expected_sell_fee = quantity * 13.0 * FEE_RATE
        expected_final_cash = cash_after_buy + quantity * 13.0 * (1.0 - FEE_RATE)

        self.assertEqual(len(run["trades"]), 2)
        self.assertEqual(
            [(trade["date"], trade["time"], trade["side"]) for trade in run["trades"]],
            [(str(D1), "09:30", "BUY"), (str(D2), "15:00", "SELL")],
        )
        self.assertAlmostEqual(run["trades"][0]["units"], quantity, places=12)
        self.assertAlmostEqual(run["trades"][0]["fee"], expected_buy_fee, places=12)
        self.assertAlmostEqual(run["trades"][1]["fee"], expected_sell_fee, places=12)
        self.assertAlmostEqual(run["metrics"]["fees_paid"],
                               expected_buy_fee + expected_sell_fee, places=12)
        self.assertAlmostEqual(run["metrics"]["final_equity"], expected_final_cash, places=10)
        self.assertAlmostEqual(run["curve"][1]["equity"],
                               cash_after_buy + quantity * 11.0, places=10)

        # 第二天价格已变化，但目标仓位不变，因此开盘不交易，也不恢复至 50%。
        second_day = run["curve"][2]
        self.assertNotAlmostEqual(second_day["open_weight_after_trade"], 0.5, places=3)
        self.assertEqual(second_day["target_weight"], 0.5)
        self.assertEqual(run["metrics"]["order_count"], 2)

    def test_signal_asof_on_target_date_is_rejected(self):
        invalid = decision(D1, D1, 0.0)

        with self.assertRaisesRegex(ValueError, "当天完整行情"):
            simulate(self.rows[:1], [invalid])

    def test_training_label_beyond_prediction_time_is_rejected(self):
        invalid = decision(D1, date(2025, 1, 3), 0.0)
        invalid["max_training_label_date"] = str(D1)

        with self.assertRaisesRegex(ValueError, "训练答案越过"):
            simulate(self.rows[:1], [invalid])


if __name__ == "__main__":
    unittest.main()
