"""Fetch the frozen extension only after the experiment protocol exists."""
import json
import math
from datetime import date, timedelta
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from step1_baseline import FIELDS, ROOT, write_csv

OUT = ROOT / "outputs/step16"
ENDPOINT = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
FIRST = date(2026, 10, 28)
LAST = date(2026, 11, 16)
CSV = OUT / "external_prices.csv"
REQUEST_PARAMS = {"secid": "0.399006", "klt": "101", "fqt": "0", "fields1": "f1,f2,f3,f4,f5,f6",
                  "fields2": "f51,f52,f53,f54,f55,f56,f57",
                  "beg": FIRST.strftime("%Y%m%d"), "end": LAST.strftime("%Y%m%d")}
CALENDAR = [FIRST + timedelta(days=i) for i in range((LAST - FIRST).days + 1)
            if (FIRST + timedelta(days=i)).weekday() < 5]


def validate(rows):
    if [r["date"] for r in rows] != CALENDAR:
        raise ValueError("Source dates differ from the frozen exchange calendar.")
    for r in rows:
        if (not all(math.isfinite(r[k]) and r[k] > 0 for k in FIELDS[1:])
                or not r["low"] <= min(r["open"], r["close"]) <= max(r["open"], r["close"]) <= r["high"]):
            raise ValueError(f"Invalid external daily bar: {r['date']}")


def main():
    if CSV.exists():
        raise SystemExit(f"Using the existing external snapshot {CSV}")
    if not (OUT / "forward_protocol.json").is_file():
        raise SystemExit("Run step16_refinement_experiment.py before downloading.")
    request = Request(f"{ENDPOINT}?{urlencode(REQUEST_PARAMS)}", headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=15) as response:
        payload = json.load(response)
    result = payload["data"]
    if payload["rc"] != 0 or result["code"] != "399006" or result["market"] != 0:
        raise ValueError("Unexpected index or inactive data source")
    rows = []
    for line in result["klines"]:
        day, opening, closing, high, low, volume, amount = line.split(",")
        rows.append(dict(zip(FIELDS, (date.fromisoformat(day), *map(float, (opening, high, low, closing, volume, amount))))))
    validate(rows)
    temp = CSV.with_suffix(".tmp")
    write_csv(temp, rows, FIELDS)
    temp.replace(CSV)
    print(f"Saved {len(rows)} market sessions; evaluation not run.")


if __name__ == "__main__":
    main()
