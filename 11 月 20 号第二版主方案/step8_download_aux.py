"""下载公开指数日线。默认只下载初始训练截止日之前的数据。"""
import argparse
import csv
import json
import math
from datetime import date
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "auxiliary"
SOURCE = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
SYMBOLS = {"csi300": "sh000300", "chinext_composite": "sz399102"}
START = date(2017, 1, 1)
LAST_HISTORY = date(2025, 10, 9)
FIRST_FORMAL = date(2025, 10, 10)
LAST_FORMAL = date(2026, 10, 23)


def download(symbol, start, end):
    records = {}
    for year in range(start.year, end.year + 1):
        first, last = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
        query = urlencode({"_var": "kline_day", "param": f"{symbol},day,{first},{last},300,"})
        with urlopen(f"{SOURCE}?{query}", timeout=15) as response:
            prefix, payload = response.read().decode("utf-8-sig").split("=", 1)
        if prefix.strip() != "kline_day":
            raise ValueError("Unexpected source response")
        obj = json.loads(payload.strip().rstrip(";"))
        if obj["code"] != 0 or not obj["data"][symbol]["day"]:
            raise ValueError(f"No unadjusted daily bars: {symbol} {first} - {last}")
        for row in obj["data"][symbol]["day"]:
            day = date.fromisoformat(row[0])
            if not first <= day <= last:
                continue
            o, c, h, low = map(float, row[1:5])
            if (day in records or not all(math.isfinite(p) and p > 0 for p in (o, c, h, low))
                    or not low <= min(o, c) <= max(o, c) <= h):
                raise ValueError(f"Duplicate date or invalid OHLC: {symbol} {day}")
            records[day] = (o, h, low, c)
    if not records:
        raise ValueError(f"No valid data for {symbol}")
    return records


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--formal", action="store_true")
    args = cli.parse_args()
    if args.formal and not (ROOT / "outputs/step8/history_protocol.json").is_file():
        raise SystemExit("先运行 step8_auxiliary.py，保存历史实验协议。")
    first, last = (FIRST_FORMAL, LAST_FORMAL) if args.formal else (START, LAST_HISTORY)
    tag = "formal" if args.formal else "history"
    OUT.mkdir(parents=True, exist_ok=True)
    for name, symbol in SYMBOLS.items():
        records = download(symbol, first, last)
        path = OUT / f"{tag}_{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "open", "high", "low", "close"])
            writer.writerows((day, *records[day]) for day in sorted(records))
        print(f"{name} {len(records)} {min(records)} {max(records)} {path}", flush=True)


if __name__ == "__main__":
    main()
