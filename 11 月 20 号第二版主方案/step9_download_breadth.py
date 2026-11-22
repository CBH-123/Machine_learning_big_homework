"""从国证指数调样表和东方财富日线构建创业板指每日上涨占比。"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import date
import json
import math
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from step1_baseline import ROOT, END, read_prices, write_csv

DATA = ROOT / "data" / "breadth"
MEMBERSHIP_START = date(2022, 12, 12)
SOURCE = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
# 行情源将中航电测的历史并入中航成飞。国证调样表仍保留逐日有效的代码。
QUOTE_ALIASES = {"300114": ("302132", date(2025, 2, 17))}


def cni_rows(path):
    # 国证下载文件的工作表使用 inlineStr，按表头校验，避免错读字段。
    xml = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as workbook:
        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
    return [[cell.findtext("s:is/s:t", namespaces=xml) for cell in row]
            for row in sheet.findall(".//s:row", xml)]


def membership():
    rows = cni_rows(DATA / "cni_399006_adjustment.xlsx")
    if rows[0] != ["开始日期", "结束日期", "样本代码", "样本简称", "所属行业", "调整类型"]:
        raise ValueError("Unknown official membership table format")
    groups = defaultdict(dict)
    for first, last, code, _, _, action in rows[1:]:
        first, last = date.fromisoformat(first), date.fromisoformat(last)
        if last < MEMBERSHIP_START:
            continue
        if not code or len(code) != 6 or not code.isdecimal() or code[:2] != "30":
            raise ValueError(f"Unexpected ChiNext code: {code}")
        if action not in {"OLD", "+", "-", "备选"} or code in groups[first, last]:
            raise ValueError(f"Unexpected membership row: {first} {code} {action}")
        groups[first, last][code] = action
    regimes = []
    previous = None
    for (first, last), statuses in sorted(groups.items()):
        active = {code for code, action in statuses.items() if action in {"OLD", "+"}}
        if len(active) != 100 or (previous is not None and
                (active - previous != {code for code, action in statuses.items() if action == "+"}
                 or previous - active != {code for code, action in statuses.items() if action == "-"})):
            raise ValueError(f"Invalid adjustment chain at {first}")
        regimes.append((first, last, active))
        previous = active
    snapshot = cni_rows(DATA / "cni_399006_history.xlsx")
    if snapshot[0][:2] != ["日期", "样本代码"]:
        raise ValueError("Unknown official snapshot table format")
    dates = {date.fromisoformat(row[0]) for row in snapshot[1:]}
    if len(dates) != 1:
        raise ValueError("Expected one comparison snapshot")
    snapshot_date = dates.pop()
    same_period = [active for first, last, active in regimes if first <= snapshot_date <= last]
    if not same_period or {row[1] for row in snapshot[1:]} != same_period[-1]:
        raise ValueError("Adjustment and comparison snapshot do not agree")
    return regimes, snapshot_date


def requested_days():
    official, _ = read_prices(ROOT / "data/raw/SZ#399006.txt")
    regimes, _ = membership()
    schedule = {}
    for row in official:
        day = row["date"]
        if MEMBERSHIP_START <= day <= END:
            applicable = [(first, active) for first, last, active in regimes if first <= day <= last]
            if not applicable:
                raise ValueError(f"Membership dates do not cover {day}")
            schedule[day] = applicable[-1]
    return schedule


def fetch_changes(code, first, last, cache):
    path = cache / f"{code}_{first}_{last}.csv"
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as f:
            return {date.fromisoformat(r["date"]): float(r["change"]) for r in csv.DictReader(f)}
    quote_code = code
    if code in QUOTE_ALIASES:
        quote_code, changed = QUOTE_ALIASES[code]
        if last >= changed:
            raise ValueError(f"Outdated membership code {code} on {last}")
    query = urlencode({
        "secid": f"0.{quote_code}", "klt": "101", "fqt": "0",
        "beg": first.strftime("%Y%m%d"), "end": last.strftime("%Y%m%d"), "lmt": "1500",
        "fields1": "f1,f2,f3,f4,f5,f6", "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
    })
    request = Request(f"{SOURCE}?{query}", headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=12) as response:
                payload = json.load(response)
            obj = payload["data"]
            if payload["rc"] != 0 or not obj or obj["code"] != quote_code or not obj["klines"]:
                raise ValueError(f"Missing daily quotes for {code}")
            result = {}
            for line in obj["klines"]:
                fields = line.split(",")
                day = date.fromisoformat(fields[0])
                if not first <= day <= last:
                    continue
                volume, change = float(fields[5]), float(fields[9])
                if day in result or not math.isfinite(change) or not math.isfinite(volume) or volume < 0:
                    raise ValueError(f"Invalid daily quote for {code} on {day}")
                if volume > 0:
                    result[day] = change
            if not result:
                raise ValueError(f"No usable daily quotes for {code}")
            temporary = path.with_suffix(".tmp")
            write_csv(temporary, [{"date": day, "change": value} for day, value in sorted(result.items())])
            temporary.replace(path)
            return result
        except (OSError, ValueError, KeyError, TypeError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    schedule = requested_days()
    days_by_code = defaultdict(list)
    for day, (_, active) in schedule.items():
        for code in active:
            days_by_code[code].append(day)
    cache = DATA / "daily_changes"
    cache.mkdir(exist_ok=True)
    quotes = {}
    print(f"Fetching {len(days_by_code)} historical member series", flush=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        tasks = {pool.submit(fetch_changes, code, days[0], days[-1], cache): code
                 for code, days in sorted(days_by_code.items())}
        for i, f in enumerate(as_completed(tasks), start=1):
            code = tasks[f]
            quotes[code] = f.result()
            if i % 25 == 0 or i == len(tasks):
                print(f"Fetched or reused {i}/{len(tasks)}", flush=True)
    records = []
    for day, (effective, active) in schedule.items():
        tradable = [quotes[code][day] for code in active if day in quotes[code]]
        signs = Counter((value > 0) - (value < 0) for value in tradable)
        records.append({"date": day, "membership_effective": effective, "members": len(active),
                        "valid_quotes": len(tradable), "up": signs[1], "down": signs[-1], "flat": signs[0],
                        "breadth": signs[1] / len(tradable) if tradable else ""})
    write_csv(DATA / "chinext_daily_breadth.csv", records)
    print(f"{len(records)} sessions, minimum daily coverage {min(r['valid_quotes'] for r in records)}/100", flush=True)


if __name__ == "__main__":
    main()
