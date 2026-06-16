#!/usr/bin/env python3
"""
refresh_salesevals.py — on-demand refresher for data/salesevals_snapshot.json.

salesevals.com has no documented JSON API and the rendered leaderboard varies in HTML
structure; rather than scrape fragile selectors daily, we keep a committed snapshot
file and refresh it manually (quarterly + on-demand). This script is the helper.

Usage:
    python3 scripts/refresh_salesevals.py

Behavior:
    1. Prints instructions to manually copy the salesevals leaderboard data
    2. Validates the JSON structure of data/salesevals_snapshot.json
    3. Reports any in-scope models from aliases.json that lack a salesevals_name match

Why manual?
    - The leaderboard updates monthly at most (Ryan's signal, not a daily-moving target)
    - Scraping HTML is fragile and would create false positives on rerenders
    - Committed snapshot = bulletproof; never breaks the dashboard
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
SNAPSHOT = ROOT / "data" / "salesevals_snapshot.json"
ALIASES = ROOT / "aliases.json"

def main():
    print("=" * 70)
    print("salesevals.com snapshot refresher (manual)")
    print("=" * 70)
    print()
    print("STEP 1: Visit https://www.salesevals.com/ and copy the current leaderboard.")
    print("STEP 2: Update data/salesevals_snapshot.json with the new rows.")
    print("STEP 3: Update 'snapshot_date' and 'data_date' fields.")
    print("STEP 4: Re-run this script to validate + check alias coverage.")
    print()
    print("Validating current snapshot…")

    if not SNAPSHOT.exists():
        print(f"  ✗ {SNAPSHOT} does not exist")
        return 1
    try:
        d = json.loads(SNAPSHOT.read_text())
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON parse failed: {e}")
        return 1

    required = {"source", "snapshot_date", "data_date", "ranked"}
    missing = required - set(d.keys())
    if missing:
        print(f"  ✗ Missing required fields: {missing}")
        return 1
    if not d["ranked"]:
        print(f"  ✗ ranked[] is empty")
        return 1
    for row in d["ranked"]:
        if not all(k in row for k in ("rank", "model", "score", "cost_per_call")):
            print(f"  ✗ malformed row: {row}")
            return 1
    print(f"  ✓ {len(d['ranked'])} rows · data_date={d['data_date']} · snapshot_date={d['snapshot_date']}")

    # Cross-check: which in-scope canonical models are matched?
    aliases = json.loads(ALIASES.read_text())["models"]
    sales_names = {r["model"] for r in d["ranked"]}
    matched = sum(1 for a in aliases if a.get("salesevals_name") in sales_names)
    unmatched_aliases = [a["display"] for a in aliases if not a.get("salesevals_name")]
    unmatched_sales = sales_names - {a.get("salesevals_name") for a in aliases}
    print()
    print(f"  Alias coverage: {matched}/{len(aliases)} in-scope models have salesevals data")
    if unmatched_aliases:
        print(f"  Aliases without salesevals_name ({len(unmatched_aliases)}):")
        for d_ in unmatched_aliases[:8]:
            print(f"    · {d_}")
    if unmatched_sales:
        print(f"  salesevals models NOT in aliases.json (consider adding):")
        for s in sorted(unmatched_sales):
            print(f"    · {s}")
    print()
    print("Snapshot is valid.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
