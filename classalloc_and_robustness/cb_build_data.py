#!/usr/bin/env python3
"""Build the data of the third real benchmark (ClassAlloc) from PortBench's public raw prices (no network, no cost).

ClassAlloc is a monthly allocation task over five of PortBench's six asset classes (equities, bonds, commodities, real
estate, cash). The cryptocurrency class is left out: its equal-weighted index is dominated by coins listed during the
period (it gains about 5,400% in 2017 and 1,100% in 2021), so it would turn the task into a bet on one class. Each class is an equal-weighted index of the benchmark's assets in that class, rebalanced daily
over the assets that have prices on both days (an asset listed after January 2015 enters after 20 sessions of prices). Decisions are made on the first trading session of every month from July
2015 (the first month with six months of history) to March 2026 (the last month with 20 sessions of outcome in the
data), each held for 20 sessions. Warm-up (lessons written by the agent): July 2015 to April 2020. Test: May 2020 to
March 2026; the decisions from July 2024 on are after the knowledge cutoff of gpt-4.1-mini (June 2024).

Input : ../portbench_prices/prices.pkl (PortBench raw prices at the pinned revision; needs numpy and pandas).
Output: classalloc/data.json (read by run_cb.py with the standard library only) and classalloc/DATA_CHECKS.json.
"""
import json, math, pickle, hashlib, collections
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "portbench_prices" / "prices.pkl"
OUT = HERE / "classalloc"; OUT.mkdir(exist_ok=True)
CLASSES = ["equities", "bonds", "commodities", "real_estate", "cash"]
WARM_END, TEST_START, CUTOFF = "2020-04-30", "2020-05-01", "2024-07-01"
HOLD, HIST, SEASON = 20, 126, 20


def main():
    raw = SRC.read_bytes(); D = pickle.loads(raw)
    sess = [s.date().isoformat() for s in D["sessions"]]
    PR = {a: {ts.date().isoformat(): float(v) for ts, v in s.items()} for a, s in D["prices"].items()}
    members = {c: sorted(a for a, cl in D["class_map"].items() if cl == c and a in PR) for c in CLASSES}
    sidx = {s: i for i, s in enumerate(sess)}
    first_i = {a: min(sidx[s] for s in PR[a] if s in sidx) for a in PR}
    # an asset listed after the start of the data enters its class index only after SEASON sessions of prices
    # (the first quotes of new listings can be erroneous, e.g. AAVE-USD gains 10,200% on its third day)
    start = {a: (first_i[a] + SEASON if first_i[a] > 0 else 0) for a in PR}
    level = {c: [1.0] for c in CLASSES}; used = {c: [0] for c in CLASSES}; daily = {c: [0.0] for c in CLASSES}
    for i in range(1, len(sess)):
        s0, s1 = sess[i - 1], sess[i]
        for c in CLASSES:
            rs = [PR[a][s1] / PR[a][s0] - 1.0 for a in members[c]
                  if i - 1 >= start[a] and s0 in PR[a] and s1 in PR[a] and PR[a][s0] > 0]
            r = float(np.mean(rs)) if rs else 0.0
            daily[c].append(r); used[c].append(len(rs)); level[c].append(level[c][-1] * (1.0 + r))
    first = {}
    for i, s in enumerate(sess):
        if i < HIST or i + HOLD >= len(sess): continue
        key = s[:7]
        if key not in first: first[key] = i
    dates = [sess[i] for i in sorted(first.values())]
    warm = [d for d in dates if d <= WARM_END]; test = [d for d in dates if d >= TEST_START]
    data = {"source": "PortBench raw prices (investor_step0_20261001/prices.pkl)", "source_sha256": hashlib.sha256(raw).hexdigest(),
            "classes": CLASSES, "members": members, "sessions": sess, "level": level,
            "decision_dates": dates, "warmup": warm, "test": test, "post_cutoff_from": CUTOFF, "hold_sessions": HOLD, "season_sessions": SEASON}
    (OUT / "data.json").write_text(json.dumps(data))
    ext = {c: dict(min=float(np.min(daily[c][1:])), max=float(np.max(daily[c][1:])),
                   days_without_member=int(sum(1 for n in used[c][1:] if n == 0)),
                   first_day_with_member=next(s for s, n in zip(sess[1:], used[c][1:]) if n > 0)) for c in CLASSES}
    yearly = collections.OrderedDict()
    for c in CLASSES:
        yearly[c] = {}
        for y in range(2015, 2027):
            idx = [i for i, s in enumerate(sess) if s.startswith(str(y))]
            if idx: yearly[c][y] = round(100 * (level[c][idx[-1]] / level[c][max(idx[0] - 1, 0)] - 1), 1)
    chk = {"n_sessions": len(sess), "first_session": sess[0], "last_session": sess[-1], "n_members": {c: len(v) for c, v in members.items()},
           "n_decisions": len(dates), "warmup": [warm[0], warm[-1], len(warm)], "test": [test[0], test[-1], len(test)],
           "test_post_cutoff": sum(1 for d in test if d >= CUTOFF), "daily_return_extremes": ext, "calendar_year_return_pct": yearly}
    (OUT / "DATA_CHECKS.json").write_text(json.dumps(chk, indent=1)); print(json.dumps(chk, indent=1))


if __name__ == "__main__":
    main()
