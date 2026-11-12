"""Aggregate counts that the paper takes from poem-level files, so that 05, 14 and 15 can run without them.

Reads results/topic1z_work_phase.parquet (built by 07) and data/five_axis/blind50_decisions.parquet. Writes
results/public_counts.json: place-classification counts, poets at the seat, poems four or more years into exile,
the blind sample size, and the per-phase poet counts and largest single-poet shares behind Figure 1's intervals.
Counts only; no poem identifiers, titles or annotations.
"""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
R = os.environ.get("DH2027_RESULTS_DIR", os.path.join(ROOT, "results"))

w = pd.read_parquet(f"{R}/topic1z_work_phase.parquet")
# same phase sample as the joint phase model in 07: poets seen both en route and after arrival
sample = w[w.phase.isin(["en route", "arrival year", "settled +1", "settled +2-3", "settled +4 or more"])]
sample = sample.groupby("person").filter(lambda g: g.phase.eq("en route").any() and g.phase.ne("en route").any())
late = w.years_in_exile >= 4
counts = {
    "works": len(w),
    "stage_counts": {k: int(v) for k, v in w.stage_adj.value_counts().sort_index().items()},
    "seat_poets": int(w[w.stage_adj == "destination"].person.nunique()),
    "works_four_or_more_years": int(late.sum()),
    "works_four_or_more_years_proxy_dated": int((late & (w.precise == 0)).sum()),
    "blind_sample_works": int(pd.read_parquet(f"{D}/five_axis/blind50_decisions.parquet")["DB_source_record_id"].nunique()),
    "phase_poets": {k: int(v) for k, v in sample.groupby("phase").person.nunique().items()},
    "phase_top_poet_share": {k: round(float(v), 6) for k, v in
                             sample.groupby("phase").person.agg(lambda s: s.value_counts(normalize=True).iloc[0]).items()},
}
with open(f"{R}/public_counts.json", "w", encoding="utf-8") as f:
    json.dump(counts, f, ensure_ascii=False, indent=1)
print(json.dumps(counts, ensure_ascii=False))
