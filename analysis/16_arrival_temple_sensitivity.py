"""Targeted checks of the arrival proxy and Bai Juyi's contribution.

Reads the existing title-corrected analysis snapshot; changes no source or main
analysis files. Excluding exchange titles affects anchor selection only, not the
outcome sample. Seat poems earlier than the alternative anchor are kept distinct
from later residence phases. Outputs contain aggregate results and episode IDs.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.environ.get("DH2027_RESULTS_DIR", os.path.join(ROOT, "results"))
SEED = 20261101
N_BOOT = 9999
PHASES = ["arrival year", "settled +1", "settled +2-3", "settled +4 or more"]
w = pd.read_parquet(os.path.join(R, "topic1z_work_phase.parquet"))


def save(frame, name):
    frame.to_csv(os.path.join(R, f"topic1s_{name}.csv"), index=False, encoding="utf-8-sig")


def fit(data, outcome, columns, contrasts, variant):
    d = data[["person", outcome] + columns].dropna()
    stats = []
    for _, group in d.groupby("person"):
        values = group[[outcome] + columns]
        centered = values - values.mean()
        x, y = centered[columns].to_numpy(), centered[outcome].to_numpy()
        stats.append((x.T @ x, x.T @ y))
    xx = np.array([s[0] for s in stats])
    xy = np.array([s[1] for s in stats])
    if not len(stats):
        raise ValueError(f"Empty model: {variant}, {outcome}")
    a_full = xx.sum(0)
    inverse = np.linalg.pinv(a_full)
    beta = inverse @ xy.sum(0)
    for weights in contrasts.values():
        if not np.allclose(np.asarray(weights) @ inverse @ a_full, weights):
            raise ValueError(f"Unidentified contrast: {variant}, {outcome}")
    rng = np.random.default_rng(SEED)
    draws = {term: [] for term in contrasts}
    for _ in range(N_BOOT):
        pick = rng.integers(len(stats), size=len(stats))
        a = xx[pick].sum(0)
        inv = np.linalg.pinv(a)
        b = inv @ xy[pick].sum(0)
        projection = inv @ a
        for term, weights in contrasts.items():
            if np.allclose(np.asarray(weights) @ projection, weights):
                draws[term].append(float(b @ weights))
    rows = []
    for term, weights in contrasts.items():
        weights = np.asarray(weights)
        samples = np.asarray(draws[term])
        lo, hi = np.percentile(samples, [2.5, 97.5])
        # Finite-replicate correction prevents an empirical tail count of zero
        # being represented as an exact zero p-value.
        tail = min(np.sum(samples <= 0), np.sum(samples >= 0))
        rows.append(dict(variant=variant, outcome=outcome, term=term,
                         works=len(d), poets=d.person.nunique(),
                         slope=float(beta @ weights), ci_low=lo, ci_high=hi,
                         boot_p=min(1.0, 2 * (tail + 1) / (len(samples) + 1)),
                         bootstrap_attempted=N_BOOT, bootstrap_valid=len(samples)))
    return rows


def phases(frame, anchors):
    d = frame.copy()
    d["anchor"] = d.episode.map(anchors)
    d["rel_alt"] = d.year - d.anchor
    d["phase_test"] = None
    valid = d.stage_adj.isin(["destination", "elsewhere"]) & d.year.notna() & d.anchor.notna()
    outside = valid & d.stage_adj.eq("elsewhere")
    d.loc[outside & d.rel_alt.le(0), "phase_test"] = "en route"
    inside = valid & d.stage_adj.eq("destination")
    for rel, label in [(0, "arrival year"), (1, "settled +1"), (2, "settled +2-3"), (3, "settled +2-3")]:
        d.loc[inside & d.rel_alt.eq(rel), "phase_test"] = label
    d.loc[inside & d.rel_alt.ge(4), "phase_test"] = "settled +4 or more"
    d.loc[inside & d.rel_alt.lt(0), "phase_test"] = "seat before alternative anchor"
    return d


original = w[w.stage_adj.eq("destination")].groupby("episode").year.min().dropna()
nonexchange = w[w.stage_adj.eq("destination") & w.title_exchange.eq(0)].groupby("episode").year.min().dropna()
anchor_check = pd.concat([original.rename("original_year"), nonexchange.rename("nonexchange_year")], axis=1)
anchor_check["shift_years"] = anchor_check.nonexchange_year - anchor_check.original_year
anchor_check["unchanged"] = anchor_check.shift_years.eq(0)
save(anchor_check.rename_axis("episode").reset_index(), "anchor_comparison")

arrival_rows = []
for variant, frame, anchors in [
    ("original anchor, full sample", w, original),
    ("original anchor, alternative-anchor episodes", w[w.episode.isin(nonexchange.index)], original),
    ("nonexchange anchor", w, nonexchange),
]:
    d = phases(frame, anchors)
    d = d[d.phase_test.isin(["en route", "arrival year"])].copy()
    d["arrival"] = d.phase_test.eq("arrival year").astype(float)
    d = d.groupby("person").filter(lambda g: g.arrival.nunique() == 2)
    arrival_rows += fit(d, "title_exchange", ["arrival", "precise"],
                        {"arrival year vs en route": [1, 0]}, variant)
save(pd.DataFrame(arrival_rows), "arrival_exchange_sensitivity")

temple_rows = []
phase_counts = []
for variant, frame in [("all poets", w), ("without Bai Juyi", w[w.poet.ne("白居易")])]:
    d = phases(frame, original)
    d = d[d.phase_test.isin(["en route"] + PHASES)].copy()
    d = d.groupby("person").filter(lambda g: g.phase_test.eq("en route").any()
                                    and g.phase_test.isin(PHASES).any())
    for phase in ["en route"] + PHASES:
        g = d[d.phase_test.eq(phase)]
        phase_counts.append(dict(variant=variant, phase=phase, works=len(g),
                                 poets=g.person.nunique(), temple_determinate=g.temple_reclusion.notna().sum(),
                                 temple_rate=g.temple_reclusion.mean(),
                                 bai_works=g.poet.eq("白居易").sum()))
    columns = []
    for i, phase in enumerate(PHASES):
        col = f"phase_{i}"
        d[col] = d.phase_test.eq(phase).astype(float)
        columns.append(col)
    for length_control in (False, True):
        xcols = columns + ["precise"] + (["loglen"] if length_control else [])
        contrasts = {phase: [float(j == i) for j in range(len(xcols))]
                     for i, phase in enumerate(PHASES)}
        contrasts["settled +1 minus arrival year"] = [-1, 1] + [0] * (len(xcols) - 2)
        rows = fit(d, "temple_reclusion", xcols, contrasts,
                   variant + (", length-controlled" if length_control else ", dating-controlled"))
        for row in rows:
            support = d[d.phase_test.eq(row["term"])]
            if row["term"] == "settled +1 minus arrival year":
                support = d[d.phase_test.eq("settled +1")]
            row["phase_works"] = len(support)
            row["phase_poets"] = support.person.nunique()
            row["interval_scope"] = "conditional on contrast estimable in bootstrap draw"
        # One family: the four phase-vs-road contrasts and the direct year contrast.
        order = sorted(range(len(rows)), key=lambda i: rows[i]["boot_p"])
        adjusted = 0
        for rank, i in enumerate(order):
            adjusted = max(adjusted, min(1.0, rows[i]["boot_p"] * (len(rows) - rank)))
            rows[i]["holm_p"] = adjusted
        temple_rows += rows
save(pd.DataFrame(temple_rows), "temple_sensitivity")
save(pd.DataFrame(phase_counts), "temple_phase_coverage")

summary = {
    "seed": SEED, "bootstrap_attempted": N_BOOT,
    "anchor_episodes": len(original), "nonexchange_anchor_episodes": len(nonexchange),
    "unchanged_anchor_episodes": int(anchor_check.unchanged.sum()),
    "shifted_anchor_episodes": int(anchor_check.shift_years.gt(0).sum()),
    "missing_nonexchange_anchor_episodes": int(anchor_check.nonexchange_year.isna().sum()),
    "original_next_year_all_poems": int(w.phase.eq("settled +1").sum()),
    "original_next_year_bai_poems": int((w.phase.eq("settled +1") & w.poet.eq("白居易")).sum()),
    "main_coding": "topic1z_work_phase.parquet: stage_adj",
}
with open(os.path.join(R, "topic1s_key_results.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(json.dumps(summary, ensure_ascii=False))
print(pd.DataFrame(arrival_rows).to_string(index=False))
print(pd.DataFrame(temple_rows).query("term == 'settled +1' or term == 'settled +1 minus arrival year'").to_string(index=False))
