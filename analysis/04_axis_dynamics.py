"""How do the five axes trade off against each other over exile time and distance,
and do individual poets follow the same pattern?

Reads results/topic1x_work_table.parquet (built by 03_explore_topic1.py) and the
author names in data/tang_exile/work_terminal.parquet.
1. Joint person-fixed-effect model: axis ~ years in exile + distance from Chang'an.
2. Co-occurrence by exile time: how landscape is paired with route / time / court.
3. Per-poet year-by-year profiles and within-poet slopes for poets with >= 3 dated years.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
R = os.environ.get("DH2027_RESULTS_DIR", os.path.join(ROOT, "results"))
SEED = 20271128
N_BOOT = 2000
AXES = ["court_institution", "landscape", "temple_reclusion", "route_displacement", "time_void"]
rng = np.random.default_rng(SEED)
key = {}


def save(df, name):
    df.to_csv(f"{R}/{name}.csv", index=False, encoding="utf-8-sig")


w = pd.read_parquet(f"{R}/topic1x_work_table.parquet")
names = (pd.read_parquet(f"{D}/tang_exile/work_terminal.parquet")
         .drop_duplicates("cbdb_person_id").set_index("cbdb_person_id").normalized_author)
w["poet"] = w.person.map(names)
w["t"] = w.years_in_exile.clip(lower=0, upper=10)
w["d500"] = w.dist_km / 500


# ---------------------------------------------------------------- 1. joint within-poet model
def fe_ols(df, ycol, xcols):
    d = df[[ycol, "person"] + xcols].dropna()
    dm = d[[ycol] + xcols] - d.groupby("person")[[ycol] + xcols].transform("mean")
    X, Y = dm[xcols].values, dm[ycol].values
    if np.linalg.matrix_rank(X) < len(xcols):
        return None, d
    return np.linalg.lstsq(X, Y, rcond=None)[0], d


LABELS = {"t": "per year in exile", "d500": "per 500 km from Chang'an", "precise": "precise vs annual-proxy date"}


def fit(df, model, xcols):
    out = []
    for ax in AXES + ["n_axes"]:
        b, d = fe_ols(df, ax, xcols)
        if b is None or d.person.nunique() < 5:
            continue
        groups = {p: g for p, g in d.groupby("person")}
        persons = list(groups)
        bs = []
        for _ in range(N_BOOT):
            pick = rng.choice(persons, len(persons))
            bd = pd.concat([groups[p].assign(person=f"{p}_{i}") for i, p in enumerate(pick)])
            bb, _ = fe_ols(bd, ax, xcols)
            if bb is not None:
                bs.append(bb)
        bs = np.array(bs)
        for j, x in enumerate(xcols):
            lo, hi = np.percentile(bs[:, j], [2.5, 97.5])
            out.append({"model": model, "axis": ax, "term": LABELS[x], "works": len(d), "poets": d.person.nunique(),
                        "slope": b[j], "ci_low": lo, "ci_high": hi,
                        "boot_p": 2 * min((bs[:, j] <= 0).mean(), (bs[:, j] >= 0).mean())})
    out = pd.DataFrame(out)
    m = out.axis != "n_axes"
    rank = out.loc[m, "boot_p"].rank(method="first")
    holm = (out.loc[m, "boot_p"] * (m.sum() - rank + 1)).clip(upper=1)
    out.loc[m, "holm_p"] = holm.loc[out.loc[m].sort_values("boot_p").index].cummax().reindex(holm.index)
    return out


# Later exile years are dominated by annual-proxy dates, which pile up on episode boundary
# years, so the time slope is re-estimated with the dating tier controlled and on subsets.
ep_end = pd.read_parquet(f"{D}/tang_exile/episode_terminal.parquet").active_end_year_max
w["precise"] = w.current_precise_exile_period_work.astype(float)
w["boundary_year"] = (w.years_in_exile == 0) | (w.year == w.episode.map(ep_end))
save(pd.crosstab([w.precision, w.boundary_year], w.years_in_exile.clip(upper=6)).reset_index(), "topic1y_dating_tier_by_exile_year")
joint = pd.concat([
    fit(w, "time + distance", ["t", "d500"]),
    fit(w, "time + distance + dating tier", ["t", "d500", "precise"]),
    fit(w, "time + dating tier (all dated works)", ["t", "precise"]),
    fit(w[w.current_precise_exile_period_work], "time only, precise dates only", ["t"]),
    fit(w[~w.boundary_year], "time only, interior years only", ["t"]),
])
save(joint, "topic1y_joint_time_distance_model")

# why time and distance must be modelled together: early-exile poems are written closer to the capital
td = w.dropna(subset=["dist_km", "years_in_exile"])
tdc = td.assign(dc=td.dist_km - td.groupby("person").dist_km.transform("mean"),
                tc=td.t - td.groupby("person").t.transform("mean"))
key["time_distance_correlation"] = {
    "works": len(td), "poets": int(td.person.nunique()),
    "spearman_pooled": float(td[["dist_km", "years_in_exile"]].corr(method="spearman").iloc[0, 1]),
    "pearson_within_poet": float(tdc[["dc", "tc"]].corr().iloc[0, 1])}
save(td.groupby("exile_time_bin", observed=True).dist_km.agg(works="size", median_km="median", mean_km="mean").reset_index(),
     "topic1y_distance_by_exile_time")

# ---------------------------------------------------------------- 2. how axes pair up over exile time
full = w.dropna(subset=AXES)
rows = []
for var in ("exile_time_bin", "dist_bin"):
    for level, g in full.groupby(var, observed=True):
        land = g[g.landscape == 1]
        rows.append({"covariate": var, "level": str(level), "works": len(g), "poets": g.person.nunique(),
                     "landscape_share": g.landscape.mean(),
                     "landscape_with_route": (land.route_displacement == 1).mean(),
                     "landscape_with_time_void": (land.time_void == 1).mean(),
                     "landscape_with_court": (land.court_institution == 1).mean(),
                     "landscape_with_temple": (land.temple_reclusion == 1).mean(),
                     "landscape_alone": (land[AXES].sum(axis=1) == 1).mean(),
                     "court_with_route": (g[g.court_institution == 1].route_displacement == 1).mean(),
                     "mean_axes": g[AXES].sum(axis=1).mean()})
save(pd.DataFrame(rows), "topic1y_pairing_by_time_and_distance")

full = full.assign(profile=full[AXES].apply(lambda r: "+".join(a.split("_")[0] for a in AXES if r[a]) or "(none)", axis=1))
top = full.profile.value_counts().head(8).index
prof = (full[full.exile_time_bin.notna()].assign(profile=lambda x: x.profile.where(x.profile.isin(top), "other"))
        .groupby(["exile_time_bin", "profile"], observed=True).size().unstack(fill_value=0))
save((prof.div(prof.sum(axis=1), axis=0)).reset_index().astype({"exile_time_bin": str}), "topic1y_profile_mix_by_exile_time")

# ---------------------------------------------------------------- 3. individual poets
dated = w.dropna(subset=["t"])
poets = dated.groupby("person").filter(lambda g: g.t.nunique() >= 3 and len(g) >= 30)
traj, pslope = [], []
for p, g in poets.groupby("person"):
    for t, gg in g.groupby(g.t.clip(upper=6)):
        traj.append({"poet": names.get(p), "years_in_exile": int(t), "works": len(gg),
                     **{ax: gg[ax].mean() for ax in AXES}, "mean_axes": gg[AXES].sum(axis=1, min_count=5).mean()})
    for ax in AXES + ["n_axes"]:
        s = g[["t", ax]].dropna()
        slope = np.polyfit(s.t, s[ax], 1)[0] if s.t.nunique() > 1 else np.nan
        pslope.append({"poet": names.get(p), "works": len(s), "years_observed": s.t.nunique(),
                       "max_years": s.t.max(), "axis": ax, "slope_per_year": slope})
save(pd.DataFrame(traj), "topic1y_poet_year_profiles")
pslope = pd.DataFrame(pslope)
wide = pslope.pivot_table(index="poet", columns="axis", values="slope_per_year")
meta = poets.groupby(poets.person.map(names)).agg(works=("t", "size"), years_observed=("t", "nunique"),
                                                   max_years=("t", "max"), precise_share=("precise", "mean"))
save(meta.join(wide).reset_index(names="poet"), "topic1y_poet_slopes")
cons = pslope.groupby("axis").slope_per_year.agg(poets="count", negative=lambda s: int((s < 0).sum()),
                                                  positive=lambda s: int((s > 0).sum()), median="median").reset_index()
cons["sign_test_p_two_sided"] = [float(binomtest(int(max(n, p)), int(n + p), 0.5).pvalue) if n + p else np.nan
                                 for n, p in zip(cons.negative, cons.positive)]
save(cons, "topic1y_poet_direction_consistency")
key["poets_with_3plus_years"] = sorted(poets.person.map(names).unique().tolist())

with open(f"{R}/topic1y_key_results.json", "w", encoding="utf-8") as f:
    json.dump(key, f, ensure_ascii=False, indent=2)
print(json.dumps(key, ensure_ascii=False))
