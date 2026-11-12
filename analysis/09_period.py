"""Does exile writing change across the Tang, and is the settling pattern the same in each period?

Reads results/topic1z_work_phase.parquet (built by 07_settling_evidence.py) and the poem bodies
in data/texts/exile_period_poem_texts.parquet. No poet spans two periods, so period is a
between-poet comparison: each poet is placed by the start year of their first linked episode
(early <= 755, mid 756-835, late >= 836) and inference is at the poet level.
1. Poet means adjusted for place of writing, years in exile, dating tier and poem length;
   early vs mid by permutation of period labels across poets, trend by poet start year,
   and leave-one-poet-out ranges.
2. The seat contrast (seat vs elsewhere, within poet) estimated separately in each period.
3. Axis rates by arrival phase in each period.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
R = os.environ.get("DH2027_RESULTS_DIR", os.path.join(ROOT, "results"))
SEED = 20271128
N_BOOT = 2000
N_PERM = 5000
MIN_POEMS = 5
AXES = ["court_institution", "landscape", "temple_reclusion", "route_displacement", "time_void"]
OUT = AXES + ["n_axes", "lex_capital", "lex_homeward", "lex_exile_self", "lex_temple", "title_exchange"]
rng = np.random.default_rng(SEED)
key = {}


def save(df, name):
    df.to_csv(f"{R}/{name}.csv", index=False, encoding="utf-8-sig")


w = pd.read_parquet(f"{R}/topic1z_work_phase.parquet")
body = pd.read_parquet(f"{D}/texts/exile_period_poem_texts.parquet").set_index("source_record_id").body
w["loglen"] = np.log(w.source_record_id.map(body).fillna("").str.replace(r"[^一-鿿]", "", regex=True)
                     .str.len().clip(lower=1))
w["n_axes"] = w[AXES].sum(axis=1, min_count=5)
w["poet_start"] = w.person.map(w.groupby("person").episode_start.min())
w["period"] = pd.cut(w.poet_start, [0, 755, 835, 1000], labels=["early", "mid", "late"]).astype(str)
key["poets_by_period"] = {p: sorted(g.poet.unique().tolist()) for p, g in w.groupby("period")}

# ---------------------------------------------------------------- 1. adjusted poet means
stage = w.stage_adj.where(w.stage_adj.isin(["capital", "destination", "elsewhere"]), "unknown")
X = pd.get_dummies(stage, drop_first=True).astype(float)
X["t"] = w.t.fillna(w.t.mean())
X["t_missing"] = w.t.isna().astype(float)
X["precise"] = w.precise
X["loglen"] = w.loglen
X["const"] = 1.0


def poet_means(y):
    m = w[y].notna() & (w.period != "nan")
    Y = w.loc[m, y].values
    beta = np.linalg.lstsq(X[m].values, Y, rcond=None)[0]
    d = pd.DataFrame({"poet": w.loc[m, "poet"], "period": w.loc[m, "period"], "start": w.loc[m, "poet_start"],
                      "adj": Y - X[m].values @ beta + Y.mean(), "raw": Y})
    pm = d.groupby("poet").agg(period=("period", "first"), start=("start", "first"), poems=("adj", "size"),
                               adjusted=("adj", "mean"), raw=("raw", "mean"))
    return pm[pm.poems >= MIN_POEMS]


def mid_minus_early(pm):
    return pm[pm.period == "mid"].adjusted.mean() - pm[pm.period == "early"].adjusted.mean()


rows, per_poet = [], []
for y in OUT:
    pm = poet_means(y)
    per_poet.append(pm.reset_index().assign(outcome=y))
    two = pm[pm.period.isin(["early", "mid"])]
    obs = mid_minus_early(two)
    perm = []
    for _ in range(N_PERM):
        lab = rng.permutation(two.period.values)
        perm.append(two.adjusted[lab == "mid"].mean() - two.adjusted[lab == "early"].mean())
    loo = [mid_minus_early(two.drop(p)) for p in two.index]
    rk = pm.start.rank().values, pm.adjusted.rank().values
    rho = np.corrcoef(*rk)[0, 1]
    rho_perm = [np.corrcoef(rk[0], rng.permutation(rk[1]))[0, 1] for _ in range(N_PERM)]
    rows.append({"outcome": y, **{f"{p}_poets": int((pm.period == p).sum()) for p in ("early", "mid", "late")},
                 **{f"{p}_adjusted": pm[pm.period == p].adjusted.mean() for p in ("early", "mid", "late")},
                 **{f"{p}_raw": pm[pm.period == p].raw.mean() for p in ("early", "mid", "late")},
                 "mid_minus_early": obs, "perm_p": float((np.abs(perm) >= abs(obs)).mean()),
                 "loo_min": min(loo), "loo_max": max(loo),
                 "spearman_start_year": rho, "spearman_perm_p": float((np.abs(rho_perm) >= abs(rho)).mean())})
res = pd.DataFrame(rows)
p = res.perm_p.sort_values()
res.loc[p.index, "holm_p"] = (p * (len(p) - np.arange(len(p)))).cummax().clip(upper=1).values
save(res, "topic1v_period_poet_level")
save(pd.concat(per_poet), "topic1v_period_poet_means")


# ---------------------------------------------------------------- 2. seat contrast within each period
def person_stats(df, y, xcols):
    d = df[[y, "person"] + xcols].dropna()
    st = []
    for _, g in d.groupby("person"):
        Xg = g[xcols].values - g[xcols].values.mean(0)
        st.append((Xg.T @ Xg, Xg.T @ (g[y].values - g[y].values.mean())))
    return st, d


seat = w[w.stage_adj.isin(["destination", "elsewhere"])].assign(dest=lambda x: (x.stage_adj == "destination").astype(float))
seat = seat.groupby("person").filter(lambda g: g.dest.nunique() == 2)
srows = []
for per, g in seat.groupby("period"):
    for y in AXES:
        # dating tier is constant within most early poets, so the per-period contrast is unadjusted
        st, d = person_stats(g, y, ["dest"])
        XX, XY = np.array([s[0] for s in st]), np.array([s[1] for s in st])
        if d.person.nunique() < 3 or np.linalg.matrix_rank(XX.sum(0)) < 1:
            continue
        b = np.linalg.solve(XX.sum(0), XY.sum(0))[0]
        bs = []
        for _ in range(N_BOOT):
            pick = rng.integers(0, len(st), len(st))
            if np.linalg.matrix_rank(XX[pick].sum(0)) == 1:
                bs.append(np.linalg.solve(XX[pick].sum(0), XY[pick].sum(0))[0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        srows.append({"period": per, "axis": y, "poets": d.person.nunique(), "works": len(d),
                      "poets_list": "|".join(sorted(g.poet.unique())), "seat_vs_elsewhere": b, "ci_low": lo, "ci_high": hi})
save(pd.DataFrame(srows), "topic1v_seat_contrast_by_period")

# ---------------------------------------------------------------- 3. arrival phases within each period
ph = (w[w.phase.notna()].groupby(["period", "phase"])
      .agg(works=("poet", "size"), poets=("poet", "nunique"), **{ax: (ax, "mean") for ax in AXES}).reset_index())
save(ph, "topic1v_phase_by_period")

with open(f"{R}/topic1v_key_results.json", "w", encoding="utf-8") as f:
    json.dump(key, f, ensure_ascii=False, indent=2)
print(json.dumps(key, ensure_ascii=False))
