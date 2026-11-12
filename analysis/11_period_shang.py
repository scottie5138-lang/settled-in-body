"""Sensitivity check: the period comparison of 09_period.py under Shang Yongliang's periodisation.

Shang (2007a, 47-48) dates the Tang periods as early 618-713, high 713-766, mid 766-827 and
late 827-907. Here a poet is placed by the start year of their first linked episode, as in
09_period.py: early <= 712, high 713-765, mid 766-826, late >= 827. Everything else is the
same as 09: poet means adjusted for place of writing, years in exile, dating tier and poem
length, poets with at least 5 poems, inference at the poet level. The rank correlation with
the poet's start year does not depend on the cut points and is reported only for reference.
1. Adjusted poet means by period; permutation tests for adjacent periods and for
   early+high vs mid (Shang's split nearest to the 09 early vs mid contrast).
2. The seat contrast (seat vs elsewhere, within poet) estimated separately in each period.
3. Axis and word rates by arrival phase in each period.
Writes results/topic1v_shang_*; it does not change any abstract number.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
R = os.environ.get("DH2027_RESULTS_DIR", os.path.join(ROOT, "results"))
SEED = 20271031
N_BOOT = 2000
N_PERM = 5000
MIN_POEMS = 5
AXES = ["court_institution", "landscape", "temple_reclusion", "route_displacement", "time_void"]
OUT = AXES + ["n_axes", "lex_capital", "lex_homeward", "lex_exile_self", "lex_temple", "title_exchange"]
PERIODS = ["early", "high", "mid", "late"]
CONTRASTS = [("early", "high"), ("high", "mid"), ("mid", "late"), ("early+high", "mid")]
rng = np.random.default_rng(SEED)
key = {"cuts": "early <= 712, high 713-765, mid 766-826, late >= 827 (Shang 2007a, 47-48)"}


def save(df, name):
    df.to_csv(f"{R}/{name}.csv", index=False, encoding="utf-8-sig")


w = pd.read_parquet(f"{R}/topic1z_work_phase.parquet")
body = pd.read_parquet(f"{D}/texts/exile_period_poem_texts.parquet").set_index("source_record_id").body
w["loglen"] = np.log(w.source_record_id.map(body).fillna("").str.replace(r"[^一-鿿]", "", regex=True)
                     .str.len().clip(lower=1))
w["n_axes"] = w[AXES].sum(axis=1, min_count=5)
w["poet_start"] = w.person.map(w.groupby("person").episode_start.min())
w["period"] = pd.cut(w.poet_start, [0, 712, 765, 826, 1000], labels=PERIODS).astype(str)
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


def group(pm, label):
    return pm[pm.period.isin(label.split("+"))]


def diff(pm, a, b):
    return group(pm, b).adjusted.mean() - group(pm, a).adjusted.mean()


rows, per_poet, crow = [], [], []
for y in OUT:
    pm = poet_means(y)
    per_poet.append(pm.reset_index().assign(outcome=y))
    rk = pm.start.rank().values, pm.adjusted.rank().values
    rho = np.corrcoef(*rk)[0, 1]
    rho_perm = [np.corrcoef(rk[0], rng.permutation(rk[1]))[0, 1] for _ in range(N_PERM)]
    rows.append({"outcome": y, **{f"{p}_poets": int((pm.period == p).sum()) for p in PERIODS},
                 **{f"{p}_adjusted": pm[pm.period == p].adjusted.mean() for p in PERIODS},
                 **{f"{p}_raw": pm[pm.period == p].raw.mean() for p in PERIODS},
                 "spearman_start_year": rho, "spearman_perm_p": float((np.abs(rho_perm) >= abs(rho)).mean())})
    for a, b in CONTRASTS:
        two = pd.concat([group(pm, a).assign(side=a), group(pm, b).assign(side=b)])
        obs = diff(pm, a, b)
        perm = []
        for _ in range(N_PERM):
            lab = rng.permutation(two.side.values)
            perm.append(two.adjusted[lab == b].mean() - two.adjusted[lab == a].mean())
        loo = [diff(pm.drop(p), a, b) for p in two.index]
        crow.append({"outcome": y, "contrast": f"{b} minus {a}", "poets_a": int((two.side == a).sum()),
                     "poets_b": int((two.side == b).sum()), "difference": obs,
                     "perm_p": float((np.abs(perm) >= abs(obs)).mean()), "loo_min": min(loo), "loo_max": max(loo)})
save(pd.DataFrame(rows), "topic1v_shang_period_poet_level")
con = pd.DataFrame(crow)
for c, g in con.groupby("contrast"):
    p = g.perm_p.sort_values()
    con.loc[p.index, "holm_p"] = (p * (len(p) - np.arange(len(p)))).cummax().clip(upper=1).values
save(con, "topic1v_shang_period_contrasts")
save(pd.concat(per_poet), "topic1v_shang_period_poet_means")


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
    for y in AXES + ["lex_capital", "lex_homeward"]:
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
        srows.append({"period": per, "outcome": y, "poets": d.person.nunique(), "works": len(d),
                      "poets_list": "|".join(sorted(g.poet.unique())), "seat_vs_elsewhere": b, "ci_low": lo, "ci_high": hi})
save(pd.DataFrame(srows), "topic1v_shang_seat_contrast_by_period")

# ---------------------------------------------------------------- 3. arrival phases within each period
ph = (w[w.phase.notna()].groupby(["period", "phase"])
      .agg(works=("poet", "size"), poets=("poet", "nunique"),
           **{c: (c, "mean") for c in AXES + ["lex_capital", "lex_homeward", "title_exchange"]}).reset_index())
save(ph, "topic1v_shang_phase_by_period")

# ---------------------------------------------------------------- 4. continuous view: the 749 cohort and the late period
# 749 is the first exile of Wang Changling, the earliest poet with a High Tang level of exchange titles; the cut
# is read off the data, so its permutation p is descriptive only. The late Tang has four poets, so its mean is
# reported with the leave-one-poet-out range.
pm_all = pd.concat(per_poet)
hrows = []
for y in ["lex_capital", "title_exchange"]:
    pm = pm_all[pm_all.outcome == y].set_index("poet")
    side = np.where(pm.start >= 749, "from 749", "before 749")
    a, b = pm.adjusted[side == "before 749"], pm.adjusted[side == "from 749"]
    perm = []
    for _ in range(N_PERM):
        lab = rng.permutation(side)
        perm.append(pm.adjusted[lab == "from 749"].mean() - pm.adjusted[lab == "before 749"].mean())
    obs = b.mean() - a.mean()
    later = pm[pm.start >= 749]
    rk = later.start.rank().values, later.adjusted.rank().values
    rho = np.corrcoef(*rk)[0, 1]
    rho_perm = [np.corrcoef(rk[0], rng.permutation(rk[1]))[0, 1] for _ in range(N_PERM)]
    late = pm[pm.period == "late"].adjusted
    loo = {p: late.drop(p).mean() for p in late.index}
    hrows.append({"outcome": y, "poets_before_749": len(a), "mean_before_749": a.mean(), "median_before_749": a.median(),
                  "poets_from_749": len(b), "mean_from_749": b.mean(), "median_from_749": b.median(),
                  "difference": obs, "perm_p_descriptive": float((np.abs(perm) >= abs(obs)).mean()),
                  "spearman_within_from_749": rho, "spearman_within_perm_p": float((np.abs(rho_perm) >= abs(rho)).mean()),
                  "late_mean": late.mean(), "late_loo_min": min(loo.values()), "late_loo_max": max(loo.values()),
                  "late_poet_with_largest_effect": max(loo, key=lambda p: abs(loo[p] - late.mean()))})
save(pd.DataFrame(hrows), "topic1v_shang_cohort_749")

with open(f"{R}/topic1v_shang_key_results.json", "w", encoding="utf-8") as f:
    json.dump(key, f, ensure_ascii=False, indent=2)
print(json.dumps(key, ensure_ascii=False))
