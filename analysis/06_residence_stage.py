"""Was the poem written at the place of exile? Stage analysis from the DB's episode destination.

Reads results/topic1x_work_table.parquet (built by 03_explore_topic1.py) and the accepted
episode destination text (active_place_text) in data/tang_exile/episode_terminal.parquet.
Each linked episode's destination is mapped by hand to Literary Map region-code prefixes
(DESTINATIONS below), and each work with a reviewed composition place is classed as
- capital:     written in Chang'an (CN6101), i.e. before departure;
- destination: composition code falls under the destination prefix;
- elsewhere:   any other place (on the road, side journeys, places after leaving);
- coarse:      composition code is an ancestor of the destination code, so cannot be told apart.
1. Coverage and stage by years in exile.
2. Axis prevalence by stage, and the landscape pairing by stage.
3. Within-poet models: stage contrast, and time in exile with and without stage controlled.
4. Per-poet stage profiles, and a sensitivity run on crosswalk-verified destinations only.
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
AXES = ["court_institution", "landscape", "temple_reclusion", "route_displacement", "time_void"]
rng = np.random.default_rng(SEED)
key = {}


def save(df, name):
    df.to_csv(f"{R}/{name}.csv", index=False, encoding="utf-8-sig")


# episode row -> (destination code prefixes, basis, note).
# crosswalk_name: the DB destination name appears among the M6-146 crosswalk names of that code.
# admin_code_inference: code inferred from the modern successor of the Tang seat; needs checking.
# Episodes whose works carry no reviewed place (杜甫 華州, 宋之問 瀧州, 岑參, ...) are not listed.
DESTINATIONS = {
    0: (["CN421"], "crosswalk_name", "荊州"),
    1: (["CN4451"], "crosswalk_name", "潮州"),
    4: (["CN4311"], "crosswalk_name", "永州"),
    9: (["CN4451", "CN46"], "admin_code_inference", "潮州 then 崖州 (Hainan)"),
    24: (["CN331082"], "admin_code_inference", "臨海"),
    28: (["CN4306"], "crosswalk_name", "岳州"),
    32: (["CN4503", "CN4507"], "admin_code_inference", "桂州 (crosswalk) and 欽州 (inferred)"),
    42: (["VN130101"], "admin_code_inference", "驩州; Vietnamese code not in crosswalk, site of most placed works"),
    43: (["CN3604"], "crosswalk_name", "江州"),
    44: (["CN3604"], "crosswalk_name", "江州"),
    45: (["CN421"], "crosswalk_name", "江陵 (DB text gives CN421)"),
    46: (["CN5203"], "admin_code_inference", "夜郎 (珍州); pardoned before arrival"),
    60: (["CN4307"], "crosswalk_name", "朗州"),
    65: (["CN431281"], "admin_code_inference", "龍標 (today 洪江 黔城)"),
    76: (["CN441204"], "admin_code_inference", "端州, seat at 高要"),
    85: (["CN5109"], "admin_code_inference", "長江縣, 遂州"),
    89: (["CN6107"], "crosswalk_name", "漢中 (興元府)"),
    90: (["CN371726"], "admin_code_inference", "濮州, seat at 鄄城"),
}
UNMAPPED = {
    19: "DB destination text missing",
    59: "DB destination text names a waypoint (干越亭), not the exile seat",
}
CAPITAL = "CN6101"

w = pd.read_parquet(f"{R}/topic1x_work_table.parquet")
ep = pd.read_parquet(f"{D}/tang_exile/episode_terminal.parquet")
names = (pd.read_parquet(f"{D}/tang_exile/work_terminal.parquet")
         .drop_duplicates("cbdb_person_id").set_index("cbdb_person_id").normalized_author)
w["poet"] = w.person.map(names)
w["t"] = w.years_in_exile.clip(lower=0, upper=10)
w["precise"] = w.current_precise_exile_period_work.astype(float)

dmap = pd.DataFrame([{"episode": i, "poet": ep.normalized_author[i], "db_destination_text": ep.active_place_text[i],
                      "start_year": ep.active_start_year_min[i], "end_year": ep.active_end_year_max[i],
                      "destination_codes": "|".join(c), "basis": b, "note": n} for i, (c, b, n) in DESTINATIONS.items()]
                    + [{"episode": i, "poet": ep.normalized_author[i], "db_destination_text": ep.active_place_text[i],
                        "start_year": ep.active_start_year_min[i], "end_year": ep.active_end_year_max[i],
                        "destination_codes": "", "basis": "unmapped", "note": n} for i, n in UNMAPPED.items()])


def stage(row):
    if pd.isna(row.episode):
        return "no_episode"
    if not isinstance(row.place_code, str) or not row.place_code:
        return "no_reviewed_place"
    if row.place_code == CAPITAL:
        return "capital"
    if int(row.episode) not in DESTINATIONS:
        return "destination_unmapped"
    codes = DESTINATIONS[int(row.episode)][0]
    if any(row.place_code.startswith(c) for c in codes):
        return "destination"
    if any(c.startswith(row.place_code) for c in codes):
        return "coarse"
    return "elsewhere"


w["stage"] = w.apply(stage, axis=1)
w["basis"] = w.episode.map(lambda e: DESTINATIONS[int(e)][1] if pd.notna(e) and int(e) in DESTINATIONS else None)
ana = w[w.stage.isin(["capital", "destination", "elsewhere"])].copy()
dmap = dmap.merge(ana.groupby("episode").stage.value_counts().unstack(fill_value=0).reset_index(), how="left", on="episode")
save(dmap.sort_values("episode"), "topic1z_destination_map")
w.drop(columns=["poet"]).to_parquet(f"{R}/topic1z_work_stage.parquet", index=False)

# ---------------------------------------------------------------- 1. coverage
cov = w.groupby("stage").agg(works=("poet", "size"), poets=("poet", "nunique"), episodes=("episode", "nunique")).reset_index()
save(cov, "topic1z_stage_coverage")
key["coverage"] = {s: int(n) for s, n in zip(cov.stage, cov.works)}
key["analysed"] = {"works": len(ana), "poets": int(ana.person.nunique()), "episodes": int(ana.episode.nunique()),
                   "poets_at_destination": int(ana[ana.stage == "destination"].person.nunique())}
save(pd.crosstab(ana.stage, ana.years_in_exile.clip(upper=6)).reset_index(), "topic1z_stage_by_exile_year")


# ---------------------------------------------------------------- 2. axis prevalence and pairing by stage
def cluster_ci(g, col):
    groups = [x[col].dropna().values for _, x in g.groupby("person")]
    groups = [x for x in groups if len(x)]
    bs = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        v = np.concatenate([groups[i] for i in pick])
        bs.append(v.mean())
    return np.percentile(bs, [2.5, 97.5])


rows = []
for s, g in ana.groupby("stage"):
    land = g[g.landscape == 1]
    r = {"stage": s, "works": len(g), "poets": g.person.nunique(),
         "mean_axes": g[AXES].sum(axis=1, min_count=5).mean(),
         "landscape_with_route": (land.route_displacement == 1).mean(),
         "landscape_with_court": (land.court_institution == 1).mean(),
         "landscape_with_temple": (land.temple_reclusion == 1).mean(),
         "landscape_with_time_void": (land.time_void == 1).mean()}
    for ax in AXES:
        lo, hi = cluster_ci(g, ax)
        r.update({ax: g[ax].mean(), f"{ax}_ci_low": lo, f"{ax}_ci_high": hi})
    rows.append(r)
save(pd.DataFrame(rows), "topic1z_axis_by_stage")


# ---------------------------------------------------------------- 3. within-poet models
def fe_ols(df, ycol, xcols):
    d = df[[ycol, "person"] + xcols].dropna()
    dm = d[[ycol] + xcols] - d.groupby("person")[[ycol] + xcols].transform("mean")
    X, Y = dm[xcols].values, dm[ycol].values
    if len(d) == 0 or np.linalg.matrix_rank(X) < len(xcols):
        return None, d
    return np.linalg.lstsq(X, Y, rcond=None)[0], d


LABELS = {"t": "per year in exile", "dest": "destination vs elsewhere", "cap": "capital vs elsewhere",
          "precise": "precise vs annual-proxy date"}


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
    for term, idx in out.groupby("term").groups.items():
        m = out.index.isin(idx) & (out.axis != "n_axes")
        p = out.loc[m, "boot_p"].sort_values()
        holm = (p * (len(p) - np.arange(len(p)))).cummax().clip(upper=1)
        out.loc[holm.index, "holm_p"] = holm
    return out


ana["dest"] = (ana.stage == "destination").astype(float)
ana["cap"] = (ana.stage == "capital").astype(float)
# poets observed both at the destination and away from it carry the within-poet stage contrast
both = ana[ana.stage != "capital"].groupby("person").filter(lambda g: g.dest.nunique() == 2)
verified = ana[ana.basis == "crosswalk_name"]
vboth = verified[verified.stage != "capital"].groupby("person").filter(lambda g: g.dest.nunique() == 2)
models = pd.concat([
    fit(ana, "stage (capital, destination vs elsewhere)", ["dest", "cap"]),
    fit(both, "destination vs elsewhere, poets seen in both", ["dest"]),
    fit(both, "time only, poets seen in both", ["t", "precise"]),
    fit(both, "time + destination, poets seen in both", ["t", "dest", "precise"]),
    fit(ana[ana.stage == "destination"], "time only, destination works only", ["t", "precise"]),
    fit(vboth, "destination vs elsewhere, crosswalk-verified destinations", ["dest"]),
])
save(models, "topic1z_within_poet_stage_models")
key["poets_seen_in_both"] = sorted(both.poet.unique().tolist())
key["works_seen_in_both"] = len(both)

# ---------------------------------------------------------------- 4. per-poet stage profiles
prof = (ana.groupby(["poet", "stage"])
        .agg(works=("t", "size"), median_years_in_exile=("t", "median"), **{ax: (ax, "mean") for ax in AXES})
        .reset_index())
save(prof, "topic1z_poet_stage_profiles")
diff = prof.pivot_table(index="poet", columns="stage", values=AXES + ["works"])
dirs = []
for ax in AXES:
    if ("destination" in diff[ax]) and ("elsewhere" in diff[ax]):
        dd = (diff[ax]["destination"] - diff[ax]["elsewhere"]).dropna()
        n = diff["works"].loc[dd.index]
        dd = dd[(n["destination"] >= 5) & (n["elsewhere"] >= 5)]
        dirs.append({"axis": ax, "poets": len(dd), "higher_at_destination": int((dd > 0).sum()),
                     "lower_at_destination": int((dd < 0).sum()), "median_difference": dd.median()})
save(pd.DataFrame(dirs), "topic1z_poet_direction_by_stage")

with open(f"{R}/topic1z_key_results.json", "w", encoding="utf-8") as f:
    json.dump(key, f, ensure_ascii=False, indent=2)
print(json.dumps(key, ensure_ascii=False))
