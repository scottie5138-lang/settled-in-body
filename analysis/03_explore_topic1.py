"""Topic 1 extension: link the five-axis profile of the 1,201 exile-period works to DB
evidence on when and where each poem was written and which exile episode it belongs to.

Covariates (all from reviewed DB chains, candidate-only for analysis purposes):
- distance of the reviewed composition place from Chang'an, via Literary Map region codes
  and the M6-146 place crosswalks (code prefix fallback, recorded per work);
- years since the start of the linked accepted episode;
- era of the reviewed work year;
- date precision tier (precise vs annual proxy).

Estimates are work-level shares with person-cluster bootstrap CIs, plus a
person-fixed-effect linear probability slope (within-author contrast).
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
CHANGAN = (34.2658, 108.9541)
rng = np.random.default_rng(SEED)
key = {}
os.makedirs(R, exist_ok=True)


def save(df, name):
    df.to_csv(f"{R}/{name}.csv", index=False, encoding="utf-8-sig")


def haversine_km(lat, lon, lat0, lon0):
    lat, lon, lat0, lon0 = map(np.radians, (lat, lon, lat0, lon0))
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------- build the work table
axis = pd.read_parquet(f"{D}/five_axis/axis_view6005_flat.parquet")
axis["value"] = axis.human_status.fillna(axis.working_status)
axis = axis[axis.value.isin(["present", "absent"])]
y = axis.assign(v=(axis.value == "present").astype(float)).pivot_table(
    index="source_record_id", columns="field_id", values="v", aggfunc="first")

screen = pd.read_parquet(f"{D}/tang_exile/full_population_m6_146p_screen_v01.parquet")
screen = screen[screen.source_record_id.isin(y.index)].set_index("source_record_id")
w = y.join(screen[["cbdb_person_id", "current_precise_exile_period_work", "reviewed_work_start_year",
                   "reviewed_work_end_year", "reviewed_place_state", "reviewed_place_text", "current_episode_ids_json"]])
w = w.rename(columns={"cbdb_person_id": "person", "reviewed_work_start_year": "year"})
key["works"] = len(w)

# place: first listed region code; fall back to the longest crosswalked prefix
xw = pd.concat([pd.read_parquet(f"{D}/tang_exile/{n}.parquet")[["cnkgraph_region_code", "cnkgraph_latitude", "cnkgraph_longitude"]]
                for n in ("full_population_m6_146e1_cnkgraph_cbdb_chgis_tgaz_place_crosswalk_v01",
                          "full_population_m6_146f_lane_p_cnkgraph_nonpolicy_place_crosswalk_v01")])
xw = xw.dropna().drop_duplicates("cnkgraph_region_code").set_index("cnkgraph_region_code")


def resolve(text):
    if not isinstance(text, str):
        return (None, None, None, "no_reviewed_place")
    code = text.replace("|", " ").split()[0].split(",")[0].strip() if text.strip(" |") else ""
    for k in range(len(code), 3, -1):
        if code[:k] in xw.index:
            r = xw.loc[code[:k]]
            return (code, r.cnkgraph_latitude, r.cnkgraph_longitude, "exact_code" if k == len(code) else "prefix_fallback")
    return (code, None, None, "code_without_coordinates")


res = w.reviewed_place_text.map(resolve)
w["place_code"], w["lat"], w["lon"], w["place_resolution"] = zip(*res)
w["lat"] = pd.to_numeric(w["lat"])
w["lon"] = pd.to_numeric(w["lon"])
w["dist_km"] = haversine_km(w.lat, w.lon, *CHANGAN)
save(w.place_resolution.value_counts().rename_axis("place_resolution").reset_index(name="works"), "topic1x_place_resolution")

# episode: the linked accepted episode whose interval contains the work year, else the latest one starting before it
ep = pd.read_parquet(f"{D}/tang_exile/episode_terminal.parquet")
ep_ids = {}
for c in ("terminal_event_id", "accepted_episode_union_id", "source_episode_id", "episode_continuity_id"):
    ep_ids.update({i: r for i, r in zip(ep[c], ep.index) if isinstance(i, str)})


def pick_episode(row):
    rows = sorted({ep_ids[i] for i in json.loads(row.current_episode_ids_json or "[]") if i in ep_ids})
    if not rows:
        return (None, np.nan, "no_linked_episode")
    cand = ep.loc[rows]
    if len(cand) == 1:
        return (cand.index[0], cand.active_start_year_min.iloc[0], "single_linked_episode")
    if pd.notna(row.year):
        inside = cand[(cand.active_start_year_min <= row.year) & (cand.active_end_year_max.fillna(9999) >= row.year)]
        if len(inside) == 1:
            return (inside.index[0], inside.active_start_year_min.iloc[0], "year_inside_one_episode")
        before = cand[cand.active_start_year_min <= row.year].sort_values("active_start_year_min")
        if len(before):
            return (before.index[-1], before.active_start_year_min.iloc[-1], "latest_episode_started_before_year")
    return (None, np.nan, "ambiguous_multi_episode")


pe = w.apply(pick_episode, axis=1)
w["episode"], w["episode_start"], w["episode_resolution"] = zip(*pe)
w["years_in_exile"] = w.year - w.episode_start
save(w.episode_resolution.value_counts().rename_axis("episode_resolution").reset_index(name="works"), "topic1x_episode_resolution")

w["dist_bin"] = pd.cut(w.dist_km, [-1, 300, 1000, 1500, 5000], labels=["<300 km", "300-999 km", "1000-1499 km", ">=1500 km"])
w["exile_time_bin"] = pd.cut(w.years_in_exile, [-100, 0, 1, 3, 100], labels=["year 0", "year 1", "years 2-3", "years 4+"])
w["era"] = pd.cut(w.year, [0, 755, 835, 1000], labels=["to 755", "756-835", "836+"])
w["precision"] = np.where(w.current_precise_exile_period_work, "precise", "annual_proxy_or_other")
w["n_axes"] = w[AXES].sum(axis=1, min_count=5)
w.reset_index().drop(columns=["current_episode_ids_json"]).to_parquet(f"{R}/topic1x_work_table.parquet", index=False)
key["coverage"] = {"with_year": int(w.year.notna().sum()), "with_coordinates": int(w.dist_km.notna().sum()),
                   "with_episode": int(w.episode.notna().sum()), "with_years_in_exile": int(w.years_in_exile.notna().sum()),
                   "persons": int(w.person.nunique()), "episodes": int(w.episode.nunique())}


# ---------------------------------------------------------------- shares by covariate bin
def cluster_ci(df, col):
    groups = [g[col].values for _, g in df.groupby("person")]
    vals = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        v = np.concatenate([groups[i] for i in pick])
        vals.append(np.nanmean(v))
    return np.nanpercentile(vals, [2.5, 97.5])


def by_bin(var):
    out = []
    for level, g in w.groupby(var, observed=True):
        for ax in AXES + ["n_axes"]:
            s = g[[ax, "person"]].dropna()
            lo, hi = cluster_ci(s, ax)
            out.append({"covariate": var, "level": str(level), "axis": ax, "works": len(s),
                        "persons": s.person.nunique(), "share_or_mean": s[ax].mean(), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(out)


bins = pd.concat([by_bin(v) for v in ("dist_bin", "exile_time_bin", "era", "precision")])
save(bins, "topic1x_axis_by_covariate")


# ---------------------------------------------------------------- within-author slopes
def fe_slope(df, ax, x):
    d = df[[ax, x, "person"]].dropna()
    d = d[d.groupby("person")[x].transform("nunique") > 1]
    if d.person.nunique() < 5:
        return None
    yd = d[ax] - d.groupby("person")[ax].transform("mean")
    xd = d[x] - d.groupby("person")[x].transform("mean")
    return float((xd * yd).sum() / (xd * xd).sum()), d


slopes = []
for x, scale, label in (("dist_km", 500.0, "per 500 km from Chang'an"), ("years_in_exile", 1.0, "per year in exile"),
                        ("lat", -5.0, "per 5 degrees further south")):
    w["_x"] = w[x].clip(upper=10) / scale if x == "years_in_exile" else w[x] / scale
    for ax in AXES + ["n_axes"]:
        r = fe_slope(w, ax, "_x")
        if r is None:
            continue
        b, d = r
        persons = d.person.unique()
        groups = {p: g for p, g in d.groupby("person")}
        bs = []
        for _ in range(N_BOOT):
            pick = rng.choice(persons, len(persons))
            bd = pd.concat([groups[p].assign(person=f"{p}_{i}") for i, p in enumerate(pick)])
            rr = fe_slope(bd, ax, "_x")
            if rr:
                bs.append(rr[0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p = 2 * min(np.mean(np.array(bs) <= 0), np.mean(np.array(bs) >= 0))
        slopes.append({"covariate": label, "axis": ax, "works": len(d), "persons": d.person.nunique(),
                       "within_author_slope": b, "ci_low": lo, "ci_high": hi, "boot_p": p})
slopes = pd.DataFrame(slopes)
ax_rows = slopes.axis != "n_axes"
order = slopes.loc[ax_rows, "boot_p"].rank(method="first")
m = ax_rows.sum()
holm = (slopes.loc[ax_rows, "boot_p"] * (m - order + 1)).clip(upper=1)
slopes.loc[ax_rows, "holm_p"] = holm.loc[slopes.loc[ax_rows].sort_values("boot_p").index].cummax().reindex(holm.index)
save(slopes, "topic1x_within_author_slopes")
w = w.drop(columns="_x")

# leave-one-author-out stability of the within-author slopes
loo = []
for x, scale, label in (("dist_km", 500.0, "per 500 km from Chang'an"), ("years_in_exile", 1.0, "per year in exile")):
    w["_x"] = w[x].clip(upper=10) / scale if x == "years_in_exile" else w[x] / scale
    for ax in ("court_institution", "route_displacement", "landscape", "n_axes"):
        full = fe_slope(w, ax, "_x")
        persons = full[1].person.unique()
        vals = [fe_slope(w[w.person != p], ax, "_x") for p in persons]
        vals = [v[0] for v in vals if v]
        loo.append({"covariate": label, "axis": ax, "full_slope": full[0], "loo_min": min(vals), "loo_max": max(vals),
                    "loo_same_sign_share": float(np.mean(np.sign(vals) == np.sign(full[0])))})
    w = w.drop(columns="_x")
save(pd.DataFrame(loo), "topic1x_slope_leave_one_author_out")

# ---------------------------------------------------------------- episodes and persons
epi = w.dropna(subset=["episode"]).groupby("episode").agg(person=("person", "first"), works=("n_axes", "size"),
                                                         **{ax: (ax, "mean") for ax in AXES})
epi = epi.join(ep[["active_start_year_min", "active_end_year_max", "active_place_text"]])
save(epi.reset_index(), "topic1x_episode_profiles")
multi = epi[epi.works >= 5].groupby("person").filter(lambda g: len(g) >= 2)
key["multi_episode_persons_ge5_works"] = int(multi.person.nunique())
# share of between-episode variance explained by person (one-way, axis means across episodes)
icc = []
for ax in AXES:
    g = multi.groupby("person")[ax]
    between = g.mean().var(ddof=1) if g.ngroups > 1 else np.nan
    within = g.var(ddof=1).mean()
    icc.append({"axis": ax, "persons": g.ngroups, "episodes": len(multi), "between_person_var": between,
                "within_person_between_episode_var": within})
save(pd.DataFrame(icc), "topic1x_episode_consistency")

with open(f"{R}/topic1x_key_results.json", "w", encoding="utf-8") as f:
    json.dump(key, f, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps(key, ensure_ascii=False, indent=1, default=str))
