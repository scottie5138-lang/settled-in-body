"""Evidence tests for the residence finding of 06_residence_stage.py.

Reads results/topic1z_work_stage.parquet (built by 06) and the 1,202 poem titles and bodies
in data/texts/exile_period_poem_texts.parquet.
1. Robustness of the destination contrast to the destination coding: evidence-adjusted seats
   (from poem titles), dropping never-arrived episodes, crosswalk-verified only, leave one poet out.
2. Arrival-anchored phases: arrival = first year with a poem written at the destination, so each
   episode runs capital -> en route -> arrival year -> settled +1 / +2-3 / +4 -> away after arrival.
3. Title genre: does the route drop come from fewer journey poems, or from every kind of poem?
4. Body lexicon: an LLM-independent check of the axes and of the phase pattern, controlled
   for poem length; the phase model is also refitted with poem length controlled.
"""
import json
import os
import re

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


w = pd.read_parquet(f"{R}/topic1z_work_stage.parquet")
tx = pd.read_parquet(f"{D}/texts/exile_period_poem_texts.parquet").set_index("source_record_id")
names = (pd.read_parquet(f"{D}/tang_exile/work_terminal.parquet")
         .drop_duplicates("cbdb_person_id").set_index("cbdb_person_id").normalized_author)
w["poet"] = w.person.map(names)
w["title"] = w.source_record_id.map(tx.title).fillna("")
w["body"] = w.source_record_id.map(tx.body).fillna("")
w["precise"] = w.current_precise_exile_period_work.astype(float)
w["t"] = w.years_in_exile.clip(lower=0, upper=10)
code = w.place_code.fillna("")

# 白居易's 江州 exile has two DB rows (episode 43: 815–818, 44: 815–817). The DH2027 decision record
# (data/dh2027_four_decisions/decisions.parquet) treats them as two source records of one exile, so they
# are merged here; otherwise episode 43's 49 poems of 818 get their own arrival anchor (818) instead of 815.
w.loc[w.episode == 43, "episode"] = 44

# Evidence-adjusted seats, each read off poem titles in data/texts (see README):
# 柳宗元's 815 poems at 柳州 (登柳州城樓...) are his second seat, inside the DB's 永州 episode;
# 宋之問's 710 poems at 越州 (郡宅中齋, 翫郡齋海榴) are written in office there before 欽州;
# 韓偓's only "destination" poem carries an inconsistent place (鄄城 code tagged 泰山) and the
# rest of the episode is in 湖南 and 江西, so the episode is treated as never arrived.
adj = w.stage.copy()
adj[(w.episode == 4) & code.str.startswith("CN4502")] = "destination"
adj[(w.episode == 32) & code.str.startswith("CN3306")] = "destination"
adj[(w.episode == 90) & (adj == "destination")] = "elsewhere"
w["stage_adj"] = adj
NEVER_ARRIVED = [46, 65, 89, 90]  # 李白 夜郎, 王昌齡 龍標, 唐彥謙 漢中, 韓偓 濮州: no poem at the seat


# ---------------------------------------------------------------- within-poet fixed-effect model
def fe_ols(df, ycol, xcols):
    d = df[[ycol, "person"] + xcols].dropna()
    dm = d[[ycol] + xcols] - d.groupby("person")[[ycol] + xcols].transform("mean")
    X, Y = dm[xcols].values, dm[ycol].values
    if len(d) == 0 or np.linalg.matrix_rank(X) < len(xcols):
        return None, d
    return np.linalg.lstsq(X, Y, rcond=None)[0], d


def fit(df, model, ys, xcols, labels=None, boot=True):
    out = []
    for ax in ys:
        b, d = fe_ols(df, ax, xcols)
        if b is None or d.person.nunique() < 5:
            continue
        bs = None
        if boot:
            groups = {p: g for p, g in d.groupby("person")}
            persons = list(groups)
            sufficient = []
            for g in groups.values():
                v = g[[ax] + xcols] - g[[ax] + xcols].mean()
                x, y = v[xcols].to_numpy(), v[ax].to_numpy()
                sufficient.append((x.T @ x, x.T @ y))
            xx = np.array([s[0] for s in sufficient])
            xy = np.array([s[1] for s in sufficient])
            bs = [[] for _ in xcols]
            for _ in range(N_BOOT):
                pick = rng.integers(len(persons), size=len(persons))
                a = xx[pick].sum(0)
                inverse = np.linalg.pinv(a)
                bb = inverse @ xy[pick].sum(0)
                projection = inverse @ a
                for j in range(len(xcols)):
                    target = np.eye(len(xcols))[j]
                    if np.allclose(target @ projection, target):
                        bs[j].append(bb[j])
        for j, x in enumerate(xcols):
            r = {"model": model, "outcome": ax, "term": (labels or {}).get(x, x), "works": len(d),
                 "poets": d.person.nunique(), "slope": b[j]}
            if boot:
                samples = np.asarray(bs[j])
                lo, hi = np.percentile(samples, [2.5, 97.5])
                r.update({"ci_low": lo, "ci_high": hi,
                          "boot_p": min(1.0, 2 * (min((samples <= 0).sum(), (samples >= 0).sum()) + 1) / (len(samples) + 1)),
                          "bootstrap_valid": len(samples), "bootstrap_attempted": N_BOOT})
            out.append(r)
    return pd.DataFrame(out)


def both_sides(df, col):
    d = df[df[col].isin(["destination", "elsewhere"])].assign(dest=lambda x: (x[col] == "destination").astype(float))
    return d.groupby("person").filter(lambda g: g.dest.nunique() == 2)


# ---------------------------------------------------------------- 1. robustness of the destination contrast
variants = {
    "as coded in 06": both_sides(w, "stage"),
    "evidence-adjusted seats": both_sides(w, "stage_adj"),
    "evidence-adjusted, never-arrived episodes dropped": both_sides(w[~w.episode.isin(NEVER_ARRIVED)], "stage_adj"),
    "crosswalk-verified destinations only": both_sides(w[w.basis == "crosswalk_name"], "stage"),
    "evidence-adjusted, without 白居易": both_sides(w[w.poet != "白居易"], "stage_adj"),
    "evidence-adjusted, precise dates only": both_sides(w[w.precise == 1], "stage_adj"),
}
rob = pd.concat([fit(d, m, AXES + ["n_axes"], ["dest"]) for m, d in variants.items()])
save(rob, "topic1z_destination_robustness")
base = variants["evidence-adjusted seats"]
loo = []
for p in sorted(base.person.unique()):
    b, d = fe_ols(base[base.person != p], "route_displacement", ["dest"])
    loo.append({"poet_left_out": names.get(p), "works": len(d), "route_destination_slope": b[0]})
save(pd.DataFrame(loo), "topic1z_destination_leave_one_poet_out")
key["robustness_route"] = {m: round(float(r.slope), 4) for m, r in
                           rob[rob.outcome == "route_displacement"].set_index("model").iterrows()}
key["leave_one_poet_out_route_range"] = [float(pd.DataFrame(loo).route_destination_slope.min()),
                                         float(pd.DataFrame(loo).route_destination_slope.max())]

# ---------------------------------------------------------------- 2. arrival-anchored phases
dest_years = w[w.stage_adj == "destination"].groupby("episode").year.min()
w["arrival_year"] = w.episode.map(dest_years)
w["rel"] = w.year - w.arrival_year
PHASES = ["capital", "en route", "arrival year", "settled +1", "settled +2-3", "settled +4 or more",
          "away after arrival", "never arrived"]


def phase(r):
    if r.stage_adj == "capital":
        return "capital"
    if r.stage_adj not in ("destination", "elsewhere") or pd.isna(r.year):
        return None
    if pd.isna(r.arrival_year):
        return "never arrived"
    if r.stage_adj == "elsewhere":
        return "en route" if r.rel <= 0 else "away after arrival"
    return {0: "arrival year", 1: "settled +1", 2: "settled +2-3", 3: "settled +2-3"}.get(int(r.rel), "settled +4 or more")


w["phase"] = w.apply(phase, axis=1)
ph = w[w.phase.notna()]
rows = []
for p in PHASES:
    g = ph[ph.phase == p]
    if not len(g):
        continue
    land = g[g.landscape == 1]
    rows.append({"phase": p, "works": len(g), "poets": g.person.nunique(), "precise_share": g.precise.mean(),
                 **{ax: g[ax].mean() for ax in AXES}, "mean_axes": g[AXES].sum(axis=1, min_count=5).mean(),
                 "landscape_with_route": (land.route_displacement == 1).mean(),
                 "landscape_with_court": (land.court_institution == 1).mean(),
                 "landscape_with_temple": (land.temple_reclusion == 1).mean()})
save(pd.DataFrame(rows), "topic1z_phase_profiles")

# within-poet contrasts of each arrived phase against the same poet's en-route poems
arr = ph[ph.phase.isin(PHASES[1:6])].copy()
arr = arr.groupby("person").filter(lambda g: (g.phase == "en route").any() and (g.phase != "en route").any())
PH_X = {"arrival year": "ph_arr", "settled +1": "ph_s1", "settled +2-3": "ph_s23", "settled +4 or more": "ph_s4"}
for p, c in PH_X.items():
    arr[c] = (arr.phase == p).astype(float)
phase_models = fit(arr, "phase vs en route (+ dating tier)", AXES + ["n_axes"], list(PH_X.values()) + ["precise"],
                   labels={**{c: p for p, c in PH_X.items()}, "precise": "precise vs annual-proxy date"})
save(phase_models, "topic1z_phase_within_poet")
key["phase_model"] = {"works": len(arr), "poets": sorted(arr.poet.unique().tolist())}

# ---------------------------------------------------------------- 3. title genre
TITLE = {
    "title_journey": r"^(早|晚|夜|初|再|重)?(發|渡|度|經|過|至|到|次|泊|入|下|上|抵|赴|出|宿|行次)|途|道中|路|流夜郎|自.{1,8}(至|抵|赴|入|向)",
    "title_parting": r"送|別",
    # 和 marks a matching poem, except in the reign name 元和 (e.g. 元和十三年...)
    "title_exchange": r"寄|贈|酬|詶|答|(?<!元)和|呈|代書",
    "title_excursion": r"登|遊|泛|望|謁|尋|訪|題",
}
clean = w.title.str.replace(r"^(相和歌辭|雜歌謠辭|琴曲歌辭|雜曲歌辭|樂府雜曲|郊廟歌辭)\s*", "", regex=True)
for c, pat in TITLE.items():
    w[c] = clean.str.contains(re.sub(r"\((?!\?)", "(?:", pat)).astype(float)
# manual review of the 24 titles whose only exchange marker is 和 (results/DH2027_24題名人工審核.csv, 2026-11-02):
# in these two Yuan Zhen titles 歸和 / 杜歸和 is a personal name, not a matching poem
w.loc[w.source_record_id.isin(["src_bbecbe9e0a9c95fa4023", "src_d6b74aacf21fbe1ef853"]), "title_exchange"] = 0.0
w["title_none"] = (w[list(TITLE)].sum(axis=1) == 0).astype(float)
gt = w[w.stage_adj.isin(["capital", "destination", "elsewhere"])]
genre = gt.groupby("stage_adj")[list(TITLE) + ["title_none"]].mean()
genre["works"] = gt.groupby("stage_adj").size()
genre["route_if_journey_title"] = gt[gt.title_journey == 1].groupby("stage_adj").route_displacement.mean()
genre["route_if_no_journey_title"] = gt[gt.title_journey == 0].groupby("stage_adj").route_displacement.mean()
genre["parting_share_of_route_poems"] = gt[gt.route_displacement == 1].groupby("stage_adj").title_parting.mean()
save(genre.reset_index(), "topic1z_title_genre_by_stage")
bt = both_sides(w, "stage_adj")
genre_models = pd.concat([
    fit(bt, "journey title share", ["title_journey", "title_parting", "title_exchange", "title_excursion"], ["dest"]),
    fit(bt, "route, title genre controlled", ["route_displacement"],
        ["dest", "title_journey", "title_parting", "title_exchange", "title_excursion"]),
    fit(bt[bt.title_journey == 0], "route, poems without a journey title", ["route_displacement"], ["dest"]),
])
save(genre_models, "topic1z_title_genre_models")

# ---------------------------------------------------------------- 4. body lexicon
LEX = {
    "lex_capital": r"長安|京華|京國|京洛|帝城|帝鄉|帝京|魏闕|北闕|金闕|丹闕|君王|天子|聖主|明主|聖朝|明時|丹墀|紫宸|鳳池|鳳闕|金馬|承明|朝廷|宸",
    "lex_road": r"驛|征|帆|棹|舟|萬里|行人|長路|道路|旅|客程|馬蹄|輪",
    "lex_temple": r"寺|僧|禪|佛|梵|上人|空門|塔|蓮宮|精舍|香火|經聲|浮屠",
    "lex_homeward": r"歸|故鄉|故園|故國|鄉|北望|望京|思家|家山",
    "lex_exile_self": r"謫|逐客|遷客|放逐|竄|流人|貶|羈|累臣|孤臣",
    # 園 inside 故園 is homeward, not dwelling; without the lookbehind every 故園 counted on both lists
    "lex_dwelling": r"居|宅|廬|(?<!故)園|圃|種|栽|鄰|卜|築|茅|籬",
}
for c, pat in LEX.items():
    w[c] = w.body.str.contains(pat).astype(float)
# longer poems hit any word list more often (arrival-year poems are the longest), so control length
w["loglen"] = np.log(w.body.str.replace(r"[^一-鿿]", "", regex=True).str.len().clip(lower=1))
PAIR = {"lex_capital": "court_institution", "lex_road": "route_displacement", "lex_temple": "temple_reclusion"}
val = []
for lx, ax in PAIR.items():
    g = w.dropna(subset=[ax])
    val.append({"lexicon": lx, "axis": ax, "works": len(g),
                "lexicon_rate_axis_present": g[g[ax] == 1][lx].mean(),
                "lexicon_rate_axis_absent": g[g[ax] == 0][lx].mean(),
                "axis_rate_lexicon_hit": g[g[lx] == 1][ax].mean(),
                "axis_rate_lexicon_miss": g[g[lx] == 0][ax].mean()})
save(pd.DataFrame(val), "topic1z_lexicon_axis_agreement")
lex_phase = ph.join(w[list(LEX)]).groupby("phase")[list(LEX)].mean().reindex([p for p in PHASES if p in set(ph.phase)])
save(lex_phase.reset_index(), "topic1z_lexicon_by_phase")
arr = arr.join(w[list(LEX) + ["loglen"]])
lex_models = fit(arr, "lexicon, phase vs en route (+ dating tier, poem length)", list(LEX),
                 list(PH_X.values()) + ["precise", "loglen"],
                 labels={**{c: p for p, c in PH_X.items()}, "precise": "precise vs annual-proxy date",
                         "loglen": "log characters"})
save(lex_models, "topic1z_lexicon_phase_within_poet")
save(fit(arr, "phase vs en route (+ dating tier, poem length)", AXES + ["n_axes"],
         list(PH_X.values()) + ["precise", "loglen"],
         labels={**{c: p for p, c in PH_X.items()}, "precise": "precise vs annual-proxy date", "loglen": "log characters"}),
     "topic1z_phase_within_poet_length_controlled")

w.drop(columns=["body"]).to_parquet(f"{R}/topic1z_work_phase.parquet", index=False)
with open(f"{R}/topic1z_evidence_key_results.json", "w", encoding="utf-8") as f:
    json.dump(key, f, ensure_ascii=False, indent=2)
print(json.dumps(key, ensure_ascii=False))
