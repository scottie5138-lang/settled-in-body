"""Beyond the route drop: what changes at the place of exile, and what does not.

Reads results/topic1z_work_phase.parquet (built by 07_settling_evidence.py) and the poem
titles and bodies in data/texts/exile_period_poem_texts.parquet.
1. Whom poets write to: central-office, local-office and religious/recluse addressees in titles.
2. Whether court writing and homeward words recede at the seat, and what carries court writing there.
3. The arrival shock: grief words, temporal void and exile self-naming around arrival.
4. The south: alien-south words (瘴, 蠻, 魑魅 ...) against named local produce (荔, 蕉, 椰 ...).
5. Ambivalence: dwelling and homeward words in the same poem.
Word-list outcomes are controlled for poem length (log characters).
Every outcome is tested with four within-poet contrasts, chosen to keep as many poets as possible:
seat vs elsewhere, arrival year vs en route, per year at the seat, per year over all dated poems.
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


w = pd.read_parquet(f"{R}/topic1z_work_phase.parquet")
w["body"] = w.source_record_id.map(pd.read_parquet(f"{D}/texts/exile_period_poem_texts.parquet")
                                   .set_index("source_record_id").body).fillna("")
title = w.title.str.replace(r"^(相和歌辭|雜歌謠辭|琴曲歌辭|雜曲歌辭|樂府雜曲|郊廟歌辭)\s*", "", regex=True)

# addressees named in titles (a title may name several)
ADDRESSEE = {
    "to_central_office": r"侍郎|尚書|中丞|舍人|學士|相公|相國|拾遺|補闕|給事|員外|郎中|御史|侍御|諫議|僕射|閣老|翰林|中書|秘書|校書|正字|博士|司業|祭酒",
    "to_local_office": r"使君|太守|刺史|明府|長史|司馬|判官|從事|錄事|司戶|少府|縣令|都督|觀察|節度|參軍|主簿|別駕|府主|長官",
    "to_religious_recluse": r"上人|禪師|僧|和尚|大師|法師|處士|山人|道士|尊師|鍊師|隱者|隱士|高士|居士|黃道士",
}
for c, pat in ADDRESSEE.items():
    w[c] = title.str.contains(pat).astype(float)
LEX = {
    "lex_grief": r"淚|愁|恨|悲|哀|泣|傷|啼|斷腸|腸斷|惆悵|涕",
    "lex_alien_south": r"瘴|蠻|夷|獠|魑魅|鬼門|炎方|炎荒|炎州|毒|蛇|虺|蜮|南荒|荒陬|殊俗|異俗|蛟|鵩",
    "lex_local_produce": r"荔|橘|柚|蕉|椰|桄榔|檳榔|朱槿|木綿|榕|甘蔗|含桃|筍|蓴|枇杷|龍眼|山茶|茗",
}
for c, pat in LEX.items():
    w[c] = w.body.str.contains(pat).astype(float)
w["dwelling_and_homeward"] = w.lex_dwelling * w.lex_homeward
# longer poems hit any word list more often, so word-list outcomes are length-controlled
w["loglen"] = np.log(w.body.str.replace(r"[^一-鿿]", "", regex=True).str.len().clip(lower=1))

OUTCOMES = {
    "addressee": list(ADDRESSEE) + ["title_exchange"],
    "what stays": ["court_institution", "lex_capital", "lex_homeward"],
    "arrival shock": ["lex_grief", "time_void", "lex_exile_self"],
    "the south": ["lex_alien_south", "lex_local_produce"],
    "ambivalence": ["lex_dwelling", "dwelling_and_homeward"],
}


# ---------------------------------------------------------------- fast within-poet model
def person_stats(df, y, xcols):
    """Per-poet demeaned X'X and X'y, so a poet-cluster bootstrap is a sum over resampled poets."""
    d = df[[y, "person"] + xcols].dropna()
    st = []
    for _, g in d.groupby("person"):
        X = g[xcols].values - g[xcols].values.mean(0)
        Y = g[y].values - g[y].values.mean()
        st.append((X.T @ X, X.T @ Y))
    return st, d


def fit(df, contrast, family, ys, xcols, term):
    out = []
    for y in ys:
        st, d = person_stats(df, y, xcols)
        # a control with no within-poet variation (dating precision in the arrival sample, where each poet's
        # poems share one precision) carries no information; drop it instead of losing the whole contrast
        used = [xcols[0]] + [c for c in xcols[1:] if d.groupby("person")[c].nunique().gt(1).any()]
        if used != xcols:
            st, d = person_stats(df, y, used)
        if d.person.nunique() < 5:
            continue
        XX = np.array([s[0] for s in st])
        XY = np.array([s[1] for s in st])
        if np.linalg.matrix_rank(XX.sum(0)) < len(used):
            continue
        b = np.linalg.solve(XX.sum(0), XY.sum(0))[0]
        bs = []
        target = np.eye(len(used))[0]
        for _ in range(N_BOOT):
            pick = rng.integers(0, len(st), len(st))
            A = XX[pick].sum(0)
            inverse = np.linalg.pinv(A)
            if np.allclose(target @ inverse @ A, target):
                bs.append((inverse @ XY[pick].sum(0))[0])
        bs = np.array(bs)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out.append({"family": family, "contrast": contrast, "term": term, "outcome": y, "works": len(d),
                    "poets": d.person.nunique(), "mean": d[y].mean(), "slope": b, "ci_low": lo, "ci_high": hi,
                    "boot_p": min(1.0, 2 * (min((bs <= 0).sum(), (bs >= 0).sum()) + 1) / (len(bs) + 1)),
                    "bootstrap_valid": len(bs), "bootstrap_attempted": N_BOOT, "controls": "+".join(used[1:])})
    return out


def with_both(df, col):
    return df.groupby("person").filter(lambda g: g[col].nunique() == 2)


seat = with_both(w[w.stage_adj.isin(["destination", "elsewhere"])]
                 .assign(dest=lambda x: (x.stage_adj == "destination").astype(float)), "dest")
arrival = with_both(w[w.phase.isin(["en route", "arrival year"])]
                    .assign(arr=lambda x: (x.phase == "arrival year").astype(float)), "arr")
at_seat = w[w.stage_adj == "destination"].groupby("person").filter(lambda g: g.t.nunique() >= 2)
dated = w.dropna(subset=["t"]).groupby("person").filter(lambda g: g.t.nunique() >= 2)
CONTRASTS = [
    ("seat vs elsewhere", seat, ["dest", "precise"], "written at the seat"),
    ("arrival year vs en route", arrival, ["arr", "precise"], "arrival year"),
    ("per year, poems at the seat", at_seat, ["t", "precise"], "per year in exile"),
    ("per year, all dated poems", dated, ["t", "precise"], "per year in exile"),
]
rows = []
for family, ys in OUTCOMES.items():
    for name, df, xc, term in CONTRASTS:
        for y in ys:
            rows += fit(df, name, family, [y], xc + (["loglen"] if y.startswith(("lex_", "dwelling")) else []), term)
res = pd.DataFrame(rows)
# Holm within each contrast across all outcomes of this script
for _, idx in res.groupby("contrast").groups.items():
    p = res.loc[idx, "boot_p"].sort_values()
    res.loc[p.index, "holm_p"] = (p * (len(p) - np.arange(len(p)))).cummax().clip(upper=1).values
save(res, "topic1w_within_poet_contrasts")

# descriptive rates by phase
PHASES = ["capital", "en route", "arrival year", "settled +1", "settled +2-3", "settled +4 or more",
          "away after arrival", "never arrived"]
allout = [y for ys in OUTCOMES.values() for y in ys]
ph = w[w.phase.notna()].groupby("phase")[allout].mean().reindex([p for p in PHASES if p in set(w.phase.dropna())])
ph.insert(0, "works", w[w.phase.notna()].groupby("phase").size())
ph.insert(1, "mean_characters", np.exp(w[w.phase.notna()].groupby("phase").loglen.mean()))
save(ph.reset_index(), "topic1w_rates_by_phase")

# per-poet direction, seat vs elsewhere, poets with >= 5 poems on each side
dirs = []
for y in allout:
    m = seat.groupby(["poet", "stage_adj"])[y].agg(["mean", "size"]).unstack()
    m = m[(m["size"]["destination"] >= 5) & (m["size"]["elsewhere"] >= 5)]
    dd = (m["mean"]["destination"] - m["mean"]["elsewhere"])
    dirs.append({"outcome": y, "poets": len(dd), "higher_at_seat": int((dd > 0).sum()),
                 "lower_at_seat": int((dd < 0).sum()), "median_difference": dd.median()})
save(pd.DataFrame(dirs), "topic1w_poet_direction_seat")

# ---------------------------------------------------------------- what carries court writing at the seat
st = w[w.stage_adj.isin(["capital", "destination", "elsewhere"])]
carry = []
for s, g in st.groupby("stage_adj"):
    c = g[g.court_institution == 1]
    carry.append({"stage": s, "poems": len(g), "court_poems": len(c),
                  "court_share_exchange_title": c.title_exchange.mean(),
                  "court_share_to_central_office": c.to_central_office.mean(),
                  "court_share_to_local_office": c.to_local_office.mean(),
                  "court_rate_if_to_central_office": g[g.to_central_office == 1].court_institution.mean(),
                  "court_rate_if_to_local_office": g[g.to_local_office == 1].court_institution.mean(),
                  "homeward_if_dwelling": g[g.lex_dwelling == 1].lex_homeward.mean(),
                  "homeward_if_no_dwelling": g[g.lex_dwelling == 0].lex_homeward.mean()})
save(pd.DataFrame(carry), "topic1w_court_carriers_and_ambivalence")

# within poets, does a dwelling word make a homeward word more or less likely in the same poem?
amb = []
for s, g in [("written at the seat", w[w.stage_adj == "destination"]), ("written elsewhere", w[w.stage_adj == "elsewhere"]),
             ("all exile-period poems", w)]:
    amb += fit(g, s, "ambivalence", ["lex_homeward"], ["lex_dwelling", "loglen"], "dwelling word in the same poem")
save(pd.DataFrame(amb), "topic1w_dwelling_homeward_association")

key["contrast_sizes"] = {n: {"works": len(df), "poets": int(df.person.nunique())} for n, df, _, _ in CONTRASTS}
with open(f"{R}/topic1w_key_results.json", "w", encoding="utf-8") as f:
    json.dump(key, f, ensure_ascii=False, indent=2)
print(json.dumps(key, ensure_ascii=False))
