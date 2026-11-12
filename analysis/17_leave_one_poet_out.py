"""Leave-one-poet-out checks for the two exploratory changes, and the exchange titles that rest on 和 alone.

Reads results/topic1z_work_phase.parquet (built by 07) and the poem bodies in data/texts. Refits the within-poet
models of 08_beyond_route.py for exchange titles (arrival-proxy year vs en route) and exile self-naming (per year
at the seat), dropping one poet at a time; point estimates only, with the same controls as 08 (dating precision
when it varies within poets, log length for word measures). Also lists the exchange titles whose only exchange
marker is 和, for a manual check of whether they are matching poems. Titles only, no poem text.
Writes results/topic1r_leave_one_poet_out.csv, results/topic1r_he_only_exchange_titles.csv and
results/topic1r_key_results.json.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
R = os.environ.get("DH2027_RESULTS_DIR", os.path.join(ROOT, "results"))

w = pd.read_parquet(f"{R}/topic1z_work_phase.parquet")
names = (pd.read_parquet(f"{D}/tang_exile/work_terminal.parquet")
         .drop_duplicates("cbdb_person_id").set_index("cbdb_person_id").normalized_author)
w["poet"] = w.person.map(names)


def fe(df, y, xcols):
    """Within-poet OLS slope of the first column; controls without within-poet variation are dropped, as in 08."""
    used = [xcols[0]] + [c for c in xcols[1:] if df.groupby("person")[c].nunique().gt(1).any()]
    d = df[[y, "person"] + used].dropna()
    X = d[used] - d.groupby("person")[used].transform("mean")
    Y = d[y] - d.groupby("person")[y].transform("mean")
    return float(np.linalg.lstsq(X.values, Y.values, rcond=None)[0][0])


def with_both(df, col):
    return df.groupby("person").filter(lambda g: g[col].nunique() == 2)


arrival = with_both(w[w.phase.isin(["en route", "arrival year"])]
                    .assign(arr=lambda x: (x.phase == "arrival year").astype(float)), "arr")
at_seat = w[w.stage_adj == "destination"].groupby("person").filter(lambda g: g.t.nunique() >= 2)
CHECKS = [("exchange titles, arrival-proxy year vs en route", arrival, "title_exchange", ["arr", "precise"]),
          ("exile self-naming, per year at the seat", at_seat, "lex_exile_self", ["t", "precise", "loglen"])]
rows = []
for label, df, y, xc in CHECKS:
    full = fe(df, y, xc)
    rows.append({"check": label, "poet_left_out": "(none)", "slope": full, "poets": df.person.nunique()})
    for p, poet in df.groupby("person").poet.first().items():
        rest = df[df.person != p]
        rows.append({"check": label, "poet_left_out": poet, "slope": fe(rest, y, xc), "poets": rest.person.nunique()})
loo = pd.DataFrame(rows)
loo.to_csv(f"{R}/topic1r_leave_one_poet_out.csv", index=False, encoding="utf-8-sig")

clean = w.title.str.replace(r"^(相和歌辭|雜歌謠辭|琴曲歌辭|雜曲歌辭|樂府雜曲|郊廟歌辭)\s*", "", regex=True)
he_only = w[clean.str.contains("(?<!元)和") & ~clean.str.contains(r"寄|贈|酬|詶|答|呈|代書")]
he_only[["source_record_id", "poet", "year", "stage_adj", "phase", "title"]].sort_values(["poet", "year"]).to_csv(
    f"{R}/topic1r_he_only_exchange_titles.csv", index=False, encoding="utf-8-sig")

key = {}
for label, g in loo[loo.poet_left_out != "(none)"].groupby("check"):
    key[label] = {"full": float(loo[(loo.check == label) & (loo.poet_left_out == "(none)")].slope.iloc[0]),
                  "min": float(g.slope.min()), "max": float(g.slope.max()),
                  "poet_at_min": g.loc[g.slope.idxmin(), "poet_left_out"], "poet_at_max": g.loc[g.slope.idxmax(), "poet_left_out"],
                  "without_bai_juyi": float(g.loc[g.poet_left_out == "白居易", "slope"].iloc[0]) if (g.poet_left_out == "白居易").any() else None}
key["exchange_titles"] = int(w.title_exchange.sum())
key["he_only_exchange_titles"] = len(he_only)
json.dump(key, open(f"{R}/topic1r_key_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(key, ensure_ascii=False, indent=1))
