"""Rebuild manuscript Table 1 and the linkage and validation inputs for Figure 1.

Reads the aggregate results and writes the comparison table as CSV and Markdown.
"""
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.environ.get("DH2027_RESULTS_DIR", os.path.join(ROOT, "results"))
OUT = os.path.join(R, "tables")
os.makedirs(OUT, exist_ok=True)


def csv(name):
    return pd.read_csv(f"{R}/{name}.csv")


def num(v):
    return f"{int(v):,}"


def pp(v):
    # percentage points with a true minus sign; values that round to zero print unsigned,
    # so a bound such as -0.03 is not shown as +0.0
    s = f"{100 * v + 0.0:+.1f}"
    return "0.0" if s in ("+0.0", "-0.0") else s.replace("-", "−")


def md(df, right):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---:" if c in right else "---" for c in cols) + "|"]
    lines += ["| " + " | ".join(str(v) if v != "" else " " for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


k1 = json.load(open(f"{R}/key_results.json", encoding="utf-8"))
k1x = json.load(open(f"{R}/topic1x_key_results.json", encoding="utf-8"))
k1z = json.load(open(f"{R}/topic1z_key_results.json", encoding="utf-8"))
funnel, cov, zc = k1["t2_tang_funnel"], k1x["coverage"], k1z["coverage"]
phase = csv("topic1z_phase_within_poet").query("outcome != 'n_axes'")
pc = json.load(open(f"{R}/public_counts.json", encoding="utf-8"))  # from 18_public_counts.py
adjusted_counts = pd.Series(pc["stage_counts"])

# Linkage and model coverage for Figure 1
w = csv("topic1w_within_poet_contrasts")
wi = w.set_index(["contrast", "outcome"])
CONTRASTS = [("seat vs elsewhere", "At the seat vs elsewhere"), ("arrival year vs en route", "Arrival-proxy year vs en route"),
             ("per year, poems at the seat", "Per year at the seat"), ("per year, all dated poems", "Per year, all dated exile poems")]
ROWS = [("Address", "title_exchange", "Titles of letters and exchanges"),
        ("Self-positioning", "lex_exile_self", "Self-naming as an exile"),
        ("Court and return", "court_institution", "Court axis"), ("Court and return", "lex_capital", "Capital-city words (京闕)"),
        ("Court and return", "lex_homeward", "Return-related words")]

def table_sample_size(contrast):
    sizes = wi.loc[contrast].loc[[outcome for _, outcome, _ in ROWS], "works"]
    lo, hi = sizes.min(), sizes.max()
    return num(lo) if lo == hi else f"{num(lo)}–{num(hi)}"


t1 = pd.DataFrame([
    ("Tang poems in the exile database snapshot", num(funnel["source_records"]), "", ""),
    ("Poems with a confirmed author", num(funnel["accepted_binding"]), "", ""),
    ("  by poets with recorded exile episodes", num(k1["t2_naive_exile_poet_corpus"]), "",
     f"{k1['t2_inflation_naive_over_effective']:.1f} times the currently confirmed exile-period corpus (this database only)"),
    ("Currently confirmed exile-period poems", num(funnel["effective"]), str(k1["t2_persons_with_effective"]),
     f"Dated precisely {num(funnel['precise'])}, by annual proxy {num(funnel['proxy'])}, other {funnel['effective_neither_flag']}"),
    ("  with five-axis annotation", num(k1x["works"]), str(cov["persons"]), f"{cov['episodes']} exile episodes"),
    ("  with a composition year", num(cov["with_year"]), "", ""),
    ("  linked to an exile episode", num(cov["with_episode"]), "", ""),
    ("  with composition coordinates", num(cov["with_coordinates"]), "", ""),
    ("Place of writing classifiable", num(k1z["analysed"]["works"]), str(k1z["analysed"]["poets"]),
     f"{k1z['analysed']['episodes']} episodes; title-corrected classification"),
    ("  at the exile seat", num(adjusted_counts["destination"]), str(pc["seat_poets"]), ""),
    ("  elsewhere (en route, side trips, after leaving)", num(adjusted_counts["elsewhere"]), "", ""),
    ("  in Chang'an before departure", num(adjusted_counts["capital"]), "", ""),
    ("Place of writing not classifiable", num(pc["works"] - 700), "",
     "415 without a reviewed place; 68 without an episode; 15 unmapped seats; 3 coarse places"),
    ("Phase trajectories (Figure 2)", f"{phase.works.min()}–{phase.works.max()}", str(phase.poets.max()),
     "Poets with poems both en route and at the seat; arrival year = first recorded year at the seat"),
    *[(f"Within-poet comparison: {lab.lower()}", table_sample_size(c), str(wi.loc[(c, 'title_exchange')].poets),
       "Table 1") for c, lab in CONTRASTS],
], columns=["Stage", "Poems", "Poets", "Note"])

# Blind validation for Figure 1 and word-list agreement for section 4
blind = csv("topic1_blind50_validation")
AXES = {"court_institution": "Court", "landscape": "Landscape", "temple_reclusion": "Temple",
        "route_displacement": "Route", "time_void": "Temporal void", "ALL": "All axes"}
t2a = pd.DataFrame([(AXES[r.axis], r.model_valued, r.agree, f"{100 * r.agreement:.1f}",
                     "" if r.axis == "ALL" else f"{100 * r.wilson_low:.1f}", f"{r.cohen_kappa:.2f}")
                    for r in blind.itertuples()],
                   columns=["Blind sample (50 poems, one human reviewer)", "Determinate comparisons", "Agreed", "Agreement %", "95% lower bound %", "κ"])
# of the 309 targeted positions, only those where both human and model gave present/absent are determinate
hard = csv("topic1_targeted309_confusion").set_index("human_status").loc[["absent", "present"], ["absent", "present"]]
t2a.loc[len(t2a)] = [f"Hard cases (targeted, {k1['t1_targeted_positions']} positions)", int(hard.values.sum()),
                     int(hard.values.trace()), f"{100 * k1['t1_targeted_model_valued_agreement']:.1f}", "",
                     f"{k1['t1_targeted_kappa']:.2f}"]
lexa = csv("topic1z_lexicon_axis_agreement")
LEX = {"lex_capital": ("Capital-city words (京闕)", "Court"), "lex_road": ("Road words (道路)", "Route"),
       "lex_temple": ("Temple words (寺院)", "Temple")}
t2b = pd.DataFrame([(LEX[r.lexicon][0], LEX[r.lexicon][1], f"{100 * r.axis_rate_lexicon_hit:.1f}", f"{100 * r.axis_rate_lexicon_miss:.1f}")
                    for r in lexa.itertuples()],
                   columns=["Word list", "Matching axis", "Marked on axis, words present %", "Marked on axis, words absent %"])

# Table 1: word and title measures across the four within-poet comparisons
# rows are labelled by construct, not by result: within a row the four comparisons can point different ways



def cell(c, o):
    r = wi.loc[(c, o)]
    star = "*" if r.holm_p < 0.05 else ""
    return f"{pp(r.slope)}{star} ({pp(r.ci_low)} to {pp(r.ci_high)})"


t3 = pd.DataFrame([(g, label, *[cell(c, o) for c, _ in CONTRASTS]) for g, o, label in ROWS],
                  columns=["", "Measure", *[f"{lab} (n={wi.loc[(c, 'title_exchange')].poets})" for c, lab in CONTRASTS]])

t1.to_csv(f"{OUT}/table1_data_linkage.csv", index=False, encoding="utf-8-sig")
pd.concat([t2a.assign(panel="A"), t2b.assign(panel="B")]).to_csv(f"{OUT}/table2_validation.csv", index=False, encoding="utf-8-sig")
t3.to_csv(f"{OUT}/table1_within_poet.csv", index=False, encoding="utf-8-sig")
with open(f"{OUT}/paper_tables.md", "w", encoding="utf-8") as f:
    f.write("*Table 1. Within-poet differences in percentage points across four comparisons: estimate (95% CI).*\n\n")
    display_rows = []
    for group, rows in t3.groupby("", sort=False):
        display_rows.append([f"***{group}***", "", "", "", ""])
        display_rows.extend(rows.iloc[:, 1:].itertuples(index=False, name=None))
    display = pd.DataFrame(display_rows, columns=t3.columns[1:])
    f.write(md(display, set(display.columns[1:])) + "\n\n")
    f.write('*Note: Pointwise 95% poet-bootstrap intervals. n counts poets. Models use poet fixed effects and, where estimable, dating precision, which is omitted in the arrival-year comparison. Word measures control log poem length. Holm correction covers 14 outcomes per comparison. No displayed estimate passes correction.*\n')
print("wrote", OUT)
