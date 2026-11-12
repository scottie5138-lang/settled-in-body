"""Figure 1: how the data are processed, from the database snapshot to the comparisons.

Replaces the paper's former Tables 1 and 2. Reads results/tables/table1_data_linkage.csv and
results/tables/table2_validation.csv (built by 15_paper_tables.py from traced result files), so every count on the
figure comes from the same files the number check reads. Left column: poem-level linkage; right column: textual
measurement and its validation; bottom: the comparisons. The figure is drawn at printed page width (7.4 in), in
inches, so its 8 pt text prints at 8 pt. Writes results/figures/fig0_process.{png,svg}.
"""
import os
import re

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.environ.get("DH2027_RESULTS_DIR", os.path.join(ROOT, "results"))
OUT = os.path.join(R, "figures")
os.makedirs(OUT, exist_ok=True)
INK, MUTED, EDGE, FILL, ACCENT = "#0b0b0b", "#52514e", "#a9a7a1", "#f3f2ef", "#2a78d6"
TITLE_PT, BODY_PT, LEAD = 8.5, 7.5, 1.4
LINE = BODY_PT * LEAD / 72  # inches per body line
PAD, GAP = 0.09, 0.22  # inner padding and vertical gap between boxes, inches

t1 = pd.read_csv(f"{R}/tables/table1_data_linkage.csv")
t1["Stage"] = t1.Stage.str.strip()  # sub-rows are indented in the table file
t1 = t1.set_index("Stage")
t2 = pd.read_csv(f"{R}/tables/table2_validation.csv")
blind = t2[t2.panel == "A"].set_index("Blind sample (50 poems, one human reviewer)")


def poems(stage):
    return t1.loc[stage, "Poems"]


def poets(stage):
    return int(t1.loc[stage, "Poets"])


def note(stage):
    return t1.loc[stage, "Note"]


dating = re.findall(r"\d+", note("Currently confirmed exile-period poems"))  # precise, annual proxy, other
unc = re.findall(r"\d+", note("Place of writing not classifiable"))  # reviewed place, episode, seat, coarse place
allax, hard = blind.loc["All axes"], blind.loc["Hard cases (targeted, 309 positions)"]
comparisons = [("Seat vs elsewhere", "Within-poet comparison: at the seat vs elsewhere"),
               ("Arrival proxy vs en route", "Within-poet comparison: arrival-proxy year vs en route"),
               ("Per year at the seat", "Within-poet comparison: per year at the seat"),
               ("Per year", "Within-poet comparison: per year, all dated exile poems"),
               ("Phase model", "Phase trajectories (Figure 2)")]
cmp_lines = [f"{label}: {poems(stage)} poems, {poets(stage)} poets" for label, stage in comparisons]

LEFT = [("Database snapshot", [f"{poems('Tang poems in the exile database snapshot')} Tang poems",
                                f"{poems('Poems with a confirmed author')} with a confirmed author",
                                f"{poems('by poets with recorded exile episodes')} by poets with exile episodes"]),
        ("Written during exile", [f"{poems('Currently confirmed exile-period poems')} poems, "
                                  f"{poets('Currently confirmed exile-period poems')} poets",
                                  f"dated precisely {dating[0]} · by annual proxy {dating[1]}",
                                  f"other {dating[2]}"]),
        ("Linked to records", [f"composition year {poems('with a composition year')}",
                               f"exile episode {poems('linked to an exile episode')} · "
                               f"coordinates {poems('with composition coordinates')}"]),
        ("Place of writing vs exile seat",
         [f"{poems('Place of writing classifiable')} classifiable ({poets('Place of writing classifiable')} poets, "
          f"{note('Place of writing classifiable').split(';')[0]})",
          f"seat {poems('at the exile seat')} · elsewhere {poems('elsewhere (en route, side trips, after leaving)')}"
          f" · Chang'an {poems(chr(105) + 'n Chang' + chr(39) + 'an before departure')}",
          f"not classifiable {poems('Place of writing not classifiable')}: no reviewed place {unc[0]},",
          f"no episode {unc[1]}, unmapped seat {unc[2]}, coarse place {unc[3]}"])]
RIGHT = [("Poem texts", ["chinese-poetry corpus,", "matched to records by text hash"]),
         ("Five-axis annotation", [f"{poems('with five-axis annotation')} poems: court, landscape, temple,",
                                   "route, temporal void",
                                   "qwen3.5:27b first pass",
                                   "→ gpt-5.6-terra poem-by-poem review",
                                   "→ claude-sonnet-5 second review of",
                                   "   medium-confidence positions"])]
WORDS = ("Word lists", ["exchange titles,", "self-naming, capital-city,", "return, dwelling, grief"])
VALID = ("Validation", [f"blind, 50 poems: {int(allax['Agreed'])}/{int(allax['Determinate comparisons'])}",
                        f"hard cases: {int(hard['Agreed'])}/{int(hard['Determinate comparisons'])}",
                        f"(κ = {hard['κ']:.2f})"])


def height(lines):
    return PAD + TITLE_PT * LEAD / 72 + 0.05 + len(lines) * LINE + PAD


W, COLW, L, RX = 7.4, 3.5, 0.05, 3.85
HALF = (COLW - 0.25) / 2
top = 0.3  # room for the column headings
left_h = sum(height(b) for _, b in LEFT) + GAP * (len(LEFT) - 1)
right_h = sum(height(b) for _, b in RIGHT) + GAP * len(RIGHT) + max(height(WORDS[1]), height(VALID[1]))
bottom_h = height([*cmp_lines[:3], cmp_lines[4]])
H = top + max(left_h, right_h) + 0.35 + bottom_h + 0.05

plt.rcParams.update({"font.family": ["Microsoft JhengHei", "DejaVu Sans"], "svg.fonttype": "none"})
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box(x, y, w, title, lines, strong=False):
    """Draw a box with its top-left corner at (x, y) in inches; return side midpoints and the bottom edge."""
    h = height(lines)
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=FILL, ec=ACCENT if strong else EDGE, lw=1.2 if strong else 0.8))
    ax.text(x + PAD, y - PAD, title, ha="left", va="top", fontsize=TITLE_PT, color=INK, weight="bold")
    ax.text(x + PAD, y - PAD - TITLE_PT * LEAD / 72 - 0.05, "\n".join(lines), ha="left", va="top",
            fontsize=BODY_PT, color=MUTED, linespacing=LEAD)
    return {"top": (x + w / 2, y), "bottom": (x + w / 2, y - h), "left": (x, y - h / 2), "right": (x + w, y - h / 2),
            "y_end": y - h}


def arrow(a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, color=MUTED, lw=0.8,
                                 shrinkA=0.5, shrinkB=0.5))


ax.text(L, H - 0.05, "Poem-level linkage", fontsize=9.5, color=INK, weight="bold", va="top")
ax.text(RX, H - 0.05, "Textual measurement", fontsize=9.5, color=INK, weight="bold", va="top")

y = H - top
left = []
for title, lines in LEFT:
    left.append(box(L, y, COLW, title, lines))
    y = left[-1]["y_end"] - GAP
y = H - top
right = []
for title, lines in RIGHT:
    right.append(box(RX, y, COLW, title, lines))
    y = right[-1]["y_end"] - GAP
words = box(RX, y, HALF, *WORDS)
valid = box(RX + HALF + 0.25, y, HALF, *VALID)

y_bottom = min(left[-1]["y_end"], words["y_end"], valid["y_end"]) - 0.35
place_models = box(L, y_bottom, COLW, "Place and phase comparisons",
                   [*cmp_lines[:3], cmp_lines[4]], strong=True)
dated_models = box(RX, y_bottom, COLW, "All dated exile poems",
                   [cmp_lines[3]], strong=True)


for a, b in zip(left, left[1:]):
    arrow(a["bottom"], b["top"])
arrow(right[0]["bottom"], right[1]["top"])
arrow(left[1]["right"], right[1]["left"])  # the confirmed exile-period poems are the ones annotated
arrow((valid["top"][0], right[1]["y_end"]), valid["top"])  # annotations go to validation
arrow(words["right"], valid["left"])  # word lists are checked against the axes
arrow(left[-1]["bottom"], place_models["top"])
# The all-dated analysis branches from linked records before place classification.
x_branch = (L + COLW + RX) / 2
x_link, y_link = left[2]["right"]
y_dated = dated_models["left"][1]
ax.plot([x_link, x_branch, x_branch], [y_link, y_link, y_dated], color=MUTED, lw=0.8)
arrow((x_branch, y_dated), dated_models["left"])

for ext in ("png", "svg"):
    fig.savefig(f"{OUT}/fig0_process.{ext}", dpi=250, facecolor="white")
print("wrote", OUT)
