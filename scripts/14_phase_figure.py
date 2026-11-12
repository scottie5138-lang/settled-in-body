"""Figure 2: the five axes by phase of exile, as within-poet differences from poems written en route.

Reads results/topic1z_phase_within_poet.csv (built by 07_settling_evidence.py) and results/public_counts.json (18), the inputs for the paper's
phase estimates. One panel per axis; each dot is the difference from en route in
percentage points with its poet-clustered bootstrap 95% CI, and en route is drawn at zero as the reference. The
"four or more years" phase is left out because only Liu Yuxi contributes it to this model. Panels run from the axes
that change to the one that persists. Writes results/figures/fig1_phase_axes.{png,svg}.
"""
import json
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.environ.get("DH2027_RESULTS_DIR", os.path.join(ROOT, "results"))
OUT = os.path.join(R, "figures")
os.makedirs(OUT, exist_ok=True)
SERIES, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
PANELS = [("route_displacement", "Route (行旅位移)"), ("temple_reclusion", "Temple (寺院隱逸)"),
          ("landscape", "Landscape (山水地景)"), ("time_void", "Temporal void (時間虛空)"),
          ("court_institution", "Court (朝廷制度)")]
PHASES = [("en route", "En\nroute"), ("arrival year", "Arrival\n(proxy)"), ("settled +1", "+1\nyear"),
          ("settled +2-3", "+2–3\nyears")]

plt.rcParams.update({"font.family": ["Microsoft JhengHei", "DejaVu Sans"], "font.size": 9,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "svg.fonttype": "none"})
ph = pd.read_csv(f"{R}/topic1z_phase_within_poet.csv").set_index(["outcome", "term"])
# poets per phase and the largest single-poet share, aggregated from the phase sample by 18_public_counts.py
pc = json.load(open(f"{R}/public_counts.json", encoding="utf-8"))
phase_poets = pd.Series(pc["phase_poets"])
# a poet-bootstrap interval says little when one poet supplies most of a phase, so intervals need both
top_share = pd.Series(pc["phase_top_poet_share"])
show_interval = (phase_poets >= 3) & (top_share <= 0.5)
fig, axes = plt.subplots(1, len(PANELS), figsize=(10, 3.4), sharey=True)
for ax, (y, title) in zip(axes, PANELS):
    ax.axhline(0, color=MUTED, lw=1, zorder=1)
    for i, (term, _) in enumerate(PHASES):
        if term == "en route":
            ax.scatter(i, 0, s=28, facecolor="white", edgecolor=SERIES, linewidth=1.5, zorder=3)
            continue
        r = ph.loc[(y, term)]
        if show_interval[term]:
            ax.plot([i, i], [100 * r.ci_low, 100 * r.ci_high], color=SERIES, lw=1.5, alpha=0.45, zorder=2)
        ax.scatter(i, 100 * r.slope, s=28, color=SERIES, edgecolor="white", linewidth=1.5, zorder=3)
        v = round(100 * r.slope)
        ax.annotate("0" if v == 0 else f"{v:+d}".replace("-", "−"), (i, 100 * r.slope), xytext=(6, -3), textcoords="offset points",
                    fontsize=7, color=INK)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    ax.set_xticks(range(len(PHASES)), [lab for _, lab in PHASES], fontsize=7.5)
    ax.set_xlim(-0.5, len(PHASES) - 0.4)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("Difference from en route\n(percentage points, within poets)")
# the model, interval rule and omitted phases are stated in the paper's figure caption, not on the image
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"{OUT}/fig1_phase_axes.{ext}", dpi=200, facecolor="white")
print("wrote", OUT)
