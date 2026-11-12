"""Figure 3: poet-level means by year of first exile, as a dot plot with one labelled row per poet.

Reads results/topic1v_shang_period_poet_means.csv (built by 11_period_shang.py). Rows run from the earliest to the
latest first exile, grouped by period (early <= 712, high 713-765, mid 766-826, late >= 827); each dot is the
adjusted poet mean, sized by the number of poems. Many poets share a first-exile year, so a scatter against the year
cannot label every dot; ordering the rows keeps every poet named and matches the rank correlation reported in the
paper. Writes results/figures/fig2_period_trend.{png,svg}.
"""
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.environ.get("DH2027_RESULTS_DIR", os.path.join(ROOT, "results"))
OUT = os.path.join(R, "figures")
os.makedirs(OUT, exist_ok=True)
SERIES, INK, MUTED, GRID, BAND = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#f3f2ef"
PANELS = [("lex_capital", "Capital-city words (京闕)"),
          ("title_exchange", "Titles of letters and exchanges (寄贈酬答)")]
PERIODS = [("early", "Early Tang"), ("high", "High Tang"), ("mid", "Mid-Tang"), ("late", "Late Tang")]
# pinyin names match the names used in the paper
PINYIN = {"駱賓王": "Luo Binwang", "宋之問": "Song Zhiwen", "沈佺期": "Shen Quanqi", "張說": "Zhang Yue",
          "張九齡": "Zhang Jiuling", "王昌齡": "Wang Changling", "李白": "Li Bai", "杜甫": "Du Fu", "高適": "Gao Shi",
          "劉長卿": "Liu Changqing", "岑參": "Cen Shen", "顏真卿": "Yan Zhenqing", "劉禹錫": "Liu Yuxi",
          "柳宗元": "Liu Zongyuan", "元稹": "Yuan Zhen", "白居易": "Bai Juyi", "韓愈": "Han Yu", "李紳": "Li Shen",
          "賈島": "Jia Dao", "李德裕": "Li Deyu", "唐彥謙": "Tang Yanqian", "韓偓": "Han Wo"}

plt.rcParams.update({"font.family": ["Microsoft JhengHei", "DejaVu Sans"], "font.size": 9,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
                     "svg.fonttype": "none"})
pm = pd.read_csv(f"{R}/topic1v_shang_period_poet_means.csv")
rows = (pm[pm.outcome == PANELS[0][0]].sort_values(["start", "poems"], ascending=[True, False])
        [["poet", "start", "period", "poems"]].reset_index(drop=True))
rows["y"] = range(len(rows))
labels = [f"{PINYIN[p]} ({int(s)})" for p, s in zip(rows.poet, rows.start)]

fig, axes = plt.subplots(1, 2, figsize=(10, 6.4), sharex=True, sharey=True)
for ax, (outcome, title) in zip(axes, PANELS):
    d = rows.merge(pm[pm.outcome == outcome][["poet", "adjusted"]], on="poet")
    for k, (code, _) in enumerate(PERIODS):
        g = rows[rows.period == code]
        if k % 2 == 0:
            ax.axhspan(g.y.min() - 0.5, g.y.max() + 0.5, color=BAND, zorder=0, lw=0)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    ax.hlines(d.y, 0, 100 * d.adjusted, color=GRID, lw=1, zorder=2)
    ax.scatter(100 * d.adjusted, d.y, s=1.1 * d.poems, color=SERIES, edgecolor="white", linewidth=0.75, zorder=3)
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=8)
    ax.set_xlabel("Adjusted poet mean (percentage scale)", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(rows.y, labels)
axes[0].set_ylim(len(rows) - 0.5, -0.5)
# period names sit in the right margin of the right panel, centred on their band
for code, name in PERIODS:
    g = rows[rows.period == code]
    axes[1].text(1.02, (g.y.min() + g.y.max()) / 2, name, transform=axes[1].get_yaxis_transform(), va="center",
                 ha="left", fontsize=8.5, color=MUTED)
fig.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"{OUT}/fig2_period_trend.{ext}", dpi=200, facecolor="white")
print("wrote", OUT)
