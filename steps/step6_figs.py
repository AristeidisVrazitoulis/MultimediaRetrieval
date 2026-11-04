#Step 6 report material: figures and the full per-class table
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from cbsr import config
    STEP6_DIR = getattr(config, "STEP6_DIR", os.path.join(config.RESULTS_DIR, "step6"))
except Exception:
    STEP6_DIR = os.path.join("results", "step6")
FIG_DIR = os.path.join(STEP6_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2de"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False})

per_class = pd.read_csv(os.path.join(STEP6_DIR, "precision_per_class.csv"))
per_query = pd.read_csv(os.path.join(STEP6_DIR, "precision_at_k.csv"))
curve = pd.read_csv(os.path.join(STEP6_DIR, "precision_curve.csv"))

N = len(per_query)
grand = per_query["precision"].mean()                    #M_avg: average over all shapes
chance = (per_class["n_shapes"] * (per_class["n_shapes"] - 1)).sum() / (N * (N - 1))
per_class["chance"] = (per_class["n_shapes"] - 1) / (N - 1)
per_class["times_chance"] = per_class["mean_precision"] / per_class["chance"]


#figure: precision@10 per class
d = per_class.sort_values("mean_precision")
fig, ax = plt.subplots(figsize=(6.6, 9.6))
ax.barh(d["class"], d["mean_precision"], color=COLORS[0], height=0.62)
ax.axvline(grand, color=INK, linestyle="--", linewidth=1.2)
ax.axvline(chance, color=INK2, linestyle=":", linewidth=1.2)
ax.text(grand + 0.008, len(d) + 0.15, f"grand average {grand:.3f}", fontsize=8, va="center")
ax.text(chance + 0.008, len(d) + 0.15, f"random ranking {chance:.3f}", fontsize=8, va="center", color=INK2)
ax.set_xlabel("mean precision@10 of the class")
ax.set_xlim(0, 0.8)
ax.set_ylim(-0.8, len(d) + 0.7)
ax.tick_params(axis="y", labelsize=7, length=0)
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "precision_per_class.png"), dpi=200)
plt.close(fig)


#figure: precision@K curves
KS = [1, 3, 5, 10, 20]
SHOWN = ["Bicycle", "Sword", "Humanoid", "BuildingNonResidential", "Skyscraper", "Vase"]
style = dict(zip(SHOWN, zip(COLORS, MARKERS)))
g = curve[curve["class"] == "ALL (grand average)"].iloc[0]


def draw(ax, classes, show_grand):
    if show_grand:
        ax.plot(KS, [g[f"p@{k}"] for k in KS], color=INK, linestyle="--", linewidth=2, label="grand average")
    ax.axhline(chance, color=INK2, linestyle=":", linewidth=1.2, label="random ranking")
    for c in classes:
        color, marker = style[c]
        row = curve[curve["class"] == c].iloc[0]
        ax.plot(KS, [row[f"p@{k}"] for k in KS], color=color, marker=marker, markersize=6,
                linewidth=2, markeredgecolor="#fcfcfb", markeredgewidth=1, label=c)
    ax.set_xticks(KS)
    ax.set_xlabel("number of results $k$")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.9))
draw(a1, SHOWN[:3], True)
a1.set_ylim(0, 1.05)
a1.set_ylabel("precision@$k$")
a1.set_title("(a) overview", fontsize=9, color=INK2)
draw(a2, SHOWN[3:], False)
a2.set_ylim(0, 0.16)
a2.set_title("(b) zoom on the three worst classes", fontsize=9, color=INK2)
handles, labels = [], []
for a in (a1, a2):
    for h, l in zip(*a.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h)
            labels.append(l)
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig(os.path.join(FIG_DIR, "precision_curve_report.png"), dpi=200)
plt.close(fig)


#full per-class table
#rows = []
#for _, r in per_class.sort_values("mean_precision", ascending=False).iterrows():
#    name = r["class"].replace("_", "\\_")
#    rows.append(f"{name} & {int(r['n_shapes'])} & {r['mean_precision']:.3f} & "
#                f"{r['times_chance']:.0f} \\\\")
#tex = ["\\begin{longtable}{lrrr}",
#       "\\toprule",
#       "Class $c$ & $n_c$ & $P_{10}(c)$ & $\\times$ random \\\\",
#       "\\midrule\\endfirsthead",
#       "\\toprule",
#       "Class $c$ & $n_c$ & $P_{10}(c)$ & $\\times$ random \\\\",
#       "\\midrule\\endhead",
#       *rows,
#       "\\bottomrule",
#       "\\end{longtable}"]
#with open(os.path.join(STEP6_DIR, "per_class_table.tex"), "w") as f:
#    f.write("\n".join(tex) + "\n")