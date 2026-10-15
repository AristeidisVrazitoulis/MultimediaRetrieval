import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from features import N_BINS, HIST_RANGES, HIST_FEATURES, SCALAR_FEATURES

FIGURES_DIR = "stats/figures_features"


def hist_matrix(df, name):
    return df[[f"{name}_{i}" for i in range(N_BINS)]].to_numpy()


def bin_centers(name):
    lo, hi = HIST_RANGES[name]
    edges = np.linspace(lo, hi, N_BINS + 1)
    return (edges[:-1] + edges[1:]) / 2


# tech tips 03c (3), Figures 1-2: one subplot per class, one line per shape.
# Good feature: lines grouped within a plot, different "signatures" across plots
def plot_histograms_per_class(df, name, classes=None, n_cols=6):
    classes = classes or sorted(df["class"].unique())
    n_rows = math.ceil(len(classes) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.2 * n_rows), sharex=True, sharey=True)
    x = bin_centers(name)
    ymax = hist_matrix(df, name).max()
    for ax, c in zip(axes.flat, classes):
        for h in hist_matrix(df[df["class"] == c], name):
            ax.plot(x, h, linewidth=1)
        ax.set_title(c, fontsize=8)
        ax.set_ylim(0, ymax)
    for ax in list(axes.flat)[len(classes):]:
        ax.axis("off")
    fig.suptitle(f"{name} histograms per class")
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}_per_class.png"), dpi=120)
    plt.close(fig)


# tech tips 03b (2): two shapes of the same class + one of a different class.
# The first two histograms should be similar, and both different from the third
def compare_three_shapes(df, same_a, same_b, other):
    rows = [df[df["file"].str.replace("\\", "/").str.endswith(s)].iloc[0] for s in (same_a, same_b, other)]
    fig, axes = plt.subplots(1, len(HIST_FEATURES), figsize=(4 * len(HIST_FEATURES), 3.2))
    for ax, name in zip(axes, HIST_FEATURES):
        x = bin_centers(name)
        for r, style in zip(rows, ["-", "-", "--"]):
            h = [r[f"{name}_{i}"] for i in range(N_BINS)]
            ax.plot(x, h, style, label=os.path.basename(os.path.dirname(r["file"])) + "/" + os.path.basename(r["file"]))
        ax.set_title(name)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "compare_" + "_".join(os.path.basename(s).split(".")[0] for s in (same_a, same_b, other)) + ".png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# tech tips 03b (2): for each scalar descriptor, the shapes with the lowest and highest
# value. Open them in the viewer: the values should match what you see
# (thin vs bulky for eccentricity, box-like vs round for rectangularity, ...)
def scalar_extremes(df, n=2):
    rows = []
    for name in SCALAR_FEATURES:
        s = df.sort_values(name)
        for kind, part in (("lowest", s.head(n)), ("highest", s.tail(n))):
            for _, r in part.iterrows():
                rows.append({"descriptor": name, "kind": kind, "value": r[name], "file": r["file"]})
    return pd.DataFrame(rows)


# a number to go with the plots of 03c (3): average L1 distance between histograms of
# shapes in the same class vs in different classes. Ratio < 1 means the descriptor
# separates classes; close to 1 means it does not
def within_vs_between(df):
    labels = df["class"].to_numpy()
    same = labels[:, None] == labels[None, :]
    off_diag = ~np.eye(len(df), dtype=bool)
    rows = []
    for name in HIST_FEATURES:
        H = hist_matrix(df, name)
        dist = np.abs(H[:, None, :] - H[None, :, :]).sum(2)
        within = dist[same & off_diag].mean()
        between = dist[~same].mean()
        rows.append({"descriptor": name, "within_class": within, "between_class": between, "ratio": within / between})
    return pd.DataFrame(rows)


def volume_reliability(df):
    # volume-based descriptors are only trustworthy when the repaired mesh is closed
    # and the shape is not "more than convex"
    closed = df["boundary_edges_after"] == 0
    bad_convexity = df["convexity"] > 1.0
    print(f"closed after repair: {closed.sum()}/{len(df)}")
    print(f"convexity > 1 (volume overestimated): {bad_convexity.sum()}")
    return df[~closed | bad_convexity][["file", "boundary_edges_before", "boundary_edges_after", "volume", "convexity"]]
