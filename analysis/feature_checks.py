import os
import math
import tempfile
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt

from cbsr.features import N_BINS, HIST_RANGES, HIST_FEATURES, SCALAR_FEATURES, extract_features
from cbsr.config import STEP3_FIGURES

FIGURES_DIR = STEP3_FIGURES


# check the descriptors on simple shapes whose values we know
# (cube of side 1, sphere of radius 0.5, rod 1 x 0.1 x 0.1). Prints computed (expected)
def validate_on_primitives():
    cube = o3d.geometry.TriangleMesh.create_box(1, 1, 1).subdivide_midpoint(4)
    sphere = o3d.geometry.TriangleMesh.create_sphere(0.5, resolution=60)
    rod = o3d.geometry.TriangleMesh.create_box(1, 0.1, 0.1).subdivide_midpoint(4)
    expected = {
        "cube":   (cube,   {"compactness": 216 / (36 * np.pi), "rectangularity": 1, "diameter": np.sqrt(3), "convexity": 1, "eccentricity": 1}),
        "sphere": (sphere, {"compactness": 1, "rectangularity": np.pi / 6, "diameter": 1, "convexity": 1, "eccentricity": 1}),
        "rod":    (rod,    {"compactness": 0.42 ** 3 / (36 * np.pi * 0.01 ** 2), "rectangularity": 1, "diameter": np.sqrt(1.02), "convexity": 1, "eccentricity": None}),
    }
    rows = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, (mesh, values) in expected.items():
            path = os.path.join(tmp, name + ".obj")
            o3d.io.write_triangle_mesh(path, mesh)
            feats = extract_features(path)
            rows[name] = {k: f"{feats[k]:.3f} ({'>> 1' if v is None else f'{v:.3f}'})" for k, v in values.items()}
    print(pd.DataFrame(rows).T.to_string())


def hist_matrix(df, name):
    return df[[f"{name}_{i}" for i in range(N_BINS)]].to_numpy()


def bin_centers(name):
    lo, hi = HIST_RANGES[name]
    edges = np.linspace(lo, hi, N_BINS + 1)
    return (edges[:-1] + edges[1:]) / 2


# Figures 1-2: one subplot per class, one line per shape.
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


# two shapes of the same class + one of a different class.
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


VOLUME_FEATURES = ["volume", "compactness", "rectangularity", "convexity"]


# volume-based descriptors are only trustworthy when the repaired mesh is closed and the
# values are physically possible (convexity <= 1, compactness >= 1)
def reliable_volume(df):
    return (df["boundary_edges_after"] == 0) & (df["convexity"] <= 1) & (df["compactness"] >= 1)


# for each scalar descriptor, the shapes with the lowest and highest
# value. Open them in the viewer: the values should match what you see
# (thin vs bulky for eccentricity, box-like vs round for rectangularity, ...).
# Volume-based descriptors only consider shapes with a reliable volume
def scalar_extremes(df, n=2):
    rows = []
    reliable = df[reliable_volume(df)]
    for name in SCALAR_FEATURES:
        s = (reliable if name in VOLUME_FEATURES else df).sort_values(name)
        for kind, part in (("lowest", s.head(n)), ("highest", s.tail(n))):
            for _, r in part.iterrows():
                rows.append({"descriptor": name, "kind": kind, "value": r[name], "file": r["file"]})
    return pd.DataFrame(rows)


# a number to go with the plots, using each class "signature" (its average
# histogram): within = average L1 distance of a shape to the signature of its own class,
# between = average L1 distance between the signatures of different classes.
# Ratio < 1 means the descriptor separates classes; close to 1 means it does not
def within_vs_between(df):
    rows = []
    for name in HIST_FEATURES:
        cols = [f"{name}_{i}" for i in range(N_BINS)]
        signatures = df.groupby("class")[cols].mean()
        H, S = df[cols].to_numpy(), signatures.loc[df["class"]].to_numpy()
        within = np.abs(H - S).sum(1).mean()
        C = signatures.to_numpy()
        between = np.abs(C[:, None, :] - C[None, :, :]).sum(2)[~np.eye(len(C), dtype=bool)].mean()
        rows.append({"descriptor": name, "within_class": within, "between_class": between, "ratio": within / between})
    return pd.DataFrame(rows)


# shapes whose volume-based descriptors are not reliable (see reliable_volume)
def volume_reliability(df):
    b = df["boundary_edges_after"]
    open_edges = b[b > 0]
    big_holes = df[b > 200]
    share = (big_holes["class"].value_counts() / df["class"].value_counts()).dropna()
    print(f"closed after repair: {(b == 0).sum()}/{len(df)}")
    print(f"open after repair: {len(open_edges)}, median boundary edges {open_edges.median():.0f}, "
          f"<= 50 edges: {(open_edges <= 50).sum()}, > 200 edges: {len(big_holes)}")
    print("classes with > 200 boundary edges in more than half of their shapes:",
          share[share > 0.5].sort_values(ascending=False).round(2).to_dict())
    over_convex = df[df["convexity"] > 1]
    print(f"convexity > 1 (volume overestimated): {len(over_convex)}", over_convex["class"].value_counts().to_dict())
    under_compact = df[df["compactness"] < 1]
    print(f"compactness < 1: {len(under_compact)}", dict(zip(under_compact["file"], under_compact["compactness"].round(3))))
    print(f"reliable volume: {reliable_volume(df).sum()}/{len(df)}")
    return df[~reliable_volume(df)][["file", "boundary_edges_before", "boundary_edges_after", "volume", "compactness", "convexity"]]
