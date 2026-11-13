# Step 6: evaluation of the retrieval quality. One script, run from the project root:  python steps/step6.py
#
# Part 1  The evaluated system = the k-nearest-neighbour search of Step 5 (kd-tree, L2) with the default
#         setting of Step 4 (standardization, Euclidean for the global block, EMD for the histograms).
#         Every shape of the database is a query (the query itself is left out of its results).
#         Measure 1: precision@k (overall, per class, curve over k, confusion of the worst classes).
# Part 2  Measure 2, the ROC curve (the whole ranking, not only the first k), and a test of two changes:
#         (a) Manhattan distance for all blocks (a kd-tree with the L1 metric then gives exactly the Step 4
#         distance), (b) one weight per block. Choosing a setting or weights with the same queries that are
#         used to report the score would be optimistic, so the queries are split 50/50 (stratified by class)
#         five times: everything is chosen on one half and the score is reported on the other half only.
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbsr import config
from cbsr.retrieval import load_database, load_prepared, BLOCKS
from cbsr.scalability import build_index, knn, index_vectors, index_sigma

K = 10                          #main K for the per-shape / per-class / grand-average metric
CURVE_KS = [1, 3, 5, 10, 20]    #K values for the precision@K curve
N_WORST_FOR_CONFUSION = 5       #how many worst classes to break down

STEP6_DIR = getattr(config, "STEP6_DIR", os.path.join(config.RESULTS_DIR, "step6"))
STEP6_FIGURES = os.path.join(STEP6_DIR, "figures")
RESULTS_CSV = os.path.join(STEP6_DIR, "precision_at_k.csv")
PER_CLASS_CSV = os.path.join(STEP6_DIR, "precision_per_class.csv")
CONFUSION_CSV = os.path.join(STEP6_DIR, "confusion_worst_classes.csv")
CURVE_CSV = os.path.join(STEP6_DIR, "precision_curve.csv")


def precision_at_k(tree, prep, db, k=K):
    #one row per query shape
    rows = []
    for _, q in db.iterrows():
        res = knn(tree, prep, q.to_dict(), k)
        tp = (res["class"] == q["class"]).sum()
        rows.append({"file": q["file"], "class": q["class"], "precision": tp / k})
    return pd.DataFrame(rows)

def confusion_breakdown(tree, prep, db, classes, k=K, top_n=5):
    rows = []
    for c in classes:
        counter = Counter()
        shapes = db[db["class"] == c]
        for _, q in shapes.iterrows():
            res = knn(tree, prep, q.to_dict(), k)
            wrong = res[res["class"] != c]["class"]
            counter.update(wrong)
        total_fp = sum(counter.values())
        for other_class, n in counter.most_common(top_n):
            rows.append({"class": c, "confused_with": other_class, "count": n,
                         "share_of_fp": n / total_fp if total_fp else 0.0})
    return pd.DataFrame(rows)

def precision_curve(tree, prep, db, ks=CURVE_KS):
    kmax = max(ks)
    per_shape = {k: [] for k in ks}
    classes = []
    for _, q in db.iterrows():
        res = knn(tree, prep, q.to_dict(), kmax)
        same = (res["class"] == q["class"]).to_numpy()
        classes.append(q["class"])
        for k in ks:
            per_shape[k].append(same[:k].sum() / k)

    df = pd.DataFrame({"class": classes, **{f"p@{k}": per_shape[k] for k in ks}})
    per_class = df.groupby("class").mean(numeric_only=True).reset_index()
    grand = df.drop(columns="class").mean().to_frame().T
    grand.insert(0, "class", "ALL (grand average)")
    return pd.concat([per_class, grand], ignore_index=True), df


def plot_precision_curve(curve_df, classes_to_highlight, path):
    plt.figure(figsize=(7, 5))
    ks = CURVE_KS
    grand = curve_df[curve_df["class"] == "ALL (grand average)"].iloc[0]
    plt.plot(ks, [grand[f"p@{k}"] for k in ks], "k--", linewidth=2, label="grand average")
    for c in classes_to_highlight:
        row = curve_df[curve_df["class"] == c]
        if len(row):
            row = row.iloc[0]
            plt.plot(ks, [row[f"p@{k}"] for k in ks], marker="o", label=c)
    plt.xlabel("K")
    plt.ylabel("precision@K")
    plt.title("precision@K vs K, grand average and selected classes")
    plt.legend(fontsize=8)
    plt.xticks(ks)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()




# ================================================================ part 2: ROC and improvement
N_SPLITS = 5
WEIGHT_GRID = (0.5, 0.7, 1.43, 2.0)      # a block weight is multiplied by one of these factors per try
PASSES = 3


# ---------------------------------------------------------------- distances between all pairs, per block

def block_matrices(prep, metric):
    # delta_j / sigma_j for every pair of database shapes. metric "euclidean": what the L2 kd-tree sees
    # (index vector = block / sigma, so the SQUARED matrix is added); "cityblock": the Step 4 distance
    # when all blocks use L1 / EMD. Each block is divided by the std of ITS metric's distance: the
    # kd-tree's L2 sigma (index_sigma) or the Step 4 sigma
    sigma = index_sigma(prep) if metric == "euclidean" else prep["sigma"]
    return {b: (cdist(prep["vectors"][b], prep["vectors"][b], metric) / sigma[b]).astype(np.float32)
            for b in BLOCKS}


def combined(M, mode, w, rows):
    if mode == "l2":
        D = np.sqrt(sum((w[b] ** 2) * M[b][rows] ** 2 for b in BLOCKS))
    else:
        D = sum(w[b] * M[b][rows] for b in BLOCKS)
    D = np.array(D, dtype=np.float32)
    D[np.arange(len(rows)), rows] = np.inf          # a query is not its own result
    return D


def precision(D, cls, rows, k):
    idx = np.argpartition(D, k, axis=1)[:, :k]
    return (cls[idx] == cls[rows][:, None]).mean(1)


def tune_weights(M, mode, cls, rows):
    # coordinate search: try to scale one block weight at a time, keep a change if the mean
    # precision@10 of the given queries goes up
    w = {b: 1.0 for b in BLOCKS}
    best = precision(combined(M, mode, w, rows), cls, rows, K).mean()
    for _ in range(PASSES):
        improved = False
        for b in BLOCKS:
            for g in WEIGHT_GRID:
                w2 = dict(w)
                w2[b] = w[b] * g
                s = precision(combined(M, mode, w2, rows), cls, rows, K).mean()
                if s > best + 1e-9:
                    best, w, improved = s, w2, True
        if not improved:
            break
    return w, best


def stratified_halves(cls, seed):
    rng = np.random.default_rng(seed)
    dev = []
    for c in np.unique(cls):
        ids = np.where(cls == c)[0]
        rng.shuffle(ids)
        dev += list(ids[: len(ids) // 2])
    dev = np.array(sorted(dev))
    return dev, np.setdiff1d(np.arange(len(cls)), dev)


# ---------------------------------------------------------------- ROC

def roc_curves(D, cls, grid):
    # one ROC curve per query: rank ALL other shapes by distance, a shape is positive if it has the class
    # of the query. TPR = found / (n_c - 1) = recall, FPR = wrong ones passed / all wrong ones.
    # The curves are resampled on a common FPR grid and averaged.
    n = len(D)
    curves, auc = np.zeros((n, len(grid))), np.zeros(n)
    for r in range(n):
        order = np.argsort(D[r], kind="stable")[:-1]        # the query itself (inf) is last
        pos = cls[order] == cls[r]
        tpr = np.concatenate([[0], np.cumsum(pos) / pos.sum()])
        fpr = np.concatenate([[0], np.cumsum(~pos) / (~pos).sum()])
        curves[r] = np.interp(grid, fpr, tpr)
        auc[r] = np.trapezoid(tpr, fpr)
    return curves, auc


def main():
    db = load_database()
    prep, from_file = load_prepared(db)
    print("Step 4 configuration:", prep["config"], "| loaded from file:", from_file)

    tree = build_index(prep)
    print(f"kd-tree over {len(db)} shapes, {index_vectors(prep).shape[1]} dimensions; K={K}")

    os.makedirs(STEP6_FIGURES, exist_ok=True)

    per_query = precision_at_k(tree, prep, db, K)
    per_query.to_csv(RESULTS_CSV, index=False)
    print("saved", RESULTS_CSV)

    per_class = (per_query.groupby("class")
                 .agg(mean_precision=("precision", "mean"), n_shapes=("precision", "size"))
                 .reset_index()
                 .sort_values("mean_precision", ascending=False))
    per_class.to_csv(PER_CLASS_CSV, index=False)
    print("saved", PER_CLASS_CSV)

    m_avg = per_query["precision"].mean()
    print(f"\nGrand average precision@{K} over {len(db)} shapes: {m_avg:.4f}")
    print("\nBest 5 classes:")
    print(per_class.head(5).to_string(index=False))
    worst = per_class.tail(N_WORST_FOR_CONFUSION)
    print(f"\nWorst {N_WORST_FOR_CONFUSION} classes:")
    print(worst.to_string(index=False))

    confusion = confusion_breakdown(tree, prep, db, worst["class"].tolist(), K)
    confusion.to_csv(CONFUSION_CSV, index=False)
    print("\nsaved", CONFUSION_CSV)
    print("\nWhat the worst classes get confused with (top confuser per class):")
    print(confusion.sort_values(["class", "count"], ascending=[True, False])
          .groupby("class").head(1).to_string(index=False))

    curve_per_class, _ = precision_curve(tree, prep, db, CURVE_KS)
    curve_per_class.to_csv(CURVE_CSV, index=False)
    print("\nsaved", CURVE_CSV)
    highlight = per_class.head(2)["class"].tolist() + worst["class"].tolist()[:2]
    plot_precision_curve(curve_per_class, highlight, os.path.join(STEP6_FIGURES, "precision_curve.png"))
    print("saved", os.path.join(STEP6_FIGURES, "precision_curve.png"))


def improvement():

    db = load_database()
    cls = db["class"].to_numpy()
    n = len(db)
    everyone = np.arange(n)
    equal = {b: 1.0 for b in BLOCKS}

    #the evaluated system (Step 6 so far) and the proposed one
    base, _ = load_prepared(db)                                           # standardize / Euclid / EMD
    new, _ = load_prepared(db, scalar_distance="manhattan", hist_distance="manhattan")
    M_base = block_matrices(base, "euclidean")
    M_new = block_matrices(new, "cityblock")

    #held-out test of the change
    rows = []
    for s in range(N_SPLITS):
        dev, test = stratified_halves(cls, s)
        w, _ = tune_weights(M_new, "l1", cls, dev)
        rows.append({"split": s,
                     "evaluated_system": precision(combined(M_base, "l2", equal, test), cls, test, K).mean(),
                     "manhattan_equal_weights": precision(combined(M_new, "l1", equal, test), cls, test, K).mean(),
                     "manhattan_tuned_weights": precision(combined(M_new, "l1", w, test), cls, test, K).mean(),
                     **{f"w_{b}": w[b] for b in BLOCKS}})
    held = pd.DataFrame(rows)
    held.to_csv(os.path.join(STEP6_DIR, "heldout_weights.csv"), index=False)
    print("precision@10 on the held-out half of the queries (5 random splits):")
    print(held.drop(columns="split").agg(["mean", "std"]).round(4).to_string())

    #final weights, tuned on all queries (the score of these is optimistic: see the held-out table)
    w_all, in_sample = tune_weights(M_new, "l1", cls, everyone)
    with open(os.path.join(STEP6_DIR, "block_weights.json"), "w") as f:
        json.dump({"weights": w_all, "precision@10_in_sample": in_sample}, f, indent=1)
    print("weights tuned on all queries:", {b: round(v, 3) for b, v in w_all.items()}, "| in-sample P@10:", round(in_sample, 4))

    #precision@k with a real kd-tree (L1 on the weighted index vectors)
    X = np.hstack([w_all[b] * new["vectors"][b] / new["sigma"][b] for b in BLOCKS])
    tree = KDTree(X, leaf_size=config.KDTREE_LEAF_SIZE, metric="manhattan")
    _, idx = tree.query(X, k=max(CURVE_KS) + 1)
    idx = np.array([[j for j in row if j != i][:max(CURVE_KS)] for i, row in enumerate(idx)])
    same = cls[idx] == cls[:, None]
    D_base = combined(M_base, "l2", equal, everyone)
    D_eq = combined(M_new, "l1", equal, everyone)
    D_tuned = combined(M_new, "l1", w_all, everyone)
    curve = pd.DataFrame({"k": CURVE_KS,
                          "evaluated_system": [precision(D_base, cls, everyone, k).mean() for k in CURVE_KS],
                          "manhattan_equal_weights": [precision(D_eq, cls, everyone, k).mean() for k in CURVE_KS],
                          "manhattan_tuned_weights_kdtree": [same[:, :k].mean() for k in CURVE_KS]})
    curve.to_csv(os.path.join(STEP6_DIR, "precision_curve_improved.csv"), index=False)
    print(curve.round(4).to_string(index=False))

    #ROC of the evaluated system and the proposed one (all queries)
    grid = np.linspace(0, 1, 1001)
    out = {}
    for name, D in [("evaluated_system", D_base), ("manhattan_tuned_weights", D_tuned)]:
        c, a = roc_curves(D, cls, grid)
        out[name] = (c.mean(0), a)
        t = lambda f: float(np.interp(f, grid, c.mean(0)))
        print(f"{name}: mean AUC {a.mean():.4f}; TPR at FPR 0.01 / 0.05 / 0.1 = {t(.01):.3f} / {t(.05):.3f} / {t(.1):.3f}")
    pd.DataFrame({"fpr": grid, **{f"tpr_{k}": v[0] for k, v in out.items()}}).to_csv(
        os.path.join(STEP6_DIR, "roc_curve.csv"), index=False)
    pd.DataFrame({"class": cls, "file": db["file"], **{f"auc_{k}": v[1] for k, v in out.items()}}).to_csv(
        os.path.join(STEP6_DIR, "auc_per_query.csv"), index=False)

    #figure
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2de"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.6))
    for ax, xmax in [(a1, 1.0), (a2, 0.1)]:
        ax.plot(grid, out["evaluated_system"][0], color="#2a78d6", linewidth=2,
                label=f"evaluated system (AUC {out['evaluated_system'][1].mean():.3f})")
        ax.plot(grid, out["manhattan_tuned_weights"][0], color="#eb6834", linewidth=2, linestyle="--",
                label=f"Manhattan, tuned weights (AUC {out['manhattan_tuned_weights'][1].mean():.3f})")
        ax.plot([0, xmax], [0, xmax], color=INK2, linestyle=":", linewidth=1.2, label="random ranking")
        ax.set_xlim(0, xmax)
        ax.set_xlabel("false positive rate")
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    a1.set_ylim(0, 1)
    a1.set_ylabel("true positive rate (recall)")
    a1.set_title("(a) full ROC curve", fontsize=9, color=INK2)
    a2.set_ylim(0, 0.6)
    a2.set_title("(b) zoom on false positive rate $\\leq$ 0.1", fontsize=9, color=INK2)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(STEP6_FIGURES, "roc_curve_report.png"), dpi=200)
    print("saved", os.path.join(STEP6_FIGURES, "roc_curve_report.png"))




if __name__ == "__main__":
    main()
    improvement()