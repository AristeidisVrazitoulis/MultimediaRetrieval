import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbsr import config
from cbsr.retrieval import load_database, load_prepared
from cbsr.scalability import build_index, knn, index_vectors

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


if __name__ == "__main__":
    main()