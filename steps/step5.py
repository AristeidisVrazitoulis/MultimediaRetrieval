import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbsr import config
from cbsr.retrieval import load_database, load_prepared
from cbsr.scalability import build_index, knn, range_search, tsne_embedding, index_vectors
from analysis.scalability_checks import (query_table, plot_query_table, compare_searches,
                                         map_precision, plot_tsne, tsne_html)


MAKE_QUERY_TABLE = True     # results/step5/query_table.csv + figures/query_table_<method>.png (renders)
COMPARE_SEARCHES = True     # results/step5/search_comparison.csv: precision@K and time, custom vs kd-tree
MAKE_TSNE = True            # results/step5/tsne_perplexity<p>.csv + figures/tsne_perplexity<p>.png/.html
K = 6                       # neighbours shown per query 

# one query shape per class: classes where Step 4 works well (Bicycle, Jet), average (Humanoid, Mug)
# and badly (Apartment)
QUERY_FILES = [
    "data/aligned/Bicycle/D00016.obj",
    "data/aligned/Jet/m1147.obj",
    "data/aligned/Humanoid/m118.obj",
    "data/aligned/Mug/D00172.obj",
    "data/aligned/Apartment/D00045.obj",
]
# t-SNE is sensitive to its settings: compare a few perplexities, then pick one
PERPLEXITIES = [10, 30, 40, 50]


db = load_database()
# the same normalization parameters and block stds as Step 4 (results/step4/prepared_params.json)
prep, from_file = load_prepared(db)
print("Step 4 configuration:", prep["config"], "| loaded from file:", from_file)

# Step 5.1: spatial search structure, built once for the whole database
tree = build_index(prep)
print(f"kd-tree over {len(db)} shapes, {index_vectors(prep).shape[1]} dimensions")
os.makedirs(config.STEP5_DIR, exist_ok=True)

# K-nearest neighbours of one shape
example = db[db.file == QUERY_FILES[0]].iloc[0].to_dict()
print(knn(tree, prep, example, K))

# R-nearest neighbours: R = median distance of the K-th neighbour over the database, so on average
# a query gets about K results; the number of results now depends on the query
d_k = tree.query(index_vectors(prep), k=K + 1)[0][:, -1]
R = float(np.median(d_k))
print(f"R = {R:.2f} (median distance to the {K}-th neighbour)")
for f in QUERY_FILES:
    q = db[db.file == f].iloc[0].to_dict()
    res = range_search(tree, prep, q, R)
    same = (res["class"] == q["class"]).sum()
    print(f"  {f}: {len(res)} shapes within R, {same} of the same class")

# Query table for the report: the same queries with the custom distance and with the kd-tree
if MAKE_QUERY_TABLE:
    table = query_table(db, prep, tree, QUERY_FILES, K)
    table.to_csv(config.QUERY_TABLE_CSV, index=False)
    print("saved", config.QUERY_TABLE_CSV)
    cache = {}
    for method in ("custom", "kdtree"):
        plot_query_table(table, method, f"query_table_{method}.png", cache)

# Is the kd-tree (L2) as good as the custom distance, and faster?
if COMPARE_SEARCHES:
    comparison = compare_searches(db, prep, tree, K)
    comparison.to_csv(config.SEARCH_COMPARISON_CSV, index=False)
    print(comparison.to_string(index=False))

# Step 5.2: t-SNE map of the feature database, coloured by class
if MAKE_TSNE:
    rows = []
    for p in PERPLEXITIES:
        emb, kl = tsne_embedding(prep, perplexity=p)
        emb.to_csv(config.TSNE_CSV.format(perplexity=p), index=False)
        title = f"t-SNE of the feature database (perplexity {p}, {config.TSNE_MAX_ITER} iterations)"
        plot_tsne(emb, title, f"tsne_perplexity{p}.png")
        tsne_html(emb, title, f"tsne_perplexity{p}.html")
        rows.append({"perplexity": p, "kl_divergence": kl, "map_precision@5": map_precision(emb)})
    print(pd.DataFrame(rows).to_string(index=False))
