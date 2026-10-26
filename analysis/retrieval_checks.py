# checks for the Step 4 design choices: how well does each distance configuration retrieve
# shapes of the query's class? (a small preview of the Step 6 evaluation)
import numpy as np
import pandas as pd

from cbsr.retrieval import METRICS


def all_distances(prep, rows):
    # combined distance from each database shape in `rows` (as query) to every database shape
    total = np.zeros((len(rows), len(prep["shapes"])))
    for blk, V in prep["vectors"].items():
        d = METRICS[prep["metrics"][blk]](V[rows][:, None, :] - V[None, :, :])
        total += d / prep["sigma"][blk]
    return total


def precision_at_k(prep, k=5, chunk=200):
    # every database shape is used once as the query (and left out of its own results);
    # precision@k = fraction of the k closest shapes that belong to the query's class
    classes = prep["shapes"]["class"].to_numpy()
    n = len(classes)
    hits = np.empty(n)
    for start in range(0, n, chunk):
        rows = np.arange(start, min(start + chunk, n))
        d = all_distances(prep, rows)
        d[np.arange(len(rows)), rows] = np.inf                 # leave the query itself out
        nearest = np.argpartition(d, k, axis=1)[:, :k]
        hits[rows] = (classes[nearest] == classes[rows][:, None]).mean(1)
    return float(hits.mean())


def compare_configurations(db, configs, prepare, k=5):
    # configs: list of dicts with the prepare_database arguments; returns one row per configuration
    rows = []
    for c in configs:
        prep = prepare(db, **c)
        rows.append({"use_log": True, "weighting": True, **c, f"precision@{k}": precision_at_k(prep, k)})
    return pd.DataFrame(rows).sort_values(f"precision@{k}", ascending=False).reset_index(drop=True)
