# Step 4: querying. Independent of any GUI: plain data in, plain data out.
import numpy as np
import pandas as pd

from .config import FEATURES_CSV, N_BINS
from .features import SCALAR_FEATURES, HIST_FEATURES

FEATURE_COLUMNS = SCALAR_FEATURES + [f"{h}_{i}" for h in HIST_FEATURES for i in range(N_BINS)]
HIST_COLUMNS = {h: [f"{h}_{i}" for i in range(N_BINS)] for h in HIST_FEATURES}


def load_database():
    return pd.read_csv(FEATURES_CSV)


#---------------------------------------------------------------- normalization (tech tips, step 1)
def fit_normalization(db, method="standardize"):
    X = db[SCALAR_FEATURES].to_numpy(float)
    if method == "minmax":
        lo, hi = X.min(0), X.max(0)
        rng = np.where(hi - lo < 1e-12, 1.0, hi - lo)
        return {"method": "minmax", "lo": lo, "rng": rng}
    elif method == "standardize":
        mu, sd = X.mean(0), X.std(0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        return {"method": "standardize", "mu": mu, "sd": sd}
    raise ValueError(f"unknown normalization method: {method}")


def apply_normalization(X, params):
    if params["method"] == "minmax":
        return (X - params["lo"]) / params["rng"]
    return (X - params["mu"]) / params["sd"]


def normalized_feature_table(db, params):
    # for inspection/report only: scalars normalized, histograms left as-is (already normalized)
    X = apply_normalization(db[SCALAR_FEATURES].to_numpy(float), params)
    return pd.concat([db[["class", "file"]].reset_index(drop=True),
                       pd.DataFrame(X, columns=SCALAR_FEATURES),
                       db[[c for h in HIST_FEATURES for c in HIST_COLUMNS[h]]].reset_index(drop=True)], axis=1)


#-------------------------------------------------------------- per-block distances
def euclidean(X, q):
    return np.linalg.norm(X - q, axis=1)


def manhattan(X, q):
    return np.abs(X - q).sum(axis=1)


def emd_1d(X, q):
    """
    histograms are already normalized to sum 1 (equal mass W=U), so EMD reduces to
    the 1D closed form: cumulative-distribution L1 distance (tech tips / lecture note
    that EMD specializes nicely for histograms; this is the standard fast 1D case)
    """
    cx, cq = np.cumsum(X, axis=1), np.cumsum(q)
    return np.abs(cx - cq).sum(axis=1)


HIST_DISTANCE_FUNCTIONS = {"euclidean": euclidean, "manhattan": manhattan, "emd": emd_1d}
SCALAR_DISTANCE_FUNCTIONS = {"euclidean": euclidean, "manhattan": manhattan}


#---------------------------------------------------------------- combined distance (tech tips, step 2)
def _block_distances(db, q_raw, scalar_params, hist_distance):
    blocks = {}

    Xs = apply_normalization(db[SCALAR_FEATURES].to_numpy(float), scalar_params)
    qs = apply_normalization(q_raw[SCALAR_FEATURES].to_numpy(float)[None, :], scalar_params)[0]
    blocks["scalars"] = SCALAR_DISTANCE_FUNCTIONS["euclidean"](Xs, qs)

    hist_fn = HIST_DISTANCE_FUNCTIONS[hist_distance]
    for h in HIST_FEATURES:
        cols = HIST_COLUMNS[h]
        Xh = db[cols].to_numpy(float)
        qh = q_raw[cols].to_numpy(float)
        blocks[h] = hist_fn(Xh, qh)

    return blocks


def query(db, query_features, k=5, scalar_params=None, hist_distance="emd", weights=None):
    if scalar_params is None:
        scalar_params = fit_normalization(db, "standardize")

    q_raw = pd.Series({c: query_features[c] for c in FEATURE_COLUMNS})
    blocks = _block_distances(db, q_raw, scalar_params, hist_distance)

    # standardize each block's distance column across the database (distance weighting),
    # then combine (equal weights by default, override via `weights` dict)
    weights = weights or {}
    total = np.zeros(len(db))
    for name, d in blocks.items():
        sd = d.std()
        d_std = d / sd if sd > 1e-12 else d
        total += weights.get(name, 1.0) * d_std

    out = db[["class", "file"]].assign(distance=total).sort_values("distance")
    out = out[out.file != query_features.get("file")]
    return out.head(k).reset_index(drop=True)