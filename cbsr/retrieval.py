# Step 4: querying. Independent of any GUI: plain data in, plain data out.
#
# A shape's feature vector consists of BLOCKS: the scalar descriptors and one histogram per
# distribution descriptor (A3, D1-D4). The distance between two shapes is computed per block and
# the block distances are summed after DISTANCE WEIGHTING (tech tips step 4): every block distance
# is divided by its standard deviation over random pairs of database shapes, so no block dominates
# only because its distances are numerically larger.
#
# Everything that depends only on the database (normalization parameters, normalized block vectors,
# per-block std) is computed ONCE by prepare_database(); steps/step4.py stores the parameters and
# the app (and later Steps 5-6) load them, so all use exactly the same numbers.
import hashlib
import json
import os
import numpy as np
import pandas as pd

from .config import (FEATURES_CSV, N_BINS, SEED, LOG_FEATURES, N_DISTANCE_PAIRS, PREPARED_PARAMS_JSON,
                     DEFAULT_NORMALIZATION, DEFAULT_SCALAR_DISTANCE, DEFAULT_HIST_DISTANCE)
from .features import SCALAR_FEATURES, HIST_FEATURES

FEATURE_COLUMNS = SCALAR_FEATURES + [f"{h}_{i}" for h in HIST_FEATURES for i in range(N_BINS)]
HIST_COLUMNS = {h: [f"{h}_{i}" for i in range(N_BINS)] for h in HIST_FEATURES}
BLOCKS = ["scalars"] + HIST_FEATURES


def load_database():
    return pd.read_csv(FEATURES_CSV)


def features_fingerprint(path=FEATURES_CSV):
    # identifies the feature table the stored parameters were fitted on. Line endings are
    # normalized first: Git on Windows checks the CSV out with CRLF, which would change the hash
    with open(path, "rb") as f:
        return hashlib.sha1(f.read().replace(b"\r\n", b"\n")).hexdigest()


# ---------------------------------------------------------------- feature normalization (tech tips, (1))

def scalar_matrix(df, log_features=LOG_FEATURES):
    # the scalar features as they go into the distance: log10 for the heavily skewed ones.
    # Used everywhere the scalars are read (fitting, database, query), so all get the same transform
    s = df[SCALAR_FEATURES].astype(float).copy()
    for c in log_features:
        s[c] = np.log10(s[c])
    return s.to_numpy()


def fit_normalization(db, method=DEFAULT_NORMALIZATION, log_features=LOG_FEATURES):
    X = scalar_matrix(db, log_features)
    if method == "minmax":
        lo, hi = X.min(0), X.max(0)
        rng = np.where(hi - lo < 1e-12, 1.0, hi - lo)
        return {"method": "minmax", "log_features": list(log_features), "lo": lo, "rng": rng}
    elif method == "standardize":
        mu, sd = X.mean(0), X.std(0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        return {"method": "standardize", "log_features": list(log_features), "mu": mu, "sd": sd}
    raise ValueError(f"unknown normalization method: {method}")


def apply_normalization(X, params):
    if params["method"] == "minmax":
        return (X - params["lo"]) / params["rng"]
    return (X - params["mu"]) / params["sd"]


def normalized_feature_table(db, params):
    # the database as it goes into the distance: scalars log10-transformed where applied and
    # normalized, histograms as stored (already normalized to sum 1)
    X = apply_normalization(scalar_matrix(db, params["log_features"]), params)
    names = [f"log10_{c}" if c in params["log_features"] else c for c in SCALAR_FEATURES]
    return pd.concat([db[["class", "file"]].reset_index(drop=True),
                      pd.DataFrame(X, columns=names),
                      db[[c for h in HIST_FEATURES for c in HIST_COLUMNS[h]]].reset_index(drop=True)], axis=1)


# ---------------------------------------------------------------- block vectors and distances

def block_vectors(features, params, hist_distance):
    # features: DataFrame (many shapes) or dict/Series (one shape) -> {block: 2D array, one row per shape}
    if isinstance(features, dict):
        features = pd.Series(features)
    df = features.to_frame().T if isinstance(features, pd.Series) else features
    out = {"scalars": apply_normalization(scalar_matrix(df, params["log_features"]), params)}
    for h in HIST_FEATURES:
        H = df[HIST_COLUMNS[h]].to_numpy(float)
        # 1D EMD between histograms with the same bins and total mass = L1 distance between the
        # cumulative histograms (times the bin width: a constant per block, cancelled by the weighting)
        out[h] = np.cumsum(H, axis=1) if hist_distance == "emd" else H
    return out


METRICS = {
    "euclidean": lambda D: np.linalg.norm(D, axis=-1),
    "manhattan": lambda D: np.abs(D).sum(axis=-1),
}


def block_metrics(scalar_distance, hist_distance):
    # EMD is computed as an L1 (manhattan) distance on the cumulative histograms
    hist_metric = "manhattan" if hist_distance == "emd" else hist_distance
    return {b: (scalar_distance if b == "scalars" else hist_metric) for b in BLOCKS}


def config_key(norm_method, scalar_distance, hist_distance, use_log=True, weighting=True):
    return f"{norm_method}|{scalar_distance}|{hist_distance}|log={use_log}|weighting={weighting}"


# ---------------------------------------------------------------- preparing the database (once)

def block_distance_std(vectors, metrics, n_pairs=N_DISTANCE_PAIRS, seed=SEED):
    # std of every block's distance over random pairs of (different) database shapes
    rng = np.random.default_rng(seed)
    n = len(vectors["scalars"])
    a, b = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
    keep = a != b
    a, b = a[keep], b[keep]
    sigma = {}
    for blk, V in vectors.items():
        s = float(METRICS[metrics[blk]](V[a] - V[b]).std())
        sigma[blk] = s if s > 1e-12 else 1.0
    return sigma


def prepare_database(db, norm_method=DEFAULT_NORMALIZATION, scalar_distance=DEFAULT_SCALAR_DISTANCE,
                     hist_distance=DEFAULT_HIST_DISTANCE, use_log=True, weighting=True,
                     n_pairs=N_DISTANCE_PAIRS, params=None, sigma=None):
    # params / sigma: pass stored values (load_prepared) to reuse them instead of fitting again
    if params is None:
        params = fit_normalization(db, norm_method, LOG_FEATURES if use_log else [])
    vectors = block_vectors(db, params, hist_distance)
    metrics = block_metrics(scalar_distance, hist_distance)
    if sigma is None:
        sigma = block_distance_std(vectors, metrics, n_pairs) if weighting else {b: 1.0 for b in BLOCKS}
    return {
        "config": {"norm_method": norm_method, "scalar_distance": scalar_distance,
                   "hist_distance": hist_distance, "use_log": use_log, "weighting": weighting},
        "params": params, "sigma": sigma, "metrics": metrics, "vectors": vectors,
        "shapes": db[["class", "file"]].reset_index(drop=True),
    }


# ---------------------------------------------------------------- querying

def distances(prep, query_features, weights=None):
    # combined distance from one query shape to every database shape
    q = block_vectors(query_features, prep["params"], prep["config"]["hist_distance"])
    weights = weights or {}
    total = np.zeros(len(prep["shapes"]))
    for blk, V in prep["vectors"].items():
        d = METRICS[prep["metrics"][blk]](V - q[blk])
        total += weights.get(blk, 1.0) * d / prep["sigma"][blk]
    return total


def query(prep, query_features, k=5, weights=None):
    # the k database shapes closest to the query (the query itself left out), closest first
    out = prep["shapes"].assign(distance=distances(prep, query_features, weights))
    out = out[out.file != query_features.get("file")]
    return out.sort_values("distance").head(k).reset_index(drop=True)


# ---------------------------------------------------------------- storing / loading the prepared parameters

def _to_json(prep):
    p = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in prep["params"].items()}
    return {"config": prep["config"], "params": p, "sigma": prep["sigma"]}


def save_prepared(preps, path=PREPARED_PARAMS_JSON, features_path=FEATURES_CSV):
    data = {"features_fingerprint": features_fingerprint(features_path), "configs": {}}
    for prep in preps:
        data["configs"][config_key(**prep["config"])] = _to_json(prep)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def load_prepared(db, norm_method=DEFAULT_NORMALIZATION, scalar_distance=DEFAULT_SCALAR_DISTANCE,
                  hist_distance=DEFAULT_HIST_DISTANCE, use_log=True, weighting=True,
                  path=PREPARED_PARAMS_JSON, features_path=FEATURES_CSV):
    # the stored parameters if they exist for this configuration AND were fitted on the current
    # feature table; otherwise prepare from scratch. Returns (prep, loaded_from_file)
    config = dict(norm_method=norm_method, scalar_distance=scalar_distance, hist_distance=hist_distance,
                  use_log=use_log, weighting=weighting)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        stored = data["configs"].get(config_key(**config))
        if stored and data["features_fingerprint"] == features_fingerprint(features_path):
            params = {k: (np.asarray(v) if isinstance(v, list) and k != "log_features" else v)
                      for k, v in stored["params"].items()}
            return prepare_database(db, **config, params=params, sigma=stored["sigma"]), True
    return prepare_database(db, **config), False
