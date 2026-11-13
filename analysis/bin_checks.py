# Step 3.2: is B = 20 bins a good choice?
#
# Two checks, both with the Step 4 custom distance (same normalization, block metrics and sigma
# weighting as cbsr/retrieval.py) and precision@K over the whole database:
#   1. other bin counts: the same B for all histograms, and 20 bins for D1 with more for the others
#   2. D1 from more surface points: D1 holds one value per surface point (10,000), so its sqrt(N)
#      limit is 100 bins; more points raise that limit. Only D1 changes, the rest stays at 20 bins
# For each setting we also measure the SAMPLING NOISE of the histograms: the L1 distance between two
# samplings (two seeds) of the same shape, divided by the mean L1 distance between different shapes.
#
# The histograms are recomputed from the same surface samples as extract_features (same seed and
# order of random draws), so B = 20 gives exactly the histograms of features.csv (checked below).
# Computing them takes about an hour; they are cached in config.BIN_CHECK_CACHE.
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cbsr import config
from cbsr.config import SEED, N_SURFACE_POINTS, N_BINS, DEFAULT_SCALAR_DISTANCE, DEFAULT_HIST_DISTANCE
from cbsr.features import (HIST_FEATURES, load_mesh, sample_surface, shape_property_values, d1_values,
                           normalized_histogram)
from cbsr.retrieval import (scalar_matrix, fit_normalization, apply_normalization, block_metrics,
                            block_distance_std, METRICS)

BINS = [5, 10, 20, 40, 50, 100]                       # check 1: the same B for all histograms
MIXED_OTHERS = [40, 50, 100]                          # check 1: D1 at N_BINS, the others at these
D1_POINTS = [N_SURFACE_POINTS, 40_000, 100_000]       # check 2: surface points for D1
D1_BINS = [20, 40, 80, 150, 300]                      # check 2: bins for D1 (only those below sqrt(points))
NOISE_EVERY = 8                                       # noise measured on every 8th shape (second seed)
K = 10                                                # precision@K, as in Step 6
CONFIGS = {"default": (DEFAULT_SCALAR_DISTANCE, DEFAULT_HIST_DISTANCE), "manhattan": ("manhattan", "manhattan")}


# ---------------------------------------------------------------- recomputing the histograms

def _shape_histograms(path, seed):
    # the same draws as extract_features: sample the surface, then the 5 distributions
    ms = load_mesh(path)              # keep the MeshSet alive while reading the mesh
    v, t = ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix()
    rng = np.random.default_rng(seed)
    points = sample_surface(v, t, N_SURFACE_POINTS, rng)
    raw = shape_property_values(points, points.mean(0), rng)
    out = {f"{h}_{b}": normalized_histogram(raw[h], h, n_bins=b) for h in HIST_FEATURES for b in BINS}
    for m in D1_POINTS:
        if m == N_SURFACE_POINTS:
            d1 = raw["D1"]
        else:
            p = sample_surface(v, t, m, np.random.default_rng(seed))
            d1 = d1_values(p, p.mean(0))
        for b in D1_BINS:
            if b < np.sqrt(m):
                out[f"D1pts_{m}_{b}"] = normalized_histogram(d1, "D1", n_bins=b)
    return out


def _shape_job(i, path):
    noise = _shape_histograms(path, SEED + 1) if i % NOISE_EVERY == 0 else None
    return _shape_histograms(path, SEED), noise


def bin_check_histograms(files, cache=config.BIN_CHECK_CACHE, recompute=False):
    # {key: (n_shapes, B) array} for every setting, plus the second-seed histograms of every
    # NOISE_EVERY-th shape under "noise_<key>" and their row indices under "noise_idx"
    if os.path.exists(cache) and not recompute:
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    jobs = Parallel(n_jobs=-1, verbose=5)(delayed(_shape_job)(i, f) for i, f in enumerate(files))
    out = {k: np.array([main[k] for main, _ in jobs]) for k in jobs[0][0]}
    out["noise_idx"] = np.array([i for i, (_, nz) in enumerate(jobs) if nz is not None])
    for k in jobs[0][0]:
        out[f"noise_{k}"] = np.array([nz[k] for _, nz in jobs if nz is not None])
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, **out)
    return out


# ---------------------------------------------------------------- evaluation

def sampling_noise(h, key):
    # L1 between two samplings of the same shape / mean L1 between different shapes
    idx = h["noise_idx"]
    a, b = h[key][idx], h[f"noise_{key}"]
    same = np.abs(a - b).sum(1).mean()
    between = np.abs(a[:, None] - a[None]).sum(-1)[np.triu_indices(len(a), 1)].mean()
    return same / between


def precision_at_k(db, hists, scalar_distance, hist_distance, k=K):
    # Step 4 custom distance between all pairs of shapes, then the fraction of the k nearest
    # neighbours (query left out) that have the query's class, averaged over all shapes
    params = fit_normalization(db, "standardize")
    vectors = {"scalars": apply_normalization(scalar_matrix(db), params)}
    for name, H in hists.items():
        vectors[name] = np.cumsum(H, axis=1) if hist_distance == "emd" else H
    metrics = block_metrics(scalar_distance, hist_distance)
    sigma = block_distance_std(vectors, metrics, chunk=20)
    n = len(db)
    labels = db["class"].to_numpy()
    hits = []
    for start in range(0, n, 20):
        rows = np.arange(start, min(start + 20, n))
        d = sum(METRICS[metrics[b]](V[rows][:, None, :] - V[None, :, :]) / sigma[b] for b, V in vectors.items())
        d[np.arange(len(rows)), rows] = np.inf
        nn = np.argpartition(d, k, axis=1)[:, :k]
        hits.append(labels[nn] == labels[rows][:, None])
    return float(np.concatenate(hits).mean())


def _row(db, h, d1_key, others_bins):
    hists = {name: h[d1_key if name == "D1" else f"{name}_{others_bins}"] for name in HIST_FEATURES}
    return {f"precision@{K} {c}": precision_at_k(db, hists, *CONFIGS[c]) for c in CONFIGS}


def bin_check(db, h):
    # check 1: other bin counts
    rows = []
    for b in BINS:
        r = {"B D1": b, "B others": b, "D1 values per bin": N_SURFACE_POINTS // b}
        r.update({f"noise {name}": sampling_noise(h, f"{name}_{b}") for name in HIST_FEATURES})
        r.update(_row(db, h, f"D1_{b}", b))
        rows.append(r)
    for b in MIXED_OTHERS:
        r = {"B D1": N_BINS, "B others": b, "D1 values per bin": N_SURFACE_POINTS // N_BINS}
        r.update({f"noise {name}": sampling_noise(h, f"{name}_{N_BINS if name == 'D1' else b}")
                  for name in HIST_FEATURES})
        r.update(_row(db, h, f"D1_{N_BINS}", b))
        rows.append(r)
    return pd.DataFrame(rows)


def d1_points_check(db, h):
    # check 2: D1 from more surface points (and so more bins allowed), the others at N_BINS
    rows = []
    for m in D1_POINTS:
        for b in D1_BINS:
            key = f"D1pts_{m}_{b}"
            if key not in h:
                continue
            r = {"D1 points": m, "sqrt(points)": int(np.sqrt(m)), "B D1": b, "D1 values per bin": m // b,
                 "noise D1": sampling_noise(h, key)}
            r.update(_row(db, h, key, N_BINS))
            rows.append(r)
    return pd.DataFrame(rows)


def run_bin_checks(features_csv=config.FEATURES_CSV, recompute=False):
    db = pd.read_csv(features_csv)
    files = db["file"].str.replace("\\", "/").tolist()
    h = bin_check_histograms(files, recompute=recompute)
    stored = db[[f"{name}_{i}" for name in HIST_FEATURES for i in range(N_BINS)]].to_numpy()
    recomputed = np.hstack([h[f"{name}_{N_BINS}"] for name in HIST_FEATURES])
    print(f"B = {N_BINS} recomputed vs {features_csv}: largest difference {np.abs(recomputed - stored).max():.1e}")

    pd.set_option("display.width", 250)
    bins = bin_check(db, h)
    bins.to_csv(config.BIN_CHECK_CSV, index=False)
    print("\nOther bin counts (noise = same shape, two samplings / different shapes):")
    print(bins.round(4).to_string(index=False))
    d1 = d1_points_check(db, h)
    d1.to_csv(config.D1_POINTS_CHECK_CSV, index=False)
    print("\nD1 from more surface points (other histograms at", N_BINS, "bins):")
    print(d1.round(4).to_string(index=False))
    return bins, d1
