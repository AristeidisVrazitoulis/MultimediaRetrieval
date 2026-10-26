# all paths and parameters of the pipeline in one place.
# Paths are relative to the project root: run the scripts from there.
import os
import numpy as np


# ---------------------------------------------------------------- data (git-ignored)
DATA_DIR = "data"
ORIGINAL_DB = os.path.join(DATA_DIR, "ShapeDatabase")     # input database
REMESHED_DB = os.path.join(DATA_DIR, "remeshed")          # Step 2.3
NORMALIZED_DB = os.path.join(DATA_DIR, "normalized")      # Step 2.5
ALIGNED_DB = os.path.join(DATA_DIR, "aligned")            # Step 3.1 (fully normalized)

# ---------------------------------------------------------------- results (CSVs + figures)
RESULTS_DIR = "results"
STEP2_DIR = os.path.join(RESULTS_DIR, "step2")
STEP3_DIR = os.path.join(RESULTS_DIR, "step3")
STEP4_DIR = os.path.join(RESULTS_DIR, "step4")
STEP2_FIGURES = os.path.join(STEP2_DIR, "figures")
STEP3_FIGURES = os.path.join(STEP3_DIR, "figures")

# Step 2
# 2.1: one row per shape of the ORIGINAL database (class, vertex/face counts, face type, bbox)
ORIGINAL_STATS_CSV = os.path.join(STEP2_DIR, "original_db_shape_stats.csv")
# 2.4: same columns, but re-scanned from the REMESHED database
REMESHED_STATS_CSV = os.path.join(STEP2_DIR, "remeshed_db_shape_stats.csv")
# 2.3: one row per shape, old -> new vertex/face counts as logged by pymeshlab
REMESHING_LOG_CSV = os.path.join(STEP2_DIR, "remeshing_log.csv")
# 2.3: the shapes selected for remeshing
REMESHING_OUTLIERS_CSV = os.path.join(STEP2_DIR, "remeshing_outliers.csv")
# 2.4: the remeshed files re-read with Open3D, to verify the log above
REMESHING_VERIFICATION_CSV = os.path.join(STEP2_DIR, "remeshing_verification.csv")
# 2.5: barycenter distance to origin + longest bbox side per normalized shape (should be ~0 and ~1)
NORMALIZATION_CHECK_CSV = os.path.join(STEP2_DIR, "normalization_check.csv")

# Step 3
ALIGNMENT_CHECK_CSV = os.path.join(STEP3_DIR, "alignment_check.csv")
FEATURES_CSV = os.path.join(STEP3_DIR, "features.csv")
FEATURE_EXTREMES_CSV = os.path.join(STEP3_DIR, "feature_extremes.csv")
FEATURE_SEPARATION_CSV = os.path.join(STEP3_DIR, "feature_class_separation.csv")
FEATURE_UNRELIABLE_VOLUME_CSV = os.path.join(STEP3_DIR, "feature_unreliable_volume.csv")

# Step 4
# the normalized feature database (one file per normalization method), as the assignment asks to store it
NORMALIZED_FEATURES_CSV = os.path.join(STEP4_DIR, "normalized_features_{method}.csv")
# fitted normalization parameters + per-block distance std, for every distance configuration;
# written by steps/step4.py, loaded by the app (and later Steps 5-6) so all use the same numbers
PREPARED_PARAMS_JSON = os.path.join(STEP4_DIR, "prepared_params.json")
# precision@K of the distance configurations, to justify the final choice in the report
DISTANCE_COMPARISON_CSV = os.path.join(STEP4_DIR, "distance_comparison.csv")

# ---------------------------------------------------------------- Step 2: remeshing
# user-chosen target inside the assignment's suggested 5K-10K range (NOT the database average)
TARGET_VERTICES = 6000
# shapes further than TOLERANCE from the target are remeshed -> band [5400, 6600]
TOLERANCE = 0.10

# ---------------------------------------------------------------- Step 3: features
N_BINS = 20                  # B
N_SURFACE_POINTS = 10000     # points sampled uniformly on the surface, pool for the histograms
N_SAMPLES = 1_000_000        # random tuples per histogram ( ~100 per dimension for D3 -> 100^3)
SEED = 42                    # fixed seed, so the same shape always gives the same features

# fixed ranges for the histograms: every shape must use the SAME bins, otherwise the
# histograms are not comparable. Shapes are normalized to the unit cube, so the
# ranges are bounded; values outside are clipped into the first/last bin.
# A3 is an angle, so [0, pi]. D1-D4: upper bound = HIST_PERCENTILE-th percentile of the
# values over the whole database, computed with calibrate_hist_ranges() (see step3_2.py)
HIST_PERCENTILE = 99.9
HIST_RANGES = {
    "A3": (0.0, np.pi),
    "D1": (0.0, 0.75),
    "D2": (0.0, 1.09),
    "D3": (0.0, 0.57),
    "D4": (0.0, 0.34),
}

# ---------------------------------------------------------------- Step 4: querying
# compactness and eccentricity span several orders of magnitude (up to ~1e9 and ~1e5): log10 keeps
# their order but stops a few extreme shapes from dominating the mean/std (or min/max)
LOG_FEATURES = ["compactness", "eccentricity"]
NORMALIZATION_METHODS = ["standardize", "minmax"]
SCALAR_DISTANCES = ["euclidean", "manhattan"]
HIST_DISTANCES = ["euclidean", "manhattan", "emd"]
DEFAULT_NORMALIZATION = "standardize"
DEFAULT_SCALAR_DISTANCE = "euclidean"
DEFAULT_HIST_DISTANCE = "emd"
# distance weighting (tech tips step 4): each block's distance is divided by its std over this many
# random pairs of database shapes, computed once and stored
N_DISTANCE_PAIRS = 20_000
