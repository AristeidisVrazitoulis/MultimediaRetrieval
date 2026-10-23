# Step 4: querying. Independent of any GUI: plain data in, plain data out.
import numpy as np
import pandas as pd

from .config import FEATURES_CSV, N_BINS
from .features import SCALAR_FEATURES, HIST_FEATURES

# the descriptor columns of the feature table (the diagnostics columns are left out)
FEATURE_COLUMNS = SCALAR_FEATURES + [f"{h}_{i}" for h in HIST_FEATURES for i in range(N_BINS)]


def load_database():
    return pd.read_csv(FEATURES_CSV)


def query(db, query_features, k=5):
    # PLACEHOLDER: z-score all columns + Euclidean distance. To be replaced in Step 4
    # (log transform, EMD for the histograms, distance weighting - see docs/tech_tips_step4.md)
    # X = db[FEATURE_COLUMNS].to_numpy(float)
    # mu, sd = X.mean(0), X.std(0) + 1e-12
    # q = np.array([query_features[c] for c in FEATURE_COLUMNS], float)
    # d = np.linalg.norm((X - mu) / sd - (q - mu) / sd, axis=1)
    # out = db[["class", "file"]].assign(distance=d).sort_values("distance")
    # return out[out.file != query_features.get("file")].head(k)   # leave out the query itself
    pass