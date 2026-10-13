import sys
import os
import glob
import argparse
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import extract_features, print_features


ALIGNED_DB = "ShapeDatabase_aligned"     # output of Step 3.1 (fully normalized shapes)

STATS_SOURCE_FOLDER = "stats/"
features_filename = STATS_SOURCE_FOLDER + "features.csv"
failed_filename = STATS_SOURCE_FOLDER + "features_failed.csv"


# Step 3.2: compute all descriptors for every shape and save them in a table
# (one row per shape, one column per feature)
def compute_database_features(db, out_filename):
    rows, failed = [], []
    for path in tqdm(sorted(glob.glob(f"{db}/*/*.obj"))):
        try:
            rows.append(extract_features(path))
        except Exception as e:
            # shapes we cannot handle are marked and skipped (see assignment notes)
            failed.append({"file": path, "error": str(e)})

    df = pd.DataFrame(rows)
    os.makedirs(STATS_SOURCE_FOLDER, exist_ok=True)
    df.to_csv(out_filename, index=False)
    pd.DataFrame(failed, columns=["file", "error"]).to_csv(failed_filename, index=False)
    print(f"Computed features for {len(rows)} shapes, failed: {len(failed)}")
    return df


# load a given shape and print its features: read them from the table if they were
# already computed, otherwise extract them now
def show_shape_features(shape, features_file=features_filename):
    path = shape if os.path.exists(shape) else os.path.join(ALIGNED_DB, shape)
    if os.path.exists(features_file):
        df = pd.read_csv(features_file)
        match = df[df["file"].apply(os.path.normpath) == os.path.normpath(path)]
        if len(match):
            print("(from " + features_file + ")")
            print_features(match.iloc[0])
            return
    print_features(extract_features(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", help="e.g. Bird/m32.obj: print the features of one shape only")
    args = parser.parse_args()

    if args.shape:
        show_shape_features(args.shape)
    else:
        compute_database_features(ALIGNED_DB, features_filename)
