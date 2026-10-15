import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import compute_database_features, show_shape_features, calibrate_hist_ranges, HIST_FEATURES
from feature_checks import (plot_histograms_per_class, compare_three_shapes, scalar_extremes,
                            within_vs_between, volume_reliability)
from mesh_vizualizer import visualize_shape


CALIBRATE_RANGES = False  # set True once to print the histogram ranges to copy into HIST_RANGES
                          # (then set it back to False: features are computed with the CURRENT HIST_RANGES)
COMPUTE_FEATURES = True   # set False to reuse stats/features.csv (computed anyway if missing)
SHOW_EXTREMES = False     # set True to open the lowest/highest shape of each descriptor in the viewer

ALIGNED_DB = "ShapeDatabase_aligned"     # output of Step 3.1 (fully normalized shapes)
EXAMPLE_SHAPE = "Bird/m32.obj"

# two shapes of the same class + one of a different class 
SAME_CLASS_A = "Car/m1513.obj"
SAME_CLASS_B = "Car/m1523.obj"
OTHER_CLASS = "Bicycle/D00904.obj"

STATS_SOURCE_FOLDER = "stats/"
features_filename = STATS_SOURCE_FOLDER + "features.csv"
extremes_filename = STATS_SOURCE_FOLDER + "feature_extremes.csv"
separation_filename = STATS_SOURCE_FOLDER + "feature_class_separation.csv"
unreliable_volume_filename = STATS_SOURCE_FOLDER + "feature_unreliable_volume.csv"


# Step 3.2
# Choose the histogram ranges: percentile of the values over the whole database.
# Copy the printed values into HIST_RANGES in features.py before computing the features
if CALIBRATE_RANGES:
    print(calibrate_hist_ranges(ALIGNED_DB))
    
# Compute all descriptors for every shape and save them in a table
# (one row per shape, one column per feature)
if COMPUTE_FEATURES or not os.path.exists(features_filename):
    compute_database_features(ALIGNED_DB, features_filename)

# Load a given shape and print its features
# (retrieved from the table if already computed, extracted otherwise)
show_shape_features(os.path.join(ALIGNED_DB, EXAMPLE_SHAPE), features_filename)


# Checking the features 
df_features = pd.read_csv(features_filename)

# Histograms per class: curves should be grouped within each plot
# and look different across plots. Saved in stats/figures_features/
for name in HIST_FEATURES:
    plot_histograms_per_class(df_features, name)

# Same-class shapes should have similar histograms, different from the third one 
compare_three_shapes(df_features, SAME_CLASS_A, SAME_CLASS_B, OTHER_CLASS)

# Average histogram distance within / between classes: ratio < 1 means the descriptor
# separates the classes
df_separation = within_vs_between(df_features)
df_separation.to_csv(separation_filename, index=False)
print(df_separation)

# Lowest and highest shapes for each scalar descriptor: their values should
# match what we see in the viewer (thin vs bulky, round vs box-like, ...)
df_extremes = scalar_extremes(df_features)
df_extremes.to_csv(extremes_filename, index=False)
# print(df_extremes)
if SHOW_EXTREMES:
    for _, r in df_extremes.iterrows():
        print(f"{r['descriptor']} {r['kind']}: {r['value']:.4f}  {r['file']}")
        visualize_shape(r['file'])

# Shapes whose volume (and so compactness, rectangularity, convexity) is not reliable:
# mesh still open after repair, or convexity > 1
volume_reliability(df_features).to_csv(unreliable_volume_filename, index=False)
