import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbsr.shape_stats import export_media_to_csv
from cbsr.viewer import visualize_avg_shape
from cbsr.remeshing import remesh_outliers, verify_resampled_files, filter_dataframe
from cbsr.normalization import normalize_database, check_normalization
from cbsr import config
from analysis.stats_plots import plot_stats
from analysis.remeshing_checks import compare_objects
    
USE_FILTER = False

# Resampling: shapes further than TOLERANCE from TARGET_VERTICES are remeshed (set in cbsr/config.py)
TARGET_VERTICES = config.TARGET_VERTICES
TOLERANCE = config.TOLERANCE    # -> band [5400, 6600]

df_sample_size = 690   # 69 classes x 10 shapes (smallest class has 15)

REMESHED_DB_NAME = config.REMESHED_DB
NORMALIZED_DB_NAME = config.NORMALIZED_DB

# output files: what each one contains is described in cbsr/config.py
original_db_stats_filename = config.ORIGINAL_STATS_CSV
remeshed_db_stats_filename = config.REMESHED_STATS_CSV
remeshing_log_filename = config.REMESHING_LOG_CSV
remeshing_verification_filename = config.REMESHING_VERIFICATION_CSV
normalization_check_filename = config.NORMALIZATION_CHECK_CSV

# Step 2.1
# Read the database and export the required data to a csv
# export_media_to_csv(config.ORIGINAL_DB, original_db_stats_filename)

# Step 2.2
# Visualize some statistics from previous csv
df = pd.read_csv(original_db_stats_filename)


# plot_stats(df) 
# # Display average object
# visualize_avg_shape(df)


# Step 2.3
# As a result, we have a new db in data/remeshed

# we might not need the whole database at first, so we take an even sample
# if USE_FILTER:
#     df = filter_dataframe(df, df_sample_size)
# remesh_outliers(df, remeshing_log_filename, TARGET_VERTICES, TOLERANCE)


# Step 2.4
# Export again the whole stats from the new db and visualize the differences
# export_media_to_csv(REMESHED_DB_NAME, remeshed_db_stats_filename)


df_updated_stats = pd.read_csv(remeshed_db_stats_filename)
plot_stats(df_updated_stats)

# We pick an extreme example of a door
# ShapeDatabase/Door/m1708.obj: 16 -> 21828 vertices
low_outlier = "Door/m1708.obj"
# ShapeDatabase/Biplane/m1120.obj: 65722 -> 6653 vertices
high_outlier = "Chess/m1595.obj"
compare_objects(low_outlier)
compare_objects(high_outlier)

verify_resampled_files(remeshing_log_filename, remeshing_verification_filename, TARGET_VERTICES, TOLERANCE)


# Step 2.5
# normalize
df_all = normalize_database(df, REMESHED_DB_NAME, NORMALIZED_DB_NAME)
check_normalization(df_all, normalization_check_filename)


