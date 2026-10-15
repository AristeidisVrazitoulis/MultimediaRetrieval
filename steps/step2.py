import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesh_exporter import export_media_to_csv
from mesh_vizualizer import visualize_avg_shape
from mesh_stats_plot import plot_stats
from remeshing import remesh_outliers, verify_resampled_files, filter_dataframe
from remeshing_validation import compare_objects
from normalizer import normalize_database, check_normalization
    
USE_FILTER = False

# Resampling: shapes further than TOLERANCE from TARGET_VERTICES are remeshed
TARGET_VERTICES = 6000
TOLERANCE = 0.10    # -> band [5400, 6600]

df_sample_size = 690   # 69 classes x 10 shapes (smallest class has 15)

REMESHED_DB_NAME = "ShapeDatabase_remeshed"
NORMALIZED_DB_NAME = "ShapeDatabase_normalized"
STATS_SOURCE_FOLDER = "stats/"

# 2.1: one row per shape of the ORIGINAL database (class, vertex/face counts, face type, bbox)
original_db_stats_filename = STATS_SOURCE_FOLDER+"original_db_shape_stats.csv"
# 2.4: same columns, but re-scanned from the REMESHED database
remeshed_db_stats_filename = STATS_SOURCE_FOLDER+"remeshed_db_shape_stats.csv"
# 2.3: one row per remeshed shape, old -> new vertex/face counts as logged by pymeshlab
remeshing_log_filename = STATS_SOURCE_FOLDER+"remeshing_log.csv"
# 2.4: the remeshed files re-read with Open3D, to verify the log above
remeshing_verification_filename = STATS_SOURCE_FOLDER+"remeshing_verification.csv"
# 2.5: barycenter distance to origin + longest bbox side per normalized shape (should be ~0 and ~1)
normalization_check_filename = STATS_SOURCE_FOLDER+"normalization_check.csv"

# Step 2.1
# Read the database and export the required data to a csv
# export_media_to_csv("ShapeDatabase", original_db_stats_filename)

# Step 2.2
# Visualize some statistics from previous csv
df = pd.read_csv(original_db_stats_filename)


plot_stats(df) 
# Display average object
visualize_avg_shape(df)


# Step 2.3
# As a result, we have a new db 'ShapeDatabase_remeshed'

# we might not need the whole database at first, so we take an even sample
if USE_FILTER:
    df = filter_dataframe(df, df_sample_size)
remesh_outliers(df, remeshing_log_filename, TARGET_VERTICES, TOLERANCE)


# Step 2.4
# Export again the whole stats from the new db and visualize the differences
export_media_to_csv(REMESHED_DB_NAME, remeshed_db_stats_filename)


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





# low outliers:  1754
# high outliers:  362
# total outliers: 2116