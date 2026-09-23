import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesh_export import open_mesh, visualize_mesh, get_mesh_data, export_media_to_csv
from mesh_stats_plot import plot_stats
from remeshing import remesh_outliers, verify_resampled_files
from remeshing_validation import compare_objects
from normalizer import normalize_database, check_normalization




REMESHED_DB_NAME = "ShapeDatabase_final"
NORMALIZED_DB_NAME = "ShapeDatabase_normalized"
STATS_SOURCE_FOLDER = "stats/"

initial_stats_filename = STATS_SOURCE_FOLDER+"shape_statistics.csv"
updated_stats_filename = STATS_SOURCE_FOLDER+"updated_shape_statistics.csv"
resample_report_filename = STATS_SOURCE_FOLDER+"resampling_report.csv"
normalization_report_filename = STATS_SOURCE_FOLDER+"normalization_report.csv"
resample_stats_verification_filename = STATS_SOURCE_FOLDER+"stats_resampled_verification.csv"

# # Step 2.1
# # Read the database and export the required data to a csv
export_media_to_csv("ShapeDatabase", initial_stats_filename)

# # Step 2.2
# # Visualize some statistics from previous csv
plot_stats(initial_stats_filename)

# Step 2.3
# Spot the outliers (<100 and >100.000 vertices) and refine or simplify accordingly
# As a result, we have a new db 'ShapeDatabase_final'

df = pd.read_csv(initial_stats_filename)
remesh_outliers(df, resample_report_filename)


# Step 2.4
# Export again the whole stats from the new db and visualize the differences
export_media_to_csv(REMESHED_DB_NAME, updated_stats_filename)
plot_stats(updated_stats_filename)
# We pick an extreme example of a door
object_filename = "Door/m1708.obj"
compare_objects(object_filename)
verify_resampled_files(resample_report_filename, resample_stats_verification_filename)


# Step 2.5
df_all = normalize_database(df, REMESHED_DB_NAME, NORMALIZED_DB_NAME)
check_normalization(df_all, normalization_report_filename)







