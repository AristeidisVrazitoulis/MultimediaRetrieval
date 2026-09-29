# use this file to play around with steps or part of the steps. It is like a draft


import pandas as pd

STATS_SOURCE_FOLDER = "stats/"

initial_stats_filename = STATS_SOURCE_FOLDER+"shape_statistics.csv"
df = pd.read_csv(initial_stats_filename)



avg_vertices = df['num_vertices'].mean()
closest_idx = (df['num_vertices'] - avg_vertices).abs().idxmin()
average_shape = df.loc[closest_idx]
print(average_shape)


