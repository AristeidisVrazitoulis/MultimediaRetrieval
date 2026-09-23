import os
import numpy as np
import open3d as o3d
import pandas as pd
from tqdm import tqdm

SOURCE_DIR = "ShapeDatabase_final"        # the corrected database from Step 2.3/2.4
NORMALIZED_DIR = "ShapeDatabase_normalized"


# weighted avg of all triangle centroids, weighted by each one's area
def barycenter(v, t):
    tri = v[t]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1)
    cents = tri.mean(1)
    return (cents * areas[:,None]).sum(0) / areas.sum()


def normalize(path, out):
    m = o3d.io.read_triangle_mesh(path)
    v, t = np.asarray(m.vertices), np.asarray(m.triangles)
    v = v - barycenter(v, t)               # center
    v = v / (v.max(0) - v.min(0)).max()    # scale to fit a unit cube
    m.vertices = o3d.utility.Vector3dVector(v)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    o3d.io.write_triangle_mesh(out, m)


def normalize_database(df, source_dir, normalized_dir):
    # derive paths from SOURCE_DIR + class + filename (same pattern as
    # save_resampled in remeshing.py) rather than df['file'] directly, so this
    # works whether df is the original shape_statistics.csv or a re-scan of
    # ShapeDatabase_final - either way we always resolve against the current
    # corrected database, not whatever root folder happened to be in df['file']
    rows = []
    for _, r in df.iterrows():
        filename = os.path.basename(r['file'])
        src = os.path.join(source_dir, r['class'], filename)
        dst = os.path.join(normalized_dir, r['class'], filename)
        rows.append({"file": src, "class": r['class'], "out": dst})

    df_all = pd.DataFrame(rows)

    for _, r in tqdm(df_all.iterrows(), total=len(df_all)):
        normalize(r['file'], r['out'])

    return df_all


# re-measures barycenter norm and bbox size, to confirm normalization worked
def check_normalization(df_all, out_filename):
    rows = []
    for _, r in df_all.iterrows():
        if os.path.exists(r['out']):
            m = o3d.io.read_triangle_mesh(r['out'])
            v, t = np.asarray(m.vertices), np.asarray(m.triangles)
            b = barycenter(v, t)
            rows.append(dict(file=r['file'], bary_norm=np.linalg.norm(b),
                              longest_side=(v.max(0) - v.min(0)).max()))

    df_norm_check = pd.DataFrame(rows)
    df_norm_check.to_csv(out_filename, index=False)
    print(df_norm_check.describe())
    return df_norm_check


if __name__ == '__main__':
    df = pd.read_csv("shape_statistics.csv")
    df_all = normalize_database(df, SOURCE_DIR, NORMALIZED_DIR)
    check_normalization(df_all, "normalization_report.csv")
