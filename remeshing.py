import pandas as pd
import pymeshlab as pml
import open3d as o3d
import numpy as np
import math
import os
import shutil


# midpoint of the suggested 5K-10K range
TARGET_VERTICES = 6000          
OUTPUT_DIR = "ShapeDatabase_final"

LOW_THRESHOLD = 100
HIGH_THRESHOLD = 100000

resample_log = []


def clean_mesh(ms):
    # fixes non-manifold geometry (duplicate/degenerate faces, non-manifold
    # edges/vertices) that would otherwise crash subdivision/decimation filters.
    # affects ~31% of our low-outlier shapes (29/94), so this is required, not optional.
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()



def refine(path, target_v=TARGET_VERTICES):
    ms = pml.MeshSet()
    ms.load_new_mesh(path)
    clean_mesh(ms)

    for _ in range(10):  #keep subdividing until we pass the target or not
        ms.meshing_surface_subdivision_midpoint(iterations=1)
        if ms.current_mesh().vertex_number() >= target_v:
            break
    return ms

#meshing_decimation_quadric_edge_collapse(targetfacenum=...) (repeatedly merges
#the edge whose removal changes the shape least until it hits the target
def simplify(path, target_v=TARGET_VERTICES):
    ms = pml.MeshSet()
    ms.load_new_mesh(path)
    clean_mesh(ms)
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_v * 2)
    return ms

def save_resampled(ms, row):
    class_dir = os.path.join(OUTPUT_DIR, row['class'])
    os.makedirs(class_dir, exist_ok=True)
    out_path = os.path.join(class_dir, os.path.basename(row['file']))
    ms.save_current_mesh(out_path)
    print(f"{row['file']}: {row['num_vertices']} -> {ms.current_mesh().vertex_number()} vertices")
    
    new_vertices = ms.current_mesh().vertex_number()
    resample_log.append({
        "file": row['file'], "class": row['class'],
        "old_vertices": row['num_vertices'], "new_vertices": new_vertices,
        "old_faces": row['num_faces'], "new_faces": ms.current_mesh().face_number(),
    })
    return out_path



def quick_stats(path):
    # independent re-check using Open3D (a different library than pymeshlab,
    # which performed the actual resampling) - catches any pymeshlab-specific
    # save/load artifacts our own in-memory logging in save_resampled() would miss
    m = o3d.io.read_triangle_mesh(path)
    v, t = np.asarray(m.vertices), np.asarray(m.triangles)
    return len(v), len(t)


def verify_resampled_files(resample_log_filename, verification_filename):
    df_log = pd.read_csv(resample_log_filename)

    rows = []
    for _, r in df_log.iterrows():
        out = os.path.join(OUTPUT_DIR, r['class'], os.path.basename(r['file']))
        if os.path.exists(out):
            nv, nf = quick_stats(out)
            rows.append(dict(file=r['file'], before_v=r['old_vertices'], after_v=nv,
                              before_f=r['old_faces'], after_f=nf))
        else:
            print(f"WARNING: expected resampled file not found: {out}")

    df_check = pd.DataFrame(rows)
    df_check.to_csv(verification_filename, index=False)

    print(df_check.describe())
    print()
    print("still > 15000 verts:", (df_check.after_v > 15000).sum())
    print("still > 100000 verts:", (df_check.after_v > 100000).sum())

    return df_check


def remesh_outliers(df, out_filename):
      # one-time full copy of the original database; save_resampled() then
    # overwrites just the outlier files in place, so ShapeDatabase_final ends
    # up as a single complete, corrected working database (no separate merge step)
    if not os.path.exists(OUTPUT_DIR):
        shutil.copytree("ShapeDatabase", OUTPUT_DIR)

    low_outliers = df[(df['num_faces'] < LOW_THRESHOLD) | (df['num_vertices'] < LOW_THRESHOLD)]
    high_outliers = df[(df['num_faces'] > HIGH_THRESHOLD) | (df['num_vertices'] > HIGH_THRESHOLD)]
    
    for _, row in low_outliers.iterrows():
        ms = refine(row['file'])
        save_resampled(ms, row)

    for _, row in high_outliers.iterrows():
        ms = simplify(row['file'])
        save_resampled(ms, row)
    
    print(f"Total outliers: {len(low_outliers)+ len(high_outliers)}")
    pd.DataFrame(resample_log).to_csv(out_filename, index=False)


# find outliers
if __name__ == '__main__':

    verify_resampled_files("resampling_report.csv", "stats_resampled_verification.csv")


    



