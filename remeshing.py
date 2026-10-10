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
    was_nonmanifold = clean_mesh(ms)

    threshold_factor = 0.02
    for _ in range(10):
        nv = ms.current_mesh().vertex_number()
        if nv >= target_v:
            break
        bbox_diag = ms.current_mesh().bounding_box().diagonal()
        ms.meshing_surface_subdivision_midpoint(threshold=pml.PureValue(bbox_diag * threshold_factor))
        threshold_factor *= 1.3   #raise threshold each pass so fewer edges qualify next round

    return ms, was_nonmanifold

#meshing_decimation_quadric_edge_collapse(targetfacenum=...) (repeatedly merges
#the edge whose removal changes the shape least until it hits the target
def simplify(path, target_v=TARGET_VERTICES):
    ms = pml.MeshSet()
    ms.load_new_mesh(path)
    was_nonmanifold = clean_mesh(ms)
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_v * 2)
    return ms, was_nonmanifold

def save_resampled(ms, row, was_nonmanifold):
    class_dir = os.path.join(OUTPUT_DIR, row['class'])
    os.makedirs(class_dir, exist_ok=True)
    out_path = os.path.join(class_dir, os.path.basename(row['file']))
    ms.save_current_mesh(out_path)
    print(f"{row['file']}: {row['num_vertices']} -> {ms.current_mesh().vertex_number()} vertices")
    
    new_vertices = ms.current_mesh().vertex_number()
    resample_log.append({
        "file": row['file'], "class": row['class'],
        "old_vertices": row['num_vertices'], "new_vertices": new_vertices,
        "old_faces": row['num_faces'], "new_faces": ms.current_mesh().face_number(), "was_nonmanifold": was_nonmanifold,
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

####fix: resamle every shape unless it is already clse to the target (+-10%)
def needs_resample(nv, target=TARGET_VERTICES, tol=0.10):
    return abs(nv - target) > target * tol

def remesh_outliers(df, out_filename, target_v=TARGET_VERTICES, tol=0.10):
      # one-time full copy of the original database; save_resampled() then
    # overwrites just the outlier files in place, so ShapeDatabase_final ends
    # up as a single complete, corrected working database (no separate merge step)
    if not os.path.exists(OUTPUT_DIR):
        shutil.copytree("ShapeDatabase", OUTPUT_DIR)

    #low_outliers = df[(df['num_faces'] < low_threshold) | (df['num_vertices'] < low_threshold)]
    #high_outliers = df[(df['num_faces'] > high_threshold) | (df['num_vertices'] > high_threshold)]
    
    to_refine = df[df['num_vertices'] < target_v * (1 - tol)]
    to_simplify = df[df['num_vertices'] > target_v * (1 + tol)]
    close_enough = df[(df['num_vertices'] >= target_v * (1 - tol)) & (df['num_vertices'] <= target_v * (1 + tol))]
    
    for _, row in to_refine.iterrows():
        ms, was_nonmanifold = refine(row['file'], target_v)
        save_resampled(ms, row, was_nonmanifold)

    for _, row in to_simplify.iterrows():
        ms, was_nonmanifold = simplify(row['file'], target_v)
        save_resampled(ms, row, was_nonmanifold)

    for _, row in close_enough.iterrows():
        class_dir = os.path.join(OUTPUT_DIR, row['class'])
        os.makedirs(class_dir, exist_ok=True)
        out_path = os.path.join(class_dir, os.path.basename(row['file']))
        shutil.copy(row['file'], out_path)
        resample_log.append({"file": row['file'], "class": row['class'], "old_vertices": row['num_vertices'], "new_vertices": row['num_vertices'], "old_faces": row['num_faces'], "new_faces": row['num_faces'], "was_nonmanifold": None,})
    
    #print(f"Total outliers: {len(low_outliers)+ len(high_outliers)}")
    #pd.DataFrame(resample_log).to_csv(out_filename, index=False)
    print(f"Refined: {len(to_refine)}, Simplified: {len(to_simplify)}, Already close: {len(close_enough)}")
    pd.DataFrame(resample_log).to_csv(out_filename, index=False)

def filter_dataframe(df, sample_size=200):
    ret_df = (df.groupby('class', group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max(1, sample_size // df['class'].nunique())), random_state=42))
            .reset_index(drop=True))
    return ret_df
    

# find outliers
if __name__ == '__main__':

    verify_resampled_files("resampling_report.csv", "stats_resampled_verification.csv")


    



