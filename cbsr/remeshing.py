import pandas as pd
import pymeshlab as pml
import open3d as o3d
import numpy as np
import math
import os
import shutil

from .config import TARGET_VERTICES, TOLERANCE, ORIGINAL_DB, REMESHED_DB, REMESHING_LOG_CSV, REMESHING_VERIFICATION_CSV
from .mesh_utils import edges

OUTPUT_DIR = REMESHED_DB



resample_log = []


def clean_mesh(ms):
    # fixes non-manifold geometry (duplicate/degenerate faces, non-manifold
    # edges/vertices) that would otherwise crash subdivision/decimation filters.
    was_nonmanifold = not ms.get_topological_measures()['is_mesh_two_manifold']
    # unreferenced vertices would inflate vertex_number() and stop refinement too early
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    # repairing edges can create new non-manifold vertices (and vice versa): repeat until manifold
    for _ in range(5):
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()
        if ms.get_topological_measures()['is_mesh_two_manifold']:
            break
    # the repair filters leave faces selected, and decimation then silently does nothing
    ms.set_selection_none()
    return was_nonmanifold


def decimate_to(ms, target_v, tol):
    # decimation targets a FACE count but we need a VERTEX count (F ~ 2V fails for
    # many-part shapes, and some shapes collapse in big jumps), so bisect on the face
    # target, always starting from the original mesh, until we land in the band
    min_v, max_v = target_v * (1 - tol), target_v * (1 + tol)
    m = ms.current_mesh()
    v, f = m.vertex_number(), m.face_number()
    if v <= max_v:
        return
    original = pml.Mesh(m.vertex_matrix(), m.face_matrix())
    lo, hi = 1, f
    face_goal = int(f * target_v / v)
    best, best_err = original, abs(v - target_v)
    for _ in range(12):
        ms.add_mesh(pml.Mesh(original.vertex_matrix(), original.face_matrix()))
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=face_goal)
        n = ms.current_mesh().vertex_number()
        if abs(n - target_v) < best_err:
            r = ms.current_mesh()
            best, best_err = pml.Mesh(r.vertex_matrix(), r.face_matrix()), abs(n - target_v)
        if min_v <= n <= max_v:
            return
        if n < min_v:
            lo, face_goal = face_goal, (face_goal + hi) // 2
        else:
            hi, face_goal = face_goal, (lo + face_goal) // 2
        if hi - lo <= 1:
            break
    ms.add_mesh(best)   # band not reachable (rare): keep the closest attempt


def edge_lengths(m):
    v, f = m.vertex_matrix(), m.face_matrix()
    e = edges(f)
    return np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)



def refine(path, target_v=TARGET_VERTICES, tol=TOLERANCE):
    ms = pml.MeshSet()
    ms.load_new_mesh(path)
    was_nonmanifold = clean_mesh(ms)

    # area-adaptive: each split edge adds one vertex, so per pass split only the
    # (target_v - nv) LONGEST edges -> big triangles first, stop right at the target
    for _ in range(50):
        nv = ms.current_mesh().vertex_number()
        if nv >= target_v:
            break
        lengths = np.sort(edge_lengths(ms.current_mesh()))[::-1]
        # we cannot define which edges to split so we set a threshold
        # the threshold cannot be larger than the number of available edges
        threshold = lengths[min(target_v - nv, len(lengths)) - 1] * 0.999
        ms.meshing_surface_subdivision_midpoint(iterations=1, threshold=pml.PureValue(threshold))
        if ms.current_mesh().vertex_number() == nv:   # nothing split, give up
            break

    decimate_to(ms, target_v, tol)   # only acts if we overshot the band
    return ms, was_nonmanifold

#meshing_decimation_quadric_edge_collapse(targetfacenum=...) (repeatedly merges
#the edge whose removal changes the shape least until it hits the target
def simplify(path, target_v=TARGET_VERTICES, tol=TOLERANCE):
    ms = pml.MeshSet()
    ms.load_new_mesh(path)
    was_nonmanifold = clean_mesh(ms)
    decimate_to(ms, target_v, tol)
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


def verify_resampled_files(resample_log_filename, verification_filename, target_v=TARGET_VERTICES, tol=TOLERANCE):
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
    print(f"still > {target_v * (1 + tol):.0f} verts:", (df_check.after_v > target_v * (1 + tol)).sum())
    print(f"still < {target_v * (1 - tol):.0f} verts:", (df_check.after_v < target_v * (1 - tol)).sum())

    return df_check

####fix: resamle every shape unless it is already clse to the target (+-10%)
def needs_resample(nv, target=TARGET_VERTICES, tol=TOLERANCE):
    return abs(nv - target) > target * tol

def remesh_outliers(df, out_filename, target_v=TARGET_VERTICES, tol=TOLERANCE):
      # one-time full copy of the original database; save_resampled() then
    # overwrites just the outlier files in place, so OUTPUT_DIR ends
    # up as a single complete, corrected working database (no separate merge step)
    if not os.path.exists(OUTPUT_DIR):
        shutil.copytree(ORIGINAL_DB, OUTPUT_DIR)
    resample_log.clear()

    #low_outliers = df[(df['num_faces'] < low_threshold) | (df['num_vertices'] < low_threshold)]
    #high_outliers = df[(df['num_faces'] > high_threshold) | (df['num_vertices'] > high_threshold)]
    
    to_refine = df[df['num_vertices'] < target_v * (1 - tol)]
    to_simplify = df[df['num_vertices'] > target_v * (1 + tol)]
    close_enough = df[(df['num_vertices'] >= target_v * (1 - tol)) & (df['num_vertices'] <= target_v * (1 + tol))]
    
    for _, row in to_refine.iterrows():
        ms, was_nonmanifold = refine(row['file'], target_v, tol)
        save_resampled(ms, row, was_nonmanifold)

    for _, row in to_simplify.iterrows():
        ms, was_nonmanifold = simplify(row['file'], target_v, tol)
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
            .apply(lambda g: g.sample(min(len(g), math.ceil(sample_size / df['class'].nunique())), random_state=42))
            .reset_index(drop=True))
    return ret_df
    

# find outliers
if __name__ == '__main__':

    verify_resampled_files(REMESHING_LOG_CSV, REMESHING_VERIFICATION_CSV)


    