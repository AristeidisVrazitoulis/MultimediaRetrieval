# Step 2.5 (translation + scaling) and Step 3.1 (PCA alignment + flipping + final scaling).
# Order (tech tips step 3): translation -> pose -> flipping -> size
import os
import numpy as np
import open3d as o3d
import pandas as pd
from tqdm import tqdm

from .mesh_utils import barycenter, scale_to_unit_cube


# ---------------------------------------------------------------- on arrays

def translate_and_scale(v, t):
    # Step 2.5: barycenter to the origin, then fit in a unit cube
    v = v - barycenter(v, t)               # center
    return scale_to_unit_cube(v)           # scale to fit a unit cube


def pca_align(v):
    cov = np.cov(v.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    e1, e2 = eigvecs[:, order[0]], eigvecs[:, order[1]]
    # PCA gives directions, not orientations: e3 could make a left-handed frame,
    # so we force a right-handed one with e1 x e2 (tech tips step 3, (2))
    e3 = np.cross(e1, e2)
    v_aligned = v @ np.column_stack([e1, e2, e3])
    return v_aligned


def moment_flip(v, t):
    # moment test over the triangle centers: f_i = sum_t sign(C_t,i) * C_t,i^2
    centers = v[t].mean(1)
    for axis in range(3):
        f = centers[:, axis]
        moment = np.sum(np.sign(f) * (f ** 2))
        if moment < 0:
            v[:, axis] = -v[:, axis]
    return v


def align_flip_scale(v, t):
    # Step 3.1: size is the LAST step (translation, pose, flipping, size): after alignment
    # the AABB equals the OBB, so this gives the tight unit-cube fit
    v = pca_align(v)
    v = moment_flip(v, t)
    return scale_to_unit_cube(v)


# ---------------------------------------------------------------- on files (database)

def normalize(path, out):
    m = o3d.io.read_triangle_mesh(path)
    v, t = np.asarray(m.vertices), np.asarray(m.triangles)
    m.vertices = o3d.utility.Vector3dVector(translate_and_scale(v, t))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    o3d.io.write_triangle_mesh(out, m, write_vertex_normals=False,
                           write_vertex_colors=False, write_triangle_uvs=False)


def align_and_flip(path, out):
    m = o3d.io.read_triangle_mesh(path)
    v = np.asarray(m.vertices)
    t = np.asarray(m.triangles)
    m.vertices = o3d.utility.Vector3dVector(align_flip_scale(v, t))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    o3d.io.write_triangle_mesh(out, m)


def normalize_database(df, source_dir, normalized_dir):
    # derive paths from source_dir + class + filename (same pattern as
    # save_resampled in remeshing.py) rather than df['file'] directly, so this
    # works whether df is original_db_shape_stats.csv or a re-scan of the remeshed
    # database - either way we always resolve against the current
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


def check_alignment(path):
    m = o3d.io.read_triangle_mesh(path)
    v = np.asarray(m.vertices)
    cov = np.cov(v.T)
    off_diag = np.abs(cov[0,1]) + np.abs(cov[0,2]) + np.abs(cov[1,2])
    return off_diag
