import numpy as np
import open3d as o3d
import os


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


def scale_to_unit_cube(v):
    # size is the LAST step (translation, pose, flipping, size): after alignment
    # the AABB equals the OBB, so this gives the tight unit-cube fit
    return v / (v.max(0) - v.min(0)).max()


def align_and_flip(path, out):
    m = o3d.io.read_triangle_mesh(path)
    v = np.asarray(m.vertices)
    t = np.asarray(m.triangles)

    v = pca_align(v)
    v = moment_flip(v, t)
    v = scale_to_unit_cube(v)

    m.vertices = o3d.utility.Vector3dVector(v)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    o3d.io.write_triangle_mesh(out, m)


def check_alignment(path):
    m = o3d.io.read_triangle_mesh(path)
    v = np.asarray(m.vertices)
    cov = np.cov(v.T)
    off_diag = np.abs(cov[0,1]) + np.abs(cov[0,2]) + np.abs(cov[1,2])
    return off_diag