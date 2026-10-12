import numpy as np
import open3d as o3d
import os


def pca_align(v):
    cov = np.cov(v.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    v_aligned = v @ eigvecs
    return v_aligned


def moment_flip(v):
    for axis in range(3):
        f = v[:, axis]
        moment = np.sum(np.sign(f) * (f ** 2))
        if moment < 0:
            v[:, axis] = -v[:, axis]
    return v


def align_and_flip(path, out):
    m = o3d.io.read_triangle_mesh(path)
    v = np.asarray(m.vertices)
    t = np.asarray(m.triangles)

    v = pca_align(v)
    v = moment_flip(v)

    m.vertices = o3d.utility.Vector3dVector(v)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    o3d.io.write_triangle_mesh(out, m)


def check_alignment(path):
    m = o3d.io.read_triangle_mesh(path)
    v = np.asarray(m.vertices)
    cov = np.cov(v.T)
    off_diag = np.abs(cov[0,1]) + np.abs(cov[0,2]) + np.abs(cov[1,2])
    return off_diag