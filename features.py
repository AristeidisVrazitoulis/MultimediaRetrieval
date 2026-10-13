import os
import numpy as np
import open3d as o3d
import pymeshlab as pml


# Step 3.2: 3D shape descriptors
# - 6 global (scalar) descriptors: area, compactness, rectangularity, diameter, convexity, eccentricity
# - 5 shape property distributions (histograms): A3, D1, D2, D3, D4

N_BINS = 20                  # B, see README/report for the argument (B << sqrt(N))
N_SURFACE_POINTS = 10000     # points sampled uniformly on the surface, pool for the histograms
N_SAMPLES = 1_000_000        # random tuples per histogram (tech tips 03c: ~100 per dimension for D3 -> 100^3)
SEED = 42                    # fixed seed, so the same shape always gives the same features

# fixed ranges for the histograms: every shape must use the SAME bins, otherwise the
# histograms are not comparable. Shapes are normalized to the unit cube, so the
# ranges are bounded; values outside are clipped into the first/last bin.
HIST_RANGES = {
    "A3": (0.0, np.pi),
    "D1": (0.0, 0.75),
    "D2": (0.0, 1.2),
    "D3": (0.0, 0.6),
    "D4": (0.0, 0.35),
}
SCALAR_FEATURES = ["area", "volume", "compactness", "rectangularity", "diameter", "convexity", "eccentricity"]
HIST_FEATURES = list(HIST_RANGES.keys())


# ---------------------------------------------------------------- mesh loading / repair

def load_mesh(path):
    # merge duplicated vertices: OBJ files often split vertices (normals/uvs), which
    # makes every edge look like a boundary edge
    ms = pml.MeshSet()
    ms.load_new_mesh(path)
    ms.meshing_merge_close_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_unreferenced_vertices()
    return ms


def repair_for_volume(ms):
    # tech tips 03b: the volume formula needs (1) consistently oriented triangles and
    # (2) no holes. We fix both with pymeshlab on a copy, used ONLY for the volume:
    # area and sampling use the original surface (hole-filling triangles are not real surface)
    ms.meshing_repair_non_manifold_edges()
    try:
        ms.meshing_re_orient_faces_coherently()
    except pml.PyMeshLabException:
        pass
    try:
        ms.meshing_close_holes(maxholesize=100000)
    except pml.PyMeshLabException:
        pass
    m = ms.current_mesh()
    return m.vertex_matrix(), m.face_matrix()


def count_boundary_edges(t):
    # edges used by exactly one triangle form the hole boundaries
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return int((counts == 1).sum())


# ---------------------------------------------------------------- basic geometry

def triangle_areas(v, t):
    tri = v[t]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def mesh_volume(v, t):
    # tech tips 03b: V = 1/6 |sum (v1 x v2) . v3|, with vi = xi - o (o = reference point)
    o = v.mean(0)
    a, b, c = v[t[:, 0]] - o, v[t[:, 1]] - o, v[t[:, 2]] - o
    return abs(np.einsum('ij,ij->i', np.cross(a, b), c).sum()) / 6.0


def convex_hull(v):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(v))
    hull, _ = pc.compute_convex_hull()
    return np.asarray(hull.vertices), np.asarray(hull.triangles)


def sample_surface(v, t, n, rng):
    # area-weighted uniform sampling on the surface: the distributions are very sensitive
    # to a good spread of the samples (tech tips 03c), while the vertices of our meshes are
    # not uniformly spread (decimation/subdivision leave dense and sparse regions)
    areas = triangle_areas(v, t)
    tri_idx = rng.choice(len(t), size=n, p=areas / areas.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    w0, w1, w2 = 1 - s, s * (1 - r2), s * r2
    tri = v[t[tri_idx]]
    return w0[:, None] * tri[:, 0] + w1[:, None] * tri[:, 1] + w2[:, None] * tri[:, 2]


# ---------------------------------------------------------------- global descriptors

def compactness(area, volume):
    # 3D version of P^2 / (4 pi A): A^3 / (36 pi V^2), equal to 1 for a sphere, > 1 otherwise
    return area ** 3 / (36 * np.pi * volume ** 2)


def principal_axes(points):
    cov = np.cov(points.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


def obb_volume(hull_points):
    # the PCA box alone is not enough: when two eigenvalues are (almost) equal, e.g. a cube
    # or a square rod, the PCA axes are arbitrary and the box comes out rotated and too big.
    # So we take the smallest of: Open3D's minimal OBB and the axis-aligned box (which is
    # the PCA box, since shapes are already PCA-aligned in Step 3.1)
    aabb = np.prod(hull_points.max(0) - hull_points.min(0))
    pts = o3d.utility.Vector3dVector(hull_points)
    minimal = o3d.geometry.OrientedBoundingBox.create_from_points_minimal(pts).volume()
    return min(aabb, minimal)


def diameter(points):
    # largest distance between two points; it is always realized by two convex hull
    # vertices, so we only need to check those (chunked to keep memory low)
    best = 0.0
    for i in range(0, len(points), 1000):
        d = np.linalg.norm(points[i:i + 1000, None, :] - points[None, :, :], axis=2)
        best = max(best, d.max())
    return best


# ---------------------------------------------------------------- shape property distributions

def random_tuples(n_points, k, n, rng):
    # tech tips 03c: draw n random k-tuples of points out of the n_points^k possibilities,
    # discarding tuples that use the same point twice. Drawing each index independently
    # is what the nested loop in the tech tips does (a fresh random() at every level)
    idx = rng.integers(0, n_points, size=(n, k))
    ok = np.ones(n, dtype=bool)
    for a in range(k):
        for b in range(a + 1, k):
            ok &= idx[:, a] != idx[:, b]
    return idx[ok]


def a3_values(p, rng, n=N_SAMPLES):
    idx = random_tuples(len(p), 3, n, rng)
    u = p[idx[:, 0]] - p[idx[:, 1]]
    w = p[idx[:, 2]] - p[idx[:, 1]]
    cos = np.einsum('ij,ij->i', u, w) / (np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1) + 1e-12)
    return np.arccos(np.clip(cos, -1, 1))


def d1_values(p, center):
    return np.linalg.norm(p - center, axis=1)


def d2_values(p, rng, n=N_SAMPLES):
    idx = random_tuples(len(p), 2, n, rng)
    return np.linalg.norm(p[idx[:, 0]] - p[idx[:, 1]], axis=1)


def d3_values(p, rng, n=N_SAMPLES):
    idx = random_tuples(len(p), 3, n, rng)
    area = 0.5 * np.linalg.norm(np.cross(p[idx[:, 1]] - p[idx[:, 0]], p[idx[:, 2]] - p[idx[:, 0]]), axis=1)
    return np.sqrt(area)


def d4_values(p, rng, n=N_SAMPLES):
    idx = random_tuples(len(p), 4, n, rng)
    a, b, c = (p[idx[:, i]] - p[idx[:, 3]] for i in range(3))
    vol = np.abs(np.einsum('ij,ij->i', np.cross(a, b), c)) / 6.0
    return np.cbrt(vol)


def normalized_histogram(values, name, n_bins=N_BINS):
    # tech tips 03c (1): h_i / sum h_i, so histograms built from different numbers of
    # samples are comparable (each bar is a fraction in [0, 1])
    lo, hi = HIST_RANGES[name]
    h, _ = np.histogram(np.clip(values, lo, hi), bins=n_bins, range=(lo, hi))
    return h / h.sum()


def shape_property_values(points, center, rng):
    return {
        "A3": a3_values(points, rng),
        "D1": d1_values(points, center),
        "D2": d2_values(points, rng),
        "D3": d3_values(points, rng),
        "D4": d4_values(points, rng),
    }


# ---------------------------------------------------------------- full extraction

def extract_features(path, return_raw=False):
    rng = np.random.default_rng(SEED)
    ms = load_mesh(path)
    v, t = ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix()

    boundary_before = count_boundary_edges(t)
    area = triangle_areas(v, t).sum()
    points = sample_surface(v, t, N_SURFACE_POINTS, rng)

    vr, tr = repair_for_volume(ms)
    boundary_after = count_boundary_edges(tr)
    volume = mesh_volume(vr, tr)

    hull_v, hull_t = convex_hull(v)
    hull_volume = mesh_volume(hull_v, hull_t)

    eigvals, _ = principal_axes(points)
    center = points.mean(0)

    feats = {
        "class": os.path.basename(os.path.dirname(path)),
        "file": path,
        "area": area,
        "volume": volume,
        "compactness": compactness(area, volume),
        "rectangularity": volume / obb_volume(hull_v),
        "diameter": diameter(hull_v),
        "convexity": volume / hull_volume,
        "eccentricity": eigvals[0] / eigvals[2],
        # diagnostics, to spot shapes where the volume (and so compactness,
        # rectangularity, convexity) is not reliable
        "boundary_edges_before": boundary_before,
        "boundary_edges_after": boundary_after,
        "hull_volume": hull_volume,
    }

    raw = shape_property_values(points, center, rng)
    for name, values in raw.items():
        for i, x in enumerate(normalized_histogram(values, name)):
            feats[f"{name}_{i}"] = x

    return (feats, raw) if return_raw else feats


def print_features(feats):
    print(f"{feats['class']} - {feats['file']}")
    for k in SCALAR_FEATURES:
        print(f"  {k:15s} {feats[k]:.4f}")
    for name in HIST_FEATURES:
        h = [feats[f"{name}_{i}"] for i in range(N_BINS)]
        print(f"  {name:15s} " + " ".join(f"{x:.2f}" for x in h))
