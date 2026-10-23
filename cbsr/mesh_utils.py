# small geometry helpers shared by several modules (vertices v: (V, 3), triangles t: (F, 3))
import numpy as np


def triangle_areas(v, t):
    tri = v[t]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


# weighted avg of all triangle centroids, weighted by each one's area
def barycenter(v, t):
    areas = triangle_areas(v, t)
    cents = v[t].mean(1)
    return (cents * areas[:, None]).sum(0) / areas.sum()


def scale_to_unit_cube(v):
    # uniform scaling: the longest side of the axis-aligned bounding box becomes 1
    return v / (v.max(0) - v.min(0)).max()


def edges(t, return_counts=False):
    # every edge of the triangles once (as a sorted vertex pair); with return_counts also
    # how many triangles use it (1 = boundary edge, 2 = interior, > 2 = non-manifold)
    e = np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    return np.unique(e, axis=0, return_counts=return_counts)
