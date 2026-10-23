# the whole preprocessing + feature extraction for ONE shape, e.g. a query shape in Step 4.
# It runs exactly the same steps, with the same settings, as the database build
# (Step 2.3 remeshing -> 2.5 normalization -> 3.1 alignment/flipping -> 3.2 features).
# The intermediate meshes go through OBJ files, like for the database: saving and
# reloading is not neutral (Open3D drops unreferenced vertices when reading), so an
# in-memory shortcut could give slightly different features than the database shapes.
import os
import tempfile
import open3d as o3d

from .config import TARGET_VERTICES, TOLERANCE
from .remeshing import needs_resample, refine, simplify
from .normalization import normalize, align_and_flip
from .features import extract_features


def process_shape(path, target_v=TARGET_VERTICES, tol=TOLERANCE):
    # vertex count as in the database statistics (Open3D, see shape_stats.py)
    nv = len(o3d.io.read_triangle_mesh(path).vertices)
    name = os.path.basename(path)
    with tempfile.TemporaryDirectory() as tmp:
        if needs_resample(nv, target_v, tol):
            ms, _ = (refine if nv < target_v else simplify)(path, target_v, tol)
            remeshed = os.path.join(tmp, "remeshed_" + name)
            ms.save_current_mesh(remeshed)
        else:
            remeshed = path                         # already in the band: used unchanged
        normalized = os.path.join(tmp, "normalized_" + name)
        normalize(remeshed, normalized)
        aligned = os.path.join(tmp, "aligned_" + name)
        align_and_flip(normalized, aligned)
        feats = extract_features(aligned)
    # report the input file, not the temporary one
    feats["file"] = path
    feats["class"] = os.path.basename(os.path.dirname(path))
    return feats
