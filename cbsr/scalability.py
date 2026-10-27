# Step 5: scalability. K-nearest / R-nearest search with a spatial search structure (kd-tree) and
# the 2D t-SNE map of the feature database. Independent of any GUI: plain data in, plain data out.
#
# The kd-tree works with ONE built-in Lp distance over ONE vector per shape (here L2), while the
# Step 4 "custom" distance is a weighted sum of per-block distances. The index vector of a shape is
# therefore the concatenation of its Step 4 block vectors (normalized scalars, histograms or
# cumulative histograms for EMD), each divided by that block's distance std sigma. With this
# scaling every block contributes on the same scale, as in the custom distance, but the blocks are
# combined by L2 over all coordinates instead of by summing the block distances.
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KDTree

from .config import SEED, KDTREE_LEAF_SIZE, TSNE_PERPLEXITY, TSNE_MAX_ITER
from .retrieval import BLOCKS, block_vectors


def index_vectors(prep, features=None):
    # features=None: the whole database (one row per shape); otherwise one query shape (dict/Series)
    if features is None:
        blocks = prep["vectors"]
    else:
        blocks = block_vectors(features, prep["params"], prep["config"]["hist_distance"])
    return np.hstack([blocks[b] / prep["sigma"][b] for b in BLOCKS])


def build_index(prep, leaf_size=KDTREE_LEAF_SIZE):
    # built once for the database; queries then only visit the tree cells close to the query
    return KDTree(index_vectors(prep), leaf_size=leaf_size, metric="euclidean")


def knn(tree, prep, query_features, k=5):
    # the k database shapes closest to the query in L2 (the query itself left out), closest first.
    # One extra neighbour is asked for, because a database query finds itself at distance 0
    q = index_vectors(prep, query_features)
    d, i = tree.query(q, k=k + 1)
    out = prep["shapes"].iloc[i[0]].assign(distance=d[0])
    out = out[out.file != query_features.get("file")]
    return out.head(k).reset_index(drop=True)


def range_search(tree, prep, query_features, r):
    # all database shapes within L2 distance r of the query (the query itself left out), closest first
    q = index_vectors(prep, query_features)
    i, d = tree.query_radius(q, r=r, return_distance=True, sort_results=True)
    out = prep["shapes"].iloc[i[0]].assign(distance=d[0])
    out = out[out.file != query_features.get("file")]
    return out.reset_index(drop=True)


def tsne_embedding(prep, perplexity=TSNE_PERPLEXITY, max_iter=TSNE_MAX_ITER, seed=SEED):
    # 2D map of the database: one point per shape, computed from the same vectors the kd-tree uses
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter, init="pca", random_state=seed)
    xy = tsne.fit_transform(index_vectors(prep))
    return prep["shapes"].assign(x=xy[:, 0], y=xy[:, 1]), float(tsne.kl_divergence_)
