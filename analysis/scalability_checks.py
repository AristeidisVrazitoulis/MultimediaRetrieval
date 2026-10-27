# checks and figures for Step 5: custom distance (brute force) vs kd-tree (L2), and the t-SNE map
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import open3d as o3d
import plotly.express as px
from sklearn.neighbors import NearestNeighbors

from cbsr import config
from cbsr.retrieval import query
from cbsr.scalability import index_vectors, knn
from analysis.retrieval_checks import all_distances

FIGURES_DIR = config.STEP5_FIGURES


# ---------------------------------------------------------------- neighbours of every database shape

def custom_neighbours(prep, k, chunk=200):
    # the k closest shapes of every database shape with the Step 4 custom distance (self left out)
    n = len(prep["shapes"])
    out = np.empty((n, k), dtype=int)
    for start in range(0, n, chunk):
        rows = np.arange(start, min(start + chunk, n))
        d = all_distances(prep, rows)
        d[np.arange(len(rows)), rows] = np.inf
        part = np.argpartition(d, k, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(d, part, 1), axis=1)
        out[rows] = np.take_along_axis(part, order, 1)
    return out


def drop_self(idx, k):
    # neighbour lists asked with k+1 neighbours: remove the query itself (not always in first
    # position, since a few shapes have exactly the same features as another shape)
    out = np.empty((len(idx), k), dtype=int)
    for row, nb in enumerate(idx):
        out[row] = nb[nb != row][:k]
    return out


def tree_neighbours(tree, prep, k):
    _, idx = tree.query(index_vectors(prep), k=k + 1)
    return drop_self(idx, k)


def check_tree_exact(tree, prep, k):
    # same L2 distance as the kd-tree, but by comparing with every shape: the kd-tree must find the same
    # neighbour distances. (Comparing shape ids instead fails for the few shapes with exactly the same
    # features as another one: which of two shapes at the same distance comes first is arbitrary)
    X = index_vectors(prep)
    d_tree = tree.query(X, k=k + 1)[0]
    d_brute = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(X).kneighbors(X)[0]
    print(f"kd-tree vs brute-force L2: largest difference in the {k} neighbour distances = "
          f"{np.abs(d_tree - d_brute).max():.1e}")


def precision(prep, neighbours):
    classes = prep["shapes"]["class"].to_numpy()
    return float((classes[neighbours] == classes[:, None]).mean())


def overlap(a, b):
    # average fraction of the k results that the two searches have in common
    return float(np.mean([len(set(x) & set(y)) / len(x) for x, y in zip(a, b)]))


def mean_query_time(search, db):
    # average time of one query (one database shape at a time, as in the app), in milliseconds
    rows = db.to_dict("records")
    t0 = time.perf_counter()
    for q in rows:
        search(q)
    return 1000 * (time.perf_counter() - t0) / len(rows)


def compare_searches(db, prep, tree, k):
    custom = custom_neighbours(prep, k)
    kd = tree_neighbours(tree, prep, k)
    check_tree_exact(tree, prep, k)
    rows = [
        {"search": "custom distance (Step 4, brute force)", f"precision@{k}": precision(prep, custom),
         "ms_per_query": mean_query_time(lambda q: query(prep, q, k), db)},
        {"search": "kd-tree (L2)", f"precision@{k}": precision(prep, kd),
         "ms_per_query": mean_query_time(lambda q: knn(tree, prep, q, k), db)},
    ]
    table = pd.DataFrame(rows)
    table["overlap_with_custom"] = [1.0, overlap(custom, kd)]
    return table


# ---------------------------------------------------------------- query table (tech tips step 5)

def query_table(db, prep, tree, query_files, k):
    # long format: one row per (query, method, rank)
    rows = []
    for f in query_files:
        q = db[db.file == f].iloc[0].to_dict()
        for method, res in (("custom", query(prep, q, k)), ("kdtree", knn(tree, prep, q, k))):
            for rank, r in res.iterrows():
                rows.append({"query": f, "query_class": q["class"], "method": method, "rank": rank + 1,
                             "file": r["file"], "class": r["class"], "distance": r["distance"]})
    return pd.DataFrame(rows)


def render(path, size=400):
    # shaded, same oblique viewpoint for every shape (shapes are normalized, so the same zoom fits all)
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.7, 0.7, 0.72])
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=size, height=size)
    vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_back_face = True
    ctr = vis.get_view_control()
    ctr.set_front([0.5, 0.35, 0.8])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.8)
    vis.poll_events()
    vis.update_renderer()
    img = np.clip(np.asarray(vis.capture_screen_float_buffer(do_render=True)), 0, 1)
    vis.destroy_window()
    return img


def plot_query_table(table, method, out_name, cache):
    # rows = query shapes, first column = query, other columns = results left to right by distance
    t = table[table.method == method]
    queries = list(dict.fromkeys(t["query"]))
    k = t["rank"].max()
    fig, axes = plt.subplots(len(queries), k + 1, figsize=(2.2 * (k + 1), 2.5 * len(queries)))
    for i, qf in enumerate(queries):
        cells = [(qf, "query")] + [(r.file, f"d = {r.distance:.2f}") for r in t[t["query"] == qf].itertuples()]
        for j, (f, label) in enumerate(cells):
            if f not in cache:
                cache[f] = render(f)
            ax = axes[i, j]
            ax.imshow(cache[f])
            ax.set_xticks([])
            ax.set_yticks([])
            name = f.split("/")[-2] + "/" + f.split("/")[-1].removesuffix(".obj")
            ax.set_xlabel(f"{name}\n{label}", fontsize=8, color="black" if j == 0 or
                          f.split("/")[-2] == qf.split("/")[-2] else "firebrick")
            if j == 0:
                for s in ax.spines.values():
                    s.set_linewidth(2)
    title = "custom distance (Step 4)" if method == "custom" else "kd-tree, L2 distance"
    fig.suptitle(f"{k} nearest neighbours, {title} (red label = other class)")
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, out_name)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("saved", out)


# ---------------------------------------------------------------- t-SNE map

def class_styles(classes):
    # 69 classes: 20 colours x 4 marker shapes, so every class gets a different (colour, marker)
    colours = plt.get_cmap("tab20").colors
    markers = ["o", "s", "^", "D"]
    return {c: (colours[i % 20], markers[i // 20 % 4]) for i, c in enumerate(sorted(set(classes)))}


def map_precision(emb, k=5):
    # fraction of the k nearest points IN THE 2D MAP that have the same class: a number that says
    # how well the colours are separated, to compare t-SNE settings
    xy = emb[["x", "y"]].to_numpy()
    _, idx = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy)
    classes = emb["class"].to_numpy()
    return float((classes[drop_self(idx, k)] == classes[:, None]).mean())


def plot_tsne(emb, title, out_name):
    styles = class_styles(emb["class"])
    fig, ax = plt.subplots(figsize=(15, 10))
    for c, g in emb.groupby("class"):
        colour, marker = styles[c]
        ax.scatter(g.x, g.y, s=14, color=colour, marker=marker, label=c, alpha=0.85, linewidths=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=3, loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=1.5, frameon=False)
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, out_name)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("saved", out)


def tsne_html(emb, title, out_name):
    # interactive version: hovering a point (brushing) shows the shape name and class
    e = emb.assign(shape=emb.file.str.split("/").str[-1])
    fig = px.scatter(e, x="x", y="y", color="class", hover_name="shape", hover_data={"class": True,
                     "x": False, "y": False}, title=title, height=800)
    fig.update_traces(marker=dict(size=6))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    out = os.path.join(FIGURES_DIR, out_name)
    fig.write_html(out, include_plotlyjs="cdn")
    print("saved", out)
