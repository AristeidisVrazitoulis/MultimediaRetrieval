# use this file to play around with steps or part of the steps. It is like a draft

# Step 2 report figures (professor feedback + tech tips). Run from the project root:
#   .venv310/bin/python steps/scratch.py
# Mesh renders use the CURRENT remeshing code in memory (independent of the database
# on disk). The histograms read the stats CSVs and the remeshed database, so run
# steps/step2.py first to make those up to date.

import sys
import os
import subprocess
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import open3d as o3d
import pymeshlab as pml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import remeshing as R


STATS_SOURCE_FOLDER = "stats/"
FIGURES_DIR = STATS_SOURCE_FOLDER + "figures_step2/"
ORIGINAL_DB = "ShapeDatabase"
REMESHED_DB = "ShapeDatabase_remeshed"

initial_stats_filename = STATS_SOURCE_FOLDER + "original_db_shape_stats.csv"
remeshed_stats_filename = STATS_SOURCE_FOLDER + "remeshed_db_shape_stats.csv"
normalization_check_filename = STATS_SOURCE_FOLDER + "normalization_check.csv"

TARGET_VERTICES = 6000
TOLERANCE = 0.10
BAND = (TARGET_VERTICES * (1 - TOLERANCE), TARGET_VERTICES * (1 + TOLERANCE))

# shapes used in the figures
REFINED_SHAPE = "Door/m1708.obj"          # 16 vertices
DECIMATED_SHAPE = "Biplane/m1120.obj"     # 65,722 vertices
NONMANIFOLD_SHAPE = "Shelf/D00881.obj"    # 50 of 400 faces on non-manifold edges
N_FACE_AREA_SHAPES = 500                  # shapes sampled for the face-area histogram (None = all)

# which figures to make
MAKE_RENDERS = True
MAKE_HISTOGRAMS = True


df = pd.read_csv(initial_stats_filename)

# average shape (Step 2.2): the shape whose vertex count is closest to the mean
avg_vertices = df['num_vertices'].mean()
closest_idx = (df['num_vertices'] - avg_vertices).abs().idxmin()
average_shape = df.loc[closest_idx]
print(average_shape)


# ---------------------------------------------------------------- rendering helpers

def ms_to_o3d(ms, red_faces=None):
    # pymeshlab mesh -> open3d mesh; vertices of red_faces (bool per face) are painted red
    m = ms.current_mesh()
    v, f = m.vertex_matrix(), m.face_matrix()
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f))
    mesh.compute_vertex_normals()
    colors = np.full((len(v), 3), 0.75)
    if red_faces is not None:
        colors[np.unique(f[red_faces])] = [0.85, 0.1, 0.1]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh


def render(mesh, width=900, height=700):
    # wireframe on shaded, same oblique viewpoint for every shape, returned as an image
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    opt.mesh_show_wireframe = True
    opt.mesh_show_back_face = True
    ctr = vis.get_view_control()
    ctr.set_front([0.5, 0.35, 0.8])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.75)
    vis.poll_events()
    vis.update_renderer()
    img = np.clip(np.asarray(vis.capture_screen_float_buffer(do_render=True)), 0, 1)
    vis.destroy_window()
    return img


def save_panels(panels, title, out_name):
    # panels: list of (image or text, caption); a text panel is used for "no result"
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.6))
    for ax, (content, caption), letter in zip(np.atleast_1d(axes), panels, "abcdefgh"):
        if isinstance(content, str):
            text = "\n".join(textwrap.fill(line, 32) for line in content.split("\n"))
            ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=11)
        else:
            ax.imshow(content)
        ax.set_xlabel(f"({letter}) {caption}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(FIGURES_DIR + out_name, dpi=150)
    plt.close(fig)
    print("saved", FIGURES_DIR + out_name)


def load(path):
    ms = pml.MeshSet()
    ms.load_new_mesh(path)
    return ms


def subdivide_without_cleaning(path):
    # runs in a separate process: on uncleaned meshes pymeshlab either raises an
    # exception or crashes the whole process (segfault), which try/except cannot catch
    code = ("import pymeshlab as pml; ms = pml.MeshSet(); ms.load_new_mesh(%r); "
            "ms.meshing_surface_subdivision_midpoint(iterations=1, threshold=pml.PercentageValue(0)); "
            "print(ms.current_mesh().vertex_number())" % path)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if p.returncode == 0:
        return f"subdivision ran: {p.stdout.strip()} vertices"
    if p.returncode < 0 or p.returncode == 139:
        return "subdivision crashed\n(segmentation fault)"
    reason = (p.stderr.strip().splitlines() or ["unknown error"])[-1]
    return "subdivision rejected the mesh:\n" + reason.replace("Details: ", "")


# ---------------------------------------------------------------- renders

def fig_refined():
    path = os.path.join(ORIGINAL_DB, REFINED_SHAPE)
    before = load(path)
    after, _ = R.refine(path, TARGET_VERTICES, TOLERANCE)
    save_panels([(render(ms_to_o3d(before)), f"before: {before.current_mesh().vertex_number()} vertices"),
                 (render(ms_to_o3d(after)), f"after refinement: {after.current_mesh().vertex_number()} vertices")],
                f"Refinement of {REFINED_SHAPE}", "refined_example.png")


def fig_decimated():
    path = os.path.join(ORIGINAL_DB, DECIMATED_SHAPE)
    before = load(path)
    after, _ = R.simplify(path, TARGET_VERTICES, TOLERANCE)
    save_panels([(render(ms_to_o3d(before)), f"before: {before.current_mesh().vertex_number()} vertices"),
                 (render(ms_to_o3d(after)), f"after simplification: {after.current_mesh().vertex_number()} vertices")],
                f"Simplification of {DECIMATED_SHAPE}", "decimated_example.png")


def fig_nonmanifold():
    path = os.path.join(ORIGINAL_DB, NONMANIFOLD_SHAPE)
    # (a) the original, faces on non-manifold edges in red. Vertices are welded first
    # (without changing the shape), otherwise every edge of a triangle soup looks open
    ms = load(path)
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_remove_duplicate_vertices()
    ms.compute_selection_by_non_manifold_edges_per_face()
    red = ms.current_mesh().face_selection_array()
    n_red, n_faces = int(red.sum()), len(red)
    # (b) what midpoint subdivision does without cleaning
    without = subdivide_without_cleaning(path)
    # (c) cleaning + refinement
    after, _ = R.refine(path, TARGET_VERTICES, TOLERANCE)
    save_panels([(render(ms_to_o3d(ms, red)), f"original: {n_red}/{n_faces} faces on non-manifold edges (red)"),
                 (without, "refinement without cleaning"),
                 (render(ms_to_o3d(after)), f"cleaning + refinement: {after.current_mesh().vertex_number()} vertices")],
                f"Non-manifold shape {NONMANIFOLD_SHAPE}", "nonmanifold_example.png")


def uniform_refine(path, target_v=TARGET_VERTICES):
    # for comparison only: classic uniform midpoint subdivision (every edge split)
    ms = load(path)
    R.clean_mesh(ms)
    while ms.current_mesh().vertex_number() < target_v:
        ms.meshing_surface_subdivision_midpoint(iterations=1, threshold=pml.PercentageValue(0))
    return ms


def fig_uniform_vs_adaptive(shape=None):
    # default shape: the one (among 1,500-3,000 vertices) with the most uneven edge
    # lengths, where the difference between the two methods is easiest to see
    if shape is None:
        cand = df[df.num_vertices.between(1500, 3000)].sample(80, random_state=0)
        def unevenness(p):
            ms = load(p)
            R.clean_mesh(ms)
            L = R.edge_lengths(ms.current_mesh())
            return L.std() / L.mean()
        path = max(cand.file, key=unevenness)
    else:
        path = os.path.join(ORIGINAL_DB, shape)
    before = load(path)
    uni = uniform_refine(path)
    ada, _ = R.refine(path, TARGET_VERTICES, TOLERANCE)
    def spread(ms):   # std/mean of edge lengths: lower = more even sampling
        L = R.edge_lengths(ms.current_mesh())
        return L.std() / L.mean()
    save_panels([(render(ms_to_o3d(before)), f"original: {before.current_mesh().vertex_number()} vertices"),
                 (render(ms_to_o3d(uni)), f"uniform: {uni.current_mesh().vertex_number()} vertices, edge spread {spread(uni):.2f}"),
                 (render(ms_to_o3d(ada)), f"adaptive: {ada.current_mesh().vertex_number()} vertices, edge spread {spread(ada):.2f}")],
                f"Uniform vs area-adaptive subdivision: {os.path.relpath(path, ORIGINAL_DB)}", "uniform_vs_adaptive.png")


# ---------------------------------------------------------------- histograms

def band_lines(ax):
    ax.axvline(TARGET_VERTICES, color="red", linestyle="--", label=f"target {TARGET_VERTICES}")
    ax.axvline(BAND[0], color="gray", linestyle=":", label=f"band [{BAND[0]:.0f}, {BAND[1]:.0f}]")
    ax.axvline(BAND[1], color="gray", linestyle=":")


def hist_vertices_before_after():
    # same log-spaced bins for both: wide and flat before, one narrow spike after
    before = pd.read_csv(initial_stats_filename).num_vertices
    after = pd.read_csv(remeshed_stats_filename).num_vertices
    bins = np.logspace(np.log10(min(before.min(), after.min())), np.log10(max(before.max(), after.max())), 26)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(before, bins=bins, alpha=0.6, label="before remeshing")
    ax.hist(after, bins=bins, alpha=0.6, label="after remeshing")
    band_lines(ax)
    ax.set_xscale("log")
    ax.set_xlabel("Number of vertices (log scale)")
    ax.set_ylabel("Number of shapes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR + "vertices_before_after.png", dpi=150)
    plt.close(fig)
    in_band = after.between(*BAND)
    print(f"after remeshing: {in_band.sum()}/{len(after)} shapes in band, mean {after.mean():.0f}")


def hist_vertices_after():
    # linear, fixed bin width of 100 vertices, with target and band
    after = pd.read_csv(remeshed_stats_filename).num_vertices
    lo, hi = np.floor(after.min() / 100) * 100, np.ceil(after.max() / 100) * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(after, bins=np.arange(lo, hi + 100, 100))
    band_lines(ax)
    ax.set_xlabel("Number of vertices")
    ax.set_ylabel("Number of shapes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR + "vertices_after.png", dpi=150)
    plt.close(fig)


def relative_face_areas(path):
    # face area divided by the MEDIAN face area of that shape: makes shapes of
    # different sizes comparable, and measures how uniform the faces are
    # (the mean is pulled up by a few huge faces, the median is not)
    m = o3d.io.read_triangle_mesh(path)
    v, t = np.asarray(m.vertices), np.asarray(m.triangles)
    a = 0.5 * np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1)
    a = a[a > 0]
    return a / np.median(a)


def hist_face_areas_before_after():
    shapes = df if N_FACE_AREA_SHAPES is None else df.sample(N_FACE_AREA_SHAPES, random_state=0)
    before, after = [], []
    for _, r in shapes.iterrows():
        name = os.path.join(r['class'], os.path.basename(r['file']))
        before.append(relative_face_areas(os.path.join(ORIGINAL_DB, name)))
        after.append(relative_face_areas(os.path.join(REMESHED_DB, name)))
    before, after = np.concatenate(before), np.concatenate(after)
    # range wide enough that (almost) nothing is cut off; y = fraction of all faces per bin
    lo = np.floor(np.log10(np.percentile(np.concatenate([before, after]), 0.1)))
    hi = np.ceil(np.log10(np.percentile(np.concatenate([before, after]), 99.9)))
    bins = np.logspace(lo, hi, 50)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(before, bins=bins, alpha=0.6, weights=np.full(len(before), 1 / len(before)), label="before remeshing")
    ax.hist(after, bins=bins, alpha=0.6, weights=np.full(len(after), 1 / len(after)), label="after remeshing")
    ax.axvline(1, color="red", linestyle="--", label="median face area of the shape")
    ax.set_xscale("log")
    ax.set_xlabel("Face area / median face area of its shape (log scale)")
    ax.set_ylabel("Fraction of faces")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR + "face_areas_before_after.png", dpi=150)
    plt.close(fig)
    for name, x in (("before", before), ("after", after)):
        print(f"relative face area {name}: 5-95% range [{np.percentile(x, 5):.3f}, {np.percentile(x, 95):.2f}], "
              f"share within 0.5x-2x of the median: {((x > 0.5) & (x < 2)).mean():.0%}")


def hist_normalization_check():
    d = pd.read_csv(normalization_check_filename)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(d.bary_norm, bins=30)
    axes[0].set_xlabel("Distance of barycenter to origin")
    axes[1].hist(d.longest_side, bins=30)
    axes[1].set_xlabel("Longest side of bounding box")
    for ax in axes:
        ax.set_ylabel("Number of shapes")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR + "normalization_check.png", dpi=150)
    plt.close(fig)
    print(f"barycenter distance max {d.bary_norm.max():.2e} | longest side min {d.longest_side.min():.6f} max {d.longest_side.max():.6f}")


os.makedirs(FIGURES_DIR, exist_ok=True)

if MAKE_RENDERS:
    fig_refined()
    fig_decimated()
    fig_nonmanifold()
    fig_uniform_vs_adaptive()

if MAKE_HISTOGRAMS:
    hist_vertices_before_after()
    hist_vertices_after()
    hist_face_areas_before_after()
    hist_normalization_check()
