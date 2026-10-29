import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import open3d as o3d
import numpy as np
import pandas as pd
from cbsr import config
from cbsr.features import SCALAR_FEATURES
from cbsr.retrieval import load_database, load_prepared, normalized_feature_table, query
from cbsr.scalability import build_index, knn, range_search, tsne_embedding

st.set_page_config(page_title="3D shape retrieval", layout="wide")

@st.cache_data
def database():
    return load_database()

def mesh_figure(path, height=300):
    m = o3d.io.read_triangle_mesh(path)
    v, t = np.asarray(m.vertices), np.asarray(m.triangles)
    fig = go.Figure(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=t[:, 0], j=t[:, 1], k=t[:, 2],
                              color="lightgray", flatshading=True,
                              lighting=dict(ambient=0.3, diffuse=0.9, specular=0.3, roughness=0.5,
                                            facenormalsepsilon=0, vertexnormalsepsilon=0),
                              lightposition=dict(x=100, y=200, z=300)))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), scene=dict(aspectmode="data"))
    return fig

db = database()

# the database side of the retrieval (normalization + distance weighting), loaded from
# results/step4/prepared_params.json (written by steps/step4.py) once per configuration
@st.cache_resource
def prepared(norm_method, scalar_distance, hist_distance):
    return load_prepared(db, norm_method, scalar_distance, hist_distance)

@st.cache_data
def normalized_table(norm_method):
    prep, _ = prepared(norm_method, config.DEFAULT_SCALAR_DISTANCE, config.DEFAULT_HIST_DISTANCE)
    return normalized_feature_table(db, prep["params"])

# Step 5: the kd-tree, built once per configuration
@st.cache_resource
def kd_tree(norm_method, scalar_distance, hist_distance):
    return build_index(prepared(norm_method, scalar_distance, hist_distance)[0])

# Step 5: t-SNE map. For the default configuration it is read from results/step5 (written by
# steps/step5.py); for the other configurations it is computed here (about a minute)
@st.cache_data
def tsne_map(norm_method, scalar_distance, hist_distance):
    stored = config.TSNE_CSV.format(perplexity=config.TSNE_PERPLEXITY)
    default = (norm_method, scalar_distance, hist_distance) == (
        config.DEFAULT_NORMALIZATION, config.DEFAULT_SCALAR_DISTANCE, config.DEFAULT_HIST_DISTANCE)
    if default and os.path.exists(stored):
        return pd.read_csv(stored)
    return tsne_embedding(prepared(norm_method, scalar_distance, hist_distance)[0])[0]


def default_index(options, value):
    return options.index(value)

SEARCHES = ["custom distance (Step 4)", "kd-tree: K nearest", "kd-tree: within distance R"]

with st.sidebar:
    shape = st.selectbox("Query shape", db.file)
    search = st.radio("Search", SEARCHES)
    if search == SEARCHES[2]:
        r = st.slider("Distance R (L2)", 0.05, 2.0, 0.45, 0.05)
    else:
        k = st.slider("Number of results (K)", 1, 10, 5)
    norm_method = st.radio("Normalization", config.NORMALIZATION_METHODS,
                           index=default_index(config.NORMALIZATION_METHODS, config.DEFAULT_NORMALIZATION))
    scalar_distance = st.radio("Scalar distance", config.SCALAR_DISTANCES,
                               index=default_index(config.SCALAR_DISTANCES, config.DEFAULT_SCALAR_DISTANCE))
    hist_distance = st.radio("Histogram distance", config.HIST_DISTANCES,
                             index=default_index(config.HIST_DISTANCES, config.DEFAULT_HIST_DISTANCE))

prep, from_file = prepared(norm_method, scalar_distance, hist_distance)
if not from_file:
    st.warning("No stored parameters for this configuration and feature table: computed them now. "
               "Run steps/step4.py to store them, so the app, the report and the evaluation use the same numbers.")

q = db[db.file == shape].iloc[0].to_dict()


tab_query, tab_features, tab_tsne, tab_eval = st.tabs(["Query", "Normalized features", "t-SNE map", "Evaluation"])

# ---------------------------------------------------------------- tab 1: query + best matches
MAX_SHOWN = 20      # a range search can return many shapes: only the closest ones are rendered

with tab_query:
    st.subheader(f"Query: {shape}")
    st.plotly_chart(mesh_figure(shape), width="stretch")

    if search == SEARCHES[0]:
        results = query(prep, q, k)
        st.subheader(f"Top {k} matches")
    elif search == SEARCHES[1]:
        results = knn(kd_tree(norm_method, scalar_distance, hist_distance), prep, q, k)
        st.subheader(f"Top {k} matches (kd-tree, L2 distance)")
    else:
        results = range_search(kd_tree(norm_method, scalar_distance, hist_distance), prep, q, r)
        st.subheader(f"{len(results)} shapes within R = {r:.2f} (kd-tree, L2 distance)")
        if len(results) > MAX_SHOWN:
            st.caption(f"showing the closest {MAX_SHOWN}")
            results = results.head(MAX_SHOWN)
    st.caption(f"{norm_method} · scalars: {scalar_distance} · histograms: {hist_distance} · "
               f"log10 on {', '.join(config.LOG_FEATURES)} · distance weighting")
    if len(results):
        cols = st.columns(min(len(results), 5))
    else:
        st.info("No database shape within this distance: increase R.")
    for i, (_, res) in enumerate(results.iterrows()):
        with cols[i % len(cols)]:
            st.plotly_chart(mesh_figure(res["file"], 220), width="stretch", key=f"r{i}")
            st.caption(f"{res['class']} · {res['file'].split('/')[-1]} · d = {res['distance']:.3f}")

# ---------------------------------------------------------------- tab 2: normalized feature database
with tab_features:
    table = normalized_table(norm_method)
    st.subheader(f"Normalized feature database ({norm_method})")
    st.caption(f"{len(table)} shapes · scalar features normalized, histograms as stored (sum to 1)")

    # the query shape first, so it is easy to find
    st.markdown("**Query shape**")
    st.dataframe(table[table.file == shape], hide_index=True)

    # check that the normalization worked: mean ≈ 0, std ≈ 1 (standardize) or min 0, max 1 (minmax)
    st.markdown("**Per-feature summary**")
    scalar_cols = [c for c in table.columns if c.removeprefix("log10_") in SCALAR_FEATURES]
    st.dataframe(table[scalar_cols].describe().loc[["mean", "std", "min", "max"]].round(3))

    st.markdown("**All shapes**")
    st.dataframe(table, hide_index=True, height=400)

    st.download_button("Download as CSV", table.to_csv(index=False),
                       file_name=f"normalized_features_{norm_method}.csv")

# ---------------------------------------------------------------- tab 3: t-SNE map (Step 5)
with tab_tsne:
    with st.spinner("computing the t-SNE map..."):
        emb = tsne_map(norm_method, scalar_distance, hist_distance)
    st.subheader("t-SNE map of the feature database")
    st.caption(f"one point per shape, coloured by class · perplexity {config.TSNE_PERPLEXITY}, "
               f"{config.TSNE_MAX_ITER} iterations · hover a point to see the shape · star = query shape")
    e = emb.assign(shape=emb.file.str.split("/").str[-1])
    fig = px.scatter(e, x="x", y="y", color="class", hover_name="shape",
                     hover_data={"class": True, "x": False, "y": False}, height=750)
    fig.update_traces(marker=dict(size=6))
    qp = e[e.file == shape]
    fig.add_trace(go.Scatter(x=qp.x, y=qp.y, mode="markers", name="query", hovertext=qp["shape"],
                             marker=dict(symbol="star", size=20, color="black", line=dict(color="white", width=1))))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- tab 4: evaluation (Step 6)
PRECISION_PER_CLASS_CSV = "results/step6/precision_per_class.csv"

@st.cache_data
def precision_per_class():
    return pd.read_csv(PRECISION_PER_CLASS_CSV)

with tab_eval:
    st.subheader("Evaluation (Step 6): precision@10")
    try:
        prec_df = precision_per_class()
        grand_avg = (prec_df["mean_precision"] * prec_df["n_shapes"]).sum() / prec_df["n_shapes"].sum()

        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Grand average precision@10", f"{grand_avg:.3f}")
            q_class = q["class"]
            row = prec_df[prec_df["class"] == q_class]
            if len(row):
                st.metric(f"Precision@10 for '{q_class}'", f"{row.iloc[0]['mean_precision']:.3f}",
                          help=f"{int(row.iloc[0]['n_shapes'])} shapes in this class")

        with col2:
            chart_df = prec_df.sort_values("mean_precision", ascending=True).set_index("class")
            st.bar_chart(chart_df["mean_precision"], height=max(300, 12 * len(chart_df)))

    except FileNotFoundError:
        st.info("Run `python steps/step6.py` first to generate the evaluation results.")