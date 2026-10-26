import streamlit as st
import plotly.graph_objects as go
import open3d as o3d
import numpy as np
from cbsr import config
from cbsr.features import SCALAR_FEATURES
from cbsr.retrieval import load_database, load_prepared, normalized_feature_table, query

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


def default_index(options, value):
    return options.index(value)

with st.sidebar:
    shape = st.selectbox("Query shape", db.file)
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


tab_query, tab_features = st.tabs(["Query", "Normalized features"])

# ---------------------------------------------------------------- tab 1: query + best matches
with tab_query:
    st.subheader(f"Query: {shape}")
    st.plotly_chart(mesh_figure(shape), width="stretch")

    st.subheader(f"Top {k} matches")
    st.caption(f"{norm_method} · scalars: {scalar_distance} · histograms: {hist_distance} · "
               f"log10 on {', '.join(config.LOG_FEATURES)} · distance weighting")
    results = query(prep, q, k)
    cols = st.columns(min(k, 5))
    for i, (_, r) in enumerate(results.iterrows()):
        with cols[i % len(cols)]:
            st.plotly_chart(mesh_figure(r["file"], 220), width="stretch", key=f"r{i}")
            st.caption(f"{r['class']} · {r['file'].split('/')[-1]} · d = {r['distance']:.3f}")

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
