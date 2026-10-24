import streamlit as st
import plotly.graph_objects as go
import open3d as o3d
import numpy as np
from cbsr.retrieval import load_database, query, fit_normalization, HIST_DISTANCE_FUNCTIONS

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

@st.cache_resource
def norm_params(method):
    return fit_normalization(db, method)

with st.sidebar:
    shape = st.selectbox("Query shape", db.file)
    k = st.slider("Number of results (K)", 1, 10, 5)
    norm_method = st.radio("Normalization", ["standardize", "minmax"])
    hist_distance = st.radio("Histogram distance", list(HIST_DISTANCE_FUNCTIONS.keys()))

q = db[db.file == shape].iloc[0].to_dict()
st.subheader(f"Query: {shape}")
st.plotly_chart(mesh_figure(shape), use_container_width=True)

st.subheader(f"Top {k} matches")
results = query(db, q, k, norm_params(norm_method), hist_distance)
cols = st.columns(min(k, 5))
for i, r in enumerate(results.itertuples()):
    with cols[i % len(cols)]:
        st.plotly_chart(mesh_figure(r.file, 220), use_container_width=True, key=f"r{i}")
        st.caption(f"{r._1} · {r.file.split('/')[-1]} · d = {r.distance:.3f}")