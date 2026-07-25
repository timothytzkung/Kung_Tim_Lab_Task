import json

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

DATA_PATH = "data/business-licences-curated-v3.geojson"

st.set_page_config(page_title="Vancouver Business Area Explorer", layout="wide")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        geojson = json.load(f)
    df = pd.json_normalize([feat["properties"] for feat in geojson["features"]])
    return df


@st.cache_data
def build_composition_matrix(df: pd.DataFrame, min_businesses: int):
    """Area x business-type matrix, row-normalized to percentages.

    Rows are localarea, columns are businesstype, values are the percentage
    of an area's businesses belonging to that type. Areas with fewer than
    min_businesses total licences are dropped before normalizing so a single
    business can't swing a whole feature.
    """
    counts_per_area = df["localarea"].value_counts()
    kept_areas = counts_per_area[counts_per_area >= min_businesses].index
    filtered = df[df["localarea"].isin(kept_areas)]

    composition = pd.crosstab(filtered["localarea"], filtered["businesstype"], normalize="index") * 100
    composition = composition.loc[counts_per_area[kept_areas].sort_values(ascending=False).index]
    return composition, counts_per_area


st.title("Vancouver Business Area Explorer")
st.caption(
    "Unit of analysis: `localarea` (25 named Vancouver neighborhoods). "
    "Each area is represented by its business-type composition — the percentage "
    "of its businesses belonging to each `businesstype`."
)

df = load_data(DATA_PATH)
all_area_counts = df["localarea"].value_counts()

st.sidebar.header("Controls")

st.sidebar.subheader("1. Minimum business-count threshold")
min_businesses = st.sidebar.slider(
    "Minimum businesses per area",
    min_value=10,
    max_value=500,
    value=100,
    step=10,
    help="Areas with fewer total licences than this are excluded before building the composition matrix.",
)

composition, counts_per_area = build_composition_matrix(df, min_businesses)
n_dropped = (counts_per_area < min_businesses).sum()
dropped_names = counts_per_area[counts_per_area < min_businesses].index.tolist()

with st.expander("Why this threshold?", expanded=False):
    st.markdown(
        f"""
The composition matrix has **{df['businesstype'].nunique()} business-type columns**, each
expressed as a percentage of an area's total business count. In a small area, a single
business shifts its column by `1 / n_businesses`, so an area with only 20 businesses can
see any single feature swing by ~5 percentage points from one licence alone — noise, not
signal about the area's real character.

At the default threshold of **{min_businesses} businesses**, one business can move a
feature by at most **{100 / min_businesses:.1f} percentage points**, which keeps the
percentages reasonably stable while still including small-but-real neighborhoods
(e.g. Shaughnessy, South Cambie, Oakridge all clear 100+).

With this threshold, **{n_dropped} area(s)** are dropped: {", ".join(dropped_names) if dropped_names else "none"}.
"Out of Town" (a handful of licences, not an actual neighborhood) is the main casualty at
low thresholds — raising the cutoff further starts trimming legitimate small
neighborhoods instead.
        """
    )

st.subheader("Composition matrix")
st.write(
    f"**{composition.shape[0]} areas** kept out of {all_area_counts.shape[0]} total, "
    f"**{composition.shape[1]} business-type features**."
)
st.dataframe(
    composition.round(2).assign(n_businesses=counts_per_area.loc[composition.index]),
    use_container_width=True,
)

st.divider()

st.sidebar.subheader("2. Number of clusters (K)")
max_k = max(2, composition.shape[0] - 1)
k = st.sidebar.slider("K (clusters)", min_value=2, max_value=min(10, max_k), value=4, step=1)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(composition.values)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(composition.values)

plot_df = pd.DataFrame(
    {
        "localarea": composition.index,
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "cluster": cluster_labels.astype(str),
        "n_businesses": counts_per_area.loc[composition.index].values,
    }
)

st.subheader(f"PCA scatter of areas, colored by K-means cluster (K={k})")
fig = px.scatter(
    plot_df,
    x="PC1",
    y="PC2",
    color="cluster",
    size="n_businesses",
    text="localarea",
    hover_data={"localarea": True, "n_businesses": True, "PC1": ":.2f", "PC2": ":.2f", "cluster": True},
)
fig.update_traces(textposition="top center")
fig.update_layout(
    legend_title_text="Cluster",
    xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)",
    yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)",
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"PCA components together explain {pca.explained_variance_ratio_.sum():.1%} of variance "
    f"in the {composition.shape[1]}-dimensional composition matrix."
)
