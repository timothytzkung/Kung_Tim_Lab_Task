
import streamlit as st
import pandas as pd
import json
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

## SETUP
st.title(
    """
IAT 461 - Business Location Explorer - LabTask06 App
    """
)
st.write("Assignment for LabTask06!")

## Data Path
DATA_PATH = "business_locations.geojson"

## CACHING

## LOAD DATA
def load_data(p=DATA_PATH):
    with open(p) as f:
        geo_json = json.load(f)
    rows = []
    for feat in geo_json["features"]:
        props = feat["properties"]
        lon, lat = feat["geometry"]["coordinates"]
        rows.append({**props, "lon": lon, "lat": lat})
    return pd.DataFrame(rows)

df = load_data()

## APP

with st.expander("See Schema"):
    st.write(df.columns)

# Tabs
map_tab, dr_tab, ex_tab = st.tabs(["Map", "Dimensionality Reduction", "Explore"])

# Side bar
st.sidebar.header("1. Select Features")


# Feature Columns
NUMERIC_COLS = ["Floor_Area_sqm", "Daily_Foot_Traffic", "Annual_Revenue_k", "Community_Impact_Score"]
selected_feats = st.sidebar.multiselect("Features to be used in models", options=NUMERIC_COLS, default=NUMERIC_COLS)
with st.expander("See Selected Feats"):
    st.write(selected_feats)

# Warning sign thing
if len(selected_feats) < 2:
    st.warning("Pick at least two features to continue")
    st.stop()

## MODEL
# Scaling
x = df[selected_feats].to_numpy()
x_scaled = StandardScaler().fit_transform(x)
with st.expander("See x_scaled:"):
    st.write(x_scaled)

# Side bar for clustering
st.sidebar.header("2. Clustering")

# Pick algs
alg = st.sidebar.selectbox("Algorithm", ["KMeans", "DBSCAN"])

# Sidebar settings
KMEANS_MIN = 2
KMEANS_MAX = 10
KMEANS_START = 2
KMEANS_STEP = 1

DBSCAN_MIN = 10
DBSCAN_MAX = 30
DBSCAN_START = 10
DBSCAN_STEP = 1

## Add exploratory feature to find optimal eps?
#TODO

# Pass function
if alg == "KMeans":
    k = st.sidebar.slider("clusters", KMEANS_MIN, KMEANS_MAX, KMEANS_START, KMEANS_STEP)
    model = KMeans(n_clusters=k)
    labels = model.fit_predict(x_scaled)
elif alg == "DBSCAN":
    k = st.sidebar.slider("samples", DBSCAN_MIN, DBSCAN_MAX, DBSCAN_START, DBSCAN_STEP)
    eps = st.sidebar.slider("epsilon", 0.05, 0.7, 0.3, 0.05)
    model = DBSCAN(eps=0.3, min_samples=k).fit(x_scaled)
    labels = model.labels_
else:
    pass


# st.write(labels)

# Catch error for undefined labels
try:
    print(labels)
except NameError:
    st.warning("There is no clustering labels")
    st.stop()
else:
    pass

# Create cluster column
df["cluster"] = pd.Categorical(labels.astype(str))
n_clusters_found = df["cluster"].nunique() # If using DBSCAN, this UI will be useful
st.metric("Number of clusters:", n_clusters_found)


with map_tab:
    # Yucky Map
    fig = px.scatter_map(
        df, lat="lat", lon="lon", zoom=10, height=550, color="cluster", map_style="carto-darkmatter"
    )
    st.plotly_chart(fig, width="stretch")


with dr_tab:
    reducer = PCA(n_components=2, random_state=42)
    embedding = reducer.fit_transform(x_scaled)
    df["dim_1"] = embedding[:, 0]
    df["dim_2"] = embedding[:, 1]

    fig_dr = px.scatter(
        df, x="dim_1", y="dim_2",
        color="cluster",
        height=550
    )
    st.plotly_chart(fig_dr, width="stretch")

with ex_tab:
    with st.expander("Look at data:"):
        st.dataframe(df.head(10))

    with st.expander("About the data:"):
        st.dataframe(df.describe())
        f"{len(df)} locations, {df["Neighborhood"].nunique()} neighbourhoods"
# st.map(df)