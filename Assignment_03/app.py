import json

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

    centroids = (
        filtered.groupby("localarea")[["geo_point_2d.lat", "geo_point_2d.lon"]]
        .mean()
        .rename(columns={"geo_point_2d.lat": "lat", "geo_point_2d.lon": "lon"})
        .loc[composition.index]
    )
    return composition, counts_per_area, centroids


# Fixed categorical order (colorblind-safe, validated) — same hue always means
# the same cluster id across both the PCA scatter and the map.
CLUSTER_PALETTE = [
    "#2a78d6",  # blue
    "#ffb300",  # orange
    "#27e7a4",  # aqua
    "#f4ff20",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]


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

composition, counts_per_area, centroids = build_composition_matrix(df, min_businesses)
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
max_k = min(len(CLUSTER_PALETTE), composition.shape[0] - 1)
k = st.sidebar.slider("K (clusters)", min_value=2, max_value=max(2, max_k), value=6, step=1)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(composition.values)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(composition.values)

cluster_str = cluster_labels.astype(str)
color_map = {str(i): CLUSTER_PALETTE[i] for i in range(k)}

plot_df = pd.DataFrame(
    {
        "localarea": composition.index,
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        "cluster": cluster_str,
        "n_businesses": counts_per_area.loc[composition.index].values,
        "lat": centroids.loc[composition.index, "lat"].values,
        "lon": centroids.loc[composition.index, "lon"].values,
    }
)

st.subheader(f"PCA scatter of areas, colored by K-means cluster (K={k})")
fig = px.scatter(
    plot_df,
    x="PC1",
    y="PC2",
    color="cluster",
    color_discrete_map=color_map,
    category_orders={"cluster": [str(i) for i in range(k)]},
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

st.divider()

st.subheader(f"Geographic view: area centroids colored by cluster (K={k})")
map_fig = px.scatter_map(
    plot_df,
    lat="lat",
    lon="lon",
    color="cluster",
    color_discrete_map=color_map,
    category_orders={"cluster": [str(i) for i in range(k)]},
    size="n_businesses",
    hover_name="localarea",
    hover_data={"n_businesses": True, "cluster": True, "lat": False, "lon": False},
    zoom=10.5,
    map_style="open-street-map",
)
map_fig.update_layout(legend_title_text="Cluster", margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(map_fig, use_container_width=True)
st.caption("Each point is an area's business centroid (mean lat/lon of its licences); size = business count.")

st.divider()

st.subheader(f"Cluster membership (K={k})")
st.caption(
    "Areas grouped by cluster, plus each cluster's top business types by average share."
)

membership = plot_df[["localarea", "cluster", "n_businesses"]].copy()
for cid in range(k):
    members = membership[membership["cluster"] == str(cid)].sort_values("n_businesses", ascending=False)
    top_types = composition.loc[members["localarea"]].mean(axis=0).sort_values(ascending=False).head(5)

    st.markdown(f"**Cluster {cid}** &nbsp; ({len(members)} area{'s' if len(members) != 1 else ''})")
    col1, col2 = st.columns([2, 3])
    with col1:
        st.dataframe(
            members.rename(columns={"localarea": "Area", "n_businesses": "# Businesses"})[
                ["Area", "# Businesses"]
            ],
            hide_index=True,
            use_container_width=True,
        )
    with col2:
        st.dataframe(
            top_types.rename("Avg. % of area's businesses").round(2),
            use_container_width=True,
        )
    
st.markdown("""
            ## B4 - Write Up
Regarding meaningful groupings, I observe that Cluster 5 (singleton containing Downtown) is rather accurate. Downtown is Vancouver's core financial/legal district, so a profile topped by legal, business-support, and consulting services standing apart from every residential neighbourhood makes sense to me. In the same way, Cluster 0's wholesale/non-food signature fits the local areas well. Mount Pleasant, Grandview-Woodland, and Strathcona are old warehouse and light-industrial eastside areas that are now mixed with some rental apartments, and Marpole has its own industrial strip along the Fraser. Cluster 4 contains Sunset and Hastings-Sunrise, which are similar working-class East Van neighbourhoods with comparable commercial strips. Cluster 2's sky-high long-term rental share fits Kitsilano and the West End very well. The West End is generally Vancouver's more densely populated area, with large rental apartment blocks

What is surprising to me, however, is that Shaughnessy is landing in the more rental-heavy Cluster 2. Shaughnessy is known to be one of the more affluent/wealthy areas with lots of heritage mansions and large properties, which is very far from an apartment-rental market. I argue this may be due to a small-sample artifact. I observe that Shaughnessy has only 355 registered businesses (the smallest of any area in the dataset), so a handful of rental-related or property-management listings can potentially swing its percentage profile enough to statistically resemble West-End/Kits, even though the underlying neighbourhood is very different. I also found Fairview anchoring the more "health-care" cluster of cluster 3 to be an interesting observation. Broadway in Fairview is a major employment hub (as demonstrated with their 10k+ registered businesses in the dataset, second only to Downtown), but it's also a hub that's built around health care facilities such as Vancouver General Hospital, BC Cancer Agency, and the BC Centre for Disease Control. As such, the fellow areas in cluster 3 (Renfrew, Victoria-Fraserview, Killarney, Dunbar, and Oakridge) don't have a comparable health care corridor. I argue that they likely land in the same cluster because they're residential areas with limited business diversity, where healthcare clinics and offices make up an outsized share of what commercial activity exists.

Ultimately, because clustering here in done on relative composition rather than absolute counts, small-business-count neighbourhoods (such as Shaughnessy and South Cambie) are just as influential as huge ones (Fairview and Downtown). I observe this is what lets a mansion district like Shaunessy get grouped with West Can's densest rental zones, and a hospital district gets grouped with more quiet family suburbs (Renfrew, Killarney, etc.). 

""")
