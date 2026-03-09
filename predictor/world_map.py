import pandas as pd
import folium
import json
from pathlib import Path
from folium.features import GeoJsonTooltip


WORLD_GEOJSON_PATH = (
    Path(__file__).resolve().parent / "data" / "world-countries.json"
)


def create_world_map_with_countries(df: pd.DataFrame) -> str:
   
    if "client_country" not in df.columns:
        raise ValueError("DataFrame must contain a 'client_country' column.")

    country_counts_series = df["client_country"].value_counts()

    # Convert to DataFrame for Choropleth and keep a dict for fast lookup
    country_counts_df = (
        country_counts_series.rename_axis("country")
        .reset_index(name="client_count")
    )

    # Handle a few common name mismatches between our data and the GeoJSON
    country_name_mapping = {
        "United States": "United States of America",
    }
    country_counts_df["country_geo"] = country_counts_df["country"].replace(
        country_name_mapping
    )

    # Dict keyed by the GeoJSON country name
    counts_by_geo_name = dict(
        zip(country_counts_df["country_geo"], country_counts_df["client_count"])
    )

    m = folium.Map(location=[20, 0], zoom_start=2)

    # Load world countries GeoJSON from local file to avoid network dependency
    with open(WORLD_GEOJSON_PATH) as f:
        geo = json.load(f)

    folium.Choropleth(
        geo_data=geo,
        data=country_counts_df,
        columns=["country_geo", "client_count"],
        key_on="feature.properties.name",
        fill_color="YlGnBu",
        fill_opacity=0.7,
        line_opacity=0.2,
        nan_fill_color="white",
        legend_name="Number of vehicle clients by country",
    ).add_to(m)

    def centroid(geometry):
        pts = []
        if geometry["type"] == "Polygon":
            pts = [(lat, lon) for lon, lat in geometry["coordinates"][0]]
        elif geometry["type"] == "MultiPolygon":
            for p in geometry["coordinates"]:
                pts += [(lat, lon) for lon, lat in p[0]]
        if not pts:
            return None
        return [
            sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts),
        ]

    centroids = {}
    for feature in geo["features"]:
        name = feature["properties"]["name"]
        feature["properties"]["client_count"] = counts_by_geo_name.get(name, 0)
        c = centroid(feature["geometry"])
        if c:
            centroids[name] = c

    # Add GeoJSON boundaries with tooltip showing country + count
    folium.GeoJson(
        geo,
        style_function=lambda _: {"color": "black", "weight": 0.5, "fillOpacity": 0},
        tooltip=GeoJsonTooltip(
            fields=["name", "client_count"],
            aliases=["Country", "Clients"],
        ),
    ).add_to(m)

    # Add text markers with the country name and client count, similar to Rwanda map
    for country_name, center in centroids.items():
        count = counts_by_geo_name.get(country_name, 0)
        if count <= 0:
            continue
        folium.Marker(
            center,
            icon=folium.DivIcon(
                html=f"<b>{country_name}<br>{count}</b>",
            ),
        ).add_to(m)

    return m._repr_html_()

