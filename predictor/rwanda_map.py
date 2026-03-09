import folium, json
from pathlib import Path
from folium.features import GeoJsonTooltip

DISTRICTS_GEOJSON_PATH = Path(__file__).resolve().parent/"data"/"rwa_adm2_simplified.geojson"

def create_rwanda_map_with_districts(df):
    with open(DISTRICTS_GEOJSON_PATH) as f: geo=json.load(f)
    counts=df['district'].value_counts().to_dict()

    def centroid(g):
        pts=[]
        if g["type"]=="Polygon": pts=[(lat,lon) for lon,lat in g["coordinates"][0]]
        if g["type"]=="MultiPolygon":
            for p in g["coordinates"]: pts+=[(lat,lon) for lon,lat in p[0]]
        return [sum(p[0] for p in pts)/len(pts),sum(p[1] for p in pts)/len(pts)] if pts else None

    centroids={}
    for f in geo["features"]:
        n=f["properties"]["shapeName"]
        f["properties"]["client_count"]=counts.get(n,0)
        c=centroid(f["geometry"])
        if c: centroids[n]=c

    m=folium.Map(location=[-1.95,29.87],zoom_start=8)

    folium.Choropleth(
        geo_data=geo,data=[[k,v] for k,v in counts.items()],
        columns=["district","client_count"],
        key_on="feature.properties.shapeName",
        fill_color="YlOrRd",legend_name="Vehicle clients by district"
    ).add_to(m)

    folium.GeoJson(
        geo,style_function=lambda _:{"color":"black","weight":1,"fillOpacity":0},
        tooltip=GeoJsonTooltip(fields=["shapeName","client_count"],aliases=["District","Clients"])
    ).add_to(m)

    for d,c in centroids.items():
        folium.Marker(c,icon=folium.DivIcon(html=f"<b>{d}<br>{counts.get(d,0)}</b>")).add_to(m)

    return m._repr_html_()