
"""
temperature_grid.py

Build temperature-colored grid polygons from point samples using a recursive
subdivision, compute subregion stats, generate Folium layers (GeoJSON and
optional raster overlay), and produce static Matplotlib plots. Includes an
optional Kriging visualization.

CLI usage (examples):
---------------------
# Basic run with defaults:
python temperature_grid.py

# Specify input CSV and min samples:
python temperature_grid.py --csv InputData.csv --min-samples 5

# Produce Folium map with raster overlay (from a transparent contour image):
python temperature_grid.py --folium-html folium_map.html --with-raster

# Change output CSV and turn on kriging plot:
python temperature_grid.py --output-centroids centroids.csv --kriging
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import geopandas as gpd
from shapely.geometry import Point, Polygon

import folium
import branca.colormap as bcm
from folium.raster_layers import ImageOverlay
# add near the other imports (optional if you want clustering)
from folium.plugins import FastMarkerCluster

from pykrige.ok import OrdinaryKriging


# -------------------------------
# Data container
# -------------------------------
@dataclass
class BuildArtifacts:
    """Holds intermediate artifacts produced by the pipeline."""
    gdf_points: gpd.GeoDataFrame
    subregions: gpd.GeoDataFrame   # polygons with stats
    image_bounds: Optional[List[List[float]]] = None  # for raster overlay [[lat_min, lon_min],[lat_max, lon_max]]
    contour_png: Optional[str] = None


# -------------------------------
# IO / Loading
# -------------------------------
def load_points_csv(path: str) -> gpd.GeoDataFrame:
    """
    Load a CSV with columns for longitude & latitude (various aliases) and temperature
    (supports 'temperature', 'Temperature (°C)', 'degC', 'degF', 'temp', etc.).
    Returns a GeoDataFrame in EPSG:4326 with a numeric 'temperature' in °F.
    """
    print(f"[loader] Loading points from '{path}'..."   )
    df = pd.read_csv(path)

    # Normalize column map (lowercased, stripped) -> original name
    norm = {c.lower().strip(): c for c in df.columns}

    # Find lon/lat
    lon_key = next((k for k in ["longitude", "lon", "lng", "x", "long"] if k in norm), None)
    lat_key = next((k for k in ["latitude", "lat", "y"] if k in norm), None)
    if not lon_key or not lat_key:
        raise ValueError(f"Could not find longitude/latitude columns in {list(df.columns)}")

    lon_col, lat_col = norm[lon_key], norm[lat_key]

    # Find temperature-like column
    temp_candidates = [
        "temperature", "temp", "degf", "deg_f", "°f", "temperature (°f)",
        "degc", "deg_c", "°c", "temperature (°c)"
    ]
    temp_key = next((k for k in temp_candidates if k in norm), None)

    if temp_key is None:
        # If no obvious temp column, try any column containing 'temp'
        temp_key = next((k for k in norm if "temp" in k), None)

    if temp_key is None:
        raise ValueError(f"Could not find a temperature column in {list(df.columns)}")

    temp_col = norm[temp_key]

    # Coerce temperature column to numeric (strip units if present)
    # Example strings: "25.3", "25.3 C", "78 F"
    temp_numeric = (
        df[temp_col]
        .astype(str)
        .str.extract(r"([-+]?\d*\.?\d+)")
        .astype(float)
        [0]
    )

    # Determine units: assume °C if column name contains 'c', else °F
    is_celsius = "c" in temp_key and "degf" not in temp_key
    if is_celsius:
        temp_f = temp_numeric * 9.0 / 5.0 + 32.0
    else:
        temp_f = temp_numeric

    # Build clean frame; drop NaNs in coords or temp
    use = pd.DataFrame({
        "longitude": pd.to_numeric(df[lon_col], errors="coerce"),
        "latitude":  pd.to_numeric(df[lat_col], errors="coerce"),
        "temperature": pd.to_numeric(temp_f, errors="coerce")
    }).dropna(subset=["longitude", "latitude", "temperature"])

    print(f"[loader] Using lon='{lon_col}', lat='{lat_col}', temp='{temp_col}' (stored in °F).")
    print(f"[loader] Loaded {len(use)} valid points from '{path}'.")

    geometry = [Point(xy) for xy in zip(use["longitude"], use["latitude"])]
    gdf = gpd.GeoDataFrame(use[["temperature"]], geometry=geometry, crs="EPSG:4326")
    return gdf


# -------------------------------
# Recursive subdivision
# -------------------------------
def recursive_subdivision_geopandas(
    gdf_points: gpd.GeoDataFrame,
    min_samples: int = 5,
    region_polygon: Optional[Polygon] = None,
    _stages_accumulator: Optional[List[List[Polygon]]] = None
) -> List[Polygon]:
    """
    Recursively subdivide the bounding region into quadrants until each quadrant
    contains at least `min_samples` points. Returns final list of polygons.
    """
    if region_polygon is not None:
        x_min, y_min, x_max, y_max = region_polygon.bounds
    else:
        x_min, y_min, x_max, y_max = gdf_points.total_bounds
        region_polygon = Polygon([
            (x_min, y_min), (x_max, y_min),
            (x_max, y_max), (x_min, y_max),
            (x_min, y_min)
        ])

    if len(gdf_points) < min_samples:
        return [region_polygon]

    mid_x = (x_min + x_max) / 2.0
    mid_y = (y_min + y_max) / 2.0

    subrects = [
        Polygon([(x_min, y_min), (mid_x, y_min), (mid_x, mid_y), (x_min, mid_y), (x_min, y_min)]),
        Polygon([(mid_x, y_min), (x_max, y_min), (x_max, mid_y), (mid_x, mid_y), (mid_x, y_min)]),
        Polygon([(x_min, mid_y), (mid_x, mid_y), (mid_x, y_max), (x_min, y_max), (x_min, mid_y)]),
        Polygon([(mid_x, mid_y), (x_max, mid_y), (x_max, y_max), (mid_x, y_max), (mid_x, mid_y)])
    ]

    if _stages_accumulator is not None:
        _stages_accumulator.append(subrects)

    subregions: List[Polygon] = []
    sub_gdfs = []
    all_ok = True

    for subrect in subrects:
        sub_gdf = gdf_points[gdf_points.intersects(subrect)]
        sub_gdfs.append((sub_gdf, subrect))
        if len(sub_gdf) < min_samples:
            all_ok = False
            break

    if all_ok:
        for sgdf, srect in sub_gdfs:
            if not sgdf.empty:
                subregions.extend(recursive_subdivision_geopandas(sgdf, min_samples, srect, _stages_accumulator))
    else:
        subregions.append(region_polygon)

    return subregions


def build_subregion_gdf(gdf_points: gpd.GeoDataFrame, min_samples: int = 5) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame of final subregion polygons in EPSG:4326."""
    final_polys = recursive_subdivision_geopandas(gdf_points, min_samples=min_samples)
    subregion_gdf = gpd.GeoDataFrame({'geometry': final_polys}, crs="EPSG:4326")
    subregion_gdf = subregion_gdf[subregion_gdf.is_valid & ~subregion_gdf.is_empty]
    return subregion_gdf


# -------------------------------
# Stats per subregion
# -------------------------------
def compute_average_temperature(
    subregion_gdf: gpd.GeoDataFrame,
    sample_points_gdf: gpd.GeoDataFrame,
    predicate: str = "intersects"
) -> gpd.GeoDataFrame:
    """
    Compute avg temperature, std dev, and sample count for each subregion polygon.
    Joins points to polygons using `predicate` (default: intersects).
    """
    if subregion_gdf.crs != sample_points_gdf.crs:
        sample_points_gdf = sample_points_gdf.to_crs(subregion_gdf.crs)

    joined = gpd.sjoin(sample_points_gdf, subregion_gdf, how='inner', predicate=predicate)

    # Aggregate per subregion
    stats = joined.groupby('index_right').agg(
        avg_temperature=('temperature', 'mean'),
        std_temperature=('temperature', 'std'),     # ← NEW
        sample_count=('temperature', 'count')
    )

    out = subregion_gdf.copy()
    out['avg_temperature'] = out.index.map(stats['avg_temperature'])
    out['std_temperature'] = out.index.map(stats['std_temperature'])   # ← NEW
    out['sample_count'] = out.index.map(stats['sample_count'])

    # Optional: if a region has only 1 sample, std is NaN; you can set it to 0 if you prefer:
    # out['std_temperature'] = out['std_temperature'].fillna(0.0)

    return out
   


# -------------------------------
# Geometry helpers
# -------------------------------
def add_centroids_wgs84(subregion_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute centroids in a local projected CRS (UTM) and return as EPSG:4326
    in a new column 'centroid'. Avoids geographic CRS centroid warning.
    """
    # Ensure WGS84 for bounds/zone calc
    gdf_wgs = subregion_gdf.to_crs(4326)

    # Pick UTM zone from dataset center
    minx, miny, maxx, maxy = gdf_wgs.total_bounds
    lon_c = (minx + maxx) / 2.0
    lat_c = (miny + maxy) / 2.0
    zone = int((lon_c + 180) // 6) + 1
    epsg_proj = 32600 + zone if lat_c >= 0 else 32700 + zone  # UTM north/south

    # Centroids in meters → back to WGS84
    centroids_proj = gdf_wgs.to_crs(epsg_proj).centroid
    centroids_wgs = centroids_proj.to_crs(4326)

    out = gdf_wgs.copy()
    out['centroid'] = centroids_wgs
    return out

    


# -------------------------------
# Folium layers
# -------------------------------
def _safe_linear_colormap(vmin: float, vmax: float, reverse: bool = False):
    """
    Always return a blue→red LinearColormap (version-proof).
    Set reverse=True to make red=min, blue=max.
    """
    colors = [
        "#3b4cc0",  # blue
        "#6f8fd6",
        "#98b6e9",
        "#f6c2a2",
        "#e67a5b",
        "#b40426",  # red
    ]
    if reverse:
        colors = list(reversed(colors))
    return bcm.LinearColormap(colors=colors, vmin=vmin, vmax=vmax)





def gdf_to_geojson(gdf: gpd.GeoDataFrame) -> dict:
    """Return a GeoJSON dict from a GeoDataFrame (keeps properties) but
    sanitizes non-serializable columns like a Shapely 'centroid'."""
    # Ensure WGS84
    if gdf.crs is None or gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    gdf2 = gdf.copy()

    # Convert centroid (Point) → numeric columns so it's JSON-serializable
    if "centroid" in gdf2.columns:
        gdf2["centroid_lon"] = gdf2["centroid"].apply(lambda p: p.x if p is not None else None)
        gdf2["centroid_lat"] = gdf2["centroid"].apply(lambda p: p.y if p is not None else None)
        gdf2 = gdf2.drop(columns=["centroid"])

    # If you have other object-typed columns that might hold complex Python objects,
    # you can drop them here as needed.

    return json.loads(gdf2.to_json())



def create_folium_map_with_layers(
    subregion_gdf: gpd.GeoDataFrame,
    image_file: Optional[str] = None,
    image_bounds: Optional[List[List[float]]] = None,
    add_centroids: bool = True,
    output_html: str = "folium_geojson_only.html",
    base_tiles: str = "CartoDB positron",
    points_gdf: Optional[gpd.GeoDataFrame] = None,
    points_as_cluster: bool = False,
    point_radius: int = 2
) -> folium.Map:
    # Ensure EPSG:4326
    if subregion_gdf.crs is None or subregion_gdf.crs.to_string() != "EPSG:4326":
        subregion_gdf = subregion_gdf.to_crs(4326)

    # Map center/bounds
    minx, miny, maxx, maxy = subregion_gdf.total_bounds
    center_lat = (miny + maxy) / 2.0
    center_lon = (minx + maxx) / 2.0

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=base_tiles)

    # Color scale (robust even if all values are equal or missing)
    
    vals = subregion_gdf['avg_temperature'].astype(float).dropna()
    if vals.empty:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmin == vmax:
            pad = 0.5 if vmin == 0.0 else 0.05 * abs(vmin)
            vmin -= pad
            vmax += pad

    # Blue=min, red=max  ✅
    cmap = _safe_linear_colormap(vmin, vmax)          # reverse=False by default
    # If you ever want the opposite: cmap = _safe_linear_colormap(vmin, vmax, reverse=True)

    cmap.caption = "Avg Temperature (°F)"
    cmap.add_to(m)


    # ✅ Use sanitized GeoJSON export
    gj = gdf_to_geojson(subregion_gdf)

    def style_function(feature):
        temp = feature["properties"].get("avg_temperature", None)
        fill_color = cmap(temp) if temp is not None else "#cccccc"
        return {"fillColor": fill_color, "color": "black", "weight": 0.5, "fillOpacity": 0.6, "stroke": False}

    tooltip = folium.GeoJsonTooltip(
    fields=["avg_temperature", "std_temperature", "sample_count"],
    aliases=["Avg Temp (°F):", "Std Dev (°F):", "Samples:"],
    localize=True,
    sticky=True
)

    poly_fg = folium.FeatureGroup(name="Temperature Grid (GeoJSON)", show=True)
    folium.GeoJson(
        data=gj,
        name="Temperature Polygons",
        style_function=style_function,
        tooltip=tooltip,
        popup=folium.GeoJsonPopup(
        fields=["avg_temperature", "std_temperature", "sample_count"],
        aliases=["Avg Temp (°F):", "Std Dev (°F):", "Samples:"],
        localize=True,
        labels=True,
        max_width=320
     ),
        highlight_function=lambda f: {"weight": 2, "color": "#333333"}
    ).add_to(poly_fg)
    poly_fg.add_to(m)

    # Optional centroid markers (uses the real Shapely centroids from the DF, which we kept)
    if add_centroids and "centroid" in subregion_gdf.columns:
        cen_fg = folium.FeatureGroup(name="Centroids", show=False)
        for _, row in subregion_gdf.dropna(subset=["centroid"]).iterrows():
            c = row["centroid"]
            t = row.get("avg_temperature", None)
            sc = row.get("sample_count", None)
            folium.CircleMarker(
                location=[c.y, c.x],
                radius=3,
                weight=1,
                fill=True,
                fill_opacity=0.9,
                popup=folium.Popup(
                    f"<b>Centroid</b><br>Lat: {c.y:.6f}, Lon: {c.x:.6f}"
                    + (f"<br>Avg Temp: {t:.2f} °F" if t is not None else "")
                    + (f"<br>Samples: {sc}" if sc is not None else ""),
                    max_width=250
                )
            ).add_to(cen_fg)
        cen_fg.add_to(m)

    if image_file and image_bounds:
        ImageOverlay(
            name="Temperature Contours (raster)",
            image=image_file,
            bounds=image_bounds,
            opacity=0.5,
            interactive=True,
            cross_origin=False,
            zindex=1
        ).add_to(m)


    # ---------- NEW: Raw points layer ----------
    if points_gdf is not None and not points_gdf.empty:
        # Ensure WGS84
        pts = points_gdf
        if pts.crs is None or pts.crs.to_string() != "EPSG:4326":
            pts = pts.to_crs(4326)



        if points_as_cluster:
            cluster_fg = folium.FeatureGroup(name="Raw Samples (cluster)", show=True, control=True)
            coords = [(geom.y, geom.x) for geom in pts.geometry if geom is not None]
            FastMarkerCluster(data=coords).add_to(cluster_fg)
            cluster_fg.add_to(m)
            print(f"✅ Added {len(coords)} raw sample points to layer 'Raw Samples (cluster)'.")
        else:
            pts_fg = folium.FeatureGroup(name="Raw Samples (points)", show=True, control=True)
            n = 0
            for _, row in pts.iterrows():
                geom = row.geometry
                if geom is None:
                    continue
                lat, lon = geom.y, geom.x
                t = row.get("temperature", None)
                folium.CircleMarker(
                    location=(lat, lon),
                    radius=max(4, point_radius),      # make them easy to see
                    color="black",
                    weight=1,
                    fill=True,
                    fill_color="black",
                    fill_opacity=1.0,
                    tooltip=(f"Temp (°F): {float(t):.2f}" if pd.notna(t) else None),
                ).add_to(pts_fg)
                n += 1
            pts_fg.add_to(m)
            print(f"✅ Added {n} raw sample points to layer 'Raw Samples (points)'.")
    # -------------------------------------------



    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[miny, minx], [maxy, maxx]])
    m.save(output_html)
    print(f"✅ Folium map saved to: {output_html}")
    return m








# -------------------------------
# Static plotting
# -------------------------------
def plot_temperature_colored_subregions(subregion_gdf: gpd.GeoDataFrame, title: str = 'Temperature-Colored Subregions') -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.cm.get_cmap("coolwarm")

    subregion_gdf.plot(
        ax=ax,
        column='avg_temperature',
        cmap=cmap,
        linewidth=0.5,
        edgecolor='black',
        legend=True,
        legend_kwds={'label': "Avg Temperature (°F)"}
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    plt.show()


def plot_rectangles_and_contours(subregion_gdf: gpd.GeoDataFrame, title: str = 'Temperature Rectangles + Contour Overlay') -> None:
    valid = subregion_gdf.dropna(subset=['avg_temperature'])
    xs = valid['centroid'].apply(lambda p: p.x).values
    ys = valid['centroid'].apply(lambda p: p.y).values
    temps = valid['avg_temperature'].values

    xi = np.linspace(xs.min(), xs.max(), 300)
    yi = np.linspace(ys.min(), ys.max(), 300)
    xi, yi = np.meshgrid(xi, yi)

    from scipy.interpolate import griddata
    zi = griddata((xs, ys), temps, (xi, yi), method='linear')

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.cm.get_cmap("coolwarm")

    subregion_gdf.plot(
        ax=ax,
        column='avg_temperature',
        cmap=cmap,
        linewidth=0.5,
        edgecolor='black'
    )
    contour = ax.contourf(xi, yi, zi, levels=20, cmap=cmap, alpha=0.5)
    plt.colorbar(contour, ax=ax, label='Avg Temperature (°F)')

    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal', adjustable='box')
    plt.show()


def plot_contour_only(subregion_gdf: gpd.GeoDataFrame, title: str = 'Temperature Contour Map (No Grid)') -> None:
    valid = subregion_gdf.dropna(subset=['avg_temperature'])
    xs = valid['centroid'].apply(lambda p: p.x).values
    ys = valid['centroid'].apply(lambda p: p.y).values
    temps = valid['avg_temperature'].values

    xi = np.linspace(xs.min(), xs.max(), 300)
    yi = np.linspace(ys.min(), ys.max(), 300)
    xi, yi = np.meshgrid(xi, yi)

    from scipy.interpolate import griddata
    zi = griddata((xs, ys), temps, (xi, yi), method='linear')

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.cm.get_cmap("coolwarm")

    contour = ax.contourf(xi, yi, zi, levels=20, cmap=cmap)
    plt.colorbar(contour, ax=ax, label="Avg Temperature (°F)")

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    plt.show()


# -------------------------------
# Raster export for Folium overlay
# -------------------------------
def save_contour_image(subregion_gdf: gpd.GeoDataFrame, image_filename: str = 'contour_overlay.png') -> List[List[float]]:
    """
    Save a transparent PNG contour image from centroid-sampled temperatures and
    return geographic bounds suitable for Folium ImageOverlay.
    """
    valid = subregion_gdf.dropna(subset=['avg_temperature'])
    xs = valid['centroid'].apply(lambda p: p.x).values
    ys = valid['centroid'].apply(lambda p: p.y).values
    temps = valid['avg_temperature'].values

    xi = np.linspace(xs.min(), xs.max(), 300)
    yi = np.linspace(ys.min(), ys.max(), 300)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    from scipy.interpolate import griddata
    zi = griddata((xs, ys), temps, (xi_grid, yi_grid), method='linear')

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    cmap = plt.cm.get_cmap("coolwarm")

    ax.contourf(xi_grid, yi_grid, zi, levels=20, cmap=cmap)
    ax.axis('off')

    plt.savefig(image_filename, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

    lat_min, lat_max = yi.min(), yi.max()
    lon_min, lon_max = xi.min(), xi.max()
    return [[lat_min, lon_min], [lat_max, lon_max]]


def create_folium_map_with_contour(
    image_file: str,
    image_bounds: List[List[float]],
    output_html: str = "folium_map.html",
    base_tiles: str = "OpenStreetMap"
) -> folium.Map:
    """
    Create a simple Folium map with just the raster (transparent contour) overlay.
    """
    center_lat = (image_bounds[0][0] + image_bounds[1][0]) / 2.0
    center_lon = (image_bounds[0][1] + image_bounds[1][1]) / 2.0

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=base_tiles)

    ImageOverlay(
        name='Temperature Contours',
        image=image_file,
        bounds=image_bounds,
        opacity=0.6,
        interactive=True,
        cross_origin=False,
        zindex=1
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_html)
    print(f"✅ Folium map saved to: {output_html}")
    return m


# -------------------------------
# Kriging visualization
# -------------------------------
def plot_kriging_contour(subregion_gdf: gpd.GeoDataFrame, title: str = 'Temperature Contour via Kriging') -> None:
    valid = subregion_gdf.dropna(subset=['avg_temperature'])
    xs = valid['centroid'].apply(lambda p: p.x).values
    ys = valid['centroid'].apply(lambda p: p.y).values
    temps = valid['avg_temperature'].values

    grid_x = np.linspace(xs.min(), xs.max(), 300)
    grid_y = np.linspace(ys.min(), ys.max(), 300)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    OK = OrdinaryKriging(xs, ys, temps, variogram_model='linear', verbose=False, enable_plotting=False)
    z_kriged, ss = OK.execute('grid', grid_x, grid_y)

    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.cm.get_cmap("coolwarm")
    contour = ax.contourf(grid_xx, grid_yy, z_kriged, levels=20, cmap=cmap)

    plt.colorbar(contour, ax=ax, label="Avg Temperature (°F)")
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    plt.show()


# -------------------------------
# Pipeline (build + folium + outputs)
# -------------------------------
def build_pipeline(
    csv_path: str = "InputData.csv",
    min_samples: int = 5,
    join_predicate: str = "intersects",
    output_centroids_csv: str = "output_centroids.csv",
    folium_html: str = "folium_geojson_only.html",
    with_raster: bool = False,
    raster_png: str = "contour_overlay.png",
    raster_html: str = "folium_map.html",
    do_static_plots: bool = True,
    do_kriging_plot: bool = False
) -> BuildArtifacts:
    """
    Run the full pipeline and return artifacts. Writes outputs as requested.
    """
    # Load points
    gdf_points = load_points_csv(csv_path)

    # Build subregions and stats
    subregions = build_subregion_gdf(gdf_points, min_samples=min_samples)
    subregions = compute_average_temperature(subregions, gdf_points, predicate=join_predicate)
    subregions = add_centroids_wgs84(subregions)

    # Save centroids CSV
    out = subregions[['centroid', 'avg_temperature']].dropna().copy()
    out['longitude'] = out['centroid'].apply(lambda p: p.x)
    out['latitude']  = out['centroid'].apply(lambda p: p.y)
    out[['longitude', 'latitude', 'avg_temperature']].to_csv(output_centroids_csv, index=False)
    print(f"✅ Wrote centroids CSV: {output_centroids_csv}")

    # Folium vector map (GeoJSON layer)
    create_folium_map_with_layers(
    subregion_gdf=subregions,
    output_html=folium_html,
    points_gdf=gdf_points,          # NEW: adds the “Raw Samples (points)” layer
    points_as_cluster=False,         # set True if you have a *lot* of points
    point_radius=2
  )
    print(f"✅ Wrote Folium map (GeoJSON): {folium_html}")

    image_bounds = None
    if with_raster:
        image_bounds = save_contour_image(subregions, image_filename=raster_png)
        create_folium_map_with_contour(raster_png, image_bounds, output_html=raster_html)
        print(f"✅ Wrote Folium raster map: {raster_html}")

    # Optional static plots
    if do_static_plots:
        plot_temperature_colored_subregions(subregions)
        plot_rectangles_and_contours(subregions)
        plot_contour_only(subregions)

    if do_kriging_plot:
        plot_kriging_contour(subregions)

    return BuildArtifacts(
        gdf_points=gdf_points,
        subregions=subregions,
        image_bounds=image_bounds,
        contour_png=(raster_png if with_raster else None)
    )


# -------------------------------
# CLI
# -------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build temperature grid and Folium layers from point data.")
    p.add_argument("--csv", dest="csv_path", default="InputData.csv", help="Path to input CSV (with longitude, latitude, temperature).")
    p.add_argument("--min-samples", type=int, default=5, help="Minimum samples required to subdivide a region.")
    p.add_argument("--predicate", default="intersects", choices=["intersects", "within"], help="Spatial join predicate for points→polygons.")
    p.add_argument("--output-centroids", default="output_centroids.csv", help="Output CSV for centroid long/lat and avg temperature.")
    p.add_argument("--folium-html", default="folium_geojson_only.html", help="Output HTML for Folium GeoJSON map.")
    p.add_argument("--with-raster", action="store_true", help="Also render transparent contour PNG and a raster Folium map.")
    p.add_argument("--raster-png", default="contour_overlay.png", help="Filename for saved transparent contour PNG.")
    p.add_argument("--raster-html", default="folium_map.html", help="Output HTML for Folium raster map.")
    p.add_argument("--no-static-plots", action="store_true", help="Disable Matplotlib static plots.")
    p.add_argument("--kriging", action="store_true", help="Show Kriging contour plot at the end.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    build_pipeline(
        csv_path=args.csv_path,
        min_samples=args.min_samples,
        join_predicate=args.predicate,
        output_centroids_csv=args.output_centroids,
        folium_html=args.folium_html,
        with_raster=args.with_raster,
        raster_png=args.raster_png,
        raster_html=args.raster_html,
        do_static_plots=not args.no_static_plots,
        do_kriging_plot=args.kriging,
    )


if __name__ == "__main__":
    main()
