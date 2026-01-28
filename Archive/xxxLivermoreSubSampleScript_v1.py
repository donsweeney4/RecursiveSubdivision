import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import numpy as np
import random
import math

import folium
from folium.raster_layers import ImageOverlay

from pykrige.ok import OrdinaryKriging

"""

Loads real data from inputData.csv.

Applies the recursive spatial subdivision algorithm.
Computes the average temperature in each final subregion.
Saves a CSV file with the centroid coordinates and average temperature of each subregion.
Plots the subregions with color corresponding to temperature (blue = cool, red = hot).

Also possible:

Temperature thresholds on the color bar,
Export to GeoJSON or Shapefile,
A dynamic map with folium.

Files Created

output_centroids.csv — contains the centroid longitude, latitude, and average temperature for each subregion.
Temperature plot — visually shows rectangular subregions colored by average temperature (blue to red scale).

"""

# Load real CSV data
df = pd.read_csv("InputData.csv")
df = df.dropna(subset=['longitude', 'latitude', 'temperature'])

# Convert to GeoDataFrame
geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
gdf = gpd.GeoDataFrame(df[['temperature']], geometry=geometry, crs="EPSG:4326")

# Recursive subdivision function
subdivision_stages = []

def recursive_subdivision_geopandas(gdf, min_samples=5, region_polygon=None, level=0):
    if region_polygon is not None:
        x_min, y_min, x_max, y_max = region_polygon.bounds
    else:
        bounds = gdf.total_bounds
        x_min, y_min, x_max, y_max = bounds
        region_polygon = Polygon([
            (x_min, y_min), (x_max, y_min),
            (x_max, y_max), (x_min, y_max),
            (x_min, y_min)
        ])

    if len(gdf) < min_samples:
        return [region_polygon]

    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2

    subrects = [
        Polygon([(x_min, y_min), (mid_x, y_min), (mid_x, mid_y), (x_min, mid_y), (x_min, y_min)]),
        Polygon([(mid_x, y_min), (x_max, y_min), (x_max, mid_y), (mid_x, mid_y), (mid_x, y_min)]),
        Polygon([(x_min, mid_y), (mid_x, mid_y), (mid_x, y_max), (x_min, y_max), (x_min, mid_y)]),
        Polygon([(mid_x, mid_y), (x_max, mid_y), (x_max, y_max), (mid_x, y_max), (mid_x, mid_y)])
    ]

    subregions = []
    sub_gdfs = []
    all_subregions_ok = True
    for subrect in subrects:
        sub_gdf = gdf[gdf.intersects(subrect)]
        sub_gdfs.append((sub_gdf, subrect))
        if len(sub_gdf) < min_samples:
            all_subregions_ok = False
            break

    subdivision_stages.append(subrects)

    if all_subregions_ok:
        for sub_gdf, subrect in sub_gdfs:
            if not sub_gdf.empty:
                subregions.extend(recursive_subdivision_geopandas(sub_gdf, min_samples, subrect, level+1))
    else:
        subregions.append(region_polygon)

    return subregions

# Apply recursive subdivision
final_subregions = recursive_subdivision_geopandas(gdf, min_samples=5)

# Create subregion GeoDataFrame
subregion_gdf = gpd.GeoDataFrame({'geometry': final_subregions}, crs="EPSG:4326")
subregion_gdf = subregion_gdf[subregion_gdf.is_valid & ~subregion_gdf.is_empty]

# Compute average temperature per subregion
def compute_average_temperature(subregion_gdf, sample_points_gdf):
    if subregion_gdf.crs != sample_points_gdf.crs:
        sample_points_gdf = sample_points_gdf.to_crs(subregion_gdf.crs)
    joined = gpd.sjoin(sample_points_gdf, subregion_gdf, how='inner', predicate='within')
    stats = joined.groupby('index_right').agg(
        avg_temperature=('temperature', 'mean'),
        sample_count=('temperature', 'count')
    )
    subregion_gdf = subregion_gdf.copy()
    subregion_gdf['avg_temperature'] = subregion_gdf.index.map(stats['avg_temperature'])
    subregion_gdf['sample_count'] = subregion_gdf.index.map(stats['sample_count'])
    subregion_gdf['centroid'] = subregion_gdf.geometry.centroid
    return subregion_gdf




subregion_gdf = compute_average_temperature(subregion_gdf, gdf)

# Save centroid coordinates and avg temperature to CSV
output_df = subregion_gdf[['centroid', 'avg_temperature']].dropna()
output_df['longitude'] = output_df['centroid'].apply(lambda p: p.x)
output_df['latitude'] = output_df['centroid'].apply(lambda p: p.y)
output_df[['longitude', 'latitude', 'avg_temperature']].to_csv("output_centroids.csv", index=False)

# Plot with color map
def plot_temperature_colored_subregions(subregion_gdf, title='Temperature-Colored Subregions'):
    fig, ax = plt.subplots(figsize=(10, 10))

    temp_values = subregion_gdf['avg_temperature'].fillna(0)
    norm = plt.Normalize(temp_values.min(), temp_values.max())
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

plot_temperature_colored_subregions(subregion_gdf)

# Combined plot: colored rectangles + contour map
def plot_rectangles_and_contours(subregion_gdf, title='Temperature Rectangles + Contour Overlay'):
    # Drop invalid data
    valid_data = subregion_gdf.dropna(subset=['avg_temperature'])

    # Extract centroid coordinates and temperature values
    xs = valid_data['centroid'].apply(lambda p: p.x).values
    ys = valid_data['centroid'].apply(lambda p: p.y).values
    temps = valid_data['avg_temperature'].values

    # Create a grid for interpolation
    xi = np.linspace(xs.min(), xs.max(), 300)
    yi = np.linspace(ys.min(), ys.max(), 300)
    xi, yi = np.meshgrid(xi, yi)

    from scipy.interpolate import griddata
    zi = griddata((xs, ys), temps, (xi, yi), method='linear')

    # Start plotting
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the temperature-colored rectangles
    temp_values = subregion_gdf['avg_temperature'].fillna(0)
    cmap = plt.cm.get_cmap("coolwarm")
    norm = plt.Normalize(temp_values.min(), temp_values.max())

    subregion_gdf.plot(
        ax=ax,
        column='avg_temperature',
        cmap=cmap,
        linewidth=0.5,
        edgecolor='black'
    )

    # Overlay the contour map
    contour = ax.contourf(xi, yi, zi, levels=20, cmap=cmap, alpha=0.5)

    # Add color bar for the contours
    plt.colorbar(contour, ax=ax, label='Avg Temperature (°F)')

    # Labels and layout
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal', adjustable='box')
    plt.show()

plot_rectangles_and_contours(subregion_gdf)

# Contour map only — no rectangle overlay
def plot_contour_only(subregion_gdf, title='Temperature Contour Map (No Grid)'):
    # Filter out missing temperature values
    valid_data = subregion_gdf.dropna(subset=['avg_temperature'])

    # Extract coordinates and temperature values
    xs = valid_data['centroid'].apply(lambda p: p.x).values
    ys = valid_data['centroid'].apply(lambda p: p.y).values
    temps = valid_data['avg_temperature'].values

    # Create interpolation grid
    xi = np.linspace(xs.min(), xs.max(), 300)
    yi = np.linspace(ys.min(), ys.max(), 300)
    xi, yi = np.meshgrid(xi, yi)

    from scipy.interpolate import griddata
    zi = griddata((xs, ys), temps, (xi, yi), method='linear')

    # Plot contour map
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.cm.get_cmap("coolwarm")

    contour = ax.contourf(xi, yi, zi, levels=20, cmap=cmap)
    plt.colorbar(contour, ax=ax, label="Avg Temperature (°F)")

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    plt.show()

plot_contour_only(subregion_gdf)


def save_contour_image(subregion_gdf, image_filename='contour_overlay.png'):
    valid_data = subregion_gdf.dropna(subset=['avg_temperature'])

    xs = valid_data['centroid'].apply(lambda p: p.x).values
    ys = valid_data['centroid'].apply(lambda p: p.y).values
    temps = valid_data['avg_temperature'].values

    xi = np.linspace(xs.min(), xs.max(), 300)
    yi = np.linspace(ys.min(), ys.max(), 300)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    from scipy.interpolate import griddata
    zi = griddata((xs, ys), temps, (xi_grid, yi_grid), method='linear')

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    cmap = plt.cm.get_cmap("coolwarm")

    # Plot contour with no axes or labels
    ax.contourf(xi_grid, yi_grid, zi, levels=20, cmap=cmap)
    ax.axis('off')

    # Save with transparent background and tight bounds
    plt.savefig(image_filename, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

    # Return geographic bounds for the image overlay
    lat_min, lat_max = yi.min(), yi.max()
    lon_min, lon_max = xi.min(), xi.max()
    return [[lat_min, lon_min], [lat_max, lon_max]]

import folium
from folium.raster_layers import ImageOverlay

def create_folium_map_with_contour(image_file, image_bounds, output_html="folium_map.html"):
    # Center of the map
    center_lat = (image_bounds[0][0] + image_bounds[1][0]) / 2
    center_lon = (image_bounds[0][1] + image_bounds[1][1]) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap")

    # Add the image overlay
    image_overlay = ImageOverlay(
        name='Temperature Contours',
        image=image_file,
        bounds=image_bounds,
        opacity=0.6,
        interactive=True,
        cross_origin=False,
        zindex=1
    )
    image_overlay.add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_html)
    print(f"✅ Folium map saved to: {output_html}")

# Save contour image and get geographic bounds
image_file = 'contour_overlay.png'
image_bounds = save_contour_image(subregion_gdf, image_file)

# Create folium map
create_folium_map_with_contour(image_file, image_bounds)




def plot_kriging_contour(subregion_gdf, title='Temperature Contour via Kriging'):
    # Filter valid data
    valid_data = subregion_gdf.dropna(subset=['avg_temperature'])
    xs = valid_data['centroid'].apply(lambda p: p.x).values
    ys = valid_data['centroid'].apply(lambda p: p.y).values
    temps = valid_data['avg_temperature'].values

    # Define grid
    grid_x = np.linspace(xs.min(), xs.max(), 300)
    grid_y = np.linspace(ys.min(), ys.max(), 300)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    # Perform Ordinary Kriging
    OK = OrdinaryKriging(xs, ys, temps, variogram_model='linear', verbose=False, enable_plotting=False)
    z_kriged, ss = OK.execute('grid', grid_x, grid_y)

    # Plot result
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.cm.get_cmap("coolwarm")
    contour = ax.contourf(grid_xx, grid_yy, z_kriged, levels=20, cmap=cmap)

    plt.colorbar(contour, ax=ax, label="Avg Temperature (°F)")
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    plt.show()

    plt.show() 

# ✅ Call it here, once
plot_kriging_contour(subregion_gdf)
 


