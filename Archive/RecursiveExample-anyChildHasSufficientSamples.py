import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import random

# Global list to capture all subdivision stages
subdivision_stages = []

def recursive_subdivision_geopandas(gdf, min_samples=5, region_polygon=None, level=0):
    """
    Recursively subdivides a GeoDataFrame of points into rectangular subregions until each
    subregion contains fewer than min_samples in all quadrants.
    """

    if region_polygon is not None:
        x_min, y_min, x_max, y_max = region_polygon.bounds
    else:
        bounds = gdf.total_bounds
        x_min, y_min, x_max, y_max = bounds
        region_polygon = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)])

    if len(gdf) < min_samples:
        return [region_polygon]

    mid_x = (x_min + x_max) / 2
    mid_y = (y_min + y_max) / 2

    # ✅ Ensure all polygons are properly closed (first point == last point)
    subrects = [
        Polygon([(x_min, y_min), (mid_x, y_min), (mid_x, mid_y), (x_min, mid_y), (x_min, y_min)]),
        Polygon([(mid_x, y_min), (x_max, y_min), (x_max, mid_y), (mid_x, mid_y), (mid_x, y_min)]),
        Polygon([(x_min, mid_y), (mid_x, mid_y), (mid_x, y_max), (x_min, y_max), (x_min, mid_y)]),
        Polygon([(mid_x, mid_y), (x_max, mid_y), (x_max, y_max), (mid_x, y_max), (mid_x, mid_y)])
    ]

    subregions = []
    should_subdivide = False
    sub_gdfs = []

    for subrect in subrects:
        sub_gdf = gdf[gdf.intersects(subrect)]
        sub_gdfs.append((sub_gdf, subrect))
        if len(sub_gdf) >= min_samples:
            should_subdivide = True

    # Save the current subrectangles (for visualization later)
    subdivision_stages.append(subrects)

    if should_subdivide:
        for sub_gdf, subrect in sub_gdfs:
            if not sub_gdf.empty:
                subregions.extend(recursive_subdivision_geopandas(sub_gdf, min_samples, subrect, level+1))
    else:
        subregions.append(region_polygon)

    return subregions

# ------------------- Example usage ----------------------

# Create example data
np.random.seed(42)
num_points = 3001

points = [Point(x, y) for x, y in (10 + np.random.rand(num_points, 2))]
gdf = gpd.GeoDataFrame({'geometry': points}, crs="EPSG:4326")

# Perform the recursive subdivision
final_subregions = recursive_subdivision_geopandas(gdf.copy(), min_samples=5)

# Create GeoDataFrame of the final subregions
if final_subregions:
    subregion_gdf = gpd.GeoDataFrame({'geometry': final_subregions}, crs="EPSG:4326")
else:
    subregion_gdf = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:4326")

# 🚨 Filter out invalid or empty geometries
subregion_gdf = subregion_gdf[subregion_gdf.is_valid & ~subregion_gdf.is_empty]

# Print the number of subregions
print(f"Number of valid final subregions: {len(subregion_gdf)}")

# ------------------- Plot each stage ----------------------


if len(gdf) < 501:
    for i, stage in enumerate(subdivision_stages):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot sample points
        gdf.plot(ax=ax, marker='o', color='blue', markersize=2, label='Sample Points')

        # Plot the subregions in this stage
        for geom in stage:
            if isinstance(geom, Polygon) and not geom.is_empty and geom.is_valid:
                x, y = geom.exterior.coords.xy
                ax.plot(x, y, color='red', linewidth=0.5)

        ax.set_title(f'Subdivision Stage {i+1}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend()
        ax.set_xlim(10, 11)
        ax.set_ylim(10, 11)
        ax.set_aspect('equal', adjustable='box')

        plt.show()
else:
    print("Sample size is large; skipping intermediate subdivision plots.")

# ------------------- Plot final subregions with random colors ----------------------

fig, ax = plt.subplots(figsize=(8, 8))

# Plot the sample points
gdf.plot(ax=ax, marker='o', color='blue', markersize=2, label='Sample Points')

# Generate a list of random colors for each polygon
num_polygons = len(subregion_gdf)
colors = ['#%06X' % random.randint(0, 0xFFFFFF) for _ in range(num_polygons)]

# Plot the final subdivided subregions with random fill colors
if not subregion_gdf.empty:
    subregion_gdf.plot(ax=ax, color=colors, edgecolor='black', linewidth=0.5)

ax.set_title('Final Subdivisions with Random Colors')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend()
ax.set_xlim(10, 11)
ax.set_ylim(10, 11)
ax.set_aspect('equal', adjustable='box')

plt.show()
