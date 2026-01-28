import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
 

def plot_final_subregions_with_random_colors(subregion_gdf, gdf=None, title='Final Subdivisions with Random Colors'):
    """
    Plots the final rectangular subregions with random fill colors.

    Parameters:
    - subregion_gdf: GeoDataFrame containing Polygon geometries of the final subregions.
    - gdf: (Optional) GeoDataFrame containing Point geometries to overlay on the plot.
    - title: Title of the plot.
    """
    # Ensure there are geometries to plot
    if subregion_gdf.empty:
        print("No subregions to plot.")
        return

    # Generate a list of random colors for each polygon
    num_polygons = len(subregion_gdf)
    colors = ['#%06X' % random.randint(0, 0xFFFFFF) for _ in range(num_polygons)]

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the sample points if provided
    if gdf is not None and not gdf.empty:
        gdf.plot(ax=ax, marker='o', color='blue', markersize=2, label='Sample Points')

    # Plot the final subdivided subregions with random fill colors
    subregion_gdf.plot(ax=ax, color=colors, edgecolor='black', linewidth=0.5)

    # Set plot title and labels
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')

    plt.show()

#########################################################################################


# Function to generate random points within a specified radius around a central point

import math
import random

def generate_random_points(center_lat, center_lon, radius_miles, num_points):
    """
    Generate random points within a specified radius around a central point,
    each with an associated random temperature between 50.0°F and 70.0°F.

    Parameters:
    - center_lat (float): Latitude of the center point in degrees.
    - center_lon (float): Longitude of the center point in degrees.
    - radius_miles (float): Radius around the center point in miles.
    - num_points (int): Number of random points to generate.

    Returns:
    - List of tuples: Each tuple contains (latitude, longitude, temperature).
    """
    points = []
    radius_km = radius_miles * 1.60934  # Convert miles to kilometers
    for _ in range(num_points):
        # Convert radius from kilometers to degrees
        radius_deg = radius_km / 111.32  # Approximate conversion

        # Generate random distance and angle
        u = random.random()
        v = random.random()
        w = radius_deg * math.sqrt(u)
        t = 2 * math.pi * v
        x = w * math.cos(t)
        y = w * math.sin(t)

        # Adjust the x-coordinate for the shrinking of the east-west distances
        new_x = x / math.cos(math.radians(center_lat))

        found_lat = center_lat + y
        found_lon = center_lon + new_x

        # Generate a random temperature between 50.0 and 70.0
        temperature = round(random.uniform(50.0, 70.0), 2)

        points.append((found_lat, found_lon, temperature))

        # Print the generated point with temperature
        print(f"latitude: {found_lat}, longitude: {found_lon}, temperature: {temperature}°F")

    return points

  


##########################################################################################

# Global list to capture all subdivision stages
subdivision_stages = []

def recursive_subdivision_geopandas(gdf, min_samples=2, region_polygon=None, level=0):
    """
    Recursively subdivides a GeoDataFrame of points into rectangular subregions.
    Subdivides only if all 4 subregions have at least min_samples points.

    Parameters:
    - gdf: GeoDataFrame containing Point geometries.
    - min_samples: Minimum number of points required in each subregion to allow further subdivision.
    - region_polygon: The current region being subdivided (used in recursion).
    - level: Current recursion depth (used for tracking).

    Returns:
    - List of Polygon geometries representing the final subregions.
    """
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

    # Define 4 subrectangles, ensuring they are closed polygons
    subrects = [
        Polygon([(x_min, y_min), (mid_x, y_min), (mid_x, mid_y), (x_min, mid_y), (x_min, y_min)]),
        Polygon([(mid_x, y_min), (x_max, y_min), (x_max, mid_y), (mid_x, mid_y), (mid_x, y_min)]),
        Polygon([(x_min, mid_y), (mid_x, mid_y), (mid_x, y_max), (x_min, y_max), (x_min, mid_y)]),
        Polygon([(mid_x, mid_y), (x_max, mid_y), (x_max, y_max), (mid_x, y_max), (mid_x, mid_y)])
    ]

    subregions = []
    sub_gdfs = []

    # Check how many samples in each subregion
    all_subregions_ok = True
    for subrect in subrects:
        sub_gdf = gdf[gdf.intersects(subrect)]
        sub_gdfs.append((sub_gdf, subrect))
        if len(sub_gdf) < min_samples:
            all_subregions_ok = False
            break  # No need to check further if any one is too small

    # Save current subdivision stage for plotting
    subdivision_stages.append(subrects)

    if all_subregions_ok:
        # Subdivide each subrectangle recursively
        for sub_gdf, subrect in sub_gdfs:
            if not sub_gdf.empty:
                subregions.extend(recursive_subdivision_geopandas(sub_gdf, min_samples, subrect, level+1))
    else:
        # Accept current parent region if subdivision is not allowed
        subregions.append(region_polygon)

    return subregions


########################################################################################################

def compute_average_temperature(subregion_gdf, sample_points_gdf):
    """
    Computes the average temperature and the number of sample points within each subregion.

    Parameters:
    - subregion_gdf (GeoDataFrame): GeoDataFrame containing polygon geometries of subregions.
    - sample_points_gdf (GeoDataFrame): GeoDataFrame containing point geometries with a 'temperature' column.

    Returns:
    - GeoDataFrame: Updated subregion_gdf with new columns 'avg_temperature' and 'sample_count'.
    """
    # Ensure both GeoDataFrames use the same CRS
    if subregion_gdf.crs != sample_points_gdf.crs:
        sample_points_gdf = sample_points_gdf.to_crs(subregion_gdf.crs)

    # Perform spatial join: assign each point to a subregion
    joined = gpd.sjoin(sample_points_gdf, subregion_gdf, how='inner', predicate='within')

    # Group by subregion index and compute mean temperature and count
    stats = joined.groupby('index_right').agg(
        avg_temperature=('temperature', 'mean'),
        sample_count=('temperature', 'count')
    )

    # Add the statistics to the subregion_gdf
    subregion_gdf = subregion_gdf.copy()
    subregion_gdf['avg_temperature'] = subregion_gdf.index.map(stats['avg_temperature'])
    subregion_gdf['sample_count'] = subregion_gdf.index.map(stats['sample_count'])

    return subregion_gdf

###########################################################################################
def print_centroid_coordinates_avg_temp_and_sample_count(subregion_gdf):
    """
    Prints the centroid coordinates, average temperature, and sample count for each subregion.

    Parameters:
    - subregion_gdf (GeoDataFrame): GeoDataFrame containing 'centroid', 'avg_temperature', and 'sample_count' columns.
    """
    for idx, row in subregion_gdf.iterrows():
        centroid = row.get('centroid')
        avg_temp = row.get('avg_temperature')
        sample_count = row.get('sample_count')

        if centroid and not centroid.is_empty:
            lon, lat = centroid.x, centroid.y
            print(f"Subregion {idx}:")
            print(f"  Centroid Coordinates: (Longitude: {lon:.6f}, Latitude: {lat:.6f})")
            print(f"  Average Temperature: {avg_temp:.2f}°F" if avg_temp is not None else "  Average Temperature: N/A")
            print(f"  Sample Count: {sample_count}" if sample_count is not None else "  Sample Count: N/A")
            print()
        else:
            print(f"Subregion {idx}: Invalid or empty centroid.\n")


#####################################################################################

# Define center coordinates for Livermore, CA
center_lat = 37.6819
center_lon = -121.7680

# Generate random points
num_points = 1000  # Adjust as needed
radius_miles = 2
random_points = generate_random_points(center_lat, center_lon, radius_miles, num_points)

# Extract latitude, longitude, and temperature from random_points
latitudes = [lat for lat, lon, temp in random_points]
longitudes = [lon for lat, lon, temp in random_points]
temperatures = [temp for lat, lon, temp in random_points]

# Create a DataFrame with the extracted data
df = pd.DataFrame({
    'latitude': latitudes,
    'longitude': longitudes,
    'temperature': temperatures
})


# Create GeoDataFrame
geometry = [Point(lon, lat) for lat, lon, _ in random_points]

# Extract temperature values
temperatures = [temp for _, _, temp in random_points]


gdf = gpd.GeoDataFrame({'temperature': temperatures}, geometry=geometry, crs="EPSG:4326")

# Now, you can apply your recursive subdivision function to 'gdf'
# Ensure your recursive_subdivision_geopandas function is defined
final_subregions = recursive_subdivision_geopandas(gdf.copy(), min_samples=2)

# Create GeoDataFrame of the final subregions
if final_subregions:
    subregion_gdf = gpd.GeoDataFrame({'geometry': final_subregions}, crs="EPSG:4326")
else:
    subregion_gdf = gpd.GeoDataFrame({'geometry': []}, crs="EPSG:4326")

# Filter out invalid or empty geometries
subregion_gdf = subregion_gdf[subregion_gdf.is_valid & ~subregion_gdf.is_empty]



# add centroid to the subregion_gdf for plotting
subregion_gdf['centroid'] = subregion_gdf.geometry.centroid

# Compute average temperature for each subregion
subregion_gdf = compute_average_temperature(subregion_gdf, gdf)

# Print the tuple with centroid coordinates and average temperature
print_centroid_coordinates_avg_temp_and_sample_count(subregion_gdf)

# Proceed with plotting  

plot_final_subregions_with_random_colors(subregion_gdf, gdf=None, title='Final Subdivisions with Random Colors')


