# Recursive Subdivision Temperature Grid

A geospatial analysis tool that builds adaptive temperature-colored grid polygons from point samples using recursive subdivision. The tool computes subregion statistics, generates interactive Folium maps, and produces static visualizations with optional Kriging interpolation.

## Features

- **Adaptive Grid Resolution**: Recursively subdivides regions based on sample density, creating finer grids where more data exists
- **Interactive Maps**: Generates Folium maps with multiple layers (GeoJSON polygons, raw sample points, contour overlays)
- **Multiple Tile Layers**: Supports CartoDB, Esri Satellite, and OpenStreetMap base layers
- **Contour Visualization**: Creates transparent contour overlays that can be exported as PNG or KML for Google Earth
- **Statistical Analysis**: Computes average temperature, standard deviation, and sample count per subregion
- **Kriging Support**: Optional Ordinary Kriging interpolation for smoother temperature surfaces
- **Flexible Input**: Auto-detects longitude/latitude and temperature columns with automatic unit conversion (°C to °F)

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

- folium
- geopandas
- matplotlib
- numpy
- pandas
- pykrige
- scipy
- shapely

## Usage

### Basic Run

```bash
python Recursive_Sub_Sample_Script_V9.py --csv InputData.csv
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--csv` | `InputData.csv` | Path to input CSV with longitude, latitude, and temperature |
| `--min-samples` | `10` | Minimum samples required to subdivide a region |
| `--min-cell-samples` | `4` | Minimum samples required to keep a subregion cell |
| `--predicate` | `intersects` | Spatial join predicate (`intersects` or `within`) |
| `--output-centroids` | `intermediate_centroids.csv` | Output CSV for centroid coordinates and avg temperature |
| `--folium-html` | `folium_geojson_only.html` | Output HTML for interactive Folium map |
| `--with-raster` | `false` | Also render transparent contour PNG and raster Folium map |
| `--raster-png` | `contour_overlay.png` | Filename for saved transparent contour PNG |
| `--raster-html` | `folium_map.html` | Output HTML for Folium raster map |
| `--no-static-plots` | `false` | Disable Matplotlib static plots |
| `--kriging` | `false` | Show Kriging contour plot |
| `--folium-output-fundamentals` | `folium_output_fundamentals.csv` | Output CSV of per-subregion statistics |

### Examples

```bash
# Specify input CSV and minimum samples threshold
python Recursive_Sub_Sample_Script_V9.py --csv MyData.csv --min-samples 5

# Generate raster overlay map
python Recursive_Sub_Sample_Script_V9.py --csv InputData.csv --with-raster

# Run with Kriging visualization, no static plots
python Recursive_Sub_Sample_Script_V9.py --csv InputData.csv --kriging --no-static-plots
```

## Input Data Format

The input CSV must contain columns for longitude, latitude, and temperature. Column names are auto-detected:

**Longitude**: `longitude`, `lon`, `lng`, `x`, `long`

**Latitude**: `latitude`, `lat`, `y`

**Temperature**: `temperature`, `temp`, `degf`, `deg_f`, `degc`, `deg_c`, or any column containing "temp"

Example CSV:
```csv
longitude,latitude,temperature
-121.7623,37.6879,72.5
-121.7612,37.6891,73.2
-121.7634,37.6868,71.8
```

## Output Files

| File | Description |
|------|-------------|
| `folium_geojson_only.html` | Interactive map with temperature grid polygons and sample points |
| `folium_map.html` | Interactive map with raster contour overlay (if `--with-raster`) |
| `intermediate_centroids.csv` | Centroid coordinates and average temperature per subregion |
| `folium_output_fundamentals.csv` | Statistics per subregion (avg temp, std dev, sample count) |
| `contour_overlay.png` | Transparent contour image for map overlays |
| `contour_overlay.kml` | KML file for Google Earth overlay |

## Algorithm

The recursive subdivision algorithm:

1. Start with the bounding box of all input points
2. If the region contains fewer than `min_samples` points, keep it as a leaf cell
3. Otherwise, divide the region into 4 equal quadrants
4. Recursively apply steps 2-3 to each quadrant
5. Filter out cells with fewer than `min_cell_samples` points
6. Compute statistics (mean, std dev) for each final cell

This creates an adaptive quad-tree structure where regions with dense data have higher resolution than sparse areas.

## License

MIT
