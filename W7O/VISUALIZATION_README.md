# SOTA Activation Zone Visualizer

Interactive map visualization tool for SOTA summits and their activation zones, similar to the R Leaflet implementation in the original W7W_LC.Rmd script.

## Features

- **Interactive Map**: Pan, zoom, and explore summits and activation zones
- **Multiple Tile Layers**: OpenStreetMap, CartoDB Positron, and CartoDB Dark
- **Detailed Popups**: Click on summits or zones for detailed information
- **Automatic Bounds**: Map automatically centers and zooms to show all data
- **Layer Control**: Toggle between different map layers
- **Measurement Tools**: Built-in distance measurement capability

## Requirements

```bash
pip install folium geopandas pandas
```

## Usage

### Basic Usage
```bash
# From association level (recommended)
cd W7O
python ../visualize_activation_zones.py --region-dir W7O_NC

# From root level
python visualize_activation_zones.py --region-dir W7O/W7O_NC

# Visualize W7W/LC summits and activation zones  
python visualize_activation_zones.py --region-dir W7W/W7W_LC
```

### Advanced Options
```bash
# Save to custom output file
python visualize_activation_zones.py --region-dir W7O/W7O_NC --output my_custom_map.html

# Don't auto-open browser (useful for scripts)
python visualize_activation_zones.py --region-dir W7O/W7O_NC --no-open
```

## Expected Directory Structure

The script expects this directory structure:
```
region-dir/
├── input/
│   └── region.geojson     # Summit locations (from SOTA API or manual)
└── output/
    ├── SUMMIT-001.geojson # Individual activation zone files
    ├── SUMMIT-002.geojson
    └── ...
```

For example:
```
W7O/W7O_NC/
├── input/
│   └── W7O_NC.geojson
└── output/
    ├── W7O_NC-001.geojson
    ├── W7O_NC-002.geojson
    └── ...
```

## Map Elements

### Summit Markers (Blue Circles)
- **Location**: Exact summit coordinates
- **Popup**: Summit code, name, elevation, coordinates
- **Tooltip**: Quick summit identification on hover

### Activation Zone Polygons (Red Areas)
- **Boundary**: 82-foot elevation contour below summit
- **Popup**: Summit code, filename, polygon area
- **Tooltip**: Quick zone identification on hover

### Controls
- **Layer Control**: Switch between map tile layers
- **Zoom Controls**: Standard pan/zoom functionality
- **Scale Bar**: Distance measurements and reference
- **Measure Tool**: Click to measure distances on map

## Output

The script generates a self-contained HTML file that can be:
- Opened in any web browser
- Shared via email or file transfer
- Embedded in websites or documentation
- Used offline (no internet required after creation)

## Examples

After running the script, you'll get an interactive map showing:
1. All summits as blue markers with detailed information
2. Activation zones as semi-transparent red polygons
3. Map title showing region and summary statistics
4. Full interactivity for exploration and analysis

This provides the same functionality as the original R Leaflet code but in Python, making it easier to integrate with the existing Python processing pipeline.
