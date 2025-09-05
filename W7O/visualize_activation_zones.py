#!/usr/bin/env python3
"""
SOTA Activation Zone Map Visualizer

This script creates an interactive map showing SOTA summits and their activation zones.
Similar to the R Leaflet visualization in the original W7W_LC.Rmd script.

Usage:
  python visualize_activation_zones.py W7O_NC
  python visualize_activation_zones.py W7O_NC W7O_WV
  python visualize_activation_zones.py W7W_LC --output maps/map.html
  
  # Cross-association combinations
  python visualize_activation_zones.py W7O/W7O_NC W7W/W7W_LC
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import geopandas as gpd
import folium
from folium import plugins
import webbrowser

def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

def load_summit_data(input_file: Path) -> gpd.GeoDataFrame:
    """Load summit data from GeoJSON file"""
    if not input_file.exists():
        raise FileNotFoundError(f"Summit input file not found: {input_file}")
    
    summits_gdf = gpd.read_file(input_file)
    logging.info(f"Loaded {len(summits_gdf)} summits from {input_file}")
    return summits_gdf

def load_activation_zones(output_dir: Path) -> gpd.GeoDataFrame:
    """Load all activation zone GeoJSON files from output directory"""
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    geojson_files = list(output_dir.glob("*.geojson"))
    if not geojson_files:
        raise FileNotFoundError(f"No GeoJSON files found in {output_dir}")
    
    activation_zones = []
    for geojson_file in geojson_files:
        try:
            gdf = gpd.read_file(geojson_file)
            # Extract summit code from filename (e.g., W7O_NC-001.geojson -> W7O/NC-001)
            summit_code = geojson_file.stem.replace('_', '/', 1).replace('_', '-', 1)
            gdf['summit_code'] = summit_code
            gdf['filename'] = geojson_file.name
            activation_zones.append(gdf)
        except Exception as e:
            logging.warning(f"Could not load {geojson_file}: {e}")
    
    if not activation_zones:
        raise ValueError("No valid activation zone files could be loaded")
    
    combined_gdf = gpd.GeoDataFrame(pd.concat(activation_zones, ignore_index=True))
    logging.info(f"Loaded {len(combined_gdf)} activation zones from {len(geojson_files)} files")
    return combined_gdf

def calculate_map_bounds(summits_gdf: gpd.GeoDataFrame, activation_zones_gdf: gpd.GeoDataFrame) -> tuple:
    """Calculate optimal map bounds to show all data"""
    # Get bounds from both datasets
    summit_bounds = summits_gdf.total_bounds  # [minx, miny, maxx, maxy]
    az_bounds = activation_zones_gdf.total_bounds
    
    # Combined bounds
    min_x = min(summit_bounds[0], az_bounds[0])
    min_y = min(summit_bounds[1], az_bounds[1])
    max_x = max(summit_bounds[2], az_bounds[2])
    max_y = max(summit_bounds[3], az_bounds[3])
    
    # Calculate center and add some padding
    center_lat = (min_y + max_y) / 2
    center_lon = (min_x + max_x) / 2
    
    # Calculate zoom level based on bounds
    lat_diff = max_y - min_y
    lon_diff = max_x - min_x
    max_diff = max(lat_diff, lon_diff)
    
    # Rough zoom level calculation
    if max_diff > 2:
        zoom = 7
    elif max_diff > 1:
        zoom = 8
    elif max_diff > 0.5:
        zoom = 9
    elif max_diff > 0.2:
        zoom = 10
    elif max_diff > 0.1:
        zoom = 11
    else:
        zoom = 12
    
    return center_lat, center_lon, zoom

def create_interactive_map(summits_gdf: gpd.GeoDataFrame, 
                          activation_zones_gdf: gpd.GeoDataFrame,
                          region_name: str) -> folium.Map:
    """Create interactive Folium map with summits and activation zones"""
    
    # Calculate map center and zoom
    center_lat, center_lon, zoom_level = calculate_map_bounds(summits_gdf, activation_zones_gdf)
    
    # Create base map with none as default (last added tileset becomes default)
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_level,
        tiles=None
    )
    
    # Add ESRI World Imagery for satellite reference
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
        name='ESRI Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add standard OpenStreetMap
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='OpenStreetMap',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)',
        name='OpenTopoMap (default)',
        overlay=False,
        control=True
    ).add_to(m)

    # Add USGS Imagery+Topo (satellite with topo overlay)
    folium.TileLayer(
        tiles='https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryTopo/MapServer/tile/{z}/{y}/{x}',
        attr='USGS National Map',
        name='USGS Topo',
        overlay=False,
        control=True
    ).add_to(m)

    # Add USGS Pure Topo (no satellite imagery - traditional topo maps)
    folium.TileLayer(
        tiles='https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}',
        attr='USGS National Map',
        name='USGS Topo (Classic)',
        overlay=False,
        control=True
    ).add_to(m)

    # Add activation zone polygons
    logging.info("Adding activation zone polygons to map...")
    for idx, row in activation_zones_gdf.iterrows():
        summit_code = row.get('summit_code', f'Zone {idx}')
        
        # Create popup with summit information
        popup_text = f"""
        <b>{summit_code}</b><br>
        File: {row.get('filename', 'Unknown')}<br>
        Area: {row.geometry.area:.2e} deg²
        """
        
        folium.GeoJson(
            row.geometry,
            style_function=lambda feature, summit_code=summit_code: {
                'fillColor': 'red',
                'color': 'darkred',
                'weight': 2,
                'fillOpacity': 0.3,
                'opacity': 0.8
            },
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"Activation Zone: {summit_code}"
        ).add_to(m)
    
    # Add summit points
    logging.info("Adding summit markers to map...")
    for idx, row in summits_gdf.iterrows():
        summit_code = row.get('code', row.get('id', f'Summit {idx}'))
        summit_name = row.get('name', row.get('title', 'Unknown'))
        elevation_m = row.get('elevationM', row.get('altM', 0))
        elevation_ft = row.get('elevationFt', row.get('altFt', 0))
        
        # Create popup with summit details
        popup_text = f"""
        <b>{summit_code}</b><br>
        <b>{summit_name}</b><br>
        Elevation: {elevation_m}m ({elevation_ft}ft)<br>
        Coordinates: {row.geometry.y:.6f}, {row.geometry.x:.6f}
        """
        
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=6,
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"{summit_code}: {summit_name}",
            color='blue',
            fill=True,
            fillColor='lightblue',
            fillOpacity=0.8,
            weight=2
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add scale bar
    plugins.MeasureControl().add_to(m)
    
    # Add title
    title_html = f'''
    <h3 align="center" style="font-size:20px"><b>SOTA Activation Zones - {region_name}</b></h3>
    <p align="center" style="font-size:14px">
    <span style="color:blue">● Summits</span> | 
    <span style="color:red">█ Activation Zones</span><br>
    {len(summits_gdf)} summits with {len(activation_zones_gdf)} activation zones
    </p>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m

def serve_map(html_file: Path, port: int = 8000):
    """Serve the HTML file via local HTTP server and open in browser"""
    import http.server
    import socketserver
    import threading
    import webbrowser
    import time
    
    os.chdir(html_file.parent)
    
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving map at http://localhost:{port}/{html_file.name}")
        
        # Open in browser after short delay
        def open_browser():
            time.sleep(1)
            webbrowser.open(f"http://localhost:{port}/{html_file.name}")
        
        threading.Thread(target=open_browser, daemon=True).start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")

def main():
    """Main function to create the visualization"""
    parser = argparse.ArgumentParser(
        description="Create interactive map of SOTA summits and activation zones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize single region
  python visualize_activation_zones.py W7O_NC
  
  # Visualize multiple regions
  python visualize_activation_zones.py W7O_NC W7O_WV
  
  # Save to custom output file
  python visualize_activation_zones.py W7W_LC --output maps/w7w_lc_map.html
  
  # Don't auto-open browser
  python visualize_activation_zones.py W7O_NC --no-open
        """
    )
    
    parser.add_argument(
        'region_dirs',
        nargs='+',
        help='One or more region directory names (e.g., W7O_NC, W7O_WV)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output HTML file path (default: maps/{regions}_activation_zones_map.html)'
    )
    
    parser.add_argument(
        '--no-open',
        action='store_true',
        help='Do not automatically open the map in browser'
    )
    
    parser.add_argument('--serve', action='store_true', 
                   help='Serve map via local HTTP server')
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Process all region directories
    all_summits = []
    all_activation_zones = []
    region_names = []
    
    for region_dir_name in args.region_dirs:
        region_dir = Path(region_dir_name)
        if not region_dir.exists():
            logging.error(f"Region directory not found: {region_dir}")
            sys.exit(1)
        
        region_name = region_dir.name
        region_names.append(region_name)
        input_dir = region_dir / "input"
        output_dir = region_dir / "output"
        
        # Find input file
        input_files = list(input_dir.glob("*.geojson"))
        if not input_files:
            logging.error(f"No GeoJSON input files found in {input_dir}")
            sys.exit(1)
        
        input_file = input_files[0]  # Use first one found
        if len(input_files) > 1:
            logging.warning(f"Multiple input files found in {region_name}, using: {input_file}")
        
        try:
            # Load data for this region
            logging.info(f"Processing region: {region_name}")
            summits_gdf = load_summit_data(input_file)
            activation_zones_gdf = load_activation_zones(output_dir)
            
            # Ensure both datasets are in WGS84
            if summits_gdf.crs != 'EPSG:4326':
                summits_gdf = summits_gdf.to_crs('EPSG:4326')
            if activation_zones_gdf.crs != 'EPSG:4326':
                activation_zones_gdf = activation_zones_gdf.to_crs('EPSG:4326')
            
            # Add region identifier to the data
            summits_gdf['region'] = region_name
            activation_zones_gdf['region'] = region_name
            
            all_summits.append(summits_gdf)
            all_activation_zones.append(activation_zones_gdf)
            
        except Exception as e:
            logging.error(f"Error processing region {region_name}: {e}")
            sys.exit(1)
    
    # Combine all regions
    if len(all_summits) > 1:
        combined_summits = gpd.GeoDataFrame(pd.concat(all_summits, ignore_index=True))
        combined_activation_zones = gpd.GeoDataFrame(pd.concat(all_activation_zones, ignore_index=True))
    else:
        combined_summits = all_summits[0]
        combined_activation_zones = all_activation_zones[0]
    
    # Determine output file path
    if args.output:
        output_file = Path(args.output)
    else:
        regions_str = "_".join(region_names).lower()
        output_file = Path(f"maps/{regions_str}_activation_zones_map.html")
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create combined map
        combined_region_name = " + ".join(region_names)
        logging.info(f"Creating interactive map for: {combined_region_name}")
        map_obj = create_interactive_map(combined_summits, combined_activation_zones, combined_region_name)
        
        # Save map
        map_obj.save(str(output_file))
        logging.info(f"Map saved to: {output_file.absolute()}")
        
        # Open in browser unless disabled
        if args.serve:
            serve_map(output_file)
        elif not args.no_open:
            logging.info("Opening map in browser...")
            webbrowser.open(f"file://{output_file.absolute()}")
        
        logging.info("Map creation complete!")
        
    except Exception as e:
        logging.error(f"Error creating map: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
