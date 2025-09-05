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
    
    # Create base map with OpenTopoMap as the only initial layer
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_level,
        tiles='https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)'
    )
    
    # Create summit selector data for JavaScript
    summit_data = []
    for idx, row in summits_gdf.iterrows():
        summit_code = row.get('code', row.get('summitCode', f'Summit-{idx}'))
        summit_name = row.get('name', row.get('title', 'Unknown'))
        
        # Find corresponding activation zone
        az_row = None
        for az_idx, az_row_candidate in activation_zones_gdf.iterrows():
            az_title = az_row_candidate.get('title', '')
            # Check if activation zone title starts with the summit code
            # e.g., "W7O/NC-001 Activation Zone" should match summit code "W7O/NC-001"
            if az_title.startswith(summit_code + ' '):
                az_row = az_row_candidate
                break
        
        summit_info = {
            'code': summit_code,
            'name': summit_name,
            'lat': row.geometry.y,
            'lon': row.geometry.x,
            'elevation_m': row.get('elevationM', row.get('altM', 0)),
            'elevation_ft': row.get('elevationFt', row.get('altFt', 0)),
            'has_az': az_row is not None
        }
        
        # Add activation zone bounds if available
        if az_row is not None and az_row.geometry is not None:
            bounds = az_row.geometry.bounds  # (minx, miny, maxx, maxy)
            summit_info['az_bounds'] = {
                'minLat': bounds[1],
                'minLon': bounds[0], 
                'maxLat': bounds[3],
                'maxLon': bounds[2]
            }
        
        summit_data.append(summit_info)

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
    
    # Add alternative tile layers AFTER all map data (these won't load until selected)
    # Limiting to just 2 essential alternatives to reduce network load
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
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add scale bar
    plugins.MeasureControl().add_to(m)
    
    # Add summit selector and title with JavaScript functionality
    summit_selector_html = f'''
    <div id="summitPanel" style="position: fixed; 
                top: 10px; 
                left: 10px; 
                width: 320px; 
                background: white; 
                border: 2px solid #ccc; 
                border-radius: 5px;
                padding: 10px;
                z-index: 1001;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                font-family: Arial, sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <h3 style="margin: 0; font-size: 16px; flex-grow: 1; text-align: center;">
                SOTA Activation Zones - {region_name}
            </h3>
            <button id="toggleSelector" style="padding: 3px 8px; background: #666; color: white; border: none; border-radius: 3px; cursor: pointer; font-weight: bold; font-size: 12px; margin-left: 10px;">
                −
            </button>
        </div>
        <div id="panelContent">
            <p style="margin: 0 0 10px 0; font-size: 12px; text-align: center;">
                <span style="color:blue">● Summits</span> | 
                <span style="color:red">█ Activation Zones</span><br>
                <span id="summitCount">{len(summits_gdf)} summits with {len(activation_zones_gdf)} activation zones</span>
            </p>
            <div style="margin-bottom: 10px;">
                <label for="summitFilter" style="font-weight: bold; font-size: 14px;">Filter Summits:</label>
                <input type="text" id="summitFilter" placeholder="Type to search..." 
                       style="width: 100%; padding: 5px; margin-top: 5px; border: 1px solid #ccc; border-radius: 3px;">
            </div>
            <div style="margin-bottom: 10px;">
                <label for="summitSelector" style="font-weight: bold; font-size: 14px;">Jump to Summit:</label>
                <select id="summitSelector" style="width: 100%; padding: 5px; margin-top: 5px; max-height: 200px;">
                    <option value="">-- Select a Summit --</option>
                </select>
            </div>
            <div style="display: flex; gap: 5px; margin-bottom: 10px;">
                <button id="prevBtn" style="flex: 1; padding: 5px; background: #28a745; color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 12px;" title="Previous Summit (Ctrl+P)">
                    ← Prev
                </button>
                <button id="nextBtn" style="flex: 1; padding: 5px; background: #28a745; color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 12px;" title="Next Summit (Ctrl+N)">
                    Next →
                </button>
            </div>
            <div style="display: flex; gap: 5px;">
                <button id="showAllBtn" style="flex: 1; padding: 5px; background: #007cba; color: white; border: none; border-radius: 3px; cursor: pointer;">
                    Show All
                </button>
            </div>
        </div>
    </div>

    <script>
        // Summit data (sorted by summit code)
        var summitData = {json.dumps(sorted(summit_data, key=lambda x: x['code']))};
        var allOptions = [];
        var currentSummitIndex = -1;
        var filteredIndices = [];
        
        // Update summit count display
        function updateSummitCount() {{
            var withAZ = summitData.filter(s => s.has_az).length;
            var withoutAZ = summitData.length - withAZ;
            document.getElementById('summitCount').innerHTML = 
                summitData.length + ' summits (' + withAZ + ' with AZ, ' + withoutAZ + ' without)';
        }}
        
        // Populate dropdown with all summits
        function populateSelector(filteredData = null) {{
            var selector = document.getElementById('summitSelector');
            var dataToUse = filteredData || summitData;
            
            // Clear existing options except first
            selector.innerHTML = '<option value="">-- Select a Summit --</option>';
            allOptions = [];
            filteredIndices = [];
            
            dataToUse.forEach(function(summit, originalIndex) {{
                var globalIndex = summitData.findIndex(s => s.code === summit.code);
                var option = document.createElement('option');
                option.value = globalIndex;
                
                // Mark summits without activation zones
                var displayText = summit.code + ': ' + summit.name;
                if (!summit.has_az) {{
                    displayText += ' (no AZ)';
                    option.style.color = '#999';
                    option.style.fontStyle = 'italic';
                }}
                
                option.text = displayText;
                selector.appendChild(option);
                allOptions.push({{
                    element: option,
                    summit: summit,
                    globalIndex: globalIndex
                }});
                filteredIndices.push(globalIndex);
            }});
        }}
        
        // Initial population
        populateSelector();
        updateSummitCount();
        
        // Filter functionality
        document.getElementById('summitFilter').addEventListener('input', function() {{
            var filterText = this.value.toLowerCase();
            if (filterText === '') {{
                populateSelector();
            }} else {{
                var filtered = summitData.filter(function(summit) {{
                    return summit.code.toLowerCase().includes(filterText) || 
                           summit.name.toLowerCase().includes(filterText);
                }});
                populateSelector(filtered);
            }}
        }});
        
        // Function to zoom to summit and its activation zone
        function zoomToSummit(summitIndex) {{
            var summit = summitData[summitIndex];
            if (!summit) return;
            
            var map = {m.get_name()};
            currentSummitIndex = summitIndex;
            
            // Update selector to show current summit
            document.getElementById('summitSelector').value = summitIndex;
            
            if (summit.az_bounds) {{
                // Zoom to activation zone bounds with padding
                var bounds = summit.az_bounds;
                var paddingLat = (bounds.maxLat - bounds.minLat) * 0.15;
                var paddingLon = (bounds.maxLon - bounds.minLon) * 0.15;
                
                var fitBounds = [
                    [bounds.minLat - paddingLat, bounds.minLon - paddingLon],
                    [bounds.maxLat + paddingLat, bounds.maxLon + paddingLon]
                ];
                map.fitBounds(fitBounds);
            }} else {{
                // Fallback: center on summit with reasonable zoom
                map.setView([summit.lat, summit.lon], 16);
            }}
        }}
        
        // Navigate to next/previous summit
        function navigateSummit(direction) {{
            if (filteredIndices.length === 0) return;
            
            var currentFilteredIndex = filteredIndices.indexOf(currentSummitIndex);
            var newFilteredIndex;
            
            if (direction === 'next') {{
                newFilteredIndex = (currentFilteredIndex + 1) % filteredIndices.length;
            }} else {{
                newFilteredIndex = currentFilteredIndex <= 0 ? filteredIndices.length - 1 : currentFilteredIndex - 1;
            }}
            
            var newSummitIndex = filteredIndices[newFilteredIndex];
            zoomToSummit(newSummitIndex);
        }}
        
        // Function to show all summits
        function showAllSummits() {{
            var map = {m.get_name()};
            currentSummitIndex = -1;
            // Calculate bounds for all summits
            if (summitData.length > 0) {{
                var minLat = Math.min(...summitData.map(s => s.lat));
                var maxLat = Math.max(...summitData.map(s => s.lat));
                var minLon = Math.min(...summitData.map(s => s.lon));
                var maxLon = Math.max(...summitData.map(s => s.lon));
                
                var bounds = [
                    [minLat, minLon],
                    [maxLat, maxLon]
                ];
                map.fitBounds(bounds, {{padding: [20, 20]}});
            }}
        }}
        
        // Toggle selector visibility
        function toggleSelector() {{
            var button = document.getElementById('toggleSelector');
            var content = document.getElementById('panelContent');
            var panel = document.getElementById('summitPanel');
            var isCollapsed = content.style.display === 'none';
            
            if (isCollapsed) {{
                content.style.display = 'block';
                button.textContent = '−';
                panel.style.width = '320px';
            }} else {{
                content.style.display = 'none';
                button.textContent = '+';
                panel.style.width = 'auto';
            }}
        }}
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(event) {{
            // Only activate if no input field is focused
            if (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'SELECT') {{
                return;
            }}
            
            if (event.ctrlKey && event.key === 'n') {{
                event.preventDefault();
                navigateSummit('next');
            }} else if (event.ctrlKey && event.key === 'p') {{
                event.preventDefault();
                navigateSummit('prev');
            }}
        }});
        
        // Event listeners
        document.getElementById('summitSelector').addEventListener('change', function() {{
            var selectedIndex = this.value;
            if (selectedIndex !== '') {{
                zoomToSummit(parseInt(selectedIndex));
            }}
        }});
        
        document.getElementById('prevBtn').addEventListener('click', function() {{
            navigateSummit('prev');
        }});
        
        document.getElementById('nextBtn').addEventListener('click', function() {{
            navigateSummit('next');
        }});
        
        document.getElementById('showAllBtn').addEventListener('click', function() {{
            showAllSummits();
            document.getElementById('summitSelector').value = '';
            document.getElementById('summitFilter').value = '';
            populateSelector();
        }});
        
        document.getElementById('toggleSelector').addEventListener('click', toggleSelector);
    </script>
    '''
    
    m.get_root().html.add_child(folium.Element(summit_selector_html))
    
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
