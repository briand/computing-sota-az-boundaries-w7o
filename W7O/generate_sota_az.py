#!/usr/bin/env python3
"""
SOTA Activation Zone Generator (Python Version for Oregon W7O)

This script:
1. Fetches summit data from SOTA API
2. Queries Oregon ImageServer for elevation data 
3. Creates 500m buffered polygons around summit coordinates (which are actually much larger than 500m from the arcgis server)
4. Computes activation zones (elevation minus 82 feet)
5. Outputs individual GeoJSON files per summit
"""

import os
import sys
import json
import requests
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from urllib.parse import urlencode
import time
from datetime import datetime, timezone


# Configuration
SOTA_API_BASE = "https://api2.sota.org.uk/api"
ARCGIS_IMAGESERVER = "https://gis.dogami.oregon.gov/arcgis/rest/services/lidar/DIGITAL_TERRAIN_MODEL_MOSAIC/ImageServer"
TILE_RADIUS_KM = 0.5  # From center to edge of raster tile (500m radius)
ACTIVATION_ZONE_HEIGHT_FT = 82
DEFAULT_ELEVATION_TOLERANCE_FT = 20.0  # Default tolerance for summit vs raster elevation validation
DEFAULT_ASSOCIATION = "W7O"

# Directory paths will be set dynamically based on association/region
CACHE_DIR = None
INPUT_DIR = None
OUTPUT_DIR = None


# Import for static analysis and type checking
if TYPE_CHECKING:
    import geopandas as gpd
    import shapely.geometry as geom
    from shapely.geometry import shape, MultiPolygon, Polygon, Point
    from shapely.ops import transform
    import pyproj
    from pyproj import CRS, Transformer
    import rasterio
    import rasterio.mask
    import rasterio.features
    from rasterio.transform import xy
    import numpy as np
    import matplotlib.pyplot as plt

import_libset = ''
# Import all required libraries at startup
try:
    import geopandas as gpd
    import shapely.geometry as geom
    from shapely.geometry import shape, MultiPolygon, Polygon, Point
    from shapely.ops import transform
    import pyproj
    from pyproj import CRS, Transformer
except ImportError as e:
    print(f"Error: Required geospatial libraries not available")
    import_libset += 'geopandas shapely pyproj '

try:
    import rasterio
    import rasterio.mask
    import rasterio.features
    from rasterio.transform import xy
    import numpy as np
except ImportError as e:
    print(f"Error: Required raster libraries not available")
    import_libset += 'rasterio numpy '

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error: Matplotlib not available")
    import_libset += 'matplotlib '

if import_libset != '':
    print(f"Install missing libraries with: 'pip install {import_libset.strip()}'")
    sys.exit(1)

def setup_logging(log_file: Optional[str] = None, quiet: bool = False):
    """Setup logging to stdout and/or file
    
    Args:
        log_file: Path to log file (optional)
        quiet: If True and log_file is specified, suppress stdout output
    """
    log_format = '%(asctime)s %(levelname)s - %(message)s'
    
    if log_file:
        # Setup file logging
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            filename=log_file,
            filemode='w'
        )
        
        # Add console logging unless quiet mode is enabled
        if not quiet:
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(logging.INFO)
            console.setFormatter(logging.Formatter(log_format))
            logging.getLogger().addHandler(console)
    else:
        # Standard stdout logging when no file specified
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            stream=sys.stdout
        )

def calculate_distance_to_edges(contour_coords: List[Tuple[float, float]], 
                               tiff_bounds: Tuple[float, float, float, float],
                               crs: str) -> Dict[str, float]:
    """Calculate minimum distance from contour points to TIFF bounding box edges in meters"""
    min_x, min_y, max_x, max_y = tiff_bounds
    
    min_dist_left = float('inf')
    min_dist_right = float('inf')
    min_dist_bottom = float('inf')
    min_dist_top = float('inf')
    min_dist_overall = float('inf')
    
    # Determine coordinate system and units
    is_geographic = crs.upper() in ['EPSG:4326', 'WGS84'] or 'GEOGCS' in crs.upper()
    is_oregon_lambert_ft = crs.upper() in ['EPSG:6557']  # Oregon GIC Lambert uses US Survey Feet
    
    for x, y in contour_coords:
        # Distance to each edge in coordinate units
        dist_left = abs(x - min_x)
        dist_right = abs(x - max_x)
        dist_bottom = abs(y - min_y)
        dist_top = abs(y - max_y)
        
        # Track minimums
        min_dist_left = min(min_dist_left, dist_left)
        min_dist_right = min(min_dist_right, dist_right)
        min_dist_bottom = min(min_dist_bottom, dist_bottom)
        min_dist_top = min(min_dist_top, dist_top)
        
        # Overall minimum distance to any edge
        min_dist_overall = min(min_dist_overall, dist_left, dist_right, dist_bottom, dist_top)
    
    # Convert to meters if needed
    if is_geographic:
        # For geographic coordinates, convert degrees to approximate meters
        # Use the latitude for more accurate conversion
        avg_lat = (min_y + max_y) / 2
        meters_per_degree = 111000 * np.cos(np.radians(avg_lat))
        
        return {
            'left': min_dist_left * meters_per_degree,
            'right': min_dist_right * meters_per_degree,
            'bottom': min_dist_bottom * 111000,  # Latitude degrees to meters
            'top': min_dist_top * 111000,
            'shortest': min_dist_overall * meters_per_degree
        }
    elif is_oregon_lambert_ft:
        # EPSG:6557 uses US Survey Feet - convert to meters
        feet_to_meters = 1.0 / 3.28084
        return {
            'left': min_dist_left * feet_to_meters,
            'right': min_dist_right * feet_to_meters,
            'bottom': min_dist_bottom * feet_to_meters,
            'top': min_dist_top * feet_to_meters,
            'shortest': min_dist_overall * feet_to_meters
        }
    else:
        # Assume already in projected meters
        return {
            'left': min_dist_left,
            'right': min_dist_right,
            'bottom': min_dist_bottom,
            'top': min_dist_top,
            'shortest': min_dist_overall
        }

def setup_directories_from_input_file(summits_file: Path, output_dir: Optional[str] = None):
    """Setup directory structure based on input file location or specified output directory"""
    global CACHE_DIR, INPUT_DIR, OUTPUT_DIR
    
    if output_dir:
        # Use specified output directory as base
        base_dir = Path(output_dir)
        logging.info(f"Using specified base directory: {base_dir}")
    else:
        # Use the directory containing the summit file as base
        base_dir = summits_file.parent
        logging.info(f"Using summit file directory as base: {base_dir}")
    
    # Set up directory paths (no input folder needed)
    CACHE_DIR = base_dir / "cache"
    OUTPUT_DIR = base_dir / "output"
    INPUT_DIR = None  # Not needed when processing from file
    
    logging.info(f"  Cache: {CACHE_DIR}")
    logging.info(f"  Output: {OUTPUT_DIR}")

def setup_regional_directories(association: str, region: str):
    """Setup directory structure for association/region"""
    global CACHE_DIR, INPUT_DIR, OUTPUT_DIR
    
    # Create region directory name (e.g., W7O_NC)
    region_dir = Path(f"{association}_{region}")
    
    # Set up directory paths
    CACHE_DIR = region_dir / "cache"
    INPUT_DIR = region_dir / "input"
    OUTPUT_DIR = region_dir / "output"
    
    logging.info(f"Using regional directory structure: {region_dir}")
    logging.info(f"  Cache: {CACHE_DIR}")
    logging.info(f"  Input: {INPUT_DIR}")
    logging.info(f"  Output: {OUTPUT_DIR}")

def ensure_directories():
    """Create cache, input, and output directories if they don't exist"""
    if CACHE_DIR is None or OUTPUT_DIR is None:
        raise ValueError("Directories not initialized. Call setup_regional_directories() or setup_directories_from_input_file() first.")
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # INPUT_DIR is optional (not used when processing from existing file)
    if INPUT_DIR is not None:
        INPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_summits_geojson(summits: List[Dict], association: str, region: str) -> Path:
    """Save summit data as GeoJSON in the input directory"""
    filename = f"{association}_{region}.geojson"
    output_path = INPUT_DIR / filename if INPUT_DIR is not None else Path(filename)
    
    # Convert SOTA summit data to GeoJSON format similar to W7W_LC.geojson
    features = []
    for summit in summits:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [summit['longitude'], summit['latitude']]
            },
            "id": summit['summitCode'],
            "properties": {
                "code": summit['summitCode'],
                "name": summit['name'],
                "title": summit['summitCode'],
                "altM": summit['altM'],
                "altFt": summit['altFt'],
                "validTo": summit['validTo'],
                "validFrom": summit['validFrom']
            }
        }
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(output_path, 'w') as f:
        json.dump(geojson, f, separators=(',', ':'), indent=2)  # Compact format like reference
    
    logging.info(f"Summit data saved to: {output_path}")
    return output_path

def load_summits_from_geojson(file_path: Path) -> List[Dict]:
    """Load summits from GeoJSON file and convert to SOTA format"""
    with open(file_path) as f:
        geojson = json.load(f)
    
    summits = []
    for feature in geojson['features']:
        lon, lat = feature['geometry']['coordinates']
        props = feature['properties']
        
        summit = {
            'summitCode': props['code'],
            'name': props['name'],
            'longitude': lon,
            'latitude': lat,
            'altM': props.get('elevationM', 0),
            'altFt': props.get('elevationFt', 0),
            'points': props.get('points', 0),
            'validTo': props.get('validTo', '2099-12-31T00:00:00Z'),
            'validFrom': props.get('validFrom', '2010-01-01T00:00:00Z')
        }
        summits.append(summit)
    
    return summits

def is_summit_valid(summit: Dict) -> bool:
    """Check if summit is currently valid (not retired)"""
    try:
        valid_to_str = summit.get('validTo', '')
        if not valid_to_str:
            return True  # Assume valid if no validTo date
        
        # Parse the ISO 8601 date string
        valid_to = datetime.fromisoformat(valid_to_str.replace('Z', '+00:00'))
        current_time = datetime.now(timezone.utc)
        
        return valid_to > current_time
        
    except (ValueError, KeyError) as e:
        logging.warning(f"Could not parse validTo date for summit {summit.get('summitCode', 'unknown')}: {e}")
        return True  # Assume valid if parsing fails

def fetch_sota_summits(association: str = "W7O", region: str = "NC") -> List[Dict]:
    """Fetch summit data from SOTA API and filter out retired summits"""
    url = f"{SOTA_API_BASE}/regions/{association}/{region}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        all_summits = data.get('summits', [])
        
        # Filter out retired summits
        valid_summits = [summit for summit in all_summits if is_summit_valid(summit)]
        
        retired_count = len(all_summits) - len(valid_summits)
        
        logging.info(f"Successfully fetched {len(all_summits)} summits from {association}/{region}")
        if retired_count > 0:
            logging.info(f"Filtered out {retired_count} retired summits")
        logging.info(f"Processing {len(valid_summits)} active summits")
        
        return valid_summits
        
    except requests.RequestException as e:
        logging.error(f"Error fetching SOTA data: {e}")
        return []

def build_imageserver_query_url(lon: float, lat: float, buffer_m: int = 1000) -> str:
    """Build Oregon ImageServer query URL for elevation data with accurate buffer calculation"""
    # Accurate conversion from meters to degrees at the given latitude
    # 1 degree of latitude ≈ 111,000 meters (constant)
    # 1 degree of longitude ≈ 111,000 * cos(latitude) meters (varies by latitude)
    
    lat_buffer_deg = buffer_m / 111000.0
    lon_buffer_deg = buffer_m / (111000.0 * np.cos(np.radians(lat)))
    
    xmin = lon - lon_buffer_deg
    ymin = lat - lat_buffer_deg 
    xmax = lon + lon_buffer_deg
    ymax = lat + lat_buffer_deg
    
    logging.debug(f"  Bounding box: {buffer_m}m buffer = {lat_buffer_deg:.6f}° lat, {lon_buffer_deg:.6f}° lon")
    logging.debug(f"  Coords: ({xmin:.6f}, {ymin:.6f}) to ({xmax:.6f}, {ymax:.6f})")
    
    params = {
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": "4326",
        "size": "512,512",
        "format": "tiff",
        "pixelType": "F32",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image"
    }
    
    return f"{ARCGIS_IMAGESERVER}/exportImage?{urlencode(params)}"

def get_cached_elevation_file(lon: float, lat: float, buffer_m: int = 1000) -> Path:
    """Get cached elevation file path for coordinates"""
    filename = f"elev_{lon:.6f}_{lat:.6f}_{buffer_m}m.tif"
    return CACHE_DIR / filename if CACHE_DIR is not None else Path(filename)

def download_elevation_data(lon: float, lat: float, buffer_m: int = 1000) -> Optional[Path]:
    """Download elevation data from Oregon ImageServer with caching"""
    cache_file = get_cached_elevation_file(lon, lat, buffer_m)
    
    # Use cached file if it exists
    if cache_file.exists():
        logging.info(f"Using cached elevation data: {cache_file}")
        return cache_file
    
    # Download new data
    url = build_imageserver_query_url(lon, lat, buffer_m)
    logging.info(f"Downloading elevation data for {lon:.6f}, {lat:.6f}...")
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            f.write(response.content)
            
        logging.info(f"Elevation data saved to cache: {cache_file}")
        return cache_file
        
    except requests.RequestException as e:
        logging.error(f"Error downloading elevation data: {e}")
        return None

def determine_needed_adjacent_tiles(distances: Dict[str, float], threshold_m: float = 10.0) -> List[str]:
    """Determine which adjacent tiles are needed based on boundary distances
    
    Args:
        distances: Dictionary with 'left', 'right', 'top', 'bottom' distances in meters
        threshold_m: Distance threshold in meters below which adjacent tiles are needed
    
    Returns:
        List of direction strings indicating which adjacent tiles to fetch
    """
    needed_directions = []
    
    if distances['left'] < threshold_m:
        needed_directions.append('left')
    if distances['right'] < threshold_m:
        needed_directions.append('right')  
    if distances['top'] < threshold_m:
        needed_directions.append('top')
        if 'left' in needed_directions:
            needed_directions.append('left-top')
        if 'right' in needed_directions:
            needed_directions.append('right-top')
    if distances['bottom'] < threshold_m:
        needed_directions.append('bottom')
        if 'left' in needed_directions:
            needed_directions.append('left-bottom')
        if 'right' in needed_directions:
            needed_directions.append('right-bottom')
    
    return needed_directions

def calculate_adjacent_tile_centers(center_lon: float, center_lat: float, buffer_m: int, 
                                   directions: List[str]) -> List[Tuple[float, float]]:
    """Calculate center coordinates for adjacent tiles in specific directions
    
    Args:
        center_lon, center_lat: Center coordinates
        buffer_m: Buffer radius in meters
        directions: List of directions to fetch ('left', 'right', 'top', 'bottom')
                   If None, returns center tile only
    
    Returns a list of (lon, lat) tuples for the requested tiles.
    """
    # Calculate approximate tile size in degrees
    # Each tile is approximately 2 * buffer_m in size
    tile_size_m = buffer_m * 2
    
    lat_step = tile_size_m / 111000.0  # Latitude degrees
    lon_step = tile_size_m / (111000.0 * np.cos(np.radians(center_lat)))  # Longitude degrees
    
    tile_centers = []
    
    # Add specific adjacent tiles based on directions
    if directions:
        for direction in directions:
            if direction == 'left':
                tile_centers.append((center_lon - lon_step, center_lat))
            elif direction == 'left-top':
                    tile_centers.append((center_lon - lon_step, center_lat + lat_step))
            elif direction == 'left-bottom':
                    tile_centers.append((center_lon - lon_step, center_lat - lat_step))
            elif direction == 'right':
                tile_centers.append((center_lon + lon_step, center_lat))
            elif direction == 'right-top':
                tile_centers.append((center_lon + lon_step, center_lat + lat_step))
            elif direction == 'right-bottom':
                tile_centers.append((center_lon + lon_step, center_lat - lat_step))
            elif direction == 'top':
                tile_centers.append((center_lon, center_lat + lat_step))
            elif direction == 'bottom':
                tile_centers.append((center_lon, center_lat - lat_step))
    
    return tile_centers

def download_adjacent_elevation_tiles(center_lon: float, center_lat: float, 
                                     needed_directions: List[str], buffer_m: int = 1000) -> List[Path]:
    """Download elevation data for specific adjacent tiles
    
    Returns a list of paths to downloaded elevation files.
    """
    tile_centers = calculate_adjacent_tile_centers(center_lon, center_lat, buffer_m, needed_directions)
    elevation_files = []
    
    direction_desc = ', '.join(needed_directions)
    logging.info(f"Downloading additional elevation tiles ({len(needed_directions)} tiles: {direction_desc})")
    
    for i, (tile_lon, tile_lat) in enumerate(tile_centers):
        logging.info(f"  Downloading adjacent tile {i}")
        
        elevation_file = download_elevation_data(tile_lon, tile_lat, buffer_m)
        if elevation_file:
            elevation_files.append(elevation_file)
        else:
            logging.warning(f"  Failed to download tile {i+1}, continuing with available tiles")
    
    logging.info(f"Successfully downloaded {len(elevation_files)}/{len(tile_centers)} elevation tiles")
    return elevation_files

def merge_elevation_tiles(elevation_files: List[Path]) -> Optional[Tuple[np.ndarray, object, object]]:
    """Merge multiple elevation TIFF files into a single elevation array
    
    Returns (merged_elevation_data, merged_transform, crs) or None if merging fails.
    """
    if not elevation_files:
        logging.error("No elevation files to merge")
        return None
    
    if len(elevation_files) == 1:
        # Single tile, just return it
        with rasterio.open(elevation_files[0]) as src:
            return src.read(1), src.transform, src.crs
    
    logging.info(f"Merging {len(elevation_files)} elevation tiles...")
    
    datasets = []
    try:
        # Open all files and collect their data
        for file_path in elevation_files:
            datasets.append(rasterio.open(file_path))
        
        # Use rasterio.merge to combine the tiles
        from rasterio.merge import merge
        
        merged_data, merged_transform = merge(datasets)
        
        # Get CRS from first dataset (should be consistent)
        merged_crs = datasets[0].crs
        
        # Close all datasets
        for ds in datasets:
            ds.close()
        
        # Merge returns a 3D array, we want the first band
        if merged_data.ndim == 3:
            merged_elevation = merged_data[0]
        else:
            merged_elevation = merged_data
        
        logging.info(f"Successfully merged tiles into {merged_elevation.shape} array")
        return merged_elevation, merged_transform, merged_crs
        
    except Exception as e:
        logging.error(f"Error merging elevation tiles: {e}")
        # Clean up any open datasets
        for ds in datasets:
            try:
                if not ds.closed:
                    ds.close()
            except:
                pass
        return None

def meters_to_feet(meters: float) -> float:
    """Convert meters to feet"""
    return meters * 3.28084

def feet_to_meters(feet: float) -> float:
    """Convert feet to meters"""
    return feet / 3.28084

def load_and_validate_elevation_data(elevation_file: Path, summit: Dict, elevation_tolerance_ft: float) -> Optional[Tuple]:
    """Load and validate elevation data from raster file
    
    Returns (elevation_data, transform, crs, tiff_bounds, activation_elevation, elevation_units, needs_larger_radius) or None
    """
    summit_alt_ft = summit['altFt']
    summit_code = summit['summitCode']
    activation_alt_ft = summit_alt_ft - ACTIVATION_ZONE_HEIGHT_FT
    
    try:
        with rasterio.open(elevation_file) as src:
            # Read elevation data
            elevation_data = src.read(1)
            transform = src.transform
            crs = src.crs
            tiff_bounds = src.bounds
            
            # Calculate and log TIFF size - Convert feet to meters for display
            tiff_width_ft = tiff_bounds[2] - tiff_bounds[0]
            tiff_height_ft = tiff_bounds[3] - tiff_bounds[1]
            tiff_width_m = tiff_width_ft / 3.28084
            tiff_height_m = tiff_height_ft / 3.28084
            
            logging.debug(f"  TIFF bounds: {tiff_bounds}")
            logging.debug(f"  TIFF size: {tiff_width_m:.0f}m × {tiff_height_m:.0f}m ({tiff_width_ft:.0f}ft × {tiff_height_ft:.0f}ft) (requested: {int(TILE_RADIUS_KM * 1000 * 2)}m × {int(TILE_RADIUS_KM * 1000 * 2)}m)")
            
            # Validate data coverage
            data_min = np.nanmin(elevation_data)
            data_max = np.nanmax(elevation_data)
            valid_data_count = np.sum(~np.isnan(elevation_data))
            total_pixels = elevation_data.size
            valid_data_ratio = valid_data_count / total_pixels
            
            logging.debug(f"  Elevation data coverage: {valid_data_ratio:.1%} ({valid_data_count}/{total_pixels} pixels)")
            
            if valid_data_ratio < 0.1:
                logging.error(f"Insufficient elevation data coverage ({valid_data_ratio:.1%}) for {summit_code}")
                return None
            
            # Determine activation elevation and units
            if abs(data_max - activation_alt_ft) < 500:  # Likely feet
                activation_elevation = activation_alt_ft
                elevation_units = "ft"
                logging.debug(f"  Detected elevation data in feet, using activation elevation: {activation_elevation}ft")
            else:  # Likely meters
                activation_elevation = feet_to_meters(activation_alt_ft)
                elevation_units = "m"
                logging.debug(f"  Detected elevation data in meters, using activation elevation: {activation_elevation:.1f}m")
            
            # Validate summit elevation against raster
            needs_larger_radius = False
            if not np.isnan(data_min) and not np.isnan(data_max):
                if elevation_units == "ft":
                    raster_max_ft = data_max
                else:
                    raster_max_ft = meters_to_feet(data_max)
                
                summit_raster_diff = abs(summit_alt_ft - raster_max_ft)
                                
                if summit_raster_diff > elevation_tolerance_ft:
                    logging.warning(f"  Summit elevation ({summit_alt_ft}ft) differs from raster maximum ({raster_max_ft:.1f}ft) by {summit_raster_diff:.1f}ft")
                    logging.warning(f"  This exceeds tolerance ({elevation_tolerance_ft:.1f}ft)")
                #    return None
                
                logging.info(f"  Summit elevation validation: SOTA {summit_alt_ft}ft vs raster max {raster_max_ft:.1f}ft (diff: {summit_raster_diff:.1f}ft)")
                            
                if raster_max_ft > summit_alt_ft:
                    logging.warning(f"  Using highest maximum for activation zone calculation to ensure conservative/smallest AZ")
                    
                    corrected_activation_alt_ft = raster_max_ft - ACTIVATION_ZONE_HEIGHT_FT
                    if elevation_units == "ft":
                        activation_elevation = corrected_activation_alt_ft
                    else:
                        activation_elevation = feet_to_meters(corrected_activation_alt_ft)
                    
                    logging.warning(f"  Updated AZ elevation: {activation_elevation:.1f}{elevation_units}")
                
                logging.info(f"  Raster elevation range: {data_min:.1f}{elevation_units} to {data_max:.1f}{elevation_units}")

                if activation_elevation < data_min:
                    logging.warning(f"  Activation elevation ({activation_elevation:.1f}{elevation_units}) is below raster minimum - will need larger radius")
                    needs_larger_radius = True
            
            return elevation_data, transform, crs, tiff_bounds, activation_elevation, elevation_units, needs_larger_radius
            
    except Exception as e:
        logging.error(f"Error loading elevation data: {e}")
        return None

def create_coordinate_mesh(elevation_data: np.ndarray, transform: object) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create coordinate mesh and clean elevation data for contour generation"""
    height, width = elevation_data.shape
    
    # Transform pixel coordinates to geographic coordinates
    xs = []
    ys = []
    for i in range(height):
        for j in range(width):
            x, y = xy(transform, i, j)
            xs.append(x)
            ys.append(y)
    
    x_coords = np.array(xs).reshape(height, width)
    y_coords = np.array(ys).reshape(height, width)
    
    # Clean elevation data
    elevation_clean = elevation_data.copy().astype(float)
    # Handle potential nodata values
    elevation_clean[elevation_clean < -1000] = np.nan
    elevation_clean[np.isnan(elevation_clean)] = np.nan
    
    return x_coords, y_coords, elevation_clean

def generate_contour_polygons(x_coords: np.ndarray, y_coords: np.ndarray, 
                            elevation_clean: np.ndarray, activation_elevation: float) -> Tuple[List, List, Optional[str]]:
    """Generate contour polygons at the activation elevation
    
    Returns (polygon_shapes, contour_coords, failure_reason)
    """
    polygon_shapes = []
    all_contour_coords = []
    contour_failure_reason = None
    
    fig, ax = plt.subplots(figsize=(1, 1))
    try:
        contour_set = ax.contour(x_coords, y_coords, elevation_clean, levels=[activation_elevation])
        
        # Extract contour paths and convert to polygons
        for level_idx, level_segs in enumerate(contour_set.allsegs):
            for seg in level_segs:
                if len(seg) < 3:
                    continue
                
                all_contour_coords.extend(seg)
                
                # Close the polygon if needed
                if not np.allclose(seg[0], seg[-1]):
                    seg = np.vstack([seg, seg[0]])
                
                try:
                    poly = Polygon(seg)
                    if poly.is_valid and not poly.is_empty:
                        polygon_shapes.append(poly)
                except Exception as e:
                    logging.warning(f"Skipping invalid contour polygon: {e}")
                    continue
                    
    except Exception as e:
        contour_failure_reason = str(e)
        logging.error(f"Error generating contours: {e}")
    finally:
        plt.close(fig)
    
    return polygon_shapes, all_contour_coords, contour_failure_reason

def select_summit_containing_polygon(polygon_shapes: List, summit_lon: float, summit_lat: float, 
                                   crs: str) -> Optional[Polygon]:
    """Select the polygon that contains the summit point"""
    if not polygon_shapes:
        return None
    
    # Transform summit point to data CRS if needed
    if crs != 'EPSG:4326':
        transformer = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
        summit_x_crs, summit_y_crs = transformer.transform(summit_lon, summit_lat)
        summit_point = Point(summit_x_crs, summit_y_crs)
        logging.debug(f"  Summit point in data CRS: ({summit_x_crs:.1f}, {summit_y_crs:.1f})")
    else:
        summit_point = Point(summit_lon, summit_lat)
        logging.debug(f"  Summit point in WGS84: ({summit_lon:.6f}, {summit_lat:.6f})")
    
    # Find polygon containing summit
    for i, poly in enumerate(polygon_shapes):
        distance = poly.distance(summit_point)
        contains = poly.contains(summit_point)
        logging.debug(f"  Polygon {i+1}: distance={distance:.3f}, contains={contains}")
        
        if contains or distance < 0.001:
            logging.info(f"  Selected polygon {i+1} containing summit point")
            return poly
    
    return None

def convert_polygon_to_geojson(polygon: Polygon, summit_code: str, crs: str) -> Optional[Dict]:
    """Convert Shapely polygon to GeoJSON feature, transforming coordinates if needed"""
    try:
        # Transform to WGS84 if needed
        if crs != 'EPSG:4326':
            logging.debug(f"  Transforming coordinates from {crs} to WGS84")
            transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
            
            def transform_coords(coords):
                return [transformer.transform(x, y) for x, y in coords]
            
            if isinstance(polygon, Polygon):
                exterior_coords = transform_coords(polygon.exterior.coords)
                holes = [transform_coords(interior.coords) for interior in polygon.interiors]
                polygon = Polygon(exterior_coords, holes)
            elif isinstance(polygon, MultiPolygon):
                transformed_polygons = []
                for poly in polygon.geoms:
                    exterior_coords = transform_coords(poly.exterior.coords)
                    holes = [transform_coords(interior.coords) for interior in poly.interiors]
                    transformed_polygons.append(Polygon(exterior_coords, holes))
                polygon = MultiPolygon(transformed_polygons)
        
        # Convert to GeoJSON
        if isinstance(polygon, Polygon):
            geom_dict = {
                "type": "Polygon",
                "coordinates": [list(polygon.exterior.coords)] + 
                              [list(interior.coords) for interior in polygon.interiors]
            }
        elif isinstance(polygon, MultiPolygon):
            geom_dict = {
                "type": "MultiPolygon", 
                "coordinates": [
                    [list(poly.exterior.coords)] + [list(interior.coords) for interior in poly.interiors]
                    for poly in polygon.geoms
                ]
            }
        else:
            logging.error(f"Unsupported geometry type: {type(polygon)}")
            return None
        
        feature = {
            "type": "Feature",
            "properties": {
                "title": summit_code + " Activation Zone",
            },
            "geometry": geom_dict
        }
        
        return feature
        
    except Exception as e:
        logging.error(f"Error converting polygon to GeoJSON: {e}")
        return None

def create_activation_zone_elevation_based(summit: Dict, elevation_tolerance_ft: float = DEFAULT_ELEVATION_TOLERANCE_FT) -> Optional[Dict]:
    """Create activation zone based on actual elevation data
    
    This implements the W7O ARM definition: 'The Activation Zone is a single, "unbroken" area 
    which can be visualized by drawing a closed shape on a map, following a contour line 82
    feet below the summit point.' Other associations may have their own definitions, especially
    regarding ft vs meters.
    
    Args:
        summit: Summit data dictionary
        elevation_tolerance_ft: Tolerance in feet for validating summit elevation against raster data maximum.
    """
    summit_code = summit['summitCode']
    summit_lat = summit['latitude']
    summit_lon = summit['longitude']
    
    logging.info(f"  Summit elevation: {summit['altFt']}ft (from SOTA API)")
    logging.info(f"  Activation zone: {summit['altFt'] - ACTIVATION_ZONE_HEIGHT_FT}ft")
    
    # Get elevation file
    elevation_file = download_elevation_data(summit_lon, summit_lat, int(TILE_RADIUS_KM * 1000))
    if not elevation_file:
        logging.error(f"Failed to get elevation data for {summit_code}")
        return None
    
    # Load and validate elevation data
    result = load_and_validate_elevation_data(elevation_file, summit, elevation_tolerance_ft)
    if not result:
        return None
    
    elevation_data, transform, crs, tiff_bounds, activation_elevation, elevation_units, needs_larger_radius = result
    
    # If activation elevation is below minimum, try with 2x radius
    if needs_larger_radius:
        logging.info(f"  Retrying with 2x radius due to activation elevation below raster minimum")
        larger_elevation_file = download_elevation_data(summit_lon, summit_lat, int(TILE_RADIUS_KM * 1000 * 2))
        if larger_elevation_file:
            larger_result = load_and_validate_elevation_data(larger_elevation_file, summit, elevation_tolerance_ft)
            if larger_result:
                elevation_data, transform, crs, tiff_bounds, activation_elevation, elevation_units, still_needs_larger = larger_result
                if not still_needs_larger:
                    logging.info(f"  Successfully obtained suitable elevation data with 2x radius")
                    elevation_file = larger_elevation_file  # Use the larger file for subsequent processing
                else:
                    logging.error(f"  Activation elevation still below minimum even with 2x radius")
                    return None
            else:
                logging.error(f"  Failed to validate 2x radius elevation data")
                return None
        else:
            logging.warning(f"  Failed to download 2x radius elevation data")
            return None
    
    # Create coordinate mesh
    x_coords, y_coords, elevation_clean = create_coordinate_mesh(elevation_data, transform)
    
    # Generate initial contour polygons
    polygon_shapes, all_contour_coords, contour_failure_reason = generate_contour_polygons(
        x_coords, y_coords, elevation_clean, activation_elevation
    )
    
    # Check if contour generation failed
    if contour_failure_reason:
        logging.error(f"Initial contour generation failed for {summit_code}: {contour_failure_reason}")
        return None
    
    if not polygon_shapes:
        logging.error(f"No contour polygons generated for {summit_code}")
        return None
    
    # Check boundary distances and fetch additional tiles if needed
    boundary_analysis = calculate_distance_to_edges(all_contour_coords, tiff_bounds, str(crs))
    needed_tiles = determine_needed_adjacent_tiles(boundary_analysis, threshold_m=10.0)

    print_analysis = lambda message: logging.info(f'{message}: margins [{', '.join([f"{k}: {v:.1f}m" for k, v in boundary_analysis.items()])}]')

    if needed_tiles:
        print_analysis("Single tile contour approaches tile boundary")
        logging.info(f"  Fetching adjacent tiles: {needed_tiles}")
        
        # Download additional tiles
        additional_files = download_adjacent_elevation_tiles(
            summit_lon, summit_lat, needed_tiles, int(TILE_RADIUS_KM * 1000)
        )
        
        if additional_files:
            # Merge elevation data
            try:
                merge_result = merge_elevation_tiles([elevation_file] + additional_files)
                
                # Re-process with merged data
                if merge_result:
                    elevation_data, transform, crs = merge_result
                    
                    # Validate and process the merged data
                    data_min = np.nanmin(elevation_data)
                    data_max = np.nanmax(elevation_data)
                    summit_alt_ft = summit['altFt']
                    
                    # Determine activation elevation and units
                    if abs(data_max - summit_alt_ft) < 500:  # Likely feet
                        activation_elevation = summit_alt_ft - ACTIVATION_ZONE_HEIGHT_FT
                        elevation_units = "ft"
                    else:  # Likely meters
                        activation_elevation = feet_to_meters(summit_alt_ft - ACTIVATION_ZONE_HEIGHT_FT)
                        elevation_units = "m"
                    
                    # Get bounds for the merged data
                    height, width = elevation_data.shape
                    from rasterio.transform import array_bounds
                    tiff_bounds = array_bounds(height, width, transform)
                    
                    x_coords, y_coords, elevation_clean = create_coordinate_mesh(elevation_data, transform)
                    polygon_shapes, all_contour_coords, contour_failure_reason = generate_contour_polygons(
                        x_coords, y_coords, elevation_clean, activation_elevation
                    )
                    
                    if contour_failure_reason:
                        logging.error(f"Multi-tile contour generation failed for {summit_code}: {contour_failure_reason}")
                        return None
                    
                    if not polygon_shapes:
                        logging.error(f"No contour polygons generated from merged tiles for {summit_code}")
                        return None
                    
                    # Recalculate boundary analysis
                    boundary_analysis = calculate_distance_to_edges(all_contour_coords, tiff_bounds, str(crs))
                    print_analysis("Multi-tile boundary analysis")
                else:
                    logging.error(f"Failed to merge elevation data for {summit_code}")
                    return None
                    
            except Exception as e:
                logging.error(f"Error processing multi-tile data for {summit_code}: {e}")
                return None
        else:
            logging.warning(f"Failed to download additional tiles for {summit_code}, proceeding with single tile")
    else:
        print_analysis("  Single tile fits contour")
    
    # Select the polygon containing the summit
    selected_polygon = select_summit_containing_polygon(polygon_shapes, summit_lon, summit_lat, str(crs))
    
    if not selected_polygon:
        logging.error(f"No polygon contains summit point for {summit_code}")
        return None
    
    # Log polygon statistics
    if elevation_units == "ft":
        area_ft2 = selected_polygon.area
        area_acres = area_ft2 / 43560
        area_m2 = area_ft2 / (3.28084 ** 2)
        logging.info(f"  Polygon area: {area_acres:.2f} acres ({area_m2:.0f} m²)")
    else:
        area_m2 = selected_polygon.area
        area_acres = area_m2 * 0.000247105
        logging.info(f"  Polygon area: {area_acres:.2f} acres ({area_m2:.0f} m²)")
    
    # Convert to GeoJSON
    feature = convert_polygon_to_geojson(selected_polygon, summit_code, str(crs))
    if not feature:
        logging.error(f"Failed to convert polygon to GeoJSON for {summit_code}")
        return None
    
    logging.info(f"  Successfully created activation zone for {summit_code}")
    return feature

def process_summit(summit: Dict, elevation_tolerance_ft: float = DEFAULT_ELEVATION_TOLERANCE_FT) -> bool:
    """Process a single summit and generate activation zone GeoJSON"""
    summit_code = summit['summitCode'].replace('/', '_').replace('-', '_')  # Make filename standard
    output_file = OUTPUT_DIR / f"{summit_code}.geojson" if OUTPUT_DIR is not None else Path(f"{summit_code}.geojson")
    
    # Skip if already processed
    if output_file.exists():
        logging.info(f"Skipping {summit['summitCode']} (already exists)")
        return True
    
    feature = create_activation_zone_elevation_based(summit, elevation_tolerance_ft)
    
    if not feature:
        logging.error(f"Failed to process {summit['summitCode']}")
        return False
    
    # Create GeoJSON document
    geojson = {
        "type": "FeatureCollection", 
        "features": [feature]
    }
    
    # Save to file
    try:
        with open(output_file, 'w') as f:
            json.dump(geojson, f, indent=2)
        logging.info(f"  Saved: {output_file}")
        return True
        
    except Exception as e:
        logging.error(f"Error saving {output_file}: {e}")
        return False

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="SOTA Activation Zone Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch summit data and process activation zones
  python generate_sota_az.py --region NC
  
  # Fetch summit data only (no processing)
  python generate_sota_az.py --fetch-only --region NC
  
  # Process from summit file using file's directory for output
  python generate_sota_az.py --summits W7O_NC/input/W7O_NC.geojson
  python generate_sota_az.py --summits /path/to/summits.geojson
  
  # Process from summit file with custom output directory
  python generate_sota_az.py --summits summits.geojson --output-dir /custom/path
  
  # Custom elevation tolerance for edge cases
  python generate_sota_az.py --summits summits.geojson --elevation-tolerance 5.0
  
  # Logging options
  python generate_sota_az.py --region NC --log-file processing.log
  python generate_sota_az.py --summits summits.geojson --log-file processing.log --quiet
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '-s', '--summits',  # Short: -s, Long: --summits
        type=str,
        help='Path to summits GeoJSON file to process'
    )
    group.add_argument(
        '-a', '--association',  # Short: -a, Long: --association  
        type=str,
        help=f'SOTA association code (e.g., W7O, W7W) - default is {DEFAULT_ASSOCIATION}'
    )
    parser.add_argument(
        '-r', '--region',  # Short: -r, Long: --region
        type=str,
        required=False,
        help='SOTA region code (e.g., NC, LC) - required when fetching from SOTA API'
    )
    parser.add_argument(
        '-o', '--output-dir',  # Short: -o, Long: --output-dir
        type=str,
        help='Base directory for cache and output folders when using --summits'
    )
    parser.add_argument(
        '-f', '--fetch-only',  # Short: -f, Long: --fetch-only
        action='store_true',
        help='Only fetch summit data from SOTA and save to input directory'
    )
    parser.add_argument(
        '-l', '--log-file',  # Short: -l, Long: --log-file
        type=str,
        help='Log file path (default: output to stdout)'
    )
    parser.add_argument(
        '-q', '--quiet',  # Short: -q, Long: --quiet
        action='store_true',
        help='Suppress stdout output when using --log-file'
    )
    parser.add_argument(
        '-t', '--elevation-tolerance',  # Short: -t, Long: --elevation-tolerance
        type=float,
        default=DEFAULT_ELEVATION_TOLERANCE_FT,
        help=f'Tolerance in feet for validating summit elevation against raster data'
    )
    
    args = parser.parse_args()
    
    # Validate --quiet is only used with --log-file
    if args.quiet and not args.log_file:
        parser.error("--quiet can only be used with --log-file")
    
    # Setup logging first so we can see debug messages
    setup_logging(args.log_file, args.quiet)
    
    # Validate arguments and determine association/region
    if args.summits and args.association:
        parser.error("Cannot specify both --summits and --association")
    
    if args.output_dir and not args.summits:
        parser.error("--output-dir can only be used with --summits")

    if not args.association:
        # This script is W7O-specific
        args.association = DEFAULT_ASSOCIATION
        logging.info("Using W7O association")

    # Handle summits file processing
    if args.summits:
        summits_file = Path(args.summits)
        if not summits_file.exists():
            parser.error(f"Summit file not found: {summits_file}")
        
        # Region is no longer auto-detected, but it's not required for file processing
        # The user can optionally specify --region but it won't affect directory structure

    # Validate region is provided for API fetch mode
    if not args.summits and not args.region:
        parser.error("--region is required when fetching from SOTA API")
    
    if args.fetch_only and not args.region:
        parser.error("--fetch-only requires --region")
    
    return args

def main():
    """Main processing function"""
    args = parse_arguments()
    
    logging.info("SOTA Activation Zone Processor")
    logging.info("=" * 50)
    
    # Determine mode and setup directories
    if args.summits:
        # Process from existing GeoJSON file
        summits_file = Path(args.summits)
        
        # Setup directories based on file location or specified output directory
        setup_directories_from_input_file(summits_file, args.output_dir)
        ensure_directories()
        
        logging.info(f"Loading summits from: {summits_file}")
        summits = load_summits_from_geojson(summits_file)
        
        # Filter out retired summits
        valid_summits = [summit for summit in summits if is_summit_valid(summit)]
        retired_count = len(summits) - len(valid_summits)
        
        logging.info(f"Loaded {len(summits)} summits from file")
        if retired_count > 0:
            logging.info(f"Filtered out {retired_count} retired summits")
        logging.info(f"Processing {len(valid_summits)} active summits")
        
        summits = valid_summits
        
    else:
        # Fetch from SOTA API
        setup_regional_directories(args.association, args.region)
        ensure_directories()
        
        logging.info(f"Fetching summit data from SOTA API: {args.association}/{args.region}")
        summits = fetch_sota_summits(args.association, args.region)
        if not summits:
            logging.error("Failed to fetch summit data")
            sys.exit(1)
        logging.info(f"Done fetching {len(summits)} summits from SOTA API")

        # Save to input directory
        summits_file = save_summits_geojson(summits, args.association, args.region)
        
    if args.fetch_only:
        logging.info(f"\nStopping now due to --fetch-only! Summit data at: {summits_file}")
        logging.info(f"To process activation zones, run:")
        logging.info(f"  python {sys.argv[0]} --summits {summits_file}")
        return
    
    logging.info(f"\nProcessing {len(summits)} summits for activation zones...")
    logging.info(f"Initial tile radius: {TILE_RADIUS_KM * 1000:.0f}m")
    logging.info(f"Activation zone height: {ACTIVATION_ZONE_HEIGHT_FT} feet")
    logging.info(f"Elevation tolerance: {args.elevation_tolerance:.1f} feet")
    
    # Process each summit  
    successful = 0
    failed = 0
    failed_summits = []  # Track which summits failed
    
    for i, summit in enumerate(summits, 1):
        logging.info(f"*** [{i}/{len(summits)}] {summit['summitCode']}: {summit['name']} ***")
        
        if process_summit(summit, args.elevation_tolerance):
            successful += 1
        else:
            failed += 1
            failed_summits.append(summit['summitCode'])
    
    logging.info(f"\n=== Processing Complete ===")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    if OUTPUT_DIR:
        logging.info(f"Output directory: {OUTPUT_DIR.absolute()}")
    
    if successful > 0:
        logging.info(f"\nGenerated {successful} activation zone files")
        logging.info(f"Each file contains elevation-based activation zone boundaries")
    
    if failed > 0:
        logging.warning(f"\n{failed} summits failed processing:")
        for summit_code in failed_summits:
            logging.warning(f"  - {summit_code}")
        logging.warning(f"\nReview individual summit logs above for specific failure reasons")

if __name__ == "__main__":
    main()
