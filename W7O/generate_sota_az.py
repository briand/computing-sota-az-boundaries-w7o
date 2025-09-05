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
BUFFER_RADIUS_KM = 0.5  # From center to edge of raster tile (500m radius)
ACTIVATION_ZONE_HEIGHT_FT = 82
DEFAULT_ELEVATION_TOLERANCE_FT = 20.0  # Default tolerance for summit vs raster elevation validation

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
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    console_format = '%(levelname)s - %(message)s'
    
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
            console.setFormatter(logging.Formatter(console_format))
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
            'overall': min_dist_overall * meters_per_degree
        }
    elif is_oregon_lambert_ft:
        # EPSG:6557 uses US Survey Feet - convert to meters
        feet_to_meters = 1.0 / 3.28084
        return {
            'left': min_dist_left * feet_to_meters,
            'right': min_dist_right * feet_to_meters,
            'bottom': min_dist_bottom * feet_to_meters,
            'top': min_dist_top * feet_to_meters,
            'overall': min_dist_overall * feet_to_meters
        }
    else:
        # Assume already in projected meters
        return {
            'left': min_dist_left,
            'right': min_dist_right,
            'bottom': min_dist_bottom,
            'top': min_dist_top,
            'overall': min_dist_overall
        }

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
    if CACHE_DIR is None or INPUT_DIR is None or OUTPUT_DIR is None:
        raise ValueError("Directories not initialized. Call setup_regional_directories() first.")
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
                "elevationM": summit['altM'],
                "elevationFt": summit.get('altFt', round(summit['altM'] * 3.28084)),
                "points": summit.get('points', 0),
                "validTo": summit.get('validTo', ''),
                "validFrom": summit.get('validFrom', '')
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
    if distances['bottom'] < threshold_m:
        needed_directions.append('bottom')
    
    return needed_directions

def calculate_adjacent_tile_centers(center_lon: float, center_lat: float, buffer_m: int = 1000, 
                                   directions: Optional[List[str]] = None) -> List[Tuple[float, float]]:
    """Calculate center coordinates for adjacent tiles in specific directions
    
    Args:
        center_lon, center_lat: Center coordinates
        buffer_m: Buffer radius in meters
        directions: List of directions to fetch ('left', 'right', 'top', 'bottom')
                   If None, returns center tile only
    
    Returns a list of (lon, lat) tuples for the requested tiles including center tile.
    """
    # Calculate approximate tile size in degrees
    # Each tile is approximately 2 * buffer_m in size
    tile_size_m = buffer_m * 2
    
    lat_step = tile_size_m / 111000.0  # Latitude degrees
    lon_step = tile_size_m / (111000.0 * np.cos(np.radians(center_lat)))  # Longitude degrees
    
    # Always include center tile
    tile_centers = [(center_lon, center_lat)]
    
    # Add specific adjacent tiles based on directions
    if directions:
        for direction in directions:
            if direction == 'left':
                tile_centers.append((center_lon - lon_step, center_lat))
            elif direction == 'right':
                tile_centers.append((center_lon + lon_step, center_lat))
            elif direction == 'top':
                tile_centers.append((center_lon, center_lat + lat_step))
            elif direction == 'bottom':
                tile_centers.append((center_lon, center_lat - lat_step))
            
            # Add corner tiles if needed for diagonal coverage
            if 'left' in directions and 'top' in directions and direction in ['left', 'top']:
                if (center_lon - lon_step, center_lat + lat_step) not in tile_centers:
                    tile_centers.append((center_lon - lon_step, center_lat + lat_step))
            if 'right' in directions and 'top' in directions and direction in ['right', 'top']:
                if (center_lon + lon_step, center_lat + lat_step) not in tile_centers:
                    tile_centers.append((center_lon + lon_step, center_lat + lat_step))
            if 'left' in directions and 'bottom' in directions and direction in ['left', 'bottom']:
                if (center_lon - lon_step, center_lat - lat_step) not in tile_centers:
                    tile_centers.append((center_lon - lon_step, center_lat - lat_step))
            if 'right' in directions and 'bottom' in directions and direction in ['right', 'bottom']:
                if (center_lon + lon_step, center_lat - lat_step) not in tile_centers:
                    tile_centers.append((center_lon + lon_step, center_lat - lat_step))
    
    return tile_centers

def download_adjacent_elevation_tiles(center_lon: float, center_lat: float, 
                                     needed_directions: List[str], buffer_m: int = 1000) -> List[Path]:
    """Download elevation data for specific adjacent tiles
    
    Returns a list of paths to downloaded elevation files.
    """
    tile_centers = calculate_adjacent_tile_centers(center_lon, center_lat, buffer_m, needed_directions)
    elevation_files = []
    
    direction_desc = ', '.join(needed_directions) if needed_directions else 'center only'
    logging.info(f"Downloading targeted elevation tiles ({len(tile_centers)} tiles: center + {direction_desc})")
    
    for i, (tile_lon, tile_lat) in enumerate(tile_centers):
        if i == 0:
            logging.info(f"  Downloading center tile: {tile_lon:.6f}, {tile_lat:.6f}")
        else:
            logging.info(f"  Downloading adjacent tile {i}: {tile_lon:.6f}, {tile_lat:.6f}")
        
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

def create_activation_zone_elevation_based(summit: Dict, elevation_tolerance_ft: float = DEFAULT_ELEVATION_TOLERANCE_FT) -> Optional[Dict]:
    """Create activation zone based on actual elevation data
    
    This implements the W7O ARM definition: 'The Activation Zone is a single, "unbroken" area 
    which can be visualized by drawing a closed shape on a map, following a contour line 82
    feet below the summit point.'
    
    Args:
        summit: Summit data dictionary
        elevation_tolerance_ft: Tolerance in feet for validating summit elevation against raster data maximum.
                               Default is 3 feet.
    """
    lon = summit['longitude']
    lat = summit['latitude']
    summit_alt_ft = summit['altFt']  # Use the integer elevation from SOTA API
    summit_code = summit['summitCode']
    
    # Calculate activation zone elevation in feet (integer arithmetic)
    # SOTA API provides integer elevationFt, ACTIVATION_ZONE_HEIGHT_FT is integer (82)
    # Result is always an integer
    activation_alt_ft = summit_alt_ft - ACTIVATION_ZONE_HEIGHT_FT
    
    logging.info(f"Processing {summit_code}: {summit['name']}")
    logging.info(f"  Summit elevation: {summit_alt_ft}ft (from SOTA API)")
    logging.info(f"  Activation zone: {activation_alt_ft}ft")
    
    # Download elevation data
    elevation_file = download_elevation_data(lon, lat, int(BUFFER_RADIUS_KM * 1000))
    if not elevation_file:
        logging.error(f"Could not download elevation data for {summit_code}")
        return None
    
    # Load elevation raster
    try:
        
        with rasterio.open(elevation_file) as src:
            # Read elevation data
            elevation_data = src.read(1)
            transform = src.transform
            crs = src.crs
            
            # Get TIFF bounds for boundary analysis
            tiff_bounds = src.bounds  # (left, bottom, right, top)
            
            # Calculate actual TIFF extent 
            # EPSG:6557 (NAD83(2011) / Oregon GIC Lambert) uses US Survey Feet as units, not meters!
            tiff_width_ft = tiff_bounds[2] - tiff_bounds[0]  # right - left
            tiff_height_ft = tiff_bounds[3] - tiff_bounds[1]  # top - bottom
            
            # Convert to meters for display
            tiff_width_m = tiff_width_ft / 3.28084
            tiff_height_m = tiff_height_ft / 3.28084
            
            logging.debug(f"  TIFF bounds: {tiff_bounds}")
            logging.debug(f"  TIFF size: {tiff_width_m:.0f}m × {tiff_height_m:.0f}m ({tiff_width_ft:.0f}ft × {tiff_height_ft:.0f}ft) (requested: {int(BUFFER_RADIUS_KM * 1000 * 2)}m × {int(BUFFER_RADIUS_KM * 1000 * 2)}m)")
            
            # Detect elevation data units by examining the data range
            data_min = np.nanmin(elevation_data)
            data_max = np.nanmax(elevation_data)
            
            if np.isnan(data_min) or np.isnan(data_max):
                logging.error(f"Invalid raster elevation data range: {data_min} to {data_max}")
                return None

            # Check if we have valid elevation data
            valid_data_count = np.sum(~np.isnan(elevation_data))
            total_pixels = elevation_data.size
            valid_data_ratio = valid_data_count / total_pixels
            
            logging.debug(f"  Elevation data coverage: {valid_data_ratio:.1%} ({valid_data_count}/{total_pixels} pixels)")
            
            if valid_data_ratio < 0.1:  # Less than 10% valid data
                logging.error(f"Insufficient elevation data coverage ({valid_data_ratio:.1%}) for {summit_code}")
                return None
            
            # Determine activation elevation in the appropriate units for the raster data
            # If the elevation data appears to be in feet (SOTA alt is close),
            # use our pre-calculated activation elevation in feet, otherwise convert to meters
            if abs(data_max - activation_alt_ft) < 500:  # Likely feet
                activation_elevation = activation_alt_ft  # Use feet directly
                elevation_units = "ft"
                logging.debug(f"  Detected elevation data in feet, using activation elevation: {activation_elevation}ft")
            else:  # Likely meters
                activation_elevation = feet_to_meters(activation_alt_ft)  # Convert to meters
                elevation_units = "m"
                logging.info(f"  Detected elevation data in meters, using activation elevation: {activation_elevation:.1f}m")
            
            # Generate contour lines at the activation elevation
            # This creates a "closed shape following a contour line 82 feet below the summit point"
            # as specified in the W7O ARM definition
            
            # Create coordinate arrays for the elevation data
            height, width = elevation_data.shape
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            
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
            
            # Replace nodata values with NaN
            elevation_clean = elevation_data.copy().astype(float)
            if hasattr(src, 'nodata') and src.nodata is not None:
                elevation_clean[elevation_data == src.nodata] = np.nan
            
            # Validate elevation data and summit elevation relationship
            # Convert raster elevation range to feet for comparison with SOTA API elevation
            if elevation_units == "ft":
                raster_max_ft = data_max
                raster_min_ft = data_min
            else:  # meters
                raster_max_ft = meters_to_feet(data_max)
                raster_min_ft = meters_to_feet(data_min)
            
            # Check if summit elevation matches raster maximum (within tolerance)
            summit_raster_diff = abs(summit_alt_ft - raster_max_ft)
            
            if summit_raster_diff > elevation_tolerance_ft:
                logging.error(f"  Summit elevation ({summit_alt_ft}ft) differs from raster maximum ({raster_max_ft:.1f}ft) by {summit_raster_diff:.1f}ft")
                logging.error(f"  This exceeds tolerance ({elevation_tolerance_ft:.1f}ft) - data validation failed")
                return None
            else:
                logging.info(f"  Summit elevation validation: SOTA {summit_alt_ft}ft vs raster max {raster_max_ft:.1f}ft (diff: {summit_raster_diff:.1f}ft)")
                            
            # Check if raster max is higher - if so, we should use that for safety
            if raster_max_ft > summit_alt_ft:
                    logging.warning(f"  Raster maximum ({raster_max_ft:.2f}ft) is higher than SOTA db summit elevation ({summit_alt_ft}ft)")
                    logging.warning(f"  Using raster maximum for activation zone calculation to ensure conservative/smallest AZ")
                    
                    # Recalculate activation elevation based on raster maximum
                    corrected_activation_alt_ft = raster_max_ft - ACTIVATION_ZONE_HEIGHT_FT
                    
                    if elevation_units == "ft":
                        activation_elevation = corrected_activation_alt_ft
                    else:  # meters
                        activation_elevation = feet_to_meters(corrected_activation_alt_ft)
                    
                    logging.warning(f"  Corrected activation elevation: {corrected_activation_alt_ft}ft ({activation_elevation:.1f}{elevation_units})")

            # The AZ is of course below the summit, but does the tile contain at least some of it?
            if elevation_units == "ft":
                range_desc = f"{data_min:.0f}ft to {data_max:.0f}ft"
            else:  # meters
                range_desc = f"{data_min:.1f}m to {data_max:.1f}m"
            
            logging.info(f"  Raster elevation range: {range_desc}")

            if activation_elevation < data_min:
                logging.warning(f"  Activation elevation ({activation_elevation:.1f}{elevation_units}) is below raster minimum - expanding tileset to cover")
                    
            # Generate contour lines at the activation elevation
            # Try single tile first, then multi-tile if contours are too close to boundaries
            
            polygon_shapes = []
            all_contour_coords = []
            contour_failure_reason = None
            needs_multi_tile = False
            needed_directions = []
            
            # First attempt: single tile contour generation
            fig, ax = plt.subplots(figsize=(1, 1))  # Small figure to save memory
            try:
                contour_set = ax.contour(x_coords, y_coords, elevation_clean, levels=[activation_elevation])
                
                # Extract contour paths and convert to polygons
                polygon_shapes = []
                all_contour_coords = []
                
                # Access contour segments from allsegs
                for level_idx, level_segs in enumerate(contour_set.allsegs):
                    for seg in level_segs:
                        if len(seg) < 3:
                            continue  # Skip invalid segments
                        
                        # Store coordinates for boundary analysis
                        all_contour_coords.extend(seg)
                        
                        # Close the polygon if it's not already closed
                        if not np.allclose(seg[0], seg[-1]):
                            seg = np.vstack([seg, seg[0]])
                        
                        # Create Shapely polygon
                        try:
                            poly = Polygon(seg)
                            if poly.is_valid and not poly.is_empty:
                                polygon_shapes.append(poly)
                        except Exception as e:
                            logging.warning(f"Skipping invalid contour polygon: {e}")
                            continue
                
                # Analyze contour boundary distances if we found contours
                if all_contour_coords:
                    distances = calculate_distance_to_edges(all_contour_coords, tiff_bounds, str(crs))
                    logging.debug(f"  Contour distances to TIFF edges - Overall: {distances['overall']:.0f}m, "
                               f"Left: {distances['left']:.0f}m, Right: {distances['right']:.0f}m, "
                               f"Bottom: {distances['bottom']:.0f}m, Top: {distances['top']:.0f}m")
                    
                    # Check if we need adjacent tiles (contour within 10m of any edge)
                    needed_directions = determine_needed_adjacent_tiles(distances, threshold_m=10.0)
                    if needed_directions:
                        needs_multi_tile = True
                        direction_desc = ', '.join(needed_directions)
                        logging.warning(f"  Contour very close to TIFF boundary - closest edges: {direction_desc} "
                                      f"(distances: {distances['overall']:.0f}m) - will fetch adjacent tiles")
                    elif distances['overall'] > 0.5 * BUFFER_RADIUS_KM:  # Far from edge (> half the radius)
                        logging.info(f"  Contour well within tile boundary ({distances['overall']:.0f}m from edge)")
                    else:
                        logging.info(f"  Contour reasonably within tile boundary ({distances['overall']:.0f}m from edge)")

            except Exception as e:
                contour_failure_reason = str(e)
                logging.error(f"Error generating contours: {e}")
                
                # Check if failure might be due to data extent issues
                summit_in_bounds = (tiff_bounds[0] <= lon <= tiff_bounds[2] and 
                                  tiff_bounds[1] <= lat <= tiff_bounds[3])
                if not summit_in_bounds:
                    logging.error(f"  Summit point ({lon:.6f}, {lat:.6f}) is outside TIFF bounds {tiff_bounds}")
                    contour_failure_reason += " - Summit outside elevation data"
                    needs_multi_tile = True  # Try multi-tile for summit outside bounds
                    needed_directions = ['left', 'right', 'top', 'bottom']  # Try all directions for summit outside bounds
                polygon_shapes = []
            finally:
                plt.close(fig)  # Clean up
            
            # Second attempt: targeted adjacent tiles if needed
            if needs_multi_tile:
                logging.info(f"  Attempting targeted adjacent tile elevation data fetch...")
                
                # Download only the needed adjacent tiles
                elevation_files = download_adjacent_elevation_tiles(lon, lat, needed_directions, int(BUFFER_RADIUS_KM * 1000))
                
                if len(elevation_files) >= 2:  # Need at least center + 1 adjacent tile
                    # Merge the tiles
                    merge_result = merge_elevation_tiles(elevation_files)
                    
                    if merge_result:
                        elevation_data, transform, crs = merge_result
                        
                        # Get new TIFF bounds from merged data
                        height, width = elevation_data.shape
                        
                        # Calculate bounds using rasterio transform functions
                        from rasterio.transform import array_bounds
                        tiff_bounds = array_bounds(height, width, transform)
                        
                        # Calculate new TIFF size for logging
                        tiff_width_units = tiff_bounds[2] - tiff_bounds[0]
                        tiff_height_units = tiff_bounds[3] - tiff_bounds[1]
                        
                        if str(crs).upper() in ['EPSG:6557']:  # Oregon Lambert in feet
                            tiff_width_m = tiff_width_units / 3.28084
                            tiff_height_m = tiff_height_units / 3.28084
                            logging.info(f"  Adjacent-tile TIFF size: {tiff_width_m:.0f}m × {tiff_height_m:.0f}m ({tiff_width_units:.0f}ft × {tiff_height_units:.0f}ft)")
                        else:
                            logging.info(f"  Adjacent-tile TIFF size: {tiff_width_units:.0f}m × {tiff_height_units:.0f}m")
                        
                        # Recalculate coordinate arrays for merged data
                        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
                        
                        xs = []
                        ys = []
                        for i in range(height):
                            for j in range(width):
                                x, y = xy(transform, i, j)
                                xs.append(x)
                                ys.append(y)
                        
                        x_coords = np.array(xs).reshape(height, width)
                        y_coords = np.array(ys).reshape(height, width)
                        
                        # Replace nodata values with NaN
                        elevation_clean = elevation_data.copy().astype(float)
                        # Note: merged data may not have consistent nodata values, rely on NaN masking
                        elevation_clean[np.isnan(elevation_clean)] = np.nan
                        elevation_clean[elevation_clean < -1000] = np.nan  # Remove obvious invalid values
                        
                        # Re-attempt contour generation with merged data
                        fig, ax = plt.subplots(figsize=(1, 1))
                        multi_tile_polygon_shapes = []
                        multi_tile_contour_coords = []
                        multi_tile_failure_reason = None
                        
                        try:
                            contour_set = ax.contour(x_coords, y_coords, elevation_clean, levels=[activation_elevation])
                            
                            # Extract contour paths and convert to polygons
                            for level_idx, level_segs in enumerate(contour_set.allsegs):
                                for seg in level_segs:
                                    if len(seg) < 3:
                                        continue
                                    
                                    multi_tile_contour_coords.extend(seg)
                                    
                                    if not np.allclose(seg[0], seg[-1]):
                                        seg = np.vstack([seg, seg[0]])
                                    
                                    try:
                                        poly = Polygon(seg)
                                        if poly.is_valid and not poly.is_empty:
                                            multi_tile_polygon_shapes.append(poly)
                                    except Exception as e:
                                        logging.warning(f"Skipping invalid adjacent-tile contour polygon: {e}")
                                        continue
                            
                            # Analyze new contour boundary distances
                            if multi_tile_contour_coords:
                                distances = calculate_distance_to_edges(multi_tile_contour_coords, tiff_bounds, str(crs))
                                logging.info(f"  Adjacent-tile contour distance to boundary: {distances['overall']:.0f}m")
                                if distances['overall'] < 10:
                                    logging.warning(f"  Adjacent-tile contour still very close to boundary ({distances['overall']:.0f}m)")
                                else:
                                    logging.info(f"  Adjacent-tile contour well positioned ({distances['overall']:.0f}m from edge)")
                            
                            # Use multi-tile polygons if they're better (contain summit and are valid)
                            if multi_tile_polygon_shapes:
                                # Check if multi-tile polygons contain the summit
                                # Transform summit point to the same CRS as the elevation data and contours
                                if str(crs) != 'EPSG:4326':
                                    transformer = Transformer.from_crs('EPSG:4326', str(crs), always_xy=True)
                                    summit_x_crs, summit_y_crs = transformer.transform(lon, lat)
                                    summit_point = Point(summit_x_crs, summit_y_crs)
                                else:
                                    summit_point = Point(lon, lat)
                                
                                # Find multi-tile polygon containing summit
                                for i, poly in enumerate(multi_tile_polygon_shapes):
                                    if poly.contains(summit_point) or poly.distance(summit_point) < 0.001:
                                        logging.info(f"  Using adjacent-tile polygon (better boundary clearance)")
                                        polygon_shapes = multi_tile_polygon_shapes
                                        all_contour_coords = multi_tile_contour_coords
                                        break
                        
                        except Exception as e:
                            multi_tile_failure_reason = str(e)
                            logging.error(f"Error generating adjacent-tile contours: {e}")
                        finally:
                            plt.close(fig)
                        
                        if len(multi_tile_polygon_shapes) > 0:
                            logging.info(f"  Adjacent-tile attempt successful: found {len(multi_tile_polygon_shapes)} contour polygon(s)")
                        else:
                            logging.warning(f"  Adjacent-tile attempt did not improve results")
                    else:
                        logging.error(f"  Failed to merge elevation tiles")
                else:
                    logging.error(f"  Insufficient elevation tiles downloaded ({len(elevation_files)} tiles)")
            
            if not polygon_shapes:
                error_msg = f"No valid contour polygons found at activation elevation for {summit_code}"
                if contour_failure_reason:
                    error_msg += f" - Reason: {contour_failure_reason}"
                logging.error(error_msg)
                return None
            
            logging.info(f"  Found {len(polygon_shapes)} contour polygon(s)")
            
            # Find the polygon that contains the summit point
            # This ensures we get the "unbroken area" that includes the summit point
            # as specified in the ARM definition
            
            # Transform summit point to the same CRS as the elevation data and contours
            if crs != 'EPSG:4326':
                transformer = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
                summit_x_crs, summit_y_crs = transformer.transform(lon, lat)
                summit_point = Point(summit_x_crs, summit_y_crs)
                logging.debug(f"  Summit point in data CRS: ({summit_x_crs:.1f}, {summit_y_crs:.1f})")
            else:
                summit_point = Point(lon, lat)
                logging.debug(f"  Summit point in WGS84: ({lon:.6f}, {lat:.6f})")
            
            activation_geometry = None
            
            # First try to find a polygon that contains the summit
            for i, poly in enumerate(polygon_shapes):
                distance = poly.distance(summit_point)
                contains = poly.contains(summit_point)
                logging.debug(f"  Polygon {i+1}: distance={distance:.3f}, contains={contains}")
                
                if contains or distance < 0.001:  # Small tolerance
                    activation_geometry = poly
                    logging.info(f"  Selected polygon {i+1} containing summit point")
                    break
            
            # If no polygon contains the summit, fail this summit
            if activation_geometry is None:
                logging.error(f"Summit not contained in any contour polygon - activation zone definition not met")
                logging.error(f"Found {len(polygon_shapes)} contour(s) but none contain the summit point")
                return None
            
            num_coords = len(activation_geometry.exterior.coords)
            logging.info(f"  Generated contour polygon with {num_coords} coordinate points")
            
            # Transform to WGS84 if needed
            if crs != 'EPSG:4326':
                logging.info(f"  Transforming coordinates from {crs} to WGS84")
                transformer = Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
                
                def transform_coords(coords):
                    return [transformer.transform(x, y) for x, y in coords]
                
                if isinstance(activation_geometry, Polygon):
                    exterior_coords = transform_coords(activation_geometry.exterior.coords)
                    holes = [transform_coords(interior.coords) for interior in activation_geometry.interiors]
                    activation_geometry = Polygon(exterior_coords, holes)
                elif isinstance(activation_geometry, MultiPolygon):
                    transformed_polygons = []
                    for poly in activation_geometry.geoms:
                        exterior_coords = transform_coords(poly.exterior.coords)
                        holes = [transform_coords(interior.coords) for interior in poly.interiors]
                        transformed_polygons.append(Polygon(exterior_coords, holes))
                    activation_geometry = MultiPolygon(transformed_polygons)
            
            # Convert to GeoJSON
            geom_dict = None
            if isinstance(activation_geometry, Polygon):
                geom_dict = {
                    "type": "Polygon",
                    "coordinates": [list(activation_geometry.exterior.coords)] + 
                                  [list(interior.coords) for interior in activation_geometry.interiors]
                }
            elif isinstance(activation_geometry, MultiPolygon):
                geom_dict = {
                    "type": "MultiPolygon", 
                    "coordinates": [
                        [list(poly.exterior.coords)] + [list(interior.coords) for interior in poly.interiors]
                        for poly in activation_geometry.geoms
                    ]
                }
            
            if geom_dict is None:
                logging.error(f"Could not create geometry for {summit_code}")
                return None
            
            feature = {
                "type": "Feature",
                "properties": {
                    "title": summit['summitCode'] + " Activation Zone",
                },
                "geometry": geom_dict
            }
            
            logging.info(f"  Successfully created elevation-based activation zone for {summit_code}")
            return feature
            
    except Exception as e:
        logging.error(f"Error processing elevation data for {summit_code}: {str(e)}")
        return None

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
        logging.error(f"Failed to process {summit['summitCode']} - could not retrieve elevation data")
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
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="SOTA Activation Zone Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From W7O directory - association auto-detected, region required
  python process_sota_az.py --fetch-only --region NC
  python process_sota_az.py --summits W7O_NC/input/W7O_NC.geojson --region NC
  python process_sota_az.py --region NC
  
  # Explicit association and region
  python process_sota_az.py --association W7O --region NC
  python process_sota_az.py --association W7W --region LC --fetch-only
  
  # Custom elevation tolerance for edge cases
  python process_sota_az.py --region NC --elevation-tolerance 5.0
  
  # Logging options
  python process_sota_az.py --region NC --log-file processing.log
  python process_sota_az.py --region NC --log-file processing.log --quiet
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--summits',
        type=str,
        help='Path to summits GeoJSON file to process'
    )
    group.add_argument(
        '--association',
        type=str,
        help='SOTA association code (e.g., W7O, W7W) - if not provided, will be derived from current directory'
    )
    
    parser.add_argument(
        '--region',
        type=str,
        required=True,
        help='SOTA region code (e.g., NC, LC) - always required'
    )
    
    parser.add_argument(
        '--fetch-only',
        action='store_true',
        help='Only fetch summit data from SOTA and save to input directory, do not process activation zones'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path (default: output to stdout)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress stdout output when using --log-file (file logging only)'
    )
    
    parser.add_argument(
        '--elevation-tolerance',
        type=float,
        default=DEFAULT_ELEVATION_TOLERANCE_FT,
        help=f'Tolerance in feet for validating summit elevation against raster data (default: {DEFAULT_ELEVATION_TOLERANCE_FT})'
    )
    
    args = parser.parse_args()
    
    # Validate --quiet is only used with --log-file
    if args.quiet and not args.log_file:
        parser.error("--quiet can only be used with --log-file")
    
    # Setup logging first so we can see debug messages
    setup_logging(args.log_file, args.quiet)
    
    # Validate arguments and determine association
    if args.summits and args.association:
        parser.error("Cannot specify both --summits and --association")
    
    if not args.summits and not args.association:
        # Try to derive association from current working directory
        current_dir = Path.cwd().name
        if current_dir in ['W7O', 'W7W', 'W6', 'W0C']:  # Common SOTA associations
            args.association = current_dir
            logging.info(f"Derived association '{current_dir}' from current directory")
        else:
            parser.error("Either --summits or --association is required when current directory doesn't match a known SOTA association (W7O, W7W, etc.)")
    
    if args.summits:
        # When processing from file, we need region and optionally association
        if not args.association:
            # Try to derive association from current working directory
            current_dir = Path.cwd().name
            if current_dir in ['W7O', 'W7W', 'W6', 'W0C']:  # Common SOTA associations
                args.association = current_dir
                logging.info(f"Derived association '{current_dir}' from current directory")
            else:
                parser.error("--association is required when current directory doesn't match a known SOTA association (W7O, W7W, etc.)")
    
    if args.fetch_only and not args.association:
        parser.error("--fetch-only requires --association and --region")
    
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
        if not summits_file.exists():
            logging.error(f"Summit file not found: {summits_file}")
            sys.exit(1)
        
        # Use the provided association and region
        setup_regional_directories(args.association, args.region)
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
        
        # Save to input directory
        summits_file = save_summits_geojson(summits, args.association, args.region)
        
        if args.fetch_only:
            logging.info(f"\nFetch complete! Summit data saved to: {summits_file}")
            logging.info(f"To process activation zones, run:")
            logging.info(f"  python {sys.argv[0]} --summits {summits_file} --region {args.region}")
            return
    
    logging.info(f"\nProcessing {len(summits)} summits for activation zones...")
    logging.info(f"Buffer radius: {BUFFER_RADIUS_KM * 1000:.0f}m")
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
