"""
GIS operations for elevation data processing and imageserver communication.
"""
import json
import requests
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlencode
import time

from config import (
    IMAGESERVER_URL, TILE_RADIUS, AZ_HEIGHT, 
    CACHE_DIR, AZ_ELEVATION_UNITS, RASTER_ELEVATION_UNITS
)

# Global variable to track if we've checked the server
_raster_units_initialized = False

try:
    import rasterio
    from rasterio.merge import merge
    from rasterio.transform import from_bounds
    from rasterio import transform as rio_transform
    import rasterio.mask
    from shapely.geometry import Point, Polygon
    import matplotlib.pyplot as plt
    from matplotlib import path
    import skimage.measure
except ImportError as e:
    logging.error(f"Missing required GIS dependencies: {e}")
    raise


def get_imageserver_metadata_file() -> Path:
    """Get path to cached imageserver metadata file."""
    from config import CACHE_DIR
    if CACHE_DIR is None:
        raise ValueError("CACHE_DIR must be set before accessing metadata")
    return CACHE_DIR / "imageserver.json"


def fetch_and_cache_imageserver_metadata() -> Optional[Dict]:
    """
    Fetch metadata from the imageserver and cache it locally.
    
    Returns:
        Metadata dictionary or None if fetch fails
    """
    metadata_file = get_imageserver_metadata_file()
    
    # Try to load from cache first
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            logging.debug("Loaded imageserver metadata from cache")
            return metadata
        except Exception as e:
            logging.warning(f"Error loading cached metadata: {e}")
    
    # Fetch fresh metadata
    try:
        url = f"{IMAGESERVER_URL}?f=json"
        logging.info(f"Fetching imageserver metadata: {url}")
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        metadata = response.json()
        
        # Cache the metadata
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logging.debug(f"Cached imageserver metadata to: {metadata_file}")
        except Exception as e:
            logging.warning(f"Failed to cache metadata: {e}")
        
        return metadata
        
    except Exception as e:
        logging.error(f"Error fetching imageserver metadata: {e}")
        return None


def determine_raster_units_from_metadata(metadata: Optional[Dict]) -> str:
    """
    Determine elevation units from imageserver metadata using spatial reference (WKID).
    
    Args:
        metadata: Imageserver metadata dictionary
        
    Returns:
        Units string ('m' for meters, 'ft' for feet)
    """
    if not metadata:
        logging.info("No ImageServer metadata available, defaulting to feet (Oregon W7O standard)")
        return "ft"
    
    try:
        # For Oregon ImageServer specifically, check if this is the DOGAMI/Oregon LiDAR service
        service_url = IMAGESERVER_URL.lower() if IMAGESERVER_URL else ""
        if "dogami.oregon.gov" in service_url or "oregon" in service_url:
            logging.info("Detected Oregon DOGAMI ImageServer - using feet for elevation values")
            return "ft"
        
        # Look for explicit elevation unit information first
        for key in ['units', 'elevationUnits', 'verticalUnits']:
            if key in metadata:
                unit_info = str(metadata[key]).lower()
                if 'feet' in unit_info or 'foot' in unit_info or 'ft' in unit_info:
                    logging.info(f"Found explicit feet units in metadata field '{key}'")
                    return 'ft'
                elif 'meter' in unit_info or 'metre' in unit_info:
                    logging.info(f"Found explicit meter units in metadata field '{key}'")
                    return 'm'
        
        # Look for spatial reference information (coordinate system, not necessarily elevation units)
        spatial_ref = metadata.get('spatialReference', {})
        wkid = spatial_ref.get('wkid')
        latest_wkid = spatial_ref.get('latestWkid')
        
        # Check both wkid and latestWkid
        wkids_to_check = [wkid, latest_wkid] if latest_wkid else [wkid]
        
        for check_wkid in wkids_to_check:
            if check_wkid:
                logging.debug(f"ImageServer spatial reference WKID: {check_wkid}")
                
                # Oregon coordinate systems and their units
                feet_systems = [6557, 2269, 2270, 3644, 3645, 102970]  # Oregon State Plane systems use feet
                meter_systems = [32610, 32611, 26910, 26911]  # UTM systems use meters
                
                if check_wkid in feet_systems:
                    logging.info(f"Detected feet-based coordinate system (WKID: {check_wkid})")
                    return 'ft'
                elif check_wkid in meter_systems:
                    logging.info(f"Detected meter-based coordinate system (WKID: {check_wkid})")
                    return 'm'
                elif check_wkid in [3857, 102100]:  # Web Mercator
                    # Web Mercator coordinates are in meters, but elevation data can be in feet
                    # For Oregon, elevation data is typically in feet even in Web Mercator
                    logging.info(f"Detected Web Mercator (WKID: {check_wkid}), checking service for elevation units")
                    if "dogami.oregon.gov" in service_url or "oregon" in service_url:
                        logging.info("Oregon elevation data in Web Mercator - using feet")
                        return 'ft'
                    else:
                        logging.info("Non-Oregon elevation data in Web Mercator - using meters")
                        return 'm'
                elif check_wkid == 4326:
                    logging.info(f"Detected geographic coordinate system (WKID: {check_wkid}), assuming meters for elevation")
                    return 'm'
        
        # Check various other possible unit indicators in metadata
        unit_indicators = [
            metadata.get("pixelType"),
            metadata.get("description", "").lower()
        ]
        
        for indicator in unit_indicators:
            if not indicator:
                continue
                
            indicator_str = str(indicator).lower()
            if "feet" in indicator_str or "ft" in indicator_str:
                logging.info("Detected elevation units: feet")
                return "ft"
            elif "meter" in indicator_str or "metre" in indicator_str:
                logging.info("Detected elevation units: meters")
                return "m"
        
        # Default for Oregon W7O (feet-based systems are standard)
        logging.info("Could not determine units from metadata, defaulting to feet (Oregon W7O standard)")
        return "ft"
        
    except Exception as e:
        logging.error(f"Error determining raster units from metadata: {e}")
        logging.info("Defaulting to feet (Oregon W7O standard)")
        return "ft"


def initialize_raster_units() -> None:
    """Initialize raster elevation units from imageserver metadata."""
    global _raster_units_initialized, RASTER_ELEVATION_UNITS
    
    if _raster_units_initialized:
        return
    
    metadata = fetch_and_cache_imageserver_metadata()
    RASTER_ELEVATION_UNITS = determine_raster_units_from_metadata(metadata)
    _raster_units_initialized = True
    
    logging.info(f"Raster elevation units: {RASTER_ELEVATION_UNITS}")


def reset_raster_units() -> None:
    """Reset raster units initialization to force re-detection."""
    global _raster_units_initialized
    _raster_units_initialized = False


def build_imageserver_query_url(lon: float, lat: float, buffer_m: int = 1000) -> str:
    """
    Build URL for querying elevation data from imageserver.
    
    Args:
        lon: Longitude in decimal degrees
        lat: Latitude in decimal degrees  
        buffer_m: Buffer distance in meters
        
    Returns:
        Complete imageserver query URL
    """
    # Convert buffer from meters to degrees (rough approximation)
    # 1 degree ≈ 111,000 meters at equator
    buffer_deg = buffer_m / 111000.0
    
    bbox = f"{lon - buffer_deg},{lat - buffer_deg},{lon + buffer_deg},{lat + buffer_deg}"
    
    params = {
        'bbox': bbox,
        'bboxSR': '4326',  # WGS84
        'size': '512,512',  # Image dimensions
        'imageSR': '4326',  # Output spatial reference
        'format': 'tiff',
        'pixelType': 'F32',  # 32-bit float
        'noDataInterpretation': 'esriNoDataMatchAny',
        'interpolation': '+RSP_BilinearInterpolation',
        'f': 'image'
    }
    
    return f"{IMAGESERVER_URL}/exportImage?{urlencode(params)}"


def get_cached_elevation_file(lon: float, lat: float, buffer_m: int = 1000, summit_code: Optional[str] = None) -> Path:
    """Get the cached filename for elevation data at given coordinates."""
    from config import CACHE_DIR, SOTA_ASSOCIATION, CURRENT_REGION
    if CACHE_DIR is None:
        raise ValueError("CACHE_DIR must be set before accessing cached files")
    
    # Include association, region, and summit code in cache filename for context
    if CURRENT_REGION and summit_code:
        # Extract just the summit number from the full summit code (e.g., "W7O/NC-001" -> "001")
        summit_num = summit_code.split('-')[-1] if '-' in summit_code else summit_code
        filename = f"{SOTA_ASSOCIATION}_{CURRENT_REGION}_{summit_num}_elev_{lon:.6f}_{lat:.6f}_{buffer_m}m.tif"
    elif CURRENT_REGION:
        filename = f"{SOTA_ASSOCIATION}_{CURRENT_REGION}_elev_{lon:.6f}_{lat:.6f}_{buffer_m}m.tif"
    else:
        # Fallback to old naming if region not set
        filename = f"elev_{lon:.6f}_{lat:.6f}_{buffer_m}m.tif"
    
    return CACHE_DIR / filename


def download_elevation_data(lon: float, lat: float, buffer_m: int = 1000, summit_code: Optional[str] = None) -> Optional[Path]:
    """
    Download elevation data for a given coordinate and buffer distance.
    
    Args:
        lon: Longitude in decimal degrees
        lat: Latitude in decimal degrees
        buffer_m: Buffer distance in meters
        summit_code: Optional summit code for cache file naming
        
    Returns:
        Path to downloaded elevation file or None if download fails
    """
    # Check cache first
    cached_file = get_cached_elevation_file(lon, lat, buffer_m, summit_code)
    if cached_file.exists():
        logging.debug(f"Using cached elevation data: {cached_file}")
        return cached_file
    
    # Download new data
    url = build_imageserver_query_url(lon, lat, buffer_m)
    logging.info(f"Downloading elevation data: {lon:.6f}, {lat:.6f}")
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Save to cache
        with open(cached_file, 'wb') as f:
            f.write(response.content)
        
        logging.info(f"  Saved elevation data: {cached_file}")
        return cached_file
        
    except Exception as e:
        logging.error(f"Error downloading elevation data: {e}")
        return None


def determine_needed_adjacent_tiles(distances: Dict[str, float], threshold_m: float = 10.0) -> List[str]:
    """
    Determine which adjacent tiles are needed based on distance to edges.
    
    Args:
        distances: Dictionary of edge distances {'north': dist, 'south': dist, 'east': dist, 'west': dist}
        threshold_m: Distance threshold in meters
        
    Returns:
        List of needed directions
    """
    needed = []
    
    if distances['north'] < threshold_m:
        needed.append('north')
    if distances['south'] < threshold_m:
        needed.append('south')
    if distances['east'] < threshold_m:
        needed.append('east')
    if distances['west'] < threshold_m:
        needed.append('west')
    
    # Handle corners
    if 'north' in needed and 'east' in needed:
        needed.append('northeast')
    if 'north' in needed and 'west' in needed:
        needed.append('northwest')
    if 'south' in needed and 'east' in needed:
        needed.append('southeast')
    if 'south' in needed and 'west' in needed:
        needed.append('southwest')
    
    return needed


def calculate_adjacent_tile_centers(center_lon: float, center_lat: float, buffer_m: int, 
                                   needed_directions: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Calculate the center coordinates for adjacent tiles.
    
    Args:
        center_lon: Center longitude
        center_lat: Center latitude
        buffer_m: Buffer distance in meters
        needed_directions: List of needed tile directions
        
    Returns:
        Dictionary mapping direction to (lon, lat) coordinates
    """
    # Convert buffer to degrees (rough approximation)
    buffer_deg = (buffer_m * 2) / 111000.0  # Full tile width
    
    adjacent_centers = {}
    
    direction_offsets = {
        'north': (0, buffer_deg),
        'south': (0, -buffer_deg),
        'east': (buffer_deg, 0),
        'west': (-buffer_deg, 0),
        'northeast': (buffer_deg, buffer_deg),
        'northwest': (-buffer_deg, buffer_deg),
        'southeast': (buffer_deg, -buffer_deg),
        'southwest': (-buffer_deg, -buffer_deg)
    }
    
    for direction in needed_directions:
        if direction in direction_offsets:
            lon_offset, lat_offset = direction_offsets[direction]
            adj_lon = center_lon + lon_offset
            adj_lat = center_lat + lat_offset
            adjacent_centers[direction] = (adj_lon, adj_lat)
    
    return adjacent_centers


def download_adjacent_elevation_tiles(center_lon: float, center_lat: float, 
                                    buffer_m: int, distances: Dict[str, float], summit_code: Optional[str] = None) -> List[Path]:
    """
    Download adjacent elevation tiles if needed for edge proximity.
    
    Args:
        center_lon: Center longitude
        center_lat: Center latitude
        buffer_m: Buffer distance in meters
        distances: Edge distances dictionary
        summit_code: Optional summit code for cache file naming
        
    Returns:
        List of paths to elevation files (including center tile)
    """
    elevation_files = []
    
    # Always include the center tile
    center_file = download_elevation_data(center_lon, center_lat, buffer_m, summit_code)
    if center_file:
        elevation_files.append(center_file)
    
    # Determine which adjacent tiles we need
    needed_directions = determine_needed_adjacent_tiles(distances, threshold_m=10.0)
    
    if needed_directions:
        logging.info(f"  Downloading {len(needed_directions)} adjacent tiles: {needed_directions}")
        
        adjacent_centers = calculate_adjacent_tile_centers(center_lon, center_lat, buffer_m, needed_directions)
        
        for direction, (adj_lon, adj_lat) in adjacent_centers.items():
            adj_file = download_elevation_data(adj_lon, adj_lat, buffer_m, summit_code)
            if adj_file:
                elevation_files.append(adj_file)
    
    logging.info(f"  Total elevation files: {len(elevation_files)}")
    return elevation_files


def merge_elevation_tiles(elevation_files: List[Path]) -> Optional[Tuple[np.ndarray, object, object]]:
    """
    Merge multiple elevation tiles into a single raster.
    
    Args:
        elevation_files: List of paths to elevation TIFF files
        
    Returns:
        Tuple of (merged_data, transform, crs) or None if merge fails
    """
    if not elevation_files:
        logging.error("No elevation files to merge")
        return None
    
    if len(elevation_files) == 1:
        # Single file - just load it
        try:
            with rasterio.open(elevation_files[0]) as src:
                return src.read(1), src.transform, src.crs
        except Exception as e:
            logging.error(f"Error reading single elevation file: {e}")
            return None
    
    # Multiple files - merge them
    try:
        sources = []
        for file_path in elevation_files:
            src = rasterio.open(file_path)
            sources.append(src)
        
        # Merge the rasters
        merged_data, merged_transform = merge(sources)
        
        # Get CRS from first source
        merged_crs = sources[0].crs
        
        # Close all sources
        for src in sources:
            src.close()
        
        # Extract first band if multiple bands
        if len(merged_data.shape) == 3:
            merged_data = merged_data[0]
        
        logging.info(f"Merged {len(elevation_files)} elevation tiles")
        return merged_data, merged_transform, merged_crs
        
    except Exception as e:
        logging.error(f"Error merging elevation tiles: {e}")
        return None


# === UNIT CONVERSION FUNCTIONS ===

def meters_to_feet(meters: float) -> float:
    """Convert meters to feet."""
    return meters * 3.28084


def feet_to_meters(feet: float) -> float:
    """Convert feet to meters."""
    return feet / 3.28084


def to_raster_units(value: float) -> float:
    """
    Convert a value from AZ_ELEVATION_UNITS to raster units.
    
    Args:
        value: Value in AZ_ELEVATION_UNITS
        
    Returns:
        Value converted to raster units
    """
    # Ensure raster units are initialized
    initialize_raster_units()
    
    # If units match, no conversion needed
    if AZ_ELEVATION_UNITS == RASTER_ELEVATION_UNITS:
        return value
    
    # Convert between units
    if AZ_ELEVATION_UNITS == "m" and RASTER_ELEVATION_UNITS == "ft":
        return meters_to_feet(value)
    elif AZ_ELEVATION_UNITS == "ft" and RASTER_ELEVATION_UNITS == "m":
        return feet_to_meters(value)
    else:
        logging.warning(f"Unknown unit conversion: {AZ_ELEVATION_UNITS} -> {RASTER_ELEVATION_UNITS}")
        return value


def from_raster_units(value_raster: float) -> float:
    """
    Convert a value from raster units to AZ_ELEVATION_UNITS.
    
    Args:
        value_raster: Value in raster units
        
    Returns:
        Value converted to AZ_ELEVATION_UNITS
    """
    # Ensure raster units are initialized
    initialize_raster_units()
    
    # If units match, no conversion needed
    if RASTER_ELEVATION_UNITS == AZ_ELEVATION_UNITS:
        return value_raster
    
    # Convert between units
    if RASTER_ELEVATION_UNITS == "ft" and AZ_ELEVATION_UNITS == "m":
        return feet_to_meters(value_raster)
    elif RASTER_ELEVATION_UNITS == "m" and AZ_ELEVATION_UNITS == "ft":
        return meters_to_feet(value_raster)
    else:
        logging.warning(f"Unknown unit conversion: {RASTER_ELEVATION_UNITS} -> {AZ_ELEVATION_UNITS}")
        return value_raster


# === ELEVATION DATA PROCESSING ===

def load_elevation_data(elevation_file: Path) -> Optional[Tuple[np.ndarray, object, object, Tuple[float, float, float, float]]]:
    """
    Load elevation data from a raster file.
    
    Args:
        elevation_file: Path to elevation raster file
        
    Returns:
        Tuple of (elevation_data, transform, crs, bounds) or None if load fails
    """
    try:
        with rasterio.open(elevation_file) as src:
            elevation_data = src.read(1)  # Read first band
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            
        logging.debug(f"Loaded elevation data: {elevation_data.shape}")
        return elevation_data, transform, crs, bounds
        
    except Exception as e:
        logging.error(f"Error loading elevation data from {elevation_file}: {e}")
        return None


def process_elevation_values(summit: Dict, elevation_data: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """
    Process elevation data to extract key values for summit analysis.
    
    Args:
        summit: Summit dictionary with elevation info
        elevation_data: Numpy array of elevation data
        
    Returns:
        Tuple of (summit_alt, min_elev, max_elev, mean_elev, median_elev, activation_alt)
    """
    # Get summit elevation from SOTA database - use field matching AZ_ELEVATION_UNITS
    from config import AZ_ELEVATION_UNITS
    if AZ_ELEVATION_UNITS == 'ft':
        summit_alt = float(summit['altFt'])
    else:
        summit_alt = float(summit['altM'])
    
    # Calculate raster statistics
    valid_data = elevation_data[~np.isnan(elevation_data)]
    if len(valid_data) == 0:
        logging.error(f"No valid elevation data found")
        return summit_alt, 0, 0, 0, 0, summit_alt - AZ_HEIGHT
    
    min_elev = float(np.min(valid_data))
    max_elev = float(np.max(valid_data))
    mean_elev = float(np.mean(valid_data))
    median_elev = float(np.median(valid_data))
    
    # Convert elevations from raster units to our working units
    min_elev = from_raster_units(min_elev)
    max_elev = from_raster_units(max_elev)
    mean_elev = from_raster_units(mean_elev)
    median_elev = from_raster_units(median_elev)
    
    # Calculate activation zone elevation
    activation_alt = summit_alt - AZ_HEIGHT
    
    logging.info(f"  SOTA elevation: {summit_alt:.1f}{AZ_ELEVATION_UNITS}")
    logging.info(f"  Raster elevation range: {min_elev:.1f} - {max_elev:.1f}{AZ_ELEVATION_UNITS}")
    logging.info(f"  Activation zone elevation: {activation_alt:.1f}{AZ_ELEVATION_UNITS}")
    
    return summit_alt, min_elev, max_elev, mean_elev, median_elev, activation_alt


def apply_elevation_business_logic(summit: Dict, summit_alt: float, activation_alt: float, 
                                 max_elev: float, min_elev: float, summit_lon: float, summit_lat: float,
                                 transform: object, crs: object, elevation_tolerance: float) -> Tuple[Optional[float], bool, Optional[Dict]]:
    """
    Apply business logic for elevation validation and return both validation result and raster data.
    
    Args:
        summit: Summit dictionary
        summit_alt: Summit elevation from SOTA
        activation_alt: Calculated activation zone elevation
        max_elev: Maximum elevation from raster
        min_elev: Minimum elevation from raster  
        summit_lon: Summit longitude
        summit_lat: Summit latitude
        transform: Raster transform object
        crs: Coordinate reference system
        elevation_tolerance: Maximum allowed elevation difference
        
    Returns:
        Tuple of (activation_alt_raster or None, validation_passed, raster_data_dict or None)
    """
    summit_code = summit['summitCode']
    
    # Create raster data dictionary for CSV tracking
    raster_data = {
        'raster_lon': summit_lon,
        'raster_lat': summit_lat, 
        'raster_elevation': max_elev,
        'elevation_difference': abs(summit_alt - max_elev)
    }
    
    # Calculate elevation difference and check tolerance
    summit_raster_diff = abs(summit_alt - max_elev)
    
    logging.info(f"  Elevation difference: {summit_raster_diff:.1f}{AZ_ELEVATION_UNITS}")
    
    # Check elevation tolerance - return raster data even if validation fails
    if summit_raster_diff > elevation_tolerance:
        logging.error(f"  Summit elevation differs from raster maximum by more than tolerance ({elevation_tolerance:.1f}{AZ_ELEVATION_UNITS}) for {summit_code}")
        logging.error(f"  SOTA: {summit_alt:.1f}{AZ_ELEVATION_UNITS}, Raster: {max_elev:.1f}{AZ_ELEVATION_UNITS}, Diff: {summit_raster_diff:.1f}{AZ_ELEVATION_UNITS}")
        return None, False, raster_data
    
    # Validation passed - calculate activation elevation in raster units
    activation_alt_raster = to_raster_units(activation_alt)
    
    # Check if activation zone elevation is achievable
    if activation_alt_raster < min_elev:
        logging.error(f"  Activation zone elevation ({activation_alt:.1f}{AZ_ELEVATION_UNITS}) is below minimum raster elevation ({min_elev:.1f}{AZ_ELEVATION_UNITS}) for {summit_code}")
        return None, False, raster_data
    
    logging.info(f"  Using activation elevation: {activation_alt:.1f}{AZ_ELEVATION_UNITS} ({activation_alt_raster:.1f}{RASTER_ELEVATION_UNITS})")
    return activation_alt_raster, True, raster_data


# === CONTOUR AND POLYGON PROCESSING ===

def create_coordinate_mesh(elevation_data: np.ndarray, transform: object) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create coordinate meshes for elevation data.
    
    Args:
        elevation_data: 2D numpy array of elevation values
        transform: Rasterio transform object
        
    Returns:
        Tuple of (x_coords, y_coords, elevation_data)
    """
    rows, cols = elevation_data.shape
    
    # Create pixel coordinate arrays
    col_coords, row_coords = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Transform to geographic coordinates
    x_coords, y_coords = rio_transform.xy(transform, row_coords.flatten(), col_coords.flatten())
    
    # Reshape back to 2D
    x_coords = np.array(x_coords).reshape(rows, cols)
    y_coords = np.array(y_coords).reshape(rows, cols)
    
    return x_coords, y_coords, elevation_data


def generate_contour_polygons(x_coords: np.ndarray, y_coords: np.ndarray, 
                            elevation_data: np.ndarray, activation_alt_raster: float) -> List:
    """
    Generate contour polygons at the activation elevation.
    
    Args:
        x_coords: 2D array of x coordinates
        y_coords: 2D array of y coordinates  
        elevation_data: 2D array of elevation values
        activation_alt_raster: Activation elevation in raster units
        
    Returns:
        List of polygon shapes
    """
    try:
        # Generate contours at activation elevation
        contours = skimage.measure.find_contours(elevation_data, activation_alt_raster)
        
        if not contours:
            logging.error(f"No contours found at elevation {activation_alt_raster}")
            return []
        
        # Convert contours to polygons
        from shapely.geometry import Polygon
        
        polygon_shapes = []
        for contour in contours:
            if len(contour) < 3:
                continue  # Need at least 3 points for a polygon
            
            # Convert contour indices to geographic coordinates
            contour_coords = []
            for point in contour:
                row, col = int(point[0]), int(point[1])
                if 0 <= row < x_coords.shape[0] and 0 <= col < x_coords.shape[1]:
                    x = x_coords[row, col]
                    y = y_coords[row, col]
                    contour_coords.append((x, y))
            
            if len(contour_coords) >= 3:
                try:
                    polygon = Polygon(contour_coords)
                    if polygon.is_valid:
                        polygon_shapes.append(polygon)
                except Exception as e:
                    logging.warning(f"Error creating polygon from contour: {e}")
        
        logging.info(f"Generated {len(polygon_shapes)} contour polygons")
        return polygon_shapes
        
    except Exception as e:
        logging.error(f"Error generating contour polygons: {e}")
        return []


def select_summit_containing_polygon(polygon_shapes: List, summit_lon: float, summit_lat: float, 
                                   summit_code: str) -> Optional[Polygon]:
    """
    Select the polygon that contains the summit point.
    
    Args:
        polygon_shapes: List of polygon shapes
        summit_lon: Summit longitude
        summit_lat: Summit latitude
        summit_code: Summit code for logging
        
    Returns:
        Selected polygon or None if no containing polygon found
    """
    from shapely.geometry import Point
    
    if not polygon_shapes:
        logging.error(f"No polygons available for summit {summit_code}")
        return None
    
    summit_point = Point(summit_lon, summit_lat)
    
    # Find polygon containing the summit
    for polygon in polygon_shapes:
        try:
            if polygon.contains(summit_point):
                logging.info(f"Found polygon containing summit {summit_code}")
                return polygon
        except Exception as e:
            logging.warning(f"Error checking polygon containment: {e}")
    
    # If no containing polygon, use the largest one
    largest_polygon = max(polygon_shapes, key=lambda p: p.area)
    logging.warning(f"No polygon contains summit {summit_code}, using largest polygon")
    return largest_polygon


def calculate_distance_to_edges(contour_coords: List[Tuple[float, float]], 
                               summit_lon: float, summit_lat: float) -> Dict[str, float]:
    """
    Calculate distances from summit to tile edges for determining if adjacent tiles are needed.
    
    Args:
        contour_coords: List of (lon, lat) coordinate tuples
        summit_lon: Summit longitude
        summit_lat: Summit latitude
        
    Returns:
        Dictionary with distances to each edge in meters
    """
    if not contour_coords:
        return {'north': float('inf'), 'south': float('inf'), 'east': float('inf'), 'west': float('inf')}
    
    lons = [coord[0] for coord in contour_coords]
    lats = [coord[1] for coord in contour_coords]
    
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    
    # Rough conversion from degrees to meters (1 degree ≈ 111 km)
    deg_to_m = 111000.0
    
    distances = {
        'north': (max_lat - summit_lat) * deg_to_m,
        'south': (summit_lat - min_lat) * deg_to_m,
        'east': (max_lon - summit_lon) * deg_to_m,
        'west': (summit_lon - min_lon) * deg_to_m
    }
    
    # Ensure all distances are positive
    for key in distances:
        distances[key] = max(0, distances[key])
    
    return distances
