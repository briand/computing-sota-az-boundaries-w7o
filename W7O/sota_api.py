"""
SOTA API interactions and GeoJSON processing for summit data.
"""
import json
import requests
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from config import SOTA_ASSOCIATION, SOTA_BASE_URL, INPUT_DIR


def fetch_sota_summits(region: str = "NC") -> List[Dict]:
    """
    Fetch summit data from SOTA API for a specific region.
    
    Args:
        region: SOTA region code (e.g., 'NC', 'LC')
        
    Returns:
        List of summit dictionaries from SOTA API
    """
    # SOTA_BASE_URL currently points to regions base: .../api/regions
    # Full pattern: <base>/<association>/<region>
    url = f"{SOTA_BASE_URL}/{SOTA_ASSOCIATION}/{region}"
    
    logging.info(f"Fetching SOTA summits: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # The new API returns {"region": {...}, "summits": [...]}
        if isinstance(data, dict) and 'summits' in data:
            summits = data['summits']
        elif isinstance(data, list):
            summits = data  # Fallback to old format
        else:
            logging.error(f"Unexpected API response format: {type(data)}")
            return []
            
        logging.info(f"Retrieved {len(summits)} summits from SOTA API")
        return summits
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching SOTA summits: {e}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing SOTA API response: {e}")
        return []


def save_summits_geojson(summits: List[Dict], region: str) -> Path:
    """
    Save summits to a GeoJSON file in the input directory.
    
    Args:
        summits: List of summit dictionaries from SOTA API
        region: SOTA region code
        
    Returns:
        Path to saved GeoJSON file
    """
    from config import INPUT_DIR, SOTA_ASSOCIATION
    
    if INPUT_DIR is None:
        raise ValueError("INPUT_DIR must be set before saving summits")
    
    # Use new naming convention: <association>_<region>_summits.geojson
    filename = f"{SOTA_ASSOCIATION}_{region}_summits.geojson"
    output_file = INPUT_DIR / filename
    
    # Convert summits to GeoJSON format
    features = []
    for summit in summits:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(summit['longitude']), float(summit['latitude'])]
            },
            "properties": summit
        }
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    # Save to file
    try:
        with open(output_file, 'w') as f:
            json.dump(geojson, f, indent=2)
        logging.info(f"Saved {len(summits)} summits to: {output_file}")
        return output_file
        
    except Exception as e:
        logging.error(f"Error saving summits to {output_file}: {e}")
        raise


def load_summits_from_geojson(file_path: Path) -> List[Dict]:
    """
    Load summit data from a GeoJSON file.
    
    Args:
        file_path: Path to the GeoJSON file
        
    Returns:
        List of summit dictionaries
    """
    try:
        with open(file_path, 'r') as f:
            geojson = json.load(f)
        
        summits = []
        for feature in geojson.get('features', []):
            # Extract properties (summit data) from the feature
            summit = feature.get('properties', {}).copy()
            
            # Ensure coordinates are available from geometry
            coords = feature.get('geometry', {}).get('coordinates', [])
            if len(coords) >= 2:
                summit['longitude'] = coords[0]
                summit['latitude'] = coords[1]
            
            # Normalize field names for compatibility
            if 'code' in summit and 'summitCode' not in summit:
                summit['summitCode'] = summit['code']
            
            summits.append(summit)
        
        logging.info(f"Loaded {len(summits)} summits from {file_path}")
        return summits
        
    except Exception as e:
        logging.error(f"Error loading summits from {file_path}: {e}")
        raise


def is_summit_valid(summit: Dict) -> bool:
    """
    Check if a summit is valid for processing (not retired, has required data).
    
    Args:
        summit: Summit dictionary
        
    Returns:
        True if summit is valid for processing
    """
    from datetime import datetime, timezone
    
    # Check for retired summits
    valid_from = summit.get('validFrom')
    valid_to = summit.get('validTo')
    
    # If validTo is set and is in the past, summit is retired
    if valid_to:
        try:
            # Parse the date string and check if it's in the past
            valid_to_date = datetime.fromisoformat(valid_to.replace('Z', '+00:00'))
            current_date = datetime.now(timezone.utc)
            if valid_to_date < current_date:
                logging.info(f"Summit {summit.get('summitCode', 'UNKNOWN')} is retired (validTo: {valid_to}), skipping.")
                return False
        except (ValueError, AttributeError):
            # If we can't parse the date, assume it's invalid
            logging.warning(f"Could not parse validTo date for summit {summit.get('summitCode', 'UNKNOWN')}: {valid_to}")
            return False
    
    # Check for required fields
    required_fields = ['summitCode', 'name', 'latitude', 'longitude', 'altM', 'altFt']
    for field in required_fields:
        if field not in summit or summit[field] is None:
            logging.warning(f"Summit {summit.get('summitCode', 'UNKNOWN')} missing required field: {field}")
            return False
    
    return True


def fetch_sota_association_regions(association: str) -> Dict[str, Dict[str, Any]]:
    """Fetch region metadata for a SOTA association.

    Queries the SOTA API endpoint:
        /api/associations/<association>

    Example: https://api2.sota.org.uk/api/associations/W7O

    Expected response JSON contains a key "regions" with a list of region dicts.
    Each region dict includes at least: regionCode, regionName (and possibly others).

    Returns:
        Mapping of regionCode -> full region dictionary.
        Returns empty dict on error or unexpected format (logs warnings).
    """
    # Derive association base by removing trailing '/regions' if present
    base = SOTA_BASE_URL
    if base.endswith('/regions'):
        base = base.rsplit('/regions', 1)[0]
    url = f"{base}/associations/{association}"
    logging.info(f"Fetching SOTA association regions: {url}")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching association regions for {association}: {e}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding association regions JSON for {association}: {e}")
        return {}

    if not isinstance(data, dict):
        logging.warning(f"Unexpected association regions payload type: {type(data)}")
        return {}

    regions_raw = data.get("regions")
    if not isinstance(regions_raw, list):
        logging.warning("Association regions JSON missing 'regions' list")
        return {}

    result: Dict[str, Dict[str, Any]] = {}
    for entry in regions_raw:
        if not isinstance(entry, dict):
            continue
        code = entry.get("regionCode")
        if not code:
            continue
        if code in result:
            logging.debug(f"Duplicate regionCode encountered: {code} (overwriting)")
        result[code] = entry

    logging.info(f"Retrieved {len(result)} regions for association {association}")
    return result


def convert_polygon_to_geojson(polygon, summit_code: str, crs: str) -> Optional[Dict]:
    """
    Convert a shapely Polygon to GeoJSON format with summit metadata.
    
    Args:
        polygon: Shapely Polygon object
        summit_code: SOTA summit code
        crs: Coordinate reference system
        
    Returns:
        GeoJSON feature dictionary or None if conversion fails
    """
    try:
        from shapely.geometry import Polygon
        
        if not isinstance(polygon, Polygon):
            logging.error(f"Expected Polygon, got {type(polygon)}")
            return None
        
        # Convert polygon coordinates to lists
        exterior_coords = list(polygon.exterior.coords)
        
        # Handle interior holes if they exist
        holes = []
        for interior in polygon.interiors:
            holes.append(list(interior.coords))
        
        # Build coordinate structure
        if holes:
            coordinates = [exterior_coords] + holes
        else:
            coordinates = [exterior_coords]
        
        # Create GeoJSON feature
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": coordinates
            },
            "properties": {
                "summitCode": summit_code,
                "description": f"SOTA Activation Zone for {summit_code}",
                "crs": crs
            }
        }
        
        return feature
        
    except Exception as e:
        logging.error(f"Error converting polygon to GeoJSON for {summit_code}: {e}")
        return None
