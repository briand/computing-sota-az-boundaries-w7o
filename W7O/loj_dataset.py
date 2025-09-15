"""Utilities for handling Lists of John (LoJ) summit datasets.

Stage: Discovery & basic loading only (no comparison logic yet).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv
import math

from config import LOJ_COORD_MATCH_TOLERANCE_M
import config as _cfg

def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in meters between two WGS84 coords."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def extract_loj_feature_coordinates(feature: Dict) -> Optional[Tuple[float, float]]:
    """Extract (lat, lon) from a LoJ GeoJSON feature supporting Point geometries.

    Returns (lat, lon) or None if unavailable.
    """
    try:
        geom = feature.get("geometry") or {}
        if geom.get("type") != "Point":
            return None
        coords = geom.get("coordinates")
        if not isinstance(coords, (list, tuple)) or len(coords) < 2:
            return None
        lon, lat = float(coords[0]), float(coords[1])
        return lat, lon
    except Exception:
        return None


def extract_loj_altitude_ft(feature: Dict) -> Optional[float]:
    """Extract altitude in feet from LoJ feature properties.

    Assumes a property named 'altFt', 'elevFt', 'elevation', or attempts to parse a
    'description' field containing a number followed by 'ft'.
    """
    props = feature.get("properties", {})
    for key in ["altFt", "elevFt", "elevation", "Elevation", "elev", "Altitude"]:
        if key in props:
            try:
                return float(props[key])
            except (TypeError, ValueError):
                pass
    desc = props.get("description") or props.get("Description")
    if isinstance(desc, str):
        import re
        m = re.search(r"(\d+(?:\.\d+)?)\s*ft", desc.lower())
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def extract_loj_name(feature: Dict) -> Optional[str]:
    """Extract a display name for a LoJ feature.

    Tries common property keys; falls back to 'name' inside properties; returns None if not found.
    """
    props = feature.get("properties", {})
    for key in ["Name", "name", "loj_name", "title", "peakName"]:
        if key in props and props[key]:
            return str(props[key])
    return None


def extract_loj_id(feature: Dict) -> Optional[str]:
    """Extract the Lists of John peak Id from feature properties.

    Checks common key casings (Id, id, ID). Returns string or None.
    """
    props = feature.get("properties", {})
    for key in ["Id", "id", "ID"]:
        if key in props and props[key] is not None and props[key] != "":
            return str(props[key])
    return None


def find_best_loj_match(lat: float, lon: float, loj_features: List[Dict], tolerance_m: float = LOJ_COORD_MATCH_TOLERANCE_M) -> Optional[Tuple[Dict, float]]:
    """Find the closest LoJ feature within tolerance to the SOTA coordinate.

    Returns tuple (feature, distance_m) or None if nothing within tolerance.
    Uses simple linear scan (acceptable due to relatively small regional dataset size).
    """
    best: Optional[Tuple[Dict, float]] = None
    for feat in loj_features:
        coord = extract_loj_feature_coordinates(feat)
        if not coord:
            continue
        lat2, lon2 = coord
        dist = haversine_meters(lat, lon, lat2, lon2)
        if dist <= tolerance_m and (best is None or dist < best[1]):
            best = (feat, dist)
    return best


def find_nearest_loj_feature(lat: float, lon: float, loj_features: List[Dict]) -> Optional[Tuple[Dict, float]]:
    """Find the nearest LoJ feature regardless of tolerance.

    Returns (feature, distance_m) or None if list empty / no coords.
    Uses linear scan; dataset size is modest so performance is acceptable.
    """
    nearest: Optional[Tuple[Dict, float]] = None
    for feat in loj_features:
        coord = extract_loj_feature_coordinates(feat)
        if not coord:
            continue
        lat2, lon2 = coord
        dist = haversine_meters(lat, lon, lat2, lon2)
        if nearest is None or dist < nearest[1]:
            nearest = (feat, dist)
    return nearest


def discover_loj_file(region: str) -> Optional[Path]:
    """Locate a LoJ JSON/GeoJSON file for the given region.

    Looks for files in the input directory matching pattern:
        loj_<association>_<region>*.json
    (case-insensitive on region)

    If multiple matches are found, returns the most recently modified.

    Returns None if no file found.
    """
    input_dir = _cfg.INPUT_DIR
    if input_dir is None:
        logging.error("INPUT_DIR not initialized before LoJ discovery")
        return None

    region_upper = region.upper()
    pattern = f"loj_{_cfg.SOTA_ASSOCIATION}_{region_upper}".lower()  # base pattern, we'll filter manually

    candidates: List[Tuple[float, Path]] = []
    try:
        for p in input_dir.glob("*.json"):
            name_lower = p.name.lower()
            if name_lower.startswith("loj_") and pattern in name_lower:
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                candidates.append((mtime, p))
    except Exception as e:
        logging.error(f"Error searching for LoJ files: {e}")
        return None

    if not candidates:
        logging.info(f"No LoJ file found matching pattern 'loj_{_cfg.SOTA_ASSOCIATION}_{region_upper}*.json'")
        return None

    # Pick most recently modified
    candidates.sort(key=lambda x: x[0], reverse=True)
    chosen = candidates[0][1]
    logging.info(f"Selected LoJ file: {chosen} (most recent of {len(candidates)})")
    return chosen


def discover_loj_csv_file(region: str) -> Optional[Path]:
    """Locate a LoJ CSV file for the given region.

    Pattern: loj_<association>_<region>*.csv (case-insensitive)
    Returns most recently modified or None.
    """
    input_dir = _cfg.INPUT_DIR
    if input_dir is None:
        logging.error("INPUT_DIR not initialized before LoJ CSV discovery")
        return None

    region_upper = region.upper()
    pattern = f"loj_{_cfg.SOTA_ASSOCIATION}_{region_upper}".lower()
    candidates: List[Tuple[float, Path]] = []
    try:
        for p in input_dir.glob("*.csv"):
            name_lower = p.name.lower()
            if name_lower.startswith("loj_") and pattern in name_lower:
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                candidates.append((mtime, p))
    except Exception as e:
        logging.error(f"Error searching for LoJ CSV files: {e}")
        return None

    if not candidates:
        logging.info(f"No LoJ CSV file found matching pattern 'loj_{_cfg.SOTA_ASSOCIATION}_{region_upper}*.csv'")
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    chosen = candidates[0][1]
    logging.info(f"Selected LoJ CSV file: {chosen} (most recent of {len(candidates)})")
    return chosen


def discover_loj_csv_association() -> Optional[Path]:
    """Locate a single association-wide LoJ CSV file.

    Pattern: loj_<association>*.csv (region fragment not required).
    Returns most recently modified or None.
    """
    input_dir = _cfg.INPUT_DIR
    if input_dir is None:
        logging.error("INPUT_DIR not initialized before association LoJ CSV discovery")
        return None
    pattern = f"loj_{_cfg.SOTA_ASSOCIATION}".lower()
    candidates: List[Tuple[float, Path]] = []
    try:
        for p in input_dir.glob("*.csv"):
            name_lower = p.name.lower()
            if name_lower.startswith("loj_") and pattern in name_lower:
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                candidates.append((mtime, p))
    except Exception as e:
        logging.error(f"Error searching for association LoJ CSV files: {e}")
        return None
    if not candidates:
        logging.info(f"No association-wide LoJ CSV file found matching pattern '{pattern}*.csv'")
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    chosen = candidates[0][1]
    logging.info(f"Selected association-wide LoJ CSV file: {chosen}")
    return chosen


def discover_loj_association_file() -> Optional[Path]:
    """Locate a single association-wide LoJ JSON/GeoJSON file.

    Pattern: loj_<association>*.json
    Returns most recently modified or None.
    """
    input_dir = _cfg.INPUT_DIR
    if input_dir is None:
        logging.error("INPUT_DIR not initialized before association LoJ discovery")
        return None
    pattern = f"loj_{_cfg.SOTA_ASSOCIATION}".lower()
    candidates: List[Tuple[float, Path]] = []
    try:
        for p in input_dir.glob("*.json"):
            name_lower = p.name.lower()
            if name_lower.startswith("loj_") and pattern in name_lower:
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                candidates.append((mtime, p))
    except Exception as e:
        logging.error(f"Error searching for association LoJ JSON files: {e}")
        return None
    if not candidates:
        logging.info(f"No association-wide LoJ JSON file found matching pattern '{pattern}*.json'")
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    chosen = candidates[0][1]
    logging.info(f"Selected association-wide LoJ JSON file: {chosen}")
    return chosen


def load_loj_csv(csv_path: Path) -> List[Dict]:
    """Load LoJ CSV rows and convert to feature-like dicts.

    Expected columns: Name,Latitude,Longitude,Elevation,Prominence,Isolation,State,Counties,Quad,Id
    Returns list of pseudo-features with geometry + properties similar to GeoJSON.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"LoJ CSV does not exist: {csv_path}")
    features: List[Dict] = []
    def _clean_str(val: Optional[str]) -> Optional[str]:
        if val is None:
            return val
        v = val.strip()
        # Collapse multi-layer leading/trailing quotes (""Name"" or """Name""")
        # down to a single quoted form "Name" but KEEP a single layer of quotes
        # to express unofficial / alternate naming.
        for q in ('"', "'"):
            if v.startswith(q) and v.endswith(q):
                # count consecutive leading/trailing quotes of same type
                lead = 0
                for ch in v:
                    if ch == q:
                        lead += 1
                    else:
                        break
                trail = 0
                for ch in reversed(v):
                    if ch == q:
                        trail += 1
                    else:
                        break
                if lead >= 2 and trail >= 2:
                    inner = v[lead:-trail].strip()
                    v = f"{q}{inner}{q}"
        return v

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row.get('Latitude', '') or 'nan')
                lon = float(row.get('Longitude', '') or 'nan')
            except ValueError:
                continue
            if any(map(lambda v: v != v, [lat, lon])):  # NaN check
                continue
            elevation = row.get('Elevation')
            name_val = _clean_str(row.get('Name'))
            # Build pseudo feature
            feature = {
                'type': 'Feature',
                'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
                'properties': {
                    'Name': name_val,
                    'Elevation': elevation,
                    'Prominence': _clean_str(row.get('Prominence')),
                    'Isolation': _clean_str(row.get('Isolation')),
                    'State': _clean_str(row.get('State')),
                    'Counties': _clean_str(row.get('Counties')),
                    'Quad': _clean_str(row.get('Quad')),
                    'Id': _clean_str(row.get('Id')),
                }
            }
            features.append(feature)
    logging.info(f"Loaded {len(features)} LoJ CSV rows")
    return features


def load_loj_geojson(loj_path: Path) -> List[Dict]:
    """Load a LoJ GeoJSON file and return list of feature dicts.

    Accepts either full GeoJSON FeatureCollection or a bare list of features.
    Basic validation only; deeper schema checks can be added later.
    Raises on read/parse errors or missing features list.
    """
    if not loj_path.exists():
        raise FileNotFoundError(f"LoJ file does not exist: {loj_path}")

    logging.info(f"Loading LoJ GeoJSON: {loj_path}")
    with open(loj_path, "r") as f:
        raw = f.read()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        # Attempt lightweight sanitization for trailing commas in arrays/objects
        logging.warning(f"Primary JSON parse failed ({e}); attempting sanitize of trailing commas")
        sanitized = _sanitize_trailing_commas(raw)
        try:
            data = json.loads(sanitized)
            logging.info("LoJ file parsed successfully after sanitization")
        except json.JSONDecodeError as e2:
            logging.error(f"Sanitized parse still failed: {e2}")
            raise

    # Determine structure
    if isinstance(data, dict):
        if data.get("type") == "FeatureCollection" and isinstance(data.get("features"), list):
            features = data["features"]
        elif "features" in data and isinstance(data["features"], list):
            features = data["features"]
        else:
            raise ValueError("LoJ GeoJSON dict missing 'features' list")
    elif isinstance(data, list):
        features = data
    else:
        raise ValueError("Unsupported LoJ GeoJSON structure (expected dict or list)")

    logging.info(f"Loaded {len(features)} LoJ features")
    return features


def _sanitize_trailing_commas(text: str) -> str:
    """Remove trailing commas before closing brackets/braces in JSON.

    This is a minimal, non-robust pass suitable for cleaning common export artifacts.
    It does not attempt full JSON5 support. Operates line by line to avoid large regex catastrophes.
    """
    import re
    # Regex to remove a comma followed by optional space/newlines then a closing ] or }
    pattern = re.compile(r",(\s*[\]\}])")
    # Apply repeatedly until stable (max a few iterations to prevent infinite loop)
    prev = None
    current = text
    for _ in range(5):
        updated = pattern.sub(r"\1", current)
        if updated == current:
            break
        current = updated
    return current


__all__ = [
    "discover_loj_file",
    "load_loj_geojson",
    "discover_loj_csv_file",
    "discover_loj_csv_association",
    "discover_loj_association_file",
    "load_loj_csv",
    "haversine_meters",
    "extract_loj_feature_coordinates",
    "extract_loj_altitude_ft",
    "extract_loj_name",
    "extract_loj_id",
    "find_best_loj_match",
    "find_nearest_loj_feature",
]
