"""
Configuration constants and global variables for SOTA activation zone processing.
"""
from pathlib import Path
from typing import Optional

# === GIS PROCESSING CONSTANTS ===
TILE_RADIUS = 500  # Initial area radius for each summit in meters
AZ_HEIGHT = 150  # Activation zone height
AZ_ELEVATION_TOLERANCE = 20.0  # EL units - if SOTA db elevation differs from raster by more than this, fail the summit
AZ_ELEVATION_UNITS = "ft"

# === SOTA CONSTANTS ===
SOTA_ASSOCIATION = "W7W"  # SOTA association code

# === GLOBAL DIRECTORY PATHS ===
# These are set by setup functions and used throughout the application
CACHE_DIR: Optional[Path] = None
INPUT_DIR: Optional[Path] = None  
OUTPUT_DIR: Optional[Path] = None

# === CURRENT PROCESSING CONTEXT ===
# Set during processing to provide context for cache file naming
CURRENT_REGION: Optional[str] = None

# === ELEVATION UNITS ===
RASTER_ELEVATION_UNITS = "ft"  # determined at runtime from imageserver metadata

# === COMPARISON / LoJ SETTINGS ===
# Maximum allowed horizontal separation (meters) between SOTA summit and LoJ feature
LOJ_COORD_MATCH_TOLERANCE_M = 50.0

# Elevation band thresholds (feet) that, if crossed between SOTA and LoJ elevations,
# should cause a match to be flagged specially (POINT_BAND_CHANGE) for diagnostic purposes.
# Example: SOTA 2999 ft vs LoJ 3001 ft crosses 3000 band.
POINT_BANDS = [1641, 3281, 4922, 6562, 8202]
