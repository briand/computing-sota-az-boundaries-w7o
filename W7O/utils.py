"""Shared utility functions for SOTA processing and comparison scripts.

This consolidates previously duplicated logic from main.py and loj_compare.py:
- Logging filename generation
- Logging setup
- Association directory setup
- Region extraction from summit filename
- Directory existence enforcement
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import config
from config import SOTA_ASSOCIATION

__all__ = [
    "generate_log_filename",
    "setup_logging",
    "setup_association_directories",
    "extract_region_from_filename",
    "ensure_directories",
]


def generate_log_filename(region: Optional[str] = None, summits_file: Optional[Path] = None, prefix: str = "log") -> str:
    """Generate a log filename.

    Args:
        region: Region code (optional)
        summits_file: Summits file path (optional)
        prefix: Prefix for the log filename (default 'log')
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if region:
        return f"{prefix}_{SOTA_ASSOCIATION}_{region}_{timestamp}.txt"
    if summits_file:
        return f"{prefix}_{summits_file.stem}_{timestamp}.txt"
    return f"{prefix}_{timestamp}.txt"


def setup_logging(log_file: str, quiet: bool = False, level: int = logging.INFO):
    """Configure logging to file and optional stdout."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    handlers = [logging.FileHandler(log_file)]
    if not quiet:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True,  # Ensure reconfiguration when called from multiple scripts
    )


def setup_association_directories(region: str):
    """Setup directory structure for association-global input/cache and region-specific outputs."""
    base_dir = Path.cwd()
    config.CACHE_DIR = base_dir / "cache"
    config.INPUT_DIR = base_dir / "input"
    config.OUTPUT_DIR = base_dir / f"{SOTA_ASSOCIATION}_{region}"
    config.CURRENT_REGION = region
    logging.debug(
        "Configured directories -> CACHE: %s INPUT: %s OUTPUT: %s", 
        config.CACHE_DIR, config.INPUT_DIR, config.OUTPUT_DIR
    )


def extract_region_from_filename(filepath: Path) -> str:
    """Extract region code from summit filename pattern: <association>_<region>_summits.geojson"""
    filename = filepath.name
    if not filename.endswith('.geojson'):
        raise ValueError(f"Summit file must be a .geojson file: {filename}")
    name_part = filename[:-8]
    parts = name_part.split('_')
    if len(parts) < 3 or parts[-1] != 'summits':
        raise ValueError("Summit filename must follow pattern '<association>_<region>_summits.geojson'")
    association = parts[0]
    if association != SOTA_ASSOCIATION:
        raise ValueError(f"Summit file association '{association}' doesn't match expected '{SOTA_ASSOCIATION}'")
    region = '_'.join(parts[1:-1])
    return region


def ensure_directories():
    """Ensure dependency directories exist."""
    for directory in [config.CACHE_DIR, config.INPUT_DIR, config.OUTPUT_DIR]:
        if directory:
            directory.mkdir(parents=True, exist_ok=True)
            logging.debug(f"Ensured directory exists: {directory}")
