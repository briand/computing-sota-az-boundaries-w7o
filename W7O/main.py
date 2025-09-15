"""
Main entry point for SOTA activation zone processing.
Handles argument parsing, logging setup, and directory management.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from config import (
    SOTA_ASSOCIATION, AZ_HEIGHT, AZ_ELEVATION_UNITS,
    TILE_RADIUS, AZ_ELEVATION_TOLERANCE
)
import config
from sota_api import fetch_sota_summits, save_summits_geojson, load_summits_from_geojson, is_summit_valid
from utils import (
    generate_log_filename, setup_logging, setup_association_directories,
    extract_region_from_filename, ensure_directories
)


## Removed duplicated utility functions now provided by utils.py


def parse_arguments():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser(
        description="SOTA Activation Zone Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch summit data and process activation zones
  python main.py --region NC
  
  # Fetch summit data only (no processing)
  python main.py --fetch-only --region NC
  
  # Process from summit file (region auto-detected from filename)
  python main.py --summits W7O/input/W7O_NC_summits.geojson
  python main.py --summits /path/to/W7O_LC_summits.geojson
  
  # Custom elevation tolerance for edge cases
  python main.py --summits W7O/input/W7O_NC_summits.geojson --elevation-tolerance 5.0
  
  # Quiet mode (log to file only, no console output)
  python main.py --region NC --quiet
  python main.py --summits W7O/input/W7O_NC_summits.geojson --quiet
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '-s', '--summits',  # Short: -s, Long: --summits
        type=str,
        help='Path to summits GeoJSON file to process'
    )
    parser.add_argument(
        '-r', '--region',  # Short: -r, Long: --region
        type=str,
        required=False,
        help='SOTA region code (e.g., NC, LC) - required when fetching from SOTA API'
    )
    parser.add_argument(
        '-f', '--fetch-only',  # Short: -f, Long: --fetch-only
        action='store_true',
        help='Only fetch summit data from SOTA and save to input directory'
    )
    parser.add_argument(
        '-q', '--quiet',  # Short: -q, Long: --quiet
        action='store_true',
        help='Suppress stdout output (logs will still be written to file)'
    )
    parser.add_argument(
        '--elevation-tolerance',  # Long: --elevation-tolerance
        type=float,
        default=50.0,
        help='Maximum elevation difference (meters) between SOTA and raster data for validation'
    )
    
    args = parser.parse_args()
    
    # Validate arguments and determine region
    if args.summits and args.region:
        parser.error("Cannot specify both --summits and --region")

    # Handle summits file processing
    if args.summits:
        summits_file = Path(args.summits)
        if not summits_file.exists():
            parser.error(f"Summit file not found: {summits_file}")

    # Validate region is provided for API fetch mode
    if not args.summits and not args.region:
        parser.error("--region is required when fetching from SOTA API")
    
    # --fetch-only only makes sense with --region (not with --summits)
    if args.fetch_only and args.summits:
        parser.error("--fetch-only cannot be used with --summits (summits already exist)")
    
    if args.fetch_only and not args.region:
        parser.error("--fetch-only requires --region")
    
    return args


def main():
    """Main processing function"""
    args = parse_arguments()
    
    # Determine log file name based on processing mode
    if args.summits:
        summits_file = Path(args.summits)
        log_file = generate_log_filename(summits_file=summits_file)
    else:
        log_file = generate_log_filename(region=args.region)
    
    # Setup logging
    setup_logging(log_file, args.quiet)
    
    logging.info("SOTA Activation Zone Processor")
    logging.info("=" * 50)
    logging.info(f"Log file: {log_file}")
    
    # Determine mode and setup directories
    if args.summits:
        # Process from existing GeoJSON file
        summits_file = Path(args.summits)
        
        # Extract region from filename
        try:
            region = extract_region_from_filename(summits_file)
            logging.info(f"Detected region '{region}' from summit filename")
        except ValueError as e:
            logging.error(f"Invalid summit filename: {e}")
            sys.exit(1)
        
        # Setup association-global input/cache and region-specific output
        setup_association_directories(region)
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
        setup_association_directories(args.region)
        ensure_directories()
        
        logging.info(f"Fetching summit data from SOTA API: {SOTA_ASSOCIATION}/{args.region}")
        summits = fetch_sota_summits(args.region)
        if not summits:
            logging.error("Failed to fetch summit data")
            sys.exit(1)
        logging.info(f"Done fetching {len(summits)} summits from SOTA API")

        # Save to input directory
        summits_file = save_summits_geojson(summits, args.region)
        
    if args.fetch_only:
        logging.info(f"\nStopping now due to --fetch-only! Summit data at: {summits_file}")
        logging.info(f"To process activation zones, run:")
        logging.info(f"  python {sys.argv[0]} --summits {summits_file}")
        # IMPORTANT: Exit before importing heavy GIS dependencies (rasterio, shapely, etc.)
        return
    
    logging.info(f"\nProcessing {len(summits)} summits for activation zones...")
    logging.info(f"Initial tile radius: {TILE_RADIUS}m")
    logging.info(f"Activation zone height: {AZ_HEIGHT} {AZ_ELEVATION_UNITS}")
    logging.info(f"Elevation tolerance: {args.elevation_tolerance:.1f} {AZ_ELEVATION_UNITS}")

    # Lazy import of processing + elevation_gis to avoid requiring GIS stack for --fetch-only mode
    try:
        from W7O.gis_processing import create_region_elevation_report_csv, process_summit  # type: ignore
        from W7O.gis_elevation import reset_raster_units  # type: ignore
    except ModuleNotFoundError as e:
        logging.error("Required GIS libraries are missing for processing mode: %s", e)
        logging.error("Install rasterio, shapely, matplotlib, scikit-image before processing activation zones.")
        logging.info("You can still run in fetch-only mode without these dependencies.")
        return

    # Reset raster units to ensure proper detection with improved logic
    reset_raster_units()

    # Create processing summary CSV file
    csv_path = create_region_elevation_report_csv()
    
    # Process each summit  
    successful = 0
    failed = 0
    failed_summits = []  # Track which summits failed
    
    for i, summit in enumerate(summits, 1):
        logging.info(f"*** [{i}/{len(summits)}] {summit['summitCode']}: {summit['name']} ***")
        
        if process_summit(summit, args.elevation_tolerance, csv_path):
            successful += 1
        else:
            failed += 1
            failed_summits.append(summit['summitCode'])
    
    logging.info(f"\n=== Processing Complete ===")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    if config.OUTPUT_DIR:
        logging.info(f"Output directory: {config.OUTPUT_DIR.absolute()}")
    logging.info(f"Processing summary: {csv_path}")
    
    if successful > 0:
        logging.info(f"\nGenerated {successful} activation zone files")
        logging.info(f"Each file contains elevation-based activation zone boundaries")
    
    if failed > 0:
        logging.warning(f"\n{failed} summits failed processing:")
        for summit_code in failed_summits:
            logging.warning(f"  - {summit_code}")


if __name__ == "__main__":
    main()
