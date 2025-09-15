"""
Summit processing orchestration, CSV tracking, and activation zone generation.
"""
import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone

from config import OUTPUT_DIR, AZ_HEIGHT, AZ_ELEVATION_UNITS, TILE_RADIUS
from sota_api import convert_polygon_to_geojson
from W7O.gis_elevation import (
    initialize_raster_units, download_elevation_data, download_adjacent_elevation_tiles,
    merge_elevation_tiles, load_elevation_data, process_elevation_values,
    apply_elevation_business_logic, create_coordinate_mesh, generate_contour_polygons,
    select_summit_containing_polygon, calculate_distance_to_edges
)


def create_region_elevation_report_csv() -> Path:
    """
    Create a CSV file for tracking region elevation processing results.
    
    Returns:
        Path to the created CSV file
    """
    from config import OUTPUT_DIR
    
    if OUTPUT_DIR is None:
        raise ValueError("OUTPUT_DIR must be set before creating CSV")
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_filename = f"region_elevation_report_{timestamp}.csv"
    csv_path = OUTPUT_DIR / csv_filename
    
    # Create CSV with headers
    headers = [
        'summit_code', 'summit_name', 'sota_longitude', 'sota_latitude', 'sota_elevation',
        'raster_longitude', 'raster_latitude', 'raster_elevation', 'elevation_difference',
        'generated_az_poly', 'failure_reason', 'processing_timestamp'
    ]
    
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
        
        logging.info(f"Created processing summary CSV: {csv_path}")
        return csv_path
        
    except Exception as e:
        logging.error(f"Error creating CSV file: {e}")
        raise


def write_summit_processing_result(csv_path: Path, summit: Dict, raster_data: Optional[Dict] = None, 
                                 success: bool = False, failure_reason: str = "") -> None:
    """
    Write a summit processing result to the CSV file.
    
    Args:
        csv_path: Path to the CSV file
        summit: Summit dictionary
        raster_data: Optional raster data dictionary with coordinates and elevation
        success: Whether processing was successful
        failure_reason: Reason for failure if not successful
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    
    # Extract basic summit info - use altitude field matching AZ_ELEVATION_UNITS
    summit_code = summit.get('summitCode', 'UNKNOWN')
    summit_name = summit.get('name', 'UNKNOWN')
    sota_lon = summit.get('longitude', 0.0)
    sota_lat = summit.get('latitude', 0.0)
    
    # Use correct altitude field based on AZ_ELEVATION_UNITS
    from config import AZ_ELEVATION_UNITS
    if AZ_ELEVATION_UNITS == 'ft':
        sota_elevation = summit.get('altFt', 0.0)
    else:
        sota_elevation = summit.get('altM', 0.0)
    
    # Extract raster info if available
    if raster_data:
        raster_lon = raster_data.get('raster_lon', 0.0)
        raster_lat = raster_data.get('raster_lat', 0.0)
        raster_elevation = raster_data.get('raster_elevation', 0.0)
        elevation_diff = raster_data.get('elevation_difference', 0.0)
    else:
        raster_lon = raster_lat = raster_elevation = elevation_diff = 0.0
    
    # Create row data
    row_data = [
        summit_code, summit_name, sota_lon, sota_lat, sota_elevation,
        raster_lon, raster_lat, raster_elevation, elevation_diff,
        'Yes' if success else 'No', failure_reason, timestamp
    ]
    
    try:
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row_data)
            
    except Exception as e:
        logging.error(f"Error writing to CSV file: {e}")


def load_and_validate_elevation_data(elevation_file: Path, summit: Dict, elevation_tolerance: float) -> Tuple[Optional[Tuple], Optional[Dict]]:
    """
    Load and validate elevation data for a summit - orchestrates the 3 main steps.
    
    Args:
        elevation_file: Path to elevation raster file
        summit: Summit dictionary
        elevation_tolerance: Tolerance for elevation validation
        
    Returns:
        Tuple of (elevation_processing_result, raster_data) where:
        - elevation_processing_result is (elevation_data, transform, crs, activation_alt_raster, needs_larger_radius) or None
        - raster_data is dictionary with coordinates and elevation info for CSV tracking
    """
    # Step 1: Load the raster data
    load_result = load_elevation_data(elevation_file)
    if not load_result:
        return None, None
    
    elevation_data, transform, crs, bounds = load_result
    
    # Step 2: Process and convert data to derive complete set of values
    summit_alt, min_elev, max_elev, mean_elev, median_elev, activation_alt = process_elevation_values(summit, elevation_data)
    
    # Step 3: Apply business logic and validation
    summit_lon = float(summit['longitude'])
    summit_lat = float(summit['latitude'])
    
    activation_alt_raster, validation_passed, raster_data = apply_elevation_business_logic(
        summit, summit_alt, activation_alt, max_elev, min_elev, summit_lon, summit_lat,
        transform, crs, elevation_tolerance
    )
    
    # Check if activation elevation is below raster minimum (needs larger radius)
    needs_larger_radius = False
    if activation_alt_raster is not None and activation_alt_raster < min_elev:
        logging.warning(f"  Activation elevation ({activation_alt_raster:.1f}) is below raster minimum - will need larger radius")
        needs_larger_radius = True
        # Don't return failure here - let the calling function try 2x radius
        return (elevation_data, transform, crs, activation_alt_raster, needs_larger_radius), raster_data
    
    if validation_passed and activation_alt_raster is not None:
        return (elevation_data, transform, crs, activation_alt_raster, needs_larger_radius), raster_data
    else:
        return None, raster_data


def get_tileset_and_process(summit_lon: float, summit_lat: float, radius_multiplier: float, summit: Dict, elevation_tolerance: float) -> Tuple[Optional[Tuple], Optional[Dict]]:
    """
    Download elevation tiles for a summit area and process them.
    
    Args:
        summit_lon: Summit longitude
        summit_lat: Summit latitude
        radius_multiplier: Multiplier for tile radius (1.0 = normal, 2.0 = larger area)
        summit: Summit dictionary
        elevation_tolerance: Elevation tolerance for validation
        
    Returns:
        Tuple of (processing_result, raster_data) where processing_result contains
        (elevation_data, transform, crs, activation_alt_raster, needs_larger_radius) or None if failed
    """
    buffer_m = int(TILE_RADIUS * radius_multiplier)
    
    # Download initial elevation data
    elevation_file = download_elevation_data(summit_lon, summit_lat, buffer_m, summit['summitCode'])
    if not elevation_file:
        logging.error(f"Failed to download elevation data for {summit['summitCode']}")
        return None, None
    
    # Load and validate the elevation data
    validation_result, raster_data = load_and_validate_elevation_data(elevation_file, summit, elevation_tolerance)
    
    return validation_result, raster_data


def process_merged_tileset(elevation_files: List[Path], summit: Dict) -> Optional[Tuple]:
    """
    Process a merged tileset to extract elevation data and coordinates.
    
    Args:
        elevation_files: List of elevation file paths to merge
        summit: Summit dictionary
        
    Returns:
        Tuple of (elevation_data, transform, crs) or None if processing fails
    """
    # Merge elevation tiles
    merge_result = merge_elevation_tiles(elevation_files)
    if not merge_result:
        logging.error(f"Failed to merge elevation tiles for {summit['summitCode']}")
        return None
    
    elevation_data, transform, crs = merge_result
    
    # Process elevation values to get stats
    summit_alt, min_elev, max_elev, mean_elev, median_elev, activation_alt = process_elevation_values(summit, elevation_data)
    
    return elevation_data, transform, crs


def calculate_contour_with_expansion(elevation_data, transform, crs, 
                                   summit: Dict, activation_alt_raster: float) -> Optional[Dict]:
    """
    Calculate contour polygons and handle edge expansion if needed.
    
    Args:
        elevation_data: Numpy array of elevation data
        transform: Raster transform object
        crs: Coordinate reference system
        summit: Summit dictionary
        activation_alt_raster: Activation elevation in raster units
        
    Returns:
        GeoJSON feature dictionary or None if calculation fails
    """
    summit_code = summit['summitCode']
    summit_lon = float(summit['longitude'])
    summit_lat = float(summit['latitude'])
    
    # Create coordinate mesh
    x_coords, y_coords, elevation_data = create_coordinate_mesh(elevation_data, transform)
    
    # Generate contour polygons
    polygon_shapes = generate_contour_polygons(x_coords, y_coords, elevation_data, activation_alt_raster)
    
    if not polygon_shapes:
        logging.error(f"No activation zone polygons generated for {summit_code}")
        return None
    
    # Select the polygon containing the summit
    selected_polygon = select_summit_containing_polygon(polygon_shapes, summit_lon, summit_lat, summit_code)
    
    if not selected_polygon:
        logging.error(f"No suitable polygon found for {summit_code}")
        return None
    
    # Convert to GeoJSON
    crs_string = str(crs) if crs else "EPSG:4326"
    feature = convert_polygon_to_geojson(selected_polygon, summit_code, crs_string)
    
    if feature:
        logging.info(f"Generated activation zone for {summit_code}")
    
    return feature


def create_activation_zone_elevation_based(summit: Dict, elevation_tolerance: float, csv_path: Optional[Path] = None) -> Optional[Dict]:
    """
    Create elevation-based activation zone polygon for a summit using the original algorithm.
    
    Args:
        summit: Summit dictionary from SOTA API
        elevation_tolerance: Tolerance in AZ_ELEVATION_UNITS for validating summit elevation against raster data maximum.
        csv_path: Optional path to CSV file for logging results
        
    Returns:
        GeoJSON feature dictionary or None if generation fails
    """
    summit_code = summit['summitCode']
    summit_lon = float(summit['longitude'])
    summit_lat = float(summit['latitude'])
    
    logging.info(f"Creating activation zone for {summit_code}")
    
    # Initialize raster units from imageserver metadata
    initialize_raster_units()
    
    # Variables for tracking results
    raster_data = None
    failure_reason = ""
    feature = None
    
    try:
        # === PASS 1: Try with normal 1x radius tileset ===
        validation_result, raster_data = get_tileset_and_process(summit_lon, summit_lat, 1.0, summit, elevation_tolerance)
        if validation_result is None:
            failure_reason = "Failed to get initial elevation data" if raster_data is None else "Elevation validation failed (tolerance exceeded)"
            logging.error(f"Failed to get initial elevation data for {summit_code}")
            return None
        
        elevation_data, transform, crs, activation_alt_raster, needs_larger_radius = validation_result
        
        # === PASS 2: If activation elevation is below minimum, try with 2x radius tileset ===
        if needs_larger_radius:
            logging.info(f"  Retrying with 2x radius due to activation elevation below raster minimum")
            larger_validation_result, larger_raster_data = get_tileset_and_process(summit_lon, summit_lat, 2.0, summit, elevation_tolerance)
            if larger_validation_result is not None:
                larger_elevation_data, larger_transform, larger_crs, larger_activation_alt_raster, still_needs_larger = larger_validation_result
                if not still_needs_larger:
                    logging.info(f"  Successfully obtained suitable elevation data with 2x radius")
                    # Use the larger tileset data
                    elevation_data, transform, crs, activation_alt_raster = larger_elevation_data, larger_transform, larger_crs, larger_activation_alt_raster
                    if larger_raster_data:  # Use the raster data from the larger tileset
                        raster_data = larger_raster_data
                else:
                    failure_reason = "Activation elevation still below minimum even with 2x radius"
                    logging.error(f"  {failure_reason}")
                    return None
            else:
                failure_reason = "Failed to get 2x radius elevation data" if larger_raster_data is None else "2x radius elevation validation failed (tolerance exceeded)"
                logging.error(f"  {failure_reason}")
                return None
        
        # === PASS 3: Generate initial contours to analyze boundary distances ===
        x_coords, y_coords, elevation_clean = create_coordinate_mesh(elevation_data, transform)
        initial_polygons = generate_contour_polygons(x_coords, y_coords, elevation_clean, activation_alt_raster)
        
        if not initial_polygons:
            logging.error(f"No initial contour polygons generated for {summit_code}")
            failure_reason = "No initial contour polygons generated"
            return None
        
        # Extract all contour coordinates for boundary analysis
        all_contour_coords = []
        for polygon in initial_polygons:
            if hasattr(polygon, 'exterior'):
                coords = list(polygon.exterior.coords)
                all_contour_coords.extend(coords)
        
        # Analyze boundary distances
        boundary_distances = calculate_distance_to_edges(all_contour_coords, summit_lon, summit_lat)
        
        # === PASS 4: Download adjacent tiles if contours approach boundaries ===
        threshold_m = 10.0  # anything less than 10m from edge triggers expansion
        edge_analysis = lambda msg: logging.info(f'{msg}: [{", ".join([f"{k}: {v:.1f}m" for k, v in boundary_distances.items()])}]')
        
        if any(dist < threshold_m for dist in boundary_distances.values()):
            edge_analysis("Contour approaches tile boundary")
            
            # Download additional tiles using the original adjacent tile algorithm
            additional_files = download_adjacent_elevation_tiles(summit_lon, summit_lat, TILE_RADIUS, boundary_distances, summit_code)
            
            if additional_files and len(additional_files) > 1:  # We have center + adjacent tiles
                logging.info(f"Processing merged tileset with {len(additional_files)} tiles")
                
                # Merge all tiles together
                merge_result = merge_elevation_tiles(additional_files)
                if not merge_result:
                    failure_reason = "Failed to merge adjacent tiles - contours approach boundary but tile merge failed"
                    logging.error(f"Failed to merge tiles for {summit_code} - summit processing failed")
                    return None
                
                # Process merged data and regenerate contours
                merged_elevation_data, merged_transform, merged_crs = merge_result
                
                # Regenerate contours with merged data
                merged_x_coords, merged_y_coords, merged_elevation_clean = create_coordinate_mesh(merged_elevation_data, merged_transform)
                merged_polygons = generate_contour_polygons(merged_x_coords, merged_y_coords, merged_elevation_clean, activation_alt_raster)
                
                if not merged_polygons:
                    failure_reason = "Merged tileset failed to generate contours - insufficient elevation data coverage"
                    logging.error(f"Merged tileset failed to generate contours for {summit_code} - summit processing failed")
                    return None
                
                logging.info(f"Successfully regenerated contours with merged tileset for {summit_code}")
                feature = calculate_contour_with_expansion(merged_elevation_data, merged_transform, merged_crs, summit, activation_alt_raster)
            else:
                failure_reason = "Adjacent tile download failed - contours approach boundary but cannot expand coverage"
                logging.error(f"Adjacent tile download failed for {summit_code} - summit processing failed")
                return None
        else:
            edge_analysis("Contour margins adequate")
            feature = calculate_contour_with_expansion(elevation_data, transform, crs, summit, activation_alt_raster)
        
        # Final check on feature generation
        if not feature:
            failure_reason = "Contour generation failed"
        
        return feature
        
    except Exception as e:
        failure_reason = f"Unexpected error: {str(e)}"
        logging.error(f"Unexpected error processing {summit_code}: {e}")
        return None
    
    finally:
        # Single point for CSV logging
        if csv_path:
            success = feature is not None
            write_summit_processing_result(csv_path, summit, raster_data, success, failure_reason)


def process_summit(summit: Dict, elevation_tolerance: float, csv_path: Optional[Path] = None) -> bool:
    """Process a single summit and generate activation zone GeoJSON"""
    from config import OUTPUT_DIR
    
    summit_code = summit['summitCode'].replace('/', '_').replace('-', '_')  # Make filename standard
    output_file = OUTPUT_DIR / f"{summit_code}.geojson" if OUTPUT_DIR is not None else Path(f"{summit_code}.geojson")

    # Skip if already processed
    if output_file.exists():
        logging.info(f"Skipping {summit['summitCode']} (already exists)")
        # Still log to CSV that it was skipped
        if csv_path:
            write_summit_processing_result(csv_path, summit, None, True, "Already processed")
        return True
    
    feature = create_activation_zone_elevation_based(summit, elevation_tolerance, csv_path)
    
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
        # Update CSV with file save failure
        if csv_path:
            write_summit_processing_result(csv_path, summit, None, False, f"File save error: {str(e)}")
        return False
