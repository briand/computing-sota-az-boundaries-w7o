# Data Coverage Test Suite

This directory contains comprehensive tests for validating the SOTA Activation Zone Processor's handling of missing or incomplete LiDAR data from the Oregon DOGAMI ImageServer.

## Test Coverage Overview

The test suite validates four primary failure modes when dealing with incomplete elevation data:

### 1. No Data Regions - Complete LiDAR Absence
**Test Coordinates:** `(44.051, -119.874)`

This location represents regions where DOGAMI has no LiDAR coverage. The ImageServer returns a TIFF file, but all elevation values are zero.

**Expected Behavior:**
- Downloads TIFF successfully (100% pixel coverage)
- Detects elevation data maximum of 0.0 meters
- Fails with: `"Activation elevation (1475.0) exceeds data maximum (0.0) - no contour possible"`
- Returns `None` (processing failure)

### 2. Partial Data Regions - Boundary Limitations
**Test Coordinates:** `(43.694, -120.158)`

This location has elevation data at the point and to the south, but missing data to the north. When creating a 500m activation zone buffer, the contours extend beyond the available data boundaries.

**Expected Behavior:**
- Downloads TIFF successfully (100% pixel coverage) 
- Detects that contours are very close to TIFF boundary (<10m)
- Warns: `"Contour very close to TIFF boundary (3m) - consider larger buffer"`
- Fails with: `"No valid contour polygons found at activation elevation"`
- Returns `None` (processing failure)

### 4. Elevation Tolerance - Minor Data Maximum Exceedance
**Test Coordinates:** `(44.051, -119.874)` with controlled elevation values

Tests the elevation tolerance feature that allows processing to continue when the activation elevation slightly exceeds the data maximum within acceptable limits (default: 3 feet).

**Expected Behavior:**
- Detects activation elevation exceeding data maximum
- Checks if excess is within tolerance (converts feet to meters as needed)
- If within tolerance: Warns and continues processing
- If exceeds tolerance: Fails with tolerance exceeded error
- Provides clear logging about tolerance decisions

### 5. Invalid Coordinates - Robustness Testing
**Test Coordinates:** `(999.0, 999.0)`

Tests the script's ability to handle completely invalid geographic coordinates.

**Expected Behavior:**
- Attempts processing but receives invalid data

### 6. ImageServer Connectivity - Infrastructure Validation
**Test Status:** Disabled by default, enable for infrastructure testing

Tests the connectivity and functionality of the configured DOGAMI ESRI ImageServer to ensure:
- Server responds to requests and provides expected capabilities
- TIFF elevation data can be downloaded for test coordinates  
- Downloaded TIFFs have valid geospatial properties and elevation data

**Expected Behavior:**
- Validates server accessibility and capabilities (Image, Metadata, Catalog)
- Downloads and validates TIFF data for multiple test coordinates
- Confirms TIFF files contain valid elevation data with proper CRS and dimensions
- Provides detailed diagnostics about server response times and data characteristics

**Enable when:**
- Testing a new ImageServer configuration
- Switching to a different association's elevation service
- Diagnosing connectivity issues with current server
- Reports NaN TIFF bounds: `"BoundingBox(left=nan, bottom=nan, right=nan, top=nan)"`
- Follows same failure path as no-data regions
- Returns `None` (processing failure)

## Test Files

### `test_data_coverage.py`
Main pytest test suite containing:

- **`TestDataCoverage`** class with comprehensive test methods:
  - `test_missing_data_region_complete_failure()` - Tests coordinate (44.051, -119.874)
  - `test_partial_data_region_insufficient_coverage()` - Tests coordinate (43.694, -120.158)  
  - `test_successful_data_region_for_comparison()` - Tests known good data region
  - `test_elevation_tolerance_functionality()` - Tests elevation tolerance feature
  - `test_error_handling_robustness()` - Tests invalid coordinates
  - `test_coverage_threshold_and_boundary_conditions()` - Documents failure modes

- **`test_script_integration_no_data_coordinates()`** - Integration test that runs both problematic coordinates through the actual processing pipeline

### `test_summits.geojson`
Sample summit data file containing both test coordinates in proper GeoJSON format for integration testing.

## Running the Tests

### Individual Test Methods
```bash
# Run specific test method
pytest test_data_coverage.py::TestDataCoverage::test_missing_data_region_complete_failure -v

# Test elevation tolerance feature
pytest test_data_coverage.py::TestDataCoverage::test_elevation_tolerance_functionality -v

# Test ImageServer connectivity (disabled by default)
# Remove @pytest.mark.skip decorator from test method to enable
pytest test_data_coverage.py::TestDataCoverage::test_imageserver_connectivity_and_functionality -v -s

# Run integration test
pytest test_data_coverage.py::test_script_integration_no_data_coordinates -v -s
```

### Full Test Suite
```bash
# Run all tests
pytest test_data_coverage.py -v

# Run with output capture for debugging
pytest test_data_coverage.py -v -s
```

### Integration Testing with Main Script
```bash
# Test both coordinates through main processing pipeline
python process_sota_az.py --summits test_summits.geojson --region TEST

# Test with custom elevation tolerance
python process_sota_az.py --summits test_summits.geojson --region TEST --elevation-tolerance 5.0
```

## Key Insights from Testing

1. **Multiple Valid Failure Modes**: The script handles different types of data issues gracefully, each with appropriate error messages and consistent `None` return values.

2. **Boundary Detection**: The script detects when activation zones would extend beyond available data and warns about insufficient buffer margins.

3. **Zero Data Handling**: Even when the ImageServer returns a TIFF file, the script properly detects when elevation values are all zeros (indicating no real LiDAR data).

4. **Elevation Tolerance**: The script provides flexibility for cases where activation elevation slightly exceeds data maximum, with configurable tolerance (default 3 feet) and clear logging of tolerance decisions.

5. **Robust Error Reporting**: All failure modes include detailed logging explaining why processing failed, enabling users to understand data limitations.

6. **Consistent Interface**: Regardless of the specific data issue, the processing function consistently returns `None` for failures, making error handling predictable in larger processing pipelines.

## Expanding the Test Suite

To add new test cases:

1. **Add new coordinates** to the integration test with expected failure patterns
2. **Create specific test methods** for new failure modes discovered
3. **Update expected behaviors** if the processing logic changes
4. **Add performance tests** for large-scale processing scenarios

The test framework provides a solid foundation for validating data coverage handling as the codebase evolves.
