# Agent Instructions: SOTA Activation Zone Processor

## Script Overview
`process_sota_python.py` is a Python 3.13 script that generates SOTA activation zone boundaries using elevation contours. It downloads DOGAMI lidar elevation data, processes it to find 82-foot contour lines below summit points, and outputs GeoJSON polygons representing activation zones.

## Key Technical Details

### Dependencies & Environment
- **Python Environment**: Requires virtual environment with geospatial packages
- **Key Libraries**: geopandas, shapely, pyproj, rasterio, numpy, matplotlib
- **Setup Commands**:
  ```bash
  configure_python_environment
  install_python_packages ["geopandas", "shapely", "pyproj", "rasterio", "numpy", "matplotlib"]
  ```

### Command Line Interface
- **Primary modes**: 
  - `--summits input/file.geojson` (process from existing file)
  - `--association W7O --region NC` (fetch and process)
  - `--fetch-only` (download summit data only)
- **Logging**: `--log-file filename.log` for file output, otherwise stdout
- **Processing**: Skips existing output files automatically

### Coordinate Systems & Data Handling
- **Input CRS**: WGS84 (EPSG:4326) from SOTA API
- **Processing CRS**: EPSG:6557 (Oregon Statewide Lambert) for accurate distance calculations
- **Elevation Units**: Auto-detects feet vs meters from data range
- **Buffer Strategy**: Requests 1km × 1km, server returns ~3.4km × 3.4km minimum tiles

### Key Processing Insights
1. **ImageServer Behavior**: DOGAMI server returns standardized ~3.4km tiles regardless of requested 1km size
2. **Distance Margins**: Expect ~1333m distances to TIFF edges (not 500m) due to generous server tiles
3. **Contour Generation**: Uses matplotlib for smooth boundaries, typically 300-500+ coordinate points
4. **Caching**: Elevation TIFFs cached locally, ~1MB each, reused across runs

### Debugging & Monitoring
- **Boundary Analysis**: Script reports distances from contours to elevation data edges
- **Coverage Validation**: Checks elevation data completeness, warns if <10% valid pixels
- **Unit Detection**: Logs whether elevation data detected as feet or meters
- **Error Handling**: Comprehensive validation for missing data, coordinate issues, contour failures

### Expected Output Patterns
- **Log Messages**: INFO level for progress, WARNING for contour issues, ERROR for failures
- **Distance Reports**: "Contour distances to TIFF edges - Overall: XXXXm" (expect 1000-1500m range)
- **TIFF Size Reports**: "TIFF size: 3416m × 3416m (requested: 1000m × 1000m)" is normal
- **Success Indicators**: "Generated contour polygon with XXX coordinate points"

### Working with the Script
1. **First Run**: Always configure Python environment and install packages
2. **Reprocessing**: Delete output files to force regeneration
3. **Single Summit Testing**: Remove specific output file, then run full command
4. **Log Analysis**: Check TIFF size and distance reports for data adequacy
5. **Error Investigation**: Look for coverage percentages and unit detection messages

### Common Issue Patterns
- **Import Errors**: Missing geospatial libraries, need package installation
- **Coordinate Errors**: Large distance values (>100km) indicate CRS handling bugs
- **Missing Data**: Low coverage percentages indicate elevation data gaps
- **Contour Failures**: Usually due to flat terrain or insufficient elevation variation

### File Structure
- **Input**: `input/W7O_NC.geojson` (summit locations)
- **Cache**: `cache/elev_*.tif` (elevation data, ~1MB each)
- **Output**: `output/W7O_NC-XXX.geojson` (activation zone polygons)
- **Logs**: Specified via `--log-file` or stdout

This script is production-ready with comprehensive logging, error handling, and validation. The ImageServer's generous tile sizes provide excellent safety margins for activation zone boundary calculations.
