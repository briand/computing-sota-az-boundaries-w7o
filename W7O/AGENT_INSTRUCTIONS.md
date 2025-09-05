# Agent Instructions: SOTA Activation Zone Processor

## Script Overview
`generate_sota_az.py` is a Python 3.13 script that generates SOTA activation zone boundaries using elevation contours. It downloads DOGAMI lidar elevation data, processes it to find 82-foot contour lines below summit points, and outputs GeoJSON polygons representing activation zones.

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
**Script Name**: `generate_sota_az.py`

**Required Arguments**:
- `--region REGION`: SOTA region code (e.g., NC, LC) - always required

**Primary Processing Modes**:
- `--summits input/file.geojson --region NC`: Process from existing GeoJSON file
- `--association W7O --region NC`: Fetch from SOTA API and process
- `--fetch-only --region NC`: Download summit data only (no processing)

**Optional Arguments**:
- `--log-file filename.log`: Write logs to file (default: stdout)
- `--quiet`: Reduce log verbosity
- `--elevation-tolerance N.N`: Custom elevation validation tolerance in feet (default: 20.0)

**Examples**:
```bash
# Process existing summit data
python generate_sota_az.py --summits W7O_NC/input/W7O_NC.geojson --region NC

# Fetch and process from API
python generate_sota_az.py --association W7O --region NC

# Fetch only (no processing)
python generate_sota_az.py --association W7O --region NC --fetch-only

# Custom elevation tolerance
python generate_sota_az.py --summits W7O_NC/input/W7O_NC.geojson --region NC --elevation-tolerance 5.0
```

**Processing Behavior**: Automatically skips existing output files

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

### Common Issue Patterns & Solutions
- **Import Errors**: Missing geospatial libraries, need package installation
- **Coordinate Errors**: Large distance values (>100km) indicate CRS handling bugs  
- **Missing Data**: Low coverage percentages indicate elevation data gaps
- **Contour Failures**: Usually due to flat terrain or insufficient elevation variation
- **"Missing return statement"**: Function processes successfully but implicitly returns None
- **"Unrecognized arguments"**: Check exact argument names with `--help`
- **"Summit file not found"**: Verify file path and working directory
- **"KeyError: 'summitCode'"**: Use `'code'` property instead in summit GeoJSON

### File Structure & JSON Formats

**Directory Layout**:
- **Input**: `W7O_NC/input/W7O_NC.geojson` (summit locations from SOTA API)
- **Cache**: `W7O_NC/cache/elev_*.tif` (elevation data, ~1MB each, auto-managed)
- **Output**: `W7O_NC/output/W7O_NC-XXX.geojson` (activation zone polygons)
- **Logs**: Specified via `--log-file` or stdout

**Input GeoJSON Format** (from SOTA API):
```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "geometry": {
      "type": "Point",
      "coordinates": [-123.383800, 45.454900]
    },
    "properties": {
      "code": "W7O/NC-031",
      "name": "Blind Cabin Ridge", 
      "title": "W7O/NC-031",
      "elevationM": 719,
      "elevationFt": 2360,
      "points": 2,
      "validTo": "2099-12-31T00:00:00Z",
      "validFrom": "2010-07-01T00:00:00Z"
    }
  }]
}
```

**Key Properties**:
- `code`: Summit identifier (used for file naming and logging)
- `name`: Human-readable summit name
- `elevationFt`: SOTA official elevation in feet
- `coordinates`: [longitude, latitude] in WGS84

**Output GeoJSON Format** (activation zones):
```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature", 
    "geometry": {
      "type": "Polygon",
      "coordinates": [[
        [-123.38456, 45.45123],
        [-123.38234, 45.45456],
        // ... 300-500+ coordinate pairs for smooth boundary
      ]]
    },
    "properties": {
      "summitCode": "W7O/NC-031",
      "summitName": "Blind Cabin Ridge",
      "activationZoneElevationFt": 2278,
      "areaAcres": 14.25,
      "areaM2": 57669
    }
  }]
}
```

**Testing with Single Summits**: Create subset files by filtering the `features` array for specific summit codes.

This script is production-ready with comprehensive logging, error handling, and validation. The ImageServer's generous tile sizes provide excellent safety margins for activation zone boundary calculations.

---

## Agent Testing & Execution Best Practices
*Added based on refactoring session findings*

### Critical Setup Sequence
1. **ALWAYS configure Python environment FIRST**
   ```python
   configure_python_environment(resourcePath="/path/to/project")
   ```

2. **Install ALL dependencies in one operation**
   ```python
   install_python_packages([
       "geopandas", "shapely", "pyproj", "rasterio", "numpy", "matplotlib"
   ], resourcePath="/path/to/project")
   ```

3. **Use FULL virtual environment Python path**
   ```bash
   # CORRECT: Full venv path from configure_python_environment
   /Users/briand/code/azcalc/.venv/bin/python script.py
   
   # WRONG: Generic python command
   python script.py
   ```

### Working Directory Management
- **CRITICAL**: Many scripts depend on specific working directories
- **Always use**: `cd /correct/directory && /full/python/path script.py`
- **Never assume**: Script can run from any directory

### Script Testing Workflow
1. **Check script help FIRST**: `script.py --help`
2. **Understand prerequisites**: Some scripts need data fetching before processing
3. **Use background execution**: `isBackground=true` for long-running processes
4. **Monitor with**: `get_terminal_output(id="terminal_id")`

### Common Failure Patterns & Solutions
- **"Required libraries not available"** → Install dependencies first
- **"File not found"** → Check working directory and script location
- **"Argument required"** → Read script help to understand required parameters
- **Long-running hangs** → Use background execution and monitor output

### macOS-Specific Considerations
- **`timeout` command not available** → Use background execution instead
- **Complex bash pipes may fail** → Keep terminal commands simple
- **Use absolute paths** → Avoid relative path confusion

### Successful Test Pattern
```python
# 1. Environment setup
configure_python_environment(resourcePath="/project/path")
install_python_packages(["required", "packages"])

# 2. Understand script
run_in_terminal("cd /project && /venv/python script.py --help")

# 3. Prerequisites (if needed)
run_in_terminal("cd /project && /venv/python script.py --fetch-only --region X")

# 4. Main execution
run_in_terminal("cd /project && /venv/python script.py --args", isBackground=true)

# 5. Monitor results
get_terminal_output(id="terminal_id")
```

This approach eliminates trial-and-error cycles and ensures reliable script execution.
