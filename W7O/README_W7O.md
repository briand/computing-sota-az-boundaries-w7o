## W7O SOTA Activation Zone Processing

The Activation #### Elevation Data Sources
- **Source**: USGS 3DEP ImageServer (`https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer`)
- **Resolution**: ~3 meter pixel resolution
- **CRS**: EPSG:6557 (NAD83(2011) / Oregon Statewide Lambert)
- **Buffer Strategy**: Request 1km × 1km around summit, server returns ~3.4km × 3.4km minimum tiles
- **Data Coverage**: Server provides generous margins (~1.3km+) around activation zones for safety
- **File Size**: Standardized ~1MB TIFF files for efficient caching and transferom the W7O ARM:
> The SOTA general rules also state that radio operations must take place within a summit's
> Activation Zone, which, in the case of the W7O-Oregon association, is an area within 82
> vertical feet of the actual summit point. The Activation Zone is a single,
> "unbroken" area which can be visualized by drawing a closed shape on a map, following a contour line 82
> feet below the summit point. Another way to describe the activation zone is any place that
> has a route to the summit point that does not dip below 82 feet of the summit point.

### Python Implementation Overview

The current Python implementation (`process_sota_python.py`) uses high-resolution contour generation to create precise activation zone boundaries following natural terrain features. The script automatically detects elevation data units (feet vs meters) and generates smooth polygon boundaries using matplotlib contour extraction.

#### Key Features
- **Contour-based boundaries**: Uses matplotlib to generate smooth elevation contours instead of blocky raster masks
- **Automatic unit detection**: Detects whether elevation data is in feet or meters and adjusts calculations accordingly
- **High-resolution output**: Generates activation zones with 300-500+ coordinate points for accurate boundaries
- **Robust caching**: Stores elevation data locally to avoid repeated downloads
- **Startup validation**: Comprehensive dependency checking with helpful error messages
- **Flexible processing modes**: Can fetch data only, process from existing files, or do both in one operation

### Command Line Usage

```bash
# Fetch W7O/NC summits and save to input directory (no processing)
python process_sota_python.py --fetch-only --association W7O --region NC

# Process summits from existing GeoJSON file
python process_sota_python.py --summits input/W7O_NC.geojson

# Fetch and process all summits in one operation
python process_sota_python.py --association W7O --region NC
```TA Activation Zone Processing

The Activation Zone, from the W7O ARM:
> The SOTA general rules also state that radio operations must take place within a summit’s
> Activation Zone, which, in the case of the W7O-Oregon association, is an area within 82
> vertical feet of the actual summit point. The Activation Zone is a single,
> “unbroken” area which can be visualized by drawing a closed shape on a map, following a contour line 82
> feet below the summit point. Another way to describe the activation zone is any place that
> has a route to the summit point that does not dip below 82 feet of the summit point.

### Input Data Sources

#### Summit Locations
- **Source**: SOTA API v2 (`https://api2.sota.org.uk/api/regions/W7O/{region}`)
- **Regions**: Oregon regions (NC, CS, CE, etc.)
- **Key Fields**: 
  - `summitCode`: Unique identifier (e.g., "W7O/NC-001")
  - `longitude`, `latitude`: WGS84 coordinates (canonical positions)
  - `altM`: Summit elevation in meters
  - `name`: Summit name
  - `validTo`: Retirement date (used to filter out retired summits)

#### Elevation Data
- **Source**: DOGAMI Oregon Digital Terrain Model Mosaic (ESRI ImageServer)
- **Service**: `DIGITAL_TERRAIN_MODEL_MOSAIC`
- **URL**: `https://gis.dogami.oregon.gov/arcgis/rest/services/lidar/DIGITAL_TERRAIN_MODEL_MOSAIC/ImageServer`
- **Format**: GeoTIFF, 32-bit float
- **Units**: Feet (automatically detected and converted)
- **Resolution**: Variable (typically 1-3 meter resolution)
- **Buffer**: 500m radius around each summit for processing

### Processing Workflow

1. **Dependency Validation**: Check all required Python libraries at startup (geopandas, rasterio, matplotlib, etc.)
2. **Summit Data Acquisition**: 
   - Fetch from SOTA API or load from existing GeoJSON file
   - Filter out retired summits using `validTo` dates
   - Save to `input/{association}_{region}.geojson` format
3. **Elevation Data Processing**: For each summit:
   - Check cache for existing elevation data (`cache/elev_{lon}_{lat}_500m.tif`)
   - Download 500m buffered elevation data from ImageServer if not cached
   - Automatically detect data units (feet vs meters) by examining value ranges
4. **Activation Zone Generation**:
   - Calculate target elevation = summit_elevation - 82 feet (25 meters)
   - Generate contour lines at activation elevation using matplotlib
   - Extract contour segments and convert to Shapely polygons
   - Select polygon containing summit point, or largest polygon if none contain summit
   - Transform coordinates to WGS84 if needed
5. **Output Generation**: Save individual GeoJSON files with simplified properties

### Cache Strategy

#### Cache File Naming
- Pattern: `elev_{lon}_{lat}_{buffer_m}m.tif`
- Coordinates use full precision (6 decimal places)
- Example: `elev_-123.456789_45.123456_500m.tif`
- Buffer fixed at 500m for all summits

#### Cache Benefits
- Eliminates repeated downloads for repeated runs
- Faster processing for re-runs
- Offline processing capability after initial download

### Per-Summit Output Files

Files are saved to `output/` subfolder with naming pattern `{ASSOCIATION}_{REGION}_{NUMBER}.geojson`:

- `output/W7O_NC_001.geojson` (W7O/NC-001)
- `output/W7O_NC_002.geojson` (W7O/NC-002)  
- `output/W7O_NC_010.geojson` (W7O/NC-010)

### Output Schema: Activation Zones GeoJSON

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[lon, lat], ...]]
      },
      "properties": {
        "title": "W7O/NC-001"
      }
    }
  ]
}
```

The output format has been designed to work with SOTLAS.

### ImageServer Query Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `bbox` | `xmin,ymin,xmax,ymax` | 500m radius bounding box in degrees |
| `bboxSR` | `4326` | Spatial reference (WGS84) |
| `size` | `512,512` | Output image dimensions |
| `format` | `tiff` | Output format |
| `pixelType` | `F32` | 32-bit float elevation values |
| `interpolation` | `RSP_BilinearInterpolation` | Resampling method |
| `f` | `image` | Response format |

### Processing Considerations

- **Contour Generation**: Uses matplotlib for high-resolution contour extraction (300-500+ coordinate points)
- **Unit Detection**: Automatically detects feet vs meters in elevation data by examining value ranges
- **Polygon Selection**: Prioritizes polygons containing summit point, falls back to largest polygon
- **Retirement Filtering**: Automatically filters out summits with `validTo` dates in the past
- **Coordinate Precision**: Maintains full coordinate precision throughout processing
- **Memory Management**: Uses small matplotlib figures and proper cleanup to minimize memory usage
- **Error Handling**: Comprehensive startup dependency validation and graceful failure handling