#!/usr/bin/env python3
"""
Manual test of ImageServer connectivity functionality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import requests
import json
import tempfile
import rasterio
import numpy as np
from W7O.generate_sota_az import build_imageserver_query_url, ARCGIS_IMAGESERVER

def test_imageserver_manually():
    """Test ImageServer connectivity manually"""
    
    print(f"Testing ImageServer: {ARCGIS_IMAGESERVER}")
    
    # Test 1: Server accessibility
    try:
        info_url = f"{ARCGIS_IMAGESERVER}?f=json"
        print(f"Testing server info URL: {info_url}")
        response = requests.get(info_url, timeout=30)
        response.raise_for_status()
        
        server_info = response.json()
        print(f"✓ Server responding: {server_info.get('name', 'Unknown service')}")
        print(f"✓ Server capabilities: {server_info.get('capabilities', 'Unknown')}")
        
    except Exception as e:
        print(f"✗ Server accessibility test failed: {e}")
        return False
    
    # Test 2: TIFF data retrieval
    test_lon, test_lat = -122.676, 45.515  # Portland area
    
    try:
        query_url = build_imageserver_query_url(test_lon, test_lat, buffer_m=1000)
        print(f"\nTesting TIFF download: {test_lon}, {test_lat}")
        print(f"Query URL: {query_url}")
        
        response = requests.get(query_url, timeout=60)
        response.raise_for_status()
        
        print(f"✓ Download successful: {len(response.content):,} bytes")
        
        # Test TIFF validity
        temp_file = Path(tempfile.mktemp(suffix='.tif'))
        try:
            with open(temp_file, 'wb') as f:
                f.write(response.content)
            
            with rasterio.open(temp_file) as src:
                print(f"✓ Valid TIFF: {src.width}x{src.height}, CRS: {src.crs}")
                elevation_data = src.read(1)
                valid_pixels = np.sum(~np.isnan(elevation_data))
                print(f"✓ Data validation: {valid_pixels:,} valid pixels")
                
        finally:
            if temp_file.exists():
                temp_file.unlink()
                
    except Exception as e:
        print(f"✗ TIFF test failed: {e}")
        return False
    
    print("\n✓ All ImageServer tests passed!")
    return True

if __name__ == "__main__":
    success = test_imageserver_manually()
    sys.exit(0 if success else 1)
