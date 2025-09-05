#!/usr/bin/env python3
"""
SOTA Activation Zone Collator

This script collates individual activation zone GeoJSON files from a folder
into a single combined GeoJSON file, for easier loading into mapping apps like CalTopo or Gaia.

Usage:
    python collate_activation_zones.py <folder_path>

Example:
    python collate_activation_zones.py W7O_NC/output
    # Creates: W7O_NC_az_all.geojson

The script:
1. Reads all .geojson files from the specified folder
2. Extracts the naming pattern (e.g., W7O_NC from W7O_NC_001.geojson)
3. Combines all features into a single FeatureCollection
4. Outputs as {pattern}_az_all.geojson
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import re


def extract_naming_pattern(filenames: List[str]) -> Optional[str]:
    """
    Extract the common naming pattern from a list of filenames.
    
    Args:
        filenames: List of filenames like ['W7O_NC_001.geojson', 'W7O_NC_002.geojson']
    
    Returns:
        Common pattern like 'W7O_NC' or None if no pattern found
    """
    if not filenames:
        return None
    
    # Pattern: extract everything before the last underscore and number
    # W7O_NC_001.geojson -> W7O_NC
    # W7W_LC_123.geojson -> W7W_LC
    pattern = re.compile(r'^(.+)_\d+\.geojson$')
    
    patterns = set()
    for filename in filenames:
        match = pattern.match(filename)
        if match:
            patterns.add(match.group(1))
    
    if len(patterns) == 1:
        return patterns.pop()
    elif len(patterns) > 1:
        print(f"Warning: Multiple naming patterns found: {patterns}")
        # Return the most common one
        return max(patterns, key=lambda p: sum(1 for f in filenames if f.startswith(p)))
    else:
        return None


def load_geojson_file(filepath: Path) -> Optional[Dict[str, Any]]:
    """
    Load a GeoJSON file and return its contents.
    
    Args:
        filepath: Path to the GeoJSON file
    
    Returns:
        GeoJSON data as dictionary or None if error
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def collate_activation_zones(folder_path: Path) -> bool:
    """
    Collate all activation zone GeoJSON files in a folder.
    
    Args:
        folder_path: Path to folder containing GeoJSON files
    
    Returns:
        True if successful, False otherwise
    """
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        return False
    
    if not folder_path.is_dir():
        print(f"Error: '{folder_path}' is not a directory")
        return False
    
    # Find all GeoJSON files (excluding any existing _az_all.geojson files)
    all_geojson_files = list(folder_path.glob("*.geojson"))
    geojson_files = [f for f in all_geojson_files if not f.name.endswith("_az_all.geojson")]
    
    if not geojson_files:
        if all_geojson_files:
            print(f"Error: Only found collated files (*_az_all.geojson) in '{folder_path}'")
            print("Please run on a folder with individual activation zone files")
        else:
            print(f"Error: No .geojson files found in '{folder_path}'")
        return False
    
    print(f"Found {len(geojson_files)} GeoJSON files in '{folder_path}'")
    
    # Extract naming pattern
    filenames = [f.name for f in geojson_files]
    pattern = extract_naming_pattern(filenames)
    
    if not pattern:
        print("Error: Could not determine naming pattern from files")
        print("Expected pattern: PREFIX_###.geojson (e.g., W7O_NC_001.geojson)")
        return False
    
    print(f"Detected naming pattern: {pattern}")
    
    # Create output filename
    output_filename = f"{pattern}_az_all.geojson"
    output_path = folder_path / output_filename
    
    # Collect all features
    all_features = []
    successful_files = 0
    
    # Sort files by name for consistent ordering
    geojson_files.sort(key=lambda f: f.name)
    
    for geojson_file in geojson_files:
        print(f"Processing: {geojson_file.name}")
        
        data = load_geojson_file(geojson_file)
        if data is None:
            continue
        
        # Validate structure
        if data.get('type') != 'FeatureCollection':
            print(f"  Warning: {geojson_file.name} is not a FeatureCollection, skipping")
            continue
        
        features = data.get('features', [])
        if not features:
            print(f"  Warning: {geojson_file.name} has no features, skipping")
            continue
        
        # Add all features from this file
        for feature in features:
            # Optionally add source file info to properties
            if feature.get('properties') is None:
                feature['properties'] = {}
            
            feature['properties']['source_file'] = geojson_file.name
            all_features.append(feature)
        
        successful_files += 1
        print(f"  Added {len(features)} feature(s)")
    
    if not all_features:
        print("Error: No valid features found in any files")
        return False
    
    # Create combined GeoJSON
    combined_geojson = {
        "type": "FeatureCollection",
        "features": all_features,
        "properties": {
            "title": f"{pattern} Activation Zones"
        }
    }
    
    # Write output file
    try:
        with open(output_path, 'w') as f:
            json.dump(combined_geojson, f, indent=2)
        
        print(f"\nSuccess! Created: {output_path}")
        print(f"Combined {len(all_features)} features from {successful_files} files")
        return True
        
    except Exception as e:
        print(f"Error writing output file: {e}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Collate SOTA activation zone GeoJSON files into a single file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collate all files in W7O_NC/output folder
  python collate_activation_zones.py W7O_NC/output
  
  # Collate files in current directory
  python collate_activation_zones.py .
  
  # Collate files in specific path
  python collate_activation_zones.py /path/to/activation/zones
  
Output:
  Creates a file named {pattern}_az_all.geojson where {pattern} is derived
  from the input filenames (e.g., W7O_NC_001.geojson -> W7O_NC_az_all.geojson)
        """
    )
    
    parser.add_argument(
        'folder',
        type=str,
        help='Path to folder containing activation zone GeoJSON files'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    folder_path = Path(args.folder)
    
    if args.verbose:
        print(f"Collating activation zones from: {folder_path.absolute()}")
    
    success = collate_activation_zones(folder_path)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
