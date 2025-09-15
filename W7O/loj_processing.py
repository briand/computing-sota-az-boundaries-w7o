"""Orchestration functions for SOTA vs Lists of John comparison.

This module extracts the large procedural "main" flow from loj_compare.py
into smaller, testable functions.

Public entry point: run(args)
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, DefaultDict
from collections import defaultdict
import logging
import csv

from config import LOJ_COORD_MATCH_TOLERANCE_M, POINT_BANDS
import config
from utils import (
    setup_association_directories,
    ensure_directories,
    setup_logging,
    generate_log_filename,
)
from sota_api import (
    fetch_sota_summits,
    save_summits_geojson,
    load_summits_from_geojson,
    is_summit_valid,
    fetch_sota_association_regions,
)
from loj_dataset import (
    discover_loj_csv_association,
    discover_loj_association_file,
    load_loj_csv,
    load_loj_geojson,
    find_best_loj_match,
    find_nearest_loj_feature,
    extract_loj_feature_coordinates,
    extract_loj_altitude_ft,
    extract_loj_name,
    extract_loj_id,
)

# ---------------- Name comparison helper -----------------

def _normalize_name(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.replace('"', '')
    return ''.join(c.lower() for c in raw if c.isalnum())

def names_match(sota_name: str, loj_name: str) -> bool:
    """Return True if names considered equivalent.

    Rules:
      1. Exact normalized equality (strip quotes, alnum lowercase).
      2. Special Mount swap: "Mount X" (SOTA order) == "X Mount" (LoJ order).
      3. High point suffix: "Foo HP" (SOTA) == "Foo" (LoJ) when remaining tokens match.
    """
    if not sota_name or not loj_name:
        return False
    n_sota = _normalize_name(sota_name)
    n_loj = _normalize_name(loj_name)
    if n_sota and n_sota == n_loj:
        return True
    # Mount swap check
    sota_tokens = [t for t in sota_name.replace('"','').split() if t]
    loj_tokens = [t for t in loj_name.replace('"','').split() if t]
    if len(sota_tokens) >= 2 and len(loj_tokens) == len(sota_tokens):
        if sota_tokens[0].lower() == 'mount' and loj_tokens[-1].lower() == 'mount':
            sota_core = ' '.join(sota_tokens[1:])
            loj_core = ' '.join(loj_tokens[:-1])
            if _normalize_name(sota_core) == _normalize_name(loj_core):
                return True
    # High point suffix: allow SOTA name ending with 'HP' to match LoJ lacking it
    # Only apply if SOTA has one extra trailing token 'HP' (case-insensitive) and remaining tokens equal.
    if len(sota_tokens) == len(loj_tokens) + 1 and sota_tokens[-1].lower() == 'hp':
        if _normalize_name(' '.join(sota_tokens[:-1])) == _normalize_name(' '.join(loj_tokens)):
            return True
    return False

# ---------------- Summit acquisition helpers -----------------

def load_or_fetch_summits(region: str) -> Path:
    expected_name = f"{config.SOTA_ASSOCIATION}_{region}_summits.geojson"
    summit_path = (config.INPUT_DIR or Path.cwd() / "input") / expected_name
    if summit_path.exists():
        logging.info(f"Summits file already exists, using cached: {summit_path}")
        try:
            summits = load_summits_from_geojson(summit_path)
            active = sum(1 for s in summits if is_summit_valid(s))
            logging.info(f"Loaded {len(summits)} summits ({active} active) from cache")
            return summit_path
        except Exception as e:  # noqa: BLE001
            logging.warning(f"Cached summits file load failed ({e}); refetching...")
    logging.info(f"Fetching SOTA summit list for {config.SOTA_ASSOCIATION}/{region}")
    summits = fetch_sota_summits(region)
    if not summits:
        raise RuntimeError(f"Failed to fetch summits for region {region}")
    summit_path = save_summits_geojson(summits, region)
    logging.info(f"Saved fetched summits to: {summit_path}")
    return summit_path

# ---------------- LoJ dataset discovery -----------------

def load_association_loj(explicit: Optional[str]) -> tuple[Path, List[Dict]]:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"Specified --loj-file not found: {p}")
        path = p
    else:
        path = discover_loj_csv_association()
        if not path:
            path = discover_loj_association_file()
    if not path:
        raise SystemExit("No association-wide LoJ file found (CSV or JSON).")
    if path.suffix.lower() == ".csv":
        feats = load_loj_csv(path)
    else:
        feats = load_loj_geojson(path)
    return path, feats

# ---------------- Region processing -----------------

def process_single_region(region: str, writer, loj_features: List[Dict]) -> tuple[int, int, Set[str], Dict[str, List[Tuple[str, float]]]]:
    summit_file = load_or_fetch_summits(region)
    sota_summits = load_summits_from_geojson(summit_file)
    sota_active = [s for s in sota_summits if is_summit_valid(s)]
    logging.info(f"Using preloaded LoJ features ({len(loj_features)}) for region {region}")

    matches = 0
    unmatched = 0
    matched_loj_ids: Set[str] = set()
    # Map of LoJ id -> list of (SOTA summit code, distance_m) for UNMATCHED_NEAREST cases referencing that LoJ peak
    nearest_reference_map: DefaultDict[str, List[Tuple[str, float]]] = defaultdict(list)
    logging.info(f"Matching {len(sota_active)} SOTA summits for region {region}")
    for summit in sota_active:
        try:
            lat = float(summit.get("latitude", 0.0))
            lon = float(summit.get("longitude", 0.0))
        except (TypeError, ValueError):
            lat, lon = 0.0, 0.0
        summit_alt_ft = summit.get("altFt") or 0.0
        summit_code = summit.get("summitCode")
        sotlas_uri = f"https://sotl.as/summits/{summit_code}" if summit_code else ""
        result = find_best_loj_match(lat, lon, loj_features, LOJ_COORD_MATCH_TOLERANCE_M)
        if result is None:
            unmatched += 1
            nearest = find_nearest_loj_feature(lat, lon, loj_features)
            nearest_name = nearest_lat = nearest_lon = nearest_alt = nearest_uri = nearest_dist = elev_offset = ""
            if nearest:
                nfeat, ndist = nearest
                ncoords = extract_loj_feature_coordinates(nfeat)
                if ncoords:
                    nearest_lat, nearest_lon = ncoords
                nearest_alt_val = extract_loj_altitude_ft(nfeat)
                if isinstance(nearest_alt_val, (int, float)):
                    nearest_alt = nearest_alt_val
                    if isinstance(summit_alt_ft, (int, float)):
                        elev_offset = summit_alt_ft - nearest_alt_val
                nearest_name = extract_loj_name(nfeat) or ""
                nearest_loj_id = extract_loj_id(nfeat) or ""
                nearest_uri = f"https://listsofjohn.com/peak/{nearest_loj_id}" if nearest_loj_id else ""
                nearest_dist = f"{ndist:.2f}"
                # Record reference for later unmatched LoJ augmentation
                if nearest_loj_id and summit_code:
                    nearest_reference_map[str(nearest_loj_id)].append((summit_code, ndist))
            writer.writerow([
                summit_code, summit.get("name"), lat, lon, summit_alt_ft,
                sotlas_uri,
                "UNMATCHED_NEAREST" if nearest else "UNMATCHED",
                nearest_name,
                nearest_lat, nearest_lon, nearest_alt,
                nearest_uri,
                nearest_dist,
                elev_offset,
                ""
            ])
            continue
        feat, dist_m = result
        feats_coords = extract_loj_feature_coordinates(feat)
        if feats_coords is None:
            unmatched += 1
            writer.writerow([
                summit_code, summit.get("name"), lat, lon, summit_alt_ft,
                sotlas_uri,
                "NO_COORDS",
                "", "", "", "",
                "", "", "", ""
            ])
            continue
        loj_lat, loj_lon = feats_coords
        loj_alt_ft = extract_loj_altitude_ft(feat) or ""
        elev_offset = ""
        if isinstance(loj_alt_ft, (int, float)) and isinstance(summit_alt_ft, (int, float)):
            elev_offset = summit_alt_ft - loj_alt_ft
        loj_name = extract_loj_name(feat) or ""
        loj_id = extract_loj_id(feat) or ""
        loj_uri = f"https://listsofjohn.com/peak/{loj_id}" if loj_id else ""
        def _norm(n: str):
            if not n:
                return ''
            n = n.replace('"', '')
            return ''.join(c.lower() for c in n if c.isalnum())
        sota_clean = _norm(summit.get("name") or "")
        loj_clean = _norm(loj_name)  # retained for potential future diagnostics
        name_equal = names_match(summit.get("name") or "", loj_name)
        status = "MATCH" if name_equal else "NAME_MISMATCH"
        # Band crossing check (only for matched coordinate pairs, regardless of name mismatch)
        # If summit_alt_ft and loj_alt_ft straddle any POINT_BANDS threshold, escalate status.
        try:
            if isinstance(summit_alt_ft, (int, float)) and isinstance(loj_alt_ft, (int, float)):
                s_alt = float(summit_alt_ft)
                l_alt = float(loj_alt_ft)
                low = min(s_alt, l_alt)
                high = max(s_alt, l_alt)
                for band in POINT_BANDS:
                    if low < band < high:
                        status = "POINT_BAND_CHANGE"
                        break
        except Exception:
            pass
        writer.writerow([
            summit_code, summit.get("name"), lat, lon, summit_alt_ft,
            sotlas_uri,
            status,
            loj_name,
            loj_lat, loj_lon, loj_alt_ft,
            loj_uri,
            f"{dist_m:.2f}", f"{elev_offset}", str(name_equal)
        ])
        matches += 1
        if loj_id:
            matched_loj_ids.add(str(loj_id))
    logging.info(f"Region {region} done: matches={matches} unmatched={unmatched}")
    return matches, unmatched, matched_loj_ids, nearest_reference_map

# ---------------- All regions -----------------

def run_all_regions(args, association_loj_path: Path, loj_features: List[Dict]):
    pseudo_region = "ALL"
    setup_association_directories(pseudo_region)
    ensure_directories()
    log_file = generate_log_filename(pseudo_region)
    setup_logging(log_file, args.quiet)
    logging.info("SOTA vs LoJ Comparison - ALL REGIONS MODE")
    logging.info("=" * 64)
    logging.info(f"Association: {config.SOTA_ASSOCIATION}  Mode: ALL REGIONS")
    logging.info(f"Log file: {log_file}")

    regions_map = fetch_sota_association_regions(config.SOTA_ASSOCIATION)
    if not regions_map:
        logging.error("No regions retrieved; aborting.")
        return
    region_codes = sorted(regions_map.keys())
    logging.info(f"Found {len(region_codes)} regions: {', '.join(region_codes)}")

    aggregate_csv = f"loj_compare_{config.SOTA_ASSOCIATION}_ALL.csv"
    aggregate_path = Path(aggregate_csv)
    headers = [
        "sota_summit_code","sota_name","sota_lat","sota_lon","sota_alt_ft","sota_sotlas_uri","match_status",
        "loj_name","loj_lat","loj_lon","loj_alt_ft","loj_uri","coord_offset_m","elev_offset_ft","name_match_equality"
    ]
    total_matches = 0
    total_unmatched = 0
    regions_processed = 0
    matched_loj_ids: Set[str] = set()
    # Aggregate of nearest references across regions
    aggregated_nearest_refs: DefaultDict[str, List[Tuple[str, float]]] = defaultdict(list)
    with open(aggregate_path, "w", newline="") as fagg:
        writer = csv.writer(fagg)
        writer.writerow(headers)
        for rcode in region_codes:
            try:
                setup_association_directories(rcode)
                ensure_directories()
                logging.info("")
                logging.info(f"=== Region {rcode} ===")
                region_matches, region_unmatched, region_matched_ids, region_nearest_refs = process_single_region(rcode, writer, loj_features)
                total_matches += region_matches
                total_unmatched += region_unmatched
                matched_loj_ids.update(region_matched_ids)
                for lid, refs in region_nearest_refs.items():
                    aggregated_nearest_refs[lid].extend(refs)
                regions_processed += 1
            except Exception as e:  # noqa: BLE001
                logging.error(f"Region {rcode} failed: {e}")
    logging.info("")
    logging.info(f"ALL REGIONS COMPLETE: regions={regions_processed} matches={total_matches} unmatched={total_unmatched}")
    logging.info(f"Aggregate CSV: {aggregate_path}")
    logging.info(f"Association-wide LoJ file: {association_loj_path}")

    all_loj_ids: Set[str] = set()
    for feat in loj_features:
        lid = extract_loj_id(feat)
        if lid:
            all_loj_ids.add(str(lid))
    unmatched_loj_ids = sorted(all_loj_ids - matched_loj_ids)
    if unmatched_loj_ids:
        unmatched_csv = f"loj_unmatched_{config.SOTA_ASSOCIATION}_ALL.csv"
        unmatched_path = Path(unmatched_csv)
        logging.info(f"Writing unmatched LoJ feature list: {unmatched_path} (count={len(unmatched_loj_ids)})")
        with open(unmatched_path, "w", newline="") as fu:
            uwriter = csv.writer(fu)
            uwriter.writerow([
                "loj_id","loj_name","loj_lat","loj_lon","loj_alt_ft","loj_uri",
                "nearest_unmatched_sota_summit_codes","nearest_unmatched_sota_min_dist_m"
            ])
            for feat in loj_features:
                lid = extract_loj_id(feat)
                if not lid or str(lid) not in unmatched_loj_ids:
                    continue
                lid_str = str(lid)
                coords = extract_loj_feature_coordinates(feat) or ("","")
                alt_ft = extract_loj_altitude_ft(feat) or ""
                name = extract_loj_name(feat) or ""
                uri = f"https://listsofjohn.com/peak/{lid}" if lid else ""
                refs = aggregated_nearest_refs.get(lid_str) or []
                if refs:
                    # deduplicate summit codes keeping shortest distance per code
                    best_by_code: Dict[str, float] = {}
                    for sc, d in refs:
                        if sc not in best_by_code or d < best_by_code[sc]:
                            best_by_code[sc] = d
                    codes_sorted = sorted(best_by_code.keys())
                    codes_joined = ";".join(codes_sorted)
                    min_dist = min(best_by_code.values())
                    min_dist_fmt = f"{min_dist:.2f}"
                else:
                    codes_joined = ""
                    min_dist_fmt = ""
                uwriter.writerow([lid, name, coords[0], coords[1], alt_ft, uri, codes_joined, min_dist_fmt])
        logging.info(f"Unmatched LoJ CSV: {unmatched_path}")
    else:
        logging.info("All LoJ features matched at least one SOTA summit (no unmatched LoJ list created)")

# ---------------- Single region -----------------

def run_single_region(args, association_loj_path: Path, loj_features: List[Dict]):
    setup_association_directories(args.region)
    ensure_directories()
    log_file = generate_log_filename(args.region)
    setup_logging(log_file, args.quiet)
    logging.info("SOTA vs LoJ Comparison - SINGLE REGION MODE")
    logging.info("=" * 64)
    logging.info(f"Association: {config.SOTA_ASSOCIATION}  Region: {args.region}")
    logging.info(f"Log file: {log_file}")

    summit_file = load_or_fetch_summits(args.region)
    sota_summits = load_summits_from_geojson(summit_file)
    sota_active = [s for s in sota_summits if is_summit_valid(s)]
    logging.info("")
    logging.info("SOTA summit list ready.")

    logging.info(f"Using association-wide LoJ file: {association_loj_path}")
    logging.info(f"Loaded {len(loj_features)} LoJ features")

    comparison_csv = f"loj_compare_{config.SOTA_ASSOCIATION}_{args.region}.csv"
    comparison_path = Path(comparison_csv)
    logging.info("")
    logging.info(f"Creating comparison CSV: {comparison_path}")

    headers = [
        "sota_summit_code","sota_name","sota_lat","sota_lon","sota_alt_ft","sota_sotlas_uri","match_status",
        "loj_name","loj_lat","loj_lon","loj_alt_ft","loj_uri","coord_offset_m","elev_offset_ft","name_match_equality"
    ]

    with open(comparison_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(headers)
        logging.info(f"Matching SOTA summits against LoJ features (tolerance {LOJ_COORD_MATCH_TOLERANCE_M:.1f} m)")
        matches = 0
        unmatched = 0
        for summit in sota_active:
            try:
                lat = float(summit.get("latitude", 0.0))
                lon = float(summit.get("longitude", 0.0))
            except (TypeError, ValueError):
                lat, lon = 0.0, 0.0
            summit_alt_ft = summit.get("altFt") or 0.0
            summit_code = summit.get("summitCode")
            sotlas_uri = f"https://sotl.as/summits/{summit_code}" if summit_code else ""
            result = find_best_loj_match(lat, lon, loj_features, LOJ_COORD_MATCH_TOLERANCE_M)
            if result is None:
                unmatched += 1
                logging.error(f"No LoJ match within {LOJ_COORD_MATCH_TOLERANCE_M:.0f}m for {summit_code}")
                nearest = find_nearest_loj_feature(lat, lon, loj_features)
                nearest_name = nearest_lat = nearest_lon = nearest_alt = nearest_uri = nearest_dist = elev_offset = ""
                if nearest:
                    nfeat, ndist = nearest
                    ncoords = extract_loj_feature_coordinates(nfeat)
                    if ncoords:
                        nearest_lat, nearest_lon = ncoords
                    nearest_alt_val = extract_loj_altitude_ft(nfeat)
                    if isinstance(nearest_alt_val, (int, float)):
                        nearest_alt = nearest_alt_val
                        if isinstance(summit_alt_ft, (int, float)):
                            elev_offset = summit_alt_ft - nearest_alt_val
                    nearest_name = extract_loj_name(nfeat) or ""
                    loj_id = extract_loj_id(nfeat) or ""
                    nearest_uri = f"https://listsofjohn.com/peak/{loj_id}" if loj_id else ""
                    nearest_dist = f"{ndist:.2f}"
                writer.writerow([
                    summit_code, summit.get("name"), lat, lon, summit_alt_ft,
                    sotlas_uri,
                    "UNMATCHED_NEAREST" if nearest else "UNMATCHED",
                    nearest_name,
                    nearest_lat, nearest_lon, nearest_alt,
                    nearest_uri,
                    nearest_dist,
                    elev_offset,
                    ""
                ])
                continue
            feat, dist_m = result
            feats_coords = extract_loj_feature_coordinates(feat)
            if feats_coords is None:
                unmatched += 1
                nearest = find_nearest_loj_feature(lat, lon, loj_features)
                nearest_name = nearest_lat = nearest_lon = nearest_alt = nearest_uri = nearest_dist = elev_offset = ""
                if nearest:
                    nfeat, ndist = nearest
                    ncoords = extract_loj_feature_coordinates(nfeat)
                    if ncoords:
                        nearest_lat, nearest_lon = ncoords
                    nearest_alt_val = extract_loj_altitude_ft(nfeat)
                    if isinstance(nearest_alt_val, (int, float)):
                        nearest_alt = nearest_alt_val
                        if isinstance(summit_alt_ft, (int, float)):
                            elev_offset = summit_alt_ft - nearest_alt_val
                    nearest_name = extract_loj_name(nfeat) or ""
                    loj_id = extract_loj_id(nfeat) or ""
                    nearest_uri = f"https://listsofjohn.com/peak/{loj_id}" if loj_id else ""
                    nearest_dist = f"{ndist:.2f}"
                writer.writerow([
                    summit_code, summit.get("name"), lat, lon, summit_alt_ft,
                    sotlas_uri,
                    "NO_COORDS_NEAREST" if nearest else "NO_COORDS",
                    nearest_name,
                    nearest_lat, nearest_lon, nearest_alt,
                    nearest_uri,
                    nearest_dist,
                    elev_offset,
                    ""
                ])
                continue
            loj_lat, loj_lon = feats_coords
            loj_alt_ft = extract_loj_altitude_ft(feat) or ""
            elev_offset = ""
            if isinstance(loj_alt_ft, (int, float)) and isinstance(summit_alt_ft, (int, float)):
                elev_offset = summit_alt_ft - loj_alt_ft
            loj_name = extract_loj_name(feat) or ""
            loj_id = extract_loj_id(feat) or ""
            loj_uri = f"https://listsofjohn.com/peak/{loj_id}" if loj_id else ""
            def _norm(n: str):
                if not n:
                    return ''
                n = n.replace('"', '')
                return ''.join(c.lower() for c in n if c.isalnum())
            sota_clean = _norm(summit.get("name") or "")
            loj_clean = _norm(loj_name)  # retained for potential future diagnostics
            name_equal = names_match(summit.get("name") or "", loj_name)
            status = "MATCH" if name_equal else "NAME_MISMATCH"
            try:
                if isinstance(summit_alt_ft, (int, float)) and isinstance(loj_alt_ft, (int, float)):
                    s_alt = float(summit_alt_ft)
                    l_alt = float(loj_alt_ft)
                    low = min(s_alt, l_alt)
                    high = max(s_alt, l_alt)
                    for band in POINT_BANDS:
                        if low < band < high:
                            status = "POINT_BAND_CHANGE"
                            break
            except Exception:
                pass
            writer.writerow([
                summit_code, summit.get("name"), lat, lon, summit_alt_ft,
                sotlas_uri,
                status,
                loj_name,
                loj_lat, loj_lon, loj_alt_ft,
                loj_uri,
                f"{dist_m:.2f}", f"{elev_offset}", str(name_equal)
            ])
            matches += 1
        logging.info(f"Matches: {matches}  Unmatched: {unmatched}")
    logging.info("")
    logging.info("Comparison stage complete.")
    logging.info(f"Output CSV: {comparison_path}")
    logging.info(f"SOTA summit file: {summit_file}")
    logging.info(f"Association-wide LoJ file: {association_loj_path}")

# ---------------- Top-level run -----------------

def run(args):
    if not args.all_regions and not args.region:
        raise SystemExit("Must specify either --region or --all-regions")

    # Pre-create base directory structure to allow LoJ discovery
    if args.all_regions:
        setup_association_directories("ALL")
    else:
        setup_association_directories(args.region)
    ensure_directories()

    association_loj_path, loj_features = load_association_loj(args.loj_file)

    if args.all_regions:
        run_all_regions(args, association_loj_path, loj_features)
    else:
        run_single_region(args, association_loj_path, loj_features)

