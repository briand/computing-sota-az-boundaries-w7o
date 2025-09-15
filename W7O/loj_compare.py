"""
Main entry point for SOTA - Lists of John compare processing.
Handles argument parsing, logging setup, and directory management.
"""
import argparse
import config
from loj_processing import run as run_processing


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="SOTA vs Lists of John comparison (Step 1: retrieve SOTA regional summit list)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch NC region summits (creates/uses cached geojson then stops)
  python loj_compare.py --region NC

  # Quiet mode (log only to file)
  python loj_compare.py --region NC --quiet
        """
    )
    parser.add_argument(
        "-r", "--region",
        required=False,
        help="SOTA region code (e.g., NC, LC)"
    )
    parser.add_argument(
        "-a", "--association",
        required=False,
        help="Override SOTA association code (e.g., W7O, W7W). Defaults to current config value.")
    parser.add_argument(
        "--all-regions",
        action="store_true",
        help="Process all regions for the association (overrides --region)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress stdout output (log file still written)"
    )
    parser.add_argument(
        "--loj-file",
        type=str,
        required=False,
        help="Explicit path to a Lists of John file (.csv preferred, or .json/.geojson)"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    # Apply association override early so downstream modules referencing config see updated value
    if args.association:
        config.SOTA_ASSOCIATION = args.association.upper()
    run_processing(args)


if __name__ == "__main__":
    main()
