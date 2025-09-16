"""Main entry point for SOTA - Lists of John compare processing with dynamic config override."""
import argparse
import importlib
import sys
import types

# Import default config first; may be replaced by --config-module
import config  # type: ignore


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
    parser.add_argument(
        "--config-module",
        type=str,
        required=False,
        help="Python module name to use instead of default 'config' (e.g., config_W7W). Must be importable."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    # If a config module override is provided, import it and alias to 'config'
    if args.config_module:
        mod_name = args.config_module
        try:
            override_mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as e:
            print(f"ERROR: Could not import config override module '{mod_name}': {e}", file=sys.stderr)
            sys.exit(2)
        # Replace existing 'config' module reference in sys.modules so downstream imports see override
        sys.modules['config'] = override_mod  # type: ignore
        # Rebind local name as well
        globals()['config'] = override_mod  # noqa: F401
    # Apply association override after potential config module replacement
    # Apply association override early so downstream modules referencing config see updated value
    if args.association:
        config.SOTA_ASSOCIATION = args.association.upper()
    # Import run_processing only after config potentially replaced
    from loj_processing import run as run_processing  # noqa: WPS433
    run_processing(args)


if __name__ == "__main__":
    main()
