#!/usr/bin/env python3
"""
Simple web-based profiling visualization tool.

Usage:
    # Start with an initial file
    python -m profile_visualizer <path_to_profiling_json> [--port PORT]

    # Start without a file (uses parent directory for file search)
    python -m profile_visualizer [--port PORT]

    Example:
    python -m profile_visualizer pie-metal/20251003_205526_profiling_result.json
    python -m profile_visualizer --port 8080
    python -m profile_visualizer --port 8080 pie-metal/20251003_205526_profiling_result.json

Features:
    - Load files from server directory dropdown
    - Drag & drop JSON files
    - Click to browse local files
"""

import argparse
import json
import sys
from pathlib import Path

from .server import start_server


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Web-based profiling visualization tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server on default port (8000) without initial file
  python -m profile_visualizer

  # Load a specific profiling file
  python -m profile_visualizer path/to/profiling.json

  # Start server on custom port
  python -m profile_visualizer --port 8080

  # Load file and use custom port
  python -m profile_visualizer path/to/profiling.json --port 8080""",
)

    parser.add_argument(
        "file",
        nargs="?",
        type=str,
        default=None,
        help="Path to profiling JSON file to load initially (optional)",
    )

    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Port number to run the web server on (default: 8000)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    initial_data = None
    search_dir = None

    # Optional: Load initial file if provided
    if args.file:
        json_path = Path(args.file)
        if not json_path.exists():
            print(f"Error: File not found: {json_path}")
            sys.exit(1)

        # Load profiling data
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                initial_data = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error: Failed to parse JSON from {json_path}: {e}")
            sys.exit(1)

        # Set search directory to the parent directory of the provided file
        search_dir = str(json_path.parent.absolute())
    else:
        # No file provided, use parent directory of script location
        script_path = Path(__file__).resolve()
        search_dir = str(script_path.parent.parent.parent)  # Go up to workspace root

    # Get static files directory
    static_dir = str(Path(__file__).parent / "static")

    # Start the server
    print(f"Starting server on port {args.port}...")
    start_server(
        port=args.port,
        search_dir=search_dir,
        initial_data=initial_data,
        static_dir=static_dir,
    )


if __name__ == "__main__":
    main()
