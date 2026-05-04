#!/usr/bin/env python3
"""
Generate WIT bindings for inferlet using componentize-py.

Usage:
    python scripts/generate_bindings.py

This generates Python bindings in src/inferlet/bindings/ that provide
typed interfaces to the WIT imports (inferlet:core/runtime, etc.).
"""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    # Get paths relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    sdk_root = project_root.parent
    pie_root = sdk_root.parent

    wit_path = pie_root / "runtime" / "wit"
    output_path = project_root / "src" / "inferlet" / "bindings"

    if not wit_path.exists():
        print(f"Error: WIT directory not found at {wit_path}", file=sys.stderr)
        return 1

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate bindings using componentize-py
    # The world is "inferlet" as defined in runtime/wit/world.wit
    cmd = [
        "componentize-py",
        "-d",
        str(wit_path),
        "-w",
        "inferlet",
        "bindings",
        str(output_path),
    ]

    print(f"Generating bindings from {wit_path}...")
    print(f"Output: {output_path}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error generating bindings:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return result.returncode
        print(result.stdout)
        print("Bindings generated successfully!")
        return 0
    except FileNotFoundError:
        print(
            "Error: componentize-py not found. Install with: pip install componentize-py",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
