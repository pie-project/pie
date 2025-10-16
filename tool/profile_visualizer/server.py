"""HTTP server for profiling visualizer."""

import json
import os
import urllib.parse
from http.server import SimpleHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, List, Any, Optional


def add_percentages_to_tree(
    nodes: List[Dict[str, Any]], root_time: Optional[float] = None
) -> None:
    """
    Add percent_of_total to each node in the profiling tree.

    Percentages are calculated relative to the root node's total time.
    This allows easy comparison of any nodes in the tree.

    Args:
        nodes: List of tree nodes (may be nested)
        root_time: Total time of the root node (calculated on first call)
    """
    if not nodes:
        return

    # First call: calculate root_time from the root nodes
    if root_time is None:
        # Use the maximum root node time as the baseline
        root_time = max((node.get("avg_latency_ms", 0) for node in nodes), default=1.0)
        if root_time == 0:
            root_time = 1.0  # Avoid division by zero

    # Add percentages to all nodes (recursively)
    for node in nodes:
        node_time = node.get("avg_latency_ms", 0)
        node["percent_of_total"] = (
            (node_time / root_time) * 100.0 if root_time > 0 else 0.0
        )

        # Recursively process children using the same root_time
        if "children" in node and node["children"]:
            add_percentages_to_tree(node["children"], root_time)


def normalize_profiling_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate profiling data format and enrich with computed fields.

    Args:
        data: Raw JSON data

    Returns:
        Validated unified_profiler data with percent_of_total added to nodes

    Raises:
        ValueError: If required fields are missing
    """
    # Check for required fields
    if "profiling_tree" not in data:
        raise ValueError(
            "Invalid profile format: missing 'profiling_tree' field. "
            "Only 'unified_profiler' format is supported. "
            "Please regenerate your profile data with the latest profiler."
        )

    if "metadata" not in data:
        raise ValueError(
            "Invalid profile format: missing 'metadata' field. "
            "Only 'unified_profiler' format is supported. "
            "Please regenerate your profile data with the latest profiler."
        )

    # Ensure format marker is present
    data["format"] = "unified_profiler"

    # Add percent_of_total to all nodes in the tree
    add_percentages_to_tree(data["profiling_tree"])

    return data


class ProfileVisualizerHandler(SimpleHTTPRequestHandler):
    """Custom handler to serve profiling visualization."""

    profiling_data = None
    search_directory = None
    root_directory = None
    static_dir = None

    def _sanitize_path(self, requested_path):
        """
        Sanitize and validate a path to prevent directory traversal attacks.

        Args:
            requested_path: The path requested by the client

        Returns:
            Sanitized absolute path if valid, root_directory if path escapes bounds
        """
        if not requested_path:
            return self.search_directory

        # Resolve to absolute path and normalize
        abs_path = Path(requested_path).resolve()
        root_path = Path(self.root_directory).resolve()

        # Check if the resolved path is within the root directory
        try:
            abs_path.relative_to(root_path)
            return str(abs_path)
        except ValueError:
            # Path escapes the root directory, return root instead
            return str(root_path)

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/" or self.path == "/index.html":
            self._serve_file("index.html", "text/html")
        elif self.path == "/styles.css":
            self._serve_file("styles.css", "text/css")
        elif self.path == "/app.js":
            self._serve_file("app.js", "application/javascript")
        elif self.path == "/data.json":
            self._serve_json(self.profiling_data)
        elif self.path == "/hardware_specs.json":
            self._serve_hardware_specs()
        elif self.path == "/detect_hardware":
            self._serve_detected_hardware()
        elif self.path.startswith("/list_files"):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            base_path = query.get("path", [""])[0]
            sanitized_path = self._sanitize_path(base_path)
            files = self._list_json_files(sanitized_path)
            self._serve_json(files)
        elif self.path.startswith("/list_dirs"):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            base_path = query.get("path", [""])[0]
            sanitized_path = self._sanitize_path(base_path)
            dirs = self._list_directories(sanitized_path)
            self._serve_json(dirs)
        elif self.path.startswith("/load_file?"):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            filepath = query.get("path", [""])[0]
            sanitized_path = self._sanitize_path(filepath)
            if os.path.exists(sanitized_path):
                try:
                    with open(sanitized_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # Normalize the data before serving
                    normalized_data = normalize_profiling_data(data)
                    self._serve_json(normalized_data)
                except (json.JSONDecodeError, ValueError) as e:
                    self.send_error(400, f"Error parsing JSON: {str(e)}")
                except Exception as e:  # pylint: disable=broad-except
                    self.send_error(400, f"Error loading file: {str(e)}")
            else:
                self.send_error(404, "File not found")
        else:
            self.send_error(404, "File not found")

    def _serve_file(self, filename, content_type):
        """Serve a static file from the static directory."""
        filepath = os.path.join(self.static_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-type", content_type)
            self.end_headers()
            self.wfile.write(content.encode())
        except FileNotFoundError:
            self.send_error(404, f"File not found: {filename}")

    def _serve_json(self, data):
        """Serve JSON data."""
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _serve_hardware_specs(self):
        """Serve hardware specifications JSON."""
        specs_path = os.path.join(self.static_dir, "..", "hardware_specs.json")
        try:
            with open(specs_path, "r", encoding="utf-8") as f:
                specs = json.load(f)
            self._serve_json(specs)
        except FileNotFoundError:
            self.send_error(404, "hardware_specs.json not found")
        except json.JSONDecodeError as e:
            self.send_error(500, f"Error parsing hardware specs: {str(e)}")

    def _serve_detected_hardware(self):
        """Serve auto-detected hardware information."""
        try:
            # Import hardware detection module
            import sys
            backend_path = os.path.join(
                os.path.dirname(self.static_dir), "..", "..", "backend",
                "backend-python"
            )
            if backend_path not in sys.path:
                sys.path.insert(0, backend_path)

            from profiler.hardware_detection import detect_hardware

            hw_info = detect_hardware()
            self._serve_json(hw_info)
        except ImportError as e:
            # Fallback if detection module not available
            self._serve_json({
                "chip": None,
                "memory_gb": None,
                "vendor": "unknown",
                "detected": False,
                "error": str(e),
            })
        except Exception as e:  # pylint: disable=broad-except
            self.send_error(500, f"Error detecting hardware: {str(e)}")

    def _list_json_files(self, base_path=""):
        """List JSON files in the specified directory (non-recursive)."""
        files = []
        if not base_path:
            base_path = self.search_directory

        print(f"[DEBUG] _list_json_files called with base_path='{base_path}'")

        abs_base = os.path.abspath(base_path)

        if os.path.exists(base_path) and os.path.isdir(base_path):
            try:
                items = os.listdir(base_path)
                for item in items:
                    filepath = os.path.join(base_path, item)
                    if os.path.isfile(filepath) and item.endswith(".json"):
                        stat = os.stat(filepath)
                        files.append(
                            {
                                "path": filepath,
                                "name": item,
                                "size": stat.st_size,
                                "mtime": stat.st_mtime,
                                "relative_path": os.path.relpath(
                                    filepath, self.root_directory
                                ),
                            }
                        )
                        print(f"[DEBUG] Added JSON file: {item}")
            except PermissionError as e:
                print(f"[DEBUG] Files: PermissionError: {e}")

        files.sort(key=lambda x: x["mtime"], reverse=True)
        print(f"[DEBUG] Returning {len(files)} JSON files")
        return files

    def _list_directories(self, base_path=""):
        """List subdirectories in the specified directory."""
        dirs = []
        if not base_path:
            base_path = self.search_directory

        print(f"[DEBUG] _list_directories called with base_path='{base_path}'")
        print(f"[DEBUG] search_directory='{self.search_directory}'")
        print(f"[DEBUG] root_directory='{self.root_directory}'")

        abs_base = os.path.abspath(base_path)
        print(f"[DEBUG] abs_base='{abs_base}'")

        if os.path.exists(base_path) and os.path.isdir(base_path):
            try:
                items = os.listdir(base_path)
                print(f"[DEBUG] Found {len(items)} items in directory")
                for item in items:
                    dirpath = os.path.join(base_path, item)
                    if os.path.isdir(dirpath) and not item.startswith("."):
                        dirs.append(
                            {
                                "path": dirpath,
                                "name": item,
                                "relative_path": os.path.relpath(
                                    dirpath, self.root_directory
                                ),
                            }
                        )
                        print(f"[DEBUG] Added directory: {item}")
            except PermissionError as e:
                print(f"[DEBUG] PermissionError: {e}")
        else:
            print(f"[DEBUG] Path does not exist or is not a directory: {base_path}")

        dirs.sort(key=lambda x: x["name"])

        # Add parent directory option, but only if it doesn't escape root_directory
        parent = os.path.dirname(abs_base)
        root_abs = Path(self.root_directory).resolve()
        parent_path = Path(parent).resolve()

        # Only add parent if it's within root_directory bounds
        if parent != abs_base:  # Not at filesystem root
            try:
                parent_path.relative_to(root_abs)
                # Parent is within allowed directory
                dirs.insert(
                    0,
                    {
                        "path": parent,
                        "name": "..",
                        "relative_path": os.path.relpath(parent, self.root_directory)
                        if parent != self.root_directory
                        else ".",
                    },
                )
                print(f"[DEBUG] Added parent directory: {parent}")
            except ValueError:
                # Parent would escape root_directory, don't add it
                print(f"[DEBUG] Parent directory {parent} outside root, not adding")

        print(f"[DEBUG] Returning {len(dirs)} directories")
        return dirs


def start_server(
    port=8000, search_dir=None, initial_data=None, static_dir=None
):
    """Start the HTTP server."""
    # Normalize initial data if provided
    if initial_data:
        initial_data = normalize_profiling_data(initial_data)

    ProfileVisualizerHandler.profiling_data = initial_data
    ProfileVisualizerHandler.search_directory = search_dir or os.getcwd()
    ProfileVisualizerHandler.root_directory = search_dir or os.getcwd()
    ProfileVisualizerHandler.static_dir = static_dir

    server = HTTPServer(("localhost", port), ProfileVisualizerHandler)

    print(f"\n{'='*60}")
    print("🔥 Profiling Visualizer Started!")
    print(f"{'='*60}")
    print(f"\nCurrent working directory: {os.getcwd()}")
    if initial_data:
        print(f"Initially loaded data: Yes")
        print(f"Timestamp: {initial_data.get('timestamp', 'N/A')}")
    print(f"\nSearch directory: {ProfileVisualizerHandler.search_directory}")
    print(f"\n👉 Open in browser: http://localhost:{port}")
    print("\nYou can:")
    print(
        f"  • Select files from the dropdown (scans: {ProfileVisualizerHandler.search_directory})"
    )
    print("  • Drag & drop JSON files into the browser")
    print("  • Click the drop zone to browse files")
    print("\nPress Ctrl+C to stop the server")
    print(f"{'='*60}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped")
