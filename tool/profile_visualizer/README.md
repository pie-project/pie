# Profiling Visualizer

A web-based tool for visualizing profiling data from the PIE backend.

## Features

- **Tree View**: Hierarchical view of profiling data with expandable nodes
- **Table View**: Sortable table showing all operations and metrics
- **Flame Graph**: Interactive flame graph with Material Design colors
- **File Browser**: Navigate directories and load JSON files from the server
- **Drag & Drop**: Drop JSON files directly into the browser
- **Percent of Total**: All percentages shown relative to total execution time

## Usage

Inside the `tool` directory:

### Start with a specific profiling file

```bash
python -m profile_visualizer path/to/profiling_result.json
```

### Start without a file (browse from workspace root)

```bash
python -m profile_visualizer
```

### Start at a specified port

```bash
python -m profile_visualizer --port 8080
```

## Structure

```bash
profile_visualizer/
├── __init__.py          # Package initialization
├── __main__.py          # Entry point (python -m profile_visualizer)
├── server.py            # HTTP server and API endpoints
├── static/
│   ├── index.html      # HTML template
│   ├── styles.css      # All CSS styles
│   └── app.js          # JavaScript application
└── README.md           # This file
```

## API Endpoints

- `GET /` - Serve the main HTML page
- `GET /data.json` - Get initially loaded profiling data
- `GET /list_files?path=<dir>` - List JSON files in directory
- `GET /list_dirs?path=<dir>` - List subdirectories
- `GET /load_file?path=<file>` - Load a specific JSON file

## Development

The server runs on `http://localhost:8000` by default.

All static files (HTML, CSS, JS) are served from the `static/` directory.
