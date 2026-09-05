"""Build command implementation for Bakery.

This module implements the `bakery build` subcommand for building
JavaScript/TypeScript, Python, and Rust inferlets into WebAssembly components.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

from rich.panel import Panel
from .console import console
from . import path as path_utils
from . import py_runtime
import typer

INFERLET_JS_PACKAGE = "@pie-project/inferlet"


def read_package_name(project_dir: Path) -> str:
    """Read the package name from Pie.toml in the project directory.

    Args:
        project_dir: Path to the project directory containing Pie.toml.

    Returns:
        The package name from [package].name in Pie.toml.

    Raises:
        FileNotFoundError: If Pie.toml doesn't exist.
        ValueError: If Pie.toml is invalid or missing [package].name.
    """
    pie_toml_path = project_dir / "Pie.toml"

    if not pie_toml_path.exists():
        raise FileNotFoundError(
            f"Pie.toml not found in {project_dir}. "
            "Please create a Pie.toml with [package].name field."
        )

    try:
        pie_toml = tomllib.loads(pie_toml_path.read_text())
    except Exception as e:
        raise ValueError(f"Failed to parse Pie.toml: {e}")

    package = pie_toml.get("package", {})
    name = package.get("name")

    if not name:
        raise ValueError(
            f"Missing [package].name in {pie_toml_path}. "
            "Please add a 'name' field under [package]."
        )

    return name


def to_python_ident(name: str) -> str:
    """Convert a package name to a valid Python identifier.

    Python identifiers use snake_case (underscores) and can't have hyphens.
    """
    # Replace hyphens with underscores
    return name.replace("-", "_")


def find_command(cmd: str) -> str | None:
    """Locate ``cmd`` on PATH or alongside the running interpreter.

    The interpreter-bin fallback matters when bakery is invoked through
    ``pie build`` via the venv's absolute path from outside the venv's
    activation: PATH won't include the venv's bin/, but tools like
    ``componentize-py`` (installed by ``factored-componentize-py``) live
    there.
    """
    candidate = Path(sys.executable).parent / cmd
    if candidate.exists():
        return str(candidate)
    if found := shutil.which(cmd):
        return found
    return None


def command_exists(cmd: str) -> bool:
    return find_command(cmd) is not None


def resolve_command(cmd: str) -> str:
    """Resolved path for ``cmd``, or the bare name as a fallback so the
    OS surfaces the usual "command not found" error."""
    return find_command(cmd) or cmd


def get_installed_python_package_path(package_name: str) -> Path | None:
    """Return an installed Python package directory, if importable."""
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.origin is None:
        return None
    package_path = Path(spec.origin).parent
    return package_path if package_path.is_dir() else None


def detect_platform(input_path: Path) -> str:
    """Auto-detect project platform (rust, javascript, or python).

    Args:
        input_path: Path to file or directory.

    Returns:
        "rust", "javascript", or "python"

    Raises:
        ValueError: If platform cannot be determined.
    """
    if input_path.is_dir():
        if (input_path / "Cargo.toml").exists():
            return "rust"
        if (input_path / "package.json").exists():
            return "javascript"
        if (input_path / "pyproject.toml").exists():
            return "python"
        # Check for main.py without pyproject.toml (simple Python project)
        if (input_path / "main.py").exists():
            return "python"
        raise ValueError(
            f"Cannot detect platform for '{input_path}'. "
            "Expected Cargo.toml (Rust), package.json (JavaScript), or pyproject.toml/main.py (Python)."
        )

    if input_path.is_file():
        ext = input_path.suffix.lower()
        if ext == ".rs":
            return "rust"
        if ext in (".js", ".ts"):
            return "javascript"
        if ext == ".py":
            return "python"
        raise ValueError(
            f"Unsupported file type: {ext}. Expected .rs, .js, .ts, or .py"
        )

    raise ValueError(f"Input '{input_path}' does not exist")


def ensure_npm_dependencies(package_dir: Path) -> None:
    """Run npm install if node_modules doesn't exist.

    Prompts the user for confirmation before running npm install. A package
    that declares no dependencies (the SDK is resolved from the checkout or
    from `@pie-project/inferlet` in node_modules) needs no install.
    """
    node_modules = package_dir / "node_modules"
    if node_modules.exists():
        return
    try:
        pkg = json.loads((package_dir / "package.json").read_text())
    except (OSError, ValueError):
        pkg = {}
    if not pkg.get("dependencies"):
        # Dev-only deps (typescript for editor support) are not needed to
        # build: esbuild bundles TS itself, and the SDK resolves from the
        # checkout or from an installed `@pie-project/inferlet`.
        return
    if not sys.stdin.isatty():
        console.print(
            "[yellow]📦 npm dependencies not installed and no terminal to ask on; "
            "building without them (run 'npm install' if the bundle needs them)[/yellow]"
        )
        return

    console.print(f"[yellow]📦 npm dependencies not found in {package_dir}[/yellow]")

    if not typer.confirm("   Run 'npm install'?", default=True):
        raise RuntimeError(
            f"npm install cancelled. Please run 'npm install' manually in {package_dir}"
        )

    with console.status("[bold green]Installing npm dependencies...[/bold green]"):
        result = subprocess.run(
            ["npm", "install", "--ignore-scripts"],
            cwd=package_dir,
            capture_output=True,
            text=True,
        )

    if result.returncode != 0:
        raise RuntimeError(f"npm install failed in {package_dir}:\n{result.stderr}")


def detect_py_input_type(input_path: Path) -> tuple[str, Path]:
    """Detect whether Python input is a single file or package directory.

    Returns:
        Tuple of (type, entry_point) where type is "file" or "package".
    """
    if input_path.is_file():
        ext = input_path.suffix.lower()
        if ext == ".py":
            return ("file", input_path)
        raise ValueError(f"Unsupported file type: {ext}. Expected .py")

    if input_path.is_dir():
        # Look for pyproject.toml or main.py
        pyproject = input_path / "pyproject.toml"
        main_py = input_path / "main.py"

        if pyproject.exists():
            # Package with pyproject.toml - look for main.py or app.py
            if main_py.exists():
                return ("package", main_py)
            app_py = input_path / "app.py"
            if app_py.exists():
                return ("package", app_py)
            raise ValueError(
                f"Directory '{input_path}' has pyproject.toml but no main.py or app.py entry point"
            )
        elif main_py.exists():
            return ("package", main_py)
        else:
            raise ValueError(
                f"Directory '{input_path}' does not contain pyproject.toml or main.py"
            )

    raise ValueError(f"Input '{input_path}' does not exist")


def generate_py_wrapper(
    user_module: Path, output_path: Path, package_name: str
) -> None:
    """Generate the WIT wrapper for Python that exports the run interface.

    The `pie:inferlet` world is a component-model-async world (`run` and the
    channel/session reads are `async func`), and componentize-py >= 0.20
    drives Python's `asyncio` from the component's own task: the wrapper's
    `run` is a coroutine and the user's `main` is simply awaited on it. No
    hand-rolled poll loop.

    Args:
        user_module: Path to the user's main Python file.
        output_path: Path to write the wrapper file.
        package_name: The package name from Pie.toml (used for the interface path).
    """
    module_name = user_module.stem

    wrapper_content = f"""# Auto-generated by bakery build --python
# This wrapper provides the WIT interface for the inferlet
# Package: {package_name}

import inspect
import json
import traceback

from wit_world import exports

# `componentize_py_types.Err` is the exception class that componentize-py
# uses to encode the Err arm of a `result<T, E>` return. Any uncaught Python
# exception from the user's main() is re-raised as this so the host receives
# a clean WIT Err (with the traceback in the message) instead of a wasm trap.
from componentize_py_types import Err as _WitErr

# Import inferlet at top level so componentize-py bundles it
import inferlet as _inferlet

# Import user module at top level so componentize-py bundles it
import {module_name} as _user_module


class Run(exports.Run):
    async def run(self, input: str) -> str:
        # Parse JSON input into a dict for the user's main()
        try:
            input_data = json.loads(input) if input else {{}}
        except json.JSONDecodeError:
            input_data = {{"input": input}}

        try:
            if hasattr(_user_module, "main"):
                result = _user_module.main(input_data)
                # Support both sync and async main()
                if inspect.isawaitable(result):
                    result = await result
                # Pass through strings; JSON-stringify everything else
                # (dict, list, primitives, dataclasses). Objects with a
                # `model_dump_json` method are handed off to it.
                if result is None:
                    return _inferlet.get_return_value() or ""
                if isinstance(result, str):
                    return result
                if hasattr(result, "model_dump_json") and callable(result.model_dump_json):
                    return result.model_dump_json()
                return json.dumps(result, default=str)
            return _inferlet.get_return_value() or ""
        except _WitErr:
            raise  # already shaped for componentize-py
        except BaseException as e:
            raise _WitErr(f"{{type(e).__name__}}: {{e}}\\n{{traceback.format_exc()}}")
"""

    output_path.write_text(wrapper_content)


def copy_dir_recursive(src: Path, dst: Path) -> None:
    """Recursively copy a directory."""
    dst.mkdir(parents=True, exist_ok=True)

    for entry in src.iterdir():
        src_path = entry
        dst_path = dst / entry.name

        if src_path.is_dir():
            copy_dir_recursive(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)


def run_componentize_py(
    wrapper_py: Path,
    output_wasm: Path,
    wit_path: Path,
    package_name: str,
    debug: bool,
) -> None:
    """Run componentize-py to compile Python code to WASM.

    Args:
        wrapper_py: Path to the Python wrapper file.
        output_wasm: Path to the output WASM file.
        wit_path: Path to the base WIT directory (sdk/interfaces).
        package_name: The package name from Pie.toml.
        debug: Enable debug mode.
    """
    # Get the working directory (where wrapper_py is located)
    work_dir = wrapper_py.parent.resolve()

    # Get the module name from the wrapper file (without .py extension)
    module_name = wrapper_py.stem

    # Componentize against the stock `pie:inferlet` world (WIT-refactor Phase 2):
    # the world now declares `export run`, so the per-package synthesized `exec`
    # world is gone. Point componentize-py straight at the vendored WIT dir.
    # Canonicalize paths for componentize-py
    output_wasm_abs = (
        output_wasm if output_wasm.is_absolute() else Path.cwd() / output_wasm
    )

    cmd = [
        resolve_command("componentize-py"),
        "-d",
        str(wit_path),
        "-w",
        "inferlet",
        "componentize",
        "-p",
        str(work_dir),
        "-o",
        str(output_wasm_abs),
        module_name,
    ]

    result = subprocess.run(
        cmd,
        cwd=work_dir,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"componentize-py failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


def get_inferlet_wit_path() -> Path:
    """Get the path to the inferlet WIT directory (sdk/rust/inferlet/wit).

    Delegates to ``path.get_wit_path`` so the resolution order (env
    override → cwd walk → bakery install location walk) stays consistent
    across modules.
    """
    return path_utils.get_wit_path()


def handle_python_build(input_path: Path, output: Path, debug: bool = False) -> None:
    """Build a Python inferlet to WASM.

    Build process:
    1. Check prerequisites (componentize-py)
    2. Read package name from Pie.toml
    3. Find inferlet and WIT paths
    4. Detect input type (file or package)
    5. Create temp directory for intermediate files
    6. Generate WIT wrapper with dynamic interface
    7. Copy user files to temp directory
    8. Copy inferlet library to temp directory
    9. Run componentize-py to compile to WASM
    """
    # Check prerequisites
    if not command_exists("componentize-py"):
        raise RuntimeError(
            "componentize-py (>= 0.25, component-model async) is required but not found.\n"
            "Install with: uv tool install componentize-py"
        )

    # Read package name from Pie.toml
    project_dir = input_path if input_path.is_dir() else input_path.parent
    package_name = read_package_name(project_dir)

    # A stock `componentize-py` build carries its own CPython, so it links
    # against the runtime's plain linker and needs no shared py-runtime
    # tree. (The factored build that imports `componentize-py-runtime` from
    # `$PIE_HOME/py-runtime` is opt-in: set BAKERY_PY_RUNTIME=1.)
    if os.environ.get("BAKERY_PY_RUNTIME") and not py_runtime.is_installed():
        console.print(
            "[dim]Python WASM runtime not installed; fetching for first use…[/dim]"
        )
        try:
            py_runtime.ensure_installed()
        except Exception as exc:
            console.print(
                f"[yellow]⚠[/yellow]  Could not fetch Python WASM runtime ({exc}). "
                "The build will continue but `pie run` will need it before "
                "the inferlet can execute."
            )

    # Resolve paths
    with console.status("[bold green]Resolving paths...[/bold green]"):
        try:
            inferlet_path = path_utils.get_inferlet_py_path()
            inferlet_src = inferlet_path / "src" / "inferlet"
        except FileNotFoundError:
            inferlet_src = get_installed_python_package_path("inferlet")
            if inferlet_src is None:
                raise FileNotFoundError(
                    "Could not find the inferlet Python package. "
                    "Install inferlet>=0.3.0 or set PIE_SDK."
                )
        wit_path = get_inferlet_wit_path()

    console.print("[bold]🏗️  Building Python inferlet...[/bold]")
    console.print(f"   Input: [blue]{input_path}[/blue]")
    console.print(f"   Output: [blue]{output}[/blue]")
    console.print(f"   Package: [dim]{package_name}[/dim]")

    # Detect input type and entry point
    input_type, entry_point = detect_py_input_type(input_path)
    console.print(
        f"   Type: [dim]{'Single file' if input_type == 'file' else 'Package'}[/dim]"
    )

    # Create temp directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        wrapper_py = temp_path / "app.py"

        with console.status(
            "[bold green]Building Python inferlet...[/bold green]"
        ) as status:
            # Step 1: Generate wrapper
            status.update("[bold green]🔧 Generating WIT wrapper...[/bold green]")
            generate_py_wrapper(entry_point, wrapper_py, package_name)

            # Step 2: Copy user files to temp directory
            status.update("[bold green]📦 Copying user files...[/bold green]")
            if input_type == "file":
                # Single file - just copy it
                dest = temp_path / entry_point.name
                shutil.copy2(entry_point, dest)
            else:
                # Package - copy all Python files from the input directory
                input_dir = input_path.resolve()
                for py_file in input_dir.glob("*.py"):
                    dest = temp_path / py_file.name
                    shutil.copy2(py_file, dest)

            # Step 3: Copy inferlet library to temp directory so it gets bundled
            status.update("[bold green]📦 Bundling inferlet library...[/bold green]")
            if inferlet_src.exists():
                inferlet_dest = temp_path / "inferlet"
                copy_dir_recursive(inferlet_src, inferlet_dest)

            # Step 4: Run componentize-py
            status.update(
                "[bold green]🔧 Compiling to WebAssembly component with componentize-py...[/bold green]"
            )
            run_componentize_py(
                wrapper_py, output, wit_path, package_name, debug
            )

    # Success
    wasm_size = output.stat().st_size if output.exists() else 0
    console.print(
        Panel(
            f"Output: [bold]{output}[/bold] ({wasm_size / 1024 / 1024:.1f} MB)",
            title="[green]✅ Build successful![/green]",
            border_style="green",
        )
    )


def detect_js_input_type(input_path: Path) -> tuple[str, Path]:
    """Detect whether JS input is a single file or package directory.

    Returns:
        Tuple of (type, entry_point) where type is "file" or "package".
    """
    if input_path.is_file():
        ext = input_path.suffix.lower()
        if ext in (".js", ".ts"):
            return ("file", input_path)
        raise ValueError(f"Unsupported file type: {ext}. Expected .js or .ts")

    if input_path.is_dir():
        package_json = input_path / "package.json"
        if not package_json.exists():
            raise ValueError(f"Directory '{input_path}' does not contain package.json")

        # Read package.json to find entry point
        package_data = json.loads(package_json.read_text())
        main = package_data.get("main", "index.js")

        entry = input_path / main
        if not entry.exists():
            raise ValueError(
                f"Entry point '{entry}' specified in package.json does not exist"
            )

        return ("package", entry)

    raise ValueError(f"Input '{input_path}' does not exist")


def get_inferlet_js_entry(project_dir: Path) -> Path:
    """Resolve the JS SDK entry from the user's install or a local SDK checkout."""
    package_entry = (
        project_dir / "node_modules" / "@pie-project" / "inferlet" / "dist" / "index.js"
    )
    if package_entry.is_file():
        return package_entry

    try:
        local_sdk = path_utils.get_inferlet_js_path()
    except FileNotFoundError:
        local_sdk = None

    if local_sdk is not None:
        source_entry = local_sdk / "src" / "index.ts"
        if source_entry.is_file():
            return source_entry

    raise FileNotFoundError(
        f"Could not find {INFERLET_JS_PACKAGE}. Run 'npm install' in "
        f"{project_dir} or set PIE_SDK to a Pie SDK checkout."
    )


def run_esbuild_user_code(entry_point: Path, output_file: Path) -> None:
    """Bundle user code with esbuild (keeps inferlet imports external)."""
    cmd = [
        "npx",
        "-y",
        "esbuild",
        str(entry_point),
        "--bundle",
        "--format=esm",
        "--platform=neutral",
        "--target=es2022",
        "--main-fields=module,main",
        f"--outfile={output_file}",
        # Keep inferlet imports external
        "--external:inferlet",
        "--external:wasi:*",
        "--external:inferlet:*",
        f"--external:{INFERLET_JS_PACKAGE}",
        f"--external:{INFERLET_JS_PACKAGE}/*",
    ]

    # Mark Node.js built-ins as external
    nodejs_builtins = [
        "fs",
        "path",
        "events",
        "os",
        "crypto",
        "child_process",
        "net",
        "http",
        "https",
        "stream",
        "util",
        "url",
        "buffer",
        "process",
        "domain",
    ]
    for builtin in nodejs_builtins:
        cmd.append(f"--external:{builtin}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"esbuild user code bundling failed:\n{result.stderr}")


def run_esbuild(
    entry_point: Path,
    output_file: Path,
    inferlet_entry: Path,
    debug: bool,
) -> None:
    """Bundle with esbuild, resolving inferlet imports."""

    if not inferlet_entry.is_file():
        raise FileNotFoundError(
            f"inferlet entry not found at '{inferlet_entry}', "
            "run npm install or set PIE_SDK"
        )

    inferlet_entry = inferlet_entry.resolve()

    cmd = [
        "npx",
        "-y",
        "esbuild",
        str(entry_point),
        "--bundle",
        "--format=esm",
        "--platform=neutral",
        "--target=es2022",
        "--main-fields=module,main",
        f"--outfile={output_file}",
        f"--alias:inferlet={inferlet_entry}",
        f"--alias:{INFERLET_JS_PACKAGE}={inferlet_entry}",
    ]

    if debug:
        cmd.append("--sourcemap=inline")
    else:
        cmd.append("--minify")

    # External WIT imports
    cmd.extend(
        [
            "--external:wasi:*",
            "--external:inferlet:*",
            f"--external:{INFERLET_JS_PACKAGE}/*",
            "--external:pie:*",
        ]
    )

    # External Node.js built-ins
    nodejs_builtins = [
        "fs",
        "path",
        "events",
        "os",
        "crypto",
        "child_process",
        "net",
        "http",
        "https",
        "stream",
        "util",
        "url",
        "buffer",
        "process",
        "domain",
    ]
    for builtin in nodejs_builtins:
        cmd.append(f"--external:{builtin}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"esbuild failed:\n{result.stderr}")


def derive_js_wit(src: Path, dst: Path) -> None:
    """Derive the JavaScript-facing WIT world from the runtime's canonical one.

    The same transformation as `sdk/inferlet/javascript/scripts/derive-js-wit.mjs`
    (the JS SDK generates its type bindings from it): componentize-js cannot
    lower an `async func` import and wasmtime types async imports distinctly,
    so every `async func` import is dropped (the host's blocking twins stay),
    `run` is exported as a plain `func`, and the wasi 0.3 imports are removed.
    """
    import re

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for entry in src.iterdir():
        if not (entry.is_file() and entry.suffix == ".wit"):
            continue
        text = entry.read_text()
        if entry.name == "world.wit":
            text = "\n".join(
                line for line in text.split("\n")
                if not re.match(r"^\s*import wasi:(http|clocks|filesystem)/", line)
            )
        elif entry.name == "run.wit":
            text = text.replace("async func", "func")
        else:
            out: list[str] = []
            for line in text.split("\n"):
                if re.match(r"^\s*[a-z0-9-]+:\s*async func", line):
                    while out and re.match(r"^\s*///", out[-1]):
                        out.pop()
                    continue
                out.append(line)
            text = "\n".join(out)
        (dst / entry.name).write_text(text)
    shutil.copytree(src / "deps", dst / "deps")


def find_componentize_js(project_dir: Path) -> list[str]:
    """The componentize-js launcher: a local install next to the project or the
    JS SDK when one exists (pinned, offline), else `npx -y`."""
    candidates = [project_dir / "node_modules" / ".bin" / "componentize-js"]
    try:
        candidates.append(path_utils.get_inferlet_js_path() / "node_modules" / ".bin" / "componentize-js")
    except FileNotFoundError:
        pass
    for c in candidates:
        if c.is_file():
            return [str(c)]
    return ["npx", "-y", "@bytecodealliance/componentize-js"]


def run_componentize_js(
    input_js: Path,
    output_wasm: Path,
    wit_path: Path,
    debug: bool,
    project_dir: Path | None = None,
) -> None:
    """Compile bundled JS to WASM component against the derived JS world."""

    js_wit = input_js.parent / "wit-js"
    derive_js_wit(wit_path, js_wit)

    cmd = [
        *find_componentize_js(project_dir or input_js.parent),
        str(input_js),
        "-o",
        str(output_wasm),
        "--wit",
        str(js_wit),
        "--world-name",
        "inferlet",
    ]

    if debug:
        cmd.append("--use-debug-build")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"componentize-js failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


def check_for_nodejs_imports(bundled_js: Path) -> None:
    """Check for Node.js-specific imports that won't work in WASM."""
    content = bundled_js.read_text()

    nodejs_modules = [
        "fs",
        "path",
        "events",
        "os",
        "crypto",
        "child_process",
        "net",
        "http",
        "https",
        "stream",
        "util",
        "url",
        "buffer",
        "process",
        "domain",
    ]

    warnings = []

    for module in nodejs_modules:
        patterns = [
            f'require("{module}")',
            f"require('{module}')",
            f'from "{module}"',
            f"from '{module}'",
            f'import "{module}"',
            f"import '{module}'",
        ]

        for pattern in patterns:
            if pattern in content:
                warnings.append(f"  - '{module}'")
                break

    if warnings:
        console.print(
            "[yellow]⚠️  Warning: The following Node.js modules were detected and will not work in WASM:[/yellow]"
        )
        for warning in warnings:
            console.print(f"[yellow]{warning}[/yellow]")
        console.print(
            "[yellow]   Consider using pure JavaScript alternatives or Pie WIT APIs instead.\n[/yellow]"
        )


def validate_user_code(bundled_js: Path, package_dir: Path) -> None:
    """Validate user code for forbidden exports using Node.js AST analysis.

    Uses acorn parser via Node.js to properly handle ES2022+ features
    like top-level await. Checks that user code doesn't export 'run' or 'main'.
    """
    # Node.js script that parses and validates the code using acorn
    # acorn is included with npm/npx, so it's always available
    validate_script = """
const fs = require("fs");
const acorn = require("acorn");

const code = fs.readFileSync(process.argv[2], "utf8");

let ast;
try {
    ast = acorn.parse(code, {
        ecmaVersion: 2022,
        sourceType: "module",
        allowAwaitOutsideFunction: true
    });
} catch (e) {
    console.error("PARSE_ERROR:" + e.message);
    process.exit(1);
}

function getPatternNames(pattern) {
    const names = [];
    if (pattern.type === "Identifier") {
        names.push(pattern.name);
    } else if (pattern.type === "ObjectPattern") {
        for (const prop of pattern.properties) {
            if (prop.type === "Property") {
                names.push(...getPatternNames(prop.value));
            } else if (prop.type === "RestElement") {
                names.push(...getPatternNames(prop.argument));
            }
        }
    } else if (pattern.type === "ArrayPattern") {
        for (const elem of pattern.elements) {
            if (elem) names.push(...getPatternNames(elem));
        }
    } else if (pattern.type === "RestElement") {
        names.push(...getPatternNames(pattern.argument));
    } else if (pattern.type === "AssignmentPattern") {
        names.push(...getPatternNames(pattern.left));
    }
    return names;
}

for (const node of ast.body) {
    let exportedNames = [];

    if (node.type === "ExportNamedDeclaration") {
        if (node.declaration) {
            const decl = node.declaration;
            if (decl.type === "FunctionDeclaration") {
                exportedNames.push(decl.id.name);
            } else if (decl.type === "VariableDeclaration") {
                for (const d of decl.declarations) {
                    exportedNames.push(...getPatternNames(d.id));
                }
            } else if (decl.type === "ClassDeclaration") {
                exportedNames.push(decl.id.name);
            }
        }
        if (node.specifiers) {
            for (const spec of node.specifiers) {
                const exported = spec.exported || spec.local;
                if (exported) exportedNames.push(exported.name);
            }
        }
    } else if (node.type === "ExportDefaultDeclaration") {
        if (node.declaration && node.declaration.id) {
            exportedNames.push(node.declaration.id.name);
        }
    }

    for (const name of exportedNames) {
        if (name === "run") {
            console.error("FORBIDDEN:run");
            process.exit(1);
        }
    }
}
console.log("OK");
"""

    # Write the validation script to a temp file and run it
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cjs", delete=False) as f:
        f.write(validate_script)
        script_path = f.name

    try:
        env = os.environ.copy()
        node_modules = package_dir / "node_modules"
        if node_modules.exists():
            env["NODE_PATH"] = str(node_modules)

        result = subprocess.run(
            ["node", script_path, str(bundled_js)],
            capture_output=True,
            text=True,
            env=env,
        )

        stderr = result.stderr.strip()

        if result.returncode != 0:
            if "Cannot find module 'acorn'" in stderr:
                console.print(
                    "[yellow]⚠️  Skipping JavaScript export validation because acorn is not installed.[/yellow]"
                )
            elif stderr.startswith("PARSE_ERROR:"):
                raise RuntimeError(f"Failed to parse JavaScript: {stderr[12:]}")
            elif stderr.startswith("FORBIDDEN:run"):
                raise RuntimeError(
                    "User code must not export 'run' - it is auto-generated.\n\n"
                    "To fix: Remove the 'export const run = { ... }' block from your code.\n"
                    "The WIT interface is now automatically created by bakery build."
                )

            else:
                raise RuntimeError(f"Validation failed: {stderr}")
    finally:
        os.unlink(script_path)


def generate_wrapper(user_bundle_path: Path, output_path: Path) -> None:
    """Generate the WIT interface wrapper."""
    user_bundle_name = user_bundle_path.name

    # Intl polyfill for WASM environment (used by @huggingface/jinja for date formatting)
    intl_polyfill = """
// Intl polyfill for WASM environment
// Provides minimal DateTimeFormat support for @huggingface/jinja
if (typeof globalThis.Intl === 'undefined') {
  const MONTHS_LONG = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December'];
  const MONTHS_SHORT = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  globalThis.Intl = {
    DateTimeFormat: function(locale, options) {
      return {
        format: function(date) {
          if (options && options.month === 'long') {
            return MONTHS_LONG[date.getMonth()];
          } else if (options && options.month === 'short') {
            return MONTHS_SHORT[date.getMonth()];
          }
          return date.toISOString();
        }
      };
    }
  };
}
"""

    wrapper_content = f"""// Auto-generated by bakery build
// This wrapper provides the WIT interface for the inferlet
{intl_polyfill}
// Import user's main function
import {{ main as userMain }} from './{user_bundle_name}';

// WIT interface export (inferlet:core/run)
export const run = {{
  async run(input) {{
    if (typeof userMain === 'function') {{
      // Parse JSON input into an object for the user's main(), mirroring
      // the Python wrapper. If input isn't valid JSON, fall back to
      // {{ input: <raw> }}.
      let inputData = {{}};
      if (input) {{
        try {{
          inputData = JSON.parse(input);
        }} catch {{
          inputData = {{ input }};
        }}
      }}
      let result;
      try {{
        result = await userMain(inputData);
      }} catch (e) {{
        // componentize-js maps a thrown *string* to the WIT
        // `result<_, string>::Err` variant; throwing an Error object
        // would trap. Coerce so users can `throw new Error(...)` idiomatically.
        if (typeof e === 'string') throw e;
        throw String(e?.message ?? e);
      }}
      // Pass strings through; JSON-stringify everything else (objects,
      // arrays, primitives). null/undefined become empty string.
      if (result == null) return '';
      if (typeof result === 'string') return result;
      return JSON.stringify(result);
    }}
    return '';
  }},
}};
"""

    output_path.write_text(wrapper_content)


def handle_rust_build(input_path: Path, output: Path) -> None:
    """Build a Rust inferlet to WASM.

    Args:
        input_path: Path to the Rust project directory (containing Cargo.toml).
        output: Output path for the .wasm file.
    """
    # Check prerequisites
    if not command_exists("cargo"):
        raise RuntimeError(
            "cargo is required but not found. Please install Rust: https://rustup.rs"
        )

    # Ensure input is a directory with Cargo.toml
    if not input_path.is_dir():
        raise ValueError(f"Rust build requires a directory, got file: {input_path}")

    cargo_toml = input_path / "Cargo.toml"
    if not cargo_toml.exists():
        raise ValueError(f"No Cargo.toml found in {input_path}")

    console.print(f"[bold]🏗️  Building Rust inferlet...[/bold]")
    console.print(f"   Input: [blue]{input_path}[/blue]")
    console.print(f"   Output: [blue]{output}[/blue]")

    # Run cargo build
    with console.status("[bold green]Running cargo build...[/bold green]"):
        cmd = [
            "cargo",
            "build",
            "--target",
            "wasm32-wasip2",
            "--release",
        ]

        result = subprocess.run(
            cmd,
            cwd=input_path,
            capture_output=True,
            text=True,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"cargo build failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    # Find the output wasm file
    # Parse Cargo.toml to get the package name
    cargo_data = tomllib.loads(cargo_toml.read_text())
    package_name = cargo_data.get("package", {}).get("name", input_path.name)
    # Cargo replaces hyphens with underscores in output file names
    wasm_name = package_name.replace("-", "_")

    wasm_path = (
        input_path / "target" / "wasm32-wasip2" / "release" / f"{wasm_name}.wasm"
    )

    if not wasm_path.exists():
        raise RuntimeError(
            f"Expected output not found at {wasm_path}\n"
            "Build may have succeeded but output location is different."
        )

    # Copy to output location
    shutil.copy2(wasm_path, output)

    # Success
    wasm_size = output.stat().st_size if output.exists() else 0
    console.print(
        Panel(
            f"Output: [bold]{output}[/bold] ({wasm_size / 1024:.1f} KB)",
            title="[green]✅ Build successful![/green]",
            border_style="green",
        )
    )


def handle_js_build(input_path: Path, output: Path, debug: bool = False) -> None:
    """Build a JavaScript/TypeScript inferlet to WASM.

    Build process:
    1. Check prerequisites (Node.js, npx)
    2. Find inferlet-js and WIT paths
    3. Ensure npm dependencies installed
    4. Detect input type (file or package)
    5. Bundle user code with esbuild
    6. Check for Node.js imports (warnings)
    7. Validate user code (no export run/main)
    8. Generate WIT wrapper
    9. Bundle wrapper with esbuild
    10. Compile to WASM with componentize-js
    """
    # Check prerequisites
    if not command_exists("node"):
        raise RuntimeError(
            "Node.js is required but not found. Please install Node.js (v18+)."
        )

    if not command_exists("npx"):
        raise RuntimeError(
            "npx is required but not found. Please install Node.js (v18+)."
        )

    # Read package name from Pie.toml
    project_dir = input_path if input_path.is_dir() else input_path.parent
    package_name = read_package_name(project_dir)

    # Ensure project npm dependencies, then resolve the JS SDK and WIT paths.
    if input_path.is_dir():
        ensure_npm_dependencies(project_dir)

    with console.status("[bold green]Resolving paths...[/bold green]"):
        inferlet_js_entry = get_inferlet_js_entry(project_dir)
        wit_path = get_inferlet_wit_path()

    console.print("[bold]🏗️  Building JS inferlet...[/bold]")
    console.print(f"   Input: [blue]{input_path}[/blue]")
    console.print(f"   Output: [blue]{output}[/blue]")

    # Detect input type
    input_type, entry_point = detect_js_input_type(input_path)
    console.print(
        f"   Type: [dim]{'Single file' if input_type == 'file' else 'Package'}[/dim]"
    )

    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        user_bundle = temp_path / "user-bundle.js"
        wrapper_js = temp_path / "wrapper.js"
        final_bundle = temp_path / "final-bundle.js"

        with console.status(
            "[bold green]Building split stages...[/bold green]"
        ) as status:
            # Step 1: Bundle user code
            status.update("[bold green]📦 Bundling user code...[/bold green]")
            run_esbuild_user_code(entry_point, user_bundle)

            # Step 2: Check for Node.js imports
            check_for_nodejs_imports(user_bundle)

            # Step 3: Validate user code
            status.update("[bold green]🔍 Validating user code...[/bold green]")
            validate_user_code(user_bundle, project_dir)

            # Step 4: Generate wrapper
            status.update("[bold green]🔧 Generating WIT wrapper...[/bold green]")
            generate_wrapper(user_bundle, wrapper_js)

            # Step 5: Bundle wrapper
            status.update("[bold green]📦 Bundling final output...[/bold green]")
            run_esbuild(wrapper_js, final_bundle, inferlet_js_entry, debug)

            # Step 6: Compile to WASM against the stock `pie:inferlet` world
            # (WIT-refactor Phase 2 — the synthesized `exec` world is gone; the
            # world declares `export run` directly).
            status.update(
                "[bold green]🔧 Compiling to WebAssembly component...[/bold green]"
            )
            run_componentize_js(final_bundle, output, wit_path, debug, project_dir)

    # Success
    wasm_size = output.stat().st_size if output.exists() else 0
    console.print(
        Panel(
            f"Output: [bold]{output}[/bold] ({wasm_size / 1024:.1f} KB)",
            title="[green]✅ Build successful![/green]",
            border_style="green",
        )
    )


def handle_build_command(
    input_path: Path,
    output: Path,
    debug: bool = False,
) -> None:
    """Handle the `bakery build` command.

    Auto-detects project platform (Rust, JavaScript, or Python) and dispatches
    to the appropriate build handler.

    Args:
        input_path: Path to the project directory or source file.
        output: Output path for the .wasm file.
        debug: Enable debug mode (JS/Python: inline source maps).
    """
    # Auto-detect platform
    platform = detect_platform(input_path)

    if platform == "rust":
        handle_rust_build(input_path, output)
    elif platform == "python":
        handle_python_build(input_path, output, debug)
    else:
        handle_js_build(input_path, output, debug)
