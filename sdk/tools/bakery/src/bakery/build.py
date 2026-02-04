"""Build command implementation for Bakery.

This module implements the `bakery build` subcommand for building
JavaScript/TypeScript, Python, and Rust inferlets into WebAssembly components.
"""

import json
import os
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Optional

from rich.panel import Panel
from .console import console
from . import path as path_utils
import typer


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


def to_wit_ident(name: str) -> str:
    """Convert a package name to a valid WIT identifier.

    WIT identifiers use kebab-case (hyphens).
    """
    # Replace underscores with hyphens
    return name.replace("_", "-")


def to_python_ident(name: str) -> str:
    """Convert a package name to a valid Python identifier.

    Python identifiers use snake_case (underscores) and can't have hyphens.
    """
    # Replace hyphens with underscores
    return name.replace("-", "_")


def command_exists(cmd: str) -> bool:
    """Check if a command is available in PATH."""
    return shutil.which(cmd) is not None


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

    Prompts the user for confirmation before running npm install.
    """
    node_modules = package_dir / "node_modules"
    if node_modules.exists():
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

    Args:
        user_module: Path to the user's main Python file.
        output_path: Path to write the wrapper file.
        package_name: The package name from Pie.toml (used for the interface path).
    """
    # Get the module name without extension
    module_name = user_module.stem

    # The wrapper imports user code and provides the WIT run export
    # All modules are bundled in the same directory by componentize-py
    # When we export `run` interface from a world, componentize-py puts it at exports.run
    wrapper_content = f"""# Auto-generated by bakery build --python
# This wrapper provides the WIT interface for the inferlet
# Package: {package_name}

# Import WIT bindings for the run export
from wit_world import exports

# Import inferlet at top level so componentize-py bundles it
import inferlet as _inferlet

# Import user module at top level so componentize-py bundles it
import {module_name} as _user_module

class Run(exports.Run):
    def run(self) -> None:
        # Call the user's main function if it exists
        if hasattr(_user_module, 'main'):
            _user_module.main()
        else:
            # Module execution happens at import time for scripts without main()
            pass
        # Signal completion if user code didn't call set_return()
        if not _inferlet.was_return_set():
            _inferlet.set_return("")
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


def generate_dynamic_wit(
    base_wit_path: Path,
    temp_wit_dir: Path,
    package_name: str,
) -> None:
    """Generate a temporary WIT directory with the dynamic export interface.

    Copies the deps/ directory from sdk/rust/inferlet/wit/ and adds
    a world.wit with the dynamic run interface. The imports are read
    from the source world.wit to avoid hardcoding.

    Args:
        base_wit_path: Path to sdk/rust/inferlet/wit/.
        temp_wit_dir: Path to the temporary WIT directory to create.
        package_name: The package name for the export interface.
    """
    # Convert package name to WIT identifier (kebab-case)
    wit_package_name = to_wit_ident(package_name)

    # Create temp WIT directory
    temp_wit_dir.mkdir(parents=True, exist_ok=True)

    # Copy the deps/ directory as-is
    deps_src = base_wit_path / "deps"
    if deps_src.exists():
        copy_dir_recursive(deps_src, temp_wit_dir / "deps")

    # Read the imports from the source world.wit
    source_world_wit = base_wit_path / "world.wit"
    include_lines = []
    if source_world_wit.exists():
        for line in source_world_wit.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("include "):
                include_lines.append(f"    {stripped}")

    includes_block = "\n".join(include_lines) if include_lines else ""

    # Add the world.wit with the dynamic run interface
    world_wit_content = f"""package pie:{wit_package_name};

// Dynamic run interface for this inferlet
interface run {{
    run: func() -> result<_, string>;
}}

// Exec world with imports and dynamic export
world exec {{
{includes_block}

    export run;
}}
"""

    (temp_wit_dir / "world.wit").write_text(world_wit_content)


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

    # Create a temporary WIT directory with the dynamic export
    temp_wit_dir = work_dir / "wit"
    generate_dynamic_wit(wit_path, temp_wit_dir, package_name)

    # Canonicalize paths for componentize-py
    output_wasm_abs = (
        output_wasm if output_wasm.is_absolute() else Path.cwd() / output_wasm
    )

    cmd = [
        "componentize-py",
        "-d",
        str(temp_wit_dir),
        "-w",
        "exec",
        "componentize",
        module_name,
        "-o",
        str(output_wasm_abs),
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
    """Get the path to the inferlet WIT directory.

    This is sdk/rust/inferlet/wit/ which contains all WIT definitions
    with properly structured deps/.

    Raises:
        FileNotFoundError: If the WIT directory cannot be found.
    """
    if pie_sdk := os.environ.get("PIE_SDK"):
        path = Path(pie_sdk) / "rust" / "inferlet" / "wit"
        if path.exists():
            return path

    current_dir = Path.cwd()
    for parent in [current_dir] + list(current_dir.parents):
        wit_path = parent / "sdk" / "rust" / "inferlet" / "wit"
        if wit_path.exists():
            return wit_path

    raise FileNotFoundError(
        "Could not find sdk/rust/inferlet/wit directory. Please set PIE_SDK environment variable."
    )


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
            "componentize-py is required but not found.\n"
            "Install with: uv tool install componentize-py"
        )

    # Read package name from Pie.toml
    project_dir = input_path if input_path.is_dir() else input_path.parent
    package_name = read_package_name(project_dir)

    # Resolve paths
    with console.status("[bold green]Resolving paths...[/bold green]"):
        inferlet_path = path_utils.get_inferlet_py_path()
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
            inferlet_src = inferlet_path / "src" / "inferlet"
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


def run_esbuild_user_code(entry_point: Path, output_file: Path) -> None:
    """Bundle user code with esbuild (keeps inferlet imports external)."""
    cmd = [
        "npx",
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
    inferlet_js_path: Path,
    debug: bool,
) -> None:
    """Bundle with esbuild, resolving inferlet imports."""

    # Validate inferlet alias target
    inferlet_entry = inferlet_js_path / "src" / "index.ts"
    if not inferlet_entry.is_file():
        raise FileNotFoundError(
            f"inferlet entry not found at '{inferlet_entry}', "
            "ensure inferlet-js/src/index.ts exists"
        )

    inferlet_entry = inferlet_entry.resolve()

    cmd = [
        "npx",
        "esbuild",
        str(entry_point),
        "--bundle",
        "--format=esm",
        "--platform=neutral",
        "--target=es2022",
        "--main-fields=module,main",
        f"--outfile={output_file}",
        f"--alias:inferlet={inferlet_entry}",
    ]

    if debug:
        cmd.append("--sourcemap=inline")
    else:
        cmd.append("--minify")

    # External WIT imports
    cmd.extend(["--external:wasi:*", "--external:inferlet:*"])

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


def run_componentize_js(
    input_js: Path,
    output_wasm: Path,
    wit_path: Path,
    debug: bool,
) -> None:
    """Compile bundled JS to WASM component."""

    cmd = [
        "npx",
        "@bytecodealliance/componentize-js",
        str(input_js),
        "-o",
        str(output_wasm),
        "--wit",
        str(wit_path),
        "--world-name",
        "exec",
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


def validate_user_code(bundled_js: Path) -> None:
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
        if (name === "main") {
            console.error("FORBIDDEN:main");
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
        # Run with NODE_PATH set to inferlet-js node_modules where acorn is available
        inferlet_js_path = path_utils.get_inferlet_js_path()
        node_modules = inferlet_js_path / "node_modules"
        env = os.environ.copy()
        env["NODE_PATH"] = str(node_modules)

        result = subprocess.run(
            ["node", script_path, str(bundled_js)],
            capture_output=True,
            text=True,
            env=env,
        )

        stderr = result.stderr.strip()

        if result.returncode != 0:
            if stderr.startswith("PARSE_ERROR:"):
                raise RuntimeError(f"Failed to parse JavaScript: {stderr[12:]}")
            elif stderr.startswith("FORBIDDEN:run"):
                raise RuntimeError(
                    "User code must not export 'run' - it is auto-generated.\n\n"
                    "To fix: Remove the 'export const run = { ... }' block from your code.\n"
                    "The WIT interface is now automatically created by bakery build."
                )
            elif stderr.startswith("FORBIDDEN:main"):
                raise RuntimeError(
                    "User code must not export 'main' - use top-level code instead.\n\n"
                    "To fix: Move your code from inside main() to the top level."
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
// WIT interface export (inferlet:core/run)
export const run = {{
  run: async () => {{
    try {{
      await import('./{user_bundle_name}');
      return {{ tag: 'ok' }};
    }} catch (e) {{
      const msg = e instanceof Error ? `${{e.message}}\\n${{e.stack}}` : String(e);
      console.log(`\\nERROR: ${{msg}}\\n`);
      return {{ tag: 'err', val: msg }};
    }}
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

    # Resolve paths
    with console.status("[bold green]Resolving paths...[/bold green]"):
        inferlet_js_path = path_utils.get_inferlet_js_path()
        wit_path = path_utils.get_wit_path()

    # Ensure npm dependencies
    ensure_npm_dependencies(inferlet_js_path)

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
            validate_user_code(user_bundle)

            # Step 4: Generate wrapper
            status.update("[bold green]🔧 Generating WIT wrapper...[/bold green]")
            generate_wrapper(user_bundle, wrapper_js)

            # Step 5: Bundle wrapper
            status.update("[bold green]📦 Bundling final output...[/bold green]")
            run_esbuild(wrapper_js, final_bundle, inferlet_js_path, debug)

            # Step 6: Compile to WASM
            status.update(
                "[bold green]🔧 Compiling to WebAssembly component...[/bold green]"
            )
            run_componentize_js(final_bundle, output, wit_path, debug)

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
