"""Create command implementation for Bakery.

This module implements the `bakery create` subcommand for creating
new JavaScript/TypeScript and Rust inferlet projects.
"""

import os
from importlib import resources
from pathlib import Path
from string import Template
from typing import Optional

from rich.panel import Panel
from .console import console
from . import path as path_utils
import typer


def get_template(name: str) -> Template:
    """Load a template file from the templates directory.

    Args:
        name: Template path relative to templates/, e.g. "typescript/index.ts.template"

    Returns:
        A string.Template object ready for substitution.
    """
    template_content = resources.files("bakery.templates").joinpath(name).read_text()
    return Template(template_content)




def validate_project_name(name: str) -> None:
    """Validate that a project name is safe.

    Raises:
        ValueError: If the project name is invalid.
    """
    # Check for path traversal
    if ".." in name:
        raise ValueError(
            f"Project name cannot contain '..' (path traversal). Got: '{name}'"
        )

    # Check for empty or whitespace-only
    if not name.strip():
        raise ValueError("Project name cannot be empty or whitespace-only")

    # Extract final component
    project_name = Path(name).name

    # Check for hidden files
    if project_name.startswith("."):
        raise ValueError(
            f"Project name cannot start with '.' (hidden file). Got: '{project_name}'"
        )


# =============================================================================
# TypeScript Generators
# =============================================================================


def generate_ts_index(project_dir: Path, name: str) -> None:
    """Generate the index.ts file for TypeScript inferlet."""
    template = get_template("typescript/index.ts.template")
    content = template.substitute(name=name)
    (project_dir / "index.ts").write_text(content)


def generate_ts_package_json(project_dir: Path, name: str) -> None:
    """Generate the package.json file for TypeScript inferlet."""
    template = get_template("typescript/package.json.template")
    content = template.substitute(name=name)
    (project_dir / "package.json").write_text(content)


def generate_tsconfig(project_dir: Path, inferlet_path: Path) -> None:
    """Generate the tsconfig.json file."""
    path_str = str(inferlet_path).replace("\\", "/")
    template = get_template("typescript/tsconfig.json.template")
    content = template.substitute(inferlet_path=path_str)
    (project_dir / "tsconfig.json").write_text(content)


# =============================================================================
# Rust Generators
# =============================================================================


def generate_rust_lib(project_dir: Path, _name: str) -> None:
    """Generate the src/lib.rs file for Rust inferlet."""
    template = get_template("rust/lib.rs.template")
    content = template.substitute()
    src_dir = project_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "lib.rs").write_text(content)


def generate_rust_cargo_toml(project_dir: Path, name: str) -> None:
    """Generate the Cargo.toml file for Rust inferlet."""
    # Use crates.io path if not in dev mode
    if pie_sdk := os.environ.get("PIE_SDK"):
        inferlet_dep = f'{{ path = "{pie_sdk}/rust/inferlet" }}'
        macros_dep = f'{{ path = "{pie_sdk}/rust/inferlet-macros" }}'
    else:
        # Try to find relative paths
        current_dir = Path.cwd()
        for parent in [current_dir] + list(current_dir.parents):
            if (parent / "sdk" / "rust" / "inferlet").exists():
                rel_path = (
                    parent.relative_to(project_dir.parent)
                    if project_dir.parent != parent
                    else Path("..")
                )
                inferlet_dep = f'{{ path = "{rel_path}/sdk/rust/inferlet" }}'
                macros_dep = f'{{ path = "{rel_path}/sdk/rust/inferlet-macros" }}'
                break
        else:
            # Fallback to relative paths assuming standard project layout
            inferlet_dep = '{ path = "../sdk/rust/inferlet" }'
            macros_dep = '{ path = "../sdk/rust/inferlet-macros" }'

    template = get_template("rust/Cargo.toml.template")
    content = template.substitute(
        name=name, inferlet_dep=inferlet_dep, macros_dep=macros_dep
    )
    (project_dir / "Cargo.toml").write_text(content)


def generate_pie_toml(project_dir: Path, name: str, language: str) -> None:
    """Generate the Pie.toml manifest file."""
    template = get_template("Pie.toml.template")
    content = template.substitute(name=name)
    (project_dir / "Pie.toml").write_text(content)


# =============================================================================
# Main Handler
# =============================================================================


def handle_create_command(
    name: str,
    rust: bool = False,
    output: Optional[Path] = None,
) -> None:
    """Handle the `bakery create` command.

    Creates a new inferlet project with:
    - TypeScript: index.ts, package.json, tsconfig.json, Pie.toml
    - Rust: src/lib.rs, Cargo.toml, Pie.toml

    Args:
        name: Name of the inferlet project.
        rust: Create a Rust project instead of TypeScript.
        output: Output directory (default: current directory).
    """
    # Validate project name
    validate_project_name(name)

    # Determine project directory
    if output:
        project_dir = output / name
    else:
        project_dir = Path(name)

    # Extract just the project name for display
    project_name = Path(name).name

    # Check if directory already exists
    if project_dir.exists():
        raise FileExistsError(f"Directory '{project_dir}' already exists")

    # Create project directory
    project_dir.mkdir(parents=True)

    language = "rust" if rust else "typescript"

    if rust:
        # Generate Rust project
        generate_rust_lib(project_dir, project_name)
        generate_rust_cargo_toml(project_dir, project_name)
        generate_pie_toml(project_dir, project_name, language)

        console.print(
            Panel(
                f"Created Rust inferlet project: [bold]{project_name}[/bold]\n\n"
                f"[blue]{project_dir}/src/lib.rs[/blue]\n"
                f"[blue]{project_dir}/Cargo.toml[/blue]\n"
                f"[blue]{project_dir}/Pie.toml[/blue]",
                title="[green]✅ Project Created[/green]",
                border_style="green",
            )
        )

        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"   cd {project_dir}")
        console.print("   cargo build --release --target wasm32-wasip2")
    else:
        # Generate TypeScript project
        try:
            inferlet_js_path = path_utils.get_inferlet_js_path()
            project_dir_abs = project_dir.resolve()
            try:
                relative_path = inferlet_js_path.resolve().relative_to(
                    project_dir_abs.parent
                )
            except ValueError:
                relative_path = inferlet_js_path.resolve()
        except FileNotFoundError:
            relative_path = Path("../inferlet-js")

        generate_ts_index(project_dir, project_name)
        generate_ts_package_json(project_dir, project_name)
        generate_tsconfig(project_dir, relative_path)
        generate_pie_toml(project_dir, project_name, language)

        console.print(
            Panel(
                f"Created TypeScript inferlet project: [bold]{project_name}[/bold]\n\n"
                f"[blue]{project_dir}/index.ts[/blue]\n"
                f"[blue]{project_dir}/package.json[/blue]\n"
                f"[blue]{project_dir}/tsconfig.json[/blue]\n"
                f"[blue]{project_dir}/Pie.toml[/blue]",
                title="[green]✅ Project Created[/green]",
                border_style="green",
            )
        )

        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"   cd {project_dir}")
        console.print(f"   bakery build . -o {project_name}.wasm")
