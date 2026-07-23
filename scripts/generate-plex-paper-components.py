#!/usr/bin/env python3

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / "tests" / "policies"


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: generate-plex-paper-components.py slug=RustType [...]")
    for argument in sys.argv[1:]:
        slug, separator, policy_type = argument.partition("=")
        if (
            not separator
            or not re.fullmatch(r"[a-z][a-z0-9-]*", slug)
            or not re.fullmatch(r"[A-Z][A-Za-z0-9]*", policy_type)
        ):
            raise SystemExit(f"invalid component specification: {argument!r}")
        root = POLICIES / f"paper-{slug}"
        (root / "src").mkdir(parents=True, exist_ok=True)
        (root / "Cargo.toml").write_text(
            f"""[package]
name = "plex-paper-{slug}"
version = "0.1.0"
edition = "2024"
publish = false

[lib]
crate-type = ["cdylib"]
path = "src/lib.rs"

[dependencies]
paper-common = {{ package = "plex-paper-common", path = "../paper-common" }}
plex = {{ package = "pie-plex-sdk", path = "../../../sdk/rust/plex" }}
"""
        )
        (root / "src" / "lib.rs").write_text(
            f"plex::export_policy!(paper_common::{policy_type});\n"
        )


if __name__ == "__main__":
    main()
