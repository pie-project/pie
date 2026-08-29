"""The cfg-feature audit: a `#[cfg(feature = "x")]` whose `x` does not exist.

A feature name in a `cfg` is not checked by the type system. If the
package's manifest does not declare the feature, the attribute is not an
error and not a build failure -- it is simply a condition that is always
false. The code under it compiles nowhere, runs nowhere, and looks
exactly like code that is merely off in this configuration.

## What this tree paid for it

The root `pie` package forwards a driver flavor to the embedded worker:

    driver-cuda  = ["worker/driver-cuda"]
    engine-metal = ["worker/engine-metal"]
    driver-dummy = ["worker/driver-dummy"]

The middle line was deleted in August 2026 when the C++ Metal driver was
retired -- correctly, because at that moment there was no Metal driver to
forward to. The Rust `engine-metal` then came back into `worker` and
`runtime`, and the root's forward was never restored. What survived was
the COMMENT that documented it, which slid down onto `driver-dummy` and
went on describing a feature that no longer existed:

    # macOS/Apple Silicon. Without this the `pie` binary has no way to
    # reach the Metal driver at all.

That comment was exactly right, and it sat directly above the wrong
line. Meanwhile `src/ops/config/template.rs` held five
`#[cfg(feature = "engine-metal")]` blocks, so `pie init` on a Mac wrote
the fallback template naming `Qwen/Qwen3-0.6B` -- a raw bf16 repo, which
the dead block's own comment says "imports fine and then fails to bind at
load", because Metal's llama path is 4-bit-only. The generated default
config could not serve.

Nothing caught it for months. `cargo` does emit `unexpected_cfgs`, and it
fired on all five -- into the warning stream of a package that no CI gate
lints, which is the same as not firing.

## Why a script and not a lint

The gates cannot reach the crates where this bug hides best. `pie`,
`runtime` and `worker` are ungated on warning count; `engine-cuda` and
`kernels-cuda` need `nvcc`; `kernels-vulkan` and `engine-vulkan` need
`slangc`; `engine-metal` and `kernels-metal` need a Mac. That is nine of
thirty-five members where `unexpected_cfgs` will not be denied on this
job, and they are disproportionately the ones with interesting `cfg`s --
a crate has feature-gated code precisely when it has optional backends.

The `kernels-cuda` in that list is the ARCHIVE crate and its reason has
since inverted, which is worth catching before someone reads it as still
true. That crate built `libpie_kernels_cuda.a` with CMake and nvcc under a
`native` feature, so `nvcc` was exactly what kept a gate off it; it was
deleted at `85c6c674b` and the JIT crate that took the name needs no toolkit
at all. Its `build.rs` writes a text file and says so in as many words, its
`cudarc` is `fallback-dynamic-loading` and optional, and NVRTC compiles the
device sources at RUN time -- so it builds on a machine that has never seen
a GPU, and `nvcc` is no longer a reason for anything.

The crate is still uncovered, for a duller reason: it is not among the
packages the workspace `cargo clippy -- -D warnings` step names. So the tally
above survives its own argument, and it is left standing rather than
recounted -- but the entry it rests on should be read as "not selected", not
as "cannot be built here".

This audit needs no compiler, no toolkit and no GPU, so it covers all of
them uniformly. It is strictly weaker than the lint on the crates the
lint reaches, and it is the only check at all on the crates it does not.

## What counts as declared

A package's `cfg(feature = "x")` is legitimate when `x` is:

  * a key in its own `[features]` table, or
  * an OPTIONAL dependency of that package -- cargo synthesizes an
    implicit feature per optional dep, so `#[cfg(feature = "transport")]`
    is valid wherever `transport = { ..., optional = true }` is written.

Features are per-package, so a name declared by a sibling does not count.
That is the whole point: `worker` declaring `engine-metal` is precisely
what made the root's missing forward invisible to a reader who grepped
the workspace for the string and found it.

## What it deliberately does not see

Only real `cfg(...)` and `cfg_attr(...)` forms are scanned, and line
comments are stripped first. `crates/worker/src/embedded_engine.rs`
writes `#[cfg(feature = "engine-…")]` inside a comment, with a literal
ellipsis, as prose standing for two real arms below it. A scanner that
matched `feature = "..."` anywhere in the file would report that as an
undeclared feature named `engine-…`, and the fix a reader would reach
for -- editing the prose -- would be a scanner appeasing itself.
"""

from __future__ import annotations

import os
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# `feature = "name"` appearing inside a cfg-ish attribute. Both `cfg` and
# `cfg_attr` are matched; so is the `all(...)`/`any(...)`/`not(...)`
# nesting, since the inner `feature = "x"` is found by scanning the
# balanced extent of the outer `cfg(`.
CFG_OPEN = re.compile(r"\bcfg(?:_attr)?\s*\(")
FEATURE = re.compile(r'feature\s*=\s*"([^"]*)"')
LINE_COMMENT = re.compile(r"//[^\n]*")
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)


def strip_comments(src: str) -> str:
    """Blank out comments, preserving byte offsets so spans stay valid.

    Doc comments (`///`, `//!`) are comments too and are stripped by the
    same rule -- a `cfg` written in documentation is documentation.
    """
    src = BLOCK_COMMENT.sub(lambda m: " " * len(m.group(0)), src)
    return LINE_COMMENT.sub(lambda m: " " * len(m.group(0)), src)


def cfg_spans(src: str) -> list[tuple[int, int]]:
    """The balanced extent of every `cfg(`/`cfg_attr(` in `src`."""
    spans = []
    for m in CFG_OPEN.finditer(src):
        depth, i = 1, m.end()
        while i < len(src) and depth:
            if src[i] == "(":
                depth += 1
            elif src[i] == ")":
                depth -= 1
            i += 1
        spans.append((m.end(), i))
    return spans


def features_used(path: Path) -> dict[str, set[Path]]:
    """Feature names named by a `cfg` in `path`, to the files naming them."""
    src = strip_comments(path.read_text(errors="replace"))
    used: dict[str, set[Path]] = {}
    for start, end in cfg_spans(src):
        for m in FEATURE.finditer(src, start, end):
            used.setdefault(m.group(1), set()).add(path)
    return used


def features_declared(manifest: dict) -> set[str]:
    """`[features]` keys plus the implicit feature of each optional dep."""
    declared = set(manifest.get("features", {}))
    for section in ("dependencies", "dev-dependencies", "build-dependencies"):
        for name, spec in (manifest.get(section) or {}).items():
            if isinstance(spec, dict) and spec.get("optional"):
                declared.add(spec.get("package", name))
                declared.add(name)
    # Target-specific dependency tables carry optional deps too.
    for target in (manifest.get("target") or {}).values():
        for section in ("dependencies", "dev-dependencies", "build-dependencies"):
            for name, spec in (target.get(section) or {}).items():
                if isinstance(spec, dict) and spec.get("optional"):
                    declared.add(spec.get("package", name))
                    declared.add(name)
    return declared


# Directories this audit does not descend into.
#
# `.claude` is the same exclusion the root `Cargo.toml` carries, and for a
# harder version of the same reason. That directory holds whole `git
# worktree` checkouts of THIS repository, one per agent, so a walk that
# descends finds every manifest in the tree again once per worktree, and
# the count is a multiple of however many worktrees exist rather than a
# property of the tree. Measured on the box this was written on: 3,612
# manifests pruned down to 79, and 5.8 seconds of walking down to 0.01 --
# before the old code went on to read and TOML-parse all 3,612 and glob
# `*.rs` beneath each, which is where the wait actually was. It also
# attributed another checkout's crates to this one. The root manifest
# excludes the directory so cargo cannot adopt a worktree's packages as
# members; this excludes it so the audit cannot report a worktree's `cfg`
# as this tree's, and so it finishes.
#
# `target` and `.git` were already filtered, but only AFTER `rglob` had
# descended into them and produced the paths: the files were skipped and
# the walk was still paid for. Pruning is where that belongs, and none of
# the three can hold a manifest this audit is about.
PRUNE = {".claude", ".git", "target"}


def manifests() -> list[Path]:
    """Every `Cargo.toml` in the tree, without descending into `PRUNE`."""
    found = []
    for parent, directories, files in os.walk(ROOT):
        directories[:] = [d for d in directories if d not in PRUNE]
        if "Cargo.toml" in files:
            found.append(Path(parent) / "Cargo.toml")
    return sorted(found)


def packages() -> list[tuple[str, Path, dict]]:
    """Every manifest in the tree that names a package, root included."""
    found = []
    for manifest_path in manifests():
        try:
            manifest = tomllib.loads(manifest_path.read_text())
        except tomllib.TOMLDecodeError as exc:
            print(f"cfg-feature-audit: cannot parse {manifest_path}: {exc}")
            sys.exit(1)
        name = manifest.get("package", {}).get("name")
        if name:
            found.append((name, manifest_path.parent, manifest))
    return found


def main() -> int:
    problems: list[str] = []
    scanned = 0
    found = packages()
    directories = [directory for _, directory, _ in found]
    for name, directory, manifest in found:
        # The root package lives at the repo root, so a naive walk of its
        # directory would sweep every crate under `crates/` and attribute
        # their features to it. A file belongs to the NEAREST enclosing
        # package, so skip anything inside a nested one.
        nested = [
            other
            for other in directories
            if other != directory and other.is_relative_to(directory)
        ]
        declared = features_declared(manifest)
        used: dict[str, set[Path]] = {}
        for source in sorted(directory.rglob("*.rs")):
            # The same prune the manifest walk takes, for the same reason: the
            # root package's directory IS the repo root, so an unpruned rglob
            # reads every agent worktree under `.claude` and reports another
            # checkout's crate as this one's dead cfg.
            if PRUNE.intersection(source.relative_to(directory).parts):
                continue
            if any(source.is_relative_to(other) for other in nested):
                continue
            for feature, files in features_used(source).items():
                used.setdefault(feature, set()).update(files)
            scanned += 1
        for feature in sorted(used):
            if feature in declared:
                continue
            where = ", ".join(
                sorted(str(f.relative_to(ROOT)) for f in used[feature])[:4]
            )
            problems.append(
                f"  {name}: `cfg(feature = \"{feature}\")` but {name} declares "
                f"no such feature -- that code compiles nowhere.\n"
                f"      named in: {where}\n"
                f"      declared: {', '.join(sorted(declared)) or '(none)'}"
            )

    if problems:
        print("cfg-feature-audit: a cfg names a feature its package does not have.")
        print()
        print("\n".join(problems))
        print()
        print(
            "Either declare the feature in that package's `[features]` table\n"
            "(forwarding to the dependency that really has it), or delete the\n"
            "dead `cfg` -- but do not leave it, because it reads as code that\n"
            "is merely off."
        )
        return 1

    print(
        f"cfg-feature-audit: {scanned} sources across {len(found)} packages; "
        f"every cfg feature is declared."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
