#!/usr/bin/env python3
"""Every package, feature and target a workflow names, resolved against cargo.

## Why this exists

A `-p` naming a crate cargo cannot see is not a narrower gate and not a
skip. Cargo exits before it compiles anything:

    error: package ID specification `engine-wgpu` did not match any packages

and so does a `--features` naming a feature no manifest declares, and so
does a `--test` naming a target that is not in the tree. Each is a hard
error on the FIRST step that reaches it, which means every step under it
runs never -- and a job that dies on step one looks, in a log nobody
opens, much like a job that had nothing to do.

This tree has now paid for that three times over, all in one branch:

* `-p engine-metal`, `-p engine-vulkan` and `-p engine-wgpu` appeared
  EIGHTEEN times in `ci.yml`, including three whole jobs, after R3 put
  all three crates in the root manifest's `exclude`.
* `cargo check -p model --features chat|contract|chat,contract` and
  `cargo test -p model --features contract` named three features R3
  deleted with `model-legacy`. The same step had already been through
  this once with a fourth name, `forward`, and was rewritten onto two
  names that were themselves about to go.
* `--features engine-wgpu` on `pie-gpu-tests`, `runtime`, `worker` and
  `pie` named a feature no package in the workspace has ever declared
  since the crate left it, and `--test shader_backends_agree` named a
  test target that is not in `crates/kernels/tests` at all.
* `build.yml`'s `aarch64-macos-metal` release leg built
  `-p pie --features engine-metal`, so an entire published artifact was
  produced by a command that could not run.

None of that was caught by anything. `scripts/ci-gate-audit.py` reads two
steps by name and checks their `-p` lists, which is what it is for and is
a fraction of the surface; the file that claimed to cover the rest --
`crates/model/tests/every_cfg_names_a_real_feature.rs`, cited in `ci.yml`
as "fails if a workflow names a package or target that does not exist" --
was itself deleted, so the claim outlived the check by some margin.

## What it checks

For every `cargo` command in every `run:` body of every workflow:

* each `-p`/`--package`/`--exclude` names a workspace member;
* each `--features` entry is declared -- a bare `f` by at least one of
  the packages the command selects, and a qualified `pkg/f` by `pkg`
  itself whenever `pkg` is a member;
* each `--test`/`--bench`/`--example`/`--bin` names a target of one of
  the selected packages.

`${{ matrix.… }}` values are resolved out of the job's own `strategy.
matrix` before any of that, so `--features ${{ matrix.cuda.feature }}` is
checked once per matrix entry rather than skipped.

## What it deliberately does not check

That the command PASSES. A red gate is a red build and CI reports it;
this is only about commands that cannot start. It also says nothing about
a step that is absent -- `ci-gate-audit.py` is the file that asks whether
the list is complete, and this one asks whether the list resolves. Two
questions, two files, and neither answers the other.

Expressions it cannot resolve (anything but a plain `matrix.<key>` or
`matrix.<key>.<field>`) are SKIPPED and counted, and the count is printed,
so a workflow that drifts into unreadable interpolation says so rather
than quietly narrowing what is covered.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - the runner installs it
    sys.exit(
        "workflow-ref-audit: needs PyYAML (`pip install pyyaml`). Unlike the\n"
        "sibling audits this one parses real YAML: matrix resolution needs the\n"
        "job's own `strategy.matrix`, and reconstructing that by indentation\n"
        "would be a second YAML parser with none of the first one's edge cases."
    )

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

EXPRESSION = re.compile(r"\$\{\{\s*([^}]*?)\s*\}\}")
MATRIX_PATH = re.compile(r"^matrix\.([A-Za-z0-9_-]+)(?:\.([A-Za-z0-9_-]+))?$")
SHELL_COMMENT = re.compile(r"^\s*#.*$", re.M)

# The flags this reads. `--exclude` is here because it takes a package
# name and cargo refuses an unknown one exactly as it refuses an unknown
# `-p`, which is a thing that is easy to forget until a sweep's exclusion
# list outlives a crate.
PACKAGE_FLAGS = {"-p", "--package", "--exclude"}
TARGET_FLAGS = {"--test": "test", "--bench": "bench", "--example": "example", "--bin": "bin"}


def workspace() -> dict[str, dict]:
    """`name -> package` for every workspace member, out of cargo itself."""
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return {p["name"]: p for p in json.loads(out.stdout)["packages"]}


def matrix_entries(job: dict) -> list[dict]:
    """One dict per matrix combination, or `[{}]` for an unmatrixed job.

    Only the axes matter, not the combinations: a value is checked once
    for each distinct value it can take, and `include`/`exclude` can only
    narrow that set. So each axis contributes its own values and the
    product is never built -- which also keeps a 2 x 8 CUDA matrix from
    turning into sixteen identical checks.
    """
    matrix = (job.get("strategy") or {}).get("matrix")
    if not isinstance(matrix, dict):
        return [{}]
    axes: dict[str, list] = {}
    for key, values in matrix.items():
        if key in ("include", "exclude") or not isinstance(values, list):
            continue
        axes[key] = values
    if not axes:
        return [{}]
    width = max(len(values) for values in axes.values())
    return [{key: values[i % len(values)] for key, values in axes.items()} for i in range(width)]


def substitute(text: str, entry: dict) -> tuple[str, int]:
    """Resolve `${{ matrix.… }}` against one matrix entry.

    Returns the text and how many expressions could not be resolved. An
    unresolved one is left as a sentinel that no flag parser will accept
    as a name, so it is skipped rather than mistaken for a package.
    """
    unresolved = 0

    def resolve(match: re.Match) -> str:
        nonlocal unresolved
        path = MATRIX_PATH.match(match.group(1))
        if path:
            axis, field = path.group(1), path.group(2)
            value = entry.get(axis)
            if field is not None and isinstance(value, dict):
                value = value.get(field)
            if isinstance(value, (str, int, float)):
                return str(value)
        unresolved += 1
        return "\x00"

    return EXPRESSION.sub(resolve, text), unresolved


def commands(body: str) -> list[list[str]]:
    """The `cargo …` invocations in a shell body, as token lists.

    Shell comments go first: this file's workflows quote retired commands
    in `#` lines inside a `run:` block, and a scanner that read those would
    report the very lines a comment exists to explain.
    """
    body = SHELL_COMMENT.sub("", body)
    body = body.replace("\\\n", " ")
    found = []
    for line in body.split("\n"):
        for piece in re.split(r"&&|\|\||;|\|", line):
            tokens = piece.split()
            if "cargo" in tokens:
                found.append(tokens[tokens.index("cargo") :])
    return found


def values_of(tokens: list[str], flag: str) -> list[str]:
    """Every value given to `flag`, in `--flag v` and `--flag=v` form."""
    found = []
    for index, token in enumerate(tokens):
        if token == flag and index + 1 < len(tokens):
            found.append(tokens[index + 1])
        elif token.startswith(f"{flag}="):
            found.append(token.split("=", 1)[1])
    return found


def check(tokens: list[str], members: dict[str, dict], where: str) -> list[str]:
    problems = []
    selected = []
    for flag in PACKAGE_FLAGS:
        for name in values_of(tokens, flag):
            if "\x00" in name:
                continue
            if name not in members:
                problems.append(
                    f"{where}: `{flag} {name}` is not a workspace member. "
                    f"Cargo refuses the whole command with `package ID "
                    f"specification ... did not match any packages`, so this "
                    f"step and everything after it in the job never runs."
                )
            elif flag != "--exclude":
                selected.append(name)

    for value in values_of(tokens, "--features"):
        if "\x00" in value:
            continue
        for feature in value.replace(",", " ").split():
            if "/" in feature:
                owner, name = feature.split("/", 1)
                owner = owner.removesuffix("?")
                if owner in members and name not in members[owner]["features"]:
                    problems.append(
                        f"{where}: `--features {feature}` but `{owner}` "
                        f"declares no feature `{name}`. Declared: "
                        f"{', '.join(sorted(members[owner]['features'])) or '(none)'}."
                    )
                continue
            if not selected:
                continue
            if any(feature in members[name]["features"] for name in selected):
                continue
            problems.append(
                f"{where}: `--features {feature}` but no package this command "
                f"selects ({', '.join(selected)}) declares it. Cargo exits "
                f"`does not have the feature`, which is a hard error and not a "
                f"narrower build. Declared: "
                + "; ".join(
                    f"{name} = {{{', '.join(sorted(members[name]['features'])) or ''}}}"
                    for name in selected
                )
                + "."
            )

    for flag, kind in TARGET_FLAGS.items():
        for name in values_of(tokens, flag):
            if "\x00" in name or not selected:
                continue
            if any(
                kind in target["kind"] and target["name"] == name
                for package in selected
                for target in members[package]["targets"]
            ):
                continue
            problems.append(
                f"{where}: `{flag} {name}` names no {kind} target of "
                f"{', '.join(selected)}. A `{flag}` naming a missing target is "
                f"not a skip -- cargo fails the step outright."
            )
    return problems


def main() -> int:
    members = workspace()
    problems: list[str] = []
    unresolved = 0
    checked = 0

    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text())
        for job_name, job in (workflow.get("jobs") or {}).items():
            entries = matrix_entries(job)
            for step in job.get("steps") or []:
                body = step.get("run")
                if not body:
                    continue
                where = f"{path.name}: {job_name}: {step.get('name') or '(unnamed step)'}"
                seen: set[str] = set()
                for entry in entries:
                    resolved, missed = substitute(body, entry)
                    unresolved += missed
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    for tokens in commands(resolved):
                        checked += 1
                        problems += check(tokens, members, where)

    problems = list(dict.fromkeys(problems))
    if problems:
        print("workflow-ref-audit: a workflow names something cargo cannot resolve\n")
        for problem in problems:
            print(f"  * {problem}")
        print(f"\n{len(problems)} problem(s).")
        return 1

    print(
        f"workflow-ref-audit: {checked} cargo commands across "
        f"{len(list(WORKFLOWS.glob('*.yml')))} workflows; every package, "
        f"feature and target resolves against {len(members)} members."
        + (f" {unresolved} expression(s) not resolvable and skipped." if unresolved else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
