#!/usr/bin/env python3
"""The kernel lint: rules every `kernels-*` crate's device tree keeps.

One check so far, the one the rest of the convention leans on:

  * every entry point in a crate's kernel tree is named by its host code.

A CUDA entry is a `__global__` function; the host names it in a symbol
string (`"::pie::attn::kv_append<...>"`), sometimes with a placeholder
spliced in by `format!` (`"::pie::linear::act_quant_e4m3{wide}<...>"`). A
placeholder stands for any identifier fragment, so `act_quant_e4m3{wide}`
reaches `act_quant_e4m3` and `act_quant_e4m3_v8`; a symbol that is nothing
but a placeholder (`"::pie::layout::{entry}"`) reaches nothing by itself,
and the names it is given must appear as literals elsewhere.

Exit status is the number of findings, capped at 1.
"""

import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# crate -> (kernel tree, host trees that may name an entry)
CRATES = {
    "kernels-cuda": (
        "crates/kernels-cuda/kernels",
        ["crates/kernels-cuda/src", "crates/engine-cuda/src"],
    ),
}

# Entry points the host does not name, each with the reason.
UNNAMED = {
    # Vendored FlashInfer, cut down when it is internalized (Incheon P3).
    "kernels-cuda": {"flashinfer/"},
}

GLOBAL = re.compile(
    r"__global__\s+(?:static\s+)?void\s+"
    r"(?:__launch_bounds__\s*\((?:[^()]|\([^()]*\))*\)\s*)?"
    r"([A-Za-z_]\w*)\s*\("
)
SYMBOL = re.compile(r'"::pie::[^"]*"')


def host_text(trees):
    """The host's string literals, op names (`"attention.decode"`) left out:
    an entry is named by a symbol or an entry literal, not by an op that
    shares its spelling."""
    text = "".join(p.read_text() for t in trees for p in (REPO / t).rglob("*.rs"))
    literals = re.findall(r'"(?:[^"\\]|\\.)*"', text, re.S)
    return "\n".join(l for l in literals if not re.fullmatch(r'"[a-z_]+\.[a-z0-9_]+"', l))


def spliced(host):
    """Regexes for symbol strings whose final segment has a placeholder."""
    out = []
    for lit in SYMBOL.findall(host):
        last = lit.strip('"').split("<", 1)[0].rsplit("::", 1)[-1]
        if "{" not in last or re.fullmatch(r"\{[^}]*\}", last):
            continue
        parts = re.split(r"\{[^}]*\}", last)
        out.append(re.compile(r"\w*".join(map(re.escape, parts))))
    return out


def unnamed_entries(crate):
    tree, hosts = CRATES[crate]
    host = host_text(hosts)
    patterns = spliced(host)
    skip = UNNAMED.get(crate, set())
    found = []
    for path in sorted(p for p in (REPO / tree).rglob("*") if p.suffix in (".cu", ".cuh")):
        rel = path.relative_to(REPO / tree).as_posix()
        if any(rel.startswith(s) for s in skip):
            continue
        text = path.read_text()
        for m in GLOBAL.finditer(text):
            name = m.group(1)
            if re.search(rf"\b{name}\b", host):
                continue
            if any(p.fullmatch(name) for p in patterns):
                continue
            line = text.count("\n", 0, m.start()) + 1
            found.append(f"{tree}/{rel}:{line}: `{name}` is named by no host code")
    return found


def main():
    findings = []
    for crate in CRATES:
        findings += unnamed_entries(crate)
    for f in findings:
        print(f)
    print(f"kernel-lint: {len(findings)} finding(s)", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
