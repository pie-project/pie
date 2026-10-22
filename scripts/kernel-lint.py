#!/usr/bin/env python3
"""The kernel lint: rules every `kernels-*` crate's device tree keeps.

Two checks so far, the ones the rest of the convention leans on:

  * every entry point in a crate's kernel tree is named by its host code;
  * every launch the host writes with a literal symbol names, by namespace
    and name, a kernel its unit (or a header the unit includes) defines.

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

# Entry points the host does not name, by path prefix, each with the reason.
UNNAMED = {}

GLOBAL = re.compile(
    r"__global__\s+(?:static\s+)?"
    r"(?:__launch_bounds__\s*\((?:[^()]|\([^()]*\))*\)\s*)?void\s+"
    r"(?:__launch_bounds__\s*\((?:[^()]|\([^()]*\))*\)\s*)?"
    r"([A-Za-z_]\w*)\s*\("
)
SYMBOL = re.compile(r'"::pie::[^"]*"')


def host_text(trees):
    """The host's string literals, op names (`"attention.decode"`) left out:
    an entry is named by a symbol or an entry literal, not by an op that
    shares its spelling."""
    text = "".join(p.read_text() for t in trees for p in (REPO / t).rglob("*.rs"))
    # Char literals first: a '"' would pair with the next string's quote.
    text = re.sub(r"'(?:\\.|[^'\\])'", "''", text)
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


UNIT_CONST = re.compile(r'const ([A-Z_0-9]+): &str = "([a-z_0-9/]+\.cuh?)";')
LAUNCH = re.compile(
    r"Fire::at\(\s*(?:if \w+ \{ ([A-Z_0-9]+) \} else \{ ([A-Z_0-9]+) \}|([A-Z_0-9]+)|\"([^\"]+)\")"
    r'\s*,\s*(?:symbol\(\s*&?(?:format!\()?\s*)?"::((?:\w+::)*)(\w*)(\{?)'
)


def namespace_at(text, pos):
    stack = []
    for m in re.finditer(r"namespace ([\w:]+) \{|\{|\}", text[:pos]):
        if m.group(1):
            stack.append(m.group(1))
        elif m.group(0) == "{":
            stack.append(None)
        elif stack:
            stack.pop()
    return "::".join(x for x in stack if x)


def unit_kernels(tree, unit, seen):
    path = tree / unit
    if unit in seen or not path.exists():
        return set()
    seen.add(unit)
    text = path.read_text()
    found = {(namespace_at(text, m.start()), m.group(1)) for m in GLOBAL.finditer(text)}
    for inc in re.findall(r'#include "([^"]+)"', text):
        found |= unit_kernels(tree, inc, seen)
    return found


def misdirected_launches(crate):
    tree = REPO / CRATES[crate][0]
    src = tree.parent / "src"
    found = []
    for path in sorted(src.rglob("*.rs")):
        text = path.read_text()
        here = dict(UNIT_CONST.findall(text))
        parent = path.parent.with_suffix(".rs")
        above = dict(UNIT_CONST.findall(parent.read_text())) if parent.exists() else {}
        for m in LAUNCH.finditer(text):
            a, b, c, literal, qual, name, spliced = m.groups()
            fn_at = max(text.rfind(k, 0, m.start()) for k in ("\nfn ", "\npub fn ", "\npub(crate) fn "))
            local = dict(UNIT_CONST.findall(text[fn_at : m.start()]))
            units = [local.get(x) or here.get(x) or above.get(x) for x in (a, b, c) if x] or [literal]
            if None in units:
                continue
            ns = qual.rstrip(":")
            kernels = set().union(*(unit_kernels(tree, u, set()) for u in units))
            hit = any(n == ns and (k.startswith(name) if spliced else k == name) for n, k in kernels)
            if not hit:
                line = text.count("\n", 0, m.start()) + 1
                rel = path.relative_to(REPO)
                found.append(f"{rel}:{line}: `{ns}::{name}` is defined in none of {units}")
    return found


def main():
    findings = []
    for crate in CRATES:
        findings += unnamed_entries(crate)
        findings += misdirected_launches(crate)
    for f in findings:
        print(f)
    print(f"kernel-lint: {len(findings)} finding(s)", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
