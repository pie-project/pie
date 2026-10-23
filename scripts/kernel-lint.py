#!/usr/bin/env python3
"""The kernel lint: rules every `kernels-*` crate's device tree keeps.

The checks, the ones the rest of the convention leans on:

  * every entry point in a crate's kernel tree is named by its host code;
  * every launch the host writes with a literal symbol names, by namespace
    (and every `::pie::` scope in it, template arguments too, exists)
    and name, a kernel its unit (or a header the unit includes) defines;
  * kernels-cuda's `src/` mirrors `kernels/`: every unit has the host
    module at its path (a tier suffix `_smNN` dropped; a unit named after its
    directory, `attn/index/index.cu`, is the directory module's, and a group
    with no host submodules keeps its units in its root), and every
    host module under a group is a unit's, nests under one (`dense/voxel`), is
    a form beside one (`select_bf16` beside `select`), is a directory module
    the kernel tree also has, or says `//! Host-only:` on its first line (as
    do its submodules, then);
  * kernels-cerebras: every `library("<group>/<name>")` the host writes is
    a unit under `kernels/`, every unit outside `prim/` is named by one, and
    every unit sits in one of the convention's groups;
  * kernels-slang: every unit sits in one of the convention's groups and
    has its host module at the mirrored path; every `pie:instantiate` entry
    is named by host code (kernels-slang's `src/` or an engine's dispatch);
    every entry and unit the host names exists; every host module has its
    test at the mirrored path in each engine's tests.

A CUDA entry is a `__global__` function; the host names it in a symbol
string (`"::pie::attn::kv_append<...>"`), sometimes with a placeholder
spliced in by `format!` (`"::pie::linear::act_quant_e4m3{wide}<...>"`). A
placeholder stands for any identifier fragment, so `act_quant_e4m3{wide}`
reaches `act_quant_e4m3` and `act_quant_e4m3_v8`; a symbol that is nothing
but a placeholder (`"::pie::layout::{entry}"`) reaches nothing by itself,
and the names it is given must appear as literals elsewhere.

A Metal entry is what the shader compiler exports after the preprocessor
runs: a `[[host_name("...")]]` a macro stamps out of a template, or a plain
`kernel void`. The host names it in a string literal, the same way, and a
launch is `Fire::at("<unit>", "<entry>")`, the unit a path under `kernels/`
or a `const FILE` naming one.

Exit status is the number of findings, capped at 1.
"""

import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent

# crate -> (kernel tree, host trees that may name an entry)
CRATES = {
    "kernels-cuda": (
        "crates/kernels-cuda/kernels",
        ["crates/kernels-cuda/src", "crates/engine-cuda/src"],
    ),
    "kernels-rocm": (
        "crates/kernels-rocm/kernels",
        ["crates/kernels-rocm/src", "crates/engine-cuda/src"],
    ),
}

# crate -> (kernel tree, host trees), for the Metal checks. Tests are hosts:
# an entry only a test fires (an f32-out probe of a decoder) is named.
METAL = {
    "kernels-metal": (
        "crates/kernels-metal/kernels",
        [
            "crates/kernels-metal/src",
            "crates/kernels-metal/tests",
            "crates/engine-metal/src",
            "crates/engine-metal/tests",
        ],
    ),
}

# Entry points the host does not name, by path prefix, each with the reason.
UNNAMED = {}

# `__launch_bounds__(...)` and HIP's `__attribute__((...))` may sit on either
# side of `void`.
_ATTRS = (
    r"(?:(?:__launch_bounds__\s*\((?:[^()]|\([^()]*\))*\)"
    r"|__attribute__\s*\(\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)\))\s*)*"
)
GLOBAL = re.compile(
    r"__global__\s+(?:static\s+)?" + _ATTRS + r"void\s+" + _ATTRS + r"([A-Za-z_]\w*)\s*\("
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
    for path in sorted(p for p in (REPO / tree).rglob("*") if p.suffix in (".cu", ".cuh", ".hip")):
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


UNIT_CONST = re.compile(r'const ([A-Z_0-9]+): &str = "([a-z_0-9/]+\.(?:cuh?|hip))";')
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


def unit_scopes(tree, unit, seen):
    """Every namespace path a unit and its headers open, and every type they
    declare (by bare name)."""
    path = tree / unit
    if unit in seen or not path.exists():
        return set(), set()
    seen.add(unit)
    text = path.read_text()
    spaces, types = set(), set()
    for m in re.finditer(r"namespace ([\w:]+) \{", text):
        outer = namespace_at(text, m.start())
        full = f"{outer}::{m.group(1)}" if outer else m.group(1)
        parts = full.split("::")
        spaces |= {"::".join(parts[: i + 1]) for i in range(len(parts))}
    types |= set(re.findall(r"\b(?:struct|class|union)\s+(\w+)", text))
    types |= set(re.findall(r"\busing\s+(\w+)\s*=", text))
    for inc in re.findall(r'#include "([^"]+)"', text):
        a, b = unit_scopes(tree, inc, seen)
        spaces |= a
        types |= b
    return spaces, types


def stray_qualifiers(literal, spaces, types):
    """The `::pie::...::` qualifiers in a launched symbol, template arguments
    included, that name neither a namespace nor a type the unit holds."""
    bad = []
    for m in re.finditer(r"::((?:\w+::)+)", literal):
        path = m.group(1).rstrip(":")
        parts = path.split("::")
        if path in spaces or not parts[0] == "pie":
            continue
        if "::".join(parts[:-1]) in spaces and parts[-1] in types:
            continue
        bad.append(path)
    return bad


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
            # The whole literal: a template argument names its own scope.
            q = text.find('"::', m.start())
            literal = text[q + 1 : text.find('"', q + 1)]
            spaces, types = set(), set()
            for u in units:
                a, b = unit_scopes(tree, u, set())
                spaces |= a
                types |= b
            for stray in stray_qualifiers(literal, spaces, types):
                line = text.count("\n", 0, m.start()) + 1
                rel = path.relative_to(REPO)
                found.append(f"{rel}:{line}: `{stray}` in `{literal}` is no scope of {units}")
    return found


def stray_symbols(crate):
    """Every `"::pie::..."` literal in a host file (a launch's, or a table's
    it fires from) whose scopes no unit the file names holds."""
    tree = REPO / CRATES[crate][0]
    src = tree.parent / "src"
    found = []
    for path in sorted(src.rglob("*.rs")):
        text = path.read_text()
        parent = path.parent.with_suffix(".rs")
        consts = dict(UNIT_CONST.findall(text))
        if parent.exists():
            consts.update(UNIT_CONST.findall(parent.read_text()))
        units = set(consts.values()) | set(re.findall(r'Fire::at\(\s*"([^"]+)"', text))
        if not units:
            continue
        spaces, types = set(), set()
        for u in units:
            a, b = unit_scopes(tree, u, set())
            spaces |= a
            types |= b
        if not spaces:
            continue
        for m in re.finditer(r'"(::pie::(?:[^"\\]|\\.|\\\n)*)"', text):
            literal = re.sub(r"\\\n\s*", "", m.group(1))
            for stray in stray_qualifiers(literal, spaces, types):
                line = text.count("\n", 0, m.start()) + 1
                rel = path.relative_to(REPO)
                found.append(f"{rel}:{line}: `{stray}` in a symbol is no scope of {sorted(units)}")
    return found


METAL_HOST_NAME = re.compile(r'host_name\(\s*((?:"[^"]*"\s*)+)\)')
METAL_PLAIN = re.compile(r"(?m)^\s*(?:\[\[kernel\]\]|kernel)\s+void\s+(\w+)\s*\(")
METAL_FILE = re.compile(r'const ([A-Z_0-9]+): &str = "([a-z_0-9/]+\.metal)";')
METAL_LAUNCH = re.compile(r'Fire::at\(\s*(?:"([^"]+)"|([A-Z_0-9]+))\s*,\s*"([^"]+)"')


def metal_entries(tree, unit, cache={}):
    """What `unit` exports: its text with quoted includes spliced (from the
    including file's directory, as `sources::resolve` does) and its macros
    expanded."""
    if unit in cache:
        return cache[unit]
    path = tree / unit

    def splice(at, text, depth=0):
        def one(m):
            target = (at / m.group(1)).resolve()
            return splice(target.parent, target.read_text(), depth + 1) if target.exists() else ""

        return re.sub(r'(?m)^\s*#include\s*"([^"]+)"\s*$', one, text) if depth < 8 else text

    text = splice(path.parent, path.read_text())
    text = re.sub(r"(?m)^\s*#include\s*<[^>]+>\s*$", "", text)
    text = subprocess.run(
        ["gcc", "-E", "-P", "-x", "c", "-"], input=text, capture_output=True, text=True
    ).stdout
    names = {"".join(re.findall(r'"([^"]*)"', g)) for g in METAL_HOST_NAME.findall(text)}
    for m in METAL_PLAIN.finditer(text):
        head = text[: m.start()]
        if "template" not in head[max(head.rfind(";"), head.rfind("}")) + 1 :]:
            names.add(m.group(1))
    cache[unit] = names
    return names


def metal_units(tree):
    return sorted(p.relative_to(tree).as_posix() for p in tree.rglob("*.metal"))


def metal_named(host):
    """The words the host's literals spell, and a pattern per literal that
    splices a `format!` placeholder into a name."""
    words = set(re.findall(r"\w+", host))
    spliced = []
    for lit in re.findall(r'"([^"\\]*\{[^"\\]*)"', host):
        if re.fullmatch(r"[\w{}:]+", lit) and re.search(r"\w\{|\}\w", lit):
            parts = re.split(r"\{[^}]*\}", lit)
            # A literal whose fixed text is a letter or two (`"{p}m{k}"`, a
            # tensor's name) names no kernel; as a pattern it would match
            # every entry with that letter in it.
            if max(map(len, parts)) < 3:
                continue
            spliced.append(re.compile(r"\w*".join(map(re.escape, parts))))
    return words, spliced


def metal_unnamed_entries(crate):
    tree, hosts = METAL[crate]
    tree = REPO / tree
    words, spliced = metal_named(host_text(hosts))
    found = []
    for unit in metal_units(tree):
        for name in sorted(metal_entries(tree, unit)):
            if name in words or any(p.fullmatch(name) for p in spliced):
                continue
            found.append(f"{METAL[crate][0]}/{unit}: `{name}` is named by no host code")
    return found


def metal_misdirected_launches(crate):
    tree, hosts = METAL[crate]
    tree = REPO / tree
    found = []
    for path in sorted(p for t in hosts for p in (REPO / t).rglob("*.rs")):
        text = path.read_text()
        consts = dict(METAL_FILE.findall(text))
        for m in METAL_LAUNCH.finditer(text):
            unit = m.group(1) or consts.get(m.group(2))
            if unit is None:
                continue
            line = text.count("\n", 0, m.start()) + 1
            rel = path.relative_to(REPO)
            if not (tree / unit).exists():
                found.append(f"{rel}:{line}: `{unit}` is no unit under {METAL[crate][0]}")
                continue
            # `.stamp("MACRO(\"name\", ...)")` instantiates the entry at run time.
            stamp = re.match(r'\s*\.stamp\(\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text[m.end() + 1 :])
            if stamp and f'\\"{m.group(3)}\\"' in stamp.group(1):
                macro = stamp.group(1).split("(", 1)[0]
                if re.search(rf"#define\s+{macro}\b", (tree / unit).read_text()):
                    continue
            if m.group(3) not in metal_entries(tree, unit):
                found.append(f"{rel}:{line}: `{m.group(3)}` is defined in none of ['{unit}']")
    return found


# kernels-cerebras: CSL units, looked up by path (`cx.library("elemwise/rmsnorm")`).
CSL_TREE = "crates/kernels-cerebras/kernels"
CSL_HOST = ["crates/kernels-cerebras/src", "crates/engine-cerebras/src"]
CSL_GROUPS = {
    "prim", "attn", "ssm", "linear", "elemwise", "conv", "gather", "sampling", "collective",
    "runtime",
}
LIBRARY = re.compile(r'library\("([^"]+)"\)')


def csl_units():
    tree = REPO / CSL_TREE
    units = {p.relative_to(tree).with_suffix("").as_posix() for p in tree.rglob("*.csl")}
    findings = []
    for unit in sorted(units):
        group = unit.split("/")[0]
        if "/" not in unit or group not in CSL_GROUPS:
            findings.append(f"{CSL_TREE}/{unit}.csl: `{group}` is not one of the convention's groups")
    named = set()
    for root in CSL_HOST:
        for path in sorted((REPO / root).rglob("*.rs")):
            text = path.read_text()
            for m in LIBRARY.finditer(text):
                named.add(m.group(1))
                if m.group(1) not in units:
                    line = text.count("\n", 0, m.start()) + 1
                    rel = path.relative_to(REPO)
                    findings.append(f"{rel}:{line}: `{m.group(1)}` is no unit under {CSL_TREE}")
    for unit in sorted(units - named):
        if not unit.startswith("prim/"):
            findings.append(f'{CSL_TREE}/{unit}.csl: no host `library("{unit}")` names it')
    return findings


# kernels-slang: one Slang tree both engines fire from. An entry is a
# `// pie:instantiate NAME ...` line; the host names it in a literal
# (`"gather_rows_copy_words"`) or splices it with `format!`
# (`"linear_kquant{}_{}_bf16"`), and names a unit in a `const` (`"linear/kquant.slang"`).
SLANG_TREE = "crates/kernels-slang/kernels"
SLANG_SRC = "crates/kernels-slang/src"
SLANG_HOSTS = [SLANG_SRC, "crates/engine-vulkan/src/dispatch", "crates/engine-wgpu/src/dispatch"]
SLANG_TESTS = ["crates/engine-vulkan/tests/engine_vulkan", "crates/engine-wgpu/tests/engine_wgpu"]
SLANG_INSTANTIATE = re.compile(r"(?m)^\s*// pie:instantiate (\w+)")
SLANG_UNIT = re.compile(r'"([a-z_0-9/]+\.slang)"')
SLANG_ENTRY = re.compile(
    r'"((?:prim|attn|ssm|linear|elemwise|conv|gather|sampling|collective|runtime)_[\w{}:]*)"'
)


def slang_tree():
    tree = REPO / SLANG_TREE
    units = {}
    for path in sorted(tree.rglob("*.slang")):
        unit = path.relative_to(tree).with_suffix("").as_posix()
        units[unit] = SLANG_INSTANTIATE.findall(path.read_text())
    return units


def slang_pattern(literal):
    """A literal's entry pattern: a `format!` placeholder stands for any
    identifier fragment, and a literal ending in `_` is a prefix (a host
    that sorts entries by family). `None` when its fixed text is too short
    to name a kernel."""
    parts = re.split(r"\{[^}]*\}", literal)
    if len(parts) > 1 and max(map(len, parts)) < 3:
        return None
    pattern = r"\w*".join(map(re.escape, parts))
    return re.compile(pattern + r"\w*" if literal.endswith("_") else pattern)


def slang_host(unit):
    """The host module that owns `unit`: its mirror, or, for a tier or
    algorithm unit (`attn/dense_tiled`), its family's (`attn/dense.rs`)."""
    head, _, stem = unit.rpartition("/")
    words = stem.split("_")
    for n in range(len(words), 0, -1):
        path = REPO / SLANG_SRC / head / ("_".join(words[:n]) + ".rs")
        if path.exists():
            return path
    return None


def slang_checks():
    units = slang_tree()
    entries = {name for names in units.values() for name in names}
    findings = []
    for unit in units:
        group = unit.split("/")[0]
        if "/" not in unit or group not in CSL_GROUPS:
            findings.append(f"{SLANG_TREE}/{unit}.slang: `{group}` is not one of the convention's groups")
        elif group != "prim" and slang_host(unit) is None:
            findings.append(f"{SLANG_TREE}/{unit}.slang: no host module at {SLANG_SRC}/{unit}.rs")
        elif group != "prim":
            host = slang_host(unit).relative_to(REPO / SLANG_SRC).as_posix()
            for tests in SLANG_TESTS:
                if not (REPO / tests / host).exists():
                    findings.append(f"{SLANG_SRC}/{host}: no test at {tests}/{host}")
    patterns = []
    for root in SLANG_HOSTS:
        for path in sorted((REPO / root).rglob("*.rs")):
            text = path.read_text()
            rel = path.relative_to(REPO)
            for m in SLANG_UNIT.finditer(text):
                if m.group(1)[: -len(".slang")] not in units:
                    line = text.count("\n", 0, m.start()) + 1
                    findings.append(f"{rel}:{line}: `{m.group(1)}` is no unit under {SLANG_TREE}")
            for m in SLANG_ENTRY.finditer(text):
                pattern = slang_pattern(m.group(1))
                if pattern is None:
                    continue
                patterns.append(pattern)
                if not any(pattern.fullmatch(e) for e in entries):
                    line = text.count("\n", 0, m.start()) + 1
                    findings.append(f"{rel}:{line}: `{m.group(1)}` names no `pie:instantiate` entry")
    for unit, names in units.items():
        for name in names:
            if not any(p.fullmatch(name) for p in patterns):
                findings.append(f"{SLANG_TREE}/{unit}.slang: `{name}` is named by no host code")
    return findings


# kernels-cuda's host mirror (conventions.md section 7).
MIRROR = ("crates/kernels-cuda/kernels", "crates/kernels-cuda/src")
MIRROR_GROUPS = ["attn", "collective", "conv", "elemwise", "gather", "linear", "runtime",
                 "sampling", "ssm"]
TIER = re.compile(r"_sm\d+a?$")


def host_mirror():
    tree, src = (REPO / d for d in MIRROR)
    stems = set()
    for cu in tree.rglob("*.cu"):
        rel = cu.relative_to(tree).with_suffix("").as_posix()
        if rel.split("/")[0] in MIRROR_GROUPS:
            stems.add(TIER.sub("", rel))
    dirs = {d.relative_to(tree).as_posix() for d in tree.rglob("*") if d.is_dir()}

    def owner(stem):
        # a unit named after its directory is the directory module's, and
        # a group with no host submodules keeps its units in its root
        head, _, leaf = stem.rpartition("/")
        if head and head.rsplit("/", 1)[-1] == leaf:
            return head
        group = stem.split("/")[0]
        if not (src / group).is_dir() and (src / f"{group}.rs").is_file():
            return group
        return stem

    owners = {owner(s) for s in stems}
    findings = []
    for stem in sorted(stems):
        if not (src / f"{owner(stem)}.rs").is_file():
            findings.append(f"{MIRROR[0]}/{stem}.cu: no host module `src/{owner(stem)}.rs`")
    for rs in sorted(src.rglob("*.rs")):
        mod = rs.relative_to(src).with_suffix("").as_posix()
        if mod.split("/")[0] not in MIRROR_GROUPS or "/" not in mod:
            continue
        parts = mod.split("/")
        nested = any("/".join(parts[:i]) in owners for i in range(2, len(parts)))
        form = any(mod.startswith(o + "_") and "/" not in mod[len(o) + 1:] for o in owners)
        def host_only(m):
            f = src / f"{m}.rs"
            return f.is_file() and f.read_text().startswith("//! Host-only:")

        under_host = any(host_only("/".join(parts[:i])) for i in range(2, len(parts)))
        if mod in owners or nested or form or mod in dirs or host_only(mod) or under_host:
            continue
        findings.append(f"{MIRROR[1]}/{mod}.rs: owns no unit, nests under none, and says no "
                        "`//! Host-only:`")
    return findings


def main():
    findings = []
    for crate in CRATES:
        findings += unnamed_entries(crate)
        findings += misdirected_launches(crate)
        findings += stray_symbols(crate)
    for crate in METAL:
        findings += metal_unnamed_entries(crate)
        findings += metal_misdirected_launches(crate)
    findings += csl_units()
    findings += slang_checks()
    findings += host_mirror()
    for f in findings:
        print(f)
    print(f"kernel-lint: {len(findings)} finding(s)", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
