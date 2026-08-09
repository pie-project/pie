"""The module-reachability audit: a file no `mod` declares is INVISIBLE.

Rust does not compile a file because it exists. It compiles a file because
some module declares it, and a `.rs` under `src/` that nothing declares is
read by nobody -- not by rustc, not by clippy, not by any test. It is not
dead code, which the compiler reports. It is code the compiler has never
seen, which nothing reports at all.

The tree has paid for this once. Commit `a86d6478d` deleted
`driver-metal-new/src/metal/bind.rs`, **886 lines the compiler had never
once read**, left behind when the module list was trimmed. The commit
message is explicit that nothing would ever have told anyone. That is the
whole reason this exists.

## Why it walks the tree instead of grepping for the stem

The obvious version -- "is there a `mod <stem>;` anywhere in the crate" --
over-reports and under-reports at once. It over-reports every `mod.rs`,
which is named by its DIRECTORY and never by its own stem, and every
`[[bin]]` target, which Cargo declares in the manifest rather than in
Rust. And it under-reports a file declared under a different parent than
the one it sits beside, which `#[path]` makes legal.

So this starts at each crate's real roots and follows declarations, which
is what rustc does.

## What counts as a declaration

`mod x;` and `pub mod x;`, with any visibility and any `cfg` -- a
`#[cfg(feature = "...")]` module is still DECLARED, and a file behind a
feature nobody enables is a different problem from a file nobody names.
An inline `mod x { ... }` declares no file and is skipped. `#[path = "y"]`
redirects, and is honoured.

A declaration `mod x;` inside `a/mod.rs` or inside `a.rs` resolves to
`a/x.rs` or `a/x/mod.rs`, which is the one rule the whole walk needs.
"""
import re, sys, pathlib, tomllib

ROOT = pathlib.Path(".")

# `mod x;` — the trailing semicolon is what distinguishes a declaration
# that names a FILE from an inline `mod x {` that does not.
# `r#` because a module may be named with a raw identifier, and
# `waker/src/lib.rs` writes `mod r#loom;`. The `r#` is not part of the
# FILE name — the file is `loom.rs` — so it is matched and dropped.
MOD = re.compile(
    r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?mod\s+(?:r#)?([A-Za-z_][A-Za-z0-9_]*)\s*;",
    re.M,
)
# `#[path = "..."]` on the line(s) before a declaration.
PATH_ATTR = re.compile(r'#\s*\[\s*path\s*=\s*"([^"]+)"\s*\]')


def roots(crate: pathlib.Path):
    """Every file Cargo itself names as an entry point."""
    manifest = crate / "Cargo.toml"
    if not manifest.is_file():
        return []
    cfg = tomllib.loads(manifest.read_text())
    out = []

    def add(rel):
        p = crate / rel
        if p.is_file():
            out.append(p)

    lib = cfg.get("lib", {})
    add(lib.get("path", "src/lib.rs"))
    for target in ("bin", "test", "bench", "example"):
        for t in cfg.get(target, []) or []:
            if "path" in t:
                add(t["path"])
            elif "name" in t and target == "bin":
                add(f"src/bin/{t['name']}.rs")
    # Cargo's autodiscovery: `src/main.rs`, `src/bin/*.rs`, and the test,
    # bench and example directories are targets whether or not the
    # manifest says so. Each is a root, not a module of anything.
    add("src/main.rs")
    for d, pat in (("src/bin", "*.rs"), ("tests", "*.rs"),
                   ("benches", "*.rs"), ("examples", "*.rs")):
        for p in sorted((crate / d).glob(pat)):
            out.append(p)
        for p in sorted((crate / d).glob("*/main.rs")):
            out.append(p)
    return out


def declared(text):
    """The module names this file declares, with any `#[path]` override."""
    out = []
    for m in MOD.finditer(text):
        # Look back over the attributes immediately above the declaration.
        head = text[max(0, m.start() - 400):m.start()]
        over = None
        tail = head.rsplit("\n", 12)[-12:]
        for line in reversed(tail):
            s = line.strip()
            if not s or s.startswith("//") or s.startswith("#["):
                p = PATH_ATTR.search(s)
                if p:
                    over = p.group(1)
                    break
                continue
            break
        out.append((m.group(1), over))
    return out


def walk(root: pathlib.Path, seen: set):
    """Follow declarations from `root`, marking every file reached."""
    stack = [root]
    while stack:
        f = stack.pop()
        if f in seen or not f.is_file():
            continue
        seen.add(f)
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        # A declaration in `a/mod.rs` or in the crate root resolves
        # beside it; one in `a.rs` resolves under `a/`.
        base = f.parent if f.name in ("mod.rs", "lib.rs", "main.rs") else f.parent / f.stem
        for name, over in declared(text):
            if over:
                stack.append((f.parent / over).resolve())
                continue
            for cand in (base / f"{name}.rs", base / name / "mod.rs"):
                if cand.is_file():
                    stack.append(cand)
                    break


def main():
    crates = sorted(p for p in (ROOT / "crates").iterdir() if (p / "Cargo.toml").is_file())
    orphans = []
    walked = 0
    for crate in crates:
        seen = set()
        for r in roots(crate):
            walk(r.resolve(), seen)
        walked += len(seen)
        for f in sorted((crate / "src").rglob("*.rs")):
            if f.resolve() not in seen:
                orphans.append(f)

    # THE SELF-VACUITY CHECK, and it is not optional. A walk that resolved
    # nothing would report zero orphans and look exactly like a clean tree
    # -- which is the failure mode of the audit this file replaces, and of
    # the kernel-vocabulary audit before it was fixed.
    if walked < 400:
        print(f"mod audit: only reached {walked} files, so the walk broke "
              f"rather than the tree being clean", file=sys.stderr)
        return 2

    if orphans:
        print("Files under src/ that no `mod` declares — the compiler has "
              "never read these:", file=sys.stderr)
        for f in orphans:
            print(f"  {f}", file=sys.stderr)
        print("\nEither declare them, or delete them. A file the compiler "
              "cannot see is not dead code; it is code no check covers.",
              file=sys.stderr)
        return 1

    print(f"mod audit: {walked} files reachable, 0 orphans")
    return 0


if __name__ == "__main__":
    sys.exit(main())
