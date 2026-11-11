#!/usr/bin/env python3
"""compare.py <reference dir> <candidate dir> [--allow glob,...]: two capture.py trees, file for file.

Prints every file that differs, is missing or is new (by manifest.json), and for a differing text file its first
differing lines (a .gz beside a file is read in its place). Files matching an --allow glob (relative paths,
fnmatch) are listed but do not fail. Exit 0 when nothing outside --allow differs.

A config's build/ is compared by what it compiles, not by its text: layout_eval.py runs each layout.csl with
its params.used, and every PE must get the same source with the same params, the layouts the same exported
names, and every source a PE gets the same code (comments aside: a parametric layout and one with literal
params build the same program)."""
import fnmatch, gzip, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layout_eval

DEFAULT_ALLOW = []


def lines(top, rel, n=None):
    p = os.path.join(top, rel)
    op = (lambda: gzip.open(p + ".gz", "rt")) if not os.path.exists(p) and os.path.exists(p + ".gz") else (lambda: open(p, errors="replace"))
    with op() as f:
        return f.readlines() if n is None else [next(f, "") for _ in range(n)]


def code(path):
    """A CSL source without its comments and blank lines (what cslc compiles)."""
    out = []
    for line in open(path, errors="replace"):
        line = re.sub(r"//.*", "", line).rstrip()
        if line:
            out.append(line)
    return out


def build(a, b, cfg):
    """The differences between two configs' builds (a list of strings, empty when they compile the same)."""
    out = []

    def layout(top):
        """A build's layout directory: build.py copied it to the build's top, wafer-compile keeps it in compiled/layout."""
        return top if os.path.exists(os.path.join(top, "layout.csl")) else os.path.join(top, "compiled", "layout")

    ta_, tb_ = os.path.join(a, cfg, "build"), os.path.join(b, cfg, "build")
    da, db = layout(ta_), layout(tb_)
    if not os.path.exists(os.path.join(db, "layout.csl")):
        return ["no candidate build"]

    def ev(d, top):
        used = os.path.join(top, "params.used")
        p = layout_eval.parse_params(open(used).read()) if os.path.exists(used) else {}
        return layout_eval.tiles(open(os.path.join(d, "layout.csl")).read(), p)

    (ta, ea), (tb, eb) = ev(da, ta_), ev(db, tb_)
    if ea != eb:
        out.append(f"exports: -{sorted(set(ea) - set(eb))} +{sorted(set(eb) - set(ea))}")
    pes = sorted(set(ta) | set(tb))
    diff = [pe for pe in pes if ta.get(pe) != tb.get(pe)]
    for pe in diff[:5]:
        out.append(f"PE {pe}: {ta.get(pe)} vs {tb.get(pe)}")
    if diff:
        out.append(f"{len(diff)} of {len(pes)} PEs differ")
    for src in sorted({s for s, _ in ta.values()} | {s for s, _ in tb.values()}):
        fa, fb = os.path.join(da, src), os.path.join(db, src)
        if not (os.path.exists(fa) and os.path.exists(fb)) or code(fa) != code(fb):
            out.append(f"source {src} differs or is missing")
    return out


def main():
    a, b = sys.argv[1], sys.argv[2]
    allow = list(DEFAULT_ALLOW)
    if "--allow" in sys.argv:
        allow += sys.argv[sys.argv.index("--allow") + 1].split(",")
    ma = json.load(open(os.path.join(a, "manifest.json")))
    mb = json.load(open(os.path.join(b, "manifest.json")))
    bad = 0
    for cfg in sorted({r.split("/")[0] for r in ma if "/build/" in r}):
        d = build(a, b, cfg)
        print(f"{cfg}/build: " + ("same program" if not d else "DIFFERS"))
        for x in d:
            print("  " + x)
        bad += bool(d)
    for rel in sorted(set(ma) | set(mb)):
        if "/build/" in rel or ma.get(rel) == mb.get(rel):
            continue
        ok = any(fnmatch.fnmatch(rel, g) for g in allow)
        state = "missing" if rel not in mb else "new" if rel not in ma else "differs"
        print(f"{'allowed ' if ok else ''}{state}: {rel}")
        if not ok:
            bad += 1
        if state == "differs" and not ok:
            try:
                la, lb = lines(a, rel), lines(b, rel)
                shown = 0
                for i in range(max(len(la), len(lb))):
                    x = la[i] if i < len(la) else "<eof>\n"
                    y = lb[i] if i < len(lb) else "<eof>\n"
                    if x != y:
                        print(f"  line {i + 1}:\n    - {x.rstrip()[:300]}\n    + {y.rstrip()[:300]}")
                        shown += 1
                        if shown == 5:
                            break
                print(f"  ({len(la)} vs {len(lb)} lines)")
            except Exception as e:
                print(f"  (not compared as text: {e})")
    print("identical" if not bad else f"{bad} file(s) differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
