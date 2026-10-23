#!/usr/bin/env python3
"""canon.py BASE_DIR CAND_DIR [LIST]: compare StableHLO dumps up to the order
ops are emitted in and the SSA names they get.

Every value is named by a hash of what computes it: the op's name, its
operands' hashes, its attributes, result types and `loc` (the scope that
emitted it), and its regions' hashes. Function arguments hash by position,
type and aliasing, so a changed argument order shows as a difference. Two
functions are equivalent when their signatures match, their returns name the
same hashes in the same order, and they hold the same multiset of op hashes
(so a dead op added or dropped shows too). Region bodies compare the same way.

Prints one line per file that differs in bytes: `equivalent` or `DIFFERENT`
with the first unmatched op; exits 1 if any is DIFFERENT. LIST (optional) is
a `diff -rq` listing to restrict the files compared.
"""
import collections
import hashlib
import os
import re
import sys

REF = re.compile(r"%[\w#]+")
OP = re.compile(r'^\s*(?:(%\w+)(?::(\d+))? = )?"([\w.]+)"\((.*?)\)(.*)$')


def h(*parts):
    return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:20]


class Parser:
    def __init__(self, lines):
        self.lines = lines
        self.i = 0

    def peek(self):
        return self.lines[self.i]

    def take(self):
        line = self.lines[self.i]
        self.i += 1
        return line

    def block(self, env, path):
        """Ops until a closing `}` line; returns (op hashes, return hashes)."""
        ops = []
        ret = None
        while True:
            line = self.peek()
            s = line.strip()
            if s.startswith("}"):
                return ops, ret
            self.take()
            m = OP.match(line)
            if not m:
                raise ValueError(f"unparsed: {line[:120]}")
            res, nres, name, operands, rest = m.groups()
            opnd = [env.get(r, "?" + r) for r in REF.findall(operands)]
            regions = []
            tail = rest
            if rest.strip() == "({":
                while True:
                    regions.append(self.region(env, path + f"/{len(regions)}"))
                    close = self.take().strip()
                    if close == "}, {":
                        continue
                    mm = re.match(r"^\}\)(.*)$", close)
                    if not mm:
                        raise ValueError(f"bad region close: {close[:80]}")
                    tail = mm.group(1)
                    break
            hv = h(name, ",".join(opnd), tail.strip(), "|".join(regions))
            if name in ("stablehlo.return", "func.return"):
                ret = h(name, ",".join(opnd), tail.strip())
            ops.append(hv)
            if res:
                if nres:
                    for j in range(int(nres)):
                        env[f"{res}#{j}"] = h(hv, str(j))
                else:
                    env[res] = hv

    def region(self, env, path):
        header = self.take().strip()
        mm = re.match(r"^\^bb\d+(?:\((.*)\))?:$", header)
        if not mm:
            raise ValueError(f"bad block header: {header[:80]}")
        args = mm.group(1) or ""
        for j, a in enumerate(re.findall(r"(%\w+): ([^,]+(?:<[^>]*>)?)", args)):
            env[a[0]] = h("barg", path, str(j), a[1])
        ops, ret = self.block(env, path)
        return h("region", ret or "", ",".join(sorted(ops)))


def canon(text):
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^func\.func (\w+) @(\w+)\((.*)\) -> \((.*)\) \{$", line)
        if not m:
            i += 1
            continue
        env = {}
        sig = []
        for j, (a, t) in enumerate(re.findall(r"(%arg\d+): ([^,{]+(?: \{[^}]*\})?)", m.group(3))):
            env[a] = h("arg", str(j), t.strip())
            sig.append(t.strip())
        p = Parser(lines[i + 1 :])
        ops, ret = p.block(env, m.group(2))
        out.append((m.group(2), sig, m.group(4), ret, collections.Counter(ops)))
        i += 1 + p.i + 1
    return out


def compare(a, b):
    ca, cb = canon(a), canon(b)
    if len(ca) != len(cb):
        return f"{len(ca)} functions against {len(cb)}"
    for fa, fb in zip(ca, cb):
        if fa[0] != fb[0]:
            return f"function {fa[0]} against {fb[0]}"
        if fa[1] != fb[1]:
            k = next(j for j, (x, y) in enumerate(zip(fa[1], fb[1])) if x != y) if len(fa[1]) == len(fb[1]) else -1
            return f"@{fa[0]}: signatures differ ({len(fa[1])} vs {len(fb[1])} args; first at {k})"
        if fa[2] != fb[2]:
            return f"@{fa[0]}: result types differ"
        if fa[3] != fb[3]:
            return f"@{fa[0]}: returns differ"
        if fa[4] != fb[4]:
            extra = fa[4] - fb[4]
            missing = fb[4] - fa[4]
            return f"@{fa[0]}: op multisets differ ({sum(extra.values())} only in base, {sum(missing.values())} only in candidate)"
    return None


def main():
    base, cand = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3:
        files = [os.path.basename(l.split(" and ")[0].split(" ", 1)[1]) for l in open(sys.argv[3]) if l.startswith("Files ")]
        only = [l for l in open(sys.argv[3]) if l.startswith("Only in")]
    else:
        files = sorted(set(os.listdir(base)) | set(os.listdir(cand)))
        only = []
    bad = 0
    for o in only:
        print("MISSING", o.strip())
        bad += 1
    eq = 0
    for f in files:
        a = open(os.path.join(base, f)).read()
        b = open(os.path.join(cand, f)).read()
        if a == b:
            continue
        try:
            why = compare(a, b)
        except ValueError as e:
            why = f"unparsed: {e}"
        if why:
            print(f"DIFFERENT {f}: {why}")
            bad += 1
        else:
            eq += 1
    print(f"canon: {eq} equivalent up to order, {bad} different")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
