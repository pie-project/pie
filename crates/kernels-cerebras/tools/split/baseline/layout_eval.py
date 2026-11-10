#!/usr/bin/env python3
"""layout_eval.py: what a role-split layout.csl places on every PE, by evaluating it.

`tiles(layout_csl, params)` runs the layout block of a layout.csl (the subset the role-split builds
write: params and consts, `for` over `@range`, `if` / `else`, `const` / `var` bindings, `@set_tile_code`)
with the cslc params `params` ({name: number}) and returns {(x, y): (source, ((param, value), ..))},
the params sorted and `memcpy_params` left out, plus the exported names. Two builds that place the
same source with the same params on every PE compile the same program, however their layouts are
written (a parametric layout and one with literal params, `emit::layout`).

`layout_eval.py <build dir> [<params K:V,..>]` prints a summary; compare.py uses `tiles` for build/."""
import re, sys

TOK = re.compile(r'\s*(?:(//[^\n]*)|(\d+\.\d*(?:[eE][-+]?\d+)?|\d+)|("[^"]*")|(@?[A-Za-z_][A-Za-z_0-9]*)|'
                 r'(==|!=|<=|>=|<<|>>|\.\{|[-+*/%<>!(){}\[\],;:=.|&^]))')


def lex(s):
    out, i = [], 0
    while i < len(s):
        m = TOK.match(s, i)
        if not m:
            if s[i:].strip() == "":
                break
            raise SyntaxError(f"at {s[i:i + 40]!r}")
        i = m.end()
        com, num, st, ident, op = m.groups()
        if com:
            continue
        if num:
            out.append(("num", float(num) if "." in num else int(num)))
        elif st:
            out.append(("str", st[1:-1]))
        elif ident:
            out.append(("id", ident))
        elif op:
            out.append(("op", op))
    return out


BIN = [("or",), ("and",), ("==", "!=", "<", "<=", ">", ">="), ("|",), ("^",), ("&",), ("<<", ">>"), ("+", "-"),
       ("*", "/", "%")]
PYOP = {"/": "//", "!": "not "}


class Gen:
    """Translates the layout subset to Python source (a parametric layout runs every PE through its
    conditions, too slow token by token)."""

    def __init__(self, toks):
        self.t, self.i = toks, 0
        self.lines = []
        self.params = []  # (name, default code or None)

    def peek(self, k=0):
        j = self.i + k
        return self.t[j] if j < len(self.t) else ("eof", None)

    def eat(self, v=None):
        tok = self.peek()
        if v is not None and tok[1] != v:
            raise SyntaxError(f"want {v!r}, got {tok} at token {self.i}")
        self.i += 1
        return tok

    def at(self, v):
        return self.peek()[1] == v

    def emit(self, ind, s):
        self.lines.append("    " * ind + s)

    def expr(self, lvl=0):
        if lvl == len(BIN):
            return self.unary()
        a = self.expr(lvl + 1)
        while self.peek()[1] in BIN[lvl]:
            op = self.eat()[1]
            b = self.expr(lvl + 1)
            if op == "/":
                a = f"_div({a}, {b})"
            else:
                a = f"({a} {op} {b})"
        return a

    def unary(self):
        if self.at("!"):
            self.eat()
            return f"(not {self.unary()})"
        if self.at("-"):
            self.eat()
            return f"(-{self.unary()})"
        return self.primary()

    def primary(self):
        k, v = self.eat()
        if k == "num":
            return repr(v)
        if k == "str":
            return repr(v)
        if k == "op" and v == "(":
            e = self.expr()
            self.eat(")")
            return f"({e})"
        if k == "op" and v == ".{":
            items = []
            while not self.at("}"):
                self.eat(".")
                n = self.eat()[1]
                self.eat("=")
                items.append(f"{n!r}: {self.expr()}")
                if self.at(","):
                    self.eat()
            self.eat("}")
            return "{" + ", ".join(items) + "}"
        if k == "id":
            if v == "true":
                return "True"
            if v == "false":
                return "False"
            if v == "if":
                self.eat("(")
                c = self.expr()
                self.eat(")")
                a = self.expr()
                self.eat("else")
                b = self.expr()
                return f"({a} if {c} else {b})"
            if v == "@as":
                self.eat("(")
                self.typ()
                self.eat(",")
                e = self.expr()
                self.eat(")")
                return e
            if v == "@import_module":
                self.skip_args()
                return "None"
            if v == "@export_name":
                self.eat("(")
                n = self.expr()
                self.eat(",")
                self.typ()
                while not self.at(")"):
                    self.eat()
                self.eat(")")
                return f"_export({n})"
            if v.startswith("@"):
                return f"_{v[1:]}({', '.join(self.args())})"
            if self.at("."):  # memcpy.get_params(x)
                self.eat(".")
                self.eat()
                self.args()
                return "None"
            return v
        raise SyntaxError(f"unexpected {k} {v!r}")

    def typ(self):
        depth = 0
        while True:
            v = self.peek()[1]
            if depth == 0 and v in (",", ")", "=", ";"):
                return
            if v in ("(", "["):
                depth += 1
            if v in (")", "]"):
                depth -= 1
            self.eat()

    def args(self):
        self.eat("(")
        out = []
        while not self.at(")"):
            out.append(self.expr())
            if self.at(","):
                self.eat()
        self.eat(")")
        return out

    def skip_args(self):
        depth = 0
        while True:
            v = self.eat()[1]
            if v in ("(", ".{", "{"):
                depth += 1
            elif v in (")", "}"):
                depth -= 1
                if depth == 0:
                    return

    def block(self, ind):
        self.eat("{")
        n = len(self.lines)
        while not self.at("}"):
            self.stmt(ind)
        self.eat("}")
        if len(self.lines) == n:
            self.emit(ind, "pass")

    def stmt(self, ind):
        k, v = self.peek()
        if v == "for":
            self.eat()
            self.eat("(")
            self.eat("@range")
            a = self.args()[1:]
            self.eat(")")
            self.eat("|")
            n = self.eat()[1]
            self.eat("|")
            self.emit(ind, f"for {n} in range({', '.join(a)}):")
            self.block(ind + 1)
            return
        if v == "if":
            kw = "if"
            while True:
                self.eat("if")
                self.eat("(")
                c = self.expr()
                self.eat(")")
                self.emit(ind, f"{kw} {c}:")
                self.block(ind + 1)
                if self.at("else"):
                    self.eat()
                    if self.at("if"):
                        kw = "elif"
                        continue
                    self.emit(ind, "else:")
                    self.block(ind + 1)
                return
        if v in ("const", "var"):
            self.eat()
            n = self.eat()[1]
            if self.at(":"):
                self.eat()
                self.typ()
            self.eat("=")
            self.emit(ind, f"{n} = {self.expr()}")
            self.eat(";")
            return
        if k == "id" and self.peek(1)[1] == "=":
            self.eat()
            self.eat("=")
            self.emit(ind, f"{v} = {self.expr()}")
            self.eat(";")
            return
        self.emit(ind, self.expr())
        self.eat(";")

    def top(self):
        while self.peek()[0] != "eof":
            v = self.peek()[1]
            if v == "param":
                self.eat()
                n = self.eat()[1]
                self.eat(":")
                self.typ()
                d = None
                if self.at("="):
                    self.eat()
                    d = self.expr()
                self.eat(";")
                self.emit(0, f"{n} = _P[{n!r}] if {n!r} in _P else {d if d is not None else '_need(' + repr(n) + ')'}")
            elif v == "layout":
                self.eat()
                self.block(0)
            else:
                self.stmt(0)


def _div(a, b):
    return a // b if isinstance(a, int) and isinstance(b, int) else a / b


def tiles(text, params):
    g = Gen(lex(text))
    g.top()
    code = "\n".join(g.lines)
    out, exports = {}, []

    def set_tile_code(x, y, src, ps=None):
        ps = {k: (int(v) if isinstance(v, bool) else v) for k, v in (ps or {}).items() if k != "memcpy_params"}
        if (x, y) in out:
            raise ValueError(f"PE {x},{y} set twice")
        out[(x, y)] = (src, tuple(sorted(ps.items())))

    def need(n):
        raise ValueError(f"param {n}: no value")

    env = {"_P": dict(params), "_div": _div, "_set_tile_code": set_tile_code, "_set_rectangle": lambda *a: None,
           "_export": exports.append, "_need": need}
    exec(compile(code, "<layout>", "exec"), env)
    return out, sorted(exports)


def number(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def parse_params(s):
    return {k: number(v) for k, v in (kv.split(":") for kv in filter(None, s.strip().split(",")))}


if __name__ == "__main__":
    import os
    d = sys.argv[1]
    p = parse_params(sys.argv[2] if len(sys.argv) > 2 else
                     (open(os.path.join(d, "params.used")).read() if os.path.exists(os.path.join(d, "params.used")) else ""))
    t, ex = tiles(open(os.path.join(d, "layout.csl")).read(), p)
    from collections import Counter
    print(len(t), "PEs", dict(Counter(s for s, _ in t.values())), len(ex), "exports")
