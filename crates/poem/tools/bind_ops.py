#!/usr/bin/env python3
"""Writes src/star/ops.rs: a Starlark binding for every op of src/ops/*.rs,
and every method of `Input` in src/forward.rs, whose parameters and result a
package can spell.

Run it after adding or changing an op; it formats what it writes. With
`--check` it writes nothing and fails if the checked-in file is not what it
would write (the test `the_bindings_are_what_bind_ops_writes` runs this).
Bindings it cannot spell live by hand in `src/star/ops_by_hand.rs`, each
named in `BY_HAND`.
"""

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULES = ["attn", "collective", "conv", "elemwise", "gather", "linear", "sampling", "ssm"]

# Rust parameter type -> (the type it is unpacked as, how it is passed on)
PARAMS = {
    "&Value": ("Value", "&{}"),
    "&Weight": ("Weight", "&{}"),
    "ValueId": ("ValueId", "{}"),
    "Option<ValueId>": ("Option<ValueId>", "{}"),
    "&Input": ("Input", "&{}"),
    "u32": ("u32", "{}"),
    "f32": ("f32", "{}"),
    "bool": ("bool", "{}"),
    "Dtype": ("Dtype", "{}"),
    "InputAxis": ("poem_ir::InputAxis", "{}"),
    "Option<u32>": ("Option<u32>", "{}"),
    "Option<f32>": ("Option<f32>", "{}"),
    "Option<bool>": ("Option<bool>", "{}"),
    "Option<&Value>": ("Option<Value>", "{}.as_ref()"),
    "Option<&Weight>": ("Option<Weight>", "{}.as_ref()"),
    "Option<&Predicate>": ("Option<Predicate>", "{}.as_ref()"),
    "Option<(u32, f32)>": ("Option<(u32, f32)>", "{}"),
    "[u32; 3]": ("[u32; 3]", "{}"),
    "Option<poem_ir::RowCount>": ("Option<poem_ir::RowCount>", "{}"),
    "[u32; 4]": ("[u32; 4]", "{}"),
    "[f32; 4]": ("[f32; 4]", "{}"),
    "[f32; 3]": ("[f32; 3]", "{}"),
    "&[u64]": ("Vec<u64>", "&{}"),
    "&[Value]": ("Vec<Value>", "&{}"),
    "&[&Value]": ("Vec<Value>", "&{}.iter().collect::<Vec<_>>()"),
    "GateActivation": ("GateActivation", "{}"),
    "RopeForm": ("RopeForm", "{}"),
    "MropeForm": ("MropeForm", "{}"),
    "PixelOrder": ("PixelOrder", "{}"),
    "TapKind": ("TapKind", "{}"),
    "ModulateForm": ("ModulateForm", "{}"),
    "Option<Yarn>": ("Option<Yarn>", "{}"),
    "Option<RopeScaling>": ("Option<RopeScaling>", "{}"),
    "Turn": ("Turn", "{}"),
    "GluAct": ("GluAct", "{}"),
    "RouteScore": ("RouteScore", "{}"),
    "Conv": ("Conv", "{}"),
    "VoxelSegment": ("VoxelSegment", "{}"),
    "RaggedMask": ("RaggedMask", "{}"),
    "&str": ("String", "&{}"),
    "u8": ("u8", "{}"),
    "impl Into<u64>": ("u64", "{}"),
    # An op that records outside any value takes the rows whose trace it is.
    "&Recorder": ("Input", "{}.recorder()"),
    # A ranking a package may leave out: the indexer's own.
    "IndexScore": ("Option<IndexScore>", "{}.unwrap_or_default()"),
}
# Parameters a package may leave out, by name, and what they then pass.
DEFAULTED = {"per_head": ("bool", "false"), "prefill_only": ("bool", "false"), "mask": ("bool", "false"), "scores": ("bool", "false"), "whole": ("bool", "false"), "lse": ("bool", "false"), "causal": ("bool", "true")}
RESULTS = {"Value", "(Value, Value)", "(Value, Option<Value>)", "(Value, Value, Value)", "()", "RaggedMask", "ValueId"}

# Bound by hand in `src/star/ops_by_hand.rs`, not here: their parameters or
# results are of types this does not spell, or their packages' spelling keeps
# an order or a default the op's own signature does not.
BY_HAND = {
    "conv::conv3d",
    "conv::upsample_nearest",
    "elemwise::norm_gate",
    "gather::gather_rows",
    "gather::split_qkv_padded",
    "inputs::readout_attention",
    "inputs::route_hints",
    "ssm::causal_conv1d",
    "ssm::causal_conv1d_chunked",
    "ssm::causal_conv1d_chunked_dilated",
    "ssm::causal_conv1d_dilated",
}

# The `Input` methods this does not bind: `on` and `partition` are bound by
# hand in `src/star/forward.rs`, `recorder` and `walk_layers` not at all.
NOT_GENERATED = ["recorder", "on", "partition", "walk_layers"]


def split(params):
    depth, cur, out = 0, "", []
    for ch in params:
        if ch in "<([":
            depth += 1
        if ch in ">)]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return [tuple(" ".join(p.split()).split(": ", 1)) for p in out]


def ops():
    pattern = re.compile(r"^pub fn (\w+)(<[^>]*>)?\((.*?)\)\s*(?:->\s*([^{]*?))?\s*\{", re.S | re.M)
    for module in MODULES:
        text = (ROOT / "src/ops" / f"{module}.rs").read_text()
        for m in pattern.finditer(text):
            yield module, m.group(1), m.group(2), split(m.group(3)), (m.group(4) or "()").strip()


def input_methods():
    text = (ROOT / "src/forward.rs").read_text()
    start = text.index("impl Input {")
    depth = 0
    for end, ch in enumerate(text[start:], start):
        depth += {"{": 1, "}": -1}.get(ch, 0)
        if depth == 0 and ch == "}":
            break
    pattern = re.compile(
        r"^    pub fn (\w+)(<[^>]*>)?\(\s*&self,?(.*?)\)\s*(?:->\s*([^{]*?))?\s*\{", re.S | re.M
    )
    for m in pattern.finditer(text[start:end]):
        yield "inputs", m.group(1), m.group(2), split(m.group(3)), (m.group(4) or "()").strip()


def entry(module, name, params, call, receiver):
    """The table row binding `call`, a template whose `{passes}` the row's
    unpacked parameters fill, but for a receiver the call names itself."""
    names = ", ".join(f'"{p}"' for p, _ in params)
    def unpacked(p, t):
        if p in DEFAULTED and DEFAULTED[p][0] == t:
            return f"Option<{t}>"
        return PARAMS[t][0]

    def passed(p, t):
        if p in DEFAULTED and DEFAULTED[p][0] == t:
            return f"{p}.unwrap_or({DEFAULTED[p][1]})"
        return PARAMS[t][1].format(p)

    takes = "".join(f"            let {p}: {unpacked(p, t)} = a.next()?;\n" for p, t in params)
    passes = ", ".join(passed(p, t) for p, t in params[int(receiver):])
    # A Rust keyword's trailing `_` is not the package's spelling.
    spelled = name.rstrip("_")
    return (
        "    Op {\n"
        f'        module: "{module}",\n'
        f'        name: "{spelled}",\n'
        f"        params: &[{names}],\n"
        "        call: |a, heap| {\n"
        f"{takes}"
        f'            result(heap, dsl("{module}.{spelled}", || {call.format(passes=passes)})?)\n'
        "        },\n"
        "    },\n"
    )


def table(rows, skipped, call, receiver=False):
    out = []
    for module, name, generic, params, result in rows:
        if receiver and name in NOT_GENERATED:
            continue
        if generic or any(t not in PARAMS for _, t in params) or result not in RESULTS:
            skipped.append(f"{module}::{name}")
            continue
        if receiver:
            params = [("inputs", "&Input")] + params
        out.append(entry(module, name, params, call.format(module=module, name=name), receiver))
    return out


def by_hand(rows):
    return (r for r in rows if f"{r[0]}::{r[1]}" not in BY_HAND)


def main():
    skipped = []
    bound = table(by_hand(ops()), skipped, "ops::{module}::{name}({{passes}})")
    inputs = table(by_hand(input_methods()), skipped, "inputs.{name}({{passes}})", receiver=True)
    out = (
        "//! Generated by `tools/bind_ops.py` from `src/ops/*.rs` and `src/forward.rs`;\n"
        "//! do not edit.\n"
        "//!\n"
        "//! Every op and `Input` method a package can spell the arguments and\n"
        "//! result of. Not bound:\n"
        + "".join(f"//! `{s}`\n" for s in skipped)
        + "\n"
        "use crate::ops;\n"
        "use crate::star::bind::spelled::*;\n"
        "use crate::star::bind::{Op, result};\n"
        "use crate::star::forward::dsl;\n\n"
        f"pub(crate) const MODULES: &[&str] = &[{', '.join(repr(m).replace(chr(39), chr(34)) for m in MODULES)}];\n\n"
        "#[cfg(test)]\n"
        f"pub(crate) const NOT_GENERATED: &[&str] = &[{', '.join(repr(m).replace(chr(39), chr(34)) for m in NOT_GENERATED)}];\n\n"
        "pub(crate) static OPS: &[Op] = &[\n" + "".join(bound) + "];\n\n"
        "pub(crate) static INPUTS: &[Op] = &[\n" + "".join(inputs) + "];\n"
    )
    out = subprocess.run(
        ["rustfmt", "--edition", "2024", "--emit", "stdout"],
        input=out, capture_output=True, text=True, check=True, cwd=ROOT,
    ).stdout
    target = ROOT / "src/star/ops.rs"
    if "--check" in sys.argv[1:]:
        if target.read_text() != out:
            sys.exit(f"{target} is not what tools/bind_ops.py writes; run it")
        return
    target.write_text(out)
    print(f"bound {len(bound)} ops and {len(inputs)} input methods; not bound: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
