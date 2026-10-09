#!/usr/bin/env python3
"""Writes src/star/ops.rs: a Starlark binding for every op of src/ops/*.rs,
and every method of `Input` in src/forward.rs, whose parameters and result a
package can spell.

Run it, then `cargo fmt`, after adding or changing an op; the test
`every_op_a_package_can_spell_is_bound` fails until it has been run.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODULES = ["attn", "collective", "elemwise", "layout", "linear", "spatial"]

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
    "Option<u32>": ("Option<u32>", "{}"),
    "Option<f32>": ("Option<f32>", "{}"),
    "Option<&Value>": ("Option<Value>", "{}.as_ref()"),
    "Option<&Weight>": ("Option<Weight>", "{}.as_ref()"),
    "Option<(u32, f32)>": ("Option<(u32, f32)>", "{}"),
    "[u32; 3]": ("[u32; 3]", "{}"),
    "[u32; 4]": ("[u32; 4]", "{}"),
    "[f32; 4]": ("[f32; 4]", "{}"),
    "&[u64]": ("Vec<u64>", "&{}"),
    "&[Value]": ("Vec<Value>", "&{}"),
    "&[&Value]": ("Vec<Value>", "&{}.iter().collect::<Vec<_>>()"),
    "GateActivation": ("GateActivation", "{}"),
    "RopeForm": ("RopeForm", "{}"),
    "MropeForm": ("MropeForm", "{}"),
    "ModulateForm": ("ModulateForm", "{}"),
    "Option<Yarn>": ("Option<Yarn>", "{}"),
    "Conv": ("Conv", "{}"),
    "VoxelSegment": ("VoxelSegment", "{}"),
    "RaggedMask": ("RaggedMask", "{}"),
    "&str": ("String", "&{}"),
    "u8": ("u8", "{}"),
    "impl Into<u64>": ("u64", "{}"),
    # An op that records outside any value takes the rows whose trace it is.
    "&Recorder": ("Input", "{}.recorder()"),
}
RESULTS = {"Value", "(Value, Value)", "(Value, Value, Value)", "()", "RaggedMask", "ValueId"}

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
        r"^    pub fn (\w+)(<[^>]*>)?\(&self,?(.*?)\)\s*(?:->\s*([^{]*?))?\s*\{", re.S | re.M
    )
    for m in pattern.finditer(text[start:end]):
        yield "inputs", m.group(1), m.group(2), split(m.group(3)), (m.group(4) or "()").strip()


def entry(module, name, params, call, receiver):
    """The table row binding `call`, a template whose `{passes}` the row's
    unpacked parameters fill, but for a receiver the call names itself."""
    names = ", ".join(f'"{p}"' for p, _ in params)
    takes = "".join(f"            let {p}: {PARAMS[t][0]} = a.next()?;\n" for p, t in params)
    passes = ", ".join(PARAMS[t][1].format(p) for p, t in params[int(receiver):])
    return (
        "    Op {\n"
        f'        module: "{module}",\n'
        f'        name: "{name}",\n'
        f"        params: &[{names}],\n"
        "        call: |a, heap| {\n"
        f"{takes}"
        f'            result(heap, dsl("{module}.{name}", || {call.format(passes=passes)})?)\n'
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


def main():
    skipped = []
    bound = table(ops(), skipped, "ops::{module}::{name}({{passes}})")
    inputs = table(input_methods(), skipped, "inputs.{name}({{passes}})", receiver=True)
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
    (ROOT / "src/star/ops.rs").write_text(out)
    print(f"bound {len(bound)} ops and {len(inputs)} input methods; not bound: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
