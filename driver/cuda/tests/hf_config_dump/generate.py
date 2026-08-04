#!/usr/bin/env python3
"""Regenerates the two C++ files derived from `model/config.hpp`.

  * `dump_hf_config.cpp` — the oracle's emitter: every `HfConfig` field, as
    JSON. What the differential compares against.
  * `../../src/model/descriptor.cpp` — the read side: a `pie.model/1` document
    back into the same struct.

Both are generated, and from the same parse (`hfconfig.py`), so a field cannot
reach one and miss the other. Hand-maintaining 219 emissions across 7 structs
would not survive one new architecture, and a field missing from the oracle is
a field missing from every comparison — silently.

    python3 driver/cuda/tests/hf_config_dump/generate.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from hfconfig import SIMPLE, parse  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
HEADER = ROOT / "src" / "model" / "config.hpp"
DUMPER = pathlib.Path(__file__).with_name("dump_hf_config.cpp")
READER = ROOT / "src" / "model" / "descriptor.cpp"

# Deliberately absent from `pie.model/1`: it rounds `head_dim` up to a head dim
# the *driver build* instantiated, so it is a property of a binary rather than
# of a model. `parse_pie_model_descriptor` recomputes it.
DERIVED = {"head_dim_kernel"}


def emit(structs, struct, var):
    """The write side: every field of `struct`, as JSON."""
    lines = []
    for f in structs[struct]:
        if f.type in SIMPLE:
            lines.append(f'    j["{f.name}"] = {var}.{f.name};')
        elif f.type == "RopeScaling":
            lines.append(f'    j["{f.name}"] = rope_scaling_name({var}.{f.name});')
        elif f.type.startswith("std::optional<"):
            lines.append(
                f'    j["{f.name}"] = {var}.{f.name} ? dump(*{var}.{f.name})'
                f' : nlohmann::json(nullptr);'
            )
        elif f.type in structs:
            lines.append(f'    j["{f.name}"] = dump({var}.{f.name});')
        else:
            raise SystemExit(f"unhandled type {f.type!r} for {struct}.{f.name}")
    return "\n".join(lines)


def read_fields(structs, struct, var):
    """The read side: every field of `struct`, out of a flat document."""
    lines = []
    for f in structs[struct]:
        if f.name in DERIVED:
            continue
        if f.type in SIMPLE:
            lines.append(f'    {var}.{f.name} = need<{f.type}>(j, "{f.name}");')
        elif f.type == "RopeScaling":
            lines.append(
                f'    {var}.{f.name} = rope_scaling_of(need<std::string>(j, "{f.name}"));'
            )
        elif f.type.startswith("std::optional<"):
            inner = f.type[len("std::optional<"):-1]
            lines.append(
                f'    if (const auto it = j.find("{f.name}");'
                f' it != j.end() && !it->is_null()) {{'
            )
            lines.append(f'        {var}.{f.name} = read_{inner}(*it);')
            lines.append("    }")
        elif f.type in structs:
            lines.append(
                f'    {var}.{f.name} = read_{f.type}(need<nlohmann::json>(j, "{f.name}"));'
            )
        else:
            raise SystemExit(f"unhandled type {f.type!r} for {struct}.{f.name}")
    return "\n".join(lines)


def render_dumper(structs, order):
    parts = [PROLOGUE.rstrip("\n") + "\n"]
    for name in order:
        if name == "HfConfig":
            continue
        parts.append(
            f"nlohmann::json dump(const pie_cuda_driver::{name}& v) {{\n"
            f"    nlohmann::json j;\n{emit(structs, name, 'v')}\n    return j;\n}}\n"
        )
    parts.append(
        f"nlohmann::json dump(const HfConfig& v) {{\n"
        f"    nlohmann::json j;\n{emit(structs, 'HfConfig', 'v')}\n    return j;\n}}\n"
    )
    parts.append(EPILOGUE)
    return "\n".join(parts)


def render_reader(structs, order):
    parts = [READER_PROLOGUE.rstrip("\n") + "\n"]
    for name in order:
        if name == "HfConfig":
            continue
        parts.append(
            f"{name} read_{name}(const nlohmann::json& j) {{\n"
            f"    {name} v;\n{read_fields(structs, name, 'v')}\n    return v;\n}}\n"
        )
    parts.append(
        f"HfConfig read_config(const nlohmann::json& j) {{\n"
        f"    HfConfig v;\n{read_fields(structs, 'HfConfig', 'v')}\n    return v;\n}}\n"
    )
    parts.append(READER_EPILOGUE)
    return "\n".join(parts)


def main():
    structs, order = parse(HEADER.read_text())
    DUMPER.write_text(render_dumper(structs, order))
    READER.write_text(render_reader(structs, order))
    total = sum(len(f) for f in structs.values())
    print(f"{DUMPER.name}: {total} fields across {len(structs)} structs")
    print(f"{READER.relative_to(ROOT)}: same fields, read side")


PROLOGUE = r'''//===-- dump_hf_config.cpp - the C++ normalizer as a golden oracle -------===//
//
// `parse_hf_config` is ~850 lines of HuggingFace-quirk normalization with 25
// `model_type` conditionals, and `pie.model/1` moves that work to import time
// in Rust. Two implementations of the same normalization is exactly the skew
// the artifact exists to kill, so the Rust port is checked against this one
// field for field rather than against a reading of the source.
//
// This is a plain C++ translation unit: `config.cpp` includes only nlohmann,
// `hf_config_json.hpp` and `kernels_manifest.hpp`, and the last of those is
// deliberately CUDA-free. So the oracle builds with a host compiler and no
// toolkit -- see `build.sh`.
//
// GENERATED by `generate.py` from `model/config.hpp`. Regenerate rather than
// edit: a field added to `HfConfig` and forgotten here would silently drop out
// of the comparison, which is the one failure this file must not have.
//
//===----------------------------------------------------------------------===//
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

#include "model/config.hpp"

namespace {

using pie_cuda_driver::HfConfig;

const char* rope_scaling_name(HfConfig::RopeScaling kind) {
    switch (kind) {
        case HfConfig::RopeScaling::None: return "none";
        case HfConfig::RopeScaling::Llama3: return "llama3";
        case HfConfig::RopeScaling::OriginalYaRN: return "original_yarn";
    }
    return "unknown";
}

'''

EPILOGUE = r'''}  // namespace

#ifndef PIE_DUMP_NO_MAIN
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: dump_hf_config <config.json>\n";
        return 2;
    }
    try {
        std::cout << dump(pie_cuda_driver::parse_hf_config(argv[1])).dump(2) << "\n";
    } catch (const std::exception& err) {
        // A refusal is part of the behavior being compared, so it leaves on
        // stdout in a shape the harness can read rather than as a crash.
        std::cout << nlohmann::json{{"error", err.what()}}.dump(2) << "\n";
        return 1;
    }
    return 0;
}
#endif
'''

READER_PROLOGUE = r'''//===-- descriptor.cpp - reading the pie.model/1 descriptor --------------===//
//
// The replacement for `parse_hf_config`. Where that one dereferences nested
// configs, probes alternate spellings, applies per-`model_type` rules and
// resolves defaults -- ~850 lines with 25 conditionals -- this reads a flat
// document whose every field is already decided.
//
// That is the whole point of `pie.model/1`: the quirks are normalized once at
// import, in Rust, and what reaches a driver is a struct written out. There is
// nothing here to disagree with the normalizer about, because there is no
// judgement left in it.
//
// `head_dim_kernel` is the one field not in the descriptor and not readable
// from one: it rounds `head_dim` up to a head dim *this driver build*
// instantiated in kernels.def, so it is recomputed here rather than carried.
//
// GENERATED by `generate.py` from `model/config.hpp`. Regenerate rather than
// edit: a field added to `HfConfig` and forgotten here would read as zero.
//
//===----------------------------------------------------------------------===//
#include "model/descriptor.hpp"

#include <stdexcept>

#include <nlohmann/json.hpp>

#include "kernels_manifest.hpp"

namespace pie_cuda_driver {

namespace {

// The descriptor is exhaustive by construction, so a missing key is a bug in
// the writer rather than a case to default around. Reading it as zero would
// turn that bug into a model that loads and answers wrongly.
template <typename T>
T need(const nlohmann::json& j, const char* key) {
    auto it = j.find(key);
    if (it == j.end()) {
        throw std::runtime_error(
            std::string("pie.model/1 descriptor: missing key '") + key + "'");
    }
    return it->get<T>();
}

HfConfig::RopeScaling rope_scaling_of(const std::string& name) {
    if (name == "none") return HfConfig::RopeScaling::None;
    if (name == "llama3") return HfConfig::RopeScaling::Llama3;
    if (name == "original_yarn") return HfConfig::RopeScaling::OriginalYaRN;
    throw std::runtime_error("pie.model/1 descriptor: unknown rope scaling " + name);
}

'''

READER_EPILOGUE = r'''}  // namespace

HfConfig parse_pie_model_descriptor(const std::string& json) {
    const auto j = nlohmann::json::parse(json);
    const auto version = need<std::string>(j, "version");
    if (version != "pie.model/1") {
        throw std::runtime_error(
            "this artifact's model descriptor is " + version +
            ", and this build reads pie.model/1; regenerate it with "
            "`pie model convert --force`");
    }
    HfConfig cfg = read_config(j);
    // Not a fact about the checkpoint: the set of instantiated head dims is a
    // property of this build (kernels.def), so it is derived here.
    cfg.head_dim_kernel = round_up_attn_head_dim(cfg.head_dim);
    return cfg;
}

}  // namespace pie_cuda_driver
'''

if __name__ == "__main__":
    main()
