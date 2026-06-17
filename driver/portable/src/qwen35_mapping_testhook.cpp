// Test-only entry points exercising the GGUF→HF tensor-name mapper and the
// GGUF hparams parser for the hybrid Qwen 3.5 / 3.6 ("qwen35" / "qwen35moe")
// architecture, in isolation — no model file, no backend, no compute.
//
// Linked into `pie_driver_portable_lib` and called from `#[cfg(test)]` Rust
// tests in pie-server (`cargo test -p pie-server --bin pie`, which runs in
// CI; the driver/portable ctest targets do not — see server/build.rs, which
// only builds the `pie_driver_portable_lib` target). In a release build no
// Rust code references these symbols, so the linker garbage-collects this
// object out of the final binary.
//
// Why this exists: the qwen35 load path is otherwise guarded only by real
// 27B/35B model boots (#694). These hooks give a cheap regression guard for
// the two pure pieces that path depends on:
//
//   1. `gguf_to_hf_name` — every GGUF tensor that `Model::build_qwen3_5_` /
//      `Model::load_qwen3_5_moe_layer_` look up by HF name must be reachable
//      through the mapper. Dropping a mapper arm (e.g. the `ssm_*` linear-
//      attention or `*_shexp` shared-expert entries) silently strands those
//      weights at load; this hook turns that into a unit failure.
//   2. `parse_gguf_hparams` — the per-layer attention pattern (`layer_types`)
//      derived from `qwen35.full_attention_interval`, and the linear-attn
//      dims read from the `ssm.*` keys.

#include <cstdio>
#include <string>
#include <vector>

#include "gguf_archive.hpp"   // gguf_to_hf_name, GgufMeta
#include "gguf_hparams.hpp"   // parse_gguf_hparams
#include "hf_config.hpp"      // Hparams

namespace {

using pie_portable_driver::GgufMeta;

// The qwen35-UNIQUE HF-canonical names `build_qwen3_5_` /
// `load_qwen3_5_moe_layer_` (GGUF path) declare — the gated-delta-rule linear
// attention and the always-active shared expert. These are the only mapper
// arms no other GGUF arch exercises: the shared arms (self_attn.*_proj +
// q/k_norm, dense mlp.gate/up/down_proj, MoE router mlp.gate + routed
// mlp.experts.*, layernorms, top-level embed/norm/lm_head) are covered
// broadly by every other model's load (qwen3 / llama / mistral / qwen3-moe),
// so re-asserting them here adds no non-redundant coverage.
//
// This list is the qwen35-unique loader SPEC — it mirrors the linear-attn +
// shared-expert declarations in model.cpp and must be updated whenever those
// builders declare a new unique tensor. Kept independent of the mapper so the
// coverage check is not tautological: it asserts the mapper can PRODUCE every
// unique name the loader CONSUMES.
const std::vector<std::string>& required_layer_hf() {
    static const std::vector<std::string> v = {
        // Linear-attention layer ('l') — gated delta rule.
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.A_log",
        "linear_attn.dt_bias",
        "linear_attn.norm.weight",
        "linear_attn.out_proj.weight",
        "linear_attn.conv1d.weight",
        // Always-active shared expert (qwen35moe).
        "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight",
        "mlp.shared_expert.down_proj.weight",
        "mlp.shared_expert_gate.weight",
    };
    return v;
}

// The GGUF tensor names (llama.cpp qwen35 convention) that map to the
// qwen35-unique HF names above. The mapper is fed exactly these; the produced
// HF names must cover the loader spec. Shared GGUF suffixes (attn_*, dense
// ffn_*, routed ffn_*_exps, norms, top level) are omitted — their mapping is
// exercised by other arches' loads.
const std::vector<std::string>& qwen35_gguf_layer_suffixes() {
    static const std::vector<std::string> v = {
        // Linear-attn delta-net inputs.
        "attn_qkv.weight", "attn_gate.weight",
        "ssm_alpha.weight", "ssm_beta.weight", "ssm_a",
        "ssm_dt.bias", "ssm_conv1d.weight", "ssm_norm.weight", "ssm_out.weight",
        // Shared-expert inputs.
        "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
        "ffn_gate_inp_shexp.weight",
    };
    return v;
}

bool produced_contains(const std::vector<std::string>& produced,
                       const std::string& want) {
    for (const auto& p : produced) {
        if (p == want) return true;
    }
    return false;
}

void add_kv(GgufMeta& m, const std::string& key, double num) {
    GgufMeta::KV kv;
    kv.key = key;
    kv.type = 0;
    kv.num_value = num;
    m.kv.emplace(key, std::move(kv));
}

// Synthetic qwen35moe metadata. `interval` and `num_layers` drive the
// layer_types derivation; the ssm.* / expert_* values are fixed knowns the
// linear-dim hook asserts against.
GgufMeta make_qwen35_meta(int interval, int num_layers) {
    GgufMeta m;
    m.general_architecture = "qwen35";
    m.has_output_weight = true;
    add_kv(m, "qwen35.block_count", num_layers);
    add_kv(m, "qwen35.full_attention_interval", interval);
    // Linear-attention dims (mapped from ssm.* in gguf_hparams.cpp).
    add_kv(m, "qwen35.ssm.group_count", 2);     // -> num K heads
    add_kv(m, "qwen35.ssm.time_step_rank", 16);  // -> num V heads
    add_kv(m, "qwen35.ssm.state_size", 128);     // -> K/V head dim
    add_kv(m, "qwen35.ssm.conv_kernel", 4);      // -> conv width
    // Shared expert + routed experts (qwen35moe).
    add_kv(m, "qwen35.expert_count", 8);
    add_kv(m, "qwen35.expert_shared_feed_forward_length", 512);
    return m;
}

}  // namespace

extern "C" {

// Returns the number of qwen35-unique loader-required HF names the mapper
// FAILS to produce from the qwen35-unique GGUF tensor set — 0 means full
// coverage. Missing names are written to stderr so a failing Rust assert is
// diagnosable.
int pie_portable_test_qwen35_mapper_missing_count() {
    using pie_portable_driver::gguf_to_hf_name;

    // Produce the HF set from the qwen35-unique GGUF vocabulary (layer 0).
    // qwen35 is not a gemma arch, so the gemma `ffn_norm` layout is off.
    std::vector<std::string> produced;
    for (const auto& s : qwen35_gguf_layer_suffixes()) {
        const std::string hf = gguf_to_hf_name("blk.0." + s, /*gemma_norm_layout=*/false);
        if (!hf.empty()) produced.push_back(hf);
    }

    int missing = 0;
    for (const auto& need : required_layer_hf()) {
        if (!produced_contains(produced, "model.layers.0." + need)) {
            std::fprintf(stderr,
                         "[qwen35-mapper-test] uncovered layer tensor: %s\n",
                         need.c_str());
            ++missing;
        }
    }
    return missing;
}

// Returns layer_types[idx] (the char 'g' or 'l', as an int) for a qwen35
// model with the given full-attention `interval` and `num_layers`, or a
// negative error code on bad input / parse shape.
int pie_portable_test_qwen35_layer_type_at(int interval, int num_layers,
                                           int idx) {
    if (interval <= 0 || num_layers <= 0 || idx < 0 || idx >= num_layers) {
        return -1;
    }
    const auto h = pie_portable_driver::parse_gguf_hparams(
        make_qwen35_meta(interval, num_layers));
    if (static_cast<int>(h.layer_types.size()) != num_layers) return -2;
    return static_cast<int>(h.layer_types[static_cast<std::size_t>(idx)]);
}

// Returns one parsed linear-attention / shared-expert dim, selected by
// `which`, from the synthetic qwen35moe metadata. Lets the Rust test assert
// the ssm.* / expert_* keys flow into the right Hparams fields.
//   0: num K heads   1: num V heads   2: K/V head dim
//   3: conv kernel   4: shared-expert intermediate size
int pie_portable_test_qwen35_parsed_linear_dim(int which) {
    const auto h = pie_portable_driver::parse_gguf_hparams(
        make_qwen35_meta(/*interval=*/4, /*num_layers=*/8));
    switch (which) {
        case 0: return h.qwen35_linear_num_k_heads;
        case 1: return h.qwen35_linear_num_v_heads;
        case 2: return h.qwen35_linear_k_head_dim;
        case 3: return h.qwen35_linear_conv_kernel;
        case 4: return h.shared_expert_intermediate_size;
        default: return -1;
    }
}

}  // extern "C"
