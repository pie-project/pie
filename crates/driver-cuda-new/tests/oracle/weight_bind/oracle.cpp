// Differential oracle for the llama-like weight binders.
//
// The subject is `bind_llama_like` / `bind_phi3` / `bind_olmo3` in
// `driver-cuda/csrc/src/model/llama_like/qwen3.cpp`, compiled as shipped
// against a stub LoadedModel. Nothing about the binders is replaced.
//
// # What is actually being pinned
//
// Two things, and the second is the one that can go wrong quietly.
//
// 1. **The names.** Every weight is fetched by a string built from a per-layer
//    prefix. Getting one wrong throws at load with the name in the message, so
//    this half is loud — but the transcript records the probe order too,
//    because when a name is present under two spellings the probe order is
//    what decides which one binds. OLMo-3 is exactly that case: it reads
//    `post_attention_layernorm` into `attn_norm`, the slot every other
//    architecture fills from `input_layernorm`.
//
// 2. **The conditionals.** `attention_bias`, `use_qk_norm` and
//    `tie_word_embeddings` each decide whether a pointer gets filled or stays
//    null, and a null pointer here is not an error — the forward path reads it
//    as "this architecture doesn't have that", and skips a bias add or an
//    RMSNorm. A port that inverted `use_qk_norm` would load, run, and produce
//    subtly wrong logits. So the transcript reports every slot of every layer
//    including the null ones, across a grid of configs where each flag is
//    varied independently.
//
// The fused-projection slots are the third case: absent means "the contract
// declined to fuse this group", and the forward path has a different code path
// for each. So the grid varies their presence too.

#include <cstdint>
#include <cstdio>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "model/llama_like/qwen3.hpp"

// The CUDA surface `tensor.cpp` names. The binders reach none of it — every
// tensor here is default-constructed and non-owning — but the linker still
// wants the symbols.
cudaError_t cudaMalloc(void**, std::size_t) { return cudaSuccess; }
cudaError_t cudaFree(void*) { return cudaSuccess; }
cudaError_t cudaGetLastError() { return cudaSuccess; }
const char* cudaGetErrorString(cudaError_t) { return "stub"; }

namespace {

using pie_cuda_driver::DeviceTensor;
using pie_cuda_driver::DType;
using pie_cuda_driver::QuantMeta;
using pie_cuda_driver::model::LoadedModel;
using pie_cuda_driver::model::Qwen3Weights;

constexpr char SEP = '\x1f';

void row(const std::string& s) { std::printf("%s\n", s.c_str()); }

std::string layer_prefix(int i) {
    return "model.layers." + std::to_string(i) + ".";
}

// Populate an engine with the canonical llama-like tensor set.
void populate_llama_like(LoadedModel& e,
                         int layers,
                         bool lm_head,
                         bool bias,
                         bool qk_norm,
                         bool qkv_fused,
                         bool gate_up_fused) {
    e.cfg.num_hidden_layers = layers;
    e.add("model.embed_tokens.weight");
    e.add("model.norm.weight");
    if (lm_head) e.add("lm_head.weight");
    for (int i = 0; i < layers; ++i) {
        const std::string p = layer_prefix(i);
        e.add(p + "input_layernorm.weight");
        e.add(p + "post_attention_layernorm.weight");
        e.add(p + "self_attn.q_proj.weight");
        e.add(p + "self_attn.k_proj.weight");
        e.add(p + "self_attn.v_proj.weight");
        e.add(p + "self_attn.o_proj.weight");
        e.add(p + "mlp.gate_proj.weight");
        e.add(p + "mlp.up_proj.weight");
        e.add(p + "mlp.down_proj.weight");
        if (bias) {
            e.add(p + "self_attn.q_proj.bias");
            e.add(p + "self_attn.k_proj.bias");
            e.add(p + "self_attn.v_proj.bias");
        }
        if (qk_norm) {
            e.add(p + "self_attn.q_norm.weight");
            e.add(p + "self_attn.k_norm.weight");
        }
        if (qkv_fused) e.add(p + "self_attn.qkv_proj.fused.weight");
        if (gate_up_fused) e.add(p + "mlp.gate_up_proj.fused.weight");
    }
}

void populate_olmo3(LoadedModel& e, int layers, bool lm_head, bool bias) {
    e.cfg.num_hidden_layers = layers;
    e.add("model.embed_tokens.weight");
    e.add("model.norm.weight");
    if (lm_head) e.add("lm_head.weight");
    for (int i = 0; i < layers; ++i) {
        const std::string p = layer_prefix(i);
        // No input_layernorm: OLMo-3 is post-norm.
        e.add(p + "post_attention_layernorm.weight");
        e.add(p + "post_feedforward_layernorm.weight");
        e.add(p + "self_attn.q_proj.weight");
        e.add(p + "self_attn.k_proj.weight");
        e.add(p + "self_attn.v_proj.weight");
        e.add(p + "self_attn.o_proj.weight");
        e.add(p + "self_attn.q_norm.weight");
        e.add(p + "self_attn.k_norm.weight");
        e.add(p + "mlp.gate_proj.weight");
        e.add(p + "mlp.up_proj.weight");
        e.add(p + "mlp.down_proj.weight");
        if (bias) {
            e.add(p + "self_attn.q_proj.bias");
            e.add(p + "self_attn.k_proj.bias");
            e.add(p + "self_attn.v_proj.bias");
        }
    }
}

// Dump every slot, including the nulls: a null here is a decision, not an
// absence.
void dump_weights(const std::string& label,
                  const LoadedModel& e,
                  const Qwen3Weights& w) {
    row("top" + std::string(1, SEP) + label + SEP + "embed" + SEP +
        e.name_of(w.embed));
    row("top" + std::string(1, SEP) + label + SEP + "final_norm" + SEP +
        e.name_of(w.final_norm));
    row("top" + std::string(1, SEP) + label + SEP + "lm_head" + SEP +
        e.name_of(w.lm_head));
    // Whether lm_head aliases embed is the whole point of the
    // tie_word_embeddings branch, and a name comparison cannot see it when
    // both resolve to the same name — so say it directly.
    row("top" + std::string(1, SEP) + label + SEP + "lm_head_aliases_embed" +
        SEP + (w.lm_head == w.embed ? "1" : "0"));
    row("top" + std::string(1, SEP) + label + SEP + "num_layers" + SEP +
        std::to_string(w.layers.size()));

    for (std::size_t i = 0; i < w.layers.size(); ++i) {
        const auto& L = w.layers[i];
        const std::string pre =
            "layer" + std::string(1, SEP) + label + SEP + std::to_string(i) + SEP;
        const std::pair<const char*, const DeviceTensor*> slots[] = {
            {"attn_norm", L.attn_norm},
            {"mlp_norm", L.mlp_norm},
            {"q_proj", L.q_proj},
            {"k_proj", L.k_proj},
            {"v_proj", L.v_proj},
            {"o_proj", L.o_proj},
            {"q_bias", L.q_bias},
            {"k_bias", L.k_bias},
            {"v_bias", L.v_bias},
            {"q_norm", L.q_norm},
            {"k_norm", L.k_norm},
            {"gate_proj", L.gate_proj},
            {"up_proj", L.up_proj},
            {"down_proj", L.down_proj},
            {"qkv_proj_fused", L.qkv_proj_fused},
            {"gate_up_proj_fused", L.gate_up_proj_fused},
        };
        for (const auto& [name, t] : slots) {
            row(pre + name + SEP + e.name_of(t));
        }
        const std::pair<const char*, const std::optional<QuantMeta>*> quants[] = {
            {"q_proj_quant", &L.q_proj_quant},
            {"k_proj_quant", &L.k_proj_quant},
            {"v_proj_quant", &L.v_proj_quant},
            {"o_proj_quant", &L.o_proj_quant},
            {"gate_proj_quant", &L.gate_proj_quant},
            {"up_proj_quant", &L.up_proj_quant},
            {"down_proj_quant", &L.down_proj_quant},
        };
        for (const auto& [name, q] : quants) {
            std::string v = "none";
            if (q->has_value()) {
                v = "kind=" + std::to_string(static_cast<int>((*q)->kind)) +
                    ",gs=" + std::to_string((*q)->group_size) +
                    ",axis=" + std::to_string((*q)->channel_axis);
            }
            row(pre + name + SEP + v);
        }
    }
}

// The probe log, deduplicated to the per-layer shape. The binder asks the same
// questions of every layer, so recording all of them would make the transcript
// grow with the layer count without saying anything new; recording layer 0's
// questions plus the total count says both.
void dump_probes(const std::string& label, const LoadedModel& e) {
    row("probe_count" + std::string(1, SEP) + label + SEP +
        std::to_string(e.probes.size()));
    for (const auto& p : e.probes) {
        // Only the questions about layer 0 and the top-level ones.
        if (p.rfind("model.layers.", 0) == 0 &&
            p.rfind("model.layers.0.", 0) != 0) {
            continue;
        }
        row("probe" + std::string(1, SEP) + label + SEP + p);
    }
}

std::string run(const std::string& label,
                LoadedModel& e,
                Qwen3Weights (*bind)(const LoadedModel&)) {
    try {
        Qwen3Weights w = bind(e);
        dump_weights(label, e, w);
        dump_probes(label, e);
        return "ok";
    } catch (const std::exception& ex) {
        row("throw" + std::string(1, SEP) + label + SEP + ex.what());
        dump_probes(label, e);
        return ex.what();
    }
}

Qwen3Weights bind_ll(const LoadedModel& e) {
    return pie_cuda_driver::model::bind_llama_like(e, false);
}

// Script 1 — the config grid. Each flag is varied independently so a port
// that wired two of them together would show up.
void script_config_grid() {
    struct Case {
        const char* label;
        bool have_lm_head;
        bool tie;
        bool bias;
        bool qk_norm;
    };
    const Case cases[] = {
        {"base", true, false, false, false},
        {"tied_no_head", false, true, false, false},
        {"untied_no_head", false, false, false, false},  // throws
        {"head_and_tie", true, true, false, false},
        {"bias", true, false, true, false},
        {"qk_norm", true, false, false, true},
        {"bias_and_qk_norm", true, false, true, true},
        // The flag is set but the tensors are absent: the binder must throw
        // rather than silently leave the slot null, because the forward path
        // reads null as "this architecture has no q_norm".
        {"qk_norm_flag_without_tensors", true, false, false, false},
    };
    for (const auto& c : cases) {
        LoadedModel e;
        e.cfg.tie_word_embeddings = c.tie;
        e.cfg.attention_bias = c.bias;
        e.cfg.use_qk_norm = c.qk_norm;
        const bool last = std::string(c.label) == "qk_norm_flag_without_tensors";
        populate_llama_like(e, 2, c.have_lm_head, c.bias, c.qk_norm, false, false);
        if (last) e.cfg.use_qk_norm = true;  // tensors were not added
        run(c.label, e, bind_ll);
    }
}

// Script 2 — the fused-projection slots, which are optional in a different
// way: absent means the contract declined to fuse, not that the model lacks
// the weights.
void script_fusion() {
    const std::pair<const char*, std::pair<bool, bool>> cases[] = {
        {"fused_neither", {false, false}},
        {"fused_qkv_only", {true, false}},
        {"fused_gate_up_only", {false, true}},
        {"fused_both", {true, true}},
    };
    for (const auto& [label, f] : cases) {
        LoadedModel e;
        populate_llama_like(e, 1, true, false, false, f.first, f.second);
        run(label, e, bind_ll);
    }
}

// Script 3 — the quant side-map. The binder pulls seven entries per layer, and
// which weight name each one is keyed on is the thing worth pinning: a swap
// between `gate_proj` and `up_proj` would hand each GEMM the other's scales.
void script_quant_sidemap() {
    LoadedModel e;
    populate_llama_like(e, 1, true, false, false, false, false);
    const std::pair<const char*, int> entries[] = {
        {"self_attn.q_proj.weight", 11},
        {"self_attn.k_proj.weight", 12},
        {"self_attn.v_proj.weight", 13},
        {"self_attn.o_proj.weight", 14},
        {"mlp.gate_proj.weight", 15},
        {"mlp.up_proj.weight", 16},
        {"mlp.down_proj.weight", 17},
    };
    for (const auto& [suffix, tag] : entries) {
        QuantMeta m;
        m.kind = QuantMeta::Kind::PerGroup;
        m.group_size = tag;
        m.channel_axis = tag % 2;
        e.add_quant(layer_prefix(0) + suffix, m);
    }
    run("quant_all", e, bind_ll);

    // A partially-quantized layer: only some projections carry metadata. The
    // mixed case is what a real fp8 checkpoint looks like.
    LoadedModel e2;
    populate_llama_like(e2, 1, true, false, false, false, false);
    QuantMeta m;
    m.kind = QuantMeta::Kind::PerChannel;
    m.group_size = 0;
    m.channel_axis = 1;
    e2.add_quant(layer_prefix(0) + "self_attn.q_proj.weight", m);
    e2.add_quant(layer_prefix(0) + "mlp.down_proj.weight", m);
    run("quant_partial", e2, bind_ll);
}

// Script 4 — the two architecture variants that reuse the same struct.
void script_variants() {
    {
        // OLMo-3 reads a different norm into `attn_norm`. Populate the same
        // tensor set llama-like uses *plus* OLMo's, so that the transcript
        // shows the binder choosing rather than taking the only option.
        LoadedModel e;
        e.cfg.use_qk_norm = true;
        populate_olmo3(e, 2, true, false);
        e.add(layer_prefix(0) + "input_layernorm.weight");
        e.add(layer_prefix(1) + "input_layernorm.weight");
        run("olmo3", e, pie_cuda_driver::model::bind_olmo3);
    }
    {
        LoadedModel e;
        e.cfg.attention_bias = true;
        populate_olmo3(e, 1, false, true);
        e.cfg.tie_word_embeddings = true;
        run("olmo3_tied_bias", e, pie_cuda_driver::model::bind_olmo3);
    }
    {
        // Phi-3 validates that the loader already split the fused tensors,
        // then delegates. The happy path.
        LoadedModel e;
        populate_llama_like(e, 2, true, false, false, false, false);
        run("phi3", e, pie_cuda_driver::model::bind_phi3);
    }
    {
        // ... and the two ways it refuses.
        LoadedModel e;
        e.cfg.num_hidden_layers = 1;
        e.add("model.embed_tokens.weight");
        e.add("model.norm.weight");
        e.add("lm_head.weight");
        e.add(layer_prefix(0) + "mlp.gate_proj.weight");
        e.add(layer_prefix(0) + "mlp.up_proj.weight");
        run("phi3_missing_qkv", e, pie_cuda_driver::model::bind_phi3);
    }
    {
        LoadedModel e;
        e.cfg.num_hidden_layers = 1;
        e.add("model.embed_tokens.weight");
        e.add("model.norm.weight");
        e.add("lm_head.weight");
        e.add(layer_prefix(0) + "self_attn.q_proj.weight");
        e.add(layer_prefix(0) + "self_attn.k_proj.weight");
        e.add(layer_prefix(0) + "self_attn.v_proj.weight");
        run("phi3_missing_gate_up", e, pie_cuda_driver::model::bind_phi3);
    }
}

// Script 5 — zero layers. The loop body never runs; everything above it still
// must.
void script_degenerate() {
    LoadedModel e;
    populate_llama_like(e, 0, true, false, false, false, false);
    run("zero_layers", e, bind_ll);

    LoadedModel e2;
    e2.cfg.num_hidden_layers = 1;
    e2.add("model.norm.weight");
    e2.add("lm_head.weight");
    run("missing_embed", e2, bind_ll);

    LoadedModel e3;
    e3.cfg.num_hidden_layers = 1;
    e3.add("model.embed_tokens.weight");
    e3.add("lm_head.weight");
    run("missing_final_norm", e3, bind_ll);
}

}  // namespace

int main() {
    script_config_grid();
    script_fusion();
    script_quant_sidemap();
    script_variants();
    script_degenerate();
    return 0;
}
