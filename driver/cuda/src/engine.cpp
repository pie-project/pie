#include "engine.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>

#include <cuda_runtime.h>

#include "cuda_check.hpp"

namespace pie_cuda_driver {

namespace {

// Llama-like name → shard axis. Returns -1 for tensors that should be
// fully replicated (norms, biases on row-parallel projections, embedding,
// etc). Convention follows pie_driver/model/qwen3.py:
//
//   * column-parallel (axis=0): Q/K/V projections, MLP gate/up, lm_head
//   * row-parallel    (axis=1): attn O proj, MLP down_proj
//   * replicated     (axis=-1): norms, embed (we keep embed full to avoid
//                               an all_gather on the embed output).
//
// Mixtral / Qwen3.5-MoE / Gemma-4 / Gemma-3n have additional weight names
// (expert FFNs, AltUp, …) — they're not in this first cut and the engine
// rejects them at the top of Engine::load when tp_size > 1.
int llama_like_shard_axis(const std::string& name) {
    auto ends_with = [&](const char* suffix) {
        const auto n = std::char_traits<char>::length(suffix);
        return name.size() >= n &&
               name.compare(name.size() - n, n, suffix) == 0;
    };
    // Column-parallel: shard along the leading (output) dim.
    // lm_head is intentionally left replicated. Tied-embedding models
    // (Qwen3-0.6B, most small Llamas) reuse `embed_tokens.weight` for the
    // output projection and don't ship a separate `lm_head.weight`; for
    // untied models the weight is duplicated on every rank, which costs
    // memory but spares an all-gather/all-reduce on every fire. Revisit if
    // the lm_head footprint dominates on large models.
    //
    // GPT-OSS attention sinks (`.sinks`) are per-head [num_attention_heads]
    // — they shard along the head axis exactly like q/k/v biases.
    if (ends_with(".q_proj.weight") || ends_with(".q_proj.bias") ||
        ends_with(".k_proj.weight") || ends_with(".k_proj.bias") ||
        ends_with(".v_proj.weight") || ends_with(".v_proj.bias") ||
        ends_with(".gate_proj.weight") ||
        ends_with(".up_proj.weight") ||
        ends_with(".sinks")) {
        return 0;
    }
    // Row-parallel: shard along the inner (input) dim.
    if (ends_with(".o_proj.weight") || ends_with(".down_proj.weight")) {
        return 1;
    }
    // Mixtral / GPT-OSS expert weights. Each expert is sharded the same
    // way as a dense MLP: w1/w3 column-parallel, w2 row-parallel.
    // Expert biases match the corresponding weight axis (b_gate/b_up
    // column-parallel; b_down replicated and applied on the leader).
    if (ends_with(".w1.weight") || ends_with(".w3.weight") ||
        ends_with(".w1.bias")   || ends_with(".w3.bias")) {
        return 0;
    }
    if (ends_with(".w2.weight")) {
        return 1;
    }
    // Qwen3.5 / Qwen3.6-MoE linear-attention. The fused `in_proj_qkv` and
    // conv1d weights have a [K1 | K2 | V] block layout that doesn't shard
    // cleanly under uniform axis-0 partitioning, so they stay replicated
    // at engine load — bind_qwen3_5 materialises the per-rank slices
    // by hand. Everything else (z gate, b/a per-head linears, out_proj,
    // dt_bias, A_log) shards along its natural axis.
    if (ends_with(".linear_attn.in_proj_z.weight") ||
        ends_with(".linear_attn.in_proj_b.weight") ||
        ends_with(".linear_attn.in_proj_a.weight") ||
        ends_with(".linear_attn.dt_bias") ||
        ends_with(".linear_attn.A_log")) {
        return 0;
    }
    if (ends_with(".linear_attn.out_proj.weight")) {
        return 1;
    }
    return -1;
}

// Whitelist of model_types we currently support TP for. The shard plan is
// llama-like and assumes the standard name layout — gemma/mixtral/MoE need
// their own plan (per-expert weights, dual-norm, etc.).
bool supports_tp(const std::string& mt) {
    return mt == "qwen3"
        || mt == "qwen2"
        || mt == "llama" || mt == "llama3"
        || mt == "mistral" || mt == "mistral3"
        || mt == "phi3"
        || mt == "olmo2" || mt == "olmo3"
        || mt == "gemma2"
        || mt == "gemma3" || mt == "gemma3_text"
        || mt == "gemma4" || mt == "gemma4_text"
        || mt == "mixtral"
        || mt == "gemma3n" || mt == "gemma3n_text"
        || mt == "gpt_oss"
        || mt == "qwen3_5" || mt == "qwen3_5_text"
        || mt == "qwen3_5_moe" || mt == "qwen3_5_moe_text";
}

}  // namespace

Engine Engine::load(const Config& boot_cfg) {
    if (boot_cfg.model.snapshot_dir.empty()) {
        throw std::runtime_error(
            "engine: model.snapshot_dir is empty — pass it in dev.toml or "
            "let the wrapper resolve it via pie_driver.hf_utils");
    }

    Engine e;
    e.boot_ = boot_cfg;

    const std::filesystem::path snapshot{boot_cfg.model.snapshot_dir};
    e.hf_ = parse_hf_config(snapshot / "config.json");

    // Bind to the requested CUDA device before we allocate anything.
    int dev_id = 0;
    {
        const auto& d = boot_cfg.model.device;
        const auto colon = d.find(':');
        if (colon != std::string::npos) {
            dev_id = std::stoi(d.substr(colon + 1));
        }
    }
    CUDA_CHECK(cudaSetDevice(dev_id));

    auto loader = SafetensorsLoader::open(snapshot);

    const int tp_size = boot_cfg.distributed.tp_size;
    const int tp_rank = boot_cfg.distributed.tp_rank;
    if (tp_size > 1 && !supports_tp(e.hf_.model_type)) {
        throw std::runtime_error(
            "engine: tensor-parallelism not yet supported for model_type='" +
            e.hf_.model_type +
            "'. Currently TP-enabled: qwen2/qwen3, llama/llama3, mistral/"
            "mistral3, phi3, olmo2/olmo3, gemma2/gemma3.");
    }
    // Sharding along the head dim requires the head/expert counts and
    // intermediate widths to all divide cleanly by tp_size. Reject early
    // with a useful message instead of failing inside `load_to_device_sharded`
    // (which sees only one tensor at a time and can't explain why).
    if (tp_size > 1) {
        const auto& hf = e.hf_;
        auto require_divisible = [&](int v, const char* name) {
            if (v <= 0 || v % tp_size != 0) {
                throw std::runtime_error(
                    std::string("engine: ") + name + "=" + std::to_string(v) +
                    " is not divisible by tp_size=" + std::to_string(tp_size) +
                    ". Sharding the head/intermediate axis requires this; "
                    "use a smaller tp_size or run single-GPU.");
            }
        };
        require_divisible(hf.num_attention_heads, "num_attention_heads");
        require_divisible(hf.num_key_value_heads, "num_key_value_heads");
        // Qwen3.5-MoE has no dense `intermediate_size`; the MLP lives
        // entirely in `moe_intermediate_size` + `shared_expert_intermediate_size`.
        const bool is_q35_moe =
            (hf.model_type == "qwen3_5_moe" ||
             hf.model_type == "qwen3_5_moe_text");
        if (!is_q35_moe) {
            require_divisible(hf.intermediate_size, "intermediate_size");
        }
        // Qwen3.5 / 3.6-MoE: linear-attention head counts must shard too.
        if (hf.model_type == "qwen3_5" || hf.model_type == "qwen3_5_text" ||
            is_q35_moe) {
            require_divisible(hf.linear_num_key_heads, "linear_num_key_heads");
            require_divisible(hf.linear_num_value_heads, "linear_num_value_heads");
        }
        if (is_q35_moe) {
            require_divisible(hf.moe_intermediate_size, "moe_intermediate_size");
            require_divisible(hf.shared_expert_intermediate_size,
                              "shared_expert_intermediate_size");
        }
    }

    const auto t0 = std::chrono::steady_clock::now();

    std::uint64_t loaded_bytes = 0;
    e.weights_.reserve(loader.num_tensors());

    // Phi-3 ships fused `qkv_proj.weight = [Hq | Hk | Hk, H]` and
    // `gate_up_proj.weight = [I | I, H]`. A naive axis-0 split of the fused
    // tensor straddles the Q/K/V (resp. gate/up) block boundaries, so each
    // rank would get a mix of unrelated rows. Instead, load each block's
    // own sharded slice into the unfused name (`q_proj.weight`, etc.) and
    // skip the fused entry — bind_phi3 then sees the unfused names already
    // present and bypasses its single-GPU view-slicing path.
    const bool unfuse_phi3 = (tp_size > 1) && (e.hf_.model_type == "phi3");
    const bool shard_q35_moe_experts = (tp_size > 1) &&
        (e.hf_.model_type == "qwen3_5_moe" ||
         e.hf_.model_type == "qwen3_5_moe_text");
    auto ends_with = [](const std::string& s, const std::string& suf) {
        return s.size() >= suf.size() &&
               s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
    };

    for (const auto& name : loader.tensor_names()) {
        // MoE routed-expert weights are 3-D `[E, *, *]`; sharding them at
        // load time avoids ever materialising the full fused tensor on a
        // rank, which is what OOM'd Qwen3.6-35B-A3B at TP=2.
        if (shard_q35_moe_experts &&
            ends_with(name, ".mlp.experts.gate_up_proj")) {
            DeviceTensor t = loader.load_to_device_moe_gate_up_sharded(
                name, tp_rank, tp_size);
            loaded_bytes += t.nbytes();
            e.weights_.emplace(name, std::move(t));
            continue;
        }
        if (shard_q35_moe_experts &&
            ends_with(name, ".mlp.experts.down_proj")) {
            DeviceTensor t = loader.load_to_device_moe_down_sharded(
                name, tp_rank, tp_size);
            loaded_bytes += t.nbytes();
            e.weights_.emplace(name, std::move(t));
            continue;
        }
        if (unfuse_phi3 && ends_with(name, ".self_attn.qkv_proj.weight")) {
            const std::string prefix = name.substr(
                0, name.size() - std::string(".self_attn.qkv_proj.weight").size());
            const std::int64_t Hq = static_cast<std::int64_t>(e.hf_.num_attention_heads) * e.hf_.head_dim;
            const std::int64_t Hk = static_cast<std::int64_t>(e.hf_.num_key_value_heads) * e.hf_.head_dim;
            DeviceTensor q = loader.load_to_device_row_range_sharded(
                name, /*row_offset=*/0, /*rows=*/Hq, tp_rank, tp_size);
            DeviceTensor k = loader.load_to_device_row_range_sharded(
                name, /*row_offset=*/Hq, /*rows=*/Hk, tp_rank, tp_size);
            DeviceTensor v = loader.load_to_device_row_range_sharded(
                name, /*row_offset=*/Hq + Hk, /*rows=*/Hk, tp_rank, tp_size);
            loaded_bytes += q.nbytes() + k.nbytes() + v.nbytes();
            e.weights_.emplace(prefix + ".self_attn.q_proj.weight", std::move(q));
            e.weights_.emplace(prefix + ".self_attn.k_proj.weight", std::move(k));
            e.weights_.emplace(prefix + ".self_attn.v_proj.weight", std::move(v));
            continue;
        }
        if (unfuse_phi3 && ends_with(name, ".mlp.gate_up_proj.weight")) {
            const std::string prefix = name.substr(
                0, name.size() - std::string(".mlp.gate_up_proj.weight").size());
            const std::int64_t I = e.hf_.intermediate_size;
            DeviceTensor g = loader.load_to_device_row_range_sharded(
                name, /*row_offset=*/0, /*rows=*/I, tp_rank, tp_size);
            DeviceTensor u = loader.load_to_device_row_range_sharded(
                name, /*row_offset=*/I, /*rows=*/I, tp_rank, tp_size);
            loaded_bytes += g.nbytes() + u.nbytes();
            e.weights_.emplace(prefix + ".mlp.gate_proj.weight", std::move(g));
            e.weights_.emplace(prefix + ".mlp.up_proj.weight", std::move(u));
            continue;
        }

        const int axis = (tp_size > 1) ? llama_like_shard_axis(name) : -1;
        DeviceTensor t = (axis >= 0)
            ? loader.load_to_device_sharded(name, axis, tp_rank, tp_size)
            : loader.load_to_device(name);
        loaded_bytes += t.nbytes();
        e.weights_.emplace(name, std::move(t));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double mib = static_cast<double>(loaded_bytes) / (1024.0 * 1024.0);

    std::cerr << "[pie-driver-cuda] loaded " << e.weights_.size() << " tensors ("
              << static_cast<std::uint64_t>(mib) << " MiB on this rank, "
              << "tp=" << tp_size << ") in " << static_cast<int>(ms)
              << " ms; arch=" << e.hf_.arch_name << " (" << e.hf_.model_type << ")\n";

    return e;
}

EngineCapabilities Engine::capabilities() const {
    EngineCapabilities c;
    c.total_pages = 0;  // populated in M1.2.2 once kv_cache lands
    c.kv_page_size = static_cast<int>(boot_.batching.kv_page_size);
    c.swap_pool_size = 0;
    c.max_batch_tokens = static_cast<int>(boot_.batching.max_batch_tokens);
    c.max_batch_size = static_cast<int>(boot_.batching.max_batch_size);
    // The runtime's `model::instruct::create` dispatches on the
    // PIE-arch key ("llama3", "gemma3", …) not HF's `architectures[0]`
    // ("LlamaForCausalLM") nor the raw HF model_type ("llama",
    // "gemma3_text"). The Python `pie_driver` normalises via the
    // `HF_TO_PIE_ARCH` table; we mirror that table here so the
    // runtime gets the same key from both backends.
    auto normalise_arch = [](const std::string& mt) -> std::string {
        if (mt == "llama")        return "llama3";
        if (mt == "gemma3_text")  return "gemma3";
        if (mt == "gemma4_text")  return "gemma4";
        if (mt == "ministral3")   return "mistral3";
        return mt;  // qwen2 / qwen3 / gemma2 / olmo3 / phi3 / mistral3 / mixtral
    };
    c.arch_name = hf_.model_type.empty()
        ? hf_.arch_name
        : normalise_arch(hf_.model_type);
    c.vocab_size = hf_.vocab_size;
    c.max_model_len = hf_.max_position_embeddings;
    c.activation_dtype = boot_.model.dtype;
    c.snapshot_dir = boot_.model.snapshot_dir;
    return c;
}

std::uint64_t Engine::total_weight_bytes() const noexcept {
    std::uint64_t n = 0;
    for (const auto& [_, t] : weights_) n += t.nbytes();
    return n;
}

const DeviceTensor& Engine::get(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end()) {
        throw std::runtime_error("engine: weight not loaded: " + name);
    }
    return it->second;
}

void Engine::insert(std::string name, DeviceTensor tensor) {
    auto [it, inserted] = weights_.emplace(std::move(name), std::move(tensor));
    if (!inserted) {
        throw std::runtime_error("engine: weight already registered: " + it->first);
    }
}

}  // namespace pie_cuda_driver
