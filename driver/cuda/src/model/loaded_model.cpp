#include "model/loaded_model.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <pie_driver_common/tensor_names.hpp>

#include "cuda_check.hpp"
#include "distributed.hpp"
#include "kernels/dequant_fp8.hpp"
#include "kernels/dtype_cast.hpp"
#include "kernels/quant_bf16_to_fp8.hpp"

#ifdef PIE_CUDA_HAS_MARLIN
#include "marlin_wrapper.hpp"
#endif

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
// rejects them at the top of LoadedModel::load when tp_size > 1.
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
    // Compressed-tensors FP8 per-channel weight_scale companion. Shape is
    // `[N]` or `[N, 1]` aligned with the weight's output axis (axis 0).
    // For column-parallel projections each rank holds [N/tp, K] of weight
    // and needs the matching [N/tp] scale slice. For row-parallel
    // projections the weight is sharded along K and every rank holds the
    // full N rows — so the scale stays replicated (not handled here;
    // axis -1 is the default fall-through).
    if (ends_with(".q_proj.weight_scale") ||
        ends_with(".k_proj.weight_scale") ||
        ends_with(".v_proj.weight_scale") ||
        ends_with(".gate_proj.weight_scale") ||
        ends_with(".up_proj.weight_scale")) {
        return 0;
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
        || mt == "mistral" || mt == "mistral3" || mt == "ministral3"
        || mt == "phi3"
        || mt == "olmo2" || mt == "olmo3"
        || mt == "gemma2"
        || mt == "gemma3" || mt == "gemma3_text"
        || mt == "gemma4" || mt == "gemma4_text"
        || mt == "mixtral"
        || mt == "gemma3n" || mt == "gemma3n_text"
        || mt == "gpt_oss"
        || mt == "qwen3_5" || mt == "qwen3_5_text"
        || mt == "qwen3_5_moe" || mt == "qwen3_5_moe_text"
        || mt == "qwen3_moe";
}

// True for any MoE model whose forward path lives in qwen3_5_moe_forward.
// All members share an all-MoE MLP layout (no dense `intermediate_size`),
// so the engine's TP divisibility checks on `intermediate_size` should
// be skipped for them.
bool is_qwen3_5_moe_arch(const std::string& mt) {
    return mt == "qwen3_5_moe" || mt == "qwen3_5_moe_text"
        || mt == "qwen3_moe";
}

// Slice a 2-D bf16 weight `[N, K]` per-rank along the given axis. Used
// by the AWQ eager-dequant path to produce per-rank shards from a full
// dequanted tensor without going through the safetensors loader's
// (which only sees AWQ packed dims). Returns the input unchanged on
// `axis = -1` (replicated).
DeviceTensor slice_bf16_per_rank(
    DeviceTensor& full, int axis, int tp_rank, int tp_size,
    const std::string& dbg_name)
{
    if (axis < 0) return std::move(full);
    const int N = static_cast<int>(full.shape()[0]);
    const int K = static_cast<int>(full.shape()[1]);
    if (axis == 0) {
        if (N % tp_size != 0) {
            throw std::runtime_error(
                "engine: col-parallel '" + dbg_name +
                "' N=" + std::to_string(N) +
                " not divisible by tp_size=" + std::to_string(tp_size));
        }
        const int N_per = N / tp_size;
        DeviceTensor out = DeviceTensor::allocate(
            DType::BF16,
            {static_cast<std::int64_t>(N_per),
             static_cast<std::int64_t>(K)});
        const std::size_t row_bytes = static_cast<std::size_t>(K) * 2;
        const auto* src = static_cast<const std::uint8_t*>(full.data())
            + static_cast<std::size_t>(tp_rank) * N_per * row_bytes;
        CUDA_CHECK(cudaMemcpyAsync(
            out.data(), src, static_cast<std::size_t>(N_per) * row_bytes,
            cudaMemcpyDeviceToDevice, /*stream=*/0));
        return out;
    }
    // axis == 1: row-parallel, slice along K.
    if (K % tp_size != 0) {
        throw std::runtime_error(
            "engine: row-parallel '" + dbg_name +
            "' K=" + std::to_string(K) +
            " not divisible by tp_size=" + std::to_string(tp_size));
    }
    const int K_per = K / tp_size;
    DeviceTensor out = DeviceTensor::allocate(
        DType::BF16,
        {static_cast<std::int64_t>(N),
         static_cast<std::int64_t>(K_per)});
    const auto* src = static_cast<const std::uint8_t*>(full.data())
        + static_cast<std::size_t>(tp_rank) * K_per * 2;
    CUDA_CHECK(cudaMemcpy2DAsync(
        out.data(),  /*dpitch=*/K_per * 2,
        src,         /*spitch=*/K * 2,
        /*width=*/K_per * 2,
        /*height=*/N,
        cudaMemcpyDeviceToDevice, /*stream=*/0));
    return out;
}

}  // namespace

LoadedModel LoadedModel::load(const Config& boot_cfg, NcclComm* tp_comm) {
    if (boot_cfg.model.snapshot_dir.empty()) {
        throw std::runtime_error(
            "engine: model.snapshot_dir is empty — pass it in dev.toml or "
            "let the wrapper resolve it via pie_driver.hf_utils");
    }

    LoadedModel e;
    e.boot_ = boot_cfg;
    const bool verbose = boot_cfg.runtime.verbose;

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

    // Compute capability — used by the runtime-quant skip (sm<89) and
    // by the eager FP8→bf16 dequant pass. cuBLASLt's native FP8 GEMM
    // requires sm89+ (Ada/Hopper); on Ampere (sm80) the dispatcher
    // falls back to dequant-then-bf16-GEMM.
    cudaDeviceProp dev_prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&dev_prop, dev_id));
    const bool fp8_native = (dev_prop.major > 8) ||
                            (dev_prop.major == 8 && dev_prop.minor >= 9);

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
        // Qwen3.5-MoE / Qwen3-MoE have no dense `intermediate_size`; the
        // MLP lives entirely in `moe_intermediate_size` (+ `shared_expert_
        // intermediate_size` for the 3.5/3.6 family — Qwen3-MoE has no
        // shared expert).
        const bool is_q35_moe = is_qwen3_5_moe_arch(hf.model_type);
        if (!is_q35_moe) {
            require_divisible(hf.intermediate_size, "intermediate_size");
        }
        // Qwen3.5 / 3.6-MoE: linear-attention head counts must shard too.
        // Qwen3-MoE has no linear-attn layers, so this check is skipped.
        const bool has_linear_attn =
            (hf.model_type == "qwen3_5" || hf.model_type == "qwen3_5_text" ||
             hf.model_type == "qwen3_5_moe" ||
             hf.model_type == "qwen3_5_moe_text");
        if (has_linear_attn) {
            require_divisible(hf.linear_num_key_heads, "linear_num_key_heads");
            require_divisible(hf.linear_num_value_heads, "linear_num_value_heads");
        }
        if (is_q35_moe) {
            require_divisible(hf.moe_intermediate_size, "moe_intermediate_size");
            // shared_expert_intermediate_size is 0 for Qwen3-MoE (no shared
            // expert); only enforce divisibility when the family actually
            // has one.
            if (hf.shared_expert_intermediate_size > 0) {
                require_divisible(hf.shared_expert_intermediate_size,
                                  "shared_expert_intermediate_size");
            }
        }
    }

    if (e.hf_.kv_cache_scheme_present) {
        std::cerr << "[pie-driver-cuda] WARNING: ckpt's "
                  << "quantization_config.kv_cache_scheme is non-null but "
                  << "the driver always uses bf16 KV cache. Generation "
                  << "may drift slightly from the calibrated reference.\n";
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
    // Multimodal text-tower extraction (Mistral3 / Llava / Qwen2.5-VL).
    // For these archs HF stores LLM weights under "language_model." and
    // vision-side weights under separate prefixes; we run the LLM only,
    // so strip the language prefix and skip the rest.
    const std::string& mm_strip = e.hf_.mm_lm_strip_prefix;
    const auto& mm_skip = e.hf_.mm_skip_prefixes;
    auto remap_name = [&](const std::string& raw) -> std::string {
        return pie_driver_common::strip_prefix(raw, mm_strip);
    };
    auto skip_tensor = [&](const std::string& raw) -> bool {
        return pie_driver_common::starts_with_any(raw, mm_skip);
    };

    for (const auto& raw_name : loader.tensor_names()) {
        if (skip_tensor(raw_name)) continue;
        const std::string name = remap_name(raw_name);
        // MoE routed-expert weights are 3-D `[E, *, *]`; sharding them at
        // load time avoids ever materialising the full fused tensor on a
        // rank, which is what OOM'd Qwen3.6-35B-A3B at TP=2.
        if (shard_q35_moe_experts &&
            pie_driver_common::ends_with(name, ".mlp.experts.gate_up_proj")) {
            DeviceTensor t = loader.load_to_device_moe_gate_up_sharded(
                raw_name, tp_rank, tp_size);
            loaded_bytes += t.nbytes();
            e.weights_.emplace(name, std::move(t));
            continue;
        }
        if (shard_q35_moe_experts &&
            pie_driver_common::ends_with(name, ".mlp.experts.down_proj")) {
            DeviceTensor t = loader.load_to_device_moe_down_sharded(
                raw_name, tp_rank, tp_size);
            loaded_bytes += t.nbytes();
            e.weights_.emplace(name, std::move(t));
            continue;
        }
        if (unfuse_phi3 &&
            pie_driver_common::ends_with(name, ".self_attn.qkv_proj.weight")) {
            const std::string prefix = name.substr(
                0, name.size() - std::string(".self_attn.qkv_proj.weight").size());
            const std::int64_t Hq = static_cast<std::int64_t>(e.hf_.num_attention_heads) * e.hf_.head_dim;
            const std::int64_t Hk = static_cast<std::int64_t>(e.hf_.num_key_value_heads) * e.hf_.head_dim;
            DeviceTensor q = loader.load_to_device_row_range_sharded(
                raw_name, /*row_offset=*/0, /*rows=*/Hq, tp_rank, tp_size);
            DeviceTensor k = loader.load_to_device_row_range_sharded(
                raw_name, /*row_offset=*/Hq, /*rows=*/Hk, tp_rank, tp_size);
            DeviceTensor v = loader.load_to_device_row_range_sharded(
                raw_name, /*row_offset=*/Hq + Hk, /*rows=*/Hk, tp_rank, tp_size);
            loaded_bytes += q.nbytes() + k.nbytes() + v.nbytes();
            e.weights_.emplace(prefix + ".self_attn.q_proj.weight", std::move(q));
            e.weights_.emplace(prefix + ".self_attn.k_proj.weight", std::move(k));
            e.weights_.emplace(prefix + ".self_attn.v_proj.weight", std::move(v));
            continue;
        }
        if (unfuse_phi3 &&
            pie_driver_common::ends_with(name, ".mlp.gate_up_proj.weight")) {
            const std::string prefix = name.substr(
                0, name.size() - std::string(".mlp.gate_up_proj.weight").size());
            const std::int64_t I = e.hf_.intermediate_size;
            DeviceTensor g = loader.load_to_device_row_range_sharded(
                raw_name, /*row_offset=*/0, /*rows=*/I, tp_rank, tp_size);
            DeviceTensor u = loader.load_to_device_row_range_sharded(
                raw_name, /*row_offset=*/I, /*rows=*/I, tp_rank, tp_size);
            loaded_bytes += g.nbytes() + u.nbytes();
            e.weights_.emplace(prefix + ".mlp.gate_proj.weight", std::move(g));
            e.weights_.emplace(prefix + ".mlp.up_proj.weight", std::move(u));
            continue;
        }

        const int axis = (tp_size > 1) ? llama_like_shard_axis(name) : -1;
        DeviceTensor t = (axis >= 0)
            ? loader.load_to_device_sharded(raw_name, axis, tp_rank, tp_size)
            : loader.load_to_device(raw_name);
        loaded_bytes += t.nbytes();
        e.weights_.emplace(name, std::move(t));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto ends_with_str = [](const std::string& s, const char* sfx) {
        return pie_driver_common::ends_with(s, sfx);
    };

    // ── Offline GPTQ INT4 (W4A16 via marlin) ──────────────────────────
    // GPTQ ships each projection as four tensors:
    //   {prefix}.qweight  int32 [K/8, N]    — packed int4 (8 nibbles/int)
    //   {prefix}.qzeros   int32 [groups, N/8]  — zero points (skip if sym=true)
    //   {prefix}.scales   fp16  [groups, N]   — per-group scales
    //   {prefix}.g_idx    int32 [K]          — desc_act permutation (skip)
    // We repack `qweight` into marlin's tile layout, cast the scales to
    // bf16, drop the unused tensors, and rename to the canonical
    // `{prefix}.weight` / `{prefix}.weight_scale_inv` so the existing
    // bind code finds them. The gemm dispatcher then routes the
    // resulting INT4_PACKED weight + bf16 scale through marlin.
    if (e.hf_.quant_method == "gptq" || e.hf_.quant_method == "awq") do {
#ifndef PIE_CUDA_HAS_MARLIN
        throw std::runtime_error(
            "engine: " + e.hf_.quant_method + " checkpoint detected but "
            "marlin is not compiled into this build. Reconfigure with "
            "-DPIE_CUDA_BUILD_MARLIN=ON.");
#else
        const bool is_awq = (e.hf_.quant_method == "awq");
        if (e.hf_.quant_bits != 4) {
            throw std::runtime_error(
                "engine: only 4-bit " + e.hf_.quant_method +
                " is supported (got bits=" +
                std::to_string(e.hf_.quant_bits) + ")");
        }
        // GPTQ goes through eager dequant when:
        //   * desc_act=true (act-order; would need permuted marlin repack)
        //   * sym=false / zero_point=true (asymmetric; marlin's kU4B8
        //     dispatch only handles sym; kU4 + has_zp would work but
        //     we hit the same issue as AWQ).
        // The launch_gptq_dequant_to_bf16 kernel handles both cases via
        // (qweight - (qzeros+1)) * scales — same autogptq convention
        // for sym (qzeros all 7 → +1 = 8 = kU4B8 bias) and asym (qzeros
        // stores zp-1 → +1 = actual zp). desc_act=true also drives the
        // optional g_idx lookup.
        const bool gptq_eager_dequant = !is_awq && (
            e.hf_.quant_desc_act ||
            e.hf_.quant_zero_point ||
            !e.hf_.quant_sym);
        // TP slicing strategy: the safetensors loader leaves GPTQ tensors
        // (`.qweight`, `.qzeros`, `.scales`, `.g_idx`) replicated on every
        // rank — the standard `llama_like_shard_axis` doesn't recognise
        // those name suffixes. Each rank holds the full tensor briefly
        // and then slices its own shard out below before the repack runs
        // per-rank. Net steady-state memory is `1/tp_size`-of-bf16 per
        // rank (same as bf16 + TP).
        // AWQ + TP works via the eager-dequant path: dequant to full
        // bf16 [N, K] on every rank, then slice per-rank. The full
        // [N, K] is briefly resident — fine for small/medium AWQ ckpts.
        // For very large ones a sharded-dequant variant could slice
        // qweight/qzeros/scales first; not implemented yet.
        if (is_awq) {
            // AWQ via marlin produces gibberish end-to-end (task #28 —
            // numerically diagnosed: same-input cos≈0.48 vs a numpy
            // reference dequant of the same ckpt). Until the marlin
            // kU4-vs-kU4B8 divergence is fixed in the kernel, run AWQ
            // through an eager dequant directly to bf16 at load time.
            // Trades the W4 memory savings (~4× larger weights) for
            // correctness — same trade-off the FP8 path makes on sm80.
            if (e.hf_.quant_group_size <= 0) {
                throw std::runtime_error(
                    "engine: AWQ ckpt missing quantization_config.group_size");
            }
            std::vector<std::string> awq_qw_names;
            for (const auto& [n, _] : e.weights_) {
                if (ends_with_str(n, ".qweight")) awq_qw_names.push_back(n);
            }
            std::uint64_t awq_freed = 0, awq_allocated = 0;
            for (const auto& qw_name : awq_qw_names) {
                const std::string prefix = qw_name.substr(
                    0, qw_name.size() - 8);  // strip ".qweight"
                const std::string qz_name = prefix + ".qzeros";
                const std::string sc_name = prefix + ".scales";
                const std::string bias_name = prefix + ".bias";
                const std::string canonical_w = prefix + ".weight";
                auto qw_it = e.weights_.find(qw_name);
                auto qz_it = e.weights_.find(qz_name);
                auto sc_it = e.weights_.find(sc_name);
                if (qw_it == e.weights_.end() || qz_it == e.weights_.end() ||
                    sc_it == e.weights_.end()) {
                    throw std::runtime_error(
                        "engine: AWQ '" + qw_name + "' missing qzeros / scales");
                }
                const DeviceTensor& qw = qw_it->second;
                const DeviceTensor& sc = sc_it->second;
                if (qw.shape().size() != 2) {
                    throw std::runtime_error(
                        "engine: AWQ qweight '" + qw_name + "' must be 2-D");
                }
                const int Kdim = static_cast<int>(qw.shape()[0]);
                const int Ndim = static_cast<int>(qw.shape()[1]) * 8;
                // Scales arrive fp16 in casperhansen-style AWQ ckpts.
                // Cast to bf16 in-place before the kernel runs — cheap
                // (one pass over [groups, N], not [K, N]).
                DeviceTensor sc_bf16 = sc.dtype() == DType::BF16
                    ? DeviceTensor()  // unused; we'll point at the original
                    : DeviceTensor::allocate(DType::BF16, sc.shape());
                const void* sc_ptr = sc.data();
                if (sc.dtype() == DType::FP16) {
                    kernels::launch_cast_fp16_to_bf16(
                        sc.data(), sc_bf16.data(), sc.numel(), /*stream=*/0);
                    sc_ptr = sc_bf16.data();
                } else if (sc.dtype() == DType::FP32) {
                    kernels::launch_cast_fp32_to_bf16(
                        sc.data(), sc_bf16.data(), sc.numel(), /*stream=*/0);
                    sc_ptr = sc_bf16.data();
                } else if (sc.dtype() != DType::BF16) {
                    throw std::runtime_error(
                        "engine: AWQ scales '" + sc_name +
                        "' has unexpected dtype " +
                        std::string(dtype_name(sc.dtype())));
                }
                // Dequant to full bf16 [N, K] first.
                DeviceTensor bf16_full = DeviceTensor::allocate(
                    DType::BF16,
                    {static_cast<std::int64_t>(Ndim),
                     static_cast<std::int64_t>(Kdim)});
                kernels::launch_awq_dequant_to_bf16(
                    qw.data(), qz_it->second.data(), sc_ptr,
                    bf16_full.data(), Kdim, Ndim, e.hf_.quant_group_size,
                    /*stream=*/0);
                awq_freed += qw.nbytes() + qz_it->second.nbytes() + sc.nbytes();

                // TP slice using the standard llama-like shard rule:
                // axis=0 (col-parallel, q/k/v/gate/up) shards the bf16
                // [N, K] along N; axis=1 (row-parallel, o/down) along
                // K; axis=-1 (everything else) stays replicated.
                DeviceTensor bf16_w = (tp_size <= 1)
                    ? std::move(bf16_full)
                    : slice_bf16_per_rank(
                          bf16_full,
                          llama_like_shard_axis(canonical_w),
                          tp_rank, tp_size, canonical_w);
                awq_allocated += bf16_w.nbytes();
                e.weights_.erase(qw_it);
                e.weights_.erase(qz_name);
                e.weights_.erase(sc_name);
                e.weights_.emplace(canonical_w, std::move(bf16_w));
                // bias (if present) is already loaded under prefix+".bias"
                // and is left as-is — engine's bf16 cleanup pass will
                // cast it down from fp16 if needed.
                (void)bias_name;
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            if (verbose) {
                std::cerr << "[pie-driver-cuda] AWQ -> bf16 eager dequant: "
                          << awq_qw_names.size() << " projections, "
                          << (awq_freed / (1024 * 1024)) << " MiB int4 -> "
                          << (awq_allocated / (1024 * 1024)) << " MiB bf16\n";
            }
            loaded_bytes = loaded_bytes - awq_freed + awq_allocated;
            break;  // Skip the marlin pass entirely.
        }

        // GPTQ eager dequant via launch_gptq_dequant_to_bf16. g_idx is
        // optional (only present for desc_act=true ckpts). Shape +
        // per-rank slice semantics mirror the AWQ path.
        if (gptq_eager_dequant) {
            if (e.hf_.quant_group_size <= 0) {
                throw std::runtime_error(
                    "engine: GPTQ eager dequant with group_size<=0 is not "
                    "supported (per-channel has no clean fallback yet)");
            }
            std::vector<std::string> qw_names;
            for (const auto& [n, _] : e.weights_) {
                if (ends_with_str(n, ".qweight")) qw_names.push_back(n);
            }
            std::uint64_t freed = 0, allocated = 0;
            for (const auto& qw_name : qw_names) {
                const std::string prefix = qw_name.substr(
                    0, qw_name.size() - 8);
                const std::string qz_name = prefix + ".qzeros";
                const std::string sc_name = prefix + ".scales";
                const std::string gi_name = prefix + ".g_idx";
                const std::string canonical_w = prefix + ".weight";
                auto qw_it = e.weights_.find(qw_name);
                auto qz_it = e.weights_.find(qz_name);
                auto sc_it = e.weights_.find(sc_name);
                if (qw_it == e.weights_.end() || qz_it == e.weights_.end() ||
                    sc_it == e.weights_.end()) {
                    throw std::runtime_error(
                        "engine: GPTQ '" + qw_name + "' missing qzeros/scales");
                }
                const DeviceTensor& qw = qw_it->second;
                const DeviceTensor& sc = sc_it->second;
                if (qw.shape().size() != 2) {
                    throw std::runtime_error(
                        "engine: GPTQ qweight '" + qw_name + "' must be 2-D");
                }
                // GPTQ qweight: [K/8, N] int32; logical dims:
                const int Kdim = static_cast<int>(qw.shape()[0]) * 8;
                const int Ndim = static_cast<int>(qw.shape()[1]);
                // Cast scales to bf16 if fp16 (small tensor; same as AWQ).
                DeviceTensor sc_bf16 = sc.dtype() == DType::BF16
                    ? DeviceTensor()
                    : DeviceTensor::allocate(DType::BF16, sc.shape());
                const void* sc_ptr = sc.data();
                if (sc.dtype() == DType::FP16) {
                    kernels::launch_cast_fp16_to_bf16(
                        sc.data(), sc_bf16.data(), sc.numel(), /*stream=*/0);
                    sc_ptr = sc_bf16.data();
                } else if (sc.dtype() == DType::FP32) {
                    kernels::launch_cast_fp32_to_bf16(
                        sc.data(), sc_bf16.data(), sc.numel(), /*stream=*/0);
                    sc_ptr = sc_bf16.data();
                } else if (sc.dtype() != DType::BF16) {
                    throw std::runtime_error(
                        "engine: GPTQ scales '" + sc_name +
                        "' has unexpected dtype " +
                        std::string(dtype_name(sc.dtype())));
                }
                // g_idx is optional — only desc_act=true ckpts ship it.
                // For sym=false / desc_act=false ckpts, the kernel uses
                // the implicit `g = k / group_size` mapping.
                auto gi_it = e.weights_.find(gi_name);
                const void* gi_ptr = (gi_it != e.weights_.end())
                    ? gi_it->second.data() : nullptr;
                if (e.hf_.quant_desc_act && !gi_ptr) {
                    throw std::runtime_error(
                        "engine: GPTQ desc_act=true ckpt missing g_idx for '" +
                        qw_name + "'");
                }
                DeviceTensor bf16_full = DeviceTensor::allocate(
                    DType::BF16,
                    {static_cast<std::int64_t>(Ndim),
                     static_cast<std::int64_t>(Kdim)});
                kernels::launch_gptq_dequant_to_bf16(
                    qw.data(), qz_it->second.data(), sc_ptr, gi_ptr,
                    bf16_full.data(), Kdim, Ndim, e.hf_.quant_group_size,
                    /*stream=*/0);
                freed += qw.nbytes() + qz_it->second.nbytes() + sc.nbytes()
                       + (gi_it != e.weights_.end() ? gi_it->second.nbytes() : 0);

                DeviceTensor bf16_w = (tp_size <= 1)
                    ? std::move(bf16_full)
                    : slice_bf16_per_rank(
                          bf16_full,
                          llama_like_shard_axis(canonical_w),
                          tp_rank, tp_size, canonical_w);
                allocated += bf16_w.nbytes();
                e.weights_.erase(qw_it);
                e.weights_.erase(qz_name);
                e.weights_.erase(sc_name);
                if (gi_it != e.weights_.end()) e.weights_.erase(gi_name);
                e.weights_.emplace(canonical_w, std::move(bf16_w));
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            if (verbose) {
                std::cerr << "[pie-driver-cuda] GPTQ -> bf16 eager dequant ("
                          << (e.hf_.quant_desc_act ? "desc_act=true" : "")
                          << (e.hf_.quant_desc_act && (!e.hf_.quant_sym || e.hf_.quant_zero_point) ? "+" : "")
                          << ((!e.hf_.quant_sym || e.hf_.quant_zero_point) ? "asym" : "")
                          << "): " << qw_names.size() << " projections, "
                          << (freed / (1024 * 1024)) << " MiB int4 -> "
                          << (allocated / (1024 * 1024)) << " MiB bf16\n";
            }
            loaded_bytes = loaded_bytes - freed + allocated;
            break;  // Skip the marlin pass.
        }

        std::vector<std::string> qw_names;
        qw_names.reserve(e.weights_.size());
        for (const auto& [n, _] : e.weights_) {
            if (ends_with_str(n, ".qweight")) qw_names.push_back(n);
        }

        const int group_size = e.hf_.quant_group_size;
        std::size_t quant_count = 0;
        std::uint64_t bytes_before = 0;
        std::uint64_t bytes_after  = 0;

        // Column-parallel (q/k/v/gate/up) shards `qweight` axis 1 (N) and
        // `scales` axis 1 (N). Row-parallel (o/down) shards qweight axis
        // 0 (K-packed) and scales axis 0 (groups). Identifying which is
        // which by name suffix, mirroring the `llama_like_shard_axis`
        // convention.
        auto is_row_parallel_proj = [&](const std::string& s) {
            return ends_with_str(s, ".o_proj.qweight")
                || ends_with_str(s, ".down_proj.qweight");
        };

        for (const auto& qw_name : qw_names) {
            const std::string prefix = qw_name.substr(0, qw_name.size() - 8);  // strip ".qweight"
            const std::string scales_name = prefix + ".scales";
            const std::string canonical_w = prefix + ".weight";
            const std::string canonical_s = prefix + ".weight_scale_inv";

            auto qw_it = e.weights_.find(qw_name);
            auto sc_it = e.weights_.find(scales_name);
            if (qw_it == e.weights_.end() || sc_it == e.weights_.end()) {
                throw std::runtime_error(
                    "engine: GPTQ tensor '" + qw_name +
                    "' missing matching '.scales' partner");
            }
            const DeviceTensor& qw = qw_it->second;
            const DeviceTensor& sc = sc_it->second;
            if (qw.shape().size() != 2 || sc.shape().size() != 2) {
                throw std::runtime_error(
                    "engine: GPTQ tensor '" + qw_name +
                    "': expected 2-D qweight/scales");
            }
            // Shape conventions differ between GPTQ and AWQ:
            //   GPTQ qweight:  [K/8, N]    (axis 0 packed)
            //   AWQ  qweight:  [K, N/8]    (axis 1 packed)
            // Decode so the rest of the loop sees logical (Ks_full, Ns_full).
            int Ks_full;
            int Ns_full;
            if (is_awq) {
                Ks_full = static_cast<int>(qw.shape()[0]);
                Ns_full = static_cast<int>(qw.shape()[1]) * 8;
            } else {
                Ks_full = static_cast<int>(qw.shape()[0]) * 8;
                Ns_full = static_cast<int>(qw.shape()[1]);
            }
            const int Ks_packed_full = Ks_full / 8;
            const int sc_groups_full = static_cast<int>(sc.shape()[0]);

            const bool row_parallel = (tp_size > 1) && is_row_parallel_proj(qw_name);
            const bool col_parallel = (tp_size > 1) && !is_row_parallel_proj(qw_name);

            // Per-rank dims after slicing.
            int Ks_packed_local = Ks_packed_full;
            int Ns_local        = Ns_full;
            int sc_groups_local = sc_groups_full;
            if (col_parallel) {
                if (Ns_full % tp_size != 0) {
                    throw std::runtime_error(
                        "engine: GPTQ col-parallel '" + qw_name +
                        "': N=" + std::to_string(Ns_full) +
                        " not divisible by tp_size=" + std::to_string(tp_size));
                }
                Ns_local = Ns_full / tp_size;
            } else if (row_parallel) {
                if (Ks_packed_full % tp_size != 0) {
                    throw std::runtime_error(
                        "engine: GPTQ row-parallel '" + qw_name +
                        "': K/8=" + std::to_string(Ks_packed_full) +
                        " not divisible by tp_size=" + std::to_string(tp_size));
                }
                if (sc_groups_full % tp_size != 0) {
                    throw std::runtime_error(
                        "engine: GPTQ row-parallel '" + qw_name +
                        "': groups=" + std::to_string(sc_groups_full) +
                        " not divisible by tp_size=" + std::to_string(tp_size) +
                        " (group_size=" + std::to_string(group_size) +
                        " must straddle the rank shard cleanly)");
                }
                Ks_packed_local = Ks_packed_full / tp_size;
                sc_groups_local = sc_groups_full / tp_size;
            }
            const int Ks_local = Ks_packed_local * 8;

            // Allocate per-rank qweight buffer, slice from the replicated
            // full tensor. Column-parallel uses a strided 2-D copy; row-
            // parallel is a contiguous chunk.
            DeviceTensor qw_local;
            if (tp_size == 1) {
                qw_local = DeviceTensor();  // unused; we use `qw` directly below
            } else {
                qw_local = DeviceTensor::allocate(
                    DType::INT32,
                    {static_cast<std::int64_t>(Ks_packed_local),
                     static_cast<std::int64_t>(Ns_local)});
                if (col_parallel) {
                    // src [Ks_packed_full, Ns_full] int32, take cols
                    // [tp_rank*Ns_local : tp_rank*Ns_local + Ns_local].
                    const std::size_t row_bytes_dst =
                        static_cast<std::size_t>(Ns_local) * sizeof(std::int32_t);
                    const std::size_t row_bytes_src =
                        static_cast<std::size_t>(Ns_full) * sizeof(std::int32_t);
                    const auto* src8 = static_cast<const std::uint8_t*>(qw.data());
                    src8 += static_cast<std::size_t>(tp_rank) * row_bytes_dst;
                    CUDA_CHECK(cudaMemcpy2DAsync(
                        qw_local.data(), row_bytes_dst,
                        src8, row_bytes_src,
                        row_bytes_dst, Ks_packed_local,
                        cudaMemcpyDeviceToDevice, /*stream=*/0));
                } else if (row_parallel) {
                    // src [Ks_packed_full, Ns_full] int32, take rows
                    // [tp_rank*Ks_packed_local : tp_rank*Ks_packed_local + Ks_packed_local].
                    const std::size_t bytes_per_rank_block =
                        static_cast<std::size_t>(Ks_packed_local) *
                        static_cast<std::size_t>(Ns_full) * sizeof(std::int32_t);
                    const auto* src8 = static_cast<const std::uint8_t*>(qw.data());
                    src8 += static_cast<std::size_t>(tp_rank) * bytes_per_rank_block;
                    CUDA_CHECK(cudaMemcpyAsync(
                        qw_local.data(), src8,
                        bytes_per_rank_block, cudaMemcpyDeviceToDevice,
                        /*stream=*/0));
                }
            }
            const void* qw_data_for_repack =
                (tp_size == 1) ? qw.data() : qw_local.data();

            // Marlin's repack output is [K/16, N*16/8] int32 = [K/16, N*2]
            // int32 = K*N/2 bytes total. Store as INT4_PACKED u8 with the
            // shape expanded to capture all bytes: [K/16, N*8] u8.
            DeviceTensor packed = DeviceTensor::allocate(
                DType::INT4_PACKED,
                {static_cast<std::int64_t>(Ks_local / 16),
                 static_cast<std::int64_t>(Ns_local * 8)});
            // For AWQ: pre-convert AWQ-format qweight `[K, N/8]` (N-axis
            // packed with `[0,2,4,6,1,3,5,7]` bit interleave) into GPTQ
            // format `[K/8, N]` (K-axis packed, linear bits). Then run
            // the standard gptq_marlin_repack pipeline. Mirrors vLLM's
            // `_convert_awq_tensor_layout` (awq_marlin.py:99-108).
            DeviceTensor awq_to_gptq_scratch;
            const void* repack_input = qw_data_for_repack;
            if (is_awq) {
                awq_to_gptq_scratch = DeviceTensor::allocate(
                    DType::INT32,
                    {static_cast<std::int64_t>(Ks_local / 8),
                     static_cast<std::int64_t>(Ns_local)});
                kernels::launch_awq_qweight_to_gptq_w4(
                    qw_data_for_repack, awq_to_gptq_scratch.data(),
                    Ks_local, Ns_local, /*stream=*/0);
                repack_input = awq_to_gptq_scratch.data();
            }
            marlin::launch_gptq_repack_w4_no_perm(
                repack_input, packed.data(),
                Ks_local, Ns_local, /*stream=*/0);

            // Convert fp16 scales → bf16, slicing per-rank.
            if (sc.dtype() != DType::FP16) {
                throw std::runtime_error(
                    "engine: GPTQ '" + scales_name +
                    "' expected fp16, got " + dtype_name(sc.dtype()));
            }
            DeviceTensor bf16_scales = DeviceTensor::allocate(
                DType::BF16,
                {static_cast<std::int64_t>(sc_groups_local),
                 static_cast<std::int64_t>(Ns_local)});
            if (tp_size == 1) {
                kernels::launch_cast_fp16_to_bf16(
                    sc.data(), bf16_scales.data(), sc.numel(), /*stream=*/0);
            } else if (col_parallel) {
                // Slice axis 1 of [groups, N_full] fp16 -> [groups, N_local] fp16,
                // then cast.
                DeviceTensor sc_local_fp16 = DeviceTensor::allocate(
                    DType::FP16,
                    {static_cast<std::int64_t>(sc_groups_local),
                     static_cast<std::int64_t>(Ns_local)});
                const std::size_t row_bytes_dst =
                    static_cast<std::size_t>(Ns_local) * sizeof(std::uint16_t);
                const std::size_t row_bytes_src =
                    static_cast<std::size_t>(Ns_full) * sizeof(std::uint16_t);
                const auto* src8 = static_cast<const std::uint8_t*>(sc.data());
                src8 += static_cast<std::size_t>(tp_rank) * row_bytes_dst;
                CUDA_CHECK(cudaMemcpy2DAsync(
                    sc_local_fp16.data(), row_bytes_dst,
                    src8, row_bytes_src,
                    row_bytes_dst, sc_groups_local,
                    cudaMemcpyDeviceToDevice, /*stream=*/0));
                kernels::launch_cast_fp16_to_bf16(
                    sc_local_fp16.data(), bf16_scales.data(),
                    sc_local_fp16.numel(), /*stream=*/0);
            } else /* row_parallel */ {
                // Slice axis 0 of [groups_full, N] fp16 -> [groups_local, N] fp16.
                DeviceTensor sc_local_fp16 = DeviceTensor::allocate(
                    DType::FP16,
                    {static_cast<std::int64_t>(sc_groups_local),
                     static_cast<std::int64_t>(Ns_local)});
                const std::size_t bytes_per_rank_block =
                    static_cast<std::size_t>(sc_groups_local) *
                    static_cast<std::size_t>(Ns_local) * sizeof(std::uint16_t);
                const auto* src8 = static_cast<const std::uint8_t*>(sc.data());
                src8 += static_cast<std::size_t>(tp_rank) * bytes_per_rank_block;
                CUDA_CHECK(cudaMemcpyAsync(
                    sc_local_fp16.data(), src8,
                    bytes_per_rank_block, cudaMemcpyDeviceToDevice,
                    /*stream=*/0));
                kernels::launch_cast_fp16_to_bf16(
                    sc_local_fp16.data(), bf16_scales.data(),
                    sc_local_fp16.numel(), /*stream=*/0);
            }
            // Marlin's W4A16 kernel reads scales in a 64-wide column-
            // interleaved layout — apply the same permutation vLLM does
            // (`marlin_permute_scales`) to match.
            kernels::launch_marlin_permute_scales_bf16(
                bf16_scales.data(),
                sc_groups_local, Ns_local, group_size, Ks_local,
                /*stream=*/0);

            // AWQ: process qzeros into marlin's expected layout.
            // GPTQ-sym ignores qzeros (passes nullptr to marlin).
            DeviceTensor marlin_qzeros;
            const std::string canonical_zp = prefix + ".weight_zero_point";
            if (is_awq) {
                const std::string qz_name = prefix + ".qzeros";
                auto qz_it = e.weights_.find(qz_name);
                if (qz_it == e.weights_.end()) {
                    throw std::runtime_error(
                        "engine: AWQ '" + qz_name + "' tensor missing");
                }
                const DeviceTensor& qz = qz_it->second;
                if (qz.dtype() != DType::INT32 || qz.shape().size() != 2) {
                    throw std::runtime_error(
                        "engine: AWQ '" + qz_name +
                        "' expected int32 [groups, N/8]");
                }
                // AWQ qzeros: [groups, N/8] int32. Marlin expects same
                // shape but with the 64-wide column permutation applied.
                marlin_qzeros = DeviceTensor::allocate(DType::INT32, qz.shape());
                kernels::launch_awq_qzero_to_marlin_w4(
                    qz.data(), marlin_qzeros.data(),
                    sc_groups_local, Ns_local, /*stream=*/0);
            }

            bytes_before += qw.nbytes() + sc.nbytes();
            bytes_after  += packed.nbytes() + bf16_scales.nbytes();
            if (marlin_qzeros.numel() > 0) {
                bytes_after += marlin_qzeros.nbytes();
            }

            // Replace under canonical names. Drop the source tensors.
            e.weights_.erase(qw_name);
            e.weights_.erase(scales_name);
            // qzeros / g_idx are unused for GPTQ-sym; AWQ keeps qzeros
            // under the canonical zero-point name.
            e.weights_.erase(prefix + ".qzeros");
            e.weights_.erase(prefix + ".g_idx");
            e.weights_.emplace(canonical_w, std::move(packed));
            e.weights_.emplace(canonical_s, std::move(bf16_scales));
            if (is_awq) {
                e.weights_.emplace(canonical_zp, std::move(marlin_qzeros));
            }

            QuantMeta meta;
            meta.kind         = QuantMeta::Kind::PerGroup;
            meta.scale        = &e.weights_.at(canonical_s);
            meta.zero_point   = is_awq ? &e.weights_.at(canonical_zp) : nullptr;
            meta.group_size   = group_size;
            meta.channel_axis = 0;
            e.set_quant_meta(canonical_w, std::move(meta));

            ++quant_count;
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        const double mib_b = static_cast<double>(bytes_before) / (1024.0 * 1024.0);
        const double mib_a = static_cast<double>(bytes_after) / (1024.0 * 1024.0);
        if (verbose) {
            std::cerr << "[pie-driver-cuda] gptq-int4 marlin: repacked "
                      << quant_count << " projections, " << static_cast<std::uint64_t>(mib_b)
                      << " -> " << static_cast<std::uint64_t>(mib_a)
                      << " MiB (gz=" << group_size << ")\n";
        }
#endif
    } while (false);

    // ── compressed-tensors FP8 (per-channel weight scales) ──────────
    // Detect the pattern: any FP8_E4M3 weight tensor accompanied by a
    // `.weight_scale` tensor (bf16 or fp32). Cast scales to fp32 (the
    // dispatcher's contract) and register QuantMeta::PerChannel. The
    // dispatcher's existing FP8 path will route through cuBLASLt's
    // native FP8 GEMM on sm89+ or the dequant fallback on sm80.
    //
    // RedHatAI's mistral3-FP8-dynamic uses a `language_model.` prefix
    // for multimodal models; that prefix-stripping is a separate
    // engine concern and not handled here. Text-only compressed-
    // tensors FP8 ckpts (e.g. DeepSeek-V3-style) flow through this
    // path directly.
    if (e.hf_.quant_method == "compressed-tensors") {
        std::vector<std::string> fp8_names;
        for (const auto& [n, t] : e.weights_) {
            if (t.dtype() == DType::FP8_E4M3 &&
                ends_with_str(n, ".weight")) fp8_names.push_back(n);
        }
        std::size_t ct_count = 0;
        for (const auto& wname : fp8_names) {
            const std::string scale_name = wname + "_scale";
            auto sc_it = e.weights_.find(scale_name);
            if (sc_it == e.weights_.end()) continue;
            const DeviceTensor& sc = sc_it->second;
            // Per-channel scale shape is `[N, 1]` or `[N]`. Cast bf16
            // / fp32 → fp32 and squeeze to a flat [N] tensor.
            const std::size_t nelems = sc.numel();
            DeviceTensor fp32_scale = DeviceTensor::allocate(
                DType::FP32, {static_cast<std::int64_t>(nelems)});
            if (sc.dtype() == DType::BF16) {
                kernels::launch_cast_bf16_to_fp32(
                    sc.data(), fp32_scale.data(), nelems, /*stream=*/0);
            } else if (sc.dtype() == DType::FP32) {
                CUDA_CHECK(cudaMemcpyAsync(
                    fp32_scale.data(), sc.data(),
                    nelems * sizeof(float), cudaMemcpyDeviceToDevice,
                    /*stream=*/0));
            } else {
                throw std::runtime_error(
                    "engine: compressed-tensors FP8 '" + scale_name +
                    "' has unexpected scale dtype " +
                    std::string(dtype_name(sc.dtype())));
            }
            // Replace under canonical name.
            const std::string canonical_s = wname + "_scale_inv";
            e.weights_.erase(scale_name);
            e.weights_.emplace(canonical_s, std::move(fp32_scale));
            // Per-tensor (scalar scale) vs per-channel ([N] scale).
            // RedHatAI/Meta-Llama-3.1-8B-FP8 ships static per-tensor;
            // mistral3-FP8-dynamic ships per-channel.
            QuantMeta meta;
            meta.kind         = (nelems == 1)
                                   ? QuantMeta::Kind::PerTensor
                                   : QuantMeta::Kind::PerChannel;
            meta.scale        = &e.weights_.at(canonical_s);
            meta.zero_point   = nullptr;
            meta.group_size   = 0;
            meta.channel_axis = 0;
            e.set_quant_meta(wname, std::move(meta));
            ++ct_count;
        }
        if (ct_count > 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
            if (verbose) {
                std::cerr << "[pie-driver-cuda] compressed-tensors FP8: "
                          << "registered " << ct_count
                          << " quantised weights\n";
            }
        } else {
            // compressed-tensors covers FP8 / INT8 / W4A16 / etc. We
            // only handle the FP8 sub-mode (weights stored as FP8_E4M3
            // with `_scale` companions). If we found no such weights,
            // the ckpt is using a sub-mode we don't recognise — refuse
            // loudly rather than silently producing bf16 with orphan
            // `_scale` tensors that the forward path will misinterpret.
            std::vector<std::string> orphan_scales;
            for (const auto& [n, _t] : e.weights_) {
                if (ends_with_str(n, "_scale") ||
                    ends_with_str(n, "_zero_point")) {
                    orphan_scales.push_back(n);
                    if (orphan_scales.size() >= 3) break;
                }
            }
            if (!orphan_scales.empty()) {
                std::string ex = orphan_scales[0];
                for (std::size_t i = 1; i < orphan_scales.size(); ++i) {
                    ex += ", " + orphan_scales[i];
                }
                throw std::runtime_error(
                    "engine: quant_method=compressed-tensors, but no "
                    "FP8_E4M3 weights with `_scale` companions found. "
                    "Got scale-like tensors (e.g. " + ex + ") suggesting "
                    "a non-FP8 sub-mode (INT8 / W4A16 / GPTQ-style). "
                    "Only the FP8 (`format=float-quantized`) sub-mode is "
                    "currently supported.");
            }
        }
    }

    // ── fp16/fp32 → bf16 sweep ────────────────────────────────────────
    // After GPTQ repack and any other dtype-aware passes, every fp16 or
    // fp32 tensor still in `weights_` (norms, biases, embed, lm_head,
    // etc.) needs to be cast to bf16 — our forward kernels are bf16-
    // only. Skip already-converted entries (BF16/INT*/FP8/etc). Some
    // GPTQ-Int4 ckpts (e.g. Qwen2-1.5B-GPTQ) ship embed_tokens / lm_head
    // as fp32 to preserve precision; we cast them down to bf16 here.
    {
        std::vector<std::string> cast_names;
        for (const auto& [n, t] : e.weights_) {
            if (t.dtype() == DType::FP16 || t.dtype() == DType::FP32) {
                // Quant scales are intentionally fp32 (cuBLASLt's
                // A_SCALE_POINTER attr requires fp32). They were either
                // loaded as fp32 or cast from bf16 by the compressed-
                // tensors pass; preserve them.
                if (ends_with_str(n, "_scale_inv")) continue;
                cast_names.push_back(n);
            }
        }
        if (!cast_names.empty()) {
            for (const auto& n : cast_names) {
                auto it = e.weights_.find(n);
                if (it == e.weights_.end()) continue;
                const DeviceTensor& src = it->second;
                DeviceTensor bf16 = DeviceTensor::allocate(DType::BF16, src.shape());
                if (src.dtype() == DType::FP16) {
                    kernels::launch_cast_fp16_to_bf16(
                        src.data(), bf16.data(), src.numel(), /*stream=*/0);
                } else if (src.dtype() == DType::FP32) {
                    kernels::launch_cast_fp32_to_bf16(
                        src.data(), bf16.data(), src.numel(), /*stream=*/0);
                } else {
                    continue;
                }
                e.weights_.erase(it);
                e.weights_.emplace(n, std::move(bf16));
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            if (verbose) {
                std::cerr << "[pie-driver-cuda] cast " << cast_names.size()
                          << " tensors {fp16,fp32} -> bf16 to match the "
                          << "GEMM pipeline\n";
            }
        }
    }

    // ── Runtime quantization ──────────────────────────────────────────
    // When `boot_cfg.model.runtime_quant == "fp8"`, walk every standard
    // llama-like projection weight (q/k/v/o/gate/up/down) and replace it
    // with a FP8_E4M3 tensor + per-tensor scale. The forward path picks
    // up the QuantMeta companion via LoadedModel::quant_meta and routes
    // through cuBLASLt FP8 GEMM. v1 restrictions:
    //   * Only model_type=qwen3 is wired through the new GEMM dispatch
    //     (qwen3_forward.cpp). Other archs still use the bf16-only
    //     shim and would break on FP8 weights.
    //   * Phi-3's fused qkv_proj has live views; refuse runtime quant
    //     until the fused-weight handling lands (M1).
    if (!boot_cfg.model.runtime_quant.empty()) do {
        const auto& mode = boot_cfg.model.runtime_quant;
        const bool is_fp8  = (mode == "fp8");
        const bool is_int8 = (mode == "int8");
        if (!is_fp8 && !is_int8) {
            throw std::runtime_error(
                "engine: unsupported runtime_quant '" + mode +
                "'. Currently supported: 'fp8' or 'int8' (per-channel "
                "symmetric W8A8 in both cases).");
        }
        const std::string& mt = e.hf_.model_type;
        const bool model_supports = (mt == "qwen3")
                                 || (mt == "qwen2")
                                 || (mt == "llama") || (mt == "llama3")
                                 || (mt == "mistral")
                                 || (mt == "qwen3_5") || (mt == "qwen3_5_text");
        if (!model_supports) {
            throw std::runtime_error(
                "engine: runtime_quant=" + mode + " is currently wired "
                "through the GEMM dispatcher only for {qwen2, qwen3, "
                "qwen3_5, qwen3_5_text, llama, llama3, mistral} (got '" +
                mt + "'). Other archs will be enabled as their forward "
                "paths migrate to ops::gemm_act_x_w.");
        }
        if (tp_size > 1 && tp_comm == nullptr) {
            throw std::runtime_error(
                "engine: runtime_quant=" + mode + " with tp_size>1 "
                "requires a NcclComm (internal: caller must pass tp_comm "
                "to LoadedModel::load).");
        }

        // On sm<89 there's no native FP8 GEMM, so the eager-dequant pass
        // below would convert every FP8 weight straight back to bf16 —
        // the round-trip is pure waste. Skip the quantization entirely
        // and warn so the user knows they got bf16-equivalent perf
        // without the memory savings (FP8 only halves weight footprint
        // on hardware where it stays packed end-to-end).
        //
        // INT8 is different — sm80 has native INT8 tensor cores, so the
        // W8A8 path is the real Ampere quant perf win.
        if (is_fp8 && !fp8_native) {
            std::cerr << "[pie-driver-cuda] runtime_quant=fp8 skipped: "
                      << "sm" << dev_prop.major << dev_prop.minor
                      << " has no native FP8 GEMM. Weights stay bf16 "
                      << "(use runtime_quant=int8 or marlin Int4 / GPTQ "
                      << "for memory + perf wins on this generation).\n";
            break;
        }

        // Snapshot tensor names first — we mutate `weights_` inside the
        // loop, which would invalidate any iterator we held.
        std::vector<std::string> names;
        names.reserve(e.weights_.size());
        for (const auto& [n, _] : e.weights_) names.push_back(n);

        auto ends_with = [&](const std::string& s, const char* sfx) {
            const auto n = std::char_traits<char>::length(sfx);
            return s.size() >= n &&
                   s.compare(s.size() - n, n, sfx) == 0;
        };
        auto is_quantizable_proj = [&](const std::string& n) {
            return ends_with(n, ".q_proj.weight")
                || ends_with(n, ".k_proj.weight")
                || ends_with(n, ".v_proj.weight")
                || ends_with(n, ".o_proj.weight")
                || ends_with(n, ".gate_proj.weight")
                || ends_with(n, ".up_proj.weight")
                || ends_with(n, ".down_proj.weight");
        };
        // Row-parallel weights are sharded along the input axis (K). Their
        // per-row absmax must be reduced across ranks so every rank picks
        // the same scale; column-parallel weights (q/k/v/gate/up) are
        // sharded along the output axis (N) and each rank's slice has a
        // distinct row range, so its local absmax is already the right one.
        auto is_row_parallel = [&](const std::string& n) {
            return ends_with(n, ".o_proj.weight")
                || ends_with(n, ".down_proj.weight");
        };

        std::size_t quantized_count = 0;
        std::uint64_t bytes_before = 0;
        std::uint64_t bytes_after  = 0;

        for (const auto& name : names) {
            if (!is_quantizable_proj(name)) continue;
            auto it = e.weights_.find(name);
            if (it == e.weights_.end()) continue;
            const DeviceTensor& src = it->second;
            if (src.dtype() != DType::BF16) continue;
            // 2-D weights only — `[N, K]` row-major. The MoE 3-D fused
            // experts stay bf16 for now (M2 follow-up).
            if (src.shape().size() != 2) continue;
            const auto rows = static_cast<int>(src.shape()[0]);
            const auto cols = static_cast<int>(src.shape()[1]);

            // Allocate FP8 buffer with the same shape (1 byte/elem vs 2)
            // and an `[N]` per-channel scale buffer. Per-channel keeps
            // outliers in one row from dragging the whole tensor's scale
            // off; on H100 cuBLASLt consumes the vector directly via
            // `CUBLASLT_MATMUL_DESC_A_SCALE_VECTOR_POINTER`, on A100 the
            // dequant fallback broadcasts row-wise.
            //
            // Two-stage path for TP correctness: stage 1 computes per-row
            // absmax into `scale.data()` (treated as a scratch buffer);
            // for row-parallel weights we then all-reduce-MAX across
            // ranks so every rank picks the same scale; stage 2 turns
            // absmax into weight_scale_inv and stage 3 casts bf16 → fp8
            // using that scale.
            const DType q_dtype = is_int8 ? DType::INT8 : DType::FP8_E4M3;
            DeviceTensor q = DeviceTensor::allocate(q_dtype, src.shape());
            DeviceTensor scale = DeviceTensor::allocate(DType::FP32, {rows});
            float* scale_ptr = static_cast<float*>(scale.data());

            kernels::launch_absmax_per_row_bf16(
                src.data(), scale_ptr, rows, cols, /*stream=*/0);
            if (tp_size > 1 && is_row_parallel(name)) {
                tp_comm->all_reduce_fp32(
                    scale_ptr,
                    static_cast<std::size_t>(rows),
                    ncclMax, /*stream=*/0);
            }
            if (is_int8) {
                kernels::launch_absmax_to_scale_inv_int8(
                    scale_ptr, rows, /*stream=*/0);
                kernels::launch_cast_bf16_to_int8_per_channel(
                    src.data(),
                    static_cast<std::int8_t*>(q.data()),
                    scale_ptr, rows, cols, /*stream=*/0);
            } else {
                kernels::launch_absmax_to_scale_inv(
                    scale_ptr, rows, /*stream=*/0);
                kernels::launch_cast_bf16_to_fp8_e4m3_per_channel(
                    src.data(),
                    static_cast<std::uint8_t*>(q.data()),
                    scale_ptr, rows, cols, /*stream=*/0);
            }

            bytes_before += src.nbytes();
            bytes_after  += q.nbytes() + scale.nbytes();

            e.weights_.erase(it);
            e.weights_.emplace(name, std::move(q));
            const std::string scale_name = name + "_scale_inv";
            e.weights_.emplace(scale_name, std::move(scale));

            QuantMeta meta;
            meta.kind         = QuantMeta::Kind::PerChannel;
            meta.scale        = &e.weights_.at(scale_name);
            meta.zero_point   = nullptr;
            meta.group_size   = 0;
            meta.channel_axis = 0;
            e.set_quant_meta(name, std::move(meta));

            ++quantized_count;
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        const double mib_before = static_cast<double>(bytes_before) / (1024.0 * 1024.0);
        const double mib_after  = static_cast<double>(bytes_after)  / (1024.0 * 1024.0);
        if (verbose) {
            std::cerr << "[pie-driver-cuda] runtime_quant=" << mode
                      << " quantised " << quantized_count << " projections: "
                      << static_cast<std::uint64_t>(mib_before) << " -> "
                      << static_cast<std::uint64_t>(mib_after) << " MiB ("
                      << static_cast<int>(100.0 * mib_after / std::max(mib_before, 1.0))
                      << "% of original)\n";
        }
    } while (false);

    // ── sm<89 eager FP8 → bf16 dequant ────────────────────────────────
    // On Ampere (sm80) and earlier, cuBLASLt has no native FP8 GEMM. The
    // dispatcher falls back to "dequant scratch then bf16 GEMM", which
    // re-materialises the full bf16 weight from FP8 on every layer of
    // every step (≈ 2.5× HBM traffic per GEMM). Walk every FP8 weight
    // here, dequant once, replace with bf16, drop the QuantMeta — then
    // the dispatcher takes the plain bf16 path. Memory savings are lost
    // (FP8 was supposed to halve weight memory) but decode throughput
    // recovers to bf16-baseline. On sm89+ this pass is a no-op.
    if (!fp8_native) {
        std::vector<std::string> fp8_weights;
        for (const auto& [n, t] : e.weights_) {
            if (t.dtype() == DType::FP8_E4M3) fp8_weights.push_back(n);
        }
        if (!fp8_weights.empty()) {
            std::uint64_t freed = 0, allocated = 0;
            // Track which `_scale_inv` companions we actually consumed so
            // we only erase those. Non-compressed-tensors FP8 ckpts
            // (Ministral-3-Instruct-2512: quant_method="fp8" with
            // bf16 scalar `weight_scale_inv` per projection) leave their
            // FP8 weights without QuantMeta — the per-arch bind dequants
            // them later using the still-resident scale tensor. Blanket-
            // erasing every `_scale_inv` here would strip those scales
            // before the bind runs.
            std::vector<std::string> consumed_scales;
            consumed_scales.reserve(fp8_weights.size());
            for (const auto& wname : fp8_weights) {
                auto qmeta_it = e.quant_meta_.find(wname);
                if (qmeta_it == e.quant_meta_.end() ||
                    !qmeta_it->second.scale) continue;
                const auto& qmeta = qmeta_it->second;
                auto w_it = e.weights_.find(wname);
                const DeviceTensor& src = w_it->second;
                DeviceTensor bf16 = DeviceTensor::allocate(
                    DType::BF16, src.shape());
                if (qmeta.kind == QuantMeta::Kind::PerChannel) {
                    const auto rows = static_cast<int>(src.shape()[0]);
                    const auto cols = static_cast<int>(src.numel() / rows);
                    kernels::launch_dequant_fp8_e4m3_to_bf16_per_channel(
                        static_cast<const std::uint8_t*>(src.data()),
                        bf16.data(),
                        static_cast<const float*>(qmeta.scale->data()),
                        rows, cols, /*stream=*/0);
                } else {
                    // Per-tensor: pull the scalar to host once.
                    float scale = 0.f;
                    CUDA_CHECK(cudaMemcpy(&scale, qmeta.scale->data(),
                        sizeof(float), cudaMemcpyDeviceToHost));
                    kernels::launch_dequant_fp8_e4m3_to_bf16(
                        static_cast<const std::uint8_t*>(src.data()),
                        bf16.data(), scale, src.numel(), /*stream=*/0);
                }
                freed     += src.nbytes();
                allocated += bf16.nbytes();
                w_it->second = std::move(bf16);
                consumed_scales.push_back(wname + "_scale_inv");
                e.quant_meta_.erase(qmeta_it);
            }
            // Drop only the `_scale_inv` companions we actually consumed.
            for (const auto& s : consumed_scales) {
                e.weights_.erase(s);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            if (verbose) {
                std::cerr << "[pie-driver-cuda] eager FP8 -> bf16 dequant "
                          << "(sm" << dev_prop.major << dev_prop.minor
                          << ", no native FP8 GEMM): " << fp8_weights.size()
                          << " weights, " << (freed / (1024 * 1024))
                          << " MiB FP8 -> " << (allocated / (1024 * 1024))
                          << " MiB bf16\n";
            }
            loaded_bytes = loaded_bytes - freed + allocated;
        }
    }

    // ── MoE per-expert → fused 3-D synthesis ──────────────────────────
    // Some compressed-tensors MoE ckpts (e.g. RedHatAI/Qwen3.6-35B-A3B-
    // FP8-dynamic) ship per-expert 2-D weights:
    //   experts.{i}.gate_proj.weight  [Im, H]
    //   experts.{i}.up_proj.weight    [Im, H]
    //   experts.{i}.down_proj.weight  [H, Im]
    // The qwen3_5_moe forward expects fused 3-D tensors:
    //   experts.gate_up_proj  [E, 2*Im, H]   (gate stacked above up)
    //   experts.down_proj     [E, H, Im]
    // Synthesize them after eager-dequant so we can keep the existing
    // forward path unchanged. Each per-expert tensor is already TP-
    // sharded along Im (via llama_like_shard_axis matching .gate_proj/
    // .up_proj/.down_proj.weight) so the fused result lands per-rank-
    // shaped without further slicing.
    if (is_qwen3_5_moe_arch(e.hf_.model_type)) {
        // Discover layer prefixes that have per-expert tensors.
        std::vector<std::string> layer_prefixes;
        {
            std::set<std::string> uniq;
            const char* suffix = ".experts.0.gate_proj.weight";
            const std::size_t suffix_len = std::strlen(suffix);
            for (const auto& [n, _] : e.weights_) {
                if (ends_with_str(n, suffix)) {
                    uniq.insert(n.substr(0, n.size() - suffix_len));
                }
            }
            layer_prefixes.assign(uniq.begin(), uniq.end());
        }
        if (!layer_prefixes.empty()) {
            std::size_t fused_count = 0;
            for (const auto& base : layer_prefixes) {
                // Count experts: walk i=0,1,... until missing.
                int E_count = 0;
                for (;;) {
                    if (e.weights_.find(base + ".experts." +
                            std::to_string(E_count) +
                            ".gate_proj.weight") == e.weights_.end()) break;
                    ++E_count;
                }
                if (E_count == 0) continue;
                // Reference shapes from expert 0.
                const auto& gp0 = e.weights_.at(
                    base + ".experts.0.gate_proj.weight");
                const auto& dp0 = e.weights_.at(
                    base + ".experts.0.down_proj.weight");
                const int Im = static_cast<int>(gp0.shape()[0]);
                const int H  = static_cast<int>(gp0.shape()[1]);
                const int Im_dn = static_cast<int>(dp0.shape()[1]);
                const std::size_t gu_per_expert = static_cast<std::size_t>(2)
                    * Im * H * sizeof(std::uint16_t);
                const std::size_t dn_per_expert = static_cast<std::size_t>(H)
                    * Im_dn * sizeof(std::uint16_t);

                DeviceTensor gu = DeviceTensor::allocate(
                    DType::BF16,
                    {static_cast<std::int64_t>(E_count),
                     static_cast<std::int64_t>(2 * Im),
                     static_cast<std::int64_t>(H)});
                DeviceTensor dn = DeviceTensor::allocate(
                    DType::BF16,
                    {static_cast<std::int64_t>(E_count),
                     static_cast<std::int64_t>(H),
                     static_cast<std::int64_t>(Im_dn)});

                for (int i = 0; i < E_count; ++i) {
                    const std::string ep = base + ".experts." +
                        std::to_string(i);
                    const auto& gw = e.weights_.at(ep + ".gate_proj.weight");
                    const auto& uw = e.weights_.at(ep + ".up_proj.weight");
                    const auto& dw = e.weights_.at(ep + ".down_proj.weight");
                    auto* gu_base = static_cast<std::uint8_t*>(gu.data())
                        + i * gu_per_expert;
                    // gate: rows [0, Im), then up: rows [Im, 2*Im).
                    CUDA_CHECK(cudaMemcpyAsync(
                        gu_base, gw.data(),
                        static_cast<std::size_t>(Im) * H * sizeof(std::uint16_t),
                        cudaMemcpyDeviceToDevice, /*stream=*/0));
                    CUDA_CHECK(cudaMemcpyAsync(
                        gu_base + Im * H * sizeof(std::uint16_t), uw.data(),
                        static_cast<std::size_t>(Im) * H * sizeof(std::uint16_t),
                        cudaMemcpyDeviceToDevice, /*stream=*/0));
                    CUDA_CHECK(cudaMemcpyAsync(
                        static_cast<std::uint8_t*>(dn.data()) + i * dn_per_expert,
                        dw.data(), dn_per_expert,
                        cudaMemcpyDeviceToDevice, /*stream=*/0));
                }
                // Erase per-expert sources, install fused.
                for (int i = 0; i < E_count; ++i) {
                    const std::string ep = base + ".experts." +
                        std::to_string(i);
                    e.weights_.erase(ep + ".gate_proj.weight");
                    e.weights_.erase(ep + ".up_proj.weight");
                    e.weights_.erase(ep + ".down_proj.weight");
                }
                e.weights_.emplace(base + ".experts.gate_up_proj", std::move(gu));
                e.weights_.emplace(base + ".experts.down_proj", std::move(dn));
                ++fused_count;
            }
            if (fused_count > 0) {
                CUDA_CHECK(cudaDeviceSynchronize());
                if (verbose) {
                    std::cerr << "[pie-driver-cuda] MoE fuse: synthesized "
                              << fused_count << " (gate_up_proj, down_proj) "
                              << "fused 3-D tensors from per-expert 2-D\n";
                }
            }
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double mib = static_cast<double>(loaded_bytes) / (1024.0 * 1024.0);

    if (verbose) {
        std::cerr << "[pie-driver-cuda] loaded " << e.weights_.size() << " tensors ("
                  << static_cast<std::uint64_t>(mib) << " MiB on this rank, "
                  << "tp=" << tp_size << ") in " << static_cast<int>(ms)
                  << " ms; arch=" << e.hf_.arch_name << " (" << e.hf_.model_type << ")\n";
    }

    return e;
}

LoadedModelCapabilities LoadedModel::capabilities() const {
    LoadedModelCapabilities c;
    c.total_pages = 0;  // populated in M1.2.2 once kv_cache lands
    c.kv_page_size = static_cast<int>(boot_.batching.kv_page_size);
    c.swap_pool_size = 0;
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

std::uint64_t LoadedModel::total_weight_bytes() const noexcept {
    std::uint64_t n = 0;
    for (const auto& [_, t] : weights_) n += t.nbytes();
    return n;
}

const DeviceTensor& LoadedModel::get(const std::string& name) const {
    auto it = weights_.find(name);
    if (it == weights_.end()) {
        throw std::runtime_error("engine: weight not loaded: " + name);
    }
    return it->second;
}

void LoadedModel::insert(std::string name, DeviceTensor tensor) {
    auto [it, inserted] = weights_.emplace(std::move(name), std::move(tensor));
    if (!inserted) {
        throw std::runtime_error("engine: weight already registered: " + it->first);
    }
}

void LoadedModel::replace(std::string name, DeviceTensor tensor) {
    weights_.insert_or_assign(std::move(name), std::move(tensor));
}

void LoadedModel::set_quant_meta(const std::string& name, QuantMeta meta) {
    if (weights_.find(name) == weights_.end()) {
        throw std::runtime_error(
            "engine::set_quant_meta: weight '" + name + "' not registered");
    }
    if (!meta.scale) {
        throw std::runtime_error(
            "engine::set_quant_meta: scale must be non-null for '" + name + "'");
    }
    auto [it, inserted] = quant_meta_.emplace(name, std::move(meta));
    if (!inserted) {
        throw std::runtime_error(
            "engine::set_quant_meta: meta already attached to '" + name + "'");
    }
}

std::optional<QuantMeta> LoadedModel::quant_meta(const std::string& name) const {
    auto it = quant_meta_.find(name);
    if (it == quant_meta_.end()) return std::nullopt;
    return it->second;
}

}  // namespace pie_cuda_driver
