#include "ops/flashinfer_moe.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_bf16.h>

#include "cutlass_fused_moe_kernels.cuh"
#include "ops/tuning_cache.hpp"

namespace pie_cuda_driver::ops {
namespace {

namespace ck = tensorrt_llm::kernels::cutlass_kernels;
namespace ce = tensorrt_llm::cutlass_extensions;
namespace tk = tensorrt_llm::kernels;

using Runner = ck::CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>;

bool env_truthy(const char* value) {
    if (value == nullptr || value[0] == '\0') return false;
    return value[0] == '1' || value[0] == 'y' || value[0] == 'Y' ||
           value[0] == 't' || value[0] == 'T' || value[0] == 'o' ||
           value[0] == 'O';
}

// Tactics are held as indices into the runner's candidate lists, not as
// configs, so that a tuning result is a pair of small integers and can be
// written to (and matched against) an on-disk cache.
struct TacticPair {
    int gemm1 = -1;
    int gemm2 = -1;
};

struct RunnerState {
    std::once_flag init_once;
    std::unique_ptr<Runner> runner;
    bool ready = false;
    std::exception_ptr init_error;
    // Populated by get_runner(); the full candidate lists plus the shape-blind
    // default pair, so the autotuner does not have to re-query them.
    std::vector<ce::CutlassGemmConfig> gemm1_tactics;
    std::vector<ce::CutlassGemmConfig> gemm2_tactics;
    TacticPair default_tactics{};
    std::mutex tune_mutex;
    std::unordered_map<std::uint64_t, TacticPair> tuned;
};

RunnerState& state() {
    constexpr int kMaxCudaDevices = 16;
    static std::array<RunnerState, kMaxCudaDevices> states;
    int device = 0;
    cudaError_t st = cudaGetDevice(&device);
    if (st != cudaSuccess) {
        throw std::runtime_error(
            std::string("flashinfer CUTLASS MoE: cudaGetDevice failed: ") +
            cudaGetErrorString(st));
    }
    if (device < 0 || device >= kMaxCudaDevices) {
        throw std::runtime_error(
            "flashinfer CUTLASS MoE: CUDA device index exceeds runner cache");
    }
    return states[device];
}

constexpr bool log_enabled() { return false; }

std::optional<ce::CutlassGemmConfig> select_first_profile(
    const std::vector<ce::CutlassGemmConfig>& configs,
    const char* name,
    int* out_index) {
    if (configs.empty()) return std::nullopt;
    *out_index = 0;
    if (log_enabled()) {
        std::fprintf(
            stderr,
            "[pie-driver-cuda] FlashInfer MoE %s selected first profile "
            "total=%zu\n",
            name, configs.size());
    }
    return configs.front();
}

std::optional<ce::CutlassGemmConfig> first_supported(
    Runner& runner,
    const std::vector<ce::CutlassGemmConfig>& configs,
    std::optional<ce::CutlassGemmConfig::EpilogueFusionType> fusion,
    int supported_index,
    const char* name,
    int* out_index) {
    int seen = 0;
    int supported = 0;
    int index = -1;
    int selected_index = -1;
    std::optional<ce::CutlassGemmConfig> selected;
    for (const auto& cfg : configs) {
        ++index;
        if (fusion && cfg.epilogue_fusion_type != *fusion) continue;
        ++seen;
        if (runner.queryOccupancyForConfig(cfg) > 0) {
            if (supported == supported_index) {
                selected = cfg;
                selected_index = index;
                *out_index = index;
            }
            ++supported;
        }
    }
    if (log_enabled()) {
        if (selected) {
            std::fprintf(
                stderr,
                "[pie-driver-cuda] FlashInfer MoE %s selected "
                "supported_index=%d raw_index=%d supported=%d seen=%d total=%zu\n",
                name, supported_index, selected_index, supported, seen, configs.size());
        } else {
            std::fprintf(
                stderr,
                "[pie-driver-cuda] FlashInfer MoE %s no tactic for "
                "supported_index=%d supported=%d seen=%d total=%zu\n",
                name, supported_index, supported, seen, configs.size());
        }
    }
    return selected;
}

std::optional<ce::CutlassGemmConfig> select_raw_profile(
    const std::vector<ce::CutlassGemmConfig>& configs,
    int raw_index,
    const char* name,
    int* out_index) {
    if (configs.empty()) return std::nullopt;
    const int index = std::min(
        std::max(0, raw_index),
        static_cast<int>(configs.size()) - 1);
    *out_index = index;
    if (log_enabled()) {
        std::fprintf(
            stderr,
            "[pie-driver-cuda] FlashInfer MoE %s selected raw_index=%d "
            "total=%zu\n",
            name, index, configs.size());
    }
    return configs[static_cast<std::size_t>(index)];
}

const char* fusion_name(ce::CutlassGemmConfig::EpilogueFusionType fusion) {
    switch (fusion) {
    case ce::CutlassGemmConfig::EpilogueFusionType::NONE: return "none";
    case ce::CutlassGemmConfig::EpilogueFusionType::FINALIZE: return "finalize";
    }
    return "unknown";
}

// The tile/cluster enums encode their shape as base-1000 digits, so they are
// unreadable as raw integers. `Undefined`(0) and `ChooseWithHeuristic`(1) are
// not shapes and are reported as-is.
std::string shape_str(int id) {
    if (id == 0) return "undef";
    if (id == 1) return "heuristic";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%dx%dx%d",
                  id / 1000000, (id % 1000000) / 1000, id % 1000);
    return std::string(buf);
}

// `sm_version` decides which of the three tile fields is live; printing the
// wrong one shows a constant `heuristic` for every tactic.
std::string tile_str(const ce::CutlassGemmConfig& cfg) {
    if (cfg.sm_version >= 120) return shape_str(static_cast<int>(cfg.tile_config_sm120));
    if (cfg.sm_version >= 100) return shape_str(static_cast<int>(cfg.tile_config_sm100));
    if (cfg.sm_version >= 90) return shape_str(static_cast<int>(cfg.tile_config_sm90));
    return shape_str(static_cast<int>(cfg.tile_config_sm80));
}

std::string config_str(const ce::CutlassGemmConfig& cfg) {
    char buf[320];
    std::snprintf(
        buf, sizeof(buf),
        "fusion=%s tma=%d swap_ab=%d sm=%d tile=%s mainloop=%d epilogue=%d "
        "cluster=%s dyn_cluster=%s fallback_cluster=%s split_k=%d stages=%d",
        fusion_name(cfg.epilogue_fusion_type),
        cfg.is_tma_warp_specialized ? 1 : 0,
        cfg.swap_ab ? 1 : 0,
        cfg.sm_version,
        tile_str(cfg).c_str(),
        static_cast<int>(cfg.mainloop_schedule),
        static_cast<int>(cfg.epilogue_schedule),
        shape_str(static_cast<int>(cfg.cluster_shape)).c_str(),
        shape_str(static_cast<int>(cfg.dynamic_cluster_shape)).c_str(),
        shape_str(static_cast<int>(cfg.fallback_cluster_shape)).c_str(),
        cfg.split_k_factor,
        cfg.stages);
    return std::string(buf);
}

void log_config(const char* name, const ce::CutlassGemmConfig& cfg) {
    if (!log_enabled()) return;
    std::fprintf(stderr, "[pie-driver-cuda] FlashInfer MoE %s tactic: %s\n",
                 name, config_str(cfg).c_str());
}

Runner& get_runner() {
    RunnerState& s = state();
    std::call_once(s.init_once, [&] {
        try {
            auto runner = std::make_unique<Runner>();
            auto gemm1 = runner->getTactics(ck::MoeGemmId::GEMM_1);
            auto gemm2 = runner->getTactics(ck::MoeGemmId::GEMM_2);
            TacticPair defaults{};
            std::optional<ce::CutlassGemmConfig> best_gemm1;
            std::optional<ce::CutlassGemmConfig> best_gemm2;
            // Default: first tactic the runner reports as occupancy-viable,
            // with the plain (NONE) epilogue on GEMM2.
            //
            // The alternative, FINALIZE, folds the topk-weighted reduction
            // and the unpermute into the GEMM epilogue, and in isolation it
            // is faster: 147.0 us against 174.6 us at GLM's shapes (M=128,
            // H=6144, I=2048, E=8, topk=8). It is not the default anyway,
            // because of how it performs that reduction. Each expert's
            // contribution to a token is committed with
            // `red.global.add.noftz.bf16x2` -- a hardware reduction-add
            // straight to global memory, in bf16 (see
            // cutlass_extensions/arch/copy_red_global.hpp). So the topk sum
            //
            //   * accumulates in bf16, rounding after each of the topk
            //     terms, where the unfused path accumulates in fp32, and
            //   * adds them in whatever order the CTAs finish in, which is
            //     not the same order twice.
            //
            // The second point makes the whole engine irreproducible: the
            // same prompt decodes to different tokens on different runs
            // whenever two logits land within the resulting ~1 ulp. That
            // was worth paying for a 16% cut of GEMM2 if it showed up end
            // to end -- it does not. Median of 5 runs on glm5.2-mini, 256
            // output tokens: 3518.8 vs 3515.8 tok/s at c=8 and 36824.6 vs
            // 36931.3 at c=128, i.e. a tie at both ends, because GEMM2 is a
            // small enough slice of the step that 27 us/layer disappears
            // into it. Determinism and an fp32 reduction for nothing.
            //
            best_gemm1 = first_supported(
                *runner, gemm1, std::nullopt, 0, "GEMM1", &defaults.gemm1);
            best_gemm2 = first_supported(
                *runner, gemm2,
                ce::CutlassGemmConfig::EpilogueFusionType::NONE,
                0, "GEMM2", &defaults.gemm2);
            if (!best_gemm2) {
                best_gemm2 = first_supported(
                    *runner, gemm2, std::nullopt, 0, "GEMM2", &defaults.gemm2);
            }
            if (!best_gemm1 || !best_gemm2) {
                throw std::runtime_error(
                    "flashinfer CUTLASS MoE: no supported BF16 tactics");
            }
            log_config("GEMM1", *best_gemm1);
            log_config("GEMM2", *best_gemm2);
            runner->setTactic(best_gemm1, best_gemm2);
            s.gemm1_tactics = std::move(gemm1);
            s.gemm2_tactics = std::move(gemm2);
            if (defaults.gemm1 < 0 || defaults.gemm2 < 0) {
                throw std::runtime_error(
                    "flashinfer CUTLASS MoE: default tactic has no index");
            }
            s.default_tactics = defaults;
            s.runner = std::move(runner);
            s.ready = true;
        } catch (...) {
            s.init_error = std::current_exception();
        }
    });

    if (!s.ready) {
        if (s.init_error) std::rethrow_exception(s.init_error);
        throw std::runtime_error("flashinfer CUTLASS MoE: runner not initialized");
    }
    return *s.runner;
}

ck::ActivationType to_cutlass_activation(MoeActivation a) {
    switch (a) {
        case MoeActivation::Swiglu: return ck::ActivationType::Swiglu;
        // Upstream's `Geglu` dispatches to `EpilogueOpDefaultFtGelu` and is
        // accepted by `supportsFusedGatedActivation`; `GegluTanh` is neither,
        // so it would drop to the unfused gate for the same math.
        case MoeActivation::Geglu:  return ck::ActivationType::Geglu;
        case MoeActivation::Relu2:
        default:                    return ck::ActivationType::Relu2;
    }
}

ck::MOEParallelismConfig parallelism_config(int tp_size, int tp_rank) {
    return ck::MOEParallelismConfig(std::max(1, tp_size), tp_rank, 1, 0);
}

// ---- Shape-aware tactic selection ----------------------------------------
//
// `getTactics` returns 109 GEMM1 and 209 GEMM2 candidates, and on SM100 every
// one of them reports positive occupancy -- so "first occupancy-viable" is
// really just "candidate 0", regardless of the problem. That is the wrong tile
// for almost every shape we run: a decode step is M=8 against N=4096, where a
// 256-row CTA tile wastes 97% of its rows, while a prefill at M=1024 wants the
// widest tile available. CUTLASS ships no shape heuristic for grouped GEMM
// here (`ChooseWithHeuristic` is not among the returned configs), so measure.
//
// `setTactic` is two pointer stores, so the tactic can be chosen per call. We
// key on the problem shape, bucket M by power of two, and tune once per key by
// coordinate descent (sweep GEMM1 with GEMM2 fixed, then the reverse).
//
// Tuning runs entirely on private buffers and a private stream, sharing only
// the (read-only, long-since-written) expert weights with the caller. That
// matters because the shape we most want to tune -- decode -- is only ever
// seen from inside `cudaStreamBeginCapture`: capture takes the very first
// step of each bucket, with no eager pass in front of it. Borrowing the
// caller's stream or workspace would need cross-stream events, and those get
// swallowed into the graph. Being self-contained means tuning is just
// ordinary work on an unrelated stream, which capture does not observe.
// Timing calls (`cudaStreamSynchronize`, `cudaMalloc`) are unsafe API in the
// capture sense, but pie captures with `cudaStreamCaptureModeRelaxed`
// (cuda_check.hpp), which permits exactly that.

struct MoeProblem {
    int num_rows;
    int hidden_size;
    int inter_size;
    int num_experts;
    int experts_per_token;
    int tp_size;
    int tp_rank;
    MoeActivation activation;
};

struct MoeBuffers {
    const std::uint16_t* input;
    const std::int32_t* token_selected_experts;
    const float* token_final_scales;
    const std::uint16_t* fc1_expert_weights;
    const std::uint16_t* fc2_expert_weights;
    std::uint16_t* output;
    std::uint8_t* workspace;
    std::int32_t* unpermuted_row_to_permuted_row;
};

void run_moe(Runner& runner, const MoeProblem& p, const MoeBuffers& b,
             cudaStream_t stream) {
    ck::QuantParams quant_params{};
    tk::LoraParams lora_params{};
    ck::MoeMinLatencyParams min_latency_params{};
    runner.runMoe(
        b.input, nullptr, false, b.token_selected_experts, b.token_final_scales,
        b.fc1_expert_weights, nullptr,
        ck::ActivationParams(to_cutlass_activation(p.activation)),
        b.fc2_expert_weights, nullptr, quant_params, p.num_rows, p.hidden_size,
        p.hidden_size, p.inter_size, p.num_experts, p.experts_per_token,
        reinterpret_cast<char*>(b.workspace), b.output,
        b.unpermuted_row_to_permuted_row,
        parallelism_config(p.tp_size, p.tp_rank), false, false, lora_params,
        false, false, false, min_latency_params, false, stream);
}

// A candidate must be at least this much faster than the incumbent to
// displace it.
constexpr float kTacticMargin = 0.98f;

int autotune_m_bucket(int m) {
    // Exact below 16 (decode concurrency lives here and the best tile changes
    // fast), power-of-two above.
    if (m <= 16) return m;
    int b = 16;
    while (b < m && b < (1 << 20)) b <<= 1;
    return b;
}

std::uint64_t tactic_key(const MoeProblem& p) {
    const auto mix = tuning_hash;
    std::uint64_t h = 0;
    h = mix(h, static_cast<std::uint64_t>(autotune_m_bucket(p.num_rows)));
    h = mix(h, static_cast<std::uint64_t>(p.hidden_size));
    h = mix(h, static_cast<std::uint64_t>(p.inter_size));
    h = mix(h, static_cast<std::uint64_t>(p.num_experts));
    h = mix(h, static_cast<std::uint64_t>(p.experts_per_token));
    h = mix(h, static_cast<std::uint64_t>(p.tp_size));
    h = mix(h, static_cast<std::uint64_t>(p.activation));
    return h;
}

// Returns the elapsed time of the fastest of `iters` runs, or -1 if the tactic
// fails. Errors here are expected (a tile can be rejected for this problem
// even when occupancy says otherwise), so they are swallowed and the tactic is
// dropped from consideration.
float time_tactic(RunnerState& s, Runner& runner, const MoeProblem& p,
                  const MoeBuffers& b, const TacticPair& t, cudaEvent_t start,
                  cudaEvent_t stop, cudaStream_t stream) {
    constexpr int kWarmup = 3;
    constexpr int kIters = 7;
    runner.setTactic(s.gemm1_tactics[t.gemm1], s.gemm2_tactics[t.gemm2]);
    try {
        for (int i = 0; i < kWarmup; ++i) run_moe(runner, p, b, stream);
    } catch (...) {
        cudaStreamSynchronize(stream);
        cudaGetLastError();
        return -1.0f;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        cudaGetLastError();
        return -1.0f;
    }
    float best = 1e30f;
    for (int i = 0; i < kIters; ++i) {
        cudaEventRecord(start, stream);
        try {
            run_moe(runner, p, b, stream);
        } catch (...) {
            cudaStreamSynchronize(stream);
            cudaGetLastError();
            return -1.0f;
        }
        cudaEventRecord(stop, stream);
        if (cudaEventSynchronize(stop) != cudaSuccess) {
            cudaGetLastError();
            return -1.0f;
        }
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, start, stop) != cudaSuccess) {
            cudaGetLastError();
            return -1.0f;
        }
        best = std::min(best, ms);
    }
    return best;
}

// Owns every allocation the tuner needs, so tuning never touches a buffer the
// caller's stream might still be using.
struct TuneArena {
    void* input = nullptr;
    void* experts = nullptr;
    void* scales = nullptr;
    void* output = nullptr;
    void* row_map = nullptr;
    void* workspace = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    ~TuneArena() {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (stream) cudaStreamDestroy(stream);
        for (void* p : {input, experts, scales, output, row_map, workspace}) {
            if (p) cudaFree(p);
        }
        cudaGetLastError();
    }

    bool alloc(std::size_t bytes, void** dst) {
        if (cudaMalloc(dst, bytes) != cudaSuccess) {
            cudaGetLastError();
            *dst = nullptr;
            return false;
        }
        return true;
    }

    bool init(const MoeProblem& p, std::size_t workspace_bytes) {
        const std::size_t rows = static_cast<std::size_t>(p.num_rows);
        const std::size_t routes = rows * p.experts_per_token;
        if (!alloc(rows * p.hidden_size * sizeof(std::uint16_t), &input) ||
            !alloc(rows * p.hidden_size * sizeof(std::uint16_t), &output) ||
            !alloc(routes * sizeof(std::int32_t), &experts) ||
            !alloc(routes * sizeof(std::int32_t), &row_map) ||
            !alloc(routes * sizeof(float), &scales) ||
            !alloc(workspace_bytes, &workspace)) {
            return false;
        }
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) !=
                cudaSuccess ||
            cudaEventCreate(&start) != cudaSuccess ||
            cudaEventCreate(&stop) != cudaSuccess) {
            const cudaError_t err = cudaGetLastError();
            std::fprintf(stderr,
                         "[pie-driver-cuda] FlashInfer MoE autotune stream/event "
                         "setup failed (%s)\n",
                         cudaGetErrorString(err));
            return false;
        }

        // Activations only have to be finite -- a tensor-core GEMM's cost does
        // not depend on its values. Routing does matter, because it decides
        // how many rows land in each expert's group; round-robin is the
        // balanced case.
        std::vector<std::int32_t> host_experts(routes);
        std::vector<float> host_scales(
            routes, 1.0f / static_cast<float>(p.experts_per_token));
        for (std::size_t i = 0; i < routes; ++i) {
            host_experts[i] = static_cast<std::int32_t>(i % p.num_experts);
        }
        const bool ok =
            cudaMemsetAsync(input, 0x3C,
                            rows * p.hidden_size * sizeof(std::uint16_t),
                            stream) == cudaSuccess &&
            cudaMemcpyAsync(experts, host_experts.data(),
                            routes * sizeof(std::int32_t),
                            cudaMemcpyHostToDevice, stream) == cudaSuccess &&
            cudaMemcpyAsync(scales, host_scales.data(), routes * sizeof(float),
                            cudaMemcpyHostToDevice, stream) == cudaSuccess &&
            cudaStreamSynchronize(stream) == cudaSuccess;
        if (!ok) {
            const cudaError_t err = cudaGetLastError();
            std::fprintf(stderr,
                         "[pie-driver-cuda] FlashInfer MoE autotune arena fill "
                         "failed (%s); using default tactic\n",
                         cudaGetErrorString(err));
        }
        return ok;
    }
};

TacticPair autotune(RunnerState& s, Runner& runner, const MoeProblem& p,
                    const MoeBuffers& live, std::size_t workspace_bytes) {
    TacticPair best = s.default_tactics;

    TuneArena arena;
    if (!arena.init(p, workspace_bytes)) {
        std::fprintf(stderr,
                     "[pie-driver-cuda] FlashInfer MoE autotune skipped for "
                     "m=%d h=%d i=%d e=%d k=%d (arena setup failed)\n",
                     p.num_rows, p.hidden_size, p.inter_size, p.num_experts,
                     p.experts_per_token);
        return best;
    }

    MoeBuffers b{};
    b.input = static_cast<const std::uint16_t*>(arena.input);
    b.token_selected_experts = static_cast<const std::int32_t*>(arena.experts);
    b.token_final_scales = static_cast<const float*>(arena.scales);
    b.fc1_expert_weights = live.fc1_expert_weights;
    b.fc2_expert_weights = live.fc2_expert_weights;
    b.output = static_cast<std::uint16_t*>(arena.output);
    b.workspace = static_cast<std::uint8_t*>(arena.workspace);
    b.unpermuted_row_to_permuted_row = static_cast<std::int32_t*>(arena.row_map);

    float best_ms = time_tactic(s, runner, p, b, best, arena.start, arena.stop,
                                arena.stream);
    const float baseline_ms = best_ms;
    int tried = 0;
    // Sweeps one of the two GEMMs, holding the other at the incumbent.
    //
    // The winner is not simply the fastest sample. Dozens of these tactics
    // land within a percent of each other, so which one records the lowest
    // time is decided by noise -- and since each tile shape accumulates in a
    // different order, letting noise pick meant the model's own output could
    // change between runs on a near-tied token. Instead every candidate within
    // `kTacticMargin` of the fastest is treated as tied, and the tie is broken
    // by the candidate's position in the list. That is stable as long as the
    // *set* of near-optimal tactics is stable, which it is; only the ordering
    // within it was noisy.
    auto sweep = [&](const std::vector<ce::CutlassGemmConfig>& configs,
                     bool is_gemm1) {
        int best_idx = is_gemm1 ? best.gemm1 : best.gemm2;
        if (best_idx < 0 || best_idx >= static_cast<int>(configs.size())) return;
        // Tune within the chosen epilogue, never across it. Which epilogue
        // GEMM2 runs decides how the topk sum is accumulated -- fp32 in a
        // separate pass, or bf16 reduction-adds in CTA completion order (see
        // `get_runner`) -- so it is a numerics decision, and a tuner chasing a
        // sub-percent timing difference must not be able to reverse it.
        const auto fusion = configs[best_idx].epilogue_fusion_type;
        std::vector<std::pair<int, float>> timings;
        timings.reserve(configs.size());
        float fastest = best_ms;
        for (int i = 0; i < static_cast<int>(configs.size()); ++i) {
            if (configs[i].epilogue_fusion_type != fusion) continue;
            if (runner.queryOccupancyForConfig(configs[i]) <= 0) continue;
            TacticPair cand = best;
            (is_gemm1 ? cand.gemm1 : cand.gemm2) = i;
            ++tried;
            const float ms = time_tactic(s, runner, p, b, cand, arena.start,
                                         arena.stop, arena.stream);
            if (ms <= 0.0f) continue;
            timings.emplace_back(i, ms);
            if (fastest < 0.0f || ms < fastest) fastest = ms;
        }
        if (fastest <= 0.0f) return;
        const float cutoff = fastest / kTacticMargin;
        float chosen_ms = best_ms;
        for (const auto& [i, ms] : timings) {
            if (ms > cutoff) continue;
            best_idx = i;
            chosen_ms = ms;
            break;
        }
        (is_gemm1 ? best.gemm1 : best.gemm2) = best_idx;
        best_ms = chosen_ms;
    };
    sweep(s.gemm1_tactics, true);
    sweep(s.gemm2_tactics, false);

    if (log_enabled()) {
        std::fprintf(
            stderr,
            "[pie-driver-cuda] FlashInfer MoE autotune m=%d h=%d i=%d e=%d "
            "k=%d: %.1f us -> %.1f us over %d tactics\n"
            "    GEMM1 %s\n    GEMM2 %s\n",
            p.num_rows, p.hidden_size, p.inter_size, p.num_experts,
            p.experts_per_token, baseline_ms * 1e3f, best_ms * 1e3f, tried,
            config_str(s.gemm1_tactics[best.gemm1]).c_str(),
            config_str(s.gemm2_tactics[best.gemm2]).c_str());
    }
    return best;
}


std::string tactic_cache_signature(const RunnerState& s) {
    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        cudaGetLastError();
        return {};
    }
    char buf[384];
    std::snprintf(buf, sizeof(buf),
                  "# pie-moe-tactics v1 sm%d%d gemm1=%zu gemm2=%zu dev=%s",
                  prop.major, prop.minor, s.gemm1_tactics.size(),
                  s.gemm2_tactics.size(), prop.name);
    return buf;
}

TuningCache& tactic_cache(const RunnerState& s) {
    static TuningCache cache("moe_tactics.txt", tactic_cache_signature(s));
    return cache;
}

// Chooses (and on first sight of a shape, measures) the tactic pair for this
// problem, then installs it on the runner.
void install_tactics(RunnerState& s, Runner& runner, const MoeProblem& p,
                     const MoeBuffers& b, std::size_t workspace_bytes) {
    const std::uint64_t key = tactic_key(p);
    std::lock_guard<std::mutex> lock(s.tune_mutex);
    TuningCache& disk = tactic_cache(s);
    auto it = s.tuned.find(key);
    if (it == s.tuned.end()) {
        TacticPair chosen{};
        if (!disk.lookup(key, &chosen.gemm1, &chosen.gemm2) ||
            chosen.gemm1 < 0 || chosen.gemm2 < 0 ||
            chosen.gemm1 >= static_cast<int>(s.gemm1_tactics.size()) ||
            chosen.gemm2 >= static_cast<int>(s.gemm2_tactics.size())) {
            chosen = autotune(s, runner, p, b, workspace_bytes);
            disk.store(key, chosen.gemm1, chosen.gemm2);
        }
        it = s.tuned.emplace(key, chosen).first;
    }
    runner.setTactic(s.gemm1_tactics[it->second.gemm1],
                     s.gemm2_tactics[it->second.gemm2]);
}

}  // namespace

bool flashinfer_cutlass_moe_enabled() { return true; }

namespace {

// `PIE_MOE_FUSED_MAX_ROWS` / `PIE_MOE_FUSED_MIN_ROWS` are documented in the
// header as the overrides for the fused path's row window, but both accessors
// returned a constant and never read the environment -- so the documented
// knobs did nothing and the window could not be swept against a measurement.
int env_int(const char* name, int fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end == v || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
        return fallback;
    }
    return static_cast<int>(parsed);
}

}  // namespace

int flashinfer_cutlass_moe_max_rows() {
    static const int v = env_int("PIE_MOE_FUSED_MAX_ROWS", 1024);
    return v;
}

namespace {

// Only an explicit override may narrow the window inside the runner. The
// callers already carry their own row caps (qwen3.5 sizes its workspace for
// `kFusedMoeMaxRows`, kimi and glm5 consult `min_rows` directly), so enforcing
// the default here would silently re-route their prefill batches to a
// different kernel -- a behaviour change dressed up as a fix.
bool fused_window_overridden() {
    static const bool on = std::getenv("PIE_MOE_FUSED_MAX_ROWS") != nullptr ||
                           std::getenv("PIE_MOE_FUSED_MIN_ROWS") != nullptr;
    return on;
}

}  // namespace

int flashinfer_cutlass_moe_min_rows() {
    static const int v = env_int("PIE_MOE_FUSED_MIN_ROWS", 0);
    return v;
}

int moe_gemv_max_tokens(int fallback) { return fallback; }

std::size_t flashinfer_cutlass_moe_workspace_bytes(
    MoeActivation activation,
    int num_rows,
    int hidden_size,
    int inter_size,
    int num_experts,
    int experts_per_token,
    int tp_size,
    int tp_rank) {
    if (num_rows <= 0 || hidden_size <= 0 || inter_size <= 0 ||
        num_experts <= 0 || experts_per_token <= 0) {
        return 0;
    }
    // A workspace query is also the arch probe: `getWorkspaceSize` walks the
    // TMA warp-specialized configs and throws when none of them has a
    // compiled launcher for this SM. The vendored generated units cover sm80
    // and (behind `PIE_HAS_SM100`) sm100, so on Hopper every config is
    // unbacked and the query throws "Could not find valid config when
    // calculating workspace size" — which used to abort load_model outright.
    // Reporting zero is what the callers already handle: they leave
    // `cutlass_ws` empty and take the non-fused expert path, the same
    // fallback the sm100-without-Blackwell-kernels case was written for.
    std::size_t bytes = 0;
    try {
        Runner& runner = get_runner();
        bytes = runner.getWorkspaceSize(
            num_rows,
            hidden_size,
            inter_size,
            num_experts,
            experts_per_token,
            to_cutlass_activation(activation),
            parallelism_config(tp_size, tp_rank),
            false,
            false,
            false,
            false,
            false);
    } catch (const std::exception& e) {
        std::fprintf(
            stderr,
            "[pie-driver-cuda] FlashInfer CUTLASS MoE unavailable on this "
            "device (%s); falling back to the unfused expert path\n",
            e.what());
        return 0;
    }
    if (log_enabled()) {
        std::fprintf(
            stderr,
            "[pie-driver-cuda] FlashInfer MoE workspace %zu MiB "
            "(rows=%d hidden=%d inter=%d experts=%d topk=%d)\n",
            bytes >> 20, num_rows, hidden_size, inter_size, num_experts,
            experts_per_token);
    }
    return bytes;
}

bool flashinfer_cutlass_moe_bf16(
    MoeActivation activation,
    const std::uint16_t* input,
    const std::int32_t* token_selected_experts,
    const float* token_final_scales,
    const std::uint16_t* fc1_expert_weights,
    const std::uint16_t* fc2_expert_weights,
    std::uint16_t* output,
    std::uint8_t* workspace,
    std::size_t workspace_bytes,
    std::int32_t* unpermuted_row_to_permuted_row,
    int num_rows,
    int hidden_size,
    int inter_size,
    int num_experts,
    int experts_per_token,
    int tp_size,
    int tp_rank,
    cudaStream_t stream) {
    if (input == nullptr || token_selected_experts == nullptr ||
        token_final_scales == nullptr || fc1_expert_weights == nullptr ||
        fc2_expert_weights == nullptr || output == nullptr ||
        workspace == nullptr || unpermuted_row_to_permuted_row == nullptr) {
        return false;
    }
    // The row window is policy, not capacity, and it belongs here: a caller
    // that decided for itself would diverge from the others the moment the
    // window moved. Declining sends the caller to its own fallback path.
    if (fused_window_overridden()) {
        if (num_rows > flashinfer_cutlass_moe_max_rows()) return false;
        if (num_rows < flashinfer_cutlass_moe_min_rows()) return false;
    }
    const std::size_t needed = flashinfer_cutlass_moe_workspace_bytes(
        activation, num_rows, hidden_size, inter_size, num_experts,
        experts_per_token, tp_size, tp_rank);
    if (needed == 0 || workspace_bytes < needed) return false;

    Runner& runner = get_runner();
    const MoeProblem problem{num_rows,   hidden_size,       inter_size,
                             num_experts, experts_per_token, tp_size,
                             tp_rank,     activation};
    const MoeBuffers buffers{input,
                             token_selected_experts,
                             token_final_scales,
                             fc1_expert_weights,
                             fc2_expert_weights,
                             output,
                             workspace,
                             unpermuted_row_to_permuted_row};
    install_tactics(state(), runner, problem, buffers, needed);
    // The runner leaves parts of its workspace untouched -- the permuted row
    // buffers are only filled for real routes, and the grouped GEMM rounds
    // each expert's row count up to a tile -- yet the padded rows are still
    // multiplied and, with a fused finalize epilogue, still scattered back.
    // Whatever the previous call left there therefore leaks into this one's
    // result. FlashInfer's own binding hides this by allocating a fresh
    // workspace per call, which a caching allocator hands back with the same
    // contents every time; pie keeps one workspace for the life of the model,
    // so the leftovers differ with every shape that ran before. That showed up
    // as the same prompt decoding to different tokens on different runs.
    cudaMemsetAsync(workspace, 0, needed, stream);
    run_moe(runner, problem, buffers, stream);
    return true;
}

}  // namespace pie_cuda_driver::ops
