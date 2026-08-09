// Differential oracle for src/store/memory_planner.rs.
//
// Compiles the REAL store/memory_planner.cpp -- all 1,221 lines of it -- and
// sweeps `plan_cuda_memory` over a grid of device shapes, model shapes and
// configurations, printing one line per case. The Rust port sweeps the
// identical grid in tests/memory_planner_parity.rs and the two transcripts
// must be byte-identical.
//
// What is stubbed, and why that is honest:
//
//   * The three CUDA queries. Stubbing them is the POINT: the planner's
//     answer is a function of (device shape, model shape, config), and no
//     single machine can present more than one device shape. The C++ has
//     therefore never been exercised on anything but whatever card the
//     developer had.
//
//   * The ~14 model workspace formulas. These live in model/ and batch/ and
//     are the planner's INPUTS, not its logic -- exactly like `cache_dir()`
//     was for the profile cache. Each stub is an affine function of the shape
//     with a distinct coefficient, so a term dropped from the arena sum shows
//     up as a different number rather than cancelling out. The Rust test
//     mirrors these coefficients exactly.
//
//   * The profile cache read. Driven from a global here so the SELECTION
//     paths (pinned / drifted / unmatched) can be swept directly; the cache's
//     own parsing is proven byte-for-byte by the profile_cache oracle.
//
// Everything else -- the budget, the ladders, the feasibility filters, the
// score, the selection order, tp_min_plan -- is the shipping code.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <optional>
#include <thread>
#include <sstream>
#include <string>
#include <vector>

#include "store/memory_planner.hpp"
#include "config.hpp"
#include "model/config.hpp"
#include "store/kv_cache.hpp"
#include "store/recurrent_state_cache.hpp"
#include "store/kv_cache_format.hpp"
#include "store/planner_profile_cache.hpp"
#include "batch/planner_calibration.hpp"
#include "gemm/gemm.hpp"

// ---------------------------------------------------------------------------
// Device stubs, driven by the sweep.
// ---------------------------------------------------------------------------
namespace {
cudaDeviceProp g_prop{};
std::size_t g_free = 0;
std::size_t g_total = 0;
// Weights already resident, as a fraction of the device. Default 0.5: a
// realistic post-load state, and far enough from both ends that the budget
// arithmetic is exercised rather than short-circuited.
double g_used_frac = 0.5;
// Set to override the fraction with an absolute figure, for the axis that
// sweeps residency directly.
std::optional<std::size_t> g_used_bytes;
bool g_calibrating = false;
bool g_envelopes = false;
bool g_rs_bf16 = true;
std::optional<pie_cuda_driver::PlannerProfileShape> g_profile;
std::string g_profile_error;
}  // namespace

cudaError_t cudaGetDevice(int* dev) { *dev = 0; return cudaSuccess; }
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int) {
    *prop = g_prop;
    return cudaSuccess;
}
cudaError_t cudaMemGetInfo(std::size_t* f, std::size_t* t) {
    *f = g_free;
    *t = g_total;
    return cudaSuccess;
}
const char* cudaGetErrorString(cudaError_t) { return "stub"; }

namespace pie_cuda_driver {

bool planner_calibration_requested() { return g_calibrating; }
void set_planner_calibration_requested(bool r) { g_calibrating = r; }

bool KvCache::envelopes_requested() { return g_envelopes; }
bool RecurrentStateCache::recurrent_state_bf16_default() { return g_rs_bf16; }

// --- KV byte formulas -------------------------------------------------------
// Distinct per-family coefficients so a mis-routed model family is visible in
// the transcript rather than producing the same page size by coincidence.
std::size_t kv_cache_device_bytes_per_page(const KvCacheFormat&, int page_size,
                                           int kv_heads, int head_dim) {
    return static_cast<std::size_t>(page_size) * kv_heads * head_dim * 2;
}
std::size_t kv_page_bytes_homogeneous(const HfConfig& cfg, int tp_size,
                                      const KvCacheFormat&) {
    const int heads = std::max(1, cfg.num_key_value_heads / std::max(1, tp_size));
    return static_cast<std::size_t>(cfg.num_hidden_layers) * heads *
           cfg.head_dim_kernel * 2 * 2;
}
std::size_t kv_page_bytes_per_layer(const HfConfig& cfg,
                                    const std::vector<int>& per_layer_head_dim,
                                    const std::vector<int>& per_layer_num_kv_heads,
                                    const std::vector<int>&, int tp_size,
                                    const KvCacheFormat&) {
    std::size_t total = 0;
    const std::size_t n = per_layer_head_dim.size();
    for (std::size_t i = 0; i < n; ++i) {
        const int heads =
            std::max(1, per_layer_num_kv_heads[i] / std::max(1, tp_size));
        total += static_cast<std::size_t>(heads) * per_layer_head_dim[i] * 2 * 2;
    }
    return total == 0 ? kv_page_bytes_homogeneous(cfg, tp_size, KvCacheFormat{})
                      : total;
}
std::size_t dsv4_compress_bytes_per_token(const HfConfig& cfg) {
    return static_cast<std::size_t>(cfg.num_hidden_layers) * 512;
}

namespace model {
std::size_t kv_page_bytes_nemotron_h(const HfConfig& cfg, int tp,
                                     const KvCacheFormat&) {
    const int heads = std::max(1, cfg.num_key_value_heads / std::max(1, tp));
    return static_cast<std::size_t>(cfg.num_hidden_layers) * heads *
           cfg.head_dim_kernel * 2 * 2 / 2;
}

// --- Workspace formulas -----------------------------------------------------
// Affine in the shape with a distinct constant each, so dropping any one term
// from the arena sum changes the total.
std::size_t workspace_bytes(const HfConfig& cfg, int max_tokens,
                            int max_output_rows, int max_intermediate,
                            int max_Hq, int max_Hk, int mtp_draft_rows) {
    return static_cast<std::size_t>(max_tokens) * cfg.hidden_size * 2 +
           static_cast<std::size_t>(max_output_rows) * 4096 +
           static_cast<std::size_t>(max_intermediate) * 128 +
           static_cast<std::size_t>(max_Hq) * 64 +
           static_cast<std::size_t>(max_Hk) * 32 +
           static_cast<std::size_t>(mtp_draft_rows) * 2048;
}
std::size_t qwen3_5_la_workspace_bytes(const HfConfig& cfg, int N, int tp) {
    return static_cast<std::size_t>(N) * cfg.hidden_size / std::max(1, tp) * 3;
}
std::size_t qwen3_5_moe_workspace_bytes(const HfConfig& cfg, int N, int tp) {
    return static_cast<std::size_t>(N) * cfg.hidden_size / std::max(1, tp) * 5;
}
std::size_t nemotron_h_workspace_bytes(const HfConfig& cfg, int N, int tp) {
    return static_cast<std::size_t>(N) * cfg.hidden_size / std::max(1, tp) * 7;
}
std::size_t gemma4_moe_workspace_bytes(const HfConfig& cfg, int N) {
    return static_cast<std::size_t>(N) * cfg.hidden_size * 11;
}
std::size_t dsv4_workspace_bytes(const HfConfig& cfg, int N, int R, int tp) {
    return (static_cast<std::size_t>(N) * cfg.hidden_size +
            static_cast<std::size_t>(R) * 8192) /
           std::max(1, tp) * 13;
}
std::size_t kimi_workspace_bytes(const HfConfig& cfg, int N, int R, int tp) {
    return (static_cast<std::size_t>(N) * cfg.hidden_size +
            static_cast<std::size_t>(R) * 8192) /
           std::max(1, tp) * 17;
}
std::size_t glm5_workspace_bytes(const HfConfig& cfg, int N, int R, int maxpos,
                                 int tp) {
    return (static_cast<std::size_t>(N) * cfg.hidden_size +
            static_cast<std::size_t>(R) * 8192 +
            static_cast<std::size_t>(maxpos) * 4) /
           std::max(1, tp) * 19;
}
std::size_t kimi_k3_workspace_bytes(const HfConfig& cfg, int N, int R, int tp) {
    return (static_cast<std::size_t>(N) * cfg.hidden_size +
            static_cast<std::size_t>(R) * 8192) /
           std::max(1, tp) * 23;
}
std::size_t nemotron_h_state_slot_bytes(const HfConfig& cfg, int mamba_layers,
                                        int tp) {
    return static_cast<std::size_t>(std::max(0, mamba_layers)) *
           cfg.hidden_size * 4 / std::max(1, tp);
}
}  // namespace model

std::size_t attention_float_workspace_bytes(const HfConfig&, const Config&,
                                            const cudaDeviceProp&, int N, int R,
                                            bool prefill_graph_capable) {
    const std::size_t base =
        static_cast<std::size_t>(N) * 512 + static_cast<std::size_t>(R) * 1024;
    return prefill_graph_capable ? base * 2 : base;
}

std::size_t persistent_input_bytes(int N, int R, int max_page_refs,
                                   int max_custom_mask_bytes) {
    return static_cast<std::size_t>(N) * 64 +
           static_cast<std::size_t>(R) * 256 +
           static_cast<std::size_t>(max_page_refs) * 4 +
           static_cast<std::size_t>(max_custom_mask_bytes);
}

namespace kernels::gemm {
std::size_t runtime_quant_scratch_bytes(const RuntimeQuantScratchSpec& spec) {
    return spec.max_tokens * static_cast<std::size_t>(spec.hidden) * 2 +
           static_cast<std::size_t>(spec.group) * 1024;
}
}  // namespace kernels::gemm

// --- Profile cache, driven from the sweep -----------------------------------
PlannerProfileKey make_planner_profile_key(const cudaDeviceProp& prop,
                                           const HfConfig& hf, int tp_size,
                                           const KvCacheFormat& fmt) {
    PlannerProfileKey k;
    k.gpu_name = prop.name;
    k.compute_major = prop.major;
    k.compute_minor = prop.minor;
    k.sm_count = prop.multiProcessorCount;
    k.kv_cache_dtype = fmt.name;
    k.tp_size = tp_size;
    k.model_type = hf.model_type;
    k.hidden_size = hf.hidden_size;
    k.num_hidden_layers = hf.num_hidden_layers;
    k.num_attention_heads = hf.num_attention_heads;
    k.num_key_value_heads = hf.num_key_value_heads;
    k.head_dim = hf.head_dim_kernel;
    return k;
}
std::optional<PlannerProfileShape> planner_profile_cache_lookup(
    const PlannerProfileKey&, std::string* error) {
    if (error != nullptr) *error = g_profile_error;
    return g_profile;
}
std::filesystem::path planner_profile_cache_path() {
    return std::filesystem::path("/stub/planner_profiles.json");
}
namespace {
std::size_t g_budget = 0;
}
void set_planner_budget_bytes(std::size_t b) { g_budget = b; }
std::size_t planner_budget_bytes() { return g_budget; }

}  // namespace pie_cuda_driver

// ---------------------------------------------------------------------------
// The sweep.
// ---------------------------------------------------------------------------
using namespace pie_cuda_driver;

namespace {

constexpr std::size_t kGiB = 1024ull * 1024 * 1024;

struct DeviceCase {
    const char* label;
    const char* name;
    int major;
    int minor;
    int sms;
    std::size_t total_gib;
};

// Every device shape the planner branches on, plus two it does not, so a
// branch added later has somewhere to show up.
const std::vector<DeviceCase> kDevices = {
    {"l40s", "NVIDIA L40S", 8, 9, 142, 45},
    {"a100", "NVIDIA A100-SXM4-80GB", 8, 0, 108, 80},
    {"h100", "NVIDIA H100 80GB HBM3", 9, 0, 132, 80},
    {"b200", "NVIDIA B200", 10, 0, 148, 180},
    {"rtx5090", "NVIDIA GeForce RTX 5090", 12, 0, 170, 32},
    {"l4", "NVIDIA L4", 8, 9, 58, 24},
    {"t4", "Tesla T4", 7, 5, 40, 16},
    {"ada6000", "NVIDIA RTX 6000 Ada", 8, 9, 142, 48},
};

enum class Fam {
    Dense,
    Qwen35,
    Qwen35Moe,
    NemotronH,
    Gemma4,
    Dsv4,
    Kimi,
    Glm5,
    KimiK3,
};

struct ModelCase {
    const char* label;
    Fam fam;
    const char* model_type;
    int hidden;
    int layers;
    int kv_heads;
    int head_dim;
};

const std::vector<ModelCase> kModels = {
    {"qwen3-8b", Fam::Dense, "qwen3", 4096, 36, 8, 128},
    {"qwen3-0.6b", Fam::Dense, "qwen3", 1024, 28, 8, 128},
    {"llama3-70b", Fam::Dense, "llama", 8192, 80, 8, 128},
    {"narrow", Fam::Dense, "llama", 2048, 24, 4, 64},
    {"qwen35", Fam::Qwen35, "qwen3_next", 4096, 48, 4, 128},
    {"qwen35moe", Fam::Qwen35Moe, "qwen3_next_moe", 2048, 48, 2, 128},
    {"nemotron", Fam::NemotronH, "nemotron_h", 4480, 62, 8, 128},
    {"gemma4", Fam::Gemma4, "gemma4", 3840, 60, 4, 256},
    {"dsv4", Fam::Dsv4, "deepseek_v4", 7168, 61, 1, 576},
    {"kimi", Fam::Kimi, "kimi", 7168, 61, 1, 576},
    {"glm5", Fam::Glm5, "glm5", 5120, 92, 1, 576},
    {"kimik3", Fam::KimiK3, "kimi_k3", 7168, 61, 1, 576},
    {"kvheavy", Fam::Dense, "llama", 8192, 96, 64, 128},
};

HfConfig make_hf(const ModelCase& m) {
    HfConfig hf;
    hf.model_type = m.model_type;
    hf.hidden_size = m.hidden;
    hf.num_hidden_layers = m.layers;
    hf.num_attention_heads = std::max(1, m.hidden / 128);
    hf.num_key_value_heads = m.kv_heads;
    hf.head_dim = m.head_dim;
    hf.head_dim_kernel = m.head_dim;
    hf.max_position_embeddings = 131072;
    hf.kv_lora_rank = 512;
    hf.qk_rope_head_dim = 64;
    hf.gemma4_enable_moe = (m.fam == Fam::Gemma4);
    hf.linear_num_key_heads = 16;
    hf.linear_num_value_heads = 32;
    hf.linear_key_head_dim = 128;
    hf.linear_value_head_dim = 128;
    hf.linear_conv_kernel_dim = 4;
    return hf;
}

std::string describe(const CudaMemoryPlan& p) {
    std::ostringstream o;
    o << "page=" << p.kv_page_size << " N=" << p.max_workspace_tokens
      << " R=" << p.max_requests << " refs=" << p.max_page_refs
      << " pgB=" << p.kv_page_bytes << " attnB=" << p.attn_float_workspace_bytes
      << " rqB=" << p.runtime_quant_scratch_bytes
      << " persB=" << p.persistent_input_bytes << " cap=["
      << p.capacity.max_forward_tokens << ',' << p.capacity.max_forward_requests
      << ',' << p.capacity.max_page_refs << ',' << p.capacity.max_logit_rows
      << ',' << p.capacity.max_prob_rows << ',' << p.capacity.max_custom_mask_bytes
      << ',' << p.capacity.max_sampler_rows << ',' << p.capacity.max_logprob_labels
      << ']';
    return o.str();
}

// Plan without emitting a row, for searches whose ANSWER is the interesting
// output rather than the plan.
std::size_t probe_budget(const DeviceCase& d, const ModelCase& m,
                         const Config& cfg);

void run_case(const std::string& id, const DeviceCase& d, const ModelCase& m,
              const Config& cfg) {
    std::memset(&g_prop, 0, sizeof(g_prop));
    std::snprintf(g_prop.name, sizeof(g_prop.name), "%s", d.name);
    g_prop.major = d.major;
    g_prop.minor = d.minor;
    g_prop.multiProcessorCount = d.sms;
    g_prop.totalGlobalMem = d.total_gib * kGiB;
    g_total = d.total_gib * kGiB;
    const std::size_t used =
        g_used_bytes.has_value()
            ? *g_used_bytes
            : static_cast<std::size_t>(static_cast<double>(g_total) * g_used_frac);
    g_free = g_total > used ? g_total - used : 0;

    const HfConfig hf = make_hf(m);
    KvCacheFormat fmt;
    fmt.name = "auto";
    kernels::gemm::RuntimeQuantScratchSpec spec;
    spec.hidden = hf.hidden_size;
    spec.group = 128;

    std::vector<int> g4_head_dim, g4_kv_heads, g4_src;
    if (m.fam == Fam::Gemma4) {
        for (int i = 0; i < hf.num_hidden_layers; ++i) {
            g4_head_dim.push_back(hf.head_dim_kernel);
            g4_kv_heads.push_back(hf.num_key_value_heads);
            g4_src.push_back(i);
        }
    }

    // `tp_min_plan` is an in-process BARRIER: with tp_size > 1 and a group key
    // it blocks until that many threads arrive. A single-threaded sweep
    // therefore deadlocks on exactly the cases the rendezvous exists for, so
    // the ranks are spawned for real. Only rank 0's plan is printed -- every
    // rank returns the same reduced plan, which is the property being checked.
    const int ranks = (cfg.distributed.tp_size > 1 &&
                       !cfg.distributed.nccl_unique_id_hex.empty())
                          ? cfg.distributed.tp_size
                          : 1;

    auto plan_once = [&]() -> std::string {
        try {
            return describe(plan_cuda_memory(
            cfg, hf, /*max_intermediate=*/hf.hidden_size * 4,
            /*max_Hq=*/hf.num_attention_heads, /*max_Hk=*/hf.num_key_value_heads,
            /*gemma4_selected=*/m.fam == Fam::Gemma4, g4_head_dim, g4_kv_heads,
            g4_src,
            /*qwen3_5_selected=*/m.fam == Fam::Qwen35 || m.fam == Fam::Qwen35Moe,
            /*qwen3_5_moe_selected=*/m.fam == Fam::Qwen35Moe,
            /*qwen3_5_linear_layers=*/m.fam == Fam::Qwen35 || m.fam == Fam::Qwen35Moe
                ? hf.num_hidden_layers * 3 / 4
                : 0,
            /*nemotron_h_selected=*/m.fam == Fam::NemotronH,
            /*nemotron_h_mamba_layers=*/m.fam == Fam::NemotronH ? 28 : 0,
            /*deepseek_v4_selected=*/m.fam == Fam::Dsv4,
            /*kimi_selected=*/m.fam == Fam::Kimi,
            /*glm5_selected=*/m.fam == Fam::Glm5,
            /*kimi_k3_selected=*/m.fam == Fam::KimiK3,
            /*prefill_graph_capable=*/m.fam == Fam::Qwen35 || m.fam == Fam::NemotronH,
            fmt, spec, /*verbose=*/true));
        } catch (const std::exception& e) {
            // The message is compared too: it is the operator-facing half of
            // the planner and the only thing a failed boot leaves behind.
            return std::string("THROWS ") + e.what();
        }
    };

    std::string out;
    std::string notes;
    if (ranks == 1) {
        // The planner's diagnostics go to `std::cerr` and are otherwise
        // invisible to its own test suite -- which is how the profile cache's
        // warnings went unexercised. Capture them: they are the operator's
        // only account of WHY a plan was chosen, so they are part of what
        // parity means. The Rust returns them as `Planned::notes` rather than
        // printing, and the two must agree line for line.
        std::ostringstream captured;
        std::streambuf* saved = std::cerr.rdbuf(captured.rdbuf());
        out = plan_once();
        std::cerr.rdbuf(saved);
        notes = captured.str();
    } else {
        std::vector<std::string> results(static_cast<std::size_t>(ranks));
        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(ranks));
        for (int i = 0; i < ranks; ++i) {
            threads.emplace_back(
                [&results, &plan_once, i] {
                    results[static_cast<std::size_t>(i)] = plan_once();
                });
        }
        for (auto& t : threads) t.join();
        out = results[0];
        // Every rank must leave with the same plan; a divergence is the whole
        // failure mode the reduction exists to prevent, so name it rather than
        // printing rank 0 and moving on.
        for (const auto& r : results) {
            if (r != out) {
                out = "RANK-DIVERGENCE";
                break;
            }
        }
        // Every rank writes the same notes to the one global `std::cerr`, and
        // the interleaving is not deterministic, so they are not compared
        // here. Nothing is lost: every note-producing path (both calibration
        // notes and all three profile-cache notes) is reached at tp=1 above,
        // where the capture is exact.
        notes = "\x01MULTIRANK";
    }
    // Newlines inside the notes would break the one-row-per-case contract;
    // fold them to a visible separator rather than dropping them.
    std::string flat;
    for (const char ch : notes) flat += (ch == '\n') ? '\x1f' : ch;
    std::cout << id << '|' << planner_budget_bytes() << '|' << out << '|' << flat
              << '\n';
}

// Runs the real path and reports only the budget it produced.
//
// Implemented by swallowing `run_case`'s row rather than by reimplementing the
// setup: a second copy of the device/HF wiring could drift from the one under
// test, and then the search would be answering a question about the wrong
// configuration.
std::size_t probe_budget(const DeviceCase& d, const ModelCase& m,
                         const Config& cfg) {
    std::ostringstream sink;
    std::streambuf* saved = std::cout.rdbuf(sink.rdbuf());
    run_case("probe", d, m, cfg);
    std::cout.rdbuf(saved);
    return planner_budget_bytes();
}

Config base_config() {
    Config c;
    c.batching.gpu_mem_utilization = 0.90;
    c.batching.memory_profile = "auto";
    c.distributed.tp_size = 1;
    c.distributed.nccl_unique_id_hex = "";
    return c;
}

}  // namespace

int main() {
    std::cout << "# memory_planner oracle v1\n";

    // 1. The full (device x model) grid at defaults.
    for (const auto& d : kDevices) {
        for (const auto& m : kModels) {
            run_case(std::string("grid/") + d.label + "/" + m.label, d, m,
                     base_config());
        }
    }

    // 2. Every profile family, on two contrasting devices.
    for (const char* profile :
         {"auto", "latency", "balanced", "throughput", "capacity", "bogus"}) {
        for (const auto& d : kDevices) {
            for (const auto& m : {kModels[0], kModels[3], kModels[5], kModels[8],
                                  kModels[12]}) {
                Config c = base_config();
                c.batching.memory_profile = profile;
                run_case(std::string("profile/") + profile + "/" + d.label + "/" +
                             m.label,
                         d, m, c);
            }
        }
    }

    // 3. Tensor parallelism -- the axis that drives most of the special cases.
    for (int tp : {1, 2, 4, 8}) {
        for (const auto& d : {kDevices[0], kDevices[2]}) {
            for (const auto& m : {kModels[0], kModels[5], kModels[6], kModels[11]}) {
                Config c = base_config();
                c.distributed.tp_size = tp;
                // A fresh key per case: the C++ registry never erases an
                // entry, so reusing one carries a stale `arrived` count into
                // the next rendezvous and it never releases.
                c.distributed.nccl_unique_id_hex = std::string("tp") +
                                                   std::to_string(tp) + d.label +
                                                   m.label;
                run_case(std::string("tp/") + std::to_string(tp) + "/" + d.label +
                             "/" + m.label,
                         d, m, c);
            }
        }
    }

    // 4. Utilization, including the values that empty the lattice.
    for (int step = 0; step <= 40; ++step) {
        const double util = 0.50 + 0.0025 * step;
        for (const auto& m : {kModels[2], kModels[8]}) {
            Config c = base_config();
            c.batching.gpu_mem_utilization = util;
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%.4f", util);
            run_case(std::string("cliff/") + buf + "/" + m.label, kDevices[0], m, c);
        }
    }
    for (double util : {0.05, 0.30, 0.50, 0.70, 0.85, 0.90, 0.95, 0.99, 1.0}) {
        for (const auto& m : {kModels[0], kModels[12]}) {
            Config c = base_config();
            c.batching.gpu_mem_utilization = util;
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%.2f", util);
            run_case(std::string("util/") + buf + "/" + m.label, kDevices[0], m, c);
        }
    }

    // 5. Weights already resident -- the `free_bytes` axis, swept up to and
    //    past the point where `usable <= used + safety` empties the budget.
    for (int used_gib : {0, 8, 20, 30, 38, 40, 41, 44, 45}) {
        g_used_bytes = static_cast<std::size_t>(used_gib) * kGiB;
        run_case(std::string("used/") + std::to_string(used_gib) + "gib",
                 kDevices[0], kModels[0], base_config());
    }
    g_used_bytes.reset();

    // 6. Pinned axes -- the single-candidate lattice.
    for (int n : {0, 512, 2048, 8192, 12288, 65536}) {
        for (int r : {0, 32, 256, 1024}) {
            Config c = base_config();
            c.batching.max_forward_tokens = static_cast<std::uint32_t>(n);
            c.batching.max_forward_requests = static_cast<std::uint32_t>(r);
            run_case(std::string("pin/") + std::to_string(n) + "/" +
                         std::to_string(r),
                     kDevices[0], kModels[0], c);
        }
    }
    for (int page : {0, 16, 32, 64}) {
        Config c = base_config();
        c.batching.kv_page_size = static_cast<std::uint32_t>(page);
        run_case(std::string("pinpage/") + std::to_string(page), kDevices[0],
                 kModels[0], c);
    }

    // 7. Calibration -- the max-area selector and its two notes.
    for (const auto& d : {kDevices[0], kDevices[4]}) {
        for (const auto& m : {kModels[0], kModels[5]}) {
            Config c = base_config();
            g_calibrating = true;
            run_case(std::string("calib/") + d.label + "/" + m.label, d, m, c);
            g_calibrating = false;
        }
    }
    // Calibration ignores the pins, which is the ratchet argument.
    {
        Config c = base_config();
        c.batching.max_forward_tokens = 1024;
        c.batching.max_forward_requests = 32;
        g_calibrating = true;
        run_case("calib/pinned-ignored", kDevices[0], kModels[0], c);
        g_calibrating = false;
    }

    // 8. The profile cache selector: hit, partial pin, drift, no match.
    {
        struct ProfCase {
            const char* label;
            const char* profile;
            int page;
            int n;
            int r;
            double budget_scale;  // 0 => budget_bytes stays 0
        };
        const std::vector<ProfCase> cases = {
            {"exact", "throughput", 16, 2048, 256, 1.0},
            {"page-only", "", 32, 0, 0, 1.0},
            {"tokens-only", "", 0, 4096, 0, 1.0},
            {"requests-only", "", 0, 0, 512, 1.0},
            {"profile-only", "latency", 0, 0, 0, 1.0},
            {"no-budget", "balanced", 16, 1024, 128, 0.0},
            {"drift-small", "balanced", 16, 1024, 128, 1.03},
            {"drift-edge", "balanced", 16, 1024, 128, 1.05},
            {"drift-over", "balanced", 16, 1024, 128, 1.20},
            {"drift-under", "balanced", 16, 1024, 128, 0.50},
            {"nomatch", "capacity", 16, 999, 0, 1.0},
            {"nomatch-profile", "nonesuch", 0, 0, 0, 1.0},
        };
        // Two passes: the first learns the real budget, the second scales it.
        Config c = base_config();
        run_case("prof/warmup", kDevices[0], kModels[0], c);
        const std::size_t real_budget = planner_budget_bytes();
        for (const auto& pc : cases) {
            PlannerProfileShape s;
            s.policy_profile = pc.profile;
            s.kv_page_size = pc.page;
            s.max_forward_tokens = pc.n;
            s.max_forward_requests = pc.r;
            s.budget_bytes =
                pc.budget_scale == 0.0
                    ? 0
                    : static_cast<std::size_t>(
                          static_cast<double>(real_budget) / pc.budget_scale);
            g_profile = s;
            run_case(std::string("prof/") + pc.label, kDevices[0], kModels[0], c);
            g_profile.reset();
        }
        // A cache that reports a complaint, and one that reports a complaint
        // AND a shape -- the C++ takes the shape anyway.
        g_profile_error = "schema version 9 is newer than this build";
        run_case("prof/complaint-only", kDevices[0], kModels[0], c);
        {
            PlannerProfileShape s;
            s.policy_profile = "throughput";
            s.kv_page_size = 16;
            s.budget_bytes = real_budget;
            g_profile = s;
            run_case("prof/complaint-and-shape", kDevices[0], kModels[0], c);
            g_profile.reset();
        }
        g_profile_error.clear();
        // The cache is ignored outside `auto`, and during calibration.
        {
            PlannerProfileShape s;
            s.policy_profile = "capacity";
            s.kv_page_size = 32;
            s.max_forward_tokens = 512;
            s.budget_bytes = real_budget;
            g_profile = s;
            Config named = base_config();
            named.batching.memory_profile = "latency";
            run_case("prof/named-profile-ignores-cache", kDevices[0], kModels[0],
                     named);
            g_calibrating = true;
            run_case("prof/calibrating-ignores-cache", kDevices[0], kModels[0], c);
            g_calibrating = false;
            g_profile.reset();
        }
        // The `!calibrating` guard on `use_profile_cache` is invisible when the
        // cached shape matches a candidate: the `if (calibrating)` branch
        // overwrites `best_it` afterwards either way, so only the NOTES differ.
        // A cache that complains, and one whose shape matches nothing, both
        // make a note fire -- and the note is the only witness that the guard
        // is doing anything at all.
        {
            g_profile_error = "entry 3: max_forward_tokens is not a number";
            g_calibrating = true;
            run_case("prof/calibrating-hides-complaint", kDevices[0], kModels[0],
                     c);
            g_calibrating = false;
            g_profile_error.clear();

            PlannerProfileShape s;
            s.policy_profile = "capacity";
            s.kv_page_size = 16;
            s.max_forward_tokens = 999;  // on no ladder
            s.budget_bytes = real_budget;
            g_profile = s;
            g_calibrating = true;
            run_case("prof/calibrating-hides-nomatch", kDevices[0], kModels[0], c);
            g_calibrating = false;
            g_profile.reset();
        }
        // `drift > tolerance` and `drift >= tolerance` differ on exactly one
        // input: drift EQUAL to 0.05 in IEEE double. `budget * 20 / 21` does
        // not land there, so scan for the integer that does. Both sides run
        // this identical scan over identical doubles, so both find the same
        // one -- or neither does, and the case reports that instead of
        // silently passing.
        //
        // `(B - m) / m` is exactly 1/20 -- hence exactly the double 0.05 --
        // iff `m == 20 * (B - m)`, i.e. `21 * m == 20 * B`, which needs
        // `21 | B`. Perturbing m instead cannot work: one step of m moves the
        // quotient by ~6e-11 while the window around 0.05 is one ulp, ~7e-18,
        // so a search over m finds nothing. Search the BUDGET instead --
        // utilization is a free dial, and one value in every 21 or so gives a
        // budget divisible by 21.
        //
        // The search runs on both sides against each side's own planner, so
        // the chosen (utilization, cached budget) pair is itself a differential
        // result: if the ported budget arithmetic drifts by one byte, the two
        // sides pick different values and the row differs.
        {
            double drift_util = 0.0;
            std::size_t drift_budget = 0;
            for (int step = 0; step < 400; ++step) {
                Config probe = base_config();
                probe.batching.gpu_mem_utilization = 0.90 - 1e-6 * step;
                const std::size_t b =
                    probe_budget(kDevices[0], kModels[0], probe);
                if (b != 0 && b % 21 == 0) {
                    drift_util = probe.batching.gpu_mem_utilization;
                    drift_budget = b;
                    break;
                }
            }
            const std::size_t exact = drift_budget / 21 * 20;
            char ub[32];
            std::snprintf(ub, sizeof(ub), "%.6f", drift_util);
            std::cout << "prof/drift-exact-scan|" << drift_budget << "|util="
                      << ub << " cached=" << exact << "|-\n";
            if (exact != 0) {
                c.batching.gpu_mem_utilization = drift_util;
                PlannerProfileShape s;
                s.policy_profile = "balanced";
                s.kv_page_size = 16;
                s.max_forward_tokens = 1024;
                s.max_forward_requests = 128;
                s.budget_bytes = exact;
                g_profile = s;
                run_case("prof/drift-exact", kDevices[0], kModels[0], c);
                g_profile.reset();
                c = base_config();
            }
        }
    }

    // 14. Tensor parallelism, single-threaded, for the DIAGNOSTICS.
    //
    // Section 3 spawns real rank threads because `tp_min_plan` is a barrier,
    // and their `std::cerr` interleaves nondeterministically -- so those rows
    // report `MULTIRANK` and carry no diagnostics at all. That silence covers
    // exactly the tp>1 special cases (the MoE TP2 knee, the Nemotron TP2
    // override, the auto prefill weight's tp>1 arm), which is why each of them
    // survived mutation while section 3 was the only tp coverage.
    //
    // `tp_min_plan` is a no-op when the key is empty, so a tp>1 plan with no
    // key runs on this thread and reports everything. It exercises the same
    // scoring; only the cross-rank reduction is skipped, and section 3 already
    // proves that.
    for (int tp : {2, 4, 8}) {
        for (const auto& d : {kDevices[0], kDevices[2], kDevices[7]}) {
            for (const auto& m : kModels) {
                Config c = base_config();
                c.distributed.tp_size = tp;
                c.distributed.nccl_unique_id_hex.clear();
                run_case(std::string("tpsolo/") + std::to_string(tp) + "/" +
                             d.label + "/" + m.label,
                         d, m, c);
                for (const char* prof : {"latency", "throughput"}) {
                    Config c2 = base_config();
                    c2.distributed.tp_size = tp;
                    c2.distributed.nccl_unique_id_hex.clear();
                    c2.batching.memory_profile = prof;
                    run_case(std::string("tpsolo/") + std::to_string(tp) + "/" +
                                 d.label + "/" + m.label + "/" + prof,
                             d, m, c2);
                }
            }
        }
    }

    // 9. The two switches that change byte counts under the plan.
    for (bool env : {false, true}) {
        for (bool bf16 : {true, false}) {
            g_envelopes = env;
            g_rs_bf16 = bf16;
            for (const auto& m : {kModels[0], kModels[4], kModels[6]}) {
                Config c = base_config();
                run_case(std::string("switch/env") + (env ? "1" : "0") + "/bf16" +
                             (bf16 ? "1" : "0") + "/" + m.label,
                         kDevices[0], m, c);
            }
        }
    }
    g_envelopes = false;
    g_rs_bf16 = true;

    // 10. Speculative drafts, which only the qwen3.5 families charge for.
    for (int drafts : {0, 1, 4, 32, 64, -1}) {
        for (const auto& m : {kModels[0], kModels[4], kModels[5]}) {
            Config c = base_config();
            c.model.mtp_num_drafts = drafts;
            run_case(std::string("mtp/") + std::to_string(drafts) + "/" + m.label,
                     kDevices[0], m, c);
        }
    }

    // 11. PIE_RS_SLOT_MULT, which resizes the recurrent pool.
    for (const char* mult : {"", "1", "2", "4", "8", "9", "0", "-3", "abc"}) {
        if (mult[0] == '\0') {
            unsetenv("PIE_RS_SLOT_MULT");
        } else {
            setenv("PIE_RS_SLOT_MULT", mult, 1);
        }
        for (const auto& m : {kModels[4], kModels[6]}) {
            Config c = base_config();
            run_case(std::string("slotmult/") + (mult[0] ? mult : "unset") + "/" +
                         m.label,
                     kDevices[0], m, c);
        }
    }
    unsetenv("PIE_RS_SLOT_MULT");

    // 12. The narrow-latency-auto collapse, swept across its boundary.
    for (int sms : {40, 64, 99, 100, 128}) {
        for (int hidden : {1024, 2048, 2049, 4096}) {
            DeviceCase d = kDevices[0];
            d.sms = sms;
            ModelCase m = kModels[0];
            m.hidden = hidden;
            run_case(std::string("narrow/") + std::to_string(sms) + "/" +
                         std::to_string(hidden),
                     d, m, base_config());
        }
    }

    // 13. tp_min_plan: the rendezvous is a no-op at tp=1 and with no key.
    for (int tp : {1, 2}) {
        for (const char* key : {"", "deadbeef"}) {
            Config c = base_config();
            c.distributed.tp_size = tp;
            c.distributed.nccl_unique_id_hex = key;
            run_case(std::string("rendezvous/tp") + std::to_string(tp) + "/" +
                         (key[0] ? "keyed" : "unkeyed"),
                     kDevices[0], kModels[0], c);
        }
    }

    return 0;
}
