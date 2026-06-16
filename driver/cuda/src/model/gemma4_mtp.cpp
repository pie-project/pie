#include "model/gemma4_mtp.hpp"

#include "config.hpp"
#include "hf_snapshot.hpp"
#include "model/gemma4.hpp"
#include "model/loaded_model.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/argmax.hpp"
#include "kernels/gather_rows.hpp"
#include "kernels/residual_add.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/sample_temp.hpp"
#include "kernels/scalar_mul.hpp"
#include "kernels/swiglu.hpp"
#include "loader/hf_config.hpp"
#include "loader/safetensors.hpp"
#include "ops/attention_flashinfer.hpp"
#include "ops/gemm.hpp"

namespace pie_cuda_driver::model {

namespace {

const DeviceTensor* find_any(
    const WeightStore& store,
    std::initializer_list<std::string> names)
{
    for (const auto& name : names) {
        auto it = store.find(name);
        if (it != store.end()) return &it->second.tensor;
    }
    return nullptr;
}

const DeviceTensor& must_any(
    const WeightStore& store,
    std::initializer_list<std::string> names,
    const char* label)
{
    if (const DeviceTensor* t = find_any(store, names)) return *t;
    std::string msg = "gemma4_mtp: missing weight for ";
    msg += label;
    msg += " (tried";
    for (const auto& name : names) {
        msg += " '";
        msg += name;
        msg += "'";
    }
    msg += ")";
    throw std::runtime_error(msg);
}

float read_scalar_once(const DeviceTensor* t) {
    if (t == nullptr || t->empty() || t->numel() == 0) return 1.f;
    if (t->dtype() == DType::FP32) {
        float v = 1.f;
        CUDA_CHECK(cudaMemcpy(&v, t->data(), sizeof(float),
                              cudaMemcpyDeviceToHost));
        return v;
    }
    if (t->dtype() == DType::BF16) {
        std::uint16_t bits = 0;
        CUDA_CHECK(cudaMemcpy(&bits, t->data(), sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost));
        const std::uint32_t f32_bits = static_cast<std::uint32_t>(bits) << 16;
        float v;
        std::memcpy(&v, &f32_bits, sizeof(float));
        return v;
    }
    return 1.f;
}

DeviceTensor make_gate_up_fused_weight(const DeviceTensor& gate,
                                       const DeviceTensor& up)
{
    if (gate.dtype() != DType::BF16 || up.dtype() != DType::BF16 ||
        gate.shape().size() != 2 || up.shape().size() != 2 ||
        gate.shape()[0] != up.shape()[0] ||
        gate.shape()[1] != up.shape()[1]) {
        throw std::runtime_error(
            "gemma4_mtp: cannot fuse gate/up projections with mismatched shapes");
    }
    const std::int64_t I = gate.shape()[0];
    const std::int64_t H = gate.shape()[1];
    DeviceTensor fused = DeviceTensor::allocate(DType::BF16, {2 * I, H});
    const std::size_t bytes =
        static_cast<std::size_t>(I) * static_cast<std::size_t>(H) *
        sizeof(std::uint16_t);
    auto* dst = static_cast<std::uint8_t*>(fused.data());
    CUDA_CHECK(cudaMemcpy(dst, gate.data(), bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(dst + bytes, up.data(), bytes,
                          cudaMemcpyDeviceToDevice));
    return fused;
}

int dim0(const DeviceTensor& t, const char* name) {
    if (t.shape().empty()) {
        throw std::runtime_error(std::string("gemma4_mtp: ") + name +
                                 " has empty shape");
    }
    return static_cast<int>(t.shape()[0]);
}

std::string layer_prefix(int i) {
    return "model.layers." + std::to_string(i) + ".";
}

std::string draft_layer_prefix(int i) {
    return "draft_model.model.layers." + std::to_string(i) + ".";
}

int find_target_kv_layer(
    const HfConfig& target_cfg,
    const std::string& layer_type)
{
    const int non_shared = std::max(
        0, target_cfg.num_hidden_layers - target_cfg.num_kv_shared_layers);
    if (target_cfg.layer_types.empty()) {
        return std::max(0, non_shared - 1);
    }
    for (int i = non_shared - 1; i >= 0; --i) {
        if (i < static_cast<int>(target_cfg.layer_types.size()) &&
            target_cfg.layer_types[i] == layer_type) {
            return i;
        }
    }
    throw std::runtime_error(
        "gemma4_mtp: no non-shared target KV layer for type '" +
        layer_type + "'");
}

bool mtp_use_target_residual_hidden() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_GEMMA4_MTP_TARGET_RESIDUAL_HIDDEN");
        return v != nullptr && std::string(v) == "1";
    }();
    return enabled;
}

bool mtp_reverse_preprojection_concat() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_GEMMA4_MTP_PREPROJ_REVERSE");
        return v != nullptr && std::string(v) == "1";
    }();
    return enabled;
}

bool mtp_disable_layer_scalar() {
    static const bool disabled = [] {
        const char* v = std::getenv("PIE_GEMMA4_MTP_DISABLE_LAYER_SCALAR");
        return v != nullptr && std::string(v) == "1";
    }();
    return disabled;
}

int mtp_position_offset() {
    static const int offset = [] {
        const char* v = std::getenv("PIE_GEMMA4_MTP_POSITION_OFFSET");
        return v == nullptr ? 1 : std::atoi(v);
    }();
    return offset;
}

bool env_bool_or(const char* name, bool fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return fallback;
    return v[0] != '0';
}

int env_int_or(const char* name, int fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return fallback;
    return std::atoi(v);
}

std::string env_string_or(const char* name, std::string fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return fallback;
    return std::string(v);
}

bool mtp_adaptive_drafts_enabled(const Gemma4MtpRuntimeConfig& runtime) {
    return env_bool_or(
        "PIE_GEMMA4_MTP_ADAPTIVE_DRAFTS", runtime.adaptive_drafts);
}

int mtp_initial_adaptive_drafts(
    const Gemma4MtpRuntimeConfig& runtime,
    int max_drafts)
{
    const int configured = env_int_or(
        "PIE_GEMMA4_MTP_INITIAL_DRAFTS", runtime.initial_drafts);
    return std::clamp(configured, 1, std::max(1, max_drafts));
}

int mtp_min_adaptive_drafts(
    const Gemma4MtpRuntimeConfig& runtime,
    int max_drafts)
{
    const int configured = env_int_or(
        "PIE_GEMMA4_MTP_MIN_DRAFTS", runtime.min_drafts);
    return std::clamp(configured, 1, std::max(1, max_drafts));
}

bool mtp_compact_draft_rows_enabled(
    const Gemma4MtpRuntimeConfig& runtime,
    const std::vector<int>& desired_drafts,
    int active_max_drafts)
{
    const std::string mode = env_string_or(
        "PIE_GEMMA4_MTP_COMPACT_DRAFT_ROWS", runtime.compact_draft_rows);
    if (mode == "off" || mode == "0") return false;
    if (mode == "on" || mode == "1") return true;
    const int M = static_cast<int>(desired_drafts.size());
    if (M < 4 || active_max_drafts <= 1) return false;
    int saved_row_steps = 0;
    for (const int d : desired_drafts) {
        saved_row_steps += std::max(0, active_max_drafts - d);
    }
    const int dense_row_steps = M * active_max_drafts;
    return saved_row_steps >= M &&
           saved_row_steps * 4 >= dense_row_steps;
}

bool mtp_profile_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_GEMMA4_MTP_PROFILE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

bool mtp_cuda_graph_enabled(const Gemma4MtpRuntimeConfig& runtime) {
    return env_bool_or("PIE_GEMMA4_MTP_CUDA_GRAPH", runtime.cuda_graph);
}

int mtp_max_draft_batch_rows(const Gemma4MtpRuntimeConfig& runtime) {
    return std::max(0, env_int_or(
        "PIE_GEMMA4_MTP_MAX_DRAFT_BATCH_ROWS",
        runtime.max_draft_batch_rows));
}

std::uint64_t mtp_profile_print_limit() {
    static const std::uint64_t limit = [] {
        const char* v = std::getenv("PIE_GEMMA4_MTP_PROFILE_LIMIT");
        if (v == nullptr || v[0] == '\0') return std::uint64_t{32};
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(v, &end, 10);
        return end == v ? std::uint64_t{32}
                        : static_cast<std::uint64_t>(parsed);
    }();
    return limit;
}

struct MtpCudaGraphKey {
    const Gemma4MtpWorkspace* workspace = nullptr;
    int rows = 0;
    int drafts = 0;
    int argmax_parts = 0;
    bool ordered_embeddings = false;

    bool operator==(const MtpCudaGraphKey& o) const noexcept {
        return workspace == o.workspace &&
               rows == o.rows &&
               drafts == o.drafts &&
               argmax_parts == o.argmax_parts &&
               ordered_embeddings == o.ordered_embeddings;
    }
};

struct MtpCudaGraphKeyHash {
    std::size_t operator()(const MtpCudaGraphKey& k) const noexcept {
        const auto p = reinterpret_cast<std::uintptr_t>(k.workspace);
        std::size_t h = static_cast<std::size_t>(p >> 4);
        h ^= static_cast<std::size_t>(k.rows) * 0x9e3779b185ebca87ull;
        h ^= static_cast<std::size_t>(k.drafts) << 17;
        h ^= static_cast<std::size_t>(k.argmax_parts) << 24;
        h ^= k.ordered_embeddings ? 0x517cc1b727220a95ull : 0ull;
        return h;
    }
};

using MtpCudaGraphMap =
    std::unordered_map<MtpCudaGraphKey, cudaGraphExec_t, MtpCudaGraphKeyHash>;

MtpCudaGraphMap& mtp_cuda_graph_cache() {
    static MtpCudaGraphMap cache;
    return cache;
}

std::mutex& mtp_cuda_graph_cache_mutex() {
    static std::mutex mu;
    return mu;
}

int mtp_desired_drafts(
    const SystemSpecDraftRequest& req,
    int max_drafts,
    const Gemma4MtpRuntimeConfig& runtime)
{
    if (!mtp_adaptive_drafts_enabled(runtime)) return max_drafts;
    if (max_drafts <= 1) return max_drafts;
    if (req.last_num_drafts <= 0 || req.last_match < 0) {
        return mtp_initial_adaptive_drafts(runtime, max_drafts);
    }
    const int min_drafts = mtp_min_adaptive_drafts(runtime, max_drafts);
    if (req.last_match >= req.last_num_drafts) {
        return std::clamp(req.last_num_drafts + 1, min_drafts, max_drafts);
    }
    return std::clamp(req.last_match + 1, min_drafts, max_drafts);
}

int mtp_argmax_parts() {
    static const int parts = [] {
        if (const char* v = std::getenv("PIE_GEMMA4_MTP_ARGMAX_PARTS");
            v != nullptr && v[0] != '\0') {
            return std::clamp(std::atoi(v), 1, 8);
        }
        if (const char* v = std::getenv("PIE_ARGMAX_PARTS");
            v != nullptr && v[0] != '\0') {
            return std::clamp(std::atoi(v), 1, 8);
        }
        return 8;
    }();
    return parts;
}

std::string mtp_kv_map_mode() {
    static const std::string mode = [] {
        const char* v = std::getenv("PIE_GEMMA4_MTP_KV_MAP");
        return (v == nullptr || v[0] == '\0') ? std::string("last_type")
                                              : std::string(v);
    }();
    return mode;
}

int choose_target_kv_layer(
    const HfConfig& target_cfg,
    const std::string& layer_type,
    int draft_idx,
    int draft_layers)
{
    const std::string mode = mtp_kv_map_mode();
    const int non_shared = std::max(
        0, target_cfg.num_hidden_layers - target_cfg.num_kv_shared_layers);
    if (mode == "tail") {
        const int start = std::max(0, non_shared - draft_layers);
        const int idx = std::min(non_shared - 1, start + draft_idx);
        if (idx >= 0 &&
            (target_cfg.layer_types.empty() ||
             idx >= static_cast<int>(target_cfg.layer_types.size()) ||
             target_cfg.layer_types[idx] == layer_type)) {
            return idx;
        }
    } else if (mode.rfind("base:", 0) == 0) {
        const int base = std::atoi(mode.c_str() + 5);
        const int idx = base + draft_idx;
        if (idx >= 0 && idx < non_shared &&
            (target_cfg.layer_types.empty() ||
             idx >= static_cast<int>(target_cfg.layer_types.size()) ||
             target_cfg.layer_types[idx] == layer_type)) {
            return idx;
        }
    }
    return find_target_kv_layer(target_cfg, layer_type);
}

struct Gemma4MtpProfile {
    bool enabled = false;
    int M = 0;
    int active_max_drafts = 0;
    int draft_steps = 0;
    int row_steps = 0;
    int layers = 0;
    bool compact_rows = false;
    bool ordered_embeddings = false;

    double host_setup_ms = 0.0;
    double host_plan_ms = 0.0;
    double host_emit_ms = 0.0;
    double upload_ms = 0.0;
    double gather_ms = 0.0;
    double input_ms = 0.0;
    double pre_projection_ms = 0.0;
    double q_ms = 0.0;
    double attention_ms = 0.0;
    double attn_out_ms = 0.0;
    double mlp_ms = 0.0;
    double final_norm_ms = 0.0;
    double score_ms = 0.0;
    double argmax_ms = 0.0;
    double copy_ms = 0.0;
    double post_projection_ms = 0.0;
    double d2h_ms = 0.0;
    double total_gpu_ms = 0.0;

    cudaEvent_t total_start = nullptr;
    cudaEvent_t total_stop = nullptr;
    cudaEvent_t stage_start = nullptr;
    cudaEvent_t stage_stop = nullptr;

    ~Gemma4MtpProfile() {
        if (total_start != nullptr) cudaEventDestroy(total_start);
        if (total_stop != nullptr) cudaEventDestroy(total_stop);
        if (stage_start != nullptr) cudaEventDestroy(stage_start);
        if (stage_stop != nullptr) cudaEventDestroy(stage_stop);
    }

    void begin(cudaStream_t stream) {
        enabled = mtp_profile_enabled();
        if (!enabled) return;
        CUDA_CHECK(cudaEventCreate(&total_start));
        CUDA_CHECK(cudaEventCreate(&total_stop));
        CUDA_CHECK(cudaEventCreate(&stage_start));
        CUDA_CHECK(cudaEventCreate(&stage_stop));
        CUDA_CHECK(cudaEventRecord(total_start, stream));
    }

    void end(cudaStream_t stream) {
        if (!enabled) return;
        CUDA_CHECK(cudaEventRecord(total_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(total_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, total_start, total_stop));
        total_gpu_ms = static_cast<double>(ms);
    }

    void add(double& dst, float ms) { dst += static_cast<double>(ms); }
};

template <class F>
void profile_mtp_cuda_stage(
    Gemma4MtpProfile& profile,
    double& dst,
    cudaStream_t stream,
    F&& fn)
{
    if (!profile.enabled) {
        fn();
        return;
    }
    CUDA_CHECK(cudaEventRecord(profile.stage_start, stream));
    fn();
    CUDA_CHECK(cudaEventRecord(profile.stage_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(profile.stage_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, profile.stage_start,
                                    profile.stage_stop));
    profile.add(dst, ms);
}

template <class F>
void profile_mtp_host_stage(
    Gemma4MtpProfile& profile,
    double& dst,
    F&& fn)
{
    if (!profile.enabled) {
        fn();
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto stop = std::chrono::steady_clock::now();
    dst += std::chrono::duration<double, std::milli>(stop - start).count();
}

void maybe_print_mtp_profile(const Gemma4MtpProfile& p) {
    if (!p.enabled) return;
    static std::atomic<std::uint64_t> seq{0};
    const std::uint64_t idx = seq.fetch_add(1, std::memory_order_relaxed) + 1;
    const std::uint64_t limit = mtp_profile_print_limit();
    if (limit != 0 && idx > limit) return;

    const double named =
        p.upload_ms + p.gather_ms + p.input_ms + p.pre_projection_ms +
        p.q_ms + p.attention_ms + p.attn_out_ms + p.mlp_ms +
        p.final_norm_ms + p.score_ms + p.argmax_ms + p.copy_ms +
        p.post_projection_ms + p.d2h_ms;
    const double other_gpu =
        p.total_gpu_ms > named ? p.total_gpu_ms - named : 0.0;
    std::cerr
        << "[pie-gemma4-mtp-profile]"
        << " seq=" << idx
        << " M=" << p.M
        << " max_drafts=" << p.active_max_drafts
        << " draft_steps=" << p.draft_steps
        << " row_steps=" << p.row_steps
        << " layers=" << p.layers
        << " compact_rows=" << (p.compact_rows ? 1 : 0)
        << " ordered=" << (p.ordered_embeddings ? 1 : 0)
        << " total_gpu_ms=" << p.total_gpu_ms
        << " upload_ms=" << p.upload_ms
        << " gather_ms=" << p.gather_ms
        << " input_ms=" << p.input_ms
        << " pre_projection_ms=" << p.pre_projection_ms
        << " q_ms=" << p.q_ms
        << " attention_ms=" << p.attention_ms
        << " attn_out_ms=" << p.attn_out_ms
        << " mlp_ms=" << p.mlp_ms
        << " final_norm_ms=" << p.final_norm_ms
        << " score_ms=" << p.score_ms
        << " argmax_ms=" << p.argmax_ms
        << " copy_ms=" << p.copy_ms
        << " post_projection_ms=" << p.post_projection_ms
        << " d2h_ms=" << p.d2h_ms
        << " other_gpu_ms=" << other_gpu
        << " host_setup_ms=" << p.host_setup_ms
        << " host_plan_ms=" << p.host_plan_ms
        << " host_emit_ms=" << p.host_emit_ms
        << "\n";
}

void load_all_safetensors(const std::filesystem::path& snapshot_dir,
                          WeightStore& store)
{
    auto loader = SafetensorsCheckpointSource::open(snapshot_dir);
    WeightStoreBuilder builder(store);
    const auto names = loader.tensor_names();
    builder.reserve(names.size());
    for (const auto& name : names) {
        const auto& info = loader.info(name);
        DeviceTensor t = DeviceTensor::allocate(info.dtype, info.shape);
        loader.copy_to_device(name, t.data(), info.shape);
        builder.insert(name, std::move(t));
    }
    builder.finalize();
    CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace

Gemma4MtpWorkspace Gemma4MtpWorkspace::allocate(
    const Gemma4MtpWeights& w,
    int max_requests,
    int max_page_refs,
    int max_drafts)
{
    const int M = std::max(1, max_requests);
    const int P = std::max(1, max_page_refs);
    const int D = std::max(1, max_drafts);
    const int Hb = w.backbone_hidden_size;
    const int H = w.cfg.hidden_size;
    const int I = std::max(1, w.max_intermediate);
    const int Hq = std::max(1, w.max_hq);
    const int V = w.cfg.vocab_size;
    const int C = std::max(1, w.num_centroids);

    Gemma4MtpWorkspace ws;
    ws.max_requests = M;
    ws.max_page_refs = P;
    ws.max_drafts = D;
    ws.hidden = DeviceTensor::allocate(DType::BF16, {M, Hb});
    ws.combined = DeviceTensor::allocate(DType::BF16, {M, 2 * Hb});
    ws.y = DeviceTensor::allocate(DType::BF16, {M, H});
    ws.norm_x = DeviceTensor::allocate(DType::BF16, {M, H});
    ws.norm_y = DeviceTensor::allocate(DType::BF16, {M, H});
    ws.q = DeviceTensor::allocate(DType::BF16, {M, Hq});
    ws.attn_out = DeviceTensor::allocate(DType::BF16, {M, Hq});
    ws.gate = DeviceTensor::allocate(DType::BF16, {M, I});
    ws.up = DeviceTensor::allocate(DType::BF16, {M, I});
    ws.gate_up_fused = DeviceTensor::allocate(DType::BF16, {M, 2 * I});
    if (w.use_ordered_embeddings) {
        ws.centroid_logits = DeviceTensor::allocate(DType::BF16, {M, C});
        const int selected = std::max(1, w.centroid_top_k) *
                             std::max(1, w.vocab_size_per_centroid);
        const int tiles = (selected + 7) / 8;
        ws.sparse_argmax_pairs =
            DeviceTensor::allocate(DType::INT64, {tiles, M});
        ws.top_centroids = DeviceBuffer<std::int32_t>::alloc(
            static_cast<std::size_t>(M) *
            static_cast<std::size_t>(std::max(1, w.centroid_top_k)));
    } else {
        ws.logits = DeviceTensor::allocate(DType::BF16, {M, V});
        ws.greedy_pairs_all = DeviceTensor::allocate(DType::INT64, {8, M});
    }

    ws.row_indices = DeviceBuffer<std::int32_t>::alloc(M);
    ws.input_ids = DeviceBuffer<std::int32_t>::alloc(M);
    ws.positions = DeviceBuffer<std::int32_t>::alloc(
        static_cast<std::size_t>(M) * static_cast<std::size_t>(std::max(D, 1)));
    ws.sampled = DeviceBuffer<std::int32_t>::alloc(M);
    ws.draft_tokens = DeviceBuffer<std::int32_t>::alloc(
        static_cast<std::size_t>(M) * static_cast<std::size_t>(D));
    ws.qo_indptr = DeviceBuffer<std::uint32_t>::alloc(M + 1);
    ws.kv_page_indices = DeviceBuffer<std::uint32_t>::alloc(P);
    ws.kv_page_indptr = DeviceBuffer<std::uint32_t>::alloc(M + 1);
    ws.kv_last_page_lens = DeviceBuffer<std::uint32_t>::alloc(M);
    ws.attn_workspaces.reserve(w.layers.size());
    ws.decode_plans.reserve(w.layers.size());
    for (std::size_t i = 0; i < w.layers.size(); ++i) {
        ws.attn_workspaces.emplace_back(AttentionWorkspace::allocate());
        ws.decode_plans.emplace_back(ops::make_decode_plan());
    }

    ws.h_row_indices.reserve(M);
    ws.h_input_ids.reserve(M);
    ws.h_positions.reserve(M);
    ws.h_sampled = PinnedHostBuffer<std::int32_t>::alloc(
        static_cast<std::size_t>(M) * static_cast<std::size_t>(D));
    ws.h_qo_indptr.reserve(M + 1);
    ws.h_kv_page_indices.reserve(P);
    ws.h_kv_page_indptr.reserve(M + 1);
    ws.h_kv_last_page_lens.reserve(M);
    return ws;
}

Gemma4MtpWeights load_gemma4_mtp_weights(
    const std::filesystem::path& snapshot_dir,
    const std::string& device,
    const HfConfig& target_cfg,
    const Gemma4Weights& target_weights,
    const Gemma4MtpRuntimeConfig& runtime,
    bool verbose)
{
    if (snapshot_dir.empty()) {
        throw std::runtime_error("gemma4_mtp: assistant snapshot dir is empty");
    }
    int dev_id = 0;
    const auto colon = device.find(':');
    const std::string id_str =
        colon == std::string::npos ? device : device.substr(colon + 1);
    if (!id_str.empty()) dev_id = std::stoi(id_str);
    CUDA_CHECK(cudaSetDevice(dev_id));

    Gemma4MtpWeights w;
    w.cfg = parse_hf_config(snapshot_dir / "config.json");
    w.backbone_hidden_size = target_cfg.hidden_size;
    w.backbone_vocab_size = target_cfg.vocab_size;
    load_all_safetensors(snapshot_dir, w.store);

    w.pre_projection = &must_any(
        w.store,
        {"pre_projection.weight", "model.pre_projection.weight",
         "draft_model.pre_projection.weight",
         "draft_model.model.pre_projection.weight"},
        "pre_projection.weight");
    w.post_projection = &must_any(
        w.store,
        {"post_projection.weight", "model.post_projection.weight",
         "draft_model.post_projection.weight",
         "draft_model.model.post_projection.weight"},
        "post_projection.weight");
    w.final_norm = &must_any(
        w.store,
        {"model.norm.weight", "draft_model.model.norm.weight"},
        "model.norm.weight");
    if (const DeviceTensor* lm = find_any(
            w.store,
            {"lm_head.weight", "draft_model.lm_head.weight",
             "model.lm_head.weight"})) {
        w.lm_head = lm;
    } else {
        w.lm_head = &must_any(
            w.store,
            {"model.embed_tokens.weight",
             "draft_model.model.embed_tokens.weight"},
            "lm_head/model.embed_tokens.weight");
    }
    w.masked_centroids = find_any(
        w.store,
        {"masked_embedding.centroids.weight",
         "draft_model.masked_embedding.centroids.weight",
         "model.masked_embedding.centroids.weight"});
    w.token_ordering = find_any(
        w.store,
        {"masked_embedding.token_ordering",
         "draft_model.masked_embedding.token_ordering",
         "model.masked_embedding.token_ordering"});
    const bool ordered_embeddings_available =
        w.cfg.gemma4_use_ordered_embeddings ||
        (w.masked_centroids != nullptr && w.token_ordering != nullptr);
    // The assistant checkpoint ships centroid/order tensors, but Pie's
    // current sparse top-token kernel is scalar and slower than the
    // tensor-core full-vocab GEMM for the Gemma4 assistant dimensions.
    // Keep the sparse path available for diagnostics.
    w.use_ordered_embeddings = false;
    if (const char* force_ordered =
            std::getenv("PIE_GEMMA4_MTP_ORDERED_EMBEDDINGS");
        ordered_embeddings_available && force_ordered != nullptr &&
        std::string(force_ordered) == "1") {
        w.use_ordered_embeddings = true;
    }
    if (const char* full_vocab = std::getenv("PIE_GEMMA4_MTP_FULL_VOCAB");
        full_vocab != nullptr && std::string(full_vocab) == "1") {
        w.use_ordered_embeddings = false;
    }
    if (w.use_ordered_embeddings) {
        if (w.masked_centroids == nullptr || w.token_ordering == nullptr) {
            throw std::runtime_error(
                "gemma4_mtp: ordered embeddings requested but "
                "masked_embedding tensors are missing");
        }
        if (w.masked_centroids->dtype() != DType::BF16 ||
            w.token_ordering->dtype() != DType::INT64) {
            throw std::runtime_error(
                "gemma4_mtp: ordered embeddings require bf16 centroids "
                "and int64 token_ordering");
        }
        w.num_centroids = w.cfg.gemma4_num_centroids > 0
            ? w.cfg.gemma4_num_centroids
            : dim0(*w.masked_centroids, "masked_embedding.centroids");
        w.centroid_top_k = w.cfg.gemma4_centroid_intermediate_top_k > 0
            ? w.cfg.gemma4_centroid_intermediate_top_k
            : 32;
        if (w.num_centroids <= 0 ||
            w.cfg.vocab_size % w.num_centroids != 0) {
            throw std::runtime_error(
                "gemma4_mtp: vocab_size is not divisible by num_centroids");
        }
        w.vocab_size_per_centroid = w.cfg.vocab_size / w.num_centroids;
        const auto& centroid_shape = w.masked_centroids->shape();
        if (centroid_shape.size() != 2 ||
            centroid_shape[0] != w.num_centroids ||
            centroid_shape[1] != w.cfg.hidden_size) {
            throw std::runtime_error(
                "gemma4_mtp: masked_embedding.centroids shape mismatch");
        }
        if (static_cast<int>(w.token_ordering->numel()) < w.cfg.vocab_size) {
            throw std::runtime_error(
                "gemma4_mtp: masked_embedding.token_ordering is too small");
        }
    }

    const int L = w.cfg.num_hidden_layers;
    w.layers.resize(L);
    w.owned_gate_up_fused.reserve(static_cast<std::size_t>(L));
    for (int i = 0; i < L; ++i) {
        const std::string p = layer_prefix(i);
        const std::string dp = draft_layer_prefix(i);
        auto& layer = w.layers[i];
        layer.input_norm = &must_any(
            w.store, {p + "input_layernorm.weight",
                      dp + "input_layernorm.weight"},
            "input_layernorm.weight");
        layer.post_attn_norm = &must_any(
            w.store, {p + "post_attention_layernorm.weight",
                      dp + "post_attention_layernorm.weight"},
            "post_attention_layernorm.weight");
        layer.pre_mlp_norm = &must_any(
            w.store, {p + "pre_feedforward_layernorm.weight",
                      dp + "pre_feedforward_layernorm.weight"},
            "pre_feedforward_layernorm.weight");
        layer.post_mlp_norm = &must_any(
            w.store, {p + "post_feedforward_layernorm.weight",
                      dp + "post_feedforward_layernorm.weight"},
            "post_feedforward_layernorm.weight");

        layer.q_proj = &must_any(
            w.store, {p + "self_attn.q_proj.weight",
                      dp + "self_attn.q_proj.weight"},
            "self_attn.q_proj.weight");
        layer.o_proj = &must_any(
            w.store, {p + "self_attn.o_proj.weight",
                      dp + "self_attn.o_proj.weight"},
            "self_attn.o_proj.weight");
        layer.q_norm = &must_any(
            w.store, {p + "self_attn.q_norm.weight",
                      dp + "self_attn.q_norm.weight"},
            "self_attn.q_norm.weight");

        layer.gate_proj = &must_any(
            w.store, {p + "mlp.gate_proj.weight",
                      dp + "mlp.gate_proj.weight"},
            "mlp.gate_proj.weight");
        layer.up_proj = &must_any(
            w.store, {p + "mlp.up_proj.weight",
                      dp + "mlp.up_proj.weight"},
            "mlp.up_proj.weight");
        layer.down_proj = &must_any(
            w.store, {p + "mlp.down_proj.weight",
                      dp + "mlp.down_proj.weight"},
            "mlp.down_proj.weight");
        w.owned_gate_up_fused.push_back(
            make_gate_up_fused_weight(*layer.gate_proj, *layer.up_proj));
        layer.gate_up_proj_fused = &w.owned_gate_up_fused.back();

        layer.layer_scalar_value = read_scalar_once(find_any(
            w.store, {p + "layer_scalar", dp + "layer_scalar"}));

        const int q_rows = dim0(*layer.q_proj, "q_proj");
        layer.head_dim = q_rows / std::max(1, w.cfg.num_attention_heads);
        const int intermediate = dim0(*layer.gate_proj, "gate_proj");
        w.max_hq = std::max(w.max_hq, q_rows);
        w.max_intermediate = std::max(w.max_intermediate, intermediate);

        const std::string layer_type =
            (i < static_cast<int>(w.cfg.layer_types.size()))
                ? w.cfg.layer_types[i]
                : std::string("full_attention");
        layer.target_kv_layer = choose_target_kv_layer(
            target_cfg, layer_type, i, L);
        const auto& target_layer = target_weights.layers[layer.target_kv_layer];
        if (target_layer.head_dim != layer.head_dim) {
            throw std::runtime_error(
                "gemma4_mtp: draft layer head_dim does not match target KV layer");
        }
        layer.window_left =
            (layer_type == "sliding_attention") ? w.cfg.sliding_window : -1;
        layer.rope_theta =
            (i < static_cast<int>(w.cfg.gemma_per_layer_rope_theta.size()))
                ? w.cfg.gemma_per_layer_rope_theta[i]
                : ((layer_type == "sliding_attention" &&
                    w.cfg.rope_local_base_freq > 0.f)
                       ? w.cfg.rope_local_base_freq
                       : w.cfg.rope_theta);
        layer.partial_rotary_factor =
            (i < static_cast<int>(
                     w.cfg.gemma_per_layer_partial_rotary_factor.size()))
                ? w.cfg.gemma_per_layer_partial_rotary_factor[i]
                : 1.0f;
    }

    if (verbose) {
        std::cerr << "[pie-driver-cuda] gemma4 MTP loaded "
                  << w.store.size() << " tensors from "
                  << snapshot_dir.string()
                  << "; layers=" << w.layers.size()
                  << " draft_hidden=" << w.cfg.hidden_size
                  << " backbone_hidden=" << w.backbone_hidden_size
                  << " ordered_embeddings=" << (w.use_ordered_embeddings ? "on" : "off")
                  << " centroids=" << w.num_centroids
                  << " centroid_top_k=" << w.centroid_top_k
                  << " max_hq=" << w.max_hq
                  << " max_intermediate=" << w.max_intermediate
                  << " kv_map=" << mtp_kv_map_mode()
                  << " pos_offset=" << mtp_position_offset()
                  << " adaptive_drafts=" << (mtp_adaptive_drafts_enabled(runtime) ? "on" : "off")
                  << " initial_drafts=" << mtp_initial_adaptive_drafts(
                         runtime, std::max(1, runtime.initial_drafts))
                  << " min_drafts=" << mtp_min_adaptive_drafts(
                         runtime, std::max(1, runtime.min_drafts))
                  << " compact_draft_rows=" << env_string_or(
                         "PIE_GEMMA4_MTP_COMPACT_DRAFT_ROWS",
                         runtime.compact_draft_rows)
                  << " cuda_graph=" << (mtp_cuda_graph_enabled(runtime) ? "on" : "off")
                  << " max_draft_batch_rows=" << mtp_max_draft_batch_rows(runtime)
                  << " argmax_parts=" << mtp_argmax_parts()
                  << " reverse_preproj=" << (mtp_reverse_preprojection_concat() ? "on" : "off")
                  << " layer_scalar=" << (mtp_disable_layer_scalar() ? "off" : "on")
                  << "\n";
    }
    return w;
}

void gemma4_mtp_draft(
    const Gemma4MtpWeights& w,
    const Gemma4Weights& target_weights,
    Gemma4MtpWorkspace& ws,
    const Gemma4MtpRuntimeConfig& runtime,
    const SystemSpecDraftInputs& in,
    std::span<pie_driver::PerRequestOutput> per_request)
{
    const int max_drafts = std::min(in.max_drafts, ws.max_drafts);
    if (max_drafts <= 0 || in.requests.empty()) return;
    const int Hb = w.backbone_hidden_size;
    const int H = w.cfg.hidden_size;
    const int V = w.cfg.vocab_size;
    const int target_vocab = w.backbone_vocab_size;
    const float eps = w.cfg.rms_norm_eps;
    cudaStream_t stream = in.cublas.stream();
    Gemma4MtpProfile profile;
    profile.begin(stream);
    const auto host_setup_start = std::chrono::steady_clock::now();

    ws.h_row_indices.clear();
    ws.h_input_ids.clear();
    ws.h_positions.clear();
    ws.h_qo_indptr.clear();
    ws.h_kv_page_indices.clear();
    ws.h_kv_page_indptr.clear();
    ws.h_kv_last_page_lens.clear();

    std::vector<int> active_req;
    std::vector<int> active_desired_drafts;
    active_req.reserve(in.requests.size());
    active_desired_drafts.reserve(in.requests.size());
    ws.h_qo_indptr.push_back(0);
    ws.h_kv_page_indptr.push_back(0);
    int active_max_drafts = 0;

    for (std::size_t j = 0; j < in.requests.size(); ++j) {
        const auto& req = in.requests[j];
        if (req.request_index < 0 ||
            req.request_index >= static_cast<int>(per_request.size()) ||
            req.source_row < 0) {
            continue;
        }
        const int r = req.request_index;
        if (in.kv_page_indptr.size() < static_cast<std::size_t>(r + 2)) {
            continue;
        }
        const std::uint32_t page_lo = in.kv_page_indptr[r];
        const std::uint32_t page_hi = in.kv_page_indptr[r + 1];
        if (page_hi <= page_lo) continue;

        const std::uint32_t prefix_len = req.source_position + 1;
        const std::uint32_t needed_pages =
            (prefix_len + static_cast<std::uint32_t>(in.page_size) - 1) /
            static_cast<std::uint32_t>(in.page_size);
        const std::uint32_t available_pages = page_hi - page_lo;
        const std::uint32_t use_pages =
            std::min(needed_pages, available_pages);
        if (use_pages == 0) continue;
        if (ws.h_kv_page_indices.size() + use_pages >
            static_cast<std::size_t>(ws.max_page_refs)) {
            break;
        }
        if (active_req.size() >= static_cast<std::size_t>(ws.max_requests)) {
            break;
        }

        active_req.push_back(static_cast<int>(j));
        const int desired_drafts =
            mtp_desired_drafts(req, max_drafts, runtime);
        active_desired_drafts.push_back(desired_drafts);
        active_max_drafts = std::max(active_max_drafts, desired_drafts);
        ws.h_row_indices.push_back(req.source_row);
        ws.h_input_ids.push_back(static_cast<std::int32_t>(req.accepted_token));
        ws.h_positions.push_back(static_cast<std::int32_t>(
            static_cast<int>(req.source_position) + mtp_position_offset()));
        ws.h_qo_indptr.push_back(static_cast<std::uint32_t>(active_req.size()));
        for (std::uint32_t p = 0; p < use_pages; ++p) {
            ws.h_kv_page_indices.push_back(in.kv_page_indices[page_lo + p]);
        }
        ws.h_kv_page_indptr.push_back(
            static_cast<std::uint32_t>(ws.h_kv_page_indices.size()));
        std::uint32_t last_len =
            prefix_len % static_cast<std::uint32_t>(in.page_size);
        if (last_len == 0) last_len = static_cast<std::uint32_t>(in.page_size);
        ws.h_kv_last_page_lens.push_back(last_len);
    }

    const int M = static_cast<int>(active_req.size());
    if (M <= 0) return;
    const int max_draft_batch_rows = mtp_max_draft_batch_rows(runtime);
    if (max_draft_batch_rows > 0 && M > max_draft_batch_rows) return;
    active_max_drafts = std::clamp(active_max_drafts, 1, max_drafts);
    const bool compact_draft_rows = mtp_compact_draft_rows_enabled(
        runtime, active_desired_drafts, active_max_drafts);
    if (compact_draft_rows && M > 1) {
        std::vector<int> order(static_cast<std::size_t>(M));
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(
            order.begin(), order.end(),
            [&](int a, int b) {
                return active_desired_drafts[static_cast<std::size_t>(a)] >
                       active_desired_drafts[static_cast<std::size_t>(b)];
            });
        bool identity = true;
        for (int i = 0; i < M; ++i) {
            if (order[static_cast<std::size_t>(i)] != i) {
                identity = false;
                break;
            }
        }
        if (!identity) {
            const auto old_active_req = active_req;
            const auto old_desired = active_desired_drafts;
            const auto old_rows = ws.h_row_indices;
            const auto old_ids = ws.h_input_ids;
            const auto old_positions = ws.h_positions;
            const auto old_pages = ws.h_kv_page_indices;
            const auto old_page_indptr = ws.h_kv_page_indptr;
            const auto old_last_lens = ws.h_kv_last_page_lens;

            ws.h_kv_page_indices.clear();
            ws.h_kv_page_indptr.clear();
            ws.h_kv_page_indptr.push_back(0);
            for (int dst = 0; dst < M; ++dst) {
                const int src = order[static_cast<std::size_t>(dst)];
                active_req[static_cast<std::size_t>(dst)] =
                    old_active_req[static_cast<std::size_t>(src)];
                active_desired_drafts[static_cast<std::size_t>(dst)] =
                    old_desired[static_cast<std::size_t>(src)];
                ws.h_row_indices[static_cast<std::size_t>(dst)] =
                    old_rows[static_cast<std::size_t>(src)];
                ws.h_input_ids[static_cast<std::size_t>(dst)] =
                    old_ids[static_cast<std::size_t>(src)];
                ws.h_positions[static_cast<std::size_t>(dst)] =
                    old_positions[static_cast<std::size_t>(src)];
                ws.h_kv_last_page_lens[static_cast<std::size_t>(dst)] =
                    old_last_lens[static_cast<std::size_t>(src)];
                const std::size_t page_lo =
                    old_page_indptr[static_cast<std::size_t>(src)];
                const std::size_t page_hi =
                    old_page_indptr[static_cast<std::size_t>(src + 1)];
                ws.h_kv_page_indices.insert(
                    ws.h_kv_page_indices.end(),
                    old_pages.begin() + static_cast<std::ptrdiff_t>(page_lo),
                    old_pages.begin() + static_cast<std::ptrdiff_t>(page_hi));
                ws.h_kv_page_indptr.push_back(static_cast<std::uint32_t>(
                    ws.h_kv_page_indices.size()));
            }
        }
        ws.h_qo_indptr.resize(static_cast<std::size_t>(M + 1));
        for (int i = 0; i <= M; ++i) {
            ws.h_qo_indptr[static_cast<std::size_t>(i)] =
                static_cast<std::uint32_t>(i);
        }
    }

    if (active_max_drafts > 1) {
        const auto base_positions = ws.h_positions;
        ws.h_positions.resize(
            static_cast<std::size_t>(M) *
            static_cast<std::size_t>(active_max_drafts));
        for (int step = 1; step < active_max_drafts; ++step) {
            for (int i = 0; i < M; ++i) {
                ws.h_positions[static_cast<std::size_t>(step) * M + i] =
                    base_positions[i] + step;
            }
        }
    }

    if (profile.enabled) {
        profile.M = M;
        profile.active_max_drafts = active_max_drafts;
        profile.compact_rows = compact_draft_rows;
        profile.ordered_embeddings = w.use_ordered_embeddings;
        profile.layers = static_cast<int>(w.layers.size());
        profile.host_setup_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - host_setup_start)
                .count();
    }

    profile_mtp_cuda_stage(profile, profile.upload_ms, stream, [&] {
        ws.row_indices.copy_from_host(std::span<const std::int32_t>(
            ws.h_row_indices.data(), static_cast<std::size_t>(M)));
        ws.positions.copy_from_host(std::span<const std::int32_t>(
            ws.h_positions.data(),
            static_cast<std::size_t>(M) *
                static_cast<std::size_t>(active_max_drafts)));
        ws.input_ids.copy_from_host(std::span<const std::int32_t>(
            ws.h_input_ids.data(), static_cast<std::size_t>(M)));
        ws.qo_indptr.copy_from_host(std::span<const std::uint32_t>(
            ws.h_qo_indptr.data(), static_cast<std::size_t>(M + 1)));
        ws.kv_page_indices.copy_from_host(std::span<const std::uint32_t>(
            ws.h_kv_page_indices.data(), ws.h_kv_page_indices.size()));
        ws.kv_page_indptr.copy_from_host(std::span<const std::uint32_t>(
            ws.h_kv_page_indptr.data(), static_cast<std::size_t>(M + 1)));
        ws.kv_last_page_lens.copy_from_host(std::span<const std::uint32_t>(
            ws.h_kv_last_page_lens.data(), static_cast<std::size_t>(M)));
    });

    if (ws.decode_plans.size() < w.layers.size() ||
        ws.attn_workspaces.size() < w.layers.size()) {
        throw std::runtime_error("gemma4_mtp: workspace missing attention plans");
    }
    auto plan_decode_rows =
        [&](int rows, cudaStream_t plan_stream, bool enable_cuda_graph) {
            for (std::size_t li = 0; li < w.layers.size(); ++li) {
                const auto& layer = w.layers[li];
                const int d = layer.head_dim;
                const int num_q_heads = w.cfg.num_attention_heads;
                const auto& target_layer =
                    target_weights.layers[layer.target_kv_layer];
                const int num_kv_heads = target_layer.num_kv_heads;
                ops::plan_attention_flashinfer_decode(
                    *ws.decode_plans[li],
                    ws.h_kv_page_indptr.data(),
                    rows, num_q_heads, num_kv_heads, d,
                    in.page_size, ws.attn_workspaces[li], plan_stream,
                    enable_cuda_graph,
                    /*full_attention_variant=*/layer.window_left < 0,
                    in.kv_cache.hnd_layout());
            }
        };

    profile_mtp_cuda_stage(profile, profile.gather_ms, stream, [&] {
        kernels::launch_gather_bf16_rows(
            static_cast<const std::uint16_t*>(
                (mtp_use_target_residual_hidden()
                     ? in.target_ws.y.data()
                     : in.target_ws.norm_x.data())),
            ws.row_indices.data(),
            static_cast<std::uint16_t*>(ws.hidden.data()),
            M, Hb, stream);
    });

    auto run_steps =
        [&](Gemma4MtpProfile& profile, cudaStream_t stream, bool plan_inside) {
        int planned_decode_rows = -1;
        for (int step = 0; step < active_max_drafts; ++step) {
        int step_rows = M;
        if (compact_draft_rows) {
            while (step_rows > 0 &&
                   active_desired_drafts[static_cast<std::size_t>(step_rows - 1)] <=
                       step) {
                --step_rows;
            }
        }
        if (step_rows <= 0) break;
        if (profile.enabled) {
            ++profile.draft_steps;
            profile.row_steps += step_rows;
        }

        const std::int32_t* input_ids =
            (step == 0) ? ws.input_ids.data() : ws.sampled.data();
        profile_mtp_cuda_stage(profile, profile.input_ms, stream, [&] {
            kernels::launch_embed_scaled_concat_bf16(
                input_ids, target_weights.embed->data(),
                static_cast<const std::uint16_t*>(ws.hidden.data()),
                static_cast<std::uint16_t*>(ws.combined.data()),
                step_rows, Hb, target_vocab,
                std::sqrt(static_cast<float>(Hb)),
                mtp_reverse_preprojection_concat(), stream);
        });

        if (plan_inside && planned_decode_rows != step_rows) {
            profile_mtp_host_stage(profile, profile.host_plan_ms, [&] {
                plan_decode_rows(
                    step_rows, stream, /*enable_cuda_graph=*/false);
            });
            planned_decode_rows = step_rows;
        }

        profile_mtp_cuda_stage(profile, profile.pre_projection_ms, stream, [&] {
            ops::gemm_act_x_wt_bf16(
                in.cublas.handle(), ws.combined.data(),
                w.pre_projection->data(), ws.y.data(),
                step_rows, H, 2 * Hb);
        });

        for (std::size_t li = 0; li < w.layers.size(); ++li) {
            const auto& layer = w.layers[li];
            const int d = layer.head_dim;
            const int num_q_heads = w.cfg.num_attention_heads;
            const int Hq = num_q_heads * d;
            const int I = dim0(*layer.gate_proj, "mtp gate_proj");
            profile_mtp_cuda_stage(profile, profile.q_ms, stream, [&] {
                kernels::launch_rmsnorm_bf16(
                    ws.y.data(), layer.input_norm->data(),
                    ws.norm_x.data(), step_rows, H, eps, stream);
                ops::gemm_act_x_wt_bf16(
                    in.cublas.handle(), ws.norm_x.data(),
                    layer.q_proj->data(), ws.q.data(), step_rows, Hq, H);
                kernels::launch_rmsnorm_bf16(
                    ws.q.data(), layer.q_norm->data(), ws.q.data(),
                    step_rows * num_q_heads, d, eps, stream);

                const std::int32_t* step_positions =
                    ws.positions.data() +
                    static_cast<std::size_t>(step) *
                        static_cast<std::size_t>(M);
                const int rotary_dim =
                    static_cast<int>(layer.partial_rotary_factor * d);
                const bool partial =
                    layer.partial_rotary_factor < 1.0f && rotary_dim > 0;
                if (partial) {
                    kernels::launch_rope_partial_bf16(
                        ws.q.data(), ws.q.data(), step_positions,
                        step_rows, num_q_heads, /*num_kv_heads=*/0, d,
                        rotary_dim, layer.rope_theta, stream);
                } else {
                    kernels::launch_rope_bf16(
                        ws.q.data(), ws.q.data(), step_positions,
                        step_rows, num_q_heads, /*num_kv_heads=*/0, d,
                        layer.rope_theta, stream);
                }
            });

            profile_mtp_cuda_stage(profile, profile.attention_ms, stream, [&] {
                ops::dispatch_attention_flashinfer_decode(
                    *ws.decode_plans[li],
                    ws.q.data(), in.kv_cache.layer_view(layer.target_kv_layer),
                    ws.attn_out.data(),
                    ws.kv_page_indices.data(), ws.kv_page_indptr.data(),
                    ws.kv_last_page_lens.data(),
                    ws.attn_workspaces[li], stream,
                    layer.window_left,
                    /*logits_soft_cap=*/0.f,
                    /*sm_scale=*/1.0f);
            });

            profile_mtp_cuda_stage(profile, profile.attn_out_ms, stream, [&] {
                ops::gemm_act_x_wt_bf16(
                    in.cublas.handle(), ws.attn_out.data(),
                    layer.o_proj->data(), ws.norm_x.data(),
                    step_rows, H, Hq, /*beta=*/0.f);
                kernels::launch_rmsnorm_residual_add_scale_rmsnorm_bf16(
                    ws.norm_x.data(), layer.post_attn_norm->data(),
                    ws.y.data(), 1.f, layer.pre_mlp_norm->data(),
                    ws.norm_x.data(), step_rows, H, eps, stream);
            });

            profile_mtp_cuda_stage(profile, profile.mlp_ms, stream, [&] {
                if (layer.gate_up_proj_fused != nullptr) {
                    ops::gemm_act_x_wt_bf16_cublas(
                        in.cublas.handle(), ws.norm_x.data(),
                        layer.gate_up_proj_fused->data(),
                        ws.gate_up_fused.data(), step_rows, 2 * I, H);
                    kernels::launch_chunked_geglu_tanh_bf16(
                        ws.gate_up_fused.data(), ws.gate.data(),
                        step_rows, I, stream);
                } else {
                    ops::gemm_act_x_wt_bf16(
                        in.cublas.handle(), ws.norm_x.data(),
                        layer.gate_proj->data(), ws.gate.data(),
                        step_rows, I, H);
                    ops::gemm_act_x_wt_bf16(
                        in.cublas.handle(), ws.norm_x.data(),
                        layer.up_proj->data(), ws.up.data(), step_rows, I, H);
                    kernels::launch_geglu_tanh_bf16(
                        ws.gate.data(), ws.up.data(), ws.gate.data(),
                        step_rows * I, stream);
                }
                ops::gemm_act_x_wt_bf16(
                    in.cublas.handle(), ws.gate.data(),
                    layer.down_proj->data(), ws.norm_x.data(),
                    step_rows, H, I, /*beta=*/0.f);
                kernels::launch_rmsnorm_residual_add_bf16(
                    ws.norm_x.data(), layer.post_mlp_norm->data(),
                    ws.y.data(), step_rows, H, eps, stream);
                if (!mtp_disable_layer_scalar() &&
                    std::abs(layer.layer_scalar_value - 1.f) > 1e-6f) {
                    kernels::launch_scalar_mul_bf16(
                        ws.y.data(), layer.layer_scalar_value,
                        static_cast<std::size_t>(step_rows) * H, stream);
                }
            });
        }

        profile_mtp_cuda_stage(profile, profile.final_norm_ms, stream, [&] {
            kernels::launch_rmsnorm_bf16(
                ws.y.data(), w.final_norm->data(), ws.norm_x.data(),
                step_rows, H, eps, stream);
        });
        if (w.use_ordered_embeddings) {
            const int selected =
                w.centroid_top_k * w.vocab_size_per_centroid;
            const int tiles = (selected + 7) / 8;
            profile_mtp_cuda_stage(profile, profile.score_ms, stream, [&] {
                ops::gemm_act_x_wt_bf16(
                    in.cublas.handle(), ws.norm_x.data(),
                    w.masked_centroids->data(), ws.centroid_logits.data(),
                    step_rows, w.num_centroids, H);
                kernels::launch_topk_centroids_bf16(
                    ws.centroid_logits.data(), ws.top_centroids.data(),
                    step_rows, w.num_centroids, w.centroid_top_k, stream);
                kernels::launch_masked_embedding_tile_argmax_pairs_bf16(
                    ws.top_centroids.data(), ws.norm_x.data(),
                    w.lm_head->data(),
                    static_cast<const std::int64_t*>(
                        w.token_ordering->data()),
                    reinterpret_cast<std::uint64_t*>(
                    ws.sparse_argmax_pairs.data()),
                    step_rows, H, w.centroid_top_k,
                    w.vocab_size_per_centroid, tiles, stream);
            });
            profile_mtp_cuda_stage(profile, profile.argmax_ms, stream, [&] {
                kernels::launch_select_global_argmax_pairs(
                    reinterpret_cast<const std::uint64_t*>(
                        ws.sparse_argmax_pairs.data()),
                    ws.sampled.data(), step_rows, tiles, stream);
            });
        } else {
            profile_mtp_cuda_stage(profile, profile.score_ms, stream, [&] {
                ops::gemm_act_x_wt_bf16(
                    in.cublas.handle(), ws.norm_x.data(),
                    w.lm_head->data(), ws.logits.data(),
                    step_rows, V, H);
            });
            const int argmax_parts = mtp_argmax_parts();
            if (argmax_parts > 1 && !ws.greedy_pairs_all.empty()) {
                profile_mtp_cuda_stage(profile, profile.argmax_ms, stream, [&] {
                    kernels::launch_argmax_bf16_partitioned_pairs(
                        ws.logits.data(),
                        reinterpret_cast<std::uint64_t*>(
                            ws.greedy_pairs_all.data()),
                        step_rows, V, argmax_parts, stream);
                    kernels::launch_select_global_argmax_pairs(
                        reinterpret_cast<const std::uint64_t*>(
                            ws.greedy_pairs_all.data()),
                        ws.sampled.data(), step_rows, argmax_parts, stream);
                });
            } else {
                profile_mtp_cuda_stage(profile, profile.argmax_ms, stream, [&] {
                    kernels::launch_argmax_bf16(
                        ws.logits.data(), ws.sampled.data(),
                        step_rows, V, stream);
                });
            }
        }

        profile_mtp_cuda_stage(profile, profile.copy_ms, stream, [&] {
            CUDA_CHECK(cudaMemcpyAsync(
                ws.draft_tokens.data() +
                    static_cast<std::size_t>(step) *
                        static_cast<std::size_t>(M),
                ws.sampled.data(),
                sizeof(std::int32_t) * static_cast<std::size_t>(step_rows),
                cudaMemcpyDeviceToDevice, stream));
        });

        if (step + 1 < active_max_drafts) {
            // The post-projected backbone hidden is only feedback for the
            // following draft step. There is no consumer after the final
            // draft token, so skip that GEMM and write directly into the
            // next-step input buffer.
            profile_mtp_cuda_stage(profile, profile.post_projection_ms, stream, [&] {
                ops::gemm_act_x_wt_bf16(
                    in.cublas.handle(), ws.norm_x.data(),
                    w.post_projection->data(), ws.hidden.data(),
                    step_rows, Hb, H, /*beta=*/0.f);
                });
        }
        }
    };

    const bool graph_steps =
        mtp_cuda_graph_enabled(runtime) &&
        !profile.enabled &&
        !compact_draft_rows &&
        active_max_drafts > 0;
    if (graph_steps) {
        profile_mtp_host_stage(profile, profile.host_plan_ms, [&] {
            plan_decode_rows(M, stream, /*enable_cuda_graph=*/true);
        });

        const MtpCudaGraphKey key{
            .workspace = &ws,
            .rows = M,
            .drafts = active_max_drafts,
            .argmax_parts = mtp_argmax_parts(),
            .ordered_embeddings = w.use_ordered_embeddings,
        };
        cudaGraphExec_t exec = nullptr;
        {
            std::lock_guard<std::mutex> lk(mtp_cuda_graph_cache_mutex());
            auto it = mtp_cuda_graph_cache().find(key);
            if (it != mtp_cuda_graph_cache().end()) exec = it->second;
        }
        if (exec == nullptr) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            cudaStream_t capture_stream = nullptr;
            CUDA_CHECK(cudaStreamCreateWithFlags(
                &capture_stream, cudaStreamNonBlocking));
            const cudaStream_t original_stream = in.cublas.stream();
            in.cublas.set_stream(capture_stream);
            Gemma4MtpProfile capture_profile;
            CUDA_CHECK(cudaStreamBeginCapture(
                capture_stream, cudaStreamCaptureModeRelaxed));
            run_steps(
                capture_profile, capture_stream,
                /*plan_inside=*/false);
            cudaGraph_t graph = nullptr;
            CUDA_CHECK(cudaStreamEndCapture(capture_stream, &graph));
            in.cublas.set_stream(original_stream);

            cudaGraphExec_t captured_exec = nullptr;
            CUDA_CHECK(cudaGraphInstantiate(
                &captured_exec, graph, nullptr, nullptr, 0));
            CUDA_CHECK(cudaGraphUpload(captured_exec, stream));
            CUDA_CHECK(cudaGraphDestroy(graph));
            CUDA_CHECK(cudaStreamDestroy(capture_stream));

            {
                std::lock_guard<std::mutex> lk(mtp_cuda_graph_cache_mutex());
                auto [it, inserted] =
                    mtp_cuda_graph_cache().emplace(key, captured_exec);
                if (inserted) {
                    exec = captured_exec;
                } else {
                    exec = it->second;
                    CUDA_CHECK(cudaGraphExecDestroy(captured_exec));
                }
            }
        }
        CUDA_CHECK(cudaGraphLaunch(exec, stream));
    } else {
        run_steps(profile, stream, /*plan_inside=*/true);
    }

    profile_mtp_cuda_stage(profile, profile.d2h_ms, stream, [&] {
        CUDA_CHECK(cudaMemcpyAsync(
            ws.h_sampled.data(), ws.draft_tokens.data(),
            sizeof(std::int32_t) * static_cast<std::size_t>(M) *
                static_cast<std::size_t>(active_max_drafts),
            cudaMemcpyDeviceToHost, stream));
    });
    CUDA_CHECK(cudaStreamSynchronize(stream));
    profile.end(stream);

    profile_mtp_host_stage(profile, profile.host_emit_ms, [&] {
        for (int step = 0; step < active_max_drafts; ++step) {
            for (int j = 0; j < M; ++j) {
                if (step >= active_desired_drafts[static_cast<std::size_t>(j)]) {
                    continue;
                }
                const auto& req = in.requests[active_req[j]];
                auto& out = per_request[req.request_index];
                const std::uint32_t token =
                    static_cast<std::uint32_t>(
                        ws.h_sampled[static_cast<std::size_t>(step) *
                                         static_cast<std::size_t>(M) +
                                     static_cast<std::size_t>(j)]);
                out.spec_tokens.push_back(token);
                out.spec_positions.push_back(req.first_draft_position + step);
            }
        }
    });
    maybe_print_mtp_profile(profile);
}

void gemma4_mtp_forward_step(
    const Gemma4MtpWeights& w,
    const Gemma4Weights& target_weights,
    Gemma4MtpWorkspace& mtp_ws,
    Qwen3Workspace& target_ws,
    KvCache& kv_cache,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* position_ids,
    const std::int32_t* base_hidden_row_indices,
    const std::int32_t* request_ids,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    int num_tokens,
    int draft_step,
    int max_global_tokens)
{
    if (num_tokens <= 0) return;
    const int Hb = w.backbone_hidden_size;
    const int H = w.cfg.hidden_size;
    const int V = w.cfg.vocab_size;
    const int target_vocab = w.backbone_vocab_size;
    const float eps = w.cfg.rms_norm_eps;
    cudaStream_t stream = cublas.stream();
    const int S = num_tokens;

    // 1. Gather hidden states from target model
    kernels::launch_gather_bf16_rows(
        static_cast<const std::uint16_t*>(
            mtp_use_target_residual_hidden()
                ? target_ws.y.data()
                : target_ws.norm_x.data()),
        base_hidden_row_indices,
        static_cast<std::uint16_t*>(mtp_ws.hidden.data()),
        S, Hb, stream);

    // 2. Embed + concat (token_ids comes from the executor for all steps)
    kernels::launch_embed_scaled_concat_bf16(
        token_ids, target_weights.embed->data(),
        static_cast<const std::uint16_t*>(mtp_ws.hidden.data()),
        static_cast<std::uint16_t*>(mtp_ws.combined.data()),
        S, Hb, target_vocab,
        std::sqrt(static_cast<float>(Hb)),
        mtp_reverse_preprojection_concat(), stream);

    // 3. Pre-projection
    ops::gemm_act_x_wt_bf16(
        cublas.handle(), mtp_ws.combined.data(),
        w.pre_projection->data(), mtp_ws.y.data(),
        S, H, 2 * Hb);

    // 4. Transformer layers
    for (std::size_t li = 0; li < w.layers.size(); ++li) {
        const auto& layer = w.layers[li];
        const int d = layer.head_dim;
        const int num_q_heads = w.cfg.num_attention_heads;
        const int Hq = num_q_heads * d;
        const int I = dim0(*layer.gate_proj, "mtp gate_proj");

        // Q projection + norm + RoPE
        kernels::launch_rmsnorm_bf16(
            mtp_ws.y.data(), layer.input_norm->data(),
            mtp_ws.norm_x.data(), S, H, eps, stream);
        ops::gemm_act_x_wt_bf16(
            cublas.handle(), mtp_ws.norm_x.data(),
            layer.q_proj->data(), mtp_ws.q.data(), S, Hq, H);
        kernels::launch_rmsnorm_bf16(
            mtp_ws.q.data(), layer.q_norm->data(), mtp_ws.q.data(),
            S * num_q_heads, d, eps, stream);
        const int rotary_dim =
            static_cast<int>(layer.partial_rotary_factor * d);
        const bool partial =
            layer.partial_rotary_factor < 1.0f && rotary_dim > 0;
        if (partial) {
            kernels::launch_rope_partial_bf16(
                mtp_ws.q.data(), mtp_ws.q.data(), position_ids,
                S, num_q_heads, 0, d,
                rotary_dim, layer.rope_theta, stream);
        } else {
            kernels::launch_rope_bf16(
                mtp_ws.q.data(), mtp_ws.q.data(), position_ids,
                S, num_q_heads, 0, d,
                layer.rope_theta, stream);
        }

        // Attention against target KV cache using original (pre-spec) page lens
        ops::dispatch_attention_flashinfer_decode(
            *mtp_ws.decode_plans[li],
            mtp_ws.q.data(), kv_cache.layer_view(layer.target_kv_layer),
            mtp_ws.attn_out.data(),
            kv_page_indices, kv_page_indptr,
            mtp_ws.kv_last_page_lens.data(),
            mtp_ws.attn_workspaces[li], stream,
            layer.window_left, 0.f, 1.0f);

        // O-proj + fused post-attn norm + residual + pre-MLP norm
        ops::gemm_act_x_wt_bf16(
            cublas.handle(), mtp_ws.attn_out.data(),
            layer.o_proj->data(), mtp_ws.norm_x.data(),
            S, H, Hq, 0.f);
        kernels::launch_rmsnorm_residual_add_scale_rmsnorm_bf16(
            mtp_ws.norm_x.data(), layer.post_attn_norm->data(),
            mtp_ws.y.data(), 1.f, layer.pre_mlp_norm->data(),
            mtp_ws.norm_x.data(), S, H, eps, stream);

        // MLP
        if (layer.gate_up_proj_fused != nullptr) {
            ops::gemm_act_x_wt_bf16_cublas(
                cublas.handle(), mtp_ws.norm_x.data(),
                layer.gate_up_proj_fused->data(),
                mtp_ws.gate_up_fused.data(), S, 2 * I, H);
            kernels::launch_chunked_geglu_tanh_bf16(
                mtp_ws.gate_up_fused.data(), mtp_ws.gate.data(),
                S, I, stream);
        } else {
            ops::gemm_act_x_wt_bf16(
                cublas.handle(), mtp_ws.norm_x.data(),
                layer.gate_proj->data(), mtp_ws.gate.data(), S, I, H);
            ops::gemm_act_x_wt_bf16(
                cublas.handle(), mtp_ws.norm_x.data(),
                layer.up_proj->data(), mtp_ws.up.data(), S, I, H);
            kernels::launch_geglu_tanh_bf16(
                mtp_ws.gate.data(), mtp_ws.up.data(), mtp_ws.gate.data(),
                S * I, stream);
        }
        ops::gemm_act_x_wt_bf16(
            cublas.handle(), mtp_ws.gate.data(),
            layer.down_proj->data(), mtp_ws.norm_x.data(),
            S, H, I, 0.f);
        kernels::launch_rmsnorm_residual_add_bf16(
            mtp_ws.norm_x.data(), layer.post_mlp_norm->data(),
            mtp_ws.y.data(), S, H, eps, stream);
        if (!mtp_disable_layer_scalar() &&
            std::abs(layer.layer_scalar_value - 1.f) > 1e-6f) {
            kernels::launch_scalar_mul_bf16(
                mtp_ws.y.data(), layer.layer_scalar_value,
                static_cast<std::size_t>(S) * H, stream);
        }
    }

    // 5. Final norm + lm_head → write to TARGET workspace's logits
    kernels::launch_rmsnorm_bf16(
        mtp_ws.y.data(), w.final_norm->data(), mtp_ws.norm_x.data(),
        S, H, eps, stream);
    ops::gemm_act_x_wt_bf16(
        cublas.handle(), mtp_ws.norm_x.data(),
        w.lm_head->data(), target_ws.logits.data(),
        S, V, H);

    // 6. Post-projection for feedback to next draft step
    if (w.post_projection != nullptr) {
        ops::gemm_act_x_wt_bf16(
            cublas.handle(), mtp_ws.norm_x.data(),
            w.post_projection->data(), mtp_ws.hidden.data(),
            S, Hb, H, 0.f);
    }
}

Gemma4MtpDiscovery discover_and_load_gemma4_mtp(
    const Config& cfg,
    const LoadedModel& engine,
    const Gemma4Weights& target_weights,
    bool is_gemma4_arch,
    int native_mtp_num_drafts,
    bool verbose)
{
    Gemma4MtpDiscovery out;
    out.snapshot_dir = cfg.model.mtp_assistant_snapshot_dir;
    if (!out.snapshot_dir.empty()) {
        out.source = "config";
    } else if (const char* env = std::getenv("PIE_GEMMA4_MTP_SNAPSHOT_DIR")) {
        out.snapshot_dir = env;
        out.source = "env";
    }
    if (is_gemma4_arch && native_mtp_num_drafts > 0 && out.snapshot_dir.empty()) {
        if (auto discovered = discover_gemma4_mtp_snapshot_dir(
                std::filesystem::path(cfg.model.snapshot_dir))) {
            out.snapshot_dir = discovered->string();
            out.source = "auto";
            if (verbose && cfg.distributed.tp_rank == 0) {
                std::cerr << "[pie-driver-cuda] Gemma4 MTP assistant "
                          << "auto-discovered: " << out.snapshot_dir
                          << "\n";
            }
        }
    }
    if (is_gemma4_arch && native_mtp_num_drafts > 0 && !out.snapshot_dir.empty()) {
        if (cfg.distributed.tp_size > 1) {
            if (verbose && cfg.distributed.tp_rank == 0) {
                std::cerr << "[pie-driver-cuda] Gemma4 MTP disabled under "
                          << "tensor parallelism for this build\n";
            }
        } else {
            out.weights.emplace(
                load_gemma4_mtp_weights(
                    std::filesystem::path(out.snapshot_dir),
                    cfg.model.device,
                    engine.hf_config(),
                    target_weights,
                    out.runtime,
                    verbose));
            if (verbose && cfg.distributed.tp_rank == 0 && !out.source.empty()) {
                std::cerr << "[pie-driver-cuda] Gemma4 MTP assistant source="
                          << out.source << "\n";
            }
        }
    } else if (is_gemma4_arch && native_mtp_num_drafts > 0 &&
               verbose && cfg.distributed.tp_rank == 0) {
        std::cerr << "[pie-driver-cuda] Gemma4 MTP system drafter not "
                  << "enabled: assistant checkpoint not found; set "
                  << "mtp_assistant_snapshot_dir or "
                  << "PIE_GEMMA4_MTP_SNAPSHOT_DIR\n";
    }
    return out;
}

}  // namespace pie_cuda_driver::model
