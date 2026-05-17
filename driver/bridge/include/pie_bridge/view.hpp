// pie_bridge/view.hpp — driver-side SoA view + AoS→SoA demux.
//
// Compatibility adapter for the C++ backends (cuda + portable). The
// wire schema in pie_bridge.h gives you `Pie*Desc` POD structs that
// mirror the rkyv archive field-for-field. The C++ handler code
// (`forward.cpp`, `plan.cpp`, sampler dispatch) wants flat SoA arrays
// for every sampler attribute, so on each `recv` we walk the new
// `samplers` Vec (AoS of tagged unions) and demultiplex into per-attr
// arenas. The arenas live in [`RequestArenas`] and are reset+refilled
// per call.
//
// Header-only: shared between cuda and portable. Both
// `driver/cuda/CMakeLists.txt` and `driver/portable/CMakeLists.txt`
// add `${PIE_BRIDGE_INCLUDE_DIR}` to their include path — the
// declarations and `inline` definitions below resolve automatically.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <pie_bridge.h>  // PieFrameDesc, PieResponseFrameDesc, PIE_*

namespace pie_driver {

// ---- Re-export new PieInProcVTable under pie_driver:: ---------------------

using ::PieInProcVTable;
using ::PieRecvResult;

// ---- Slice -----------------------------------------------------------------

template <typename T>
struct PieSlice {
    const T* ptr = nullptr;
    std::size_t len = 0;

    constexpr std::size_t size() const noexcept { return len; }
    constexpr bool empty() const noexcept { return len == 0; }
    constexpr const T* data() const noexcept { return ptr; }
    template <typename U>
    std::span<const U> as() const noexcept {
        return std::span<const U>(reinterpret_cast<const U*>(ptr), len);
    }
};

// Type-erased slice (no template param). Used by fields whose element
// type is opaque (path bytes / raw blobs); cast via `.as<T>()`.
struct PieByteSlice {
    const void* ptr = nullptr;
    std::size_t len = 0;

    constexpr std::size_t size() const noexcept { return len; }
    constexpr bool empty() const noexcept { return len == 0; }
    template <typename U>
    std::span<const U> as() const noexcept {
        return std::span<const U>(reinterpret_cast<const U*>(ptr), len);
    }
};

template <typename T>
constexpr PieSlice<T> slice_from(const T* ptr, std::size_t len) noexcept {
    return PieSlice<T>{ptr, len};
}
inline PieSlice<std::uint32_t> slice_from_u32(const std::uint32_t* p, std::size_t n) noexcept {
    return PieSlice<std::uint32_t>{p, n};
}
inline PieSlice<std::uint64_t> slice_from_u64(const std::uint64_t* p, std::size_t n) noexcept {
    return PieSlice<std::uint64_t>{p, n};
}
inline PieSlice<std::uint8_t> slice_from_u8(const std::uint8_t* p, std::size_t n) noexcept {
    return PieSlice<std::uint8_t>{p, n};
}
inline PieSlice<float> slice_from_f32(const float* p, std::size_t n) noexcept {
    return PieSlice<float>{p, n};
}
inline PieSlice<std::int64_t> slice_from_i64(const std::int64_t* p, std::size_t n) noexcept {
    return PieSlice<std::int64_t>{p, n};
}

// ---- Driver-internal method tag constants ---------------------------------
//
// In the wire format these are split into (payload_kind, sub_op). The
// polling loop in [`InProcServer`] synthesizes a single tag here before
// invoking the handler so the per-backend dispatch stays a flat switch.

constexpr std::uint32_t PIE_METHOD_FORWARD              = 0;
constexpr std::uint32_t PIE_METHOD_COPY_D2H             = 1;
constexpr std::uint32_t PIE_METHOD_COPY_H2D             = 2;
constexpr std::uint32_t PIE_METHOD_COPY_D2D             = 3;
constexpr std::uint32_t PIE_METHOD_COPY_H2H             = 4;
constexpr std::uint32_t PIE_METHOD_LOAD_ADAPTER         = 5;
constexpr std::uint32_t PIE_METHOD_SAVE_ADAPTER         = 6;
constexpr std::uint32_t PIE_METHOD_ZO_INITIALIZE_ADAPTER = 7;
constexpr std::uint32_t PIE_METHOD_ZO_UPDATE_ADAPTER    = 8;
constexpr std::uint32_t PIE_METHOD_HEALTH               = 9;
constexpr std::uint32_t PIE_METHOD_RS_COPY_D2H          = 10;
constexpr std::uint32_t PIE_METHOD_RS_COPY_H2D          = 11;
constexpr std::uint32_t PIE_METHOD_RS_COPY_D2D          = 12;
constexpr std::uint32_t PIE_METHOD_RS_COPY_H2H          = 13;

// Driver-side sampler IDs (DISTRIBUTION=0, MULTINOMIAL=1, …, ENTROPY=10).
// The wire schema orders variants differently (`Sampler::Multinomial`
// has kind=0); [`fill_forward_view`] remaps via `kNewToOldSamplerKind`
// when filling the SoA arrays. This is a backward-compat tail for the
// existing kernel dispatch code; deletion candidate if the dispatch
// code is ever rewritten to consume schema variant order directly.
constexpr std::uint32_t SAMPLER_DISTRIBUTION = 0;
constexpr std::uint32_t SAMPLER_MULTINOMIAL  = 1;
constexpr std::uint32_t SAMPLER_TOP_K        = 2;
constexpr std::uint32_t SAMPLER_TOP_P        = 3;
constexpr std::uint32_t SAMPLER_MIN_P        = 4;
constexpr std::uint32_t SAMPLER_TOP_K_TOP_P  = 5;
constexpr std::uint32_t SAMPLER_EMBEDDING    = 6;
constexpr std::uint32_t SAMPLER_RAW_LOGITS   = 7;
constexpr std::uint32_t SAMPLER_LOGPROB      = 8;
constexpr std::uint32_t SAMPLER_LOGPROBS     = 9;
constexpr std::uint32_t SAMPLER_ENTROPY      = 10;

// ---- ForwardRequest view (driver-side SoA layout) -------------------------

struct PieForwardRequestView {
    // Token data
    PieSlice<std::uint32_t> token_ids;
    PieSlice<std::uint32_t> position_ids;

    // KV cache
    PieSlice<std::uint32_t> kv_page_indices;
    PieSlice<std::uint32_t> kv_page_indptr;
    PieSlice<std::uint32_t> kv_last_page_lens;
    PieSlice<std::uint32_t> qo_indptr;
    PieSlice<std::uint32_t> rs_slot_ids;
    PieSlice<std::uint8_t>  rs_slot_flags;

    // Attention / logit masks
    PieSlice<std::uint32_t> flattened_masks;
    PieSlice<std::uint32_t> mask_indptr;
    PieSlice<std::uint32_t> logit_masks;
    PieSlice<std::uint32_t> logit_mask_indptr;

    // Sampling (CSR)
    PieSlice<std::uint32_t> sampling_indices;
    PieSlice<std::uint32_t> sampling_indptr;

    // Sampler attributes (SoA — demultiplexed from new AoS samplers Vec).
    PieSlice<std::uint32_t> sampler_types;
    PieSlice<float>         sampler_temperatures;
    PieSlice<std::uint32_t> sampler_top_k;
    PieSlice<float>         sampler_top_p;
    PieSlice<float>         sampler_min_p;
    PieSlice<std::uint32_t> sampler_seeds;
    PieSlice<std::uint32_t> sampler_label_ids;     // Logprobs.token_ids concatenated
    PieSlice<std::uint32_t> sampler_label_indptr;  // CSR into sampler_label_ids
    PieSlice<std::uint32_t> request_num_samplers;  // per-request count

    // Adapter bindings (Vec<AdapterBinding> → flat lists). Portable
    // reinterprets these as int64 (`adapter_indices.as<int64_t>()`).
    PieSlice<std::int64_t> adapter_indices;       // -1 sentinel for None
    PieSlice<std::int64_t> adapter_seeds;         // -1 sentinel for None

    // Speculative decoding
    PieSlice<std::uint32_t> spec_token_ids;
    PieSlice<std::uint32_t> spec_position_ids;
    PieSlice<std::uint32_t> spec_indptr;
    PieSlice<std::uint8_t>  output_spec_flags;

    PieSlice<std::uint64_t> context_ids;

    std::uint32_t driver_id;
    std::uint8_t  single_token_mode;
    std::uint8_t  has_user_mask;

    constexpr std::size_t size() const noexcept { return token_ids.size(); }
    constexpr bool empty() const noexcept { return token_ids.empty(); }
};

// ---- ForwardResponse view (driver-side SoA layout, write side) -----------

struct PieForwardResponseView {
    std::uint32_t num_requests;

    PieSlice<std::uint32_t> tokens_indptr;
    PieSlice<std::uint32_t> tokens;

    PieSlice<std::uint32_t> dists_req_indptr;
    PieSlice<std::uint32_t> dists_kv_indptr;
    PieSlice<std::uint32_t> dists_ids;
    PieSlice<float>         dists_probs;

    PieSlice<std::uint32_t> logits_req_indptr;
    PieSlice<std::uint32_t> logits_byte_indptr;
    PieSlice<std::uint8_t>  logits_bytes;

    PieSlice<std::uint32_t> logprobs_req_indptr;
    PieSlice<std::uint32_t> logprobs_val_indptr;
    PieSlice<float>         logprobs_values;

    PieSlice<std::uint32_t> entropies_indptr;
    PieSlice<float>         entropies;
};

// ---- Top-level request / response views -----------------------------------

struct PieInProcRequestView {
    std::uint32_t method;     // PIE_METHOD_*
    std::uint32_t driver_id;

    // Forward variant — populated when method == PIE_METHOD_FORWARD.
    PieForwardRequestView forward;

    // Copy variant — populated when method ∈ {COPY_D2H, COPY_H2D,
    // COPY_D2D, COPY_H2H}.
    PieSlice<std::uint32_t> copy_srcs;
    PieSlice<std::uint32_t> copy_dsts;

    // Adapter variant — populated when method ∈ {LOAD/SAVE/ZO_INIT/
    // ZO_UPDATE}. Path is bytes (UTF-8); empty when absent.
    std::uint64_t   adapter_id;
    PieByteSlice    adapter_path;
};

struct PieStatusResponseView {
    std::int32_t status;
};

struct PieInProcResponseView {
    std::uint32_t            method;
    std::int32_t             status;
    PieForwardResponseView   forward;
};

// ---- Per-batch arenas ------------------------------------------------------

/// Scratch vectors that back the slice pointers in `PieForwardRequestView`
/// for fields the wire schema stores in AoS form (samplers, adapter
/// bindings, BRLE masks). Reset + refill per `recv` call.
struct RequestArenas {
    // Sampler SoA demuxed from PieSamplerDesc[].
    std::vector<std::uint32_t> sampler_types;
    std::vector<float>         sampler_temperatures;
    std::vector<std::uint32_t> sampler_top_k;
    std::vector<float>         sampler_top_p;
    std::vector<float>         sampler_min_p;
    std::vector<std::uint32_t> sampler_seeds;
    std::vector<std::uint32_t> sampler_label_ids;
    std::vector<std::uint32_t> sampler_label_indptr;
    std::vector<std::uint32_t> request_num_samplers;

    // Adapter binding flat lists (-1 sentinel for None). Both i64
    // because portable reinterprets via `.as<int64_t>()`.
    std::vector<std::int64_t> adapter_indices;
    std::vector<std::int64_t> adapter_seeds;

    // BRLE masks: the bridge ships Vec<Brle>; the driver-side code path
    // expects a flat run-length buffer with per-ROW byte offsets. We
    // concatenate each Brle's buffer into `flattened_masks` and build
    // `mask_byte_indptr[r..r+1]` per query row. Same shape for the
    // (per-request) logit mask buffers.
    std::vector<std::uint32_t> flattened_masks;
    std::vector<std::uint32_t> mask_byte_indptr;
    std::vector<std::uint32_t> logit_masks_flat;
    std::vector<std::uint32_t> logit_mask_byte_indptr;

    // Per-call response-side scratches for ResponseBuilder fast-path.
    std::vector<std::uint8_t> output_spec_flags;

    void clear() {
        sampler_types.clear();
        sampler_temperatures.clear();
        sampler_top_k.clear();
        sampler_top_p.clear();
        sampler_min_p.clear();
        sampler_seeds.clear();
        sampler_label_ids.clear();
        sampler_label_indptr.clear();
        request_num_samplers.clear();
        adapter_indices.clear();
        adapter_seeds.clear();
        flattened_masks.clear();
        mask_byte_indptr.clear();
        logit_masks_flat.clear();
        logit_mask_byte_indptr.clear();
        output_spec_flags.clear();
    }
};

// ---- Conversion API --------------------------------------------------------

namespace detail {

// Wire schema's Sampler variant order (Multinomial=0, …, Entropy=10) →
// driver-side sampler-ID space (DISTRIBUTION=0, MULTINOMIAL=1, …,
// ENTROPY=10). See SAMPLER_* constants above for the rationale.
constexpr std::uint32_t kNewToOldSamplerKind[11] = {
    SAMPLER_MULTINOMIAL,    // new 0 → old 1
    SAMPLER_TOP_K,          // 1 → 2
    SAMPLER_TOP_P,          // 2 → 3
    SAMPLER_MIN_P,          // 3 → 4
    SAMPLER_TOP_K_TOP_P,    // 4 → 5
    SAMPLER_EMBEDDING,      // 5 → 6
    SAMPLER_DISTRIBUTION,   // 6 → 0
    SAMPLER_RAW_LOGITS,     // 7 → 7
    SAMPLER_LOGPROB,        // 8 → 8
    SAMPLER_LOGPROBS,       // 9 → 9
    SAMPLER_ENTROPY,        // 10 → 10
};

inline void fill_forward_view(const PieForwardRequestDesc& f,
                              std::uint32_t driver_id,
                              RequestArenas& arenas,
                              PieForwardRequestView& out) {
    out.driver_id = driver_id;
    out.single_token_mode = f.single_token_mode;
    out.has_user_mask = f.has_user_mask;

    // Pass-through slices (same shape, just different ptr/len naming).
    out.token_ids         = slice_from(f.token_ids_ptr, f.token_ids_len);
    out.position_ids      = slice_from(f.position_ids_ptr, f.position_ids_len);
    out.kv_page_indices   = slice_from(f.kv_page_indices_ptr, f.kv_page_indices_len);
    out.kv_page_indptr    = slice_from(f.kv_page_indptr_ptr, f.kv_page_indptr_len);
    out.kv_last_page_lens = slice_from(f.kv_last_page_lens_ptr, f.kv_last_page_lens_len);
    out.qo_indptr         = slice_from(f.qo_indptr_ptr, f.qo_indptr_len);
    out.rs_slot_ids       = slice_from(f.rs_slot_ids_ptr, f.rs_slot_ids_len);
    out.rs_slot_flags     = slice_from(f.rs_slot_flags_ptr, f.rs_slot_flags_len);

    // BRLE masks come over the wire as Vec<Brle>. The driver code path
    // wants a flat run-length buffer plus per-ROW byte offsets. Walk
    // the descriptor array and rebuild both arenas. Note that the
    // schema's `mask_indptr` / `logit_mask_indptr` are per-REQUEST
    // partitions of Vec<Brle> — we don't use them here; the per-row
    // layout is reconstructed directly from the Brle vector.
    arenas.mask_byte_indptr.clear();
    arenas.mask_byte_indptr.reserve(f.masks_len + 1);
    arenas.mask_byte_indptr.push_back(0);
    for (std::size_t i = 0; i < f.masks_len; ++i) {
        const PieBrleDesc& b = f.masks_ptr[i];
        arenas.flattened_masks.insert(arenas.flattened_masks.end(),
                                      b.buffer_ptr, b.buffer_ptr + b.buffer_len);
        arenas.mask_byte_indptr.push_back(
            static_cast<std::uint32_t>(arenas.flattened_masks.size()));
    }
    out.flattened_masks = slice_from(arenas.flattened_masks.data(), arenas.flattened_masks.size());
    out.mask_indptr     = slice_from(arenas.mask_byte_indptr.data(), arenas.mask_byte_indptr.size());

    arenas.logit_mask_byte_indptr.clear();
    arenas.logit_mask_byte_indptr.reserve(f.logit_mask_indptr_len);
    arenas.logit_mask_byte_indptr.push_back(0);
    if (f.logit_mask_indptr_len >= 1) {
        for (std::size_t r = 1; r < f.logit_mask_indptr_len; ++r) {
            const std::uint32_t lo = f.logit_mask_indptr_ptr[r - 1];
            const std::uint32_t hi = f.logit_mask_indptr_ptr[r];
            for (std::uint32_t i = lo; i < hi; ++i) {
                const PieBrleDesc& b = f.logit_masks_ptr[i];
                arenas.logit_masks_flat.insert(arenas.logit_masks_flat.end(),
                                               b.buffer_ptr, b.buffer_ptr + b.buffer_len);
            }
            arenas.logit_mask_byte_indptr.push_back(
                static_cast<std::uint32_t>(arenas.logit_masks_flat.size()));
        }
    }
    out.logit_masks       = slice_from(arenas.logit_masks_flat.data(), arenas.logit_masks_flat.size());
    out.logit_mask_indptr = slice_from(arenas.logit_mask_byte_indptr.data(), arenas.logit_mask_byte_indptr.size());
    out.sampling_indices  = slice_from(f.sampling_indices_ptr, f.sampling_indices_len);
    out.sampling_indptr   = slice_from(f.sampling_indptr_ptr, f.sampling_indptr_len);
    out.spec_token_ids    = slice_from(f.spec_token_ids_ptr, f.spec_token_ids_len);
    out.spec_position_ids = slice_from(f.spec_position_ids_ptr, f.spec_position_ids_len);
    out.spec_indptr       = slice_from(f.spec_indptr_ptr, f.spec_indptr_len);
    out.output_spec_flags = slice_from(f.output_spec_flags_ptr, f.output_spec_flags_len);
    out.context_ids       = slice_from(f.context_ids_ptr, f.context_ids_len);

    // Demultiplex AoS samplers into SoA arenas. The driver handler reads
    // one entry per sampler from each per-attribute array. For sampler
    // variants that don't carry a given attribute (e.g. TopK has no p),
    // we fill 0 / sentinel.
    const std::size_t n = f.samplers_len;
    arenas.sampler_types.resize(n);
    arenas.sampler_temperatures.resize(n);
    arenas.sampler_top_k.resize(n);
    arenas.sampler_top_p.resize(n);
    arenas.sampler_min_p.resize(n);
    arenas.sampler_seeds.resize(n);
    arenas.sampler_label_ids.clear();
    arenas.sampler_label_indptr.clear();
    arenas.sampler_label_indptr.reserve(n + 1);
    arenas.sampler_label_indptr.push_back(0);

    for (std::size_t i = 0; i < n; ++i) {
        const PieSamplerDesc& s = f.samplers_ptr[i];
        const std::uint8_t k = s.kind;
        arenas.sampler_types[i] = (k < 11) ? kNewToOldSamplerKind[k] : 0;
        arenas.sampler_temperatures[i] = s.temperature;
        arenas.sampler_top_k[i] = 0;
        arenas.sampler_top_p[i] = 1.0f;
        arenas.sampler_min_p[i] = 0.0f;
        switch (k) {
            case 1: // TopK
                arenas.sampler_top_k[i] = s.k;
                break;
            case 2: // TopP
                arenas.sampler_top_p[i] = s.p;
                break;
            case 3: // MinP
                arenas.sampler_min_p[i] = s.p;
                break;
            case 4: // TopKTopP
                arenas.sampler_top_k[i] = s.k;
                arenas.sampler_top_p[i] = s.p;
                break;
            case 6: // Dist
                arenas.sampler_top_k[i] = s.num_tokens;
                break;
            default:
                break;
        }
        // Sentinel: s.seed == 0 → no caller-provided seed (driver-side
        // dispatch treats 0 the same way the old `seed_has==false`
        // branch did).
        arenas.sampler_seeds[i] = s.seed;
        // Logprob.token_id / Logprobs.token_ids → concatenate, record offset.
        if (k == 8 /*Logprob*/) {
            arenas.sampler_label_ids.push_back(s.token_id);
        } else if (k == 9 /*Logprobs*/) {
            const std::uint32_t* tids = s.token_ids_ptr;
            const std::size_t tlen = s.token_ids_len;
            arenas.sampler_label_ids.insert(arenas.sampler_label_ids.end(), tids, tids + tlen);
        }
        arenas.sampler_label_indptr.push_back(
            static_cast<std::uint32_t>(arenas.sampler_label_ids.size()));
    }

    // Per-request sampler counts: difference of sampler_indptr.
    const std::size_t r = f.sampler_indptr_len > 0 ? f.sampler_indptr_len - 1 : 0;
    arenas.request_num_samplers.resize(r);
    for (std::size_t i = 0; i < r; ++i) {
        arenas.request_num_samplers[i] =
            f.sampler_indptr_ptr[i + 1] - f.sampler_indptr_ptr[i];
    }

    // Adapter bindings → two parallel flat lists. The wire format
    // already uses i64 with -1 sentinels for both fields, matching what
    // portable consumes via `.as<int64_t>()` — this is a pure
    // demultiplex, no value conversion.
    const std::size_t b = f.adapter_bindings_len;
    arenas.adapter_indices.resize(b);
    arenas.adapter_seeds.resize(b);
    for (std::size_t i = 0; i < b; ++i) {
        const PieAdapterBindingDesc& bind = f.adapter_bindings_ptr[i];
        arenas.adapter_indices[i] = bind.adapter_id;
        arenas.adapter_seeds[i] = bind.seed;
    }

    // Wire the arena vectors to the view's slices.
    out.sampler_types         = slice_from(arenas.sampler_types.data(), arenas.sampler_types.size());
    out.sampler_temperatures  = slice_from(arenas.sampler_temperatures.data(), arenas.sampler_temperatures.size());
    out.sampler_top_k         = slice_from(arenas.sampler_top_k.data(), arenas.sampler_top_k.size());
    out.sampler_top_p         = slice_from(arenas.sampler_top_p.data(), arenas.sampler_top_p.size());
    out.sampler_min_p         = slice_from(arenas.sampler_min_p.data(), arenas.sampler_min_p.size());
    out.sampler_seeds         = slice_from(arenas.sampler_seeds.data(), arenas.sampler_seeds.size());
    out.sampler_label_ids     = slice_from(arenas.sampler_label_ids.data(), arenas.sampler_label_ids.size());
    out.sampler_label_indptr  = slice_from(arenas.sampler_label_indptr.data(), arenas.sampler_label_indptr.size());
    out.request_num_samplers  = slice_from(arenas.request_num_samplers.data(), arenas.request_num_samplers.size());
    out.adapter_indices       = slice_from(arenas.adapter_indices.data(), arenas.adapter_indices.size());
    out.adapter_seeds         = slice_from(arenas.adapter_seeds.data(), arenas.adapter_seeds.size());
}

}  // namespace detail

/// Populate `out` from `frame`, using `arenas` as scratch for fields
/// that need AoS→SoA demultiplexing. After this returns, `out`'s slice
/// pointers are valid until `frame` is invalidated (commit_response on
/// the matching req_id) AND `arenas` is reused for the next request.
///
/// `out.method` is set to a synthetic `PIE_METHOD_*` tag.
inline void build_request_view(const PieFrameDesc& frame,
                               RequestArenas& arenas,
                               PieInProcRequestView& out) {
    arenas.clear();
    out = PieInProcRequestView{};
    out.driver_id = frame.driver_id;

    const std::uint8_t kind = frame.payload.kind;
    switch (kind) {
        case PIE_REQUEST_PAYLOAD_FORWARD:
            out.method = PIE_METHOD_FORWARD;
            detail::fill_forward_view(frame.payload.forward, frame.driver_id, arenas, out.forward);
            break;
        case PIE_REQUEST_PAYLOAD_COPY: {
            const PieCopyRequestDesc& c = frame.payload.copy;
            // CopyDir variant order: D2H=0, H2D=1, D2D=2, H2H=3 →
            // driver methods 1, 2, 3, 4.
            const bool rs = c.resource == PIE_COPY_RESOURCE_RS;
            out.method = (rs ? PIE_METHOD_RS_COPY_D2H : PIE_METHOD_COPY_D2H) + c.dir;
            out.copy_srcs = slice_from(c.srcs_ptr, c.srcs_len);
            out.copy_dsts = slice_from(c.dsts_ptr, c.dsts_len);
            break;
        }
        case PIE_REQUEST_PAYLOAD_ADAPTER: {
            const PieAdapterRequestDesc& a = frame.payload.adapter;
            // AdapterOp: Load=0 / Save=1 / ZoInit=2 / ZoUpdate=3 →
            // driver methods 5, 6, 7, 8.
            out.method = PIE_METHOD_LOAD_ADAPTER + a.op;
            out.adapter_id = a.adapter_id;
            out.adapter_path.ptr = a.path_ptr;
            out.adapter_path.len = a.path_len;
            break;
        }
        case PIE_REQUEST_PAYLOAD_HEALTH:
            out.method = PIE_METHOD_HEALTH;
            break;
        default:
            // Leave method at 0 / forward as default-initialized; the
            // worker treats unknown methods as bad_method.
            out.method = static_cast<std::uint32_t>(-1);
            break;
    }
}

/// Inverse: pack a filled response view into a new
/// `PieResponseFrameDesc`. The descriptor's slice pointers alias `view`'s
/// scratch (no copy); caller must keep `view` alive across the
/// `send_response` call.
inline void build_response_desc(std::uint32_t driver_id,
                                const PieInProcResponseView& view,
                                PieResponseFrameDesc& out) {
    out = PieResponseFrameDesc{};
    out.driver_id = driver_id;
    out.aborted = 0;

    if (view.method == PIE_METHOD_FORWARD) {
        out.payload.kind = PIE_RESPONSE_PAYLOAD_FORWARD;
        PieForwardResponseDesc& fr = out.payload.forward;
        fr.num_requests = view.forward.num_requests;

        fr.tokens_indptr_ptr = view.forward.tokens_indptr.data();
        fr.tokens_indptr_len = view.forward.tokens_indptr.size();
        fr.tokens_ptr        = view.forward.tokens.data();
        fr.tokens_len        = view.forward.tokens.size();

        fr.dists_req_indptr_ptr = view.forward.dists_req_indptr.data();
        fr.dists_req_indptr_len = view.forward.dists_req_indptr.size();
        fr.dists_kv_indptr_ptr  = view.forward.dists_kv_indptr.data();
        fr.dists_kv_indptr_len  = view.forward.dists_kv_indptr.size();
        fr.dists_ids_ptr        = view.forward.dists_ids.data();
        fr.dists_ids_len        = view.forward.dists_ids.size();
        fr.dists_probs_ptr      = view.forward.dists_probs.data();
        fr.dists_probs_len      = view.forward.dists_probs.size();

        fr.logits_req_indptr_ptr   = view.forward.logits_req_indptr.data();
        fr.logits_req_indptr_len   = view.forward.logits_req_indptr.size();
        fr.logits_byte_indptr_ptr  = view.forward.logits_byte_indptr.data();
        fr.logits_byte_indptr_len  = view.forward.logits_byte_indptr.size();
        fr.logits_bytes_ptr        = view.forward.logits_bytes.data();
        fr.logits_bytes_len        = view.forward.logits_bytes.size();

        fr.logprobs_req_indptr_ptr  = view.forward.logprobs_req_indptr.data();
        fr.logprobs_req_indptr_len  = view.forward.logprobs_req_indptr.size();
        fr.logprobs_val_indptr_ptr  = view.forward.logprobs_val_indptr.data();
        fr.logprobs_val_indptr_len  = view.forward.logprobs_val_indptr.size();
        fr.logprobs_values_ptr      = view.forward.logprobs_values.data();
        fr.logprobs_values_len      = view.forward.logprobs_values.size();

        fr.entropies_indptr_ptr = view.forward.entropies_indptr.data();
        fr.entropies_indptr_len = view.forward.entropies_indptr.size();
        fr.entropies_ptr        = view.forward.entropies.data();
        fr.entropies_len        = view.forward.entropies.size();
    } else {
        // Everything else (copy / adapter / health / unknown) just
        // produces a StatusResponse with the int code.
        out.payload.kind = PIE_RESPONSE_PAYLOAD_STATUS;
        out.payload.status.status = view.status;
    }
}

}  // namespace pie_driver
