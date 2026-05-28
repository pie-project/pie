/* pie_bridge.h — C ABI for non-Rust drivers (cuda/portable C++, Python
 * via ctypes). Every symbol below is emitted by `#[schema]` on the
 * corresponding Rust type in `driver/bridge/src/schema.rs`. Symbol
 * names derive mechanically from the type name in snake_case:
 *
 *   * Reader:     pie_<type>_<field>(...)
 *   * Parse:      pie_parse_<type>(bytes, len) -> const Pie<T>*
 *   * Descriptor: Pie<T>Desc (C-friendly mirror)
 *   * Builder:    pie_build_<type>(const Pie<T>Desc*, out, cap) -> size_t
 *   * Enum kind:  pie_<enum>_kind(p) -> uint8_t (data enums)
 *   * Enum value: pie_<enum>_value(p) -> uint8_t (unit enums)
 *   * Enum cast:  pie_<enum>_as_<variant>(p) -> const PieV*
 *
 * # Usage (C++):
 *
 *   const PieFrame* f = pie_parse_frame(bytes, len);
 *   if (!f) { handle error }
 *   const PieRequestPayload* p = pie_frame_payload(f);
 *   switch (pie_request_payload_kind(p)) {
 *     case PIE_REQUEST_PAYLOAD_FORWARD: {
 *       const PieForwardRequest* fr = pie_request_payload_as_forward(p);
 *       const uint32_t* tokens; size_t n;
 *       pie_forward_request_token_ids(fr, &tokens, &n);
 *       ...
 *     }
 *     ...
 *   }
 *
 * # Lifetime: pointers returned by pie_parse_* and pie_*_at remain
 * valid for as long as the input byte buffer remains live. Do not free
 * any of them.
 */

#ifndef PIE_BRIDGE_H
#define PIE_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ----- Opaque archived types ----- */

typedef struct PieFrame              PieFrame;
typedef struct PieResponseFrame      PieResponseFrame;
typedef struct PieRequestPayload     PieRequestPayload;
typedef struct PieResponsePayload    PieResponsePayload;
typedef struct PieForwardRequest     PieForwardRequest;
typedef struct PieForwardResponse    PieForwardResponse;
typedef struct PieCopyRequest        PieCopyRequest;
typedef struct PieAdapterRequest     PieAdapterRequest;
typedef struct PieStatusResponse     PieStatusResponse;
typedef struct PieAdapterBinding     PieAdapterBinding;
typedef struct PieBrle               PieBrle;
typedef struct PieSampler            PieSampler;
typedef struct PieCopyDir            PieCopyDir;
typedef struct PieCopyResource       PieCopyResource;
typedef struct PieAdapterOp          PieAdapterOp;

/* ----- Discriminant constants ----- */

#define PIE_REQUEST_PAYLOAD_FORWARD  0
#define PIE_REQUEST_PAYLOAD_COPY     1
#define PIE_REQUEST_PAYLOAD_ADAPTER  2
#define PIE_REQUEST_PAYLOAD_HEALTH   3

#define PIE_RESPONSE_PAYLOAD_FORWARD 0
#define PIE_RESPONSE_PAYLOAD_STATUS  1

#define PIE_SAMPLER_MULTINOMIAL    0
#define PIE_SAMPLER_TOP_K          1
#define PIE_SAMPLER_TOP_P          2
#define PIE_SAMPLER_MIN_P          3
#define PIE_SAMPLER_TOP_K_TOP_P    4
#define PIE_SAMPLER_EMBEDDING      5
#define PIE_SAMPLER_DIST           6
#define PIE_SAMPLER_RAW_LOGITS     7
#define PIE_SAMPLER_LOGPROB        8
#define PIE_SAMPLER_LOGPROBS       9
#define PIE_SAMPLER_ENTROPY        10

#define PIE_COPY_DIR_D2H  0
#define PIE_COPY_DIR_H2D  1
#define PIE_COPY_DIR_D2D  2
#define PIE_COPY_DIR_H2H  3

#define PIE_COPY_RESOURCE_KV  0
#define PIE_COPY_RESOURCE_RS  1

#define PIE_ADAPTER_OP_LOAD        0
#define PIE_ADAPTER_OP_SAVE        1
#define PIE_ADAPTER_OP_ZO_INIT     2
#define PIE_ADAPTER_OP_ZO_UPDATE   3

/* ===========================================================
 * Parse entry points
 * =========================================================== */

const PieFrame*         pie_parse_frame         (const uint8_t* bytes, size_t len);
const PieResponseFrame* pie_parse_response_frame(const uint8_t* bytes, size_t len);

/* ===========================================================
 * Frame
 * =========================================================== */

uint32_t                 pie_frame_driver_id(const PieFrame*);
const PieRequestPayload* pie_frame_payload  (const PieFrame*);

uint8_t                   pie_request_payload_kind       (const PieRequestPayload*);
const PieForwardRequest*  pie_request_payload_as_forward (const PieRequestPayload*);
const PieCopyRequest*     pie_request_payload_as_copy    (const PieRequestPayload*);
const PieAdapterRequest*  pie_request_payload_as_adapter (const PieRequestPayload*);

/* ===========================================================
 * ResponseFrame
 * =========================================================== */

uint32_t                  pie_response_frame_driver_id(const PieResponseFrame*);
uint8_t                   pie_response_frame_aborted  (const PieResponseFrame*);
const PieResponsePayload* pie_response_frame_payload  (const PieResponseFrame*);

uint8_t                   pie_response_payload_kind       (const PieResponsePayload*);
const PieForwardResponse* pie_response_payload_as_forward (const PieResponsePayload*);
const PieStatusResponse*  pie_response_payload_as_status  (const PieResponsePayload*);

/* ===========================================================
 * ForwardRequest — slice fields share the (out_ptr, out_len) shape
 * =========================================================== */

void pie_forward_request_token_ids        (const PieForwardRequest*, const uint32_t**, size_t*);
void pie_forward_request_position_ids     (const PieForwardRequest*, const uint32_t**, size_t*);
void pie_forward_request_kv_page_indices  (const PieForwardRequest*, const uint32_t**, size_t*);
void pie_forward_request_kv_page_indptr   (const PieForwardRequest*, const uint32_t**, size_t*);
void pie_forward_request_kv_last_page_lens(const PieForwardRequest*, const uint32_t**, size_t*);
void pie_forward_request_qo_indptr        (const PieForwardRequest*, const uint32_t**, size_t*);
void pie_forward_request_rs_slot_ids      (const PieForwardRequest*, const uint32_t**, size_t*);
void pie_forward_request_rs_slot_flags    (const PieForwardRequest*, const uint8_t**,  size_t*);
/* `masks` and `logit_masks` are arrays of nested PieBrle structs;
 * use the `_len`/`_at` pair below instead of a flat slice accessor.
 * `mask_indptr` / `logit_mask_indptr` partition the Vec<Brle> per
 * request (length = num_requests + 1). */
void pie_forward_request_mask_indptr      (const PieForwardRequest*, const uint32_t**, size_t*);
void pie_forward_request_logit_mask_indptr(const PieForwardRequest*, const uint32_t**, size_t*);
size_t        pie_forward_request_masks_len(const PieForwardRequest*);
const PieBrle* pie_forward_request_masks_at (const PieForwardRequest*, size_t i);
size_t        pie_forward_request_logit_masks_len(const PieForwardRequest*);
const PieBrle* pie_forward_request_logit_masks_at (const PieForwardRequest*, size_t i);
void pie_forward_request_sampling_indices (const PieForwardRequest*, const uint32_t**, size_t*);
void pie_forward_request_sampling_indptr  (const PieForwardRequest*, const uint32_t**, size_t*);
void pie_forward_request_sampler_indptr   (const PieForwardRequest*, const uint32_t**, size_t*);
void pie_forward_request_spec_token_ids   (const PieForwardRequest*, const uint32_t**, size_t*);
void pie_forward_request_spec_position_ids(const PieForwardRequest*, const uint32_t**, size_t*);
void pie_forward_request_spec_indptr      (const PieForwardRequest*, const uint32_t**, size_t*);
void pie_forward_request_context_ids      (const PieForwardRequest*, const uint64_t**, size_t*);
void pie_forward_request_output_spec_flags(const PieForwardRequest*, const uint8_t**,  size_t*);

uint8_t pie_forward_request_single_token_mode(const PieForwardRequest*);
uint8_t pie_forward_request_has_user_mask    (const PieForwardRequest*);

size_t            pie_forward_request_samplers_len(const PieForwardRequest*);
const PieSampler* pie_forward_request_samplers_at (const PieForwardRequest*, size_t i);

size_t                   pie_forward_request_adapter_bindings_len(const PieForwardRequest*);
const PieAdapterBinding* pie_forward_request_adapter_bindings_at (const PieForwardRequest*, size_t i);

/* AdapterBinding fields — `-1` means unbound for both. */
int64_t pie_adapter_binding_adapter_id(const PieAdapterBinding*);
int64_t pie_adapter_binding_seed      (const PieAdapterBinding*);

/* ===========================================================
 * Sampler — union of all variant fields. Each accessor returns the
 * value for variants that carry the field; default for others.
 * =========================================================== */

uint8_t  pie_sampler_kind       (const PieSampler*);
float    pie_sampler_temperature(const PieSampler*);
uint32_t pie_sampler_k          (const PieSampler*);
float    pie_sampler_p          (const PieSampler*);
uint32_t pie_sampler_num_tokens (const PieSampler*);
uint32_t pie_sampler_token_id   (const PieSampler*);
void     pie_sampler_token_ids  (const PieSampler*, const uint32_t** out_ptr, size_t* out_len);
/* Multinomial only: `seed == 0` means "use a fresh per-fire random seed". */
uint32_t pie_sampler_seed       (const PieSampler*);

/* ===========================================================
 * CopyRequest / AdapterRequest
 * =========================================================== */

const PieCopyDir* pie_copy_request_dir (const PieCopyRequest*);
void              pie_copy_request_srcs(const PieCopyRequest*, const uint32_t**, size_t*);
void              pie_copy_request_dsts(const PieCopyRequest*, const uint32_t**, size_t*);
const PieCopyResource* pie_copy_request_resource(const PieCopyRequest*);

uint8_t pie_copy_dir_value(const PieCopyDir*);
uint8_t pie_copy_resource_value(const PieCopyResource*);

const PieAdapterOp* pie_adapter_request_op        (const PieAdapterRequest*);
uint64_t            pie_adapter_request_adapter_id(const PieAdapterRequest*);
void                pie_adapter_request_path      (const PieAdapterRequest*, const char**, size_t*);

uint8_t pie_adapter_op_value(const PieAdapterOp*);

/* ===========================================================
 * StatusResponse
 * =========================================================== */

int32_t pie_status_response_status(const PieStatusResponse*);

/* ===========================================================
 * Builder API — descriptor structs + pie_build_<type>
 *
 * Each `#[schema]` type has a matching `Pie<T>Desc`. Vec<T> fields
 * become (ptr, len); Option<T> becomes (has, value) for primitives or
 * (ptr, len)/null for String; nested schema types embed their Desc.
 * Tagged enums embed a `kind` byte plus the union of variant fields.
 * =========================================================== */

typedef struct PieAdapterBindingDesc {
    int64_t adapter_id;  /* -1 = unbound */
    int64_t seed;        /* -1 = no caller-provided seed */
} PieAdapterBindingDesc;

typedef struct PieBrleDesc {
    const uint32_t* buffer_ptr;  size_t buffer_len;
    uint64_t total_size;
} PieBrleDesc;

/* PieBrle accessors: buffer is the run-length array, total_size is the
 * total boolean count this BRLE represents. */
void     pie_brle_buffer    (const PieBrle*, const uint32_t**, size_t*);
uint64_t pie_brle_total_size(const PieBrle*);

typedef struct PieSamplerDesc {
    uint8_t  kind;
    float    temperature;
    uint32_t seed;       /* Multinomial only: 0 = fresh per-fire random seed */
    uint32_t k;
    float    p;
    uint32_t num_tokens;
    uint32_t token_id;
    const uint32_t* token_ids_ptr;
    size_t          token_ids_len;
} PieSamplerDesc;

typedef struct PieForwardRequestDesc {
    const uint32_t* token_ids_ptr;          size_t token_ids_len;
    const uint32_t* position_ids_ptr;       size_t position_ids_len;
    const uint32_t* kv_page_indices_ptr;    size_t kv_page_indices_len;
    const uint32_t* kv_page_indptr_ptr;     size_t kv_page_indptr_len;
    const uint32_t* kv_last_page_lens_ptr;  size_t kv_last_page_lens_len;
    const uint32_t* qo_indptr_ptr;          size_t qo_indptr_len;
    const uint32_t* rs_slot_ids_ptr;        size_t rs_slot_ids_len;
    const uint8_t*  rs_slot_flags_ptr;      size_t rs_slot_flags_len;
    const PieBrleDesc* masks_ptr;           size_t masks_len;
    const uint32_t* mask_indptr_ptr;        size_t mask_indptr_len;
    const PieBrleDesc* logit_masks_ptr;     size_t logit_masks_len;
    const uint32_t* logit_mask_indptr_ptr;  size_t logit_mask_indptr_len;
    const uint32_t* sampling_indices_ptr;   size_t sampling_indices_len;
    const uint32_t* sampling_indptr_ptr;    size_t sampling_indptr_len;
    const PieSamplerDesc* samplers_ptr;     size_t samplers_len;
    const uint32_t* sampler_indptr_ptr;     size_t sampler_indptr_len;
    const PieAdapterBindingDesc* adapter_bindings_ptr; size_t adapter_bindings_len;
    const uint32_t* spec_token_ids_ptr;     size_t spec_token_ids_len;
    const uint32_t* spec_position_ids_ptr;  size_t spec_position_ids_len;
    const uint32_t* spec_indptr_ptr;        size_t spec_indptr_len;
    const uint8_t*  output_spec_flags_ptr;  size_t output_spec_flags_len;
    const uint64_t* context_ids_ptr;        size_t context_ids_len;
    uint8_t single_token_mode;
    uint8_t has_user_mask;
} PieForwardRequestDesc;

typedef struct PieForwardResponseDesc {
    uint32_t num_requests;

    const uint32_t* tokens_indptr_ptr;        size_t tokens_indptr_len;
    const uint32_t* tokens_ptr;               size_t tokens_len;

    const uint32_t* dists_req_indptr_ptr;     size_t dists_req_indptr_len;
    const uint32_t* dists_kv_indptr_ptr;      size_t dists_kv_indptr_len;
    const uint32_t* dists_ids_ptr;            size_t dists_ids_len;
    const float*    dists_probs_ptr;          size_t dists_probs_len;

    const uint32_t* logits_req_indptr_ptr;    size_t logits_req_indptr_len;
    const uint32_t* logits_byte_indptr_ptr;   size_t logits_byte_indptr_len;
    const uint8_t*  logits_bytes_ptr;         size_t logits_bytes_len;

    const uint32_t* logprobs_req_indptr_ptr;  size_t logprobs_req_indptr_len;
    const uint32_t* logprobs_val_indptr_ptr;  size_t logprobs_val_indptr_len;
    const float*    logprobs_values_ptr;      size_t logprobs_values_len;

    const uint32_t* entropies_indptr_ptr;     size_t entropies_indptr_len;
    const float*    entropies_ptr;            size_t entropies_len;

    const uint32_t* spec_indptr_ptr;          size_t spec_indptr_len;
    const uint32_t* spec_tokens_ptr;          size_t spec_tokens_len;
    const uint32_t* spec_positions_ptr;       size_t spec_positions_len;

    uint32_t probe_wire_parse_us;
    uint32_t probe_plan_us;
    uint32_t probe_h2d_us;
    uint32_t probe_kernel_launch_us;
    uint32_t probe_sync_us;
    uint32_t probe_response_build_us;
} PieForwardResponseDesc;

typedef struct PieCopyRequestDesc {
    uint8_t dir;
    const uint32_t* srcs_ptr; size_t srcs_len;
    const uint32_t* dsts_ptr; size_t dsts_len;
    uint8_t resource;
} PieCopyRequestDesc;

typedef struct PieAdapterRequestDesc {
    uint8_t  op;
    uint64_t adapter_id;
    /* Meaningful only when `op == PIE_ADAPTER_OP_LOAD`; empty otherwise. */
    const uint8_t* path_ptr; size_t path_len;
} PieAdapterRequestDesc;

typedef struct PieStatusResponseDesc {
    int32_t status;
} PieStatusResponseDesc;

typedef struct PieRequestPayloadDesc {
    uint8_t kind;
    PieForwardRequestDesc forward;
    PieCopyRequestDesc    copy;
    PieAdapterRequestDesc adapter;
} PieRequestPayloadDesc;

typedef struct PieResponsePayloadDesc {
    uint8_t kind;
    PieForwardResponseDesc forward;
    PieStatusResponseDesc  status;
} PieResponsePayloadDesc;

typedef struct PieFrameDesc {
    uint32_t              driver_id;
    PieRequestPayloadDesc payload;
} PieFrameDesc;

typedef struct PieResponseFrameDesc {
    uint32_t               driver_id;
    uint8_t                aborted;
    PieResponsePayloadDesc payload;
} PieResponseFrameDesc;

/* Builders: serialize a descriptor into `out_buf`. Returns bytes
 * written, or 0 on encode failure / insufficient out_buf_cap. */
size_t pie_build_frame             (const PieFrameDesc*,           uint8_t* out_buf, size_t out_buf_cap);
size_t pie_build_response_frame    (const PieResponseFrameDesc*,   uint8_t* out_buf, size_t out_buf_cap);
size_t pie_build_forward_request   (const PieForwardRequestDesc*,  uint8_t* out_buf, size_t out_buf_cap);
size_t pie_build_forward_response  (const PieForwardResponseDesc*, uint8_t* out_buf, size_t out_buf_cap);
size_t pie_build_copy_request      (const PieCopyRequestDesc*,     uint8_t* out_buf, size_t out_buf_cap);
size_t pie_build_adapter_request   (const PieAdapterRequestDesc*,  uint8_t* out_buf, size_t out_buf_cap);
size_t pie_build_status_response   (const PieStatusResponseDesc*,  uint8_t* out_buf, size_t out_buf_cap);
size_t pie_build_adapter_binding   (const PieAdapterBindingDesc*,  uint8_t* out_buf, size_t out_buf_cap);

/* Sizing: returns the encoded byte count for a descriptor without
 * writing. Use to size a target buffer before calling pie_build_*.
 * Returns 0 on encode failure. Internally serializes-and-discards,
 * so prefer calling the builder directly when you already have a
 * sufficient buffer. */
size_t pie_size_frame              (const PieFrameDesc*);
size_t pie_size_response_frame     (const PieResponseFrameDesc*);
size_t pie_size_forward_request    (const PieForwardRequestDesc*);
size_t pie_size_forward_response   (const PieForwardResponseDesc*);
size_t pie_size_copy_request       (const PieCopyRequestDesc*);
size_t pie_size_adapter_request    (const PieAdapterRequestDesc*);
size_t pie_size_status_response    (const PieStatusResponseDesc*);
size_t pie_size_adapter_binding    (const PieAdapterBindingDesc*);

/* ===========================================================
 * In-process vtable — direct-FFI handoff between Rust (runtime)
 * and a same-process C++ driver. Exchanges PieFrameDesc /
 * PieResponseFrameDesc pointers directly; no rkyv encode/decode
 * on this path. The shmem path (used by Python drivers) goes
 * through pie_parse_<t> / pie_build_<t> with rkyv bytes.
 *
 * Lifetime contract:
 *   - `recv` writes *out_request pointing to a PieFrameDesc that
 *     remains valid until the matching `send_response(req_id)`.
 *   - Every slice pointer inside that descriptor (and its nested
 *     sub-descriptors) shares the same lifetime.
 *   - `send_response` must copy any data it needs synchronously;
 *     the response descriptor and its slices are invalid after
 *     the call returns.
 * =========================================================== */

typedef int32_t PieRecvResult;

typedef struct PieInProcVTable {
    PieRecvResult (*recv)(
        void* ctx,
        const PieFrameDesc** out_request,
        uint32_t*            out_req_id);

    void (*send_response)(
        void*                       ctx,
        uint32_t                    req_id,
        const PieResponseFrameDesc* response);

    void* ctx;
} PieInProcVTable;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PIE_BRIDGE_H */
