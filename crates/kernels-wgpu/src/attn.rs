#![allow(clippy::too_many_arguments)]
//! Attention, and the KV writes that feed it.
//!
//! The head dim is an axis everywhere here and its points are the checkpoint
//! geometries the tree is actually compiled for -- 64 (llama-3.2, gpt-oss),
//! 128 (llama, qwen), 256 (qwen3.5), 512 (gemma4 full-attn). A checkpoint
//! whose width is not a point of the axis has no pipeline, which used to be a
//! runtime PSO failure naming a string and is now a fact the table states.

use kernels::{Axis, Cap, KernelSig, kernel};

use crate::axes::*;

pub static KERNELS: &[KernelSig] = &[
    // 1 in split_qkv.wgsl
    // Three results, which is what makes the row's `Out` indices load-bearing:
    // the reorder reads how many values a kernel produces off the row, and a
    // statement writing three states them all after its inputs.
    kernel!(split_qkv_bf16 "split_qkv_bf16", file = Some("attn/split_qkv.wgsl"), launch = kernels::LaunchRule::SplitPacked,
    operands = kernels::operands![
        packed: Buf <- kernels::Source::In(0),
        q: BufMut <- kernels::Source::Out(0),
        k: BufMut <- kernels::Source::Out(1),
        v: BufMut <- kernels::Source::Out(2),
        params: Buf <- kernels::Source::Param(0),
    ]),
    // 1 in attn_gate.wgsl
    kernel!(gate "gate", axes = &[BF16]),
    // 1 in kv_append.wgsl
    // The first row to name STATE. The cache is not an operand — no traced
    // value stands for it, because it outlives the fire — so the pointers come
    // from the driver's pool through `Resolver::kv` and the row is what asks.
    kernel!(kv_append "kv_append", file = Some("attn/kv_write.wgsl"), launch = kernels::LaunchRule::PerHead,
        operands = kernels::operands![
            k_new: Buf <- kernels::Source::In(0),
            v_new: Buf <- kernels::Source::In(1),
            k_cache: BufMut <- kernels::Source::KvKeys,
            v_cache: BufMut <- kernels::Source::KvValues,
            pos: I32s <- kernels::Source::Positions,
            head_dim: I32 <- kernels::Source::Param(0),
            // The POOL's, not the statement's: `max_ctx * head_dim` for the
            // pool the driver allocated.
            k_head_stride: Usize <- kernels::Source::KvHeadStride,
            k_seq_stride: Usize <- kernels::Source::KvSeqStride,
        ],
        // The STATEMENT's head width, not the fire's -- see
        // `kernels::KernelSig::head_param`. This row already hands the kernel
        // `head_dim` as param 0; the grid asked the fire for the same
        // quantity, and gemma-4 states two.
        head_param = Some(0),
        axes = &[BF16]),
    // 1 in kv_append_paged.wgsl
    // Sparse indices, and the gaps are stated. Buffers 4, 6-9 and 11 belong to
    // a shared ring ABI this kernel does not read; a row is positional, so it
    // lists them as `Unbound` rather than closing the gap and shifting
    // everything after.
    kernel!(kv_append_paged "kv_append_paged", file = Some("attn/kv_write.wgsl"), launch = kernels::LaunchRule::PerHead,
        operands = kernels::operands![
            k_new: Buf <- kernels::Source::In(0),
            v_new: Buf <- kernels::Source::In(1),
            k_pages: BufMut <- kernels::Source::KvKeys,
            v_pages: BufMut <- kernels::Source::KvValues,
            ring_4: Buf,
            head_dim: I32 <- kernels::Source::Param(0),
            ring_6: Buf,
            ring_7: Buf,
            ring_8: Buf,
            ring_9: Buf,
            page_size: I32 <- kernels::Source::KvPageSize,
            ring_11: Buf,
            n_kv_heads: I32 <- kernels::Source::Param(1),
            // The normalized physical destination: `fire_csr` already
            // computes both from the ring positions.
            w_page: U32s <- kernels::Source::KvWritePage,
            w_off: U32s <- kernels::Source::KvWriteOffset,
            ring_15: Buf,
        ],
        // BOTH halves of the head shape, from the two params this row already
        // hands the kernel. The kernel addresses the pool as
        // `(slot * n_kv_heads + head) * head_dim + channel` with the
        // statement's numbers; a grid built from the fire's covered
        // `[256, 16]` where the statement said `[512, 4]`, so on every
        // gemma-4 full-attention layer the top half of every KV head was
        // never written and heads 4..15 landed in the next token's rows.
        head_param = Some(0),
        heads_param = Some(1),
        axes = &[BF16]),
    // 1 in logit_softcap.wgsl
    // gemma's logit softcap: `cap * tanh(x / cap)`, applied to the readout so
    // no logit runs away. A statement and not a mode -- a deployment without
    // one names nothing here rather than passing an infinite cap.
    kernel!(logit_softcap "logit_softcap", file = Some("attn/logit_softcap.wgsl"),
        launch = kernels::LaunchRule::Elementwise,
        operands = kernels::operands![
            logits: Buf <- kernels::Source::In(0),
            out: BufMut <- kernels::Source::Out(0),
            // `SoftcapParams`: cap then n, packed.
            params: Buf <- kernels::Source::Param(0),
        ],
        axes = &[BF16]),
    // 1 in attn_gate.wgsl
    kernel!(q_gate_split "q_gate_split", axes = &[BF16]),
    // 7 in sdpa_paged.wgsl.
    //
    // NOT a clean product, and the row says so by listing its tails. `_p32`
    // and `_p32_sg8` are the same template at `<…, 32, true, 32>` and
    // `<…, 32, true, 8>` where the plain form is `<…, 0, false, 32>`, so they
    // are points of a page-shape axis rather than kernels — but they are only
    // compiled at the head dims that wanted them, so a `head dim × page shape`
    // product would name five entrypoints no shader instantiates.
    //
    // `lacks` on this and on `sdpa_vector_decode`: this tree has no page-mask
    // substitution path either, so an `attn.q` tap with a PageMaskSink is
    // unservable here, and no capture variant exists so neither can publish
    // scores. The declaration says so instead of a pipeline build discovering
    // it.
    // Seventeen buffers, and the row is the only place they are written down.
    // Six are the FIRE's tables — the positions, which request owns each
    // token, the page CSR, the mask and its enable — and the ROW names which,
    // because a text cannot state this fire's data. `sinks` stays a gap:
    // gpt-oss reads it and `llama_like` has none, so the row keeps the slot
    // and no statement fills it until a text that has sinks does.
    kernel!(sdpa_paged_decode "sdpa_paged_decode", file = Some("attn/sdpa_paged.wgsl"),
    launch = kernels::LaunchRule::SdpaVector,
    operands = kernels::operands![
        queries: Buf <- kernels::Source::In(0),
        k_pages: Buf <- kernels::Source::KvKeys,
        v_pages: Buf <- kernels::Source::KvValues,
        out: BufMut <- kernels::Source::Out(0),
        gqa_factor: I32 <- kernels::Source::Param(0),
        position_ids: I32s <- kernels::Source::Positions,
        req_of_token: I32s <- kernels::Source::RequestOfToken,
        kv_page_indices: U32s <- kernels::Source::KvPageIndices,
        kv_page_indptr: U32s <- kernels::Source::KvPageIndptr,
        page_size: I32 <- kernels::Source::KvPageSize,
        n_kv_heads: I32 <- kernels::Source::Param(1),
        scale: F32 <- kernels::Source::ParamF32(2),
        attention_mask: U8s <- kernels::Source::AttentionMask,
        attention_mask_stride: U32 <- kernels::Source::AttentionMaskStride,
        attention_mask_enabled: U8s <- kernels::Source::AttentionMaskEnabled,
        window: I32 <- kernels::Source::Param(4),
        sinks: Buf,
    ],
    lacks = &[Cap::Scores, Cap::PageMaskSink],
    axes = &[Axis {
        what: "head dim and page shape",
        points: &["_bfloat16_d_64", "_bfloat16_d_128", "_bfloat16_d_256",
                  "_bfloat16_d_512", "_bfloat16_d_64_p32",
                  "_bfloat16_d_128_p32", "_bfloat16_d_64_p32_sg8"],
    }]),
    // 1 in sdpa_paged.wgsl
    // The SAME template at `sinks = true`, so the same row with one slot
    // filled. A sink is a per-head learned logit that joins the softmax
    // without a value behind it -- gpt-oss's, and the reason the slot has been
    // open on `sdpa_paged_decode` since the rows were written.
    kernel!(sdpa_paged_decode_sink "sdpa_paged_decode_sink", file = Some("attn/sdpa_paged.wgsl"),
    launch = kernels::LaunchRule::SdpaVector,
    operands = kernels::operands![
        queries: Buf <- kernels::Source::In(0),
        k_pages: Buf <- kernels::Source::KvKeys,
        v_pages: Buf <- kernels::Source::KvValues,
        out: BufMut <- kernels::Source::Out(0),
        gqa_factor: I32 <- kernels::Source::Param(0),
        position_ids: I32s <- kernels::Source::Positions,
        req_of_token: I32s <- kernels::Source::RequestOfToken,
        kv_page_indices: U32s <- kernels::Source::KvPageIndices,
        kv_page_indptr: U32s <- kernels::Source::KvPageIndptr,
        page_size: I32 <- kernels::Source::KvPageSize,
        n_kv_heads: I32 <- kernels::Source::Param(1),
        scale: F32 <- kernels::Source::ParamF32(2),
        attention_mask: U8s <- kernels::Source::AttentionMask,
        attention_mask_stride: U32 <- kernels::Source::AttentionMaskStride,
        attention_mask_enabled: U8s <- kernels::Source::AttentionMaskEnabled,
        window: I32 <- kernels::Source::Param(4),
        sinks: Buf <- kernels::Source::Weight(0),
    ],
    lacks = &[Cap::Scores, Cap::PageMaskSink],
    axes = &[BF16, Axis { what: "head dim", points: &["_d_64"] }]),
    // 1 in sdpa_paged_mma.wgsl
    // Metal's MMA entrypoint names over a scalar body -- see that shader's
    // header for why the name is an ABI point and not a hardware claim. The
    // OPERANDS are `sdpa_paged_tiled`'s exactly, because the two shaders take
    // the same eleven buffers and the same seven scalars; what differs on
    // Metal is the threadgroup width, which is `LaunchRule::SdpaMma`, and what
    // differs here is nothing at all.
    //
    // Stating them is not cosmetic. Upstream's lowering picks this row for a
    // sinked prefill, and while it was UNSTATED the plan supplied FIVE scalars
    // into a uniform block the shader reads SEVEN fields of --
    // `every_launchs_scalars_land_where_its_module_reads_them` says so by name.
    //
    // The pitch stays `Source::AttentionMaskStride` where `kernels-metal`
    // states `Param(3)`: this table's documented divergence, and the reason a
    // user mask works on this backend. See `DELIBERATE` in
    // `tests/entrypoints.rs`.
    kernel!(sdpa_paged_mma "sdpa_paged_mma", file = Some("attn/sdpa_paged_mma.wgsl"),
        launch = kernels::LaunchRule::SdpaMma,
        operands = kernels::operands![
        queries: Buf <- kernels::Source::In(0),
        k_pages: Buf <- kernels::Source::KvKeys,
        v_pages: Buf <- kernels::Source::KvValues,
        out: BufMut <- kernels::Source::Out(0),
        gqa_factor: I32 <- kernels::Source::Param(0),
        position_ids: I32s <- kernels::Source::Positions,
        req_of_token: I32s <- kernels::Source::RequestOfToken,
        kv_page_indices: U32s <- kernels::Source::KvPageIndices,
        kv_page_indptr: U32s <- kernels::Source::KvPageIndptr,
        page_size: I32 <- kernels::Source::KvPageSize,
        n_kv_heads: I32 <- kernels::Source::Param(1),
        scale: F32 <- kernels::Source::ParamF32(2),
        attention_mask: U8s <- kernels::Source::AttentionMask,
        attention_mask_stride: U32 <- kernels::Source::AttentionMaskStride,
        attention_mask_enabled: U8s <- kernels::Source::AttentionMaskEnabled,
        window: I32 <- kernels::Source::Param(4),
        sinks: Buf,
        n_rows: I32 <- kernels::Source::Rows,
    ],
        lacks = &[Cap::Scores, Cap::PageMaskSink],
        axes = &[BF16, Axis { what: "head dim", points: &["_d_64"] }]),
    // 1 in sdpa_paged_mma.wgsl
    // The same template at `sinks = true`, exactly as `sdpa_paged_tiled_sink`
    // is to `sdpa_paged_tiled`.
    kernel!(sdpa_paged_mma_sink "sdpa_paged_mma_sink", file = Some("attn/sdpa_paged_mma.wgsl"),
        launch = kernels::LaunchRule::SdpaMma,
        operands = kernels::operands![
        queries: Buf <- kernels::Source::In(0),
        k_pages: Buf <- kernels::Source::KvKeys,
        v_pages: Buf <- kernels::Source::KvValues,
        out: BufMut <- kernels::Source::Out(0),
        gqa_factor: I32 <- kernels::Source::Param(0),
        position_ids: I32s <- kernels::Source::Positions,
        req_of_token: I32s <- kernels::Source::RequestOfToken,
        kv_page_indices: U32s <- kernels::Source::KvPageIndices,
        kv_page_indptr: U32s <- kernels::Source::KvPageIndptr,
        page_size: I32 <- kernels::Source::KvPageSize,
        n_kv_heads: I32 <- kernels::Source::Param(1),
        scale: F32 <- kernels::Source::ParamF32(2),
        attention_mask: U8s <- kernels::Source::AttentionMask,
        attention_mask_stride: U32 <- kernels::Source::AttentionMaskStride,
        attention_mask_enabled: U8s <- kernels::Source::AttentionMaskEnabled,
        window: I32 <- kernels::Source::Param(4),
        sinks: Buf <- kernels::Source::Weight(0),
        n_rows: I32 <- kernels::Source::Rows,
    ],
        lacks = &[Cap::Scores, Cap::PageMaskSink],
        axes = &[BF16, Axis { what: "head dim", points: &["_d_64"] }]),
    // 4 in sdpa_paged.wgsl
    // `sdpa_paged_decode`'s seventeen operands in the same order, plus an
    // eighteenth: the fire's true row count. The grid rounds the rows up to
    // whole tiles -- see `kernels::LaunchRule::SdpaTiled` -- so the threads of
    // a partial last tile are past the end and this is what tells them.
    //
    // The pitch stays `Source::AttentionMaskStride` where `kernels-metal`
    // states `Param(3)`, which is this table's documented divergence and the
    // reason a user mask works on this backend at all. See `DELIBERATE` in
    // `tests/entrypoints.rs`.
    kernel!(sdpa_paged_tiled "sdpa_paged_tiled", file = Some("attn/sdpa_paged.wgsl"),
        launch = kernels::LaunchRule::SdpaTiled,
        operands = kernels::operands![
        queries: Buf <- kernels::Source::In(0),
        k_pages: Buf <- kernels::Source::KvKeys,
        v_pages: Buf <- kernels::Source::KvValues,
        out: BufMut <- kernels::Source::Out(0),
        gqa_factor: I32 <- kernels::Source::Param(0),
        position_ids: I32s <- kernels::Source::Positions,
        req_of_token: I32s <- kernels::Source::RequestOfToken,
        kv_page_indices: U32s <- kernels::Source::KvPageIndices,
        kv_page_indptr: U32s <- kernels::Source::KvPageIndptr,
        page_size: I32 <- kernels::Source::KvPageSize,
        n_kv_heads: I32 <- kernels::Source::Param(1),
        scale: F32 <- kernels::Source::ParamF32(2),
        attention_mask: U8s <- kernels::Source::AttentionMask,
        attention_mask_stride: U32 <- kernels::Source::AttentionMaskStride,
        attention_mask_enabled: U8s <- kernels::Source::AttentionMaskEnabled,
        window: I32 <- kernels::Source::Param(4),
        sinks: Buf,
        n_rows: I32 <- kernels::Source::Rows,
    ],
        lacks = &[Cap::Scores, Cap::PageMaskSink],
        axes = &[BF16, Axis { what: "head dim", points: &["_d_64", "_d_128", "_d_256", "_d_512"] }]),
    // 1 in sdpa_paged.wgsl
    // The same template at `sinks = true`, exactly as `sdpa_paged_decode_sink`
    // is to `sdpa_paged_decode`.
    kernel!(sdpa_paged_tiled_sink "sdpa_paged_tiled_sink", file = Some("attn/sdpa_paged.wgsl"),
        launch = kernels::LaunchRule::SdpaTiled,
        operands = kernels::operands![
        queries: Buf <- kernels::Source::In(0),
        k_pages: Buf <- kernels::Source::KvKeys,
        v_pages: Buf <- kernels::Source::KvValues,
        out: BufMut <- kernels::Source::Out(0),
        gqa_factor: I32 <- kernels::Source::Param(0),
        position_ids: I32s <- kernels::Source::Positions,
        req_of_token: I32s <- kernels::Source::RequestOfToken,
        kv_page_indices: U32s <- kernels::Source::KvPageIndices,
        kv_page_indptr: U32s <- kernels::Source::KvPageIndptr,
        page_size: I32 <- kernels::Source::KvPageSize,
        n_kv_heads: I32 <- kernels::Source::Param(1),
        scale: F32 <- kernels::Source::ParamF32(2),
        attention_mask: U8s <- kernels::Source::AttentionMask,
        attention_mask_stride: U32 <- kernels::Source::AttentionMaskStride,
        attention_mask_enabled: U8s <- kernels::Source::AttentionMaskEnabled,
        window: I32 <- kernels::Source::Param(4),
        sinks: Buf <- kernels::Source::Weight(0),
        n_rows: I32 <- kernels::Source::Rows,
    ],
        lacks = &[Cap::Scores, Cap::PageMaskSink],
        axes = &[BF16, Axis { what: "head dim", points: &["_d_64"] }]),
    // 1 in sdpa_paged.wgsl
    kernel!(sdpa_paged_tiled_strided "sdpa_paged_tiled_strided",
        axes = &[BF16, Axis { what: "head dim", points: &["_d_256"] }]),
    // 3 in sdpa_vector.wgsl
    // Dense 0..10, and the row is where the WIDTHS live: the four strides are
    // `const constant size_t&` — eight bytes — while the params channel is
    // `u32`. A driver handing a four-byte slot to an eight-byte read gives the
    // kernel the next scalar as this one's high half, so the row's `Usize`
    // says widen and the stage does.
    kernel!(sdpa_vector_decode "sdpa_vector_decode", file = Some("attn/sdpa_vector.wgsl"), launch = kernels::LaunchRule::SdpaVector,
        operands = kernels::operands![
            queries: Buf <- kernels::Source::In(0),
            keys: Buf <- kernels::Source::KvKeys,
            values: Buf <- kernels::Source::KvValues,
            out: BufMut <- kernels::Source::Out(0),
            gqa_factor: I32 <- kernels::Source::Param(0),
            n: I32 <- kernels::Source::Param(1),
            k_head_stride: Usize <- kernels::Source::KvHeadStride,
            k_seq_stride: Usize <- kernels::Source::KvSeqStride,
            v_head_stride: Usize <- kernels::Source::KvHeadStride,
            v_seq_stride: Usize <- kernels::Source::KvSeqStride,
            scale: F32 <- kernels::Source::ParamF32(2),
        ],
        lacks = &[Cap::Scores, Cap::PageMaskSink],
        axes = &[BF16, Axis { what: "head dim", points: &["_d_64", "_d_128", "_d_256"] }]),
    // 1 in sdpa_sliding.wgsl
    kernel!(sdpa_vector_decode_sink "sdpa_vector_decode_sink",
        axes = &[BF16, Axis { what: "head dim", points: &["_d_64"] }]),
    // 2 in sdpa_sliding.wgsl
    // `sdpa_vector_decode` over a SLIDING window, and the window is an
    // operand rather than a flag -- the port's rule that a per-fire choice the
    // C++ made at encode time becomes data on the dispatch.
    //
    // Two row pitches the contiguous form does not have: gemma reads its query
    // out of a wider buffer than it writes.
    kernel!(sdpa_vector_decode_swa "sdpa_vector_decode_swa",
        file = Some("attn/sdpa_sliding.wgsl"),
        launch = kernels::LaunchRule::SdpaVector,
        operands = kernels::operands![
            queries: Buf <- kernels::Source::In(0),
            keys: Buf <- kernels::Source::KvKeys,
            values: Buf <- kernels::Source::KvValues,
            out: BufMut <- kernels::Source::Out(0),
            gqa_factor: I32 <- kernels::Source::Param(0),
            n: I32 <- kernels::Source::Param(1),
            k_head_stride: Usize <- kernels::Source::KvHeadStride,
            k_seq_stride: Usize <- kernels::Source::KvSeqStride,
            v_head_stride: Usize <- kernels::Source::KvHeadStride,
            v_seq_stride: Usize <- kernels::Source::KvSeqStride,
            scale: F32 <- kernels::Source::ParamF32(2),
            window: I32 <- kernels::Source::Param(3),
            q_row_stride: I32 <- kernels::Source::Param(4),
            o_row_stride: I32 <- kernels::Source::Param(5),
        ],
        lacks = &[Cap::Scores, Cap::PageMaskSink],
        axes = &[BF16, Axis { what: "head dim", points: &["_d_256", "_d_512"] }]),
];

use crate::routine::{Bind, Buf, BufMut, Ctx, Env, Fire, I32s, Routine, U8s, U32s, Usize};
use kernels::routine::Refusal;
use kernels::shader::{elementwise, elementwise_rows};

/// The head widths `sdpa_paged_*` is built for.
const PAGED_DIMS: [i32; 4] = [64, 128, 256, 512];

/// The head widths `sdpa_vector_decode` is built for.
const VECTOR_DIMS: [i32; 3] = [64, 128, 256];

/// The head widths the sliding-window decode is built for.
const SWA_DIMS: [i32; 2] = [256, 512];

/// `sdpa_paged_decode`, by head width.
///
/// Four entries where the row beside it states SEVEN points. The other three
/// -- `_d_64_p32`, `_d_128_p32`, `_d_64_p32_sg8` -- are deliberately not
/// reachable from here, and `kernels-vulkan`'s
/// `the_page_shape_tails_are_one_real_variant_and_one_bare_name` is where the
/// measurement lives, and this crate's
/// `the_entrypoints_that_ignore_the_window_are_exactly_the_ones_named_p32` is
/// the same claim asked of this tree. In short: `_p32` sets `PIE_FAST_FULL`, which pins the
/// key run's start to zero, so a caller who asked for a window would get FULL
/// attention and no error; and `_p32_sg8` is byte-identical to `_p32`, so a
/// caller who asked for a short group would get the ordinary one. Both
/// failures are silent and neither has a caller -- no text names a tail and
/// the driver's reachability census lists none -- so a routine that could
/// spell them would only be a way to reach them by mistake.
const PAGED_DECODE: [&str; 4] = [
    "sdpa_paged_decode_bfloat16_d_64",
    "sdpa_paged_decode_bfloat16_d_128",
    "sdpa_paged_decode_bfloat16_d_256",
    "sdpa_paged_decode_bfloat16_d_512",
];

/// `sdpa_paged_tiled`, by head width. All four are real and all four are
/// reachable.
const PAGED_TILED: [&str; 4] = [
    "sdpa_paged_tiled_bfloat16_d_64",
    "sdpa_paged_tiled_bfloat16_d_128",
    "sdpa_paged_tiled_bfloat16_d_256",
    "sdpa_paged_tiled_bfloat16_d_512",
];

/// `sdpa_vector_decode`, by head width.
const VECTOR_DECODE: [&str; 3] = [
    "sdpa_vector_decode_bfloat16_d_64",
    "sdpa_vector_decode_bfloat16_d_128",
    "sdpa_vector_decode_bfloat16_d_256",
];

/// `sdpa_vector_decode_swa`, by head width.
const VECTOR_SWA: [&str; 2] = [
    "sdpa_vector_decode_swa_bfloat16_d_256",
    "sdpa_vector_decode_swa_bfloat16_d_512",
];

/// The query rows one tiled workgroup covers, and its lane count on each axis.
///
/// `sdpa_paged.slang` and `sdpa_paged_mma.slang` both declare
/// `[numthreads(32, 32, 1)]` under `PIE_TILED`, and the tile height is the
/// same 32: the staged key run is shared across 32 query rows, which is the
/// whole reason the tiled form exists beside the decode one.
const TILE: u32 = 32;

/// The y extent of a tiled attention workgroup, from `@workgroup_size(32, 8)`.
///
/// Separate from `TILE` because they are two different numbers that happen
/// to be one on the sibling: the tile is 32 ROWS of work and the workgroup is
/// eight lanes tall, each sweeping four of them.
const TILE_LANES: u32 = 8;

/// Which point of an axis a head width is.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a width the shader tree does not carry. `Narrow`
/// rather than `Wide` even at 1024, because the axis is a LIST and not an
/// interval: 512 is compiled and 384 is not, so "too large" is not what is
/// wrong with either.
fn head_point(head_dim: i32, points: &[i32]) -> Result<usize, Refusal> {
    points
        .iter()
        .position(|d| *d == head_dim)
        .ok_or(Refusal::Narrow {
            what: "the head width",
            at: i64::from(head_dim),
        })
}

/// The vector shape: one workgroup per (query head, row), one lane per head
/// dimension.
///
/// `sdpa_vector.slang`, `sdpa_sliding.slang` and `sdpa_paged.slang`'s decode
/// branch all declare `[numthreads(PIE_HEAD_DIM, 1, 1)]` and all three read
/// `gl_NumWorkGroups.x` back as the query head count, so the x extent in
/// THREADS is the product and the driver's `div_ceil` gives the head count
/// back exactly. It gives it back exactly only because the head width IS the
/// workgroup width -- which is why `head_dim` is a point of the axis here
/// rather than a number the caller chooses.
///
/// # Errors
///
/// [`Refusal::Empty`] for a head count or row count that is zero or negative,
/// and [`Refusal::Grid`] if the x product does not fit a `u32`.
fn vector_grid(head_dim: i32, q_heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if q_heads <= 0 {
        return Err(Refusal::Empty {
            what: "query heads",
        });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    // ONE LANE PER PAIR, and this is where `kernels-vulkan`'s arithmetic does
    // NOT transfer. Its shaders run one lane per channel and it multiplies by
    // `head_dim`; every wgpu decode shader here is
    // `@workgroup_size(PIE_PAIRS)` with `PIE_PAIRS = head_dim / 2`, because a
    // lane packs two bf16 channels into a word. `driver-wgpu::geometry`'s
    // `Rule::SdpaVector` says the same thing as
    // `module.local.at(0) * dims.q_heads`, and refuses outright when the
    // module's width and the fire's head width disagree.
    //
    // Copying vulkan's line would ask for twice the lanes: the second half of
    // every head would run with `d_out` past the row and, because the bodies
    // guard on `arrayLength`, write nothing and report success.
    if head_dim % 2 != 0 {
        return Err(Refusal::Narrow {
            what: "the head width is not a whole number of bf16 pairs",
            at: i64::from(head_dim),
        });
    }
    let x = q_heads
        .unsigned_abs()
        .checked_mul(head_dim.unsigned_abs() / 2)
        .ok_or(Refusal::Grid {
            what: "query heads * the head width in pairs",
            at: i64::from(q_heads) * i64::from(head_dim) / 2,
        })?;
    Ok([x, rows.unsigned_abs(), 1])
}

/// The tiled shape: one 32 x 32 workgroup per (query head, block of 32 rows).
///
/// The row count is rounded UP to whole tiles, which is what the `n_rows`
/// push scalar is for: the last tile's rows past the end have to know they are
/// past it, and the grid cannot tell them.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty head or row count, and [`Refusal::Grid`]
/// if either axis does not fit a `u32` once multiplied by the tile.
fn tiled_grid(q_heads: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if q_heads <= 0 {
        return Err(Refusal::Empty {
            what: "query heads",
        });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    let x = q_heads
        .unsigned_abs()
        .checked_mul(TILE)
        .ok_or(Refusal::Grid {
            what: "query heads * the tile's lane count",
            at: i64::from(q_heads) * i64::from(TILE),
        })?;
    // The tile is 32 ROWS and the workgroup is 32 x 8, so a whole tile is
    // `TILE_LANES` lanes on y and not `TILE`. `kernels-vulkan` multiplies by
    // its own 32 because its module is square; `driver-wgpu::geometry`'s
    // `Rule::SdpaTiled` states `module.local.at(1) * rows.div_ceil(32)`.
    let y = rows
        .unsigned_abs()
        .div_ceil(TILE)
        .checked_mul(TILE_LANES)
        .ok_or(Refusal::Grid {
            what: "rows rounded up to whole tiles",
            at: i64::from(rows),
        })?;
    Ok([x, y, 1])
}

/// The KV write shape: one lane per head dimension, one row per head, one
/// plane per token.
///
/// # Errors
///
/// [`Refusal::Empty`] for any of the three being zero or negative.
fn head_grid(head_dim: i32, heads: i32, depth: i32) -> Result<[u32; 3], Refusal> {
    if head_dim <= 0 {
        return Err(Refusal::Empty {
            what: "the head width",
        });
    }
    if heads <= 0 {
        return Err(Refusal::Empty { what: "heads" });
    }
    if depth <= 0 {
        return Err(Refusal::Empty { what: "tokens" });
    }
    Ok([
        head_dim.unsigned_abs(),
        heads.unsigned_abs(),
        depth.unsigned_abs(),
    ])
}

/// One packed row cut into three, at two boundaries the caller does not state.
///
/// `q_width` and `kv_width` ride in `params` rather than in the push block --
/// the shader reads them out of a `SplitQkvParams` struct -- so this signature
/// cannot check them and does not pretend to. What it does state is
/// `packed_width`, which is `q_width + 2 * kv_width` and is the extent the
/// grid needs; the shader recomputes the same sum from its own copy and
/// guards on it, so a `packed_width` that disagreed would leave a tail of the
/// row uncopied rather than write out of bounds.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty row or an empty rectangle.
pub fn split_qkv_bf16(
    ctx: &Ctx<'_>,
    packed: Buf,
    q: BufMut,
    k: BufMut,
    v: BufMut,
    params: Buf,
    packed_width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "attn/split_qkv.wgsl",
            entrypoint: "split_qkv_bf16",
            lanes: elementwise_rows(*packed_width, *rows)?,
        },
        &[packed.v(), q.v(), k.v(), v.v(), params.v()],
    )
}

/// qwen3.5's attention gate: `attn *= sigmoid(gate)`, in place.
///
/// `row_stride` is a real argument and not a convenience. When it is zero the
/// shader falls back to `gl_NumWorkGroups.x * 256`, which is the width ROUNDED
/// UP to whole workgroups -- so the fallback is the true row pitch only when
/// the width is already a multiple of 256, and is silently too large
/// otherwise. Pass the pitch unless the row is known to be aligned.
///
/// The tail past the row is guarded by the output's own descriptor range
/// (`GetDimensions`), not by anything here, which is why this body has no
/// width to check against a length.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty rectangle.
pub fn gate(
    ctx: &Ctx<'_>,
    attn: BufMut,
    gate: Buf,
    row_stride: i32,
    width: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "attn/gate.wgsl",
            entrypoint: "gate_bfloat16",
            lanes: elementwise_rows(*width, *rows)?,
        },
        &[attn.v(), gate.v(), row_stride.v()],
    )
}

/// qwen3.5's `[query|gate]` split: one interleaved head row cut in two.
///
/// `q_heads` is grid-only and load-bearing: the shader reads it back as
/// `gl_NumWorkGroups.y` to compute the default row pitches. The y axis is one
/// workgroup per head because `[numthreads]` is `(256, 1, 1)`, so the count
/// comes back exactly.
///
/// # Errors
///
/// Whatever `head_grid` refuses, with the head COUNT on the y axis rather
/// than a token depth.
pub fn q_gate_split(
    ctx: &Ctx<'_>,
    qg: Buf,
    q_out: BufMut,
    gate_out: BufMut,
    head_dim: i32,
    qg_row_stride: i32,
    out_row_stride: i32,
    q_heads: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "attn/gate.wgsl",
            entrypoint: "q_gate_split_bfloat16",
            lanes: head_grid(head_dim, *q_heads, *rows)?,
        },
        &[
            qg.v(),
            q_out.v(),
            gate_out.v(),
            head_dim.v(),
            qg_row_stride.v(),
            out_row_stride.v(),
        ],
    )
}

/// The contiguous KV append: one token into the slot `pos` names.
///
/// The two strides are the POOL's, in elements, and they are
/// [`Usize`] because the shader declares them `uint2`. That is also why this
/// is the family that made `driver-vulkan::encode::words` align an extent to
/// its own width: `Push { int head_dim; uint2 k_head_stride; uint2
/// k_seq_stride; }` leaves a four-byte hole after `head_dim`, and a packer
/// that concatenated would push twenty bytes where the module declares
/// twenty-four.
///
/// One token, hence the `1` on the z axis: `pos` is read at index 0 and the
/// destination row is the same for every head.
///
/// # Errors
///
/// See `head_grid`.
pub fn kv_append(
    ctx: &Ctx<'_>,
    k_new: Buf,
    v_new: Buf,
    k_cache: BufMut,
    v_cache: BufMut,
    pos: I32s,
    head_dim: i32,
    k_head_stride: Usize,
    k_seq_stride: Usize,
    heads: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "attn/kv_write.wgsl",
            entrypoint: "kv_append_bfloat16",
            lanes: head_grid(head_dim, *heads, 1)?,
        },
        &[
            k_new.v(),
            v_new.v(),
            k_cache.v(),
            v_cache.v(),
            pos.v(),
            head_dim.v(),
            k_head_stride.v(),
            k_seq_stride.v(),
        ],
    )
}

/// The paged KV append: many tokens, each into the page and offset its own
/// entry names.
///
/// Sixteen arguments, six of which reach the device.
///
/// `kv_write.slang`'s paged branch reads only the four KV planes and the two
/// write tables; bindings 4 through 9 and 11 are the Metal ring ABI's slots
/// and no Vulkan module compiles them. slangc emits no `OpDecorate Binding`
/// for a global it did not compile, so the module's own answer is
/// `{0, 1, 2, 3, 10, 11}` -- six used slots inside a twelve-wide layout.
///
/// The LAYOUT keeps its holes: `Device::build` sizes the descriptor set over
/// `0..declared.bindings`, so `w_page` still lands at descriptor 10. The CALL
/// does not: `Device::slots` skips every unused index while writing
/// descriptors, and `encode::dispatch` refuses any list whose length is not
/// `declared.bindings - holes()`. So the six placeholders that would push
/// `w_page` to the tenth argument are exactly the six that make the dispatch
/// too long.
///
/// The SIGNATURE keeps them anyway, under `_` names. It is the operand list
/// the trace fills and the row states, and `shader_backends_agree.rs` holds
/// the two against each other while the kernel is stated twice. What a body
/// may do is decline to forward one -- a subsequence, never a reordering,
/// which `tests/routines.rs` measures both halves of.
///
/// # Errors
///
/// See `head_grid`.
pub fn kv_append_paged(
    ctx: &Ctx<'_>,
    k_new: Buf,
    v_new: Buf,
    k_pages: BufMut,
    v_pages: BufMut,
    _ring_4: Buf,
    head_dim: i32,
    _ring_6: Buf,
    _ring_7: Buf,
    _ring_8: Buf,
    _ring_9: Buf,
    page_size: i32,
    _ring_11: Buf,
    n_kv_heads: i32,
    w_page: U32s,
    w_off: U32s,
    _ring_15: Buf,
    tokens: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "attn/kv_write.wgsl",
            entrypoint: "kv_append_paged_bfloat16",
            lanes: head_grid(head_dim, n_kv_heads, *tokens)?,
        },
        &[
            k_new.v(),
            v_new.v(),
            k_pages.v(),
            v_pages.v(),
            head_dim.v(),
            page_size.v(),
            n_kv_heads.v(),
            w_page.v(),
            w_off.v(),
        ],
    )
}

/// gemma's final logit softcap: `cap * tanh(x / cap)`.
///
/// The cap rides in `params` with a trailing word nothing reads, so the only
/// extent this body states is the element count.
///
/// # Errors
///
/// [`Refusal::Empty`] for an empty readout.
pub fn logit_softcap(
    ctx: &Ctx<'_>,
    logits: Buf,
    out: BufMut,
    params: Buf,
    n: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "attn/logit_softcap.wgsl",
            entrypoint: "logit_softcap_bfloat16",
            lanes: elementwise(*n, 1)?,
        },
        &[logits.v(), out.v(), params.v()],
    )
}

/// Paged decode attention: one query row per workgroup row, walking the pages
/// its request owns.
///
/// Ten buffers, and six of them are the FIRE's tables rather than the
/// statement's values -- the positions, which request owns each token, the
/// page CSR, the mask and its enable. `sdpa_paged.slang` declares `sinks` at
/// binding 10 unconditionally, but nothing reads it without `PIE_WITH_SINK`
/// and slangc decorates no binding for a global it compiled out, so this form
/// takes ten arguments and [`sdpa_paged_decode_sink`] -- whose module really
/// does decorate 10 -- takes eleven. One text, two signatures, and the
/// module's decoration set is the only thing that says which is which.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a head width off `PAGED_DIMS`, and whatever
/// `vector_grid` refuses.
pub fn sdpa_paged_decode(
    ctx: &Ctx<'_>,
    queries: Buf,
    k_pages: Buf,
    v_pages: Buf,
    out: BufMut,
    gqa_factor: i32,
    position_ids: I32s,
    req_of_token: I32s,
    kv_page_indices: U32s,
    kv_page_indptr: U32s,
    page_size: i32,
    n_kv_heads: i32,
    scale: f32,
    attention_mask: U8s,
    attention_mask_stride: u32,
    attention_mask_enabled: U8s,
    window: i32,
    // FORWARDED HERE, where `kernels-vulkan` takes it as `_sinks` and drops
    // it. `slangc` emits no binding for a global the un-sinked variant never
    // reads; WGSL declares it in source and `naga` keeps it, so this module
    // declares eleven `@group(0)` bindings and the layout is built from that.
    // Skipping the slot binds the window scalar's block where the sink plane
    // belongs. §8c, in the direction it was written for.
    sinks: Buf,
    head_dim: Env<i32>,
    q_heads: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "attn/sdpa_paged.wgsl",
            entrypoint: PAGED_DECODE[head_point(*head_dim, &PAGED_DIMS)?],
            lanes: vector_grid(*head_dim, *q_heads, *rows)?,
        },
        &[
            queries.v(),
            k_pages.v(),
            v_pages.v(),
            out.v(),
            gqa_factor.v(),
            position_ids.v(),
            req_of_token.v(),
            kv_page_indices.v(),
            kv_page_indptr.v(),
            page_size.v(),
            n_kv_heads.v(),
            scale.v(),
            attention_mask.v(),
            attention_mask_stride.v(),
            attention_mask_enabled.v(),
            window.v(),
            sinks.v(),
        ],
    )
}

/// Paged decode with attention sinks.
///
/// A sink is a per-head learned logit that joins the softmax with no value
/// behind it -- gpt-oss's, and the one point compiled is `d_64` because that
/// is gpt-oss's head width. The signature is [`sdpa_paged_decode`]'s
/// unchanged, with `sinks` actually read.
///
/// # Errors
///
/// [`Refusal::Narrow`] for any head width but 64, and whatever
/// `vector_grid` refuses.
pub fn sdpa_paged_decode_sink(
    ctx: &Ctx<'_>,
    queries: Buf,
    k_pages: Buf,
    v_pages: Buf,
    out: BufMut,
    gqa_factor: i32,
    position_ids: I32s,
    req_of_token: I32s,
    kv_page_indices: U32s,
    kv_page_indptr: U32s,
    page_size: i32,
    n_kv_heads: i32,
    scale: f32,
    attention_mask: U8s,
    attention_mask_stride: u32,
    attention_mask_enabled: U8s,
    window: i32,
    sinks: Buf,
    head_dim: Env<i32>,
    q_heads: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    head_point(*head_dim, &[64])?;
    ctx.dispatch(
        Fire {
            module: "attn/sdpa_paged.wgsl",
            entrypoint: "sdpa_paged_decode_sink_bfloat16_d_64",
            lanes: vector_grid(*head_dim, *q_heads, *rows)?,
        },
        &[
            queries.v(),
            k_pages.v(),
            v_pages.v(),
            out.v(),
            gqa_factor.v(),
            position_ids.v(),
            req_of_token.v(),
            kv_page_indices.v(),
            kv_page_indptr.v(),
            page_size.v(),
            n_kv_heads.v(),
            scale.v(),
            attention_mask.v(),
            attention_mask_stride.v(),
            attention_mask_enabled.v(),
            window.v(),
            sinks.v(),
        ],
    )
}

/// Paged prefill attention: a tile of 32 query rows against one staged run of
/// keys.
///
/// The difference from [`sdpa_paged_decode`] is not the arithmetic but the
/// sharing. The decode form gives one workgroup to each (head, query row) and
/// each walks the whole key run alone; this one stages the run once for 32
/// rows, which is why it exists and why `n_rows` appears: the grid rounds up
/// to whole tiles, so the last tile has rows past the end and only this scalar
/// can tell them so.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a head width off `PAGED_DIMS`, and whatever
/// `tiled_grid` refuses.
pub fn sdpa_paged_tiled(
    ctx: &Ctx<'_>,
    queries: Buf,
    k_pages: Buf,
    v_pages: Buf,
    out: BufMut,
    gqa_factor: i32,
    position_ids: I32s,
    req_of_token: I32s,
    kv_page_indices: U32s,
    kv_page_indptr: U32s,
    page_size: i32,
    n_kv_heads: i32,
    scale: f32,
    attention_mask: U8s,
    attention_mask_stride: u32,
    attention_mask_enabled: U8s,
    window: i32,
    // FORWARDED HERE, where `kernels-vulkan` takes it as `_sinks` and drops
    // it. `slangc` emits no binding for a global the un-sinked variant never
    // reads; WGSL declares it in source and `naga` keeps it, so this module
    // declares eleven `@group(0)` bindings and the layout is built from that.
    // Skipping the slot binds the window scalar's block where the sink plane
    // belongs. §8c, in the direction it was written for.
    sinks: Buf,
    n_rows: i32,
    head_dim: Env<i32>,
    q_heads: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "attn/sdpa_paged.wgsl",
            entrypoint: PAGED_TILED[head_point(*head_dim, &PAGED_DIMS)?],
            lanes: tiled_grid(*q_heads, n_rows)?,
        },
        &[
            queries.v(),
            k_pages.v(),
            v_pages.v(),
            out.v(),
            gqa_factor.v(),
            position_ids.v(),
            req_of_token.v(),
            kv_page_indices.v(),
            kv_page_indptr.v(),
            page_size.v(),
            n_kv_heads.v(),
            scale.v(),
            attention_mask.v(),
            attention_mask_stride.v(),
            attention_mask_enabled.v(),
            window.v(),
            sinks.v(),
            n_rows.v(),
        ],
    )
}

/// Paged prefill with attention sinks. One point, `d_64`.
///
/// # Errors
///
/// [`Refusal::Narrow`] for any head width but 64, and whatever `tiled_grid`
/// refuses.
pub fn sdpa_paged_tiled_sink(
    ctx: &Ctx<'_>,
    queries: Buf,
    k_pages: Buf,
    v_pages: Buf,
    out: BufMut,
    gqa_factor: i32,
    position_ids: I32s,
    req_of_token: I32s,
    kv_page_indices: U32s,
    kv_page_indptr: U32s,
    page_size: i32,
    n_kv_heads: i32,
    scale: f32,
    attention_mask: U8s,
    attention_mask_stride: u32,
    attention_mask_enabled: U8s,
    window: i32,
    sinks: Buf,
    n_rows: i32,
    head_dim: Env<i32>,
    q_heads: Env<i32>,
) -> Result<(), Refusal> {
    head_point(*head_dim, &[64])?;
    ctx.dispatch(
        Fire {
            module: "attn/sdpa_paged.wgsl",
            entrypoint: "sdpa_paged_tiled_sink_bfloat16_d_64",
            lanes: tiled_grid(*q_heads, n_rows)?,
        },
        &[
            queries.v(),
            k_pages.v(),
            v_pages.v(),
            out.v(),
            gqa_factor.v(),
            position_ids.v(),
            req_of_token.v(),
            kv_page_indices.v(),
            kv_page_indptr.v(),
            page_size.v(),
            n_kv_heads.v(),
            scale.v(),
            attention_mask.v(),
            attention_mask_stride.v(),
            attention_mask_enabled.v(),
            window.v(),
            sinks.v(),
            n_rows.v(),
        ],
    )
}

/// Paged prefill whose query and output rows have their own pitches.
///
/// TWO more push scalars than [`sdpa_paged_tiled`] and they are the whole
/// difference: `PIE_STRIDED` adds `q_row_pitch` and `o_row_pitch` to the end
/// of the block, so a checkpoint that reads its queries out of a wider buffer
/// than it writes has somewhere to say so. One point, `d_256`, which is
/// qwen3.5's.
///
/// # Errors
///
/// [`Refusal::Narrow`] for any head width but 256, and whatever
/// `tiled_grid` refuses.
pub fn sdpa_paged_tiled_strided(
    ctx: &Ctx<'_>,
    queries: Buf,
    k_pages: Buf,
    v_pages: Buf,
    out: BufMut,
    gqa_factor: i32,
    position_ids: I32s,
    req_of_token: I32s,
    kv_page_indices: U32s,
    kv_page_indptr: U32s,
    page_size: i32,
    n_kv_heads: i32,
    scale: f32,
    attention_mask: U8s,
    attention_mask_stride: u32,
    attention_mask_enabled: U8s,
    window: i32,
    // FORWARDED HERE, where `kernels-vulkan` takes it as `_sinks` and drops
    // it. `slangc` emits no binding for a global the un-sinked variant never
    // reads; WGSL declares it in source and `naga` keeps it, so this module
    // declares eleven `@group(0)` bindings and the layout is built from that.
    // Skipping the slot binds the window scalar's block where the sink plane
    // belongs. §8c, in the direction it was written for.
    sinks: Buf,
    n_rows: i32,
    q_row_pitch: i32,
    o_row_pitch: i32,
    head_dim: Env<i32>,
    q_heads: Env<i32>,
) -> Result<(), Refusal> {
    head_point(*head_dim, &[256])?;
    ctx.dispatch(
        Fire {
            module: "attn/sdpa_paged.wgsl",
            entrypoint: "sdpa_paged_tiled_strided_bfloat16_d_256",
            lanes: tiled_grid(*q_heads, n_rows)?,
        },
        &[
            queries.v(),
            k_pages.v(),
            v_pages.v(),
            out.v(),
            gqa_factor.v(),
            position_ids.v(),
            req_of_token.v(),
            kv_page_indices.v(),
            kv_page_indptr.v(),
            page_size.v(),
            n_kv_heads.v(),
            scale.v(),
            attention_mask.v(),
            attention_mask_stride.v(),
            attention_mask_enabled.v(),
            window.v(),
            sinks.v(),
            n_rows.v(),
            q_row_pitch.v(),
            o_row_pitch.v(),
        ],
    )
}

/// The cooperative-matrix prefill.
///
/// The same eighteen operands and the same 32-row tile as
/// [`sdpa_paged_tiled`]: the module's push block is field for field identical.
/// What differs is inside -- the `Q.K^T` and `P.V` inner products go through
/// `coopMatMulAdd`.
///
/// The tier does NOT change the entrypoint. `sdpa_paged_mma.slang` names
/// `sdpa_paged_mma_bfloat16_d_64` on both its plain and its `@coopmat`
/// instantiate lines, and which module a name resolves to is the driver's
/// choice from the device's capability tiers, not this body's. That is why a
/// [`Fire`] carries a name and no tier.
///
/// # Errors
///
/// [`Refusal::Narrow`] for any head width but 64, and whatever `tiled_grid`
/// refuses.
pub fn sdpa_paged_mma(
    ctx: &Ctx<'_>,
    queries: Buf,
    k_pages: Buf,
    v_pages: Buf,
    out: BufMut,
    gqa_factor: i32,
    position_ids: I32s,
    req_of_token: I32s,
    kv_page_indices: U32s,
    kv_page_indptr: U32s,
    page_size: i32,
    n_kv_heads: i32,
    scale: f32,
    attention_mask: U8s,
    attention_mask_stride: u32,
    attention_mask_enabled: U8s,
    window: i32,
    // FORWARDED HERE, where `kernels-vulkan` takes it as `_sinks` and drops
    // it. `slangc` emits no binding for a global the un-sinked variant never
    // reads; WGSL declares it in source and `naga` keeps it, so this module
    // declares eleven `@group(0)` bindings and the layout is built from that.
    // Skipping the slot binds the window scalar's block where the sink plane
    // belongs. §8c, in the direction it was written for.
    sinks: Buf,
    n_rows: i32,
    head_dim: Env<i32>,
    q_heads: Env<i32>,
) -> Result<(), Refusal> {
    head_point(*head_dim, &[64])?;
    ctx.dispatch(
        Fire {
            module: "attn/sdpa_paged_mma.wgsl",
            entrypoint: "sdpa_paged_mma_bfloat16_d_64",
            lanes: tiled_grid(*q_heads, n_rows)?,
        },
        &[
            queries.v(),
            k_pages.v(),
            v_pages.v(),
            out.v(),
            gqa_factor.v(),
            position_ids.v(),
            req_of_token.v(),
            kv_page_indices.v(),
            kv_page_indptr.v(),
            page_size.v(),
            n_kv_heads.v(),
            scale.v(),
            attention_mask.v(),
            attention_mask_stride.v(),
            attention_mask_enabled.v(),
            window.v(),
            sinks.v(),
            n_rows.v(),
        ],
    )
}

/// The cooperative-matrix prefill with attention sinks.
///
/// # Errors
///
/// [`Refusal::Narrow`] for any head width but 64, and whatever `tiled_grid`
/// refuses.
pub fn sdpa_paged_mma_sink(
    ctx: &Ctx<'_>,
    queries: Buf,
    k_pages: Buf,
    v_pages: Buf,
    out: BufMut,
    gqa_factor: i32,
    position_ids: I32s,
    req_of_token: I32s,
    kv_page_indices: U32s,
    kv_page_indptr: U32s,
    page_size: i32,
    n_kv_heads: i32,
    scale: f32,
    attention_mask: U8s,
    attention_mask_stride: u32,
    attention_mask_enabled: U8s,
    window: i32,
    sinks: Buf,
    n_rows: i32,
    head_dim: Env<i32>,
    q_heads: Env<i32>,
) -> Result<(), Refusal> {
    head_point(*head_dim, &[64])?;
    ctx.dispatch(
        Fire {
            module: "attn/sdpa_paged_mma.wgsl",
            entrypoint: "sdpa_paged_mma_sink_bfloat16_d_64",
            lanes: tiled_grid(*q_heads, n_rows)?,
        },
        &[
            queries.v(),
            k_pages.v(),
            v_pages.v(),
            out.v(),
            gqa_factor.v(),
            position_ids.v(),
            req_of_token.v(),
            kv_page_indices.v(),
            kv_page_indptr.v(),
            page_size.v(),
            n_kv_heads.v(),
            scale.v(),
            attention_mask.v(),
            attention_mask_stride.v(),
            attention_mask_enabled.v(),
            window.v(),
            sinks.v(),
            n_rows.v(),
        ],
    )
}

/// The dense, unpaged decode: one contiguous cache, addressed by four strides.
///
/// The four strides are [`Usize`] because `sdpa_vector.slang` declares them
/// `uint2`. Here they need no alignment hole -- two `int`s precede them, so
/// the first already starts on an eight-byte boundary -- which is exactly why
/// `driver-vulkan::encode::words` pads before an extent rather than after one.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a head width off `VECTOR_DIMS`, and whatever
/// `vector_grid` refuses.
pub fn sdpa_vector_decode(
    ctx: &Ctx<'_>,
    queries: Buf,
    keys: Buf,
    values: Buf,
    out: BufMut,
    gqa_factor: i32,
    n: i32,
    k_head_stride: Usize,
    k_seq_stride: Usize,
    v_head_stride: Usize,
    v_seq_stride: Usize,
    scale: f32,
    head_dim: Env<i32>,
    q_heads: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "attn/sdpa_vector.wgsl",
            entrypoint: VECTOR_DECODE[head_point(*head_dim, &VECTOR_DIMS)?],
            lanes: vector_grid(*head_dim, *q_heads, *rows)?,
        },
        &[
            queries.v(),
            keys.v(),
            values.v(),
            out.v(),
            gqa_factor.v(),
            n.v(),
            k_head_stride.v(),
            k_seq_stride.v(),
            v_head_stride.v(),
            v_seq_stride.v(),
            scale.v(),
        ],
    )
}

/// The dense decode over a SLIDING window.
///
/// The window is an operand and not a flag -- the port's rule that a per-fire
/// choice the C++ made at encode time becomes data on the dispatch -- and it
/// comes with two row pitches the contiguous form does not have, because
/// gemma reads its query out of a wider buffer than it writes.
///
/// # Errors
///
/// [`Refusal::Narrow`] for a head width off `SWA_DIMS`, and whatever
/// `vector_grid` refuses.
pub fn sdpa_vector_decode_swa(
    ctx: &Ctx<'_>,
    queries: Buf,
    keys: Buf,
    values: Buf,
    out: BufMut,
    gqa_factor: i32,
    n: i32,
    k_head_stride: Usize,
    k_seq_stride: Usize,
    v_head_stride: Usize,
    v_seq_stride: Usize,
    scale: f32,
    window: i32,
    q_row_stride: i32,
    o_row_stride: i32,
    head_dim: Env<i32>,
    q_heads: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    ctx.dispatch(
        Fire {
            module: "attn/sdpa_sliding.wgsl",
            entrypoint: VECTOR_SWA[head_point(*head_dim, &SWA_DIMS)?],
            lanes: vector_grid(*head_dim, *q_heads, *rows)?,
        },
        &[
            queries.v(),
            keys.v(),
            values.v(),
            out.v(),
            gqa_factor.v(),
            n.v(),
            k_head_stride.v(),
            k_seq_stride.v(),
            v_head_stride.v(),
            v_seq_stride.v(),
            scale.v(),
            window.v(),
            q_row_stride.v(),
            o_row_stride.v(),
        ],
    )
}

/// The dense decode with attention sinks.
///
/// It lives in `sdpa_sliding.slang` rather than in `sdpa_vector.slang`, which
/// is why it takes the SLIDING signature: the window and the two row pitches
/// are in the push block whether or not a caller wants a window, and `sinks`
/// takes binding 4, which the windowed form leaves undeclared. So this is
/// [`sdpa_vector_decode_swa`] plus one buffer, and not
/// [`sdpa_vector_decode`] plus one buffer.
///
/// # Errors
///
/// [`Refusal::Narrow`] for any head width but 64, and whatever
/// `vector_grid` refuses.
pub fn sdpa_vector_decode_sink(
    ctx: &Ctx<'_>,
    queries: Buf,
    keys: Buf,
    values: Buf,
    out: BufMut,
    sinks: Buf,
    gqa_factor: i32,
    n: i32,
    k_head_stride: Usize,
    k_seq_stride: Usize,
    v_head_stride: Usize,
    v_seq_stride: Usize,
    scale: f32,
    window: i32,
    q_row_stride: i32,
    o_row_stride: i32,
    head_dim: Env<i32>,
    q_heads: Env<i32>,
    rows: Env<i32>,
) -> Result<(), Refusal> {
    head_point(*head_dim, &[64])?;
    ctx.dispatch(
        Fire {
            module: "attn/sdpa_vector.wgsl",
            entrypoint: "sdpa_vector_decode_sink_bfloat16_d_64",
            lanes: vector_grid(*head_dim, *q_heads, *rows)?,
        },
        &[
            queries.v(),
            keys.v(),
            values.v(),
            out.v(),
            sinks.v(),
            gqa_factor.v(),
            n.v(),
            k_head_stride.v(),
            k_seq_stride.v(),
            v_head_stride.v(),
            v_seq_stride.v(),
            scale.v(),
            window.v(),
            q_row_stride.v(),
            o_row_stride.v(),
        ],
    )
}

/// The sixteen, in the order the rows above name them.
pub static ROUTINES: &[Routine] = &[
    crate::routine!(split_qkv_bf16),
    crate::routine!(gate, in_place = &[(0, 0)]),
    crate::routine!(kv_append),
    crate::routine!(kv_append_paged),
    crate::routine!(logit_softcap),
    crate::routine!(q_gate_split),
    crate::routine!(sdpa_paged_decode),
    crate::routine!(sdpa_paged_decode_sink),
    crate::routine!(sdpa_paged_mma),
    crate::routine!(sdpa_paged_mma_sink),
    crate::routine!(sdpa_paged_tiled),
    crate::routine!(sdpa_paged_tiled_sink),
    crate::routine!(sdpa_paged_tiled_strided),
    crate::routine!(sdpa_vector_decode),
    crate::routine!(sdpa_vector_decode_sink),
    crate::routine!(sdpa_vector_decode_swa),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The two grids that are NOT `kernels-vulkan`'s, pinned as numbers.
    ///
    /// This family's bodies were ported from vulkan's and its seven family
    /// tests were NOT: two of them assert `q_heads * head_dim` for the vector
    /// extent and a square tile for the tiled one, and both are wrong here.
    /// Shipping them would have been seven green tests asserting the sibling's
    /// arithmetic about this backend's shaders.
    ///
    /// What replaces them is this, plus `tests/routines.rs`, which drives all
    /// sixteen bodies over every head width their tables carry and checks each
    /// dispatch's buffer count against the parsed `naga` module.
    #[test]
    fn the_two_grids_that_are_not_vulkans_are_the_ones_the_driver_states() {
        // ONE LANE PER PAIR: every decode shader is
        // `@workgroup_size(PIE_PAIRS)` with `PIE_PAIRS = head_dim / 2`, so a
        // 64-wide head over 8 query heads is 8 * 32 lanes and not 8 * 64.
        // `driver-wgpu::geometry`'s `Rule::SdpaVector` says
        // `module.local.at(0) * dims.q_heads` for the same reason.
        assert_eq!(vector_grid(64, 8, 7).expect("a real point"), [256, 7, 1]);
        assert_eq!(vector_grid(128, 4, 1).expect("a real point"), [256, 1, 1]);

        // THE TILE IS 32 ROWS AND THE WORKGROUP IS 32 x 8. Vulkan multiplies
        // the row-tile count by its own square 32; here y is `TILE_LANES`.
        // `Rule::SdpaTiled` states `module.local.at(1) * rows.div_ceil(32)`.
        assert_eq!(tiled_grid(8, 32).expect("one whole tile"), [256, 8, 1]);
        assert_eq!(tiled_grid(8, 33).expect("two tiles"), [256, 16, 1]);
        assert_eq!(tiled_grid(8, 1).expect("a partial tile"), [256, 8, 1]);

        // And the head width has to be a whole number of pairs, because half
        // a pair is a lane that would read one channel and write two.
        assert!(matches!(vector_grid(65, 8, 7), Err(Refusal::Narrow { .. })));
    }

    /// An empty extent is refused rather than dispatched.
    #[test]
    fn an_empty_head_count_or_row_count_is_refused() {
        assert!(matches!(vector_grid(64, 0, 7), Err(Refusal::Empty { .. })));
        assert!(matches!(vector_grid(64, 8, 0), Err(Refusal::Empty { .. })));
        assert!(matches!(tiled_grid(0, 32), Err(Refusal::Empty { .. })));
        assert!(matches!(tiled_grid(8, 0), Err(Refusal::Empty { .. })));
        assert!(matches!(head_grid(0, 8, 7), Err(Refusal::Empty { .. })));
        assert!(matches!(head_grid(64, 8, 0), Err(Refusal::Empty { .. })));
    }

    /// A head width the tree does not carry is refused by NAME.
    ///
    /// The bodies index a literal spelling table with the head width, so an
    /// unknown one must not reach it: `PAGED_DIMS` carries four and a fifth
    /// would panic on the index rather than refuse.
    #[test]
    fn a_head_width_the_tree_does_not_carry_is_refused_by_name() {
        assert!(head_point(96, &PAGED_DIMS).is_err());
        assert!(head_point(512, &VECTOR_DIMS).is_err());
        assert!(head_point(64, &SWA_DIMS).is_err());
        // And the ones it does carry resolve to their own index.
        assert_eq!(head_point(64, &PAGED_DIMS).expect("carried"), 0);
        assert_eq!(head_point(512, &PAGED_DIMS).expect("carried"), 3);
        assert_eq!(head_point(256, &SWA_DIMS).expect("carried"), 0);
    }
}
