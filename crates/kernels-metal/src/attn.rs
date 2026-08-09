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
    // 1 in split_qkv.metal
    // Three results, which is what makes the row's `Out` indices load-bearing:
    // the reorder reads how many values a kernel produces off the row, and a
    // statement writing three states them all after its inputs.
    kernel!(split_qkv_bf16 "split_qkv_bf16", file = Some("attn/split_qkv.metal"), launch = kernels::LaunchRule::SplitPacked,
        operands = kernels::operands![
            packed: Buf <- kernels::Source::In(0),
            q: BufMut <- kernels::Source::Out(0),
            k: BufMut <- kernels::Source::Out(1),
            v: BufMut <- kernels::Source::Out(2),
            params: Buf <- kernels::Source::Param(0),
        ]),
    // 1 in attn_gate.metal
    kernel!(gate "gate", axes = &[BF16]),
    // 1 in kv_append.metal
    // The first row to name STATE. The cache is not an operand — no traced
    // value stands for it, because it outlives the fire — so the pointers come
    // from the driver's pool through `Resolver::kv` and the row is what asks.
    kernel!(kv_append "kv_append", file = Some("attn/kv_write.metal"), launch = kernels::LaunchRule::PerHead,
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
        axes = &[BF16]),
    // 1 in kv_append_paged.metal
    // Sparse indices, and the gaps are stated. Buffers 4, 6-9 and 11 belong to
    // a shared ring ABI this kernel does not read; a row is positional, so it
    // lists them as `Unbound` rather than closing the gap and shifting
    // everything after.
    kernel!(kv_append_paged "kv_append_paged", file = Some("attn/kv_write.metal"), launch = kernels::LaunchRule::PerHead,
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
        axes = &[BF16]),
    // 1 in logit_softcap.metal
    kernel!(logit_softcap "logit_softcap", axes = &[BF16]),
    // 1 in attn_gate.metal
    kernel!(q_gate_split "q_gate_split", axes = &[BF16]),
    // 7 in sdpa_paged.metal.
    //
    // NOT a clean product, and the row says so by listing its tails. `_p32`
    // and `_p32_sg8` are the same template at `<…, 32, true, 32>` and
    // `<…, 32, true, 8>` where the plain form is `<…, 0, false, 32>`, so they
    // are points of a page-shape axis rather than kernels — but they are only
    // compiled at the head dims that wanted them, so a `head dim × page shape`
    // product would name five entrypoints no shader instantiates.
    //
    // `lacks` on this and on `sdpa_vector_decode`: Metal has no page-mask
    // substitution path, so an `attn.q` tap with a PageMaskSink is unservable
    // here, and no capture variant exists so neither can publish scores. The
    // declaration says so instead of a C++ throw discovering it.
    // Seventeen buffers, and the row is the only place they are written down.
    // Six are the FIRE's tables — the positions, which request owns each
    // token, the page CSR, the mask and its enable — and the ROW names which,
    // because a text cannot state this fire's data. `sinks` stays a gap:
    // gpt-oss reads it and `llama_like` has none, so the row keeps the slot
    // and no statement fills it until a text that has sinks does.
    kernel!(sdpa_paged_decode "sdpa_paged_decode", file = Some("attn/sdpa_paged.metal"),
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
        attention_mask_stride: U32 <- kernels::Source::Param(3),
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
    // 1 in sdpa_paged.metal
    kernel!(sdpa_paged_decode_sink "sdpa_paged_decode_sink",
        axes = &[BF16, Axis { what: "head dim", points: &["_d_64"] }]),
    // 1 in sdpa_paged_mma.metal
    kernel!(sdpa_paged_mma "sdpa_paged_mma",
        axes = &[BF16, Axis { what: "head dim", points: &["_d_64"] }]),
    // 1 in sdpa_paged_mma.metal
    kernel!(sdpa_paged_mma_sink "sdpa_paged_mma_sink",
        axes = &[BF16, Axis { what: "head dim", points: &["_d_64"] }]),
    // 4 in sdpa_paged.metal
    kernel!(sdpa_paged_tiled "sdpa_paged_tiled",
        axes = &[BF16, Axis { what: "head dim", points: &["_d_64", "_d_128", "_d_256", "_d_512"] }]),
    // 1 in sdpa_paged.metal
    kernel!(sdpa_paged_tiled_sink "sdpa_paged_tiled_sink",
        axes = &[BF16, Axis { what: "head dim", points: &["_d_64"] }]),
    // 1 in sdpa_paged.metal
    kernel!(sdpa_paged_tiled_strided "sdpa_paged_tiled_strided",
        axes = &[BF16, Axis { what: "head dim", points: &["_d_256"] }]),
    // 3 in sdpa_vector.metal
    // Dense 0..10, and the row is where the WIDTHS live: the four strides are
    // `const constant size_t&` — eight bytes — while the params channel is
    // `u32`. A driver handing a four-byte slot to an eight-byte read gives the
    // kernel the next scalar as this one's high half, so the row's `Usize`
    // says widen and the stage does.
    kernel!(sdpa_vector_decode "sdpa_vector_decode", file = Some("attn/sdpa_vector.metal"), launch = kernels::LaunchRule::SdpaVector,
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
    // 1 in sdpa_sliding.metal
    kernel!(sdpa_vector_decode_sink "sdpa_vector_decode_sink",
        axes = &[BF16, Axis { what: "head dim", points: &["_d_64"] }]),
    // 2 in sdpa_sliding.metal
    kernel!(sdpa_vector_decode_swa "sdpa_vector_decode_swa",
        axes = &[BF16, Axis { what: "head dim", points: &["_d_256", "_d_512"] }]),
];
