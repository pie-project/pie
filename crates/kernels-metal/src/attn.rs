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
    kernel!(kv_append "kv_append", file = Some("attn/kv_write.metal"), launch = kernels::LaunchRule::PerHead, axes = &[BF16]),
    // 1 in kv_append_paged.metal
    kernel!(kv_append_paged "kv_append_paged", file = Some("attn/kv_write.metal"), launch = kernels::LaunchRule::PerHead, axes = &[BF16]),
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
    kernel!(sdpa_paged_decode "sdpa_paged_decode", file = Some("attn/sdpa_paged.metal"),
    launch = kernels::LaunchRule::SdpaVector,
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
    kernel!(sdpa_vector_decode "sdpa_vector_decode", file = Some("attn/sdpa_vector.metal"), launch = kernels::LaunchRule::SdpaVector,
        lacks = &[Cap::Scores, Cap::PageMaskSink],
        axes = &[BF16, Axis { what: "head dim", points: &["_d_64", "_d_128", "_d_256"] }]),
    // 1 in sdpa_sliding.metal
    kernel!(sdpa_vector_decode_sink "sdpa_vector_decode_sink",
        axes = &[BF16, Axis { what: "head dim", points: &["_d_64"] }]),
    // 2 in sdpa_sliding.metal
    kernel!(sdpa_vector_decode_swa "sdpa_vector_decode_swa",
        axes = &[BF16, Axis { what: "head dim", points: &["_d_256", "_d_512"] }]),
];
