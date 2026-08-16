use std::ffi::CString;

/// One header, and the name an `#include` spells to reach it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    /// What `#include "…"` must say, relative to `kernels/`.
    pub name: &'static str,
    /// The text, carried in the binary by [`include_str!`].
    pub text: &'static str,
}

/// The impersonation layer: headers wearing NVIDIA's and the standard
/// library's filenames, carried because the source that reaches for them is
/// source we do not own and the spelling is the contract.
///
/// Goes to every compile, with [`LIBRARY`].
///
/// # The three lists are the header set, and they are maintained by hand
///
/// NVRTC matches `includeNames[]` against the **literal string in the
/// `#include` directive** -- there is no search path and no resolution -- so
/// an entry's `name` is exactly what the includer writes, and a header that
/// is not in the set does not exist as far as a compile is concerned.
/// **Adding a file under `kernels/` or `shim/` means adding a line to one of
/// these.**
///
/// Forgetting is caught by `every_device_include_resolves` in this file's
/// tests: it walks the set's own quoted `#include`s and fails on the first one
/// nothing carries, at `cargo test` and on a machine with no GPU. What it
/// cannot catch is a file nothing includes yet, or one reached only by an
/// angled spelling. The compiler catches neither, because a short list still
/// compiles.
///
/// 18 entries, from `shim/`.
#[rustfmt::skip]
pub const SHIM: &[Header] = &[
    Header { name: "array", text: include_str!("../shim/array") },
    Header { name: "cassert", text: include_str!("../shim/cassert") },
    Header { name: "cooperative_groups.h", text: include_str!("../shim/cooperative_groups.h") },
    Header { name: "cstddef", text: include_str!("../shim/cstddef") },
    Header { name: "cstdint", text: include_str!("../shim/cstdint") },
    Header { name: "cstring", text: include_str!("../shim/cstring") },
    Header { name: "cuda.h", text: include_str!("../shim/cuda.h") },
    Header { name: "cuda/cmath", text: include_str!("../shim/cuda/cmath") },
    Header { name: "cuda/std/limits", text: include_str!("../shim/cuda/std/limits") },
    Header { name: "cuda/std/optional", text: include_str!("../shim/cuda/std/optional") },
    Header { name: "cuda_bf16.h", text: include_str!("../shim/cuda_bf16.h") },
    Header { name: "cuda_fp16.h", text: include_str!("../shim/cuda_fp16.h") },
    Header { name: "cuda_fp4.h", text: include_str!("../shim/cuda_fp4.h") },
    Header { name: "cuda_fp8.h", text: include_str!("../shim/cuda_fp8.h") },
    Header { name: "cuda_pipeline.h", text: include_str!("../shim/cuda_pipeline.h") },
    Header { name: "cuda_runtime.h", text: include_str!("../shim/cuda_runtime.h") },
    Header { name: "math_constants.h", text: include_str!("../shim/math_constants.h") },
    Header { name: "type_traits", text: include_str!("../shim/type_traits") },
];

/// This crate's own device text: every `__global__` template a unit compiles
/// and the prelude they are written over.
///
/// Goes to every compile, with [`SHIM`]. See [`SHIM`] for the rule this list
/// is maintained under.
///
/// 89 entries, from `kernels/`, minus the upstream subtrees.
#[rustfmt::skip]
pub const LIBRARY: &[Header] = &[
    Header { name: "attn/attention_flashinfer.cuh", text: include_str!("../kernels/attn/attention_flashinfer.cuh") },
    Header { name: "attn/attention_mla_fa2.cuh", text: include_str!("../kernels/attn/attention_mla_fa2.cuh") },
    Header { name: "attn/attention_mla_naive.cuh", text: include_str!("../kernels/attn/attention_mla_naive.cuh") },
    Header { name: "attn/attention_naive.cuh", text: include_str!("../kernels/attn/attention_naive.cuh") },
    Header { name: "attn/attention_naive_paged.cuh", text: include_str!("../kernels/attn/attention_naive_paged.cuh") },
    Header { name: "attn/attention_score_capture.cuh", text: include_str!("../kernels/attn/attention_score_capture.cuh") },
    Header { name: "attn/attention_score_post.cuh", text: include_str!("../kernels/attn/attention_score_post.cuh") },
    Header { name: "attn/attention_xqa.cuh", text: include_str!("../kernels/attn/attention_xqa.cuh") },
    Header { name: "attn/attention_xqa_mha.cuh", text: include_str!("../kernels/attn/attention_xqa_mha.cuh") },
    Header { name: "attn/attn_res.cuh", text: include_str!("../kernels/attn/attn_res.cuh") },
    Header { name: "attn/attn_sink.cuh", text: include_str!("../kernels/attn/attn_sink.cuh") },
    Header { name: "attn/dsa_indexer.cuh", text: include_str!("../kernels/attn/dsa_indexer.cuh") },
    Header { name: "attn/dsv4_compress.cuh", text: include_str!("../kernels/attn/dsv4_compress.cuh") },
    Header { name: "attn/fa2.cuh", text: include_str!("../kernels/attn/fa2.cuh") },
    Header { name: "attn/fa4.cuh", text: include_str!("../kernels/attn/fa4.cuh") },
    Header { name: "attn/head_dim_pad.cuh", text: include_str!("../kernels/attn/head_dim_pad.cuh") },
    Header { name: "attn/kimi_mla.cuh", text: include_str!("../kernels/attn/kimi_mla.cuh") },
    Header { name: "attn/kv_paged.cuh", text: include_str!("../kernels/attn/kv_paged.cuh") },
    Header { name: "attn/mla_paged.cuh", text: include_str!("../kernels/attn/mla_paged.cuh") },
    Header { name: "attn/pack_dense_mask.cuh", text: include_str!("../kernels/attn/pack_dense_mask.cuh") },
    Header { name: "attn/page_compact.cuh", text: include_str!("../kernels/attn/page_compact.cuh") },
    Header { name: "attn/qkv_fused.cuh", text: include_str!("../kernels/attn/qkv_fused.cuh") },
    Header { name: "attn/softcap.cuh", text: include_str!("../kernels/attn/softcap.cuh") },
    Header { name: "attn/split_packed.cuh", text: include_str!("../kernels/attn/split_packed.cuh") },
    Header { name: "cascade/merge_states.cuh", text: include_str!("../kernels/cascade/merge_states.cuh") },
    Header { name: "comm/all_reduce.cuh", text: include_str!("../kernels/comm/all_reduce.cuh") },
    Header { name: "gemm/gemv.cuh", text: include_str!("../kernels/gemm/gemv.cuh") },
    Header { name: "graph/supergraph.cuh", text: include_str!("../kernels/graph/supergraph.cuh") },
    Header { name: "prelude/kv_paged_addr.cuh", text: include_str!("../kernels/prelude/kv_paged_addr.cuh") },
    Header { name: "layout/deinterleave.cuh", text: include_str!("../kernels/layout/deinterleave.cuh") },
    Header { name: "layout/embed.cuh", text: include_str!("../kernels/layout/embed.cuh") },
    Header { name: "layout/envelope.cuh", text: include_str!("../kernels/layout/envelope.cuh") },
    Header { name: "layout/envelope_device.cuh", text: include_str!("../kernels/layout/envelope_device.cuh") },
    Header { name: "layout/gather_rows.cuh", text: include_str!("../kernels/layout/gather_rows.cuh") },
    Header { name: "layout/gather_rows_tile.cuh", text: include_str!("../kernels/layout/gather_rows_tile.cuh") },
    Header { name: "layout/gather_tokens.cuh", text: include_str!("../kernels/layout/gather_tokens.cuh") },
    Header { name: "layout/geometry.cuh", text: include_str!("../kernels/layout/geometry.cuh") },
    Header { name: "layout/graph_pad.cuh", text: include_str!("../kernels/layout/graph_pad.cuh") },
    Header { name: "layout/slot_ops.cuh", text: include_str!("../kernels/layout/slot_ops.cuh") },
    Header { name: "layout/split_gate_up.cuh", text: include_str!("../kernels/layout/split_gate_up.cuh") },
    Header { name: "mlp/gaussian_topk.cuh", text: include_str!("../kernels/mlp/gaussian_topk.cuh") },
    Header { name: "mlp/swiglu.cuh", text: include_str!("../kernels/mlp/swiglu.cuh") },
    Header { name: "mlp/swiglu_tile.cuh", text: include_str!("../kernels/mlp/swiglu_tile.cuh") },
    Header { name: "moe/dsv4_routing.cuh", text: include_str!("../kernels/moe/dsv4_routing.cuh") },
    Header { name: "moe/expert_offsets.cuh", text: include_str!("../kernels/moe/expert_offsets.cuh") },
    Header { name: "moe/moe_dispatch.cuh", text: include_str!("../kernels/moe/moe_dispatch.cuh") },
    Header { name: "moe/moe_fused_tile.cuh", text: include_str!("../kernels/moe/moe_fused_tile.cuh") },
    Header { name: "moe/moe_grouped_gemm.cuh", text: include_str!("../kernels/moe/moe_grouped_gemm.cuh") },
    Header { name: "moe/moe_grouped_gemm_tile.cuh", text: include_str!("../kernels/moe/moe_grouped_gemm_tile.cuh") },
    Header { name: "moe/topk_sigmoid.cuh", text: include_str!("../kernels/moe/topk_sigmoid.cuh") },
    Header { name: "moe/topk_softmax.cuh", text: include_str!("../kernels/moe/topk_softmax.cuh") },
    Header { name: "moe/topk_softmax_tile.cuh", text: include_str!("../kernels/moe/topk_softmax_tile.cuh") },
    Header { name: "norm/add_bias.cuh", text: include_str!("../kernels/norm/add_bias.cuh") },
    Header { name: "norm/altup.cuh", text: include_str!("../kernels/norm/altup.cuh") },
    Header { name: "norm/altup_aux.cuh", text: include_str!("../kernels/norm/altup_aux.cuh") },
    Header { name: "norm/dsv4_hc.cuh", text: include_str!("../kernels/norm/dsv4_hc.cuh") },
    Header { name: "norm/elementwise.cuh", text: include_str!("../kernels/norm/elementwise.cuh") },
    Header { name: "norm/rmsnorm.cuh", text: include_str!("../kernels/norm/rmsnorm.cuh") },
    Header { name: "norm/rmsnorm_rasr_tile.cuh", text: include_str!("../kernels/norm/rmsnorm_rasr_tile.cuh") },
    Header { name: "norm/rmsnorm_tile.cuh", text: include_str!("../kernels/norm/rmsnorm_tile.cuh") },
    Header { name: "prelude/device.cuh", text: include_str!("../kernels/prelude/device.cuh") },
    Header { name: "prelude/fp8.cuh", text: include_str!("../kernels/prelude/fp8.cuh") },
    Header { name: "prelude/half2.cuh", text: include_str!("../kernels/prelude/half2.cuh") },
    Header { name: "prelude/mma.cuh", text: include_str!("../kernels/prelude/mma.cuh") },
    Header { name: "quant/dequant_fp4.cuh", text: include_str!("../kernels/quant/dequant_fp4.cuh") },
    Header { name: "quant/dequant_fp8.cuh", text: include_str!("../kernels/quant/dequant_fp8.cuh") },
    Header { name: "quant/dequant_wna16.cuh", text: include_str!("../kernels/quant/dequant_wna16.cuh") },
    Header { name: "quant/dequant_wna16_tile.cuh", text: include_str!("../kernels/quant/dequant_wna16_tile.cuh") },
    Header { name: "quant/dtype_cast.cuh", text: include_str!("../kernels/quant/dtype_cast.cuh") },
    Header { name: "quant/mxfp4_marlin.cuh", text: include_str!("../kernels/quant/mxfp4_marlin.cuh") },
    Header { name: "quant/quant_bf16_to_fp8.cuh", text: include_str!("../kernels/quant/quant_bf16_to_fp8.cuh") },
    Header { name: "quant/quant_bf16_to_mxfp4.cuh", text: include_str!("../kernels/quant/quant_bf16_to_mxfp4.cuh") },
    Header { name: "quant/transcode.cuh", text: include_str!("../kernels/quant/transcode.cuh") },
    Header { name: "quant/wna16_gemv_tile.cuh", text: include_str!("../kernels/quant/wna16_gemv_tile.cuh") },
    Header { name: "rope/rope.cuh", text: include_str!("../kernels/rope/rope.cuh") },
    Header { name: "rope/rope_tile.cuh", text: include_str!("../kernels/rope/rope_tile.cuh") },
    Header { name: "prelude/rope.cuh", text: include_str!("../kernels/prelude/rope.cuh") },
    Header { name: "sample/argmax.cuh", text: include_str!("../kernels/sample/argmax.cuh") },
    Header { name: "sample/argmax_tile.cuh", text: include_str!("../kernels/sample/argmax_tile.cuh") },
    Header { name: "ssm/causal_conv1d.cuh", text: include_str!("../kernels/ssm/causal_conv1d.cuh") },
    Header { name: "ssm/gated_delta_net.cuh", text: include_str!("../kernels/ssm/gated_delta_net.cuh") },
    Header { name: "ssm/gated_delta_net_prep.cuh", text: include_str!("../kernels/ssm/gated_delta_net_prep.cuh") },
    Header { name: "ssm/kda.cuh", text: include_str!("../kernels/ssm/kda.cuh") },
    Header { name: "ssm/nemotron_h.cuh", text: include_str!("../kernels/ssm/nemotron_h.cuh") },
    Header { name: "tile/alternatives.cuh", text: include_str!("../kernels/tile/alternatives.cuh") },
    Header { name: "vision/gemma4_audio.cuh", text: include_str!("../kernels/vision/gemma4_audio.cuh") },
    Header { name: "vision/gemma4_naive_kernels.cuh", text: include_str!("../kernels/vision/gemma4_naive_kernels.cuh") },
    Header { name: "vision/gemma4_vision.cuh", text: include_str!("../kernels/vision/gemma4_vision.cuh") },
    Header { name: "vision/qwen3_vl_tower.cuh", text: include_str!("../kernels/vision/qwen3_vl_tower.cuh") },
    Header { name: "vision/tower_naive_kernels.cuh", text: include_str!("../kernels/vision/tower_naive_kernels.cuh") },
];

/// The internalised FlashInfer and XQA closure.
///
/// Kept out of [`DEVICE_HEADERS`] and handed only to the units that ask for
/// it: this is somebody else's attention library and `nvrtcCreateProgram`
/// copies every byte it is given, so a `norm` compile does not carry it. See
/// [`SHIM`] for the rule this list is maintained under.
///
/// # 39 of the 79 entries name a file that is already here, on purpose
///
/// NVRTC does no path resolution, so a file two directives reach by two
/// spellings needs an entry per spelling. The upstream trees moved in INTACT
/// -- which is why not one upstream byte had to change -- so they still reach
/// their siblings the way they always did, in two forms:
///
/// * `../cp_async.cuh`, from `attention/decode.cuh`. Thirteen of these, and
///   they sort to the top of this list. `comm/` reaches its siblings exactly
///   as `attention/` does, which is why internalising that directory added
///   one spelling -- `../fp4_layout.cuh` -- and reused `../utils.cuh` and
///   `../vec_dtypes.cuh` unchanged.
/// * `cp_async.cuh`, bare, from a file in the same directory. Twenty-six of
///   these, and they sort in among the real entries.
///
/// **Neither form is a typo and neither may be "corrected" to the path it
/// duplicates.** An entry whose `name` does not match the tail of its
/// `include_str!` path is one of these 39, it costs a pointer, and deleting
/// one stops the compile at the first directive that spelled it that way.
///
/// 79 entries, from `kernels/flashinfer` and `kernels/xqa`.
#[rustfmt::skip]
pub const UPSTREAM: &[Header] = &[
    Header { name: "../cp_async.cuh", text: include_str!("../kernels/flashinfer/cp_async.cuh") },
    Header { name: "../fastdiv.cuh", text: include_str!("../kernels/flashinfer/fastdiv.cuh") },
    Header { name: "../fp4_layout.cuh", text: include_str!("../kernels/flashinfer/fp4_layout.cuh") },
    Header { name: "../frag_layout_swizzle.cuh", text: include_str!("../kernels/flashinfer/frag_layout_swizzle.cuh") },
    Header { name: "../layout.cuh", text: include_str!("../kernels/flashinfer/layout.cuh") },
    Header { name: "../math.cuh", text: include_str!("../kernels/flashinfer/math.cuh") },
    Header { name: "../mma.cuh", text: include_str!("../kernels/flashinfer/mma.cuh") },
    Header { name: "../page.cuh", text: include_str!("../kernels/flashinfer/page.cuh") },
    Header { name: "../permuted_smem.cuh", text: include_str!("../kernels/flashinfer/permuted_smem.cuh") },
    Header { name: "../pos_enc.cuh", text: include_str!("../kernels/flashinfer/pos_enc.cuh") },
    Header { name: "../profiler.cuh", text: include_str!("../kernels/flashinfer/profiler.cuh") },
    Header { name: "../utils.cuh", text: include_str!("../kernels/flashinfer/utils.cuh") },
    Header { name: "../vec_dtypes.cuh", text: include_str!("../kernels/flashinfer/vec_dtypes.cuh") },

    Header { name: "flashinfer/attention/cascade.cuh", text: include_str!("../kernels/flashinfer/attention/cascade.cuh") },
    Header { name: "flashinfer/attention/decode.cuh", text: include_str!("../kernels/flashinfer/attention/decode.cuh") },
    Header { name: "flashinfer/attention/default_decode_params.cuh", text: include_str!("../kernels/flashinfer/attention/default_decode_params.cuh") },
    Header { name: "flashinfer/attention/default_prefill_params.cuh", text: include_str!("../kernels/flashinfer/attention/default_prefill_params.cuh") },
    Header { name: "flashinfer/attention/mask.cuh", text: include_str!("../kernels/flashinfer/attention/mask.cuh") },
    Header { name: "flashinfer/attention/mla.cuh", text: include_str!("../kernels/flashinfer/attention/mla.cuh") },
    Header { name: "flashinfer/attention/mla_params.cuh", text: include_str!("../kernels/flashinfer/attention/mla_params.cuh") },
    Header { name: "flashinfer/attention/prefill.cuh", text: include_str!("../kernels/flashinfer/attention/prefill.cuh") },
    Header { name: "flashinfer/attention/state.cuh", text: include_str!("../kernels/flashinfer/attention/state.cuh") },
    Header { name: "flashinfer/attention/variant_helper.cuh", text: include_str!("../kernels/flashinfer/attention/variant_helper.cuh") },
    Header { name: "flashinfer/attention/variants.cuh", text: include_str!("../kernels/flashinfer/attention/variants.cuh") },
    Header { name: "flashinfer/comm/trtllm_allreduce_fusion.cuh", text: include_str!("../kernels/flashinfer/comm/trtllm_allreduce_fusion.cuh") },
    Header { name: "flashinfer/comm/vllm_custom_all_reduce.cuh", text: include_str!("../kernels/flashinfer/comm/vllm_custom_all_reduce.cuh") },
    Header { name: "flashinfer/cp_async.cuh", text: include_str!("../kernels/flashinfer/cp_async.cuh") },
    Header { name: "flashinfer/fastdiv.cuh", text: include_str!("../kernels/flashinfer/fastdiv.cuh") },
    Header { name: "flashinfer/fp4_layout.cuh", text: include_str!("../kernels/flashinfer/fp4_layout.cuh") },
    Header { name: "flashinfer/frag_layout_swizzle.cuh", text: include_str!("../kernels/flashinfer/frag_layout_swizzle.cuh") },
    Header { name: "flashinfer/layout.cuh", text: include_str!("../kernels/flashinfer/layout.cuh") },
    Header { name: "flashinfer/math.cuh", text: include_str!("../kernels/flashinfer/math.cuh") },
    Header { name: "flashinfer/mma.cuh", text: include_str!("../kernels/flashinfer/mma.cuh") },
    Header { name: "flashinfer/page.cuh", text: include_str!("../kernels/flashinfer/page.cuh") },
    Header { name: "flashinfer/permuted_smem.cuh", text: include_str!("../kernels/flashinfer/permuted_smem.cuh") },
    Header { name: "flashinfer/pos_enc.cuh", text: include_str!("../kernels/flashinfer/pos_enc.cuh") },
    Header { name: "flashinfer/profiler.cuh", text: include_str!("../kernels/flashinfer/profiler.cuh") },
    Header { name: "flashinfer/utils.cuh", text: include_str!("../kernels/flashinfer/utils.cuh") },
    Header { name: "flashinfer/vec_dtypes.cuh", text: include_str!("../kernels/flashinfer/vec_dtypes.cuh") },
    Header { name: "xqa/barriers.cuh", text: include_str!("../kernels/xqa/barriers.cuh") },
    Header { name: "xqa/cuda_hint.cuh", text: include_str!("../kernels/xqa/cuda_hint.cuh") },
    Header { name: "xqa/defines.h", text: include_str!("../kernels/xqa/defines.h") },
    Header { name: "xqa/ldgsts.cuh", text: include_str!("../kernels/xqa/ldgsts.cuh") },
    Header { name: "xqa/mha.cuh", text: include_str!("../kernels/xqa/mha.cuh") },
    Header { name: "xqa/mha.h", text: include_str!("../kernels/xqa/mha.h") },
    Header { name: "xqa/mhaUtils.cuh", text: include_str!("../kernels/xqa/mhaUtils.cuh") },
    Header { name: "xqa/mha_components.cuh", text: include_str!("../kernels/xqa/mha_components.cuh") },
    Header { name: "xqa/mha_stdheaders.cuh", text: include_str!("../kernels/xqa/mha_stdheaders.cuh") },
    Header { name: "xqa/mma.cuh", text: include_str!("../kernels/xqa/mma.cuh") },
    Header { name: "xqa/platform.h", text: include_str!("../kernels/xqa/platform.h") },
    Header { name: "xqa/specDec.h", text: include_str!("../kernels/xqa/specDec.h") },
    Header { name: "xqa/utils.cuh", text: include_str!("../kernels/xqa/utils.cuh") },
    Header { name: "xqa/utils.h", text: include_str!("../kernels/xqa/utils.h") },
    Header { name: "barriers.cuh", text: include_str!("../kernels/xqa/barriers.cuh") },
    Header { name: "cascade.cuh", text: include_str!("../kernels/flashinfer/attention/cascade.cuh") },
    Header { name: "cp_async.cuh", text: include_str!("../kernels/flashinfer/cp_async.cuh") },
    Header { name: "cuda_hint.cuh", text: include_str!("../kernels/xqa/cuda_hint.cuh") },
    Header { name: "defines.h", text: include_str!("../kernels/xqa/defines.h") },
    Header { name: "fastdiv.cuh", text: include_str!("../kernels/flashinfer/fastdiv.cuh") },
    Header { name: "layout.cuh", text: include_str!("../kernels/flashinfer/layout.cuh") },
    Header { name: "ldgsts.cuh", text: include_str!("../kernels/xqa/ldgsts.cuh") },
    Header { name: "mask.cuh", text: include_str!("../kernels/flashinfer/attention/mask.cuh") },
    Header { name: "math.cuh", text: include_str!("../kernels/flashinfer/math.cuh") },
    Header { name: "mha.h", text: include_str!("../kernels/xqa/mha.h") },
    Header { name: "mhaUtils.cuh", text: include_str!("../kernels/xqa/mhaUtils.cuh") },
    Header { name: "mha_components.cuh", text: include_str!("../kernels/xqa/mha_components.cuh") },
    Header { name: "mha_stdheaders.cuh", text: include_str!("../kernels/xqa/mha_stdheaders.cuh") },
    Header { name: "mla_params.cuh", text: include_str!("../kernels/flashinfer/attention/mla_params.cuh") },
    Header { name: "mma.cuh", text: include_str!("../kernels/flashinfer/mma.cuh") },
    Header { name: "page.cuh", text: include_str!("../kernels/flashinfer/page.cuh") },
    Header { name: "platform.h", text: include_str!("../kernels/xqa/platform.h") },
    Header { name: "prefill.cuh", text: include_str!("../kernels/flashinfer/attention/prefill.cuh") },
    Header { name: "specDec.h", text: include_str!("../kernels/xqa/specDec.h") },
    Header { name: "state.cuh", text: include_str!("../kernels/flashinfer/attention/state.cuh") },
    Header { name: "utils.cuh", text: include_str!("../kernels/flashinfer/utils.cuh") },
    Header { name: "utils.h", text: include_str!("../kernels/xqa/utils.h") },
    Header { name: "variant_helper.cuh", text: include_str!("../kernels/flashinfer/attention/variant_helper.cuh") },
    Header { name: "variants.cuh", text: include_str!("../kernels/flashinfer/attention/variants.cuh") },
    Header { name: "vec_dtypes.cuh", text: include_str!("../kernels/flashinfer/vec_dtypes.cuh") },
];

/// What every compile in this crate resolves an `#include` against, unless the
const SHIMMED: [Header; SHIM.len() + LIBRARY.len()] =
    join::<{ SHIM.len() + LIBRARY.len() }>(SHIM, LIBRARY);

/// See [`SHIMMED`].
pub const DEVICE_HEADERS: &[Header] = &SHIMMED;

/// [`DEVICE_HEADERS`] plus [`UPSTREAM`] — what a unit compiling upstream
pub const ALL_HEADERS: &[Header] =
    &join::<{ SHIM.len() + LIBRARY.len() + UPSTREAM.len() }>(&SHIMMED, UPSTREAM);

/// `[T] ++ [U] -> [T; N]` at compile time.
const fn join<const N: usize>(left: &[Header], right: &[Header]) -> [Header; N] {
    let mut out = [Header { name: "", text: "" }; N];
    let mut w = 0;
    let mut i = 0;
    while i < left.len() {
        out[w] = left[i];
        w += 1;
        i += 1;
    }
    let mut j = 0;
    while j < right.len() {
        out[w] = right[j];
        w += 1;
        j += 1;
    }
    out
}

/// The text [`LIBRARY`] carries under `name`, by the spelling an `#include`
/// reaches it with.
///
/// # Why a root does not `include_str!` its own file
///
/// [`LIBRARY`] is the whole of `kernels/` as the binary carries it, and
/// [`DEVICE_HEADERS`] hands all of it to every compile. A root that also wrote
/// `include_str!("../kernels/layout/slot_ops.cuh")` was stating a second time
/// what this list already holds, in a second spelling — a `../` path whose
/// depth is a fact about where the *Rust* file sits, so moving `layout.rs` to
/// `layout/mod.rs` broke every root declared in it. Naming the carried file is
/// the whole of what a root has to say.
///
/// # A name nothing answers to does not compile
///
/// That is the point of the `const fn` and the [`panic!`]: `include_str!`
/// failed at compile time on a path that did not exist, and giving that up for
/// a runtime lookup would move a typo from `cargo check` to the first fire of
/// that kernel on a GPU. Const evaluation refuses instead, and the diagnostic
/// points at the declaration, which is where the misspelled name is written.
#[must_use]
pub const fn carried(name: &'static str) -> &'static str {
    let mut i = 0;
    while i < LIBRARY.len() {
        if str_eq(LIBRARY[i].name, name) {
            return LIBRARY[i].text;
        }
        i += 1;
    }
    panic!("no file under `kernels/` is carried under that name")
}

/// [`carried`], as a value rather than a refusal to compile.
///
/// The launch path names its file at run time, so a miss there has to be
/// answerable — see [`crate::jit::Root::of`].
#[must_use]
pub fn text_of(name: &str) -> Option<&'static str> {
    LIBRARY.iter().find(|header| header.name == name).map(|header| header.text)
}

/// `a == b`, in a `const` context.
///
/// `str::eq` is not `const`, and the comparison has to happen during const
/// evaluation or [`carried`] cannot refuse at compile time.
pub(crate) const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// The header set as the two parallel arrays `nvrtcCreateProgram` wants,
pub fn as_nvrtc_arrays(headers: &[Header]) -> Result<(Vec<CString>, Vec<CString>), String> {
    let mut texts = Vec::with_capacity(headers.len());
    let mut names = Vec::with_capacity(headers.len());
    for header in headers {
        texts.push(
            CString::new(header.text)
                .map_err(|_| format!("header `{}` contains a NUL", header.name))?,
        );
        names.push(
            CString::new(header.name)
                .map_err(|_| format!("header name `{}` contains a NUL", header.name))?,
        );
    }
    Ok((texts, names))
}

/// Every carried header reachable from `root`, or the first `#include` the set
pub fn reachable(from: &str, root: &str, headers: &[Header]) -> Result<Vec<&'static str>, String> {
    let mut seen: Vec<&'static str> = Vec::new();
    let mut queue: Vec<(&str, &str)> = vec![(from, root)];
    while let Some((at, text)) = queue.pop() {
        for included in quoted_includes(text) {
            let Some(header) = headers.iter().find(|h| h.name == included) else {
                return Err(format!(
                    "`{from}` reaches `{included}` from `{at}`, and the header set it \
                     compiles against does not carry it -- NVRTC resolves against the \
                     set and nothing else, so this compiles nowhere"
                ));
            };
            if !seen.contains(&header.name) {
                seen.push(header.name);
                queue.push((header.name, header.text));
            }
        }
    }
    Ok(seen)
}

/// FNV-1a 64 over every header's name and text, in table order.
#[must_use]
pub fn digest(headers: &[Header]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for header in headers {
        hash = fold(hash, header.name.as_bytes());
        hash = fold(hash, &[0]);
        hash = fold(hash, header.text.as_bytes());
        hash = fold(hash, &[0]);
    }
    hash
}

/// Every quoted `#include` in `source`, in order of appearance.
#[must_use]
pub fn quoted_includes(source: &str) -> Vec<&str> {
    source
        .lines()
        .filter_map(|line| {
            let rest = line.strip_prefix("#include")?;
            let rest = rest.strip_prefix(|c: char| c == ' ' || c == '\t')?;
            rest.trim_start().strip_prefix('"')?.split('"').next()
        })
        .collect()
}

/// FNV-1a 64 over bytes, the fold [`digest`] is built out of.
pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    fold(FNV_OFFSET_BASIS, bytes)
}

/// The algorithm's offset basis, and the same one
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// One more chunk into a running FNV-1a, so a digest over several fields is
fn fold(mut hash: u64, bytes: &[u8]) -> u64 {
    /// The algorithm's prime.
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::Headers;

    /// The set a unit chooses decides what resolves — checked with a header
    #[test]
    fn a_units_header_choice_decides_what_resolves() {
        let upstream = UPSTREAM
            .iter()
            .find(|v| !Headers::Library.set().iter().any(|l| l.name == v.name))
            .expect("the upstream closure carries a header the library does not");
        let root = format!("#include \"{}\"\n", upstream.name);

        let refused = reachable("a/unit", &root, Headers::Library.set())
            .expect_err("the library set does not carry an upstream header");
        assert!(refused.contains(upstream.name), "a refusal names the include: {refused}");

        let resolved = reachable("a/unit", &root, Headers::LibraryAndUpstream.set())
            .expect("the upstream set carries it");
        assert!(resolved.contains(&upstream.name), "and reports what it reached: {resolved:?}");
    }

    /// Reachability is transitive and reports the header it came THROUGH,
    #[test]
    fn an_include_two_headers_deep_is_reached_and_blamed_precisely() {
        let carried = [
            Header { name: "one.cuh", text: "#include \"two.cuh\"\n" },
            Header { name: "two.cuh", text: "#include \"three.cuh\"\n" },
        ];
        let root = "#include \"one.cuh\"\n";

        let why = reachable("a/unit", root, &carried).expect_err("`three.cuh` is not carried");
        assert!(why.contains("`three.cuh`"), "the include that is missing: {why}");
        assert!(why.contains("from `two.cuh`"), "and the header that reaches it: {why}");
        assert!(why.contains("`a/unit`"), "and the unit that would not compile: {why}");

        let full = [carried[0], carried[1], Header { name: "three.cuh", text: "" }];
        assert_eq!(
            reachable("a/unit", root, &full).expect("all three are carried"),
            ["one.cuh", "two.cuh", "three.cuh"]
        );
    }

    /// Every header the crate carries is self-consistent.
    #[test]
    fn every_device_include_resolves() {
        for header in ALL_HEADERS {
            for included in quoted_includes(header.text) {
                assert!(
                    ALL_HEADERS.iter().any(|h| h.name == included),
                    "`{}` includes `{included}`, which the set does not carry",
                    header.name
                );
            }
        }
    }

    /// The set is what a compile is keyed on, so two different sets must not
    #[test]
    fn the_digest_moves_when_any_header_does() {
        let base = digest(DEVICE_HEADERS);
        assert_eq!(base, digest(DEVICE_HEADERS), "and is stable");

        let edited = [Header { name: DEVICE_HEADERS[0].name, text: "// not what it was" }];
        assert_ne!(base, digest(&edited), "text is in the key");

        let renamed = [Header { name: "norm/somewhere_else.cuh", text: DEVICE_HEADERS[0].text }];
        assert_ne!(base, digest(&renamed), "and so is the name it resolves by");

        let split = [Header { name: "a", text: "bc" }, Header { name: "d", text: "e" }];
        let joined = [Header { name: "ab", text: "c" }, Header { name: "d", text: "e" }];
        assert_ne!(digest(&split), digest(&joined));
    }

    #[test]
    fn only_column_zero_quoted_includes_are_directives() {
        let source = "\
#include \"a.cuh\"
  #include \"indented.cuh\"
#include <cuda_bf16.h>
const char* s = \"#include \\\"in_a_string.cuh\\\"\";
#include\t\"tabbed.cuh\"
";
        assert_eq!(quoted_includes(source), vec!["a.cuh", "tabbed.cuh"]);
    }

    /// The arrays handed to NVRTC are the table, in order and complete.
    #[test]
    fn the_nvrtc_arrays_are_the_table() {
        let (texts, names) = as_nvrtc_arrays(DEVICE_HEADERS).expect("no NULs in a source");
        assert_eq!(texts.len(), DEVICE_HEADERS.len());
        assert_eq!(names.len(), DEVICE_HEADERS.len());
        for (at, header) in DEVICE_HEADERS.iter().enumerate() {
            assert_eq!(names[at].to_str().unwrap(), header.name);
            assert_eq!(texts[at].to_str().unwrap(), header.text);
        }
    }
}
