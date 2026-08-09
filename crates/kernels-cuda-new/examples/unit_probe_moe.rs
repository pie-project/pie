//! Does every `moe` unit compile under NVRTC, and does every row it states
//! resolve to a mangled symbol?
//!
//! # The question this answers
//!
//! A row is three strings — a template path, an element type, and the symbol a
//! trace names — and nothing in the type system ties them to the `.cuh` that
//! has to hold the template. A row can name a kernel that was renamed in the
//! split, a template that takes four arguments where the row supplies one, or
//! a header that stopped compiling under NVRTC the day it grew an
//! `#include <cstdint>`. Every one of those is a clean `cargo build` and a
//! fire that fails at run time, on a machine with a GPU, in a process serving
//! tokens.
//!
//! So this compiles each of `moe`'s units the way `runtime::cache` will,
//! against the header set carried in the binary, with one
//! `nvrtcAddNameExpression` per row — and refuses to be satisfied by anything
//! less than a lowered name for EVERY row and a non-empty cubin.
//!
//! ```text
//! cargo run -p kernels-cuda-new --features cuda-13 --example unit_probe_moe
//! ```
//!
//! # The second table: kernels no row names
//!
//! Twenty-nine of this family's thirty-eight entry points are reachable only
//! through an ahead-of-time launcher, because their geometry fits no rule
//! [`kernels_cuda_new::runtime::launch`] evaluates — a fixed algorithmic block
//! width, a grid over routes or padded blocks rather than output rows, a grid
//! of exactly one block, a two-dimensional block, dynamic shared memory sized
//! from an operand VALUE, or a vectorised twin chosen at run time from a
//! pointer's alignment. Each `.cuh` says which and why beside the kernel, and
//! [`kernels_cuda_new::families::moe`] groups them.
//!
//! NVRTC PARSES those templates as part of the unit that holds them and does
//! not instantiate them, and a template that only fails when instantiated is
//! exactly the failure a parse does not find — the wmma call sites most of
//! all, whose fragment types do not exist until an element type is supplied.
//! So the second table instantiates all twenty-nine anyway, through rows this
//! probe owns and nothing fires — carried here rather than in
//! [`kernels_cuda_new::families::moe`] because a `Unit` with rows in it is a
//! claim that the rows can be LAUNCHED, and these cannot.
//! `LaunchRule::Unstated` is the honest spelling of that: the row has not
//! said.
//!
//! `moe/moe_grouped_gemm` is a unit ONLY here. Its single kernel builds a
//! `dim3(N / 64, max_blocks)` grid over a host-computed block bound, so the
//! family states no unit for it and nothing else in this crate would ever
//! hand the file to NVRTC.
//!
//! # Why this probe carries the wmma call sites
//!
//! `moe_decode_wmma_by_token`, `moe_decode_wmma_by_route` and
//! `moe_grouped_gemm` are the family's three `wmma::fragment` users, and they
//! are the reason it was unmigratable until `csrc/src/pie_mma.cuh` landed.
//! Under nvcc those fragments come from `<mma.h>`; under NVRTC they come from
//! the shim, which implements exactly `16×16×16`, `bf16 × bf16 → f32`, A
//! `row_major`, B `col_major`, store `mem_row_major` and `static_assert`s on
//! anything else. Whether the shim covers this family is therefore not a
//! reading question — it is whether these three instantiate. They are the
//! last three rows of the second table.
//!
//! # One template stays parse-only
//!
//! `moe_decode_gemv<T, ActByToken, kWarps, kUnroll>` is swept over eleven
//! `(kWarps, kUnroll)` rungs by a host macro and takes four template
//! arguments, where [`DeviceKernel::instantiation`] spells exactly one — a
//! type. The `_by_token` and `_by_route` wrappers below pin the two rungs the
//! decode path actually uses; nvcc instantiates all eleven from
//! `moe_dispatch.cu` on every ahead-of-time build, which is the gate that
//! covers the rest.
//!
//! # Why this file carries a `cfg` fence
//!
//! Every example in this crate that touches `cudarc` is declared in
//! `Cargo.toml` with `required-features = ["_cuda"]`, because `cargo test`
//! with no features builds every example and an example naming
//! [`kernels_cuda_new::runtime`] does not exist in a feature-free build. This
//! one has no such entry: it was written by a migration that owns three files
//! and `Cargo.toml` is not one of them, and two agents editing the manifest at
//! once is a merge conflict in the one file that has to parse for anything to
//! build at all. So the fence below is the workaround and the manifest entry
//! is the fix — delete the fence when the entry lands.

#[cfg(not(feature = "_cuda"))]
fn main() {
    eprintln!(
        "unit_probe_moe asks NVRTC to compile things and needs a CUDA backend:\n  \
         cargo run -p kernels-cuda-new --features cuda-13 --example unit_probe_moe"
    );
}

#[cfg(feature = "_cuda")]
fn main() {
    probe::main();
}

#[cfg(feature = "_cuda")]
mod probe {
    use std::time::Duration;

    use kernels::KernelSig;
    use kernels::kernel;
    use kernels::operands;
    use kernels_cuda_new::device::DeviceKernel;
    use kernels_cuda_new::runtime::nvrtc;
    use kernels_cuda_new::source;
    use kernels_cuda_new::unit::Unit;

    /// `moe/dsv4_routing`'s second kernel: one thread per token, a grid of
    /// `ceil(tokens / 256)` blocks over a table the rule cannot see, because
    /// the hash lookup's row count is the TOKEN count and its width is `K`.
    const DSV4_ROUTING_UNROWED: Unit = Unit {
        name: "moe/dsv4_routing",
        root: include_str!("../csrc/src/moe/dsv4_routing.cuh"),
        rows: DSV4_ROUTING_UNROWED_ROWS,
        options: &[],
    };

    /// The eight routing templates `moe/topk_softmax`'s unit leaves out — the
    /// block form at 64 threads, the fused-router form at 64, five warp rungs
    /// at one warp each, and the sigmoid-with-bias form at 64, twice, because
    /// a checkpoint's router logits are bf16 or fp32 and the launcher picks.
    /// A routing block's width is its shared expert slab's stride, not a
    /// tiling choice, and no rule states 64.
    const TOPK_SOFTMAX_UNROWED: Unit = Unit {
        name: "moe/topk_softmax",
        root: include_str!("../csrc/src/moe/topk_softmax.cuh"),
        rows: TOPK_SOFTMAX_UNROWED_ROWS,
        options: &[],
    };

    /// The eighteen dispatch templates `moe/moe_dispatch`'s unit leaves out:
    /// the pointer builders that launch one block, the two counting sorts that
    /// size shared memory from `num_experts`, the vectorised twins a run-time
    /// alignment test picks, the two reorders whose grid rows are the INPUT's,
    /// and the four decode GEMM entry points whose grid is a route count.
    const MOE_DISPATCH_UNROWED: Unit = Unit {
        name: "moe/moe_dispatch",
        root: include_str!("../csrc/src/moe/moe_dispatch.cuh"),
        rows: MOE_DISPATCH_UNROWED_ROWS,
        options: &[],
    };

    /// The whole of `moe/moe_grouped_gemm`, which is a unit nowhere else.
    ///
    /// Its one kernel launches `dim3(N / kNTile, max_blocks)` — a second grid
    /// axis counting the padded expert blocks a host-side align pass produced
    /// — so the family states no unit for it, and without this row NVRTC would
    /// never see the file at all.
    const MOE_GROUPED_GEMM_UNROWED: Unit = Unit {
        name: "moe/moe_grouped_gemm",
        root: include_str!("../csrc/src/moe/moe_grouped_gemm.cuh"),
        rows: MOE_GROUPED_GEMM_UNROWED_ROWS,
        options: &[],
    };

    static DSV4_ROUTING_UNROWED_ROWS: &[DeviceKernel] = &[DeviceKernel {
        sig: &UNROWED_SIGS[0],
        template_path: "moe::device::hash_route_lookup",
        elem: "device::bf16",
    }];

    static TOPK_SOFTMAX_UNROWED_ROWS: &[DeviceKernel] = &[
        DeviceKernel {
            sig: &UNROWED_SIGS[1],
            template_path: "moe::device::topk_softmax",
            elem: "device::bf16",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[2],
            template_path: "moe::device::router_topk_softmax",
            elem: "device::bf16",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[3],
            template_path: "moe::device::topk_softmax_warp_x1",
            elem: "device::bf16",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[4],
            template_path: "moe::device::topk_softmax_warp_x2",
            elem: "device::bf16",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[5],
            template_path: "moe::device::topk_softmax_warp_x4",
            elem: "device::bf16",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[6],
            template_path: "moe::device::topk_softmax_warp_x8",
            elem: "device::bf16",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[7],
            template_path: "moe::device::topk_softmax_warp_x16",
            elem: "device::bf16",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[8],
            template_path: "moe::device::topk_sigmoid_bias",
            elem: "device::bf16",
        },
        // The one place this family spells an element type the prelude does not
        // name. `device::Elem` has no `float` specialisation and there is no
        // `device::f32`, so `topk_softmax.cuh` declares both in `moe::device`
        // for its own `Logit<T>` -- and the row has to reach them by the path
        // they actually live at. The prelude naming them would delete this.
        DeviceKernel {
            sig: &UNROWED_SIGS[9],
            template_path: "moe::device::topk_sigmoid_bias",
            elem: "moe::device::f32",
        },
    ];

    static MOE_DISPATCH_UNROWED_ROWS: &[DeviceKernel] = &[
        DeviceKernel {
            sig: &UNROWED_SIGS[10],
            template_path: "moe::device::scatter_add_weighted",
            elem: "device::bf16",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[11],
            template_path: "moe::device::batched_weighted_sum",
            elem: "device::bf16",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[12],
            template_path: "moe::device::token_batched_weighted_sum_vec",
            elem: "device::bf16",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[13],
            template_path: "moe::device::token_batched_weighted_sum_add_vec",
            elem: "device::bf16",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[14],
            template_path: "moe::device::build_dual_gemm_ptrs",
            elem: "device::bf16",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[15],
            template_path: "moe::device::build_moe_ptrs_decode",
            elem: "device::bf16",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[16],
            template_path: "moe::device::build_moe_ptrs_decode_batched",
            elem: "device::bf16",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[17],
            template_path: "moe::device::build_moe_ptrs_aligned",
            elem: "device::bf16",
        },
        // The two counting sorts, and the only rows in this family whose
        // element type is an INDEX type: both are templated over the index
        // width and instantiated at `i32`, because what they sort is a route
        // table and not an activation.
        DeviceKernel {
            sig: &UNROWED_SIGS[18],
            template_path: "moe::device::moe_align_decode",
            elem: "device::i32",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[19],
            template_path: "moe::device::moe_bucket_exact",
            elem: "device::i32",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[20],
            template_path: "moe::device::gather_moe_aligned_inputs_vec",
            elem: "device::bf16",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[21],
            template_path: "moe::device::reorder_moe_aligned_output",
            elem: "device::bf16",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[22],
            template_path: "moe::device::reorder_moe_aligned_output_vec",
            elem: "device::bf16",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[23],
            template_path: "moe::device::transpose_expert_scales",
            elem: "device::u8",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[24],
            template_path: "moe::device::moe_decode_gemv_by_token",
            elem: "device::bf16",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[25],
            template_path: "moe::device::moe_decode_gemv_by_route",
            elem: "device::bf16",
        },
        // The wmma call sites. `pie_mma.cuh`'s fragments do not exist until an
        // element type is supplied, so these two rows are what turns "the shim
        // is the right shape" from a reading into a measurement.
        DeviceKernel {
            sig: &UNROWED_SIGS[26],
            template_path: "moe::device::moe_decode_wmma_by_token",
            elem: "device::bf16",
        },
        DeviceKernel {
            sig: &UNROWED_SIGS[27],
            template_path: "moe::device::moe_decode_wmma_by_route",
            elem: "device::bf16",
        },
    ];

    static MOE_GROUPED_GEMM_UNROWED_ROWS: &[DeviceKernel] = &[DeviceKernel {
        sig: &UNROWED_SIGS[28],
        template_path: "moe::device::moe_grouped_gemm",
        elem: "device::bf16",
    }];

    /// The contracts these probe rows are written against.
    ///
    /// Operands in the kernels' order and `LaunchRule::Unstated` throughout:
    /// what is being asked is whether NVRTC can instantiate the template, and
    /// an operand list is what a `DeviceKernel` needs to exist, not what this
    /// probe checks. Symbols mirror `kernels_cuda::moe`'s where that family
    /// declares the same kernel and are descriptive where it does not — no
    /// dispatcher reads them, because a row here is never launched.
    #[rustfmt::skip]
    static UNROWED_SIGS: [KernelSig; 29] = [
        kernel!(hash_route_lookup "moe::hash_route_lookup",
            file = Some("moe/dsv4_routing.cuh"),
            operands = operands![token_ids: I32s, tid2eid: I64s, logits: Buf,
                                 topk_idx: I32sMut, topk_w: F32sMut,
                                 tokens: I32, vocab_size: I32, e: I32, k: I32,
                                 renormalize: Bool, routed_scaling_factor: F32]),
        kernel!(topk_softmax "moe::topk_softmax_bf16",
            file = Some("moe/topk_softmax.cuh"),
            operands = operands![logits: Buf, act: Buf, bias: Buf,
                                 topk_idx: I32sMut, topk_w: F32sMut,
                                 num_experts: I32, k: I32, hidden: I32]),
        kernel!(router_topk_softmax "moe::router_topk_softmax_bf16",
            file = Some("moe/topk_softmax.cuh"),
            operands = operands![router_weight: Buf, act: Buf, bias: Buf,
                                 topk_idx: I32sMut, topk_w: F32sMut,
                                 num_experts: I32, k: I32, hidden: I32]),
        kernel!(topk_softmax_warp_x1 "moe::topk_softmax_warp_x1_bf16",
            file = Some("moe/topk_softmax.cuh"),
            operands = operands![logits: Buf, topk_idx: I32sMut, topk_w: F32sMut,
                                 num_experts: I32, k: I32]),
        kernel!(topk_softmax_warp_x2 "moe::topk_softmax_warp_x2_bf16",
            file = Some("moe/topk_softmax.cuh"),
            operands = operands![logits: Buf, topk_idx: I32sMut, topk_w: F32sMut,
                                 num_experts: I32, k: I32]),
        kernel!(topk_softmax_warp_x4 "moe::topk_softmax_warp_x4_bf16",
            file = Some("moe/topk_softmax.cuh"),
            operands = operands![logits: Buf, topk_idx: I32sMut, topk_w: F32sMut,
                                 num_experts: I32, k: I32]),
        kernel!(topk_softmax_warp_x8 "moe::topk_softmax_warp_x8_bf16",
            file = Some("moe/topk_softmax.cuh"),
            operands = operands![logits: Buf, topk_idx: I32sMut, topk_w: F32sMut,
                                 num_experts: I32, k: I32]),
        kernel!(topk_softmax_warp_x16 "moe::topk_softmax_warp_x16_bf16",
            file = Some("moe/topk_softmax.cuh"),
            operands = operands![logits: Buf, topk_idx: I32sMut, topk_w: F32sMut,
                                 num_experts: I32, k: I32]),
        kernel!(topk_sigmoid_bias_bf16 "moe::topk_sigmoid_bias_bf16",
            file = Some("moe/topk_softmax.cuh"),
            operands = operands![logits: Buf, correction_bias: F32s,
                                 topk_idx: I32sMut, topk_w: F32sMut,
                                 num_experts: I32, k: I32, normalize: I32,
                                 routed_scaling_factor: F32]),
        kernel!(topk_sigmoid_bias_fp32 "moe::topk_sigmoid_bias_fp32",
            file = Some("moe/topk_softmax.cuh"),
            operands = operands![logits: F32s, correction_bias: F32s,
                                 topk_idx: I32sMut, topk_w: F32sMut,
                                 num_experts: I32, k: I32, normalize: I32,
                                 routed_scaling_factor: F32]),
        kernel!(scatter_add_weighted "moe::scatter_add_weighted_bf16",
            file = Some("moe/moe_dispatch.cuh"),
            operands = operands![out: BufMut, src: Buf, dst_idx: I32s,
                                 row_weights: F32s, hidden: I32]),
        kernel!(batched_weighted_sum "moe::batched_weighted_sum_bf16",
            file = Some("moe/moe_dispatch.cuh"),
            operands = operands![out: BufMut, src: Buf, weights: F32s,
                                 batch: I32, hidden: I32]),
        kernel!(token_batched_weighted_sum_vec "moe::token_batched_weighted_sum_vec_bf16",
            file = Some("moe/moe_dispatch.cuh"),
            operands = operands![out: BufMut, src: Buf, weights: F32s,
                                 top_k: I32, hidden_vec: I32]),
        kernel!(token_batched_weighted_sum_add_vec "moe::token_batched_weighted_sum_add_vec_bf16",
            file = Some("moe/moe_dispatch.cuh"),
            operands = operands![out: BufMut, src: Buf, weights: F32s,
                                 top_k: I32, hidden_vec: I32]),
        kernel!(build_dual_gemm_ptrs "moe::build_dual_gemm_ptrs_bf16",
            file = Some("moe/moe_dispatch.cuh"),
            operands = operands![act: Buf, w0: Buf, w1: Buf,
                                 out0: BufMut, out1: BufMut,
                                 act_ptrs: BufArrayOut, w_ptrs: BufArrayOut,
                                 out_ptrs: BufArrayOutMut]),
        kernel!(build_moe_ptrs_decode "moe::build_moe_ptrs_decode_bf16",
            file = Some("moe/moe_dispatch.cuh"),
            operands = operands![topk_idx: I32s, topk_w: F32s,
                                 gate_up_base: Buf, down_base: Buf, norm_x: Buf,
                                 expert_gate_up: BufMut, expert_act: BufMut,
                                 expert_out: BufMut,
                                 a_gu_ptrs: BufArrayOut, b_gu_ptrs: BufArrayOut,
                                 c_gu_ptrs: BufArrayOutMut,
                                 a_dn_ptrs: BufArrayOut, b_dn_ptrs: BufArrayOut,
                                 c_dn_ptrs: BufArrayOutMut, weights_out: F32sMut,
                                 top_k: I32, h: I32, i_moe: I32]),
        kernel!(build_moe_ptrs_decode_batched "moe::build_moe_ptrs_decode_batched_bf16",
            file = Some("moe/moe_dispatch.cuh"),
            operands = operands![topk_idx: I32s, topk_w: F32s,
                                 gate_up_base: Buf, down_base: Buf, norm_x: Buf,
                                 expert_gate_up: BufMut, expert_act: BufMut,
                                 expert_out: BufMut,
                                 a_gu_ptrs: BufArrayOut, b_gu_ptrs: BufArrayOut,
                                 c_gu_ptrs: BufArrayOutMut,
                                 a_dn_ptrs: BufArrayOut, b_dn_ptrs: BufArrayOut,
                                 c_dn_ptrs: BufArrayOutMut, weights_out: F32sMut,
                                 num_tokens: I32, top_k: I32, h: I32, i_moe: I32]),
        kernel!(build_moe_ptrs_aligned "moe::build_moe_ptrs_aligned_bf16",
            file = Some("moe/moe_dispatch.cuh"),
            operands = operands![expert_ids: I32s, gate_up_base: Buf, down_base: Buf,
                                 aligned_in: Buf, aligned_gate_up: BufMut,
                                 aligned_act: BufMut, aligned_out: BufMut,
                                 a_gu_ptrs: BufArrayOut, b_gu_ptrs: BufArrayOut,
                                 c_gu_ptrs: BufArrayOutMut,
                                 a_dn_ptrs: BufArrayOut, b_dn_ptrs: BufArrayOut,
                                 c_dn_ptrs: BufArrayOutMut,
                                 max_blocks: I32, block_size: I32, h: I32,
                                 i_moe: I32, routed_blocks: I32,
                                 shared_gate_up_base: Buf, shared_down_base: Buf]),
        kernel!(moe_align_decode "moe::moe_align_decode",
            file = Some("moe/moe_dispatch.cuh"),
            operands = operands![topk_idx: I32s, sorted_route_ids: I32sMut,
                                 expert_ids: I32sMut, route_to_aligned_row: I32sMut,
                                 num_routes: I32, num_experts: I32, block_size: I32,
                                 max_blocks: I32, num_tokens_past_padded: I32sMut]),
        kernel!(moe_bucket_exact "moe::moe_bucket_exact",
            file = Some("moe/moe_dispatch.cuh"),
            operands = operands![topk_idx: I32s, sorted_route_ids: I32sMut,
                                 route_to_sorted_row: I32sMut, counts_out: I32sMut,
                                 num_routes: I32, num_experts: I32]),
        kernel!(gather_moe_aligned_inputs_vec "moe::gather_moe_aligned_inputs_vec_bf16",
            file = Some("moe/moe_dispatch.cuh"),
            operands = operands![norm_x: Buf, sorted_route_ids: I32s,
                                 aligned_in: BufMut, num_routes: I32,
                                 aligned_rows: I32, top_k: I32, hidden_vec: I32,
                                 shared_row_begin: I32, num_tokens: I32]),
        kernel!(reorder_moe_aligned_output "moe::reorder_moe_aligned_output_bf16",
            file = Some("moe/moe_dispatch.cuh"),
            operands = operands![aligned_out: Buf, sorted_route_ids: I32s,
                                 route_out: BufMut, num_routes: I32,
                                 aligned_rows: I32, hidden: I32,
                                 shared_row_begin: I32, num_tokens: I32,
                                 shared_out: BufMut]),
        kernel!(reorder_moe_aligned_output_vec "moe::reorder_moe_aligned_output_vec_bf16",
            file = Some("moe/moe_dispatch.cuh"),
            operands = operands![aligned_out: Buf, sorted_route_ids: I32s,
                                 route_out: BufMut, num_routes: I32,
                                 aligned_rows: I32, hidden_vec: I32,
                                 shared_row_begin: I32, num_tokens: I32,
                                 shared_out: BufMut]),
        kernel!(transpose_expert_scales "moe::transpose_expert_scales_u8",
            file = Some("moe/moe_dispatch.cuh"),
            operands = operands![src: U8s, dst: U8sMut, n: I32, kg: I32]),
        kernel!(moe_decode_gemv_by_token "moe::moe_decode_gemv_by_token_bf16",
            file = Some("moe/moe_dispatch.cuh"),
            operands = operands![topk_idx: I32s, act: Buf, weight_base: Buf,
                                 out: BufMut, top_k: I32, k: I32, n: I32,
                                 expert_stride: I64]),
        kernel!(moe_decode_gemv_by_route "moe::moe_decode_gemv_by_route_bf16",
            file = Some("moe/moe_dispatch.cuh"),
            operands = operands![topk_idx: I32s, act: Buf, weight_base: Buf,
                                 out: BufMut, top_k: I32, k: I32, n: I32,
                                 expert_stride: I64]),
        kernel!(moe_decode_wmma_by_token "moe::moe_decode_wmma_by_token_bf16",
            file = Some("moe/moe_dispatch.cuh"),
            operands = operands![topk_idx: I32s, act: Buf, weight_base: Buf,
                                 out: BufMut, top_k: I32, k: I32, n: I32,
                                 expert_stride: I64]),
        kernel!(moe_decode_wmma_by_route "moe::moe_decode_wmma_by_route_bf16",
            file = Some("moe/moe_dispatch.cuh"),
            operands = operands![topk_idx: I32s, act: Buf, weight_base: Buf,
                                 out: BufMut, top_k: I32, k: I32, n: I32,
                                 expert_stride: I64]),
        kernel!(moe_grouped_gemm "moe::moe_grouped_gemm_bf16",
            file = Some("moe/moe_grouped_gemm.cuh"),
            operands = operands![a: Buf, weight_base: Buf, c: BufMut,
                                 expert_ids: I32s, n: I32, k: I32]),
    ];

    /// What one unit's compile came to.
    struct Report {
        unit: &'static str,
        rows: usize,
        lowered: usize,
        millis: f64,
        cubin: usize,
        verdict: Result<(), String>,
    }

    pub fn main() {
        let arch = kernels_cuda_new::runtime::cache::arch().unwrap_or("sm_89");
        println!("NVRTC version: {}", version());
        println!("architecture:  {arch}");
        println!("header set:    {} headers carried in the binary", source::DEVICE_HEADERS.len());

        println!("\nthe units, and the rows they state:\n");
        let mut stated: Vec<Report> = Vec::new();
        for unit in kernels_cuda_new::families::moe::UNITS {
            stated.push(probe(unit, arch));
        }
        table(&stated);

        println!("\nkernels no row names, instantiated anyway:\n");
        let unrowed: Vec<Report> = [
            &DSV4_ROUTING_UNROWED,
            &TOPK_SOFTMAX_UNROWED,
            &MOE_DISPATCH_UNROWED,
            &MOE_GROUPED_GEMM_UNROWED,
        ]
        .into_iter()
        .map(|unit| probe(unit, arch))
        .collect();
        table(&unrowed);

        let all: Vec<&Report> = stated.iter().chain(unrowed.iter()).collect();
        let failed: Vec<&&Report> = all.iter().filter(|row| row.verdict.is_err()).collect();
        println!();
        if failed.is_empty() {
            let rows: usize = all.iter().map(|row| row.rows).sum();
            let bytes: usize = all.iter().map(|row| row.cubin).sum();
            let millis: f64 = all.iter().map(|row| row.millis).sum();
            println!(
                "{} compiles, {rows} instantiations, {rows} lowered names, {bytes} bytes of\n\
                 cubin, {millis:.0} ms. Every template a row names exists in the unit its `file`\n\
                 claims and instantiates at the element type the row states — including the\n\
                 three `wmma::fragment` call sites, which is `pie_mma.cuh` covering this family\n\
                 measured rather than read.",
                all.len()
            );
        } else {
            for row in &failed {
                println!("{}: {}", row.unit, row.verdict.as_ref().unwrap_err());
            }
            std::process::exit(1);
        }
    }

    fn table(rows: &[Report]) {
        println!("  {:<22} {:>4} {:>8} {:>9} {:>11}", "unit", "rows", "lowered", "ms", "cubin");
        for row in rows {
            let mark = if row.verdict.is_ok() { "OK" } else { "FAILED" };
            println!(
                "  {:<22} {:>4} {:>8} {:>9.1} {:>11}  {mark}",
                row.unit, row.rows, row.lowered, row.millis, row.cubin
            );
        }
    }

    /// Compile one unit and check the two things a fire depends on: a lowered
    /// name per row, and an image to load.
    fn probe(unit: &Unit, arch: &str) -> Report {
        match nvrtc::compile(unit, arch) {
            Ok(compiled) => {
                let mut verdict = Ok(());
                if compiled.lowered.len() != unit.rows.len() {
                    verdict = Err(format!(
                        "{} rows, {} lowered names",
                        unit.rows.len(),
                        compiled.lowered.len()
                    ));
                } else if let Some((symbol, _)) =
                    compiled.lowered.iter().find(|(_, mangled)| mangled.is_empty())
                {
                    verdict = Err(format!("`{symbol}` lowered to the empty string"));
                } else if compiled.cubin.is_empty() {
                    verdict = Err("the compile succeeded and produced no cubin".into());
                }
                if !compiled.log.trim().is_empty() {
                    println!("  {} said:\n{}", unit.name, compiled.log.trim());
                }
                Report {
                    unit: unit.name,
                    rows: unit.rows.len(),
                    lowered: compiled.lowered.len(),
                    millis: duration_ms(compiled.elapsed),
                    cubin: compiled.cubin.len(),
                    verdict,
                }
            }
            Err(why) => Report {
                unit: unit.name,
                rows: unit.rows.len(),
                lowered: 0,
                millis: 0.0,
                cubin: 0,
                verdict: Err(why.to_string()),
            },
        }
    }

    fn duration_ms(elapsed: Duration) -> f64 {
        elapsed.as_secs_f64() * 1e3
    }

    /// `libnvrtc`'s own version, so a compile that behaves differently on another
    /// machine can be told apart from one that behaves differently on this one.
    fn version() -> String {
        use cudarc::nvrtc::sys as nv;
        let (mut major, mut minor) = (0, 0);
        // SAFETY: both are live out-parameters for the call's duration.
        let code = unsafe { nv::nvrtcVersion(&raw mut major, &raw mut minor) };
        if code == nv::nvrtcResult::NVRTC_SUCCESS {
            format!("{major}.{minor}")
        } else {
            format!("unavailable ({code:?})")
        }
    }
}
