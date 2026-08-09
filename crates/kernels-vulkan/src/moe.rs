//! Routing, and every projection that selects an expert.
//!
//! Filed by what the kernel DOES rather than by the file it sits in:
//! `affine_qmm_t_routed` lives in `moe/qmm_t_routed.comp` beside its dense
//! twin, but a routed matmul reads an expert slot and is only reachable from
//! a mixture. This is the caller-set rule `.wiki/kernel-refactor.md` §7 uses
//! to settle the same question on the CUDA side.
//!
//! Declaring the axes is what surfaced the one real coverage gap here, and
//! then closed it: `qmv_routed` was compiled for ONE affine format where the
//! dense `qmv_fast` had six, so a Qwen3-MoE or routed gemma-4 at any other
//! format had no pipeline at all. The five missing instantiations are in
//! `moe/qmv_routed.comp` now, with the evidence for widening rather than
//! refusing. `.wiki/kernel-metal-refactor.md` §9 records it.

use kernels::{KernelSig, kernel};

use crate::axes::*;

pub static KERNELS: &[KernelSig] = &[
    kernel!(combine_sorted "combine_sorted", file = Some("moe/route.comp"),
    launch = kernels::LaunchRule::RouteRows,
    operands = kernels::operands![
        y: Buf <- kernels::Source::In(0),
        expert_weights: Buf <- kernels::Source::In(1),
        out: BufMut <- kernels::Source::Out(0),
        // `ExpertCombineParams` -- a POINTER where the scalars are, so the
        // slot is packed rather than one buffer per number.
        params: Buf <- kernels::Source::Param(0),
        inv: Buf <- kernels::Source::In(2),
    ]),
    kernel!(route_gather "route_gather", file = Some("moe/route.comp"),
    launch = kernels::LaunchRule::RouteRows,
    operands = kernels::operands![
        x: Buf <- kernels::Source::In(0),
        out: BufMut <- kernels::Source::Out(0),
        perm: Buf <- kernels::Source::In(1),
        params: Buf <- kernels::Source::Param(0),
    ]),
    // FIVE outputs, and that is the shape of the thing: a sort states the
    // permutation, the per-row expert, the per-tile expert, and the inverse
    // the combine reads back. A text that named fewer would leave the combine
    // reading whatever was in the buffer.
    kernel!(route_sort "route_sort", file = Some("moe/route.comp"),
    // ONE workgroup, whatever the row count: the sort reduces across every
    // (row, slot) pair through workgroup-scoped atomics and stripes them over its
    // own lanes. `RouterLane` -- which this shared until the row axis landed
    // on it -- would launch one copy per row, each clearing and rewriting the
    // permutation the others are reading.
    launch = kernels::LaunchRule::RouterSort,
    operands = kernels::operands![
        expert_ids: Buf <- kernels::Source::In(0),
        perm: BufMut <- kernels::Source::Out(0),
        row_expert: BufMut <- kernels::Source::Out(1),
        tile_expert: BufMut <- kernels::Source::Out(2),
        params: Buf <- kernels::Source::Param(0),
        inv: BufMut <- kernels::Source::Out(3),
    ]),
    // 9 in moe/qmm_t_routed.comp
    kernel!(mxfp4_qmm_t_routed_bias "mxfp4_qmm_t_routed_bias", axes = &[BF16, TILE_M, TILE_N]),
    // 1 in moe/qmv_routed.comp
    //
    // This row used to name no operands, which made it the one unstated row in
    // the table that is provably REACHABLE. `model-compiler`'s routed-QMV site
    // picks the symbol with a `match` on the weight repr --
    // `WeightRepr::Mxfp4Marlin => "mxfp4_qmv_routed_bias"` against
    // `affine_qmv_routed{_bias}` for everything else -- and then makes ONE
    // `with_params` call for both arms. So a driver does try to bind this, and
    // an operand list it cannot read is a failure at launch rather than dead
    // code.
    //
    // The list below is not invented to fill the hole. `qmv.metal` generates
    // this symbol from `instantiate_gptoss_qmv` with `fn = qmv_routed_bias` --
    // the SAME macro and the SAME template function as
    // `affine_qmv_routed_bias`, differing only in the codec and the
    // group/bits point, neither of which appears in the signature. The twelve
    // parameters are therefore identical operand for operand, and this row is
    // `qmv_routed_bias`'s copied across rather than reconstructed.
    //
    // It agrees with the shader too: dense buffer numbering puts `biases` at 2,
    // and `moe/qmv_routed.comp` declares 0/1/3/4/5/6 under `PIE_MXFP4`,
    // omitting exactly that one because the MXFP4 codec has no separate bias
    // plane. Metal takes the pointer and ignores it for the same reason, so the
    // slot stays in the ABI and stays unread. `--bindings` checks this.
    kernel!(mxfp4_qmv_routed_bias "mxfp4_qmv_routed_bias",
    file = Some("moe/qmv_routed.comp"),
    launch = kernels::LaunchRule::RoutedQmv,
    operands = kernels::operands![
        w: Buf <- kernels::Source::Weight(0),
        scales: Buf <- kernels::Source::Weight(1),
        biases: Buf <- kernels::Source::Weight(2),
        x: Buf <- kernels::Source::In(0),
        y: BufMut <- kernels::Source::Out(0),
        in_vec_size: I32 <- kernels::Source::Param(0),
        out_vec_size: I32 <- kernels::Source::Param(1),
        bias: Buf <- kernels::Source::Weight(3),
        expert_ids: Buf <- kernels::Source::In(1),
        x_slot_stride: I32 <- kernels::Source::Param(2),
        x_row_stride: I32 <- kernels::Source::Param(3),
        slots_per_row: I32 <- kernels::Source::Param(4),
    ],
    axes = &[BF16, GROUP_32, BITS_4]),
    // 54 in moe/qmm_t_routed.comp
    kernel!(qmm_t_routed "affine_qmm_t_routed", axes = &[BF16, GROUP, BITS, TILE_M, TILE_N]),
    // 9 in moe/qmm_t_routed.comp
    kernel!(qmm_t_routed_fp16 "affine_qmm_t_routed_fp16",
        axes = &[BF16, GROUP_64, BITS_4, TILE_M, TILE_N]),
    // 1 in moe/qmv_routed.comp
    // ONE affine format, and that is the kernel's design rather than a gap:
    // `AffineQ::group_size` is a constant, so a second group point would name
    // an instantiation that dequantises at 64 whatever it claims. A routed
    // checkpoint at another group is meant to fail by name when its pipeline
    // is built -- which `entrypoint()` now does at the call instead of in the
    // SPIR-V module lookup.
    kernel!(qmv_routed "affine_qmv_routed", file = Some("moe/qmv_routed.comp"),
    launch = kernels::LaunchRule::RoutedQmv,
    operands = kernels::operands![
        w: Buf <- kernels::Source::Weight(0),
        scales: Buf <- kernels::Source::Weight(1),
        biases: Buf <- kernels::Source::Weight(2),
        x: Buf <- kernels::Source::In(0),
        y: BufMut <- kernels::Source::Out(0),
        in_vec_size: I32 <- kernels::Source::Param(0),
        out_vec_size: I32 <- kernels::Source::Param(1),
        // The unbiased variant; `affine_qmv_routed_bias` is the symbol that
        // reads it.
        bias: Buf,
        // What makes it routed: the slot the row's expert lives in.
        expert_ids: Buf <- kernels::Source::In(1),
        x_slot_stride: I32 <- kernels::Source::Param(2),
        x_row_stride: I32 <- kernels::Source::Param(3),
        slots_per_row: I32 <- kernels::Source::Param(4),
    ],
    axes = &[BF16, GROUP_64, BITS_4]),
    // 1 in moe/qmv_routed.comp
    kernel!(qmv_routed_bias "affine_qmv_routed_bias", file = Some("moe/qmv_routed.comp"),
    launch = kernels::LaunchRule::RoutedQmv,
    operands = kernels::operands![
        w: Buf <- kernels::Source::Weight(0),
        scales: Buf <- kernels::Source::Weight(1),
        biases: Buf <- kernels::Source::Weight(2),
        x: Buf <- kernels::Source::In(0),
        y: BufMut <- kernels::Source::Out(0),
        in_vec_size: I32 <- kernels::Source::Param(0),
        out_vec_size: I32 <- kernels::Source::Param(1),
        bias: Buf <- kernels::Source::Weight(3),
        expert_ids: Buf <- kernels::Source::In(1),
        x_slot_stride: I32 <- kernels::Source::Param(2),
        x_row_stride: I32 <- kernels::Source::Param(3),
        slots_per_row: I32 <- kernels::Source::Param(4),
    ],
    axes = &[BF16, GROUP_64, BITS_4]),
    // 1 in moe/route.comp
    kernel!(router_topk "router_topk", file = Some("moe/route.comp"),
    launch = kernels::LaunchRule::RouterLane,
    operands = kernels::operands![
        logits: Buf <- kernels::Source::In(0),
        expert_ids: BufMut <- kernels::Source::Out(0),
        expert_weights: BufMut <- kernels::Source::Out(1),
        params: Buf <- kernels::Source::Param(0),
        // The unscaled variant reads it and does nothing with it; the slot is
        // positional so it is listed, and `router_topk_scaled` is the symbol
        // that means it.
        per_expert_scale: Buf,
    ],
    axes = &[BF16]),
    // 1 in moe/route.comp
    kernel!(router_topk_scaled "router_topk_scaled", file = Some("moe/route.comp"),
    launch = kernels::LaunchRule::RouterLane,
    operands = kernels::operands![
        logits: Buf <- kernels::Source::In(0),
        expert_ids: BufMut <- kernels::Source::Out(0),
        expert_weights: BufMut <- kernels::Source::Out(1),
        params: Buf <- kernels::Source::Param(0),
        per_expert_scale: Buf <- kernels::Source::In(1),
    ],
    axes = &[BF16]),
    kernel!(shared_expert_combine "shared_expert_combine", file = Some("moe/route.comp"),
    launch = kernels::LaunchRule::RouteRows,
    operands = kernels::operands![
        routed: Buf <- kernels::Source::In(0),
        shared: Buf <- kernels::Source::In(1),
        gate: Buf <- kernels::Source::In(2),
        // May alias `routed`, which the driver does not need to know: an
        // alias is two names for one address and the binding is by address.
        out: BufMut <- kernels::Source::Out(0),
        width: U32 <- kernels::Source::Param(0),
    ]),
    kernel!(shared_expert_combine_strided "shared_expert_combine_strided",
    file = Some("moe/route.comp"),
    launch = kernels::LaunchRule::RouteRows,
    operands = kernels::operands![
        routed: Buf <- kernels::Source::In(0),
        shared: Buf <- kernels::Source::In(1),
        gate: Buf <- kernels::Source::In(2),
        out: BufMut <- kernels::Source::Out(0),
        width: U32 <- kernels::Source::Param(0),
        row_pitch: I32 <- kernels::Source::Param(1),
    ]),
];
