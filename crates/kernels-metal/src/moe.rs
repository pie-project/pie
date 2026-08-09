//! Routing, and every projection that selects an expert.
//!
//! Filed by what the kernel DOES rather than by the file it sits in:
//! `affine_qmm_t_routed` lives in `quantized_qmm_t.metal` beside its dense
//! twin, but a routed matmul reads an expert slot and is only reachable from
//! a mixture. This is the caller-set rule `.wiki/kernel-refactor.md` §7 uses
//! to settle the same question on the CUDA side.
//!
//! Declaring the axes is what surfaced the one real coverage gap here, and
//! then closed it: `qmv_routed` was compiled for ONE affine format where the
//! dense `qmv_fast` had six, so a Qwen3-MoE or routed gemma-4 at any other
//! format had no pipeline at all. The five missing instantiations are in
//! `quantized_qmv.metal` now, with the evidence for widening rather than
//! refusing. `.wiki/kernel-metal-refactor.md` §9 records it.

use kernels::{KernelSig, kernel};

use crate::axes::*;

pub static KERNELS: &[KernelSig] = &[
    kernel!(combine_sorted "combine_sorted", file = Some("moe/route.metal"),
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
    kernel!(route_gather "route_gather", file = Some("moe/route.metal"),
    launch = kernels::LaunchRule::RouteRows,
    // This statement's rows are the SORTED STACK, not the fire's tokens --
    // `MoeRouteParams::padded`, the fifth word. It shares `RouteRows` with
    // `combine_sorted`, whose rows ARE the fire's, which is why the extent is
    // stated on the row and not decided by the rule. Given the fire's count
    // the gather ran over a quarter of its own output at `top_k = 4` and left
    // the rest whatever the arena held.
    rows_param = Some(4),
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
    kernel!(route_sort "route_sort", file = Some("moe/route.metal"),
    // ONE threadgroup, whatever the row count: the sort reduces across every
    // (row, slot) pair through threadgroup atomics and stripes them over its
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
    // 9 in quantized_qmm_t.metal
    kernel!(mxfp4_qmm_t_routed_bias "mxfp4_qmm_t_routed_bias", axes = &[BF16, TILE_M, TILE_N]),
    // 1 in quantized_qmv.metal
    //
    // This row named no operands, which made it the one unstated row in the
    // table that is provably REACHABLE. `model-compiler`'s routed-QMV site
    // picks the symbol with a `match` on the weight repr --
    // `WeightRepr::Mxfp4Marlin => "mxfp4_qmv_routed_bias"` against
    // `affine_qmv_routed{_bias}` for everything else -- and then makes ONE
    // `with_params` call for both arms. So a driver does try to bind this, and
    // an operand list it cannot read is a failure at launch rather than dead
    // code.
    //
    // Found from the Vulkan side, by intersecting the operand-less rows with
    // every symbol literal in `model-compiler`: of the 57, exactly this one
    // survived. `kernels-vulkan` states it identically.
    //
    // The list below is not invented to fill the hole. `qmv.metal` generates
    // this symbol from `instantiate_gptoss_qmv` with `fn = qmv_routed_bias` --
    // the SAME macro and the SAME template function as `qmv_routed_bias`
    // directly above, differing only in the codec and the group/bits point,
    // neither of which appears in the signature. The twelve parameters are
    // therefore identical operand for operand, and this is that row's list
    // copied across rather than reconstructed.
    //
    // `biases` stays in the ABI and stays unread: the MXFP4 codec has no
    // separate bias plane, so the kernel takes the pointer and ignores it. A
    // row is positional, so dropping the slot would shift everything after it.
    //
    // Which is why it is `Buf` with NO source, the same way `qmv_routed`'s
    // unread `bias` is. Copying the affine row wholesale gave it
    // `Weight(2)` -- and a slot nothing reads then ate the index the slot
    // the kernel DOES read needed, pushing `bias` to `Weight(3)`.
    //
    // `Weight(3)` is unreachable for this codec by counting:
    // `MatW::scale_names` yields `.scales` alone for `Mxfp4Marlin`, so the
    // list is two names long before `routed_qmv` appends `.bias` and three
    // after. The affine row's indices are right for THREE codec names
    // (`w`, `.scales`, `.zeros`), and that difference is the only thing the
    // two rows may disagree about.
    //
    // Measured on `mlx-community/gpt-oss-20b-MXFP4-Q4`: the expert bank
    // publishes `weight`, `scales` and `bias` and no `biases` at all, and
    // the bias is `[32, 2880]` -- one value per output row, where a zero
    // point would be `[32, 2880, 90]` per group beside the scales.
    kernel!(mxfp4_qmv_routed_bias "mxfp4_qmv_routed_bias",
    file = Some("quant/qmv.metal"),
    launch = kernels::LaunchRule::RoutedQmv,
    // The matvec's row axis is `out_vec_size`, the second word, and not the
    // output rectangle's width: a routed projection writes a whole token's
    // `k` results end to end, so the value is `k` times as wide as one
    // result. See `dsl::metal::routed_qmv`.
    grid_param = Some(1),
    operands = kernels::operands![
        w: Buf <- kernels::Source::Weight(0),
        scales: Buf <- kernels::Source::Weight(1),
        biases: Buf,
        x: Buf <- kernels::Source::In(0),
        y: BufMut <- kernels::Source::Out(0),
        in_vec_size: I32 <- kernels::Source::Param(0),
        out_vec_size: I32 <- kernels::Source::Param(1),
        bias: Buf <- kernels::Source::Weight(2),
        expert_ids: Buf <- kernels::Source::In(1),
        x_slot_stride: I32 <- kernels::Source::Param(2),
        x_row_stride: I32 <- kernels::Source::Param(3),
        slots_per_row: I32 <- kernels::Source::Param(4),
    ],
    axes = &[BF16, GROUP_32, BITS_4]),
    // 54 in quantized_qmm_t.metal
    kernel!(qmm_t_routed "affine_qmm_t_routed", axes = &[BF16, GROUP, BITS, TILE_M, TILE_N]),
    // 9 in quantized_qmm_t.metal
    kernel!(qmm_t_routed_fp16 "affine_qmm_t_routed_fp16",
        axes = &[BF16, GROUP_64, BITS_4, TILE_M, TILE_N]),
    // 1 in quantized_qmv.metal
    // ONE affine format, and that is the kernel's design rather than a gap:
    // `AffineQ::group_size` is a constant, so a second group point would name
    // an instantiation that dequantises at 64 whatever it claims. A routed
    // checkpoint at another group is meant to fail by name when its pipeline
    // is built -- which `entrypoint()` now does at the call instead of in the
    // Metal compiler.
    kernel!(qmv_routed "affine_qmv_routed", file = Some("quant/qmv.metal"),
    launch = kernels::LaunchRule::RoutedQmv,
    // The matvec's row axis is `out_vec_size`, the second word, and not the
    // output rectangle's width: a routed projection writes a whole token's
    // `k` results end to end, so the value is `k` times as wide as one
    // result. See `dsl::metal::routed_qmv`.
    grid_param = Some(1),
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
    // 1 in quantized_qmv.metal
    kernel!(qmv_routed_bias "affine_qmv_routed_bias", file = Some("quant/qmv.metal"),
    launch = kernels::LaunchRule::RoutedQmv,
    // The matvec's row axis is `out_vec_size`, the second word, and not the
    // output rectangle's width: a routed projection writes a whole token's
    // `k` results end to end, so the value is `k` times as wide as one
    // result. See `dsl::metal::routed_qmv`.
    grid_param = Some(1),
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
    // 1 in moe_route.metal
    kernel!(router_topk "router_topk", file = Some("moe/route.metal"),
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
    // 1 in moe_route.metal
    kernel!(router_topk_scaled "router_topk_scaled", file = Some("moe/route.metal"),
    launch = kernels::LaunchRule::RouterLane,
    operands = kernels::operands![
        logits: Buf <- kernels::Source::In(0),
        expert_ids: BufMut <- kernels::Source::Out(0),
        expert_weights: BufMut <- kernels::Source::Out(1),
        params: Buf <- kernels::Source::Param(0),
        per_expert_scale: Buf <- kernels::Source::In(1),
    ],
    axes = &[BF16]),
    kernel!(shared_expert_combine "shared_expert_combine", file = Some("moe/route.metal"),
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
    file = Some("moe/route.metal"),
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
