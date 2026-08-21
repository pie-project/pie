
use kernels::{Bind, Fire};
use kernels_macros::routine;
use crate::jit::{Ctx, Launch, Root, aligned16};
use crate::jit::Abi;
use crate::jit::abi::Tensor;
use crate::jit::abi::bf16;
use kernels::routine::{Asks, Const, In, InOut, Out, Region, Stride};

use kernels::keys;
use kernels::Refusal;

use core::ffi::c_void;

const BLOCK: u32 = 256;

const WARP: u32 = 32;

const FLOAT: u32 = 4;

const SORT_BLOCK: u32 = 1024;

const DISPATCH_BLOCK: u32 = 256;

const MOE_VEC_WIDTH: i32 = 8;

const GEMV_WARPS: i32 = 4;

const FRAG: i32 = 16;

const GEMM_WARPS: u32 = 4;

#[allow(clippy::cast_possible_wrap)]
const N_TILE: i32 = FRAG * GEMM_WARPS as i32;

pub const MOE_ALIGNED_BLOCK_MIN: i32 = 16;

pub const MOE_ALIGNED_BLOCK_MAX: i32 = 64;

#[must_use]
const fn rms(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK).smem((BLOCK / WARP) * FLOAT)
}

#[must_use]
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, BLOCK)
}

#[must_use]
const fn elementwise_rows(rows: u32, width: u32) -> Launch {
    Launch::grid([rows, width.div_ceil(BLOCK), 1], [BLOCK, 1, 1])
}

#[must_use]
const fn router_lane(rows: u32) -> Launch {
    const ROUTER_BLOCK: u32 = 64;

    Launch::per_row(rows, ROUTER_BLOCK)
}

const MAX_EXPERTS: i32 = 512;

fn per_route<P, Q>(dst: Region<P>, routes: Region<Q>) -> Result<i32, Refusal> {
    if dst.width % routes.width != 0 {
        return Err(Refusal::Narrow {
            what: "the row is not a whole number of routes",
            at: i64::from(dst.width),
        });
    }
    Ok(dst.width / routes.width)
}

fn routed_rows<P>(out_rows: i32, out_width: i32, aligned: Region<P>) -> Result<i32, Refusal> {
    let routes = out_rows.saturating_mul(out_width) / aligned.width;
    if routes <= 0 {
        return Err(Refusal::Empty { what: "the routed row count" });
    }
    Ok(routes)
}

#[routine(bf16)]
pub fn topk_sigmoid<T>(
    ctx: &Ctx<'_>,
    logits: In<Tensor<T>>,
    topk_idx: Out<Tensor<i32>>,
    topk_w: Out<Tensor<f32>>,
    correction_bias: Option<Const<Tensor<f32>>>,
    renormalize: Const<bool>,
    routed_scaling_factor: Const<f32>) -> Result<(), Refusal> {

    let renormalize = *renormalize;
    let routed_scaling_factor = *routed_scaling_factor;

    let rect = logits.all("num_experts")?;
    let routed = topk_idx.all("the routed fanout")?;
    let (e, k) = (rect.width, routed.width);
    if e > MAX_EXPERTS {
        return Err(Refusal::Wide {
            what: "num_experts, which the router stages in shared memory",
            at: i64::from(e),
            max: i64::from(MAX_EXPERTS),
        });
    }
    ctx.fire(Fire::at("moe/topk_sigmoid.cuh", crate::jit::symbol(&format!("::pie::moe::topk_sigmoid<{}>", T::CPP))).apply(rms(rect.rows.unsigned_abs())), &[
                rect.ptr.arg(),
                routed.ptr.arg(),
                topk_w.arg(),
                correction_bias.arg(),
                e.arg(),
                k.arg(),
                renormalize.arg(),
                routed_scaling_factor.arg(),
            ])
}

#[routine(bf16)]
pub fn topk_sqrtsoftplus<T>(
    ctx: &Ctx<'_>,
    logits: In<Tensor<T>>,
    topk_idx: Out<Tensor<i32>>,
    topk_w: Out<Tensor<f32>>,
    correction_bias: Option<Const<Tensor<f32>>>,
    renormalize: Const<bool>,
    routed_scaling_factor: Const<f32>) -> Result<(), Refusal> {

    let renormalize = *renormalize;
    let routed_scaling_factor = *routed_scaling_factor;

    let rect = logits.all("num_experts")?;
    let routed = topk_idx.all("the routed fanout")?;
    let (e, k) = (rect.width, routed.width);
    if e > MAX_EXPERTS {
        return Err(Refusal::Wide {
            what: "num_experts, which the router stages in shared memory",
            at: i64::from(e),
            max: i64::from(MAX_EXPERTS),
        });
    }
    ctx.fire(Fire::at("moe/dsv4_routing.cuh", crate::jit::symbol(&format!("::pie::moe::topk_sqrtsoftplus<{}>", T::CPP))).apply(rms(rect.rows.unsigned_abs())), &[
                rect.ptr.arg(),
                routed.ptr.arg(),
                topk_w.arg(),
                correction_bias.arg(),
                e.arg(),
                k.arg(),
                renormalize.arg(),
                routed_scaling_factor.arg(),
            ])
}

#[routine]
pub fn hash_route_lookup(
    ctx: &Ctx<'_>,
    token_ids: In<Tensor<i32>>,
    tid2eid: Const<Tensor<i64>>,
    logits: In<Tensor<bf16>>,
    topk_idx: Out<Tensor<i32>>,
    topk_w: Out<Tensor<f32>>,
    vocab_size: Const<i32>,
    renormalize: Const<bool>,
    routed_scaling_factor: Const<f32>) -> Result<(), Refusal> {

    let vocab_size = *vocab_size;
    let renormalize = *renormalize;
    let routed_scaling_factor = *routed_scaling_factor;

             const DSV4_BLOCK: u32 = 256;

    let rect = logits.all("num_experts")?;
    let routed = topk_idx.all("the routed fanout")?;
    let (tokens, num_experts, top_k) = (token_ids.rows, rect.width, routed.width);
    ctx.fire(Fire::at("moe/dsv4_routing.cuh", "::pie::moe::hash_route_lookup<::pie::bf16>").apply(Launch::flat(tokens.unsigned_abs(), DSV4_BLOCK)), &[
                token_ids.arg(),
                tid2eid.arg(),
                rect.ptr.arg(),
                routed.ptr.arg(),
                topk_w.arg(),
                tokens.arg(),
                vocab_size.arg(),
                num_experts.arg(),
                top_k.arg(),
                renormalize.arg(),
                routed_scaling_factor.arg(),
            ])
}

#[routine(bf16)]
pub fn topk_softmax<T>(
    ctx: &Ctx<'_>,
    logits: In<Tensor<T>>,
    topk_idx: Out<Tensor<i32>>,
    topk_w: Out<Tensor<f32>>) -> Result<(), Refusal> {
    let rect = logits.all("num_experts")?;
    let routed = topk_idx.all("the routed fanout")?;
    let (num_experts, k) = (rect.width, routed.width);
    if num_experts > MAX_EXPERTS {
        return Err(Refusal::Wide {
            what: "num_experts, which the router stages in shared memory",
            at: i64::from(num_experts),
            max: i64::from(MAX_EXPERTS),
        });
    }
    ctx.fire(Fire::at("moe/topk_softmax.cuh", crate::jit::symbol(&format!("::pie::moe::topk_softmax<{}>", T::CPP))).apply(router_lane(rect.rows.unsigned_abs())), &[
                rect.ptr.arg(),
                core::ptr::null::<bf16>().arg(),
                core::ptr::null::<bf16>().arg(),
                routed.ptr.arg(),
                topk_w.arg(),
                num_experts.arg(),
                k.arg(),
                0_i32.arg(),
            ])
}

#[routine]
pub fn topk_sigmoid_bias_fp32(
    ctx: &Ctx<'_>,
    logits: In<Tensor<f32>>,
    correction_bias: Const<Tensor<f32>>,
    topk_idx: Out<Tensor<i32>>,
    topk_w: Out<Tensor<f32>>,
    normalize: Const<bool>,
    routed_scaling_factor: Const<f32>) -> Result<(), Refusal> {

    let normalize = *normalize;
    let routed_scaling_factor = *routed_scaling_factor;

    let rect = logits.all("num_experts")?;
    let routed = topk_idx.all("the routed fanout")?;
    let (num_experts, k) = (rect.width, routed.width);
    if num_experts > MAX_EXPERTS {
        return Err(Refusal::Wide {
            what: "num_experts, which the router stages in shared memory",
            at: i64::from(num_experts),
            max: i64::from(MAX_EXPERTS),
        });
    }
    ctx.fire(Fire::at("moe/topk_softmax.cuh", "::pie::moe::topk_sigmoid_bias<::pie::moe::f32>").apply(router_lane(rect.rows.unsigned_abs())), &[
                rect.ptr.arg(),
                correction_bias.arg(),
                routed.ptr.arg(),
                topk_w.arg(),
                num_experts.arg(),
                k.arg(),
                i32::from(normalize).arg(),
                routed_scaling_factor.arg(),
            ])
}

#[routine(bf16)]
pub fn apply_per_expert_scale<T>(
    ctx: &Ctx<'_>,
    topk_idx: In<Tensor<i32>>,
    topk_w: InOut<Tensor<f32>>,
    per_expert_scale: Const<Tensor<T>>,

) -> Result<(), Refusal> {
    let total = topk_w.rows.saturating_mul(topk_w.width);
    ctx.fire(Fire::at("moe/topk_softmax.cuh", crate::jit::symbol(&format!("::pie::moe::apply_per_expert_scale<{}>", T::CPP))).apply(elementwise(total.unsigned_abs())), &[topk_idx.arg(), topk_w.arg(), per_expert_scale.arg(), total.arg()])
}

pub const fn supported(m: i32, n: i32, k: i32) -> Result<(), Refusal> {

    const SHORT_K: i32 = 512;

    if m > FRAG {
        return Err(Refusal::Wide {
            what: "M, which must be exactly one 16-row fragment",
            at: m as i64,
            max: FRAG as i64,
        });
    }
    if m < FRAG {
        return Err(Refusal::Narrow {
            what: "M, which must be exactly one 16-row fragment",
            at: m as i64,
        });
    }
    if k > SHORT_K {
        return Err(Refusal::Wide {
            what: "K, above which cuBLAS wins",
            at: k as i64,
            max: SHORT_K as i64,
        });
    }
    if n % N_TILE != 0 {
        return Err(Refusal::Narrow { what: "N, in whole 64-wide tiles", at: n as i64 });
    }
    if k % FRAG != 0 {
        return Err(Refusal::Narrow { what: "K, in whole 16-deep fragments", at: k as i64 });
    }
    Ok(())
}

#[routine(bf16, driver)]
pub fn moe_grouped_gemm<T>(
    ctx: &Ctx<'_>,
    a: In<Tensor<T>>,
    weight_base: Const<Tensor<T>>,
    expert_ids: In<Tensor<i32>>,
    c: InOut<Tensor<T>>,
    max_blocks: Const<i32>,
    m: Const<i32>) -> Result<(), Refusal> {

    let rows = max_blocks.saturating_mul(*m);
    let dst = c.over(rows, "N, the destination's width")?;
    let act = a.over(rows, "K, the activation's width")?;
    let (n, k) = (dst.width, act.width);
    supported(*m, n, k)?;
    ctx.fire(Fire::at("moe/moe_grouped_gemm.cuh", crate::jit::symbol(&format!("::pie::moe::moe_grouped_gemm<{}>", T::CPP))).apply(Launch::grid(
                [(n / N_TILE).unsigned_abs(), max_blocks.unsigned_abs(), 1],
                [GEMM_WARPS * 32, 1, 1],
            )), &[
                act.ptr.arg(),
                weight_base.arg(),
                dst.ptr.arg(),
                expert_ids.arg(),
                n.arg(),
                k.arg(),
            ])
}

#[routine(bf16)]
pub fn moe_gate_up_decode_gemv<T>(
    ctx: &Ctx<'_>,
    topk_idx: In<Tensor<i32>>,
    norm_x: In<Tensor<T>>,
    gate_up_base: Const<Tensor<T>>,
    expert_gate_up: Out<Tensor<T>>,

) -> Result<(), Refusal> {
    let routed = topk_idx.all("the route width")?;
    let dst = expert_gate_up.all("the routed destination's width")?;
    let i_moe = per_route(dst, routed)?;
    let src = norm_x.all("H, the hidden size")?;
    let (num_tokens, top_k, h) = (routed.rows, routed.width, src.width);
    let routes = num_tokens * top_k;
    let n = 2 * i_moe;
    if h % MOE_VEC_WIDTH != 0 {
        return Err(Refusal::Narrow { what: "H, in whole float4 loads of 8", at: i64::from(h) });
    }
    ctx.fire(Fire::at("moe/moe_dispatch.cuh", crate::jit::symbol(&format!("::pie::moe::moe_decode_gemv_by_token<{}>", T::CPP))).apply(Launch::grid(
                [n.unsigned_abs().div_ceil(GEMV_WARPS.unsigned_abs()), routes.unsigned_abs(), 1],
                [WARP, GEMV_WARPS.unsigned_abs(), 1],
            )), &[
                routed.ptr.arg(),
                src.ptr.arg(),
                gate_up_base.arg(),
                dst.ptr.arg(),
                top_k.arg(),
                h.arg(),
                n.arg(),
                (i64::from(n) * i64::from(h)).arg(),
            ])
}

#[routine(bf16)]
pub fn moe_down_decode_gemv<T>(
    ctx: &Ctx<'_>,
    topk_idx: In<Tensor<i32>>,
    expert_act: In<Tensor<T>>,
    down_base: Const<Tensor<T>>,
    expert_out: Out<Tensor<T>>) -> Result<(), Refusal> {
    let routed = topk_idx.all("the route width")?;
    let dst = expert_out.all("the routed destination's width")?;
    let h = per_route(dst, routed)?;
    let act = expert_act.all("I_moe, the per-expert intermediate size")?;
    let (num_tokens, top_k, i_moe) = (routed.rows, routed.width, act.width);
    let routes = num_tokens * top_k;
    if i_moe % MOE_VEC_WIDTH != 0 {
        return Err(Refusal::Narrow {
            what: "I_moe, in whole float4 loads of 8",
            at: i64::from(i_moe),
        });
    }
    ctx.fire(Fire::at("moe/moe_dispatch.cuh", crate::jit::symbol(&format!("::pie::moe::moe_decode_gemv_by_route<{}>", T::CPP))).apply(Launch::grid(
                [h.unsigned_abs().div_ceil(GEMV_WARPS.unsigned_abs()), routes.unsigned_abs(), 1],
                [WARP, GEMV_WARPS.unsigned_abs(), 1],
            )), &[
                routed.ptr.arg(),
                act.ptr.arg(),
                down_base.arg(),
                dst.ptr.arg(),
                top_k.arg(),
                i_moe.arg(),
                h.arg(),
                (i64::from(h) * i64::from(i_moe)).arg(),
            ])
}

#[routine]
pub fn transpose_expert_scales_u8(
    ctx: &Ctx<'_>,
    src: Const<Tensor<u8>>,
    dst: Out<Tensor<u8>>,
    num_experts: Const<i32>,
    n: Const<i32>,
    k_groups: Const<i32>) -> Result<(), Refusal> {
    let num_experts = *num_experts;
    let n = *n;
    let k_groups = *k_groups;
    const BX: u32 = 32;
    const BY: u32 = 8;
    ctx.fire(Fire::at("moe/moe_dispatch.cuh", "::pie::moe::transpose_expert_scales<::pie::u8>").apply(Launch::grid(
                [
                    k_groups.unsigned_abs().div_ceil(BX),
                    n.unsigned_abs().div_ceil(BY),
                    num_experts.unsigned_abs(),
                ],
                [BX, BY, 1],
            )), &[src.arg(), dst.ptr.arg(), n.arg(), k_groups.arg()])
}

#[routine(whole, untraced, driver)]
pub fn build_moe_ptrs_aligned_bf16(
    ctx: &Ctx<'_>,
    expert_ids: In<Tensor<i32>>,
    gate_up_base: Const<Tensor<bf16>>,
    down_base: Const<Tensor<bf16>>,
    aligned_in: In<Tensor<bf16>>,
    aligned_gate_up: Out<Tensor<bf16>>,
    aligned_act: Out<Tensor<bf16>>,
    aligned_out: Out<Tensor<bf16>>,
    // BARE, AND THE TYPE SYSTEM IS THE REASON. A mark carries its operand as
    // `Tensor<E>`, and `E` must be an `Elem`; `*const bf16` is not one, so a
    // pointer-to-pointer ARRAY has no carrier a mark could wrap. That is not a
    // gap in this signature — `#[routine]` names the bare pointer *"THE STATED
    // ABSENCE"* and admits it precisely here, on a row the trace never binds.
    //
    // The three rectangles above ARE marks because a staging buffer is a
    // `Tensor<bf16>`; these six are addresses of addresses, which no rectangle
    // describes.
    a_gu_ptrs: *mut *const bf16,
    b_gu_ptrs: *mut *const bf16,
    c_gu_ptrs: *mut *mut bf16,
    a_dn_ptrs: *mut *const bf16,
    b_dn_ptrs: *mut *const bf16,
    c_dn_ptrs: *mut *mut bf16,
    // Null when the text has no shared expert; the rewrite below makes it safe.
    shared_gate_up_base: Const<Tensor<bf16>>,
    shared_down_base: Const<Tensor<bf16>>,
    max_blocks: i32,
    block_size: i32,
    routed_blocks: i32) -> Result<(), Refusal>
{
    let (shared_gate_up_base, shared_down_base) = (shared_gate_up_base.v, shared_down_base.v);

    let aligned_rows = max_blocks.saturating_mul(block_size);
    let hidden = aligned_out.over(aligned_rows, "H, the hidden size")?;
    let inter = aligned_act.over(aligned_rows, "I_moe, the per-expert intermediate size")?;
    let (h, i_moe) = (hidden.width, inter.width);
    let routed_blocks = if shared_gate_up_base.is_null() || shared_down_base.is_null() {
        max_blocks
    } else {
        routed_blocks
    };
    ctx.fire(Fire::at("moe/moe_dispatch.cuh", "::pie::moe::build_moe_ptrs_aligned<::pie::bf16>").apply(Launch::flat(max_blocks.unsigned_abs(), DISPATCH_BLOCK)), &[
                expert_ids.arg(),
                gate_up_base.arg(),
                down_base.arg(),
                aligned_in.arg(),
                aligned_gate_up.arg(),
                inter.ptr.arg(),
                hidden.ptr.arg(),
                a_gu_ptrs.arg(),
                b_gu_ptrs.arg(),
                c_gu_ptrs.arg(),
                a_dn_ptrs.arg(),
                b_dn_ptrs.arg(),
                c_dn_ptrs.arg(),
                max_blocks.arg(),
                block_size.arg(),
                h.arg(),
                i_moe.arg(),
                routed_blocks.arg(),
                shared_gate_up_base.arg(),
                shared_down_base.arg(),
            ])
}

#[routine(bf16, whole)]
pub fn reorder_moe_aligned_output<T>(
    ctx: &Ctx<'_>,
    aligned_out: In<Tensor<T>>,
    sorted_route_ids: In<Tensor<i32>>,
    route_out: Out<Tensor<T>>) -> Result<(), Refusal>
where

    *const T: Abi + kernels::Bind<crate::jit::ArgValue>,
    *mut T: Abi + kernels::Bind<crate::jit::ArgValue>,
    T: kernels::Elem<Write = *mut T>,
    <T as kernels::Elem>::Read: Into<*const T>,
    <T as kernels::Elem>::Write: Into<*mut T>,
{

    fn ptr_of<T>(p: impl Into<*const T>) -> *const c_void {
        p.into().cast()
    }

    fn ptr_of_mut<T>(p: impl Into<*mut T>) -> *const c_void {
        p.into().cast_const().cast()
    }

    #[must_use]
    fn moe_vectorizable(a: *const c_void, b: *const c_void, hidden: i32) -> bool {
    hidden % MOE_VEC_WIDTH == 0 && aligned16(a) && aligned16(b)
    }

    let aligned = aligned_out.all("the aligned rectangle's width")?;
    let num_routes = routed_rows(route_out.rows, route_out.width, aligned)?;
    let (aligned_rows, hidden, num_tokens) =
        (sorted_route_ids.rows, aligned.width, route_out.rows);

    let shared_row_begin = -1;
    let vectorizable =
        moe_vectorizable(ptr_of(aligned.ptr), ptr_of_mut(route_out.ptr), hidden);
    let width = if vectorizable { hidden / MOE_VEC_WIDTH } else { hidden };
    let instantiation = if vectorizable {
        format!("::pie::moe::reorder_moe_aligned_output_vec<{}>", T::CPP)
    } else {
        format!("::pie::moe::reorder_moe_aligned_output<{}>", T::CPP)
    };
    ctx.fire(Fire::at("moe/moe_dispatch.cuh", crate::jit::symbol(&instantiation)).apply(Launch::grid(
                [aligned_rows.unsigned_abs(), width.unsigned_abs().div_ceil(DISPATCH_BLOCK), 1],
                [DISPATCH_BLOCK, 1, 1],
            )), &[
                aligned.ptr.arg(),
                sorted_route_ids.arg(),
                route_out.arg(),
                num_routes.arg(),
                aligned_rows.arg(),
                width.arg(),
                shared_row_begin.arg(),
                num_tokens.arg(),
                core::ptr::null_mut::<T>().arg(),
            ])
}

#[routine(whole)]
pub fn moe_align_decode(
    ctx: &Ctx<'_>,
    topk_idx: In<Tensor<i32>>,
    sorted_route_ids: Out<Tensor<i32>>,
    expert_ids: Out<Tensor<i32>>,
    route_to_aligned_row: Out<Tensor<i32>>,
    num_experts: Const<i32>,
    block_size: Const<i32>,
    max_blocks: Const<i32>,

) -> Result<(), Refusal> {
    #[must_use]
    const fn router_sort(n_experts: u32) -> Launch {
    Launch::per_row(1, SORT_BLOCK).smem((3 * n_experts + 34) * FLOAT)
    }

    let num_routes = topk_idx.rows.saturating_mul(topk_idx.width);
    ctx.fire(Fire::at("moe/moe_dispatch.cuh", "::pie::moe::moe_align_decode<::pie::i32>").apply(router_sort(num_experts.unsigned_abs())), &[
                topk_idx.arg(),
                sorted_route_ids.arg(),
                expert_ids.arg(),
                route_to_aligned_row.arg(),
                num_routes.arg(),
                num_experts.arg(),
                block_size.arg(),
                max_blocks.arg(),
                core::ptr::null_mut::<i32>().arg(),
            ])
}

#[routine(whole)]
pub fn moe_bucket_exact(
    ctx: &Ctx<'_>,
    topk_idx: In<Tensor<i32>>,
    sorted_route_ids: Out<Tensor<i32>>,
    route_to_sorted_row: Out<Tensor<i32>>,
    counts_out: Out<Tensor<i32>>) -> Result<(), Refusal> {

    let num_routes = topk_idx.rows.saturating_mul(topk_idx.width);
    let counts = counts_out.all("num_experts")?;
    let num_experts = counts.width;
    ctx.fire(Fire::at("moe/moe_dispatch.cuh", "::pie::moe::moe_bucket_exact<::pie::i32>").apply(Launch::grid([1, 1, 1], [SORT_BLOCK, 1, 1])
                .smem((3 * num_experts.unsigned_abs() + 1) * FLOAT)), &[
                topk_idx.arg(),
                sorted_route_ids.arg(),
                route_to_sorted_row.arg(),
                counts.ptr.arg(),
                num_routes.arg(),
                num_experts.arg(),
            ])
}

#[routine(bf16, whole)]
pub fn gather_moe_aligned_inputs<T>(
    ctx: &Ctx<'_>,
    norm_x: In<Tensor<T>>,
    sorted_route_ids: In<Tensor<i32>>,
    aligned_in: Out<Tensor<T>>,
    top_k: Const<i32>,
    tokens: Const<i32>) -> Result<(), Refusal> {

    let top_k = *top_k;

    let tokens = *tokens;
    let (aligned_rows, hidden) = (sorted_route_ids.rows, aligned_in.width);

    let top_k = top_k;
    let num_tokens = tokens;
    let num_routes = num_tokens.saturating_mul(top_k);
    ctx.fire(Fire::at("moe/moe_dispatch.cuh", crate::jit::symbol(&format!("::pie::moe::gather_moe_aligned_inputs<{}>", T::CPP))).apply(elementwise_rows(aligned_rows.unsigned_abs(), hidden.unsigned_abs())), &[
                norm_x.arg(),
                sorted_route_ids.arg(),
                aligned_in.arg(),
                num_routes.arg(),
                aligned_rows.arg(),
                top_k.arg(),
                hidden.arg(),
                (-1i32).arg(),
                num_tokens.arg(),
            ])
}

#[routine(bf16)]
pub fn token_batched_weighted_sum<T>(
    ctx: &Ctx<'_>,
    out: Out<Tensor<T>>,
    src: In<Tensor<T>>,
    weights: In<Tensor<f32>>,

) -> Result<(), Refusal> {

    let fan = weights.all("the routed fanout")?;
    let top_k = fan.width;
    ctx.fire(Fire::at("moe/moe_dispatch.cuh", crate::jit::symbol(&format!("::pie::moe::token_batched_weighted_sum<{}>", T::CPP))).apply(elementwise_rows(out.rows.unsigned_abs(), out.width.unsigned_abs())), &[out.arg(), src.ptr.arg(), fan.ptr.arg(), top_k.arg(), out.width.arg()])
}

#[routine(bf16)]
pub fn token_batched_weighted_sum_add<T>(
    ctx: &Ctx<'_>,
    src: In<Tensor<T>>,
    weights: In<Tensor<f32>>,
    out: InOut<Tensor<T>>) -> Result<(), Refusal> {
    let fan = weights.all("the routed fanout")?;
    let top_k = fan.width;
    ctx.fire(Fire::at("moe/moe_dispatch.cuh", crate::jit::symbol(&format!("::pie::moe::token_batched_weighted_sum_add<{}>", T::CPP))).apply(elementwise_rows(out.rows.unsigned_abs(), out.width.unsigned_abs())), &[out.arg(), src.ptr.arg(), fan.ptr.arg(), top_k.arg(), out.width.arg()])
}

#[routine(bf16, internal)]
pub fn scalar_weighted_add<T>(
    ctx: &Ctx<'_>,
    out: Out<Tensor<T>>,
    src: In<Tensor<T>>,
    weight: Const<f32>,

) -> Result<(), Refusal> {
    let n = out.rows.saturating_mul(out.width);
    ctx.fire(Fire::at("moe/moe_dispatch.cuh", crate::jit::symbol(&format!("::pie::moe::scalar_weighted_add<{}>", T::CPP))).apply(elementwise(n.unsigned_abs())), &[out.arg(), src.ptr.arg(), weight.arg(), n.arg()])
}

#[routine(bf16, whole)]
pub fn add_moe_route_bias<T>(
    ctx: &Ctx<'_>,
    out: InOut<Tensor<T>>,
    bias: Const<Tensor<T>>,
    topk_idx: In<Tensor<i32>>,
    out_stride: Const<i32>) -> Result<(), Refusal> {

    let num_routes = topk_idx.rows.saturating_mul(topk_idx.width);
    let dst = out.all("the bias column count")?;

    let out_stride = Stride(*out_stride);

    if dst.width > out_stride.0 {
        return Err(Refusal::Wide {
            what: "the bias column count against the destination's row pitch",
            at: i64::from(dst.width),
            max: i64::from(out_stride.0),
        });
    }
    ctx.fire(Fire::at("moe/moe_dispatch.cuh", crate::jit::symbol(&format!("::pie::moe::add_moe_route_bias<{}>", T::CPP))).apply(rms(num_routes.unsigned_abs())), &[
                dst.ptr.arg(),
                bias.arg(),
                topk_idx.arg(),
                num_routes.arg(),
                dst.width.arg(),
                out_stride.arg(),
            ])
}

#[must_use]
pub fn moe_aligned_block(routes: i32, num_experts: i32) -> i32 {
    if num_experts <= 0 {
        return MOE_ALIGNED_BLOCK_MIN;
    }
    let per_expert = routes / num_experts;
    let mut block = MOE_ALIGNED_BLOCK_MIN;
    while block * 2 <= MOE_ALIGNED_BLOCK_MAX && block * 2 <= per_expert {
        block *= 2;
    }
    block
}

pub static EXPERT_OFFSETS_ROOT: Root = Root::new("moe/expert_offsets.cuh");

#[routine(untraced)]
pub fn flashinfer_cutlass_moe_bf16(
    _ctx: &Ctx<'_>,
    _x: In<Tensor<bf16>>,
    _experts: In<Tensor<c_void>>,
    _weights: Const<Tensor<c_void>>,
    _out: Out<Tensor<bf16>>,
    _tokens: i32,
    _hidden: i32) -> Result<(), Refusal> {
    Err(Refusal::Absent {
        what: "the fused CUTLASS MoE leg, retired with its instantiation seam \
               rather than carried: the aligned leg is the only leg left, and \
               `moe_cutlass_max_rows = 0` is what selects it",
    })
}
