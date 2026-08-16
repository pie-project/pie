//! The MoE family: routing, dispatch, and the expert matmuls between them.
//!
//! Four routers pick experts per token; an alignment pass buckets and
//! gathers rows per expert; the matmuls run them; a combine pass weights
//! the results back onto the tokens.

#![allow(clippy::too_many_arguments)]

use crate::jit::{Ctx, Family, Launch, Root, Routine, aligned16};
use crate::{driver_bound, routine};
use crate::jit::Abi;
use crate::jit::abi::Inst;
use crate::jit::abi::bf16;
use kernels::routine::{Env, InOut, Param, Region, Stride};
// `keys` is imported as the module, not the facts inside it: `stated_source`
// only emits a source when the path's second-to-last segment is `keys`.
use kernels::keys;
use kernels::Bank;
use kernels::In;
use kernels::Out;
use kernels::Refusal;
use kernels::Unbound;
use kernels::Weight;
// `#[kernels_macros::routine]` is spelled out in full at each use: attribute
// macros and `macro_rules!` share a namespace, so importing it would collide
// with `use crate::{driver_bound, routine}` above.

use core::ffi::c_void;

// These mirror native launch constants (`runtime/launch.rs`,
// `moe_dispatch.cuh`) rather than being independent; `MOE_VEC_WIDTH` is eight
// bf16, one `uint4`.
const BLOCK: u32 = 256;

const WARP: u32 = 32;

const FLOAT: u32 = 4;

const SORT_BLOCK: u32 = 1024;

const DISPATCH_BLOCK: u32 = 256;

const MOE_VEC_WIDTH: i32 = 8;

const GEMV_WARPS: i32 = 4;

// Extents are checked either by reaching a grid axis (`Launch::empty()`
// refuses a zero) or by the launcher building a view over the operand
// before reading its width. `moe_grouped_gemm_bf16` and
// `build_moe_ptrs_aligned_bf16` are driver ops whose extents skip this
// filter, so their own bodies are the only guard.

// `moe_grouped_gemm.cuh` supplies `FRAG` (`kFrag`), `GEMM_WARPS`
// (`kGemmWarps`), and `N_TILE` (`kNTile`, the N-axis tile).
const FRAG: i32 = 16;

const GEMM_WARPS: u32 = 4;

#[allow(clippy::cast_possible_wrap)]
const N_TILE: i32 = FRAG * GEMM_WARPS as i32;

/// The smallest block the aligned MoE path is ever padded to.
pub const MOE_ALIGNED_BLOCK_MIN: i32 = 16;

/// The largest, and the cap is a measurement rather than a limit.
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
/// The expert ceiling all four routers share — a shared-memory bound,
/// distinct from the "does this rectangle exist" check each view already
/// makes. [`hash_route_lookup`] never enforces it.
const MAX_EXPERTS: i32 = 512;

/// The width one route holds, of a rectangle whose full width is
/// `routes.width` of them.
///
/// Shared by [`moe_gate_up_decode_gemv`] and [`moe_down_decode_gemv`] so the
/// divisibility rule lives in one place. Takes [`Region`]s, not `i32`s, so a
/// zero divisor can't already have slipped in disguised as a positive width.
fn per_route<P, Q>(dst: Region<P>, routes: Region<Q>) -> Result<i32, Refusal> {
    if dst.width % routes.width != 0 {
        return Err(Refusal::Narrow {
            what: "the row is not a whole number of routes",
            at: i64::from(dst.width),
        });
    }
    Ok(dst.width / routes.width)
}

/// [`reorder_moe_aligned_output`]'s route count.
///
/// `aligned` is a [`Region`], [`per_route`]'s reason. The quotient still
/// needs its own guard: a positive divisor and numerator can still divide to
/// nothing.
fn routed_rows<P>(out_rows: i32, out_width: i32, aligned: Region<P>) -> Result<i32, Refusal> {
    let routes = out_rows.saturating_mul(out_width) / aligned.width;
    if routes <= 0 {
        return Err(Refusal::Empty { what: "the routed row count" });
    }
    Ok(routes)
}

/// `moe::topk_sigmoid_bf16` — the sigmoid router, one block per token.
///
/// # Safety
///
/// `logits`, `topk_idx` and `topk_w` each address their own `rows * width`
/// elements; `correction_bias` is null or one float per expert.
#[kernels_macros::routine]
pub fn topk_sigmoid<T>(
    ctx: &Ctx,
    logits: In<0, T>,
    topk_idx: Out<0, i32>,
    topk_w: Out<1, f32>,
    // Null at runtime when the router has no per-expert bias (most
    // routers); deepseek-style routers set it. `Bank<0, f32>` reads the
    // statement's positional weight slot rather than a named one.
    correction_bias: Bank<0, f32>,
    // Both total deployment facts with defaults, so no refusal is needed
    // here. Named after the `__global__`'s own parameters for readability;
    // binding is by type, so the name need not match `MoeNormTopk`.
    renormalize: Env<keys::MoeNormTopk>,
    routed_scaling_factor: Env<keys::MoeRoutedScaling>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
{
    // One block per token, so nothing else would catch a zero-wide
    // rectangle; these two views are the only guard.
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
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "moe/topk_sigmoid.cuh",
            &format!("::pie::moe::topk_sigmoid<{}>", T::CPP),
            rms(rect.rows.unsigned_abs()),
            &[
                rect.ptr.arg(),
                routed.ptr.arg(),
                topk_w.ptr.arg(),
                correction_bias.ptr.arg(),
                e.arg(),
                k.arg(),
                renormalize.arg(),
                routed_scaling_factor.arg(),
            ],
        )
    }
}

/// `moe::topk_sqrtsoftplus_bf16` — DeepSeek-V4's sqrt-softplus router.
///
/// # Safety
///
/// As [`topk_sigmoid`].
#[kernels_macros::routine]
pub fn topk_sqrtsoftplus<T>(
    ctx: &Ctx,
    logits: In<0, T>,
    topk_idx: Out<0, i32>,
    topk_w: Out<1, f32>,
    correction_bias: Bank<0, f32>,
    renormalize: Env<keys::MoeNormTopk>,
    routed_scaling_factor: Env<keys::MoeRoutedScaling>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
{
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
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "moe/dsv4_routing.cuh",
            &format!("::pie::moe::topk_sqrtsoftplus<{}>", T::CPP),
            rms(rect.rows.unsigned_abs()),
            &[
                rect.ptr.arg(),
                routed.ptr.arg(),
                topk_w.ptr.arg(),
                correction_bias.ptr.arg(),
                e.arg(),
                k.arg(),
                renormalize.arg(),
                routed_scaling_factor.arg(),
            ],
        )
    }
}

/// `moe::hash_route_lookup` — DeepSeek-V4's hashed expert sets.
///
/// # Safety
///
/// `token_ids` is `[tokens]` i32 in `[0, vocab_size)`; `tid2eid`
/// `[vocab_size, top_k]` i64; `logits` `[tokens, num_experts]` bf16;
/// `topk_idx`/`topk_w` writable `[tokens, top_k]`. Only `tokens` is bounded.
#[kernels_macros::routine]
pub fn hash_route_lookup(
    ctx: &Ctx,
    token_ids: In<0, i32>,
    // `Weight<0, _>`, not `WeightNamed`: they agree today, but would diverge
    // for a `scale.`-prefixed name, which `WeightNamed` refuses.
    tid2eid: Weight<0, *const i64>,
    logits: In<1, bf16>,
    topk_idx: Out<0, i32>,
    topk_w: Out<1, f32>,
    // `tid2eid`'s vocabulary axis; no operand carries it, so it comes from
    // the model instead.
    vocab_size: Env<keys::Vocab>,
    renormalize: Env<keys::MoeNormTopk>,
    routed_scaling_factor: Env<keys::MoeRoutedScaling>,
) -> Result<(), Refusal> {
             const DSV4_BLOCK: u32 = 256;

    // `token_ids` gets no view: its rows reach the grid axis directly and
    // its width is never read.
    let rect = logits.all("num_experts")?;
    let routed = topk_idx.all("the routed fanout")?;
    let (tokens, num_experts, top_k) = (token_ids.rows, rect.width, routed.width);
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "moe/dsv4_routing.cuh",
            "::pie::moe::hash_route_lookup<::pie::bf16>",
            Launch::flat(tokens.unsigned_abs(), DSV4_BLOCK),
            &[
                token_ids.ptr.arg(),
                tid2eid.ptr.arg(),
                rect.ptr.arg(),
                routed.ptr.arg(),
                topk_w.ptr.arg(),
                tokens.arg(),
                vocab_size.arg(),
                num_experts.arg(),
                top_k.arg(),
                renormalize.arg(),
                routed_scaling_factor.arg(),
            ],
        )
    }
}

/// `moe::topk_softmax_bf16` — the softmax router's BLOCK form.
///
/// # Safety
///
/// `logits` addresses `logits.rows * logits.width` live elements and
/// `topk_idx` / `topk_w` `topk_idx.rows * topk_idx.width` writable ones.
#[kernels_macros::routine]
pub fn topk_softmax<T>(
    ctx: &Ctx,
    logits: In<0, T>,
    topk_idx: Out<0, i32>,
    topk_w: Out<1, f32>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
{
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
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "moe/topk_softmax.cuh",
            &format!("::pie::moe::topk_softmax<{}>", T::CPP),
            router_lane(rect.rows.unsigned_abs()),
            &[
                rect.ptr.arg(),
                core::ptr::null::<bf16>().arg(),
                core::ptr::null::<bf16>().arg(),
                routed.ptr.arg(),
                topk_w.ptr.arg(),
                num_experts.arg(),
                k.arg(),
                0_i32.arg(),
            ],
        )
    }
}

/// `moe::topk_sigmoid_bias_fp32` — sigmoid routing with the correction bias
/// in fp32.
///
/// # Safety
///
/// `logits` addresses `tokens * num_experts` live floats; `correction_bias`
/// must be `num_experts` live floats and never null — null here is a fault.
#[kernels_macros::routine]
pub fn topk_sigmoid_bias_fp32(
    ctx: &Ctx,
    logits: In<0, f32>,
    correction_bias: Weight<0, *const f32>,
    topk_idx: Out<0, i32>,
    topk_w: Out<1, f32>,
    // Same fact as [`topk_sigmoid`]'s `renormalize`, spelled `normalize`
    // here to match this `__global__`'s own parameter name.
    normalize: Env<keys::MoeNormTopk>,
    routed_scaling_factor: Env<keys::MoeRoutedScaling>,
) -> Result<(), Refusal> {
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
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "moe/topk_softmax.cuh",
            "::pie::moe::topk_sigmoid_bias<::pie::moe::f32>",
            router_lane(rect.rows.unsigned_abs()),
            &[
                rect.ptr.arg(),
                correction_bias.ptr.arg(),
                routed.ptr.arg(),
                topk_w.ptr.arg(),
                num_experts.arg(),
                k.arg(),
                // This kernel takes `int` where its cousins take `bool`;
                // `i32::from` doesn't deref like `.arg()` does, hence `**`.
                i32::from(**normalize).arg(),
                routed_scaling_factor.arg(),
            ],
        )
    }
}

/// `moe::apply_per_expert_scale_bf16` — folds a per-expert scale into the
/// router weights, in place.
///
/// # Safety
///
/// `topk_idx` and `topk_w` each address `topk_w.rows * topk_w.width` live
/// elements, and `per_expert_scale` one per expert named by any of them.
#[kernels_macros::routine]
pub fn apply_per_expert_scale<T>(
    ctx: &Ctx,
    topk_idx: In<0, i32>,
    // Aliases result 0 via `in_place = &[(0, 1)]`, but reads as `In<1, ..>`:
    // `Out<0, ..>` would silently switch the query to `out_width(0)`.
    topk_w: InOut<1, f32>,
    // Positional, not named: this parameter's `scale.`-prefixed name is one
    // `WeightNamed` would refuse (a dangling sentinel), yet the kernel reads
    // it as a real per-expert array.
    per_expert_scale: Weight<0, *const T>,
    // Route count is `topk_w.rows * topk_w.width` (tokens * top_k), not
    // `topk_w.rows` alone — `topk_w` is `[Tokens, top_k]`.
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
{
    let total = topk_w.rows.saturating_mul(topk_w.width);
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "moe/topk_softmax.cuh",
            &format!("::pie::moe::apply_per_expert_scale<{}>", T::CPP),
            elementwise(total.unsigned_abs()),
            &[topk_idx.ptr.arg(), topk_w.ptr.arg(), per_expert_scale.ptr.arg(), total.arg()],
        )
    }
}

/// Whether the short-K grouped GEMM can compute this rectangle at all.
pub const fn supported(m: i32, n: i32, k: i32) -> Result<(), Refusal> {
    /// The reduction bound past which the grouped GEMM stops paying.
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

/// `moe::moe_grouped_gemm_bf16` — the short-K grouped GEMM, one launch over
/// the padded rectangle.
///
/// # Safety
///
/// The four pointers must be device allocations of the shapes above, live on
/// `stream` until the launch completes.
#[kernels_macros::routine]
pub fn moe_grouped_gemm<T>(
    ctx: &Ctx,
    a: In<0, T>,
    // Named, not positional: this driver op reads the bank via the
    // resolver, not the flat arg run.
    weight_base: Weight<0, *const T>,
    // N is `c.width`; the row count (`max_blocks * block_size`) is this
    // launcher's own product, so `over()` builds the view rather than
    // reading it off `c`.
    c: Out<0, T>,
    // The statement's input 2 (`stage`) is never bound here; the launcher
    // takes it once, as `c`.
    expert_ids: In<1, i32>,
    // Adjacent, same-typed `i32`s that swap with no type error: `m` is
    // `block_size` (param 0) and must equal `FRAG`; `max_blocks` is param 1.
    max_blocks: i32,
    m: i32,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // `supported` lets N=0/K=0 through (both divide evenly), and a
    // zero-extent grid launches nothing rather than panicking; these two
    // views catch that instead.
    let rows = max_blocks.saturating_mul(m);
    let dst = c.over(rows, "N, the destination's width")?;
    let act = a.over(rows, "K, the activation's width")?;
    let (n, k) = (dst.width, act.width);
    supported(m, n, k)?;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "moe/moe_grouped_gemm.cuh",
            &format!("::pie::moe::moe_grouped_gemm<{}>", T::CPP),
            Launch::grid(
                [(n / N_TILE).unsigned_abs(), max_blocks.unsigned_abs(), 1],
                [GEMM_WARPS * 32, 1, 1],
            ),
            &[
                act.ptr.arg(),
                weight_base.ptr.arg(),
                dst.ptr.arg(),
                expert_ids.ptr.arg(),
                n.arg(),
                k.arg(),
            ],
        )
    }
}

/// `moe::moe_gate_up_decode_gemv_bf16` — the decode gate/up leg, one fused
/// GEMV. `I_moe` is `out_width / top_k`, divided out via [`per_route`].
///
/// # Errors
///
/// [`Refusal::Absent`] from the three views; [`per_route`]'s
/// [`Refusal::Narrow`] on the fanout or a non-whole-float4 `H`.
///
/// # Safety
///
/// `topk_idx` is `[num_tokens, top_k]` i32, `norm_x` `[num_tokens, H]` bf16,
/// `gate_up_base` the `[experts, 2*I_moe, H]` weight, `expert_gate_up`
/// writable for `[num_tokens * top_k, 2*I_moe]` bf16.
#[kernels_macros::routine]
pub fn moe_gate_up_decode_gemv<T>(
    ctx: &Ctx,
    topk_idx: In<0, i32>,
    norm_x: In<1, T>,
    gate_up_base: Weight<0, *const T>,
    expert_gate_up: Out<0, T>,
    // `topk_idx.width` is the route width (a width, not a row count);
    // `expert_gate_up.width` arrives undivided — [`per_route`] divides it.
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
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
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "moe/moe_dispatch.cuh",
            &format!("::pie::moe::moe_decode_gemv_by_token<{}>", T::CPP),
            Launch::grid(
                [n.unsigned_abs().div_ceil(GEMV_WARPS.unsigned_abs()), routes.unsigned_abs(), 1],
                [WARP, GEMV_WARPS.unsigned_abs(), 1],
            ),
            &[
                routed.ptr.arg(),
                src.ptr.arg(),
                gate_up_base.ptr.arg(),
                dst.ptr.arg(),
                top_k.arg(),
                h.arg(),
                n.arg(),
                (i64::from(n) * i64::from(h)).arg(),
            ],
        )
    }
}

/// `moe::moe_down_decode_gemv_bf16` — the decode down leg, reading the
/// gate/up leg's activated output. `i_moe` (`in_width(1)`) is untouched;
/// [`per_route`] divides `out_width` for `H` instead.
///
/// # Errors
///
/// [`moe_gate_up_decode_gemv`]'s three views, "I_moe" for "H"; or a
/// non-whole-float4 `i_moe`.
///
/// # Safety
///
/// `expert_act` is `[num_tokens * top_k, I_moe]` bf16 (the gate/up leg's
/// SwiGLU output), `down_base` the `[experts, H, I_moe]` weight,
/// `expert_out` writable for `[num_tokens * top_k, H]` bf16.
#[kernels_macros::routine]
pub fn moe_down_decode_gemv<T>(
    ctx: &Ctx,
    topk_idx: In<0, i32>,
    expert_act: In<1, T>,
    down_base: Weight<0, *const T>,
    expert_out: Out<0, T>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
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
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "moe/moe_dispatch.cuh",
            &format!("::pie::moe::moe_decode_gemv_by_route<{}>", T::CPP),
            Launch::grid(
                [h.unsigned_abs().div_ceil(GEMV_WARPS.unsigned_abs()), routes.unsigned_abs(), 1],
                [WARP, GEMV_WARPS.unsigned_abs(), 1],
            ),
            &[
                routed.ptr.arg(),
                act.ptr.arg(),
                down_base.ptr.arg(),
                dst.ptr.arg(),
                top_k.arg(),
                i_moe.arg(),
                h.arg(),
                (i64::from(h) * i64::from(i_moe)).arg(),
            ],
        )
    }
}

/// `moe::transpose_expert_scales_u8` — the MXFP4 group-scale relayout.
///
/// # Safety
///
/// `src` and `dst` are `num_experts * n * k_groups` bytes each and must not
/// overlap: the kernel writes `dst[e][j][i]` from `src[e][i][j]`.
#[kernels_macros::routine]
pub fn transpose_expert_scales_u8(
    ctx: &Ctx,
    // Unbound today; `Weight<0, _>` since this statement's one bank is
    // named, not positional.
    src: Weight<0, *const u8>,
    dst: Out<0, u8>,
    // `n` stays a bare `i32`, not `Env<i32>`: `fact_of` would match its name
    // against "n"/"rows"/"tokens" and silently derive the launch's row
    // count instead of this bank's own dimension.
    num_experts: i32,
    n: i32,
    k_groups: i32,
) -> Result<(), Refusal> {
    const BX: u32 = 32;
    const BY: u32 = 8;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "moe/moe_dispatch.cuh",
            "::pie::moe::transpose_expert_scales<::pie::u8>",
            Launch::grid(
                [
                    k_groups.unsigned_abs().div_ceil(BX),
                    n.unsigned_abs().div_ceil(BY),
                    num_experts.unsigned_abs(),
                ],
                [BX, BY, 1],
            ),
            &[src.ptr.arg(), dst.ptr.arg(), n.arg(), k_groups.arg()],
        )
    }
}

/// `moe::build_moe_ptrs_aligned_bf16` — fills the six pointer arrays a pair
/// of batched GEMMs index through.
///
/// # Safety
///
/// The six pointer arrays are device arrays of at least `max_blocks`
/// pointers each; `shared_gate_up_base`/`shared_down_base` may be null (the
/// rewrite above is what makes that safe).
#[kernels_macros::routine]
pub fn build_moe_ptrs_aligned_bf16(
    ctx: &Ctx,
    // Driver-dispatched: `operand()` never validates this against the
    // statement, so the order (inputs, outputs, weights) is kept correct by
    // hand.
    expert_ids: In<0, i32>,
    // Positional, not named: this driver op reads both banks from the flat
    // `args` run at a literal index.
    gate_up_base: Bank<0, bf16>,
    down_base: Bank<1, bf16>,
    aligned_in: In<1, bf16>,
    // Three staging buffers: `[aligned, 2*I]`, `[aligned, I]`, `[aligned,
    // H]`. Only outputs 1 and 2 carry a width view; output 0's width
    // (`2 * i_moe`) is computed below.
    aligned_gate_up: Out<0, bf16>,
    aligned_act: Out<1, bf16>,
    aligned_out: Out<2, bf16>,
    // Driver-owned workspace, not trace values: the batched-cuBLAS fallback
    // reads these from inside `moe_grouped_gemm_bf16`'s own body, so
    // declaring them as results would free them (liveness) too early.
    a_gu_ptrs: Env<*mut *const bf16>,
    b_gu_ptrs: Env<*mut *const bf16>,
    c_gu_ptrs: Env<*mut *mut bf16>,
    a_dn_ptrs: Env<*mut *const bf16>,
    b_dn_ptrs: Env<*mut *const bf16>,
    c_dn_ptrs: Env<*mut *mut bf16>,
    // `routed_blocks` is `max_blocks`, rewritten below when the shared pair
    // is null.
    max_blocks: i32,
    block_size: i32,
    routed_blocks: i32,
    // Null when the text has no shared expert; `Unbound`, not a slot, since
    // they're unplaceable rather than mis-numbered.
    shared_gate_up_base: Unbound<*const bf16>,
    shared_down_base: Unbound<*const bf16>,
) -> Result<(), Refusal>
{
    let (a_gu_ptrs, b_gu_ptrs, c_gu_ptrs) = (*a_gu_ptrs, *b_gu_ptrs, *c_gu_ptrs);
    let (a_dn_ptrs, b_dn_ptrs, c_dn_ptrs) = (*a_dn_ptrs, *b_dn_ptrs, *c_dn_ptrs);
    // A zero width wouldn't panic: every pointer-array entry would alias the
    // base address instead, and both GEMMs would read it as correct.
    let aligned_rows = max_blocks.saturating_mul(block_size);
    let hidden = aligned_out.over(aligned_rows, "H, the hidden size")?;
    let inter = aligned_act.over(aligned_rows, "I_moe, the per-expert intermediate size")?;
    let (h, i_moe) = (hidden.width, inter.width);
    let (shared_gate_up_base, shared_down_base) =
        (shared_gate_up_base.ptr, shared_down_base.ptr);
    let routed_blocks = if shared_gate_up_base.is_null() || shared_down_base.is_null() {
        max_blocks
    } else {
        routed_blocks
    };
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "moe/moe_dispatch.cuh",
            "::pie::moe::build_moe_ptrs_aligned<::pie::bf16>",
            Launch::flat(max_blocks.unsigned_abs(), DISPATCH_BLOCK),
            &[
                expert_ids.ptr.arg(),
                gate_up_base.ptr.arg(),
                down_base.ptr.arg(),
                aligned_in.ptr.arg(),
                aligned_gate_up.ptr.arg(),
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
            ],
        )
    }
}
/// `moe::reorder_moe_aligned_output_bf16` — scatters an aligned GEMM's
/// output rows back to their routes.
///
/// # Safety
///
/// `aligned_out` is `[aligned_rows, hidden]` bf16, `sorted_route_ids`
/// `[aligned_rows]` i32, `route_out` writable for `[num_routes, hidden]`
/// bf16.
#[kernels_macros::routine]
pub fn reorder_moe_aligned_output<T>(
    ctx: &Ctx,
    aligned_out: In<0, T>,
    sorted_route_ids: In<1, i32>,
    route_out: Out<0, T>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    #[must_use]
    fn moe_vectorizable(a: *const c_void, b: *const c_void, hidden: i32) -> bool {
    hidden % MOE_VEC_WIDTH == 0 && aligned16(a) && aligned16(b)
    }

    // `route_out` gets no view: both numbers are guarded only via
    // [`routed_rows`]'s quotient.
    let aligned = aligned_out.all("the aligned rectangle's width")?;
    let num_routes = routed_rows(route_out.rows, route_out.width, aligned)?;
    let (aligned_rows, hidden, num_tokens) =
        (sorted_route_ids.rows, aligned.width, route_out.rows);
    // No shared tail is ever supplied, so `vectorizable`'s check on a null
    // pointer passes trivially — not a real guard.
    let shared_row_begin = -1;
    let vectorizable =
        moe_vectorizable(aligned.ptr.cast(), route_out.ptr.cast_const().cast(), hidden);
    let width = if vectorizable { hidden / MOE_VEC_WIDTH } else { hidden };
    let instantiation = if vectorizable {
        &format!("::pie::moe::reorder_moe_aligned_output_vec<{}>", T::CPP)
    } else {
        &format!("::pie::moe::reorder_moe_aligned_output<{}>", T::CPP)
    };
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "moe/moe_dispatch.cuh",
            instantiation,
            Launch::grid(
                [aligned_rows.unsigned_abs(), width.unsigned_abs().div_ceil(DISPATCH_BLOCK), 1],
                [DISPATCH_BLOCK, 1, 1],
            ),
            &[
                aligned.ptr.arg(),
                sorted_route_ids.ptr.arg(),
                route_out.ptr.arg(),
                num_routes.arg(),
                aligned_rows.arg(),
                width.arg(),
                shared_row_begin.arg(),
                num_tokens.arg(),
                core::ptr::null_mut::<T>().arg(),
            ],
        )
    }
}

/// `moe::moe_align_decode` — the block-padded counting sort: buckets routes
/// by expert into blocks.
///
/// # Safety
///
/// `topk_idx` is `[num_routes]` i32 in `[0, num_experts)`; `sorted_route_ids`
/// and `route_to_aligned_row` are writable `[num_routes]`, `expert_ids`
/// `[max_blocks]`. `block_size * max_blocks` is the padded row count.
#[kernels_macros::routine]
pub fn moe_align_decode(
    ctx: &Ctx,
    topk_idx: In<0, i32>,
    sorted_route_ids: Out<0, i32>,
    expert_ids: Out<1, i32>,
    route_to_aligned_row: Out<2, i32>,
    num_experts: Param<0, i32>,
    block_size: Param<1, i32>,
    max_blocks: Param<2, i32>,
    // Always null: no caller supplies a scratch buffer, and the kernel
    // guards every write to it on non-null.
) -> Result<(), Refusal> {
    #[must_use]
    const fn router_sort(n_experts: u32) -> Launch {
    Launch::per_row(1, SORT_BLOCK).smem((3 * n_experts + 34) * FLOAT)
    }

    let num_routes = topk_idx.rows.saturating_mul(topk_idx.width);
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "moe/moe_dispatch.cuh",
            "::pie::moe::moe_align_decode<::pie::i32>",
            router_sort(num_experts.unsigned_abs()),
            &[
                topk_idx.ptr.arg(),
                sorted_route_ids.ptr.arg(),
                expert_ids.ptr.arg(),
                route_to_aligned_row.ptr.arg(),
                num_routes.arg(),
                num_experts.arg(),
                block_size.arg(),
                max_blocks.arg(),
                core::ptr::null_mut::<i32>().arg(),
            ],
        )
    }
}

// The params run is its own index space: a param wrapper that advanced the
// operand counters would let a bare pointer silently bind one slot along.
// Checked by `cargo check` alone.
const _: () = {
    let d = <moe_align_decode as kernels::Derivation>::DERIVED;
    assert!(d.len() == 7);
    assert!(matches!(d[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[2].source, Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(matches!(d[3].source, Some(kernels::Source::Slot(kernels::Kind::Out, 2))));
    assert!(matches!(d[4].source, Some(kernels::Source::Slot(kernels::Kind::Param, 0))));
    assert!(matches!(d[5].source, Some(kernels::Source::Slot(kernels::Kind::Param, 1))));
    assert!(matches!(d[6].source, Some(kernels::Source::Slot(kernels::Kind::Param, 2))));
};

/// `moe::moe_bucket_exact` — the unpadded sort: exact per-expert counts, no
/// block padding.
///
/// # Safety
///
/// `topk_idx` is `[num_routes]` i32 in `[0, num_experts)`; `sorted_route_ids`
/// and `route_to_sorted_row` are writable `[num_routes]`, `counts_out`
/// `[num_experts]`. An out-of-range id indexes past the shared slab.
#[kernels_macros::routine]
pub fn moe_bucket_exact(
    ctx: &Ctx,
    topk_idx: In<0, i32>,
    sorted_route_ids: Out<0, i32>,
    route_to_sorted_row: Out<1, i32>,
    counts_out: Out<2, i32>,
) -> Result<(), Refusal> {
    // A bare product: the grid is a literal `[1, 1, 1]`, so nothing catches
    // a zero-route sort; only the expert count has a guard, via the view.
    let num_routes = topk_idx.rows.saturating_mul(topk_idx.width);
    let counts = counts_out.all("num_experts")?;
    let num_experts = counts.width;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "moe/moe_dispatch.cuh",
            "::pie::moe::moe_bucket_exact<::pie::i32>",
            Launch::grid([1, 1, 1], [SORT_BLOCK, 1, 1])
                .smem((3 * num_experts.unsigned_abs() + 1) * FLOAT),
            &[
                topk_idx.ptr.arg(),
                sorted_route_ids.ptr.arg(),
                route_to_sorted_row.ptr.arg(),
                counts.ptr.arg(),
                num_routes.arg(),
                num_experts.arg(),
            ],
        )
    }
}

/// `moe::gather_moe_aligned_inputs_bf16` — gathers token rows into the
/// aligned, block-padded rectangle.
///
/// # Safety
///
/// `norm_x` is `[num_tokens, hidden]` bf16, `sorted_route_ids`
/// `[aligned_rows]` i32, `aligned_in` writable for `[aligned_rows, hidden]`
/// bf16.
#[kernels_macros::routine]
pub fn gather_moe_aligned_inputs<T>(
    ctx: &Ctx,
    norm_x: In<0, T>,
    sorted_route_ids: In<1, i32>,
    aligned_in: Out<0, T>,
    // `RowsTotal` is the ALIGNED row count here, not the token count. Must
    // equal `moe_align_decode`'s `tokens * top_k`, the sentinel row's key.
    tokens: Env<keys::RowsTotal>,
    // From `Geometry.experts_per_token`, not `DispatchCtx::experts_per_token`
    // (always 0 on this driver). Already refuses `Unstated` for `<= 0`.
    top_k: Env<keys::ExpertsPerToken>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let (aligned_rows, hidden) = (sorted_route_ids.rows, aligned_in.width);
    // `Env<T>` derefs to `T` and the key derefs to its value: two derefs
    // down.
    let top_k = **top_k;
    let num_tokens = **tokens;
    let num_routes = num_tokens.saturating_mul(top_k);
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "moe/moe_dispatch.cuh",
            &format!("::pie::moe::gather_moe_aligned_inputs<{}>", T::CPP),
            elementwise_rows(aligned_rows.unsigned_abs(), hidden.unsigned_abs()),
            &[
                norm_x.ptr.arg(),
                sorted_route_ids.ptr.arg(),
                aligned_in.ptr.arg(),
                num_routes.arg(),
                aligned_rows.arg(),
                top_k.arg(),
                hidden.arg(),
                // No shared tail for this leg either.
                (-1i32).arg(),
                num_tokens.arg(),
            ],
        )
    }
}

/// `moe::token_batched_weighted_sum_bf16` — the combine: weighted sum of
/// each token's top-k expert outputs.
///
/// # Safety
///
/// `src` is `[num_tokens, top_k, hidden]` bf16, `weights` `[num_tokens,
/// top_k]` f32, `out` writable for `[num_tokens, hidden]` bf16.
#[kernels_macros::routine]
pub fn token_batched_weighted_sum<T>(
    ctx: &Ctx,
    out: Out<0, T>,
    src: In<0, T>,
    weights: In<1, f32>,
    // Off `weights` (operand 1), not `out`/`src`: the fanout rides in as
    // the router's own weight vector.
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // Only `weights` has a guard; `out`'s numbers reach the grid axis
    // directly.
    let fan = weights.all("the routed fanout")?;
    let top_k = fan.width;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "moe/moe_dispatch.cuh",
            &format!("::pie::moe::token_batched_weighted_sum<{}>", T::CPP),
            elementwise_rows(out.rows.unsigned_abs(), out.width.unsigned_abs()),
            &[out.ptr.arg(), src.ptr.arg(), fan.ptr.arg(), top_k.arg(), out.width.arg()],
        )
    }
}

/// `moe::token_batched_weighted_sum_add_bf16` — the same combine,
/// accumulating into `out` rather than overwriting it.
///
/// # Safety
///
/// As [`token_batched_weighted_sum`], and `out` is read as well as
/// written.
#[kernels_macros::routine]
pub fn token_batched_weighted_sum_add<T>(
    ctx: &Ctx,
    // `in_place = &[(0, 2)]` aliases input 2 with result 0, so indices are
    // stated explicitly.
    out: Out<0, T>,
    src: In<0, T>,
    weights: In<1, f32>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let fan = weights.all("the routed fanout")?;
    let top_k = fan.width;
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "moe/moe_dispatch.cuh",
            &format!("::pie::moe::token_batched_weighted_sum_add<{}>", T::CPP),
            elementwise_rows(out.rows.unsigned_abs(), out.width.unsigned_abs()),
            &[out.ptr.arg(), src.ptr.arg(), fan.ptr.arg(), top_k.arg(), out.width.arg()],
        )
    }
}

/// `moe::scalar_weighted_add_bf16` — `out += weight * src` over a flat run.
///
/// # Safety
///
/// `out` and `src` each address `n` live elements and may alias exactly
/// (`in_place: &[(0, 0)]`); `out` is read as well as written.
#[kernels_macros::routine]
pub fn scalar_weighted_add<T>(
    ctx: &Ctx,
    out: Out<0, T>,
    src: In<0, T>,
    // The residual scale, caller-chosen, with no arm or fire behind it.
    weight: f32,
    // `n` is `out.rows * out.width`; valid off either operand since the
    // device row aliases both exactly (`in_place: &[(0, 0)]`).
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    let n = out.rows.saturating_mul(out.width);
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "moe/moe_dispatch.cuh",
            &format!("::pie::moe::scalar_weighted_add<{}>", T::CPP),
            elementwise(n.unsigned_abs()),
            &[out.ptr.arg(), src.ptr.arg(), weight.arg(), n.arg()],
        )
    }
}

/// `moe::add_moe_route_bias_bf16` — adds each route's expert bias onto that
/// route's output row.
///
/// # Errors
///
/// [`Refusal::Absent`] naming "the bias column count" when `out` states no
/// row width, and [`Refusal::Wide`] when that width exceeds `out_stride`.
///
/// # Safety
///
/// `out` is writable bf16 `[num_routes, out_stride]` and read as well as
/// written; `bias` is `[num_experts, cols]` bf16, `topk_idx` `[num_routes]`
/// i32 with every entry a valid expert.
#[kernels_macros::routine]
pub fn add_moe_route_bias<T>(
    ctx: &Ctx,
    out: Out<0, T>,
    // `Bank<0, _>`, not `In<0, _>`: both are arity-1 reads with no type
    // error, but `In(0)` would hand the kernel `x` where it expects the
    // bias table.
    bias: Bank<0, T>,
    topk_idx: In<1, i32>,
    // `Param<0, i32>`, not the region beside it: `Region::stride` equals
    // its width today, so a view here would compare a number against
    // itself.
    out_stride: Param<0, i32>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // A bare product; a zero is caught only by `rms`'s grid.
    let num_routes = topk_idx.rows.saturating_mul(topk_idx.width);
    let dst = out.all("the bias column count")?;
    // Typed here so the comparison below can't swap pitch and width.
    let out_stride = Stride(*out_stride);
    // The bound the kernel itself never checks: without this, a bias wider
    // than the destination's pitch runs the add off each row's end.
    if dst.width > *out_stride {
        return Err(Refusal::Wide {
            what: "the bias column count against the destination's row pitch",
            at: i64::from(dst.width),
            max: i64::from(*out_stride),
        });
    }
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "moe/moe_dispatch.cuh",
            &format!("::pie::moe::add_moe_route_bias<{}>", T::CPP),
            rms(num_routes.unsigned_abs()),
            &[
                dst.ptr.arg(),
                bias.ptr.arg(),
                topk_idx.ptr.arg(),
                num_routes.arg(),
                dst.width.arg(),
                out_stride.0.arg(),
            ],
        )
    }
}

/// The aligned MoE path's block size for one forward, from that batch's
/// route and expert counts.
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

/// `moe/expert_offsets.cuh` — the CUTLASS fused MoE's routing front-end,
/// carried and fired by nothing.
///
/// Exists only so `tests/every_carried_file_is_reachable.rs` can find the
/// file; kept rather than deleted, like `src/tile.rs`'s five.
pub static EXPERT_OFFSETS_ROOT: Root = Root::new("moe/expert_offsets.cuh");

/// `moe::flashinfer_cutlass_moe_bf16` — the fused routed block, retired.
///
/// Still nameable in a trace, but nothing runs it: carrying CUTLASS for
/// NVRTC would cost a ~13.9MB `include_str!` closure. The aligned leg
/// replaced it, selected by `moe_cutlass_max_rows = 0`.
///
/// # Errors
///
/// Always. See above.
#[kernels_macros::routine]
pub fn flashinfer_cutlass_moe_bf16(
    _ctx: &Ctx,
    _x: *const bf16,
    _experts: *const c_void,
    _weights: *const c_void,
    _out: *mut bf16,
    // Always `Err`, so nothing ever reads this column.
    _tokens: i32,
    _hidden: i32,
) -> Result<(), Refusal> {
    Err(Refusal::Absent {
        what: "the fused CUTLASS MoE leg, retired with its instantiation seam \
               rather than carried: the aligned leg is the only leg left, and \
               `moe_cutlass_max_rows = 0` is what selects it",
    })
}

const _: () = {
    assert!(<build_moe_ptrs_aligned_bf16 as ::kernels::Derivation>::DERIVED.len() == 18);
    assert!(matches!(<build_moe_ptrs_aligned_bf16 as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 0))));
    assert!(matches!(<build_moe_ptrs_aligned_bf16 as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 1))));
    assert!(matches!(<build_moe_ptrs_aligned_bf16 as ::kernels::Derivation>::DERIVED[3].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(<build_moe_ptrs_aligned_bf16 as ::kernels::Derivation>::DERIVED[6].source, Some(kernels::Source::Slot(kernels::Kind::Out, 2))));
    assert!(<build_moe_ptrs_aligned_bf16 as ::kernels::Derivation>::DERIVED[7].source.is_none());
    assert!(<build_moe_ptrs_aligned_bf16 as ::kernels::Derivation>::DERIVED[12].source.is_none());
    assert!(<build_moe_ptrs_aligned_bf16 as ::kernels::Derivation>::DERIVED[16].source.is_none());
    assert!(<build_moe_ptrs_aligned_bf16 as ::kernels::Derivation>::DERIVED[17].source.is_none());
    // `source.is_none()` above holds for the six pointer arrays and the two
    // shared-expert bases: nothing supplies any of them.

    assert!(<moe_grouped_gemm as ::kernels::Derivation>::DERIVED.len() == 6);
    assert!(kernels::source_is_named(&<moe_grouped_gemm as ::kernels::Derivation>::DERIVED[1].source, <kernels::keys::NamedWeight as kernels::keys::Fact>::KEY));
    assert!(matches!(<moe_grouped_gemm as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(<moe_grouped_gemm as ::kernels::Derivation>::DERIVED[3].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));

    // Both routers' bias slots: `Weight(0)`-sourced and non-nullable.
    assert!(!<topk_sigmoid as ::kernels::Derivation>::DERIVED[3].nullable);
    assert!(!<topk_sqrtsoftplus as ::kernels::Derivation>::DERIVED[3].nullable);
    assert!(matches!(<topk_sigmoid as ::kernels::Derivation>::DERIVED[3].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 0))));
    assert!(matches!(<topk_sqrtsoftplus as ::kernels::Derivation>::DERIVED[3].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 0))));
    assert!(<topk_sigmoid as ::kernels::Derivation>::DERIVED.len() == 6);
    assert!(<topk_sqrtsoftplus as ::kernels::Derivation>::DERIVED.len() == 6);
    assert!(matches!(<topk_sigmoid as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<topk_sigmoid as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(<topk_sigmoid as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(kernels::source_is_named(&<topk_sigmoid as ::kernels::Derivation>::DERIVED[4].source, <kernels::keys::MoeNormTopk as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&<topk_sigmoid as ::kernels::Derivation>::DERIVED[5].source, <kernels::keys::MoeRoutedScaling as kernels::keys::Fact>::KEY));
    assert!(matches!(<topk_sqrtsoftplus as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(kernels::source_is_named(&<topk_sqrtsoftplus as ::kernels::Derivation>::DERIVED[4].source, <kernels::keys::MoeNormTopk as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&<topk_sqrtsoftplus as ::kernels::Derivation>::DERIVED[5].source, <kernels::keys::MoeRoutedScaling as kernels::keys::Fact>::KEY));

    assert!(<hash_route_lookup as ::kernels::Derivation>::DERIVED.len() == 8);
    assert!(matches!(<hash_route_lookup as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(kernels::source_is_named(&<hash_route_lookup as ::kernels::Derivation>::DERIVED[1].source, <kernels::keys::NamedWeight as kernels::keys::Fact>::KEY));
    assert!(matches!(<hash_route_lookup as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(<hash_route_lookup as ::kernels::Derivation>::DERIVED[2].stated);
    assert!(kernels::source_is_named(&<hash_route_lookup as ::kernels::Derivation>::DERIVED[6].source, <kernels::keys::MoeNormTopk as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&<hash_route_lookup as ::kernels::Derivation>::DERIVED[7].source, <kernels::keys::MoeRoutedScaling as kernels::keys::Fact>::KEY));
    assert!(<topk_sigmoid_bias_fp32 as ::kernels::Derivation>::DERIVED.len() == 6);
    assert!(matches!(<topk_sigmoid_bias_fp32 as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(kernels::source_is_named(&<topk_sigmoid_bias_fp32 as ::kernels::Derivation>::DERIVED[1].source, <kernels::keys::NamedWeight as kernels::keys::Fact>::KEY));
    assert!(matches!(<topk_sigmoid_bias_fp32 as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(<topk_sigmoid_bias_fp32 as ::kernels::Derivation>::DERIVED[3].source, Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(kernels::source_is_named(&<topk_sigmoid_bias_fp32 as ::kernels::Derivation>::DERIVED[4].source, <kernels::keys::MoeNormTopk as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&<topk_sigmoid_bias_fp32 as ::kernels::Derivation>::DERIVED[5].source, <kernels::keys::MoeRoutedScaling as kernels::keys::Fact>::KEY));

    assert!(<moe_align_decode as ::kernels::Derivation>::DERIVED.len() == 7);

    // `[3]` is `RowsTotal`, not `keys::Rows`: for this statement, `rows.count`
    // is the ALIGNED row count, and the wrong fact would over-run `norm_x`.
    assert!(matches!(<gather_moe_aligned_inputs as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<gather_moe_aligned_inputs as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(<gather_moe_aligned_inputs as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(kernels::source_is_named(&<gather_moe_aligned_inputs as ::kernels::Derivation>::DERIVED[3].source, <kernels::keys::RowsTotal as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&<gather_moe_aligned_inputs as ::kernels::Derivation>::DERIVED[4].source, <kernels::keys::ExpertsPerToken as kernels::keys::Fact>::KEY));
    assert!(<gather_moe_aligned_inputs as ::kernels::Derivation>::DERIVED.len() == 5);

    assert!(<reorder_moe_aligned_output as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(matches!(<reorder_moe_aligned_output as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<reorder_moe_aligned_output as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(<reorder_moe_aligned_output as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(<gather_moe_aligned_inputs as ::kernels::Derivation>::DERIVED.len() == 5);

    assert!(<topk_softmax as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(matches!(<topk_softmax as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<topk_softmax as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(<topk_softmax as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::Out, 1))));

    assert!(<apply_per_expert_scale as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(matches!(<apply_per_expert_scale as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<apply_per_expert_scale as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(kernels::source_is_named(&<apply_per_expert_scale as ::kernels::Derivation>::DERIVED[2].source, <kernels::keys::NamedWeight as kernels::keys::Fact>::KEY));
    assert!(<apply_per_expert_scale as ::kernels::Derivation>::DERIVED[0].stated);
    assert!(<apply_per_expert_scale as ::kernels::Derivation>::DERIVED[1].stated);
    assert!(<apply_per_expert_scale as ::kernels::Derivation>::DERIVED[2].stated);
    assert!(<moe_gate_up_decode_gemv as ::kernels::Derivation>::DERIVED.len() == 4);
    assert!(matches!(<moe_gate_up_decode_gemv as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<moe_gate_up_decode_gemv as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(kernels::source_is_named(&<moe_gate_up_decode_gemv as ::kernels::Derivation>::DERIVED[2].source, <kernels::keys::NamedWeight as kernels::keys::Fact>::KEY));
    assert!(matches!(<moe_gate_up_decode_gemv as ::kernels::Derivation>::DERIVED[3].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(<moe_down_decode_gemv as ::kernels::Derivation>::DERIVED.len() == 4);
    assert!(matches!(<moe_down_decode_gemv as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(kernels::source_is_named(&<moe_down_decode_gemv as ::kernels::Derivation>::DERIVED[2].source, <kernels::keys::NamedWeight as kernels::keys::Fact>::KEY));
    assert!(matches!(<moe_down_decode_gemv as ::kernels::Derivation>::DERIVED[3].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(<moe_bucket_exact as ::kernels::Derivation>::DERIVED.len() == 4);
    // All four indices: a swap here is a same-typed permutation (all
    // `[Tokens, top_k]` i32) that nothing downstream would catch.
    assert!(matches!(<moe_bucket_exact as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<moe_bucket_exact as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(<moe_bucket_exact as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(matches!(<moe_bucket_exact as ::kernels::Derivation>::DERIVED[3].source, Some(kernels::Source::Slot(kernels::Kind::Out, 2))));
    // All four `stated`, which licenses reading them by counted index.
    assert!(<moe_bucket_exact as ::kernels::Derivation>::DERIVED[0].stated);
    assert!(<moe_bucket_exact as ::kernels::Derivation>::DERIVED[1].stated);
    assert!(<moe_bucket_exact as ::kernels::Derivation>::DERIVED[2].stated);
    assert!(<moe_bucket_exact as ::kernels::Derivation>::DERIVED[3].stated);
    assert!(<token_batched_weighted_sum as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(<token_batched_weighted_sum_add as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(matches!(<token_batched_weighted_sum as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(<token_batched_weighted_sum_add as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(<token_batched_weighted_sum as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(<token_batched_weighted_sum as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<token_batched_weighted_sum_add as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(<token_batched_weighted_sum_add as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(<token_batched_weighted_sum_add as ::kernels::Derivation>::DERIVED[1].stated);

    assert!(<add_moe_route_bias as ::kernels::Derivation>::DERIVED.len() == 4);
    // `[1]` and `[3]`: the hazard `bias`'s and `out_stride`'s own comments
    // describe — same arity, easy to mis-slot.
    assert!(matches!(<add_moe_route_bias as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 0))));
    assert!(matches!(<add_moe_route_bias as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(<add_moe_route_bias as ::kernels::Derivation>::DERIVED[3].source, Some(kernels::Source::Slot(kernels::Kind::Param, 0))));

    assert!(<scalar_weighted_add as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(matches!(<scalar_weighted_add as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(<scalar_weighted_add as ::kernels::Derivation>::DERIVED[2].source.is_none());
};

// A view moves the refusal but not the index: each width now arrives as
// `Region::width`, built from one stated operand slot. If that slot moved,
// the refusal would keep naming the same word while pointing elsewhere.
const _: () = {
    // `max_blocks * m` is the padded row count only if both stay params, not
    // operand-derived — a `Source` on either would silently change what the
    // product means.
    assert!(matches!(<moe_grouped_gemm as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(<moe_grouped_gemm as ::kernels::Derivation>::DERIVED[4].source.is_none());
    assert!(<moe_grouped_gemm as ::kernels::Derivation>::DERIVED[5].source.is_none());

    // Same reason, one routine over: swapping `aligned_act`/`aligned_out`
    // (1, 2) would succeed too, building I_moe's view over H's buffer.
    assert!(matches!(<build_moe_ptrs_aligned_bf16 as ::kernels::Derivation>::DERIVED[5].source, Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(<build_moe_ptrs_aligned_bf16 as ::kernels::Derivation>::DERIVED[13].source.is_none());
    assert!(<build_moe_ptrs_aligned_bf16 as ::kernels::Derivation>::DERIVED[14].source.is_none());

    // "The routed fanout" comes off `topk_idx`, an OUTPUT: the router writes
    // the top-k table, so its width is the fanout.
    assert!(matches!(<topk_sqrtsoftplus as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(<hash_route_lookup as ::kernels::Derivation>::DERIVED[3].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    // `moe_down_decode_gemv` takes "the route width" off input 0, the same
    // slot its gate/up twin does; only the twin was pinned.
    assert!(matches!(<moe_down_decode_gemv as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));

    // `cols` (output 0) and `out_stride` (param 0) are pinned in different
    // channels: a lowering that widened one without the other is the case
    // this bounds check exists for.
    assert!(matches!(<add_moe_route_bias as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
};

/// This family's routine table — every launcher above, as the bind side
/// reaches them.
pub static ROUTINES: &[Routine] = &[
    routine!(topk_sigmoid_bf16 = topk_sigmoid::<bf16>, ),
    routine!(topk_sqrtsoftplus_bf16 = topk_sqrtsoftplus::<bf16>, ),
    // `driver_bound!`, not `routine!`: the fused leg's operands are a
    // workspace, an arch probe and a tactic cache, never derived.
    driver_bound!(flashinfer_cutlass_moe_bf16),
    routine!(hash_route_lookup, ),
    routine!(topk_softmax_bf16 = topk_softmax::<bf16>, ),
    routine!(topk_sigmoid_bias_fp32, ),
    routine!(apply_per_expert_scale_bf16 = apply_per_expert_scale::<bf16>, in_place = &[(0, 1)], ),
    routine!(moe_grouped_gemm_bf16 = moe_grouped_gemm::<bf16>, in_place = &[(0, 2)], ),
    routine!(moe_gate_up_decode_gemv_bf16 = moe_gate_up_decode_gemv::<bf16>, ),
    routine!(moe_down_decode_gemv_bf16 = moe_down_decode_gemv::<bf16>, ),
    routine!(transpose_expert_scales_u8, ),
    routine!(build_moe_ptrs_aligned_bf16, whole, ),
    routine!(reorder_moe_aligned_output_bf16 = reorder_moe_aligned_output::<bf16>, whole, ),
    routine!(moe_align_decode, whole, ),
    routine!(moe_bucket_exact, whole, ),
    routine!(gather_moe_aligned_inputs_bf16 = gather_moe_aligned_inputs::<bf16>, whole, ),
    routine!(token_batched_weighted_sum_bf16 = token_batched_weighted_sum::<bf16>, ),
    routine!(token_batched_weighted_sum_add_bf16 = token_batched_weighted_sum_add::<bf16>, in_place = &[(0, 2)], ),
    routine!(scalar_weighted_add_bf16 = scalar_weighted_add::<bf16>, ),
    // The kernel accumulates, so result 0 must be input 0's buffer;
    // `tests/stated_columns.rs`'s `DIVERGED_IN_PLACE` holds this.
    routine!(add_moe_route_bias_bf16 = add_moe_route_bias::<bf16>, whole, in_place = &[(0, 0)], ),
];

/// `moe`, as a trace names it.
pub static FAMILY: Family = crate::family!(ROUTINES);

// `apply_per_expert_scale`'s `topk_w` is `InOut<1, f32>`, pinned since
// `Out<0, ..>` would silently switch the query to `out_width(0)`.
const _: () = {
    let d = <apply_per_expert_scale as kernels::Derivation>::DERIVED;
    assert!(matches!(d[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
};
