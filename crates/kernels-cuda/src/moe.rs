//! The MoE family: routing, dispatch, and the expert matmuls between them.
//!
//! Four routers pick experts per token; an alignment pass buckets and
//! gathers rows per expert; the matmuls run them; a combine pass weights
//! the results back onto the tokens.

use kernels::{Bind, Fire};
use kernels_macros::routine;
use crate::jit::{Ctx, Launch, Root, aligned16};
use crate::jit::Abi;
use crate::jit::abi::Tensor;
use crate::jit::abi::{MaybeConst, bf16};
use kernels::routine::{Asks, Const, In, InOut, Out, Region, Stride};
// `keys` is imported as the module, not the facts inside it: `stated_source`
// only emits a source when the path's second-to-last segment is `keys`.
use kernels::keys;
use kernels::Refusal;
// `#[routine]` is spelled out in full at each use: attribute
// macros and `macro_rules!` share a namespace, so importing it would collide
// with `use crate::{untraced, routine}` above.

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
#[routine(bf16)]
pub fn topk_sigmoid<T>(
    ctx: &Ctx<'_>,
    logits: In<Tensor<T>>,
    topk_idx: Out<Tensor<i32>>,
    topk_w: Out<Tensor<f32>>,
    // NULL AT RUNTIME when the router has no per-expert bias, which is most
    // routers; deepseek-style routers set it. `dsl::cuda::topk_sigmoid`
    // states no `weights:` at all, so the chain's `Weight(0)` half refuses on
    // every real fire and the null binds — and `MaybeConst` is the carrier
    // that says so, which is what keeps the statement from reading one
    // operand short. `Provenance::Either` was this claim before the marks.
    correction_bias: Option<Const<Tensor<f32>>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let renormalize = ctx.ask::<bool, keys::MoeNormTopk>()?;
    let routed_scaling_factor = ctx.ask::<f32, keys::MoeRoutedScaling>()?;

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

/// `moe::topk_sqrtsoftplus_bf16` — DeepSeek-V4's sqrt-softplus router.
///
/// # Safety
///
/// As [`topk_sigmoid`].
#[routine(bf16)]
pub fn topk_sqrtsoftplus<T>(
    ctx: &Ctx<'_>,
    logits: In<Tensor<T>>,
    topk_idx: Out<Tensor<i32>>,
    topk_w: Out<Tensor<f32>>,
    // As [`topk_sigmoid`]'s, and null for the same reason.
    correction_bias: Option<Const<Tensor<f32>>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let renormalize = ctx.ask::<bool, keys::MoeNormTopk>()?;
    let routed_scaling_factor = ctx.ask::<f32, keys::MoeRoutedScaling>()?;

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

/// `moe::hash_route_lookup` — DeepSeek-V4's hashed expert sets.
///
/// # Safety
///
/// `token_ids` is `[tokens]` i32 in `[0, vocab_size)`; `tid2eid`
/// `[vocab_size, top_k]` i64; `logits` `[tokens, num_experts]` bf16;
/// `topk_idx`/`topk_w` writable `[tokens, top_k]`. Only `tokens` is bounded.
#[routine]
pub fn hash_route_lookup(
    ctx: &Ctx<'_>,
    token_ids: In<Tensor<i32>>,
    // `Weight<0, _>`, not `WeightNamed`: they agree today, but would diverge
    // for a `scale.`-prefixed name, which `WeightNamed` refuses.
    tid2eid: Const<Tensor<i64>>,
    logits: In<Tensor<bf16>>,
    topk_idx: Out<Tensor<i32>>,
    topk_w: Out<Tensor<f32>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let vocab_size = ctx.ask::<i32, keys::Vocab>()?;
    let renormalize = ctx.ask::<bool, keys::MoeNormTopk>()?;
    let routed_scaling_factor = ctx.ask::<f32, keys::MoeRoutedScaling>()?;

             const DSV4_BLOCK: u32 = 256;

    // `token_ids` gets no view: its rows reach the grid axis directly and
    // its width is never read.
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

/// `moe::topk_softmax_bf16` — the softmax router's BLOCK form.
///
/// # Safety
///
/// `logits` addresses `logits.rows * logits.width` live elements and
/// `topk_idx` / `topk_w` `topk_idx.rows * topk_idx.width` writable ones.
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

/// `moe::topk_sigmoid_bias_fp32` — sigmoid routing with the correction bias
/// in fp32.
///
/// # Safety
///
/// `logits` addresses `tokens * num_experts` live floats; `correction_bias`
/// must be `num_experts` live floats and never null — null here is a fault.
#[routine]
pub fn topk_sigmoid_bias_fp32(
    ctx: &Ctx<'_>,
    logits: In<Tensor<f32>>,
    correction_bias: Const<Tensor<f32>>,
    topk_idx: Out<Tensor<i32>>,
    topk_w: Out<Tensor<f32>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let normalize = ctx.ask::<bool, keys::MoeNormTopk>()?;
    let routed_scaling_factor = ctx.ask::<f32, keys::MoeRoutedScaling>()?;

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
                // This kernel takes `int` where its cousins take `bool`;
                // `i32::from` doesn't deref like `.arg()` does, hence `**`.
                i32::from(normalize).arg(),
                routed_scaling_factor.arg(),
            ])
}

/// `moe::apply_per_expert_scale_bf16` — folds a per-expert scale into the
/// router weights, in place.
///
/// # Safety
///
/// `topk_idx` and `topk_w` each address `topk_w.rows * topk_w.width` live
/// elements, and `per_expert_scale` one per expert named by any of them.
#[routine(bf16)]
pub fn apply_per_expert_scale<T>(
    ctx: &Ctx<'_>,
    topk_idx: In<Tensor<i32>>,
    // Aliases result 0 via `in_place = &[(0, 1)]`, but reads as `In<1, ..>`:
    // `Out<0, ..>` would silently switch the query to `out_width(0)`.
    topk_w: InOut<Tensor<f32>>,
    // Positional, not named: this parameter's `scale.`-prefixed name is one
    // `WeightNamed` would refuse (a dangling sentinel), yet the kernel reads
    // it as a real per-expert array.
    per_expert_scale: Const<Tensor<T>>,
    // Route count is `topk_w.rows * topk_w.width` (tokens * top_k), not
    // `topk_w.rows` alone — `topk_w` is `[Tokens, top_k]`.


) -> Result<(), Refusal> {
    let total = topk_w.rows.saturating_mul(topk_w.width);
    ctx.fire(Fire::at("moe/topk_softmax.cuh", crate::jit::symbol(&format!("::pie::moe::apply_per_expert_scale<{}>", T::CPP))).apply(elementwise(total.unsigned_abs())), &[topk_idx.arg(), topk_w.arg(), per_expert_scale.arg(), total.arg()])
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
#[routine(bf16, driver)]
pub fn moe_grouped_gemm<T>(
    ctx: &Ctx<'_>,
    a: In<Tensor<T>>,
    // Named, not positional: this driver op reads the bank via the resolver,
    // not the flat arg run.
    weight_base: Const<Tensor<T>>,
    expert_ids: In<Tensor<i32>>,
    // THIRD OPERAND AND FIRST RESULT, which is what `in_place = &[(0, 2)]`
    // said: the statement places `stage` as input 2 and declares the gemm's
    // destination on top of it. The launcher takes the address once.
    //
    // N is `c.width`; the row count (`max_blocks * block_size`) is this
    // launcher's own product, so `over()` builds the view rather than
    // reading it off `c`.
    c: InOut<Tensor<T>>,
    // TWO PLAIN `i32`s AT HEAD -- neither a mark nor a key -- and the
    // migration gave them `keys::MoeMaxBlocks`/`keys::MoeAlignedRows`, which
    // no driver answers, so this GEMM refused `Unstated` every time. They are
    // the alignment's own product and the statement's to carry.
    //
    // Adjacent, same-typed and swappable with no type error: `m` is the
    // ALIGNED row count and `max_blocks` the block ceiling.
    max_blocks: Const<i32>,
    m: Const<i32>) -> Result<(), Refusal> {
    // `supported` lets N=0/K=0 through (both divide evenly), and a
    // zero-extent grid launches nothing rather than panicking; these two
    // views catch that instead.
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
#[routine(bf16)]
pub fn moe_gate_up_decode_gemv<T>(
    ctx: &Ctx<'_>,
    topk_idx: In<Tensor<i32>>,
    norm_x: In<Tensor<T>>,
    gate_up_base: Const<Tensor<T>>,
    expert_gate_up: Out<Tensor<T>>,
    // `topk_idx.width` is the route width (a width, not a row count);
    // `expert_gate_up.width` arrives undivided — [`per_route`] divides it.



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

/// `moe::transpose_expert_scales_u8` — the MXFP4 group-scale relayout.
///
/// # Safety
///
/// `src` and `dst` are `num_experts * n * k_groups` bytes each and must not
/// overlap: the kernel writes `dst[e][j][i]` from `src[e][i][j]`.
#[routine]
pub fn transpose_expert_scales_u8(
    ctx: &Ctx<'_>,
    // Unbound today; `Weight<0, _>` since this statement's one bank is
    // named, not positional.
    src: Const<Tensor<u8>>,
    dst: Out<Tensor<u8>>,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    num_experts: i32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    n: i32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    k_groups: i32) -> Result<(), Refusal> {
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

/// `moe::build_moe_ptrs_aligned_bf16` — fills the six pointer arrays a pair
/// of batched GEMMs index through.
///
/// # Safety
///
/// The six pointer arrays are device arrays of at least `max_blocks`
/// pointers each; `shared_gate_up_base`/`shared_down_base` may be null (the
/// rewrite above is what makes that safe).
#[routine(whole, untraced, driver)]
pub fn build_moe_ptrs_aligned_bf16(
    ctx: &Ctx<'_>,
    // Driver-dispatched: `operand()` never validates this against the
    // statement, so the order (inputs, outputs, weights) is kept correct by
    // hand.
    expert_ids: In<Tensor<i32>>,
    // Positional, not named: this driver op reads both banks from the flat
    // `args` run at a literal index.
    gate_up_base: Const<Tensor<bf16>>,
    down_base: Const<Tensor<bf16>>,
    aligned_in: In<Tensor<bf16>>,
    // Three staging buffers: `[aligned, 2*I]`, `[aligned, I]`, `[aligned,
    // H]`. Only outputs 1 and 2 carry a width view; output 0's width
    // (`2 * i_moe`) is computed below.
    aligned_gate_up: Out<Tensor<bf16>>,
    aligned_act: Out<Tensor<bf16>>,
    aligned_out: Out<Tensor<bf16>>,
    // Driver-owned workspace, not trace values: Env<the batched-cuBLAS fallback, keys::Unstated>
    // reads these from inside `moe_grouped_gemm_bf16`'s own body, so
    // declaring them as results would free them (liveness) too early.
    a_gu_ptrs: *mut *const bf16,
    b_gu_ptrs: *mut *const bf16,
    c_gu_ptrs: *mut *mut bf16,
    a_dn_ptrs: *mut *const bf16,
    b_dn_ptrs: *mut *const bf16,
    c_dn_ptrs: *mut *mut bf16,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    max_blocks: i32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    block_size: i32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    routed_blocks: i32,
    // Null when the text has no shared expert; `Unbound`, not a slot, since
    // they're unplaceable rather than mis-numbered.
    shared_gate_up_base: *const bf16,
    shared_down_base: *const bf16) -> Result<(), Refusal>
{
    let (a_gu_ptrs, b_gu_ptrs, c_gu_ptrs) = (a_gu_ptrs, b_gu_ptrs, c_gu_ptrs);
    let (a_dn_ptrs, b_dn_ptrs, c_dn_ptrs) = (a_dn_ptrs, b_dn_ptrs, c_dn_ptrs);
    // A zero width wouldn't panic: every pointer-array entry would alias the
    // base address instead, and both GEMMs would read it as correct.
    let aligned_rows = max_blocks.saturating_mul(block_size);
    let hidden = aligned_out.over(aligned_rows, "H, the hidden size")?;
    let inter = aligned_act.over(aligned_rows, "I_moe, the per-expert intermediate size")?;
    let (h, i_moe) = (hidden.width, inter.width);
    let (shared_gate_up_base, shared_down_base) =
        (shared_gate_up_base, shared_down_base);
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
/// `moe::reorder_moe_aligned_output_bf16` — scatters an aligned GEMM's
/// output rows back to their routes.
///
/// # Safety
///
/// `aligned_out` is `[aligned_rows, hidden]` bf16, `sorted_route_ids`
/// `[aligned_rows]` i32, `route_out` writable for `[num_routes, hidden]`
/// bf16.
#[routine(bf16, whole)]
pub fn reorder_moe_aligned_output<T>(
    ctx: &Ctx<'_>,
    aligned_out: In<Tensor<T>>,
    sorted_route_ids: In<Tensor<i32>>,
    route_out: Out<Tensor<T>>) -> Result<(), Refusal>
where
    // BOTH CARRIERS ARE ADDRESSES HERE: the alignment probe casts them and the
    // launch spends them, and only a pointee that says so can do either.
    *const T: Abi + kernels::Bind<crate::jit::ArgValue>,
    // AND THE WRITE SIDE NAMES ITS OWN TRAIT. `Out`/`InOut` bind through
    // `BindMut`, not `Bind`, so that a plane whose read and write carriers
    // are ONE TYPE can still say which way a slot is driven -- see
    // `kernels::routine::BindMut`. Here the two carriers already differ, so
    // the blanket impl over `*mut T` makes this the same obligation twice;
    // it is spelled because `<T as Elem>::Write` is opaque under a generic
    // `T` and the compiler cannot see that it is this pointer.
    *mut T: Abi + kernels::Bind<crate::jit::ArgValue>,
    T: kernels::Elem<Write = *mut T>,
    <T as kernels::Elem>::Read: Into<*const T>,
    <T as kernels::Elem>::Write: Into<*mut T>,
{
    /// A read carrier as the untyped address the alignment probe wants.
    ///
    /// `Elem::Read` is an associated type, so a body cannot `.cast()` it: the
    /// bound above says it CONVERTS to `*const T`, which is the one thing a
    /// pointee's read carrier is guaranteed to be on this plane.
    fn ptr_of<T>(p: impl Into<*const T>) -> *const c_void {
        p.into().cast()
    }

    /// The same for a write carrier.
    fn ptr_of_mut<T>(p: impl Into<*mut T>) -> *const c_void {
        p.into().cast_const().cast()
    }

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

/// `moe::moe_align_decode` — the block-padded counting sort: buckets routes
/// by expert into blocks.
///
/// # Safety
///
/// `topk_idx` is `[num_routes]` i32 in `[0, num_experts)`; `sorted_route_ids`
/// and `route_to_aligned_row` are writable `[num_routes]`, `expert_ids`
/// `[max_blocks]`. `block_size * max_blocks` is the padded row count.
#[routine(whole)]
pub fn moe_align_decode(
    ctx: &Ctx<'_>,
    topk_idx: In<Tensor<i32>>,
    sorted_route_ids: Out<Tensor<i32>>,
    expert_ids: Out<Tensor<i32>>,
    route_to_aligned_row: Out<Tensor<i32>>,
    num_experts: Const<i32>,
    // THE TWO THE ALIGNMENT TAKES, WHICH WERE `Param<1>` AND `Param<2>`. A
    // block size and a block ceiling are the statement's own -- two fires of
    // one deployment align the same way -- so both fail `ask`'s test and no
    // driver answers `keys::MoeBlockSize` or `keys::MoeMaxBlocks`.
    block_size: Const<i32>,
    max_blocks: Const<i32>,
    // Always null: no caller supplies a scratch buffer, and the kernel
    // guards every write to it on non-null.


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

// The params run is its own index space: a param wrapper that advanced the
// operand counters would let a bare pointer silently bind one slot along.
// Checked by `cargo check` alone.
const _: () = {
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(moe_align_decode);
    // SEVEN, NOT FIVE: the block size and the block ceiling came back onto the
    // signature as `Const<i32>`. They were `Param<1>`/`Param<2>` at HEAD and
    // the migration made them asks no driver answers, so this routine could
    // not fire at all.
    assert!(d.len() == 7);
    assert!(matches!(d[5], Some(kernels::Source::Slot(kernels::Kind::Param, 1))));
    assert!(matches!(d[6], Some(kernels::Source::Slot(kernels::Kind::Param, 2))));
    assert!(matches!(d[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(d[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(d[2], Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(matches!(d[3], Some(kernels::Source::Slot(kernels::Kind::Out, 2))));
    assert!(matches!(d[4], Some(kernels::Source::Slot(kernels::Kind::Param, 0))));
    // Always null: no caller supplies a scratch counter, and nothing binds
    // one, so the column's last entry is the empty source.
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.
};

/// `moe::moe_bucket_exact` — the unpadded sort: exact per-expert counts, no
/// block padding.
///
/// # Safety
///
/// `topk_idx` is `[num_routes]` i32 in `[0, num_experts)`; `sorted_route_ids`
/// and `route_to_sorted_row` are writable `[num_routes]`, `counts_out`
/// `[num_experts]`. An out-of-range id indexes past the shared slab.
#[routine(whole)]
pub fn moe_bucket_exact(
    ctx: &Ctx<'_>,
    topk_idx: In<Tensor<i32>>,
    sorted_route_ids: Out<Tensor<i32>>,
    route_to_sorted_row: Out<Tensor<i32>>,
    counts_out: Out<Tensor<i32>>) -> Result<(), Refusal> {
    // A bare product: the grid is a literal `[1, 1, 1]`, so nothing catches
    // a zero-route sort; only the expert count has a guard, via the view.
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

/// `moe::gather_moe_aligned_inputs_bf16` — gathers token rows into the
/// aligned, block-padded rectangle.
///
/// # Safety
///
/// `norm_x` is `[num_tokens, hidden]` bf16, `sorted_route_ids`
/// `[aligned_rows]` i32, `aligned_in` writable for `[aligned_rows, hidden]`
/// bf16.
#[routine(bf16, whole)]
pub fn gather_moe_aligned_inputs<T>(
    ctx: &Ctx<'_>,
    norm_x: In<Tensor<T>>,
    sorted_route_ids: In<Tensor<i32>>,
    aligned_in: Out<Tensor<T>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let top_k = ctx.ask::<i32, keys::ExpertsPerToken>()?;

    let tokens = ctx.ask::<i32, keys::RowsTotal>()?;
    let (aligned_rows, hidden) = (sorted_route_ids.rows, aligned_in.width);
    // `Env<T>` derefs to `T` and the key derefs to its value: two derefs
    // down.
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
                // No shared tail for this leg either.
                (-1i32).arg(),
                num_tokens.arg(),
            ])
}

/// `moe::token_batched_weighted_sum_bf16` — the combine: weighted sum of
/// each token's top-k expert outputs.
///
/// # Safety
///
/// `src` is `[num_tokens, top_k, hidden]` bf16, `weights` `[num_tokens,
/// top_k]` f32, `out` writable for `[num_tokens, hidden]` bf16.
#[routine(bf16)]
pub fn token_batched_weighted_sum<T>(
    ctx: &Ctx<'_>,
    out: Out<Tensor<T>>,
    src: In<Tensor<T>>,
    weights: In<Tensor<f32>>,
    // Off `weights` (operand 1), not `out`/`src`: the fanout rides in as
    // the router's own weight vector.


) -> Result<(), Refusal> {
    // Only `weights` has a guard; `out`'s numbers reach the grid axis
    // directly.
    let fan = weights.all("the routed fanout")?;
    let top_k = fan.width;
    ctx.fire(Fire::at("moe/moe_dispatch.cuh", crate::jit::symbol(&format!("::pie::moe::token_batched_weighted_sum<{}>", T::CPP))).apply(elementwise_rows(out.rows.unsigned_abs(), out.width.unsigned_abs())), &[out.arg(), src.ptr.arg(), fan.ptr.arg(), top_k.arg(), out.width.arg()])
}

/// `moe::token_batched_weighted_sum_add_bf16` — the same combine,
/// accumulating into `out` rather than overwriting it.
///
/// # Safety
///
/// As [`token_batched_weighted_sum`], and `out` is read as well as
/// written.
#[routine(bf16)]
pub fn token_batched_weighted_sum_add<T>(
    ctx: &Ctx<'_>,
    src: In<Tensor<T>>,
    weights: In<Tensor<f32>>,
    // LAST, AND THAT IS THE INDEX. `in_place = &[(0, 2)]` used to state this
    // beside the row: result 0 is the statement's third input, the residual
    // this launch folds into. The mark says it now, and it says it by SITTING
    // THIRD -- the slot is the position among the operand marks.
    out: InOut<Tensor<T>>) -> Result<(), Refusal> {
    let fan = weights.all("the routed fanout")?;
    let top_k = fan.width;
    ctx.fire(Fire::at("moe/moe_dispatch.cuh", crate::jit::symbol(&format!("::pie::moe::token_batched_weighted_sum_add<{}>", T::CPP))).apply(elementwise_rows(out.rows.unsigned_abs(), out.width.unsigned_abs())), &[out.arg(), src.ptr.arg(), fan.ptr.arg(), top_k.arg(), out.width.arg()])
}

/// `moe::scalar_weighted_add_bf16` — `out += weight * src` over a flat run.
///
/// # Safety
///
/// `out` and `src` each address `n` live elements and may alias exactly
/// (`in_place: &[(0, 0)]`); `out` is read as well as written.
#[routine(bf16, internal)]
pub fn scalar_weighted_add<T>(
    ctx: &Ctx<'_>,
    out: Out<Tensor<T>>,
    src: In<Tensor<T>>,
    // The residual scale, caller-chosen, with no arm or fire behind it:
    // the statement that fires this row states the value itself, so it
    // claims the next params-run slot rather than asking for one.
    weight: Const<f32>,
    // `n` is `out.rows * out.width`; valid off either operand since the
    // device row aliases both exactly (`in_place: &[(0, 0)]`).
) -> Result<(), Refusal> {
    let n = out.rows.saturating_mul(out.width);
    ctx.fire(Fire::at("moe/moe_dispatch.cuh", crate::jit::symbol(&format!("::pie::moe::scalar_weighted_add<{}>", T::CPP))).apply(elementwise(n.unsigned_abs())), &[out.arg(), src.ptr.arg(), weight.arg(), n.arg()])
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
#[routine(bf16, whole)]
pub fn add_moe_route_bias<T>(
    ctx: &Ctx<'_>,
    out: InOut<Tensor<T>>,
    // `Weight<0, *const _>`, not `In<0, *const _>`: both are arity-1 reads with no type
    // error, but `In(0)` would hand the kernel `x` where it expects the
    // bias table.
    bias: Const<Tensor<T>>,
    topk_idx: In<Tensor<i32>>,
    // THE STATEMENT'S PITCH, WHICH WAS `Param<0, i32>`. No driver answers
    // `keys::OutStride`, so this fired `Unstated` while it was an ask.
    out_stride: Const<i32>) -> Result<(), Refusal> {
    // A bare product; a zero is caught only by `rms`'s grid.
    let num_routes = topk_idx.rows.saturating_mul(topk_idx.width);
    let dst = out.all("the bias column count")?;
    // Typed here so the comparison below can't swap pitch and width.
    let out_stride = Stride(*out_stride);
    // The bound the kernel itself never checks: without this, a bias wider
    // than the destination's pitch runs the add off each row's end.
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
#[routine(untraced)]
pub fn flashinfer_cutlass_moe_bf16(
    _ctx: &Ctx<'_>,
    _x: *const bf16,
    _experts: *const c_void,
    _weights: *const c_void,
    _out: *mut bf16,
    // Always `Err`, so nothing ever reads this column.
    _tokens: i32,
    _hidden: i32) -> Result<(), Refusal> {
    Err(Refusal::Absent {
        what: "the fused CUTLASS MoE leg, retired with its instantiation seam \
               rather than carried: the aligned leg is the only leg left, and \
               `moe_cutlass_max_rows = 0` is what selects it",
    })
}

const _: () = {
    assert!(<build_moe_ptrs_aligned_bf16 as ::kernels::Derivation>::DERIVED.len() == 18);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(build_moe_ptrs_aligned_bf16)[1], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(build_moe_ptrs_aligned_bf16)[2], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 1)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(build_moe_ptrs_aligned_bf16)[3], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(build_moe_ptrs_aligned_bf16)[6], Some(kernels::Source::Slot(kernels::Kind::Out, 2))));
    assert!(kernels::routine::sources::<crate::jit::Cuda, _, _>(build_moe_ptrs_aligned_bf16)[7].is_none());
    assert!(kernels::routine::sources::<crate::jit::Cuda, _, _>(build_moe_ptrs_aligned_bf16)[12].is_none());
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.
    // `source.is_none()` above holds for the six pointer arrays and the two
    // shared-expert bases: nothing supplies any of them.

    // SIX: the block ceiling and the aligned row count came back onto the
    // signature. They were bare `i32` arguments at HEAD -- no mark, no key
    // -- and the migration invented asks for them.
    assert!(<moe_grouped_gemm as ::kernels::Derivation>::DERIVED.len() == 6);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(moe_grouped_gemm::<bf16>)[1], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(moe_grouped_gemm::<bf16>)[2], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(moe_grouped_gemm::<bf16>)[3], Some(kernels::Source::Alias(2, 0))));

    // BOTH ROUTERS' BIAS SLOTS ARE NULLABLE, and the assertion is inverted
    // from what it used to pin. Neither router's DSL builder states a
    // `weights:` run, so the chain refuses and the binder binds null on every
    // real fire; a column that claimed the plane was required made the arity
    // rule read one operand more than any statement places.
    assert!(<topk_sigmoid as ::kernels::Derivation>::DERIVED[3].nullable);
    assert!(<topk_sqrtsoftplus as ::kernels::Derivation>::DERIVED[3].nullable);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(topk_sigmoid::<bf16>)[3], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(topk_sqrtsoftplus::<bf16>)[3], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(<topk_sigmoid as ::kernels::Derivation>::DERIVED.len() == 4);
    assert!(<topk_sqrtsoftplus as ::kernels::Derivation>::DERIVED.len() == 4);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(topk_sigmoid::<bf16>)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(topk_sigmoid::<bf16>)[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(topk_sigmoid::<bf16>)[2], Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(topk_sqrtsoftplus::<bf16>)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));

    assert!(<hash_route_lookup as ::kernels::Derivation>::DERIVED.len() == 5);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(hash_route_lookup)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(hash_route_lookup)[1], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(hash_route_lookup)[2], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(<topk_sigmoid_bias_fp32 as ::kernels::Derivation>::DERIVED.len() == 4);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(topk_sigmoid_bias_fp32)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(topk_sigmoid_bias_fp32)[1], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(topk_sigmoid_bias_fp32)[2], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(topk_sigmoid_bias_fp32)[3], Some(kernels::Source::Slot(kernels::Kind::Out, 1))));

    assert!(<moe_align_decode as ::kernels::Derivation>::DERIVED.len() == 7);

    // `[3]` is `RowsTotal`, not `keys::Rows`: for this statement, `rows.count`
    // is the ALIGNED row count, and the wrong fact would over-run `norm_x`.
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(gather_moe_aligned_inputs::<bf16>)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(gather_moe_aligned_inputs::<bf16>)[1], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(gather_moe_aligned_inputs::<bf16>)[2], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.
    assert!(<gather_moe_aligned_inputs as ::kernels::Derivation>::DERIVED.len() == 3);

    assert!(<reorder_moe_aligned_output as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(reorder_moe_aligned_output::<bf16>)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(reorder_moe_aligned_output::<bf16>)[1], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(reorder_moe_aligned_output::<bf16>)[2], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(<gather_moe_aligned_inputs as ::kernels::Derivation>::DERIVED.len() == 3);

    assert!(<topk_softmax as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(topk_softmax::<bf16>)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(topk_softmax::<bf16>)[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(topk_softmax::<bf16>)[2], Some(kernels::Source::Slot(kernels::Kind::Out, 1))));

    assert!(<apply_per_expert_scale as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(apply_per_expert_scale::<bf16>)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(apply_per_expert_scale::<bf16>)[1], Some(kernels::Source::Alias(1, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(apply_per_expert_scale::<bf16>)[2], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(<moe_gate_up_decode_gemv as ::kernels::Derivation>::DERIVED.len() == 4);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(moe_gate_up_decode_gemv::<bf16>)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(moe_gate_up_decode_gemv::<bf16>)[1], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(moe_gate_up_decode_gemv::<bf16>)[2], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(moe_gate_up_decode_gemv::<bf16>)[3], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(<moe_down_decode_gemv as ::kernels::Derivation>::DERIVED.len() == 4);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(moe_down_decode_gemv::<bf16>)[1], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(moe_down_decode_gemv::<bf16>)[2], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(moe_down_decode_gemv::<bf16>)[3], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(<moe_bucket_exact as ::kernels::Derivation>::DERIVED.len() == 4);
    // All four indices: a swap here is a same-typed permutation (all
    // `[Tokens, top_k]` i32) that nothing downstream would catch.
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(moe_bucket_exact)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(moe_bucket_exact)[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(moe_bucket_exact)[2], Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(moe_bucket_exact)[3], Some(kernels::Source::Slot(kernels::Kind::Out, 2))));
    // All four `stated`, which licenses reading them by counted index.
    assert!(<token_batched_weighted_sum as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(<token_batched_weighted_sum_add as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(token_batched_weighted_sum::<bf16>)[2], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(token_batched_weighted_sum_add::<bf16>)[2], Some(kernels::Source::Alias(2, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(token_batched_weighted_sum::<bf16>)[0], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(token_batched_weighted_sum::<bf16>)[1], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(token_batched_weighted_sum_add::<bf16>)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(token_batched_weighted_sum_add::<bf16>)[1], Some(kernels::Source::Slot(kernels::Kind::In, 1))));

    // FOUR: the output pitch came back as `Const<i32>`, `Param<0>`
    // again, after a spell as an ask no driver answered.
    assert!(<add_moe_route_bias as ::kernels::Derivation>::DERIVED.len() == 4);
    // `[1]` and `[3]`: the hazard `bias`'s and `out_stride`'s own comments
    // describe — same arity, easy to mis-slot.
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(add_moe_route_bias::<bf16>)[1], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(add_moe_route_bias::<bf16>)[2], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.

    assert!(<scalar_weighted_add as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(scalar_weighted_add::<bf16>)[0], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // `weight` claims the params run through the float reading of the
    // channel now, rather than answering `None` as an unstated fact.
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(scalar_weighted_add::<bf16>)[2], Some(kernels::Source::Slot(kernels::Kind::ParamF32, 0))));
};

// A view moves the refusal but not the index: each width now arrives as
// `Region::width`, built from one stated operand slot. If that slot moved,
// the refusal would keep naming the same word while pointing elsewhere.
const _: () = {
    // `max_blocks * m` is the padded row count only if both stay params, not
    // operand-derived — a `Source` on either would silently change what the
    // product means.
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(moe_grouped_gemm::<bf16>)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.

    // Same reason, one routine over: swapping `aligned_act`/`aligned_out`
    // (1, 2) would succeed too, building I_moe's view over H's buffer.
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(build_moe_ptrs_aligned_bf16)[5], Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(kernels::routine::sources::<crate::jit::Cuda, _, _>(build_moe_ptrs_aligned_bf16)[13].is_none());
    assert!(kernels::routine::sources::<crate::jit::Cuda, _, _>(build_moe_ptrs_aligned_bf16)[14].is_none());

    // "The routed fanout" comes off `topk_idx`, an OUTPUT: the router writes
    // the top-k table, so its width is the fanout.
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(topk_sqrtsoftplus::<bf16>)[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(hash_route_lookup)[3], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    // `moe_down_decode_gemv` takes "the route width" off input 0, the same
    // slot its gate/up twin does; only the twin was pinned.
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(moe_down_decode_gemv::<bf16>)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));

    // `cols` (output 0) and `out_stride` (param 0) are pinned in different
    // channels: a lowering that widened one without the other is the case
    // this bounds check exists for.
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(add_moe_route_bias::<bf16>)[0], Some(kernels::Source::Alias(0, 0))));
};


// `apply_per_expert_scale`'s `topk_w` is `In<1, *mut f32>`, pinned since
// `Out<0, ..>` would silently switch the query to `out_width(0)`.
const _: () = {
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(apply_per_expert_scale::<bf16>);
    assert!(matches!(d[1], Some(kernels::Source::Alias(1, 0))));
};
