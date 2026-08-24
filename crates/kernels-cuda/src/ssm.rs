#![allow(clippy::too_many_arguments)]

use crate::jit::Abi;
use crate::jit::abi::Tensor;
use crate::jit::abi::{MaybeConst, bf16};
use crate::jit::{Ctx, Launch};
use crate::views::{MoeBanks, RecurrentState};
use kernels::Refusal;
use kernels::raises::Struct;
use kernels::routine::{Cache, Const, In, Out};
use kernels::{Bind, Fire};
use kernels_macros::routine;

use core::ffi::c_void;

const RULE_BLOCK: u32 = 256;

const WARP: u32 = 32;

const FLOAT: u32 = 4;

#[must_use]
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, RULE_BLOCK)
}

#[must_use]
fn per_head_elementwise(rows: u32, heads: u32, head_dim: u32) -> Launch {
    const SINK_BLOCK_MIN: u32 = WARP;

    const SINK_BLOCK_MAX: u32 = 128;

    Launch::grid(
        [rows, heads, 1],
        [head_dim.clamp(SINK_BLOCK_MIN, SINK_BLOCK_MAX), 1, 1],
    )
}

#[must_use]
const fn gated_rms(rows: u32, heads: u32) -> Launch {
    Launch::grid([rows, heads, 1], [RULE_BLOCK, 1, 1])
}

#[must_use]
const fn recurrent_scan(rows: u32, heads: u32, k_d: u32) -> Launch {
    const SCAN_BLOCK: u32 = 128;

    Launch::grid([rows, heads, 1], [SCAN_BLOCK, 1, 1])
        .smem(k_d.saturating_mul(2).saturating_mul(FLOAT))
}

#[must_use]
const fn warp_tiled_scan(rows: u32, heads: u32, value_width: u32) -> Launch {
    const SCAN_WARPS: u32 = 4;

    Launch::grid(
        [rows, heads, value_width.div_ceil(SCAN_WARPS)],
        [SCAN_WARPS * WARP, 1, 1],
    )
}

#[must_use]
const fn kda_shmem(d: u32) -> u32 {
    3u32.saturating_mul(d).saturating_mul(FLOAT)
}

const PTRS_BLOCK: u32 = 256;

const GDN_BLOCK: u32 = 128;

/// The claimed `Ssm` points are spelled at bf16 and NOWHERE ELSE, and each
/// one has its own reason.
///
/// The two conv routines are `#[routine(bf16, ..)]` because of their
/// signatures: the optional bias rides `MaybeConst<T>`, whose `Abi` impl
/// `jit::abi` writes one pointee at a time. The chunked recurrence is pinned
/// because its PROLOGUE is — `qwen_gdn_qk_norm` and `qwen_gdn_v_gates` are
/// fired at `::pie::bf16` by literal symbols, and the packed row is the only
/// operand whose element the point quantifies over.
///
/// A point quantifies over `Scalar`, so a claim states the pin as a refusal
/// BY NAME rather than widening it with a cast no kernel stands behind — the
/// `gate.sigmoid_mul` precedent. A second element wants a second spelling of
/// the launcher, not a cast here.
fn at_bf16<T: kernels::points::Scalar>(what: &'static str) -> Result<(), Refusal> {
    if T::CPP == <bf16 as kernels::Elem>::CPP {
        Ok(())
    } else {
        Err(Refusal::Absent { what })
    }
}

/// The conv's two numbers, as the routines ask for them.
///
/// `c` is the CHANNEL count and it is read, not stated: a depthwise conv
/// runs one channel per column, so the channel count IS the operand's row.
/// `k` is the kernel width and it is stated, because it lives in the
/// `[channels, width]` weight and a `Const` carries an address with no
/// rectangle behind it.
fn conv_shape<T: kernels::Elem>(x: In<Tensor<T>>, conv_width: u32) -> Result<(i32, i32), Refusal> {
    let rect = x.all("the conv's channel count")?;
    let k = i32::try_from(conv_width).map_err(|_| Refusal::Wide {
        what: "the conv width this statement states",
        at: i64::from(conv_width),
        max: i64::from(i32::MAX),
    })?;
    Ok((rect.width, k))
}

/// The `Ssm` family, claimed. Five of seven points land; the other two are
/// measured backlog rows, and each absence is a stated one.
///
/// # The three GDN points are one seam, and W10 finished cutting it
///
/// * `ssm.gdn_prep` — CLAIMED, one launch. It used to resolve through
///   `qwen_gdn_post_conv_prep_bf16`, which is a different statement wearing
///   this point's name: that routine takes the post-conv `qkv` this
///   declaration has no slot for, writes FIVE rectangles where the
///   statement states one, and asks for five geometry scalars on top. An
///   executor firing it had to reach backwards through the plan for the
///   missing operand and carve three scratch columns for the missing
///   results — and hand over the two halves of `[b | a]` and of
///   `[g_log | beta]` as pointer offsets into packed rows, which is a row
///   stride of `v_heads` claimed for bytes whose stride is `2 * v_heads`.
///   `qwen_gdn_ba_gates` is that arithmetic with the packing kept: the
///   projection in as the matmul wrote it, the decay row out as the two
///   recurrences read it. Nothing is staged and nothing is cut.
/// * `ssm.gated_delta` — CLAIMED, and structurally identical to the
///   chunked form below: [`GdnShape::stage`] then the step scan. The five
///   compact f32 planes the scan takes are not five rows carved out of one
///   at offsets nothing declares — they are the same two prologue kernels
///   the window fires, over the same two packed rectangles the statement
///   carries.
/// * `ssm.gated_delta_chunked` — CLAIMED, and by a body rather than a
///   `canon` retag, because no single routine in this file spells it. The
///   window's prologue (the q/k l2-norm, the value widen, the fused decay
///   row's cut) and the scan are separate launches, and WHICH scan is a
///   per-fire choice on `k_dim` and `v_dim` — the branch reads
///   `kernels/ssm/gated_delta_net.cuh`, not a config flag. See
///   [`Ssm::gated_delta_chunked`]'s own body for the three arms and what
///   each one is bounded by.
///
///   This is where the l2-norm ended up and why: it cannot live on
///   `ssm.gdn_prep` at all, because that declaration carries no `qkv` slot.
///   It is the recurrence points that take the packed row, so it is the
///   recurrence points that own the norm — the chunked body said so first
///   and the step form now says it too.
/// * `ssm.kda_step` / `ssm.kda_chunked` — CLAIMED, and by bodies for the
///   same reason `gated_delta_chunked` is: no routine in this file spells
///   either. The legacy KDA leg staged NINE launches per mixer, and the new
///   text keeps the four that are statements (one packed projection, one
///   packed conv, the `f` and `b` projections) and leaves the other five
///   here — a packed-plane cut with two l2 norms and a widen in it, the
///   gate/beta cook, and the recurrence. The two points' own docs say what
///   each launch is and where it came from.
///
///   The chunked one also carries a finding: `kda_prefill_batched` had
///   never been fired by anything in this tree, and it is bit-identical to
///   a per-token loop of the decode step — the same fp32 state through the
///   same slab, the same warp-per-row reductions. So W2's
///   rounding-trajectory law holds here with no arm to choose, and
///   `tests/kda_recurrence.rs` is where that is measured rather than
///   argued.
#[kernels_macros::claims]
impl kernels::points::Ssm for Ctx<'_> {
    fn causal_conv1d<T: kernels::points::Scalar>(
        &self,
        x: In<Tensor<T>>,
        weight: Const<Tensor<T>>,
        state: Cache<kernels::raises::Struct<RecurrentState>>,
        conv_width: u32,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("ssm.causal_conv1d at an element other than bf16")?;
        let (c, k) = conv_shape(x, conv_width)?;
        causal_conv1d_update_batched::<bf16>(
            self,
            In {
                ptr: x.ptr.cast::<bf16>(),
                rows: x.rows,
                width: x.width,
            },
            Const::new(weight.v.cast::<bf16>()),
            // The statement carries ONE weight, so there is no bias plane
            // to hand over. Every shipping conv1d in this tree agrees —
            // the legacy `ConvW` states `bias: None` — and a checkpoint
            // that grew one would state a second `Const` on the point.
            None,
            Out {
                ptr: y.ptr.cast::<bf16>(),
                rows: y.rows,
                width: y.width,
            },
            Const::new(c),
            Const::new(k),
            state.raised(),
        )
    }

    fn causal_conv1d_chunked<T: kernels::points::Scalar>(
        &self,
        x: In<Tensor<T>>,
        indptr: In<Tensor<i32>>,
        weight: Const<Tensor<T>>,
        state: Cache<kernels::raises::Struct<RecurrentState>>,
        conv_width: u32,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("ssm.causal_conv1d_chunked at an element other than bf16")?;
        let (c, k) = conv_shape(x, conv_width)?;
        causal_conv1d_prefill_batched::<bf16>(
            self,
            In {
                ptr: x.ptr.cast::<bf16>(),
                rows: x.rows,
                width: x.width,
            },
            Const::new(weight.v.cast::<bf16>()),
            None,
            Out {
                ptr: y.ptr.cast::<bf16>(),
                rows: y.rows,
                width: y.width,
            },
            Const::new(c),
            Const::new(k),
            state.raised(),
            // A STATEMENT THAT NAMES A CACHE ROW LEAVES ITS TAIL THERE. A
            // chunked conv that skipped the write-back would hand the next
            // fire a stale window, so there is no second reading for the
            // point to state; the driver has never bound the flag any other
            // way either (`fire/launch.rs` builds every `Gdn` view with
            // `write_state: true`).
            Const::new(true),
            indptr,
        )
    }

    /// Qwen's gated-delta prologue: the packed `[b | a]` projection in, the
    /// packed `[g_log | beta]` decay row out.
    ///
    /// ONE LAUNCH, AND EXACTLY THE DECLARATION'S SLOTS. This point used to
    /// resolve through `qwen_gdn_post_conv_prep_bf16`, which is a different
    /// statement wearing this one's name: that routine takes the
    /// post-convolution `qkv` this declaration has no slot for, and writes
    /// five rectangles where this one states one. An executor firing it had
    /// to reach backwards through the plan for the missing operand and carve
    /// three scratch columns for the missing results — and, worse, hand the
    /// two halves of `ba` and the two halves of `gates` over as pointer
    /// offsets into packed rows, which is a row stride of `v_heads` claimed
    /// for bytes whose stride is `2 * v_heads`. True at one token, false at
    /// two, and silent either way.
    ///
    /// `qwen_gdn_ba_gates` is that arithmetic with the packing kept: it
    /// reads the projection as the matmul wrote it and writes the decay row
    /// as [`Ssm::gated_delta`] and [`Ssm::gated_delta_chunked`] read it. The
    /// three planes the old routine also wrote belong to the recurrence
    /// points, which declare the `qkv` they are cut from; both stage them
    /// now (see [`GdnShape::stage`]).
    ///
    /// # `v_heads` is read, not stated
    ///
    /// The declaration states no scalar, and it does not need to: the
    /// operand IS `[b | a]`, so the value-head count is half its width. A
    /// `Const` restating it could disagree with the rectangle it divides.
    fn gdn_prep<T: kernels::points::Scalar>(
        &self,
        ba: In<Tensor<T>>,
        dt_bias: Const<Tensor<T>>,
        a_log: Const<Tensor<f32>>,
        gates: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("ssm.gdn_prep at an element other than bf16")?;
        let src = ba.all("the `[b | a]` projection")?;
        let dst = gates.all("the fused `[g_log | beta]` row")?;
        if src.width % 2 != 0 {
            return Err(Refusal::Narrow {
                what: "the `[b | a]` projection's row, which halves into the value heads",
                at: i64::from(src.width),
            });
        }
        let v_h = src.width / 2;
        // THE RESULT IS THE OPERAND'S SHAPE ON f32 — `program.rs`'s width
        // rule says so (`sized(rows(0), width(0), Dt::F32)`) and the kernel
        // strides both by the same `2 * v_heads`, so a rectangle that
        // disagreed would be written past rather than partially.
        if dst.width != src.width || dst.rows != src.rows {
            return Err(Refusal::Narrow {
                what: "the fused `[g_log | beta]` row, against the projection it is derived from",
                at: i64::from(dst.width),
            });
        }
        const PREP_BLOCK: u32 = 256;
        self.fire(
            Fire::at(
                "ssm/gated_delta_net_prep.cuh",
                "::pie::ssm::qwen_gdn_ba_gates<::pie::bf16>",
            )
            .apply(Launch::grid(
                [
                    src.rows.unsigned_abs(),
                    v_h.unsigned_abs().div_ceil(PREP_BLOCK),
                    1,
                ],
                [PREP_BLOCK, 1, 1],
            )),
            &[
                ba.ptr.cast::<bf16>().arg(),
                a_log.v.arg(),
                dt_bias.v.cast::<bf16>().arg(),
                gates.ptr.arg(),
                src.rows.arg(),
                v_h.arg(),
            ],
        )
    }

    /// The gated-delta rule for one token per request.
    ///
    /// THE SAME BODY AS [`Ssm::gated_delta_chunked`] WITH A DIFFERENT SCAN,
    /// and that is the point of writing it: the prologue is a property of
    /// the two rectangles a statement carries, not of how many tokens are in
    /// them. `qkv` is packed by `conv_dim` and `gates` by `2 * v_heads`; the
    /// scans read five COMPACT f32 planes. [`GdnShape::stage`] is the cut, in a
    /// kernel, at any token count — where the executor's own pointer
    /// arithmetic was right only at one.
    ///
    /// The l2-norm rides here for the reason
    /// [`Ssm::gated_delta_chunked`] states: `ssm.gdn_prep` declares no `qkv`
    /// slot, so the norm cannot be its arithmetic; this point declares the
    /// packed row and the four numbers that divide it, so it owns it.
    fn gated_delta<T: kernels::points::Scalar>(
        &self,
        qkv: In<Tensor<T>>,
        z: In<Tensor<T>>,
        gates: In<Tensor<f32>>,
        state: Cache<kernels::raises::Struct<RecurrentState>>,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
        y: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        // `z` IS THE OUT-NORM'S GATE and this point does not spend it — the
        // chunked form's reading, in the same words.
        let _ = z;
        at_bf16::<T>("ssm.gated_delta at an element other than bf16")?;

        let shape = GdnShape::of(qkv, gates, k_heads, v_heads, k_dim, v_dim)?;
        // The result is `[tokens, v_heads * v_dim]` and the scan writes it
        // that way, `out[(r * v_heads + h) * v_dim + v]`.
        let result = y.all("the recurrence's result")?;
        if i64::from(result.width) != i64::from(shape.v_h) * i64::from(shape.v_d) {
            return Err(Refusal::Narrow {
                what: "the recurrence's result row, against the stated value heads",
                at: i64::from(result.width),
            });
        }
        if state.ptr.is_null() {
            return Err(Refusal::Null {
                what: "the recurrent view this statement names",
            });
        }
        let staged = shape.stage(self, qkv.ptr.cast::<bf16>(), gates.ptr)?;
        shape.step(self, &staged, state.ptr, y)
    }

    /// The gated-delta rule over a prefill window, GQA and all.
    ///
    /// # The prologue is this point's, not `gdn_prep`'s
    ///
    /// The legacy text ran the q/k l2-norm inside its prep launch, and the
    /// decode lane's staging shim still reaches BACKWARDS through the plan's
    /// dataflow to hand `qwen_gdn_post_conv_prep_bf16` a `qkv` its statement
    /// never carried (`baker-smoke/src/smoke.rs:1156-1166`). That reach is
    /// the tell. `ssm.gdn_prep` DECLARES three slots — `ba`, `dt_bias`,
    /// `a_log` — and not one of them is the packed row, so the l2-norm
    /// cannot be its arithmetic. This point declares `qkv` and the four head
    /// numbers that divide it, so the norm is HERE, and the body owns it:
    ///
    ///   1. `qwen_gdn_qk_norm` — the two key-head planes, l2-normalised and
    ///      widened, `q` carrying the `1/sqrt(k_dim)` scale the decode
    ///      prologue applies (`driver_internal.rs:101`). Compact in
    ///      `k_heads` heads, which is what the GQA scans below read.
    ///   2. `qwen_gdn_v_gates` — the value slice widened out of the packed
    ///      row, and `gates` cut into its two COMPACT halves. The cut is a
    ///      kernel and not a pointer offset: the scans index
    ///      `g_log[t * v_heads + h]` while `gates` strides by `2 * v_heads`
    ///      per token, and the two agree only at ONE token. The decode
    ///      shim used to take that offset by hand and was right for exactly
    ///      as long as its fires were one row wide.
    ///   3. one of three scans.
    ///
    /// # The scratch
    ///
    /// Named device slabs from [`Ctx::scratch`] — the same grow-on-demand
    /// arena `attn::xqa` takes its two workspaces from and `sample` its
    /// argmax pairs. A body that staged nothing could not claim this point
    /// at all; a body that asked the DRIVER to stage it would be putting
    /// five rectangles nothing declares on the operand column.
    ///
    /// A named slab is per PROCESS, not per fire, so the planes are alive
    /// only between the prologue and the scan that reads them — which is
    /// exactly the window a fire owns, since the launches are issued in
    /// order on one stream. Two fires racing the same plane would share
    /// them, and that is the property every `Ctx::scratch` caller already
    /// has.
    ///
    /// # Which scan, and why the token count is not the question
    ///
    /// The legacy text guarded its three arms on TOKENS
    /// (`GuardPred::TokensLE(64)` for the warp-tiled arm, `TokensLE(4096)`
    /// for the cached one). Neither bound is in a kernel. Every one of these
    /// scans walks its request's window with `for (int t = 0; t < T; ++t)`
    /// and no cap — the guards were a performance dial, and the shipping
    /// projection turns BOTH off (`warp_tiled: false`, `cached_max: 0`,
    /// `model-legacy/src/qwen_3_5/project.rs:233-235`), leaving the fla arm
    /// to serve every window length. What IS in the kernels is shape:
    ///
    /// * `chunk_gated_delta_prefill_batched_fla<_, 128, 128>` is GQA-NATIVE
    ///   (`h_k = h / (V_h / K_h)`, `gated_delta_net.cuh:1440-1451`), so the
    ///   `repeat_interleave` the legacy staged is not a step this arm needs
    ///   at all. Its bounds are `BK_MAX = 128 >= k_dim` and a `BV = 128`
    ///   that must DIVIDE `v_dim` — the block returns early past `v_dim`,
    ///   and the token loop's `__syncthreads()` would hang on the lanes that
    ///   left.
    /// * `chunk_gated_delta_prefill_batched_warp_tiled_gqa` is GQA-native
    ///   too and takes any `v_dim`; it is bounded by `MAX_K_PER_LANE = 8`
    ///   over 32 lanes — `k_dim <= 256` (`gated_delta_net.cuh:469`).
    /// * `chunk_gated_delta_prefill_batched` is the only one that is NOT:
    ///   it takes `V_h` alone and indexes q/k by it, which is precisely what
    ///   the legacy composite's `repeat_interleave_heads_fp32` was FOR. Past
    ///   both bounds above, this body stages that repeat itself.
    ///
    /// The order between the first two is not arbitrary. fla holds its state
    /// in bf16 registers and rounds it TWICE per token, which is what the
    /// per-token form does through HBM (`gated_delta_net.cuh:1414-1420`
    /// records the two as bit-identical) and what the decode step is pinned
    /// to (`recurrent_step_batched_gqa_smem` rounds `state * g` before
    /// adding delta ON PURPOSE — "Qwen3.5-0.8B diverged from the HF
    /// reference trajectory at the SECOND decoded token" without it,
    /// `gated_delta_net.cuh:1370-1381`). The warp-tiled arm carries fp32
    /// state across the whole window instead: more accurate per step, and
    /// agreeing with neither. A prefill whose tail is handed to the decode
    /// step must round the way the step rounds, so fla goes first and
    /// warp-tiled serves only the shapes fla cannot take.
    ///
    /// # The slot the tail lands in
    ///
    /// `state_bf16`, `KLast = false`: `slab + slot * slot_stride_elems +
    /// h * k_dim * v_dim`, then `k * v_dim + v`. That is the decode step's
    /// addressing verbatim (`gated_delta_net.cuh:1291-1293, 1345`) and the
    /// `[v_heads, k_dim, v_dim]` bf16 slab `baker-smoke` allocates. The
    /// element is PINNED rather than read because a `Cache` slot's dtype
    /// column is `Opaque` — the pool chose it, and on this plane the pool
    /// chose bf16.
    fn gated_delta_chunked<T: kernels::points::Scalar>(
        &self,
        qkv: In<Tensor<T>>,
        indptr: In<Tensor<i32>>,
        z: In<Tensor<T>>,
        gates: In<Tensor<f32>>,
        state: Cache<kernels::raises::Struct<RecurrentState>>,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
        y: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        // `z` IS THE OUT-NORM'S GATE and this point does not spend it. The
        // statement carries it because the recurrence and the gate are one
        // arm of the text's `merge!`, and `norm.rmsnorm_gated` downstream is
        // what reads it — the decode lane says the same thing in the same
        // words (`baker-smoke/src/smoke.rs:1215-1217`).
        let _ = z;
        at_bf16::<T>("ssm.gated_delta_chunked at an element other than bf16")?;

        let window = Chunked::of(qkv, indptr, gates, k_heads, v_heads, k_dim, v_dim)?;
        let shape = window.g;
        // The result is `[tokens, v_heads * v_dim]` and the scans write it
        // that way, `out[(t * v_heads + h) * v_dim + v]`. A narrower
        // rectangle would be written PAST rather than partially, so the
        // width is checked here and not left to the walk that sized it.
        let result = y.all("the recurrence's result")?;
        if i64::from(result.width) != i64::from(shape.v_h) * i64::from(shape.v_d) {
            return Err(Refusal::Narrow {
                what: "the recurrence's result row, against the stated value heads",
                at: i64::from(result.width),
            });
        }
        if state.ptr.is_null() {
            return Err(Refusal::Null {
                what: "the recurrent view this statement names",
            });
        }
        let rsv = unsafe { &*state.ptr };
        let staged = shape.stage(self, qkv.ptr.cast::<bf16>(), gates.ptr)?;
        window.scan(self, &staged, rsv, y.ptr)
    }

    /// Kimi's KDA rule for one token per request.
    ///
    /// # What the TEXT carries and what this body stages
    ///
    /// The legacy leg fired NINE launches per mixer
    /// (`model-legacy/src/kimi_k3/forward/mod.rs:199-283`) and the new text
    /// keeps four of them, because it says the same arithmetic in fewer
    /// rectangles:
    ///
    /// | legacy | the text |
    /// | --- | --- |
    /// | `q_proj`, `k_proj`, `v_proj` — three matmuls | ONE `gemm.matmul` on a packed `[3 * heads * head_dim, hidden]` bank |
    /// | `q_conv1d`, `k_conv1d`, `v_conv1d` — three convs | ONE `ssm.causal_conv1d` on the packed plane, over `3 * heads * head_dim` channels |
    /// | `f_a`, `f_b`, `b` — three matmuls | the same three; `f` and `b` arrive as VALUES |
    ///
    /// So the matmuls are not this body's and neither is the convolution:
    /// `mixed` arrives post-conv, `f` and `b` arrive projected. The other
    /// five ARE this body's, and they are exactly the ones with no
    /// statement of their own to stand in — two casts, an l2 norm, the
    /// gate/beta cook and the recurrence:
    ///
    ///   1. `kda_qkv_prep` — the packed row cut into three COMPACT f32
    ///      planes, `q` and `k` l2-normalised at the STATED `norm_eps` and
    ///      `v` widened. ONE launch where the legacy fired three
    ///      (`l2norm_scale_bf16_to_fp32` twice and `bf16_to_fp32` once)
    ///      over rectangles its three separate convs had already left
    ///      compact. `kernels/ssm/kda.cuh` transcribes those three and says
    ///      what it normalises OVER, which is the whole plane and not the
    ///      head — the shipped reading, and a debt it names rather than
    ///      quietly settles.
    ///   2. `kda_gate_beta` — `g = -exp(a_log[h]) * softplus(f + dt_bias)`
    ///      per CHANNEL and `beta = sigmoid(b)` per head. Unchanged from
    ///      the legacy launch, argument for argument.
    ///   3. `kda_recurrent_step_batched` — the delta rule against the
    ///      slot's `[heads, head_dim, head_dim]` f32 state.
    ///
    /// # The scratch
    ///
    /// Named device slabs from [`Ctx::scratch`], the arena
    /// `Ssm::gated_delta_chunked` takes its five planes from and for the
    /// same reason: a body that asked the DRIVER to stage them would be
    /// putting five rectangles nothing declares on the operand column. Two
    /// slabs and five planes, because the recurrences take the planes as
    /// separate pointers — `[q | k | v]` in one and `[gate | beta]` in the
    /// other, cut at the offsets `stage` computes and nowhere else.
    ///
    /// # The state's element is F32 here and bf16 next door
    ///
    /// `kda_recurrent_step_batched` takes `float* __restrict__ state_base`
    /// (`kernels/ssm/kda.cuh`), where every gated-delta scan in this file
    /// is instantiated `<::pie::ssm::state_bf16, ..>`. A `Cache` slot's
    /// dtype column is `Opaque` — the pool chose it — so this is a pin
    /// against the kernel rather than a reading of the slot, and the pool
    /// already spells both (`driver-cuda/src/layout/recurrent_layout.rs`'s
    /// `recurrent_is_bf16`).
    fn kda_step<T: kernels::points::Scalar>(
        &self,
        mixed: In<Tensor<T>>,
        f: In<Tensor<T>>,
        b: In<Tensor<T>>,
        dt_bias: Const<Tensor<f32>>,
        a_log: Const<Tensor<f32>>,
        state: Cache<kernels::raises::Struct<RecurrentState>>,
        heads: u32,
        head_dim: u32,
        norm_eps: f32,
        y: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("ssm.kda_step at an element other than bf16")?;
        let shape = Kda::of(mixed, f, b, heads, head_dim, norm_eps, y)?;
        let rsv = raised(state)?;
        let staged = shape.stage(self, mixed, f, b, dt_bias, a_log)?;
        kda_recurrent_step_batched(
            self,
            plane(staged.q_norm, shape.n, shape.w),
            plane(staged.k_norm, shape.n, shape.w),
            plane(staged.v, shape.n, shape.w),
            plane(staged.gate, shape.n, shape.w),
            plane(staged.beta, shape.n, shape.h),
            Out {
                ptr: y.ptr,
                rows: shape.n,
                width: shape.w,
            },
            Const::new(shape.h),
            Const::new(shape.d),
            rsv,
        )
    }

    /// [`Ssm::kda_step`] over a prefill window — the same prologue, and the
    /// window form of the same recurrence.
    ///
    /// # `kda_prefill_batched`, and why it is not "the step in a loop"
    ///
    /// It IS the step in a loop, and that is the finding rather than the
    /// objection. Both kernels live in `kernels/ssm/kda.cuh`, share the
    /// warp-per-`v`-row mapping, and carry the state as F32 THROUGH THE
    /// SLAB — `row[ki] = sv` twice per token, read back by the next token
    /// from the same address. There is no register-resident window and no
    /// bf16 round trip, so the two rounding trajectories are not merely
    /// close: the prefill kernel's per-token body is the decode kernel's,
    /// character for character, with `rh` replaced by `t * H + h` and the
    /// whole thing wrapped in the window's `for (t = begin; t < end; ++t)`.
    ///
    /// That matters because of W2's law: A PREFILL TAIL MUST LEAVE THE
    /// STATE THE DECODE STEP WOULD HAVE LEFT. The gated-delta family had to
    /// CHOOSE its arm to satisfy it (fla rounds to bf16 twice per token
    /// because the step does, and warp-tiled — more accurate — is
    /// therefore the fallback rather than the default). KDA has no such
    /// choice to make: one arithmetic, two loop nests.
    ///
    /// `kernels-cuda/tests/kda_recurrence.rs` MEASURES it rather than
    /// asserting it, because the legacy never fired this kernel. It ran the
    /// step form even on prefill, one launch per token, and
    /// `.wiki/driver/new-horizon.md:7652` lists `ssm::kda_prefill_batched`
    /// among the seven symbols with "nothing at all — not a test, not a
    /// comment, only the row and its wrapper". A kernel nothing has ever
    /// run is not one to trust on a reading of its source, so the test runs
    /// a two-window handoff against a per-request step loop and compares
    /// the slab bit for bit.
    ///
    /// # The CSR is the request count and nothing else
    ///
    /// `kda_prefill_batched` reads `qo_indptr[r]` and `qo_indptr[r + 1]`
    /// and takes its `r` off the CSR operand's own row count, which is the
    /// pairing every other `*_chunked` routine in this file uses rather
    /// than a `Const` restating it.
    fn kda_chunked<T: kernels::points::Scalar>(
        &self,
        mixed: In<Tensor<T>>,
        indptr: In<Tensor<i32>>,
        f: In<Tensor<T>>,
        b: In<Tensor<T>>,
        dt_bias: Const<Tensor<f32>>,
        a_log: Const<Tensor<f32>>,
        state: Cache<kernels::raises::Struct<RecurrentState>>,
        heads: u32,
        head_dim: u32,
        norm_eps: f32,
        y: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("ssm.kda_chunked at an element other than bf16")?;
        let shape = Kda::of(mixed, f, b, heads, head_dim, norm_eps, y)?;
        if indptr.rows <= 0 {
            return Err(Refusal::Empty {
                what: "the query CSR this statement names",
            });
        }
        let rsv = raised(state)?;
        let staged = shape.stage(self, mixed, f, b, dt_bias, a_log)?;
        kda_prefill_batched(
            self,
            plane(staged.q_norm, shape.n, shape.w),
            plane(staged.k_norm, shape.n, shape.w),
            plane(staged.v, shape.n, shape.w),
            plane(staged.gate, shape.n, shape.w),
            plane(staged.beta, shape.n, shape.h),
            Out {
                ptr: y.ptr,
                rows: shape.n,
                width: shape.w,
            },
            Const::new(shape.h),
            Const::new(shape.d),
            rsv,
            indptr,
        )
    }
}

/// The recurrent view a `Cache` slot names, or the refusal that says it was
/// not there.
fn raised(
    state: Cache<kernels::raises::Struct<RecurrentState>>,
) -> Result<In<Struct<RecurrentState>>, Refusal> {
    if state.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    Ok(state.raised())
}

/// One staged plane, as the routines want it: a rectangle whose rows are the
/// fire's and whose width the shape states.
const fn plane(ptr: *mut f32, rows: i32, width: i32) -> In<Tensor<f32>> {
    In {
        ptr: ptr.cast_const(),
        rows,
        width,
    }
}

/// KDA's geometry, read ONCE and checked against the four rectangles that
/// carry it.
///
/// `heads` and `head_dim` are STATED because a packed `[q | k | v]` row
/// cannot be divided by reading it, and everything else is READ: `n` is the
/// fire's own row count, and the widths are checked against the two stated
/// numbers rather than trusted.
#[derive(Clone, Copy)]
struct Kda {
    n: i32,
    h: i32,
    d: i32,
    /// `h * d` — ONE plane of the packed row, and the width of every f32
    /// plane the recurrence reads.
    w: i32,
    eps: f32,
}

/// The five f32 planes the KDA recurrences take, carved out of the named
/// scratch.
#[derive(Clone, Copy)]
struct KdaStaged {
    q_norm: *mut f32,
    k_norm: *mut f32,
    v: *mut f32,
    gate: *mut f32,
    beta: *mut f32,
}

impl Kda {
    fn of<T: kernels::points::Scalar>(
        mixed: In<Tensor<T>>,
        f: In<Tensor<T>>,
        b: In<Tensor<T>>,
        heads: u32,
        head_dim: u32,
        norm_eps: f32,
        y: Out<Tensor<f32>>,
    ) -> Result<Kda, Refusal> {
        fn stated(n: u32, what: &'static str) -> Result<i32, Refusal> {
            match i32::try_from(n) {
                Ok(0) => Err(Refusal::Empty { what }),
                Ok(n) => Ok(n),
                Err(_) => Err(Refusal::Wide {
                    what,
                    at: i64::from(n),
                    max: i64::from(i32::MAX),
                }),
            }
        }
        let h = stated(heads, "the KDA heads this statement states")?;
        let d = stated(head_dim, "the KDA head width this statement states")?;
        let wide = i64::from(h) * i64::from(d);
        let w = i32::try_from(wide).map_err(|_| Refusal::Wide {
            what: "the KDA plane the two stated head numbers multiply out to",
            at: wide,
            max: i64::from(i32::MAX),
        })?;

        // THE ROW THE TWO NUMBERS ADD UP TO, three times over. A width that
        // disagrees is a statement whose head numbers are not this
        // rectangle's, and the prologue would cut the value plane at the
        // wrong column — so it refuses rather than slicing on faith.
        let packed = mixed.all("the post-convolution `[q | k | v]` row")?;
        if i64::from(packed.width) != 3 * i64::from(w) {
            return Err(Refusal::Narrow {
                what: "the post-convolution `[q | k | v]` row, against the two stated head numbers",
                at: i64::from(packed.width),
            });
        }
        let n = packed.rows;

        // `f` IS PER CHANNEL AND `b` IS PER HEAD, which is KDA's whole
        // shape: the forget gate is a `[heads, head_dim]` decay where the
        // gated-delta rule's is one column per head, and beta is the
        // column. That reading is checkable exactly here, and
        // `kda_gate_beta` reads the two apart — `raw_g[(t * H + h) * D + d]`
        // against `raw_beta[t * H + h]`.
        let forget = f.all("the forget projection this statement hands over")?;
        if i64::from(forget.width) != i64::from(w) {
            return Err(Refusal::Narrow {
                what: "the forget projection's row, against the two stated head numbers",
                at: i64::from(forget.width),
            });
        }
        let beta = b.all("the beta projection this statement hands over")?;
        if i64::from(beta.width) != i64::from(h) {
            return Err(Refusal::Narrow {
                what: "the beta projection's row, against the stated head count",
                at: i64::from(beta.width),
            });
        }
        if forget.rows != n || beta.rows != n {
            return Err(Refusal::Narrow {
                what: "the forget and beta projections do not carry the packed row's rows",
                at: i64::from(forget.rows.min(beta.rows)),
            });
        }

        // The result is `[rows, heads * head_dim]` and the recurrences write
        // it that way, `out[(t * H + h) * D + v]`. A narrower rectangle
        // would be written PAST rather than partially, so the width is
        // checked here and not left to the walk that sized it.
        let result = y.all("the recurrence's result")?;
        if i64::from(result.width) != i64::from(w) {
            return Err(Refusal::Narrow {
                what: "the recurrence's result row, against the two stated head numbers",
                at: i64::from(result.width),
            });
        }
        Ok(Kda {
            n,
            h,
            d,
            w,
            eps: norm_eps,
        })
    }

    /// `n * width` f32 elements, refused rather than wrapped.
    fn plane_elems(self, width: i32, what: &'static str) -> Result<usize, Refusal> {
        let elems = i64::from(self.n) * i64::from(width);
        usize::try_from(elems).map_err(|_| Refusal::Wide {
            what,
            at: elems,
            max: i64::MAX,
        })
    }

    /// The prologue: two launches, five planes.
    fn stage<T: kernels::points::Scalar>(
        self,
        ctx: &Ctx<'_>,
        mixed: In<Tensor<T>>,
        f: In<Tensor<T>>,
        b: In<Tensor<T>>,
        dt_bias: Const<Tensor<f32>>,
        a_log: Const<Tensor<f32>>,
    ) -> Result<KdaStaged, Refusal> {
        const PREP_BLOCK: u32 = 128;

        /// `q`, `k`, `v` — the three the cut writes, one block each per row.
        /// `kda_qkv_prep` reads which one off `blockIdx.y`.
        const PLANES: u32 = 3;

        let Kda { n, h, d, w, eps } = self;
        let wide = self.plane_elems(w, "the `[q | k | v]` plane this fire stages")?;
        let decay = self.plane_elems(h, "the beta column this fire stages")?;

        // Three planes in the first slab and two in the second, because the
        // recurrences take them as five separate pointers: `q`, `k`, `v`,
        // then `gate` and `beta`.
        let qkv = ctx
            .scratch("ssm::kda_qkv", 3 * wide * core::mem::size_of::<f32>())?
            .cast::<f32>();
        let gb = ctx
            .scratch(
                "ssm::kda_gates",
                (wide + decay) * core::mem::size_of::<f32>(),
            )?
            .cast::<f32>();
        let staged = KdaStaged {
            q_norm: qkv,
            k_norm: unsafe { qkv.add(wide) },
            v: unsafe { qkv.add(2 * wide) },
            gate: gb,
            beta: unsafe { gb.add(wide) },
        };

        ctx.fire(
            Fire::at("ssm/kda.cuh", "::pie::ssm::kda_qkv_prep<::pie::bf16, 128>")
                .apply(Launch::grid([n.unsigned_abs(), PLANES, 1], [PREP_BLOCK, 1, 1])),
            &[
                mixed.ptr.cast::<bf16>().arg(),
                staged.q_norm.arg(),
                staged.k_norm.arg(),
                staged.v.arg(),
                w.arg(),
                eps.arg(),
            ],
        )?;

        // The gate/beta cook, through the routine the legacy fired — same
        // launcher, same arguments, and `lower_bound` left at the zero that
        // launcher pins, which is the softplus branch. The sigmoid branch
        // past it is a reading no kimi text has ever stated.
        kda_gate_beta::<bf16>(
            ctx,
            In {
                ptr: f.ptr.cast::<bf16>(),
                rows: n,
                width: w,
            },
            In {
                ptr: b.ptr.cast::<bf16>(),
                rows: n,
                width: h,
            },
            a_log,
            dt_bias,
            Out {
                ptr: staged.gate,
                rows: n,
                width: w,
            },
            Out {
                ptr: staged.beta,
                rows: n,
                width: h,
            },
            Const::new(d),
        )?;
        Ok(staged)
    }
}

/// The gated-delta rule's geometry, read ONCE and checked against the two
/// rectangles that carry it.
///
/// SHARED BY THE STEP AND THE WINDOW, because the two points differ in
/// exactly two things and neither is here: how the request boundary arrives
/// (a row count vs a CSR) and which scan runs. Everything before that — the
/// four head numbers, the packed row they must add up to, the fused
/// `[g_log | beta]` row, and the five compact f32 planes the scans read — is
/// one description, so it is written once.
///
/// The four head numbers are STATED on the point — a packed row cannot be
/// divided by reading it — and everything else is READ: `n` is the window's
/// own row count and `conv_dim` its width.
#[derive(Clone, Copy)]
struct GdnShape {
    n: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
    conv_dim: i32,
}

/// [`GdnShape`] over a prefill window: the same geometry, plus the request
/// boundary the scans walk.
///
/// `r` is the CSR's row count, the pairing every other `*_chunked` routine in
/// this file uses rather than a `Const` restating it.
#[derive(Clone, Copy)]
struct Chunked {
    g: GdnShape,
    r: i32,
    qo_indptr: *const u32,
}

/// The five f32 planes the scans take, carved out of the named scratch.
#[derive(Clone, Copy)]
struct Staged {
    q_norm: *mut f32,
    k_norm: *mut f32,
    v: *mut f32,
    g_log: *mut f32,
    beta: *mut f32,
}

impl GdnShape {
    fn of<T: kernels::points::Scalar>(
        qkv: In<Tensor<T>>,
        gates: In<Tensor<f32>>,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
    ) -> Result<GdnShape, Refusal> {
        fn stated(n: u32, what: &'static str) -> Result<i32, Refusal> {
            match i32::try_from(n) {
                Ok(0) => Err(Refusal::Empty { what }),
                Ok(n) => Ok(n),
                Err(_) => Err(Refusal::Wide {
                    what,
                    at: i64::from(n),
                    max: i64::from(i32::MAX),
                }),
            }
        }
        let k_h = stated(k_heads, "the key heads this statement states")?;
        let v_h = stated(v_heads, "the value heads this statement states")?;
        let k_d = stated(k_dim, "the key head width this statement states")?;
        let v_d = stated(v_dim, "the value head width this statement states")?;
        if v_h % k_h != 0 {
            return Err(Refusal::Narrow {
                what: "v_h per k_h",
                at: i64::from(v_h),
            });
        }
        let packed = qkv.all("the post-convolution qkv")?;
        // THE ROW THE FOUR NUMBERS ADD UP TO. A width that disagrees is a
        // statement whose head numbers are not this rectangle's, and the
        // prologue would read the value slice at the wrong column — so it
        // refuses rather than slicing on faith.
        let want = 2 * i64::from(k_h) * i64::from(k_d) + i64::from(v_h) * i64::from(v_d);
        if i64::from(packed.width) != want {
            return Err(Refusal::Narrow {
                what: "the post-convolution qkv's row, against the four stated head numbers",
                at: i64::from(packed.width),
            });
        }
        let fused = gates.all("the fused `[g_log | beta]` row")?;
        if i64::from(fused.width) != 2 * i64::from(v_h) {
            return Err(Refusal::Narrow {
                what: "the fused `[g_log | beta]` row, against the stated value heads",
                at: i64::from(fused.width),
            });
        }
        Ok(GdnShape {
            n: packed.rows,
            k_h,
            v_h,
            k_d,
            v_d,
            conv_dim: packed.width,
        })
    }

    /// `n * heads * width` f32 elements, refused rather than wrapped.
    fn plane(self, heads: i32, width: i32, what: &'static str) -> Result<usize, Refusal> {
        let elems = i64::from(self.n) * i64::from(heads) * i64::from(width);
        usize::try_from(elems).map_err(|_| Refusal::Wide {
            what,
            at: elems,
            max: i64::MAX,
        })
    }

    /// The prologue: two launches, five planes.
    ///
    /// BOTH RECURRENCE POINTS COME THROUGH HERE, and that is the whole
    /// answer to the strideless mark. The scans read five COMPACT planes —
    /// `q_norm[(t * K_h + h) * K_d]`, `g_log[t * V_h + h]` — while the two
    /// rectangles a statement carries are PACKED rows: `qkv` strides by
    /// `conv_dim` and `gates` by `2 * V_h`. Cutting a compact plane out of
    /// either with a pointer offset claims the cut's width as the row
    /// stride, which is true at one token and false at two. So the cut is a
    /// kernel, at every token count, for the step exactly as for the window.
    fn stage(self, ctx: &Ctx<'_>, qkv: *const bf16, gates: *const f32) -> Result<Staged, Refusal> {
        const PREP_BLOCK: u32 = 128;

        let GdnShape {
            n,
            k_h,
            v_h,
            k_d,
            v_d,
            conv_dim,
        } = self;
        let key = self.plane(k_h, k_d, "the key plane this recurrence stages")?;
        let val = self.plane(v_h, v_d, "the value plane this recurrence stages")?;
        let decay = self.plane(v_h, 1, "the decay plane this recurrence stages")?;

        // Two planes per slab and one slab per shape, because the scans take
        // the halves as separate pointers: `q` then `k`, `g_log` then `beta`.
        let qk = ctx
            .scratch("ssm::gdn_chunk_qk", 2 * key * core::mem::size_of::<f32>())?
            .cast::<f32>();
        let v = ctx
            .scratch("ssm::gdn_chunk_v", val * core::mem::size_of::<f32>())?
            .cast::<f32>();
        let gb = ctx
            .scratch(
                "ssm::gdn_chunk_gates",
                2 * decay * core::mem::size_of::<f32>(),
            )?
            .cast::<f32>();
        let staged = Staged {
            q_norm: qk,
            k_norm: unsafe { qk.add(key) },
            v,
            g_log: gb,
            beta: unsafe { gb.add(decay) },
        };

        #[allow(clippy::cast_precision_loss)]
        let q_scale = (k_d as f32).sqrt().recip();
        ctx.fire(
            Fire::at(
                "ssm/gated_delta_net_prep.cuh",
                "::pie::ssm::qwen_gdn_qk_norm<::pie::bf16, 128>",
            )
            .apply(Launch::grid(
                [n.unsigned_abs(), k_h.unsigned_abs(), 1],
                [PREP_BLOCK, 1, 1],
            )),
            &[
                qkv.arg(),
                staged.q_norm.arg(),
                staged.k_norm.arg(),
                k_h.arg(),
                k_d.arg(),
                conv_dim.arg(),
                q_scale.arg(),
            ],
        )?;
        ctx.fire(
            Fire::at(
                "ssm/gated_delta_net_prep.cuh",
                "::pie::ssm::qwen_gdn_v_gates<::pie::bf16, 128>",
            )
            .apply(Launch::grid(
                [n.unsigned_abs(), v_h.unsigned_abs(), 1],
                [PREP_BLOCK, 1, 1],
            )),
            &[
                qkv.arg(),
                gates.arg(),
                staged.v.arg(),
                staged.g_log.arg(),
                staged.beta.arg(),
                k_h.arg(),
                v_h.arg(),
                k_d.arg(),
                v_d.arg(),
                conv_dim.arg(),
            ],
        )?;
        Ok(staged)
    }

    /// The step scan, over the five planes [`GdnShape::stage`] just wrote.
    ///
    /// THE ROW COUNT IS THE REQUEST COUNT HERE, and it is read rather than
    /// stated because the point's own name is what makes it true: a step
    /// form is one token per request, which is the fact (`qo_one`) the lane
    /// carrying this statement was selected on. `recurrent_step_batched_gqa`
    /// indexes both `slot_ids[r]` and the token row by the same `blockIdx.y`
    /// (`gated_delta_net.cuh:1275-1284`), so the two numbers are one number
    /// and a `Const` restating it could only disagree with the rectangle.
    fn step(
        self,
        ctx: &Ctx<'_>,
        staged: &Staged,
        rsv: *const crate::views::RecurrentView,
        out: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let GdnShape {
            n,
            k_h,
            v_h,
            k_d,
            v_d,
            ..
        } = self;
        let key = In {
            ptr: staged.q_norm.cast_const(),
            rows: n,
            width: k_h.saturating_mul(k_d),
        };
        recurrent_gated_delta_step_batched_gqa_state_bf16(
            ctx,
            key,
            In {
                ptr: staged.k_norm.cast_const(),
                rows: key.rows,
                width: key.width,
            },
            In {
                ptr: staged.v.cast_const(),
                rows: n,
                width: v_h.saturating_mul(v_d),
            },
            In {
                ptr: staged.g_log.cast_const(),
                rows: n,
                width: v_h,
            },
            In {
                ptr: staged.beta.cast_const(),
                rows: n,
                width: v_h,
            },
            out,
            Const::new(k_h),
            Const::new(v_h),
            Const::new(k_d),
            Const::new(v_d),
            Const::new(n),
            In {
                ptr: rsv,
                rows: 0,
                width: 0,
            },
        )
    }
}

impl Chunked {
    fn of<T: kernels::points::Scalar>(
        qkv: In<Tensor<T>>,
        indptr: In<Tensor<i32>>,
        gates: In<Tensor<f32>>,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
    ) -> Result<Chunked, Refusal> {
        let g = GdnShape::of(qkv, gates, k_heads, v_heads, k_dim, v_dim)?;
        if indptr.rows <= 0 {
            return Err(Refusal::Empty {
                what: "the query CSR this statement names",
            });
        }
        Ok(Chunked {
            g,
            r: indptr.rows,
            // The CSR is `[requests + 1]` u32 boundaries; every scan in this
            // file reads it that way and the declaration spells its slot
            // `i32` because a boundary buffer is an ordinary device
            // rectangle, not because the kernel signs it.
            qo_indptr: indptr.ptr.cast::<u32>(),
        })
    }

    /// The scan, chosen on the shape. See the point's own doc for what each
    /// arm is bounded by and why the order is what it is.
    fn scan(
        self,
        ctx: &Ctx<'_>,
        staged: &Staged,
        rsv: &crate::views::RecurrentView,
        out: *mut f32,
    ) -> Result<(), Refusal> {
        const BK_MAX_FLA: i32 = 128;

        const BV_FLA: u32 = 128;

        /// `MAX_K_PER_LANE * 32` — `kernels/ssm/gated_delta_net.cuh:469`.
        const WARP_TILED_K_MAX: i32 = 256;

        let Chunked { g, r, qo_indptr } = self;
        let GdnShape {
            k_h,
            v_h,
            k_d,
            v_d,
            ..
        } = g;
        let (rows, heads) = (r.unsigned_abs(), v_h.unsigned_abs());
        let state_base = rsv.slab;
        if state_base.is_null() {
            return Err(Refusal::Null {
                what: "the recurrent slab this statement's cache row names",
            });
        }
        let slot_ids = rsv.slot_ids;
        let slot_stride_elems = rsv.slot_stride_elems;
        // A STATEMENT THAT NAMES A CACHE ROW LEAVES ITS TAIL THERE — the
        // chunked conv's reading, and the same one here.
        let write_state = true;

        if k_d <= BK_MAX_FLA && v_d.unsigned_abs() % BV_FLA == 0 {
            return ctx.fire(
                Fire::at(
                    "ssm/gated_delta_net.cuh",
                    "::pie::ssm::chunk_gated_delta_prefill_batched_fla<::pie::ssm::state_bf16, 128, 128>",
                )
                .apply(
                    Launch::grid([v_d.unsigned_abs() / BV_FLA, rows, heads], [BV_FLA, 1, 1])
                        .smem(2 * BK_MAX_FLA.unsigned_abs() * FLOAT),
                ),
                &[
                    staged.q_norm.cast_const().arg(),
                    staged.k_norm.cast_const().arg(),
                    staged.v.cast_const().arg(),
                    staged.g_log.cast_const().arg(),
                    staged.beta.cast_const().arg(),
                    state_base.arg(),
                    slot_ids.arg(),
                    qo_indptr.arg(),
                    slot_stride_elems.arg(),
                    out.arg(),
                    k_h.arg(),
                    v_h.arg(),
                    k_d.arg(),
                    v_d.arg(),
                    write_state.arg(),
                    // No commit-length clamp and no per-row mask: this point
                    // states neither, and a null is what "the pass is
                    // uniform" spells at both kernels.
                    MaybeConst::<i32>::none().arg(),
                    MaybeConst::<u8>::none().arg(),
                ],
            );
        }

        if k_d <= WARP_TILED_K_MAX {
            return ctx.fire(
                Fire::at(
                    "ssm/gated_delta_net.cuh",
                    "::pie::ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa<::pie::ssm::state_bf16, false>",
                )
                .apply(warp_tiled_scan(rows, heads, v_d.unsigned_abs())),
                &[
                    staged.q_norm.cast_const().arg(),
                    staged.k_norm.cast_const().arg(),
                    staged.v.cast_const().arg(),
                    staged.g_log.cast_const().arg(),
                    staged.beta.cast_const().arg(),
                    state_base.arg(),
                    slot_ids.arg(),
                    qo_indptr.arg(),
                    slot_stride_elems.arg(),
                    out.arg(),
                    k_h.arg(),
                    v_h.arg(),
                    k_d.arg(),
                    v_d.arg(),
                    write_state.arg(),
                    MaybeConst::<u8>::none().arg(),
                ],
            );
        }

        // PAST BOTH GQA-NATIVE ARMS, and this is where the legacy composite
        // lives: the per-token form indexes q and k by `V_h`, so the key
        // heads have to be broadcast up to the value heads first. One more
        // scratch slab, one more launch, and the scan reads a `k_h == v_h`
        // rectangle.
        let (q_norm, k_norm) = if v_h == k_h {
            (staged.q_norm.cast_const(), staged.k_norm.cast_const())
        } else {
            let wide = g.plane(v_h, k_d, "the repeated key plane this window stages")?;
            let rep = ctx
                .scratch(
                    "ssm::gdn_chunk_repeat",
                    2 * wide * core::mem::size_of::<f32>(),
                )?
                .cast::<f32>();
            let (q, k) = (rep, unsafe { rep.add(wide) });
            for (src, dst) in [(staged.q_norm, q), (staged.k_norm, k)] {
                ctx.fire(
                    Fire::at(
                        "ssm/gated_delta_net_prep.cuh",
                        "::pie::ssm::repeat_interleave_heads_fp32<::pie::ssm::f32>",
                    )
                    .apply(gated_rms(g.n.unsigned_abs(), heads)),
                    &[
                        src.cast_const().arg(),
                        dst.arg(),
                        k_h.arg(),
                        v_h.arg(),
                        k_d.arg(),
                        (v_h / k_h).arg(),
                    ],
                )?;
            }
            (q.cast_const(), k.cast_const())
        };
        ctx.fire(
            Fire::at(
                "ssm/gated_delta_net.cuh",
                "::pie::ssm::chunk_gated_delta_prefill_batched<::pie::ssm::state_bf16, false>",
            )
            .apply(
                Launch::grid([rows, heads, 1], [GDN_BLOCK, 1, 1])
                    .smem(2 * k_d.unsigned_abs() * FLOAT),
            ),
            &[
                q_norm.arg(),
                k_norm.arg(),
                staged.v.cast_const().arg(),
                staged.g_log.cast_const().arg(),
                staged.beta.cast_const().arg(),
                state_base.arg(),
                slot_ids.arg(),
                qo_indptr.arg(),
                slot_stride_elems.arg(),
                out.arg(),
                v_h.arg(),
                k_d.arg(),
                v_d.arg(),
            ],
        )
    }
}

#[routine(bf16, canon = "ssm.causal_conv1d", out(y = like(x)))]
pub fn causal_conv1d_update_batched<T>(
    ctx: &Ctx<'_>,
    x: In<Tensor<T>>,
    weight: Const<Tensor<T>>,
    bias: Option<Const<Tensor<T>>>,
    y: Out<Tensor<T>>,
    c: Const<i32>,
    k: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
) -> Result<(), Refusal>
where
    MaybeConst<T>: Abi,
{
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    let state_base = rsv.conv_slab;
    let slot_stride_elems = rsv.conv_stride;
    let slot_ids = rsv.slot_ids;

    #[must_use]
    const fn split_packed(rows: u32, in_width: u32) -> Launch {
        Launch::grid([in_width.div_ceil(RULE_BLOCK), rows, 1], [RULE_BLOCK, 1, 1])
    }

    let r = x.rows;
    ctx.fire(
        Fire::at(
            "ssm/causal_conv1d.cuh",
            crate::jit::symbol(&format!(
                "::pie::ssm::causal_conv1d_update_batched<{}>",
                T::CPP
            )),
        )
        .apply(split_packed(r.unsigned_abs(), c.unsigned_abs())),
        &[
            x.arg(),
            weight.arg(),
            bias.arg(),
            state_base.arg(),
            slot_ids.arg(),
            slot_stride_elems.arg(),
            y.arg(),
            r.arg(),
            c.arg(),
            k.arg(),
        ],
    )
}

pub fn causal_conv1d_prefill_noact<T>(
    ctx: &Ctx<'_>,
    x: *const T,
    weight: *const T,
    bias: MaybeConst<T>,
    y: *mut T,
    state_out: *mut T,
    n: i32,
    channels: i32,
    k: i32,
) -> Result<(), Refusal>
where
    T: kernels::Elem,
    *const T: Abi,
    *mut T: Abi,
    MaybeConst<T>: Abi,
{
    ctx.fire(
        Fire::at(
            "ssm/causal_conv1d.cuh",
            crate::jit::symbol(&format!(
                "::pie::ssm::causal_conv1d_prefill<{}, false>",
                T::CPP
            )),
        )
        .apply(Launch::grid([channels.unsigned_abs(), 1, 1], [64, 1, 1])),
        &[
            x.arg(),
            weight.arg(),
            bias.arg(),
            y.arg(),
            state_out.arg(),
            n.arg(),
            channels.arg(),
            k.arg(),
        ],
    )
}

#[routine(bf16, canon = "ssm.causal_conv1d_chunked", out(y = like(x)))]
pub fn causal_conv1d_prefill_batched<T>(
    ctx: &Ctx<'_>,
    x: In<Tensor<T>>,
    weight: Const<Tensor<T>>,
    bias: Option<Const<Tensor<T>>>,
    y: Out<Tensor<T>>,
    c: Const<i32>,
    k: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
    write_state: Const<bool>,
    qo_indptr: In<Tensor<i32>>,
) -> Result<(), Refusal>
where
    MaybeConst<T>: Abi,
{
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    let state_out_base = rsv.conv_slab;
    let slot_stride_elems = rsv.conv_stride;
    // The request count is the CSR operand's own row count -- the pairing,
    // not a `Const` restating it.
    let r = qo_indptr.rows;
    let write_state = *write_state;
    let slot_ids = rsv.slot_ids;
    let qo_indptr = qo_indptr.ptr as *const u32;
    const CONV_CHANNEL_TILE_FROM: i32 = 8;

    const CONV_TILE: u32 = 128;

    const CONV_PER_CHANNEL_BLOCK: u32 = 64;

    let (rows, chans) = (r.unsigned_abs(), c.unsigned_abs());

    let (instantiation, launch) = if r >= CONV_CHANNEL_TILE_FROM {
        (
            crate::jit::symbol(&format!(
                "::pie::ssm::causal_conv1d_prefill_batched_channel_tile<{}>",
                T::CPP
            )),
            Launch::grid([chans.div_ceil(CONV_TILE), rows, 1], [CONV_TILE, 1, 1]),
        )
    } else {
        (
            crate::jit::symbol(&format!(
                "::pie::ssm::causal_conv1d_prefill_batched<{}>",
                T::CPP
            )),
            Launch::grid([chans, rows, 1], [CONV_PER_CHANNEL_BLOCK, 1, 1]),
        )
    };
    ctx.fire(
        Fire::at("ssm/causal_conv1d.cuh", instantiation).apply(launch),
        &[
            x.arg(),
            weight.arg(),
            bias.arg(),
            y.arg(),
            state_out_base.arg(),
            slot_ids.arg(),
            qo_indptr.arg(),
            slot_stride_elems.arg(),
            c.arg(),
            k.arg(),
            write_state.arg(),
            MaybeConst::<u8>::none().arg(),
            MaybeConst::<i32>::none().arg(),
        ],
    )
}

#[routine]
pub fn bf16_to_fp32(
    ctx: &Ctx<'_>,
    x: In<Tensor<c_void>>,
    y: Out<Tensor<f32>>,
) -> Result<(), Refusal> {
    let dst = y.all("element count")?;
    let n = dst.elements();
    if n <= 0 {
        return Err(Refusal::Empty {
            what: "element count",
        });
    }
    let count = n.unsigned_abs();
    let elems = count as usize;
    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::widen<::pie::bf16>",
        )
        .apply(elementwise(count)),
        &[x.arg(), y.arg(), elems.arg()],
    )
}

#[routine]
pub fn fp32_to_bf16(
    ctx: &Ctx<'_>,
    x: In<Tensor<f32>>,
    y: Out<Tensor<c_void>>,
) -> Result<(), Refusal> {
    let dst = y.all("element count")?;
    let n = dst.elements();
    if n <= 0 {
        return Err(Refusal::Empty {
            what: "element count",
        });
    }
    let count = n.unsigned_abs();
    let elems = count as usize;
    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::narrow<::pie::bf16>",
        )
        .apply(elementwise(count)),
        &[x.arg(), y.arg(), elems.arg()],
    )
}

#[routine]
pub fn repeat_interleave_heads_fp32(
    ctx: &Ctx<'_>,
    in_: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    k_h: Const<i32>,
    v_h: Const<i32>,
    d: Const<i32>,
) -> Result<(), Refusal> {
    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::repeat_interleave_heads_fp32<::pie::ssm::f32>",
        )
        .apply(gated_rms(in_.rows.unsigned_abs(), v_h.unsigned_abs())),
        &[
            in_.arg(),
            out.arg(),
            k_h.arg(),
            v_h.arg(),
            d.arg(),
            (*v_h / *k_h).arg(),
        ],
    )
}

#[routine]
pub fn l2norm_scale_bf16_to_fp32(
    ctx: &Ctx<'_>,
    x: In<Tensor<c_void>>,
    y: Out<Tensor<f32>>,
    eps: Const<f32>,
) -> Result<(), Refusal> {
    let eps = *eps;

    #[must_use]
    const fn per_row_narrow(rows: u32) -> Launch {
        const PER_ROW_NARROW_BLOCK: u32 = 128;

        Launch::per_row(rows, PER_ROW_NARROW_BLOCK)
    }

    let dst = y.all("the normalised row")?;
    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::l2norm_scale<::pie::bf16, 128>",
        )
        .apply(per_row_narrow(dst.rows.unsigned_abs())),
        &[x.arg(), y.arg(), dst.width.arg(), 1.0f32.arg(), eps.arg()],
    )
}

#[routine(bf16)]
pub fn kda_gate_beta<T>(
    ctx: &Ctx<'_>,
    raw_g: In<Tensor<T>>,
    raw_beta: In<Tensor<T>>,
    a_log: Const<Tensor<f32>>,
    dt_bias: Const<Tensor<f32>>,
    gate_out: Out<Tensor<f32>>,
    beta_out: Out<Tensor<f32>>,
    d: Const<i32>,
) -> Result<(), Refusal> {
    let betas = beta_out.all("the KDA head count")?;
    let t = betas.rows;

    let h = betas.width;
    ctx.fire(
        Fire::at(
            "ssm/kda.cuh",
            crate::jit::symbol(&format!("::pie::ssm::kda_gate_beta<{}>", T::CPP)),
        )
        .apply(per_head_elementwise(
            t.unsigned_abs(),
            h.unsigned_abs(),
            d.unsigned_abs(),
        )),
        &[
            raw_g.arg(),
            raw_beta.arg(),
            a_log.arg(),
            dt_bias.arg(),
            gate_out.arg(),
            beta_out.arg(),
            t.arg(),
            h.arg(),
            d.arg(),
            0.0f32.arg(),
        ],
    )
}

#[routine(bf16, canon = "norm.rmsnorm_gated_by", out(out = like(g)))]
pub fn kda_o_norm_gated<T>(
    ctx: &Ctx<'_>,
    o: In<Tensor<f32>>,
    g: In<Tensor<T>>,
    weight: Const<Tensor<f32>>,
    out: Out<Tensor<T>>,
    h: Const<i32>,
    d: Const<i32>,
    eps: Const<f32>,
) -> Result<(), Refusal> {
    let eps = *eps;

    ctx.fire(
        Fire::at(
            "ssm/kda.cuh",
            crate::jit::symbol(&format!("::pie::ssm::kda_o_norm_gated<{}>", T::CPP)),
        )
        .apply(per_head_elementwise(
            out.rows.unsigned_abs(),
            h.unsigned_abs(),
            d.unsigned_abs(),
        )),
        &[
            o.arg(),
            g.arg(),
            weight.arg(),
            out.arg(),
            h.arg(),
            d.arg(),
            eps.arg(),
        ],
    )
}

#[routine(whole)]
pub fn kda_recurrent_step_batched(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    gate: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    h: Const<i32>,
    d: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };
    let state_base = rsv.slab;
    let slot_ids = rsv.slot_ids;
    let slot_stride_elems = rsv.slot_stride_elems;
    // One row per request: the statement's `[Requests, H, D]` result is the
    // launch rectangle, so the count is the result's own row count.
    let r = out.rows;
    const KDA_STEP_BLOCK: u32 = 256;
    ctx.fire(
        Fire::at("ssm/kda.cuh", "::pie::ssm::kda_recurrent_step_batched").apply(
            Launch::grid(
                [r.unsigned_abs(), h.unsigned_abs(), 1],
                [KDA_STEP_BLOCK, 1, 1],
            )
            .smem(kda_shmem(d.unsigned_abs())),
        ),
        &[
            q_norm.arg(),
            k_norm.arg(),
            v.arg(),
            gate.arg(),
            beta.arg(),
            state_base.arg(),
            slot_ids.arg(),
            slot_stride_elems.arg(),
            out.arg(),
            h.arg(),
            d.arg(),
        ],
    )
}

#[routine(whole, out(out = split(v, d)))]
pub fn kda_prefill_batched(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    gate: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    h: Const<i32>,
    d: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
    qo_indptr: In<Tensor<i32>>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };
    // The request count is the CSR operand's own row count.
    let r = qo_indptr.rows;
    let state_base = rsv.slab;
    let slot_ids = rsv.slot_ids;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let slot_stride_elems = rsv.slot_stride_elems;
    const KDA_PREFILL_MAX_WARPS: i32 = 32;
    ctx.fire(
        Fire::at("ssm/kda.cuh", "::pie::ssm::kda_prefill_batched").apply(
            Launch::grid(
                [r.unsigned_abs(), h.unsigned_abs(), 1],
                [d.min(KDA_PREFILL_MAX_WARPS).unsigned_abs() * WARP, 1, 1],
            )
            .smem(kda_shmem(d.unsigned_abs())),
        ),
        &[
            q_norm.arg(),
            k_norm.arg(),
            v.arg(),
            gate.arg(),
            beta.arg(),
            state_base.arg(),
            slot_ids.arg(),
            qo_indptr.arg(),
            slot_stride_elems.arg(),
            out.arg(),
            h.arg(),
            d.arg(),
        ],
    )
}

#[routine]
pub fn nemotron_prepare_mamba_params(
    ctx: &Ctx<'_>,
    a_log: Const<Tensor<bf16>>,
    d: Const<Tensor<bf16>>,
    dt_bias: Const<Tensor<bf16>>,
    a: Out<Tensor<f32>>,
    d_f32: Out<Tensor<f32>>,
    dt_bias_f32: Out<Tensor<f32>>,
    num_heads: Const<i32>,
) -> Result<(), Refusal> {
    ctx.fire(
        Fire::at(
            "ssm/nemotron_h.cuh",
            "::pie::ssm::prepare_mamba_params<::pie::bf16>",
        )
        .apply(elementwise(num_heads.unsigned_abs())),
        &[
            a_log.arg(),
            d.arg(),
            dt_bias.arg(),
            a.arg(),
            d_f32.arg(),
            dt_bias_f32.arg(),
            num_heads.arg(),
        ],
    )
}

#[routine]
pub fn nemotron_prepare_mamba_dt_da(
    ctx: &Ctx<'_>,
    dt: In<Tensor<bf16>>,
    a: In<Tensor<f32>>,
    dt_bias: In<Tensor<f32>>,
    dt_out: Out<Tensor<f32>>,
    da_out: Out<Tensor<f32>>,
) -> Result<(), Refusal> {
    let src = dt.all("rows * num_heads")?;
    let num_heads = src.width;
    let total = src.elements();
    if total <= 0 {
        return Err(Refusal::Empty {
            what: "rows * num_heads",
        });
    }
    ctx.fire(
        Fire::at(
            "ssm/nemotron_h.cuh",
            "::pie::ssm::prepare_mamba_dt_da<::pie::bf16>",
        )
        .apply(elementwise(total.unsigned_abs())),
        &[
            dt.arg(),
            a.arg(),
            dt_bias.arg(),
            dt_out.arg(),
            da_out.arg(),
            total.arg(),
            num_heads.arg(),
            0.0f32.arg(),
        ],
    )
}

#[routine(bf16, out(y = like(x)))]
pub fn zamba_rmsnorm_gated<T>(
    ctx: &Ctx<'_>,
    x: In<Tensor<T>>,
    gate: In<Tensor<T>>,
    weight: Const<Tensor<T>>,
    y: Out<Tensor<T>>,
    n_groups: Const<i32>,
    eps: Const<f32>,
) -> Result<(), Refusal> {
    let eps = *eps;

    let src = x.all("the normalised row")?;
    let gates = gate.all("the normalised row")?;
    let hidden = src.width;

    let gate_stride = gates.stride;
    ctx.fire(
        Fire::at(
            "ssm/nemotron_h.cuh",
            crate::jit::symbol(&format!("::pie::ssm::zamba_rmsnorm_gated<{}>", T::CPP)),
        )
        .apply(gated_rms(src.rows.unsigned_abs(), n_groups.unsigned_abs())),
        &[
            x.arg(),
            gate.arg(),
            weight.arg(),
            y.arg(),
            hidden.arg(),
            gate_stride.arg(),
            (hidden / *n_groups).arg(),
            eps.arg(),
        ],
    )
}

#[routine]
pub fn nemotron_mamba_split_bf16(
    ctx: &Ctx<'_>,
    projected: In<Tensor<c_void>>,
    gate: Out<Tensor<c_void>>,
    conv_in: Out<Tensor<c_void>>,
    dt: Out<Tensor<c_void>>,
) -> Result<(), Refusal> {
    const SPLIT_BLOCK: u32 = 256;

    let src = projected.all("a split extent")?;
    let gates = gate.all("a split extent")?;
    let conv = conv_in.all("a split extent")?;
    let heads = dt.all("a split extent")?;

    let n = src.rows;

    let projection_dim = src.stride;
    let intermediate = gates.width;
    let conv_dim = conv.width;
    let num_heads = heads.width;

    let ungated = gate.ptr.is_null();

    let total = src.elements();
    let conv_dt_total = n.saturating_mul(conv_dim.saturating_add(num_heads));
    if ungated && conv_dt_total <= 0 {
        return Err(Refusal::Empty {
            what: "rows * (conv_dim + num_heads)",
        });
    }
    if ungated {
        return ctx.fire(
            Fire::at("ssm/nemotron_h.cuh", "::pie::ssm::mamba_split_conv_dt").apply(Launch::grid(
                [conv_dt_total.unsigned_abs().div_ceil(SPLIT_BLOCK), 1, 1],
                [SPLIT_BLOCK, 1, 1],
            )),
            &[
                projected.arg(),
                conv_in.arg(),
                dt.arg(),
                projection_dim.arg(),
                intermediate.arg(),
                conv_dim.arg(),
                num_heads.arg(),
                conv_dt_total.arg(),
            ],
        );
    }
    ctx.fire(
        Fire::at("ssm/nemotron_h.cuh", "::pie::ssm::mamba_split").apply(Launch::grid(
            [total.unsigned_abs().div_ceil(SPLIT_BLOCK), 1, 1],
            [SPLIT_BLOCK, 1, 1],
        )),
        &[
            projected.arg(),
            gate.arg(),
            conv_in.arg(),
            dt.arg(),
            projection_dim.arg(),
            intermediate.arg(),
            conv_dim.arg(),
            num_heads.arg(),
            total.arg(),
        ],
    )
}

#[routine(whole)]
pub fn nemotron_mamba_ssm_batched_bf16(
    ctx: &Ctx<'_>,
    conv_out: In<Tensor<c_void>>,
    dt_precomputed: In<Tensor<f32>>,
    dt: In<Tensor<f32>>,
    a: In<Tensor<f32>>,
    d: In<Tensor<f32>>,
    dt_bias: In<Tensor<f32>>,
    da_precomputed: In<Tensor<f32>>,
    y: Out<Tensor<c_void>>,
    num_heads: Const<i32>,
    head_dim: Const<i32>,
    state_size: Const<i32>,
    n_groups: Const<i32>,
    conv_dim: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
    qo_indptr: In<Tensor<i32>>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    let ssm_state_base = rsv.slab;
    let slot_ids = rsv.slot_ids;
    // The request count is the CSR operand's own row count, and the token
    // rows are the result's rectangle; `rows != r` below is the prefill test.
    let r = qo_indptr.rows;
    let rows = y.rows;
    let qo_indptr = qo_indptr.ptr as *const u32;
    const SSM_PREFILL_BLOCK: u32 = 512;

    const SSM_DECODE_BLOCK: u32 = 256;

    let intermediate = num_heads.saturating_mul(*head_dim);
    let sequence_prefill = rows != r;
    let smem = 2 * state_size.unsigned_abs() * FLOAT;
    let (rows, heads) = (r.unsigned_abs(), num_heads.unsigned_abs());

    let (instantiation, launch) = if sequence_prefill {
        (
            "::pie::ssm::mamba_ssm_batched_prefill_reg",
            Launch::grid(
                [
                    rows,
                    heads,
                    head_dim.unsigned_abs().div_ceil(SSM_PREFILL_BLOCK / WARP),
                ],
                [SSM_PREFILL_BLOCK, 1, 1],
            )
            .smem(smem),
        )
    } else {
        (
            "::pie::ssm::mamba_ssm_batched_warp",
            Launch::grid([rows, heads, 1], [SSM_DECODE_BLOCK, 1, 1]).smem(smem),
        )
    };
    ctx.fire(
        Fire::at("ssm/nemotron_h.cuh", instantiation).apply(launch),
        &[
            conv_out.arg(),
            dt.arg(),
            a.arg(),
            d.arg(),
            dt_bias.arg(),
            dt_precomputed.arg(),
            da_precomputed.arg(),
            ssm_state_base.arg(),
            slot_ids.arg(),
            qo_indptr.arg(),
            y.arg(),
            num_heads.arg(),
            head_dim.arg(),
            state_size.arg(),
            n_groups.arg(),
            conv_dim.arg(),
            intermediate.arg(),
            0.0f32.arg(),
        ],
    )
}

#[routine(whole)]
pub fn build_nemotron_moe_ptrs_decode_batched_bf16(
    ctx: &Ctx<'_>,
    topk_idx: In<Tensor<i32>>,
    topk_w: In<Tensor<f32>>,
    norm_x: In<Tensor<c_void>>,
    top_k: Const<i32>,
    hidden: Const<i32>,
    intermediate: Const<i32>,
    banks: In<Struct<MoeBanks>>,
) -> Result<(), Refusal> {
    if banks.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the MoE bank view this statement names",
        });
    }
    let banks = unsafe { &*banks.ptr };
    // The routed fanout is the top-k table's own width, and the row count is
    // its rows: the statement placed that operand, so neither is a fact.
    let n = topk_idx.rows;
    let top_k = *top_k;
    let hidden = *hidden;
    let intermediate = *intermediate;
    let up_weight_ptrs = banks.up_weight_ptrs;
    let down_weight_ptrs = banks.down_weight_ptrs;
    let expert_up = banks.expert_up;
    let expert_act = banks.expert_act;
    let expert_out = banks.expert_out;
    let a_up_ptrs = banks.a_up_ptrs;
    let b_up_ptrs = banks.b_up_ptrs;
    let c_up_ptrs = banks.c_up_ptrs;
    let a_down_ptrs = banks.a_down_ptrs;
    let b_down_ptrs = banks.b_down_ptrs;
    let c_down_ptrs = banks.c_down_ptrs;
    let weights_out = banks.route_weights;
    let routes = n.saturating_mul(top_k);
    ctx.fire(
        Fire::at(
            "ssm/nemotron_h.cuh",
            "::pie::ssm::build_nemotron_moe_ptrs_decode_batched",
        )
        .apply(Launch::grid(
            [routes.unsigned_abs().div_ceil(PTRS_BLOCK), 1, 1],
            [PTRS_BLOCK, 1, 1],
        )),
        &[
            topk_idx.arg(),
            topk_w.arg(),
            up_weight_ptrs.arg(),
            down_weight_ptrs.arg(),
            norm_x.arg(),
            expert_up.arg(),
            expert_act.arg(),
            expert_out.arg(),
            a_up_ptrs.arg(),
            b_up_ptrs.arg(),
            c_up_ptrs.arg(),
            a_down_ptrs.arg(),
            b_down_ptrs.arg(),
            c_down_ptrs.arg(),
            weights_out.arg(),
            routes.arg(),
            top_k.arg(),
            hidden.arg(),
            intermediate.arg(),
        ],
    )
}

#[routine(whole)]
pub fn build_nemotron_moe_ptrs_aligned_bf16(
    ctx: &Ctx<'_>,
    expert_ids: In<Tensor<i32>>,
    aligned_in: In<Tensor<c_void>>,
    max_blocks: Const<i32>,
    block_size: Const<i32>,
    hidden: Const<i32>,
    intermediate: Const<i32>,
    banks: In<Struct<MoeBanks>>,
) -> Result<(), Refusal> {
    if banks.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the MoE bank view this statement names",
        });
    }
    let banks = unsafe { &*banks.ptr };
    let max_blocks = *max_blocks;
    let block_size = *block_size;
    let hidden = *hidden;
    let intermediate = *intermediate;
    let up_weight_ptrs = banks.up_weight_ptrs;
    let down_weight_ptrs = banks.down_weight_ptrs;
    let aligned_up = banks.aligned_up;
    let aligned_act = banks.aligned_act;
    let aligned_out = banks.aligned_out;
    let a_up_ptrs = banks.a_up_ptrs;
    let b_up_ptrs = banks.b_up_ptrs;
    let c_up_ptrs = banks.c_up_ptrs;
    let a_down_ptrs = banks.a_down_ptrs;
    let b_down_ptrs = banks.b_down_ptrs;
    let c_down_ptrs = banks.c_down_ptrs;
    ctx.fire(
        Fire::at(
            "ssm/nemotron_h.cuh",
            "::pie::ssm::build_nemotron_moe_ptrs_aligned",
        )
        .apply(Launch::grid(
            [max_blocks.unsigned_abs().div_ceil(PTRS_BLOCK), 1, 1],
            [PTRS_BLOCK, 1, 1],
        )),
        &[
            expert_ids.arg(),
            up_weight_ptrs.arg(),
            down_weight_ptrs.arg(),
            aligned_in.arg(),
            aligned_up.arg(),
            aligned_act.arg(),
            aligned_out.arg(),
            a_up_ptrs.arg(),
            b_up_ptrs.arg(),
            c_up_ptrs.arg(),
            a_down_ptrs.arg(),
            b_down_ptrs.arg(),
            c_down_ptrs.arg(),
            max_blocks.arg(),
            block_size.arg(),
            hidden.arg(),
            intermediate.arg(),
        ],
    )
}

#[derive(Clone, Copy)]
struct Shape {
    r: i32,
    k_h: i32,
    v_h: i32,
    k_d: i32,
    v_d: i32,
}

struct Operands {
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    g_log: *const f32,
    beta: *const f32,
    state_base: *mut c_void,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: i64,
    out: *mut f32,
    write_state: bool,
}

fn chunk_prefill(
    ctx: &Ctx<'_>,
    fla: &'static str,
    per_token: &'static str,
    ops: &Operands,
    shape: Shape,
) -> Result<(), Refusal> {
    const BK_MAX_FLA: i32 = 128;

    const BV_FLA: u32 = 128;

    let Shape {
        r,
        k_h,
        v_h,
        k_d,
        v_d,
    } = shape;
    let (rows, heads) = (r.unsigned_abs(), v_h.unsigned_abs());
    if k_d <= BK_MAX_FLA && v_d.unsigned_abs() % BV_FLA == 0 {
        return ctx.fire(
            Fire::at("ssm/gated_delta_net.cuh", fla).apply(
                Launch::grid([v_d.unsigned_abs() / BV_FLA, rows, heads], [BV_FLA, 1, 1])
                    .smem(2 * BK_MAX_FLA.unsigned_abs() * FLOAT),
            ),
            &[
                ops.q_norm.arg(),
                ops.k_norm.arg(),
                ops.v.arg(),
                ops.g_log.arg(),
                ops.beta.arg(),
                ops.state_base.arg(),
                ops.slot_ids.arg(),
                ops.qo_indptr.arg(),
                ops.slot_stride_elems.arg(),
                ops.out.arg(),
                k_h.arg(),
                v_h.arg(),
                k_d.arg(),
                v_d.arg(),
                ops.write_state.arg(),
                MaybeConst::<i32>::none().arg(),
                MaybeConst::<u8>::none().arg(),
            ],
        );
    }
    ctx.fire(
        Fire::at("ssm/gated_delta_net.cuh", per_token).apply(
            Launch::grid([rows, heads, 1], [GDN_BLOCK, 1, 1]).smem(2 * k_d.unsigned_abs() * FLOAT),
        ),
        &[
            ops.q_norm.arg(),
            ops.k_norm.arg(),
            ops.v.arg(),
            ops.g_log.arg(),
            ops.beta.arg(),
            ops.state_base.arg(),
            ops.slot_ids.arg(),
            ops.qo_indptr.arg(),
            ops.slot_stride_elems.arg(),
            ops.out.arg(),
            v_h.arg(),
            k_d.arg(),
            v_d.arg(),
        ],
    )
}

fn cached(
    ctx: &Ctx<'_>,
    instantiation: &'static str,
    ops: &Operands,
    shape: Shape,
) -> Result<(), Refusal> {
    let Shape {
        r, v_h, k_d, v_d, ..
    } = shape;
    ctx.fire(
        Fire::at("ssm/gated_delta_net.cuh", instantiation).apply(
            Launch::grid([r.unsigned_abs(), v_h.unsigned_abs(), 1], [GDN_BLOCK, 1, 1])
                .smem(k_d.unsigned_abs() * v_d.unsigned_abs() * FLOAT),
        ),
        &[
            ops.q_norm.arg(),
            ops.k_norm.arg(),
            ops.v.arg(),
            ops.g_log.arg(),
            ops.beta.arg(),
            ops.state_base.arg(),
            ops.slot_ids.arg(),
            ops.qo_indptr.arg(),
            ops.slot_stride_elems.arg(),
            ops.out.arg(),
            v_h.arg(),
            k_d.arg(),
            v_d.arg(),
            ops.write_state.arg(),
            MaybeConst::<u8>::none().arg(),
        ],
    )
}

#[routine]
pub fn chunk_gated_delta_prefill_batched(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    k_h: Const<i32>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
    qo_indptr: In<Tensor<i32>>,
    write_state: Const<bool>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    // The request count is the CSR operand's own row count.
    let r = qo_indptr.rows;
    let state_base = rsv.slab;
    let slot_ids = rsv.slot_ids;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let slot_stride_elems = rsv.slot_stride_elems;
    let write_state = *write_state;
    chunk_prefill(
        ctx,
        "::pie::ssm::chunk_gated_delta_prefill_batched_fla<::pie::ssm::f32, 128, 128>",
        "::pie::ssm::chunk_gated_delta_prefill_batched<::pie::ssm::f32, false>",
        &Operands {
            q_norm: q_norm.ptr,
            k_norm: k_norm.ptr,
            v: v.ptr,
            g_log: g_log.ptr,
            beta: beta.ptr,
            state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out: out.ptr,
            write_state,
        },
        Shape {
            r,
            k_h: *k_h,
            v_h: *v_h,
            k_d: *k_d,
            v_d: *v_d,
        },
    )
}

#[routine]
pub fn chunk_gated_delta_prefill_batched_state_bf16(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    k_h: Const<i32>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
    qo_indptr: In<Tensor<i32>>,
    write_state: Const<bool>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    // The request count is the CSR operand's own row count.
    let r = qo_indptr.rows;
    let state_base = rsv.slab;
    let slot_ids = rsv.slot_ids;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let slot_stride_elems = rsv.slot_stride_elems;
    let write_state = *write_state;
    chunk_prefill(
        ctx,
        "::pie::ssm::chunk_gated_delta_prefill_batched_fla<::pie::ssm::state_bf16, 128, 128>",
        "::pie::ssm::chunk_gated_delta_prefill_batched<::pie::ssm::state_bf16, false>",
        &Operands {
            q_norm: q_norm.ptr,
            k_norm: k_norm.ptr,
            v: v.ptr,
            g_log: g_log.ptr,
            beta: beta.ptr,
            state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out: out.ptr,
            write_state,
        },
        Shape {
            r,
            k_h: *k_h,
            v_h: *v_h,
            k_d: *k_d,
            v_d: *v_d,
        },
    )
}

#[routine]
pub fn chunk_gated_delta_prefill_batched_cached(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
    qo_indptr: In<Tensor<i32>>,
    write_state: Const<bool>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    // The request count is the CSR operand's own row count.
    let r = qo_indptr.rows;
    let state_base = rsv.slab;
    let slot_ids = rsv.slot_ids;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let slot_stride_elems = rsv.slot_stride_elems;
    let write_state = *write_state;
    cached(
        ctx,
        "::pie::ssm::chunk_gated_delta_prefill_batched_cached<::pie::ssm::f32, false>",
        &Operands {
            q_norm: q_norm.ptr,
            k_norm: k_norm.ptr,
            v: v.ptr,
            g_log: g_log.ptr,
            beta: beta.ptr,
            state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out: out.ptr,
            write_state,
        },
        Shape {
            r,
            k_h: 0,
            v_h: *v_h,
            k_d: *k_d,
            v_d: *v_d,
        },
    )
}

#[routine]
pub fn chunk_gated_delta_prefill_batched_cached_state_bf16(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
    qo_indptr: In<Tensor<i32>>,
    write_state: Const<bool>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    // The request count is the CSR operand's own row count.
    let r = qo_indptr.rows;
    let state_base = rsv.slab;
    let slot_ids = rsv.slot_ids;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let slot_stride_elems = rsv.slot_stride_elems;
    let write_state = *write_state;
    cached(
        ctx,
        "::pie::ssm::chunk_gated_delta_prefill_batched_cached<::pie::ssm::state_bf16, false>",
        &Operands {
            q_norm: q_norm.ptr,
            k_norm: k_norm.ptr,
            v: v.ptr,
            g_log: g_log.ptr,
            beta: beta.ptr,
            state_base,
            slot_ids,
            qo_indptr,
            slot_stride_elems,
            out: out.ptr,
            write_state,
        },
        Shape {
            r,
            k_h: 0,
            v_h: *v_h,
            k_d: *k_d,
            v_d: *v_d,
        },
    )
}

#[routine(canon = "ssm.gated_delta")]
pub fn recurrent_gated_delta_step_batched_gqa_state_bf16(
    ctx: &Ctx<'_>,
    q_norm_kh: In<Tensor<f32>>,
    k_norm_kh: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    k_h: Const<i32>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>,
    r: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    let r = *r;
    let state_base = rsv.slab;
    let slot_ids = rsv.slot_ids;
    let slot_stride_elems = rsv.slot_stride_elems;
    const SMEM_BV: u32 = 128;

    const GDN_SMEM_ARM_WIDTH: i32 = 128;

    if *v_h % *k_h != 0 {
        return Err(Refusal::Narrow {
            what: "v_h per k_h",
            at: i64::from(*v_h),
        });
    }

    let (instantiation, launch) = if *v_d == GDN_SMEM_ARM_WIDTH && *k_d == GDN_SMEM_ARM_WIDTH {
        (
            "::pie::ssm::recurrent_step_batched_gqa_smem<::pie::ssm::gqa_smem_bv>",
            Launch::grid(
                [
                    v_d.unsigned_abs().div_ceil(SMEM_BV),
                    r.unsigned_abs(),
                    v_h.unsigned_abs(),
                ],
                [SMEM_BV, 1, 1],
            )
            .smem(k_d.unsigned_abs() * SMEM_BV * 2 + 2 * k_d.unsigned_abs() * FLOAT),
        )
    } else {
        (
            "::pie::ssm::recurrent_step_batched_gqa<::pie::ssm::state_bf16, false>",
            Launch::grid([r.unsigned_abs(), v_h.unsigned_abs(), 1], [GDN_BLOCK, 1, 1])
                .smem(2 * k_d.unsigned_abs() * FLOAT),
        )
    };
    ctx.fire(
        Fire::at("ssm/gated_delta_net.cuh", instantiation).apply(launch),
        &[
            q_norm_kh.arg(),
            k_norm_kh.arg(),
            v.arg(),
            g_log.arg(),
            beta.arg(),
            state_base.arg(),
            slot_ids.arg(),
            slot_stride_elems.arg(),
            out.arg(),
            k_h.arg(),
            v_h.arg(),
            k_d.arg(),
            v_d.arg(),
        ],
    )
}

#[routine]
pub fn recurrent_gated_delta_step_batched(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>,
    r: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    let r = *r;
    let state_base = rsv.slab;
    let slot_ids = rsv.slot_ids;
    let slot_stride_elems = rsv.slot_stride_elems;
    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net.cuh",
            "::pie::ssm::recurrent_step_batched<::pie::ssm::f32, false>",
        )
        .apply(recurrent_scan(
            r.unsigned_abs(),
            v_h.unsigned_abs(),
            k_d.unsigned_abs(),
        )),
        &[
            q_norm.arg(),
            k_norm.arg(),
            v.arg(),
            g_log.arg(),
            beta.arg(),
            state_base.arg(),
            slot_ids.arg(),
            slot_stride_elems.arg(),
            out.arg(),
            v_h.arg(),
            k_d.arg(),
            v_d.arg(),
        ],
    )
}

#[routine]
pub fn recurrent_gated_delta_step_batched_state_bf16(
    ctx: &Ctx<'_>,
    q_norm: In<Tensor<f32>>,
    k_norm: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>,
    r: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    let r = *r;
    let state_base = rsv.slab;
    let slot_ids = rsv.slot_ids;
    let slot_stride_elems = rsv.slot_stride_elems;
    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net.cuh",
            "::pie::ssm::recurrent_step_batched<::pie::ssm::state_bf16, false>",
        )
        .apply(recurrent_scan(
            r.unsigned_abs(),
            v_h.unsigned_abs(),
            k_d.unsigned_abs(),
        )),
        &[
            q_norm.arg(),
            k_norm.arg(),
            v.arg(),
            g_log.arg(),
            beta.arg(),
            state_base.arg(),
            slot_ids.arg(),
            slot_stride_elems.arg(),
            out.arg(),
            v_h.arg(),
            k_d.arg(),
            v_d.arg(),
        ],
    )
}

#[routine]
pub fn recurrent_gated_delta_step_batched_gqa(
    ctx: &Ctx<'_>,
    q_norm_kh: In<Tensor<f32>>,
    k_norm_kh: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    k_h: Const<i32>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>,
    r: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    let r = *r;
    let state_base = rsv.slab;
    let slot_ids = rsv.slot_ids;
    let slot_stride_elems = rsv.slot_stride_elems;
    if *v_h % *k_h != 0 {
        return Err(Refusal::Narrow {
            what: "v_h per k_h",
            at: i64::from(*v_h),
        });
    }
    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net.cuh",
            "::pie::ssm::recurrent_step_batched_gqa<::pie::ssm::f32, false>",
        )
        .apply(recurrent_scan(
            r.unsigned_abs(),
            v_h.unsigned_abs(),
            k_d.unsigned_abs(),
        )),
        &[
            q_norm_kh.arg(),
            k_norm_kh.arg(),
            v.arg(),
            g_log.arg(),
            beta.arg(),
            state_base.arg(),
            slot_ids.arg(),
            slot_stride_elems.arg(),
            out.arg(),
            k_h.arg(),
            v_h.arg(),
            k_d.arg(),
            v_d.arg(),
        ],
    )
}

#[routine]
pub fn chunk_gated_delta_prefill_batched_warp_tiled_gqa(
    ctx: &Ctx<'_>,
    q_norm_kh: In<Tensor<f32>>,
    k_norm_kh: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    k_h: Const<i32>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
    qo_indptr: In<Tensor<i32>>,
    write_state: Const<bool>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    // The request count is the CSR operand's own row count.
    let r = qo_indptr.rows;
    let state_base = rsv.slab;
    let slot_ids = rsv.slot_ids;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let slot_stride_elems = rsv.slot_stride_elems;
    let write_state = *write_state;
    if *v_h % *k_h != 0 {
        return Err(Refusal::Narrow {
            what: "v_h per k_h",
            at: i64::from(*v_h),
        });
    }
    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net.cuh",
            "::pie::ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa<::pie::ssm::f32, false>",
        )
        .apply(warp_tiled_scan(
            r.unsigned_abs(),
            v_h.unsigned_abs(),
            v_d.unsigned_abs(),
        )),
        &[
            q_norm_kh.arg(),
            k_norm_kh.arg(),
            v.arg(),
            g_log.arg(),
            beta.arg(),
            state_base.arg(),
            slot_ids.arg(),
            qo_indptr.arg(),
            slot_stride_elems.arg(),
            out.arg(),
            k_h.arg(),
            v_h.arg(),
            k_d.arg(),
            v_d.arg(),
            write_state.arg(),
            core::ptr::null::<u8>().arg(),
        ],
    )
}

#[routine]
pub fn chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16(
    ctx: &Ctx<'_>,
    q_norm_kh: In<Tensor<f32>>,
    k_norm_kh: In<Tensor<f32>>,
    v: In<Tensor<f32>>,
    g_log: In<Tensor<f32>>,
    beta: In<Tensor<f32>>,
    out: Out<Tensor<f32>>,
    k_h: Const<i32>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>,
    rsv: In<Struct<RecurrentState>>,
    qo_indptr: In<Tensor<i32>>,
    write_state: Const<bool>,
) -> Result<(), Refusal> {
    if rsv.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the recurrent view this statement names",
        });
    }
    let rsv = unsafe { &*rsv.ptr };

    // The request count is the CSR operand's own row count.
    let r = qo_indptr.rows;
    let state_base = rsv.slab;
    let slot_ids = rsv.slot_ids;
    let qo_indptr = qo_indptr.ptr as *const u32;
    let slot_stride_elems = rsv.slot_stride_elems;
    let write_state = *write_state;
    if *v_h % *k_h != 0 {
        return Err(Refusal::Narrow {
            what: "v_h per k_h",
            at: i64::from(*v_h),
        });
    }
    ctx.fire(Fire::at("ssm/gated_delta_net.cuh", "::pie::ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa<::pie::ssm::state_bf16, false>").apply(warp_tiled_scan(r.unsigned_abs(), v_h.unsigned_abs(), v_d.unsigned_abs())), &[
                q_norm_kh.arg(),
                k_norm_kh.arg(),
                v.arg(),
                g_log.arg(),
                beta.arg(),
                state_base.arg(),
                slot_ids.arg(),
                qo_indptr.arg(),
                slot_stride_elems.arg(),
                out.arg(),
                k_h.arg(),
                v_h.arg(),
                k_d.arg(),
                v_d.arg(),
                write_state.arg(),
                core::ptr::null::<u8>().arg(),
            ])
}

#[routine(untraced)]
pub fn verify_stash_store(
    _ctx: &Ctx<'_>,
    _mixed_qkv: In<Tensor<bf16>>,
    _a: In<Tensor<bf16>>,
    _b: In<Tensor<bf16>>,
    _tokens: i32,
) -> Result<(), Refusal> {
    Err(Refusal::Absent {
        what: "the verify-stash slab: `RecurrentStateLayout` allocates \
                                 conv state, recurrent state and the MTP pending hidden, \
                                 and none of the three is this pool",
    })
}

#[routine(untraced)]
pub fn verify_stash_load(
    _ctx: &Ctx<'_>,
    _mixed_qkv: Out<Tensor<bf16>>,
    _a: Out<Tensor<bf16>>,
    _b: Out<Tensor<bf16>>,
    _tokens: i32,
) -> Result<(), Refusal> {
    Err(Refusal::Absent {
        what: "the verify-stash slab; see `verify_stash_store`",
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slab {
    Conv,
    Recurrent,
}

#[derive(Clone, Copy, Debug)]
pub struct Gdn {
    pub k_h: i32,
    pub v_h: i32,
    pub k_d: i32,
    pub v_d: i32,
    pub conv_dim: i32,
    pub conv_k: i32,
    pub n_groups: i32,
    pub conv_stride_elems: i64,
    pub state_stride_elems: i64,
    pub slot_ids_d: *const i32,
    pub write_state: bool,
}
