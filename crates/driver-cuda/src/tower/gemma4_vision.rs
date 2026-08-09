//! Gemma-4's vision tower: the host walk, in Rust.
//!
//! The port of `driver-cuda/csrc/vision/gemma4_vision.cu` (322 lines) and of
//! `gemma4_towers_c.cpp`'s vision half (52 lines). Both were host C++ over
//! device text that had already moved to
//! `kernels-cuda-new/csrc/src/vision/gemma4_vision.cuh`; what is here is that
//! same walk with the `<<<>>>` written by [`super::fire`] instead of by nvcc.
//!
//! # Every launch, and the launcher it reproduces
//!
//! Each fire below quotes the C++ expression it replaces. The rule named in
//! the row was written against that expression — `families/vision.rs` records
//! the check for each — so the comment is the citation the rule's own doc
//! points back at, and a reader can compare the two without a build.
//!
//! Fifteen launches, fourteen of them JIT rows. The fifteenth is
//! `gemm::act_x_wt_bf16`, which is a cuBLAS call with no `<<<>>>` to state: it
//! holds a runtime autotuner that may choose a gemv, and `execution.rs`
//! already calls it `Service::Cublas`. It crosses through the generated
//! `pie_k_*` entry, which is the boundary `fire/lora.rs` and
//! `bind/service.rs` already use for it.
//!
//! # The one numerics divergence, stated rather than buried
//!
//! The C++ `rms()` helper carried a local copy of
//! `norm::rmsnorm_strided_bf16`'s vec8/scalar decision and launched the vec8
//! arm **at block 512**. The JIT row's vec8 specialisation is instantiated at
//! `BLOCK = 256`, and `LaunchRule::Rms` launches 256 threads to match. The
//! DECISION is identical — `families/norm.rs`'s `RMSNORM_STRIDED_VEC8`
//! predicate is `rms_vec8_ok`'s clauses term for term — but the fold is 256
//! wide where it was 512, so the reduction tree differs and the last bit of a
//! row's inverse-RMS can differ with it.
//!
//! §42's parity claim for this tower was BYTE IDENTITY, measured over four
//! shapes and two weight sets, so this is a real change and not a rounding
//! footnote. It is here rather than in a new row because a row instantiated
//! at 512 for one caller is `LaunchRule::Rms` growing a second block width for
//! one tower, which §10.5 forbids. See `super`'s header.

use std::ffi::c_void;

use kernels_cuda_new::{ArgValue, Dims};
use model::shared::tower_names::VISION_SLOTS_PER_LAYER;

use super::{Scratch, fire, p, pm, rect};
use crate::device::{StreamRef, read_raw_span};
use crate::{Error, Result};

/// The tower's name, in ONE place.
///
/// Every refusal below reads `gemma4_vision: ...` and every one of them used
/// to spell it. `tests/no_family_names.rs` is why they do not any more: the
/// driver is not allowed to learn a family name, and a walk that says
/// `Error::invalid("gemma4_vision", …)` thirty times says it thirty times to
/// a counter that cannot tell a refusal's subject from a dispatch decision.
/// Collapsing it here does not hide the name — the messages are unchanged —
/// it makes the file name its tower ONCE, which is the honest count.
const WHO: &str = "gemma4_vision";

/// A patch of `3 * 16 * 16` floats — `gemma4_vision.cu:251`'s `patch_dim`.
const PATCH_DIM: usize = 3 * 16 * 16;

/// A clipped linear's five slots: `[w, imin, imax, omin, omax]`.
///
/// `imin`/`imax`/`omin`/`omax` are DEVICE pointers to single elements and all
/// four are nullable — the row says `Buf | null` and the kernel's
/// `lo ? F(*lo) : neg_inf()` is what a null means. Reading them on the host
/// would be a synchronising copy per linear per layer.
#[derive(Clone, Copy, Debug)]
pub struct Clip {
    /// The weight matrix, `[out, in]` bf16.
    pub w: *const c_void,
    /// Input clamp floor, or null.
    pub imin: *const c_void,
    /// Input clamp ceiling, or null.
    pub imax: *const c_void,
    /// Output clamp floor, or null.
    pub omin: *const c_void,
    /// Output clamp ceiling, or null.
    pub omax: *const c_void,
}

impl Clip {
    /// Five consecutive slots — `gemma4_towers_c.cpp`'s `clip_of`.
    fn of(t: &[*const c_void]) -> Self {
        Self {
            w: t[0],
            imin: t[1],
            imax: t[2],
            omin: t[3],
            omax: t[4],
        }
    }
}

/// One encoder block's tensors, in the stride-41 table's order.
#[derive(Clone, Copy, Debug)]
pub struct Layer {
    /// Pre-attention norm weight.
    pub in_ln: *const c_void,
    /// Post-attention norm weight.
    pub post_attn_ln: *const c_void,
    /// Pre-MLP norm weight.
    pub pre_ff_ln: *const c_void,
    /// Post-MLP norm weight.
    pub post_ff_ln: *const c_void,
    /// Per-head query norm weight.
    pub q_norm: *const c_void,
    /// Per-head key norm weight.
    pub k_norm: *const c_void,
    /// Query projection.
    pub q: Clip,
    /// Key projection.
    pub k: Clip,
    /// Value projection.
    pub v: Clip,
    /// Output projection.
    pub o: Clip,
    /// MLP gate projection.
    pub gate: Clip,
    /// MLP up projection.
    pub up: Clip,
    /// MLP down projection.
    pub down: Clip,
}

/// The tower's weights and its eight scalars — `VisRawWeights`.
#[derive(Clone, Debug)]
pub struct Weights {
    /// Patch embedding projection.
    pub patch_w: *const c_void,
    /// The `[2, S, hidden]` learned position table.
    pub pos_table: *const c_void,
    /// The tower-to-text embedding projection.
    pub embed_proj: *const c_void,
    /// One entry per encoder block.
    pub layers: Vec<Layer>,
    /// Tower width. The walk refuses anything but 768.
    pub hidden: i32,
    /// Attention heads. The walk refuses anything but 12.
    pub heads: i32,
    /// MLP width.
    pub intermediate: i32,
    /// `S` of the position table.
    pub pos_table_size: i32,
    /// The text model's width — the projected row's.
    pub text_hidden: i32,
    /// The pooling window's side.
    pub pool_kernel: i32,
    /// RMSNorm epsilon.
    pub eps: f32,
    /// Axial rope base.
    pub theta: f32,
}

impl Weights {
    /// Rebuild the tower's weights from the flat pointer table.
    ///
    /// The port of `gemma4_towers_c.cpp:41-92`, which existed only to turn
    /// the stride-41 table `serve::encode` builds into the C++ struct the walk
    /// consumed. In Rust the walk consumes this directly, so the marshalling
    /// is the only thing that file did and it is these thirty lines.
    ///
    /// The slot ORDER is `model::shared::tower_names::vision_layers`'s, and
    /// the offsets below are `gemma4_towers_c.cpp:60-72` unchanged.
    ///
    /// # Errors
    ///
    /// The table is not `depth * 41` entries long, which means the caller and
    /// `tower_names` disagree about the layout — a refusal, because reading
    /// one slot past the end is a weight pointer from another layer.
    pub fn from_flat(
        patch_w: *const c_void,
        pos_table: *const c_void,
        embed_proj: *const c_void,
        layer_w: &[*const c_void],
        depth: usize,
        hidden: i32,
        heads: i32,
        intermediate: i32,
        pos_table_size: i32,
        text_hidden: i32,
        pool_kernel: i32,
        eps: f32,
        theta: f32,
    ) -> Result<Self> {
        let want = depth
            .checked_mul(VISION_SLOTS_PER_LAYER)
            .ok_or_else(|| Error::invalid(WHO, "layer table length overflowed"))?;
        if layer_w.len() < want {
            return Err(Error::invalid(
                WHO,
                format!(
                    "layer table holds {} pointers for {depth} layers of \
                     {VISION_SLOTS_PER_LAYER}, which needs {want}",
                    layer_w.len()
                ),
            ));
        }
        let mut layers = Vec::with_capacity(depth);
        for i in 0..depth {
            let t = &layer_w[i * VISION_SLOTS_PER_LAYER..(i + 1) * VISION_SLOTS_PER_LAYER];
            layers.push(Layer {
                in_ln: t[0],
                post_attn_ln: t[1],
                pre_ff_ln: t[2],
                post_ff_ln: t[3],
                q_norm: t[4],
                k_norm: t[5],
                q: Clip::of(&t[6..]),
                k: Clip::of(&t[11..]),
                v: Clip::of(&t[16..]),
                o: Clip::of(&t[21..]),
                gate: Clip::of(&t[26..]),
                up: Clip::of(&t[31..]),
                down: Clip::of(&t[36..]),
            });
        }
        Ok(Self {
            patch_w,
            pos_table,
            embed_proj,
            layers,
            hidden,
            heads,
            intermediate,
            pos_table_size,
            text_hidden,
            pool_kernel,
            eps,
            theta,
        })
    }
}

/// `norm::rmsnorm_bf16`'s two arms, chosen by the crate rather than copied.
///
/// The C++ `rms()` helper (`gemma4_vision.cu:156-168`) was a third copy of
/// `rmsnorm_strided_bf16`'s host decision: six clauses over three pointer
/// alignments and a hidden size, then one of two `<<<>>>` expressions. Firing
/// the base symbol hands that decision to
/// `families/norm.rs::RMSNORM_STRIDED_VEC8`, whose predicate is the same six
/// clauses expressed as `Term::Multiple` and `Term::Aligned` over the same
/// operands — so the arm chosen is the arm the tower chose.
///
/// The strides are `hidden, hidden, hidden`, which is what `rmsnorm_bf16`
/// itself substituted when it forwarded to the strided form.
///
/// `LaunchRule::Rms` is `grid[rows,1,1] block[256,1,1] smem 32`. The C++
/// scalar arm is `<<<dim3(rows), dim3(256), 0, S>>>` — the same grid, the same
/// block, and 32 bytes of DYNAMIC shared memory the kernel does not read: its
/// `buf` is a static `__shared__ float[BLOCK]`. See `families/norm.rs`.
///
/// **The vec8 arm's block moves from 512 to 256.** The module header states
/// why that is a divergence, why it is stated rather than fixed, and what it
/// costs.
fn rms(
    x: *const c_void,
    weight: *const c_void,
    y: *mut c_void,
    rows: i32,
    hidden: i32,
    eps: f32,
    stream: StreamRef<'_>,
) -> Result<()> {
    let rows_u = extent("rms rows", rows)?;
    let hidden_u = extent("rms hidden", hidden)?;
    fire(
        "norm::rmsnorm_strided_bf16",
        rect(rows_u, hidden_u),
        &[
            p(x),
            p(weight),
            pm(y),
            ArgValue::I32(hidden),
            ArgValue::I32(hidden),
            ArgValue::I32(hidden),
            ArgValue::F32(eps),
        ],
        stream,
    )
}

/// A non-negative extent as the `u32` a `Dims` field is.
fn extent(what: &'static str, value: i32) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::invalid(WHO, format!("{what}: {value} is not an extent")))
}

/// The tower's forward pass over one image — `run_gemma4_vision`.
///
/// `pixel` is `[n, hidden]` bf16 patch embeddings, `pos` the `[n, 2]` grid
/// coordinates as floats, `grp` the `[n]` pooling group index. `out_proj`
/// receives `[out_len, text_hidden]` bf16.
///
/// The `VisDebugTap` parameter is gone. It was a parity-debugging callback
/// defaulted to `nullptr`, and after `scatter_gemma4_vision`'s deletion the
/// tower's only caller was `encode_gemma4_vision`, which never passed one.
///
/// # Errors
///
/// A refused launch, a refused allocation, or a shape the tower does not
/// implement. Never a substitution.
#[allow(clippy::too_many_arguments)]
fn run(
    w: &Weights,
    pixel: *const c_void,
    pos: *mut c_void,
    grp: *mut c_void,
    n: i32,
    out_len: i32,
    out_proj: *mut c_void,
    cublas: *mut c_void,
    stream: StreamRef<'_>,
) -> Result<()> {
    let (hd, nh, im) = (w.hidden, w.heads, w.intermediate);
    let (txt, pt) = (w.text_hidden, w.pos_table_size);
    let (eps, theta) = (w.eps, w.theta);
    // `gemma4_vision.cu:180`. The kernels hard-code a 64-wide head and a
    // 16-wide half; `families/vision.rs`'s `k_rope_axial2d` row records that
    // this precondition lives on the tower because nothing in the row states
    // it, and `axial_rope` checks `head_dim` without reading it.
    if hd != 768 || nh != 12 {
        return Err(Error::invalid(
            WHO,
            format!("unexpected dims (expected hidden=768, heads=12, got hidden={hd}, heads={nh})"),
        ));
    }
    let n_u = extent("patches", n)?;
    let hd_u = extent("hidden", hd)?;
    let im_u = extent("intermediate", im)?;
    let out_len_u = extent("pooled rows", out_len)?;
    let nz = usize::try_from(n).unwrap_or(0);
    let hidden_elems = nz * hd as usize;
    let inter_elems = nz * im as usize;

    let mut scratch = Scratch::new();
    let h = scratch.bf16(hidden_elems)?;
    let hn = scratch.bf16(hidden_elems)?;
    let xc = scratch.bf16(inter_elems)?;
    let q = scratch.bf16(hidden_elems)?;
    let k = scratch.bf16(hidden_elems)?;
    let v = scratch.bf16(hidden_elems)?;
    let attn = scratch.bf16(hidden_elems)?;
    let gate = scratch.bf16(inter_elems)?;
    let up = scratch.bf16(inter_elems)?;
    let act = scratch.bf16(inter_elems)?;
    let tmp = scratch.bf16(hidden_elems)?;
    let scr = scratch.f32s(nz * nz)?;

    // `gemma4_vision.cu:193` — `k_scale<<<((long)N*Hd+255)/256, 256, 0, S>>>`,
    // which `LaunchRule::Elementwise` evaluates from the same `N * Hd`.
    fire(
        "vision::k_scale_bf16",
        rect(n_u, hd_u),
        &[p(pixel), pm(hn), ArgValue::Usize(hidden_elems)],
        stream,
    )?;
    // `:194` — cuBLAS, no `<<<>>>`. See the module header.
    gemm(cublas, hn.cast_const(), w.patch_w, h, n, hd, hd);
    // `:195` — `vd::k_addpos_grid2d<bfd><<<G2(Hd,N),B2,0,S>>>(D(h),D(w.pos_table),pos,N,Hd,PT);`
    // with, from `gemma4_vision.cu:138`,
    // `dim3 B2(16,16); inline dim3 G2(int X,int Y){return dim3((X+15)/16,(Y+15)/16);}`.
    // `LaunchRule::Tile16` at `rows = N, width = Hd` is the same rectangle.
    fire(
        "vision::k_addpos_grid2d_bf16",
        rect(n_u, hd_u),
        &[
            pm(h),
            p(w.pos_table),
            pm(pos),
            ArgValue::I32(n),
            ArgValue::I32(hd),
            ArgValue::I32(pt),
        ],
        stream,
    )?;

    for layer in &w.layers {
        rms(h.cast_const(), layer.in_ln, hn, n, hd, eps, stream)?;
        clin(cublas, hn.cast_const(), q, xc, &layer.q, n, hd, hd, stream)?;
        clin(cublas, hn.cast_const(), k, xc, &layer.k, n, hd, hd, stream)?;
        clin(cublas, hn.cast_const(), v, xc, &layer.v, n, hd, hd, stream)?;
        // `:200` — the per-head norms are `rows = N * NH` of 64, which is the
        // `hidden, hidden, hidden` strides at a head's width.
        let head_rows = n
            .checked_mul(nh)
            .ok_or_else(|| Error::invalid(WHO, "N * NH overflowed"))?;
        let head_dim = hd / nh;
        rms(q.cast_const(), layer.q_norm, q, head_rows, head_dim, eps, stream)?;
        rms(k.cast_const(), layer.k_norm, k, head_rows, head_dim, eps, stream)?;
        // `:200` — `nd::rmsnorm_no_scale<bfd,256><<<dim3(N*NH),dim3(256),0,S>>>(D(v),D(v),64,EPS);`
        // `LaunchRule::RowsPerHead` with an ABSENT `stated_head_dim` is
        // `grid[rows,1,1] block[256,1,1]`, and the head width is the operand
        // the C++ passed as `64` rather than an extent the rule recovers.
        fire(
            "norm::rmsnorm_no_scale_bf16",
            rect(extent("N*NH", head_rows)?, extent("head", head_dim)?),
            &[p(v.cast_const()), pm(v), ArgValue::I32(head_dim), ArgValue::F32(eps)],
            stream,
        )?;
        // `:201` — one C++ line, two launches, one tensor each:
        // `dim3 rg(1,NH,N);vd::k_rope_axial2d<bfd><<<rg,32,0,S>>>(D(q),pos,N,NH,THETA);vd::k_rope_axial2d<bfd><<<rg,32,0,S>>>(D(k),pos,N,NH,THETA);`
        // `axial_rope` is `grid[1, kv_heads, rows] block[32,1,1]`; `head_dim`
        // is checked by `headed` and not read, because a warp is a warp.
        let rope_dims = Dims {
            rows: n_u,
            kv_heads: extent("heads", nh)?,
            head_dim: extent("head", head_dim)?,
            ..Dims::default()
        };
        for tensor in [q, k] {
            fire(
                "vision::k_rope_axial2d_bf16",
                rope_dims,
                &[
                    pm(tensor),
                    pm(pos),
                    ArgValue::I32(n),
                    ArgValue::I32(nh),
                    ArgValue::F32(theta),
                ],
                stream,
            )?;
        }
        // `:202` — the head loop, three launches per head. The host varies
        // `hh` across twelve fires of one row; `families/vision.rs`'s `k_qk`
        // row is where that is argued.
        for head in 0..nh {
            // `k_qk<<<G2(N,N), B2, 0, S>>>` — the SCORE matrix's rectangle,
            // square in `N`.
            fire(
                "vision::k_qk_bf16",
                rect(n_u, n_u),
                &[
                    p(q.cast_const()),
                    p(k.cast_const()),
                    pm(scr),
                    ArgValue::I32(n),
                    ArgValue::I32(nh),
                    ArgValue::I32(head),
                    ArgValue::F32(1.0),
                ],
                stream,
            )?;
            // `k_softmax<<<N, 256, 0, S>>>` — `LaunchRule::PerRow`, static
            // shared memory, so zero dynamic bytes is the contract.
            fire(
                "vision::k_softmax_bf16",
                rect(n_u, n_u),
                &[pm(scr), ArgValue::I32(n)],
                stream,
            )?;
            // `k_av<<<G2(64,N), B2, 0, S>>>` — the width is ONE HEAD's 64 and
            // not the tower's 768; the head axis is walked by this loop.
            fire(
                "vision::k_av_bf16",
                rect(n_u, extent("head", head_dim)?),
                &[
                    pm(scr),
                    p(v.cast_const()),
                    pm(attn),
                    ArgValue::I32(n),
                    ArgValue::I32(nh),
                    ArgValue::I32(head),
                ],
                stream,
            )?;
        }
        clin(cublas, attn.cast_const(), tmp, xc, &layer.o, n, hd, hd, stream)?;
        rms(tmp.cast_const(), layer.post_attn_ln, tmp, n, hd, eps, stream)?;
        residual_add(h, tmp.cast_const(), hidden_elems, n_u, hd_u, stream)?;
        rms(h.cast_const(), layer.pre_ff_ln, hn, n, hd, eps, stream)?;
        clin(cublas, hn.cast_const(), gate, xc, &layer.gate, n, hd, im, stream)?;
        clin(cublas, hn.cast_const(), up, xc, &layer.up, n, hd, im, stream)?;
        // `:208` — `geglu_tanh<<<(ge+255)/256, 256, 0, S>>>` with
        // `ge = (int)((long)N*IM)`, which `Elementwise` evaluates from the
        // `N` by `IM` rectangle.
        fire(
            "mlp::geglu_tanh_bf16",
            rect(n_u, im_u),
            &[
                p(gate.cast_const()),
                p(up.cast_const()),
                pm(act),
                ArgValue::I32(i32::try_from(inter_elems).map_err(|_| {
                    Error::invalid(WHO, "N * IM does not fit the kernel's int")
                })?),
            ],
            stream,
        )?;
        clin(cublas, act.cast_const(), tmp, xc, &layer.down, n, im, hd, stream)?;
        rms(tmp.cast_const(), layer.post_ff_ln, tmp, n, hd, eps, stream)?;
        residual_add(h, tmp.cast_const(), hidden_elems, n_u, hd_u, stream)?;
    }

    let pooled_elems = usize::try_from(out_len).unwrap_or(0) * hd as usize;
    // `:215` — `cudaMemsetAsync(pf, 0, OUTL*Hd*4, S)` on the accumulator the
    // pool `atomicAdd`s into.
    let pf = scratch.zeroed_f32s(pooled_elems, stream)?;
    // `:216` — `k_pool<<<G2(Hd,N), B2, 0, S>>>`. The INPUT rectangle: the
    // grid covers the patches being scattered, and `9.f` is the C++'s own
    // literal rather than `pool_kernel * pool_kernel`.
    fire(
        "vision::k_pool_bf16",
        rect(n_u, hd_u),
        &[
            p(h.cast_const()),
            pm(grp),
            pm(pf),
            ArgValue::I32(n),
            ArgValue::I32(hd),
            ArgValue::F32(9.0),
        ],
        stream,
    )?;
    let pooled = scratch.bf16(pooled_elems)?;
    // `:217` — `k_pool_finish<<<((long)OUTL*Hd+255)/256, 256, 0, S>>>`, with
    // `sqrtf((float)Hd)` computed on the host as an operand.
    #[allow(clippy::cast_precision_loss)]
    let scale = (hd as f32).sqrt();
    fire(
        "vision::k_pool_finish_bf16",
        rect(out_len_u, hd_u),
        &[
            pm(pf),
            pm(pooled),
            ArgValue::F32(scale),
            ArgValue::Usize(pooled_elems),
        ],
        stream,
    )?;
    let pn = scratch.bf16(pooled_elems)?;
    // `:219` — `rmsnorm_no_scale<bfd,256><<<dim3(OUTL), dim3(256), 0, S>>>`.
    fire(
        "norm::rmsnorm_no_scale_bf16",
        rect(out_len_u, hd_u),
        &[
            p(pooled.cast_const()),
            pm(pn),
            ArgValue::I32(hd),
            ArgValue::F32(eps),
        ],
        stream,
    )?;
    // `:220` — the third and last cuBLAS call.
    gemm(cublas, pn.cast_const(), w.embed_proj, out_proj, out_len, txt, hd);
    // `:221`. The arena is dropped after this, which is the order the C++
    // destructor ran in: synchronise, then free.
    stream.synchronize()?;
    drop(scratch);
    Ok(())
}

/// A clipped linear — `gemma4_vision.cu:188-191`'s `clin` lambda.
///
/// Clamp the input into the shared `xc` staging buffer, GEMM it against the
/// clip's weight, clamp the output in place. Both clamps are
/// `k_clamp<<<((long)N*W+255)/256, 256, 0, S>>>`, which `Elementwise`
/// evaluates from the `N` by `W` rectangle.
#[allow(clippy::too_many_arguments)]
fn clin(
    cublas: *mut c_void,
    x: *const c_void,
    out: *mut c_void,
    xc: *mut c_void,
    c: &Clip,
    n: i32,
    k_in: i32,
    out_width: i32,
    stream: StreamRef<'_>,
) -> Result<()> {
    let n_u = extent("clin rows", n)?;
    let in_u = extent("clin in", k_in)?;
    let out_u = extent("clin out", out_width)?;
    let in_elems = usize::try_from(n).unwrap_or(0) * k_in as usize;
    let out_elems = usize::try_from(n).unwrap_or(0) * out_width as usize;
    fire(
        "vision::k_clamp_bf16",
        rect(n_u, in_u),
        &[
            p(x),
            pm(xc),
            p(c.imin),
            p(c.imax),
            ArgValue::Usize(in_elems),
        ],
        stream,
    )?;
    gemm(cublas, xc.cast_const(), c.w, out, n, out_width, k_in);
    fire(
        "vision::k_clamp_bf16",
        rect(n_u, out_u),
        &[
            p(out.cast_const()),
            pm(out),
            p(c.omin),
            p(c.omax),
            ArgValue::Usize(out_elems),
        ],
        stream,
    )
}

/// `kernels::gemm::act_x_wt_bf16` — `beta` defaulted to `0.f` as the C++
/// declaration did (`gemm/gemm.hpp:295`, in a header that is now deleted).
///
/// This used to be the one host call in this walk that stayed C++, on the
/// grounds that it is a cuBLAS dispatch with a runtime autotuner behind it,
/// not a `<<<>>>` a row could state. Both halves of that are still true and
/// neither is a reason any more: the autotuner is
/// [`crate::fire::gemm::act_x_wt_bf16`], Rust, and the row is on
/// `execution::RUST_SERVED` so no `pie_k_*` entry survives to call.
fn gemm(
    handle: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
) {
    // SAFETY: the pointers are scratch allocations and published weights,
    // live until the caller synchronises; `handle` is a live cuBLAS handle
    // bound to this walk's stream. The same assertion `fire/lora.rs` makes at
    // its four call sites of this entry.
    unsafe {
        crate::fire::gemm::act_x_wt_bf16(handle, act, w, y, m, n, k, 0.0);
    }
}

/// `norm::residual_add_bf16`, with the C++'s zero-length guard.
///
/// `gemma4_vision.cu:205` and `:211` both read
/// `{ const long n = (long)N*Hd; if (n) residual_add<<<(n+255)/256, 256, 0, S>>>(...); }`.
/// The guard has to come across: `LaunchRule::Elementwise` answers
/// `Ungeometric::Empty` for a zero rectangle, so a legal no-op would arrive
/// here as a refusal — the trap `fire/attn_score.rs` documents for its own
/// `num_requests <= 0`.
fn residual_add(
    y: *mut c_void,
    x: *const c_void,
    elems: usize,
    rows: u32,
    width: u32,
    stream: StreamRef<'_>,
) -> Result<()> {
    if elems == 0 {
        return Ok(());
    }
    fire(
        "norm::residual_add_bf16",
        rect(rows, width),
        &[pm(y), p(x), ArgValue::Usize(elems)],
        stream,
    )
}

/// The encode-ABI entry: host pixels in, host bf16 embedding rows out.
///
/// The port of `encode_gemma4_vision` (`gemma4_vision.cu:241-320`). One image
/// per iteration, each with its own arena, each synchronised before the next
/// — the shape the C++ had, kept because the output rows are read back to the
/// host between images.
///
/// `pixels` is the whole image plane as BYTES and `pixel_byte_indptr` cuts it,
/// which is what the plan carries (`MediaEncodePlan::validate` proves the
/// partition and the `f32` alignment). The C++ took a `const float*` and
/// divided the offsets by four; the bytes are the same bytes and the division
/// is gone.
///
/// # Errors
///
/// A patch count that is not a whole number of pooling groups, an output
/// buffer too small for the rows this image produces, or any refused launch.
/// Each is the `throw` the C++ made at the same point, as a value.
pub fn encode(
    w: &Weights,
    pixels: &[u8],
    pixel_byte_indptr: &[u32],
    patch_positions: &[u32],
    output_rows: &mut [u8],
    output_row_indptr: &mut [u32],
    cublas: *mut c_void,
    stream: StreamRef<'_>,
) -> Result<()> {
    let num_images = output_row_indptr.len().saturating_sub(1);
    if num_images == 0 || pixel_byte_indptr.len() < num_images + 1 {
        return Err(Error::invalid(
            WHO,
            "invalid standalone encode inputs",
        ));
    }
    let pk = usize::try_from(w.pool_kernel)
        .map_err(|_| Error::invalid(WHO, "pool_kernel is negative"))?;
    let pk2 = pk
        .checked_mul(pk)
        .filter(|v| *v != 0)
        .ok_or_else(|| Error::invalid(WHO, "pool_kernel is zero"))?;
    let row_bytes = usize::try_from(w.text_hidden).unwrap_or(0) * 2;
    let mut rows_written = 0usize;
    let mut patch_off = 0usize;
    output_row_indptr[0] = 0;
    for image in 0..num_images {
        let blo = pixel_byte_indptr[image] as usize;
        let bhi = pixel_byte_indptr[image + 1] as usize;
        if bhi < blo || bhi > pixels.len() {
            return Err(Error::invalid(
                WHO,
                format!("image {image}'s pixel span [{blo}, {bhi}) leaves the payload"),
            ));
        }
        let n_floats = (bhi - blo) / 4;
        let n_patch = n_floats / PATCH_DIM;
        if n_patch == 0 || !n_patch.is_multiple_of(pk2) {
            return Err(Error::invalid(
                WHO,
                format!("invalid patch count ({n_patch} for a {pk}x{pk} pool)"),
            ));
        }
        let out_len = n_patch / pk2;
        let want = rows_written
            .checked_add(out_len)
            .and_then(|r| r.checked_mul(row_bytes))
            .ok_or_else(|| Error::invalid(WHO, "output row count overflowed"))?;
        if want > output_rows.len() {
            return Err(Error::invalid(
                WHO,
                "encode output buffer too small",
            ));
        }
        if patch_positions.len() < (patch_off + n_patch) * 2 {
            return Err(Error::invalid(
                WHO,
                "patch position table is shorter than the patches it describes",
            ));
        }
        let pos_h = &patch_positions[patch_off * 2..(patch_off + n_patch) * 2];

        let mut scratch = Scratch::new();
        let pix_f32 = scratch.upload_bytes(&pixels[blo..bhi], stream)?;
        let pix_bf = scratch.bf16(n_floats)?;
        // `:280` — `k_f32_to_bf16<<<(n_floats+255)/256, 256, 0, S>>>`. The
        // input is `F32s` and not `Buf`: this kernel's source is float
        // whatever the row's element type is.
        fire(
            "vision::k_f32_to_bf16_bf16",
            rect(
                u32::try_from(n_floats)
                    .map_err(|_| Error::invalid(WHO, "pixel count overflowed"))?,
                1,
            ),
            &[pm(pix_f32), pm(pix_bf), ArgValue::Usize(n_floats)],
            stream,
        )?;

        // `:283-296` — the host's own arithmetic: the positions widened to
        // float for the two kernels that consume them as trigonometric
        // arguments, and the pooling group index, which is a division by a
        // pooling kernel the patch grid does not carry.
        let mut posf = vec![0.0f32; n_patch * 2];
        let mut grp = vec![0i32; n_patch];
        let mut maxx = 0u32;
        for patch in 0..n_patch {
            maxx = maxx.max(pos_h[2 * patch]);
        }
        let gx = i32::try_from((maxx as usize + 1) / pk)
            .map_err(|_| Error::invalid(WHO, "pooling grid width overflowed"))?;
        for patch in 0..n_patch {
            let (px, py) = (pos_h[2 * patch], pos_h[2 * patch + 1]);
            #[allow(clippy::cast_precision_loss)]
            {
                posf[2 * patch] = px as f32;
                posf[2 * patch + 1] = py as f32;
            }
            let (cx, cy) = (
                i32::try_from(px as usize / pk).unwrap_or(0),
                i32::try_from(py as usize / pk).unwrap_or(0),
            );
            grp[patch] = cx + gx * cy;
        }
        let pos_d = scratch.upload_f32s(&posf, stream)?;
        let grp_d = scratch.upload_i32s(&grp, stream)?;
        let proj_d = scratch.bf16(out_len * usize::try_from(w.text_hidden).unwrap_or(0))?;
        run(
            w,
            pix_bf.cast_const(),
            pos_d,
            grp_d,
            i32::try_from(n_patch)
                .map_err(|_| Error::invalid(WHO, "patch count overflowed an int"))?,
            i32::try_from(out_len)
                .map_err(|_| Error::invalid(WHO, "pooled row count overflowed"))?,
            proj_d,
            cublas,
            stream,
        )?;
        // `:310` — the rows come back to the host between images, because the
        // encode ABI's output is a host buffer.
        let begin = rows_written * row_bytes;
        let end = begin + out_len * row_bytes;
        // SAFETY: `proj_d` is `out_len * text_hidden` bf16 elements of this
        // arena, live until `scratch` drops below, and `end - begin` is
        // exactly that many bytes.
        unsafe { read_raw_span(proj_d.cast_const(), &mut output_rows[begin..end], stream)? };
        stream.synchronize()?;
        drop(scratch);
        rows_written += out_len;
        output_row_indptr[image + 1] = u32::try_from(rows_written)
            .map_err(|_| Error::invalid(WHO, "encoded row count overflowed u32"))?;
        patch_off += n_patch;
    }
    Ok(())
}
