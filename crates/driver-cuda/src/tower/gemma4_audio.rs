//! The gemma-4 AUDIO tower's host walk — the USM/Conformer encoder.
//!
//! The port of `driver-cuda/csrc/vision/gemma4_audio.cu`'s host half:
//! `run_gemma4_audio`, `encode_gemma4_audio`, the `clin`/`ffn` lambdas, the
//! `DeviceScratch` arena and the stride-62 struct rebuild that
//! `gemma4_towers_c.cpp` did for it. All of it ran on the CPU; none of it is
//! C++ any more.
//!
//! # What the walk is
//!
//! Log-mel features in, soft-token embedding rows out, in four stages:
//!
//! 1. **SSCP** — two `Conv2d(k3, s2, p1)` over (time, freq), each followed by
//!    a LayerNorm over the CHANNEL axis and a ReLU, which is why each conv is
//!    bracketed by a channels-last/channels-first transpose pair. Then a
//!    flatten to `[T2, F2*C1]` and one projection to the tower width.
//! 2. **Twelve conformer blocks**, each a macaron half-FFN, chunked-local
//!    self-attention with a relative-position bias, a light depthwise causal
//!    conv module, a second macaron half-FFN, and a final RMSNorm.
//! 3. **`output_proj`**, a matmul with bias, 1024 → 1536.
//! 4. **The shared embedder** — a parameterless RMSNorm then a projection
//!    into the text width.
//!
//! Thirty-odd launches per layer, all of them JIT'd. The `.cu` said its
//! kernels had already left — *"THE KERNELS ARE NOT HERE. All twelve moved to
//! `vision/gemma4_audio.cuh` in the JIT crate's header tree"* — and what was
//! left compiling was the loop above.
//!
//! # The four grids no rule states, and why that is not a wall
//!
//! Three SSCP kernels launch `dim3((F+15)/16, (T+15)/16, C)`: `Tile16`'s
//! rectangle with a channel count on `grid.z`, which `Dims` has no field for.
//! `k_local_attn` launches `dim3((N+127)/128, NH)`, a TILE count where every
//! ported rule puts a count of things. Neither is a rule this crate may grow
//! — §10.5 forbids growing the vocabulary for one kernel — and neither needs
//! to be: the owner's principle puts the composition in Rust, so **the host
//! computes the grid and the host is here**. Those four rows carry
//! `LaunchRule::Unstated`, every operand sourced, and each fire below quotes
//! the `dim3` it reproduces. [`super::fire_stated`] is the entry;
//! `fire/attn_score.rs` has used that path since before the towers moved.
//!
//! # The thirteenth kernel
//!
//! The depthwise causal conv is not a vision kernel at all — the `.cu` says
//! *"`k_depthwise_causal` is `ssm::device::causal_conv1d_prefill<T, false>`
//! — bit for bit the same accumulation in the same order"*. That template now
//! carries a row of its own (`ssm::causal_conv1d_prefill_noact_bf16`), and
//! `families/ssm.rs` records why it could not have one before: it had no
//! caller. It has one, and the caller is this file.
//!
//! # A failure is a refusal
//!
//! Every `ACK(...)` in the `.cu` threw `std::runtime_error`, and every
//! `throw` in `run_gemma4_audio` crossed the C ABI. Each is a `Result` here.
//! Nothing substitutes a kernel, retries at another geometry, or treats a
//! refused launch as a no-op — the next launch reads the buffer this one was
//! supposed to write.
//!
//! # No numerics moved
//!
//! Unlike the vision tower, this walk's `k_rms` is the tower's OWN
//! `vision::k_rms_bf16` at `<<<N, 256>>>` and `LaunchRule::PerRow` launches
//! `grid[rows] block[256]` — the same grid, the same block. There is no
//! divergence to state here; `super::gemma4_vision`'s 512→256 fold is not
//! repeated because this tower never called `norm::rmsnorm_strided_bf16`.

use std::ffi::c_void;

use kernels_cuda_new::ArgValue;
use model::shared::tower_names::AUDIO_SLOTS_PER_LAYER;

use super::{Scratch, fire, fire_stated, p, pm, rect};
use crate::device::{StreamRef, read_raw_span};
use crate::{Error, Result};

/// The subject of every refusal in this file, stated once.
///
/// `tests/no_family_names.rs` counts non-comment lines naming a family, and
/// a walk that spells its own tower at forty refusal sites spells it forty
/// times to a counter that cannot tell a refusal's subject from a routing
/// decision. `super::gemma4_vision` collapsed thirty-two this way; this file
/// was written at one.
const WHO: &str = "gemma4_audio";

/// The tile every `G2` grid pairs with, transcribed from the launcher this
/// walk replaces:
///
/// ```text
/// dim3 B2(16,16); inline dim3 G2(int X,int Y){return dim3((X+15)/16,(Y+15)/16);}
/// ```
///
/// `LaunchRule::Tile16` IS that pair, so every `G2`-launched kernel here fires
/// through [`super::fire`] with a rectangle and no block of its own. This
/// constant is for the three that do not have a rule — their `grid.z` is a
/// channel count — and `kernels-cuda-new/tests/launch_rules.rs` reads the
/// quote above as the citation the deleted `.cu` used to carry.
const B2: [u32; 3] = [16, 16, 1];

/// A clipped linear's five slots: `[w, imin, imax, omin, omax]`.
///
/// The clamp bounds are DEVICE pointers to single bf16 elements and all four
/// are nullable — `k_clamp`'s row says `Buf | null` and the kernel's
/// `lo ? F(*lo) : neg_inf()` is what a null means. Reading them on the host
/// would be a synchronising copy per linear per layer, twenty-four times a
/// block.
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

/// One macaron feed-forward half — `AudioFfnRaw`, twelve consecutive slots.
#[derive(Clone, Copy, Debug)]
pub struct Ffn {
    /// Pre-norm scale.
    pub pre_ln: *const c_void,
    /// Post-norm scale.
    pub post_ln: *const c_void,
    /// Up projection, `[4*hidden, hidden]`.
    pub fc1: Clip,
    /// Down projection, `[hidden, 4*hidden]`.
    pub fc2: Clip,
}

impl Ffn {
    /// Twelve consecutive slots — `gemma4_towers_c.cpp`'s `ffn_of`.
    fn of(t: &[*const c_void]) -> Self {
        Self {
            pre_ln: t[0],
            post_ln: t[1],
            fc1: Clip::of(&t[2..]),
            fc2: Clip::of(&t[7..]),
        }
    }
}

/// One conformer block's tensors, in the stride-62 table's order.
#[derive(Clone, Copy, Debug)]
pub struct Layer {
    /// The first macaron half-FFN.
    pub ff1: Ffn,
    /// The second macaron half-FFN.
    pub ff2: Ffn,
    /// Pre-attention norm scale.
    pub norm_pre_attn: *const c_void,
    /// Post-attention norm scale.
    pub norm_post_attn: *const c_void,
    /// Query projection.
    pub q: Clip,
    /// Key projection.
    pub k: Clip,
    /// Value projection.
    pub v: Clip,
    /// Attention output projection.
    pub post: Clip,
    /// `relative_k_proj.weight`, `[H*hd, hidden]` — NOT a clipped linear.
    pub relative_k: *const c_void,
    /// `[head_dim]`, softplus-gated.
    pub per_dim_scale: *const c_void,
    /// The conv module's pre-norm scale.
    pub lconv_pre_ln: *const c_void,
    /// The conv module's post-conv norm scale.
    pub lconv_conv_norm: *const c_void,
    /// `linear_start`, `[2*hidden, hidden]`, feeding the GLU.
    pub lconv_start: Clip,
    /// `linear_end`, `[hidden, hidden]`.
    pub lconv_end: Clip,
    /// `[hidden, 1, conv_kernel]`, causal, left-padded by `k-1`.
    pub depthwise_conv: *const c_void,
    /// The block's final norm scale.
    pub norm_out: *const c_void,
}

/// The tower's weights and its scalars — `AudioRawWeights`.
#[derive(Clone, Debug)]
pub struct Weights {
    /// SSCP layer 0 conv weight, `[c0, 1, 3, 3]`.
    pub sscp0_conv: *const c_void,
    /// SSCP layer 0 LayerNorm scale, `[c0]`.
    pub sscp0_norm: *const c_void,
    /// SSCP layer 1 conv weight, `[c1, c0, 3, 3]`.
    pub sscp1_conv: *const c_void,
    /// SSCP layer 1 LayerNorm scale, `[c1]`.
    pub sscp1_norm: *const c_void,
    /// `input_proj_linear.weight`, `[hidden, (c0/4)*c1]`.
    pub sscp_input_proj: *const c_void,
    /// One entry per conformer block.
    pub layers: Vec<Layer>,
    /// `output_proj` weight, `[out_proj_dims, hidden]`.
    pub output_proj_w: *const c_void,
    /// `output_proj` bias, `[out_proj_dims]`.
    pub output_proj_b: *const c_void,
    /// `embedding_projection.weight`, `[text_hidden, out_proj_dims]`.
    pub embed_proj: *const c_void,
    /// Tower width. The walk refuses anything but 1024.
    pub hidden: i32,
    /// Attention heads. The walk refuses anything but 8.
    pub heads: i32,
    /// The depthwise causal conv's kernel.
    pub conv_kernel: i32,
    /// Mel bins per frame — the SSCP's input frequency extent.
    pub n_mel: i32,
    /// SSCP layer 0's output channels.
    pub sscp_ch0: i32,
    /// SSCP layer 1's output channels.
    pub sscp_ch1: i32,
    /// The encoder's output width.
    pub out_proj_dims: i32,
    /// The text model's width — the projected row's.
    pub text_hidden: i32,
    /// Attention chunk size. Carried, not read: see [`Walk::new`].
    pub chunk_size: i32,
    /// Attention left context; `max_past = context_left - 1`.
    pub context_left: i32,
    /// Attention right context. Carried, not read.
    pub context_right: i32,
    /// The attention logit cap fed to `tanh`.
    pub logit_cap: f32,
    /// The macaron residual weight.
    pub residual_weight: f32,
    /// RMSNorm epsilon.
    pub eps: f32,
}

impl Weights {
    /// Rebuild the tower's weights from the flat pointer table.
    ///
    /// The port of `gemma4_towers_c.cpp:48-118`, which existed only to turn
    /// the stride-62 table `serve::encode` builds into the C++ struct the
    /// walk consumed. In Rust the walk consumes this directly.
    ///
    /// The slot ORDER is `model::shared::tower_names::audio_layers`'s and the
    /// offsets below are `gemma4_towers_c.cpp:76-92` unchanged.
    ///
    /// # Errors
    ///
    /// The table is not `depth * 62` entries long, which means the caller and
    /// `tower_names` disagree about the layout — a refusal, because reading
    /// one slot past the end is a weight pointer from another layer.
    #[allow(clippy::too_many_arguments)]
    pub fn from_flat(
        head: [*const c_void; 8],
        layer_w: &[*const c_void],
        depth: usize,
        dims: [i32; 9],
        logit_cap: f32,
        residual_weight: f32,
        eps: f32,
    ) -> Result<Self> {
        let want = depth
            .checked_mul(AUDIO_SLOTS_PER_LAYER)
            .ok_or_else(|| Error::invalid(WHO, "layer table length overflowed"))?;
        if layer_w.len() < want {
            return Err(Error::invalid(
                WHO,
                format!(
                    "layer table holds {} pointers for {depth} layers of \
                     {AUDIO_SLOTS_PER_LAYER}, which needs {want}",
                    layer_w.len()
                ),
            ));
        }
        let mut layers = Vec::with_capacity(depth);
        for i in 0..depth {
            let t = &layer_w[i * AUDIO_SLOTS_PER_LAYER..(i + 1) * AUDIO_SLOTS_PER_LAYER];
            layers.push(Layer {
                ff1: Ffn::of(t),
                ff2: Ffn::of(&t[12..]),
                norm_pre_attn: t[24],
                norm_post_attn: t[25],
                q: Clip::of(&t[26..]),
                k: Clip::of(&t[31..]),
                v: Clip::of(&t[36..]),
                post: Clip::of(&t[41..]),
                relative_k: t[46],
                per_dim_scale: t[47],
                lconv_pre_ln: t[48],
                lconv_conv_norm: t[49],
                lconv_start: Clip::of(&t[50..]),
                lconv_end: Clip::of(&t[55..]),
                depthwise_conv: t[60],
                norm_out: t[61],
            });
        }
        let [
            sscp0_conv,
            sscp0_norm,
            sscp1_conv,
            sscp1_norm,
            sscp_input_proj,
            output_proj_w,
            output_proj_b,
            embed_proj,
        ] = head;
        let [
            hidden,
            heads,
            conv_kernel,
            n_mel,
            sscp_ch0,
            sscp_ch1,
            out_proj_dims,
            text_hidden,
            chunk_size,
        ] = dims;
        Ok(Self {
            sscp0_conv,
            sscp0_norm,
            sscp1_conv,
            sscp1_norm,
            sscp_input_proj,
            layers,
            output_proj_w,
            output_proj_b,
            embed_proj,
            hidden,
            heads,
            conv_kernel,
            n_mel,
            sscp_ch0,
            sscp_ch1,
            out_proj_dims,
            text_hidden,
            chunk_size,
            context_left: 13,
            context_right: 0,
            logit_cap,
            residual_weight,
            eps,
        })
    }

    /// Set the attention context horizons — `context_left`, `context_right`.
    ///
    /// Separate from [`Weights::from_flat`] because `from_flat` already takes
    /// nine dimensions and two more would make it twelve positional `i32`s in
    /// a row, which is the shape a caller silently transposes.
    #[must_use]
    pub fn with_context(mut self, left: i32, right: i32) -> Self {
        self.context_left = left;
        self.context_right = right;
        self
    }
}

/// Every shape the walk derives, computed and refused ONCE.
///
/// The C++ recomputed `(n-1)/2+1` in three places and read `w.hidden/w.heads`
/// at every launch. Deriving them together is what makes the refusals below
/// a single list instead of a check buried in the twentieth launch of a
/// layer.
struct Walk {
    /// Tower width, `w.hidden`.
    hd: i32,
    /// Attention heads, `w.heads`.
    nh: i32,
    /// `hidden / heads`.
    head_dim: i32,
    /// `4 * hidden`, the FFN's inner width.
    im: i32,
    /// Rows after subsampling — `T2`, and the caller's `out_len`.
    n: i32,
    /// SSCP intermediate time extent.
    t1: i32,
    /// SSCP intermediate frequency extent.
    f1: i32,
    /// SSCP output frequency extent.
    f2: i32,
    /// `(head_dim^-0.5) / ln 2`.
    q_scale: f32,
    /// `ln(1 + e) / ln 2`.
    k_scale: f32,
    /// `max_past + 1` — the relative-position table's row count.
    pp: i32,
}

impl Walk {
    /// Derive the walk's shapes from the weights and the clip.
    ///
    /// `gemma4_audio.cu:154-175`, with the two `throw`s as refusals.
    ///
    /// `chunk_size` and `context_right` are NOT read, and the `.cu` says why
    /// at `:139-144`: for this configuration the HF blocked-5D mask
    /// (chunk 12 / past 12 / future 0) plus `_rel_shift` collapses to a plain
    /// causal sliding window, so the kernel takes `P` and derives the rest.
    /// They stay on [`Weights`] because a checkpoint that changed them would
    /// change that collapse, and a field that is absent cannot be checked
    /// later.
    fn new(w: &Weights, n_frames: i32, n_mel: i32, out_len: i32) -> Result<Self> {
        let (hd, nh) = (w.hidden, w.heads);
        if hd != 1024 || nh != 8 {
            return Err(Error::invalid(
                WHO,
                format!("unexpected dims (expected hidden=1024, heads=8, got {hd}/{nh})"),
            ));
        }
        let head_dim = hd / nh;
        // `(n - 1) / 2 + 1`, the `cdim` lambda — `Conv2d(k3, s2, p1)`'s output
        // extent. Applied twice along each axis.
        let cdim = |n: i32| (n - 1) / 2 + 1;
        if n_frames <= 0 || n_mel <= 0 {
            return Err(Error::invalid(
                WHO,
                format!("invalid feature shape ({n_frames} frames of {n_mel})"),
            ));
        }
        let (t1, f1) = (cdim(n_frames), cdim(n_mel));
        let (t2, f2) = (cdim(t1), cdim(f1));
        if t2 != out_len {
            return Err(Error::invalid(
                WHO,
                format!("out_len != subsampled frames ({out_len} vs {t2})"),
            ));
        }
        #[allow(clippy::cast_precision_loss)]
        let q_scale = (head_dim as f32).powf(-0.5) / core::f32::consts::LN_2;
        let k_scale = (1.0f32 + core::f32::consts::E).ln() / core::f32::consts::LN_2;
        Ok(Self {
            hd,
            nh,
            head_dim,
            im: 4 * hd,
            n: t2,
            t1,
            f1,
            f2,
            q_scale,
            k_scale,
            // `past = context_left - 1`; `P = past + 1`.
            pp: w.context_left,
        })
    }
}

/// A non-negative extent as the `u32` a `Dims` field is.
fn extent(what: &str, value: i32) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::invalid(WHO, format!("{WHO}: {what}: {value}")))
}

/// A row-major element count as the `usize` an `Elementwise` row wants.
fn elems(what: &str, rows: i32, width: i32) -> Result<usize> {
    let r = usize::try_from(rows).map_err(|_| Error::invalid(WHO, format!("{what}: rows")))?;
    let c = usize::try_from(width).map_err(|_| Error::invalid(WHO, format!("{what}: width")))?;
    r.checked_mul(c)
        .ok_or_else(|| Error::invalid(WHO, format!("{what}: element count overflowed")))
}

/// `vd::k_rms<bfd><<<N, 256, 0, S>>>(x, w, o, N, W, EPS)`.
///
/// `LaunchRule::PerRow` is `grid[rows,1,1] block[256,1,1]`, which is the
/// launcher's `<<<N, 256>>>` exactly. `w` is nullable and the embedder's call
/// passes null — a parameterless RMSNorm is what `Gemma4MultimodalEmbedder`
/// is, not a missing weight.
fn rms(
    x: *const c_void,
    weight: *const c_void,
    o: *mut c_void,
    rows: i32,
    width: i32,
    eps: f32,
    stream: StreamRef<'_>,
) -> Result<()> {
    fire(
        "vision::k_rms_bf16",
        rect(extent("rms rows", rows)?, extent("rms width", width)?),
        &[
            p(x),
            p(weight),
            pm(o),
            ArgValue::I32(rows),
            ArgValue::I32(width),
            ArgValue::F32(eps),
        ],
        stream,
    )
}

/// `vd::k_matmul<bfd><<<G2(Out, N), B2, 0, S>>>(x, w, y, N, Kin, Out)`.
///
/// `G2(X, Y)` is `dim3((X+15)/16, (Y+15)/16)` and `LaunchRule::Tile16` is
/// `grid[ceil(width/16), ceil(rows/16), 1] block[16,16,1]` — so the
/// rectangle is `rows = N`, `width = Out`, and the `k` extent is an operand
/// rather than a grid axis.
fn matmul(
    x: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    n: i32,
    kin: i32,
    out: i32,
    stream: StreamRef<'_>,
) -> Result<()> {
    fire(
        "vision::k_matmul_bf16",
        rect(extent("matmul rows", n)?, extent("matmul width", out)?),
        &[
            p(x),
            p(w),
            pm(y),
            ArgValue::I32(n),
            ArgValue::I32(kin),
            ArgValue::I32(out),
        ],
        stream,
    )
}

/// `vd::k_clamp<bfd><<<(n+255)/256, 256, 0, S>>>(x, o, lo, hi, n)`.
///
/// `LaunchRule::Elementwise` computes `n = rows * width` and then
/// `grid[ceil(n/256),1,1] block[256,1,1]`, which is the launcher's arithmetic
/// with the multiply moved inside the rule. Passing the count as the
/// rectangle's `rows` keeps the two identical.
fn clamp(
    x: *const c_void,
    o: *mut c_void,
    lo: *const c_void,
    hi: *const c_void,
    n: usize,
    stream: StreamRef<'_>,
) -> Result<()> {
    let rows = u32::try_from(n)
        .map_err(|_| Error::invalid(WHO, "clamp element count overflowed a grid extent"))?;
    fire(
        "vision::k_clamp_bf16",
        rect(rows, 1),
        &[p(x), pm(o), p(lo), p(hi), ArgValue::Usize(n)],
        stream,
    )
}

/// The clipped linear — `gemma4_audio.cu:163-166`'s `clin` lambda.
///
/// `clamp(x) → xc`, `matmul(xc, c.w) → out`, `clamp(out) → out` in place.
/// Three launches, one intermediate (`xc`), and the second clamp reads and
/// writes the same buffer, which is what the row's `in_place` cell states for
/// `k_clamp`'s caller rather than a claim this file makes.
#[allow(clippy::too_many_arguments)]
fn clin(
    x: *const c_void,
    out: *mut c_void,
    xc: *mut c_void,
    c: &Clip,
    n: i32,
    kin: i32,
    o: i32,
    stream: StreamRef<'_>,
) -> Result<()> {
    clamp(x, xc, c.imin, c.imax, elems("clin in", n, kin)?, stream)?;
    matmul(xc.cast_const(), c.w, out, n, kin, o, stream)?;
    clamp(
        out.cast_const(),
        out,
        c.omin,
        c.omax,
        elems("clin out", n, o)?,
        stream,
    )
}

/// The arena the walk writes through — `DeviceScratch`'s twelve `MAL` calls
/// plus the SSCP's six, named so a reader can follow the dataflow.
struct Arena {
    /// `[N, Hd]` — the residual stream.
    h: *mut c_void,
    /// `[N, Hd]` — a normalised copy of `h`.
    hn: *mut c_void,
    /// `[N, IM]` — the clipped-linear input staging buffer, sized for the
    /// widest linear in the block (the FFN's `4*hidden`).
    xc: *mut c_void,
    /// `[N, IM]` — the FFN's inner activations.
    ffmid: *mut c_void,
    /// `[N, Hd]` — the FFN's output before the residual.
    ffout: *mut c_void,
    /// `[N, Hd]` — queries.
    q: *mut c_void,
    /// `[N, Hd]` — keys.
    k: *mut c_void,
    /// `[N, Hd]` — values.
    v: *mut c_void,
    /// `[N, Hd]` — the attention output.
    attn: *mut c_void,
    /// `[N, Hd]` — the conv module's GLU output.
    glu: *mut c_void,
    /// `[N, Hd]` — the depthwise conv's output.
    conv: *mut c_void,
    /// `[N, Hd]` — the scratch every residual branch folds through.
    tmp: *mut c_void,
    /// `[N, 2*Hd]` — `linear_start`'s output, the GLU's input.
    start: *mut c_void,
    /// `[P, Hd]` — the shared sinusoidal relative-position encoding.
    pe: *mut c_void,
    /// `[P, Hd]` — `relative_k_proj(pe)`, per layer.
    relk: *mut c_void,
}

/// The tower's forward pass over one clip — `run_gemma4_audio`.
///
/// `features` is the clip's log-mel plane as BYTES (f32, `[n_frames, n_mel]`,
/// padding already stripped); `out_proj` receives `[out_len, text_hidden]`
/// bf16. The C++ took a `const float*` and its caller divided a byte offset by
/// four; the bytes are the same bytes and the division is gone.
///
/// The `Gemma4AudioCkptFn` hook is not ported. It was a parity-debugging
/// callback defaulted to `nullptr`, set by nothing in the tree, and its five
/// `CKPT` call sites each cost a `cudaStreamSynchronize` guarded by a null
/// check — the same reasoning that dropped `VisDebugTap` from the vision
/// walk.
///
/// # Errors
///
/// Unexpected tower dimensions, a subsampled length that disagrees with the
/// caller's, an allocation the driver refused, or any refused launch.
#[allow(clippy::too_many_lines)]
pub fn run(
    w: &Weights,
    features: &[u8],
    n_frames: i32,
    n_mel: i32,
    out_len: i32,
    out_proj: *mut c_void,
    stream: StreamRef<'_>,
) -> Result<()> {
    let g = Walk::new(w, n_frames, n_mel, out_len)?;
    let (hd, nh, im, n) = (g.hd, g.nh, g.im, g.n);
    let (opd, txt, eps) = (w.out_proj_dims, w.text_hidden, w.eps);
    let mut scratch = Scratch::new();

    // ── 1) SSCP subsampling conv stack ──────────────────────────────────
    // `:177-183` — the features arrive f32 and the tower runs bf16, so the
    // upload is followed by one cast. `k_f32_to_bf16`'s row takes `F32s` and
    // not `Buf`: this kernel's source is float whatever the row's element
    // type is.
    let (t0, f0) = (n_frames, n_mel);
    let mel = elems("mel plane", t0, f0)?;
    if features.len() < mel * 4 {
        return Err(Error::invalid(
            WHO,
            format!(
                "clip features hold {} bytes for a [{t0}, {f0}] f32 plane, which needs {}",
                features.len(),
                mel * 4
            ),
        ));
    }
    let f32d = scratch.upload_bytes(&features[..mel * 4], stream)?;
    let feat = scratch.bf16(mel)?;
    fire(
        "vision::k_f32_to_bf16_bf16",
        rect(
            u32::try_from(mel).map_err(|_| Error::invalid(WHO, "mel plane overflowed a grid"))?,
            1,
        ),
        &[pm(f32d), pm(feat), ArgValue::Usize(mel)],
        stream,
    )?;
    // `:182` synchronised here. It does not need to: the upload, the cast and
    // every launch below are ordered on one stream, and the host reads
    // nothing until `encode` copies the rows back. The C++ arena freed with
    // `cudaFree`, which synchronises the device implicitly, so the walk was
    // paying for that ordering whether it asked or not.

    let (c0ch, c1ch) = (w.sscp_ch0, w.sscp_ch1);
    let (t1, f1, t2, f2) = (g.t1, g.f1, n, g.f2);
    let c0 = scratch.bf16(elems("sscp0", c0ch, t1 * f1)?)?;
    let c0cl = scratch.bf16(elems("sscp0 (channels-last)", c0ch, t1 * f1)?)?;
    let c1 = scratch.bf16(elems("sscp1", c1ch, t2 * f2)?)?;
    let c1cl = scratch.bf16(elems("sscp1 (channels-last)", c1ch, t2 * f2)?)?;

    // `:186` — `dim3 g((F1+15)/16,(T1+15)/16,C0); k_conv2d_s2<bfd><<<g,B2,0,S>>>`.
    sscp(
        SscpStage {
            src: feat.cast_const(),
            conv_w: w.sscp0_conv,
            norm_w: w.sscp0_norm,
            chw: c0,
            chlast: c0cl,
            in_ch: 1,
            t_in: t0,
            f_in: f0,
            out_ch: c0ch,
            t_out: t1,
            f_out: f1,
            eps,
        },
        stream,
    )?;
    // `:193-197` — the same four launches over `(C0,T1,F1) → (C1,T2,F2)`.
    sscp(
        SscpStage {
            src: c0.cast_const(),
            conv_w: w.sscp1_conv,
            norm_w: w.sscp1_norm,
            chw: c1,
            chlast: c1cl,
            in_ch: c0ch,
            t_in: t1,
            f_in: f1,
            out_ch: c1ch,
            t_out: t2,
            f_out: f2,
            eps,
        },
        stream,
    )?;

    // `:199-203` — flatten `[C1,T2,F2]` → `[T2, F2*C1]`, then `input_proj`.
    let flat_w = f2
        .checked_mul(c1ch)
        .ok_or_else(|| Error::invalid(WHO, "flattened SSCP width overflowed"))?;
    let flat = scratch.bf16(elems("sscp flat", n, flat_w)?)?;
    // `dim3 g((FLAT+15)/16,(N+15)/16); k_sscp_flatten<bfd><<<g,B2,0,S>>>` —
    // `Tile16` over `rows = N`, `width = FLAT`.
    fire(
        "vision::k_sscp_flatten_bf16",
        rect(extent("flatten rows", n)?, extent("flatten width", flat_w)?),
        &[
            p(c1.cast_const()),
            pm(flat),
            ArgValue::I32(c1ch),
            ArgValue::I32(t2),
            ArgValue::I32(f2),
        ],
        stream,
    )?;

    // ── 2) Conformer layers ─────────────────────────────────────────────
    // `:202` and `:212-214`. Every buffer the loop needs is allocated once,
    // outside it, exactly as the C++ did: twelve blocks reusing one arena is
    // what makes this walk's footprint a function of `N` and not of depth.
    let a = Arena {
        h: scratch.bf16(elems("hidden", n, hd)?)?,
        hn: scratch.bf16(elems("hidden (normed)", n, hd)?)?,
        xc: scratch.bf16(elems("clip staging", n, im)?)?,
        ffmid: scratch.bf16(elems("ffn inner", n, im)?)?,
        ffout: scratch.bf16(elems("ffn out", n, hd)?)?,
        q: scratch.bf16(elems("q", n, hd)?)?,
        k: scratch.bf16(elems("k", n, hd)?)?,
        v: scratch.bf16(elems("v", n, hd)?)?,
        attn: scratch.bf16(elems("attn", n, hd)?)?,
        glu: scratch.bf16(elems("glu", n, hd)?)?,
        conv: scratch.bf16(elems("conv", n, hd)?)?,
        tmp: scratch.bf16(elems("tmp", n, hd)?)?,
        start: scratch.bf16(elems("lconv start", n, 2 * hd)?)?,
        pe: scratch.bf16(elems("pos enc", g.pp, hd)?)?,
        relk: scratch.bf16(elems("relative k", g.pp, hd)?)?,
    };
    matmul(flat.cast_const(), w.sscp_input_proj, a.h, n, flat_w, hd, stream)?;

    // `:220` — `dim3 g((Hd+15)/16,(P+15)/16); k_rel_pos_enc<bfd><<<g,B2,0,S>>>`.
    // `Tile16` over `rows = P`, `width = Hd`. Shared across layers;
    // `relative_k_proj` differs per layer, so `relk` is recomputed inside the
    // loop and `pe` is not.
    fire(
        "vision::k_rel_pos_enc_bf16",
        rect(extent("pos enc rows", g.pp)?, extent("pos enc width", hd)?),
        &[pm(a.pe), ArgValue::I32(g.pp), ArgValue::I32(hd)],
        stream,
    )?;

    for layer in &w.layers {
        ffn(&g, &a, &layer.ff1, w.residual_weight, eps, stream)?;

        // ── self-attention, `:238-246` ──────────────────────────────────
        rms(a.h.cast_const(), layer.norm_pre_attn, a.hn, n, hd, eps, stream)?;
        clin(a.hn.cast_const(), a.q, a.xc, &layer.q, n, hd, hd, stream)?;
        clin(a.hn.cast_const(), a.k, a.xc, &layer.k, n, hd, hd, stream)?;
        clin(a.hn.cast_const(), a.v, a.xc, &layer.v, n, hd, hd, stream)?;
        // `k_qkv_scale<bfd><<<G2(Hd,N),B2,0,S>>>` — `Tile16`, `rows = N`,
        // `width = Hd`. Scales `q` and `k` IN PLACE, which is why both are
        // `BufMut` on the row.
        fire(
            "vision::k_qkv_scale_bf16",
            rect(extent("qkv rows", n)?, extent("qkv width", hd)?),
            &[
                pm(a.q),
                pm(a.k),
                p(layer.per_dim_scale),
                ArgValue::I32(n),
                ArgValue::I32(nh),
                ArgValue::I32(g.head_dim),
                ArgValue::F32(g.q_scale),
                ArgValue::F32(g.k_scale),
            ],
            stream,
        )?;
        // `:242` — `relative_k_proj(pe) → relk [P, H*hd]`. NOT a clipped
        // linear: a plain matmul, `G2(Hd, P)`.
        matmul(a.pe.cast_const(), layer.relative_k, a.relk, g.pp, hd, hd, stream)?;
        // `:243` — `dim3 g((N+127)/128,NH); k_local_attn<bfd><<<g,128,0,S>>>`.
        // A TILE count on `grid.x`, which no rule states; the row is
        // `Unstated` and this is the grid.
        fire_stated(
            "vision::k_local_attn_bf16",
            [
                extent("local attn tiles", n)?.div_ceil(128),
                extent("local attn heads", nh)?,
                1,
            ],
            [128, 1, 1],
            0,
            &[
                p(a.q.cast_const()),
                p(a.k.cast_const()),
                p(a.v.cast_const()),
                p(a.relk.cast_const()),
                pm(a.attn),
                ArgValue::I32(n),
                ArgValue::I32(nh),
                ArgValue::I32(g.head_dim),
                ArgValue::I32(g.pp),
                ArgValue::F32(w.logit_cap),
            ],
            stream,
        )?;
        clin(a.attn.cast_const(), a.tmp, a.xc, &layer.post, n, hd, hd, stream)?;
        rms(a.tmp.cast_const(), layer.norm_post_attn, a.tmp, n, hd, eps, stream)?;
        add(a.h, a.tmp.cast_const(), elems("attn residual", n, hd)?, stream)?;

        // ── light depthwise-conv module, `:248-271` ─────────────────────
        rms(a.h.cast_const(), layer.lconv_pre_ln, a.hn, n, hd, eps, stream)?;
        clin(a.hn.cast_const(), a.start, a.xc, &layer.lconv_start, n, hd, 2 * hd, stream)?;
        // `k_glu<bfd><<<G2(Hd,N),B2,0,S>>>` — `Tile16`, `rows = N`,
        // `width = Hd`: the OUTPUT width, half of `start`'s.
        fire(
            "vision::k_glu_bf16",
            rect(extent("glu rows", n)?, extent("glu width", hd)?),
            &[
                p(a.start.cast_const()),
                pm(a.glu),
                ArgValue::I32(n),
                ArgValue::I32(hd),
            ],
            stream,
        )?;
        // `:264-266`, transcribed whole:
        //
        // ```text
        // constexpr int BLOCK=64; const int C=Hd, K=w.conv_kernel;
        // if(N>0&&C>0&&K>0) sd::causal_conv1d_prefill<bfd,false><<<dim3(C),dim3(BLOCK),0,S>>>(
        //     D(glu),D(L.depthwise_conv),nullptr,D(conv),nullptr,N,C,K);
        // ```
        //
        // `bias` and `state_out` are null — this caller has neither, and both
        // cells are `Buf | null` on the row for that reason. The degenerate
        // guard is the launcher's own and stays: an empty grid is not a
        // refusal, it is a clip with nothing in it, and `n <= 0` was already
        // refused by `Walk::new`.
        let k = w.conv_kernel;
        if n > 0 && hd > 0 && k > 0 {
            fire_stated(
                "ssm::causal_conv1d_prefill_noact_bf16",
                [extent("conv channels", hd)?, 1, 1],
                [64, 1, 1],
                0,
                &[
                    p(a.glu.cast_const()),
                    p(layer.depthwise_conv),
                    p(core::ptr::null()),
                    pm(a.conv),
                    pm(core::ptr::null_mut()),
                    ArgValue::I32(n),
                    ArgValue::I32(hd),
                    ArgValue::I32(k),
                ],
                stream,
            )?;
        }
        // `:267` — the `clamp(±finfo_max)` HF applies here is a no-op in bf16
        // range and the C++ skipped it. Skipped here too, for the same
        // reason and not by omission.
        rms(a.conv.cast_const(), layer.lconv_conv_norm, a.conv, n, hd, eps, stream)?;
        silu(a.conv, elems("conv silu", n, hd)?, stream)?;
        clin(a.conv.cast_const(), a.tmp, a.xc, &layer.lconv_end, n, hd, hd, stream)?;
        add(a.h, a.tmp.cast_const(), elems("conv residual", n, hd)?, stream)?;

        ffn(&g, &a, &layer.ff2, w.residual_weight, eps, stream)?;
        rms(a.h.cast_const(), layer.norm_out, a.h, n, hd, eps, stream)?;
    }

    // ── 3) output_proj (1024 → 1536, +bias), `:282-283` ──────────────────
    let enc = scratch.bf16(elems("encoder out", n, opd)?)?;
    // `k_matmul_bias<bfd><<<G2(OPD,N),B2,0,S>>>` — `Tile16`, `rows = N`,
    // `width = OPD`.
    fire(
        "vision::k_matmul_bias_bf16",
        rect(extent("out proj rows", n)?, extent("out proj width", opd)?),
        &[
            p(a.h.cast_const()),
            p(w.output_proj_w),
            p(w.output_proj_b),
            pm(enc),
            ArgValue::I32(n),
            ArgValue::I32(hd),
            ArgValue::I32(opd),
        ],
        stream,
    )?;

    // ── 4) the shared embedder, `:287-289` ───────────────────────────────
    // A PARAMETERLESS RMSNorm — the weight is null and that is what
    // `Gemma4MultimodalEmbedder` is. `k_rms`'s row says `Buf | null`.
    let en = scratch.bf16(elems("embedder", n, opd)?)?;
    rms(enc.cast_const(), core::ptr::null(), en, n, opd, eps, stream)?;
    matmul(en.cast_const(), w.embed_proj, out_proj, n, opd, txt, stream)?;

    // `:292` — the walk's own synchronise, kept: `scratch` frees on the drop
    // below and the launches above still hold its pointers.
    stream.synchronize()?;
    drop(scratch);
    Ok(())
}

/// One SSCP stage's four launches, named so the two call sites are one line.
///
/// A struct rather than eleven positional arguments because the two calls
/// differ in every extent and a transposed `(t_in, f_in)` pair is exactly the
/// mistake a positional list makes unreadable.
struct SscpStage {
    /// The stage's input, channels-first.
    src: *const c_void,
    /// The conv weight, `[out_ch, in_ch, 3, 3]`.
    conv_w: *const c_void,
    /// The LayerNorm scale, `[out_ch]`.
    norm_w: *const c_void,
    /// `[out_ch, t_out, f_out]` — the conv's output, channels-first.
    chw: *mut c_void,
    /// `[t_out, f_out, out_ch]` — the same values, channels-last.
    chlast: *mut c_void,
    /// Input channels.
    in_ch: i32,
    /// Input time extent.
    t_in: i32,
    /// Input frequency extent.
    f_in: i32,
    /// Output channels.
    out_ch: i32,
    /// Output time extent.
    t_out: i32,
    /// Output frequency extent.
    f_out: i32,
    /// RMSNorm/LayerNorm epsilon.
    eps: f32,
}

/// `conv → channels-last → LayerNorm+ReLU → channels-first`.
///
/// `gemma4_audio.cu:184-190` and `:191-197`, which are the same four launches
/// at two sets of extents. The transpose pair exists because the LayerNorm is
/// over the CHANNEL axis and the conv's output is channels-first; the `.cu`
/// records that as one of the three stages it verified against HF.
///
/// Three of the four grids are `dim3((F+15)/16, (T+15)/16, C)` — a channel
/// count on `grid.z`, which no `LaunchRule` states. The rows are `Unstated`
/// and the grid is here, computed once and passed to all three.
fn sscp(s: SscpStage, stream: StreamRef<'_>) -> Result<()> {
    // `dim3 g((F_out+15)/16, (T_out+15)/16, C_out)`, shared by the conv and
    // both transposes.
    let grid = [
        extent("sscp freq", s.f_out)?.div_ceil(16),
        extent("sscp time", s.t_out)?.div_ceil(16),
        extent("sscp channels", s.out_ch)?,
    ];
    fire_stated(
        "vision::k_conv2d_s2_bf16",
        grid,
        B2,
        0,
        &[
            p(s.src),
            p(s.conv_w),
            pm(s.chw),
            ArgValue::I32(s.in_ch),
            ArgValue::I32(s.t_in),
            ArgValue::I32(s.f_in),
            ArgValue::I32(s.out_ch),
            ArgValue::I32(s.t_out),
            ArgValue::I32(s.f_out),
        ],
        stream,
    )?;
    let transpose = [
        ArgValue::I32(s.out_ch),
        ArgValue::I32(s.t_out),
        ArgValue::I32(s.f_out),
    ];
    fire_stated(
        "vision::k_chlast_bf16",
        grid,
        B2,
        0,
        &[
            p(s.chw.cast_const()),
            pm(s.chlast),
            transpose[0],
            transpose[1],
            transpose[2],
        ],
        stream,
    )?;
    // `LaunchRule::PerRowNarrow` is `grid[rows,1,1] block[128,1,1]`, and the
    // rows are the `T*F` spatial positions with the channel axis as the
    // width. In place: it reads and writes `chlast`.
    //
    // The two launchers this one call replaces, verbatim — `launch_rules.rs`
    // reads them as the citation `per_row_narrow`'s transcription was written
    // from, and they are the reason this function takes its extents as a
    // struct: the only difference between them is which of `(T1,F1,C0)` and
    // `(T2,F2,C1)` is substituted.
    //
    // ```text
    // vd::k_layernorm_relu<bfd><<<T1*F1,128,0,S>>>(D(c0cl),D(w.sscp0_norm),D(c0cl),T1*F1,C0,EPS);
    // vd::k_layernorm_relu<bfd><<<T2*F2,128,0,S>>>(D(c1cl),D(w.sscp1_norm),D(c1cl),T2*F2,C1,EPS);
    // ```
    let rows = s
        .t_out
        .checked_mul(s.f_out)
        .ok_or_else(|| Error::invalid(WHO, "SSCP row count overflowed"))?;
    fire(
        "vision::k_layernorm_relu_bf16",
        rect(extent("ln rows", rows)?, extent("ln width", s.out_ch)?),
        &[
            p(s.chlast.cast_const()),
            p(s.norm_w),
            pm(s.chlast),
            ArgValue::I32(rows),
            ArgValue::I32(s.out_ch),
            ArgValue::F32(s.eps),
        ],
        stream,
    )?;
    fire_stated(
        "vision::k_chfirst_bf16",
        grid,
        B2,
        0,
        &[
            p(s.chlast.cast_const()),
            pm(s.chw),
            transpose[0],
            transpose[1],
            transpose[2],
        ],
        stream,
    )
}

/// `vd::k_add<bfd><<<(n+255)/256, 256, 0, S>>>(a, b, n)` — `a += b`.
///
/// `LaunchRule::Elementwise`, and the row's `in_place` cell already states
/// that operand 0 is both the read and the write.
fn add(y: *mut c_void, x: *const c_void, n: usize, stream: StreamRef<'_>) -> Result<()> {
    let rows = u32::try_from(n)
        .map_err(|_| Error::invalid(WHO, "residual element count overflowed a grid extent"))?;
    fire(
        "vision::k_add_bf16",
        rect(rows, 1),
        &[pm(y), p(x), ArgValue::Usize(n)],
        stream,
    )
}

/// `vd::k_silu<bfd><<<(n+255)/256, 256, 0, S>>>(x, x, n)`, in place.
fn silu(x: *mut c_void, n: usize, stream: StreamRef<'_>) -> Result<()> {
    let rows = u32::try_from(n)
        .map_err(|_| Error::invalid(WHO, "silu element count overflowed a grid extent"))?;
    fire(
        "vision::k_silu_bf16",
        rect(rows, 1),
        &[p(x.cast_const()), pm(x), ArgValue::Usize(n)],
        stream,
    )
}

/// The macaron half-FFN — `gemma4_audio.cu:223-231`'s `ffn` lambda.
///
/// `rms → clin(fc1) → silu → clin(fc2) → rms → axpy`, six composed steps over
/// two intermediates (`ffmid` at `[N, 4*hidden]` and `ffout` at `[N, hidden]`)
/// and one in-place scale-and-accumulate onto the residual stream. This is
/// the shape the owner's principle names directly: kernels producing
/// intermediate results, composed by host code, and the host code is this.
fn ffn(
    g: &Walk,
    a: &Arena,
    f: &Ffn,
    residual_weight: f32,
    eps: f32,
    stream: StreamRef<'_>,
) -> Result<()> {
    let (n, hd, im) = (g.n, g.hd, g.im);
    rms(a.h.cast_const(), f.pre_ln, a.hn, n, hd, eps, stream)?;
    clin(a.hn.cast_const(), a.ffmid, a.xc, &f.fc1, n, hd, im, stream)?;
    silu(a.ffmid, elems("ffn silu", n, im)?, stream)?;
    clin(a.ffmid.cast_const(), a.ffout, a.xc, &f.fc2, n, im, hd, stream)?;
    rms(a.ffout.cast_const(), f.post_ln, a.ffout, n, hd, eps, stream)?;
    // `k_axpy<bfd><<<(N*Hd+255)/256, 256, 0, S>>>(h, ffout, RW, N*Hd)` —
    // `h += RW * ffout`, the macaron half-step's residual weight.
    let count = elems("ffn residual", n, hd)?;
    let rows = u32::try_from(count)
        .map_err(|_| Error::invalid(WHO, "axpy element count overflowed a grid extent"))?;
    fire(
        "vision::k_axpy_bf16",
        rect(rows, 1),
        &[
            pm(a.h),
            p(a.ffout.cast_const()),
            ArgValue::F32(residual_weight),
            ArgValue::Usize(count),
        ],
        stream,
    )
}

/// Frames after subsampling — `gemma4_audio.hpp`'s
/// `gemma4_audio_subsampled_len`, `floor((n-1)/2)+1` twice.
#[must_use]
pub fn subsampled_len(n_frames: i32) -> i32 {
    let conv = |n: i32| (n + 2 - 3) / 2 + 1;
    conv(conv(n_frames))
}

/// The encode-ABI entry: host log-mel features in, host bf16 rows out.
///
/// The port of `encode_gemma4_audio` (`gemma4_audio.cu:307-351`). One clip per
/// iteration, each with its own arena, each synchronised before the next —
/// the shape the C++ had, kept because the output rows are read back to the
/// host between clips.
///
/// `features` is the whole feature plane as BYTES and `feature_byte_indptr`
/// cuts it, which is what the plan carries.
///
/// # Errors
///
/// A feature span that is not a whole number of mel frames, an output buffer
/// too small for the rows this clip produces, or any refused launch. Each is
/// the `throw` the C++ made at the same point, as a value.
pub fn encode(
    w: &Weights,
    features: &[u8],
    feature_byte_indptr: &[u32],
    output_rows: &mut [u8],
    output_row_indptr: &mut [u32],
    stream: StreamRef<'_>,
) -> Result<()> {
    let num_clips = output_row_indptr.len().saturating_sub(1);
    if num_clips == 0 || feature_byte_indptr.len() < num_clips + 1 {
        return Err(Error::invalid(WHO, "invalid standalone encode inputs"));
    }
    let n_mel = w.n_mel;
    let mel = usize::try_from(n_mel)
        .ok()
        .filter(|v| *v != 0)
        .ok_or_else(|| Error::invalid(WHO, "the tower states no mel bin count"))?;
    let row_bytes = usize::try_from(w.text_hidden).unwrap_or(0) * 2;
    let mut rows_written = 0usize;
    output_row_indptr[0] = 0;
    for clip in 0..num_clips {
        let blo = feature_byte_indptr[clip] as usize;
        let bhi = feature_byte_indptr[clip + 1] as usize;
        if bhi < blo || bhi > features.len() {
            return Err(Error::invalid(
                WHO,
                format!("clip {clip}'s feature span [{blo}, {bhi}) leaves the payload"),
            ));
        }
        let floats = (bhi - blo) / 4;
        let frames = floats / mel;
        if frames == 0 || !floats.is_multiple_of(mel) {
            return Err(Error::invalid(
                WHO,
                format!("invalid feature shape ({floats} floats over {mel} mel bins)"),
            ));
        }
        let frames_i = i32::try_from(frames)
            .map_err(|_| Error::invalid(WHO, "mel frame count overflowed an int"))?;
        let rows = subsampled_len(frames_i);
        let rows_u = usize::try_from(rows)
            .map_err(|_| Error::invalid(WHO, "subsampled row count is negative"))?;
        let want = rows_written
            .checked_add(rows_u)
            .and_then(|r| r.checked_mul(row_bytes))
            .ok_or_else(|| Error::invalid(WHO, "output row count overflowed"))?;
        if want > output_rows.len() {
            return Err(Error::invalid(WHO, "encode output buffer too small"));
        }

        let mut scratch = Scratch::new();
        let projected = scratch.bf16(
            rows_u
                .checked_mul(usize::try_from(w.text_hidden).unwrap_or(0))
                .ok_or_else(|| Error::invalid(WHO, "projected row buffer overflowed"))?,
        )?;
        run(
            w,
            &features[blo..bhi],
            frames_i,
            n_mel,
            rows,
            projected,
            stream,
        )?;
        // `:342-346` — the rows come back to the host between clips, because
        // the encode ABI's output is a host buffer.
        let begin = rows_written * row_bytes;
        let end = begin + rows_u * row_bytes;
        // SAFETY: `projected` is `rows * text_hidden` bf16 elements of this
        // arena, live until `scratch` drops below, and `end - begin` is
        // exactly that many bytes.
        unsafe { read_raw_span(projected.cast_const(), &mut output_rows[begin..end], stream)? };
        stream.synchronize()?;
        drop(scratch);
        rows_written += rows_u;
        output_row_indptr[clip + 1] = u32::try_from(rows_written)
            .map_err(|_| Error::invalid(WHO, "encoded row count overflowed u32"))?;
    }
    Ok(())
}
