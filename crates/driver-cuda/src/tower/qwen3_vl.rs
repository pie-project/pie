//! Qwen3-VL's vision tower: the host walk, in Rust.
//!
//! The port of `driver-cuda/csrc/vision/qwen3_vl_tower.cu` (522 lines),
//! `qwen3_vl_tower_c.cpp` (114) and `vis_helpers.cpp` (136) — 772 lines that
//! were host C++ over device text which had already moved to
//! `kernels-cuda/kernels/vision/qwen3_vl_tower.cuh` and
//! `tower_naive_kernels.cuh`. What is here is that same walk with the
//! `<<<>>>` written by [`super::fire`] and [`super::fire_stated`] instead of
//! by nvcc, and with the last `.cu` in `driver-cuda/csrc/vision/` deleted
//! behind it.
//!
//! **`qwen3_vl_tower.cu` held ZERO `__global__` and sixteen `<<<>>>`.** It was
//! a host program wearing a `.cu` extension because `<<<>>>` needs nvcc to
//! parse, and the sixteen launches named nine kernels that were already rows
//! in `families::vision`. That is the whole of why this file exists.
//!
//! # Every launch, and the launcher it reproduces
//!
//! Sixteen `<<<>>>` at `qwen3_vl_tower.cu:122, 147, 148, 164, 165, 168, 169,
//! 236, 246, 249, 257, 259, 261, 263, 459, 498`. Each fire below quotes the
//! line it replaces; where one fire stands for two identical launches (the
//! two `k_merge_gather` arms are two fires, the two `k_f32_to_bf16` are one)
//! the comment cites both. `:459` is the DEAD batched arm's — see
//! [`scatter`] — and is cited beside `:498`, the live one, because the
//! expression is the same and the dead arm's is the one being retired.
//!
//! Fifteen of the sixteen are JIT rows. The sixteenth is not a `<<<>>>` at
//! all: `qwen3vl_vis_gemm_bf16` is a cuBLAS dispatch with a runtime autotuner
//! behind it, and that autotuner is now `fire::gemm::act_x_wt_bf16` — Rust,
//! reached directly, where it used to cross through the generated
//! `pie_k_gemm_act_x_wt_bf16`. `fire/lora.rs` and `tower::gemma4_vision` take
//! the same route.
//!
//! # What is still C++, and what retires it
//!
//! Two callees, both calls rather than programs:
//!
//! * `gemm::act_x_wt_bf16` — cuBLAS, above.
//! * The FlashInfer prefill trio, in [`attn`]. That is the FA2 lattice, and
//!   **north star §5 step 8 retires it**; this step does not port it and does
//!   not delete `csrc/attn/attention_flashinfer.cu`. `fire/attn_score.rs` is
//!   the shape being followed: a Rust walk calling a driver-owned launcher
//!   across the C ABI is a walk in Rust.
//!
//! # The 2-D RoPE is not gemma's
//!
//! `qwen3_vl_tower.cu:14-17`: the ViT rope *"uses transformers' rotate_half
//! layout (head_dim split into a row-half and a col-half, then full
//! rotate_half over each — NOT gemma's interleaved pairs)"*. That is inside
//! `k_split_rope_qkv` and nothing here changes it, but it is the reason the
//! two towers cannot share a rope row.
//!
//! # Parity
//!
//! `qwen3_vl_tower.cu:21-23`: *"NOT yet parity-verified — a faithful first
//! draft to be checked against `scripts/qwen3_vl_vision_parity_ref.py` dumps
//! (`/tmp/qwen3_vl_vision_parity/`). Per MULTIMODAL.md §11, parity is
//! bf16-vs-bf16 rel_rms + cosine, not max_abs."* Still true, and this port
//! changes no arithmetic: the same kernels are fired at the same geometries
//! over the same buffers in the same order.
//!
//! The two open PARITY TODOs are carried at the code they qualify —
//! [`interp_pos_embed`] (the linspace endpoints and the floor/ceil clamp) and
//! the within-group order the merge gather assumes.
//!
//! # The parity checkpoint hook is gone
//!
//! `set_qwen3vl_vision_ckpt` and the `emit_ckpt` lambda it fed
//! (`qwen3_vl_tower.cu:82-85, 227-235`) are not reproduced. It was a
//! `Qwen3VLVisionCkptFn` defaulted to `nullptr` that nothing in the tree ever
//! set, and every call site began `if(!g_qvis_ckpt)return;` followed by a
//! `cudaStreamSynchronize` — a debugging tap that cost a declaration and, had
//! it been armed, a drain per layer. `tower::gemma4_vision` dropped its
//! `VisDebugTap` for the same reason and by the same evidence.
//!
//! # Two dead knobs, and the measurements they carry
//!
//! `constexpr bool VTIM = false` (`:212`, `:409`) gated `PIE_VIS_TIMING`
//! phase timing: patch / per-layer attention / layers-total / merger, over
//! `cudaEvent` pairs *"computed after the final sync so it doesn't perturb"*,
//! plus a host-side `steady_clock` accumulator for the pure-CPU interpolation
//! work. `constexpr bool GEMM_ONLY = false` (`:218`) was an ablation:
//! *"skip the per-layer elementwise launches (norm/split/rope/gelu/bias) so
//! VTIM's `rest` measures GEMM-only time — isolates tensor-core MFU from the
//! memory-bound elementwise stalls between GEMMs. Output is garbage under
//! this flag; timing only."* Neither knob is reproduced — a `constexpr false`
//! is not a feature — but what they were FOR is recorded here, because the
//! next person to ask "where does this tower's time go?" is asking the
//! question those two answered.

pub mod attn;

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use kernels_cuda::vision;

use super::call;
use crate::device::{
    Allocator, DeviceBuffer, StreamRef, copy_raw_span, fill_raw_span, read_raw_span,
};
use crate::{Error, Result};

/// The tower's name, in ONE place — `tower::gemma4_vision::WHO`'s reason.
///
/// `qwen3vl_vision` and not `qwen3_vl`: it is the string the C++ threw with
/// (`"qwen3vl_vision: "` at `qwen3_vl_tower.cu:88`), so every refusal below
/// reads exactly as the `throw` it replaces — which matters, because
/// `execution.rs`'s `WALKED` entry quoted two of them by text.
const WHO: &str = "qwen3vl_vision";

/// Pointers per encoder block in the flat table — `qwen3_vl_tower_c.cpp:69`'s
/// `block_w + i * 12`.
const SLOTS_PER_BLOCK: usize = 12;

/// Pointers per merger (main or deepstack) — `qwen3_vl_tower_c.cpp:39`'s
/// six-pointer table `[norm.g, norm.b, fc1.w, fc1.b, fc2.w, fc2.b]`.
const SLOTS_PER_MERGER: usize = 6;

// ── The arena's slots ────────────────────────────────────────────────────
//
// THE INDICES ARE DISJOINT BY DESIGN AND THAT IS LOAD-BEARING. The walk pulls
// raw addresses out of the arena and holds them across calls that may grow
// OTHER slots — `run` holds `h`/`hn`/`qkv`/… while `run_merger` sizes its
// three, and `scatter` holds the pixel and output slabs while `run` sizes all
// of its. Growing a slot frees and re-allocates that slot's buffer, so two
// live users of ONE index would be a dangling launch argument. No index below
// is used by two frames at once.

/// The residual stream, `[N, hidden]`.
const S_H: usize = 0;
/// The normed copy the projections read, `[N, hidden]`.
const S_HN: usize = 1;
/// The fused QKV projection, `[N, 3 * hidden]`.
const S_QKV: usize = 2;
/// Queries, `[N, hidden]`.
const S_Q: usize = 3;
/// Keys, `[N, hidden]`.
const S_K: usize = 4;
/// Values, `[N, hidden]`.
const S_V: usize = 5;
/// The attention output, `[N, hidden]`.
const S_ATTN: usize = 6;
/// The MLP's hidden activation, `[N, intermediate]`.
const S_MID: usize = 7;
/// A merger's normed patches, `[n_patch, hidden]`.
const S_MERGE_NORMED: usize = 8;
/// A merger's 2x2-grouped rows, `[n_token, 4 * hidden]`.
const S_MERGE_GROUPED: usize = 9;
/// A merger's MLP activation, `[n_token, 4 * hidden]`.
const S_MERGE_MID: usize = 10;
/// The uploaded pixel plane as `f32`.
const S_PIX_F32: usize = 11;
/// The same plane cast to bf16.
const S_PIX_BF: usize = 12;
/// The main merger's output for one image, `[n_token, out_hidden]`.
const S_MAIN: usize = 13;
/// The first deepstack merger's output; `S_DEEP + d` is the `d`-th.
const S_DEEP: usize = 14;

/// LayerNorm parameters — `QVisLayerNorm`: gamma + beta, both nullable.
///
/// `eps` is not here for the reason the C++ header gives: it lives on
/// [`Weights::ln_eps`], one value for the whole tower.
#[derive(Clone, Copy, Debug)]
pub struct LayerNorm {
    /// Weight `[dim]`, or null.
    pub g: *const c_void,
    /// Bias `[dim]`, or null.
    pub b: *const c_void,
}

impl LayerNorm {
    /// Two consecutive slots — `qwen3_vl_tower_c.cpp:23`'s `ln(g, b)`.
    const fn of(g: *const c_void, b: *const c_void) -> Self {
        Self { g, b }
    }
}

/// One linear projection — `QVisLinear`: weight `[out, in]` row-major plus an
/// optional bias `[out]`.
#[derive(Clone, Copy, Debug)]
pub struct Linear {
    /// The weight matrix.
    pub w: *const c_void,
    /// The bias, or null for no bias.
    pub b: *const c_void,
}

impl Linear {
    /// Two consecutive slots — `qwen3_vl_tower_c.cpp:30`'s `lin(w, b)`.
    const fn of(w: *const c_void, b: *const c_void) -> Self {
        Self { w, b }
    }
}

/// One pre-norm ViT block — `QVisBlock`, in the stride-12 table's order.
#[derive(Clone, Copy, Debug)]
pub struct Block {
    /// Pre-attention norm.
    pub norm1: LayerNorm,
    /// Pre-MLP norm.
    pub norm2: LayerNorm,
    /// The fused `[3 * hidden, hidden]` projection (split after the matmul).
    pub qkv: Linear,
    /// The output projection, `[hidden, hidden]`.
    pub o: Linear,
    /// MLP up, `[intermediate, hidden]`.
    pub fc1: Linear,
    /// MLP down, `[hidden, intermediate]`.
    pub fc2: Linear,
}

/// A patch merger, main or deepstack — `QVisMerger`.
///
/// The 2x2 spatial merge groups four consecutive patch rows (the input is
/// already in spatial-merge order, because [`merge_reorder`] put it there)
/// into a `4 * hidden` vector, then `fc1` -> GELU -> `fc2` -> `out_hidden`.
#[derive(Clone, Copy, Debug)]
pub struct Merger {
    /// `[hidden]` for the main merger, `[4 * hidden]` for a deepstack one.
    pub norm: LayerNorm,
    /// `[4 * hidden, 4 * hidden]` (+bias).
    pub fc1: Linear,
    /// `[out_hidden, 4 * hidden]` (+bias).
    pub fc2: Linear,
    /// False = main (norm BEFORE the shuffle), true = deepstack
    /// (`use_postshuffle_norm=True`, norm AFTER it over `4 * hidden`).
    pub is_postshuffle: bool,
}

impl Merger {
    /// The six-pointer merger table — `qwen3_vl_tower_c.cpp:39`'s
    /// `merger_of(t, postshuffle)`.
    fn of(t: &[*const c_void], is_postshuffle: bool) -> Self {
        Self {
            norm: LayerNorm::of(t[0], t[1]),
            fc1: Linear::of(t[2], t[3]),
            fc2: Linear::of(t[4], t[5]),
            is_postshuffle,
        }
    }
}

/// The tower's weights and config — `QwenVisRawWeights`.
///
/// The reference geometry, from `vision/qwen3_vl_tower.hpp`'s header (the
/// `Qwen/Qwen3-VL-2B-Instruct` config against transformers 5.9
/// `modeling_qwen3_vl.py`): depth 24, hidden 1024, heads 16 (head_dim 64),
/// intermediate 4096, patch 16, temporal_patch 2, spatial_merge 2 (2x2 -> 4
/// patches per token), out_hidden 2048 (= text hidden), num_position_embeddings
/// 2304 (a 48x48 learned absolute table, bilinearly interpolated to the grid
/// and ADDED), 2-D RoPE theta 10000, `hidden_act` gelu_pytorch_tanh, LayerNorm
/// eps 1e-6. The deepstack mergers tap layers {5, 11, 17} and are added into
/// the text decoder at LLM layers 0/1/2 on image rows.
///
/// Nothing here checks those numbers. The tower runs whatever the loader
/// published, exactly as the C++ did — the ONE shape check is the
/// `hidden == heads * head_dim` identity in [`run`], because a mismatch there
/// is not slow, it is out of bounds.
#[derive(Clone, Debug)]
pub struct Weights {
    /// `patch_embed.proj`: a Conv3d flattened to a matmul
    /// `[hidden, in_channels * temporal_patch * patch * patch]` plus a bias.
    pub patch: Linear,
    /// The learned absolute position-embedding table,
    /// `[num_pos_embed, hidden]`, bf16 on the device.
    pub pos_embed: *const c_void,
    /// The encoder blocks, in depth order.
    pub blocks: Vec<Block>,
    /// The main patch merger.
    pub merger: Merger,
    /// The deepstack mergers, in tap order.
    pub deepstack: Vec<Merger>,
    /// Which block each deepstack merger taps after.
    pub deepstack_layer_idx: Vec<i32>,
    /// Model width.
    pub hidden: i32,
    /// Attention heads.
    pub heads: i32,
    /// `hidden / heads`, computed once at marshalling.
    pub head_dim: i32,
    /// MLP width.
    pub intermediate: i32,
    /// Spatial patch edge.
    pub patch_size: i32,
    /// Temporal patch depth.
    pub temporal_patch_size: i32,
    /// The spatial-merge edge; the merge unit is its square.
    pub spatial_merge_size: i32,
    /// Input channels.
    pub in_channels: i32,
    /// The text hidden size the mergers project into.
    pub out_hidden: i32,
    /// Rows in the absolute position-embedding table.
    pub num_pos_embed: i32,
    /// The table's grid edge, `round(sqrt(num_pos_embed))`.
    pub num_grid_per_side: i32,
    /// LayerNorm epsilon.
    pub ln_eps: f32,
    /// 2-D RoPE theta.
    pub rope_theta: f32,
}

impl Weights {
    /// Rebuild the tower from the flat pointer tables — the whole of
    /// `qwen3_vl_tower_c.cpp`.
    ///
    /// That file was 114 lines and was *"marshalling only — every launch and
    /// every byte of host prep is `qwen3_vl_tower.cu`'s"*. In Rust the walk
    /// consumes this struct directly, so the marshalling is the only thing it
    /// did, and it is these thirty lines: `block_w` at stride
    /// [`SLOTS_PER_BLOCK`], `merger_w` and each `deepstack_w` at stride
    /// [`SLOTS_PER_MERGER`], with the strides NAMED instead of open-coded as
    /// `t[0]`..`t[11]` off a `const void* const*`.
    ///
    /// `head_dim` and `num_grid_per_side` are derived here exactly as the
    /// C++ derived them (`heads > 0 ? hidden / heads : 0` and
    /// `lround(sqrt(num_pos_embed))`) so that the walk never recomputes
    /// either.
    ///
    /// # Errors
    ///
    /// A table shorter than the depth or merger count it is supposed to
    /// describe — a refusal, because reading one slot past the end is a
    /// weight pointer from another block, and every one of them is a live
    /// device address that would launch rather than fault.
    #[allow(clippy::too_many_arguments)]
    pub fn from_flat(
        patch_w: *const c_void,
        patch_b: *const c_void,
        pos_embed: *const c_void,
        block_w: &[*const c_void],
        depth: usize,
        merger_w: &[*const c_void],
        deepstack_w: &[*const c_void],
        deepstack_layers: &[i32],
        hidden: i32,
        heads: i32,
        intermediate: i32,
        patch_size: i32,
        temporal_patch: i32,
        merge_size: i32,
        in_channels: i32,
        out_hidden: i32,
        num_pos_embed: i32,
        ln_eps: f32,
        rope_theta: f32,
    ) -> Result<Self> {
        let want = depth
            .checked_mul(SLOTS_PER_BLOCK)
            .ok_or_else(|| Error::invalid(WHO, "block table length overflowed"))?;
        if block_w.len() < want {
            return Err(Error::invalid(
                WHO,
                format!(
                    "block table holds {} pointers for {depth} blocks of \
                     {SLOTS_PER_BLOCK}, which needs {want}",
                    block_w.len()
                ),
            ));
        }
        if merger_w.len() < SLOTS_PER_MERGER {
            return Err(Error::invalid(
                WHO,
                format!(
                    "the main merger needs {SLOTS_PER_MERGER} pointers, not {}",
                    merger_w.len()
                ),
            ));
        }
        let num_deep = deepstack_layers.len();
        let want_deep = num_deep
            .checked_mul(SLOTS_PER_MERGER)
            .ok_or_else(|| Error::invalid(WHO, "deepstack table length overflowed"))?;
        if deepstack_w.len() < want_deep {
            return Err(Error::invalid(
                WHO,
                format!(
                    "deepstack table holds {} pointers for {num_deep} mergers of \
                     {SLOTS_PER_MERGER}, which needs {want_deep}",
                    deepstack_w.len()
                ),
            ));
        }
        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            let t = &block_w[i * SLOTS_PER_BLOCK..(i + 1) * SLOTS_PER_BLOCK];
            blocks.push(Block {
                norm1: LayerNorm::of(t[0], t[1]),
                norm2: LayerNorm::of(t[2], t[3]),
                qkv: Linear::of(t[4], t[5]),
                o: Linear::of(t[6], t[7]),
                fc1: Linear::of(t[8], t[9]),
                fc2: Linear::of(t[10], t[11]),
            });
        }
        let deepstack = (0..num_deep)
            .map(|d| {
                Merger::of(&deepstack_w[d * SLOTS_PER_MERGER..(d + 1) * SLOTS_PER_MERGER], true)
            })
            .collect();
        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let num_grid_per_side = f64::from(num_pos_embed).sqrt().round() as i32;
        Ok(Self {
            patch: Linear::of(patch_w, patch_b),
            pos_embed,
            blocks,
            merger: Merger::of(merger_w, false),
            deepstack,
            deepstack_layer_idx: deepstack_layers.to_vec(),
            hidden,
            heads,
            head_dim: if heads > 0 { hidden / heads } else { 0 },
            intermediate,
            patch_size,
            temporal_patch_size: temporal_patch,
            spatial_merge_size: merge_size,
            in_channels,
            out_hidden,
            num_pos_embed,
            num_grid_per_side,
            ln_eps,
            rope_theta,
        })
    }

    /// `in_channels * temporal_patch * patch * patch` — 1536 at the reference
    /// geometry, and the width of one patch row of the pixel plane.
    ///
    /// `run_qwen3vl_vision`'s `PATCH_DIM` (`qwen3_vl_tower.cu:190`) and
    /// `scatter_qwen3vl_vision`'s (`:387`), which were the same expression
    /// written twice.
    fn patch_dim(&self) -> Result<i32> {
        self.in_channels
            .checked_mul(self.temporal_patch_size)
            .and_then(|v| v.checked_mul(self.patch_size))
            .and_then(|v| v.checked_mul(self.patch_size))
            .filter(|v| *v > 0)
            .ok_or_else(|| Error::invalid(WHO, "the patch geometry is empty or overflowed"))
    }

    /// The merge unit, `spatial_merge_size ^ 2` — four at the reference
    /// geometry, and the number of patch rows one merged token consumes.
    fn merge_unit(&self) -> Result<i32> {
        self.spatial_merge_size
            .checked_mul(self.spatial_merge_size)
            .filter(|v| *v > 0)
            .ok_or_else(|| Error::invalid(WHO, "spatial_merge_size is zero or overflowed"))
    }
}

/// A count as `usize`, refusing a negative rather than saturating.
fn count(what: &'static str, value: i32) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::invalid(WHO, format!("{what}: {value} is negative")))
}

// ── Host helpers mirroring transformers `vision_utils` ───────────────────

/// The spatial-merge reorder permutation: `perm[k]` is the source patch index
/// for output position `k`, for one `(t, h, w)` grid.
///
/// `qwen3_vl_tower.cu:319-336`. Mirrors the `reorder` in
/// `get_vision_bilinear_indices_and_weights` / `get_vision_position_ids`:
///
/// ```text
/// reorder = (h_idx[:, :, None, None] * w + w_idx[None, None, :, :])
///           .transpose(1,2).flatten().repeat(t)
/// ```
///
/// where `h_idx = arange(h).view(h/m, m)` and `w_idx = arange(w).view(w/m, m)`
/// — i.e. iterate blocks `(bh, bw)`, then within-block `(ih, iw)`, giving
/// `src = bh*m*w + ih*w + bw*m + iw` (+ `frame*h*w` for `t > 1`).
///
/// This is why `k_merge_gather` is a plain concatenation of four consecutive
/// rows where HF needs a five-way reshape: the host has already put the
/// patches in the order the gather assumes.
///
/// PARITY TODO (`qwen3_vl_tower.cu:344`, carried): the WITHIN-GROUP order of
/// the four rows a merged token concatenates is asserted by this permutation
/// and by the kernel together; the reference dump is what settles it.
fn merge_reorder(t: i32, h: i32, w: i32, m: i32) -> Vec<usize> {
    let mut perm = Vec::new();
    for f in 0..t {
        for bh in 0..h / m {
            for bw in 0..w / m {
                for ih in 0..m {
                    for iw in 0..m {
                        let src = f * h * w + (bh * m + ih) * w + (bw * m + iw);
                        perm.push(src.max(0).unsigned_abs() as usize);
                    }
                }
            }
        }
    }
    perm
}

/// `(row, col)` RoPE position ids per patch, in spatial-merge order.
///
/// `qwen3_vl_tower.cu:340-353`. `row = bh*m + ih`, `col = bw*m + iw`, in the
/// same nesting as [`merge_reorder`] — matches `get_vision_position_ids`.
/// Floats because `k_split_rope_qkv` consumes them as trigonometric
/// arguments.
#[allow(clippy::cast_precision_loss)]
fn vision_rope_positions(t: i32, h: i32, w: i32, m: i32) -> Vec<f32> {
    let mut pos = Vec::new();
    for _ in 0..t {
        for bh in 0..h / m {
            for bw in 0..w / m {
                for ih in 0..m {
                    for iw in 0..m {
                        pos.push((bh * m + ih) as f32);
                        pos.push((bw * m + iw) as f32);
                    }
                }
            }
        }
    }
    pos
}

/// Bilinear-interpolate the `[num_pos_embed, hidden]` absolute position table
/// to the `(h, w)` grid, in spatial-merge order, giving `[t*h*w, hidden]`.
///
/// `qwen3_vl_tower.cu:355-390`. Mirrors
/// `get_vision_bilinear_indices_and_weights` followed by
/// `(pos_embed(idx) * weight).sum(0)`. `table` is the `[num_pos_embed,
/// hidden]` device table already copied to the host as `f32`.
///
/// PARITY TODO (`qwen3_vl_tower.cu:359-361`, carried verbatim): *"verify
/// against `scripts/qwen3_vl_vision_parity_ref.py` `pos_embed_interp` — esp.
/// the `linspace(0, side-1, h/w)` endpoints and the floor/ceil clamp at
/// `side-1`."*
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn interp_pos_embed(
    table: &[f32],
    side: i32,
    hidden: i32,
    t: i32,
    h: i32,
    w: i32,
    m: i32,
) -> Vec<f32> {
    let hid = hidden.max(0) as usize;
    let idx = |gh: i32, gw: i32| (gh * side + gw).max(0) as usize;
    // `linspace(0, side-1, n)` evaluated at `i`.
    let frac = |n: i32, i: i32| -> f32 {
        if n == 1 { 0.0 } else { i as f32 * (side - 1) as f32 / (n - 1) as f32 }
    };
    // The per-(row, col) interpolated embedding, in ROW-MAJOR (un-reordered)
    // order; the gather below puts it in spatial-merge order.
    let mut lin = vec![0.0f32; (h.max(0) as usize) * (w.max(0) as usize) * hid];
    for i in 0..h {
        let hg = frac(h, i);
        let hf = hg as i32;
        let hc = (hf + 1).min(side - 1);
        let hfr = hg - hf as f32;
        for j in 0..w {
            let wg = frac(w, j);
            let wf = wg as i32;
            let wc = (wf + 1).min(side - 1);
            let wfr = wg - wf as f32;
            let (w00, w01) = ((1.0 - hfr) * (1.0 - wfr), (1.0 - hfr) * wfr);
            let (w10, w11) = (hfr * (1.0 - wfr), hfr * wfr);
            let (p00, p01) = (idx(hf, wf) * hid, idx(hf, wc) * hid);
            let (p10, p11) = (idx(hc, wf) * hid, idx(hc, wc) * hid);
            let o = ((i as usize) * (w.max(0) as usize) + j as usize) * hid;
            for c in 0..hid {
                lin[o + c] = w00 * table[p00 + c]
                    + w01 * table[p01 + c]
                    + w10 * table[p10 + c]
                    + w11 * table[p11 + c];
            }
        }
    }
    // Gather into spatial-merge order, repeated over `t` frames.
    let perm = merge_reorder(t, h, w, m);
    let plane = (h.max(0) as usize) * (w.max(0) as usize);
    let mut out = vec![0.0f32; perm.len() * hid];
    for (k, src) in perm.iter().enumerate() {
        // Frame-independent: the table is per spatial position.
        let s = if plane == 0 { 0 } else { src % plane };
        out[k * hid..(k + 1) * hid].copy_from_slice(&lin[s * hid..(s + 1) * hid]);
    }
    out
}

// ── The one cache, which was two statics and a per-call `cudaMallocAsync` ──

/// One grid's precomputed side inputs, on the device.
///
/// Held for the process, which is what the C++ did: `pe_cache`'s entries came
/// from `cudaMalloc` (not `cudaMallocAsync`) and were never freed, because a
/// grid seen once is seen again on the next image of the same size.
#[derive(Debug)]
struct Grid {
    /// `[n_patch, 2]` `(row, col)` ids.
    rope: DeviceBuffer,
    /// `[n_patch, hidden]` interpolated position embeddings, bf16.
    pe: DeviceBuffer,
}

/// Everything the tower keeps between forward passes.
///
/// # What the interpolation cache became, and why
///
/// The C++ had TWO function-local statics, each with its own `std::mutex`:
///
/// * `tbl_mu` / `tbl_key` / `tbl_cache` (`:411`) — the host `f32` copy of the
///   absolute position table, keyed by the device pointer, *"so we don't
///   re-do the ~5 MB D2H + convert on every forward pass. Forward passes for
///   one model are serialized by the engine; the mutex guards the (rare)
///   first-touch."*
/// * `pe_mu` / `pe_cache` (`:426-427`) — a `std::map<tuple<int,int,int>,
///   pair<float*, bf*>>` of `cudaMalloc`'d rope/pe buffers, because *"rope
///   positions + interpolated pos-embed are a deterministic function of the
///   grid — identical for every same-size image. Cache the device buffers by
///   grid so the CPU interp + bf16 convert + H2D run ONCE, not per image."*
///
/// Both become fields of ONE `Mutex<Res>`, which is the change worth making
/// rather than transliterating. Three reasons:
///
/// 1. **The two caches are one invariant.** `pe_cache`'s entries are computed
///    FROM `tbl_cache`; keying them separately let a table swap leave stale
///    grids behind. One lock, one key check, one eviction — [`Res::table`]
///    clears `grids` when the table pointer changes, which the C++ never did.
/// 2. **`std::map` was ordered for no reason and hashed for none either.** A
///    `BTreeMap<(i32, i32, i32), Grid>` is the same structure with the same
///    ordering, holds a handful of entries, and needs no `Hash` on a key that
///    is three integers.
/// 3. **A `Mutex` here is what `Allocator` wants anyway.** The grids own
///    [`DeviceBuffer`]s rather than raw `cudaMalloc` results, so they are
///    freed if the cache is ever cleared instead of leaked by construction —
///    and a `DeviceBuffer` is `Send` without an `unsafe impl` because
///    `DevPtr` is a `usize`, so this whole struct is `Send` by derivation.
///    Compare `kernels_cuda::gemm::quant`'s `DequantWeightCache`, which
///    needs `unsafe impl Send` for a cache holding raw pointers.
///
/// # And the arena, which is the third thing the C++ needed and did not name
///
/// `run_qwen3vl_vision` and `run_merger` allocated their scratch with
/// `cudaMallocAsync` and released it with `cudaFreeAsync` — stream-ordered,
/// so neither call drained the stream. That is not decoration: the scatter's
/// closing comment is *"No sync here: all work is stream-ordered on S… \
/// Draining mid-forward only stalled the CPU from queuing the LLM layers."*
/// [`super::Scratch`], which the gemma-4 towers use, is `cudaMalloc` and
/// `cudaFree` — and BOTH synchronise the whole device. A per-forward
/// `Scratch` would therefore have put roughly twenty full device drains
/// exactly where that measurement says not to.
///
/// So the arena is persistent and grow-only: [`Res::slot`] hands out an
/// address, reallocating only when the request exceeds what it holds. Steady
/// state is zero allocations and zero drains per forward; a new image size
/// pays one free + one alloc per grown slot, which is the high-water-mark
/// shape the tree already uses for shared-memory caps.
#[derive(Debug)]
struct Res {
    /// Owns the grids and the arena slots.
    alloc: Allocator,
    /// The device address `table` was read from — the C++'s `tbl_key`.
    table_key: usize,
    /// `[num_pos_embed * hidden]` as `f32` on the host — the C++'s
    /// `tbl_cache`.
    table: Vec<f32>,
    /// Per-`(t, h, w)` rope ids and interpolated position embeddings — the
    /// C++'s `pe_cache`.
    grids: BTreeMap<(i32, i32, i32), Grid>,
    /// The walk's scratch, indexed by the `S_*` constants.
    slots: Vec<Option<DeviceBuffer>>,
}

/// The one instance — the C++'s two function-local statics, joined.
fn res() -> &'static Mutex<Res> {
    static RES: OnceLock<Mutex<Res>> = OnceLock::new();
    RES.get_or_init(|| {
        Mutex::new(Res {
            alloc: Allocator::new(),
            table_key: 0,
            table: Vec::new(),
            grids: BTreeMap::new(),
            slots: Vec::new(),
        })
    })
}

impl Res {
    /// The address of slot `index`, sized to at least `bytes`.
    ///
    /// Grow-only: a slot that is already big enough is returned as it is, so
    /// the steady state allocates nothing. See the struct docs for why this
    /// is not [`super::Scratch`].
    fn slot(&mut self, index: usize, bytes: usize) -> Result<*mut c_void> {
        if self.slots.len() <= index {
            self.slots.resize_with(index + 1, || None);
        }
        let want = bytes.max(1);
        if self.slots[index].as_ref().is_none_or(|b| b.len() < want) {
            // Freed BEFORE the replacement is taken, not after: `cudaFree`
            // and `cudaMalloc` both synchronise the device, so holding two
            // allocations at once would double the peak for no benefit.
            self.slots[index] = None;
            self.slots[index] = Some(self.alloc.alloc(want)?);
        }
        self.slots[index]
            .as_ref()
            .map(DeviceBuffer::as_ptr)
            .ok_or_else(|| Error::invalid(WHO, "an arena slot went missing"))
    }

    /// Slot `index`, sized to `src` and filled from it on `stream`.
    fn upload(&mut self, index: usize, src: &[u8], stream: StreamRef<'_>) -> Result<*mut c_void> {
        let pointer = self.slot(index, src.len())?;
        let buffer = self.slots[index]
            .as_mut()
            .ok_or_else(|| Error::invalid(WHO, "an arena slot went missing"))?;
        buffer.copy_from_host(src, stream)?;
        Ok(pointer)
    }

    /// Ensure the host `f32` copy of the absolute position table is current.
    ///
    /// `qwen3_vl_tower.cu:409-423`. The read is ~5 MB at the reference
    /// geometry (2304 x 1024 bf16), which is why it is cached at all; the
    /// key is the device pointer, so a reload that republishes the table
    /// invalidates it.
    ///
    /// The C++ used a blocking `cudaMemcpy`. This queues the read on the
    /// walk's stream and drains once — the same one-off stall, on the stream
    /// the caller already owns, and it happens on first touch rather than per
    /// forward.
    fn table(&mut self, w: &Weights, stream: StreamRef<'_>) -> Result<()> {
        let key = w.pos_embed as usize;
        if self.table_key == key && !self.table.is_empty() {
            return Ok(());
        }
        let rows = count("num_pos_embed", w.num_pos_embed)?;
        let hidden = count("hidden", w.hidden)?;
        let elems = rows
            .checked_mul(hidden)
            .ok_or_else(|| Error::invalid(WHO, "the position table overflowed"))?;
        let mut raw = vec![0u8; elems * 2];
        // SAFETY: `pos_embed` is a published weight of `[num_pos_embed,
        // hidden]` bf16 — the shape the loader allocated it at — and the span
        // is exactly that many bytes. `stream` is live for the borrow and is
        // drained below before `raw` is read.
        unsafe { read_raw_span(w.pos_embed, &mut raw, stream)? };
        stream.synchronize()?;
        self.table = raw
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect();
        // A NEW TABLE INVALIDATES EVERY GRID, which the C++ did not do: its
        // `pe_cache` outlived a `tbl_key` change and would have served
        // embeddings interpolated from the previous model's table. Nothing in
        // the tree republishes a vision table today, so this is a latent bug
        // fixed in passing rather than a behaviour change anyone can observe.
        self.grids.clear();
        self.table_key = key;
        Ok(())
    }

    /// The `(rope, pe)` device buffers for one grid, computing them once.
    ///
    /// `qwen3_vl_tower.cu:428-441`'s `grid_rope_pe` lambda: the CPU interp,
    /// the bf16 convert and the H2D run ONCE per distinct `(t, h, w)`, not
    /// per image.
    fn grid(
        &mut self,
        w: &Weights,
        key: (i32, i32, i32),
        stream: StreamRef<'_>,
    ) -> Result<(*mut c_void, *mut c_void)> {
        if !self.grids.contains_key(&key) {
            let (gt, gh, gw) = key;
            let merge = w.spatial_merge_size;
            let rope_h = vision_rope_positions(gt, gh, gw, merge);
            let pe_h =
                interp_pos_embed(&self.table, w.num_grid_per_side, w.hidden, gt, gh, gw, merge);
            // f32 -> bf16 on the HOST, as the C++ did: this is a one-off per
            // grid, and `k_f32_to_bf16` is for the pixel plane, which is per
            // forward. Truncating rather than round-to-nearest-even, which is
            // what `__float2bfloat16` does not do — see below.
            let mut pe_bf = Vec::with_capacity(pe_h.len() * 2);
            for value in &pe_h {
                pe_bf.extend_from_slice(&bf16_bytes(*value));
            }
            // SAFETY: `f32` is plain data with no padding, so the run is
            // readable as bytes for its own length. `Scratch::upload_f32s`
            // makes the same reinterpretation.
            let rope_bytes = unsafe {
                std::slice::from_raw_parts(rope_h.as_ptr().cast::<u8>(), rope_h.len() * 4)
            };
            let mut rope = self.alloc.alloc(rope_bytes.len().max(1))?;
            rope.copy_from_host(rope_bytes, stream)?;
            let mut pe = self.alloc.alloc(pe_bf.len().max(1))?;
            pe.copy_from_host(&pe_bf, stream)?;
            self.grids.insert(key, Grid { rope, pe });
        }
        let grid = self
            .grids
            .get(&key)
            .ok_or_else(|| Error::invalid(WHO, "the grid cache lost an entry it just filled"))?;
        Ok((grid.rope.as_ptr(), grid.pe.as_ptr()))
    }
}

/// `__float2bfloat16`'s bytes: round-to-nearest-even on the discarded half.
///
/// The C++ called NVIDIA's `__float2bfloat16` for the position-embedding
/// convert (`qwen3_vl_tower.cu:435`), which rounds to nearest even. A plain
/// `>> 16` truncation is the OTHER rounding and differs by one ulp on half
/// the values, which is exactly the class of silent numerics drift a port is
/// supposed not to introduce — so the rounding is written out rather than
/// approximated.
fn bf16_bytes(value: f32) -> [u8; 2] {
    let bits = value.to_bits();
    if value.is_nan() {
        // A quiet NaN, as the intrinsic produces: preserve the sign and set
        // the top mantissa bit rather than rounding a NaN payload.
        return (((bits >> 16) as u16) | 0x0040).to_le_bytes();
    }
    let rounded = bits + 0x0000_7fff + ((bits >> 16) & 1);
    ((rounded >> 16) as u16).to_le_bytes()
}

// ── The walk ─────────────────────────────────────────────────────────────

/// `kernels::gemm::act_x_wt_bf16` — `y[M,N] = x[M,K] @ W[N,K]^T`, bf16 in and
/// out with fp32 accumulate.
///
/// `vis_helpers.cpp:33`'s whole body. `beta = 1` fuses a residual add, which
/// is what the o-projection and fc2 epilogues use: the GEMM writes straight
/// into the residual stream instead of a separate add kernel.
///
/// The one host call in this walk that is not a `<<<>>>`: a cuBLAS dispatch
/// with a runtime autotuner behind it, which is
/// [`crate::fire::gemm::act_x_wt_bf16`]. Same entry
/// `tower::gemma4_vision::gemm` uses.
#[allow(clippy::too_many_arguments)]
fn gemm(
    handle: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) {
    // SAFETY: the pointers are arena allocations and published weights, live
    // until the caller synchronises; `handle` is a live cuBLAS handle bound to
    // this walk's stream — which the C++ required in the same words:
    // *"The caller guarantees the handle's stream == the `S` passed to the
    // forward, so the bias kernel below orders correctly after the GEMM."*
    unsafe {
        crate::fire::gemm::act_x_wt_bf16(handle, act, w, y, m, n, k, beta);
    }
}

/// A projection with its optional bias — `qwen3_vl_tower.cu:119-123`'s
/// `gemm_bias`. `m` rows, `o` outputs, `k` inputs.
#[allow(clippy::too_many_arguments)]
fn gemm_bias(
    cublas: *mut c_void,
    x: *const c_void,
    lin: &Linear,
    y: *mut c_void,
    m: i32,
    o: i32,
    k: i32,
    stream: StreamRef<'_>,
) -> Result<()> {
    gemm(cublas, x, lin.w, y, m, o, k, 0.0);
    if lin.b.is_null() {
        return Ok(());
    }
    // `qwen3_vl_tower.cu:122` —
    // `vd::k_bias<bfd><<<((long)M*O+255)/256,256,0,S>>>(D(y),D(lin.b),(long)M,O);`
    // `m` crosses as a `usize` and `o` as an `int`, which is the kernel's own
    // asymmetry: the bias index is `i % n` on a 64-bit `i`.
    let rows = count("bias rows", m)?;
    call("vision::k_bias_bf16", stream, |ctx| vision::k_bias_bf16(ctx, y, lin.b, rows, o))
}

/// `fc1 -> GELU -> fc2`, with bias on both — `qwen3_vl_tower.cu:143-150`.
///
/// `erf_gelu`: false = `gelu_pytorch_tanh` (the ViT block MLP), true = the
/// exact erf GELU (`nn.GELU()` with `approximate='none'`, both patch
/// mergers). Two kernels and two rows because they are two FUNCTIONS —
/// `families/vision.rs` records that merging them by name once changed
/// numerics silently.
#[allow(clippy::too_many_arguments)]
fn mlp(
    cublas: *mut c_void,
    input: *const c_void,
    mid: *mut c_void,
    out: *mut c_void,
    fc1: &Linear,
    fc2: &Linear,
    n: i32,
    d_in: i32,
    d_mid: i32,
    d_out: i32,
    erf_gelu: bool,
    stream: StreamRef<'_>,
) -> Result<()> {
    gemm_bias(cublas, input, fc1, mid, n, d_mid, d_in, stream)?;
    let elems = count("mlp rows", n)?
        .checked_mul(count("mlp width", d_mid)?)
        .ok_or_else(|| Error::invalid(WHO, "N * Dmid overflowed"))?;
    if erf_gelu {
        // `:147` —
        // `vd::k_gelu_erf<bfd><<<((long)N*Dmid+255)/256,256,0,S>>>(D(mid),D(mid),(long)N*Dmid);`
        call("vision::k_gelu_erf_bf16", stream, |ctx| {
            vision::k_gelu_erf_bf16(ctx, mid.cast_const(), mid, elems)
        })?;
    } else {
        // `:148` —
        // `vd::k_gelu_tanh<bfd><<<((long)N*Dmid+255)/256,256,0,S>>>(D(mid),D(mid),(long)N*Dmid);`
        call("vision::k_gelu_tanh_bf16", stream, |ctx| {
            vision::k_gelu_tanh_bf16(ctx, mid.cast_const(), mid, elems)
        })?;
    }
    gemm_bias(cublas, mid.cast_const(), fc2, out, n, d_out, d_mid, stream)
}

/// One patch merger over `h[n_patch, hidden]` -> `out[n_token, out_hidden]`.
///
/// `qwen3_vl_tower.cu:155-176`.
///
/// * main: LayerNorm over `hidden` on `h`, then the 2x2 group (-> `4*hidden`),
///   then `fc1` -> GELU -> `fc2`.
/// * deepstack: the 2x2 group first, then LayerNorm over `4*hidden`
///   (`use_postshuffle_norm=True`), then the MLP.
#[allow(clippy::too_many_arguments)]
fn run_merger(
    st: &mut Res,
    cublas: *mut c_void,
    m: &Merger,
    h: *const c_void,
    n_patch: i32,
    n_token: i32,
    hidden: i32,
    merge_unit: i32,
    out_hidden: i32,
    eps: f32,
    out: *mut c_void,
    stream: StreamRef<'_>,
) -> Result<()> {
    // `W` = 4096 at the reference geometry.
    let width = merge_unit
        .checked_mul(hidden)
        .ok_or_else(|| Error::invalid(WHO, "merge_unit * hidden overflowed"))?;
    let patch_elems = count("merger patches", n_patch)?
        .checked_mul(count("hidden", hidden)?)
        .ok_or_else(|| Error::invalid(WHO, "n_patch * hidden overflowed"))?;
    let group_elems = count("merged tokens", n_token)?
        .checked_mul(count("merged width", width)?)
        .ok_or_else(|| Error::invalid(WHO, "n_token * W overflowed"))?;
    let normed = st.slot(S_MERGE_NORMED, patch_elems * 2)?;
    let grouped = st.slot(S_MERGE_GROUPED, group_elems * 2)?;
    let mid = st.slot(S_MERGE_MID, group_elems * 2)?;

    // The gather's geometry, at both call sites, is
    // `dim3 B2(16,16); inline dim3 G2(int X,int Y){return dim3((X+15)/16,(Y+15)/16);}`
    // from `qwen3_vl_tower.cu:139`, so `<<<G2(W,n_token),B2,0,S>>>` is
    // `dim3((W+15)/16, (n_token+15)/16)` over `dim3(16,16)` — and `W` is the
    // `merge_unit * hidden` the routine recovers from its own operands.
    if m.is_postshuffle {
        // `:168` — `vd::k_merge_gather<bfd><<<G2(W,n_token),B2,0,S>>>(D(h),D(grouped),n_token,merge_unit,hidden);`
        call("vision::k_merge_gather_bf16", stream, |ctx| {
            vision::k_merge_gather_bf16(ctx, h, grouped, n_token, merge_unit, hidden)
        })?;
        // `:169` — `vd::k_layernorm<bfd><<<n_token,256,0,S>>>(D(grouped),D(m.norm.g),D(m.norm.b),D(grouped),n_token,W,eps);`
        // The kernel's `__shared__` is static, so zero dynamic bytes is the
        // whole contract.
        call("vision::k_layernorm_bf16", stream, |ctx| {
            vision::k_layernorm_bf16(
                ctx,
                grouped.cast_const(),
                m.norm.g,
                m.norm.b,
                grouped,
                n_token,
                width,
                eps,
            )
        })?;
    } else {
        // `:164` — `vd::k_layernorm<bfd><<<n_patch,256,0,S>>>(D(h),D(m.norm.g),D(m.norm.b),D(normed),n_patch,hidden,eps);`
        call("vision::k_layernorm_bf16", stream, |ctx| {
            vision::k_layernorm_bf16(ctx, h, m.norm.g, m.norm.b, normed, n_patch, hidden, eps)
        })?;
        // `:165` — `vd::k_merge_gather<bfd><<<G2(W,n_token),B2,0,S>>>(D(normed),D(grouped),n_token,merge_unit,hidden);`
        call("vision::k_merge_gather_bf16", stream, |ctx| {
            vision::k_merge_gather_bf16(
                ctx,
                normed.cast_const(),
                grouped,
                n_token,
                merge_unit,
                hidden,
            )
        })?;
    }
    // `:171` — the mergers take the ERF gelu.
    mlp(
        cublas,
        grouped.cast_const(),
        mid,
        out,
        &m.fc1,
        &m.fc2,
        n_token,
        width,
        width,
        out_hidden,
        true,
        stream,
    )
}

/// The tower over `num_img` images concatenated row-wise —
/// `run_qwen3vl_vision` (`qwen3_vl_tower.cu:186-303`).
///
/// `pixel` / `rope_pos` / `pos_embed_interp` are `[Ntot, ...]` with
/// `Ntot = Σ n_patch`; the per-row layer kernels run over all `Ntot` rows at
/// once, attention is block-diagonal per image (one FlashInfer multi-sequence
/// prefill), and the mergers loop per image. `out_main` and each `out_deep[d]`
/// are `[Σ NTOK, out_hidden]` with per-image segments. `num_img == 1` is the
/// single-image case and is the only one [`scatter`] reaches today — see its
/// docs for the measurement that turned batching off.
///
/// The multi-image capability is kept because it is this function's real
/// signature: `n_patch_h` is what the attention call consumes to build its
/// block-diagonal plan, and collapsing it to one image would delete the only
/// thing that makes `attn::attend` a multi-sequence call.
///
/// # Errors
///
/// `hidden != heads*head_dim`, a patch count that is not a whole number of
/// merge groups, a refused allocation or a refused launch. Each is the
/// `throw` the C++ made at the same point, as a value — and the first two are
/// the refusals `execution.rs`'s `WALKED` entry quoted.
#[allow(clippy::too_many_arguments)]
fn run(
    st: &mut Res,
    w: &Weights,
    pixel: *const c_void,
    rope_pos: *const c_void,
    pos_embed_interp: *const c_void,
    n_patch_h: &[i32],
    out_main: *mut c_void,
    out_deep: &[*mut c_void],
    cublas: *mut c_void,
    stream: StreamRef<'_>,
) -> Result<()> {
    let (hd, nh, head) = (w.hidden, w.heads, w.head_dim);
    let (im_width, out) = (w.intermediate, w.out_hidden);
    let (eps, theta) = (w.ln_eps, w.rope_theta);
    let unit = w.merge_unit()?;
    let patch_dim = w.patch_dim()?;
    let num_img = n_patch_h.len();

    // Per-image row and token offsets — `:194-195`.
    let mut off = vec![0i32; num_img + 1];
    let mut tok = vec![0i32; num_img + 1];
    for (i, &rows) in n_patch_h.iter().enumerate() {
        off[i + 1] = off[i]
            .checked_add(rows)
            .ok_or_else(|| Error::invalid(WHO, "the batched patch count overflowed"))?;
        tok[i + 1] = tok[i] + rows / unit;
    }
    let n = off[num_img];
    // `:196-197`, both messages verbatim: `execution.rs` cites them as this
    // walk's refusals, and a refusal that changed its words would leave that
    // citation naming nothing.
    if hd != nh * head {
        return Err(Error::invalid(WHO, "hidden != heads*head_dim"));
    }
    if unit == 0 || n % unit != 0 {
        return Err(Error::invalid(WHO, "n_patch not divisible by merge^2"));
    }

    let rows = count("patch rows", n)?;
    let hidden_elems = rows
        .checked_mul(count("hidden", hd)?)
        .ok_or_else(|| Error::invalid(WHO, "N * hidden overflowed"))?;
    let inter_elems = rows
        .checked_mul(count("intermediate", im_width)?)
        .ok_or_else(|| Error::invalid(WHO, "N * intermediate overflowed"))?;
    // `:199-204`, minus `tmp`: the C++ allocated `bf* tmp=MAL((long)N*Hd)`
    // and freed it in the same list without ever naming it again. A dead
    // allocation is not a measurement.
    let h = st.slot(S_H, hidden_elems * 2)?;
    let hn = st.slot(S_HN, hidden_elems * 2)?;
    let qkv = st.slot(S_QKV, hidden_elems * 6)?;
    let q = st.slot(S_Q, hidden_elems * 2)?;
    let k = st.slot(S_K, hidden_elems * 2)?;
    let v = st.slot(S_V, hidden_elems * 2)?;
    let attn_out = st.slot(S_ATTN, hidden_elems * 2)?;
    let mid = st.slot(S_MID, inter_elems * 2)?;

    // `:226` — patch embed: the Conv3d as a matmul `[hidden, PATCH_DIM]`
    // (+bias) over `pixel[N, PATCH_DIM]`.
    gemm_bias(cublas, pixel, &w.patch, h, n, hd, patch_dim, stream)?;
    // `:236` — `vd::k_add_pe<bfd><<<((long)N*Hd+255)/256,256,0,S>>>(D(h),D(pos_embed_interp),(long)N*Hd);`
    // The `pe` operand is the HOST-interpolated table.
    call("vision::k_add_pe_bf16", stream, |ctx| {
        vision::k_add_pe_bf16(ctx, h, pos_embed_interp, hidden_elems)
    })?;

    let mut deep_written = 0usize;
    for (li, layer) in w.blocks.iter().enumerate() {
        // ── attention: norm1 -> qkv -> rope -> attn -> o -> residual ──
        //
        // Fused epilogues, as the C++ had them (`:241-244`): the qkv bias
        // folds into the split, q/k rope share one launch, and the
        // o-projection writes the residual directly (cuBLAS `beta=1`) so
        // `h += attn @ Wo^T` in place — only the o-bias remains as a kernel.
        //
        // `:246` — `vd::k_layernorm<bfd><<<N,256,0,S>>>(D(h),D(L.norm1.g),D(L.norm1.b),D(hn),N,Hd,EPS);`
        call("vision::k_layernorm_bf16", stream, |ctx| {
            vision::k_layernorm_bf16(
                ctx,
                h.cast_const(),
                layer.norm1.g,
                layer.norm1.b,
                hn,
                n,
                hd,
                eps,
            )
        })?;
        // `:247` — cuBLAS, no `<<<>>>`; the bias is NOT applied here, which
        // is why this is `gemm` and not `gemm_bias`.
        gemm(cublas, hn.cast_const(), layer.qkv.w, qkv, n, 3 * hd, hd, 0.0);
        // `:249` —
        // `vd::k_split_rope_qkv<bfd><<<dim3(NH,N),HEAD/2,0,S>>>(D(qkv),D(L.qkv.b),D(q),D(k),D(v),rope_pos,N,NH,HEAD,THETA);`
        //
        // The GRID is one block per (head, row) and the BLOCK is `HEAD/2` —
        // not 128, because widening it would be a performance decision taken
        // on the tower owner's behalf: 96 idle lanes over a 32-wide half,
        // correct and four times the launch.
        call("vision::k_split_rope_qkv_bf16", stream, |ctx| {
            vision::k_split_rope_qkv_bf16(
                ctx,
                qkv.cast_const(),
                layer.qkv.b,
                q,
                k,
                v,
                rope_pos,
                n,
                nh,
                head,
                theta,
            )
        })?;
        // `:254` — full bidirectional attention over each image's patches.
        // The softmax scale is applied INSIDE flashinfer, which is why
        // nothing here scales the scores.
        attn::attend(q.cast_const(), k, v, attn_out, n_patch_h, nh, head, stream)?;
        // `:256` — `beta = 1.0f`: the residual add, fused into the GEMM.
        gemm(cublas, attn_out.cast_const(), layer.o.w, h, n, hd, hd, 1.0);
        if !layer.o.b.is_null() {
            // `:257` — `vd::k_bias<bfd><<<((long)N*Hd+255)/256,256,0,S>>>(D(h),D(L.o.b),(long)N,Hd);`
            call("vision::k_bias_bf16", stream, |ctx| {
                vision::k_bias_bf16(ctx, h, layer.o.b, rows, hd)
            })?;
        }
        // ── mlp: norm2 -> fc1 -> gelu(+bias) -> fc2 -> residual ──
        //
        // `:259` — `vd::k_layernorm<bfd><<<N,256,0,S>>>(D(h),D(L.norm2.g),D(L.norm2.b),D(hn),N,Hd,EPS);`
        call("vision::k_layernorm_bf16", stream, |ctx| {
            vision::k_layernorm_bf16(
                ctx,
                h.cast_const(),
                layer.norm2.g,
                layer.norm2.b,
                hn,
                n,
                hd,
                eps,
            )
        })?;
        // `:260` — cuBLAS; fc1's bias folds into the activation below.
        gemm(cublas, hn.cast_const(), layer.fc1.w, mid, n, im_width, hd, 0.0);
        // `:261` — `vd::k_gelu_bias<bfd><<<((long)N*IM+255)/256,256,0,S>>>(D(mid),D(L.fc1.b),N,IM);`
        // Fired unconditionally, bias or not: the bias is nullable and the
        // kernel adds nothing when it is null.
        call("vision::k_gelu_bias_bf16", stream, |ctx| {
            vision::k_gelu_bias_bf16(ctx, mid, layer.fc1.b, n, im_width)
        })?;
        // `:262` — `beta = 1.0f` again: the second residual, fused.
        gemm(cublas, mid.cast_const(), layer.fc2.w, h, n, hd, im_width, 1.0);
        if !layer.fc2.b.is_null() {
            // `:263` — `vd::k_bias<bfd><<<((long)N*Hd+255)/256,256,0,S>>>(D(h),D(L.fc2.b),(long)N,Hd);`
            call("vision::k_bias_bf16", stream, |ctx| {
                vision::k_bias_bf16(ctx, h, layer.fc2.b, rows, hd)
            })?;
        }
        // ── deepstack tap: post-block, before the next layer, per image ──
        //
        // `:265-275`. The index into `w.deepstack` is `d` and the output goes
        // to `out_deep[deep_written]`, which are different counters when a
        // tap's layer index repeats — reproduced exactly rather than
        // simplified, because the smoke test drives depth-1 weights with
        // three taps all at layer 0 and that is the case they differ in.
        for (d, &at) in w.deepstack_layer_idx.iter().enumerate() {
            if at != i32::try_from(li).unwrap_or(i32::MAX) || deep_written >= out_deep.len() {
                continue;
            }
            let Some(merger) = w.deepstack.get(d) else {
                return Err(Error::invalid(
                    WHO,
                    format!("deepstack tap {d} has no merger behind it"),
                ));
            };
            let merger = *merger;
            for (image, &ni) in n_patch_h.iter().enumerate() {
                let base = count("image row offset", off[image])? * count("hidden", hd)? * 2;
                let dst = count("image token offset", tok[image])? * count("out_hidden", out)? * 2;
                run_merger(
                    st,
                    cublas,
                    &merger,
                    h.cast_const().wrapping_byte_add(base),
                    ni,
                    ni / unit,
                    hd,
                    unit,
                    out,
                    eps,
                    out_deep[deep_written].wrapping_byte_add(dst),
                    stream,
                )?;
            }
            deep_written += 1;
        }
    }
    // `:284-287` — the main merger, per image.
    for (image, &ni) in n_patch_h.iter().enumerate() {
        let base = count("image row offset", off[image])? * count("hidden", hd)? * 2;
        let dst = count("image token offset", tok[image])? * count("out_hidden", out)? * 2;
        let merger = w.merger;
        run_merger(
            st,
            cublas,
            &merger,
            h.cast_const().wrapping_byte_add(base),
            ni,
            ni / unit,
            hd,
            unit,
            out,
            eps,
            out_main.wrapping_byte_add(dst),
            stream,
        )?;
    }
    Ok(())
}

/// Encode every image and scatter its merged tokens into the fire's hidden
/// rows — `scatter_qwen3vl_vision` (`qwen3_vl_tower.cu:392-521`).
///
/// `pixels` is the whole pixel plane as BYTES and `pixel_byte_indptr` cuts it,
/// which is what the plan carries; the C++ took a `const float*` and divided
/// the offsets by four, naming a type it never dereferenced on the host.
/// `grids` is `[t, h, w]` per image, `anchor_rows` the destination row of each
/// image's first merged token.
///
/// `out_hidden` is not a parameter: the C++ entry took it twice — once as
/// `scatter_qwen3vl_vision`'s argument and once on the weights struct — and
/// `qwen3_vl_tower_c.cpp` set both from the same value. [`Weights::out_hidden`]
/// is the single source now.
///
/// # The batched arm is DEAD, and its measurement is the reason
///
/// `qwen3_vl_tower.cu:443-450` carried a `constexpr bool uniform = false` and
/// `const int g0t=0,g0h=0,g0w=0,np0=0` — a whole second arm, unreachable AND
/// unrunnable if reached (a zero grid interpolates a zero-row table). It is
/// not ported. What is ported is why it is off:
///
/// > Uniform batch test: all images share a grid and are page-aligned →
/// > encode them in ONE batched tower pass (bigger GEMMs, per-image
/// > block-diag attention). **OFF by default: measured ~6% SLOWER here — the
/// > per-image vision GEMMs are already compute-efficient at `M=n_patch`, so
/// > batching to `M=Σn_patch` gives no GEMM win while the multi-seq attention
/// > + larger buffers add overhead.**
///
/// [`run`] keeps its multi-image signature so that measurement can be re-taken
/// without rewriting the walk; what is gone is a dead branch that could not
/// have taken it.
///
/// # No synchronise, and that is also a measurement
///
/// `:513-517`: *"No sync here: all work is stream-ordered on S; the decoder
/// layers that follow (same stream) see the scattered embeddings, and the
/// fire's final sync settles the async pixel H2D before the host pixel buffer
/// is reused. Draining mid-forward only stalled the CPU from queuing the LLM
/// layers."* This function drains exactly once and only on the first forward
/// after a weight publish, inside [`Res::table`]; see [`Res`] for why the
/// arena is persistent rather than a [`super::Scratch`], which is the same
/// measurement read a second time.
///
/// # Errors
///
/// A pixel span that leaves the payload, a grid or anchor table shorter than
/// the image count, a refused allocation, or any refused launch.
#[allow(clippy::too_many_arguments)]
pub fn scatter(
    w: &Weights,
    pixels: &[u8],
    pixel_byte_indptr: &[u32],
    grids: &[u32],
    anchor_rows: &[u32],
    hidden_rows: *mut c_void,
    n_rows: i32,
    deepstack_scratch: *mut c_void,
    num_deep: i32,
    cublas: *mut c_void,
    stream: StreamRef<'_>,
) -> Result<()> {
    // `:396` — `if(vin.weights==nullptr || vin.num_images<=0) return;`
    let num_images = anchor_rows.len();
    if num_images == 0 || w.blocks.is_empty() {
        return Ok(());
    }
    if pixel_byte_indptr.len() < num_images + 1 || grids.len() < 3 * num_images {
        return Err(Error::invalid(
            WHO,
            format!(
                "{num_images} images need {} indptr entries and {} grid values, not {} and {}",
                num_images + 1,
                3 * num_images,
                pixel_byte_indptr.len(),
                grids.len()
            ),
        ));
    }
    let out = w.out_hidden;
    let unit = w.merge_unit()?;
    let patch_dim = count("patch_dim", w.patch_dim()?)?;
    let out_bytes = count("out_hidden", out)?
        .checked_mul(2)
        .ok_or_else(|| Error::invalid(WHO, "out_hidden overflowed"))?;
    let deep = count("num_deep", num_deep)?;

    // `:397-401` — zero the FULL deepstack scratch so the decoder can add it
    // into hidden as a plain whole-tensor residual: non-image rows contribute
    // zero, which mirrors HF `_deepstack_process` updating only the visual
    // rows.
    if !deepstack_scratch.is_null() && deep > 0 {
        let bytes = deep
            .checked_mul(count("n_rows", n_rows)?)
            .and_then(|v| v.checked_mul(out_bytes))
            .ok_or_else(|| Error::invalid(WHO, "the deepstack scratch size overflowed"))?;
        // SAFETY: the fire owns `deepstack_scratch` as `[num_deep, n_rows,
        // out_hidden]` bf16 and passed both extents; the fill is exactly that
        // many bytes and is ordered on the walk's stream.
        unsafe { fill_raw_span(deepstack_scratch, 0, bytes, stream)? };
    }

    let mut st = res().lock().unwrap_or_else(|e| e.into_inner());
    let st = &mut *st;
    // THE LOCK IS HELD FOR THE WHOLE WALK, where the C++ took its two mutexes
    // only around their caches and relied on the engine serializing forward
    // passes for one model. The arena is what changed the requirement: slabs
    // reused across calls cannot be shared by two concurrent walks, so the
    // serialization the C++ ASSUMED is now enforced. A second caller waits;
    // it does not interleave into `h`.
    st.table(w, stream)?;

    // `:485-511` — the per-image arm, which is the live one.
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
        let n_patch = n_floats / patch_dim;
        // `:490` — `if(n_patch<=0) continue;`
        if n_patch == 0 {
            continue;
        }
        let n_patch_i = i32::try_from(n_patch)
            .map_err(|_| Error::invalid(WHO, "an image's patch count overflowed an int"))?;
        let key = (
            i32::try_from(grids[3 * image]).unwrap_or(i32::MAX),
            i32::try_from(grids[3 * image + 1]).unwrap_or(i32::MAX),
            i32::try_from(grids[3 * image + 2]).unwrap_or(i32::MAX),
        );
        let n_token = n_patch / count("merge unit", unit)?;
        let anchor = anchor_rows[image] as usize;

        // `:495-497` — the pixel plane, uploaded then cast.
        let pix_f32 = st.upload(S_PIX_F32, &pixels[blo..bhi], stream)?;
        let pix_bf = st.slot(S_PIX_BF, n_floats * 2)?;
        // `:498` (and `:459`, the dead batched arm's identical launch) —
        // `vd::k_f32_to_bf16<bfd><<<(n_floats+255)/256,256,0,S>>>(pix_f32_d,D(pix_bf_d),n_floats);`
        // The SOURCE is float whatever the destination's element type is.
        call("vision::k_f32_to_bf16_bf16", stream, |ctx| {
            vision::k_f32_to_bf16_bf16(ctx, pix_f32.cast_const(), pix_bf, n_floats)
        })?;
        // `:499` — the cached rope ids and interpolated position embeddings.
        let (rope_d, pe_d) = st.grid(w, key, stream)?;
        let token_bytes = n_token
            .checked_mul(out_bytes)
            .ok_or_else(|| Error::invalid(WHO, "an image's output size overflowed"))?;
        let main_d = st.slot(S_MAIN, token_bytes)?;
        let mut deep_d = Vec::with_capacity(deep);
        for d in 0..deep {
            deep_d.push(st.slot(S_DEEP + d, token_bytes)?);
        }
        // `:504`.
        run(
            st,
            w,
            pix_bf.cast_const(),
            rope_d.cast_const(),
            pe_d.cast_const(),
            &[n_patch_i],
            main_d,
            &deep_d,
            cublas,
            stream,
        )?;
        // `:505-509` — the scatter proper: the merged tokens into the fire's
        // hidden rows at the anchor, and each deepstack merger's into its own
        // plane of the scratch.
        let at = anchor
            .checked_mul(out_bytes)
            .ok_or_else(|| Error::invalid(WHO, "an anchor offset overflowed"))?;
        // SAFETY: `hidden_rows` is `[n_rows, out_hidden]` bf16 and the caller
        // stated both extents; `anchor + n_token <= n_rows` is the plan's
        // invariant, the same one the C++ wrote through. `main_d` is this
        // walk's arena slot, sized to exactly `token_bytes` above, and the
        // two spans are distinct allocations.
        unsafe {
            copy_raw_span(
                hidden_rows.wrapping_byte_add(at),
                main_d.cast_const(),
                token_bytes,
                stream,
            )?;
        }
        if deepstack_scratch.is_null() {
            continue;
        }
        for (d, src) in deep_d.iter().enumerate() {
            let plane = d
                .checked_mul(count("n_rows", n_rows)?)
                .and_then(|v| v.checked_mul(out_bytes))
                .and_then(|v| v.checked_add(at))
                .ok_or_else(|| Error::invalid(WHO, "a deepstack offset overflowed"))?;
            // SAFETY: as above, into plane `d` of `[num_deep, n_rows,
            // out_hidden]` — the buffer this function memset whole at entry.
            unsafe {
                copy_raw_span(
                    deepstack_scratch.wrapping_byte_add(plane),
                    src.cast_const(),
                    token_bytes,
                    stream,
                )?;
            }
        }
    }
    Ok(())
}
