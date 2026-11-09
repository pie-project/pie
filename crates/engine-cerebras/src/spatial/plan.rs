//! The plan layer: from a model ([`super::model::ModelSpec`]) to the wafer
//! program's compile parameters (`PLAN.md` beside this file).
//!
//! [`SpatialPlan::derive`] picks a kernel family and computes every
//! parameter of its build from the model's dims and the wafer: the grid
//! shape and per-PE splits (`PX PY Hc Dc Gs Qh Ic`), the batch and pool
//! (`B NB MC PS E`), the grids' placement (`GW GH`) and the head's layout
//! (`HH Vr F Gx Gy`). Per-PE memory is an estimate fitted to the image ends
//! the builds reported; the link stays the final word. For Qwen3-0.6B and
//! Qwen3-8B it gives the parameters of the builds validated on CS-3.

use super::model::refuse;
use super::model::{ModelSpec, Need, Refused, out_json_params};

/// The fabric a program is laid out on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaferDims {
    /// The usable rectangle (the fabric less the memcpy frame: 7 columns, 2 rows).
    pub width: usize,
    pub height: usize,
    /// A PE's memory: the image must end below it.
    pub pe_bytes: usize,
}

impl WaferDims {
    /// CS-3: a 762 x 1172 fabric at offsets (4, 1); 48 KB a PE.
    pub const CS3: WaferDims = WaferDims {
        width: 755,
        height: 1170,
        pe_bytes: 0xc000,
    };
}

/// What a build is for; the rest follows from the model and the wafer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Goals {
    /// Lanes a pass (`B`).
    pub lanes: usize,
    /// The lanes the grid's shape must hold (rows are added until it does):
    /// one grid shape serves every build up to it.
    pub grid_lanes: usize,
    /// KV page size (`PS`): PIE's page size.
    pub page_size: usize,
    /// Pool blocks a PE (`NB`), pinned; `None`: the largest power of two up
    /// to `max_blocks` that fits.
    pub blocks: Option<usize>,
    pub max_blocks: usize,
    /// Lanes an MLP chunk (`MC`), pinned; `None`: two where the batch
    /// splits evenly in f32 builds (serve2d's choice), one in bf16 ones (its
    /// buffer costs about two lanes there).
    pub mlp_chunk: Option<usize>,
    /// Forced/output tokens a lane a launch (`CAP`); `None`: serve2d's 64,
    /// halved until the head's token buffers fit, or 16 in a pipeline.
    pub cap: Option<usize>,
    /// Pipeline the layers (serve2d_pipe): micro-batches of B go round.
    pub pipeline: bool,
    /// Copy pages on the host (`KVCP` 0): no copy kernel, ~0.9 KB a PE back.
    pub host_copy: bool,
    /// A paged8b grid's rows (`PY`), pinned (a multiple of 4); `None`: the
    /// fewest in steps of 16 that hold `grid_lanes`. serve8b_region at
    /// Llama 3.1 8B's inter wants 152 (the 160 the steps give is 6 columns
    /// too wide for its head and strip).
    pub rows: Option<usize>,
}

impl Default for Goals {
    fn default() -> Goals {
        Goals {
            lanes: 4,
            grid_lanes: 8,
            page_size: 16,
            blocks: None,
            max_blocks: 8,
            mlp_chunk: None,
            cap: None,
            pipeline: false,
            host_copy: false,
            rows: None,
        }
    }
}

impl Goals {
    pub fn lanes(lanes: usize) -> Goals {
        Goals {
            lanes,
            grid_lanes: lanes.max(8),
            ..Goals::default()
        }
    }
}

/// The kernel family a plan builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// `serve2d` GRID 1: `qwen2d_paged` grids (f32, square, two
    /// query heads a kv head, hidden unpadded) and head2d beside them.
    Serve2d,
    /// `serve2d_pipe`: the same grids as `qwen2d_pring` ring stages.
    Serve2dPipe,
    /// `qwen2d_paged8b`: one layer grid, bf16 weights and KV, any GQA ratio,
    /// rectangular and hidden-padded. No full program places it yet.
    Paged8b,
}

/// One layer's grid: PX columns (kv dims, MLP rows) x PY rows (hidden).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    pub px: usize,
    pub py: usize,
    /// Hidden dims a row (`hidden` zero-padded to `py * hc`).
    pub hc: usize,
    pub hidden: usize,
    /// Head dims a column, `gs` columns a kv head.
    pub dc: usize,
    pub gs: usize,
    pub qh: usize,
    /// MLP rows a column (`inter` zero-padded to `px * ic`).
    pub ic: usize,
    pub lanes: usize,
    pub page_size: usize,
    pub blocks: usize,
    pub entries: usize,
    pub mlp_chunk: usize,
    /// bf16 weights (`W16`) and KV (`KV16`).
    pub w16: bool,
    pub kv16: bool,
    pub host_copy: bool,
    /// The projections stay u4g64 in PE memory (`W4`, `spatial::u4`; on
    /// the bf16 units only).
    pub w4: bool,
}

impl Grid {
    /// The KV pool a layer: pages of `page_size`.
    pub fn pages(&self) -> usize {
        self.blocks * self.py
    }

    /// A lane's most positions: E pages a row, below the RoPE tables' 4096.
    pub fn max_context(&self) -> usize {
        (self.entries * self.py * self.page_size).min(ROPE_POSITIONS)
    }
}

/// The LM head (head2d): `blocks` blocks of `c` chunk columns (plus the VM
/// column each) and `hh` rows of `vr` vocabulary rows, feeder row `hh / 2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Head {
    pub c: usize,
    pub hc: usize,
    pub hh: usize,
    pub vr: usize,
    pub gx: usize,
    pub gy: usize,
    pub gb: usize,
    pub blocks: usize,
    pub vmc: usize,
    pub untied: bool,
    /// f32 tiles, exact (0.6B); else bf16 tiles, one pass, packed-key argmax (8B).
    pub f32: bool,
}

impl Head {
    pub fn width(&self) -> usize {
        self.blocks * (self.c + self.vmc)
    }
}

/// The full program: the grids' placement and the head beside them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Program {
    /// Grids across and down, serpentine; layer 0 top right.
    pub gw: usize,
    pub gh: usize,
    pub cap: usize,
    /// serve2d_pipe: micro-batches a launch and lanes a head step.
    pub mmax: Option<usize>,
    pub hb: Option<usize>,
    /// The rectangle (grids + head).
    pub width: usize,
    pub height: usize,
}

/// A model on the wafer: the family, its grid, the head, the program when
/// one places the grids, and why not when none does.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialPlan {
    pub spec: ModelSpec,
    pub family: Family,
    pub layers: usize,
    pub vocab: usize,
    pub grid: Grid,
    pub head: Head,
    pub program: Option<Program>,
    /// A grid PE's estimated bytes, of `pe_bytes`.
    pub grid_bytes: usize,
    pub pe_bytes: usize,
    pub needs: Vec<Need>,
    /// Why no full program places the model (empty when `program` is set).
    pub blockers: Vec<String>,
}

// ---- What the kernels fix ---------------------------------------------------

/// Head dims a column: RoPE pairs Dc/2 dims from each half of a head.
const DC: usize = 8;
/// One page entry a lane a row (every build so far).
const ENTRIES: usize = 1;
/// RoPE angles from three 16-entry tables: positions below 16^3.
pub(super) const ROPE_POSITIONS: usize = 4096;
/// `Tc` (a single sequence's KV slots a row): unused by GRID 1 builds,
/// carried at serve2d's value.
pub(crate) const TC: usize = 32;
/// head2d's vocabulary rows a PE are a multiple of 32 (the VM's sums are
/// exact only then).
const VR_STEP: usize = 32;
/// The bf16 head (8B, head2d v10): 256 region rows, 32 hidden dims a PE, at
/// most ~19.5 KB of tables a PE beside its buffers and code.
const HEAD16_ROWS: usize = 256;
const HEAD16_HC: usize = 32;
pub(super) const HEAD16_TABLES: usize = 19_500;
/// The f32 one-block head beside serve2d's grids: tables a PE, at most.
const HEAD32_TABLES: usize = 16_384;
/// serve2d: `in_tokens` + `out_tokens` at the root take B x CAP words each,
/// 4 KB together at most (B 16 at CAP 64 ran out); the pipe's take MMAX x B
/// x CAP, 1024 words at most. Pipelines run up to 8 micro-batches, head
/// steps of up to 8 lanes.
pub(super) const TOKENS_SERVE: usize = 512;
const TOKENS_PIPE: usize = 1024;
const MMAX: usize = 8;
const HEAD_BATCH: usize = 8;

// ---- Per-PE memory, fitted to the builds' image ends ------------------------
//
// A grid PE holds code and runtime, its weight tiles, the norm/RoPE tables,
// the KV pool (NB blocks of Dc x PS keys and values), a lane's buffers per
// lane and the MLP chunk's buffers. The weights, tables and pool are exact;
// the code and the lane's scratch are fitted:
// - serve2d (f32): image ends B 8 NB 8 MC 2 = 0xb180, B 16 NB 4 MC 2 =
//   0xb790, B 16 NB 8 does not link (model: 51.6 KB);
// - qwen2d_paged8b (bf16): B 1 NB 4 = 0xb490; the model
//   is within 0.1 KB of every other build in that table and fails exactly
//   the two that did not link (B 8 NB 2, B 9 NB 1; with KVCP 0, B 8 NB 4).

const SERVE2D_FIXED: usize = 24_984;
const SERVE2D_LANE_MISC: usize = 50;
const PAGED8B_FIXED: usize = 21_404;
/// The page-copy kernel (paged8b: 18.4 KB of code with it, 17.5 KB without).
const COPY_CODE: usize = 922;
/// A ring stage's handoff code (qwen2d_pring over qwen2d_paged at B 8:
/// 0x3e0 more, of which the received `[x | tab]` is 416 B).
const RING_CODE: usize = 576;

impl Grid {
    pub(super) fn weight_bytes(&self) -> usize {
        if self.w4 {
            // The six u4 tiles (codes, groups' scales and biases, segments)
            // and the decode scratch (`u4::pe_bytes`).
            return super::u4::pe_bytes(
                self.hc,
                self.dc,
                self.gs,
                self.qh,
                self.ic,
                1,
                self.lanes == 1,
            );
        }
        // wqkv Hc x (Qh + 2) Dc, wo Qh Dc x Hc, wgu Hc x 2 Ic, wd Ic x Hc.
        let elems = self.hc * ((2 * self.qh + 2) * self.dc + 3 * self.ic);
        elems * if self.w16 { 2 } else { 4 }
    }

    fn table_bytes(&self) -> usize {
        // gamma1/2 [Hc], qkn_w [2][Qh + 1][DH], rot_c/rot_s [3][16][DH].
        let dh = self.dc / 2;
        4 * (2 * self.hc + 2 * (self.qh + 1) * dh + 2 * 3 * 16 * dh)
    }

    pub(super) fn block_bytes(&self) -> usize {
        2 * self.dc * self.page_size * if self.kv16 { 2 } else { 4 }
    }

    fn lane_bytes(&self) -> usize {
        let (tab, qkv) = (2 + 3 * self.entries, (self.qh + 2) * self.dc + 1);
        let scores = self.qh * self.entries * self.page_size;
        if self.w16 {
            // paged8b: x, its table, QKV, scores (buffers shared by phase).
            4 * (self.hc + tab + qkv + scores)
        } else {
            // qwen2d_paged in serve2d: x and the handoff's x, the table, QKV,
            // scores and their scratch, RoPE rows, softmax and misc.
            let rope = 2 * (self.qh + 1) * (self.dc / 2);
            4 * (2 * self.hc + tab + qkv + 2 * scores + rope + SERVE2D_LANE_MISC)
        }
    }

    /// A PE's estimated bytes.
    /// The fixed bytes [`Grid::bytes`] counts (code and fixed data).
    pub fn fixed_bytes(&self) -> usize {
        let copy = if self.host_copy { COPY_CODE } else { 0 };
        (if self.w16 {
            PAGED8B_FIXED
        } else {
            SERVE2D_FIXED
        }) - copy
    }

    pub fn bytes(&self, ring: bool) -> usize {
        let fixed = if self.w16 {
            PAGED8B_FIXED
        } else {
            SERVE2D_FIXED
        };
        let copy = if self.host_copy { COPY_CODE } else { 0 };
        let ring = if ring {
            RING_CODE + 4 * self.lanes * (self.hc + 2 + 3 * self.entries)
        } else {
            0
        };
        let w4 = if self.w4 { W4_CODE } else { 0 };
        fixed - copy
            + w4
            + ring
            + self.weight_bytes()
            + self.table_bytes()
            + self.blocks * self.block_bytes()
            + self.lanes * self.lane_bytes()
            + 4 * self.mlp_chunk * (3 * self.ic + 1)
    }
}

impl Grid {
    /// One more layer's bytes on a PE holding several (a region's): its
    /// weights (without the u4 decode scratch, shared), tables and pool.
    pub fn layer_bytes(&self) -> usize {
        let weights = if self.w4 {
            super::u4::tiles(self.hc, self.dc, self.gs, self.qh, self.ic, self.lanes == 1)
                .iter()
                .map(|t| 2 * t.layer_words())
                .sum()
        } else {
            self.weight_bytes()
        };
        weights + self.table_bytes() + self.blocks * self.block_bytes()
    }

    /// A region PE's estimated bytes at `nl` layers (qwen2d_region: the
    /// grid kernel's model, `REGION_EXTRA` more code, the layers past the
    /// first by [`Grid::layer_bytes`]).
    pub fn region_bytes(&self, nl: usize, ring: bool) -> usize {
        self.bytes(ring) + REGION_EXTRA + nl.saturating_sub(1) * self.layer_bytes()
    }
}

/// qwen2d_region's image over the grid model at the same tiles (link probe,
/// 2026-11-04: Qwen3-14B u4 regions 256 x 144, RHc 36, RDc 4, RIc 68, RNL 2:
/// 47,488 B against 43,510 + `W4_CODE` + staging).
pub const REGION_EXTRA: usize = 1_314;
/// A W4 unit image over the grid model (the u4 matvec's code and what the
/// model does not count; link probe, 2026-11-04: the Qwen3-14B u4 grid
/// 128 x 144, Hc 36, Ic 136: 46,048 B against 43,496 + staging).
pub const W4_CODE: usize = 1_688;
/// KV page staging (`KVST` 1, B 1 builds without a ring): a unit PE's
/// bytes, and each layer past a region's first (link probe, 2026-11-04,
/// Qwen3-14B u4 B 1: KVST 1 against 0, grid 864 B, RNL 2 region 976 B).
pub const KVST_BYTES: usize = 864;
pub const KVST_LAYER_BYTES: usize = 112;

/// serve8b_region's head PE (`head_pe.csl`, HEADMODE 2, MODE 0, untied, no
/// VM block) at `vr` vocab rows and the grid's `hc` (the head holds 2 `hc`
/// dims a row): the bf16 rows, the per-row f32 vectors, the max-by-halving
/// scratch (`VP`, `vr` padded to a power of two), the kept logits rows
/// (`hlb` lanes) and, on a ring of `mmax` micro-batches, the pipe's code and
/// per-lane state. Link probes (2026-11-04, Qwen3-14B u4 pipe x12, B 1,
/// `vr` 64-140, `hc` 29-36: this bounds every probe, at most ~370 B high).
/// With `u4` (`W4H` 1) the bf16 rows are one u4g64 tile ([`head_u4_bytes`]).
#[must_use]
pub fn head_bytes(
    vr: usize,
    hc: usize,
    lanes: usize,
    hlb: usize,
    mmax: Option<usize>,
    u4: bool,
) -> usize {
    let vp = vr.next_power_of_two().max(32);
    let pipe = mmax.map_or(0, |m| HEAD_PIPE + 4 * m * lanes);
    // Per-row f32 vectors: 20 B a row bf16; 14 B a row fits the u4 probes.
    let rows = if u4 {
        head_u4_bytes(vr, hc) + vr * 14
    } else {
        vr * (4 * hc + 20)
    };
    HEAD_FIXED + rows + 60 * hc + 8 * vp + 4 * vr * hlb.saturating_sub(1) + pipe
}

/// A `W4H` head PE's u4 part (`codegen::w4_head`): the tile (`2 hc` dims
/// by `vr` rows, a pad), the u4 scratch at NP 3 (decode buffer,
/// accumulator, widened scales, x's parts, `wmv4`'s f32 scratch, the
/// header copy) and `HEAD_W4_CODE`. Link probes (2026-11-04, B 1 heads
/// with their final root's tokens, W4H 1): Vr 132, 72 dims, CAP 64: 39,392
/// B (bf16: 46,256); Vr 228, 80 dims, CAP 32: 47,280 B; pipe x12 Vr 228:
/// head 47,424 B, final root 48,944 B at CAP 16. The model is within 50 B
/// above each (the per-row term is 14 B on u4 heads).
#[must_use]
pub fn head_u4_bytes(vr: usize, hc: usize) -> usize {
    let t = super::u4::head_tile(2 * hc, vr);
    let np = 3;
    4 * super::u4::head_words(t)
        + 2 * (t.chunk() * (t.n + 1) + 1)
        + 4 * (t.n + 1)
        + 4 * t.n
        + 2 * 3 * np * t.r
        + 8 * t.r
        + 2 * t.header()
        + HEAD_W4_CODE
}
pub const HEAD_W4_CODE: usize = 1_140;

/// The embedding strip's PE (`embed_pe.csl`): `ve` vocab rows of `4 hc`
/// dims, bf16 or (`u4`, `W4H` 1) [`super::u4::embed_layout`]'s rows, and
/// its f32 row and lanes. Link probes (2026-11-04, Qwen3-14B u4 B 1, Ve
/// 132, Hc 36): 47,184 B bf16, 21,664 B u4; Ve 228, Hc 40: 32,112 B u4. The planner did not check this
/// image before: at 200k+ vocabularies it overflows like the head's.
#[must_use]
pub fn embed_bytes(ve: usize, hc: usize, lanes: usize, u4: bool) -> usize {
    let he = 4 * hc;
    let rows = if u4 {
        let (_, rw, n32) = super::u4::embed_layout(he, ve);
        4 * n32 + 2 * rw + EMBED_W4_CODE
    } else {
        2 * ve * he
    };
    EMBED_FIXED + 4 * he * (lanes + 1) + rows
}
pub const EMBED_FIXED: usize = 8_016;
pub const EMBED_W4_CODE: usize = 828;

/// The final root's head PE: [`head_bytes`] and its forced and argmax token
/// rows (`in_tokens`, `out_tokens`: `mmax` x `lanes` x `cap` u32 each; probe:
/// the Qwen3-14B pipe x12 final root at CAP 32, 49,408 B, overflowed the
/// PE, which no grid, region or plain head probe showed; 0427, 0428).
#[must_use]
pub fn final_root_bytes(
    vr: usize,
    hc: usize,
    lanes: usize,
    hlb: usize,
    mmax: Option<usize>,
    cap: usize,
    u4: bool,
) -> usize {
    head_bytes(vr, hc, lanes, hlb, mmax, u4) + 8 * mmax.unwrap_or(1) * lanes * cap
}

/// [`head_bytes`]' constant (code, runtime and the small buffers) and the
/// ring's part over one shape's B 1 head (45,808 B unpiped, 46,320 at MMAX 12).
pub const HEAD_FIXED: usize = 20_288;
pub const HEAD_PIPE: usize = 464;

impl Grid {
    /// The KV staging slots' bytes on a unit PE of `nl` layers.
    pub fn staging_bytes(nl: usize) -> usize {
        KVST_BYTES + nl.saturating_sub(1) * KVST_LAYER_BYTES
    }
}

/// The largest even divisor of `n` at most `cap` (a two-phase all-reduce's
/// phase-1 group: 8 and 16 cost about the same, 32 more, as measured).
fn group(n: usize, cap: usize) -> Option<usize> {
    (2..=cap.min(n))
        .rev()
        .find(|g| g % 2 == 0 && n.is_multiple_of(*g))
}

impl SpatialPlan {
    /// The plan for `spec` on `wafer`, or why no grid holds it.
    pub fn derive(
        spec: &ModelSpec,
        wafer: WaferDims,
        goals: Goals,
    ) -> Result<SpatialPlan, Refused> {
        let name = &spec.name;
        if spec.hybrid.is_some() {
            return refuse(format!(
                "{name}: Gated DeltaNet layers (plan it with plan_q35::Q35Plan)"
            ));
        }
        if !spec.head_dim.is_multiple_of(DC) || !(spec.head_dim / DC).is_multiple_of(2) {
            return refuse(format!(
                "{name}: head_dim {} does not split into an even number of {DC}-dim column slices",
                spec.head_dim
            ));
        }
        if goals.lanes == 0 || goals.page_size == 0 {
            return refuse("a plan wants at least one lane and a page size");
        }
        let gs = spec.head_dim / DC;
        let px = spec.kv_heads * gs;
        if px > wafer.width {
            return refuse(format!(
                "{name}: {px} columns a grid (kv heads x {gs}) exceed the wafer"
            ));
        }
        let qh = spec.group();
        let mut why_not_serve2d = Vec::new();
        if qh != 2 {
            why_not_serve2d.push(format!("{qh} query heads a kv head (qwen2d_paged runs 2)"));
        }
        if !spec.hidden.is_multiple_of(px) || !spec.inter.is_multiple_of(px) {
            why_not_serve2d.push(format!(
                "hidden {} or inter {} does not split over {px} square rows/columns",
                spec.hidden, spec.inter
            ));
        }
        if spec.weights == poem_ir::Dtype::U4g64 {
            why_not_serve2d.push("u4g64 weights run on the bf16 units (W4) only".into());
        }
        let pipeline = goals.pipeline;
        if why_not_serve2d.is_empty() {
            let grid = Self::grid(spec, px, px, false, &goals);
            let bytes = grid.bytes(pipeline);
            if bytes <= wafer.pe_bytes {
                return Self::serve2d(spec, wafer, goals, grid);
            }
            why_not_serve2d.push(format!(
                "the f32 grid wants {bytes} bytes a PE (of {}) at B {} NB {}",
                wafer.pe_bytes, grid.lanes, grid.blocks
            ));
        }
        let mut plan = Self::paged8b(spec, wafer, goals, px)?;
        plan.blockers.insert(
            0,
            format!(
                "serve2d (the only full program) wants square f32 grids: {}",
                why_not_serve2d.join("; ")
            ),
        );
        Ok(plan)
    }

    /// A grid of `px` x `py` at `goals`' batch, its pool deepest that fits
    /// unless pinned.
    fn grid(spec: &ModelSpec, px: usize, py: usize, w16: bool, goals: &Goals) -> Grid {
        let lanes = goals.lanes;
        let mlp_chunk = goals
            .mlp_chunk
            .unwrap_or(if !w16 && lanes.is_multiple_of(2) {
                2
            } else {
                1
            });
        let mut grid = Grid {
            px,
            py,
            hc: spec.hidden.div_ceil(py),
            hidden: spec.hidden,
            dc: DC,
            gs: spec.head_dim / DC,
            qh: spec.group(),
            ic: spec.inter.div_ceil(px),
            lanes,
            page_size: goals.page_size,
            blocks: goals.blocks.unwrap_or(1),
            entries: ENTRIES,
            mlp_chunk,
            w16,
            kv16: w16,
            host_copy: goals.host_copy,
            w4: w16 && spec.weights == poem_ir::Dtype::U4g64,
        };
        if goals.blocks.is_none() {
            let mut nb = 1;
            while nb * 2 <= goals.max_blocks
                && (Grid {
                    blocks: nb * 2,
                    ..grid
                })
                .bytes(goals.pipeline)
                    <= WaferDims::CS3.pe_bytes
            {
                nb *= 2;
            }
            grid.blocks = nb;
        }
        grid
    }

    /// serve2d / serve2d_pipe: the grids serpentine, head2d east of them.
    fn serve2d(
        spec: &ModelSpec,
        wafer: WaferDims,
        goals: Goals,
        grid: Grid,
    ) -> Result<SpatialPlan, Refused> {
        let (p, layers, vmc) = (grid.py, spec.layers, 1);
        let mut blockers = Vec::new();
        let mut best: Option<(Program, Head)> = None;
        // Every split of the layers into GW x GH (GW >= 2; the feeder row
        // between layer 0's next strip and the last: GH >= 4) whose grids
        // and one-block head fit; the smallest rectangle wins.
        for gw in 2..=layers {
            if layers % gw != 0 {
                continue;
            }
            let gh = layers / gw;
            let (width, height) = (gw * p + p + vmc, gh * p);
            if gh < 4 || width > wafer.width || height > wafer.height {
                continue;
            }
            let Some(head) = Self::head32(spec, &grid, height, gh) else {
                continue;
            };
            if best.is_none_or(|(b, _)| width * height < b.width * b.height) {
                let program = Program {
                    gw,
                    gh,
                    cap: 0,
                    mmax: None,
                    hb: None,
                    width,
                    height,
                };
                best = Some((program, head));
            }
        }
        let head = match best {
            Some((_, head)) => head,
            None => {
                blockers.push(format!(
                    "no GW x GH = {layers} split of {p} x {p} grids (GW >= 2, GH >= 4) with the \
                     head beside them fits {} x {}",
                    wafer.width, wafer.height
                ));
                Self::head32(spec, &grid, usize::MAX, usize::MAX).unwrap_or(Head {
                    c: p,
                    hc: grid.hc,
                    hh: 0,
                    vr: 0,
                    gx: 0,
                    gy: 0,
                    gb: 1,
                    blocks: 1,
                    vmc,
                    untied: !spec.tied,
                    f32: true,
                })
            }
        };
        let b = grid.lanes;
        let program = best.map(|(mut program, _)| {
            if goals.pipeline {
                let cap = goals.cap.unwrap_or(16);
                program.cap = cap;
                program.mmax = Some((TOKENS_PIPE / (b * cap)).clamp(1, MMAX));
                program.hb = Some(
                    (1..=HEAD_BATCH.min(b))
                        .rev()
                        .find(|h| b.is_multiple_of(*h))
                        .unwrap_or(1),
                );
            } else {
                let mut cap = goals.cap.unwrap_or(64);
                while b * cap > TOKENS_SERVE && cap > 2 {
                    cap /= 2;
                }
                program.cap = cap;
            }
            program
        });
        Ok(SpatialPlan {
            spec: spec.clone(),
            family: if goals.pipeline {
                Family::Serve2dPipe
            } else {
                Family::Serve2d
            },
            layers,
            vocab: spec.vocab,
            grid,
            head,
            program,
            grid_bytes: grid.bytes(goals.pipeline),
            pe_bytes: wafer.pe_bytes,
            needs: spec.needs(),
            blockers,
        })
    }

    /// serve2d's head: one block of C = PY chunk columns, f32 tiles, no
    /// taller than the grids, its feeder row (HH / 2) below layer 0's strip
    /// and the next and above the last. The fewest vocabulary rows a PE
    /// (multiples of 32) that fit.
    pub(super) fn head32(spec: &ModelSpec, grid: &Grid, height: usize, gh: usize) -> Option<Head> {
        Self::head_s2d(spec, grid, height, gh, true)
    }

    /// [`Self::head32`]'s head with f32 or (`!f32`) bf16 tiles, in the same
    /// bytes of tables a PE.
    pub(super) fn head_s2d(
        spec: &ModelSpec,
        grid: &Grid,
        height: usize,
        gh: usize,
        f32: bool,
    ) -> Option<Head> {
        let p = grid.py;
        let tables = if spec.tied { 1 } else { 2 };
        let el = if f32 { 4 } else { 2 };
        (1..)
            .map(|k| k * VR_STEP)
            .take_while(|vr| tables * el * grid.hc * vr <= HEAD32_TABLES)
            .find_map(|vr| {
                let hh = spec.vocab.div_ceil(vr).next_multiple_of(2);
                let f = hh / 2;
                let fits =
                    hh < height && f > 2 * p - 1 && f < (gh.saturating_sub(1)).saturating_mul(p);
                let (gx, gy) = (group(p, 16)?, group(f, 18)?);
                fits.then_some(Head {
                    c: p,
                    hc: grid.hc,
                    hh,
                    vr,
                    gx,
                    gy,
                    gb: 1,
                    blocks: 1,
                    vmc: 1,
                    untied: !spec.tied,
                    f32,
                })
            })
    }

    /// qwen2d_paged8b: rows added (in steps of 16) until `grid_lanes` lanes
    /// with one pool block fit a PE; then the build's batch and pool.
    fn paged8b(
        spec: &ModelSpec,
        wafer: WaferDims,
        goals: Goals,
        px: usize,
    ) -> Result<SpatialPlan, Refused> {
        let name = &spec.name;
        let sizing = Goals {
            lanes: goals.grid_lanes.max(goals.lanes),
            blocks: Some(1),
            ..goals
        };
        // The fewest rows (steps of 16) whose grid fits (a u4g64 model's by
        // its u4 bytes: fewer rows a layer, so more layers on the wafer),
        // or the rows pinned.
        let py = match goals.rows {
            Some(py) if py == 0 || !py.is_multiple_of(4) || py > wafer.height => {
                return refuse(format!(
                    "{name}: {py} grid rows (a pinned PY is a multiple of 4 within the wafer)"
                ));
            }
            Some(py) => Some(py),
            None => (1..)
                .map(|k| 16 * k)
                .take_while(|&py| py <= wafer.height)
                .find(|&py| Self::grid(spec, px, py, true, &sizing).bytes(false) <= wafer.pe_bytes),
        }
        .ok_or_else(|| {
                let tall = Self::grid(spec, px, wafer.height / 16 * 16, true, &sizing);
                Refused(format!(
                    "{name}: no {px}-column bf16 grid holds {} lanes: even {} rows want {} bytes a PE (of {})",
                    sizing.lanes,
                    tall.py,
                    tall.bytes(false),
                    wafer.pe_bytes
                ))
            })?;
        let grid = Self::grid(spec, px, py, true, &goals);
        let bytes = grid.bytes(false);
        if bytes > wafer.pe_bytes {
            return refuse(format!(
                "{name}: the {px} x {py} grid at B {} NB {} wants {bytes} bytes a PE (of {})",
                grid.lanes, grid.blocks, wafer.pe_bytes
            ));
        }
        // The bf16 head (head2d v10): 32 dims a PE, 256 rows, the fewest
        // blocks (even) whose tables fit.
        let c = spec.hidden.div_ceil(HEAD16_HC);
        let tables = if spec.tied { 1 } else { 2 };
        let head = (1..)
            .map(|k| 2 * k)
            .take_while(|nb| nb * (c + 1) <= wafer.width)
            .find_map(|nb| {
                let vr = spec.vocab.div_ceil(nb * HEAD16_ROWS);
                (tables * 2 * HEAD16_HC * vr <= HEAD16_TABLES).then(|| Head {
                    c,
                    hc: HEAD16_HC,
                    hh: HEAD16_ROWS,
                    vr,
                    gx: group(c, 16).unwrap_or(2),
                    gy: group(HEAD16_ROWS / 2, 18).unwrap_or(2),
                    gb: group(nb, 4).unwrap_or(2),
                    blocks: nb,
                    vmc: 1,
                    untied: !spec.tied,
                    f32: false,
                })
            })
            .ok_or_else(|| {
                Refused(format!(
                    "{name}: no head2d shape holds a {}-row vocabulary",
                    spec.vocab
                ))
            })?;
        // Where its grids would go: whole grids on the wafer beside the head.
        let (across, down) = (wafer.width / px, wafer.height / py);
        let mut blockers = vec![format!(
            "no full program places {px} x {py} grids yet (serve2d's diagonal relay also wants PX >= PY)"
        )];
        let beside = (wafer.width - head.width().min(wafer.width)) / px * down;
        if across * down < spec.layers + 1 {
            blockers.push(format!(
                "{} grids of {px} x {py} ({} PEs) and the head ({} PEs) on {} x {}: {across} x {down} = {} \
                 grids fit the whole wafer, {beside} beside a {} x {} head",
                spec.layers,
                spec.layers * px * py,
                head.width() * (head.hh + 1),
                wafer.width,
                wafer.height,
                across * down,
                head.width(),
                head.hh + 1
            ));
        }
        Ok(SpatialPlan {
            spec: spec.clone(),
            family: Family::Paged8b,
            layers: spec.layers,
            vocab: spec.vocab,
            grid,
            head,
            program: None,
            grid_bytes: bytes,
            pe_bytes: wafer.pe_bytes,
            // serve8b places lm_head itself.
            needs: spec
                .needs()
                .into_iter()
                .filter(|n| *n != Need::LmHeadPlacement)
                .collect(),
            blockers,
        })
    }

    /// The build's compile parameters, in the order the requests list them
    /// (the compiler's `--params`).
    pub fn params(&self) -> Vec<(&'static str, usize)> {
        let (g, h) = (&self.grid, &self.head);
        let b = |x: bool| usize::from(x);
        let mut out = vec![
            ("PX", g.px),
            ("PY", g.py),
            ("Hc", g.hc),
            ("HR", g.hidden),
            ("Dc", g.dc),
            ("Gs", g.gs),
            ("Qh", g.qh),
            ("Ic", g.ic),
        ];
        match self.family {
            Family::Paged8b => {
                out.extend([
                    ("PS", g.page_size),
                    ("E", g.entries),
                    ("W16", b(g.w16)),
                    ("NP", 3),
                    ("KV16", b(g.kv16)),
                    ("MC", g.mlp_chunk),
                    ("PROF", 0),
                    ("iters", 1),
                    ("NB", g.blocks),
                    ("B", g.lanes),
                ]);
                if g.host_copy {
                    out.push(("KVCP", 0));
                }
            }
            Family::Serve2d | Family::Serve2dPipe => {
                let pipe = self.family == Family::Serve2dPipe;
                let p = self.program.unwrap_or(Program {
                    gw: 0,
                    gh: 0,
                    cap: 0,
                    mmax: None,
                    hb: None,
                    width: 0,
                    height: 0,
                });
                out.extend([("Tc", TC), ("W16", 0)]);
                if !pipe {
                    out.push(("NP", 3));
                }
                out.push(("KV16", 0));
                if !pipe {
                    out.push(("PROF", 0));
                }
                out.push(("GRID", 1));
                if pipe {
                    out.push(("HEADMODE", 2));
                }
                out.extend([
                    ("B", g.lanes),
                    ("PS", g.page_size),
                    ("NB", g.blocks),
                    ("E", g.entries),
                    ("MC", g.mlp_chunk),
                    ("GW", p.gw),
                    ("GH", p.gh),
                ]);
                if !pipe {
                    out.push(("HEADMODE", 2));
                }
                out.extend([
                    ("F", h.hh / 2),
                    ("HH", h.hh),
                    ("Vr", h.vr),
                    ("Gx", h.gx),
                    ("Gy", h.gy),
                    ("Gb", h.gb),
                    ("vocab", self.vocab),
                    ("PASSES", 3),
                    ("UNTIED", b(h.untied)),
                    ("CERT", 0),
                    ("MODE", 0),
                    ("OVL", 0),
                    ("VMC", h.vmc),
                    ("WF32", 1),
                ]);
                if !pipe {
                    out.push(("CAP", p.cap));
                }
                out.extend([("DBG", 0), ("iters", 1)]);
                if pipe {
                    out.extend([
                        ("MMAX", p.mmax.unwrap_or(1)),
                        ("CAP", p.cap),
                        ("HB", p.hb.unwrap_or(1)),
                    ]);
                }
                if g.host_copy {
                    out.push(("KVCP", 0));
                }
            }
        }
        out
    }

    /// The standalone bf16 head's parameters (`head2d_v10`; the 8B head).
    pub fn head_params(&self) -> Vec<(&'static str, usize)> {
        let h = &self.head;
        vec![
            ("W", h.width()),
            ("H", h.hh),
            ("C", h.c),
            ("Hc", h.hc),
            ("Vr", h.vr),
            ("Gx", h.gx),
            ("Gy", h.gy),
            ("Gb", h.gb),
            ("vocab", self.vocab),
            ("PASSES", if h.f32 { 3 } else { 1 }),
            ("UNTIED", usize::from(h.untied)),
            ("CERT", 0),
            ("MODE", if h.f32 { 0 } else { 2 }),
            ("OVL", 0),
            ("VMC", h.vmc),
            ("WF32", usize::from(h.f32)),
            ("iters", 1),
            ("SCHAIN", 0),
        ]
    }

    /// `params` as a request's `extra` string (`K:V,...`).
    pub fn extra(params: &[(&'static str, usize)]) -> String {
        params
            .iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Checks a compiled program's params (its `out.json`) against this
    /// plan's structural ones: a program built for another model's dims is
    /// refused before its weights are placed.
    pub fn check_program(&self, get: impl Fn(&str) -> Option<usize>) -> Result<(), Refused> {
        let want = self.params();
        for key in [
            "PX", "PY", "Hc", "HR", "Dc", "Gs", "Qh", "Ic", "vocab", "UNTIED",
        ] {
            let Some(&(_, w)) = want.iter().find(|(k, _)| *k == key) else {
                continue;
            };
            match get(key) {
                Some(v) if v == w => {}
                v => {
                    return refuse(format!(
                        "the wafer program's {key} is {v:?}; the model wants {w} ({})",
                        Self::extra(&want)
                    ));
                }
            }
        }
        if let (Some(p), Some(gw), Some(gh)) = (self.program, get("GW"), get("GH"))
            && gw * gh != p.gw * p.gh
        {
            return refuse(format!(
                "the wafer program holds {} layers; the model has {}",
                gw * gh,
                self.layers
            ));
        }
        Ok(())
    }
}

impl SpatialPlan {
    /// [`ModelSpec::check_program`] on a program's `out.json`, then the
    /// plan's own structural params (a 2-D build for these dims).
    pub fn check(&self, out_json: &serde_json::Value) -> Result<(), Refused> {
        let get = out_json_params(out_json);
        self.spec.check_program(&get)?;
        if get("HEADMODE").is_some() {
            self.check_program(&get)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::model::check;
    use super::*;
    use poem_ir::Dtype;
    use std::collections::BTreeMap;

    fn spec(name: &str) -> Result<ModelSpec, Refused> {
        let catalog = poem_compiler::catalog::repository();
        let deployment = catalog
            .parse(name)
            .unwrap_or_else(|| panic!("{name} parses"));
        ModelSpec::read(&deployment.trace(poem_ir::Platform::Cerebras))
    }

    fn map(extra: &str) -> BTreeMap<String, usize> {
        extra
            .split(',')
            .map(|kv| {
                let (k, v) = kv.split_once(':').expect("k:v");
                (k.to_string(), v.parse().expect("a number"))
            })
            .collect()
    }

    fn plan(spec: &ModelSpec, goals: Goals) -> SpatialPlan {
        SpatialPlan::derive(spec, WaferDims::CS3, goals).expect("a plan")
    }

    /// Asserts the plan's params equal a request's `extra`, key for key.
    fn same(params: &[(&'static str, usize)], extra: &str) {
        let ours = map(&SpatialPlan::extra(params));
        assert_eq!(
            ours,
            map(extra),
            "\nours:  {}\ntoday: {extra}",
            SpatialPlan::extra(params)
        );
    }

    #[test]
    fn qwen3_dims_come_from_the_trace() {
        let s = spec("qwen3-0.6b-bf16-kv-bf16").expect("qwen3-0.6b reads");
        assert_eq!(
            (
                s.layers, s.hidden, s.q_heads, s.kv_heads, s.head_dim, s.inter, s.vocab
            ),
            (28, 1024, 16, 8, 128, 3072, 151_936)
        );
        assert!(s.tied && s.qk_norm && s.rope_table.is_none());
        assert_eq!(
            (s.rope_theta, s.rms_eps, s.weights, s.kv),
            (1e6, 1e-6, Dtype::Bf16, Dtype::Bf16)
        );
        assert!((s.sm_scale - 128f32.sqrt().recip()).abs() < 1e-7);
        let s = spec("qwen3-8b-bf16-kv-bf16").expect("qwen3-8b reads");
        assert_eq!(
            (s.layers, s.hidden, s.q_heads, s.kv_heads, s.inter, s.tied),
            (36, 4096, 32, 8, 12288, false)
        );
        let s = spec("qwen3-0.6b-bf16-kv-bf16-yarn4").expect("a stretched qwen3 reads");
        assert_eq!(s.needs(), vec![Need::RopeTable]);
    }

    /// Qwen3-0.6B: serve2d's batched hardware builds and the pipeline's,
    /// from the model's dims and the batch alone.
    #[test]
    fn qwen3_0_6b_plans_todays_serve2d_builds() {
        let s = spec("qwen3-0.6b-bf16-kv-bf16").expect("qwen3-0.6b reads");
        let base = "PX:128,PY:128,Hc:8,HR:1024,Dc:8,Gs:16,Qh:2,Ic:24,Tc:32,W16:0,NP:3,KV16:0,PROF:0,\
                    GRID:1,PS:16,E:1,GW:4,GH:7,HEADMODE:2,F:396,HH:792,Vr:192,Gx:16,Gy:18,Gb:1,\
                    vocab:151936,PASSES:3,UNTIED:0,CERT:0,MODE:0,OVL:0,VMC:1,WF32:1,DBG:0,iters:1";
        // The four batched builds: B, NB, MC, CAP.
        for (b, nb, mc, cap) in [(4, 8, 2, 64), (1, 8, 1, 64), (8, 8, 2, 64), (16, 4, 2, 32)] {
            let p = plan(&s, Goals::lanes(b));
            assert_eq!(p.family, Family::Serve2d);
            assert!(p.blockers.is_empty() && p.needs.is_empty(), "{p:?}");
            same(
                &p.params(),
                &format!("{base},B:{b},NB:{nb},MC:{mc},CAP:{cap}"),
            );
            let prm: BTreeMap<_, _> = p.params().into_iter().collect();
            let pairs: Vec<(&str, usize)> = prm.iter().map(|(k, v)| (*k, *v)).collect();
            let lay =
                super::super::program::ProgramSpec::of_pairs(&pairs).expect("a serve2d layout");
            assert_eq!((lay.layers(), lay.head_dim()), (28, 128));
            assert!(p.check_program(|k| prm.get(k).copied()).is_ok());
        }
        let p = plan(&s, Goals::lanes(16));
        assert_eq!((p.grid.pages(), p.grid.max_context()), (512, 2048));
        assert_eq!(p.program.map(|p| (p.width, p.height)), Some((641, 896)));

        // The pipelines: B, NB, MMAX, CAP, HB.
        let base = "PX:128,PY:128,Hc:8,HR:1024,Dc:8,Gs:16,Qh:2,Ic:24,Tc:32,W16:0,KV16:0,GRID:1,\
                    HEADMODE:2,PS:16,E:1,MC:2,GW:4,GH:7,F:396,HH:792,Vr:192,Gx:16,Gy:18,Gb:1,\
                    vocab:151936,PASSES:3,UNTIED:0,CERT:0,MODE:0,OVL:0,VMC:1,WF32:1,DBG:0,iters:1";
        for (b, nb, mmax, cap, hb) in [
            (4, 8, 8, 16, 4),
            (4, 8, 8, 32, 4),
            (8, 8, 8, 16, 8),
            (16, 4, 4, 16, 8),
        ] {
            let goals = Goals {
                pipeline: true,
                cap: Some(cap),
                ..Goals::lanes(b)
            };
            let p = plan(&s, goals);
            assert_eq!(p.family, Family::Serve2dPipe);
            same(
                &p.params(),
                &format!("{base},B:{b},NB:{nb},MMAX:{mmax},CAP:{cap},HB:{hb}"),
            );
        }
    }

    /// Qwen3-8B: the paged8b layer builds (128 x 160) and the bf16 head.
    #[test]
    fn qwen3_8b_plans_todays_paged8b_builds() {
        let s = spec("qwen3-8b-bf16-kv-bf16").expect("qwen3-8b reads");
        let base = "PX:128,PY:160,Hc:26,HR:4096,Dc:8,Gs:16,Qh:4,Ic:96,PS:16,E:1,W16:1,NP:3,KV16:1,\
                    MC:1,PROF:0,iters:1";
        // (B, NB pinned or not, KVCP 0): today's NB 4 at B 1 and 2 was a
        // test matrix's choice; the deepest pool would be 8 there.
        for (b, nb, auto, host) in [
            (1, 4, false, false),
            (2, 4, false, false),
            (4, 4, true, false),
            (8, 1, true, false),
            (8, 3, false, true),
        ] {
            let goals = Goals {
                blocks: (!auto).then_some(nb),
                host_copy: host,
                ..Goals::lanes(b)
            };
            let p = plan(&s, goals);
            assert_eq!(p.family, Family::Paged8b);
            assert!(p.program.is_none() && p.grid_bytes <= p.pe_bytes, "{p:?}");
            let kvcp = if host { ",KVCP:0" } else { "" };
            same(&p.params(), &format!("{base},NB:{nb},B:{b}{kvcp}"));
        }
        // Builds that did not link don't fit the model either.
        for (b, nb, host) in [(8, 2, false), (8, 4, true)] {
            let goals = Goals {
                blocks: Some(nb),
                host_copy: host,
                ..Goals::lanes(b)
            };
            assert!(
                SpatialPlan::derive(&s, WaferDims::CS3, goals).is_err(),
                "B {b} NB {nb}"
            );
        }
        // B 9 (did not link at 160 rows) takes a taller grid.
        assert_eq!(plan(&s, Goals::lanes(9)).grid.py, 176);
        let p = plan(&s, Goals::lanes(1));
        assert_eq!(p.grid.blocks, 8);
        same(
            &p.head_params(),
            "W:516,H:256,C:128,Hc:32,Vr:149,Gx:16,Gy:16,Gb:4,vocab:151936,PASSES:1,UNTIED:1,\
             CERT:0,MODE:2,OVL:0,VMC:1,WF32:0,iters:1,SCHAIN:0",
        );
        // No full program: serve2d wants Qh 2, and 36 grids don't fit.
        assert!(
            p.blockers[0].contains("4 query heads a kv head"),
            "{:?}",
            p.blockers
        );
        assert!(
            p.blockers.iter().any(|b| b.contains("5 x 7 = 35 grids")),
            "{:?}",
            p.blockers
        );
    }

    /// Qwen3-14B: a layer grid fits a PE; 40 of them don't fit the wafer.
    #[test]
    fn qwen3_14b_plans_a_grid_that_does_not_fit_the_wafer() {
        let s = spec("qwen3-14b-bf16-kv-bf16").expect("qwen3-14b reads");
        assert_eq!(
            (s.hidden, s.q_heads, s.kv_heads, s.inter, s.layers),
            (5120, 40, 8, 17408, 40)
        );
        let p = plan(&s, Goals::lanes(4));
        assert_eq!(p.family, Family::Paged8b);
        assert_eq!((p.grid.px, p.grid.qh, p.grid.ic), (128, 5, 136));
        assert!(p.grid_bytes <= p.pe_bytes);
        assert!(p.program.is_none());
        let wafer = p
            .blockers
            .iter()
            .find(|b| b.contains("40 grids"))
            .expect("a wafer blocker");
        assert!(
            wafer.contains(&format!("{} x {}", p.grid.px, p.grid.py)),
            "{wafer}"
        );
    }

    /// Llama 3.1 8B reads, without q/k norms and with llama3-scaled RoPE: it
    /// plans as a paged8b grid that needs the identity-norm variant.
    #[test]
    fn llama_3_plans_with_the_identity_qk_norm() {
        let s = spec("llama31-8b-bf16-kv-bf16").expect("llama 3.1 8b reads");
        assert!(!s.qk_norm && !s.tied);
        assert_eq!(
            (s.hidden, s.inter, s.vocab, s.rope_theta),
            (4096, 14336, 128_256, 500_000.0)
        );
        let p = plan(&s, Goals::lanes(4));
        assert_eq!(p.needs, vec![Need::IdentityQkNorm, Need::RopeTable]);
        assert_eq!((p.family, p.grid.ic, p.grid.qh), (Family::Paged8b, 112, 4));
    }

    /// A u4g64 deployment reads, and plans its grid by the u4 byte model:
    /// fewer rows a layer than its bf16 deployment's, W4 on.
    #[test]
    fn u4g64_plans_smaller_grids() {
        for id in ["qwen3-8b", "qwen3-14b", "qwen3-32b"] {
            let u = spec(&format!("{id}-u4g64-kv-bf16")).expect("u4g64 reads");
            let b = spec(&format!("{id}-bf16-kv-bf16")).expect("bf16 reads");
            assert_eq!(u.weights, Dtype::U4g64);
            let (pu, pb) = (plan(&u, Goals::lanes(1)), plan(&b, Goals::lanes(1)));
            assert!(pu.grid.w4 && !pb.grid.w4);
            assert_eq!(pu.family, Family::Paged8b);
            // On the bf16 grid the u4 tiles and their decode scratch (a
            // constant now) take less than the bf16 tiles, and the u4 grid
            // has fewer rows; except at Qwen3-32B's 400 rows (Hc 13, so the
            // fewest rows the planner looks for are tiny tiles): every PE of
            // a group's rows keeps the group's scales and biases for its
            // outputs, 8 bytes an output a group at Hc < 64 against Hc / 2
            // of codes, and with the u4 code's bytes it plans more rows.
            let on_bf16 = Grid {
                w4: true,
                ..pb.grid
            };
            let smaller = on_bf16.weight_bytes() < pb.grid.weight_bytes();
            assert_eq!(smaller, id != "qwen3-32b", "{id}");
            assert_eq!(pu.grid.py < pb.grid.py, id != "qwen3-32b", "{id}");
        }
    }

    /// Models whose structure the kernels don't implement are refused by name.
    #[test]
    fn other_structures_are_refused() {
        for (name, why) in [
            ("qwen3-30b-a3b-bf16-kv-bf16", "mixture of experts"),
            ("gemma4-12b-bf16-kv-bf16", ""),
            ("qwen3-0.6b-bf16-kv-e4m3", "E4m3 KV"),
        ] {
            let err = spec(name).expect_err(name);
            assert!(err.0.contains(why), "{name}: {err}");
        }
    }

    fn out_json(extra: &str) -> serde_json::Value {
        let params: serde_json::Map<_, _> = map(extra)
            .into_iter()
            .map(|(k, v)| (k, serde_json::json!(v.to_string())))
            .collect();
        serde_json::json!({ "params": params })
    }

    /// The load check: a program for the model's dims passes, one for other
    /// dims, a strip program (serve28, no longer served) or an untied model
    /// is refused.
    #[test]
    fn programs_are_checked_against_the_trace() {
        let catalog = poem_compiler::catalog::repository();
        let trace = |name: &str| {
            catalog
                .parse(name)
                .expect(name)
                .trace(poem_ir::Platform::Cerebras)
        };
        let small = trace("qwen3-0.6b-bf16-kv-bf16");
        let s = ModelSpec::read(&small).expect("qwen3-0.6b reads");
        let p = plan(&s, Goals::lanes(4));
        let serve2d = out_json(&SpatialPlan::extra(&p.params()));
        assert!(check(&small, &serve2d).is_ok());
        assert!(p.check(&serve2d).is_ok());
        let wrong = out_json(&SpatialPlan::extra(&p.params()).replace("vocab:151936", "vocab:100"));
        assert!(
            check(&small, &wrong)
                .expect_err("vocab")
                .0
                .contains("vocab is 100")
        );
        let serve28 = "kernel_cols:64,hidden:1024,heads:2,head_dim:128,inter:3072,positions:4096,\
                       attn_rows:128,mlp_rows:128,band_rows:16,seg_rows:32,head_rows:128,vocab_rows:1200,\
                       prompt_max:64,vocab:151936,layers:28,lps:7,iters:64";
        let err = check(&small, &out_json(serve28)).expect_err("serve28");
        assert!(err.0.contains("no longer served"), "{err}");
        // Qwen3-8B's lm_head is its own: today's placement would read the
        // embedding for it.
        let big = trace("qwen3-8b-bf16-kv-bf16");
        let err = check(&big, &serve2d).expect_err("untied");
        assert!(err.0.contains("untied lm_head"), "{err}");
    }

    /// Prints whether serve8b_region (option A) builds each Qwen3 model at
    /// B 1, bf16 and u4g64 (`--nocapture`).
    #[test]
    fn region_fits_print() {
        for id in ["qwen3-8b", "qwen3-14b", "qwen3-32b"] {
            for w in ["bf16", "u4g64"] {
                let name = format!("{id}-{w}-kv-bf16");
                let s = spec(&name).expect(&name);
                let goals = Goals {
                    lanes: 1,
                    grid_lanes: 4,
                    host_copy: true,
                    ..Goals::default()
                };
                let p = plan(&s, goals);
                let mut cfg = super::super::build::ServeConfig::new(goals);
                cfg.region = true;
                let g = &p.grid;
                match super::super::build::program(&p, &cfg) {
                    Ok(spec) => println!(
                        "FIT {name}: plan grid {}x{} Hc {} bytes {} -> {:?} {:?} {:?} {:?}",
                        g.px,
                        g.py,
                        g.hc,
                        p.grid_bytes,
                        spec.design,
                        spec.rectangle(),
                        match &spec.layers {
                            super::super::program::Layers::Units { grid, .. } =>
                                format!("grid {}x{} Hc {}", grid.px, grid.py, grid.hc),
                            _ => String::new(),
                        },
                        match &spec.layers {
                            super::super::program::Layers::Units { units, .. } => format!(
                                "regions {}x{} RNL {} RC {} RR {} NCOL {} RHc {} RIc {}",
                                units.rpx,
                                units.rpy,
                                units.rnl,
                                units.rc,
                                units.rr,
                                units.ncol,
                                units.rhc,
                                units.ric
                            ),
                            _ => String::new(),
                        }
                    ),
                    Err(e) => println!(
                        "FIT {name}: grid {}x{} Hc {} bytes {} -> {e}",
                        g.px, g.py, g.hc, p.grid_bytes
                    ),
                }
            }
        }
    }

    /// The head and strip byte model at W4H 1 against its link probes.
    #[test]
    fn head_u4_model_bounds_probes() {
        for (vr, hc, mmax, cap, probe) in [
            (132, 36, None, 64, 39_392),
            (228, 40, None, 32, 47_280),
            (228, 40, Some(12), 16, 48_944),
        ] {
            let m = final_root_bytes(vr, hc, 1, 1, mmax, cap, true);
            assert!(
                m >= probe && m <= probe + 400,
                "Vr {vr}: model {m} vs probe {probe}"
            );
        }
        for (ve, hc, u4, probe) in [
            (132, 36, false, 47_184),
            (132, 36, true, 21_664),
            (228, 40, true, 32_112),
        ] {
            let m = embed_bytes(ve, hc, 1, u4);
            assert!(
                m >= probe && m <= probe + 200,
                "Ve {ve} u4 {u4}: model {m} vs probe {probe}"
            );
        }
    }

    /// u4g64 feasibility of the 27B-31B class on serve8b_region
    /// (`--nocapture`): each model as a Qwen3-style dense stand-in of the
    /// same per-layer u4 bytes (head_dim 128; attention rows the shape allows,
    /// the rest of the projections folded into `inter`), its own vocab and
    /// tying, then at Qwen3's vocab (the head alone), at B 1 and on the ring,
    /// the fewest KV blocks.
    #[test]
    #[ignore]
    fn u4_feasibility_print() {
        let base = spec("qwen3-14b-u4g64-kv-bf16").expect("14b");
        // (name, hidden, layers, q, kv, inter, vocab, tied)
        let models = [
            ("qwen3-14b", 5120, 40, 40, 8, 17408, 151_936, false),
            ("qwen36-27b", 5120, 64, 48, 8, 20680, 248_320, false),
            ("gemma4-12b", 3840, 48, 40, 8, 15648, 262_144, true),
            ("gemma4-31b", 5376, 60, 72, 8, 23552, 262_144, true),
            ("muse-glimmer-30b", 6656, 52, 32, 8, 20824, 202_048, false),
            ("qwen3-32b", 5120, 64, 64, 8, 25600, 151_936, false),
        ];
        for (name, hidden, layers, q, kv, inter, vocab, tied) in models {
            for v in [vocab, 151_936] {
                for pipeline in [false, true] {
                    let s = ModelSpec {
                        name: format!("{name}-u4g64"),
                        hidden,
                        layers,
                        q_heads: q,
                        kv_heads: kv,
                        inter,
                        vocab: v,
                        tied,
                        ..base.clone()
                    };
                    let goals = Goals {
                        lanes: 1,
                        grid_lanes: 1,
                        host_copy: true,
                        pipeline,
                        blocks: Some(1),
                        ..Goals::default()
                    };
                    let tag = format!("{name} vocab {v} pipe {}", u8::from(pipeline));
                    // The planner's head2d gate is not serve8b_region's head
                    // (`build::place_at` sizes that): plan at a small vocab, then
                    // place at the model's.
                    let small = ModelSpec {
                        vocab: 32_000,
                        ..s.clone()
                    };
                    let p = match SpatialPlan::derive(&small, WaferDims::CS3, goals) {
                        Ok(p) => SpatialPlan { vocab: v, ..p },
                        Err(e) => {
                            println!("FEAS {tag}: no plan: {e}");
                            continue;
                        }
                    };
                    if v == vocab && !pipeline {
                        // A grid PE's bytes by part, at the plan's grid.
                        let g = &p.grid;
                        let elems = g.hc * ((2 * g.qh + 2) * g.dc + 3 * g.ic);
                        let ideal = elems * 9 / 16;
                        let codes = elems / 2;
                        println!(
                            "PARTS {name}: grid {}x{} Hc {} Ic {}: u4 tiles+scratch {} B (codes {codes}, codes+scales {ideal}), tables {}, KV block {}, lane {}, total {}",
                            g.px,
                            g.py,
                            g.hc,
                            g.ic,
                            g.weight_bytes(),
                            g.table_bytes(),
                            g.block_bytes(),
                            g.lane_bytes(),
                            g.bytes(false)
                        );
                    }
                    let mut cfg = super::super::build::ServeConfig::new(goals);
                    cfg.region = true;
                    match super::super::build::params(&p, &cfg) {
                        Ok((d, kv)) => {
                            let get = |k: &str| {
                                kv.iter()
                                    .find(|(n, _)| *n == k)
                                    .map_or(String::new(), |(_, v)| v.to_string())
                            };
                            println!(
                                "FEAS {tag}: {d:?} grid {}x{} Hc {} Ic {}, regions {}x{} RNL {} {}x{}, NCOL {}, NB {}, Vr {}, CAP {}",
                                get("PX"),
                                get("PY"),
                                get("Hc"),
                                get("Ic"),
                                get("RPX"),
                                get("RPY"),
                                get("RNL"),
                                get("RC"),
                                get("RR"),
                                get("NCOL"),
                                get("NB"),
                                get("Vr"),
                                get("CAP")
                            );
                        }
                        Err(e) => println!(
                            "FEAS {tag}: {e}\n  closest: {}",
                            super::super::build::placement_report(&p, pipeline)
                        ),
                    }
                }
            }
        }
    }

    /// u4 scale duplication and the uniform one-layer layout (`--nocapture`):
    /// per model (the feasibility stand-ins),
    /// a unit PE's bytes at 128-column one-layer units sharing `UNITS` PEs
    /// (rows: the most that fit, and the most with group-aligned hidden
    /// tiles, `64 % Hc == 0` or `Hc % 64 == 0`), as the kernel holds scales
    /// (`dup`) and with each scale on one PE (`own`: wo and wd by a cross-PE
    /// per-group reduction, plus its partials' buffer), and the u4 head.
    #[test]
    #[ignore]
    fn u4_dedupe_print() {
        // Unit PEs of a wafer less the head, strip and VM columns.
        const UNITS: usize = 757_000;
        let base = spec("qwen3-14b-u4g64-kv-bf16").expect("14b");
        let most = WaferDims::CS3.pe_bytes - 768;
        let models = [
            ("qwen3-14b", 5120, 40, 40, 8, 17408, 151_936),
            ("qwen36-27b", 5120, 64, 48, 8, 20680, 248_320),
            ("gemma4-12b", 3840, 48, 40, 8, 15648, 262_144),
            ("gemma4-31b", 5376, 60, 72, 8, 23552, 262_144),
            ("muse-glimmer-30b", 6656, 52, 32, 8, 20824, 202_048),
            ("qwen3-32b", 5120, 64, 64, 8, 25600, 151_936),
        ];
        let goals = Goals {
            lanes: 1,
            grid_lanes: 1,
            host_copy: true,
            blocks: Some(1),
            ..Goals::default()
        };
        for (name, hidden, layers, q, kv, inter, vocab) in models {
            let s = ModelSpec {
                hidden,
                layers,
                q_heads: q,
                kv_heads: kv,
                inter,
                vocab,
                ..base.clone()
            };
            let pmax = (UNITS / (layers * 128)) / 4 * 4;
            let aligned = (4..=pmax).rev().step_by(4).find(|&py| {
                let hc = hidden.div_ceil(py);
                64 % hc == 0 || hc % 64 == 0
            });
            let vr = vocab.div_ceil(1152);
            for (tag, py) in [("most", Some(pmax)), ("aligned", aligned)] {
                let Some(py) = py else { continue };
                let g = SpatialPlan::grid(&s, 128, py, true, &goals);
                let head = final_root_bytes(vr + vr % 2, g.hc, 1, 1, None, 32, true);
                let t = super::super::u4::tiles(g.hc, g.dc, g.gs, g.qh, g.ic, true);
                let sb = super::super::u4::scale_bytes(&t);
                let (dup, own): (usize, usize) =
                    sb.iter().fold((0, 0), |(a, b), &(d, o)| (a + d, b + o));
                // The reduction's partials, [g][n + 1] f32 of the widest tile it
                // serves, past what the decode buffer (free by then) holds.
                let buf = 2 * (super::super::u4::BUF_WORDS + 1);
                let part = t
                    .iter()
                    .zip(&sb)
                    .filter(|(_, (d, o))| o < d)
                    .map(|(t, _)| (4 * t.g * (t.n + 1)).saturating_sub(buf))
                    .max()
                    .unwrap_or(0);
                let dupb = g.bytes(false) + Grid::staging_bytes(1);
                let ownb = dupb - dup + own + part;
                println!(
                    "DEDUP {name} {tag}: {layers} x 128x{py} Hc {} Dc {} Qh {} Ic {}: scales dup {dup} own {own} (wqkv {:?} wo {:?} wgu {:?} wd {:?}); PE dup {dupb} ({:+}), own {ownb} ({:+}); head Vr {vr} {head} B",
                    g.hc,
                    g.dc,
                    g.qh,
                    g.ic,
                    sb[0],
                    sb[3],
                    sb[4],
                    sb[5],
                    dupb as isize - most as isize,
                    ownb as isize - most as isize
                );
            }
        }
    }

    /// A model's serve8b_region params at B 1 (`PARAMS_OF=<name>`,
    /// `--nocapture`): the request's `extra`.
    #[test]
    #[ignore]
    fn region_params_print() {
        let name = std::env::var("PARAMS_OF").expect("PARAMS_OF");
        let s = spec(&name).expect("spec");
        let pipeline = std::env::var("PIPE").is_ok_and(|v| v == "1");
        let grid_lanes = std::env::var("GRID_LANES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        let goals = Goals {
            lanes: 1,
            grid_lanes,
            host_copy: true,
            pipeline,
            ..Goals::default()
        };
        let p = plan(&s, goals);
        let mut cfg = super::super::build::ServeConfig::new(goals);
        cfg.region = true;
        let (d, kv) = super::super::build::params(&p, &cfg).expect("params");
        let spec = super::super::program::ProgramSpec::from_params(d, &kv).expect("spec");
        println!(
            "PARAMS {:?} {:?} {}",
            d,
            spec.rectangle(),
            kv.iter()
                .map(|(k, v)| format!("{k}:{v}"))
                .collect::<Vec<_>>()
                .join(",")
        );
    }

    /// The unit PE bytes behind a 14B/32B u4 layout (`--nocapture`): the
    /// grid at serve8b_region's widest head-fitting PY (144) and the
    /// 256 x 240 region at 3..5 layers a PE (the byte model of one layer,
    /// the rest's weights, tables and pool blocks added).
    #[test]
    fn unit_bytes_print() {
        for (name, py) in [
            ("qwen3-14b-u4g64-kv-bf16", 144),
            ("qwen3-32b-u4g64-kv-bf16", 144),
        ] {
            let s = spec(name).expect(name);
            let goals = Goals {
                lanes: 1,
                grid_lanes: 1,
                host_copy: true,
                blocks: Some(1),
                ..Goals::default()
            };
            let g = SpatialPlan::grid(&s, 128, py, true, &goals);
            let r = Grid {
                px: 256,
                py: 240,
                hc: s.hidden.div_ceil(240),
                dc: 4,
                gs: 32,
                ic: s.inter.div_ceil(256),
                ..g
            };
            let per = r.weight_bytes() + r.table_bytes() + r.blocks * r.block_bytes();
            println!(
                "UNIT {name}: grid 128x{py} Hc {} Ic {} = {} B (u4 weights {}); region 256x240 RHc {} RIc {}: {}",
                g.hc,
                g.ic,
                g.bytes(false),
                g.weight_bytes(),
                r.hc,
                r.ic,
                (3..=5)
                    .map(|nl| format!("RNL {nl} = {} B", r.bytes(false) + (nl - 1) * per))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }

    /// The byte model at given unit shapes (`GRID_AT=hidden,inter,qh,py,ic`,
    /// `--nocapture`), to compare with a link probe's image sizes.
    #[test]
    #[ignore]
    fn grid_bytes_at() {
        let v: Vec<usize> = std::env::var("GRID_AT")
            .expect("GRID_AT")
            .split(',')
            .map(|x| x.parse().unwrap())
            .collect();
        let s = spec("qwen3-14b-u4g64-kv-bf16").unwrap();
        let goals = Goals {
            lanes: 1,
            grid_lanes: 1,
            host_copy: true,
            blocks: Some(1),
            ..Goals::default()
        };
        let g = Grid {
            hidden: v[0],
            hc: v[0].div_ceil(v[3]),
            qh: v[2],
            py: v[3],
            ic: v[4],
            ..SpatialPlan::grid(&s, 128, v[3], true, &goals)
        };
        let _ = v[1];
        println!(
            "BYTES grid {} (u4 {}) region nl1 {} layer {}",
            g.bytes(false),
            g.weight_bytes(),
            g.region_bytes(1, false),
            g.layer_bytes()
        );
        // GRID_AT=...,px,dc,gs,nl: a region of those
        if v.len() > 5 {
            let r = Grid {
                px: v[5],
                dc: v[6],
                gs: v[7],
                ..g
            };
            println!(
                "BYTES region {}x{} nl {} = {} ring {} (grid ring {})",
                r.px,
                r.py,
                v[8],
                r.region_bytes(v[8], false),
                r.region_bytes(v[8], true),
                g.bytes(true)
            );
        }
    }

    /// Prints every model's plan (`--nocapture`): the table in PLAN.md.
    #[test]
    fn plans_print() {
        for (name, b) in [
            ("qwen3-0.6b-bf16-kv-bf16", 4),
            ("qwen3-8b-bf16-kv-bf16", 4),
            ("qwen3-14b-bf16-kv-bf16", 4),
            ("qwen3-32b-bf16-kv-bf16", 4),
            ("llama31-8b-bf16-kv-bf16", 4),
            ("qwen3-8b-u4g64-kv-bf16", 4),
            ("qwen3-14b-u4g64-kv-bf16", 4),
            ("qwen3-32b-u4g64-kv-bf16", 4),
        ] {
            let s = spec(name).expect(name);
            match SpatialPlan::derive(&s, WaferDims::CS3, Goals::lanes(b)) {
                Ok(p) => println!(
                    "{name}: {:?} {} bytes {}/{} needs {:?}\n  head {}\n  blockers {:?}",
                    p.family,
                    SpatialPlan::extra(&p.params()),
                    p.grid_bytes,
                    p.pe_bytes,
                    p.needs,
                    SpatialPlan::extra(&p.head_params()),
                    p.blockers
                ),
                Err(e) => println!("{name}: refused: {e}"),
            }
        }
    }
}
