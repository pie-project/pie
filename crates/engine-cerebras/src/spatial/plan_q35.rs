//! The plan for a hybrid model (Qwen3.5 dense): Gated DeltaNet units and
//! gated-attention units, one square f32 grid a layer as serve2d's, in the
//! model's layer order, and serve2d's f32 head beside them.
//!
//! Both units keep serve2d's split of the hidden state (row `py` holds
//! hidden chunk `py`, `Hc` dims) and of the MLP (`Ic` rows a column). They
//! differ in what a column holds:
//!
//! - a Gated DeltaNet unit gives each k head `GK` columns; column `c` holds
//!   `DK` dims of its head's q and k and `DV` of each of its `VH` v heads'
//!   v and z (and every v head's b and a, replicated). The recurrent state
//!   lies on the grid as the delta rule reads it: PE (`py`, `c`) holds row
//!   `k = py` of each v head's `k_dim x v_dim` state, over column `c`'s v
//!   dims. So `k_dim` is the grid's height, and a step's two state reads
//!   (`S^T k`, `S^T q`) are one all-reduce down each column.
//! - an attention unit gives each kv head `Gs` columns of `Dc` dims, as
//!   qwen2d_paged, with `Qh` query heads a kv head and each query head's
//!   gate beside it. RoPE turns `ROT` dims a head: the placement lays each
//!   turned pair in one column (the kernels turn a column's two halves), the
//!   unturned dims after them with a turn of zero.

use super::model::refuse;
use super::model::{LayerKind, ModelSpec, Refused};
use super::plan::{Goals, Grid, Head, Program, SpatialPlan, WaferDims};

/// A Gated DeltaNet unit's splits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnUnit {
    pub p: usize,
    pub hc: usize,
    pub ic: usize,
    /// Columns a k head, its dims a column, v heads a k head, and the v
    /// dims of each a column.
    pub gk: usize,
    pub dk: usize,
    pub vh: usize,
    pub dv: usize,
    pub conv_width: usize,
    pub lanes: usize,
    pub mlp_chunk: usize,
}

/// serve2d's fitted fixed bytes a PE (code, runtime, collectives), which
/// the units share; a lane's misc scratch as serve2d's.
const FIXED: usize = 24_984;
const LANE_MISC: usize = 50;

impl GdnUnit {
    /// A column's projection outputs: q and k (`DK` each), v and z (`VH x
    /// DV` each), and each v head's b and a.
    pub fn proj(&self) -> usize {
        2 * self.dk + 2 * self.vh * self.dv + 2 * self.vh
    }

    /// The conv's channels a column (q, k, v).
    pub fn conv_channels(&self) -> usize {
        2 * self.dk + self.vh * self.dv
    }

    /// A PE's estimated bytes (f32 weights and state).
    pub fn bytes(&self) -> usize {
        let weights = self.hc * self.proj()
            + self.conv_channels() * self.conv_width
            + self.vh * self.dv * self.hc
            + 3 * self.hc * self.ic;
        // gamma1/2, the gated norm's weight, -exp(A_log) and dt_bias.
        let tables = 2 * self.hc + self.vh * self.dv + 2 * self.vh;
        // x and the handoff's x, the projection and its sum of squares, the
        // state row, the conv's window.
        let lane = 2 * self.hc
            + self.proj()
            + 1
            + self.vh * self.dv
            + (self.conv_width - 1) * self.conv_channels()
            + LANE_MISC;
        FIXED + 4 * (weights + tables + self.lanes * lane + self.mlp_chunk * (3 * self.ic + 1))
    }
}

/// An attention unit: qwen2d_paged's grid with gated query heads and a
/// partial RoPE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttnUnit {
    pub grid: Grid,
    /// Dims a head RoPE turns.
    pub rot: usize,
}

impl AttnUnit {
    /// A PE's estimated bytes: the paged grid's, plus the gates' weights
    /// and their lane buffer.
    pub fn bytes(&self) -> usize {
        let g = &self.grid;
        g.bytes(false) + 4 * g.qh * g.dc * (g.hc + g.lanes)
    }
}

/// A hybrid model on the wafer.
#[derive(Debug, Clone, PartialEq)]
pub struct Q35Plan {
    pub spec: ModelSpec,
    pub kinds: Vec<LayerKind>,
    pub gdn: GdnUnit,
    pub attn: AttnUnit,
    pub head: Head,
    /// The grids' placement and the head's (the smallest rectangle).
    pub placement: Program,
    /// The placement once nothing blocks a build.
    pub program: Option<Program>,
    pub pe_bytes: usize,
    /// Why no program places the model (empty when `program` is set).
    pub blockers: Vec<String>,
}

impl Q35Plan {
    pub fn derive(spec: &ModelSpec, wafer: WaferDims, goals: Goals) -> Result<Q35Plan, Refused> {
        let name = &spec.name;
        let Some(hy) = &spec.hybrid else {
            return refuse(format!("{name}: not a hybrid (plan it with SpatialPlan)"));
        };
        let g = hy.gdn;
        if goals.lanes == 0 || goals.page_size == 0 {
            return refuse("a plan wants at least one lane and a page size");
        }
        // The state's k rows are the grid's rows.
        let p = g.k_dim;
        let gk = p / g.k_heads;
        let fits_columns = gk >= 2
            && p.is_multiple_of(g.k_heads)
            && g.k_dim.is_multiple_of(gk)
            && g.v_dim.is_multiple_of(gk);
        if !fits_columns {
            return refuse(format!(
                "{name}: {} k heads of {} do not split over a {p}-column grid",
                g.k_heads, g.k_dim
            ));
        }
        if !spec.hidden.is_multiple_of(p) || !spec.inter.is_multiple_of(p) {
            return refuse(format!(
                "{name}: hidden {} or inter {} does not split over {p} rows/columns",
                spec.hidden, spec.inter
            ));
        }
        let lanes = goals.lanes;
        let mlp_chunk = goals
            .mlp_chunk
            .unwrap_or(if lanes.is_multiple_of(2) { 2 } else { 1 });
        let gdn = GdnUnit {
            p,
            hc: spec.hidden / p,
            ic: spec.inter / p,
            gk,
            dk: g.k_dim / gk,
            vh: g.v_heads / g.k_heads,
            dv: g.v_dim / gk,
            conv_width: g.conv_width,
            lanes,
            mlp_chunk,
        };
        // Attention: Gs columns a kv head of Dc dims, turned pairs a column.
        let gs = p / spec.kv_heads;
        if !p.is_multiple_of(spec.kv_heads) || !spec.head_dim.is_multiple_of(gs) {
            return refuse(format!(
                "{name}: {} kv heads of {} do not split over {p} columns",
                spec.kv_heads, spec.head_dim
            ));
        }
        let dc = spec.head_dim / gs;
        let pairs = hy.rotary_dim / 2;
        // A column turns at most Dc/2 pairs; the turned pairs must fit the
        // head's columns.
        if !dc.is_multiple_of(2) || pairs > gs * (dc / 2) {
            return refuse(format!(
                "{name}: {pairs} RoPE pairs over {gs} columns of {dc} dims"
            ));
        }
        let mut grid = Grid {
            px: p,
            py: p,
            hc: spec.hidden / p,
            hidden: spec.hidden,
            dc,
            gs,
            qh: spec.group(),
            ic: spec.inter / p,
            lanes,
            page_size: goals.page_size,
            blocks: goals.blocks.unwrap_or(1),
            entries: 1,
            mlp_chunk,
            w16: false,
            kv16: false,
            host_copy: goals.host_copy,
        };
        let mut attn = AttnUnit {
            grid,
            rot: hy.rotary_dim,
        };
        if goals.blocks.is_none() {
            let mut nb = 1;
            while nb * 2 <= goals.max_blocks
                && (AttnUnit {
                    grid: Grid {
                        blocks: nb * 2,
                        ..grid
                    },
                    ..attn
                })
                .bytes()
                    <= wafer.pe_bytes
            {
                nb *= 2;
            }
            grid.blocks = nb;
            attn.grid = grid;
        }
        let mut blockers = Vec::new();
        for (what, bytes) in [("Gated DeltaNet", gdn.bytes()), ("attention", attn.bytes())] {
            if bytes > wafer.pe_bytes {
                blockers.push(format!(
                    "the {what} unit wants {bytes} bytes a PE (of {}) at B {lanes}",
                    wafer.pe_bytes
                ));
            }
        }
        // serve2d's placement: GW x GH grids serpentine, head east.
        let (layers, vmc) = (spec.layers, 1);
        let mut best: Option<(Program, Head)> = None;
        for gw in 2..=layers {
            if !layers.is_multiple_of(gw) {
                continue;
            }
            let gh = layers / gw;
            let (width, height) = (gw * p + p + vmc, gh * p);
            if gh < 4 || width > wafer.width || height > wafer.height {
                continue;
            }
            let Some(head) = SpatialPlan::head32(spec, &grid, height, gh) else {
                continue;
            };
            if best.is_none_or(|(b, _)| width * height < b.width * b.height) {
                let mut cap = goals.cap.unwrap_or(64);
                while lanes * cap > super::plan::TOKENS_SERVE && cap > 2 {
                    cap /= 2;
                }
                let program = Program {
                    gw,
                    gh,
                    cap,
                    mmax: None,
                    hb: None,
                    width,
                    height,
                };
                best = Some((program, head));
            }
        }
        let Some((program, head)) = best else {
            return refuse(format!(
                "{name}: no GW x GH = {layers} split of {p} x {p} grids with the head beside \
                 them fits {} x {}",
                wafer.width, wafer.height
            ));
        };
        blockers.push(
            "the qwen35 units' CSL (qwen35_gdn, qwen35_attn) is not built: no program yet".into(),
        );
        Ok(Q35Plan {
            spec: spec.clone(),
            kinds: hy.kinds.clone(),
            gdn,
            attn,
            head,
            placement: program,
            program: blockers.is_empty().then_some(program),
            pe_bytes: wafer.pe_bytes,
            blockers,
        })
    }

    /// The build's parameters: the units', the layer mix (bit `l` of
    /// `KINDS`: layer `l` attends), the placement and the head.
    pub fn params(&self) -> Vec<(&'static str, usize)> {
        let (g, a, h, program) = (&self.gdn, &self.attn.grid, &self.head, &self.placement);
        let kinds = self
            .kinds
            .iter()
            .enumerate()
            .filter(|(_, k)| **k == LayerKind::Attention)
            .fold(0usize, |m, (l, _)| m | 1 << l);
        vec![
            ("Q35", 1),
            ("PX", g.p),
            ("PY", g.p),
            ("Hc", g.hc),
            ("Ic", g.ic),
            ("GK", g.gk),
            ("DK", g.dk),
            ("VH", g.vh),
            ("DV", g.dv),
            ("CW", g.conv_width),
            ("Gs", a.gs),
            ("Dc", a.dc),
            ("Qh", a.qh),
            ("ROT", self.attn.rot),
            ("PS", a.page_size),
            ("NB", a.blocks),
            ("E", a.entries),
            ("B", g.lanes),
            ("MC", g.mlp_chunk),
            ("KINDS", kinds),
            ("GW", program.gw),
            ("GH", program.gh),
            ("CAP", program.cap),
            ("HEADMODE", 2),
            ("F", h.hh / 2),
            ("HH", h.hh),
            ("Vr", h.vr),
            ("Gx", h.gx),
            ("Gy", h.gy),
            ("vocab", self.spec.vocab),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::model::ModelSpec;

    fn spec(name: &str) -> ModelSpec {
        let c = poem_compiler::catalog::repository();
        let t = c
            .parse(name)
            .expect(name)
            .trace(poem_ir::Platform::Cerebras);
        ModelSpec::read(&t).expect(name)
    }

    /// Qwen3.5-0.8B: 24 grids of 128 x 128 (18 Gated DeltaNet, 6 attention)
    /// 4 across and 6 down, the 248k head beside them; each unit fits a PE.
    #[test]
    fn qwen35_0_8b_places_both_units() {
        let s = spec("qwen35-d0.8b-bf16-kv-bf16");
        let p = Q35Plan::derive(&s, WaferDims::CS3, Goals::lanes(1)).expect("a plan");
        let g = p.gdn;
        assert_eq!(
            (g.p, g.hc, g.ic, g.gk, g.dk, g.vh, g.dv),
            (128, 8, 28, 8, 16, 1, 16)
        );
        assert_eq!((g.proj(), g.conv_channels()), (66, 48));
        let a = p.attn.grid;
        assert_eq!((a.gs, a.dc, a.qh, p.attn.rot), (64, 4, 4, 64));
        assert!(g.bytes() <= p.pe_bytes, "gdn {}", g.bytes());
        assert!(p.attn.bytes() <= p.pe_bytes, "attn {}", p.attn.bytes());
        assert_eq!(
            p.kinds
                .iter()
                .filter(|k| **k == LayerKind::Attention)
                .count(),
            6
        );
        let pl = p.placement;
        assert_eq!((pl.gw * pl.gh, pl.cap), (24, 64));
        assert!(pl.width <= 755 && pl.height <= 1170, "{pl:?}");
        assert!(p.head.vr * p.head.hh >= 248_320);
        // The CSL is the one blocker left.
        assert_eq!(p.blockers.len(), 1, "{:?}", p.blockers);
        assert!(p.blockers[0].contains("not built"));
        let prm = p.params();
        let get = |k: &str| prm.iter().find(|(n, _)| *n == k).map(|(_, v)| *v);
        assert_eq!(get("KINDS"), Some(0x888888));
        assert_eq!(get("Q35"), Some(1));
    }

    /// The larger dense ones split (32 v heads over 16 k heads: VH 2), but
    /// serve2d's f32 head holds too few vocabulary rows a PE at their
    /// hidden for 248k rows beside 32 grids: they want the bf16 head.
    #[test]
    fn qwen35_4b_and_9b_want_the_bf16_head() {
        for name in ["qwen35-d4b-bf16-kv-bf16", "qwen35-d9b-bf16-kv-bf16"] {
            let s = spec(name);
            let err = Q35Plan::derive(&s, WaferDims::CS3, Goals::lanes(1)).expect_err(name);
            assert!(err.0.contains("head beside them"), "{err}");
        }
    }

    /// A non-hybrid model is not this plan's.
    #[test]
    fn a_dense_model_is_refused() {
        let s = spec("qwen3-0.6b-bf16-kv-bf16");
        assert!(Q35Plan::derive(&s, WaferDims::CS3, Goals::lanes(1)).is_err());
    }
}
