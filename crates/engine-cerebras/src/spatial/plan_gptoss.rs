//! gpt-oss on the wafer: a layer as two units and their layout, from linked
//! sizes.
//!
//! A layer is a slot of `PX` x (`ra` + `rm`) PEs: an attention unit of
//! `PX` x `ra` (qwen2d_paged8b_hr's attention: kv heads x Gs columns of Dc
//! head dims, `ceil(hidden / ra)` hidden dims a row, the layer's KV pool) over
//! a MoE grid of `PX` x `rm` (`moe_mxfp4`: every expert resident and sliced
//! like a dense MLP, `ceil(hidden / rm)` dims a row, `ceil(inter / PX)` a
//! column, so a token's top-k experts run on every PE of the grid). The
//! slots tile the wafer `across` x `down`; the bf16 head (head2d, `k` MoE rows
//! a chunk column) and the embedding strip stand east of them.
//!
//! Why two units: a layer's slot is ~30k PEs (24 layers and the head on
//! 755 x 1170), and the per-PE cost that is not weights decides the fit.
//! Measured (cslc links, `cs_readelf --syms` on an interior PE):
//! - attention and the MoE on one grid (qwen2d_paged8b_hr with the MoE as its
//!   MLP, 128 x 234, Hc 13): 34.3 KB a PE besides the experts, 52.5 KB with
//!   them even at four codes a u16. It cannot fit.
//! - the attention unit alone (gptoss_attn: no MLP, no q/k norm, long-vector
//!   softmax): 15.7 KB of code; 38,400 B a PE at Hc 74, B 1, E 7, NB 7;
//! - the MoE grid alone at Hc 15, Ic 23, 32 experts: 47,056 B a PE at B 1,
//!   +320 B a lane (24,448 B of it the experts, the rest code and buffers).
//!
//! MoE storage a PE (moe_mxfp4): each expert's gate/up tile (Hc x 2 Ic) and
//! down tile (Ic x Hc) as three planes of ceil(L / 3) input rows of MXFP4
//! codes (three a u16), then a word of two scale bases and each tile's 2R
//! scale offsets as nibbles (gpt-oss-20b's widest tile spans 14).

use super::model::{ModelSpec, Refused, refuse};
use super::plan::{Grid, HEAD16_TABLES, ROPE_POSITIONS, WaferDims};

/// PE memory less the stack's margin.
const PE_LIMIT: usize = 0xc000 - 768;

/// The MoE grid's bytes a PE besides [`MoeUnit::buffers`] and the experts:
/// code, runtime, collectives, x and the host kernel's leftovers (fitted to
/// the Hc 15, Ic 23, NE 32, B 1 link: 47,056 B).
const MOE_FIXED: usize = 17_966;
/// A lane's bytes the host kernel adds beside the MoE's own (x, its table).
const MOE_LANE_HOST: usize = 166;

/// gptoss_attn's links (no MLP, no q/k norm; its biases and sinks in) run
/// this far under paged8b's grid model plus the softmax's scratch
/// ([`AttnUnit::softmax_scratch`]); the model keeps the smallest gap.
const ATTN_LESS: usize = 2_382;

/// The embedding strip's tables a PE, at most (serve8b's: 116 dims x 132
/// rows, bf16).
const STRIP_TABLES: usize = 30_624;
/// head2d's region rows (serve8b's 1152: the feeder row in the middle).
const HEAD_ROWS: usize = 1152;
/// Head dims a column (Dc) and the KV page size the attention unit runs.
const DC: usize = 4;
const PAGE: usize = 16;

/// One MoE grid's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeUnit {
    pub px: usize,
    pub py: usize,
    pub hc: usize,
    pub ic: usize,
    pub experts: usize,
    pub top_k: usize,
    pub lanes: usize,
    pub mlp_chunk: usize,
}

impl MoeUnit {
    /// Splits `hidden` over `py` rows and `inter` over `px` columns.
    pub fn new(spec: &ModelSpec, px: usize, py: usize, lanes: usize) -> Result<MoeUnit, Refused> {
        let Some(m) = &spec.moe else {
            return refuse(format!("{}: not a mixture of experts", spec.name));
        };
        Ok(MoeUnit {
            px,
            py,
            hc: spec.hidden.div_ceil(py),
            ic: spec.inter.div_ceil(px),
            experts: m.experts,
            top_k: m.top_k,
            lanes,
            mlp_chunk: 1,
        })
    }

    fn g2(&self) -> usize {
        2 * self.ic
    }

    /// An expert's u16 words a PE (moe_mxfp4's `XW`): code planes, the base
    /// word, the scale offsets.
    pub fn expert_words(&self) -> usize {
        let g2 = self.g2();
        self.hc.div_ceil(3) * g2
            + self.ic.div_ceil(3) * self.hc
            + 1
            + (2 * g2).div_ceil(4)
            + (2 * self.hc).div_ceil(4)
    }

    /// The unit's buffers a PE: router tile, biases, decode scratch, lanes.
    pub fn buffers(&self) -> usize {
        let (g2, ne, tk, hc, ic) = (self.g2(), self.experts, self.top_k, self.hc, self.ic);
        let rmax = g2.max(hc);
        // The SwiGLU (every slot at once) borrows the decode's buffers.
        let swn = tk * ic;
        let msc = (4 * (2 * g2).div_ceil(4).max((2 * hc).div_ceil(4))).max(swn);
        let dec = (hc.div_ceil(3) * g2).max(ic.div_ceil(3) * hc).max(2 * swn);
        2 * (hc * ne + 1)
            + 4 * (ne + ne.div_ceil(self.py) * g2 + ne.div_ceil(self.px) * hc + 2)
            + 4 * (dec + msc + (2 * rmax).max(swn) + tk)
            + 4 * (self.mlp_chunk * tk * g2 + tk * ic + swn)
            + self.lanes * (4 * (ne + 1) + 2 * tk + 4 * tk)
    }

    /// The experts' bytes a PE (`mx`, u32 words).
    pub fn expert_bytes(&self) -> usize {
        4 * (self.experts * self.expert_words()).div_ceil(2)
    }

    /// A PE's bytes.
    pub fn bytes(&self) -> usize {
        MOE_FIXED
            + self.lanes.saturating_sub(1) * MOE_LANE_HOST
            + self.buffers()
            + self.expert_bytes()
    }

    pub fn pes(&self) -> usize {
        self.px * self.py
    }
}

/// One layer's attention unit: qwen2d_paged8b_hr's attention alone, bf16
/// weights and KV, pages copied by the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttnUnit {
    pub grid: Grid,
}

impl AttnUnit {
    /// `ra` rows, `lanes` lanes of `entries` pages a row, `blocks` pool pages
    /// a row.
    pub fn new(
        spec: &ModelSpec,
        ra: usize,
        lanes: usize,
        entries: usize,
        blocks: usize,
    ) -> AttnUnit {
        let gs = spec.head_dim / DC;
        AttnUnit {
            grid: Grid {
                px: spec.kv_heads * gs,
                py: ra,
                hc: spec.hidden.div_ceil(ra),
                hidden: spec.hidden,
                dc: DC,
                gs,
                qh: spec.q_heads / spec.kv_heads,
                ic: 1,
                lanes,
                page_size: PAGE,
                blocks,
                entries,
                mlp_chunk: 1,
                w16: true,
                kv16: true,
                host_copy: true,
            },
        }
    }

    /// A PE's bytes (gptoss_attn).
    pub fn bytes(&self) -> usize {
        self.grid.bytes(false) + self.softmax_scratch() - ATTN_LESS
    }

    /// The softmax's scratch (`xs`): half a PE's score rows, f32.
    pub fn softmax_scratch(&self) -> usize {
        let g = &self.grid;
        4 * (g.entries * g.page_size).div_ceil(2) * g.qh * g.lanes
    }
}

/// The whole program's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GptossLayout {
    /// Slots across and down; slot `PX` x (`ra` + `rm`).
    pub across: usize,
    pub down: usize,
    pub ra: usize,
    pub rm: usize,
    pub attn: AttnUnit,
    pub moe: MoeUnit,
    /// MoE rows a head chunk column (`head_c` columns of `k` Hc dims, + the VM
    /// column), `vr` vocabulary rows a head PE over `HEAD_ROWS` rows.
    pub k: usize,
    pub head_c: usize,
    pub vr: usize,
    /// The embedding strip: `strip_c` columns of `j` MoE rows' Hc dims.
    pub j: usize,
    pub strip_c: usize,
    pub width: usize,
    pub height: usize,
}

impl GptossLayout {
    /// The layout for `lanes` lanes of up to `context` positions: slots of
    /// as many rows as `down` bands give, the attention unit's rows (and with
    /// them the MoE grid's) chosen for the largest KV pool, the narrowest head
    /// and strip whose tables fit.
    pub fn derive(
        spec: &ModelSpec,
        wafer: WaferDims,
        lanes: usize,
        context: usize,
    ) -> Result<GptossLayout, Refused> {
        let name = &spec.name;
        if spec.moe.is_none() {
            return refuse(format!("{name}: not a mixture of experts"));
        }
        let context = context.min(ROPE_POSITIONS);
        let px = spec.kv_heads * (spec.head_dim / DC);
        let mut best: Option<((usize, usize), GptossLayout)> = None;
        let mut why = Vec::new();
        for down in 1..=spec.layers {
            let height = wafer.height / down;
            let across = spec.layers.div_ceil(down);
            for ra in 8..height {
                let rm = height - ra;
                let entries = context.div_ceil(ra * PAGE);
                // The pool: every lane's pages a row, and what else fits.
                let need = lanes * entries;
                let a0 = AttnUnit::new(spec, ra, lanes, entries, need);
                if a0.bytes() > PE_LIMIT {
                    continue;
                }
                let blocks = need + (PE_LIMIT - a0.bytes()) / a0.grid.block_bytes();
                let attn = AttnUnit::new(spec, ra, lanes, entries, blocks.max(need));
                let moe = MoeUnit::new(spec, px, rm, lanes)?;
                if moe.bytes() > PE_LIMIT {
                    why.push(format!(
                        "{down} bands, ra {ra}: MoE grid {px} x {rm} wants {} B a PE",
                        moe.bytes()
                    ));
                    continue;
                }
                let Some((k, head_c, vr)) = Self::head(spec, moe.hc, rm) else {
                    continue;
                };
                // The strip: the most MoE rows a column whose tables fit.
                let Some(j) = (1..=8).rev().find(|j| 2 * j * moe.hc * vr <= STRIP_TABLES) else {
                    continue;
                };
                let strip_c = rm.div_ceil(j);
                let width = across * px + head_c + 1 + strip_c;
                if width > wafer.width || HEAD_ROWS > wafer.height {
                    why.push(format!("{down} bands, ra {ra}: {width} columns"));
                    continue;
                }
                // The largest pool, then the MoE grid's slack.
                let slack = (blocks * ra, PE_LIMIT - moe.bytes());
                let l = GptossLayout {
                    across,
                    down,
                    ra,
                    rm,
                    attn,
                    moe,
                    k,
                    head_c,
                    vr,
                    j,
                    strip_c,
                    width,
                    height: down * height,
                };
                if best.as_ref().is_none_or(|(s, _)| slack > *s) {
                    best = Some((slack, l));
                }
            }
        }
        best.map(|(_, l)| l).ok_or_else(|| {
            Refused(format!(
                "{name}: no layout holds {} layers at {lanes} lanes of {context} positions ({})",
                spec.layers,
                why.last()
                    .map(String::as_str)
                    .unwrap_or("no attention unit fits")
            ))
        })
    }

    /// The narrowest head over `rm` MoE rows of `hc` dims: `k` rows a chunk
    /// column, its tables within head2d's.
    fn head(spec: &ModelSpec, hc: usize, rm: usize) -> Option<(usize, usize, usize)> {
        let vr = spec.vocab.div_ceil(HEAD_ROWS);
        (1..=8)
            .rev()
            .find(|k| 2 * k * hc * vr <= HEAD16_TABLES)
            .map(|k| (k, rm.div_ceil(k), vr))
    }

    /// Lanes' positions the pool holds a layer.
    pub fn pool_positions(&self) -> usize {
        self.attn.grid.blocks * self.ra * PAGE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poem_ir::Platform;

    fn spec(name: &str) -> ModelSpec {
        let t = poem_compiler::catalog::repository()
            .parse(name)
            .unwrap()
            .trace(Platform::Cerebras);
        ModelSpec::read(&t).unwrap()
    }

    fn unit(py: usize, hc: usize, ne: usize, lanes: usize, mc: usize) -> MoeUnit {
        MoeUnit {
            px: 32,
            py,
            hc,
            ic: 23,
            experts: ne,
            top_k: 4,
            lanes,
            mlp_chunk: mc,
        }
    }

    /// The MoE model reproduces its links within 48 B.
    #[test]
    fn the_moe_model_is_fitted_to_the_links() {
        let links = [
            (unit(36, 15, 32, 1, 1), 47_056),
            (unit(36, 15, 32, 2, 1), 47_408),
            (unit(36, 15, 32, 4, 1), 48_000),
            (unit(36, 15, 32, 4, 2), 48_720),
            (unit(36, 14, 32, 1, 1), 46_400),
            (unit(36, 13, 8, 1, 1), 27_488),
        ];
        for (u, top) in links {
            let d = u.bytes() as isize - top as isize;
            assert!(d.abs() <= 48, "{u:?}: model {} link {top}", u.bytes());
        }
        assert_eq!(unit(36, 15, 32, 1, 1).expert_bytes(), 24_448);
    }

    /// The attention model never runs under gptoss_attn's links, and at most
    /// 48 B over (hc, lanes, entries, blocks -> image end).
    #[test]
    fn the_attention_model_covers_the_links() {
        let s = spec("gptoss-20b-bf16-mxfp4-kv-bf16");
        let links = [
            ((74, 1, 7, 7), 38_400),
            ((74, 2, 7, 14), 46_160),
            ((74, 1, 6, 8), 37_872),
        ];
        for ((hc, b, e, nb), top) in links {
            let mut a = AttnUnit::new(&s, 2880usize.div_ceil(hc), b, e, nb);
            a.grid.hc = hc;
            let m = a.bytes();
            assert!(
                m >= top && m <= top + 48,
                "hc {hc} B {b} E {e} NB {nb}: model {m} link {top}"
            );
        }
    }

    /// gpt-oss-20b: 24 slots of attention over MoE and the head fit the
    /// wafer at 1 and 2 lanes of 4096 positions and 4 of 1024; not 4 of 4096
    /// nor 8.
    #[test]
    fn gptoss_20b_fits_the_wafer_as_attention_and_moe_units() {
        let s = spec("gptoss-20b-bf16-mxfp4-kv-bf16");
        for (lanes, ctx) in [(1, 4096), (2, 4096), (4, 1024)] {
            let l = GptossLayout::derive(&s, WaferDims::CS3, lanes, ctx).unwrap();
            assert!(
                l.across * l.down >= 24 && l.width <= 755 && l.height <= 1170,
                "{l:?}"
            );
            assert!(l.attn.bytes() <= PE_LIMIT && l.moe.bytes() <= PE_LIMIT);
            assert!(l.pool_positions() >= lanes * ctx);
            println!(
                "B {lanes} x {ctx}: {} x {} slots of 128 x ({} + {}), attention Hc {} E {} NB {} ({} B, pool {} positions), \
                 MoE Hc {} Ic {} ({} B), head {} x {} cols (k {}, Vr {}), strip {} (j {}), {} x {}",
                l.across,
                l.down,
                l.ra,
                l.rm,
                l.attn.grid.hc,
                l.attn.grid.entries,
                l.attn.grid.blocks,
                l.attn.bytes(),
                l.pool_positions(),
                l.moe.hc,
                l.moe.ic,
                l.moe.bytes(),
                l.head_c + 1,
                HEAD_ROWS,
                l.k,
                l.vr,
                l.strip_c,
                l.j,
                l.width,
                l.height
            );
        }
        // Lanes are bounded by the MoE grid's ~300 B a lane (B 8 wants 49.4 KB at
        // Hc 15) and, at long context, by the attention unit's scores.
        assert!(GptossLayout::derive(&s, WaferDims::CS3, 4, 4096).is_err());
        assert!(GptossLayout::derive(&s, WaferDims::CS3, 8, 512).is_err());
    }
}
