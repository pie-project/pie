//! The plan for Gemma 4 dense (E2B, E4B): one square grid a layer, the
//! paged attention grid's split (qwen2d_paged), in the model's layer order,
//! with each layer's per-layer embedding on its own grid.
//!
//! A unit is the paged grid with bf16 weights (paged8b's storage, f32
//! arithmetic) and what Gemma adds:
//!
//! - The reading: a sliding unit attends the last `window` positions with
//!   `Dc = head_dim / Gs` dims a column; a global unit `global_head_dim /
//!   Gs`, RoPE on the first `global_rotary / 2` pairs at span `d/2` (the
//!   placement lays each turned pair in one column, the unturned with a
//!   turn of zero, as Qwen3.5's partial RoPE).
//! - Sandwich norms: the attention's and the MLP's outputs are RMS-normed
//!   before their residual adds (two more `[Hc]` gammas and an all-reduce
//!   each), v is RMS-normed without a weight, the softmax scale is 1 (the
//!   placement folds the kernels' `1/sqrt(d)` back into q's norm weights).
//! - A GELU-tanh GLU (`Ic = inter_l / P`: E2B's shared tail runs twice as
//!   wide).
//! - The PLE block: column `c` holds `ple_dim / P` dims of the layer's
//!   table for row `r`'s vocab range (`vocab / P` rows); a token's row comes
//!   down each column from the row that holds it. The context half projects
//!   the embedding `x0`, which rides the handoff beside `x` (`Hc` a row), as
//!   q does `x` (partials down a column), normed over `ple_dim` across the
//!   row. Then `gelu(gate x) * ple` projects back as o does and normed, and
//!   the layer's scalar scales the residual.
//! - The shared tail: an owning layer the tail reads (E2B: 13 and 14) puts
//!   the token's k and v (`Dc` each a column) on the handoff; each tail
//!   layer appends them to its own pool, so a tail unit is a q-only unit
//!   with a copy of its source's kv.
//!
//! The head is serve8b's bf16 head2d over the tied embedding, scaling the
//! gathered embedding by `sqrt(hidden)`; the softcap is monotone, so the
//! greedy argmax needs nothing, and the VM's sampling reads `cap · tanh(z /
//! cap)` ([`G4Need::SoftcapSampling`]).

use super::model::{Refused, refuse};
use super::model_g4::{G4Spec, Reading};
use super::plan::{Goals, Grid, WaferDims};

/// What the units don't do yet that the plan already sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum G4Need {
    /// The sliding window's lower bound in the visibility bitmaps: the host
    /// clears bits below `pos + 1 - window` (pages wholly below it leave
    /// the lane's entries). Placement only, but untested past 512 tokens.
    WindowBits,
    /// Sampling over softcapped logits (the VM's top-p/temperature path).
    SoftcapSampling,
}

/// One layer's unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct G4Unit {
    pub grid: Grid,
    pub reading: Reading,
    /// Dims a head RoPE turns.
    pub rot: usize,
    /// The layer computes k and v (else it appends its source's from the
    /// handoff).
    pub owns_kv: bool,
    /// The layer's k and v ride the handoff to the shared tail.
    pub shares_kv: bool,
    /// PLE dims a column and vocab rows a PE row (0: no PLE).
    pub ple_c: usize,
    pub ple_rows: usize,
}

impl G4Unit {
    /// A PE's estimated bytes: the paged grid's at bf16 weights, plus the
    /// sandwich norms' gammas, the PLE table slice, its three projections'
    /// slices (bf16), the handoff's `x0` and the shared kv.
    pub fn bytes(&self) -> usize {
        let g = &self.grid;
        let mut b = g.bytes(true) + 4 * 3 * g.hc;
        if !self.owns_kv {
            // No k/v projection: wqkv holds q alone.
            b -= 2 * g.hc * 2 * g.dc;
        }
        if self.ple_c > 0 {
            b += 2 * self.ple_rows * self.ple_c // the table
                + 2 * 3 * g.hc * self.ple_c // model_proj, gate, proj slices
                + 4 * (g.hc + 1) // the PLE norm's gamma and the scalar
                + 4 * g.lanes * (g.hc + 3 * self.ple_c); // x0, relay, row, gate
        }
        if self.shares_kv || !self.owns_kv {
            b += 4 * g.lanes * 2 * g.dc;
        }
        b
    }
}

/// Gemma 4 on the wafer.
#[derive(Debug, Clone, PartialEq)]
pub struct G4Plan {
    pub spec: G4Spec,
    pub units: Vec<G4Unit>,
    /// Grids across and down (serpentine), the grids' rectangle.
    pub gw: usize,
    pub gh: usize,
    /// Columns east of the grids left for the head, and the head's bytes a
    /// PE there (bf16 embedding rows only).
    pub head_cols: usize,
    pub head_bytes: usize,
    pub pe_bytes: usize,
    pub needs: Vec<G4Need>,
    /// Why no program places the model (empty when one does).
    pub blockers: Vec<String>,
}

/// The bf16 head's fixed bytes a PE (head2d_v10 as serve8b builds it,
/// less its tiles).
const HEAD_FIXED: usize = 14_000;

impl G4Plan {
    pub fn derive(spec: &G4Spec, wafer: WaferDims, goals: Goals) -> Result<G4Plan, Refused> {
        let name = &spec.name;
        if goals.lanes == 0 || goals.page_size == 0 {
            return refuse("a plan wants at least one lane and a page size");
        }
        // The grid's side: the narrowest that splits every dim, a kv head's
        // columns holding an even number of each reading's dims.
        let p = 128;
        let gs = p / spec.kv_heads;
        let split = |d: usize| d.is_multiple_of(gs) && (d / gs).is_multiple_of(2);
        if !spec.hidden.is_multiple_of(p)
            || !p.is_multiple_of(spec.kv_heads)
            || spec.global_kv_heads != spec.kv_heads
            || !split(spec.head_dim)
            || !split(spec.global_head_dim)
            || spec.inter.iter().any(|i| !i.is_multiple_of(p))
        {
            return refuse(format!(
                "{name}: hidden {}, {} kv heads of {}/{} or the MLP widths do not split over {p}",
                spec.hidden, spec.kv_heads, spec.head_dim, spec.global_head_dim
            ));
        }
        let (ple_c, ple_rows) = match spec.ple_dim {
            Some(d) if d.is_multiple_of(p) && spec.vocab.is_multiple_of(p) => {
                (d / p, spec.vocab / p)
            }
            Some(d) => return refuse(format!("{name}: PLE width {d} does not split over {p}")),
            None => (0, 0),
        };
        let lanes = goals.lanes;
        let mlp_chunk = goals
            .mlp_chunk
            .unwrap_or(if lanes.is_multiple_of(2) { 2 } else { 1 });
        let shared: Vec<bool> = (0..spec.layers())
            .map(|l| {
                spec.kv_source
                    .iter()
                    .enumerate()
                    .any(|(t, &s)| s == l && t != l)
            })
            .collect();
        let units: Vec<G4Unit> = (0..spec.layers())
            .map(|l| {
                let (d, rot) = match spec.readings[l] {
                    Reading::Sliding => (spec.head_dim, spec.head_dim),
                    Reading::Global => (spec.global_head_dim, spec.global_rotary),
                };
                // A sliding pool holds the window (pages a lane: window / PS + 1).
                let blocks = goals.blocks.unwrap_or(match spec.readings[l] {
                    Reading::Sliding => (spec.window / goals.page_size + 1).div_ceil(p).max(1),
                    Reading::Global => 1,
                });
                G4Unit {
                    grid: Grid {
                        px: p,
                        py: p,
                        hc: spec.hidden / p,
                        hidden: spec.hidden,
                        dc: d / gs,
                        gs,
                        qh: spec.q_heads / spec.kv_heads,
                        ic: spec.inter[l] / p,
                        lanes,
                        page_size: goals.page_size,
                        blocks,
                        entries: 1,
                        mlp_chunk,
                        w16: true,
                        kv16: false,
                        host_copy: goals.host_copy,
                    },
                    reading: spec.readings[l],
                    rot,
                    owns_kv: spec.kv_source[l] == l,
                    shares_kv: shared[l],
                    ple_c,
                    ple_rows,
                }
            })
            .collect();
        let mut blockers = Vec::new();
        for (l, u) in units.iter().enumerate() {
            if u.bytes() > wafer.pe_bytes {
                blockers.push(format!(
                    "layer {l}'s unit wants {} bytes a PE (of {}) at B {lanes}",
                    u.bytes(),
                    wafer.pe_bytes
                ));
            }
        }
        // Grids serpentine, the head in the columns east of them: the
        // fewest grids across whose rows fit the height (the last row may
        // be part empty), leaving the head the most columns.
        let layers = spec.layers();
        let Some(gw) =
            (1..=layers).find(|w| layers.div_ceil(*w) * p <= wafer.height && w * p < wafer.width)
        else {
            return refuse(format!(
                "{name}: {layers} grids of {p} x {p} do not fit the wafer"
            ));
        };
        let gh = layers.div_ceil(gw);
        let head_cols = wafer.width - gw * p;
        let head_pes = head_cols * wafer.height;
        let head_bytes = HEAD_FIXED + (2 * spec.vocab * spec.hidden).div_ceil(head_pes);
        if head_bytes > wafer.pe_bytes {
            blockers.push(format!(
                "the head's {} bf16 rows over {head_cols} x {} PEs want {head_bytes} bytes a PE",
                spec.vocab, wafer.height
            ));
        }
        let mut needs = vec![G4Need::SoftcapSampling];
        if units.iter().any(|u| u.reading == Reading::Sliding) {
            needs.insert(0, G4Need::WindowBits);
        }
        Ok(G4Plan {
            spec: spec.clone(),
            units,
            gw,
            gh,
            head_cols,
            head_bytes,
            pe_bytes: wafer.pe_bytes,
            needs,
            blockers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::model_g4::G4Spec;

    fn spec(name: &str) -> G4Spec {
        let catalog = poem_compiler::catalog::repository();
        G4Spec::read(
            &catalog
                .parse(name)
                .expect(name)
                .trace(poem_ir::Platform::Cerebras),
        )
        .expect(name)
    }

    fn show(p: &G4Plan) {
        let mut seen = Vec::new();
        for (l, u) in p.units.iter().enumerate() {
            let key = (u.reading, u.owns_kv, u.shares_kv, u.grid.ic);
            if !seen.contains(&key) {
                seen.push(key);
                println!(
                    "  layer {l:2}: {:?} Dc {} Ic {} owns {} shares {} blocks {} -> {} B",
                    u.reading,
                    u.grid.dc,
                    u.grid.ic,
                    u.owns_kv,
                    u.shares_kv,
                    u.grid.blocks,
                    u.bytes()
                );
            }
        }
        println!(
            "  grids {} x {}, head {} cols at {} B/PE, needs {:?}, blockers {:?}",
            p.gw, p.gh, p.head_cols, p.head_bytes, p.needs, p.blockers
        );
    }

    /// E2B: every unit fits a PE at B 1, the 35 grids sit 4 x 9 with 243
    /// columns east for the head.
    #[test]
    fn e2b_plans() {
        let s = spec("gemma4-e2b-bf16-kv-bf16");
        let p = G4Plan::derive(&s, WaferDims::CS3, Goals::lanes(1)).expect("plans");
        show(&p);
        let u = |l: usize| p.units[l];
        assert_eq!(
            (u(0).grid.dc, u(4).grid.dc, u(0).grid.gs, u(0).grid.qh),
            (2, 4, 128, 8)
        );
        assert_eq!(
            (u(0).grid.ic, u(15).grid.ic, u(0).ple_c, u(0).ple_rows),
            (48, 96, 2, 2048)
        );
        assert!(u(13).shares_kv && u(14).shares_kv && !u(12).shares_kv && !u(15).owns_kv);
        assert_eq!((p.gw, p.gh, p.head_cols), (4, 9, 243));
        assert!(p.blockers.is_empty(), "{:?}", p.blockers);
    }

    /// Prints E2B's and E4B's plans at 1 and 4 lanes (`--nocapture`).
    #[test]
    fn plans_print() {
        for name in ["gemma4-e2b-bf16-kv-bf16", "gemma4-e4b-bf16-kv-bf16"] {
            for b in [1, 4] {
                println!("{name} B {b}:");
                show(&G4Plan::derive(&spec(name), WaferDims::CS3, Goals::lanes(b)).expect(name));
            }
        }
    }
}
