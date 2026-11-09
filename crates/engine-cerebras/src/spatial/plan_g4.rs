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
    /// A PE's estimated bytes, fitted to the linked ELFs of gemma4_layer at
    /// E2B's dims (`G4_FIXED`: its code, task table and fixed data): the
    /// paged grid's data at bf16 weights, plus the sandwich norms' gammas,
    /// the three PLE projections' slices (bf16), the handoff's `x0` and PLE
    /// row, and the shared kv. The PLE tables live in the head region.
    pub fn bytes(&self) -> usize {
        let g = &self.grid;
        let mut b = G4_FIXED + G4_LANE * g.lanes + 26 * g.ic + 228 * g.dc + g.bytes(false)
            - g.fixed_bytes()
            + 4 * 3 * g.hc;
        if !self.owns_kv {
            // No k/v projection: wqkv holds q alone.
            b -= 2 * g.hc * 2 * g.dc;
        }
        if self.ple_c > 0 {
            b += 2 * 3 * g.hc * self.ple_c // model_proj, gate, proj slices
                + 4 * (g.hc + 1) // the PLE norm's gamma and the scalar
                + 4 * g.lanes * (g.hc + 4 * self.ple_c); // x0, prow, relay, row, gate
        }
        if self.shares_kv || !self.owns_kv {
            b += 4 * g.lanes * 2 * g.dc;
        }
        b
    }
}

/// gemma4_layer's fixed bytes a PE (its code is ~27.4 KB: qwen35_attn's
/// plus the norms, GELU, PLE and the W16 matvec), and what the grid's data
/// terms leave out a lane (scratch the unit adds) and an `ic` and a `dc`
/// (the MLP's and the heads' buffers), fitted to the linked sizes in
/// [`tests::units_match_their_linked_size`].
const G4_FIXED: usize = 27_846;
const G4_LANE: usize = 1_108;

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
    /// serve_g4's layout of the plan (`None`: blocked).
    pub program: Option<G4Program>,
    /// Why no program places the model (empty when one does).
    pub blockers: Vec<String>,
}

/// serve_g4's head and PLE region: head2d's one f32 block of `p` chunk
/// columns and `vmc` (its stand-in VM column) east of the grids, `hh`
/// region rows of `vr` vocabulary rows (the feeder row between them), its
/// root's `gx` x `gy` groups; east of it `npc` columns of PLE PEs holding
/// `rpp` vocabulary rows of their strip's tables; `cap` tokens a launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct G4Program {
    pub hh: usize,
    pub vr: usize,
    pub gx: usize,
    pub gy: usize,
    pub gb: usize,
    pub vmc: usize,
    pub npc: usize,
    pub rpp: usize,
    pub cap: usize,
    /// Bytes a PE: head2d's block, a PLE PE.
    pub head_pe: usize,
    pub ple_pe: usize,
}

/// head2d's f32 block's bytes a PE less its tiles (`Hc Vr + 1` words):
/// linked at E2B (37,648 bytes at Hc 12, Vr 228).
const HEAD_FIXED: usize = 26_700;
/// A PLE PE's bytes less its table slice (`RPP x NWW` words): linked at E2B
/// (42,736 bytes at RPP 18, NWW 512).
const PLE_FIXED: usize = 5_872;

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
        // head2d's block (f32 tiles of the tied embedding): `hh` region rows
        // and the feeder row, as tall as the grids (even); its stand-in VM
        // column; then the PLE region in the columns left.
        let vmc = 1;
        let hh = (gh * p).next_multiple_of(2);
        let vr = spec.vocab.div_ceil(hh);
        let hc = spec.hidden / p;
        let head_pe = HEAD_FIXED + 4 * (hc * vr + 1);
        let npc = head_cols.saturating_sub(p + vmc);
        let nww = gw * p * ple_c / 2;
        let rpp = if npc == 0 {
            0
        } else {
            spec.vocab.div_ceil(npc * p)
        };
        let ple_pe = PLE_FIXED + 4 * rpp * nww;
        let head_bytes = head_pe.max(ple_pe);
        if hh + 1 > wafer.height || npc == 0 {
            blockers.push(format!(
                "head2d's {hh} rows and the PLE region do not fit {} x {} beside {gw} x {gh} grids",
                wafer.width, wafer.height
            ));
        }
        if head_pe > wafer.pe_bytes || ple_pe > wafer.pe_bytes {
            blockers.push(format!(
                "the head ({head_pe} bytes a PE: {vr} vocabulary rows) or the PLE region ({ple_pe}: \
                 {rpp} rows over {npc} columns) over {} bytes",
                wafer.pe_bytes
            ));
        }
        // head2d's root groups: Gx 16 columns, Gy 18 rows a group dividing
        // the feeder's halves.
        let (gx, gy, gb) = (16, 18, 1);
        if !p.is_multiple_of(gx) || !(hh / 2).is_multiple_of(gy) {
            blockers.push(format!(
                "head2d's groups ({gx} x {gy}) do not split {p} columns and {} rows",
                hh / 2
            ));
        }
        let program = blockers.is_empty().then_some(G4Program {
            hh,
            vr,
            gx,
            gy,
            gb,
            vmc,
            npc,
            rpp,
            cap: goals.cap.unwrap_or(64),
            head_pe,
            ple_pe,
        });
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
            program,
            blockers,
        })
    }

    /// serve_g4's parameters (`layout.csl`'s order), `None` when no program
    /// places the plan.
    #[must_use]
    pub fn params(&self) -> Option<Vec<(&'static str, usize)>> {
        let pr = self.program.as_ref()?;
        let (s, u0) = (&self.spec, &self.units[0]);
        let g = &u0.grid;
        let nown = s.owners();
        if (0..s.layers()).any(|l| (l < nown) != (s.kv_source[l] == l)) {
            return None;
        }
        let kinds = (0..s.layers())
            .filter(|&l| s.readings[l] == Reading::Global)
            .fold(0usize, |m, l| m | 1 << l);
        let dc = |r: Reading| {
            self.units
                .iter()
                .find(|u| u.reading == r)
                .map_or(0, |u| u.grid.dc)
        };
        let ic_tail = self.units.get(nown).map_or(g.ic, |u| u.grid.ic);
        let f32_bits = |x: f32| x.to_bits() as usize;
        let ple = s.ple_dim.unwrap_or(0) as f32;
        Some(vec![
            ("P", g.px),
            ("Hc", g.hc),
            ("Gs", g.gs),
            ("Qh", g.qh),
            ("DcS", dc(Reading::Sliding)),
            ("DcG", dc(Reading::Global)),
            ("IcS", g.ic),
            ("IcT", ic_tail),
            ("PC", u0.ple_c),
            ("KINDS", kinds),
            ("NL", s.layers()),
            ("NOWN", nown),
            ("PS", g.page_size),
            (
                "NB",
                self.units.iter().map(|u| u.grid.blocks).max().unwrap_or(1),
            ),
            ("E", g.entries),
            ("B", g.lanes),
            ("MC", g.mlp_chunk),
            ("GW", self.gw),
            ("GH", self.gh),
            ("ESB", f32_bits((s.hidden as f32).sqrt())),
            (
                "PSB",
                f32_bits(ple.sqrt() * std::f32::consts::FRAC_1_SQRT_2),
            ),
            ("NP", 3),
            ("F", pr.hh / 2),
            ("HH", pr.hh),
            ("Vr", pr.vr),
            ("Gx", pr.gx),
            ("Gy", pr.gy),
            ("Gb", pr.gb),
            ("vocab", s.vocab),
            // head2d's f32 head as serve2d builds it: three passes, exact.
            ("PASSES", 3),
            ("UNTIED", 0),
            ("CERT", 0),
            ("MODE", 0),
            ("OVL", 0),
            ("VMC", pr.vmc),
            ("WF32", 1),
            ("CAP", pr.cap),
            ("NPC", pr.npc),
            ("RPP", pr.rpp),
            ("WIN", s.window),
            // One kv head spans the grid: all-reduces in groups of 16
            // (0386: 499 vs 434 tok/s, the same tokens).
            ("G1", if g.gs == g.px { 16 } else { 0 }),
        ])
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

    /// E2B's params at B 1: the program built and verified on the CS-3
    /// (plus the sliding window and G1: 0386),
    /// read back as a serve_g4 program.
    #[test]
    fn e2b_params_are_the_chip_programs() {
        use crate::spatial::codegen::Design;
        use crate::spatial::program::ProgramSpec;
        let s = spec("gemma4-e2b-bf16-kv-bf16");
        let p = G4Plan::derive(&s, WaferDims::CS3, Goals::lanes(1)).expect("plans");
        let text: Vec<String> = p
            .params()
            .expect("a program")
            .iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect();
        assert_eq!(
            text.join(","),
            "P:128,Hc:12,Gs:128,Qh:8,DcS:2,DcG:4,IcS:48,IcT:96,PC:2,KINDS:17734058512,NL:35,\
             NOWN:15,PS:16,NB:1,E:1,B:1,MC:1,GW:4,GH:9,ESB:1109181553,PSB:1093993715,NP:3,F:576,\
             HH:1152,Vr:228,Gx:16,Gy:18,Gb:1,vocab:262144,PASSES:3,UNTIED:0,CERT:0,MODE:0,OVL:0,\
             VMC:1,WF32:1,CAP:64,NPC:114,RPP:18,WIN:512,G1:16"
        );
        let pr = p.program.expect("a program");
        assert_eq!((pr.head_pe, pr.ple_pe), (37_648, 42_736));
        let params: Vec<(String, String)> = p
            .params()
            .expect("a program")
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        assert_eq!(ProgramSpec::infer_design(&params), Ok(Design::ServeG4));
        let prog = ProgramSpec::from_params(Design::ServeG4, &params).expect("a program");
        assert_eq!(prog.to_params(), params);
        assert_eq!(prog.rectangle(), (755, 1153));
        assert_eq!(prog.layers(), 35);
        let g = prog.gemma().expect("gemma");
        assert_eq!((g.source(15), g.source(19), g.source(34)), (13, 14, 14));
        assert_eq!((g.dc(4), g.dc(5), g.ic(14), g.ic(15)), (4, 2, 48, 96));
        // Strips 0-7 whole, strip 8's three layers east of its filler.
        let areas: Vec<(i32, i32, i32, i32)> = prog
            .grid_areas()
            .iter()
            .map(|r| (r.x, r.y, r.w, r.h))
            .collect();
        assert_eq!(areas, vec![(0, 0, 512, 1024), (128, 1024, 384, 128)]);
        assert_eq!(prog.origin(32), (384, 1024));
        assert_eq!(prog.origin(34), (128, 1024));
        assert_eq!(prog.root(), (512 + 72, 576));
        assert_eq!(prog.positions(), 2048);
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

    /// The unit's bytes against gemma4_layer linked at E2B's dims
    /// (SDK 2.10.1: the ELF's allocated sections): never
    /// under, at most 1.5 KB over.
    #[test]
    fn units_match_their_linked_size() {
        let p = G4Plan::derive(
            &spec("gemma4-e2b-bf16-kv-bf16"),
            WaferDims::CS3,
            Goals::lanes(1),
        )
        .expect("plans");
        let at = |l: usize, lanes: usize, mc: usize, shares: bool| {
            let mut u = p.units[l];
            u.grid.lanes = lanes;
            u.grid.mlp_chunk = mc;
            u.shares_kv = shares;
            u.bytes()
        };
        // (layer, lanes, MLP chunk, shares kv, linked bytes)
        for (l, lanes, mc, shares, linked) in [
            (0, 1, 1, true, 37_434),   // sliding, Dc 2, Ic 48
            (4, 1, 1, false, 39_530),  // global, Dc 4, Ic 48
            (19, 1, 1, false, 44_618), // shared tail, global, Ic 96
            (19, 2, 2, false, 47_614),
            (4, 4, 2, true, 46_034),
        ] {
            let est = at(l, lanes, mc, shares);
            assert!(
                est >= linked && est <= linked + 1536,
                "layer {l} B {lanes}: {est} vs {linked}"
            );
        }
    }
}
