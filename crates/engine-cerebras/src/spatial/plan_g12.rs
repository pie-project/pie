//! Gemma 4 dense in u4g64 on serve8b_region (`G12 1`): the planner. A
//! [`G4Spec`] without per-layer embeddings or a shared kv tail (Gemma-4-12B)
//! takes gemma4_unit grids and regions (`codegen::serve8b_region_g12`), the
//! u4g64 head and strip (`W4H 1`), B 1.
//!
//! Shapes: `PX` columns hold the sliding kv heads (`Dc = kv_heads head_dim /
//! PX` dims a column, `Gs = head_dim / Dc` a head, `Qh` query heads a kv
//! head); a global layer's one kv head goes on every copy of `GsG =
//! global_head_dim / Dc` columns (`PX / GsG` copies of `QhG` query heads):
//! the same `Dc` for both kinds, so one pool and one lane layout. The
//! geometry is serve8b_region's (`build::place_at`'s rules: a band of grids,
//! `RC x RR` regions of `RNL` layers, the column of `NCOL` grids, the head
//! and the strip east), searched over grid rows and region shapes; every
//! unit image kind (grid or region by its layers' kinds) must fit its PE by
//! [`unit_bytes`] (the u4 symbols exact, the tables, pool and buffers by the
//! kernel's sizes, its code calibrated by link probes), the head and the
//! strip by `plan::final_root_bytes` / `plan::embed_bytes`.

use super::model_g4::{G4Spec, Reading};
use super::plan::{Goals, WaferDims};
use super::serve8b::Kernel;
use super::u4g12::{self, G12, KindShape, RB};

/// A unit PE's image must end this far below the top of its memory (the
/// stack is not counted by the linker; `build::STACK_MARGIN`).
const STACK_MARGIN: usize = 768;
/// The regions' Y phase-1 group (it must divide `RPY`) and the grids'.
const RGY: usize = 16;
/// The u4 decode buffer (u16 words): 512 frees 1 KB a grid PE at 12B's tiles.
const W4BUF: usize = 512;
/// Grid columns: 128, as serve8b_region's searches.
const PX: usize = 128;

/// gemma4_unit's code and fixed data a PE past what [`unit_bytes`] counts,
/// by image class: link probe 2026-11-05 (scratch/g12/lp/g12e, with the c_bc drains: Gemma-4-12B,
/// grids 128 x 96 Hc 40, regions 128 x 240 RHc 16 RNL 2, NB 1, PS 16, WIN
/// 1024, W4BUF 512, KVST 0): grid sliding 44,416 B, grid global 47,120,
/// region SS 43,920, region GS 46,720; each constant makes its class exact
/// there. A unit with a global layer lays its lanes out at QhG heads and,
/// mixed, carries the spread and both kinds' route words: more code.
const CODE_GRID_S: usize = 24_402;
const CODE_GRID_G: usize = 24_930;
const CODE_REGION_S: usize = 24_970;
const CODE_REGION_MIX: usize = 25_738;

/// A placement: the grid and region kernels, the counts, the head's rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct G12Shape {
    pub grid: Kernel,
    pub region: Kernel,
    pub rnl: usize,
    pub rc: usize,
    pub rr: usize,
    pub band: usize,
    pub ncol: usize,
    pub hh: usize,
    pub vr: usize,
    pub gx: usize,
    pub gy: usize,
    pub cap: usize,
    /// Each image kind's bytes (grid S/G, region by kinds), the head's
    /// final root and the strip.
    pub bytes: Vec<(String, usize)>,
    pub cost: usize,
}

/// The plan: the model's facts in gemma4_unit's terms and its placement.
#[derive(Debug, Clone, PartialEq)]
pub struct G12Plan {
    pub name: String,
    pub hidden: usize,
    pub vocab: usize,
    pub layers: usize,
    pub dc: usize,
    pub gs: usize,
    pub qh: usize,
    pub ic: usize,
    pub knobs: G12,
    pub page_size: usize,
    pub blocks: usize,
    pub shape: Option<G12Shape>,
    pub blockers: Vec<String>,
}

/// The kernel's buffers that scale with the shape, its tables and pool (the
/// rest is [`unit_bytes`]'s code constant).
fn data_bytes(
    k: Kernel,
    glob: &[bool],
    ks: &[KindShape; 2],
    nb: usize,
    ps: usize,
    np: usize,
) -> usize {
    let (hc, dc, ic, dh, nl) = (k.hc, k.dc, k.ic, k.dc / 2, glob.len());
    let qm = u4g12::unit_qm(ks, glob);
    let nh = qm + 1;
    let nk = if glob.iter().any(|g| *g) && glob.iter().any(|g| !g) {
        2
    } else {
        1
    };
    let t = [u4g12::tiles(k, ks[0]), u4g12::tiles(k, ks[1])];
    let all = t.iter().flat_map(|t| [t[0], t[1], t[2], t[4]]);
    let nmax = all.clone().map(|t| t.n).max().unwrap_or(0);
    let pl = all
        .map(|t| t.chunk_in(W4BUF) * (t.n + 1))
        .max()
        .unwrap_or(0);
    let xmax = hc.max(qm * dc).max(ic).max(ps).max(dh);
    let tables = 4 * nl * hc + 4 * nl * (3 * hc + 1) + 4 * nl * 2 * nh * dh + 2 * 4 * nk * RB * dh;
    let pool = 2 * 2 * (nl * nb * dc * ps + 1);
    let lane = 4 * (hc + 2 * nh * dh + dc + 1) + 4 * (2 * ic + 1) + 4 * ic.max(dh);
    let u4 = 2 * (pl + 1) + 4 * (nmax + 1) + 4 * nmax + 2 * 2 * np * xmax + 2 * np * xmax;
    tables + pool + lane + u4
}

/// A unit PE's bytes: its u4 symbols (exactly the kernel's), tables, pool,
/// buffers and its class's code.
#[must_use]
pub fn unit_bytes(
    k: Kernel,
    glob: &[bool],
    ks: &[KindShape; 2],
    nb: usize,
    ps: usize,
    np: usize,
) -> usize {
    let weights: usize = [0, 1, 2, 4]
        .iter()
        .map(|&m| 2 * (u4g12::symbol_words(k, ks, glob, W4BUF, m) + 1))
        .sum::<usize>()
        + 2 * 4;
    let (has_s, has_g) = (glob.iter().any(|g| !g), glob.iter().any(|g| *g));
    let code = match (glob.len() > 1, has_s, has_g) {
        (false, _, false) => CODE_GRID_S,
        (false, _, true) => CODE_GRID_G,
        (true, _, false) => CODE_REGION_S,
        (true, _, true) => CODE_REGION_MIX,
    };
    code + weights + data_bytes(k, glob, ks, nb, ps, np)
}

/// A unit's cost for ranking placements (as `build::unit_cost`: the u4
/// elements a PE at ~2 cycles each, the tiles' group epilogues and
/// segments, the collectives' hops).
fn unit_cost(k: Kernel, s: KindShape) -> usize {
    let t = u4g12::tiles(k, s);
    let elems: usize = [0, 1, 2, 3, 4].iter().map(|&m| t[m].r * t[m].n).sum();
    let groups: usize = [0, 1, 2, 3, 4]
        .iter()
        .map(|&m| t[m].g * (120 + 2 * t[m].n) + 60 * t[m].segs_in(W4BUF))
        .sum();
    8000 + elems * 2 + 20 * (k.px + k.py) + groups
}

impl G12Plan {
    /// The plan for `spec` (B 1, one KV block unless pinned).
    #[must_use]
    pub fn derive(spec: &G4Spec, wafer: WaferDims, goals: Goals) -> G12Plan {
        let mut blockers = Vec::new();
        let layers = spec.layers();
        if goals.lanes != 1 || goals.pipeline {
            blockers.push(format!(
                "gemma4_unit runs B 1 without a ring (asked B {})",
                goals.lanes
            ));
        }
        if spec.ple_dim.is_some() {
            blockers.push("per-layer embeddings (serve_g4 places them)".into());
        }
        if (0..layers).any(|l| spec.kv_source[l] != l) {
            blockers.push("a shared kv tail (serve_g4 places it)".into());
        }
        if spec.inter.iter().any(|&i| i != spec.inter[0]) {
            blockers.push("MLP widths differ by layer".into());
        }
        if spec.global_kv_heads != 1 && spec.readings.contains(&Reading::Global) {
            blockers.push(format!(
                "{} global kv heads (one is replicated)",
                spec.global_kv_heads
            ));
        }
        if layers > 64 {
            blockers.push("more than 64 layers (KINDS is a u64)".into());
        }
        let kvd = spec.kv_heads * spec.head_dim;
        let dc = kvd / PX;
        let gs = if dc > 0 { spec.head_dim / dc } else { 0 };
        let gsg = if dc > 0 { spec.global_head_dim / dc } else { 0 };
        let copies = if gsg > 0 { PX / gsg } else { 0 };
        if dc * PX != kvd
            || dc % 2 != 0
            || gs * dc != spec.head_dim
            || gsg * dc != spec.global_head_dim
            || copies == 0
            || PX % gsg != 0
            || spec.q_heads % copies != 0
            || spec.q_heads % spec.kv_heads != 0
        {
            blockers.push(format!(
                "kv {kvd} dims over {PX} columns: Dc {dc}, head_dim {} / global {} do not split evenly",
                spec.head_dim, spec.global_head_dim
            ));
        }
        let ic = spec.inter[0] / PX;
        if ic * PX != spec.inter[0] {
            blockers.push(format!("inter {} over {PX} columns", spec.inter[0]));
        }
        let kinds: u64 = (0..layers.min(64))
            .filter(|&l| spec.readings[l] == Reading::Global)
            .map(|l| 1u64 << l)
            .sum();
        let knobs = G12 {
            kinds,
            qhg: if copies > 0 { spec.q_heads / copies } else { 0 },
            gsg,
            win: spec.window,
            w4buf: W4BUF,
            esb: (spec.hidden as f32).sqrt().to_bits(),
            kv16: true,
        };
        let mut plan = G12Plan {
            name: spec.name.clone(),
            hidden: spec.hidden,
            vocab: spec.vocab,
            layers,
            dc,
            gs,
            qh: if spec.kv_heads > 0 {
                spec.q_heads / spec.kv_heads
            } else {
                0
            },
            ic,
            knobs,
            page_size: goals.page_size,
            blocks: goals.blocks.unwrap_or(1),
            shape: None,
            blockers,
        };
        if plan.blockers.is_empty() {
            plan.shape = plan.search(wafer);
            if plan.shape.is_none() {
                plan.blockers
                    .push("no serve8b_region placement of gemma4_unit units fits".into());
            }
        }
        plan
    }

    fn ks(&self) -> [KindShape; 2] {
        u4g12::kinds(self.qh, self.gs, &self.knobs)
    }

    /// The units' layers in serve8b_region's order: band grids, then the
    /// regions (serpentine by columns), then the column's grids.
    fn unit_layers(
        &self,
        band: usize,
        nreg: usize,
        rnl: usize,
        ncol: usize,
    ) -> (Vec<Vec<bool>>, Vec<Vec<bool>>) {
        let g = |l: usize| self.knobs.global(l);
        let mut grids: Vec<Vec<bool>> = (0..band).map(|l| vec![g(l)]).collect();
        let regions: Vec<Vec<bool>> = (0..nreg)
            .map(|r| (0..rnl).map(|j| g(band + r * rnl + j)).collect())
            .collect();
        grids.extend((0..ncol).map(|k| vec![g(band + nreg * rnl + k)]));
        (grids, regions)
    }

    /// The placement at grid rows `py` and these regions, if it fits.
    fn place(
        &self,
        wafer: WaferDims,
        py: usize,
        (rpy, rnl, rc, rr): (usize, usize, usize, usize),
    ) -> Option<G12Shape> {
        let hid = self.hidden;
        let (hc, rhc) = (hid.div_ceil(py), hid.div_ceil(rpy));
        let grid = Kernel {
            px: PX,
            py,
            hc,
            dc: self.dc,
            gs: self.gs,
            ic: self.ic,
        };
        let region = Kernel {
            px: PX,
            py: rpy,
            hc: rhc,
            dc: self.dc,
            gs: self.gs,
            ic: self.ic,
        };
        let band = rc + 1;
        let ncol = self.layers.checked_sub(band + rc * rr * rnl)?;
        let (c, vmc) = (py / 2, 1);
        let hh = py * (1 + ncol);
        let vr = self.vocab.div_ceil(hh);
        let vr = vr + vr % 2;
        let f = hh / 2;
        let width = rc * PX + PX + c + vmc + py / 4;
        // The layout's and the handoffs' rules (serve8b_region::Tables: the
        // last region hands down into the column's first grid from its rows).
        if py % 4 != 0
            || py > 2 * PX
            || rpy % RGY != 0
            || py % RGY != 0
            || rpy > 2 * PX
            || py > PX
            || !(rc % 2 == 0 || rr == 1)
            || c * 2 * hc < hid
            || width > wafer.width
            || hh + 1 > wafer.height
            || py + rr * rpy > wafer.height
            || py * (1 + ncol) > wafer.height
            || f < py
            || f >= py * ncol
        {
            return None;
        }
        // The head's groups: Gx divides C with its final root clear of the
        // VM column; Gy the divisor of F nearest its square root.
        let gx = [8, 4, 2]
            .into_iter()
            .find(|&gx| c % gx == 0 && (c / gx / 2) * gx + gx / 2 + 1 < c)?;
        let gy = (1..=f)
            .filter(|d| f % d == 0)
            .min_by_key(|&d| (d * d).abs_diff(f))?;
        let most = wafer.pe_bytes - STACK_MARGIN;
        let ks = self.ks();
        let (grids, regions) = self.unit_layers(band, rc * rr, rnl, ncol);
        let mut bytes: Vec<(String, usize)> = Vec::new();
        let tag = |g: &[bool]| {
            g.iter()
                .map(|&x| if x { 'G' } else { 'S' })
                .collect::<String>()
        };
        for (k, units, what) in [(grid, &grids, "grid"), (region, &regions, "region")] {
            for u in units {
                let name = format!("{what} {}", tag(u));
                if bytes.iter().any(|b| b.0 == name) {
                    continue;
                }
                let b = unit_bytes(k, u, &ks, self.blocks, self.page_size, 1);
                if b > most {
                    return None;
                }
                bytes.push((name, b));
            }
        }
        let cap = [64, 32, 16, 8]
            .into_iter()
            .find(|&cap| super::plan::final_root_bytes(vr, hc, 1, 1, None, cap, true) <= most)?;
        bytes.push((
            "head final root".into(),
            super::plan::final_root_bytes(vr, hc, 1, 1, None, cap, true),
        ));
        let strip = super::plan::embed_bytes(vr, hc, 1, true);
        if strip > most {
            return None;
        }
        bytes.push(("strip".into(), strip));
        let cost = grids
            .iter()
            .map(|u| unit_cost(grid, ks[usize::from(u[0])]))
            .sum::<usize>()
            + regions
                .iter()
                .flat_map(|u| u.iter().map(|&g| unit_cost(region, ks[usize::from(g)])))
                .sum::<usize>();
        Some(G12Shape {
            grid,
            region,
            rnl,
            rc,
            rr,
            band,
            ncol,
            hh,
            vr,
            gx,
            gy,
            cap,
            bytes,
            cost,
        })
    }

    /// The cheapest placement that fits (ties: the larger margin).
    fn search(&self, wafer: WaferDims) -> Option<G12Shape> {
        let mut best: Option<G12Shape> = None;
        for py in (64..=PX).step_by(16) {
            for rpy in (64..=2 * PX).step_by(16) {
                for rnl in 1..=2 {
                    for rc in 1..=4 {
                        for rr in 1..=8 {
                            let Some(s) = self.place(wafer, py, (rpy, rnl, rc, rr)) else {
                                continue;
                            };
                            let worst =
                                |s: &G12Shape| s.bytes.iter().map(|b| b.1).max().unwrap_or(0);
                            if best
                                .as_ref()
                                .is_none_or(|b| (s.cost, worst(&s)) < (b.cost, worst(b)))
                            {
                                best = Some(s);
                            }
                        }
                    }
                }
            }
        }
        best
    }

    /// serve8b_region's parameters for the plan (`ProgramSpec::from_params`
    /// with `Design::Serve8bRegion`), `None` if it does not place.
    #[must_use]
    pub fn params(&self) -> Option<Vec<(&'static str, String)>> {
        let s = self.shape.as_ref()?;
        let (g, r, k) = (s.grid, s.region, &self.knobs);
        let p: Vec<(&'static str, usize)> = vec![
            ("PX", g.px),
            ("W4", 1),
            ("PY", g.py),
            ("Hc", g.hc),
            ("Dc", g.dc),
            ("Gs", g.gs),
            ("Ic", g.ic),
            ("RPX", r.px),
            ("RPY", r.py),
            ("RHc", r.hc),
            ("RDc", r.dc),
            ("RGs", r.gs),
            ("RIc", r.ic),
            ("RNL", s.rnl),
            ("RGY", RGY),
            ("GY", RGY),
            ("RC", s.rc),
            ("RR", s.rr),
            ("NCOL", s.ncol),
            ("RTR", 4),
            ("HR", self.hidden),
            ("Qh", self.qh),
            ("QhG", k.qhg),
            ("GsG", k.gsg),
        ];
        let mut out: Vec<(&'static str, String)> =
            p.into_iter().map(|(a, b)| (a, b.to_string())).collect();
        out.push(("KINDS", k.kinds.to_string()));
        for (a, b) in [
            ("WIN", k.win),
            ("W4BUF", k.w4buf),
            ("ESB", k.esb as usize),
            ("PS", self.page_size),
            ("NB", self.blocks),
            ("E", 1),
            ("B", 1),
            ("MC", 1),
            ("W16", 1),
            ("NP", 1),
            ("KV16", 1),
            ("KVCP", 0),
            ("PROF", 0),
            ("TIME", 0),
            ("HROUTE", 1),
            ("ROT8", 1),
            ("NOLAYER", 0),
            ("KVST", 0),
            ("HEADMODE", 2),
            ("HH", s.hh),
            ("Vr", s.vr),
            ("Gx", s.gx),
            ("Gy", s.gy),
            ("vocab", self.vocab),
            ("PASSES", 3),
            ("MODE", 0),
            ("OVL", 0),
            ("VMC", 1),
            ("iters", 1),
            ("W4H", 1),
            ("CAP", s.cap),
            ("G12", 1),
        ] {
            out.push((a, b.to_string()));
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec12() -> G4Spec {
        let trace = poem_compiler::catalog::repository()
            .parse("gemma4-12b-u4g64-kv-bf16")
            .expect("gemma4-12b")
            .trace(poem_ir::Platform::Cerebras);
        G4Spec::read(&trace).expect("gemma4-12b reads")
    }

    fn goals() -> Goals {
        Goals {
            lanes: 1,
            grid_lanes: 1,
            host_copy: true,
            blocks: Some(1),
            page_size: 16,
            ..Goals::default()
        }
    }

    /// The byte model at the probe's shapes is the link probe (by the
    /// calibration), class by class.
    #[test]
    fn unit_bytes_match_the_probe() {
        let knobs = G12 {
            kinds: 0,
            qhg: 4,
            gsg: 32,
            win: 1024,
            w4buf: W4BUF,
            esb: 0,
            kv16: true,
        };
        let ks = u4g12::kinds(2, 16, &knobs);
        let grid = Kernel {
            px: 128,
            py: 96,
            hc: 40,
            dc: 16,
            gs: 16,
            ic: 120,
        };
        let region = Kernel {
            px: 128,
            py: 240,
            hc: 16,
            dc: 16,
            gs: 16,
            ic: 120,
        };
        for (k, glob, want) in [
            (grid, vec![false], 44_416),
            (grid, vec![true], 47_120),
            (region, vec![false, false], 43_920),
            (region, vec![true, false], 46_720),
        ] {
            assert_eq!(unit_bytes(k, &glob, &ks, 1, 16, 1), want, "{k:?} {glob:?}");
        }
    }

    /// Gemma-4-12B (u4g64) plans as linked at full size (scratch/g12/full,
    /// 2026-11-05: 713 x 1153, every image within 48,384 B): grids 128 x 96,
    /// regions 128 x 240 of two layers, 4 x 4, eleven grids down the column.
    #[test]
    fn gemma4_12b_plans_as_linked() {
        let s = spec12();
        assert_eq!(
            (
                s.hidden,
                s.layers(),
                s.q_heads,
                s.kv_heads,
                s.head_dim,
                s.global_head_dim
            ),
            (3840, 48, 16, 8, 256, 512)
        );
        let plan = G12Plan::derive(&s, WaferDims::CS3, goals());
        assert!(plan.blockers.is_empty(), "{:?}", plan.blockers);
        let got: std::collections::BTreeMap<&str, String> =
            plan.params().unwrap().into_iter().collect();
        let linked = "PX:128,W4:1,PY:96,Hc:40,Dc:16,Gs:16,Ic:120,RPX:128,RPY:240,RHc:16,RDc:16,RGs:16,RIc:120,RNL:2,RGY:16,GY:16,RC:4,RR:4,NCOL:11,RTR:4,HR:3840,Qh:2,QhG:4,GsG:32,KINDS:142971416741920,WIN:1024,W4BUF:512,ESB:1115152118,PS:16,NB:1,E:1,B:1,MC:1,W16:1,NP:1,KV16:1,KVCP:0,PROF:0,TIME:0,HROUTE:1,ROT8:1,NOLAYER:0,KVST:0,HEADMODE:2,HH:1152,Vr:228,Gx:8,Gy:24,vocab:262144,PASSES:3,MODE:0,OVL:0,VMC:1,iters:1,W4H:1,CAP:64,G12:1";
        let want: std::collections::BTreeMap<&str, String> = linked
            .split(',')
            .filter_map(|kv| kv.split_once(':'))
            .map(|(k, v)| (k, v.to_string()))
            .collect();
        assert_eq!(
            got,
            want,
            "bytes {:?}",
            plan.shape.as_ref().map(|s| &s.bytes)
        );
        let program = super::super::program::ProgramSpec::from_params(
            super::super::codegen::Design::Serve8bRegion,
            &plan
                .params()
                .unwrap()
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect::<Vec<_>>(),
        )
        .expect("the plan's program");
        assert_eq!(program.gemma12(), Some(plan.knobs));
    }

    /// E2B (per-layer embeddings, a shared tail) is serve_g4's, not this plan's.
    #[test]
    fn e2b_is_refused() {
        let trace = poem_compiler::catalog::repository()
            .parse("gemma4-e2b-bf16-kv-bf16")
            .expect("e2b")
            .trace(poem_ir::Platform::Cerebras);
        let s = G4Spec::read(&trace).expect("e2b reads");
        let plan = G12Plan::derive(&s, WaferDims::CS3, goals());
        assert!(
            plan.shape.is_none() && plan.blockers.iter().any(|b| b.contains("per-layer")),
            "{:?}",
            plan.blockers
        );
    }
}
