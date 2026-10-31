//! Building a wafer program in PIE: the plan and a small serve config made
//! into a [`ProgramSpec`] ([`program`]: each design's components from the
//! plan, the [`Pins`], the head's VM, the parameters rendered from the spec
//! and checked against the layout and the registry of validated shapes),
//! then the generated sources ([`super::codegen`]), `cslc`, the compile
//! cache, and `pie_manifest.json` recording the spec ([`super::manifest`]).

use std::path::PathBuf;
use std::time::Instant;

use super::codegen::Design;
use super::guest4::wafer::VmLimits;
use super::manifest::Manifest;
use super::paged::Shape;
use super::plan::{Family, Goals, SpatialPlan};
use super::program::Stated::{Either, Knob, Typed};
use super::program::{Head, HeadKind, Kv, Layers, Model, PageCopy, ProgramSpec, Stage, Stated};
use super::serve8b::Kernel;
use super::serve8b_region::{Units, VmBlock};
use super::shapes;

type R<T> = std::result::Result<T, String>;

/// The eta VM image a build carries on the head's VM columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VmImage {
    /// Ops, registers and pool words the image holds.
    pub max_ops: usize,
    pub nreg: usize,
    pub pool_words: usize,
    /// The op-class feature mask the image compiles.
    pub feat: usize,
    /// The out ring: words a step and depth.
    pub ring_w: usize,
    pub ring_d: usize,
    /// Per-op profile counters.
    pub prof: bool,
    /// Per-step host input words a turn (`vm_sin`; 0: none).
    pub step_input: usize,
    /// Logits rows a turn (`vm_rows`; multi-row programs).
    pub rows: usize,
    /// Per-lane programs: the bank arena's words (`vm_lprog` 1, `bank_words`).
    pub lane_programs: Option<usize>,
    /// The opcodes the image compiles (`opmask_lo`, `opmask_hi`); `None`: all.
    pub opmask: Option<(u32, u32)>,
    /// Lanes the VM holds (`vm_lanes`; `None`: MMAX x B).
    pub lanes: Option<usize>,
    /// Top-p walks eta's order in rounds (`vm_pwalk`, the layout's default
    /// 1); `Some(false)`: the list engine.
    pub pwalk: Option<bool>,
    /// The guest4hz image's options: the canonical sum (`vm_sumpipe`, 3 by
    /// row chunks, 1 pipelined, 2 concatenating) and the last hub's level-2
    /// list in wide registers (`vm_l2w`); `None`: the layout's default
    /// (serve2d_pipe's: 3 and 1, as measured fastest on CS-3).
    pub sumpipe: Option<usize>,
    pub l2w: Option<bool>,
}

impl VmImage {
    /// The image's layout parameters. Ones left at their defaults are not
    /// passed, so they don't appear in the manifest either; readers take
    /// absent as the default.
    fn params(&self) -> Vec<(&'static str, usize)> {
        let mut out = vec![
            ("VM", 1),
            ("max_ops", self.max_ops),
            ("nreg", self.nreg),
            ("pool_words", self.pool_words),
            ("feat", self.feat),
            ("ring_w", self.ring_w),
            ("ring_d", self.ring_d),
            ("vm_prof", usize::from(self.prof)),
        ];
        if self.step_input > 0 {
            out.push(("vm_sin", self.step_input));
        }
        if self.rows > 1 {
            out.push(("vm_rows", self.rows));
        }
        if let Some(words) = self.lane_programs {
            out.extend([("vm_lprog", 1), ("bank_words", words)]);
        }
        if let Some((lo, hi)) = self.opmask {
            out.extend([("opmask_lo", lo as usize), ("opmask_hi", hi as usize)]);
        }
        if let Some(lanes) = self.lanes {
            out.push(("vm_lanes", lanes));
        }
        if let Some(pwalk) = self.pwalk {
            out.push(("vm_pwalk", usize::from(pwalk)));
        }
        if let Some(sumpipe) = self.sumpipe {
            out.push(("vm_sumpipe", sumpipe));
        }
        if let Some(l2w) = self.l2w {
            out.push(("vm_l2w", usize::from(l2w)));
        }
        out
    }

    /// As the second VM set (set B, `cols` VMB columns): the
    /// layout's `b_*` parameters, every one passed.
    fn params_b(&self, cols: usize) -> Vec<(&'static str, usize)> {
        let (lo, hi) = self.opmask.unwrap_or((0xffff_ffff, 0xffff_ffff));
        vec![
            ("VMB", cols),
            ("b_max_ops", self.max_ops),
            ("b_nreg", self.nreg),
            ("b_pool_words", self.pool_words),
            ("b_feat", self.feat),
            ("b_ring_w", self.ring_w),
            ("b_ring_d", self.ring_d),
            ("b_vm_prof", usize::from(self.prof)),
            ("b_vm_lanes", self.lanes.unwrap_or(1)),
            ("b_vm_rows", self.rows),
            ("b_vm_lprog", usize::from(self.lane_programs.is_some())),
            ("b_bank_words", self.lane_programs.unwrap_or(1)),
            ("b_vm_pwalk", usize::from(self.pwalk.unwrap_or(true))),
            ("b_vm_sin", self.step_input),
            ("b_opmask_lo", lo as usize),
            ("b_opmask_hi", hi as usize),
        ]
    }
}

/// Speculative decode on the head (serve2d_pipe): history, lookup table and
/// output capacities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Spec {
    pub history: usize,
    pub table: usize,
    pub out: usize,
}

/// What a serving build is for, beyond the model: the plan's goals and the
/// head-side options. Nothing here is a free-form parameter string. `pins`
/// overrides a few plan-derived shape fields ([`Pins`]), for small test
/// shapes the plan's rules don't produce.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServeConfig {
    pub goals: Goals,
    /// serve2d_pipe: micro-batches a launch and lanes a head step.
    pub mmax: Option<usize>,
    pub hb: Option<usize>,
    /// The eta VM image; `None`: no VM.
    pub vm: Option<VmImage>,
    /// The second resident VM set (serve2d_pipe): its image and its
    /// columns (`VMB`), east of the first set's.
    pub vm_b: Option<(VmImage, usize)>,
    /// Virtual logits lanes a launch (`HLM`, serve2d_pipe).
    pub hlm: Option<usize>,
    pub spec: Option<Spec>,
    /// The fused naive sampler on the head.
    pub smp: bool,
    /// A paged8b plan as serve8b_region's units (grids and several-layer
    /// regions) instead of serve8b's strips; with the `pipeline` goal, as
    /// serve8b_region_pipe's ring (`mmax` micro-batches).
    pub region: bool,
    pub pins: Pins,
}

impl ServeConfig {
    pub fn new(goals: Goals) -> ServeConfig {
        ServeConfig {
            goals,
            mmax: None,
            hb: None,
            vm: None,
            vm_b: None,
            hlm: None,
            spec: None,
            smp: false,
            region: false,
            pins: Pins::default(),
        }
    }
}

impl ServeConfig {
    /// `[cerebras.spatial.serve]`: the build PIE compiles when no `program`
    /// directory is configured. Keys (all optional):
    /// - the plan's goals: `lanes`, `grid_lanes`, `page_size` (default
    ///   `page_size`, PIE's), `blocks`, `max_blocks`, `mlp_chunk`, `cap`,
    ///   `pipeline`, `host_copy`;
    /// - serve2d_pipe: `mmax`, `hb`, `hlm`, `smp`, and `[spec]` (`history`,
    ///   `table`, `out`);
    /// - `region`: a paged8b (8B) plan as serve8b_region's units;
    /// - `[vm]`: `max_ops`, `nreg`, `pool_words`, `feat`, `ring_w` (default
    ///   1), `ring_d` (32), `prof`, `step_input`, `rows`, `bank_words`
    ///   (per-lane programs), `opmask_lo`/`opmask_hi`, `lanes` (`vm_lanes`),
    ///   `pwalk` (`vm_pwalk`), `sumpipe` (`vm_sumpipe`), `l2w` (`vm_l2w`);
    /// - `[vm_b]` (serve2d_pipe's second VM set): the same keys
    ///   for its image (the layout's `b_*`) and `cols` (`VMB`);
    /// - `[pins]`: [`Pins`] by their layout names (`Tc`, `E`, `GW`, `GH`,
    ///   `F`, `HH`, `Vr`, `Gx`, `Gy`, `CAP`, `GRID`, `MMAX`, `KVST`, `NP`),
    ///   for shapes the plan does not place (simulator-sized models).
    pub fn from_toml(t: &toml::Table, page_size: usize) -> R<ServeConfig> {
        fn int(t: &toml::Table, k: &str) -> R<Option<usize>> {
            match t.get(k) {
                None => Ok(None),
                Some(v) => v
                    .as_integer()
                    .and_then(|i| usize::try_from(i).ok())
                    .map(Some)
                    .ok_or_else(|| {
                        format!("[cerebras.spatial.serve] `{k}` is not a non-negative integer")
                    }),
            }
        }
        fn flag(t: &toml::Table, k: &str) -> R<bool> {
            match t.get(k) {
                None => Ok(false),
                Some(v) => v
                    .as_bool()
                    .ok_or_else(|| format!("[cerebras.spatial.serve] `{k}` is not a boolean")),
            }
        }
        let known = [
            "lanes",
            "grid_lanes",
            "page_size",
            "blocks",
            "max_blocks",
            "mlp_chunk",
            "cap",
            "pipeline",
            "host_copy",
            "mmax",
            "hb",
            "hlm",
            "smp",
            "spec",
            "vm",
            "vm_b",
            "pins",
            "region",
        ];
        if let Some(k) = t.keys().find(|k| !known.contains(&k.as_str())) {
            return Err(format!("[cerebras.spatial.serve]: unknown key `{k}`"));
        }
        let lanes = int(t, "lanes")?.unwrap_or(Goals::default().lanes);
        let base = Goals::lanes(lanes);
        let goals = Goals {
            lanes,
            grid_lanes: int(t, "grid_lanes")?.unwrap_or(base.grid_lanes),
            page_size: int(t, "page_size")?.unwrap_or(page_size),
            blocks: int(t, "blocks")?,
            max_blocks: int(t, "max_blocks")?.unwrap_or(base.max_blocks),
            mlp_chunk: int(t, "mlp_chunk")?,
            cap: int(t, "cap")?,
            pipeline: flag(t, "pipeline")?,
            host_copy: flag(t, "host_copy")?,
        };
        let spec = match t.get("spec") {
            None => None,
            Some(v) => {
                let s = v
                    .as_table()
                    .ok_or("[cerebras.spatial.serve] `spec` is not a table")?;
                Some(Spec {
                    history: int(s, "history")?.unwrap_or(1),
                    table: int(s, "table")?.unwrap_or(1),
                    out: int(s, "out")?.unwrap_or(1),
                })
            }
        };
        let image = |key: &str| -> R<Option<(VmImage, Option<usize>)>> {
            let Some(v) = t.get(key) else { return Ok(None) };
            {
                let m = v
                    .as_table()
                    .ok_or_else(|| format!("[cerebras.spatial.serve] `{key}` is not a table"))?;
                let need = |k: &str| {
                    int(m, k)?.ok_or_else(|| format!("[cerebras.spatial.serve.{key}] needs `{k}`"))
                };
                let lo = int(m, "opmask_lo")?;
                let hi = int(m, "opmask_hi")?;
                Ok(VmImage {
                    max_ops: need("max_ops")?,
                    nreg: need("nreg")?,
                    pool_words: need("pool_words")?,
                    feat: need("feat")?,
                    ring_w: int(m, "ring_w")?.unwrap_or(1),
                    ring_d: int(m, "ring_d")?.unwrap_or(32),
                    prof: flag(m, "prof")?,
                    step_input: int(m, "step_input")?.unwrap_or(0),
                    rows: int(m, "rows")?.unwrap_or(1),
                    lane_programs: int(m, "bank_words")?,
                    opmask: (lo.is_some() || hi.is_some()).then(|| {
                        (
                            lo.unwrap_or(0xffff_ffff) as u32,
                            hi.unwrap_or(0xffff_ffff) as u32,
                        )
                    }),
                    lanes: int(m, "lanes")?,
                    pwalk: m.get("pwalk").map(|_| flag(m, "pwalk")).transpose()?,
                    sumpipe: int(m, "sumpipe")?,
                    l2w: m.get("l2w").map(|_| flag(m, "l2w")).transpose()?,
                })
                .map(|img| Some((img, int(m, "cols").ok().flatten())))
            }
        };
        let vm = image("vm")?.map(|(img, _)| img);
        let vm_b = match image("vm_b")? {
            Some((img, Some(cols))) if cols > 0 => Some((img, cols)),
            Some(_) => {
                return Err(
                    "[cerebras.spatial.serve.vm_b] needs `cols` (VMB, its VM columns)".into(),
                );
            }
            None => None,
        };
        Ok(ServeConfig {
            goals,
            mmax: int(t, "mmax")?,
            hb: int(t, "hb")?,
            vm,
            vm_b,
            hlm: int(t, "hlm")?,
            spec,
            smp: flag(t, "smp")?,
            region: flag(t, "region")?,
            pins: match t.get("pins") {
                None => Pins::default(),
                Some(v) => {
                    let m = v
                        .as_table()
                        .ok_or("[cerebras.spatial.serve] `pins` is not a table")?;
                    let mut pins = Pins::default();
                    for k in m.keys() {
                        pins.set(k, int(m, k)?.unwrap_or(0))?;
                    }
                    pins
                }
            },
        })
    }
}

/// The program PIE builds for `spec` under `cfg` on `target`: the plan,
/// its parameters, the compile (cached). Returns the output directory.
pub fn build_for(
    spec: &super::model::ModelSpec,
    cfg: &ServeConfig,
    target: crate::sdk::Target,
    system: bool,
) -> R<PathBuf> {
    let plan = SpatialPlan::derive(spec, super::plan::WaferDims::CS3, cfg.goals)
        .map_err(|e| e.to_string())?;
    let program = program(&plan, cfg)?;
    tracing::info!(design = program.design.name(), model = %spec.name, "spatial build: the plan's program");
    compile(&program, target, system)
}

/// The design a plan's family builds: paged8b grids go into serve8b.
pub fn design(plan: &SpatialPlan) -> R<Design> {
    match plan.family {
        Family::Serve2d => Ok(Design::Serve2d),
        Family::Serve2dPipe => Ok(Design::Serve2dPipe),
        Family::Paged8b => Ok(Design::Serve8b),
    }
}

/// serve8b's placement of a paged8b plan on `wafer`: `GW` grids across, `GH`
/// strips down, the first `GS` strips full and the rest one grid short
/// (layers = GH (GW - 1) + GS); beside them the head (PY / 2 columns + its VM
/// column, HH = GH * PY rows, Vr = ceil(vocab / HH) vocab rows a PE) and the
/// PY / 4-wide embedding strip. The fewest strips, then the fewest grids
/// across, that fit (serve8b/layout.csl's asserts included).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Serve8bPlace {
    pub gw: usize,
    pub gh: usize,
    pub gs: usize,
    pub hh: usize,
    pub vr: usize,
    pub gx: usize,
    pub gy: usize,
}

pub fn place_serve8b(plan: &SpatialPlan, wafer: super::plan::WaferDims) -> R<Serve8bPlace> {
    let (px, py, layers, vocab) = (plan.grid.px, plan.grid.py, plan.layers, plan.vocab);
    if py % 4 != 0 || py < px || py > 2 * px {
        return Err(format!(
            "serve8b wants PY a multiple of 4 in [PX, 2 PX]; the plan's grid is {px} x {py}"
        ));
    }
    let vmc = 1;
    for gh in 1..=wafer.height / py {
        for gw in 2..=wafer.width / px {
            let width = gw * px + py / 2 + vmc + py / 4;
            let hh = gh * py;
            if width > wafer.width || hh + 1 > wafer.height || gh * (gw - 1) + gh < layers {
                continue;
            }
            let gs = layers.saturating_sub(gh * (gw - 1)).max(1);
            let gs = if gs % 2 == 1 && gs != gh { gs + 1 } else { gs };
            if gs > gh || !(gw >= 3 || gh - gs <= 1) {
                continue;
            }
            let f = hh / 2;
            if f < py || f >= (gh - 1) * py {
                continue;
            }
            let c = py / 2;
            let gx = [16, 8, 4, 2]
                .into_iter()
                .find(|g| c % g == 0 && c / g >= 2)
                .unwrap_or(1);
            let gx = gx.min(8);
            let gy = (2..=24).rev().step_by(2).find(|g| f % g == 0).unwrap_or(2);
            return Ok(Serve8bPlace {
                gw,
                gh,
                gs,
                hh,
                vr: vocab.div_ceil(hh),
                gx,
                gy,
            });
        }
    }
    Err(format!(
        "{}: {layers} grids of {px} x {py} and serve8b's head do not fit {} x {}",
        plan.spec.name, wafer.width, wafer.height
    ))
}

/// serve8b's program for a paged8b plan: the plan's grids in strips
/// ([`place_serve8b`]), head2d_v10 and the embedding strip.
fn serve8b(plan: &SpatialPlan, cfg: &ServeConfig) -> R<Draft> {
    if cfg.vm.is_some() || cfg.spec.is_some() || cfg.smp || cfg.goals.pipeline {
        return Err("serve8b has no eta VM, speculative decode, fused sampler or pipeline".into());
    }
    let g = &plan.grid;
    let p = place_serve8b(plan, super::plan::WaferDims::CS3)?;
    let cap = cfg.goals.cap.unwrap_or_else(|| {
        let mut c = 64;
        while g.lanes * c > 512 && c > 1 {
            c /= 2;
        }
        c
    });
    let spec = ProgramSpec::new(
        Design::Serve8b,
        Model {
            hidden: g.hidden,
            qh: g.qh,
            vocab: plan.vocab,
            cap,
        },
        Layers::Strips {
            grid: kernel(g),
            gw: p.gw,
            gh: p.gh,
            full: p.gs,
        },
        Stage::Single,
        Head {
            kind: HeadKind::Untied,
            hh: p.hh,
            vr: p.vr,
            gx: p.gx,
            vm_cols: 1,
            vmb_cols: 0,
            hlm: None,
            hlbn: None,
            defer: false,
        },
        Kv::Paged {
            shape: pool(g, g.py),
            // Page copies by the host (KVCP 0): with the fast softmax (~600 B a
            // grid PE) B 1 NP 3 fits only without the copy kernels (1,504 B
            // free, 736 with them).
            copy: PageCopy::Host,
            // KV page staging: a slot a PE where it fits (B 1; at B 4 the grid
            // has no room for it).
            staging: usize::from(g.lanes == 1),
        },
    );
    let mut st: Vec<Stated> = ["PX", "PY", "Hc", "HR", "Dc", "Gs", "Qh", "Ic", "PS", "E"]
        .map(Typed)
        .to_vec();
    st.extend([
        Knob("MC", g.mlp_chunk),
        Knob("W16", usize::from(g.w16)),
        Knob("NP", 3),
        Knob("KV16", usize::from(g.kv16)),
        Knob("PROF", 0),
        Knob("TIME", 0),
        Typed("GW"),
        Typed("GH"),
        Typed("GS"),
        Knob("HEADMODE", 2),
        Typed("HH"),
        Typed("Vr"),
        Typed("Gx"),
        Knob("Gy", p.gy),
        Typed("vocab"),
        Knob("PASSES", 3),
        Knob("MODE", 0),
        Knob("OVL", 0),
        Typed("VMC"),
        Typed("CAP"),
        Knob("iters", 1),
        Typed("B"),
        Typed("NB"),
        Typed("KVCP"),
        Typed("KVST"),
    ]);
    Ok(Draft { spec, stated: st })
}

/// serve8b_region's placement of a paged8b plan ("option A"): the
/// plan's grid for the first and last layers (a top band of `RC RPX / PX
/// + 1` grids, `NCOL` grids down the column beside the head) and between
/// them `RC` x `RR` regions of `RNL` layers, twice the grid's width (`RGs`
/// 2 Gs columns a kv head of `RDc` Dc / 2 dims, `RIc` Ic / 2) and `RPY` 240
/// rows (`RHc` = ceil(hidden / RPY)). RC 2, RNL 3, RR 4 as built; `NCOL`
/// the layers left. The head as serve8b's, `HH` = the grid column's rows.
fn serve8b_region(plan: &SpatialPlan, cfg: &ServeConfig, design: Design) -> R<Draft> {
    if cfg.spec.is_some() || cfg.smp || cfg.vm_b.is_some() {
        return Err(
            "serve8b_region has no speculative decode, fused sampler or second VM set".into(),
        );
    }
    // serve8b_region_pipe (`pipeline`): micro-batches round the units, in
    // the shapes validated on hardware only (see `refused`).
    let pipe = cfg
        .goals
        .pipeline
        .then(|| pipe_shape(plan.grid.lanes, cfg))
        .transpose()?;
    let g = &plan.grid;
    let wafer = super::plan::WaferDims::CS3;
    let (rc, rnl, rr, rpy, rgy) = (2, 3, 4, 240, 16);
    let (rpx, band) = (2 * g.px, 2 * 2 * g.px / g.px + 1);
    if !g.dc.is_multiple_of(4) || !g.ic.is_multiple_of(2) || plan.layers < band + rc * rr * rnl + 1
    {
        return Err(format!(
            "{}: serve8b_region's option A does not take this grid ({g:?})",
            plan.spec.name
        ));
    }
    let ncol = plan.layers - band - rc * rr * rnl;
    let c = g.py / 2;
    // With an eta VM (VMX): the guest4hz VM as a 5-column block east of
    // the strip, Vr 160 vocab rows a head PE (32 a VM column: required) and
    // the fewest head rows HH >= vocab / Vr, even (F = HH / 2), whose feeder
    // half F has an even row group Gy <= 28 (Gy must divide F; evenness and
    // the cap of 28, the group the Vr 160 fits were measured at, are this
    // plan's choices, not head2d_v10's limits); without, the head as tall
    // as the grid column, VMC 1.
    let (hh, vr, vmc, vw) = match cfg.vm {
        Some(_) => {
            let vr = 160;
            let mut hh = plan.vocab.div_ceil(vr);
            hh += hh % 2;
            while !(4..=28).step_by(2).any(|x| (hh / 2) % x == 0) {
                hh += 2;
            }
            (hh, vr, 0, 5)
        }
        None => {
            let hh = g.py * (1 + ncol);
            (hh, plan.vocab.div_ceil(hh), 1, 0)
        }
    };
    let width = rc * rpx + g.px + c + vmc + g.py / 4 + vw;
    if width > wafer.width || hh + 1 > wafer.height || g.py + rr * rpy > wafer.height {
        return Err(format!(
            "{}: serve8b_region's {width} x {} does not fit the wafer",
            plan.spec.name,
            hh + 1
        ));
    }
    let f = hh / 2;
    if f < g.py || f >= g.py * ncol {
        return Err(format!(
            "{}: serve8b_region's feeder row {f} is not between the grids' {} and {}",
            plan.spec.name,
            g.py,
            g.py * ncol
        ));
    }
    let gx = [16, 8, 4, 2]
        .into_iter()
        .find(|x| c.is_multiple_of(*x) && c / x >= 2)
        .unwrap_or(1)
        .min(8);
    let gy_most = if cfg.vm.is_some() { 28 } else { 24 };
    let gy = (2..=gy_most)
        .rev()
        .step_by(2)
        .find(|x| f % x == 0)
        .unwrap_or(2);
    let cap = cfg.goals.cap.unwrap_or_else(|| {
        let mut c = 64;
        while g.lanes * c > 512 && c > 1 {
            c /= 2;
        }
        c
    });
    let cap = match pipe {
        Some((_, pcap)) if cap > pcap && cfg.goals.cap.is_some() => {
            return Err(format!(
                "serve8b_region_pipe at B {} holds CAP {pcap} at most (its hardware-validated fit), not {cap}",
                g.lanes
            ));
        }
        Some((_, pcap)) => cap.min(pcap),
        None => cap,
    };
    let units = Units {
        rpx,
        rpy,
        rhc: g.hidden.div_ceil(rpy),
        rdc: g.dc / 2,
        rgs: 2 * g.gs,
        ric: g.ic / 2,
        rnl,
        rc,
        rr,
        ncol,
        rtr: 4,
        gy: g.gs,
        rgy,
        hroute: true,
        rot8: true,
        nolayer: false,
        band,
        vm: None,
    };
    let mut st: Vec<Stated> = Vec::new();
    // KV page staging at B 1 (at B 4 the units have no room for it).
    let staging = usize::from(g.lanes == 1 && pipe.is_none());
    st.push(Typed("KVST"));
    let stage = match pipe {
        Some((mmax, _)) => {
            st.extend([Typed("PIPE"), Typed("MMAX")]);
            Stage::Ring { mmax }
        }
        None => Stage::Single,
    };
    let mut units = units;
    let mut hlbn = None;
    if let Some(vm) = cfg.vm {
        // The guest4hz VM image on the block; one VM lane a head lane, and the
        // head keeps one lane's logits rows at B > 1 (room on its PEs).
        if vm.lanes.is_some_and(|l| l != g.lanes) || vm.lane_programs.is_none() {
            return Err(
                "serve8b_region's VM holds per-lane programs (`bank_words`), one lane a head lane"
                    .into(),
            );
        }
        let (lo, hi) = vm.opmask.unwrap_or((0xffff_ffff, 0xffff_ffff));
        units.vm = Some(VmBlock {
            max_ops: vm.max_ops,
            nreg: vm.nreg,
            pool_words: vm.pool_words,
            // Several lanes: 128 bank words a lane at most (the B 4 fit: every
            // current program's live window, 33-108 words, fits it).
            bank_words: if g.lanes > 1 {
                vm.lane_programs.unwrap_or(0).min(128 * g.lanes)
            } else {
                vm.lane_programs.unwrap_or(0)
            },
            lanes: g.lanes,
        });
        st.extend([
            Typed("VMX"),
            Typed("VMREAL"),
            Typed("max_ops"),
            Typed("nreg"),
            Typed("pool_words"),
            Knob("feat", vm.feat),
            Knob("opmask_lo", lo as usize),
            Knob("opmask_hi", hi as usize),
            Knob("ring", vm.ring_d),
            Knob("ring_w", vm.ring_w),
            Knob("vm_rows", vm.rows),
            Knob("vm_sin", vm.step_input),
            Knob("vm_lprog", 1),
            Knob("vm_pwalk", usize::from(vm.pwalk.unwrap_or(true))),
            Typed("bank_words"),
            Knob("vm_prof", usize::from(vm.prof)),
            Typed("vm_lanes"),
            // The VM options the fitted builds use: the last hub's level-2
            // list in wide; the pipelined canonical sum (2, concatenating, is
            // B 1's once hardware shows its gain).
            Knob("VML2W", 1),
            Knob("VMSUMPIPE", 1),
        ]);
        if g.lanes > 1 {
            hlbn = Some(1);
            st.push(Typed("HLBN"));
        }
    }
    let spec = ProgramSpec::new(
        design,
        Model {
            hidden: g.hidden,
            qh: g.qh,
            vocab: plan.vocab,
            cap,
        },
        Layers::Units {
            grid: kernel(g),
            units,
        },
        stage,
        Head {
            kind: HeadKind::Untied,
            hh,
            vr,
            gx,
            vm_cols: vmc,
            vmb_cols: 0,
            hlm: None,
            hlbn,
            defer: false,
        },
        Kv::Paged {
            shape: pool(g, g.py.min(rpy)),
            copy: PageCopy::Host,
            staging,
        },
    );
    st.extend(
        [
            "PX", "PY", "Hc", "Dc", "Gs", "Ic", "RPX", "RPY", "RHc", "RDc", "RGs", "RIc", "RNL",
            "RGY", "RC", "RR", "NCOL", "RTR", "HR", "Qh", "PS", "NB", "E",
        ]
        .map(Typed),
    );
    st.extend([
        Knob("MC", g.mlp_chunk),
        Knob("W16", usize::from(g.w16)),
        Knob("KV16", usize::from(g.kv16)),
        Typed("KVCP"),
        Knob("PROF", 0),
        Knob("TIME", 0),
        Typed("HROUTE"),
        Typed("ROT8"),
        Typed("NOLAYER"),
        Knob("HEADMODE", 2),
        Typed("HH"),
        Typed("Vr"),
        Typed("Gx"),
        Knob("Gy", gy),
        Typed("vocab"),
        Knob("PASSES", 3),
        Knob("MODE", 0),
        Knob("OVL", 0),
        Typed("VMC"),
        Typed("CAP"),
        Knob("iters", 1),
        Typed("B"),
        Knob("NP", if pipe.is_some() { 1 } else { 3 }),
    ]);
    Ok(Draft { spec, stated: st })
}

/// serve8b_region_pipe's micro-batches (`mmax`, or the most validated) and
/// CAP for `lanes`: only the validated shapes; the rest are unrun.
fn pipe_shape(lanes: usize, cfg: &ServeConfig) -> R<(usize, usize)> {
    if cfg.vm.is_some() {
        return Err("serve8b_region_pipe has no eta VM yet (VMX unrun on the ring)".into());
    }
    let (most, cap) = shapes::ran(Design::Serve8bRegionPipe, lanes).ok_or_else(|| {
        format!(
            "serve8b_region_pipe runs B 4, 2 or 1 (validated on hardware); B {lanes} is unsupported"
        )
    })?;
    match cfg.mmax.unwrap_or(most) {
        m if (1..=most).contains(&m) => Ok((m, cap)),
        m => Err(format!(
            "serve8b_region_pipe at B {lanes} runs 1 to {most} micro-batches (validated on hardware), not {m}"
        )),
    }
}

/// The layout's parameters: (name, has a default), in declaration order.
pub fn layout_params(layout: &str) -> Vec<(String, bool)> {
    layout
        .lines()
        .filter_map(|l| l.trim_start().strip_prefix("param "))
        .filter_map(|rest| {
            let (name, ty) = rest.split_once(':')?;
            let name = name.trim();
            (!ty.contains("color") && !name.is_empty())
                .then(|| (name.to_string(), ty.contains('=')))
        })
        .collect()
}

/// A planned program before its overrides and its rendering: the spec's
/// components, and the parameters the build states.
struct Draft {
    spec: ProgramSpec,
    stated: Vec<Stated>,
}

/// The plan's grid as a kernel shape.
fn kernel(g: &super::plan::Grid) -> Kernel {
    Kernel {
        px: g.px,
        py: g.py,
        hc: g.hc,
        dc: g.dc,
        gs: g.gs,
        ic: g.ic,
    }
}

/// The plan's KV pool on `rows` pool rows.
fn pool(g: &super::plan::Grid, rows: usize) -> Shape {
    Shape {
        lanes: g.lanes,
        rows,
        page_size: g.page_size,
        blocks: g.blocks,
        entries: g.entries,
    }
}

/// A VM image's limits as the VM layout holds them.
fn limits(img: &VmImage) -> VmLimits {
    let (lo, hi) = img.opmask.unwrap_or((0xffff_ffff, 0xffff_ffff));
    VmLimits {
        max_ops: img.max_ops,
        nreg: img.nreg,
        pool_words: img.pool_words,
        feat: img.feat as u32,
        opmask: u64::from(lo) | u64::from(hi) << 32,
        sin: img.step_input,
        lprog: img.lane_programs.is_some(),
        bank_words: img.lane_programs.unwrap_or(0),
    }
}

/// serve2d's (or serve2d_pipe's) program for a serve2d plan: the plan's
/// square f32 grids and head, the ring's micro-batches, the head's VM.
fn serve2d(plan: &SpatialPlan, cfg: &ServeConfig, design: Design) -> R<Draft> {
    let pipe = design == Design::Serve2dPipe;
    if !pipe && (cfg.spec.is_some() || cfg.smp || cfg.mmax.is_some()) {
        return Err("speculative decode, the fused sampler and MMAX are serve2d_pipe's".into());
    }
    if cfg.vm_b.is_some() && !pipe {
        return Err("a second VM set (`vm_b`) is serve2d_pipe's".into());
    }
    let (g, h) = (&plan.grid, &plan.head);
    let p = plan.program.unwrap_or(super::plan::Program {
        gw: 0,
        gh: 0,
        cap: 0,
        mmax: None,
        hb: None,
        width: 0,
        height: 0,
    });
    let mmax = cfg.mmax.unwrap_or(p.mmax.unwrap_or(1));
    let spec = ProgramSpec::new(
        design,
        Model {
            hidden: g.hidden,
            qh: g.qh,
            vocab: plan.vocab,
            cap: p.cap,
        },
        Layers::Grids {
            grid: kernel(g),
            gw: p.gw,
            gh: p.gh,
            tc: super::plan::TC,
        },
        match (pipe, mmax) {
            (true, m) if m > 0 => Stage::Ring { mmax: m },
            _ => Stage::Single,
        },
        Head {
            kind: HeadKind::Tied,
            hh: h.hh,
            vr: h.vr,
            gx: h.gx,
            vm_cols: h.vmc,
            vmb_cols: cfg.vm_b.map_or(0, |(_, cols)| cols),
            hlm: cfg.hlm.filter(|_| pipe),
            hlbn: None,
            defer: false,
        },
        // The serve2d layouts have the copy kernels (they declare no KVCP).
        Kv::Paged {
            shape: pool(g, g.py),
            copy: PageCopy::Kernels,
            staging: 0,
        },
    );
    let mut st: Vec<Stated> = ["PX", "PY", "Hc", "HR", "Dc", "Gs", "Qh", "Ic", "Tc"]
        .map(Typed)
        .to_vec();
    st.extend([
        Knob("W16", 0),
        Knob("KV16", 0),
        Typed("GRID"),
        Knob("HEADMODE", 2),
        Either("B", g.lanes),
        Either("PS", g.page_size),
        Either("NB", g.blocks),
        Either("E", g.entries),
        Knob("MC", g.mlp_chunk),
        Typed("GW"),
        Typed("GH"),
        Knob("F", h.hh / 2),
        Typed("HH"),
        Typed("Vr"),
        Typed("Gx"),
        Knob("Gy", h.gy),
        Knob("Gb", h.gb),
        Typed("vocab"),
        Knob("PASSES", 3),
        Knob("UNTIED", usize::from(h.untied)),
        Knob("CERT", 0),
        Knob("MODE", 0),
        Knob("OVL", 0),
        Typed("VMC"),
        Knob("WF32", 1),
        Typed("CAP"),
        Knob("DBG", 0),
        Knob("iters", 1),
    ]);
    if pipe {
        let sp = cfg.spec.unwrap_or(Spec {
            history: 1,
            table: 1,
            out: 1,
        });
        st.extend([
            Either("MMAX", mmax),
            Knob("HB", cfg.hb.unwrap_or(p.hb.unwrap_or(1))),
            Knob("SPEC", usize::from(cfg.spec.is_some())),
            Knob("SPEC_H", sp.history),
            Knob("SPEC_T", sp.table),
            Knob("SPEC_O", sp.out),
            Knob("SMP", usize::from(cfg.smp)),
        ]);
        if cfg.hlm.is_some() {
            st.push(Typed("HLM"));
        }
    } else {
        st.extend([Knob("NP", 3), Knob("PROF", 0)]);
    }
    if let Some(vm) = cfg.vm {
        st.extend(vm.params().into_iter().map(|(k, v)| Either(k, v)));
        // serve2d_pipe's sum options as measured fastest on CS-3: the
        // layout's own defaults are the older 1 and 0 (it is kept as run).
        if pipe {
            if vm.sumpipe.is_none() {
                st.push(Knob("vm_sumpipe", 3));
            }
            if vm.l2w.is_none() {
                st.push(Knob("vm_l2w", 1));
            }
        }
    }
    if let Some((img, cols)) = cfg.vm_b {
        st.extend(img.params_b(cols).into_iter().map(|(k, v)| Either(k, v)));
    }
    Ok(Draft { spec, stated: st })
}

/// Shape overrides of a planned build (`[cerebras.spatial.serve.pins]`):
/// for small test shapes the plan's rules do not produce. Each is a typed
/// field of the program, applied to its component (or a kernel knob), then
/// checked like any build: the layout must declare it and the registry
/// must not refuse the shape.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Pins {
    /// `Tc`: a one-sequence grid's KV positions a PE row.
    pub tc: Option<usize>,
    /// `E`: a lane's pages a pool row.
    pub entries: Option<usize>,
    /// `GW`, `GH`: grids across and down.
    pub gw: Option<usize>,
    pub gh: Option<usize>,
    /// `F`: the head's feeder row (a knob; the host reads `HH / 2`).
    pub feeder: Option<usize>,
    /// `HH`, `Vr`, `Gx`, `Gy`: the head's rows, vocabulary rows a PE and
    /// its collectives' groups (`Gy` a knob).
    pub hh: Option<usize>,
    pub vr: Option<usize>,
    pub gx: Option<usize>,
    pub gy: Option<usize>,
    /// `CAP`: iterations a launch.
    pub cap: Option<usize>,
    /// `GRID`: 0, one sequence's grids (refused by the registry); 1, paged.
    pub grid: Option<usize>,
    /// `MMAX`: a ring's micro-batches.
    pub mmax: Option<usize>,
    /// `KVST`: KV staging slots a PE.
    pub staging: Option<usize>,
    /// `NP`: the layer kernel's passes (a knob).
    pub passes: Option<usize>,
}

impl Pins {
    /// The pins' layout names, as `[pins]` takes them.
    pub const NAMES: &[&str] = &[
        "Tc", "E", "GW", "GH", "F", "HH", "Vr", "Gx", "Gy", "CAP", "GRID", "MMAX", "KVST", "NP",
    ];

    /// Pins `name` (a layout parameter's name, [`Pins::NAMES`]) to `value`.
    pub fn set(&mut self, name: &str, value: usize) -> R<()> {
        let slot = match name {
            "Tc" => &mut self.tc,
            "E" => &mut self.entries,
            "GW" => &mut self.gw,
            "GH" => &mut self.gh,
            "F" => &mut self.feeder,
            "HH" => &mut self.hh,
            "Vr" => &mut self.vr,
            "Gx" => &mut self.gx,
            "Gy" => &mut self.gy,
            "CAP" => &mut self.cap,
            "GRID" => &mut self.grid,
            "MMAX" => &mut self.mmax,
            "KVST" => &mut self.staging,
            "NP" => &mut self.passes,
            _ => {
                return Err(format!(
                    "[cerebras.spatial.serve.pins] `{name}` is not a pin ({})",
                    Pins::NAMES.join(", ")
                ));
            }
        };
        *slot = Some(value);
        Ok(())
    }

    /// The pins set, by layout name.
    fn named(&self) -> Vec<(&'static str, usize)> {
        [
            ("Tc", self.tc),
            ("E", self.entries),
            ("GW", self.gw),
            ("GH", self.gh),
            ("F", self.feeder),
            ("HH", self.hh),
            ("Vr", self.vr),
            ("Gx", self.gx),
            ("Gy", self.gy),
            ("CAP", self.cap),
            ("GRID", self.grid),
            ("MMAX", self.mmax),
            ("KVST", self.staging),
            ("NP", self.passes),
        ]
        .into_iter()
        .filter_map(|(k, v)| v.map(|v| (k, v)))
        .collect()
    }

    /// The pins applied to `d`'s components, each then stated with its
    /// value.
    fn apply(&self, d: &mut Draft) {
        let spec = &mut d.spec;
        if let Some(v) = self.tc
            && let Layers::Grids { tc, .. } = &mut spec.layers
        {
            *tc = v;
        }
        if let Some(v) = self.gw {
            match &mut spec.layers {
                Layers::Grids { gw, .. } | Layers::Strips { gw, .. } => *gw = v,
                Layers::Units { .. } => {}
            }
        }
        if let Some(v) = self.gh {
            match &mut spec.layers {
                Layers::Grids { gh, .. } | Layers::Strips { gh, .. } => *gh = v,
                Layers::Units { .. } => {}
            }
        }
        spec.head.hh = self.hh.unwrap_or(spec.head.hh);
        spec.head.vr = self.vr.unwrap_or(spec.head.vr);
        spec.head.gx = self.gx.unwrap_or(spec.head.gx);
        spec.model.cap = self.cap.unwrap_or(spec.model.cap);
        if let Kv::Paged { shape, staging, .. } = &mut spec.kv {
            shape.entries = self.entries.unwrap_or(shape.entries);
            *staging = self.staging.unwrap_or(*staging);
        }
        if self.grid == Some(0) {
            spec.kv = Kv::Sequence;
        }
        match (self.mmax, spec.stage) {
            (Some(0), Stage::Ring { .. }) => spec.stage = Stage::Single,
            (Some(m), Stage::Ring { .. }) => spec.stage = Stage::Ring { mmax: m },
            _ => {}
        }
        for (k, v) in self.named() {
            d.stated
                .retain(|s| !matches!(*s, Typed(n) | Knob(n, _) | Either(n, _) if n == k));
            d.stated.push(Either(k, v));
        }
    }
}

/// The program `plan` builds for `cfg`: its components from the plan and
/// the config (the design's own rules, then the pins and the VM's sampler
/// on the final geometry), its parameters rendered from them in the
/// layout's order, checked against the layout and the registry of
/// validated shapes.
pub fn program(plan: &SpatialPlan, cfg: &ServeConfig) -> R<ProgramSpec> {
    let d = match design(plan)? {
        Design::Serve8b if cfg.region && cfg.goals.pipeline => Design::Serve8bRegionPipe,
        Design::Serve8b if cfg.region => Design::Serve8bRegion,
        _ if cfg.region => {
            return Err(format!(
                "{}: `region` places paged8b (8B) plans only",
                plan.spec.name
            ));
        }
        d => d,
    };
    if !matches!(
        d,
        Design::Serve8b | Design::Serve8bRegion | Design::Serve8bRegionPipe
    ) && plan.program.is_none()
        && cfg.pins.gw.is_none()
    {
        return Err(format!(
            "{}: no full program places the model: {}",
            plan.spec.name,
            plan.blockers.join("; ")
        ));
    }
    let mut draft = match d {
        Design::Serve8b => serve8b(plan, cfg)?,
        Design::Serve8bRegion | Design::Serve8bRegionPipe => serve8b_region(plan, cfg, d)?,
        Design::Serve2d | Design::Serve2dPipe => serve2d(plan, cfg, d)?,
    };
    cfg.pins.apply(&mut draft);
    // The head's VM on the final geometry (serve8b_region's is its block).
    if matches!(d, Design::Serve2d | Design::Serve2dPipe)
        && let Some(vm) = cfg.vm
    {
        let b = cfg.vm_b.map(|(img, _)| limits(&img));
        draft.spec.sampler = draft.spec.head_sampler(limits(&vm), b, false);
    }
    let files = d.files()?;
    let layout = &files
        .iter()
        .find(|f| f.0 == "layout.csl")
        .ok_or("no layout.csl")?
        .1;
    let decl = layout_params(layout);
    let named = |s: &Stated| match *s {
        Typed(n) | Knob(n, _) | Either(n, _) => n,
    };
    for (name, has_default) in &decl {
        if !*has_default && !draft.stated.iter().any(|s| named(s) == name) {
            return Err(format!("{}: the layout's `{name}` has no value", d.name()));
        }
    }
    // Plan keys another layout takes (NP, PROF on serve2d_pipe) are not
    // passed; pins and the VM images' keys must be the layout's.
    let config_keys: Vec<&str> = cfg
        .pins
        .named()
        .iter()
        .map(|p| p.0)
        .chain(
            cfg.vm
                .filter(|_| !matches!(d, Design::Serve8bRegion | Design::Serve8bRegionPipe))
                .map(|vm| vm.params())
                .unwrap_or_default()
                .iter()
                .map(|p| p.0),
        )
        .chain(
            cfg.vm_b
                .map(|(img, cols)| img.params_b(cols))
                .unwrap_or_default()
                .iter()
                .map(|p| p.0),
        )
        .collect();
    if let Some(bad) = config_keys
        .iter()
        .find(|k| !decl.iter().any(|(n, _)| n == *k))
    {
        return Err(format!("{}: no layout parameter `{bad}`", d.name()));
    }
    let names: Vec<String> = decl.into_iter().map(|(n, _)| n).collect();
    let spec = draft.spec.compose(&draft.stated, &names)?;
    shapes::check(d, &spec.to_params())?;
    spec.read_back()
}

/// The compile parameters of `plan` built for `cfg`: [`program`]'s,
/// rendered.
pub fn params(plan: &SpatialPlan, cfg: &ServeConfig) -> R<(Design, Vec<(String, String)>)> {
    let spec = program(plan, cfg)?;
    Ok((spec.design, spec.to_params()))
}

/// The compile cache's key: the sources, the parameters, the fabric.
fn key(files: &[super::codegen::File], params: &[(String, String)], fabric: &str) -> String {
    let mut h = blake3::Hasher::new();
    for (name, text) in files {
        h.update(name.as_bytes())
            .update(&[0])
            .update(text.as_bytes())
            .update(&[0]);
    }
    for (k, v) in params {
        h.update(k.as_bytes())
            .update(b":")
            .update(v.as_bytes())
            .update(b",");
    }
    h.update(fabric.as_bytes());
    h.finalize().to_hex().to_string()
}

/// The program's rectangle ([`ProgramSpec::rectangle`]) for `params`.
#[cfg(test)]
fn rectangle(design: Design, params: &[(String, String)]) -> R<(u32, u32)> {
    let (w, h) = ProgramSpec::from_params(design, params)?.rectangle();
    Ok((w as u32, h as u32))
}

/// `program`, from the compile cache or compiled into it
/// (the engine's disk cache, `PIE_CEREBRAS_CACHE`). The output directory gets
/// `pie_manifest.json`. `system`: laid out on the whole wafer.
pub fn compile(program: &ProgramSpec, target: crate::sdk::Target, system: bool) -> R<PathBuf> {
    use crate::sdk::{Arch, Compile, Cslc};
    let (design, params) = (program.design, &program.to_params());
    shapes::check(design, params)?;
    let files = design.files()?;
    let (w, h) = program.rectangle();
    let (w, h) = (w as u32, h as u32);
    let (fabric_dims, fabric_offsets) = crate::sdk::cslc::memcpy_fabric(w, h);
    let fabric_dims = if system {
        crate::device::wafer_dims(target)
    } else {
        fabric_dims
    };
    let fabric = format!("{fabric_dims:?}|{fabric_offsets:?}|{target:?}|{system}");
    let hex = key(&files, params, &fabric);
    let dir = crate::device::disk_cache().join(format!("{}-{}", design.name(), &hex[..32]));
    let out = dir.join("out");
    let bin = out.join("bin");
    if Manifest::read(&out).is_some_and(|m| m.key == hex) && bin.join("out_rpc.json").is_file() {
        tracing::info!(dir = %out.display(), design = design.name(), "spatial build: program cached");
        return Ok(out);
    }
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for (name, text) in &files {
        std::fs::write(dir.join(name), text).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let compile = Compile {
        arch: Arch::from(target),
        layout: dir.join("layout.csl"),
        fabric_dims,
        fabric_offsets,
        params: params.to_vec(),
        memcpy: true,
        channels: 1,
        import_paths: Vec::new(),
        out: out.clone(),
    };
    let started = Instant::now();
    tracing::info!(dir = %out.display(), design = design.name(), "spatial build: compiling");
    if let Err(e) = Cslc::find().and_then(|cslc| cslc.compile(&compile)) {
        let _ = std::fs::remove_dir_all(&out);
        return Err(format!("compiling {}: {e}", design.name()));
    }
    let m = Manifest {
        program: program.clone(),
        key: hex,
        fabric_dims,
        fabric_offsets,
        system,
    };
    m.write(&out)?;
    tracing::info!(
        s = started.elapsed().as_secs_f32(),
        "spatial build: compiled"
    );
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_params_read_with_defaults() {
        let (_, f) = (Design::Serve2dPipe, Design::Serve2dPipe.files().unwrap());
        let decl = layout_params(&f[0].1);
        assert!(decl.iter().any(|(n, d)| n == "PX" && !d));
        assert!(decl.iter().any(|(n, d)| n == "VM" && *d));
        assert!(!decl.iter().any(|(n, _)| n.starts_with("c_")));
    }

    #[test]
    fn key_changes_with_params() {
        let f = Design::Serve2dPipe.files().unwrap();
        let a = key(&f, &[("B".into(), "2".into())], "x");
        let b = key(&f, &[("B".into(), "4".into())], "x");
        assert_ne!(a, b);
    }
}

/// Small builds for the fabric simulator and tests.
pub mod presets {
    use super::*;
    use crate::spatial::model::ModelSpec;
    use crate::spatial::plan::WaferDims;

    /// The tiny Qwen3 shape the simulator runs (hidden 64, 8 layers, 2 q / 1 kv
    /// heads of 32, inter 64, vocab 250).
    pub fn spec() -> ModelSpec {
        ModelSpec {
            name: "tiny".into(),
            layers: 8,
            hidden: 64,
            q_heads: 2,
            kv_heads: 1,
            head_dim: 32,
            inter: 64,
            vocab: 250,
            tied: true,
            qk_norm: true,
            rope_theta: 1.0e6,
            rope_table: None,
            rms_eps: 1e-6,
            sm_scale: 1.0 / 32f32.sqrt(),
            weights: poem_ir::Dtype::F32,
            kv: poem_ir::Dtype::F32,
        }
    }

    /// A tiny simulator build of serve2d_pipe: the plan's model dims, a
    /// hand-placed program and head (the plan places none at this size), and
    /// the multi-lane VM.
    pub fn pipe_config() -> (SpatialPlan, ServeConfig) {
        let goals = Goals {
            pipeline: true,
            cap: Some(8),
            blocks: Some(2),
            mlp_chunk: Some(1),
            ..Goals::lanes(2)
        };
        let plan = SpatialPlan::derive(&spec(), WaferDims::CS3, goals).expect("a tiny plan");
        let cfg = ServeConfig {
            mmax: Some(2),
            hb: Some(2),
            hlm: Some(4),
            vm: Some(VmImage {
                max_ops: 48,
                nreg: 6,
                pool_words: 64,
                feat: 99,
                ring_w: 1,
                ring_d: 32,
                prof: false,
                step_input: 0,
                rows: 1,
                lane_programs: None,
                opmask: None,
                lanes: None,
                pwalk: None,
                sumpipe: None,
                l2w: None,
            }),
            pins: Pins {
                tc: Some(1),
                entries: Some(2),
                gw: Some(2),
                gh: Some(4),
                feeder: Some(8),
                hh: Some(16),
                vr: Some(32),
                gx: Some(4),
                gy: Some(4),
                cap: Some(8),
                ..Pins::default()
            },
            ..ServeConfig::new(goals)
        };
        (plan, cfg)
    }
}

#[cfg(test)]
mod tiny {
    use super::presets::*;
    use super::*;
    use crate::spatial::plan::WaferDims;

    /// The same parameters as the out.json of the build the simulator ran,
    /// with the sum options the layout defaults to (3 and 1) stated by the
    /// plan.
    #[test]
    fn tiny_pipe_params_match_the_simulated_build() {
        let (plan, cfg) = pipe_config();
        let (d, p) = params(&plan, &cfg).unwrap();
        assert_eq!(d, Design::Serve2dPipe);
        let want = "PX:4,PY:4,Hc:16,HR:64,Dc:8,Gs:4,Qh:2,Ic:16,Tc:1,W16:0,KV16:0,GRID:1,HEADMODE:2,B:2,PS:16,\
                    NB:2,E:2,MC:1,MMAX:2,HB:2,GW:2,GH:4,F:8,HH:16,Vr:32,Gx:4,Gy:4,Gb:1,vocab:250,PASSES:3,\
                    UNTIED:0,CERT:0,MODE:0,OVL:0,VMC:1,WF32:1,CAP:8,DBG:0,SPEC:0,SPEC_H:1,SPEC_T:1,SPEC_O:1,\
                    SMP:0,HLM:4,iters:1,VM:1,max_ops:48,nreg:6,pool_words:64,feat:99,ring_w:1,ring_d:32,vm_prof:0,\
                    vm_sumpipe:3,vm_l2w:1";
        let mut want: Vec<(String, String)> = want
            .split(',')
            .map(|kv| {
                kv.split_once(':')
                    .map(|(k, v)| (k.trim().to_string(), v.to_string()))
                    .unwrap()
            })
            .collect();
        let mut got = p.clone();
        want.sort();
        got.sort();
        assert_eq!(got, want);
    }

    #[test]
    fn vm_options_reach_the_params_and_the_layout_is_checked() {
        let (plan, mut cfg) = pipe_config();
        let vm = cfg.vm.as_mut().unwrap();
        vm.step_input = 4;
        vm.opmask = Some((0x0000_ffff, 0));
        let (_, p) = params(&plan, &cfg).unwrap();
        let get = |k: &str| p.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
        assert_eq!(
            (get("vm_sin"), get("opmask_lo"), get("opmask_hi")),
            (Some("4".into()), Some("65535".into()), Some("0".into()))
        );
        assert_eq!(get("vm_rows"), None);
        // vm_lprog is not in this layout's snapshot yet: refused, not dropped.
        cfg.vm.as_mut().unwrap().lane_programs = Some(256);
        let declared = layout_params(&Design::Serve2dPipe.files().unwrap()[0].1);
        if !declared.iter().any(|(n, _)| n == "vm_lprog") {
            assert!(params(&plan, &cfg).unwrap_err().contains("vm_lprog"));
        }
    }

    /// serve2d's one-sequence grids (GRID 0) are refused, by the plan's
    /// pins and by an explicit compile alike.
    #[test]
    fn serve2d_grid_0_is_refused() {
        let (plan, mut cfg) = pipe_config();
        cfg.goals.pipeline = false;
        cfg.vm = None;
        cfg.mmax = None;
        cfg.hb = None;
        cfg.hlm = None;
        cfg.spec = None;
        cfg.smp = false;
        let plan2 =
            SpatialPlan::derive(&plan.spec, crate::spatial::plan::WaferDims::CS3, cfg.goals)
                .unwrap();
        let (d, p) = params(&plan2, &cfg).unwrap();
        assert_eq!(d, Design::Serve2d);
        assert!(shapes::check(d, &p).is_ok());
        cfg.pins.set("GRID", 0).unwrap();
        assert!(
            params(&plan2, &cfg)
                .unwrap_err()
                .contains("GRID:0 is unsupported")
        );
        let grid0: Vec<(String, String)> = p
            .iter()
            .map(|(k, v)| match k.as_str() {
                "GRID" => (k.clone(), "0".to_string()),
                _ => (k.clone(), v.clone()),
            })
            .collect();
        let program = ProgramSpec::from_params(Design::Serve2d, &grid0).unwrap();
        assert!(
            compile(&program, crate::sdk::Target::Wse3, false)
                .unwrap_err()
                .contains("hung on hardware")
        );
        let explicit = vec![("GRID".to_string(), "0".to_string())];
        assert!(shapes::check(Design::Serve2dPipe, &explicit).is_ok());
    }

    /// serve2d_pipe: only the shapes hardware ran right, pins included.
    #[test]
    fn serve2d_pipe_refuses_its_unvalidated_shapes() {
        let p = |s: &str| -> Vec<(String, String)> {
            s.split(',')
                .map(|kv| {
                    kv.split_once(':')
                        .map(|(k, v)| (k.to_string(), v.to_string()))
                        .unwrap()
                })
                .collect()
        };
        let bad = |s: &str| shapes::check(Design::Serve2dPipe, &p(s)).unwrap_err();
        for ok in [
            "B:4,HB:4,MMAX:8,CAP:32",
            "B:4,HB:4,MMAX:4,CAP:16,VM:1",
            "B:2,HB:2,MMAX:2,CAP:8,VM:1",
        ] {
            assert!(shapes::check(Design::Serve2dPipe, &p(ok)).is_ok(), "{ok}");
        }
        assert!(bad("B:16,HB:4,MMAX:4,CAP:12").contains("B > 8 is unsupported"));
        assert!(bad("B:8,HB:8,MMAX:8,CAP:16").contains("HB > 4 is unsupported"));
        assert!(bad("B:8,HB:4,MMAX:4,CAP:16").contains("unrun"));
        assert!(bad("B:4,HB:4,MMAX:9,CAP:16").contains("unrun"));
        assert!(bad("B:4,HB:4,MMAX:8,CAP:16,VM:1").contains("unrun"));
        // The validated symmetric 8/8 VMCONC, and nothing else of it.
        let conc = "B:4,HB:4,MMAX:4,CAP:16,VM:1,VMB:6,vm_lanes:8,b_vm_lanes:8,VMCONC:1";
        assert!(shapes::check(Design::Serve2dPipe, &p(conc)).is_ok());
        assert!(bad(&conc.replace("b_vm_lanes:8", "b_vm_lanes:2")).contains("VMCONC runs only"));
        assert!(bad(&conc.replace("MMAX:4", "MMAX:8")).contains("VMCONC runs only"));
        let (plan, mut cfg) = pipe_config();
        assert!(params(&plan, &cfg).is_ok());
        cfg.hb = Some(1);
        assert!(params(&plan, &cfg).unwrap_err().contains("unrun"));
    }

    /// `[pins]` takes the typed pins by their layout names, and only those.
    #[test]
    fn pins_are_typed() {
        let t: toml::Table = toml::from_str("[pins]\nGW = 3\nKVST = 1\n").unwrap();
        let pins = ServeConfig::from_toml(&t, 16).unwrap().pins;
        assert_eq!((pins.gw, pins.staging, pins.gh), (Some(3), Some(1), None));
        let t: toml::Table = toml::from_str("[pins]\nPASSES = 1\n").unwrap();
        assert!(
            ServeConfig::from_toml(&t, 16)
                .unwrap_err()
                .contains("`PASSES` is not a pin")
        );
    }

    #[test]
    fn serve_config_reads_its_table() {
        let t: toml::Table = toml::from_str(
            "lanes = 2\npipeline = true\ncap = 8\nmmax = 2\nhb = 2\n[vm]\nmax_ops = 48\nnreg = 6\npool_words = 64\nfeat = 99\nopmask_hi = 0\n",
        )
        .unwrap();
        let c = ServeConfig::from_toml(&t, 16).unwrap();
        assert_eq!(
            (
                c.goals.lanes,
                c.goals.page_size,
                c.goals.pipeline,
                c.goals.cap
            ),
            (2, 16, true, Some(8))
        );
        assert_eq!((c.mmax, c.hb), (Some(2), Some(2)));
        let vm = c.vm.unwrap();
        assert_eq!(
            (vm.max_ops, vm.ring_w, vm.ring_d, vm.opmask),
            (48, 1, 32, Some((0xffff_ffff, 0)))
        );
        let bad: toml::Table = toml::from_str("lanse = 2\n").unwrap();
        assert!(
            ServeConfig::from_toml(&bad, 16)
                .unwrap_err()
                .contains("lanse")
        );
    }

    /// The second VM set: `[vm_b]` with its columns, as the layout's
    /// VMB and `b_*` params, the program one VMB wider.
    #[test]
    fn a_second_vm_set_reaches_the_layout() {
        let (plan, mut cfg) = pipe_config();
        let t: toml::Table = toml::from_str(
            "[vm_b]\ncols = 1\nmax_ops = 64\nnreg = 8\npool_words = 128\nfeat = 7\nlanes = 2\nbank_words = 900\n",
        )
        .unwrap();
        let (img, cols) = ServeConfig::from_toml(&t, 16).unwrap().vm_b.unwrap();
        assert_eq!((img.max_ops, img.lane_programs, cols), (64, Some(900), 1));
        let one = rectangle(Design::Serve2dPipe, &params(&plan, &cfg).unwrap().1).unwrap();
        cfg.vm_b = Some((img, cols));
        let (_, p) = params(&plan, &cfg).unwrap();
        let get = |k: &str| p.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
        assert_eq!(
            (
                get("VMB"),
                get("b_max_ops"),
                get("b_vm_lprog"),
                get("b_bank_words")
            ),
            (
                Some("1".into()),
                Some("64".into()),
                Some("1".into()),
                Some("900".into())
            )
        );
        assert_eq!(
            rectangle(Design::Serve2dPipe, &p).unwrap(),
            (one.0 + 1, one.1)
        );
        let no_cols: toml::Table =
            toml::from_str("[vm_b]\nmax_ops = 64\nnreg = 8\npool_words = 128\nfeat = 7\n").unwrap();
        assert!(
            ServeConfig::from_toml(&no_cols, 16)
                .unwrap_err()
                .contains("cols")
        );
    }

    /// The csl README's `[vm_b]` (serve2d_pipe's set B as hardware ran it)
    /// parses to the hardware build's `b_*` parameters with the fitted limits.
    #[test]
    fn the_readme_vm_b_is_the_fitted_set_b() {
        let readme = include_str!("csl/README.md");
        let at = readme
            .find("[cerebras.spatial.serve.vm_b]")
            .expect("the README's [vm_b]");
        let block: String = readme[at..readme[at..].find("```").map_or(readme.len(), |e| at + e)]
            .lines()
            .skip(1)
            .map(|l| format!("{}\n", l.trim()))
            .collect();
        let t: toml::Table = toml::from_str(&format!("[vm_b]\n{block}")).unwrap();
        let (img, cols) = ServeConfig::from_toml(&t, 16).unwrap().vm_b.unwrap();
        let got: Vec<(&str, usize)> = img.params_b(cols);
        let want = "VMB:6,b_max_ops:136,b_nreg:6,b_pool_words:904,b_feat:495,b_ring_w:8,b_ring_d:16,b_vm_prof:0,\
                    b_vm_lanes:2,b_vm_rows:2,b_vm_lprog:1,b_bank_words:640,b_vm_pwalk:1,b_vm_sin:0,\
                    b_opmask_lo:1878015419,b_opmask_hi:256";
        let want: Vec<(&str, usize)> = want
            .split(',')
            .map(|kv| {
                kv.split_once(':')
                    .map(|(k, v)| (k, v.parse().unwrap()))
                    .unwrap()
            })
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn a_plan_without_a_program_is_refused() {
        let goals = Goals {
            pipeline: true,
            ..Goals::lanes(2)
        };
        let plan = SpatialPlan::derive(&spec(), WaferDims::CS3, goals).unwrap();
        assert!(
            params(&plan, &ServeConfig::new(goals))
                .unwrap_err()
                .contains("no full program")
        );
    }
}

#[cfg(test)]
mod qwen3_8b {
    use super::*;
    use crate::spatial::model::ModelSpec;
    use crate::spatial::plan::WaferDims;

    fn spec() -> ModelSpec {
        let catalog = poem_compiler::catalog::repository();
        let d = catalog
            .parse("qwen3-8b-bf16-kv-bf16")
            .expect("qwen3-8b parses");
        ModelSpec::read(&d.trace(poem_ir::Platform::Cerebras)).expect("qwen3-8b reads")
    }

    fn sorted(extra: &str) -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = extra
            .split(',')
            .map(|kv| {
                kv.split_once(':')
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .unwrap()
            })
            .collect();
        v.sort();
        v
    }

    /// The two serve8b builds validated on CS-3 with real Qwen3-8B (B 1 with
    /// copy kernels, B 4 with host copies) come out of the plan and a config.
    #[test]
    fn the_plan_places_qwen3_8b_as_the_hardware_builds() {
        let base = "PX:128,PY:144,Hc:29,HR:4096,Dc:8,Gs:16,Qh:4,Ic:96,PS:16,E:1,MC:1,W16:1,NP:3,KV16:1,PROF:0,TIME:0,\
                    GW:5,GH:8,GS:4,HEADMODE:2,HH:1152,Vr:132,Gx:8,Gy:24,vocab:151936,PASSES:3,MODE:0,OVL:0,VMC:1,\
                    CAP:64,iters:1";
        for (b, nb, host, np) in [(1, 1, false, 3), (4, 1, true, 1)] {
            let goals = Goals {
                blocks: Some(nb),
                host_copy: host,
                grid_lanes: b,
                ..Goals::lanes(b)
            };
            let plan = SpatialPlan::derive(&spec(), WaferDims::CS3, goals).expect("a plan");
            let mut cfg = ServeConfig::new(goals);
            if np != 3 {
                cfg.pins.set("NP", np).unwrap();
            }
            let (d, p) = params(&plan, &cfg).unwrap();
            assert_eq!(d, Design::Serve8b);
            let mut got = p.clone();
            got.sort();
            let _ = host;
            let want = format!("{base},B:{b},NB:{nb},KVCP:0,KVST:{}", usize::from(b == 1))
                .replace("NP:3", &format!("NP:{np}"));
            assert_eq!(got, sorted(&want), "B {b}");
            // The model accepts the program (untied, 36 layers in strips of 5
            // and 4), and the engine reads it as serve8b's layout.
            let get = |k: &str| p.iter().find(|e| e.0 == k).and_then(|e| e.1.parse().ok());
            plan.spec
                .check_program(get)
                .expect("the model takes serve8b");
            let lay = crate::spatial::program::ProgramSpec::from_params(d, &p).expect("serve8b");
            assert_eq!(
                (lay.layers(), lay.chunk_cols(), lay.root()),
                (36, 72, (640 + 36, 576))
            );
            assert_eq!(lay.paged().map(|s| (s.lanes, s.pages())), Some((b, 144)));
            assert!(lay.host_copy());
        }
    }

    /// serve8b_region's two hardware builds come out of the plan with
    /// `region`.
    #[test]
    fn the_plan_places_qwen3_8b_regions_as_the_hardware_builds() {
        let base = "PX:128,PY:144,Hc:29,Dc:8,Gs:16,Ic:96,RPX:256,RPY:240,RHc:18,RDc:4,RGs:32,RIc:48,RNL:3,RGY:16,\
                    RC:2,RR:4,NCOL:7,RTR:4,HR:4096,Qh:4,PS:16,NB:1,E:1,MC:1,W16:1,KV16:1,KVCP:0,PROF:0,TIME:0,\
                    HROUTE:1,ROT8:1,NOLAYER:0,HEADMODE:2,HH:1152,Vr:132,Gx:8,Gy:24,vocab:151936,PASSES:3,MODE:0,\
                    OVL:0,VMC:1,CAP:64,iters:1";
        // The grids of the serve8b builds (B 1 plans for page-copy kernels;
        // serve8b_region copies by the host either way).
        for (b, np, host) in [(1, 3, false), (4, 1, true)] {
            let goals = Goals {
                blocks: Some(1),
                host_copy: host,
                grid_lanes: b,
                ..Goals::lanes(b)
            };
            let plan = SpatialPlan::derive(&spec(), WaferDims::CS3, goals).expect("a plan");
            let mut cfg = ServeConfig::new(goals);
            cfg.region = true;
            if np != 3 {
                cfg.pins.set("NP", np).unwrap();
            }
            let (d, p) = params(&plan, &cfg).unwrap();
            assert_eq!(d, Design::Serve8bRegion);
            let mut got = p.clone();
            got.sort();
            assert_eq!(
                got,
                sorted(&format!(
                    "{base},B:{b},NP:{np},KVST:{}",
                    usize::from(b == 1)
                )),
                "B {b}"
            );
            let get = |k: &str| p.iter().find(|e| e.0 == k).and_then(|e| e.1.parse().ok());
            plan.spec
                .check_program(get)
                .expect("the model takes serve8b_region");
            let lay =
                crate::spatial::program::ProgramSpec::from_params(d, &p).expect("serve8b_region");
            assert_eq!((lay.layers(), lay.root()), (36, (676, 576)));
            let p2: Vec<(String, String)> = p.clone();
            assert_eq!(rectangle(d, &p2).unwrap(), (749, 1153));
        }
    }

    /// `region` with `pipeline` is the validated serve8b_region_pipe builds (B 4 x 4
    /// micro-batches at CAP 16, B 2 x 8 and B 1 x 12 at CAP 32); more
    /// micro-batches, a larger CAP, other lanes or grids, the VM or staging
    /// are refused, by the plan and by pins.
    #[test]
    fn the_plan_places_qwen3_8b_pipe_as_validated_and_refuses_the_rest() {
        let base = "PX:128,PY:144,Hc:29,Dc:8,Gs:16,Ic:96,RPX:256,RPY:240,RHc:18,RDc:4,RGs:32,RIc:48,RNL:3,RGY:16,\
                    RC:2,RR:4,NCOL:7,RTR:4,HR:4096,Qh:4,PS:16,NB:1,E:1,MC:1,W16:1,KV16:1,KVCP:0,PROF:0,TIME:0,\
                    HROUTE:1,ROT8:1,NOLAYER:0,HEADMODE:2,HH:1152,Vr:132,Gx:8,Gy:24,vocab:151936,PASSES:3,MODE:0,\
                    OVL:0,VMC:1,iters:1,NP:1,PIPE:1,KVST:0";
        let cfg_for = |b: usize, mmax: Option<usize>| {
            // The ring's grids are serve8b's B 4 grids at every B.
            let goals = Goals {
                blocks: Some(1),
                host_copy: true,
                grid_lanes: 4,
                pipeline: true,
                ..Goals::lanes(b)
            };
            let plan = SpatialPlan::derive(&spec(), WaferDims::CS3, goals).expect("a plan");
            let mut cfg = ServeConfig::new(goals);
            cfg.region = true;
            cfg.mmax = mmax;
            (plan, cfg)
        };
        for (b, m, cap) in [(4, 4, 16), (2, 8, 32), (1, 12, 32)] {
            let (plan, cfg) = cfg_for(b, None);
            let (d, p) = params(&plan, &cfg).unwrap();
            assert_eq!(d, Design::Serve8bRegionPipe);
            let mut got = p.clone();
            got.sort();
            assert_eq!(
                got,
                sorted(&format!("{base},B:{b},MMAX:{m},CAP:{cap}")),
                "B {b}"
            );
            assert_eq!(rectangle(d, &p).unwrap(), (749, 1153));
            // Fewer micro-batches are the same program's.
            let (plan, cfg) = cfg_for(b, Some(2));
            assert!(params(&plan, &cfg).is_ok());
            // More are not.
            let (plan, cfg) = cfg_for(b, Some(m + 1));
            assert!(params(&plan, &cfg).unwrap_err().contains("micro-batches"));
            // Nor are pins past the validated shapes.
            for pin in [("MMAX", m + 1), ("CAP", 2 * cap), ("KVST", 1)] {
                let (plan, mut cfg) = cfg_for(b, None);
                cfg.pins.set(pin.0, pin.1).unwrap();
                assert!(
                    params(&plan, &cfg).unwrap_err().contains("unsupported"),
                    "{pin:?}"
                );
            }
        }
        // B 8 has not run.
        let (plan, cfg) = cfg_for(8, None);
        assert!(params(&plan, &cfg).unwrap_err().contains("B 8"));
        assert!(
            shapes::check(
                Design::Serve8bRegionPipe,
                &sorted(&format!("{base},B:8,MMAX:2,CAP:16"))
            )
            .is_err()
        );
        assert!(
            shapes::check(
                Design::Serve8bRegionPipe,
                &sorted(&format!("{base},B:1,MMAX:12,CAP:32"))
            )
            .is_ok()
        );
        // B 2 on B 2's own grid (PY 128) is unrun.
        let goals = Goals {
            blocks: Some(1),
            host_copy: true,
            grid_lanes: 2,
            pipeline: true,
            ..Goals::lanes(2)
        };
        let plan = SpatialPlan::derive(&spec(), WaferDims::CS3, goals).expect("a plan");
        let mut cfg = ServeConfig::new(goals);
        cfg.region = true;
        assert!(params(&plan, &cfg).unwrap_err().contains("PY"));
    }

    /// With the guest4hz VM image the region plan is the real-VM builds (the
    /// VM block, Vr 160, HH 952, HLBN 1 at B 4; VML2W 1, and bank_words 512
    /// at B 4 from their later fit), staging at B 1.
    #[test]
    fn the_plan_places_qwen3_8b_regions_with_the_vm() {
        let base = "PX:128,PY:144,Hc:29,Dc:8,Gs:16,Ic:96,RPX:256,RPY:240,RHc:18,RDc:4,RGs:32,RIc:48,RNL:3,RGY:16,\
                    RC:2,RR:4,NCOL:7,RTR:4,HR:4096,Qh:4,PS:16,NB:1,E:1,MC:1,W16:1,KV16:1,KVCP:0,PROF:0,TIME:0,\
                    HROUTE:1,ROT8:1,NOLAYER:0,HEADMODE:2,HH:952,Vr:160,Gx:8,Gy:28,vocab:151936,PASSES:3,MODE:0,\
                    OVL:0,VMC:0,CAP:64,iters:1,VMX:1,VMREAL:1,max_ops:104,nreg:12,pool_words:211,feat:1007,\
                    opmask_lo:3757088255,opmask_hi:256,ring:16,ring_w:5,vm_rows:5,vm_sin:5,vm_lprog:1,vm_pwalk:1,\
                    vm_prof:0,VML2W:1,VMSUMPIPE:1";
        let vm = VmImage {
            max_ops: 104,
            nreg: 12,
            pool_words: 211,
            feat: 1007,
            ring_w: 5,
            ring_d: 16,
            prof: false,
            step_input: 5,
            rows: 5,
            lane_programs: Some(768),
            opmask: Some((3_757_088_255, 256)),
            lanes: None,
            pwalk: Some(true),
            sumpipe: None,
            l2w: None,
        };
        for (b, np, host, extra) in [
            (1, 3, false, "bank_words:768,vm_lanes:1,B:1,NP:3,KVST:1"),
            (
                4,
                1,
                true,
                "bank_words:512,vm_lanes:4,HLBN:1,B:4,NP:1,KVST:0",
            ),
        ] {
            let goals = Goals {
                blocks: Some(1),
                host_copy: host,
                grid_lanes: b,
                ..Goals::lanes(b)
            };
            let plan = SpatialPlan::derive(&spec(), WaferDims::CS3, goals).expect("a plan");
            let mut cfg = ServeConfig::new(goals);
            cfg.region = true;
            cfg.vm = Some(vm);
            if np != 3 {
                cfg.pins.set("NP", np).unwrap();
            }
            let (d, p) = params(&plan, &cfg).unwrap();
            assert_eq!(d, Design::Serve8bRegion);
            let mut got = p.clone();
            got.sort();
            assert_eq!(got, sorted(&format!("{base},{extra}")), "B {b}");
            assert_eq!(rectangle(d, &p).unwrap(), (753, 1152));
            let lay =
                crate::spatial::program::ProgramSpec::from_params(d, &p).expect("serve8b_region");
            assert_eq!(lay.layers(), 36);
        }
    }
}
