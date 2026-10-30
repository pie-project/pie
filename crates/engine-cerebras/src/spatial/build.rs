//! Building a wafer program in PIE: the plan's parameters plus a small serve
//! config, generated sources ([`super::codegen`]), `cslc`, the compile cache,
//! and a manifest saying what was built.
//!
//! Before this, the engine loaded a directory compiled outside PIE and
//! guessed the design from `out.json` fields, RPC arities and ELF symbol
//! sizes. A directory built here carries `pie_manifest.json` (the design and
//! its parameters), and readers take the design from it
//! ([`super::manifest`]).

use std::path::PathBuf;
use std::time::Instant;

use super::codegen::Design;
use super::manifest::Manifest;
use super::plan::{Family, Goals, SpatialPlan};

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
        out
    }

    /// As the second VM set (guest4h2s's set B, `cols` VMB columns): the
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
/// overrides a plan-derived structural parameter by name. It exists for small
/// test shapes the plan's rules don't produce, and is checked against the
/// layout's parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServeConfig {
    pub goals: Goals,
    /// serve2d_pipe: micro-batches a launch and lanes a head step.
    pub mmax: Option<usize>,
    pub hb: Option<usize>,
    /// The eta VM image; `None`: no VM.
    pub vm: Option<VmImage>,
    /// guest4h2s's second resident VM set (serve2d_pipe): its image and its
    /// columns (`VMB`), east of the first set's.
    pub vm_b: Option<(VmImage, usize)>,
    /// Virtual logits lanes a launch (`HLM`, serve2d_pipe).
    pub hlm: Option<usize>,
    pub spec: Option<Spec>,
    /// The fused naive sampler on the head.
    pub smp: bool,
    /// A paged8b plan as serve8b_region's units (grids and several-layer
    /// regions) instead of serve8b's strips.
    pub region: bool,
    pub pins: Vec<(&'static str, usize)>,
}

impl ServeConfig {
    pub fn new(goals: Goals) -> ServeConfig {
        ServeConfig { goals, mmax: None, hb: None, vm: None, vm_b: None, hlm: None, spec: None, smp: false, region: false, pins: Vec::new() }
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
    ///   `pwalk` (`vm_pwalk`);
    /// - `[vm_b]` (serve2d_pipe's second VM set, guest4h2s): the same keys
    ///   for its image (the layout's `b_*`) and `cols` (`VMB`);
    /// - `[pins]`: layout parameters by name, for shapes the plan does not
    ///   place (simulator-sized models); checked against the layout.
    pub fn from_toml(t: &toml::Table, page_size: usize) -> R<ServeConfig> {
        fn int(t: &toml::Table, k: &str) -> R<Option<usize>> {
            match t.get(k) {
                None => Ok(None),
                Some(v) => v
                    .as_integer()
                    .and_then(|i| usize::try_from(i).ok())
                    .map(Some)
                    .ok_or_else(|| format!("[cerebras.spatial.serve] `{k}` is not a non-negative integer")),
            }
        }
        fn flag(t: &toml::Table, k: &str) -> R<bool> {
            match t.get(k) {
                None => Ok(false),
                Some(v) => v.as_bool().ok_or_else(|| format!("[cerebras.spatial.serve] `{k}` is not a boolean")),
            }
        }
        let known = [
            "lanes", "grid_lanes", "page_size", "blocks", "max_blocks", "mlp_chunk", "cap", "pipeline",
            "host_copy", "mmax", "hb", "hlm", "smp", "spec", "vm", "vm_b", "pins", "region",
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
                let s = v.as_table().ok_or("[cerebras.spatial.serve] `spec` is not a table")?;
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
                let m = v.as_table().ok_or_else(|| format!("[cerebras.spatial.serve] `{key}` is not a table"))?;
                let need = |k: &str| int(m, k)?.ok_or_else(|| format!("[cerebras.spatial.serve.{key}] needs `{k}`"));
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
                    opmask: (lo.is_some() || hi.is_some())
                        .then(|| (lo.unwrap_or(0xffff_ffff) as u32, hi.unwrap_or(0xffff_ffff) as u32)),
                    lanes: int(m, "lanes")?,
                    pwalk: m.get("pwalk").map(|_| flag(m, "pwalk")).transpose()?,
                })
                .map(|img| Some((img, int(m, "cols").ok().flatten())))
            }
        };
        let vm = image("vm")?.map(|(img, _)| img);
        let vm_b = match image("vm_b")? {
            Some((img, Some(cols))) if cols > 0 => Some((img, cols)),
            Some(_) => return Err("[cerebras.spatial.serve.vm_b] needs `cols` (VMB, its VM columns)".into()),
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
                None => Vec::new(),
                Some(v) => {
                    let m = v.as_table().ok_or("[cerebras.spatial.serve] `pins` is not a table")?;
                    let mut out = Vec::new();
                    for k in m.keys() {
                        let v = int(m, k)?.unwrap_or(0);
                        // Pins name layout parameters; the names live as long as the program.
                        out.push((&*Box::leak(k.clone().into_boxed_str()), v));
                    }
                    out
                }
            },
        })
    }
}

/// The program PIE builds for `spec` under `cfg` on `target`: the plan,
/// its parameters, the compile (cached). Returns the output directory.
pub fn build_for(
    spec: &super::plan::ModelSpec,
    cfg: &ServeConfig,
    target: crate::sdk::Target,
    system: bool,
) -> R<PathBuf> {
    let plan = SpatialPlan::derive(spec, super::plan::WaferDims::CS3, cfg.goals).map_err(|e| e.to_string())?;
    let (design, params) = params(&plan, cfg)?;
    tracing::info!(design = design.name(), model = %spec.name, "spatial build: the plan's program");
    compile(design, &params, target, system)
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
        return Err(format!("serve8b wants PY a multiple of 4 in [PX, 2 PX]; the plan's grid is {px} x {py}"));
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
            let gx = [16, 8, 4, 2].into_iter().find(|g| c % g == 0 && c / g >= 2).unwrap_or(1);
            let gx = gx.min(8);
            let gy = (2..=24).rev().step_by(2).find(|g| f % g == 0).unwrap_or(2);
            return Ok(Serve8bPlace { gw, gh, gs, hh, vr: vocab.div_ceil(hh), gx, gy });
        }
    }
    Err(format!("{}: {layers} grids of {px} x {py} and serve8b's head do not fit {} x {}", plan.spec.name, wafer.width, wafer.height))
}

/// serve8b's compile parameters, in the requests' order.
fn serve8b_params(plan: &SpatialPlan, cfg: &ServeConfig) -> R<Vec<(&'static str, usize)>> {
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
    Ok(vec![
        ("PX", g.px), ("PY", g.py), ("Hc", g.hc), ("HR", g.hidden), ("Dc", g.dc), ("Gs", g.gs), ("Qh", g.qh),
        ("Ic", g.ic), ("PS", g.page_size), ("E", g.entries), ("MC", g.mlp_chunk), ("W16", usize::from(g.w16)),
        ("NP", 3), ("KV16", usize::from(g.kv16)), ("PROF", 0), ("TIME", 0), ("GW", p.gw), ("GH", p.gh),
        ("GS", p.gs), ("HEADMODE", 2), ("HH", p.hh), ("Vr", p.vr), ("Gx", p.gx), ("Gy", p.gy),
        ("vocab", plan.vocab), ("PASSES", 3), ("MODE", 0), ("OVL", 0), ("VMC", 1), ("CAP", cap), ("iters", 1),
        ("B", g.lanes), ("NB", g.blocks), ("KVCP", usize::from(!g.host_copy)),
        // KV page staging: a slot a PE where it fits (B 1; at B 4 the grid
        // has no room for it).
        ("KVST", usize::from(g.lanes == 1)),
    ])
}

/// serve8b_region's placement of a paged8b plan (DESIGN.md's option A): the
/// plan's grid for the first and last layers (a top band of `RC RPX / PX
/// + 1` grids, `NCOL` grids down the column beside the head) and between
/// them `RC` x `RR` regions of `RNL` layers, twice the grid's width (`RGs`
/// 2 Gs columns a kv head of `RDc` Dc / 2 dims, `RIc` Ic / 2) and `RPY` 240
/// rows (`RHc` = ceil(hidden / RPY)). RC 2, RNL 3, RR 4 as built; `NCOL`
/// the layers left. The head as serve8b's, `HH` = the grid column's rows.
fn serve8b_region_params(plan: &SpatialPlan, cfg: &ServeConfig) -> R<Vec<(&'static str, usize)>> {
    if cfg.spec.is_some() || cfg.smp || cfg.goals.pipeline || cfg.vm_b.is_some() {
        return Err("serve8b_region has no speculative decode, fused sampler, pipeline or second VM set".into());
    }
    let g = &plan.grid;
    let wafer = super::plan::WaferDims::CS3;
    let (rc, rnl, rr, rpy, rgy) = (2, 3, 4, 240, 16);
    let (rpx, band) = (2 * g.px, 2 * 2 * g.px / g.px + 1);
    if g.dc % 4 != 0 || g.ic % 2 != 0 || plan.layers < band + rc * rr * rnl + 1 {
        return Err(format!("{}: serve8b_region's option A does not take this grid ({g:?})", plan.spec.name));
    }
    let ncol = plan.layers - band - rc * rr * rnl;
    let c = g.py / 2;
    // With an eta VM (VMX, VM5.md): Fork 3's VM as a 5-column block east of
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
        return Err(format!("{}: serve8b_region's {width} x {} does not fit the wafer", plan.spec.name, hh + 1));
    }
    let f = hh / 2;
    if f < g.py || f >= g.py * ncol {
        return Err(format!("{}: serve8b_region's feeder row {f} is not between the grids' {} and {}", plan.spec.name, g.py, g.py * ncol));
    }
    let gx = [16, 8, 4, 2].into_iter().find(|x| c % x == 0 && c / x >= 2).unwrap_or(1).min(8);
    let gy_most = if cfg.vm.is_some() { 28 } else { 24 };
    let gy = (2..=gy_most).rev().step_by(2).find(|x| f % x == 0).unwrap_or(2);
    let cap = cfg.goals.cap.unwrap_or_else(|| {
        let mut c = 64;
        while g.lanes * c > 512 && c > 1 {
            c /= 2;
        }
        c
    });
    // KV page staging at B 1 (at B 4 the units have no room for it).
    let mut out = vec![("KVST", usize::from(g.lanes == 1))];
    if let Some(vm) = cfg.vm {
        // Fork 3's VM image on the block; one VM lane a head lane, and the
        // head keeps one lane's logits rows at B > 1 (room on its PEs).
        if vm.lanes.is_some_and(|l| l != g.lanes) || vm.lane_programs.is_none() {
            return Err("serve8b_region's VM holds per-lane programs (`bank_words`), one lane a head lane".into());
        }
        let (lo, hi) = vm.opmask.unwrap_or((0xffff_ffff, 0xffff_ffff));
        out.extend([
            ("VMX", 1), ("VMREAL", 1), ("max_ops", vm.max_ops), ("nreg", vm.nreg), ("pool_words", vm.pool_words),
            ("feat", vm.feat), ("opmask_lo", lo as usize), ("opmask_hi", hi as usize), ("ring", vm.ring_d),
            ("ring_w", vm.ring_w), ("vm_rows", vm.rows), ("vm_sin", vm.step_input), ("vm_lprog", 1),
            ("vm_pwalk", usize::from(vm.pwalk.unwrap_or(true))),
            // Several lanes: 128 bank words a lane at most (Fork 11's B 4 fit:
            // every current program's live window, 33-108 words, admits).
            ("bank_words", if g.lanes > 1 { vm.lane_programs.unwrap_or(0).min(128 * g.lanes) } else { vm.lane_programs.unwrap_or(0) }),
            ("vm_prof", usize::from(vm.prof)), ("vm_lanes", g.lanes),
            // Fork 3's VM options as Fork 11 fit them: the last hub's level-2
            // list in wide; the pipelined canonical sum (2, concatenating, is
            // B 1's once hardware shows its gain).
            ("VML2W", 1), ("VMSUMPIPE", 1),
        ]);
        if g.lanes > 1 {
            out.push(("HLBN", 1));
        }
    }
    out.extend([
        ("PX", g.px), ("PY", g.py), ("Hc", g.hc), ("Dc", g.dc), ("Gs", g.gs), ("Ic", g.ic), ("RPX", rpx),
        ("RPY", rpy), ("RHc", g.hidden.div_ceil(rpy)), ("RDc", g.dc / 2), ("RGs", 2 * g.gs), ("RIc", g.ic / 2),
        ("RNL", rnl), ("RGY", rgy), ("RC", rc), ("RR", rr), ("NCOL", ncol), ("RTR", 4), ("HR", g.hidden),
        ("Qh", g.qh), ("PS", g.page_size), ("NB", g.blocks), ("E", g.entries), ("MC", g.mlp_chunk),
        ("W16", usize::from(g.w16)), ("KV16", usize::from(g.kv16)), ("KVCP", 0), ("PROF", 0), ("TIME", 0),
        ("HROUTE", 1), ("ROT8", 1), ("NOLAYER", 0), ("HEADMODE", 2), ("HH", hh), ("Vr", vr),
        ("Gx", gx), ("Gy", gy), ("vocab", plan.vocab), ("PASSES", 3), ("MODE", 0), ("OVL", 0), ("VMC", vmc),
        ("CAP", cap), ("iters", 1), ("B", g.lanes), ("NP", 3),
    ]);
    Ok(out)
}

/// Builds PIE refuses to compile: serve2d's one-sequence qwen2d grids
/// (`GRID` 0) loaded and then hung on hardware (0150), and nothing has
/// checked them since the paged and ring work.
fn refused(design: Design, params: &[(String, String)]) -> R<()> {
    let grid = params.iter().find(|(k, _)| k == "GRID").map(|(_, v)| v.trim());
    if design == Design::Serve2d && grid == Some("0") {
        return Err("serve2d GRID:0 is unsupported (hung on hardware, 0150); use GRID:1 with B:1 for one sequence".into());
    }
    Ok(())
}

/// The layout's parameters: (name, has a default), in declaration order.
pub fn layout_params(layout: &str) -> Vec<(String, bool)> {
    layout
        .lines()
        .filter_map(|l| l.trim_start().strip_prefix("param "))
        .filter_map(|rest| {
            let (name, ty) = rest.split_once(':')?;
            let name = name.trim();
            (!ty.contains("color") && !name.is_empty()).then(|| (name.to_string(), ty.contains('=')))
        })
        .collect()
}

/// The compile parameters of `plan` built for `cfg`, in the layout's order,
/// checked against the layout: every parameter without a default is set,
/// and nothing unknown is passed.
pub fn params(plan: &SpatialPlan, cfg: &ServeConfig) -> R<(Design, Vec<(String, String)>)> {
    let d = match design(plan)? {
        Design::Serve8b if cfg.region => Design::Serve8bRegion,
        _ if cfg.region => return Err(format!("{}: `region` places paged8b (8B) plans only", plan.spec.name)),
        d => d,
    };
    if !matches!(d, Design::Serve8b | Design::Serve8bRegion) && plan.program.is_none() && !cfg.pins.iter().any(|p| p.0 == "GW") {
        return Err(format!("{}: no full program places the model: {}", plan.spec.name, plan.blockers.join("; ")));
    }
    let mut set: Vec<(&'static str, usize)> = match d {
        Design::Serve8b => serve8b_params(plan, cfg)?,
        Design::Serve8bRegion => serve8b_region_params(plan, cfg)?,
        _ => plan.params(),
    };
    let put = |set: &mut Vec<(&'static str, usize)>, k: &'static str, v: usize| match set.iter_mut().find(|e| e.0 == k) {
        Some(e) => e.1 = v,
        None => set.push((k, v)),
    };
    if d == Design::Serve2dPipe {
        if let Some(m) = cfg.mmax {
            put(&mut set, "MMAX", m);
        }
        if let Some(h) = cfg.hb {
            put(&mut set, "HB", h);
        }
        let sp = cfg.spec.unwrap_or(Spec { history: 1, table: 1, out: 1 });
        put(&mut set, "SPEC", usize::from(cfg.spec.is_some()));
        put(&mut set, "SPEC_H", sp.history);
        put(&mut set, "SPEC_T", sp.table);
        put(&mut set, "SPEC_O", sp.out);
        put(&mut set, "SMP", usize::from(cfg.smp));
        if let Some(h) = cfg.hlm {
            put(&mut set, "HLM", h);
        }
    } else if !matches!(d, Design::Serve8b | Design::Serve8bRegion) && (cfg.spec.is_some() || cfg.smp || cfg.mmax.is_some()) {
        return Err("speculative decode, the fused sampler and MMAX are serve2d_pipe's".into());
    }
    let vm_b: Vec<(&'static str, usize)> = cfg.vm_b.map(|(img, cols)| img.params_b(cols)).unwrap_or_default();
    if cfg.vm_b.is_some() && d != Design::Serve2dPipe {
        return Err("a second VM set (`vm_b`) is serve2d_pipe's".into());
    }
    // serve8b_region names its VM image's params itself (serve8b_region_params).
    let vm = cfg.vm.filter(|_| d != Design::Serve8bRegion);
    let vm_keys: Vec<&'static str> = vm
        .map(|vm| vm.params())
        .unwrap_or_default()
        .iter()
        .chain(&vm_b)
        .map(|p| p.0)
        .collect();
    if let Some(vm) = vm {
        for (k, v) in vm.params() {
            put(&mut set, k, v);
        }
    }
    for &(k, v) in &vm_b {
        put(&mut set, k, v);
    }
    for &(k, v) in &cfg.pins {
        put(&mut set, k, v);
    }
    let files = d.files()?;
    let layout = &files.iter().find(|f| f.0 == "layout.csl").ok_or("no layout.csl")?.1;
    let decl = layout_params(layout);
    let mut out = Vec::new();
    for (name, has_default) in &decl {
        match set.iter().find(|(k, _)| k == name) {
            Some((_, v)) => out.push((name.clone(), v.to_string())),
            None if *has_default => {}
            None => return Err(format!("{}: the layout's `{name}` has no value", d.name())),
        }
    }
    refused(d, &out)?;
    let unknown: Vec<&str> = set.iter().map(|(k, _)| *k).filter(|k| !decl.iter().any(|(n, _)| n == k)).collect();
    // Plan keys the other 2-D layout takes (NP, PROF on serve2d) are dropped silently;
    // pins and config keys must exist.
    let config_keys: Vec<&str> = cfg.pins.iter().map(|p| p.0).chain(vm_keys.iter().copied()).collect();
    if let Some(bad) = unknown.iter().find(|k| config_keys.contains(k)) {
        return Err(format!("{}: no layout parameter `{bad}`", d.name()));
    }
    Ok((d, out))
}

/// The compile cache's key: the sources, the parameters, the fabric.
fn key(files: &[super::codegen::File], params: &[(String, String)], fabric: &str) -> String {
    let mut h = blake3::Hasher::new();
    for (name, text) in files {
        h.update(name.as_bytes()).update(&[0]).update(text.as_bytes()).update(&[0]);
    }
    for (k, v) in params {
        h.update(k.as_bytes()).update(b":").update(v.as_bytes()).update(b",");
    }
    h.update(fabric.as_bytes());
    h.finalize().to_hex().to_string()
}

/// The program's rectangle from its parameters (grids + head + its VM columns).
fn rectangle(design: Design, params: &[(String, String)]) -> R<(u32, u32)> {
    let get = |k: &str| -> R<u32> {
        params
            .iter()
            .find(|(n, _)| n == k)
            .and_then(|(_, v)| v.parse().ok())
            .ok_or_else(|| format!("{}: no `{k}`", design.name()))
    };
    if design == Design::Serve8bRegion {
        // serve8b_region's layout.csl: the band and regions (RC RPX), the grid
        // column (PX), the head and the strip; down the regions, the grid
        // column or the head's rows and its feeder row.
        let (px, py, hh, vmc) = (get("PX")?, get("PY")?, get("HH")?, get("VMC")?);
        let head = get("HEADMODE")? == 2;
        let (hw, ew) = if head { (py / 2 + vmc, py / 4) } else { (0, 0) };
        let lh = (py + get("RR")? * get("RPY")?).max(py * (1 + get("NCOL")?));
        let rh = if head { lh.max(hh + 1) } else { lh };
        // VMX: the VM block, 5 columns east of the strip.
        let vw = if head && get("VMX").unwrap_or(0) == 1 { 5 } else { 0 };
        return Ok((get("RC")? * get("RPX")? + px + hw + ew + vw, rh));
    }
    let (px, py, gw, gh, hh, vmc) = (get("PX")?, get("PY")?, get("GW")?, get("GH")?, get("HH")?, get("VMC")?);
    match design {
        // layout.csl: the grids then the head's PY + VMC columns; down GH
        // grids or the head's HH rows and its feeder row (HEADMODE 2).
        Design::Serve2d => Ok((gw * px + py + vmc, (gh * py).max(hh + 1))),
        // serve2d_pipe: VMT = VMC + VMB VM columns (guest4h2s's two sets).
        Design::Serve2dPipe => {
            let vmb = get("VMB").unwrap_or(0);
            Ok((gw * px + py + vmc + vmb, (gh * py).max(hh + 1)))
        }
        // serve8b's layout.csl: the grids, the head's PY/2 + VMC columns and
        // the PY/4-wide embedding strip (HEADMODE 2; neither in mode 0).
        Design::Serve8b => {
            let head = get("HEADMODE")? == 2;
            let (hw, ew) = if head { (py / 2 + vmc, py / 4) } else { (0, 0) };
            Ok((gw * px + hw + ew, if head { (gh * py).max(hh + 1) } else { gh * py }))
        }
        Design::Serve8bRegion => unreachable!("handled above"),
    }
}

/// `design` with `params`, from the compile cache or compiled into it
/// ([`crate::device::disk_cache`]). The output directory gets
/// `pie_manifest.json`. `system`: laid out on the whole wafer.
pub fn compile(design: Design, params: &[(String, String)], target: crate::sdk::Target, system: bool) -> R<PathBuf> {
    use crate::sdk::{Arch, Compile, Cslc};
    refused(design, params)?;
    let files = design.files()?;
    let (w, h) = rectangle(design, params)?;
    let (fabric_dims, fabric_offsets) = crate::sdk::cslc::memcpy_fabric(w, h);
    let fabric_dims = if system { crate::device::wafer_dims(target) } else { fabric_dims };
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
    let kvst = params.iter().find(|(k, _)| k == "KVST").and_then(|(_, v)| v.parse::<usize>().ok());
    let m = Manifest {
        design: design.name().to_string(),
        params: params.to_vec(),
        key: hex,
        fabric_dims,
        fabric_offsets,
        system,
        vm: design.vm_image().map(String::from),
        kvst: kvst.filter(|&s| s > 0),
        kvst_pack: kvst.filter(|&s| s > 0).map(|_| design.kv_staging().0.to_string()),
        kvst_aliased: kvst.filter(|&s| s > 0).map(|_| design.kv_staging().1),
    };
    m.write(&out)?;
    tracing::info!(s = started.elapsed().as_secs_f32(), "spatial build: compiled");
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
    use crate::spatial::plan::{ModelSpec, WaferDims};

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

    /// Fork 4's simulator build (scratch tiny/pvm): the plan's model dims, a
    /// hand-placed program and head (the plan places none at this size), and
    /// the multi-lane VM.
    pub fn pipe_config() -> (SpatialPlan, ServeConfig) {
        let goals = Goals { pipeline: true, cap: Some(8), blocks: Some(2), mlp_chunk: Some(1), ..Goals::lanes(2) };
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
            }),
            pins: vec![
                ("Tc", 1), ("E", 2), ("GW", 2), ("GH", 4), ("F", 8), ("HH", 16), ("Vr", 32),
                ("Gx", 4), ("Gy", 4), ("CAP", 8),
            ],
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

    /// The same parameters as the out.json of the build the simulator ran.
    #[test]
    fn tiny_pipe_params_match_the_simulated_build() {
        let (plan, cfg) = pipe_config();
        let (d, p) = params(&plan, &cfg).unwrap();
        assert_eq!(d, Design::Serve2dPipe);
        let want = "PX:4,PY:4,Hc:16,HR:64,Dc:8,Gs:4,Qh:2,Ic:16,Tc:1,W16:0,KV16:0,GRID:1,HEADMODE:2,B:2,PS:16,\
                    NB:2,E:2,MC:1,MMAX:2,HB:2,GW:2,GH:4,F:8,HH:16,Vr:32,Gx:4,Gy:4,Gb:1,vocab:250,PASSES:3,\
                    UNTIED:0,CERT:0,MODE:0,OVL:0,VMC:1,WF32:1,CAP:8,DBG:0,SPEC:0,SPEC_H:1,SPEC_T:1,SPEC_O:1,\
                    SMP:0,HLM:4,iters:1,VM:1,max_ops:48,nreg:6,pool_words:64,feat:99,ring_w:1,ring_d:32,vm_prof:0";
        let mut want: Vec<(String, String)> = want
            .split(',')
            .map(|kv| kv.split_once(':').map(|(k, v)| (k.trim().to_string(), v.to_string())).unwrap())
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
        assert_eq!((get("vm_sin"), get("opmask_lo"), get("opmask_hi")), (Some("4".into()), Some("65535".into()), Some("0".into())));
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
        let plan2 = SpatialPlan::derive(&plan.spec, crate::spatial::plan::WaferDims::CS3, cfg.goals).unwrap();
        let (d, p) = params(&plan2, &cfg).unwrap();
        assert_eq!(d, Design::Serve2d);
        assert!(refused(d, &p).is_ok());
        cfg.pins.push(("GRID", 0));
        assert!(params(&plan2, &cfg).unwrap_err().contains("GRID:0 is unsupported"));
        let explicit = vec![("GRID".to_string(), "0".to_string())];
        assert!(compile(Design::Serve2d, &explicit, crate::sdk::Target::Wse3, false).unwrap_err().contains("0150"));
        assert!(refused(Design::Serve2dPipe, &explicit).is_ok());
    }

    #[test]
    fn serve_config_reads_its_table() {
        let t: toml::Table = toml::from_str(
            "lanes = 2\npipeline = true\ncap = 8\nmmax = 2\nhb = 2\n[vm]\nmax_ops = 48\nnreg = 6\npool_words = 64\nfeat = 99\nopmask_hi = 0\n",
        )
        .unwrap();
        let c = ServeConfig::from_toml(&t, 16).unwrap();
        assert_eq!((c.goals.lanes, c.goals.page_size, c.goals.pipeline, c.goals.cap), (2, 16, true, Some(8)));
        assert_eq!((c.mmax, c.hb), (Some(2), Some(2)));
        let vm = c.vm.unwrap();
        assert_eq!((vm.max_ops, vm.ring_w, vm.ring_d, vm.opmask), (48, 1, 32, Some((0xffff_ffff, 0))));
        let bad: toml::Table = toml::from_str("lanse = 2\n").unwrap();
        assert!(ServeConfig::from_toml(&bad, 16).unwrap_err().contains("lanse"));
    }

    /// guest4h2s's second VM set: `[vm_b]` with its columns, as the layout's
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
        assert_eq!((get("VMB"), get("b_max_ops"), get("b_vm_lprog"), get("b_bank_words")), (Some("1".into()), Some("64".into()), Some("1".into()), Some("900".into())));
        assert_eq!(rectangle(Design::Serve2dPipe, &p).unwrap(), (one.0 + 1, one.1));
        let no_cols: toml::Table = toml::from_str("[vm_b]\nmax_ops = 64\nnreg = 8\npool_words = 128\nfeat = 7\n").unwrap();
        assert!(ServeConfig::from_toml(&no_cols, 16).unwrap_err().contains("cols"));
    }

    #[test]
    fn a_plan_without_a_program_is_refused() {
        let goals = Goals { pipeline: true, ..Goals::lanes(2) };
        let plan = SpatialPlan::derive(&spec(), WaferDims::CS3, goals).unwrap();
        assert!(params(&plan, &ServeConfig::new(goals)).unwrap_err().contains("no full program"));
    }
}


#[cfg(test)]
mod qwen3_8b {
    use super::*;
    use crate::spatial::plan::{ModelSpec, WaferDims};

    fn spec() -> ModelSpec {
        let catalog = poem_compiler::catalog::repository();
        let d = catalog.parse("qwen3-8b-bf16-kv-bf16").expect("qwen3-8b parses");
        ModelSpec::read(&d.trace(poem_ir::Platform::Cerebras)).expect("qwen3-8b reads")
    }

    fn sorted(extra: &str) -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = extra
            .split(',')
            .map(|kv| kv.split_once(':').map(|(k, v)| (k.to_string(), v.to_string())).unwrap())
            .collect();
        v.sort();
        v
    }

    /// 0115_serve8b_real's two builds come out of the plan and a config.
    #[test]
    fn the_plan_places_qwen3_8b_as_0115() {
        let base = "PX:128,PY:144,Hc:29,HR:4096,Dc:8,Gs:16,Qh:4,Ic:96,PS:16,E:1,MC:1,W16:1,NP:3,KV16:1,PROF:0,TIME:0,\
                    GW:5,GH:8,GS:4,HEADMODE:2,HH:1152,Vr:132,Gx:8,Gy:24,vocab:151936,PASSES:3,MODE:0,OVL:0,VMC:1,\
                    CAP:64,iters:1";
        for (b, nb, host, np) in [(1, 1, false, 3), (4, 1, true, 1)] {
            let goals = Goals { blocks: Some(nb), host_copy: host, grid_lanes: b, ..Goals::lanes(b) };
            let plan = SpatialPlan::derive(&spec(), WaferDims::CS3, goals).expect("a plan");
            let mut cfg = ServeConfig::new(goals);
            if np != 3 {
                cfg.pins.push(("NP", np));
            }
            let (d, p) = params(&plan, &cfg).unwrap();
            assert_eq!(d, Design::Serve8b);
            let mut got = p.clone();
            got.sort();
            let want = format!("{base},B:{b},NB:{nb},KVCP:{},KVST:{}", usize::from(!host), usize::from(b == 1))
                .replace("NP:3", &format!("NP:{np}"));
            assert_eq!(got, sorted(&want), "B {b}");
            // The model accepts the program (untied, 36 layers in strips of 5
            // and 4), and the engine reads it as serve8b's layout.
            let get = |k: &str| p.iter().find(|e| e.0 == k).and_then(|e| e.1.parse().ok());
            plan.spec.check_program(get).expect("the model takes serve8b");
            let lay = crate::spatial::serve2d::Layout2d::of(get).expect("a layout").expect("serve8b");
            assert_eq!((lay.layers(), lay.chunk_cols(), lay.root()), (36, 72, (640 + 36, 576)));
            assert_eq!(lay.paged.map(|s| (s.lanes, s.pages())), Some((b, 144)));
            assert_eq!(lay.host_copy, host);
        }
    }

    /// 0125_serve8b_region's two builds come out of the plan with `region`.
    #[test]
    fn the_plan_places_qwen3_8b_regions_as_0125() {
        let base = "PX:128,PY:144,Hc:29,Dc:8,Gs:16,Ic:96,RPX:256,RPY:240,RHc:18,RDc:4,RGs:32,RIc:48,RNL:3,RGY:16,\
                    RC:2,RR:4,NCOL:7,RTR:4,HR:4096,Qh:4,PS:16,NB:1,E:1,MC:1,W16:1,KV16:1,KVCP:0,PROF:0,TIME:0,\
                    HROUTE:1,ROT8:1,NOLAYER:0,HEADMODE:2,HH:1152,Vr:132,Gx:8,Gy:24,vocab:151936,PASSES:3,MODE:0,\
                    OVL:0,VMC:1,CAP:64,iters:1";
        // The grids of 0115's builds (serve8b's B 1 plans for page-copy
        // kernels; serve8b_region copies by the host either way).
        for (b, np, host) in [(1, 3, false), (4, 1, true)] {
            let goals = Goals { blocks: Some(1), host_copy: host, grid_lanes: b, ..Goals::lanes(b) };
            let plan = SpatialPlan::derive(&spec(), WaferDims::CS3, goals).expect("a plan");
            let mut cfg = ServeConfig::new(goals);
            cfg.region = true;
            if np != 3 {
                cfg.pins.push(("NP", np));
            }
            let (d, p) = params(&plan, &cfg).unwrap();
            assert_eq!(d, Design::Serve8bRegion);
            let mut got = p.clone();
            got.sort();
            assert_eq!(got, sorted(&format!("{base},B:{b},NP:{np},KVST:{}", usize::from(b == 1))), "B {b}");
            let get = |k: &str| p.iter().find(|e| e.0 == k).and_then(|e| e.1.parse().ok());
            plan.spec.check_program(get).expect("the model takes serve8b_region");
            let lay = crate::spatial::serve2d::Layout2d::of(get).expect("a layout").expect("serve8b_region");
            assert_eq!((lay.layers(), lay.root()), (36, (676, 576)));
            let p2: Vec<(String, String)> = p.clone();
            assert_eq!(rectangle(d, &p2).unwrap(), (749, 1153));
        }
    }

    /// With Fork 3's VM image the region plan is Fork 11's real-VM builds
    /// (scratch/fork11/fvr_params.txt: the VM block, Vr 160, HH 952, HLBN 1
    /// at B 4; VML2W 1, and bank_words 512 at B 4 from their later fit),
    /// staging at B 1.
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
        };
        for (b, np, host, extra) in [
            (1, 3, false, "bank_words:768,vm_lanes:1,B:1,NP:3,KVST:1"),
            (4, 1, true, "bank_words:512,vm_lanes:4,HLBN:1,B:4,NP:1,KVST:0"),
        ] {
            let goals = Goals { blocks: Some(1), host_copy: host, grid_lanes: b, ..Goals::lanes(b) };
            let plan = SpatialPlan::derive(&spec(), WaferDims::CS3, goals).expect("a plan");
            let mut cfg = ServeConfig::new(goals);
            cfg.region = true;
            cfg.vm = Some(vm);
            if np != 3 {
                cfg.pins.push(("NP", np));
            }
            let (d, p) = params(&plan, &cfg).unwrap();
            assert_eq!(d, Design::Serve8bRegion);
            let mut got = p.clone();
            got.sort();
            assert_eq!(got, sorted(&format!("{base},{extra}")), "B {b}");
            assert_eq!(rectangle(d, &p).unwrap(), (753, 1152));
            let get = |k: &str| p.iter().find(|e| e.0 == k).and_then(|e| e.1.parse().ok());
            let lay = crate::spatial::serve2d::Layout2d::of(get).expect("a layout").expect("serve8b_region");
            assert_eq!(lay.layers(), 36);
        }
    }
}
