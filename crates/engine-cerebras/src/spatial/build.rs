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
        ServeConfig { goals, mmax: None, hb: None, vm: None, hlm: None, spec: None, smp: false, region: false, pins: Vec::new() }
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
            "host_copy", "mmax", "hb", "hlm", "smp", "spec", "vm", "pins", "region",
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
        let vm = match t.get("vm") {
            None => None,
            Some(v) => {
                let m = v.as_table().ok_or("[cerebras.spatial.serve] `vm` is not a table")?;
                let need = |k: &str| int(m, k)?.ok_or_else(|| format!("[cerebras.spatial.serve.vm] needs `{k}`"));
                let lo = int(m, "opmask_lo")?;
                let hi = int(m, "opmask_hi")?;
                Some(VmImage {
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
            }
        };
        Ok(ServeConfig {
            goals,
            mmax: int(t, "mmax")?,
            hb: int(t, "hb")?,
            vm,
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
    if cfg.vm.is_some() || cfg.spec.is_some() || cfg.smp || cfg.goals.pipeline {
        return Err("serve8b_region has no eta VM, speculative decode, fused sampler or pipeline".into());
    }
    let g = &plan.grid;
    let wafer = super::plan::WaferDims::CS3;
    let (rc, rnl, rr, rpy, rgy) = (2, 3, 4, 240, 16);
    let (rpx, band) = (2 * g.px, 2 * 2 * g.px / g.px + 1);
    if g.dc % 4 != 0 || g.ic % 2 != 0 || plan.layers < band + rc * rr * rnl + 1 {
        return Err(format!("{}: serve8b_region's option A does not take this grid ({g:?})", plan.spec.name));
    }
    let ncol = plan.layers - band - rc * rr * rnl;
    let hh = g.py * (1 + ncol);
    let (c, vmc) = (g.py / 2, 1);
    let width = rc * rpx + g.px + c + vmc + g.py / 4;
    if width > wafer.width || hh + 1 > wafer.height || g.py + rr * rpy > wafer.height {
        return Err(format!("{}: serve8b_region's {width} x {} does not fit the wafer", plan.spec.name, hh + 1));
    }
    let f = hh / 2;
    let gx = [16, 8, 4, 2].into_iter().find(|x| c % x == 0 && c / x >= 2).unwrap_or(1).min(8);
    let gy = (2..=24).rev().step_by(2).find(|x| f % x == 0).unwrap_or(2);
    let cap = cfg.goals.cap.unwrap_or_else(|| {
        let mut c = 64;
        while g.lanes * c > 512 && c > 1 {
            c /= 2;
        }
        c
    });
    Ok(vec![
        ("PX", g.px), ("PY", g.py), ("Hc", g.hc), ("Dc", g.dc), ("Gs", g.gs), ("Ic", g.ic), ("RPX", rpx),
        ("RPY", rpy), ("RHc", g.hidden.div_ceil(rpy)), ("RDc", g.dc / 2), ("RGs", 2 * g.gs), ("RIc", g.ic / 2),
        ("RNL", rnl), ("RGY", rgy), ("RC", rc), ("RR", rr), ("NCOL", ncol), ("RTR", 4), ("HR", g.hidden),
        ("Qh", g.qh), ("PS", g.page_size), ("NB", g.blocks), ("E", g.entries), ("MC", g.mlp_chunk),
        ("W16", usize::from(g.w16)), ("KV16", usize::from(g.kv16)), ("KVCP", 0), ("PROF", 0), ("TIME", 0),
        ("HROUTE", 1), ("ROT8", 1), ("NOLAYER", 0), ("HEADMODE", 2), ("HH", hh), ("Vr", plan.vocab.div_ceil(hh)),
        ("Gx", gx), ("Gy", gy), ("vocab", plan.vocab), ("PASSES", 3), ("MODE", 0), ("OVL", 0), ("VMC", vmc),
        ("CAP", cap), ("iters", 1), ("B", g.lanes), ("NP", 3),
    ])
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
    let vm_keys: Vec<&'static str> = cfg.vm.map(|vm| vm.params()).unwrap_or_default().iter().map(|p| p.0).collect();
    if let Some(vm) = cfg.vm {
        for (k, v) in vm.params() {
            put(&mut set, k, v);
        }
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
        return Ok((get("RC")? * get("RPX")? + px + hw + ew, rh));
    }
    let (px, py, gw, gh, hh, vmc) = (get("PX")?, get("PY")?, get("GW")?, get("GH")?, get("HH")?, get("VMC")?);
    match design {
        // layout.csl: the grids then the head's PY + VMC columns; down GH
        // grids or the head's HH rows and its feeder row (HEADMODE 2).
        Design::Serve2d | Design::Serve2dPipe => Ok((gw * px + py + vmc, (gh * py).max(hh + 1))),
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
    let m = Manifest {
        design: design.name().to_string(),
        params: params.to_vec(),
        key: hex,
        fabric_dims,
        fabric_offsets,
        system,
        vm: design.vm_image().map(String::from),
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
            let want = format!("{base},B:{b},NB:{nb},KVCP:{}", usize::from(!host)).replace("NP:3", &format!("NP:{np}"));
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
            assert_eq!(got, sorted(&format!("{base},B:{b},NP:{np}")), "B {b}");
            let get = |k: &str| p.iter().find(|e| e.0 == k).and_then(|e| e.1.parse().ok());
            plan.spec.check_program(get).expect("the model takes serve8b_region");
            let lay = crate::spatial::serve2d::Layout2d::of(get).expect("a layout").expect("serve8b_region");
            assert_eq!((lay.layers(), lay.root()), (36, (676, 576)));
            let p2: Vec<(String, String)> = p.clone();
            assert_eq!(rectangle(d, &p2).unwrap(), (749, 1153));
        }
    }
}
