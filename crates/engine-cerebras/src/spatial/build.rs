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
    pub pins: Vec<(&'static str, usize)>,
}

impl ServeConfig {
    pub fn new(goals: Goals) -> ServeConfig {
        ServeConfig { goals, mmax: None, hb: None, vm: None, hlm: None, spec: None, smp: false, pins: Vec::new() }
    }
}

/// The design a plan's family builds.
pub fn design(plan: &SpatialPlan) -> R<Design> {
    match plan.family {
        Family::Serve2d => Ok(Design::Serve2d),
        Family::Serve2dPipe => Ok(Design::Serve2dPipe),
        Family::Paged8b => Err(format!(
            "{}: the plan picks qwen2d_paged8b layer grids, and no full program places them yet ({})",
            plan.spec.name,
            plan.blockers.join("; ")
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
            (!ty.contains("color") && !name.is_empty()).then(|| (name.to_string(), ty.contains('=')))
        })
        .collect()
}

/// The compile parameters of `plan` built for `cfg`, in the layout's order,
/// checked against the layout: every parameter without a default is set,
/// and nothing unknown is passed.
pub fn params(plan: &SpatialPlan, cfg: &ServeConfig) -> R<(Design, Vec<(String, String)>)> {
    let d = design(plan)?;
    if plan.program.is_none() && !cfg.pins.iter().any(|p| p.0 == "GW") {
        return Err(format!("{}: no full program places the model: {}", plan.spec.name, plan.blockers.join("; ")));
    }
    let mut set: Vec<(&'static str, usize)> = plan.params();
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
    } else if cfg.spec.is_some() || cfg.smp || cfg.mmax.is_some() {
        return Err("speculative decode, the fused sampler and MMAX are serve2d_pipe's".into());
    }
    if let Some(vm) = cfg.vm {
        for (k, v) in [
            ("VM", 1),
            ("max_ops", vm.max_ops),
            ("nreg", vm.nreg),
            ("pool_words", vm.pool_words),
            ("feat", vm.feat),
            ("ring_w", vm.ring_w),
            ("ring_d", vm.ring_d),
            ("vm_prof", usize::from(vm.prof)),
        ] {
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
    let config_keys: Vec<&str> = cfg.pins.iter().map(|p| p.0).collect();
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
    let (px, py, gw, gh, hh, vmc) = (get("PX")?, get("PY")?, get("GW")?, get("GH")?, get("HH")?, get("VMC")?);
    // layout.csl: the grids then the head's PY + VMC columns; down GH grids
    // or the head's HH rows and its feeder row (HEADMODE 2).
    Ok((gw * px + py + vmc, (gh * py).max(hh + 1)))
}

/// `design` with `params`, from the compile cache or compiled into it
/// ([`crate::device::disk_cache`]). The output directory gets
/// `pie_manifest.json`. `system`: laid out on the whole wafer.
pub fn compile(design: Design, params: &[(String, String)], target: crate::sdk::Target, system: bool) -> R<PathBuf> {
    use crate::sdk::{Arch, Compile, Cslc};
    let files = design.files()?;
    let (w, h) = rectangle(design, params)?;
    let fabric_dims = if system { crate::device::wafer_dims(target) } else { (w + 7, h + 2) };
    let fabric_offsets = (4, 1);
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
            vm: Some(VmImage { max_ops: 48, nreg: 6, pool_words: 64, feat: 99, ring_w: 1, ring_d: 32, prof: false }),
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
    fn a_plan_without_a_program_is_refused() {
        let goals = Goals { pipeline: true, ..Goals::lanes(2) };
        let plan = SpatialPlan::derive(&spec(), WaferDims::CS3, goals).unwrap();
        assert!(params(&plan, &ServeConfig::new(goals)).unwrap_err().contains("no full program"));
    }
}
