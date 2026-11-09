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
    /// serve8b_region(_pipe): the units' op cuts (`SPD` 1, codegen's
    /// `Variant::speed`; cerebras-8b-speed).
    pub speed: bool,
    /// serve8b_region(_pipe), with `speed`: the softmax against a carried
    /// log-sum-exp, no max all-reduce (`LSE` 1, codegen's `Variant::lse`).
    pub lse: bool,
    /// serve8b_region(_pipe), with `lse`: the second op cuts (`OP2` 1,
    /// codegen's `Variant::ops2`).
    pub ops2: bool,
    /// serve8b_region(_pipe), with `ops2` at B 1: q, k and v as one matvec
    /// (`QKV1` 1, codegen's `Variant::qkv1`).
    pub qkv1: bool,
    /// serve8b_region(_pipe), with `ops2`: the regions' Y all-reduces with FIFO
    /// dual-input merges (`ARF` 1, codegen's `Variant::arf`).
    pub arf: bool,
    /// With `arf`: also the regions' X full all-reduces (`ARF` 3).
    pub arfx: bool,
    /// serve8b_region(_pipe), with `ops2`: an IEEE f16 build (`F16` 1);
    /// `f16acc` also the regions' f16-accumulating matvecs (`F16` 2).
    pub f16: bool,
    pub f16acc: bool,
    /// With `f16acc`: the grids' weight matvecs accumulate in f16 too (`F16` 3).
    pub f16accg: bool,
    /// serve8b_region(_pipe), with `arf`: the region kernel's bit-exact code
    /// cuts (`MEM` 1, codegen's `Variant::mem`).
    pub mem: bool,
    /// serve8b_region(_pipe), with `ops2`: the regions' RoPE tables once a
    /// pass (`OP3` 1, codegen's `Variant::op3`).
    pub op3: bool,
    /// serve8b_region(_pipe), with `arf`, `op3` or `mem`: the grids take those
    /// levers too (`GRD`, codegen's `Variant::grid`).
    pub grid_levers: bool,
    /// With `f16acc`: the regions' f16 probability copy clamped with 0515's redo
    /// window (`F16CL` 1, codegen's `Variant::f16cl`; Llama: 0526).
    pub f16_clamp: bool,
    /// With `arfx`: the regions' XA all-reduces on their own colours (`XAC` 1,
    /// codegen's `Variant::xac`).
    pub xac: bool,
    /// With `qkv1`, `ops2` and `f16acc` (B 1): the units' op cuts (`TRM`,
    /// codegen's `Variant::trim`: 1 the exact cuts, 2 x's sums of squares by
    /// a scalar loop, 8 exp at degree 4; 4 the grids too with `f16accg`).
    pub trim: bool,
    /// With `arfx` and `xac`: the segment all-reduce as streamed halves too
    /// (`ARF` 7, codegen's arfs_ops; the host's X set as segment lines).
    pub arfs: bool,
    /// With `arf`: the ARF Y streamed halves in groups of 16 rows (`ARY` 16;
    /// the host's route words).
    pub ary16: bool,
    /// serve8b_region(_pipe): vertical handoffs on centred columns (`HVC` 1;
    /// the host's tables).
    pub hvc: bool,
    /// With `trim` (any model, Qwen3-8B included): the regions' MLP act in f16 on
    /// every PE of a column from the gate-up all-reduce, no act all-reduce
    /// (`TRM` bit 1024, with 4096; not 8192: the grids lose with it, 0538; inexact,
    /// but golden on Llama and, in host emulation, on Qwen3-8B).
    pub silu16: bool,
    /// With `trim` (any model): the regions' x g in f16, its
    /// f16 gammas made in place on the first pass (`TRM` bits 512 + 4096), and
    /// the grids' (8192, when they take the cuts); inexact.
    pub xg16: bool,
    /// With `trim`: the head's column argmax in one streamed round of packed
    /// keys, the exact pair chains only on a tie (`TRM` bit 16384; exact).
    pub head_pack: bool,
    /// With `silu16`: its two scales in f16 after the conversion (`TRM` 2048;
    /// inexact; 0540 sp540_t27 2,033.3 tok/s, golden, 0 mismatches).
    pub silu16h: bool,
    /// With `silu16`: its wide tail (`SLW` 1: no f16 subnormal flush of a sig(a) at
    /// a < -10; exact to ~1 ulp there; +18 cycles a layer). Qwen3-8B needs it (1024 alone
    /// flips 3 of 512 teacher-forced tokens in host emulation, tools/emu/tf_qwen3_8b.py);
    /// Llama's t29 ran without.
    pub silu16w: bool,
    /// The head's argmax round from bf16(x) alone (`MODE` 2) instead of the
    /// exact three-part x (`MODE` 0): a third of the head's matvec.
    pub head_one_pass: bool,
    /// serve8b_region, with `lse`: the attention vectorized over a kv head's
    /// query heads (HV): one-position kernel pages (`PS` 1, `NB` the plan's
    /// blocks x PIE's page size, `E` 5). PIE keeps its page size; the
    /// engine expands each PIE page into its kernel pages
    /// ([`super::wafer::Wafer::page_as`]).
    pub hv: bool,
    /// serve8b_region: lever D (`YD`), every layer but the first and last
    /// in regions of `RPY` 208 (no band; the last grid at (XC, YD)).
    pub lever_d: bool,
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
            speed: false,
            lse: false,
            ops2: false,
            qkv1: false,
            arf: false,
            arfx: false,
            f16: false,
            f16acc: false,
            f16accg: false,
            mem: false,
            op3: false,
            grid_levers: false,
            f16_clamp: false,
            xac: false,
            trim: false,
            arfs: false,
            ary16: false,
            hvc: false,
            silu16: false,
            xg16: false,
            head_pack: false,
            silu16h: false,
            silu16w: false,
            head_one_pass: false,
            hv: false,
            lever_d: false,
            pins: Pins::default(),
        }
    }
}

impl ServeConfig {
    /// `[cerebras.spatial.serve]`: the build PIE compiles when no `program`
    /// directory is configured. Keys (all optional):
    /// - the plan's goals: `lanes`, `grid_lanes`, `page_size` (default
    ///   `page_size`, PIE's), `blocks`, `max_blocks`, `mlp_chunk`, `cap`,
    ///   `pipeline`, `host_copy`, `rows` (a paged8b grid's PY);
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
            "rows",
            "speed",
            "lse",
            "ops2",
            "qkv1",
            "arf",
            "arfx",
            "f16",
            "f16acc",
            "f16accg",
            "mem",
            "op3",
            "grid_levers",
            "f16_clamp",
            "xac",
            "trim",
            "arfs",
            "ary16",
            "hvc",
            "silu16",
            "xg16",
            "head_pack",
            "silu16h",
            "silu16w",
            "head_one_pass",
            "hv",
            "lever_d",
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
            rows: int(t, "rows")?,
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
            speed: flag(t, "speed")?,
            lse: flag(t, "lse")?,
            ops2: flag(t, "ops2")?,
            qkv1: flag(t, "qkv1")?,
            arf: flag(t, "arf")?,
            arfx: flag(t, "arfx")?,
            f16: flag(t, "f16")?,
            f16acc: flag(t, "f16acc")?,
            f16accg: flag(t, "f16accg")?,
            mem: flag(t, "mem")?,
            op3: flag(t, "op3")?,
            grid_levers: flag(t, "grid_levers")?,
            f16_clamp: flag(t, "f16_clamp")?,
            xac: flag(t, "xac")?,
            trim: flag(t, "trim")?,
            arfs: flag(t, "arfs")?,
            ary16: flag(t, "ary16")?,
            hvc: flag(t, "hvc")?,
            silu16: flag(t, "silu16")?,
            xg16: flag(t, "xg16")?,
            head_pack: flag(t, "head_pack")?,
            silu16h: flag(t, "silu16h")?,
            silu16w: flag(t, "silu16w")?,
            head_one_pass: flag(t, "head_one_pass")?,
            hv: flag(t, "hv")?,
            lever_d: flag(t, "lever_d")?,
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
    if spec.moe.is_some() {
        let program = gptoss_program(spec, cfg)?;
        tracing::info!(design = program.design.name(), model = %spec.name, "spatial build: the plan's program");
        return compile(&program, target, system);
    }
    if spec.hybrid.is_some() {
        let program = q35_program(spec, cfg)?;
        tracing::info!(design = program.design.name(), model = %spec.name, "spatial build: the plan's program");
        return compile(&program, target, system);
    }
    let plan = SpatialPlan::derive(spec, super::plan::WaferDims::CS3, cfg.goals)
        .map_err(|e| e.to_string())?;
    let program = program(&plan, cfg)?;
    tracing::info!(design = program.design.name(), model = %spec.name, "spatial build: the plan's program");
    compile(&program, target, system)
}

/// The serve_g4 program PIE builds for Gemma 4 `spec` under `cfg`: its
/// plan's parameters, the compile (cached).
pub fn build_g4(
    spec: &super::model_g4::G4Spec,
    cfg: &ServeConfig,
    target: crate::sdk::Target,
    system: bool,
) -> R<PathBuf> {
    let program = g4_program(spec, cfg)?;
    tracing::info!(design = program.design.name(), model = %spec.name, "spatial build: the plan's program");
    compile(&program, target, system)
}

/// Gemma 4's serve_g4 program: its plan's parameters, as given (the
/// config's pins and VM images are serve2d's and serve8b's).
pub fn g4_program(spec: &super::model_g4::G4Spec, cfg: &ServeConfig) -> R<ProgramSpec> {
    let plan = super::plan_g4::G4Plan::derive(spec, super::plan::WaferDims::CS3, cfg.goals)
        .map_err(|e| e.to_string())?;
    let Some(params) = plan.params() else {
        return Err(format!(
            "{}: no serve_g4 program places the model: {}",
            spec.name,
            plan.blockers.join("; ")
        ));
    };
    if cfg.region || cfg.vm.is_some() || cfg.pins.gw.is_some() {
        return Err(format!(
            "{}: serve_g4 takes no `region`, VM image or pins",
            spec.name
        ));
    }
    let params: super::program::Params = params
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    ProgramSpec::from_params(Design::ServeG4, &params)
}

/// A hybrid model's serve_q35 program: its plan's parameters, as given
/// (the config's pins and VM images are serve2d's and serve8b's).
pub fn q35_program(spec: &super::model::ModelSpec, cfg: &ServeConfig) -> R<ProgramSpec> {
    let plan = super::plan_q35::Q35Plan::derive(spec, super::plan::WaferDims::CS3, cfg.goals)
        .map_err(|e| e.to_string())?;
    if plan.program.is_none() {
        return Err(format!(
            "{}: no full program places the model: {}",
            spec.name,
            plan.blockers.join("; ")
        ));
    }
    if cfg.region || cfg.vm.is_some() || cfg.pins.gw.is_some() {
        return Err(format!(
            "{}: serve_q35 takes no `region`, VM image or pins",
            spec.name
        ));
    }
    let params: super::program::Params = plan
        .params()
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    ProgramSpec::from_params(Design::ServeQ35, &params)
}

/// A MoE model's serve_gptoss program: the validated layout's params
/// (the f32 pool, `KV16` 0), its lanes and page size the config's.
pub fn gptoss_program(spec: &super::model::ModelSpec, cfg: &ServeConfig) -> R<ProgramSpec> {
    if cfg.region || cfg.vm.is_some() || cfg.pins.gw.is_some() || cfg.mmax.is_some() {
        return Err(format!(
            "{}: serve_gptoss takes no `region`, ring, VM image or pins",
            spec.name
        ));
    }
    let params =
        super::plan_gptoss::program_params(spec, cfg.goals.lanes.max(1), cfg.goals.page_size)
            .map_err(|e| e.to_string())?;
    ProgramSpec::from_params(Design::ServeGptoss, &params)
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
        Model::dims(g.hidden, g.qh, plan.vocab, cap),
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

/// The regions' Y phase-1 group (it must divide `RPY`).
const RGY: usize = 16;

/// serve8b_region's units: the grid (one layer a unit, the band and the
/// column), `RC` x `RR` regions of `RPX` x `RPY` holding `RNL` layers each,
/// and the head's rows (`hh`, `vr`, `vmc`) that go with them.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Placement {
    grid: super::plan::Grid,
    rpx: usize,
    rpy: usize,
    rnl: usize,
    rc: usize,
    rr: usize,
    band: usize,
    ncol: usize,
    hh: usize,
    vr: usize,
    vmc: usize,
    /// The search's cost estimate (lower is faster; see `unit_cost`).
    cost: usize,
}

/// A unit layer's estimated cycles: a fixed part (norms, attention,
/// collectives), the matvecs by the PE's tile elements (bf16 about 0.6
/// cycles an element on the simulator) and the collectives' hops. Only for
/// ranking placements.
fn unit_cost(g: &super::plan::Grid) -> usize {
    let elems = g.hc * ((2 * g.qh + 2) * g.dc + 3 * g.ic);
    8000 + elems * 6 / 10 + 20 * (g.px + g.py)
}

/// A unit PE's image must end this far below the top of its memory (the
/// stack is not counted by the linker).
const STACK_MARGIN: usize = 768;

/// The placement at `grid` with these regions when it fits the wafer and
/// its PEs: the shape rules of the layout (and `program::serve8b_region`),
/// the head beside the column, the feeder row between the grids' rows, the
/// byte model of `plan::Grid` for grid and region PEs (`check_bytes`).
#[allow(clippy::too_many_arguments)]
fn place_at(
    plan: &SpatialPlan,
    grid: super::plan::Grid,
    (rpx, rpy, rnl, rc, rr): (usize, usize, usize, usize, usize),
    vm: bool,
    ring: bool,
    check_bytes: bool,
    grow_head: bool,
) -> Option<Placement> {
    let wafer = super::plan::WaferDims::CS3;
    let g = grid;
    if rpx == 0 || rpy == 0 || !(rc * rpx).is_multiple_of(g.px) || !rpx.is_multiple_of(g.px) {
        return None;
    }
    let wide = rpx / g.px;
    let (rgs, ric) = (wide * g.gs, g.ic / wide);
    let rdc = g.dc / wide;
    if !g.py.is_multiple_of(4)
        || !rpy.is_multiple_of(RGY)
        || rdc * wide != g.dc
        || !rdc.is_multiple_of(2)
        || ric * wide != g.ic
        || g.hidden.div_ceil(rpy) * rpy < g.hidden
    {
        return None;
    }
    let band = rc * rpx / g.px + 1;
    let ncol = plan.layers.checked_sub(band + rc * rr * rnl)?;
    let c = g.py / 2;
    let (hh, vr, vmc, vw) = if vm {
        let vr = 160;
        let mut hh = plan.vocab.div_ceil(vr);
        hh += hh % 2;
        while !(4..=28).step_by(2).any(|x| (hh / 2) % x == 0) {
            hh += 2;
        }
        (hh, vr, 0, 5)
    } else {
        // The grid column's height; with `grow_head` (option A) the first
        // taller one (to 2 YL at most: the feeder half above the last
        // layer) whose strip tiles (Vr rows of 4 Hc dims) pass STRIP_TABLE,
        // with a feeder half F that a row group of 16-24 divides, when the
        // column's own does not and a taller one does (Qwen3-8B's 1152 rows
        // hold as is; Llama's 3 column grids are short for its vocabulary).
        // Vr even (head2d_v10's row pairs).
        let vr_of = |hh: usize| plan.vocab.div_ceil(hh).next_multiple_of(2);
        let fits = |hh: usize| {
            vr_of(hh) * 4 * g.hc * 2 <= STRIP_TABLE
                && (16..=24).step_by(2).any(|x| (hh / 2).is_multiple_of(x))
        };
        let mut hh = g.py * (1 + ncol);
        if grow_head
            && !fits(hh)
            && let Some(taller) = (hh..(2 * g.py * ncol).min(wafer.height))
                .step_by(2)
                .find(|&hh| fits(hh))
        {
            hh = taller;
        }
        (hh, vr_of(hh), 1, 0)
    };
    let width = rc * rpx + g.px + c + vmc + g.py / 4 + vw;
    if c * 2 * g.hc < g.hidden {
        return None;
    }
    let f = hh / 2;
    if width > wafer.width
        || hh + 1 > wafer.height
        || g.py + rr * rpy > wafer.height
        || g.py * (1 + ncol) > wafer.height
        || f < g.py
        || f >= g.py * ncol
    {
        return None;
    }
    let region = super::plan::Grid {
        px: rpx,
        py: rpy,
        hc: g.hidden.div_ceil(rpy),
        dc: rdc,
        gs: rgs,
        ic: ric,
        ..g
    };
    // The deepest pool (up to the plan's) both unit kinds hold; B 1 without
    // a ring stages KV pages (`KVST` 1, as `serve8b_region` builds it).
    let most = wafer.pe_bytes - STACK_MARGIN;
    let staging = |nl: usize| {
        if g.lanes == 1 && !ring {
            super::plan::Grid::staging_bytes(nl)
        } else {
            0
        }
    };
    let fits = |nb: usize| {
        let (g, r) = (
            super::plan::Grid { blocks: nb, ..g },
            super::plan::Grid {
                blocks: nb,
                ..region
            },
        );
        g.bytes(ring) + staging(1) <= most
            && (rc * rr == 0 || r.region_bytes(rnl, ring) + staging(rnl) <= most)
    };
    // The head's PEs (no VM block): their rows at CAP 1 (serve8b_region then
    // picks the CAP the final root holds, `plan::final_root_bytes`) and the
    // embedding strip's.
    let head_fits =
        super::plan::final_root_bytes(vr, g.hc, g.lanes, g.lanes.min(4), ring.then_some(12), 1)
            <= most
            && super::plan::embed_bytes(vr, g.hc, g.lanes) <= most;
    if check_bytes && !vm && !head_fits {
        return None;
    }
    let (g, region) = if check_bytes {
        let nb = (1..=g.blocks).rev().find(|&nb| fits(nb))?;
        (
            super::plan::Grid { blocks: nb, ..g },
            super::plan::Grid {
                blocks: nb,
                ..region
            },
        )
    } else {
        (g, region)
    };
    let cost = (band + ncol) * unit_cost(&g) + rc * rr * rnl * unit_cost(&region);
    Some(Placement {
        grid: g,
        rpx,
        rpy,
        rnl,
        rc,
        rr,
        band,
        ncol,
        hh,
        vr,
        vmc,
        cost,
    })
}

/// serve8b_region's placement for `plan`: option A (`RC` 2, `RNL` 3, `RR`
/// 4, `RPY` 240, regions twice the grid's width; the shape built and run
/// for Qwen3-8B) when it takes the plan's grid; else the cheapest
/// placement (`unit_cost`) that fits, over grids of the plan's columns and
/// at least its rows (16-row steps), regions one or two grids wide of
/// 16-row steps, 1 to 4 layers a region, up to 4 x 8 regions.
fn placement(plan: &SpatialPlan, vm: bool, ring: bool) -> R<Placement> {
    let g = plan.grid;
    if g.dc.is_multiple_of(4)
        && g.ic.is_multiple_of(2)
        && let Some(p) = place_at(plan, g, (2 * g.px, 240, 3, 2, 4), vm, ring, false, true)
    {
        return Ok(p);
    }
    let wafer = super::plan::WaferDims::CS3;
    let mut best: Option<Placement> = None;
    for py in (g.py..=wafer.height / 2).step_by(16) {
        let grid = super::plan::Grid {
            py,
            hc: g.hidden.div_ceil(py),
            ..g
        };
        for wide in [1, 2] {
            for rpy in (64..=wafer.height - py).step_by(16) {
                for rnl in 1..=4 {
                    for rc in 1..=4 {
                        for rr in 1..=8 {
                            let at = (wide * g.px, rpy, rnl, rc, rr);
                            if let Some(p) = place_at(plan, grid, at, vm, ring, true, false)
                                && best.is_none_or(|b| p.cost < b.cost)
                            {
                                best = Some(p);
                            }
                        }
                    }
                }
            }
        }
    }
    best.ok_or_else(|| {
        format!(
            "{}: no serve8b_region placement of {} layers fits the wafer (grids of {} columns, regions 1-2 grids wide, 1-4 layers a region)",
            plan.spec.name, plan.layers, g.px
        )
    })
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
    if cfg.ops2 && !cfg.lse {
        return Err("serve8b_region's second op cuts (`ops2`) need `lse`".into());
    }
    if cfg.qkv1 && !cfg.ops2 {
        return Err("serve8b_region's one QKV matvec (`qkv1`) needs `ops2`".into());
    }
    if (cfg.arf || cfg.f16 || cfg.f16acc) && !cfg.ops2 {
        return Err("serve8b_region's `arf` / `f16` / `f16acc` need `ops2`".into());
    }
    if cfg.f16accg && !cfg.f16acc {
        return Err("serve8b_region's grid f16 accumulation (`f16accg`) needs `f16acc`".into());
    }
    if cfg.arfx && !cfg.arf {
        return Err("serve8b_region's X all-reduces (`arfx`) need `arf`".into());
    }
    if cfg.op3 && !cfg.ops2 {
        return Err("serve8b_region's `op3` needs `ops2`".into());
    }
    if cfg.mem && !cfg.arf {
        return Err("serve8b_region's code cuts (`mem`) need `arf`".into());
    }
    if cfg.grid_levers && !(cfg.arf || cfg.op3) {
        return Err("serve8b_region's `grid_levers` needs `arf` or `op3`".into());
    }
    if cfg.f16_clamp && !cfg.f16acc {
        return Err("serve8b_region's `f16_clamp` needs `f16acc`".into());
    }
    if cfg.xac && !cfg.arfx {
        return Err("serve8b_region's `xac` needs `arfx`".into());
    }
    if cfg.trim && !(cfg.qkv1 && cfg.ops2 && cfg.f16acc && cfg.arfx) {
        return Err("serve8b_region's `trim` needs `qkv1`, `ops2`, `f16acc` and `arfx`".into());
    }
    if cfg.silu16w && !cfg.silu16 {
        return Err("serve8b_region's `silu16w` needs `silu16`".into());
    }
    if cfg.silu16h && !cfg.silu16 {
        return Err("serve8b_region's `silu16h` needs `silu16`".into());
    }
    if (cfg.silu16 || cfg.xg16 || cfg.head_pack) && !cfg.trim {
        return Err("serve8b_region's `silu16`, `xg16` and `head_pack` need `trim`".into());
    }
    if cfg.arfs && !(cfg.arfx && cfg.xac) {
        return Err("serve8b_region's `arfs` needs `arfx` and `xac`".into());
    }
    if cfg.ary16 && !cfg.arf {
        return Err("serve8b_region's `ary16` needs `arf`".into());
    }
    if cfg.hv && !cfg.lse {
        return Err("serve8b_region's HV (`hv`) needs `lse` (and `speed`)".into());
    }
    // Lever D (`lever_d`): its fixed all-region layout; else the placement
    // (option A, or the search when option A does not fit).
    let (gv, rpx, rpy, rnl, rc, rr, band, ncol, hh, vr, vmc, rgy, yd) = if cfg.lever_d {
        let g = &plan.grid;
        let wafer = super::plan::WaferDims::CS3;
        // Lever D: layer 0's grid, RC x RR regions of RNL layers from row 0 at
        // RPY 208 (RGY 16 divides it; RR 5 is the most the wafer's rows hold),
        // the remaining NCOL layers' grids down the column.
        let (rc, rnl, rpy, rgy) = (2, 3, if cfg.lever_d { 208 } else { 240 }, 16);
        let rpx = 2 * g.px;
        let (rr, band) = if cfg.lever_d {
            let rr = 5;
            if plan.layers < 2 + rc * rr * rnl || rr * rpy > wafer.height {
                return Err(format!(
                    "{}: serve8b_region's lever D takes {} layers or more, not {}",
                    plan.spec.name,
                    2 + rc * rr * rnl,
                    plan.layers
                ));
            }
            (rr, 1)
        } else {
            (4, 2 * 2 * g.px / g.px + 1)
        };
        if !g.dc.is_multiple_of(4)
            || !g.ic.is_multiple_of(2)
            || plan.layers < band + rc * rr * rnl + 1
        {
            return Err(format!(
                "{}: serve8b_region's option A does not take this grid ({g:?})",
                plan.spec.name
            ));
        }
        let ncol = plan.layers - band - rc * rr * rnl;
        // The last layer's grid row (the layout's YL): the head's feeder half F
        // lies in [PY, YL).
        // One column grid: beside the snake's last region (row RR - 2, as
        // 0472 ran); more: beside the U's (row 1), so they fit below it.
        let yd = match (cfg.lever_d, ncol) {
            (false, _) => 0,
            (true, 1) => (rr - 2) * rpy,
            (true, _) => rpy,
        };
        let yl = if cfg.lever_d {
            yd + (ncol - 1) * g.py
        } else {
            g.py * ncol
        };
        // The head's height as option A's (its hardware-validated heads: Llama
        // 864, Qwen3-8B 1152): its column grids' count.
        let ncol_a = if cfg.lever_d {
            plan.layers.saturating_sub(2 * 2 + 1 + rc * 4 * rnl)
        } else {
            ncol
        };
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
                // The grid column's height; taller (to 2 YL at most: the feeder
                // half above the last layer) while the strip's tiles (Vr rows
                // of 4 Hc dims) pass STRIP_TABLE, with a feeder half F that a
                // row group of 16-24 divides (Qwen3-8B's 1152 rows hold as is;
                // Llama's 3 column grids are short for its vocabulary).
                // Vr even (head2d_v10's row pairs).
                let vr_of = |hh: usize| plan.vocab.div_ceil(hh).next_multiple_of(2);
                let fits = |hh: usize| {
                    vr_of(hh) * 4 * g.hc * 2 <= STRIP_TABLE
                        && (16..=24).step_by(2).any(|x| (hh / 2).is_multiple_of(x))
                };
                let mut hh = g.py * (1 + ncol_a);
                if !fits(hh) {
                    hh = (hh..(2 * g.py * ncol_a).min(wafer.height))
                        .step_by(2)
                        .find(|&hh| fits(hh))
                        .ok_or_else(|| {
                            format!(
                                "{}: serve8b_region's strip holds no {}-row vocabulary beside {ncol} column grids of PY {}",
                                plan.spec.name, plan.vocab, g.py
                            )
                        })?;
                }
                (hh, vr_of(hh), 1, 0)
            }
        };
        let width = rc * rpx + g.px + c + vmc + g.py / 4 + vw;
        let lh = if cfg.lever_d {
            (rr * rpy).max(yd + ncol * g.py)
        } else {
            g.py + rr * rpy
        };
        if width > wafer.width || hh + 1 > wafer.height || lh > wafer.height {
            return Err(format!(
                "{}: serve8b_region's {width} x {} does not fit the wafer",
                plan.spec.name,
                hh + 1
            ));
        }
        let f = hh / 2;
        if f < g.py || f >= yl {
            return Err(format!(
                "{}: serve8b_region's feeder row {f} is not between the grids' {} and {yl}",
                plan.spec.name, g.py
            ));
        }
        (*g, rpx, rpy, rnl, rc, rr, band, ncol, hh, vr, vmc, rgy, yd)
    } else {
        let pl = placement(plan, cfg.vm.is_some(), pipe.is_some())?;
        let Placement {
            rpx,
            rpy,
            rnl,
            rc,
            rr,
            band,
            ncol,
            hh,
            vr,
            vmc,
            ..
        } = pl;
        let rgy = RGY;
        (
            pl.grid, rpx, rpy, rnl, rc, rr, band, ncol, hh, vr, vmc, rgy, 0,
        )
    };
    let g = &gv;
    let c = g.py / 2;
    // The grids' Y phase-1 group: Gs, or the largest even divisor of PY
    // below it (152 rows: 8).
    let grid_gy = (2..=g.gs)
        .rev()
        .step_by(2)
        .find(|x| g.py.is_multiple_of(*x))
        .unwrap_or(2);
    let f = hh / 2;
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
    // The final root's head PE holds `in_tokens`/`out_tokens` of MMAX x B x
    // CAP (`plan::final_root_bytes`): the largest CAP (halving) that fits it.
    let cap = if cfg.vm.is_some() {
        cap
    } else {
        let hlb = g.lanes.min(4);
        let mmax = pipe.map(|(m, _)| m);
        let most = super::plan::WaferDims::CS3.pe_bytes - STACK_MARGIN;
        let fits =
            |c: usize| super::plan::final_root_bytes(vr, g.hc, g.lanes, hlb, mmax, c) <= most;
        let mut c = cap;
        while c > 1 && !fits(c) {
            c /= 2;
        }
        if !fits(c) || (cfg.goals.cap.is_some() && c < cap) {
            return Err(format!(
                "serve8b_region: the final root's head PE (Vr {vr}, {} dims, B {}, MMAX {}) holds CAP {c} at most, not {cap}",
                2 * g.hc,
                g.lanes,
                mmax.unwrap_or(1)
            ));
        }
        c
    };
    let units = Units {
        rpx,
        rpy,
        rhc: g.hidden.div_ceil(rpy),
        rdc: g.dc * g.px / rpx,
        rgs: g.gs * rpx / g.px,
        ric: g.ic * g.px / rpx,
        rnl,
        rc,
        rr,
        ncol,
        // Lever D: the grid's bottom row carries three re-chunk legs and E.
        rtr: if cfg.lever_d { 5 } else { 4 },
        gy: grid_gy,
        rgy,
        hroute: true,
        rot8: true,
        nolayer: false,
        band,
        yd,
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
            // Llama: no q/k norms, its own epsilon (the units' sources
            // follow, codegen's `Variant`).
            qk_norm: plan.spec.qk_norm,
            eps_bits: if plan.spec.rms_eps == super::model::KERNEL_EPS {
                0
            } else {
                plan.spec.rms_eps.to_bits()
            },
            speed: cfg.speed,
            lse: cfg.lse,
            ops2: cfg.ops2,
            qkv1: cfg.qkv1,
            arf: if cfg.arfs {
                7
            } else if cfg.arfx {
                3
            } else {
                usize::from(cfg.arf)
            },
            f16: if cfg.f16accg {
                3
            } else if cfg.f16acc {
                2
            } else {
                usize::from(cfg.f16)
            },
            // (MEM 3 where the q/k norm is the identity: q's scale a constant)
            mem: if cfg.mem {
                if plan.spec.qk_norm { 1 } else { 3 }
            } else {
                0
            },
            op3: usize::from(cfg.op3),
            // (the grids' levers: ARF 1, OP3 2, MEM 4; MEM's only with ARF)
            grid: if cfg.grid_levers {
                usize::from(cfg.arf)
                    | (usize::from(cfg.op3) << 1)
                    | (usize::from(cfg.mem && cfg.arf) << 2)
            } else {
                0
            },
            f16cl: cfg.f16_clamp,
            xac: cfg.xac,
            // (TRM: every bit, the grids' with their f16 accumulation; 0533: TRM 479
            // 1,903.1 / 1,902.5 tok/s and TRM 255 1,897.4 / 1,897.7, both = golden)
            // (q/k-norm models, Qwen3-8B, take the same bits: 0534's "241, the exact bits
            // alone" kept the teacher-forced set against a golden of the u4 model; against the
            // bf16 model's own f32 golden (512 tokens) qt22's stream is golden and
            // the host emulation of every bit keeps it)
            trim: if cfg.trim {
                1019
                    | (usize::from(cfg.grid_levers && cfg.f16accg) << 2)
                    // (silu16: TRM 1024, the regions' act in f16, no act all-reduce; inexact;
                    // with 4096, 512's f16 gammas in place, so that the region fits beside 512)
                    | (if cfg.silu16 { 1024 | 4096 } else { 0 })
                    // (not 8192, the grids too: 0538 sp538_t20, grids +0.14k / +0.11k)
                    // (xg16: TRM 512 + 4096, x g in f16 with in-place f16 gammas; inexact)
                    | (if cfg.xg16 { 512 | 4096 } else { 0 })
                    // (and 8192: the grids' x g in f16, when they take the cuts)
                    | (usize::from(cfg.xg16 && cfg.grid_levers && cfg.f16accg) << 13)
                    // (head_pack: TRM 16384, the head's packed argmax round; exact)
                    | (usize::from(cfg.head_pack) << 14)
                    // (silu16h: TRM 2048, the SiLU's scales in f16)
                    | (usize::from(cfg.silu16h && cfg.silu16) << 11)
            } else {
                0
            },
            ary: if cfg.ary16 { 16 } else { 0 },
            hvc: cfg.hvc,
            slw: cfg.trim && cfg.silu16 && cfg.silu16w,
            ..Model::dims(g.hidden, g.qh, plan.vocab, cap)
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
            shape: if cfg.hv {
                // HV: one-position kernel pages, PIE's pages' worth a block.
                Shape {
                    page_size: 1,
                    blocks: g.blocks * g.page_size,
                    entries: 5,
                    ..pool(g, g.py.min(rpy))
                }
            } else {
                pool(g, g.py.min(rpy))
            },
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
    // Stated only where they differ from the layout's defaults (Qwen3's
    // builds keep their parameters).
    if grid_gy != g.gs {
        st.push(Typed("GY"));
    }
    if !plan.spec.qk_norm {
        st.push(Typed("QKN"));
    }
    if plan.spec.rms_eps != super::model::KERNEL_EPS {
        st.push(Typed("EPS"));
    }
    if cfg.speed {
        st.push(Typed("SPD"));
    }
    if cfg.lse {
        st.push(Typed("LSE"));
    }
    if cfg.ops2 {
        st.push(Typed("OP2"));
    }
    if cfg.qkv1 {
        st.push(Typed("QKV1"));
    }
    if cfg.arf {
        st.push(Typed("ARF"));
    }
    if cfg.f16 || cfg.f16acc {
        st.push(Typed("F16"));
    }
    if cfg.mem {
        st.push(Typed("MEM"));
    }
    if cfg.op3 {
        st.push(Typed("OP3"));
    }
    if cfg.grid_levers {
        st.push(Typed("GRD"));
    }
    if cfg.f16_clamp {
        st.push(Typed("F16CL"));
    }
    if cfg.xac {
        st.push(Typed("XAC"));
    }
    if cfg.trim {
        st.push(Typed("TRM"));
    }
    if cfg.ary16 {
        st.push(Typed("ARY"));
    }
    if cfg.hvc {
        st.push(Typed("HVC"));
    }
    if cfg.trim && cfg.silu16 && cfg.silu16w {
        st.push(Typed("SLW"));
    }
    if cfg.lever_d {
        st.push(Typed("YD"));
    }
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
        Knob("MODE", if cfg.head_one_pass { 2 } else { 0 }),
        Knob("OVL", 0),
        Typed("VMC"),
        Typed("CAP"),
        Knob("iters", 1),
        Typed("B"),
        Knob("NP", if pipe.is_some() { 1 } else { 3 }),
    ]);
    Ok(Draft { spec, stated: st })
}

/// The bytes of a serve8b_region strip PE's embedding tiles (`Vr` rows of
/// 4 `Hc` bf16 dims) the head's height is chosen for: 32 KiB (Qwen3-8B's
/// validated strip holds 30.6 KB).
const STRIP_TABLE: usize = 32 * 1024;

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
        Model::dims(g.hidden, g.qh, plan.vocab, p.cap),
        Layers::Grids {
            grid: kernel(g),
            gw: p.gw,
            gh: p.gh,
            tc: super::plan::TC,
            hybrid: None,
            gemma: None,
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
                Layers::Units { .. } | Layers::Gptoss { .. } => {}
            }
        }
        if let Some(v) = self.gh {
            match &mut spec.layers {
                Layers::Grids { gh, .. } | Layers::Strips { gh, .. } => *gh = v,
                Layers::Units { .. } | Layers::Gptoss { .. } => {}
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
        Design::ServeQ35 => return Err("serve_q35 builds from a hybrid plan (q35_program)".into()),
        Design::ServeG4 => return Err("serve_g4 builds from a Gemma 4 plan (g4_program)".into()),
        Design::ServeGptoss => {
            return Err("serve_gptoss builds from a MoE plan (gptoss_program)".into());
        }
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
    let files = program.files()?;
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
            norm_plus_one: false,
            hybrid: None,
            moe: None,
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

    pub(super) fn sorted(extra: &str) -> Vec<(String, String)> {
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

    /// Lever D for Qwen3-8B (36 layers: 1 + 10 x 3 + 5): the U order, the
    /// five column grids from (512, 208), the head as option A's (HH 1152);
    /// with `hv`, `speed`, `lse`, `head_one_pass` (the payload speed8b_qld).
    #[test]
    fn the_plan_places_qwen3_8b_lever_d() {
        let goals = Goals {
            blocks: Some(1),
            host_copy: false,
            grid_lanes: 1,
            ..Goals::lanes(1)
        };
        let plan = SpatialPlan::derive(&spec(), WaferDims::CS3, goals).expect("a plan");
        let mut cfg = ServeConfig::new(goals);
        cfg.region = true;
        cfg.speed = true;
        cfg.lse = true;
        cfg.hv = true;
        cfg.lever_d = true;
        cfg.head_one_pass = true;
        cfg.pins.set("NP", 1).unwrap();
        let (d, p) = params(&plan, &cfg).unwrap();
        let mut got = p.clone();
        got.sort();
        assert_eq!(
            got,
            sorted(
                "PX:128,PY:144,Hc:29,Dc:8,Gs:16,Ic:96,RPX:256,RPY:208,RHc:20,RDc:4,RGs:32,RIc:48,RNL:3,\
                 RGY:16,RC:2,RR:5,NCOL:5,YD:208,RTR:5,HR:4096,Qh:4,PS:1,NB:16,E:5,MC:1,W16:1,KV16:1,KVCP:0,\
                 PROF:0,TIME:0,HROUTE:1,ROT8:1,NOLAYER:0,HEADMODE:2,HH:1152,Vr:132,Gx:8,Gy:24,vocab:151936,\
                 PASSES:3,MODE:2,OVL:0,VMC:1,CAP:64,iters:1,B:1,NP:1,KVST:1,SPD:1,LSE:1"
            )
        );
        let lay =
            crate::spatial::program::ProgramSpec::from_params(d, &p).expect("serve8b_region D");
        assert_eq!(lay.layers(), 36);
        let num = |k: &str| p.iter().find(|e| e.0 == k).and_then(|e| e.1.parse().ok());
        plan.spec
            .check_program(num)
            .expect("the model takes the lever D program");
        assert_eq!(rectangle(d, &p).unwrap(), (749, 1153));
    }

    /// The Pie bundle's Qwen3-8B config (qx4: lever D + ops2 + qkv1 + arf +
    /// arfx + mem + op3 + f16acc + f16accg) with `grid_levers` and `xac`
    /// states 0529's sp529_qx7 program (TIME 1 there; KVST the engine's).
    #[test]
    fn the_plan_places_qwen3_8b_grid_levers_and_xac_as_0529() {
        let goals = Goals {
            blocks: Some(1),
            host_copy: false,
            grid_lanes: 1,
            ..Goals::lanes(1)
        };
        let plan = SpatialPlan::derive(&spec(), WaferDims::CS3, goals).expect("a plan");
        let mut cfg = ServeConfig::new(goals);
        (
            cfg.region,
            cfg.speed,
            cfg.lse,
            cfg.hv,
            cfg.lever_d,
            cfg.head_one_pass,
        ) = (true, true, true, true, true, true);
        (cfg.ops2, cfg.qkv1, cfg.arf, cfg.arfx, cfg.mem, cfg.op3) =
            (true, true, true, true, true, true);
        (cfg.f16acc, cfg.f16accg, cfg.grid_levers, cfg.xac) = (true, true, true, true);
        cfg.pins.set("NP", 1).unwrap();
        let (_, p) = params(&plan, &cfg).unwrap();
        let mut got = p.clone();
        got.retain(|(k, _)| k != "KVST");
        got.sort();
        assert_eq!(
            got,
            sorted(
                "PX:128,PY:144,Hc:29,Dc:8,Gs:16,Ic:96,RPX:256,RPY:208,RHc:20,RDc:4,RGs:32,RIc:48,RNL:3,\
                 RGY:16,RC:2,RR:5,NCOL:5,YD:208,RTR:5,HR:4096,Qh:4,PS:1,NB:16,E:5,MC:1,W16:1,KV16:1,KVCP:0,\
                 PROF:0,TIME:0,HROUTE:1,ROT8:1,NOLAYER:0,HEADMODE:2,HH:1152,Vr:132,Gx:8,Gy:24,vocab:151936,\
                 PASSES:3,MODE:2,OVL:0,VMC:1,CAP:64,iters:1,B:1,NP:1,SPD:1,LSE:1,OP2:1,QKV1:1,ARF:3,MEM:1,\
                 OP3:1,F16:3,GRD:7,XAC:1"
            )
        );
    }

    /// ... and with `trim`, `arfs` and `hvc` states 0535's sp535_qt15 (TIME 1 there) but
    /// TRM 1023: q/k-norm models take every op cut since qwen3-8b-speed (0535 ran 245).
    #[test]
    fn the_plan_places_qwen3_8b_trim_arfs_hvc_as_0535() {
        let goals = Goals {
            blocks: Some(1),
            host_copy: false,
            grid_lanes: 1,
            ..Goals::lanes(1)
        };
        let plan = SpatialPlan::derive(&spec(), WaferDims::CS3, goals).expect("a plan");
        let mut cfg = ServeConfig::new(goals);
        (
            cfg.region,
            cfg.speed,
            cfg.lse,
            cfg.hv,
            cfg.lever_d,
            cfg.head_one_pass,
        ) = (true, true, true, true, true, true);
        (cfg.ops2, cfg.qkv1, cfg.arf, cfg.arfx, cfg.mem, cfg.op3) =
            (true, true, true, true, true, true);
        (cfg.f16acc, cfg.f16accg, cfg.grid_levers, cfg.xac) = (true, true, true, true);
        (cfg.trim, cfg.arfs, cfg.hvc) = (true, true, true);
        cfg.pins.set("NP", 1).unwrap();
        let (_, p) = params(&plan, &cfg).unwrap();
        let mut got = p.clone();
        got.retain(|(k, _)| k != "KVST");
        got.sort();
        assert_eq!(
            got,
            sorted(
                "PX:128,PY:144,Hc:29,Dc:8,Gs:16,Ic:96,RPX:256,RPY:208,RHc:20,RDc:4,RGs:32,RIc:48,RNL:3,\
                 RGY:16,RC:2,RR:5,NCOL:5,YD:208,RTR:5,HR:4096,Qh:4,PS:1,NB:16,E:5,MC:1,W16:1,KV16:1,KVCP:0,\
                 PROF:0,TIME:0,HROUTE:1,ROT8:1,NOLAYER:0,HEADMODE:2,HH:1152,Vr:132,Gx:8,Gy:24,vocab:151936,\
                 PASSES:3,MODE:2,OVL:0,VMC:1,CAP:64,iters:1,B:1,NP:1,SPD:1,LSE:1,OP2:1,QKV1:1,ARF:7,MEM:1,\
                 OP3:1,F16:3,GRD:7,XAC:1,TRM:1023,HVC:1"
            )
        );
    }

    /// Qwen3-8B with the Llama levers: qg3's program (sp535_qt15: lever D, HV,
    /// ARF 7, F16 3, GRD 7, XAC, HVC) + `silu16`, `silu16h`, `xg16`, `head_pack` states
    /// TRM 32767, as Llama's t29 (its MEM stays 1: q/k norms).
    #[test]
    fn the_plan_places_qwen3_8b_with_the_llama_levers() {
        let goals = Goals {
            blocks: Some(1),
            host_copy: false,
            grid_lanes: 1,
            ..Goals::lanes(1)
        };
        let plan = SpatialPlan::derive(&spec(), WaferDims::CS3, goals).expect("a plan");
        let mut cfg = ServeConfig::new(goals);
        cfg.region = true;
        (
            cfg.speed,
            cfg.lse,
            cfg.ops2,
            cfg.qkv1,
            cfg.head_one_pass,
            cfg.hv,
            cfg.lever_d,
        ) = (true, true, true, true, true, true, true);
        (
            cfg.arf,
            cfg.arfx,
            cfg.arfs,
            cfg.mem,
            cfg.op3,
            cfg.f16acc,
            cfg.f16accg,
            cfg.grid_levers,
        ) = (true, true, true, true, true, true, true, true);
        (cfg.xac, cfg.trim, cfg.hvc) = (true, true, true);
        cfg.pins.set("NP", 1).unwrap();
        let want = "PX:128,PY:144,Hc:29,Dc:8,Gs:16,Ic:96,RPX:256,RPY:208,RHc:20,RDc:4,RGs:32,RIc:48,RNL:3,\
                    RGY:16,RC:2,RR:5,NCOL:5,YD:208,RTR:5,HR:4096,Qh:4,PS:1,NB:16,E:5,MC:1,W16:1,KV16:1,KVCP:0,\
                    PROF:0,TIME:0,HROUTE:1,ROT8:1,NOLAYER:0,HEADMODE:2,HH:1152,Vr:132,Gx:8,Gy:24,vocab:151936,\
                    PASSES:3,MODE:2,OVL:0,VMC:1,CAP:64,iters:1,B:1,NP:1,SPD:1,LSE:1,OP2:1,QKV1:1,ARF:7,MEM:1,\
                    OP3:1,F16:3,GRD:7,XAC:1,HVC:1";
        let got = |cfg: &ServeConfig| {
            let (_, p) = params(&plan, cfg).unwrap();
            let mut g = p.clone();
            g.retain(|(k, _)| k != "KVST");
            g.sort();
            g
        };
        // trim alone: every op cut (TRM 1023 = 1019 + the grids' 4), as Llama's
        assert_eq!(got(&cfg), sorted(&format!("{want},TRM:1023")));
        (cfg.silu16, cfg.silu16h, cfg.xg16, cfg.head_pack) = (true, true, true, true);
        assert_eq!(got(&cfg), sorted(&format!("{want},TRM:32767")));
        // + silu16w: the f16 SiLU's wide tail (SLW 1), regions only
        cfg.silu16w = true;
        assert_eq!(got(&cfg), sorted(&format!("{want},TRM:32767,SLW:1")));
        let (d, p) = params(&plan, &cfg).unwrap();
        let num = |k: &str| p.iter().find(|e| e.0 == k).and_then(|e| e.1.parse().ok());
        plan.spec.check_program(num).expect("the model takes it");
        let lay = crate::spatial::program::ProgramSpec::from_params(d, &p).expect("serve8b_region");
        assert!(lay.model.qk_norm);
        for (n, text) in d.files_with(lay.model.variant()).unwrap() {
            assert_eq!(text.contains("(trm 1024)"), n == "region_pe.csl", "{n}");
            assert_eq!(
                text.contains("(slw) e' = e 2^9"),
                n == "region_pe.csl",
                "{n}"
            );
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

#[cfg(test)]
mod llama31_8b {
    use super::*;
    use crate::spatial::codegen::Variant;
    use crate::spatial::model::ModelSpec;
    use crate::spatial::plan::WaferDims;

    fn spec() -> ModelSpec {
        let catalog = poem_compiler::catalog::repository();
        let d = catalog
            .parse("llama31-8b-bf16-kv-bf16")
            .expect("llama31-8b parses");
        ModelSpec::read(&d.trace(poem_ir::Platform::Cerebras)).expect("llama31-8b reads")
    }

    fn cfg_for(b: usize, pipeline: bool) -> (SpatialPlan, ServeConfig) {
        let goals = Goals {
            blocks: Some(1),
            host_copy: true,
            grid_lanes: 4,
            pipeline,
            rows: Some(152),
            ..Goals::lanes(b)
        };
        let plan = SpatialPlan::derive(&spec(), WaferDims::CS3, goals).expect("a plan");
        let mut cfg = ServeConfig::new(goals);
        cfg.region = true;
        (plan, cfg)
    }

    /// Llama 3.1 8B as serve8b_region's units: 152-row grids (GY 8; 160
    /// would be 761 columns), 3 column grids and a head grown to 864 rows so
    /// the strip's tiles hold the vocabulary, the units built without q/k
    /// norms at eps 1e-5. The model takes the program; the full rectangle
    /// fits the wafer.
    #[test]
    fn the_plan_places_llama_3_regions() {
        let (plan, cfg) = cfg_for(1, false);
        let (d, p) = params(&plan, &cfg).unwrap();
        assert_eq!(d, Design::Serve8bRegion);
        let get = |k: &str| p.iter().find(|e| e.0 == k).and_then(|e| e.1.parse().ok());
        for (k, v) in [
            ("PY", 152),
            ("Hc", 27),
            ("Ic", 112),
            ("RIc", 56),
            ("GY", 8),
            ("NCOL", 3),
            ("HH", 864),
            ("Vr", 150),
            ("Gx", 4),
            ("Gy", 24),
            ("QKN", 0),
            ("EPS", 1e-5f32.to_bits() as usize),
        ] {
            assert_eq!(get(k), Some(v), "{k}");
        }
        plan.spec
            .check_program(get)
            .expect("the model takes serve8b_region");
        assert_eq!(rectangle(d, &p).unwrap(), (755, 1112));
        let lay = ProgramSpec::from_params(d, &p).expect("serve8b_region");
        assert_eq!(lay.layers(), 32);
        assert!(!lay.model.qk_norm);
        // The generated units differ from Qwen3's by the variant alone.
        let qwen = d.files().unwrap();
        let llama = d.files_with(lay.model.variant()).unwrap();
        for ((n, a), (_, b)) in qwen.iter().zip(&llama) {
            let unit = *n == "grid_pe.csl" || *n == "region_pe.csl";
            assert_eq!(a == b, !unit, "{n}");
            if unit {
                assert!(a.contains("allreduce_seg(ssq3_dsd, NH * B);"));
                assert!(!b.contains("allreduce_seg(ssq3_dsd, NH * B);"), "{n}");
                assert!(b.contains("const EPS: f32 = 1.0e-5;"), "{n}");
            }
        }
        // Qwen3-8B's model refuses a Llama program, and Llama a Qwen3 one.
        let qwen3 = poem_compiler::catalog::repository()
            .parse("qwen3-8b-bf16-kv-bf16")
            .unwrap()
            .trace(poem_ir::Platform::Cerebras);
        let qwen3 = ModelSpec::read(&qwen3).unwrap();
        assert!(qwen3.check_program(get).is_err());
        let qwen_get = |k: &str| match k {
            "QKN" | "EPS" => None,
            _ => get(k),
        };
        assert!(plan.spec.check_program(qwen_get).is_err());
    }

    /// `lever_d` + `hv` (with `speed`, `lse`, `head_one_pass`, CAP 128):
    /// the plan states 0472's hardware build (lever D, Llama as
    /// 1 + 10 x 3 + 1 at RPY 208; HV's one-position pages), NP pinned to 1 as there,
    /// but for its TIME 1.
    #[test]
    fn the_plan_places_llama_3_lever_d_as_0472() {
        let (plan, mut cfg) = cfg_for(1, false);
        cfg.goals.cap = Some(128);
        cfg.speed = true;
        cfg.lse = true;
        cfg.head_one_pass = true;
        cfg.hv = true;
        cfg.lever_d = true;
        cfg.pins.set("NP", 1).unwrap();
        let plan = SpatialPlan::derive(&plan.spec, WaferDims::CS3, cfg.goals).unwrap();
        let (d, p) = params(&plan, &cfg).unwrap();
        assert_eq!(d, Design::Serve8bRegion);
        let want = "PX:128,PY:152,Hc:27,Dc:8,Gs:16,Ic:112,RPX:256,RPY:208,RHc:20,RDc:4,RGs:32,RIc:56,RNL:3,\
                    RGY:16,GY:8,RC:2,RR:5,NCOL:1,YD:624,RTR:5,HR:4096,Qh:4,PS:1,NB:16,E:5,MC:1,W16:1,KV16:1,\
                    KVCP:0,PROF:0,TIME:0,HROUTE:1,ROT8:1,QKN:0,EPS:925353388,NOLAYER:0,HEADMODE:2,HH:864,\
                    Vr:150,Gx:4,Gy:24,vocab:128256,PASSES:3,MODE:2,OVL:0,VMC:1,CAP:128,iters:1,B:1,NP:1,\
                    SPD:1,LSE:1";
        let mut got = p.clone();
        got.retain(|(k, _)| k != "KVST");
        got.sort();
        assert_eq!(got, super::qwen3_8b::sorted(want));
        assert_eq!(rectangle(d, &p).unwrap().1, 1040);
        let lay = ProgramSpec::from_params(d, &p).expect("serve8b_region D");
        assert_eq!(lay.layers(), 32);
        let num = |k: &str| p.iter().find(|e| e.0 == k).and_then(|e| e.1.parse().ok());
        plan.spec
            .check_program(num)
            .expect("the model takes the lever D program");
        // HV without LSE, and D on a layer count it cannot take, are refused.
        cfg.lse = false;
        assert!(params(&plan, &cfg).is_err());
    }

    /// The Pie bundle's Llama config (x4: ops2 + qkv1 + arf + arfx + mem + op3 +
    /// f16acc + f16accg) with `grid_levers` states 0522's sp522_g7 program
    /// (TIME 1 there; KVST the engine's).
    #[test]
    fn the_plan_places_llama_3_grid_levers_as_0522() {
        let (plan, mut cfg) = cfg_for(1, false);
        cfg.goals.cap = Some(128);
        (
            cfg.speed,
            cfg.lse,
            cfg.ops2,
            cfg.qkv1,
            cfg.head_one_pass,
            cfg.hv,
            cfg.lever_d,
        ) = (true, true, true, true, true, true, true);
        (
            cfg.arf,
            cfg.arfx,
            cfg.mem,
            cfg.op3,
            cfg.f16acc,
            cfg.f16accg,
            cfg.grid_levers,
        ) = (true, true, true, true, true, true, true);
        cfg.pins.set("NP", 1).unwrap();
        let plan = SpatialPlan::derive(&plan.spec, WaferDims::CS3, cfg.goals).unwrap();
        let (_, p) = params(&plan, &cfg).unwrap();
        let want = "PX:128,PY:152,Hc:27,Dc:8,Gs:16,Ic:112,RPX:256,RPY:208,RHc:20,RDc:4,RGs:32,RIc:56,RNL:3,\
                    RGY:16,GY:8,RC:2,RR:5,NCOL:1,YD:624,RTR:5,HR:4096,Qh:4,PS:1,NB:16,E:5,MC:1,W16:1,KV16:1,\
                    KVCP:0,PROF:0,TIME:0,HROUTE:1,ROT8:1,QKN:0,EPS:925353388,NOLAYER:0,HEADMODE:2,HH:864,\
                    Vr:150,Gx:4,Gy:24,vocab:128256,PASSES:3,MODE:2,OVL:0,VMC:1,CAP:128,iters:1,B:1,NP:1,\
                    SPD:1,LSE:1,OP2:1,QKV1:1,ARF:3,F16:3,MEM:3,OP3:1,GRD:7";
        let mut got = p.clone();
        got.retain(|(k, _)| k != "KVST");
        got.sort();
        assert_eq!(got, super::qwen3_8b::sorted(want));
        // + f16_clamp + xac: 0527's sp527_gx7
        (cfg.f16_clamp, cfg.xac) = (true, true);
        let (_, p) = params(&plan, &cfg).unwrap();
        let mut got = p.clone();
        got.retain(|(k, _)| k != "KVST");
        got.sort();
        assert_eq!(
            got,
            super::qwen3_8b::sorted(&format!("{want},F16CL:1,XAC:1"))
        );
        // + trim + ary16 + hvc: TRM 511, ARY 16, HVC 1 (0534's sp534_t14); + arfs: ARF 7
        (cfg.trim, cfg.ary16, cfg.hvc) = (true, true, true);
        let (_, p) = params(&plan, &cfg).unwrap();
        let mut got = p.clone();
        got.retain(|(k, _)| k != "KVST");
        got.sort();
        assert_eq!(
            got,
            super::qwen3_8b::sorted(&format!("{want},F16CL:1,XAC:1,TRM:1023,ARY:16,HVC:1"))
        );
        cfg.arfs = true;
        let (d, p) = params(&plan, &cfg).unwrap();
        let mut got = p.clone();
        got.retain(|(k, _)| k != "KVST");
        got.sort();
        let want7 = want.replace("ARF:3", "ARF:7");
        assert_eq!(
            got,
            super::qwen3_8b::sorted(&format!("{want7},F16CL:1,XAC:1,TRM:1023,ARY:16,HVC:1"))
        );
        let lay = ProgramSpec::from_params(d, &p).unwrap();
        for (n, text) in d.files_with(lay.model.variant()).unwrap() {
            if n == "region_pe.csl" || n == "grid_pe.csl" {
                assert!(
                    text.contains("(trm) y starts at 0")
                        && text.contains("(trm 2)")
                        && text.contains("(trm 8)"),
                    "{n}"
                );
                assert!(
                    text.contains("(trm) both halves' q heads in one op")
                        && text.contains("(trm 16)"),
                    "{n}"
                );
                assert!(text.contains("(trm 64)") && text.contains("(arfs)"), "{n}");
            }
        }
        // + silu16: TRM 6143 (0538's sp538_t19, 1,997.6 tok/s): the regions' act in f16, no act all-reduce
        cfg.silu16 = true;
        let (d, p) = params(&plan, &cfg).unwrap();
        let mut got = p.clone();
        got.retain(|(k, _)| k != "KVST");
        got.sort();
        assert_eq!(
            got,
            super::qwen3_8b::sorted(&format!("{want7},F16CL:1,XAC:1,TRM:6143,ARY:16,HVC:1"))
        );
        let lay = ProgramSpec::from_params(d, &p).unwrap();
        for (n, text) in d.files_with(lay.model.variant()).unwrap() {
            assert_eq!(text.contains("(trm 1024)"), n == "region_pe.csl", "{n}");
        }
        // + xg16: the grids' x g in f16 too (8192; 512 and 4096 already)
        cfg.xg16 = true;
        let (_, p) = params(&plan, &cfg).unwrap();
        let mut got = p.clone();
        got.retain(|(k, _)| k != "KVST");
        got.sort();
        assert_eq!(
            got,
            super::qwen3_8b::sorted(&format!("{want7},F16CL:1,XAC:1,TRM:14335,ARY:16,HVC:1"))
        );
        // + head_pack: TRM 30719
        cfg.head_pack = true;
        let (_, p) = params(&plan, &cfg).unwrap();
        let mut got = p.clone();
        got.retain(|(k, _)| k != "KVST");
        got.sort();
        assert_eq!(
            got,
            super::qwen3_8b::sorted(&format!("{want7},F16CL:1,XAC:1,TRM:30719,ARY:16,HVC:1"))
        );
        // + silu16h: TRM 32767 (everything)
        cfg.silu16h = true;
        let (_, p) = params(&plan, &cfg).unwrap();
        let mut got = p.clone();
        got.retain(|(k, _)| k != "KVST");
        got.sort();
        assert_eq!(
            got,
            super::qwen3_8b::sorted(&format!("{want7},F16CL:1,XAC:1,TRM:32767,ARY:16,HVC:1"))
        );
    }

    /// `ops2` on 0472's config states 0480's program: the same parameters
    /// plus OP2 1 (0480 ran OP2's unit files hand-copied into 0472's build;
    /// the codegen's files for these parameters are byte-identical to them).
    #[test]
    fn the_plan_places_llama_3_lever_d_with_ops2_as_0480() {
        let (plan, mut cfg) = cfg_for(1, false);
        cfg.goals.cap = Some(128);
        cfg.speed = true;
        cfg.lse = true;
        cfg.ops2 = true;
        cfg.head_one_pass = true;
        cfg.hv = true;
        cfg.lever_d = true;
        cfg.pins.set("NP", 1).unwrap();
        let plan = SpatialPlan::derive(&plan.spec, WaferDims::CS3, cfg.goals).unwrap();
        let (d, p) = params(&plan, &cfg).unwrap();
        let mut off = cfg.clone();
        off.ops2 = false;
        let (_, mut base) = params(&plan, &off).unwrap();
        base.push(("OP2".to_string(), "1".to_string()));
        let (mut got, mut want) = (p.clone(), base);
        got.sort();
        want.sort();
        assert_eq!(got, want);
        let lay = ProgramSpec::from_params(d, &p).expect("serve8b_region D + OP2");
        assert!(lay.model.ops2 && lay.layers() == 32);
    }

    /// `speed` states SPD 1 (and `head_one_pass` MODE 2): the units' files
    /// carry the op cuts (the exp on both kernels, the scores' and RoPE's on
    /// the regions only), the rest is the default build's.
    #[test]
    fn speed_states_spd_and_patches_the_units() {
        let (plan, mut cfg) = cfg_for(1, false);
        cfg.speed = true;
        cfg.head_one_pass = true;
        let (d, p) = params(&plan, &cfg).unwrap();
        let get = |k: &str| p.iter().find(|e| e.0 == k).map(|e| e.1.clone());
        assert_eq!(get("SPD").as_deref(), Some("1"));
        assert_eq!(get("MODE").as_deref(), Some("2"));
        let lay = ProgramSpec::from_params(d, &p).unwrap();
        assert!(lay.model.speed);
        let base = d
            .files_with(Variant {
                speed: false,
                ..lay.model.variant()
            })
            .unwrap();
        let fast = d.files_with(lay.model.variant()).unwrap();
        for ((n, a), (_, b)) in base.iter().zip(&fast) {
            match *n {
                "grid_pe.csl" => {
                    assert!(b.contains("fn exp_fast(") && !a.contains("fn exp_fast("));
                    assert!(!b.contains("fn q_parts("));
                }
                "region_pe.csl" => {
                    assert!(b.contains("fn exp_fast(") && b.contains("fn q_parts("));
                    assert!(b.contains("fn rope_vec("));
                }
                _ => assert_eq!(a, b, "{n}"),
            }
        }
    }

    /// `lse` (with `speed`) states LSE 1: both units' files drop the max
    /// all-reduce (the attention runs as a loop that a bad pass repeats) and
    /// the collectives' soft divide/modulo; `lse` without `speed` is refused.
    #[test]
    fn lse_states_lse_and_patches_both_units() {
        let (plan, mut cfg) = cfg_for(1, false);
        cfg.speed = true;
        cfg.lse = true;
        let (d, p) = params(&plan, &cfg).unwrap();
        let get = |k: &str| p.iter().find(|e| e.0 == k).map(|e| e.1.clone());
        assert_eq!(get("LSE").as_deref(), Some("1"));
        let lay = ProgramSpec::from_params(d, &p).unwrap();
        assert!(lay.model.lse && lay.model.speed);
        let base = d
            .files_with(Variant {
                lse: false,
                ..lay.model.variant()
            })
            .unwrap();
        let lse = d.files_with(lay.model.variant()).unwrap();
        for ((n, a), (_, b)) in base.iter().zip(&lse) {
            match *n {
                "grid_pe.csl" | "region_pe.csl" => {
                    assert!(a.contains("allreduce(true, red_dsd, NQB);"), "{n}");
                    assert!(!b.contains("allreduce(true,"), "{n}");
                    assert!(
                        b.contains("while (apass < 8)") && b.contains("fn gsh()"),
                        "{n}"
                    );
                    assert!(!b.contains("  const r = axis_id % ga();\n"), "{n}");
                    assert!(b.contains("    ar_qt = axis_id >> gsh();\n"), "{n}");
                    assert!(b.contains("fn q_parts("), "{n}");
                    assert_eq!(b.contains("fn rope_vec("), *n == "region_pe.csl", "{n}");
                    assert!(b.contains("const HV: bool = PS == 1;"), "{n}");
                }
                _ => assert_eq!(a, b, "{n}"),
            }
        }
        let alone = Variant {
            speed: false,
            ..lay.model.variant()
        };
        assert!(d.files_with(alone).is_err());
    }

    /// `ops2` (with `lse`) states OP2 1: both units' SiLU element runs as
    /// scalars at MC 1, set_axis returns on the axis it already has, PROF 1
    /// splits the MLP's marks; nothing else changes; without `lse` it is
    /// refused.
    #[test]
    fn ops2_states_op2_and_patches_both_units() {
        let (plan, mut cfg) = cfg_for(1, false);
        cfg.speed = true;
        cfg.lse = true;
        cfg.ops2 = true;
        let (d, p) = params(&plan, &cfg).unwrap();
        let get = |k: &str| p.iter().find(|e| e.0 == k).map(|e| e.1.clone());
        assert_eq!(get("OP2").as_deref(), Some("1"));
        let lay = ProgramSpec::from_params(d, &p).unwrap();
        assert!(lay.model.ops2);
        let base = d
            .files_with(Variant {
                ops2: false,
                ..lay.model.variant()
            })
            .unwrap();
        let op2 = d.files_with(lay.model.variant()).unwrap();
        for ((n, a), (_, b)) in base.iter().zip(&op2) {
            match *n {
                "grid_pe.csl" | "region_pe.csl" => {
                    assert!(!a.contains("fn silu1(") && b.contains("fn silu1("), "{n}");
                    assert!(b.contains("  if (axis_cur == want) { return; }\n"), "{n}");
                    assert!(
                        b.contains("    mark(14);\n") && !a.contains("mark(14)"),
                        "{n}"
                    );
                }
                _ => assert_eq!(a, b, "{n}"),
            }
        }
        let alone = Variant {
            lse: false,
            speed: true,
            ..lay.model.variant()
        };
        assert!(d.files_with(alone).is_err());
        cfg.lse = false;
        assert!(params(&plan, &cfg).is_err());
    }

    /// `qkv1` (with `ops2`) states QKV1 1: both units run one QKV matvec
    /// over `wqa`'s fused rows (wqv, wv as dummies); nothing else changes;
    /// without `ops2` it is refused.
    #[test]
    fn qkv1_states_qkv1_and_patches_both_units() {
        let (plan, mut cfg) = cfg_for(1, false);
        cfg.speed = true;
        cfg.lse = true;
        cfg.ops2 = true;
        cfg.qkv1 = true;
        let (d, p) = params(&plan, &cfg).unwrap();
        let get = |k: &str| p.iter().find(|e| e.0 == k).map(|e| e.1.clone());
        assert_eq!(get("QKV1").as_deref(), Some("1"));
        let lay = ProgramSpec::from_params(d, &p).unwrap();
        assert!(lay.model.qkv1);
        let base = d
            .files_with(Variant {
                qkv1: false,
                ..lay.model.variant()
            })
            .unwrap();
        let one = d.files_with(lay.model.variant()).unwrap();
        for ((n, a), (_, b)) in base.iter().zip(&one) {
            match *n {
                "grid_pe.csl" | "region_pe.csl" => {
                    assert!(
                        a.contains("wqv_dsd, xb, Hc, sc_s);") && !b.contains("wqv_dsd, xb"),
                        "{n}"
                    );
                    assert!(
                        b.contains("{QKW} -> qkv[i] }), wqa_dsd, xb, Hc, sc_s);"),
                        "{n}"
                    );
                    assert!(b.contains("var wv: [2]WS align(16)"), "{n}");
                }
                _ => assert_eq!(a, b, "{n}"),
            }
        }
        cfg.ops2 = false;
        assert!(params(&plan, &cfg).is_err());
    }

    /// `arf` states ARF 1 (the regions' Y all-reduces, grids unchanged);
    /// `f16acc` states F16 2 (KV and the head in f16 in both units' files,
    /// the regions' f16-accumulating matvecs); without `ops2` both are refused.
    #[test]
    fn arf_and_f16acc_state_their_params_and_patch_the_units() {
        let (plan, mut cfg) = cfg_for(1, false);
        cfg.speed = true;
        cfg.lse = true;
        cfg.ops2 = true;
        cfg.qkv1 = true;
        cfg.arf = true;
        cfg.arfx = true;
        cfg.f16acc = true;
        cfg.mem = true;
        cfg.op3 = true;
        let (d, p) = params(&plan, &cfg).unwrap();
        let get = |k: &str| p.iter().find(|e| e.0 == k).map(|e| e.1.clone());
        assert_eq!(get("ARF").as_deref(), Some("3"));
        assert_eq!(get("F16").as_deref(), Some("2"));
        assert_eq!(get("MEM").as_deref(), Some("3"));
        assert_eq!(get("OP3").as_deref(), Some("1"));
        let lay = ProgramSpec::from_params(d, &p).unwrap();
        assert_eq!((lay.model.arf, lay.model.f16, lay.model.mem), (3, 2, 3));
        let base = d
            .files_with(Variant {
                arf: 0,
                f16: 0,
                mem: 0,
                op3: 0,
                ..lay.model.variant()
            })
            .unwrap();
        let new = d.files_with(lay.model.variant()).unwrap();
        for ((n, a), (_, b)) in base.iter().zip(&new) {
            match *n {
                "region_pe.csl" => {
                    assert!(!a.contains("fn arf_ar(") && b.contains("fn arf_ar("), "{n}");
                    assert!(
                        b.contains("wmvh(gb, wgu_dsd, xn_dsd, Hc, 2 * Ic, false);"),
                        "{n}"
                    );
                    assert!(b.contains("@as(@fp16(), @bitcast(f32, u))"), "{n}");
                    assert!(
                        b.contains("const R2X: u16 =") && b.contains("prof[slot] += lo - t_lo;"),
                        "{n}"
                    );
                    assert!(
                        b.contains("fn axis_xa()") && b.contains("3 * NCOL + 3"),
                        "{n}"
                    );
                    assert!(b.contains("rope_tab();   // (op3)"), "{n}");
                    assert!(
                        b.contains("const QSCL: f32") && !b.contains("const qkn_a"),
                        "{n}"
                    );
                }
                "grid_pe.csl" => {
                    assert!(!b.contains("fn arf_ar(") && !b.contains("fn wmvh("), "{n}");
                    assert!(b.contains("@as(@fp16(), @bitcast(f32, u))"), "{n}");
                }
                "head_pe.csl" => {
                    assert!(
                        a.contains("]bf16);") && !b.contains("]bf16);") && b.contains("@fp16()"),
                        "{n}"
                    );
                }
                _ => assert_eq!(a, b, "{n}"),
            }
        }
        cfg.ops2 = false;
        assert!(params(&plan, &cfg).is_err());
        cfg.ops2 = true;
        cfg.arf = false;
        assert!(params(&plan, &cfg).is_err(), "mem / arfx without arf");
        cfg.mem = false;
        assert!(params(&plan, &cfg).is_err(), "arfx without arf");
    }

    /// `grid_levers` states GRD (ARF 1, OP3 2, MEM 4) and patches the grid
    /// kernel with the regions' levers; the regions' sources stay as without it.
    #[test]
    fn grid_levers_state_grd_and_patch_the_grids() {
        let (plan, mut cfg) = cfg_for(1, false);
        (cfg.speed, cfg.lse, cfg.ops2, cfg.qkv1, cfg.arf, cfg.arfx) =
            (true, true, true, true, true, true);
        (cfg.f16acc, cfg.f16accg, cfg.mem, cfg.op3) = (true, true, true, true);
        let (d, p0) = params(&plan, &cfg).unwrap();
        cfg.grid_levers = true;
        let (_, p) = params(&plan, &cfg).unwrap();
        let get = |k: &str| p.iter().find(|e| e.0 == k).map(|e| e.1.clone());
        assert_eq!(get("GRD").as_deref(), Some("7"));
        assert!(p0.iter().all(|e| e.0 != "GRD"));
        let base = ProgramSpec::from_params(d, &p0).unwrap();
        let lay = ProgramSpec::from_params(d, &p).unwrap();
        assert_eq!(lay.model.grid, 7);
        let a = d.files_with(base.model.variant()).unwrap();
        let b = d.files_with(lay.model.variant()).unwrap();
        for ((n, x), (_, y)) in a.iter().zip(&b) {
            if *n == "grid_pe.csl" {
                assert!(
                    !x.contains("fn arf_ar(")
                        && y.contains("fn arf_ar(")
                        && y.contains("fn axis_xa()"),
                    "{n}"
                );
                assert!(
                    y.contains("rope_tab();   // (op3)") && y.contains("fn rope_vec()"),
                    "{n}"
                );
                assert!(
                    y.contains("const QSCL: f32") && !y.contains("const qkn_a"),
                    "{n}"
                );
            } else {
                assert_eq!(x, y, "{n}");
            }
        }
        cfg.arf = false;
        cfg.arfx = false;
        cfg.mem = false;
        let (_, p) = params(&plan, &cfg).unwrap();
        assert_eq!(
            p.iter().find(|e| e.0 == "GRD").map(|e| e.1.as_str()),
            Some("2")
        );
        cfg.op3 = false;
        assert!(params(&plan, &cfg).is_err(), "grid_levers with no lever");
    }

    /// The ring at PY 152: B 1 x 12 at CAP 32 ran on hardware (0402) and
    /// builds; B 4 x 4 has not run there and stays refused.
    #[test]
    fn the_llama_3_ring_b1_ran_on_hardware() {
        let (plan, cfg) = cfg_for(1, true);
        let (d, p) = params(&plan, &cfg).unwrap();
        assert_eq!(d, Design::Serve8bRegionPipe);
        let get = |k: &str| p.iter().find(|e| e.0 == k).map(|e| e.1.as_str());
        assert_eq!(
            (get("PY"), get("B"), get("MMAX"), get("CAP")),
            (Some("152"), Some("1"), Some("12"), Some("32"))
        );
        let (plan, cfg) = cfg_for(4, true);
        let e = params(&plan, &cfg).unwrap_err();
        assert!(e.contains("PY:152 is unsupported"), "{e}");
    }

    /// Without the rows pin the plan's 160 rows are too wide for the head
    /// and strip (option A at 160 rows is 761 columns; the placement search
    /// finds no other shape), and serve8b's strips do not hold 32 layers.
    #[test]
    fn llama_3_wants_its_rows_pinned() {
        let goals = Goals {
            blocks: Some(1),
            host_copy: true,
            grid_lanes: 4,
            ..Goals::lanes(1)
        };
        let plan = SpatialPlan::derive(&spec(), WaferDims::CS3, goals).expect("a plan");
        let mut cfg = ServeConfig::new(goals);
        cfg.region = true;
        let g = plan.grid;
        assert_eq!(g.py, 160);
        assert!(
            place_at(
                &plan,
                g,
                (2 * g.px, 240, 3, 2, 4),
                false,
                false,
                false,
                true
            )
            .is_none()
        );
        let e = params(&plan, &cfg).unwrap_err();
        assert!(e.contains("no serve8b_region placement"), "{e}");
        cfg.region = false;
        assert!(params(&plan, &cfg).is_err());
    }

    /// The checkpoint config's llama3 scaling gives the trace's table.
    #[test]
    fn the_config_rope_scaling_is_the_traces_table() {
        let config = serde_json::json!({
            "rope_theta": 500000.0,
            "max_position_embeddings": 131072,
            "rope_scaling": {
                "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192, "rope_type": "llama3"
            }
        });
        let r = crate::spatial::wafer::Rotary::of_config(&config, 128)
            .unwrap()
            .unwrap();
        let s = spec();
        let want = &s.rope_table.as_ref().unwrap().inv_freq;
        let got = r.table.as_ref().unwrap();
        assert_eq!(got.len(), 64);
        for (a, b) in got.iter().zip(want) {
            assert!((a - b).abs() <= 1e-6 * b.abs(), "{a} {b}");
        }
        assert_eq!(r.inv_freq(128).unwrap(), *got);
        assert!(r.inv_freq(64).is_err());
        assert!(r.theta_only("serve2d").is_err());
        // transformers 5 spells it rope_parameters; a plain config has none.
        let v5 = serde_json::json!({ "rope_parameters": config["rope_scaling"].clone() });
        let mut v5 = v5;
        v5["rope_parameters"]["rope_theta"] = serde_json::json!(500000.0);
        let r5 = crate::spatial::wafer::Rotary::of_config(&v5, 128)
            .unwrap()
            .unwrap();
        assert_eq!(r5.table, r.table);
        let plain = serde_json::json!({ "rope_theta": 1.0e6 });
        let p = crate::spatial::wafer::Rotary::of_config(&plain, 128)
            .unwrap()
            .unwrap();
        assert!(p.table.is_none());
    }
}
