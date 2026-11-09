//! The resident wafer model in process: a compiled grid program (serve2d,
//! serve2d_pipe, serve8b, serve8b_region, serve8b_region_pipe) loaded
//! through a [`Device`], its routes and weights placed from the checkpoint,
//! requests answered by launches.
//!
//! A batched program takes a launch of lanes over PIE's KV pages
//! ([`WaferModel::fire_batch`]); a one-sequence program (serve2d `GRID` 0)
//! takes tokens at positions ([`WaferModel::forward`]), its KV cache
//! persisting until a request starts at 0. The head's eta VM, when the
//! program has one, samples in place of the head's argmax.
//!
//! The SDK wants every runtime call on the thread that made the first
//! runtime, so the model lives on a thread of its own ([`Wafer`]).

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;

use super::ckpt::Ckpt;
use super::device::Device;
use super::program::ProgramSpec;
use crate::sdk::{Platform, Rect, Runtime, Sdk};

/// `in_tokens`: take the step's own argmax.
pub const SAMPLE: u32 = 0xFFFF_FFFF;

type R<T> = std::result::Result<T, String>;

/// 16-bit elements (one a word, as a 16-bit memcpy gives them) two a word,
/// the even element in the low half.
pub fn pack16(cells: &[u32]) -> Vec<u32> {
    cells
        .chunks(2)
        .map(|p| (p[0] & 0xffff) | (p.get(1).copied().unwrap_or(0) & 0xffff) << 16)
        .collect()
}

/// [`pack16`]'s words back to `n` elements, one a word.
pub fn unpack16(words: &[u32], n: usize) -> Vec<u32> {
    words
        .iter()
        .flat_map(|&w| [w & 0xffff, w >> 16])
        .take(n)
        .collect()
}

/// The program a compiled output directory holds
/// ([`super::manifest::read_program`]: the manifest's spec, or `out.json`'s
/// params for a directory compiled outside PIE). Nothing is guessed: the
/// PE images' `vmlane` symbol, where readable, must be the size the spec's
/// VM ABI says, and `gring` at least.
pub fn read_design(program: &Path) -> R<ProgramSpec> {
    let spec = super::manifest::read_program(program)?;
    check_images(program, &spec)?;
    Ok(spec)
}

/// The PE images' account of the VM ABI against the spec's: the `vmlane`
/// control (a virtual lane's words, every lane of a ring launch) and the
/// `gring` header. A symbol no image defines is not checked.
fn check_images(program: &Path, spec: &ProgramSpec) -> R<()> {
    let Some(abi) = spec.vm_abi() else {
        return Ok(());
    };
    use super::guest4::lane::symbol_words;
    if let (Some(m), Some(shape)) = (spec.mmax(), spec.paged())
        && let Some(words) = symbol_words(program, "vmlane")
        && words != abi.lane_words * m * shape.lanes
    {
        return Err(format!(
            "{}: `vmlane` is {words} words; a {} image of {} virtual lanes holds {} a lane",
            program.display(),
            spec.design.vm_image().unwrap_or("VM"),
            m * shape.lanes,
            abi.lane_words
        ));
    }
    if let Some(words) = symbol_words(program, "gring")
        && words < abi.ring_words
    {
        return Err(format!(
            "{}: `gring` is {words} words; the {} image's header is {}",
            program.display(),
            spec.design.vm_image().unwrap_or("VM"),
            abi.ring_words
        ));
    }
    Ok(())
}

/// Time spent waiting on the wafer thread's launches, all fires together: the host's share of a token is
/// the rest of its wall time.
pub(crate) static MODEL_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Adds its lifetime to [`MODEL_NS`].
pub(crate) struct Busy(std::time::Instant);

impl Busy {
    pub(crate) fn start() -> Busy {
        Busy(std::time::Instant::now())
    }
}

impl Drop for Busy {
    fn drop(&mut self) {
        let ns = u64::try_from(self.0.elapsed().as_nanos()).unwrap_or(u64::MAX);
        MODEL_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Which logits a forward hands back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Logits {
    None,
    Last,
    All,
}

/// The program on the fabric, weights placed.
pub struct WaferModel {
    rt: Box<dyn Device>,
    design: ProgramSpec,
    /// Launches and layer steps run so far (the warm-up's included).
    launches: u64,
    steps: u64,
    /// Batched launches' lane-passes (tokens through the model).
    lane_steps: u64,
    /// The head's eta VM, when the build has one (set A), and a second VM
    /// set's (B: its own columns and image) beside it.
    vm: Option<Vm>,
    vm_b: Option<Vm>,
    /// A host-tier page copy is between its staging launch and its memcpy
    /// (the slots may alias a scratch buffer a fire uses): fires wait.
    staged: bool,
    /// The VM control (`vmlane`) the head holds, as last written: a launch
    /// writes it only when it changes.
    vm_written: Vec<u32>,
    /// A build with recurrent layers (serve_q35): whose state lane slot 0
    /// holds, and every state's tokens so far.
    states: States,
}

/// The recurrent states of a serve_q35 build. The wafer holds one lane
/// slot's (B 1): `resident`'s, its tokens `tokens[resident]`; any other
/// key's state is its tokens, replayed from a fresh state when a lane runs
/// on it (as a lane's KV is brought back on a one-sequence build).
#[derive(Default)]
struct States {
    resident: Option<u32>,
    tokens: std::collections::HashMap<u32, Vec<u32>>,
    /// Tokens replayed to bring a state back, for the logs.
    replayed: u64,
}

/// The head VM's state on the host: its shape and ring, the `gmode` the
/// head PEs hold, and the steps run since the loaded program's homes were
/// seeded (the ring's position).
struct Vm {
    lay: super::guest4::wafer::VmLayout,
    ring: super::guest4::lane::Ring,
    gmode: u32,
    steps: u64,
    loaded: bool,
    /// A per-lane-program image's arena and windows as placed.
    slots: super::guest4::slots::Slots,
}

pub use super::guest4::slots::StepInput;

/// One VM step's answer: the token the VM sampled (the head embedded it),
/// the step's `out` words, the pool after it.
pub type VmStep = (u32, Vec<u32>, Vec<u32>);

/// A rectangle of PEs from the layouts' `usize` coordinates.
pub(crate) fn rect(x: usize, y: usize, w: usize, h: usize) -> Rect {
    Rect {
        x: x as i32,
        y: y as i32,
        w: w as i32,
        h: h as i32,
    }
}

/// RoPE's inverse frequencies over `d` dims, as f32 (as numpy rounds them).
pub(crate) fn inv_freq(rope_theta: f64, d: usize) -> Vec<f32> {
    (0..d / 2)
        .map(|k| rope_theta.powf(-((2 * k) as f64) / d as f64) as f32)
        .collect()
}

/// RoPE's frequencies a placement writes: `theta`'s, unless a table is
/// stated (a llama3/YaRN scaling: the trace's, or the checkpoint config's
/// `rope_scaling`), `d / 2` inverse frequencies.
#[derive(Debug, Clone, PartialEq)]
pub struct Rotary {
    pub theta: f64,
    pub table: Option<Vec<f32>>,
}

impl Rotary {
    #[must_use]
    pub fn theta(theta: f64) -> Rotary {
        Rotary { theta, table: None }
    }

    /// The inverse frequencies over `d` dims: the table's, or theta's.
    pub fn inv_freq(&self, d: usize) -> R<Vec<f32>> {
        match &self.table {
            None => Ok(inv_freq(self.theta, d)),
            Some(t) if t.len() == d / 2 => Ok(t.clone()),
            Some(t) => Err(format!(
                "RoPE table of {} frequencies for a head of {d} dims",
                t.len()
            )),
        }
    }

    /// Theta, on placements that build frequencies from theta alone.
    pub fn theta_only(&self, what: &str) -> R<f64> {
        match self.table {
            None => Ok(self.theta),
            Some(_) => Err(format!(
                "{what}'s placement builds RoPE from theta; a scaled table (llama3/YaRN) \
                 is serve8b's and serve8b_region's"
            )),
        }
    }

    /// The checkpoint config's: `rope_theta` (or `rope_parameters`'), and
    /// its `rope_scaling` turned into a table over `d` dims (`None`: no
    /// theta stated).
    pub fn of_config(config: &serde_json::Value, d: usize) -> R<Option<Rotary>> {
        let params = &config["rope_parameters"];
        let Some(theta) = config["rope_theta"]
            .as_f64()
            .or_else(|| params["rope_theta"].as_f64())
        else {
            return Ok(None);
        };
        let scaling = match &config["rope_scaling"] {
            serde_json::Value::Object(_) => &config["rope_scaling"],
            _ => params,
        };
        let max = config["max_position_embeddings"]
            .as_u64()
            .and_then(|m| u32::try_from(m).ok());
        let table = match poem::rope::RopeScaling::of(&JsonDict(scaling), max)
            .map_err(|e| format!("config.json's rope scaling: {e}"))?
        {
            None => None,
            Some(s) => {
                let pairs = u32::try_from(d / 2).map_err(|e| e.to_string())?;
                let f = s
                    .freqs(theta as f32, pairs, pairs)
                    .map_err(|e| format!("config.json's rope scaling: {e}"))?;
                if f.mscale != 1.0 {
                    return Err(format!(
                        "config.json's rope scaling scales cos/sin by {} (YaRN): not placed",
                        f.mscale
                    ));
                }
                Some(f.inv_freq)
            }
        };
        Ok(Some(Rotary { theta, table }))
    }
}

/// A `rope_scaling` dict of a config.json, read by key.
struct JsonDict<'a>(&'a serde_json::Value);

impl poem::rope::Entries for JsonDict<'_> {
    fn number(&self, key: &str) -> anyhow::Result<Option<f64>> {
        match &self.0[key] {
            serde_json::Value::Null => Ok(None),
            v => v
                .as_f64()
                .map(Some)
                .ok_or_else(|| anyhow::anyhow!("rope scaling's `{key}` is a number, not {v}")),
        }
    }
    fn numbers(&self, key: &str) -> anyhow::Result<Option<Vec<f64>>> {
        match &self.0[key] {
            serde_json::Value::Null => Ok(None),
            serde_json::Value::Array(a) => a
                .iter()
                .map(|v| {
                    v.as_f64()
                        .ok_or_else(|| anyhow::anyhow!("rope scaling's `{key}` holds {v}"))
                })
                .collect::<anyhow::Result<_>>()
                .map(Some),
            v => anyhow::bail!("rope scaling's `{key}` is a list, not {v}"),
        }
    }
    fn text(&self, key: &str) -> anyhow::Result<Option<String>> {
        match &self.0[key] {
            serde_json::Value::Null => Ok(None),
            v => v
                .as_str()
                .map(|t| Some(t.to_string()))
                .ok_or_else(|| anyhow::anyhow!("rope scaling's `{key}` is a string, not {v}")),
        }
    }
    fn flag(&self, key: &str) -> anyhow::Result<Option<bool>> {
        match &self.0[key] {
            serde_json::Value::Null => Ok(None),
            v => v
                .as_bool()
                .map(Some)
                .ok_or_else(|| anyhow::anyhow!("rope scaling's `{key}` is a bool, not {v}")),
        }
    }
}

/// What a load places: a checkpoint's weights, or nothing (a host-protocol
/// check of a shape too large to fill).
#[derive(Clone, Copy)]
pub enum Weights<'a> {
    Place(&'a Ckpt),
    Skip,
}

/// Refuses `ckpt` where a component of `design` cannot place it; RoPE:
/// `rope` (the model's), else the checkpoint config's (its theta and
/// scaling), else theta 1e6.
fn check_weights(design: &ProgramSpec, ckpt: &Ckpt, rope: Option<Rotary>) -> R<Rotary> {
    for c in design.components() {
        c.validate(design, ckpt)?;
    }
    // (serve_gptoss places its own YaRN tables and mscale, serve_gptoss::yarn_inv_freq:
    // the generic table's refusal of an mscale is not its)
    let config = match Rotary::of_config(ckpt.config(), design.head_dim()) {
        Err(_) if design.gptoss().is_some() => None,
        r => r?,
    };
    if let (Some(rope), Some(config)) = (&rope, &config)
        && ((rope.theta - config.theta).abs() > 1e-6 * config.theta.abs()
            || rope.table.is_some() != config.table.is_some())
    {
        tracing::warn!(
            theta = rope.theta,
            config = config.theta,
            "spatial wafer: the checkpoint config's RoPE differs from the model's"
        );
    }
    Ok(rope.or(config).unwrap_or(Rotary::theta(1.0e6)))
}

impl WaferModel {
    /// [`WaferModel::open_with`] on a Hugging Face directory, RoPE from its
    /// config.
    pub fn open(program: &Path, ckpt_dir: &Path, platform: Platform) -> R<WaferModel> {
        WaferModel::open_with(program, Ckpt::open(ckpt_dir)?, platform, None)
    }

    /// Loads `program` and places `ckpt`'s weights; RoPE is `rope` (the
    /// model's, from its trace), else the checkpoint config's (else theta
    /// 1e6).
    pub fn open_with(
        program: &Path,
        ckpt: Ckpt,
        platform: Platform,
        rope: Option<Rotary>,
    ) -> R<WaferModel> {
        let design = read_design(program)?;
        // The VM's out ring's words, from the program's own account.
        let ring = design.vm_abi().map_or(0, |abi| abi.ring_words);
        let sdk = Sdk::open().map_err(|e| e.to_string())?;
        let mut rt = Runtime::new(sdk, program, platform, "WARNING").map_err(|e| e.to_string())?;
        rt.load();
        rt.run();
        WaferModel::on_device(Box::new(rt), design, Weights::Place(&ckpt), rope, ring)
    }

    /// The loaded program `design` on `dev` (loaded and running): its
    /// routes, `init`, the weights placed, the VM read, a warm-up request.
    /// `ring`: the VM's out ring's words (0 without a VM).
    pub fn on_device(
        mut rt: Box<dyn Device>,
        design: ProgramSpec,
        weights: Weights,
        rope: Option<Rotary>,
        ring: usize,
    ) -> R<WaferModel> {
        let rope = match weights {
            Weights::Place(ckpt) => Some(check_weights(&design, ckpt, rope)?),
            Weights::Skip => rope,
        };
        let started = Instant::now();
        // The route tables `init` applies.
        for c in design.components() {
            c.routes(&design, &mut *rt)?;
        }
        // Every PE takes its place, queue colors and routes before any
        // traffic.
        if rt.has_symbol("init") {
            rt.launch("init", &[]).map_err(|e| e.to_string())?;
        }
        tracing::info!(
            s = started.elapsed().as_secs_f32(),
            "spatial wafer: program loaded"
        );
        let started = Instant::now();
        if let (Weights::Place(ckpt), Some(rope)) = (weights, &rope) {
            let w = super::component::Weights { ckpt, rope };
            for c in design.components() {
                c.place(&design, &w, &mut *rt)?;
            }
        }
        tracing::info!(
            s = started.elapsed().as_secs_f32(),
            "spatial wafer: weights placed"
        );
        let mut model = WaferModel {
            rt,
            design,
            launches: 0,
            lane_steps: 0,
            vm: None,
            vm_b: None,
            staged: false,
            vm_written: Vec::new(),
            steps: 0,
            states: States::default(),
        };
        if let Some(lay) = model.design.vm() {
            let words = ring;
            let ring = super::guest4::lane::read_ring(&mut *model.rt, lay.hub(), words)?;
            super::guest4::wafer::set_gmode(&mut *model.rt, &lay, 0)?;
            tracing::info!(
                ?ring,
                ops = lay.max_ops,
                registers = lay.nreg,
                pool = lay.pool_words,
                "spatial wafer: the head has the eta VM"
            );
            model.vm = Some(Vm {
                lay,
                ring,
                gmode: 0,
                steps: 0,
                loaded: false,
                slots: super::guest4::slots::Slots::default(),
            });
            if let Some(lay) = model.design.vm_b() {
                let ring = super::guest4::lane::read_ring(&mut *model.rt, lay.hub(), words)?;
                tracing::info!(
                    ?ring,
                    ops = lay.max_ops,
                    registers = lay.nreg,
                    pool = lay.pool_words,
                    bank = lay.bank_words,
                    "spatial wafer: the head has a second VM set (B)"
                );
                model.vm_b = Some(Vm {
                    lay,
                    ring,
                    gmode: 0,
                    steps: 0,
                    loaded: false,
                    slots: super::guest4::slots::Slots::default(),
                });
            }
        }
        // One request (position 0; later requests overwrite its KV) so the
        // first real one does not pay first-launch costs.
        let started = Instant::now();
        if model.design.paged().is_some() {
            // Every lane one step at slot 0 of pool page 0 (later fires write
            // over it): a warm-up.
            let warm = super::serve2d::BatchLane {
                pos: 0,
                slot: 0,
                pages: vec![0],
                visible: None,
                forced: vec![0],
                state: None,
            };
            model.fire_batch(&[warm], 1, false)?;
        } else {
            model.run(&[0], 1, 0, false)?;
        }
        tracing::info!(
            s = started.elapsed().as_secs_f32(),
            "spatial wafer: warm-up request"
        );
        Ok(model)
    }

    pub fn design(&self) -> &ProgramSpec {
        &self.design
    }

    /// The profile counters (`prof`, 16 words a PE) of every layer unit
    /// (serve8b_region's grid areas, in layer order) and of the head block:
    /// for each, the max and the mean over its PEs a slot. The units count
    /// under TIME 1 (PROF 0: slot 0 the pass; PROF 1: a slot a phase), the
    /// head always. `zero` writes them back to 0 after the read.
    #[allow(clippy::type_complexity)]
    pub fn profile(&mut self, zero: bool) -> R<Vec<(Rect, Vec<u32>, Vec<f64>)>> {
        const NP: usize = 16;
        let mut rects = self.design.grid_areas();
        let hx = self.design.head_x();
        rects.push(rect(
            hx,
            0,
            self.design.chunk_cols() + self.design.vmc(),
            self.design.hh() + 1,
        ));
        let mut out = Vec::new();
        for r in rects {
            let a = self.rt.get("prof", r, NP)?;
            let n = a.len() / NP;
            let mut mx = vec![0u32; NP];
            let mut mean = vec![0f64; NP];
            for pe in a.chunks(NP) {
                for s in 0..NP {
                    mx[s] = mx[s].max(pe[s]);
                    mean[s] += pe[s] as f64 / n as f64;
                }
            }
            if zero {
                self.rt.write("prof", r, NP, &vec![0u32; a.len()])?;
            }
            out.push((r, mx, mean));
        }
        Ok(out)
    }

    /// The head block's profile counters per PE (16 words a PE, row-major over
    /// [`Self::profile`]'s head rectangle), and that rectangle's width.
    pub fn profile_head_raw(&mut self) -> R<(Vec<u32>, usize)> {
        let hx = self.design.head_x();
        let w = self.design.chunk_cols() + self.design.vmc();
        let a = self
            .rt
            .get("prof", rect(hx, 0, w, self.design.hh() + 1), 16)?;
        Ok((a, w))
    }

    /// Why the head's VM cannot take `table` (no VM, or a table past its
    /// limits or its ring's width), if it cannot.
    pub fn vm_refuses(&self, table: &super::guest4::Table) -> Option<String> {
        let Some(vm) = &self.vm else {
            return Some("the wafer program has no VM".into());
        };
        if !table.per_step.is_empty() {
            return Some(
                "the host writes some of its homes every step; this path writes no step input"
                    .into(),
            );
        }
        if vm.ring.lanes != 1 {
            return Some(format!(
                "the VM time-multiplexes {} lanes; this path drives one",
                vm.ring.lanes
            ));
        }
        vm.lay
            .refuses_head(table)
            .or_else(|| vm.ring.refuses_outputs(&table.outputs))
    }

    /// Loads `table` on the head's VM, its homes seeded with `seeds` (a
    /// lane's carried state): the VM's ring starts over.
    pub fn vm_load(
        &mut self,
        table: &super::guest4::Table,
        seeds: &[super::guest4::wafer::Seed],
    ) -> R<()> {
        if let Some(why) = self.vm_refuses(table) {
            return Err(why);
        }
        let vm = self.vm.as_mut().ok_or("the wafer program has no VM")?;
        super::guest4::wafer::load(&mut *self.rt, &vm.lay, table, seeds)?;
        vm.gmode = 1;
        vm.steps = 0;
        vm.loaded = true;
        Ok(())
    }

    /// Loads `table` on a multi-lane VM (`gring` lanes L, serve2d_pipe's
    /// lanes m·B + b): the program once for all lanes, every bank zeroed
    /// (each lane is seeded when a request takes it), the ring starting over.
    pub fn vm_load_lanes(&mut self, table: &super::guest4::Table) -> R<()> {
        let vm = self.vm.as_mut().ok_or("the wafer program has no VM")?;
        if let Some(why) = vm
            .lay
            .refuses(table)
            .or_else(|| vm.ring.refuses_outputs(&table.outputs))
        {
            return Err(why);
        }
        super::guest4::wafer::load(&mut *self.rt, &vm.lay, table, &[])?;
        let lanes = vm.ring.lanes as usize;
        let empty: Vec<Vec<super::guest4::wafer::Seed>> = (0..lanes).map(|_| Vec::new()).collect();
        super::guest4::wafer::write_banks(&mut *self.rt, &vm.lay, &empty)?;
        if vm.lay.sin > 0 {
            let entries = if table.per_step.is_empty() {
                Vec::new()
            } else {
                super::guest4::slots::step_entries(&vm.lay, table)?
            };
            let words = super::guest4::slots::gsdesc(&vec![entries; lanes]);
            super::guest4::wafer::write_gsdesc(&mut *self.rt, &vm.lay, &words)?;
        }
        vm.gmode = 1;
        vm.steps = 0;
        vm.loaded = true;
        Ok(())
    }

    /// Lane `lane`'s bank seeded with `seeds` (its request's carried state),
    /// the other lanes' banks as they stand.
    pub fn vm_seed_bank(&mut self, lane: usize, seeds: &[super::guest4::wafer::Seed]) -> R<()> {
        use super::guest4::wafer;
        let vm = self.vm.as_ref().ok_or("the wafer program has no VM")?;
        let lanes = vm.ring.lanes as usize;
        if lane >= lanes {
            return Err(format!("lane {lane} of a VM of {lanes}"));
        }
        let mut images = wafer::read_banks(&mut *self.rt, &vm.lay, lanes)?;
        // The lane's bank alone, laid out as a VM of one lane, spread into
        // its slot.
        let one = wafer::bank_images(
            &vm.lay,
            &[seeds
                .iter()
                .map(|s| wafer::Seed {
                    words: s.words.clone(),
                    ..*s
                })
                .collect()],
        )?;
        let bank = vm.lay.pool_words + (vm.lay.nreg - 1) * vm.lay.vr;
        for (src, dst) in one.iter().zip(images.iter_mut()) {
            for pe in 0..=vm.lay.rows {
                let at = (pe * lanes + lane) * bank;
                dst[at..at + bank].copy_from_slice(&src[pe * bank..(pe + 1) * bank]);
            }
        }
        wafer::write_bank_images(&mut *self.rt, &vm.lay, &images)
    }

    /// The head's VM, if the program has one: its layout and out ring.
    pub fn vm_shape(&self) -> Option<(super::guest4::wafer::VmLayout, super::guest4::lane::Ring)> {
        self.vm.as_ref().map(|vm| (vm.lay, vm.ring))
    }

    /// The second VM set's (B), if the program has one.
    pub fn vm_shape_b(
        &self,
    ) -> Option<(super::guest4::wafer::VmLayout, super::guest4::lane::Ring)> {
        self.vm_b.as_ref().map(|vm| (vm.lay, vm.ring))
    }

    /// A per-lane-program image's lanes: each its program (`None`: idle),
    /// the arena and windows compacted again, a lane on the same program
    /// keeping its window's state, a lane in `parked` getting the window it
    /// left back, a lane in `seeds` starting from them. The windows of the
    /// lanes that leave (their program gone or changed, or their bank
    /// seeded for another), by lane.
    /// `set` 0 places set A's lanes, 1 set B's.
    pub fn vm_place_lanes(
        &mut self,
        set: usize,
        programs: &[Option<super::guest4::slots::LaneProgram>],
        seeds: &[Option<Vec<super::guest4::wafer::Seed>>],
        parked: &[Option<super::guest4::slots::Parked>],
    ) -> R<Vec<Option<super::guest4::slots::Parked>>> {
        use super::guest4::{slots::Slots, wafer};
        let vm = if set == 0 {
            self.vm.as_mut()
        } else {
            self.vm_b.as_mut()
        };
        let vm = vm.ok_or(if set == 0 {
            "the wafer program has no VM"
        } else {
            "the wafer program has no VM set B"
        })?;
        if !vm.lay.lprog {
            return Err("the VM image runs one program for every lane (no vm_lprog)".into());
        }
        if programs.len() != vm.ring.lanes as usize {
            return Err(format!(
                "{} lanes for a VM of {}",
                programs.len(),
                vm.ring.lanes
            ));
        }
        let slots = Slots::plan(&vm.lay, programs)?;
        let banks = if vm.loaded {
            wafer::read_bank_words(&mut *self.rt, &vm.lay, vm.lay.bank_words)?
        } else {
            Vec::new()
        };
        let old = if vm.loaded {
            vm.slots.clone()
        } else {
            Slots::default()
        };
        let images = slots.rebank(&vm.lay, &old, &banks, seeds, parked)?;
        let left = (0..programs.len())
            .map(|l| {
                let was = old.lanes.get(l)?.as_ref()?;
                let stays = slots.lanes[l].as_ref().is_some_and(|s| s.key == was.key)
                    && !matches!(seeds.get(l), Some(Some(_)))
                    && !matches!(parked.get(l), Some(Some(_)));
                if stays {
                    None
                } else {
                    old.park(&vm.lay, l, &banks)
                }
            })
            .collect();
        wafer::load_slots(&mut *self.rt, &vm.lay, &slots, &images, !vm.loaded)?;
        // A first load writes gmode 1 (load_slots); a later placement leaves
        // gmode as the device has it (0 after argmax launches), for vm_fire
        // to set.
        let fresh = !vm.loaded;
        if fresh {
            vm.steps = 0;
            vm.gmode = 1;
        }
        vm.loaded = true;
        vm.slots = slots;
        // `gmode` is one word over both sets' head: set A's copy tracks it.
        if fresh && let Some(a) = self.vm.as_mut() {
            a.gmode = 1;
        }
        Ok(left)
    }

    /// The VM lanes a launch carries: set A's, then set B's (global lane g
    /// is head lane g; g past set A's lanes is set B's lane g − A's).
    pub fn vm_lanes(&self) -> (usize, usize) {
        let lanes = |vm: &Option<Vm>| vm.as_ref().map_or(0, |vm| vm.ring.lanes as usize);
        (lanes(&self.vm), lanes(&self.vm_b))
    }

    /// One lap of a multi-lane VM launch over both VM sets: `lanes` (head
    /// lane g is global VM lane g, [`WaferModel::vm_lanes`]; idle lanes
    /// repeat a live one) each forced its token, the VM sampling the next
    /// one of each lane in `sampled` from its set's bank. Every other
    /// lane's turn is skipped (its gmode_l 0: the head's argmax, no bank
    /// swap, no section, no ring row), so an idle or waiting lane never
    /// runs a program. Each set counts only its own turns (its lanes in
    /// head-lane order, its ring, its step input), so every lap gives each
    /// set exactly its lanes. Per global lane: the token the VM sampled,
    /// the turn's `out` words, its pool after it (empty for a skipped
    /// lane).
    pub fn vm_fire(
        &mut self,
        lanes: &[super::serve2d::BatchLane],
        sampled: &[bool],
        inputs: &[StepInput],
    ) -> R<Vec<VmStep>> {
        use super::guest4::wafer;
        self.unstaged()?;
        if self.design.hybrid().is_some() || self.design.q35r() {
            return Err(
                "the head's VM samples no serve_q35 lane (its states run on fire_batch)".into(),
            );
        }
        let (na, nb) = self.vm_lanes();
        let grid = &self.design;
        let n = na + nb;
        let fits = if nb > 0 {
            lanes.len() == n
        } else {
            n <= lanes.len()
        };
        if na == 0 || lanes.len() != grid.launch_lanes() || !fits {
            return Err(format!(
                "a VM launch of {} lanes on VM sets of {na} and {nb} lanes ({} lanes a launch)",
                lanes.len(),
                grid.launch_lanes()
            ));
        }
        let is_sampled = |g: usize| g < n && sampled.get(g).copied().unwrap_or(false);
        let ranges = [0..na, na..n];
        for (set, range) in ranges.iter().enumerate() {
            let vm = if set == 0 {
                self.vm.as_ref()
            } else {
                self.vm_b.as_ref()
            };
            let Some(vm) = vm else { continue };
            if range.clone().any(is_sampled) && !vm.loaded {
                return Err(format!("set {set}'s lanes sampled with no program loaded"));
            }
            // The step-input ring has `ring_d` slots: a launch's turns,
            // skips included, must fit it.
            if range.len() > vm.ring.depth as usize {
                return Err(format!(
                    "a launch of {} turns on set {set}'s ring of {}",
                    range.len(),
                    vm.ring.depth
                ));
            }
        }
        // `gmode` covers both sets' head: set A's copy tracks it.
        let a = self.vm.as_mut().ok_or("the wafer program has no VM")?;
        if a.gmode != 1 {
            wafer::set_gmode(&mut *self.rt, &a.lay, 1)?;
            a.gmode = 1;
        }
        for (set, range) in ranges.iter().enumerate() {
            let vm = if set == 0 {
                self.vm.as_ref()
            } else {
                self.vm_b.as_ref()
            };
            let Some(vm) = vm else { continue };
            let mine: Vec<StepInput> = range
                .clone()
                .map(|g| inputs.get(g).cloned().flatten())
                .collect();
            if vm.lay.sin > 0 {
                let images =
                    super::guest4::slots::step_images(&vm.lay, vm.ring.depth as usize, &mine);
                wafer::write_step(&mut *self.rt, &vm.lay, &images)?;
            } else if mine.iter().any(Option::is_some) {
                return Err(format!("step input for set {set}, whose image has none"));
            }
            // vm_conc: the set's rows this launch, before every launch.
            if grid.vm_conc() {
                wafer::write_conc_rows(&mut *self.rt, &vm.lay, vm.ring, range.len() as u32)?;
            }
        }
        // VMCONC: a VM launch rebinds the head's argmax chain to set B's
        // rows, so no lane may take the head's argmax (gmode_l 0). Every lane
        // is a VM lane; one this launch does not sample is a skipped turn
        // instead (its forced prefix past the launch's one lap): routed to
        // its set and counted in its rows, bank untouched, token 0, which no
        // one reads (only sampled lanes' tokens come back).
        let conc = grid.vm_conc();
        let mask: Vec<u32> = (0..lanes.len())
            .map(|g| u32::from(conc || is_sampled(g)))
            .collect();
        let nforce: Vec<u32> = (0..lanes.len())
            .map(|g| {
                let forced = lanes[g].forced.len() as u32;
                if is_sampled(g) {
                    forced
                } else {
                    forced.max(1) + 1
                }
            })
            .collect();
        let sets: Vec<u32> = (0..lanes.len())
            .map(|g| u32::from(g >= na && g < n))
            .collect();
        self.launches += 1;
        self.steps += 1;
        self.lane_steps += n as u64;
        let turns = super::serve2d::VmTurns {
            sampled: &mask,
            sets: &sets,
            nforce: conc.then_some(&nforce[..]),
        };
        let (tokens, _) = super::serve2d::fire_lanes(
            grid,
            &mut *self.rt,
            lanes,
            1,
            false,
            Some(&turns),
            &mut self.vm_written,
        )?;
        let mut out: Vec<VmStep> = tokens
            .iter()
            .take(n)
            .map(|t| {
                (
                    t.first().copied().unwrap_or_default(),
                    Vec::new(),
                    Vec::new(),
                )
            })
            .collect();
        for (set, range) in ranges.iter().enumerate() {
            let vm = if set == 0 {
                self.vm.as_mut()
            } else {
                self.vm_b.as_mut()
            };
            let Some(vm) = vm else { continue };
            // The set's ring takes a row a sampled turn of its own (its
            // `ostep`, its lanes in turn order): this launch's are the
            // `count` before its count now, read back so the host cannot
            // drift from it.
            let count = range.clone().filter(|&g| is_sampled(g)).count();
            if count == 0 {
                continue;
            }
            let ostep = wafer::read_ostep(&mut *self.rt, &vm.lay)?;
            if ostep != vm.steps + count as u64 {
                tracing::warn!(
                    set,
                    ostep,
                    expected = vm.steps + count as u64,
                    count,
                    "spatial wafer: a VM set's sampled-turn count is not the host's"
                );
            }
            let first = ostep.saturating_sub(count as u64);
            let mut rows =
                wafer::read_out_rows(&mut *self.rt, &vm.lay, vm.ring, first, count)?.into_iter();
            vm.steps = ostep;
            let pools = if vm.lay.lprog {
                let hub = wafer::read_hub_bank(&mut *self.rt, &vm.lay)?;
                vm.slots.pools(&vm.lay, &hub)
            } else {
                wafer::read_bank_pools(&mut *self.rt, &vm.lay, range.len())?
            };
            for (k, g) in range.clone().enumerate() {
                if is_sampled(g) {
                    out[g].1 = rows.next().unwrap_or_default();
                    out[g].2 = pools.get(k).cloned().unwrap_or_default();
                }
            }
        }
        Ok(out)
    }

    /// The last launch's logits rows of the first HLB lanes (y x 1/rms in
    /// f32: on a VM build, what the VM sampled from).
    pub fn lane_planes(&mut self) -> R<Vec<Vec<f32>>> {
        let grid = &self.design;
        super::serve2d::logits_rows(grid, &mut *self.rt, grid.logit_lanes())
    }

    /// The logits row the VM sampled its last step from
    /// ([`super::guest4::wafer::read_plane`]): for checks.
    pub fn vm_plane(&mut self) -> R<Vec<f32>> {
        let vm = self.vm.as_ref().ok_or("the wafer program has no VM")?;
        super::guest4::wafer::read_plane(&mut *self.rt, &vm.lay)
    }

    /// `token` at position `start` with the VM picking the next one: one
    /// launch of one iteration with `gmode` 1 (the VM runs the loaded
    /// program on that row's logits and the head embeds its token), then
    /// the step's `out` words and the VM's pool.
    pub fn vm_step(&mut self, token: u32, start: usize) -> R<VmStep> {
        self.unstaged()?;
        let grid = &self.design;
        let vm = self.vm.as_mut().ok_or("the wafer program has no VM")?;
        if !vm.loaded {
            return Err("no program is loaded on the VM".into());
        }
        if vm.gmode != 1 {
            super::guest4::wafer::set_gmode(&mut *self.rt, &vm.lay, 1)?;
            vm.gmode = 1;
        }
        self.launches += 1;
        self.steps += 1;
        let (sampled, _) = super::serve2d::run(grid, &mut *self.rt, &[token], 1, start, false)?;
        let mut rows =
            super::guest4::wafer::read_out(&mut *self.rt, &vm.lay, vm.ring, vm.steps, 1)?;
        vm.steps += 1;
        let pool = super::guest4::wafer::read_pool(&mut *self.rt, &vm.lay)?;
        let token = sampled
            .first()
            .copied()
            .ok_or("the launch answered no token")?;
        Ok((token, rows.pop().unwrap_or_default(), pool))
    }

    /// A batched build's launch ([`super::serve2d::fire`]).
    pub fn fire_batch(
        &mut self,
        lanes: &[super::serve2d::BatchLane],
        steps: usize,
        want_logits: bool,
    ) -> R<super::serve2d::BatchOut> {
        self.unstaged()?;
        let grid = &self.design;
        // The head's own argmax: the VM idles.
        if let Some(vm) = self.vm.as_mut()
            && vm.gmode != 0
        {
            super::guest4::wafer::set_gmode(&mut *self.rt, &vm.lay, 0)?;
            vm.gmode = 0;
        }
        self.launches += 1;
        self.steps += steps as u64;
        self.lane_steps += (lanes.len() * steps) as u64;
        if self.launches.is_multiple_of(64) {
            tracing::info!(
                launches = self.launches,
                steps = self.steps,
                lane_steps = self.lane_steps,
                lanes = lanes.len(),
                io = %self.rt.io_summary(),
                "spatial wafer: batch totals"
            );
        }
        let recurrent = grid.hybrid().is_some() || grid.q35r();
        if recurrent {
            self.seat_state(lanes)?;
        }
        let out = super::serve2d::fire(
            &self.design,
            &mut *self.rt,
            lanes,
            steps,
            want_logits,
            &mut self.vm_written,
        )?;
        if recurrent {
            self.ran_state(lanes, steps, &out.0);
        }
        Ok(out)
    }

    /// Lane slot 0's recurrent state brought to the launch's lane: zeroed
    /// for a fresh start, kept when it is the lane's at the lane's first
    /// slot, else the lane's tokens before it replayed from zero (on the
    /// lane's pages, which they rewrite as they were). A scratch launch (no
    /// state) leaves it nobody's.
    fn seat_state(&mut self, lanes: &[super::serve2d::BatchLane]) -> R<()> {
        let [lane] = lanes else {
            return Err(format!(
                "a serve_q35 launch carries one lane (B 1), not {}",
                lanes.len()
            ));
        };
        let Some(seat) = lane.state else {
            self.states.resident = None;
            return Ok(());
        };
        let at = lane.slot as usize;
        if seat.fresh {
            if at != 0 {
                return Err(format!(
                    "a fresh recurrent state {} at slot {at}: its tokens before are unknown",
                    seat.key
                ));
            }
            self.reset_state()?;
            self.states.tokens.insert(seat.key, Vec::new());
            self.states.resident = Some(seat.key);
            return Ok(());
        }
        let held = self.states.tokens.get(&seat.key);
        if self.states.resident == Some(seat.key) && held.is_some_and(|t| t.len() == at) {
            return Ok(());
        }
        let Some(before) = held.filter(|t| t.len() >= at).map(|t| t[..at].to_vec()) else {
            return Err(format!(
                "recurrent state {} at slot {at}: the wafer knows {} of its tokens",
                seat.key,
                held.map_or(0, Vec::len)
            ));
        };
        self.reset_state()?;
        self.states.resident = None;
        let cap = self.design.cap().saturating_sub(1).max(1);
        let shift = lane.pos.wrapping_sub(lane.slot);
        for (i, chunk) in before.chunks(cap).enumerate() {
            let first = (i * cap) as u32;
            let replay = super::serve2d::BatchLane {
                pos: first.wrapping_add(shift),
                slot: first,
                pages: lane.pages.clone(),
                visible: None,
                forced: chunk.to_vec(),
                state: None,
            };
            super::serve2d::fire(
                &self.design,
                &mut *self.rt,
                &[replay],
                chunk.len(),
                false,
                &mut self.vm_written,
            )?;
        }
        self.states.replayed += at as u64;
        tracing::info!(
            key = seat.key,
            tokens = at,
            total = self.states.replayed,
            "spatial wafer: a recurrent state replayed from its tokens"
        );
        self.states.tokens.insert(seat.key, before);
        self.states.resident = Some(seat.key);
        Ok(())
    }

    /// The state's tokens after a launch: the forced ones, then each
    /// iteration past them fed the argmax before.
    fn ran_state(
        &mut self,
        lanes: &[super::serve2d::BatchLane],
        steps: usize,
        argmax: &[Vec<u32>],
    ) {
        let (Some(lane), Some(key)) = (lanes.first(), self.states.resident) else {
            return;
        };
        let out = argmax.first().map_or(&[][..], Vec::as_slice);
        let f = lane.forced.len().min(steps);
        let fed = lane.forced[..f].iter().copied().chain(
            out.iter()
                .skip(f.saturating_sub(1))
                .take(steps - f)
                .copied(),
        );
        let tokens = self.states.tokens.entry(key).or_default();
        tokens.truncate(lane.slot as usize);
        tokens.extend(fed);
    }

    /// Lane slot 0's recurrent state zeroed (its conv window and state).
    fn reset_state(&mut self) -> R<()> {
        // (serve8b_region Q35R: a pass at write slot 0 zeroes the lane's state)
        if self.design.q35r() {
            return Ok(());
        }
        self.rt
            .launch("reset", &[0])
            .map_err(|e| format!("the recurrent state's reset: {e}"))
    }

    /// `copy_state`: state `dst` becomes `src`'s (its tokens; the wafer's
    /// slot replays them when a lane runs on it).
    pub fn copy_state(&mut self, src: u32, dst: u32) -> R<()> {
        let tokens = self
            .states
            .tokens
            .get(&src)
            .cloned()
            .ok_or_else(|| format!("copy_state from recurrent state {src}, which never ran"))?;
        self.states.tokens.insert(dst, tokens);
        if self.states.resident == Some(dst) {
            self.states.resident = None;
        }
        Ok(())
    }

    /// A batched build's whole-page copies (every layer's pool).
    pub fn copy_pages(&mut self, pairs: &[(u32, u32)]) -> R<()> {
        let grid = &self.design;
        grid.kv.copy_pages(grid, &mut *self.rt, pairs)
    }

    /// Refuses a fire while a host-tier page copy is mid-way.
    fn unstaged(&self) -> R<()> {
        if self.staged {
            return Err("a kv page copy to or from the host did not finish; no fire runs over its staging slots".into());
        }
        Ok(())
    }

    /// Bytes of one page's host copy, if the host can keep this program's
    /// pages (the host tier): its cells at the KV's own width (bf16 two a
    /// word, f32 one).
    pub fn page_bytes(&self) -> Option<usize> {
        let grid = &self.design;
        grid.kv.page_bytes(grid)
    }

    /// `pages`' cells off the device, a page each, packed at the KV's width
    /// (the host tier's suspension). Only between launches.
    pub fn page_out(&mut self, pages: &[u32]) -> R<Vec<Vec<u32>>> {
        let grid = &self.design;
        let bf16 = grid.kv_bytes() == 2;
        // A staged copy is one with no fire inside it: the flag stays up
        // past an error midway, and fires refuse while it is.
        self.staged = true;
        let pages: Vec<Vec<u32>> = grid
            .kv
            .read_pages(grid, &mut *self.rt, pages)?
            .into_iter()
            .map(|w| if bf16 { pack16(&w) } else { w })
            .collect();
        self.staged = false;
        Ok(pages)
    }

    /// Host copies back into device pages (a resumed lane's).
    pub fn page_in(&mut self, pages: &[(u32, Vec<u32>)]) -> R<()> {
        let grid = &self.design;
        let elements = grid
            .kv
            .page_words(grid)
            .ok_or("the host keeps no pages of this build")?;
        let bf16 = grid.kv_bytes() == 2;
        self.staged = true;
        let cells: Vec<(u32, Vec<u32>)> = pages
            .iter()
            .map(|(p, w)| {
                (
                    *p,
                    if bf16 {
                        unpack16(w, elements)
                    } else {
                        w.clone()
                    },
                )
            })
            .collect();
        grid.kv.write_pages(grid, &mut *self.rt, &cells)?;
        self.staged = false;
        Ok(())
    }

    /// One launch: head iterations 0..=steps, iteration i forcing forced[i]
    /// (past its end, the argmax). The argmax after each step; the last
    /// step's logits when asked.
    fn run(
        &mut self,
        forced: &[u32],
        steps: usize,
        start: usize,
        want_logits: bool,
    ) -> R<(Vec<u32>, Option<Vec<f32>>)> {
        self.unstaged()?;
        self.launches += 1;
        self.steps += steps as u64;
        // The head's own argmax: the VM idles.
        if let Some(vm) = self.vm.as_mut()
            && vm.gmode != 0
        {
            super::guest4::wafer::set_gmode(&mut *self.rt, &vm.lay, 0)?;
            vm.gmode = 0;
        }
        super::serve2d::run(
            &self.design,
            &mut *self.rt,
            forced,
            steps,
            start,
            want_logits,
        )
    }

    /// The wafer's token loop: `tokens` at positions `start..`, then `extra`
    /// more steps each fed the step before's argmax, in one launch where it
    /// fits (a long feed goes first in pieces). The argmax after every step:
    /// `tokens.len() + extra` of them, the last `extra + 1` the greedy
    /// continuation. Nothing but the argmaxes comes back.
    pub fn greedy(&mut self, tokens: &[u32], start: usize, extra: usize) -> R<Vec<u32>> {
        let piece = self.design.cap() - 1;
        let room = self.design.positions().saturating_sub(start + tokens.len());
        let extra = extra.min(piece.saturating_sub(1)).min(room);
        // The feed's head in pieces that leave the tail and the loop one launch.
        let tail = tokens.len().min(piece - extra).max(1);
        let head = tokens.len() - tail;
        let mut argmax = Vec::with_capacity(tokens.len() + extra);
        if head > 0 {
            let (a, _) = self.forward(&tokens[..head], start, Logits::None)?;
            argmax.extend(a);
        }
        let (a, _) = self.run(&tokens[head..], tail + extra, start + head, false)?;
        argmax.extend(a);
        Ok(argmax)
    }

    /// The tokens at positions `start..`: the argmax after each, and the
    /// logits `which` asks for. A launch takes at most `iters - 1` tokens;
    /// longer runs go in pieces (the cache carries on between them), and
    /// every row's logits take a launch a token.
    pub fn forward(
        &mut self,
        tokens: &[u32],
        start: usize,
        which: Logits,
    ) -> R<(Vec<u32>, Vec<Vec<f32>>)> {
        let mut argmax = Vec::with_capacity(tokens.len());
        let mut logits = Vec::new();
        if which == Logits::All {
            for (i, &tok) in tokens.iter().enumerate() {
                let (a, l) = self.run(&[tok], 1, start + i, true)?;
                argmax.extend(a);
                logits.extend(l);
            }
            return Ok((argmax, logits));
        }
        let piece = self.design.cap() - 1;
        let pieces = tokens.len().div_ceil(piece);
        for (i, chunk) in tokens.chunks(piece).enumerate() {
            let want = which == Logits::Last && i + 1 == pieces;
            let (a, l) = self.run(chunk, chunk.len(), start + i * piece, want)?;
            argmax.extend(a);
            logits.extend(l);
        }
        Ok((argmax, logits))
    }
}

enum Job {
    Forward {
        tokens: Vec<u32>,
        start: usize,
        which: Logits,
    },
    Greedy {
        tokens: Vec<u32>,
        start: usize,
        extra: usize,
    },
    /// A token loop posted without waiting (the role-split program): its
    /// answer on the posted channel.
    PostGreedy {
        tokens: Vec<u32>,
        start: usize,
        extra: usize,
    },
    /// A batched build's launch (answered on the batch channel).
    Batch {
        lanes: Vec<super::serve2d::BatchLane>,
        steps: usize,
        logits: bool,
    },
    /// A batched build's launch posted without waiting on it: answered on
    /// its own channel, then `landed` is told.
    Posted {
        lanes: Vec<super::serve2d::BatchLane>,
        steps: usize,
        reply: mpsc::Sender<BatchAnswer>,
        landed: Box<dyn FnOnce() + Send>,
    },
    /// A batched build's whole-page copies (answered on the batch channel).
    Copy {
        pairs: Vec<(u32, u32)>,
    },
    /// A recurrent state's copy (the batch channel).
    CopyState {
        src: u32,
        dst: u32,
    },
    /// Pages to and from the host tier (the batch channel: the pages'
    /// words as the answer's rows).
    PageOut {
        pages: Vec<u32>,
    },
    PageIn {
        pages: Vec<(u32, Vec<u32>)>,
    },
    /// A program onto the head's VM, and one VM step (the VM channel).
    VmLoad {
        program: std::sync::Arc<super::guest4::lane::Program>,
        seeds: Vec<super::guest4::wafer::Seed>,
    },
    VmStep {
        token: u32,
        start: usize,
    },
    /// A multi-lane VM: a program for every lane, one lane's bank seeded,
    /// one lap of a launch.
    VmLoadLanes {
        program: std::sync::Arc<super::guest4::lane::Program>,
    },
    VmSeedBank {
        lane: usize,
        seeds: Vec<super::guest4::wafer::Seed>,
    },
    VmFire {
        lanes: Vec<super::serve2d::BatchLane>,
        sampled: Vec<bool>,
        inputs: Vec<StepInput>,
    },
    /// A per-lane-program image's lanes placed.
    VmPlaceLanes {
        set: usize,
        programs: Vec<Option<super::guest4::slots::LaneProgram>>,
        seeds: Vec<Option<Vec<super::guest4::wafer::Seed>>>,
        parked: Vec<Option<super::guest4::slots::Parked>>,
        left: mpsc::Sender<Vec<Option<super::guest4::slots::Parked>>>,
    },
}

impl Job {
    /// The job's name in the slow-job line.
    fn kind(&self) -> &'static str {
        match self {
            Job::Forward { .. } => "forward",
            Job::Greedy { .. } => "greedy",
            Job::PostGreedy { .. } => "post_greedy",
            Job::Batch { .. } => "batch",
            Job::Posted { .. } => "posted",
            Job::Copy { .. } => "copy",
            Job::CopyState { .. } => "copy_state",
            Job::PageOut { .. } => "page_out",
            Job::PageIn { .. } => "page_in",
            Job::VmLoad { .. } => "vm_load",
            Job::VmStep { .. } => "vm_step",
            Job::VmLoadLanes { .. } => "vm_load_lanes",
            Job::VmSeedBank { .. } => "vm_seed_bank",
            Job::VmFire { .. } => "vm_fire",
            Job::VmPlaceLanes { .. } => "vm_place_lanes",
        }
    }
}

/// Logs a wafer-thread job that took over 50 ms when it ends (dropped); a
/// launch (`batch`, `posted`) says so itself when slow for its steps.
struct SlowJob(&'static str, std::time::Instant);

impl Drop for SlowJob {
    fn drop(&mut self) {
        let ms = self.1.elapsed().as_secs_f64() * 1e3;
        if ms > 50.0 && !matches!(self.0, "batch" | "posted") {
            tracing::warn!(kind = self.0, ms, "spatial wafer: a slow job");
        }
    }
}

type Answer = R<(Vec<u32>, Vec<Vec<f32>>)>;
/// Each lane's tokens and logits rows.
pub(crate) type BatchAnswer = R<super::serve2d::BatchOut>;

/// A [`WaferModel`] on its own thread.
pub struct Wafer {
    jobs: Option<mpsc::Sender<Job>>,
    answers: mpsc::Receiver<Answer>,
    batch_answers: mpsc::Receiver<BatchAnswer>,
    /// The batched build's paging (`None`: one sequence), and its
    /// iterations a launch.
    pub paged: Option<super::paged::Shape>,
    pub cap: usize,
    /// Lanes a batched launch carries (MMAX micro-batches of B on a ring),
    /// and how many of them may want logits.
    pub launch_lanes: usize,
    pub logit_lanes: usize,
    /// The KV positions one sequence may hold (a lane's pages on a batched
    /// build).
    pub positions: usize,
    /// The head's eta VM (its shape and out ring), when the build has one,
    /// and a second VM set's (B).
    pub vm: Option<(super::guest4::wafer::VmLayout, super::guest4::lane::Ring)>,
    pub vm_b: Option<(super::guest4::wafer::VmLayout, super::guest4::lane::Ring)>,
    /// Bytes of one page's host copy (`None`: the host keeps no pages of
    /// this build).
    pub page_bytes: Option<usize>,
    vm_answers: mpsc::Receiver<R<Vec<VmStep>>>,
    thread: Option<std::thread::JoinHandle<()>>,
    pub vocab: usize,
    pub hidden: usize,
    /// The build has recurrent layers (serve_q35): lanes carry their state.
    pub recurrent: bool,
    /// Kernel pages a PIE page (1: the program's pages are PIE's). A build
    /// whose pages are shorter than PIE's (HV, `PS` 1) keeps PIE's page
    /// size: PIE page `g` is kernel pages `g * group .. (g + 1) * group`
    /// (offset `o` in kernel page `g * group + o / PS`), expanded here at
    /// the handle: the lanes' page tables, the page copies and the host
    /// tier's pages. `paged` and `page_bytes` are PIE's view.
    pub group: u32,
    /// `PIE_CEREBRAS_PAGE_ORDER` (`ar:<py>:<gy>`): kernel pages relabelled
    /// onto the rows by all-reduce slack ([`super::paged::RowOrder`]),
    /// after the group expansion, on every page the handle passes on.
    pub row_order: Option<super::paged::RowOrder>,
    /// The role-split program ([`super::split`]).
    pub split: bool,
    /// The prep's posted-loop length (its meta's `LOOP`; `PIE_CEREBRAS_SPLIT_LOOP` overrides it).
    pub split_loop: Option<usize>,
    /// Posted token loops' answers (the role-split program).
    posted: mpsc::Receiver<Answer>,
}

/// The handle's page order from `PIE_CEREBRAS_PAGE_ORDER`, over a kernel
/// pool of `paged`'s pages (unset or no pool: none).
fn row_order_env(paged: Option<super::paged::Shape>) -> R<Option<super::paged::RowOrder>> {
    match (std::env::var("PIE_CEREBRAS_PAGE_ORDER").ok(), paged) {
        (Some(spec), Some(shape)) => {
            let o = super::paged::RowOrder::parse(&spec, &shape)?;
            tracing::info!(
                "PIE_CEREBRAS_PAGE_ORDER {spec}: rows first {:?}",
                &o.rows()[..4.min(o.rows().len())]
            );
            Ok(Some(o))
        }
        _ => Ok(None),
    }
}

/// One launch's account: how long the wafer thread waited for it (`idle_ms`,
/// since its previous job), how long it ran (`ms`, tables to tokens), its
/// passes and lanes, and the kernel pages its lanes name.
fn launch_line(
    kind: &str,
    lanes: &[super::serve2d::BatchLane],
    steps: usize,
    idle_from: std::time::Instant,
    t0: std::time::Instant,
) {
    tracing::info!(
        kind,
        steps,
        lanes = lanes.len(),
        pages = lanes.iter().map(|l| l.pages.len()).sum::<usize>(),
        pos = lanes.first().map_or(0, |l| l.pos),
        idle_ms = (t0 - idle_from).as_secs_f64() * 1e3,
        tables_ms =
            super::serve2d::TABLES_NS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6,
        ms = t0.elapsed().as_secs_f64() * 1e3,
        "spatial wafer: launch"
    );
}

/// A page job's account (`copy_pages`, `page_out`, `page_in`): its kernel
/// pages, the wafer thread's wait for it and its time.
fn job_line(kind: &str, pages: usize, idle_from: std::time::Instant, t0: std::time::Instant) {
    tracing::info!(
        kind,
        pages,
        idle_ms = (t0 - idle_from).as_secs_f64() * 1e3,
        ms = t0.elapsed().as_secs_f64() * 1e3,
        "spatial wafer: page job"
    );
}

impl Wafer {
    /// Loads `program` (the compiler's output directory) on `platform` and
    /// places `ckpt` (RoPE `rope`, else the checkpoint config's); returns
    /// once the model answers.
    pub fn open(
        program: PathBuf,
        ckpt: Ckpt,
        platform: Platform,
        rope: Option<Rotary>,
    ) -> R<Wafer> {
        if super::split::is_split(&program) {
            return Wafer::open_split(program, &ckpt, platform);
        }
        let (job_tx, job_rx) = mpsc::channel::<Job>();
        let (ans_tx, ans_rx) = mpsc::channel::<Answer>();
        let (batch_tx, batch_rx) = mpsc::channel::<BatchAnswer>();
        let (vm_tx, vm_rx) = mpsc::channel::<R<Vec<VmStep>>>();
        let (ready_tx, ready_rx) = mpsc::channel::<R<Wafer>>();
        let thread = std::thread::Builder::new()
            .name("spatial-wafer".into())
            .spawn(move || {
                let mut model = match WaferModel::open_with(&program, ckpt, platform, rope) {
                    Ok(m) => {
                        let g = &m.design;
                        let (paged, (launch_lanes, logit_lanes)) =
                            (g.paged(), (g.launch_lanes(), g.logit_lanes()));
                        // The facts the engine reads; the channels are filled in below.
                        let _ = ready_tx.send(Ok(Wafer {
                            jobs: None,
                            answers: mpsc::channel().1,
                            batch_answers: mpsc::channel().1,
                            vm_answers: mpsc::channel().1,
                            paged,
                            cap: m.design.cap(),
                            launch_lanes,
                            logit_lanes,
                            positions: m.design.positions(),
                            vm: m.vm.as_ref().map(|vm| (vm.lay, vm.ring)),
                            vm_b: m.vm_b.as_ref().map(|vm| (vm.lay, vm.ring)),
                            page_bytes: m.page_bytes(),
                            thread: None,
                            vocab: m.design.vocab(),
                            hidden: m.design.hidden(),
                            recurrent: m.design.hybrid().is_some() || m.design.q35r(),
                            group: 1,
                            row_order: match row_order_env(m.design.paged()) {
                                Ok(o) => o,
                                Err(e) => {
                                    let _ = ready_tx.send(Err(e));
                                    return;
                                }
                            },
                            split: false,
            split_loop: None,
                            posted: mpsc::channel().1,
                        }));
                        m
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                // The end of the previous job: a launch's `idle_ms` is the
                // wafer thread's wait for it (the host's share between launches).
                let mut idle_from = std::time::Instant::now();
                for job in job_rx {
                    // (Every job the wafer thread runs, timed when slow: the
                    // host-stall hunt.)
                    let job_t0 = std::time::Instant::now();
                    let job_kind = job.kind();
                    let _slow = SlowJob(job_kind, job_t0);
                    let answer = match job {
                        Job::Batch {
                            lanes,
                            steps,
                            logits,
                        } => {
                            let t0 = std::time::Instant::now();
                            let reply = model.fire_batch(&lanes, steps, logits);
                            launch_line("batch", &lanes, steps, idle_from, t0);
                            idle_from = std::time::Instant::now();
                            if batch_tx.send(reply).is_err() {
                                break;
                            }
                            continue;
                        }
                        Job::Posted {
                            lanes,
                            steps,
                            reply,
                            landed,
                        } => {
                            // The poster may have gone (its lanes diverged):
                            // the launch still ran, in order.
                            let t0 = std::time::Instant::now();
                            let _ = reply.send(model.fire_batch(&lanes, steps, false));
                            launch_line("posted", &lanes, steps, idle_from, t0);
                            landed();
                            idle_from = std::time::Instant::now();
                            continue;
                        }
                        Job::VmLoad { program, seeds } => {
                            let reply = model.vm_load(&program.table, &seeds).map(|()| Vec::new());
                            if vm_tx.send(reply).is_err() {
                                break;
                            }
                            continue;
                        }
                        Job::VmLoadLanes { program } => {
                            let reply = model.vm_load_lanes(&program.table).map(|()| Vec::new());
                            if vm_tx.send(reply).is_err() {
                                break;
                            }
                            continue;
                        }
                        Job::VmSeedBank { lane, seeds } => {
                            let reply = model.vm_seed_bank(lane, &seeds).map(|()| Vec::new());
                            if vm_tx.send(reply).is_err() {
                                break;
                            }
                            continue;
                        }
                        Job::VmPlaceLanes {
                            set,
                            programs,
                            seeds,
                            parked,
                            left,
                        } => {
                            let reply =
                                model
                                    .vm_place_lanes(set, &programs, &seeds, &parked)
                                    .map(|out| {
                                        let _ = left.send(out);
                                        Vec::new()
                                    });
                            if vm_tx.send(reply).is_err() {
                                break;
                            }
                            continue;
                        }
                        Job::VmFire {
                            lanes,
                            sampled,
                            inputs,
                        } => {
                            let reply = model.vm_fire(&lanes, &sampled, &inputs);
                            if vm_tx.send(reply).is_err() {
                                break;
                            }
                            continue;
                        }
                        Job::VmStep { token, start } => {
                            let reply = model.vm_step(token, start).map(|step| vec![step]);
                            if vm_tx.send(reply).is_err() {
                                break;
                            }
                            continue;
                        }
                        Job::PageOut { pages } => {
                            let t0 = std::time::Instant::now();
                            let reply = model.page_out(&pages).map(|words| (words, Vec::new()));
                            job_line("page_out", pages.len(), idle_from, t0);
                            idle_from = std::time::Instant::now();
                            if batch_tx.send(reply).is_err() {
                                break;
                            }
                            continue;
                        }
                        Job::PageIn { pages } => {
                            let t0 = std::time::Instant::now();
                            let reply = model.page_in(&pages).map(|()| (Vec::new(), Vec::new()));
                            job_line("page_in", pages.len(), idle_from, t0);
                            idle_from = std::time::Instant::now();
                            if batch_tx.send(reply).is_err() {
                                break;
                            }
                            continue;
                        }
                        Job::CopyState { src, dst } => {
                            let reply = model
                                .copy_state(src, dst)
                                .map(|()| (Vec::new(), Vec::new()));
                            if batch_tx.send(reply).is_err() {
                                break;
                            }
                            continue;
                        }
                        Job::Copy { pairs } => {
                            let t0 = std::time::Instant::now();
                            let reply = model.copy_pages(&pairs).map(|()| (Vec::new(), Vec::new()));
                            job_line("copy_pages", pairs.len(), idle_from, t0);
                            idle_from = std::time::Instant::now();
                            if batch_tx.send(reply).is_err() {
                                break;
                            }
                            continue;
                        }
                        Job::Forward {
                            tokens,
                            start,
                            which,
                        } => model.forward(&tokens, start, which),
                        Job::Greedy {
                            tokens,
                            start,
                            extra,
                        } => model.greedy(&tokens, start, extra).map(|a| (a, Vec::new())),
                        Job::PostGreedy { .. } => {
                            Err("posted token loops are the role-split program's".into())
                        }
                    };
                    idle_from = std::time::Instant::now();
                    tracing::info!(
                        launches = model.launches,
                        steps = model.steps,
                        "spatial wafer: totals"
                    );
                    if ans_tx.send(answer).is_err() {
                        break;
                    }
                }
            })
            .map_err(|e| format!("spawning the wafer thread: {e}"))?;
        let mut facts = ready_rx
            .recv()
            .map_err(|_| "the wafer thread died while loading".to_string())??;
        let row_order = facts.row_order.take();
        Ok(Wafer {
            row_order,
            jobs: Some(job_tx),
            answers: ans_rx,
            batch_answers: batch_rx,
            vm_answers: vm_rx,
            thread: Some(thread),
            posted: mpsc::channel().1,
            ..facts
        })
    }

    /// The role-split program ([`super::split`]) on its own thread: one
    /// sequence, greedy rows by the wafer token loop, logits rows by `lgt`.
    fn open_split(program: PathBuf, ckpt: &Ckpt, platform: Platform) -> R<Wafer> {
        let spec = super::split::SplitSpec::from_env()?.ok_or("PIE_CEREBRAS_SPLIT is unset")?;
        let split_loop = spec.int("LOOP").map(|v| v as usize);
        let cfg = ckpt.config();
        let vocab = cfg["text_config"]["vocab_size"]
            .as_u64()
            .or_else(|| cfg["vocab_size"].as_u64())
            .ok_or("the checkpoint config states no vocab_size")? as usize;
        let (job_tx, job_rx) = mpsc::channel::<Job>();
        let (ans_tx, ans_rx) = mpsc::channel::<Answer>();
        let (post_tx, post_rx) = mpsc::channel::<Answer>();
        let (ready_tx, ready_rx) = mpsc::channel::<R<(usize, usize, usize)>>();
        let thread = std::thread::Builder::new()
            .name("spatial-wafer".into())
            .spawn(move || {
                let open = || -> R<super::split::SplitModel> {
                    let sdk = Sdk::open().map_err(|e| e.to_string())?;
                    let mut rt =
                        Runtime::new(sdk, &program, platform, "WARNING").map_err(|e| e.to_string())?;
                    let t0 = Instant::now();
                    rt.load();
                    rt.run();
                    tracing::info!(s = t0.elapsed().as_secs_f64(), "split: program loaded");
                    super::split::SplitModel::open(Box::new(rt), &spec, vocab)
                };
                let mut m = match open() {
                    Ok(m) => {
                        let _ = ready_tx.send(Ok((m.cap, m.positions, m.hidden)));
                        m
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                let mut idle_from = Instant::now();
                for job in job_rx {
                    let t0 = Instant::now();
                    if let Job::PostGreedy {
                        tokens,
                        start,
                        extra,
                    } = job
                    {
                        let a = m.greedy(&tokens, start, extra).map(|a| (a, Vec::new()));
                        tracing::info!(
                            rows = tokens.len(),
                            steps = extra + 1,
                            pos = start,
                            idle_ms = (t0 - idle_from).as_secs_f64() * 1e3,
                            ms = t0.elapsed().as_secs_f64() * 1e3,
                            "spatial wafer: split token loop launch (posted)"
                        );
                        idle_from = Instant::now();
                        if post_tx.send(a).is_err() {
                            break;
                        }
                        continue;
                    }
                    let answer = match job {
                        Job::Greedy {
                            tokens,
                            start,
                            extra,
                        } => {
                            let a = m.greedy(&tokens, start, extra).map(|a| (a, Vec::new()));
                            tracing::info!(
                                rows = tokens.len(),
                                steps = extra + 1,
                                pos = start,
                                idle_ms = (t0 - idle_from).as_secs_f64() * 1e3,
                                ms = t0.elapsed().as_secs_f64() * 1e3,
                                "spatial wafer: split token loop launch"
                            );
                            a
                        }
                        Job::Forward {
                            tokens,
                            start,
                            which,
                        } => match which {
                            Logits::None => m.forward(&tokens, start, false).map(|(a, _)| (a, Vec::new())),
                            Logits::Last => m
                                .forward(&tokens, start, true)
                                .map(|(a, l)| (a, l.into_iter().collect())),
                            Logits::All => (|| {
                                let (mut am, mut rows) = (Vec::new(), Vec::new());
                                for (i, &t) in tokens.iter().enumerate() {
                                    let (a, l) = m.forward(&[t], start + i, true)?;
                                    am.extend(a);
                                    rows.extend(l);
                                }
                                Ok((am, rows))
                            })(),
                        },
                        _ => Err("the role-split program runs one sequence: no batched, paged or VM jobs".into()),
                    };
                    idle_from = Instant::now();
                    tracing::info!(launches = m.launches, steps = m.steps, "spatial wafer: totals");
                    if ans_tx.send(answer).is_err() {
                        break;
                    }
                }
            })
            .map_err(|e| format!("spawning the wafer thread: {e}"))?;
        let (cap, positions, hidden) = ready_rx
            .recv()
            .map_err(|_| "the wafer thread died while loading".to_string())??;
        Ok(Wafer {
            jobs: Some(job_tx),
            answers: ans_rx,
            batch_answers: mpsc::channel().1,
            paged: None,
            cap,
            launch_lanes: 1,
            logit_lanes: 1,
            positions,
            vm: None,
            vm_b: None,
            page_bytes: None,
            vm_answers: mpsc::channel().1,
            thread: Some(thread),
            vocab,
            hidden,
            recurrent: false,
            group: 1,
            row_order: None,
            split: true,
            split_loop,
            posted: post_rx,
        })
    }

    /// Pages PIE's way: `page_size` positions a page, a whole number of
    /// the program's (which then divide its blocks a row). Sets
    /// [`Wafer::group`] and turns `paged` and `page_bytes` into PIE's
    /// view; a page size the program's pages do not tile is left alone
    /// (the load's check then refuses it).
    pub fn page_as(&mut self, page_size: usize) {
        let Some(shape) = self.paged else { return };
        if self.group != 1 || shape.page_size == 0 || page_size <= shape.page_size {
            return;
        }
        let g = page_size / shape.page_size;
        if g * shape.page_size != page_size || !shape.blocks.is_multiple_of(g) {
            return;
        }
        self.group = g as u32;
        self.paged = Some(super::paged::Shape {
            page_size,
            blocks: shape.blocks / g,
            ..shape
        });
        self.page_bytes = self.page_bytes.map(|b| b * g);
    }

    /// PIE page `p`'s kernel page `o` (of `group`), relabelled by the page
    /// order when there is one.
    fn kpage(&self, p: u32, o: u32) -> u32 {
        let k = p * self.group + o;
        self.row_order.as_ref().map_or(k, |r| r.map(k))
    }

    /// PIE page ids as the program's (each `group` kernel pages, in order).
    fn kernel_pages(&self, pages: &[u32]) -> Vec<u32> {
        let g = self.group;
        pages
            .iter()
            .flat_map(|&p| (0..g).map(move |o| self.kpage(p, o)))
            .collect()
    }

    fn kernel_lanes(
        &self,
        mut lanes: Vec<super::serve2d::BatchLane>,
    ) -> Vec<super::serve2d::BatchLane> {
        if self.group > 1 || self.row_order.is_some() {
            for l in &mut lanes {
                l.pages = self.kernel_pages(&l.pages);
            }
        }
        lanes
    }

    /// [`WaferModel::fire_batch`] on the wafer's thread.
    pub fn fire_batch(
        &mut self,
        lanes: Vec<super::serve2d::BatchLane>,
        steps: usize,
        logits: bool,
    ) -> BatchAnswer {
        let _busy = Busy::start();
        let lanes = self.kernel_lanes(lanes);
        self.jobs
            .as_ref()
            .ok_or("the wafer is closed")?
            .send(Job::Batch {
                lanes,
                steps,
                logits,
            })
            .map_err(|_| "the wafer thread went away".to_string())?;
        self.batch_answers
            .recv()
            .map_err(|_| "the wafer thread went away".to_string())?
    }

    /// [`WaferModel::fire_batch`] of argmax only, posted: the wafer's
    /// thread runs it after every job before, answers on `reply`, then
    /// calls `landed`. Jobs posted after it run after it.
    pub fn post_batch(
        &mut self,
        lanes: Vec<super::serve2d::BatchLane>,
        steps: usize,
        reply: mpsc::Sender<BatchAnswer>,
        landed: Box<dyn FnOnce() + Send>,
    ) -> R<()> {
        let lanes = self.kernel_lanes(lanes);
        self.jobs
            .as_ref()
            .ok_or("the wafer is closed")?
            .send(Job::Posted {
                lanes,
                steps,
                reply,
                landed,
            })
            .map_err(|_| "the wafer thread went away".to_string())
    }

    /// Why the head's VM cannot take `table`, if it cannot (as
    /// [`WaferModel::vm_refuses`]).
    pub fn vm_refuses(&self, table: &super::guest4::Table) -> Option<String> {
        let Some((lay, ring)) = &self.vm else {
            return Some("the wafer program has no VM".into());
        };
        if !table.per_step.is_empty() {
            return Some(
                "the host writes some of its homes every step; this path writes no step input"
                    .into(),
            );
        }
        if ring.lanes != 1 {
            return Some(format!(
                "the VM time-multiplexes {} lanes; this path drives one",
                ring.lanes
            ));
        }
        lay.refuses_head(table)
            .or_else(|| ring.refuses_outputs(&table.outputs))
    }

    /// [`WaferModel::vm_load`] on the wafer's thread.
    pub fn vm_load(
        &mut self,
        program: std::sync::Arc<super::guest4::lane::Program>,
        seeds: Vec<super::guest4::wafer::Seed>,
    ) -> R<()> {
        let _busy = Busy::start();
        self.jobs
            .as_ref()
            .ok_or("the wafer is closed")?
            .send(Job::VmLoad { program, seeds })
            .map_err(|_| "the wafer thread went away".to_string())?;
        self.vm_answers
            .recv()
            .map_err(|_| "the wafer thread went away".to_string())?
            .map(drop)
    }

    /// [`WaferModel::vm_load_lanes`] on the wafer's thread.
    pub fn vm_load_lanes(
        &mut self,
        program: std::sync::Arc<super::guest4::lane::Program>,
    ) -> R<()> {
        let _busy = Busy::start();
        self.vm_job(Job::VmLoadLanes { program }).map(drop)
    }

    /// [`WaferModel::vm_seed_bank`] on the wafer's thread.
    pub fn vm_seed_bank(&mut self, lane: usize, seeds: Vec<super::guest4::wafer::Seed>) -> R<()> {
        let _busy = Busy::start();
        self.vm_job(Job::VmSeedBank { lane, seeds }).map(drop)
    }

    /// [`WaferModel::vm_fire`] on the wafer's thread.
    pub fn vm_fire(
        &mut self,
        lanes: Vec<super::serve2d::BatchLane>,
        sampled: Vec<bool>,
        inputs: Vec<StepInput>,
    ) -> R<Vec<VmStep>> {
        let _busy = Busy::start();
        let lanes = self.kernel_lanes(lanes);
        self.vm_job(Job::VmFire {
            lanes,
            sampled,
            inputs,
        })
    }

    /// [`WaferModel::vm_place_lanes`] on the wafer's thread.
    pub fn vm_place_lanes(
        &mut self,
        set: usize,
        programs: Vec<Option<super::guest4::slots::LaneProgram>>,
        seeds: Vec<Option<Vec<super::guest4::wafer::Seed>>>,
        parked: Vec<Option<super::guest4::slots::Parked>>,
    ) -> R<Vec<Option<super::guest4::slots::Parked>>> {
        let _busy = Busy::start();
        let (tx, rx) = mpsc::channel();
        self.vm_job(Job::VmPlaceLanes {
            set,
            programs,
            seeds,
            parked,
            left: tx,
        })?;
        rx.recv()
            .map_err(|_| "the wafer thread went away".to_string())
    }

    fn vm_job(&mut self, job: Job) -> R<Vec<VmStep>> {
        self.jobs
            .as_ref()
            .ok_or("the wafer is closed")?
            .send(job)
            .map_err(|_| "the wafer thread went away".to_string())?;
        self.vm_answers
            .recv()
            .map_err(|_| "the wafer thread went away".to_string())?
    }

    /// [`WaferModel::vm_step`] on the wafer's thread.
    pub fn vm_step(&mut self, token: u32, start: usize) -> R<VmStep> {
        let _busy = Busy::start();
        self.jobs
            .as_ref()
            .ok_or("the wafer is closed")?
            .send(Job::VmStep { token, start })
            .map_err(|_| "the wafer thread went away".to_string())?;
        self.vm_answers
            .recv()
            .map_err(|_| "the wafer thread went away".to_string())??
            .pop()
            .ok_or_else(|| "the VM answered no step".to_string())
    }

    /// [`WaferModel::copy_pages`] on the wafer's thread.
    /// [`WaferModel::page_out`] on the wafer's thread.
    pub fn page_out(&mut self, pages: Vec<u32>) -> R<Vec<Vec<u32>>> {
        let _busy = Busy::start();
        let g = self.group as usize;
        let pages = self.kernel_pages(&pages);
        self.jobs
            .as_ref()
            .ok_or("the wafer is closed")?
            .send(Job::PageOut { pages })
            .map_err(|_| "the wafer thread went away".to_string())?;
        let words = self
            .batch_answers
            .recv()
            .map_err(|_| "the wafer thread went away".to_string())?
            .map(|(words, _)| words)?;
        // A PIE page's words: its kernel pages' in order.
        Ok(words.chunks(g.max(1)).map(|c| c.concat()).collect())
    }

    /// [`WaferModel::page_in`] on the wafer's thread.
    pub fn page_in(&mut self, pages: Vec<(u32, Vec<u32>)>) -> R<()> {
        let _busy = Busy::start();
        let g = self.group;
        let pages = if g > 1 || self.row_order.is_some() {
            let mut out = Vec::new();
            for (p, words) in pages {
                let n = words.len() / g as usize;
                if n * g as usize != words.len() {
                    return Err(format!(
                        "page {p}'s host copy is {} words, not {g} kernel pages' worth",
                        words.len()
                    ));
                }
                for (o, c) in words.chunks(n.max(1)).enumerate() {
                    out.push((self.kpage(p, o as u32), c.to_vec()));
                }
            }
            out
        } else {
            pages
        };
        self.jobs
            .as_ref()
            .ok_or("the wafer is closed")?
            .send(Job::PageIn { pages })
            .map_err(|_| "the wafer thread went away".to_string())?;
        self.batch_answers
            .recv()
            .map_err(|_| "the wafer thread went away".to_string())?
            .map(drop)
    }

    /// [`WaferModel::copy_state`] on the wafer's thread, after every job
    /// before it.
    pub fn copy_state(&mut self, src: u32, dst: u32) -> R<()> {
        self.jobs
            .as_ref()
            .ok_or("the wafer is closed")?
            .send(Job::CopyState { src, dst })
            .map_err(|_| "the wafer thread went away".to_string())?;
        self.batch_answers
            .recv()
            .map_err(|_| "the wafer thread went away".to_string())?
            .map(drop)
    }

    pub fn copy_pages(&mut self, pairs: Vec<(u32, u32)>) -> R<()> {
        let _busy = Busy::start();
        let g = self.group;
        let pairs: Vec<(u32, u32)> = pairs
            .into_iter()
            .flat_map(|(s, d)| (0..g).map(move |o| (s, d, o)))
            .map(|(s, d, o)| (self.kpage(s, o), self.kpage(d, o)))
            .collect();
        self.jobs
            .as_ref()
            .ok_or("the wafer is closed")?
            .send(Job::Copy { pairs })
            .map_err(|_| "the wafer thread went away".to_string())?;
        self.batch_answers
            .recv()
            .map_err(|_| "the wafer thread went away".to_string())?
            .map(drop)
    }

    /// [`WaferModel::greedy`] on the wafer's thread.
    pub fn greedy(&mut self, tokens: &[u32], start: usize, extra: usize) -> R<Vec<u32>> {
        let _busy = Busy::start();
        self.jobs
            .as_ref()
            .ok_or("the wafer is closed")?
            .send(Job::Greedy {
                tokens: tokens.to_vec(),
                start,
                extra,
            })
            .map_err(|_| "the wafer thread went away".to_string())?;
        self.answers
            .recv()
            .map_err(|_| "the wafer thread went away".to_string())?
            .map(|(argmax, _)| argmax)
    }

    /// Posts a token loop (the role-split program): its answer comes from
    /// [`Wafer::take_posted`], in order.
    pub fn post_greedy(&mut self, tokens: &[u32], start: usize, extra: usize) -> R<()> {
        self.jobs
            .as_ref()
            .ok_or("the wafer is closed")?
            .send(Job::PostGreedy {
                tokens: tokens.to_vec(),
                start,
                extra,
            })
            .map_err(|_| "the wafer thread went away".to_string())
    }

    /// A posted loop's argmax after each row (as [`Wafer::greedy`]'s):
    /// waiting for it, or `None` if it has not landed.
    pub fn take_posted(&mut self, wait: bool) -> Option<R<Vec<u32>>> {
        let got = if wait {
            let _busy = Busy::start();
            self.posted
                .recv()
                .map_err(|_| "the wafer thread went away".to_string())
        } else {
            match self.posted.try_recv() {
                Ok(a) => Ok(a),
                Err(mpsc::TryRecvError::Empty) => return None,
                Err(_) => Err("the wafer thread went away".to_string()),
            }
        };
        Some(got.and_then(|a| a.map(|(argmax, _)| argmax)))
    }

    pub fn forward(&mut self, tokens: &[u32], start: usize, which: Logits) -> Answer {
        let _busy = Busy::start();
        self.jobs
            .as_ref()
            .ok_or("the wafer is closed")?
            .send(Job::Forward {
                tokens: tokens.to_vec(),
                start,
                which,
            })
            .map_err(|_| "the wafer thread went away".to_string())?;
        self.answers
            .recv()
            .map_err(|_| "the wafer thread went away".to_string())?
    }
}

impl Drop for Wafer {
    fn drop(&mut self) {
        // Closing the job channel ends the thread, which stops the program.
        self.jobs = None;
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An HV build (PS 1, NB 16, E 5 on 152 rows) paged PIE's way (16):
    /// 152 PIE pages of 16, each 16 kernel pages in order; a 531-position
    /// lane on PIE pages 0..34 makes a kernel table (at most 4 a row).
    #[test]
    fn hv_pages_as_pie_pages() {
        let mut w = Wafer {
            jobs: None,
            answers: mpsc::channel().1,
            batch_answers: mpsc::channel().1,
            vm_answers: mpsc::channel().1,
            paged: Some(super::super::paged::Shape {
                lanes: 1,
                rows: 152,
                page_size: 1,
                blocks: 16,
                entries: 5,
            }),
            cap: 128,
            launch_lanes: 1,
            logit_lanes: 1,
            positions: 760,
            vm: None,
            vm_b: None,
            page_bytes: Some(100),
            thread: None,
            vocab: 0,
            hidden: 0,
            recurrent: false,
            group: 1,
            row_order: None,
            split: false,
            split_loop: None,
            posted: mpsc::channel().1,
        };
        let kernel = w.paged.unwrap();
        w.page_as(16);
        let pie = w.paged.unwrap();
        assert_eq!(
            (w.group, pie.page_size, pie.pages(), w.page_bytes),
            (16, 16, 152, Some(1600))
        );
        assert_eq!(w.kernel_pages(&[2, 0])[..3], [32, 33, 34]);
        assert_eq!(w.kernel_pages(&[2, 0])[16..18], [0, 1]);
        let lane = super::super::paged::Lane {
            pos: 530,
            slot: 530,
            pages: w.kernel_pages(&(0..34).collect::<Vec<u32>>()),
            visible: None,
        };
        assert!(super::super::paged::table(&kernel, &[lane], 1).is_ok());
        // A second call, or a page size the kernel's do not tile, changes nothing.
        w.page_as(16);
        assert_eq!(w.group, 16);
        // The page order relabels the expanded kernel pages (row 104 first).
        w.row_order = Some(super::super::paged::RowOrder::parse("ar:208:16", &kernel).unwrap());
        assert_eq!(w.kernel_pages(&[0])[..3], [104, 103, 105]);
        assert_eq!(
            w.kernel_lanes(vec![super::super::serve2d::BatchLane {
                pos: 0,
                slot: 0,
                pages: vec![0],
                visible: None,
                forced: vec![],
                state: None,
            }])[0]
                .pages[..2],
            [104, 103]
        );
    }

    /// `PLACE_CKPT=<dir> PLACE_PARAMS=<K:V,...>`: a real checkpoint
    /// validated and placed on the program of those params on a recording
    /// device (no chip, no simulator: every write's symbol and size checked
    /// against nothing but the host's own account). The chip's load errors
    /// before the first launch show here (e.g. Gemma-12B's tied head).
    #[test]
    #[ignore]
    fn place_on_a_recorder() {
        let dir = std::env::var("PLACE_CKPT").expect("PLACE_CKPT");
        let params = super::super::program::parse_params(
            &std::env::var("PLACE_PARAMS").expect("PLACE_PARAMS"),
        );
        let design = ProgramSpec::from_params(ProgramSpec::infer_design(&params).unwrap(), &params)
            .expect("the program");
        let ckpt = Ckpt::open(Path::new(&dir)).expect("the checkpoint");
        let theta = check_weights(&design, &ckpt, None).expect("the checkpoint fits");
        let mut rec = super::super::device::Recorder::default().forgetful();
        for c in design.components() {
            c.routes(&design, &mut rec).expect("routes");
        }
        let w = super::super::component::Weights {
            ckpt: &ckpt,
            rope: &theta,
        };
        for c in design.components() {
            let t = Instant::now();
            c.place(&design, &w, &mut rec)
                .unwrap_or_else(|e| panic!("{}: {e}", c.name()));
            eprintln!("placed {} in {:.1}s", c.name(), t.elapsed().as_secs_f32());
        }
        eprintln!("{} calls", rec.take().len());
    }

    #[test]
    fn bf16_pages_pack_two_a_word_and_back() {
        let cells: Vec<u32> = (0..7).map(|i| 0x3f80 + i).collect();
        let words = pack16(&cells);
        assert_eq!(words.len(), 4);
        assert_eq!(words[0], 0x3f80 | 0x3f81 << 16);
        assert_eq!(unpack16(&words, cells.len()), cells);
    }
}
