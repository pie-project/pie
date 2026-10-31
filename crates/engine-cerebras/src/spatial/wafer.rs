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

/// What a load places: a checkpoint's weights, or nothing (a host-protocol
/// check of a shape too large to fill).
#[derive(Clone, Copy)]
pub enum Weights<'a> {
    Place(&'a Ckpt),
    Skip,
}

/// Refuses `ckpt` where a component of `design` cannot place it; RoPE's
/// theta: `rope_theta` (the model's), else the checkpoint
/// config's, else 1e6.
fn check_weights(design: &ProgramSpec, ckpt: &Ckpt, rope_theta: Option<f64>) -> R<f64> {
    for c in design.components() {
        c.validate(design, ckpt)?;
    }
    let config_theta = ckpt.config()["rope_theta"].as_f64();
    if let (Some(theta), Some(config)) = (rope_theta, config_theta)
        && (theta - config).abs() > 1e-6 * config.abs()
    {
        tracing::warn!(
            theta,
            config,
            "spatial wafer: the checkpoint config's rope_theta differs from the model's"
        );
    }
    Ok(rope_theta.or(config_theta).unwrap_or(1.0e6))
}

impl WaferModel {
    /// [`WaferModel::open_with`] on a Hugging Face directory, RoPE's theta
    /// from its config.
    pub fn open(program: &Path, ckpt_dir: &Path, platform: Platform) -> R<WaferModel> {
        WaferModel::open_with(program, Ckpt::open(ckpt_dir)?, platform, None)
    }

    /// Loads `program` and places `ckpt`'s weights; RoPE's theta is
    /// `rope_theta` (the model's, from its trace), else the checkpoint
    /// config's (else 1e6).
    pub fn open_with(
        program: &Path,
        ckpt: Ckpt,
        platform: Platform,
        rope_theta: Option<f64>,
    ) -> R<WaferModel> {
        let design = read_design(program)?;
        // The VM's out ring's words, from the program's own account.
        let ring = design.vm_abi().map_or(0, |abi| abi.ring_words);
        let sdk = Sdk::open().map_err(|e| e.to_string())?;
        let mut rt = Runtime::new(sdk, program, platform, "WARNING").map_err(|e| e.to_string())?;
        rt.load();
        rt.run();
        WaferModel::on_device(
            Box::new(rt),
            design,
            Weights::Place(&ckpt),
            rope_theta,
            ring,
        )
    }

    /// The loaded program `design` on `dev` (loaded and running): its
    /// routes, `init`, the weights placed, the VM read, a warm-up request.
    /// `ring`: the VM's out ring's words (0 without a VM).
    pub fn on_device(
        mut rt: Box<dyn Device>,
        design: ProgramSpec,
        weights: Weights,
        rope_theta: Option<f64>,
        ring: usize,
    ) -> R<WaferModel> {
        let rope_theta = match weights {
            Weights::Place(ckpt) => Some(check_weights(&design, ckpt, rope_theta)?),
            Weights::Skip => rope_theta,
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
        if let (Weights::Place(ckpt), Some(rope_theta)) = (weights, rope_theta) {
            let w = super::component::Weights { ckpt, rope_theta };
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
        super::serve2d::fire(
            grid,
            &mut *self.rt,
            lanes,
            steps,
            want_logits,
            &mut self.vm_written,
        )
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
        let bf16 = grid.layers.kv_bytes() == 2;
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
        let bf16 = grid.layers.kv_bytes() == 2;
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
}

impl Wafer {
    /// Loads `program` (the compiler's output directory) on `platform` and
    /// places `ckpt` (RoPE's theta `rope_theta`, else the checkpoint
    /// config's); returns once the model answers.
    pub fn open(
        program: PathBuf,
        ckpt: Ckpt,
        platform: Platform,
        rope_theta: Option<f64>,
    ) -> R<Wafer> {
        let (job_tx, job_rx) = mpsc::channel::<Job>();
        let (ans_tx, ans_rx) = mpsc::channel::<Answer>();
        let (batch_tx, batch_rx) = mpsc::channel::<BatchAnswer>();
        let (vm_tx, vm_rx) = mpsc::channel::<R<Vec<VmStep>>>();
        let (ready_tx, ready_rx) = mpsc::channel::<R<Wafer>>();
        let thread = std::thread::Builder::new()
            .name("spatial-wafer".into())
            .spawn(move || {
                let mut model = match WaferModel::open_with(&program, ckpt, platform, rope_theta) {
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
                        }));
                        m
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                for job in job_rx {
                    let answer = match job {
                        Job::Batch {
                            lanes,
                            steps,
                            logits,
                        } => {
                            let reply = model.fire_batch(&lanes, steps, logits);
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
                            let _ = reply.send(model.fire_batch(&lanes, steps, false));
                            landed();
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
                            let reply = model.page_out(&pages).map(|words| (words, Vec::new()));
                            if batch_tx.send(reply).is_err() {
                                break;
                            }
                            continue;
                        }
                        Job::PageIn { pages } => {
                            let reply = model.page_in(&pages).map(|()| (Vec::new(), Vec::new()));
                            if batch_tx.send(reply).is_err() {
                                break;
                            }
                            continue;
                        }
                        Job::Copy { pairs } => {
                            let reply = model.copy_pages(&pairs).map(|()| (Vec::new(), Vec::new()));
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
                    };
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
        let facts = ready_rx
            .recv()
            .map_err(|_| "the wafer thread died while loading".to_string())??;
        Ok(Wafer {
            jobs: Some(job_tx),
            answers: ans_rx,
            batch_answers: batch_rx,
            vm_answers: vm_rx,
            thread: Some(thread),
            ..facts
        })
    }

    /// [`WaferModel::fire_batch`] on the wafer's thread.
    pub fn fire_batch(
        &mut self,
        lanes: Vec<super::serve2d::BatchLane>,
        steps: usize,
        logits: bool,
    ) -> BatchAnswer {
        let _busy = Busy::start();
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
        self.jobs
            .as_ref()
            .ok_or("the wafer is closed")?
            .send(Job::PageOut { pages })
            .map_err(|_| "the wafer thread went away".to_string())?;
        self.batch_answers
            .recv()
            .map_err(|_| "the wafer thread went away".to_string())?
            .map(|(words, _)| words)
    }

    /// [`WaferModel::page_in`] on the wafer's thread.
    pub fn page_in(&mut self, pages: Vec<(u32, Vec<u32>)>) -> R<()> {
        let _busy = Busy::start();
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

    pub fn copy_pages(&mut self, pairs: Vec<(u32, u32)>) -> R<()> {
        let _busy = Busy::start();
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

    #[test]
    fn bf16_pages_pack_two_a_word_and_back() {
        let cells: Vec<u32> = (0..7).map(|i| 0x3f80 + i).collect();
        let words = pack16(&cells);
        assert_eq!(words.len(), 4);
        assert_eq!(words[0], 0x3f80 | 0x3f81 << 16);
        assert_eq!(unpack16(&words, cells.len()), cells);
    }
}
