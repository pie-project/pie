//! The Neural Engine's state for one load, as sites: the dense MLPs, and
//! each shape of wide projection. A site owns its surfaces viewed as GPU
//! tensors, its layers' banks cut for the split, which layer's weights are
//! staged, and its program, compiled off the serving thread. All sites
//! share one hand-off and one status flag.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use kernels_metal::ane::ffn::{self, Ffn, MAX_ROWS, MIN_ROWS, Memory, Shape};
use kernels_metal::ane::{Element, Handoff, Surface, linear};
use kernels_metal::linear::ane::{self as kernels, Projection, Set, Shared, Source, Target};
use kernels_metal::{Bank, Ctx, Error, Tensor};
use poem_ir::Dtype;

use super::banks::{self, Mlp, Split};
use super::{Columns, Plan};
use crate::device::{Buffer, Context, Handles, Pipelines};
use crate::encode::Sink;
use crate::error::{Fault, Result};

struct Layer {
    split: Split,
    source: Source,
}

/// What a dispatch's weight value names.
#[derive(Clone, Copy)]
enum Key {
    Mlp(u32),
    Projection(usize, u32),
}

type Compiled<T> = Arc<OnceLock<std::result::Result<T, String>>>;

struct MlpSite {
    shape: Shape,
    layers: BTreeMap<u32, Layer>,
    shared: Shared,
    sets: [Set; 2],
    up_rows: Tensor,
    rotated: Tensor,
    ffn: Compiled<Ffn>,
    /// The layer whose weights are in the set of its parity.
    staged: AtomicU32,
    _memory: Arc<Memory>,
}

struct ProjectionLayer {
    projection: Projection,
    /// The GPU's leading rows.
    view: Bank,
}

struct ProjectionSite {
    shape: linear::Shape,
    layers: BTreeMap<u32, ProjectionLayer>,
    shared: Shared,
    sets: [Vec<Target>; 2],
    rotated: Tensor,
    program: Compiled<linear::Linear>,
    staged: AtomicU32,
    _memory: Arc<linear::Memory>,
}

/// What a step's completion handler asks: whether any Neural Engine
/// evaluation or join of the step failed.
pub type Verdict = Box<dyn Fn() -> Option<String> + Send>;

pub struct Private {
    handoff: Handoff,
    mlp: Option<MlpSite>,
    projections: Vec<ProjectionSite>,
    /// By the weight value a dispatch names.
    keys: BTreeMap<u32, Key>,
    /// The join's non-finite flag, read on the host after each step.
    status: usize,
    /// How many MLPs have been handed to the Neural Engine.
    splits: AtomicU64,
    /// How many projections have.
    column_splits: AtomicU64,
    _buffers: Vec<Buffer>,
}

fn view(
    device: &Context,
    handles: &Handles,
    surface: &Surface,
    rows: u32,
    width: u32,
    keep: &mut Vec<Buffer>,
) -> Result<Tensor> {
    let at = std::ptr::NonNull::new(surface.base()).ok_or(Fault::Device {
        call: "IOSurfaceGetBaseAddress",
        why: "a surface has no base address".into(),
    })?;
    // SAFETY: the surface is page-aligned shared memory that the site's
    // memory keeps alive as long as the buffer.
    let buffer = unsafe { Buffer::foreign(device, at, surface.bytes()) }?;
    let dtype = match surface.element() {
        Element::Int8 => Dtype::I8,
        Element::Fp16 => Dtype::F16,
    };
    let tensor = Tensor::new(
        handles.bind(&buffer, 0, surface.bytes())?,
        rows,
        width,
        dtype,
    );
    keep.push(buffer);
    Ok(tensor)
}

fn target(
    device: &Context,
    handles: &Handles,
    weights: &Surface,
    scale: &Surface,
    rows: u32,
    keep: &mut Vec<Buffer>,
) -> Result<Target> {
    Ok(Target {
        weights: view(device, handles, weights, rows, weights.stride(), keep)?,
        stride: weights.stride(),
        scale: view(device, handles, scale, rows, 1, keep)?,
        scale_stride: scale.stride() / 2,
    })
}

/// The share of a split the Neural Engine takes, in percent: `PIE_ANE_PERCENT`,
/// else 47. The two engines must finish together: on an M-series box
/// without MPP serving Qwen3.6-27B at 640 rows, 16 of the MLP's 34 units
/// was the floor of a curve that rose on both sides (14: +5%, 18: +8%,
/// 8: +19%), and the projections split at the same share land a 640-row
/// prefill in 3.44 s against the GPU's 5.51 s alone.
fn percent() -> u32 {
    std::env::var("PIE_ANE_PERCENT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(47)
}

/// How many 512-wide units of `all` the Neural Engine takes.
fn share(all: u32) -> u32 {
    ((all * percent() + 50) / 100).clamp(1, all.saturating_sub(1).max(1))
}

/// The MLP's units: eight where the GPU reads prompts through MPP, else
/// the share.
fn mlp_units(intermediate: u32) -> u32 {
    let all = intermediate / ffn::UNIT;
    if kernels_metal::tuning::current().qmm_mpp {
        8.clamp(1, all.saturating_sub(1).max(1))
    } else {
        share(all)
    }
}

fn fault(why: String) -> Fault {
    Fault::Device { call: "ane", why }
}

/// The tensors every site shares: the rotation signs and the status flag.
struct Common {
    signs: Tensor,
    status: Tensor,
}

fn mlp_site(
    device: &Context,
    handles: &Handles,
    common: &Common,
    mlps: &[Mlp],
    keep: &mut Vec<Buffer>,
    keys: &mut BTreeMap<u32, Key>,
) -> Result<Option<MlpSite>> {
    let (hidden, intermediate) = (mlps[0].gate_up.codes.width, mlps[0].intermediate);
    let Ok(shape) = Shape::new(hidden, intermediate, mlp_units(intermediate)) else {
        return Ok(None);
    };
    let memory = Arc::new(Memory::new(&shape).map_err(fault)?);
    let inputs = memory
        .inputs
        .iter()
        .map(|s| {
            Ok((
                view(device, handles, s, shape.segment, MAX_ROWS, keep)?,
                s.stride(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let token_scale = view(device, handles, &memory.token_scale, 1, MAX_ROWS, keep)?;
    let partial = (
        view(device, handles, &memory.partial, hidden + 1, MAX_ROWS, keep)?,
        memory.partial.stride() / 2,
    );
    let set = |w: &ffn::Weights, keep: &mut Vec<Buffer>| -> Result<Set> {
        Ok(Set {
            gate: w
                .gate
                .iter()
                .map(|s| target(device, handles, s, &w.gate_scale, shape.ane, keep))
                .collect::<Result<_>>()?,
            up: w
                .up
                .iter()
                .map(|s| target(device, handles, s, &w.up_scale, shape.ane, keep))
                .collect::<Result<_>>()?,
            down: w
                .down
                .iter()
                .map(|s| target(device, handles, s, &w.down_scale, hidden, keep))
                .collect::<Result<_>>()?,
        })
    };
    let sets = [set(&memory.sets[0], keep)?, set(&memory.sets[1], keep)?];
    let up_rows = banks::staging(device, handles, MAX_ROWS, shape.gpu, Dtype::Bf16, keep)?;
    let rotated = banks::staging(device, handles, MAX_ROWS, hidden, Dtype::F16, keep)?;
    let shared = Shared {
        inputs,
        token_scale,
        partial,
        signs: common.signs,
        status: common.status,
    };
    let mut layers = BTreeMap::new();
    for mlp in mlps
        .iter()
        .filter(|m| m.intermediate == intermediate && m.gate_up.codes.width == hidden)
    {
        keys.insert(mlp.key.0, Key::Mlp(mlp.layer));
        let scales = [
            banks::staging(device, handles, shape.ane, 1, Dtype::F16, keep)?,
            banks::staging(device, handles, shape.ane, 1, Dtype::F16, keep)?,
            banks::staging(device, handles, hidden, 1, Dtype::F16, keep)?,
        ];
        let split = banks::split(handles, mlp, shape.gpu)?;
        layers.insert(
            mlp.layer,
            Layer {
                split,
                source: Source {
                    gate_up: mlp.gate_up,
                    down: mlp.down,
                    scales,
                },
            },
        );
    }
    eprintln!(
        "PIE_ANE: {} MLP layers split, Neural Engine takes {} of {intermediate} columns",
        layers.len(),
        shape.ane
    );
    Ok(Some(MlpSite {
        shape,
        layers,
        shared,
        sets,
        up_rows,
        rotated,
        ffn: Arc::new(OnceLock::new()),
        staged: AtomicU32::new(u32::MAX),
        _memory: memory,
    }))
}

#[allow(clippy::too_many_arguments)]
fn projection_site(
    device: &Context,
    handles: &Handles,
    common: &Common,
    index: usize,
    (k, n): (u32, u32),
    projections: &[banks::Projection],
    keep: &mut Vec<Buffer>,
    keys: &mut BTreeMap<u32, Key>,
) -> Result<Option<ProjectionSite>> {
    let Ok(shape) = linear::Shape::new(k, n, share(n / ffn::UNIT)) else {
        return Ok(None);
    };
    let memory = Arc::new(linear::Memory::new(&shape).map_err(fault)?);
    let inputs = memory
        .inputs
        .iter()
        .map(|s| {
            Ok((
                view(device, handles, s, shape.segment, MAX_ROWS, keep)?,
                s.stride(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let token_scale = view(device, handles, &memory.token_scale, 1, MAX_ROWS, keep)?;
    let partial = (
        view(device, handles, &memory.partial, shape.ane, MAX_ROWS, keep)?,
        memory.partial.stride() / 2,
    );
    let set = |w: &linear::Weights, keep: &mut Vec<Buffer>| -> Result<Vec<Target>> {
        w.w.iter()
            .map(|s| target(device, handles, s, &w.scale, shape.ane, keep))
            .collect()
    };
    let sets = [set(&memory.sets[0], keep)?, set(&memory.sets[1], keep)?];
    let rotated = banks::staging(device, handles, MAX_ROWS, k, Dtype::F16, keep)?;
    let shared = Shared {
        inputs,
        token_scale,
        partial,
        signs: common.signs,
        status: common.status,
    };
    let mut layers = BTreeMap::new();
    for p in projections {
        keys.insert(p.key.0, Key::Projection(index, p.layer));
        let scale = banks::staging(device, handles, shape.ane, 1, Dtype::F16, keep)?;
        layers.insert(
            p.layer,
            ProjectionLayer {
                projection: Projection {
                    bank: p.bank,
                    scale,
                    first: shape.keep,
                    rows: shape.ane,
                    segment: shape.segment,
                },
                view: banks::rows_of(handles, &p.bank, 0, shape.keep)?,
            },
        );
    }
    eprintln!(
        "PIE_ANE: {} projections of {k} -> {n} split, Neural Engine takes {} columns",
        layers.len(),
        shape.ane
    );
    Ok(Some(ProjectionSite {
        shape,
        layers,
        shared,
        sets,
        rotated,
        program: Arc::new(OnceLock::new()),
        staged: AtomicU32::new(u32::MAX),
        _memory: memory,
    }))
}

/// One program to compile, off the serving thread.
enum Job {
    Mlp(Compiled<Ffn>, Arc<Memory>, Shape),
    Linear(Compiled<linear::Linear>, Arc<linear::Memory>, linear::Shape),
}

impl Job {
    fn run(self, cache: &std::path::Path) {
        let started = std::time::Instant::now();
        match self {
            Job::Mlp(slot, memory, shape) => {
                let result = Ffn::compile(&shape, &memory, cache);
                match &result {
                    Ok(_) => eprintln!(
                        "PIE_ANE: Neural Engine MLP program ready in {:.1}s",
                        started.elapsed().as_secs_f64()
                    ),
                    Err(why) => eprintln!(
                        "PIE_ANE: the Neural Engine MLP program did not compile ({why}); the GPU runs the MLP alone"
                    ),
                }
                let _ = slot.set(result);
            }
            Job::Linear(slot, memory, shape) => {
                let result = linear::Linear::compile(&shape, &memory, cache);
                match &result {
                    Ok(_) => eprintln!(
                        "PIE_ANE: Neural Engine {} -> {} program ready in {:.1}s",
                        shape.k,
                        shape.n,
                        started.elapsed().as_secs_f64()
                    ),
                    Err(why) => eprintln!(
                        "PIE_ANE: the Neural Engine {} -> {} program did not compile ({why}); the GPU runs it alone",
                        shape.k, shape.n
                    ),
                }
                let _ = slot.set(result);
            }
        }
    }
}

pub fn load(
    device: &Context,
    handles: &Handles,
    mlps: &[Mlp],
    projections: &[((u32, u32), Vec<banks::Projection>)],
) -> Result<Option<Private>> {
    use objc2_metal::MTLDevice;
    let mut keep = Vec::new();
    let signs = banks::staging(
        device,
        handles,
        1,
        ffn::INTERMEDIATE_BLOCK,
        Dtype::F32,
        &mut keep,
    )?;
    let bytes: Vec<u8> = ffn::signs().iter().flat_map(|v| v.to_le_bytes()).collect();
    banks::host(&keep[keep.len() - 1])?[..bytes.len()].copy_from_slice(&bytes);
    let status = banks::staging(device, handles, 1, 1, Dtype::U32, &mut keep)?;
    let status_host = banks::host(&keep[keep.len() - 1])?.as_mut_ptr() as usize;
    let common = Common { signs, status };
    let mut keys = BTreeMap::new();

    let mlp = if mlps.is_empty() {
        None
    } else {
        mlp_site(device, handles, &common, mlps, &mut keep, &mut keys)?
    };
    let mut sites = Vec::new();
    for (shape, group) in projections {
        let index = sites.len();
        if let Some(site) = projection_site(
            device, handles, &common, index, *shape, group, &mut keep, &mut keys,
        )? {
            sites.push(site);
        }
    }
    if mlp.is_none() && sites.is_empty() {
        return Ok(None);
    }

    // The row scales, once, on the GPU.
    let frame = device.frame()?;
    let pipelines = Pipelines::new();
    let sink = Sink::new(device, &frame, &pipelines, handles);
    if let Some(site) = &mlp {
        for layer in site.layers.values() {
            kernels::row_scales(&sink, &site.shape, &layer.source, signs)
                .map_err(|e| fault(e.to_string()))?;
        }
    }
    for site in &sites {
        for layer in site.layers.values() {
            kernels::projection_scale(&sink, &layer.projection, signs)
                .map_err(|e| fault(e.to_string()))?;
        }
    }
    frame.commit()?;

    let event = || {
        device
            .device()
            .newSharedEvent()
            .ok_or(fault("no shared event".into()))
    };
    let handoff = Handoff::new(event()?, event()?);

    // Every program compiles on one thread, in order, so serving starts
    // while they do and the compiler service sees one at a time.
    let mut jobs = Vec::new();
    if let Some(site) = &mlp {
        jobs.push(Job::Mlp(
            site.ffn.clone(),
            site._memory.clone(),
            site.shape.clone(),
        ));
    }
    for site in &sites {
        jobs.push(Job::Linear(
            site.program.clone(),
            site._memory.clone(),
            site.shape.clone(),
        ));
    }
    std::thread::Builder::new()
        .name("pie-ane-compile".into())
        .spawn(move || {
            let cache = kernels_metal::ane::cache_directory();
            for job in jobs {
                job.run(&cache);
            }
        })
        .map_err(|e| fault(e.to_string()))?;
    Ok(Some(Private {
        handoff,
        mlp,
        projections: sites,
        keys,
        status: status_host,
        splits: AtomicU64::new(0),
        column_splits: AtomicU64::new(0),
        _buffers: keep,
    }))
}

impl Private {
    /// The hand-off a live encode fences on.
    pub fn handoff(&self) -> &Handoff {
        &self.handoff
    }

    /// How the programs' compiles went: `None` while any is still running,
    /// `Some(Ok)` once every one may be planned on, `Some(Err)` with the
    /// first reason when one was refused and the GPU keeps that work.
    #[must_use]
    pub fn compiled(&self) -> Option<std::result::Result<(), String>> {
        let mut outcome = Ok(());
        if let Some(site) = &self.mlp
            && let Err(why) = site.ffn.get()?
        {
            outcome = Err(why.clone());
        }
        for site in &self.projections {
            if let Err(why) = site.program.get()?
                && outcome.is_ok()
            {
                outcome = Err(why.clone());
            }
        }
        Some(outcome)
    }

    /// How many MLPs the Neural Engine has been handed so far.
    #[must_use]
    pub fn splits(&self) -> u64 {
        self.splits.load(Ordering::Acquire)
    }

    /// How many projections it has.
    #[must_use]
    pub fn column_splits(&self) -> u64 {
        self.column_splits.load(Ordering::Acquire)
    }

    fn rows_fit(&self, rows: u32) -> bool {
        (MIN_ROWS..=MAX_ROWS).contains(&rows) && !self.handoff.retired()
    }

    /// A split for the MLP whose gate-up weight is `gate_up`, over `rows`,
    /// if the Neural Engine can take it: the rows fit a procedure, the
    /// program is compiled, the layer is one of the split ones, and nothing
    /// has retired the hand-off.
    pub fn plan(&self, gate_up: poem_ir::ValueId, rows: u32) -> Option<Plan> {
        let Some(Key::Mlp(layer)) = self.keys.get(&gate_up.0).copied() else {
            return None;
        };
        if !self.rows_fit(rows) {
            return None;
        }
        let site = self.mlp.as_ref()?;
        let evaluation = site.ffn.get()?.as_ref().ok()?.evaluation(rows)?;
        let split = site.layers.get(&layer)?.split;
        Some(Plan {
            split,
            up_rows: Tensor {
                rows,
                ..site.up_rows
            },
            rows,
            layer,
            at: self.handoff.allot(),
            evaluation,
        })
    }

    /// A column split for the projection whose weight is `w`, over `rows`,
    /// on the same terms.
    pub fn plan_columns(&self, w: poem_ir::ValueId, rows: u32) -> Option<Columns> {
        let Some(Key::Projection(index, layer)) = self.keys.get(&w.0).copied() else {
            return None;
        };
        if !self.rows_fit(rows) {
            return None;
        }
        let site = self.projections.get(index)?;
        let evaluation = site.program.get()?.as_ref().ok()?.evaluation(rows)?;
        let view = site.layers.get(&layer)?.view;
        Some(Columns {
            view,
            keep: site.shape.keep,
            rows,
            site: index,
            layer,
            at: self.handoff.allot(),
            evaluation,
        })
    }

    fn stage_mlp(
        &self,
        ctx: &Ctx<'_>,
        site: &MlpSite,
        layer: u32,
    ) -> std::result::Result<(), Error> {
        let Some(source) = site.layers.get(&layer) else {
            return Ok(());
        };
        let set = &site.sets[(layer & 1) as usize];
        kernels::stage(ctx, &site.shape, &source.source, site.shared.signs, set)?;
        site.staged.store(layer, Ordering::Release);
        Ok(())
    }

    /// Before the GPU's share of an MLP: stage this layer's weights if they
    /// are not, hand the activations over, and queue the Neural Engine
    /// behind the signal.
    pub fn before(&self, ctx: &Ctx<'_>, plan: &Plan, x: Tensor) -> std::result::Result<(), Error> {
        let site = self.mlp.as_ref().expect("a plan names a site");
        if site.staged.load(Ordering::Acquire) != plan.layer {
            self.stage_mlp(ctx, site, plan.layer)?;
        }
        let rotated = Tensor {
            rows: plan.rows,
            ..site.rotated
        };
        kernels::prepare(ctx, &site.shared, x, rotated, plan.at.ready)?;
        if let Some(Ok(ffn)) = site.ffn.get() {
            let binding = &ffn.evaluations[plan.evaluation].1[(plan.layer & 1) as usize];
            self.handoff
                .start(&ffn.program, binding, plan.at.ready, plan.at.done);
            self.splits.fetch_add(1, Ordering::AcqRel);
        }
        Ok(())
    }

    /// After the GPU's share of an MLP: stage the next layer's weights into
    /// the other set while the Neural Engine runs, then wait on it and join.
    pub fn after(&self, ctx: &Ctx<'_>, plan: &Plan, out: Tensor) -> std::result::Result<(), Error> {
        let site = self.mlp.as_ref().expect("a plan names a site");
        self.stage_mlp(ctx, site, plan.layer + 1)?;
        kernels::join(ctx, &site.shared, out, plan.at.done)
    }

    fn stage_projection(
        &self,
        ctx: &Ctx<'_>,
        site: &ProjectionSite,
        layer: u32,
    ) -> std::result::Result<(), Error> {
        let Some(source) = site.layers.get(&layer) else {
            return Ok(());
        };
        let set = &site.sets[(layer & 1) as usize];
        kernels::stage_projection(ctx, &source.projection, site.shared.signs, set)?;
        site.staged.store(layer, Ordering::Release);
        Ok(())
    }

    /// Before the GPU's columns of a projection: as [`Private::before`].
    pub fn before_columns(
        &self,
        ctx: &Ctx<'_>,
        plan: &Columns,
        x: Tensor,
    ) -> std::result::Result<(), Error> {
        let site = &self.projections[plan.site];
        if site.staged.load(Ordering::Acquire) != plan.layer {
            self.stage_projection(ctx, site, plan.layer)?;
        }
        let rotated = Tensor {
            rows: plan.rows,
            ..site.rotated
        };
        kernels::prepare(ctx, &site.shared, x, rotated, plan.at.ready)?;
        if let Some(Ok(program)) = site.program.get() {
            let binding = &program.evaluations[plan.evaluation].1[(plan.layer & 1) as usize];
            self.handoff
                .start(&program.program, binding, plan.at.ready, plan.at.done);
            self.column_splits.fetch_add(1, Ordering::AcqRel);
        }
        Ok(())
    }

    /// After the GPU's columns: stage the next layer, wait, and land the
    /// Neural Engine's columns beside them.
    pub fn after_columns(
        &self,
        ctx: &Ctx<'_>,
        plan: &Columns,
        out: Tensor,
    ) -> std::result::Result<(), Error> {
        let site = &self.projections[plan.site];
        self.stage_projection(ctx, site, plan.layer + 1)?;
        kernels::join_columns(ctx, &site.shared, out, plan.keep, plan.at.done)
    }

    /// What a step's completion handler asks: whether any Neural Engine
    /// evaluation or join of the step failed. A failure retires the split.
    pub fn verdict(&self) -> Option<Verdict> {
        let handoff = self.handoff.clone();
        let seen = handoff.failures();
        let status = self.status;
        Some(Box::new(move || {
            // SAFETY: `status` is the host address of a shared-storage u32
            // the load keeps alive; the join writes it atomically.
            let flag = unsafe { &*(status as *const std::sync::atomic::AtomicU32) };
            let bad = flag.swap(0, Ordering::AcqRel) != 0;
            if bad {
                handoff.fail("the Neural Engine's output was not finite");
            }
            (bad || handoff.failures() > seen)
                .then(|| "a Neural Engine evaluation of this step failed".to_string())
        }))
    }
}
