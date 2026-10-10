//! The Neural Engine's state for one load: its surfaces viewed as GPU
//! tensors, each layer's banks cut for the split, which layer's weights are
//! staged, and the program, compiled off the serving thread.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use kernels_metal::ane::ffn::{self, Ffn, MAX_ROWS, MIN_ROWS, Memory, Shape};
use kernels_metal::ane::{Element, Handoff, Surface};
use kernels_metal::linear::ane::{self as kernels, Set, Shared, Source, Target};
use kernels_metal::{Ctx, Error, Tensor};
use poem_ir::Dtype;

use super::Plan;
use super::banks::{self, Mlp, Split};
use crate::device::{Buffer, Context, Handles, Pipelines};
use crate::encode::Sink;
use crate::error::{Fault, Result};

struct Layer {
    split: Split,
    source: Source,
}

pub struct Private {
    handoff: Handoff,
    ffn: Arc<OnceLock<std::result::Result<Ffn, String>>>,
    shape: Shape,
    /// By layer index.
    layers: BTreeMap<u32, Layer>,
    /// Layer index by the op's `gate_up` value.
    ordinals: BTreeMap<u32, u32>,
    shared: Shared,
    sets: [Set; 2],
    up_rows: Tensor,
    rotated: Tensor,
    /// The join's non-finite flag, read on the host after each step.
    status: usize,
    /// The layer whose weights are in the set of its parity.
    staged: AtomicU32,
    /// How many MLPs have been handed to the Neural Engine.
    splits: AtomicU64,
    _buffers: Vec<Buffer>,
    _memory: Arc<Memory>,
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
    // SAFETY: the surface is page-aligned shared memory that `_memory`
    // keeps alive as long as the buffer.
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

/// How many 512-channel units the Neural Engine takes: `PIE_ANE_UNITS`,
/// else eight where the GPU reads prompts through MPP, else 47 percent.
/// The two engines must finish together: on an M-series box without MPP
/// serving Qwen3.6-27B at 640 rows, 16 of 34 units was the floor of a
/// curve that rose on both sides (14: +5%, 18: +8%, 8: +19%).
fn units(intermediate: u32) -> u32 {
    let all = intermediate / ffn::UNIT;
    kernels_metal::ane::units()
        .unwrap_or(if kernels_metal::tuning::current().qmm_mpp {
            8
        } else {
            (all * 47 + 50) / 100
        })
        .clamp(1, all.saturating_sub(1).max(1))
}

pub fn load(device: &Context, handles: &Handles, mlps: &[Mlp]) -> Result<Option<Private>> {
    use objc2_metal::MTLDevice;
    let (hidden, intermediate) = (mlps[0].gate_up.codes.width, mlps[0].intermediate);
    let Ok(shape) = Shape::new(hidden, intermediate, units(intermediate)) else {
        return Ok(None);
    };
    let fault = |why: String| Fault::Device { call: "ane", why };
    let memory = Arc::new(Memory::new(&shape).map_err(fault)?);
    let mut keep = Vec::new();
    let inputs = memory
        .inputs
        .iter()
        .map(|s| {
            Ok((
                view(device, handles, s, shape.segment, MAX_ROWS, &mut keep)?,
                s.stride(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let token_scale = view(device, handles, &memory.token_scale, 1, MAX_ROWS, &mut keep)?;
    let partial = (
        view(
            device,
            handles,
            &memory.partial,
            hidden + 1,
            MAX_ROWS,
            &mut keep,
        )?,
        memory.partial.stride() / 2,
    );
    let mut set = |w: &ffn::Weights| -> Result<Set> {
        Ok(Set {
            gate: w
                .gate
                .iter()
                .map(|s| target(device, handles, s, &w.gate_scale, shape.ane, &mut keep))
                .collect::<Result<_>>()?,
            up: w
                .up
                .iter()
                .map(|s| target(device, handles, s, &w.up_scale, shape.ane, &mut keep))
                .collect::<Result<_>>()?,
            down: w
                .down
                .iter()
                .map(|s| target(device, handles, s, &w.down_scale, hidden, &mut keep))
                .collect::<Result<_>>()?,
        })
    };
    let sets = [set(&memory.sets[0])?, set(&memory.sets[1])?];
    let up_rows = banks::staging(device, handles, MAX_ROWS, shape.gpu, Dtype::Bf16, &mut keep)?;
    let rotated = banks::staging(device, handles, MAX_ROWS, hidden, Dtype::F16, &mut keep)?;
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
    let shared = Shared {
        inputs,
        token_scale,
        partial,
        signs,
        status,
    };

    let mut layers = BTreeMap::new();
    let mut ordinals = BTreeMap::new();
    for mlp in mlps
        .iter()
        .filter(|m| m.intermediate == intermediate && m.gate_up.codes.width == hidden)
    {
        ordinals.insert(mlp.key.0, mlp.layer);
        let scales = [
            banks::staging(device, handles, shape.ane, 1, Dtype::F16, &mut keep)?,
            banks::staging(device, handles, shape.ane, 1, Dtype::F16, &mut keep)?,
            banks::staging(device, handles, hidden, 1, Dtype::F16, &mut keep)?,
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
    let frame = device.frame()?;
    let pipelines = Pipelines::new();
    let sink = Sink::new(device, &frame, &pipelines, handles);
    for layer in layers.values() {
        kernels::row_scales(&sink, &shape, &layer.source, signs)
            .map_err(|e| fault(e.to_string()))?;
    }
    frame.commit()?;

    let event = device
        .device()
        .newSharedEvent()
        .ok_or(fault("no shared event".into()))?;
    let handoff = Handoff::new(event);
    let ffn = Arc::new(OnceLock::new());
    let (compiled, held, compile_shape) = (ffn.clone(), memory.clone(), shape.clone());
    std::thread::Builder::new()
        .name("pie-ane-compile".into())
        .spawn(move || {
            let started = std::time::Instant::now();
            let result = Ffn::compile(
                &compile_shape,
                &held,
                &kernels_metal::ane::cache_directory(),
            );
            match &result {
                Ok(_) => eprintln!(
                    "PIE_ANE: Neural Engine program ready in {:.1}s",
                    started.elapsed().as_secs_f64()
                ),
                Err(why) => eprintln!(
                    "PIE_ANE: the Neural Engine program did not compile ({why}); the GPU runs the MLP alone"
                ),
            }
            let _ = compiled.set(result);
        })
        .map_err(|e| fault(e.to_string()))?;
    eprintln!(
        "PIE_ANE: {} MLP layers split, Neural Engine takes {} of {intermediate} columns",
        layers.len(),
        shape.ane
    );
    Ok(Some(Private {
        handoff,
        ffn,
        shape,
        layers,
        ordinals,
        shared,
        sets,
        up_rows,
        rotated,
        status: status_host,
        staged: AtomicU32::new(u32::MAX),
        splits: AtomicU64::new(0),
        _buffers: keep,
        _memory: memory,
    }))
}

impl Private {
    fn ffn(&self) -> Option<&Ffn> {
        self.ffn.get()?.as_ref().ok()
    }

    /// The hand-off a live encode fences on.
    pub fn handoff(&self) -> &Handoff {
        &self.handoff
    }

    /// How the program's compile went: `None` while it is still running,
    /// `Some(Ok)` once a plan may be made, `Some(Err)` with the reason when
    /// the GPU is keeping the whole MLP.
    #[must_use]
    pub fn compiled(&self) -> Option<std::result::Result<(), String>> {
        self.ffn
            .get()
            .map(|result| result.as_ref().map(|_| ()).map_err(Clone::clone))
    }

    /// How many MLPs the Neural Engine has been handed so far.
    #[must_use]
    pub fn splits(&self) -> u64 {
        self.splits.load(Ordering::Acquire)
    }

    /// A split for the MLP whose gate-up weight is `gate_up`, over `rows`,
    /// if the Neural Engine can take it: the rows fit a procedure, the
    /// program is compiled, the layer is one of the split ones, and nothing
    /// has retired the hand-off.
    pub fn plan(&self, gate_up: poem_ir::ValueId, rows: u32) -> Option<Plan> {
        if !(MIN_ROWS..=MAX_ROWS).contains(&rows) || self.handoff.retired() {
            return None;
        }
        let evaluation = self.ffn()?.evaluation(rows)?;
        let layer = *self.ordinals.get(&gate_up.0)?;
        let split = self.layers.get(&layer)?.split;
        Some(Plan {
            split,
            up_rows: Tensor {
                rows,
                ..self.up_rows
            },
            rows,
            layer,
            ready: self.handoff.next(),
            done: self.handoff.next(),
            evaluation,
        })
    }

    fn stage(&self, ctx: &Ctx<'_>, layer: u32) -> std::result::Result<(), Error> {
        let Some(source) = self.layers.get(&layer) else {
            return Ok(());
        };
        let set = &self.sets[(layer & 1) as usize];
        kernels::stage(ctx, &self.shape, &source.source, self.shared.signs, set)?;
        self.staged.store(layer, Ordering::Release);
        Ok(())
    }

    /// Before the GPU's share: stage this layer's weights if they are not,
    /// hand the activations over, and queue the Neural Engine behind the
    /// signal.
    pub fn before(&self, ctx: &Ctx<'_>, plan: &Plan, x: Tensor) -> std::result::Result<(), Error> {
        if self.staged.load(Ordering::Acquire) != plan.layer {
            self.stage(ctx, plan.layer)?;
        }
        let rotated = Tensor {
            rows: plan.rows,
            ..self.rotated
        };
        kernels::prepare(ctx, &self.shared, x, rotated, plan.ready)?;
        if let Some(ffn) = self.ffn() {
            let binding = &ffn.evaluations[plan.evaluation].1[(plan.layer & 1) as usize];
            self.handoff
                .start(&ffn.program, binding, plan.ready, plan.done);
            self.splits.fetch_add(1, Ordering::AcqRel);
        }
        Ok(())
    }

    /// After the GPU's share: stage the next layer's weights into the other
    /// set while the Neural Engine runs, then wait on it and join.
    pub fn after(&self, ctx: &Ctx<'_>, plan: &Plan, out: Tensor) -> std::result::Result<(), Error> {
        self.stage(ctx, plan.layer + 1)?;
        kernels::join(ctx, &self.shared, out, plan.done)
    }

    /// What a step's completion handler asks: whether any Neural Engine
    /// evaluation or join of the step failed. A failure retires the split.
    pub fn verdict(&self) -> Option<Box<dyn Fn() -> Option<String> + Send>> {
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
