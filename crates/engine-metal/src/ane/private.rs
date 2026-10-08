use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};

use ane::ffn::{Ffn, MAX_ROWS, MIN_ROWS, Memory, SEGMENT, Shape};
use ane::handoff::Handoff;
use ane::private::Surface;
use kernels_metal::linear::ane::{self as kernels, Join, Pack, Rows, Target};
use kernels_metal::{Bank, Ctx, Error, Tensor};
use poem_ir::Dtype;

use super::Plan;
use super::banks::{self, Mlp, Split};
use crate::device::{Buffer, Context, Handles, Pipelines};
use crate::encode::Sink;
use crate::error::{Fault, Result};

struct Set {
    gate: Vec<Target>,
    up: Vec<Target>,
    down: Vec<Target>,
}

struct Layer {
    split: Split,
    gate_up: Bank,
    down: Bank,
    scales: [Tensor; 3],
}

pub struct Private {
    handoff: Handoff,
    ffn: Arc<OnceLock<std::result::Result<Ffn, String>>>,
    shape: Shape,
    layers: BTreeMap<u32, Layer>,
    inputs: Vec<(Tensor, u32)>,
    token_scale: Tensor,
    partial: (Tensor, u32),
    sets: [Set; 2],
    up_rows: Tensor,
    rotated: Tensor,
    signs: Tensor,
    status: (Tensor, usize),
    staged: AtomicU32,
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
    let buffer = unsafe { Buffer::foreign(device, at, surface.bytes()) }?;
    let dtype = if surface.int8 { Dtype::I8 } else { Dtype::F16 };
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

fn units(intermediate: u32) -> u32 {
    let all = intermediate / ane::ffn::UNIT;
    ane::units()
        .unwrap_or(if kernels_metal::tuning::current().qmm_mpp {
            8
        } else {
            all * 3 / 5
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
                view(device, handles, s, SEGMENT, MAX_ROWS, &mut keep)?,
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
    let mut set = |w: &ane::ffn::Weights| -> Result<Set> {
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
        ane::ffn::INTERMEDIATE_BLOCK,
        Dtype::F32,
        &mut keep,
    )?;
    let sign_values = ane::ffn::signs();
    let bytes: Vec<u8> = sign_values.iter().flat_map(|v| v.to_le_bytes()).collect();
    banks::host(&keep[keep.len() - 1])?[..bytes.len()].copy_from_slice(&bytes);
    let status = banks::staging(device, handles, 1, 1, Dtype::U32, &mut keep)?;
    let status_host = banks::host(&keep[keep.len() - 1])?.as_mut_ptr() as usize;

    let mut layers = BTreeMap::new();
    for mlp in mlps
        .iter()
        .filter(|m| m.intermediate == intermediate && m.gate_up.codes.width == hidden)
    {
        let scales = [
            banks::staging(device, handles, shape.ane, 1, Dtype::F16, &mut keep)?,
            banks::staging(device, handles, shape.ane, 1, Dtype::F16, &mut keep)?,
            banks::staging(device, handles, hidden, 1, Dtype::F16, &mut keep)?,
        ];
        let split = banks::split(device, handles, mlp, shape.gpu, &mut keep)?;
        layers.insert(
            mlp.layer,
            Layer {
                split,
                gate_up: mlp.gate_up,
                down: mlp.down,
                scales,
            },
        );
    }
    let frame = device.frame()?;
    let pipelines = Pipelines::new();
    let sink = Sink::new(device, &frame, &pipelines, handles);
    for layer in layers.values() {
        row_scales(&sink, &shape, layer, signs).map_err(|e| fault(e.to_string()))?;
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
            let result = Ffn::compile(&compile_shape, &held, &ane::private::cache_directory());
            match &result {
                Ok(_) => eprintln!("PIE_ANE: Neural Engine program ready in {:.1}s", started.elapsed().as_secs_f64()),
                Err(why) => eprintln!("PIE_ANE: the Neural Engine program did not compile ({why}); the GPU runs the MLP alone"),
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
        inputs,
        token_scale,
        partial,
        sets,
        up_rows,
        rotated,
        signs,
        status: (status, status_host),
        staged: AtomicU32::new(u32::MAX),
        _buffers: keep,
        _memory: memory,
    }))
}

fn row_scales(
    ctx: &Ctx<'_>,
    shape: &Shape,
    layer: &Layer,
    signs: Tensor,
) -> std::result::Result<(), Error> {
    let gate = Rows {
        first: shape.gpu,
        rows: shape.ane,
        input: 0,
        span: shape.hidden,
        intermediate: false,
    };
    let up = Rows {
        first: shape.intermediate + shape.gpu,
        ..gate
    };
    let down = Rows {
        first: 0,
        rows: shape.hidden,
        input: shape.gpu,
        span: shape.ane,
        intermediate: true,
    };
    kernels::row_scale(ctx, &layer.gate_up, &gate, signs, layer.scales[0])?;
    kernels::row_scale(ctx, &layer.gate_up, &up, signs, layer.scales[1])?;
    kernels::row_scale(ctx, &layer.down, &down, signs, layer.scales[2])
}

impl Private {
    fn ffn(&self) -> Option<&Ffn> {
        self.ffn.get()?.as_ref().ok()
    }

    pub fn plan(&self, layer: u32, rows: u32) -> Option<Plan> {
        if !(MIN_ROWS..=MAX_ROWS).contains(&rows) || self.handoff.retired() {
            return None;
        }
        let evaluation = self.ffn()?.evaluation(rows)?;
        let split = self.layers.get(&layer)?.split;
        Some(Plan {
            split,
            up_rows: Tensor {
                rows,
                ..self.up_rows
            },
            layer,
            rows,
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
        let shape = &self.shape;
        for (k, (gate, up)) in set.gate.iter().zip(&set.up).enumerate() {
            let rows = Rows {
                first: shape.gpu,
                rows: shape.ane,
                input: k as u32 * SEGMENT,
                span: SEGMENT,
                intermediate: false,
            };
            kernels::weights(
                ctx,
                &source.gate_up,
                &rows,
                self.signs,
                source.scales[0],
                gate,
            )?;
            let rows = Rows {
                first: shape.intermediate + shape.gpu,
                ..rows
            };
            kernels::weights(
                ctx,
                &source.gate_up,
                &rows,
                self.signs,
                source.scales[1],
                up,
            )?;
        }
        let mut input = shape.gpu;
        for (&span, down) in shape.down.iter().zip(&set.down) {
            let rows = Rows {
                first: 0,
                rows: shape.hidden,
                input,
                span,
                intermediate: true,
            };
            kernels::weights(ctx, &source.down, &rows, self.signs, source.scales[2], down)?;
            input += span;
        }
        self.staged.store(layer, Ordering::Release);
        Ok(())
    }

    pub fn before(&self, ctx: &Ctx<'_>, plan: &Plan, x: Tensor) -> std::result::Result<(), Error> {
        if self.staged.load(Ordering::Acquire) != plan.layer {
            self.stage(ctx, plan.layer)?;
        }
        let rotated = Tensor {
            rows: plan.rows,
            ..self.rotated
        };
        kernels::rotate(ctx, x, self.signs, rotated, self.token_scale)?;
        let last = self.inputs.len() - 1;
        for (k, &(packed, stride)) in self.inputs.iter().enumerate() {
            let stamp = if k == last { plan.ready as u32 } else { 0 };
            kernels::pack(
                ctx,
                rotated,
                packed,
                &Pack {
                    channel: k as u32 * SEGMENT,
                    width: SEGMENT,
                    stride,
                    stamp,
                },
            )?;
        }
        if let Some(ffn) = self.ffn() {
            let binding = &ffn.evaluations[plan.evaluation].1[(plan.layer & 1) as usize];
            self.handoff
                .start(&ffn.program, binding, plan.ready, plan.done);
        }
        Ok(())
    }

    pub fn after(&self, ctx: &Ctx<'_>, plan: &Plan, out: Tensor) -> std::result::Result<(), Error> {
        self.stage(ctx, plan.layer + 1)?;
        let (partial, stride) = self.partial;
        let join = Join {
            stride,
            stamp: plan.done as u32,
        };
        kernels::split_join(ctx, out, partial, self.token_scale, self.status.0, &join)
    }

    pub fn event(&self) -> &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLEvent> {
        objc2::runtime::ProtocolObject::from_ref(self.handoff.event())
    }

    pub fn verdict(&self) -> Option<Box<dyn Fn() -> Option<String> + Send>> {
        let handoff = self.handoff.clone();
        let seen = handoff.failures();
        let status = self.status.1;
        Some(Box::new(move || {
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
