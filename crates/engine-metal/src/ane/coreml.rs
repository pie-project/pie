use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use kernels_metal::Tensor;
use kernels_metal::linear::ane as kernels;
use model_ir::Dtype;

use super::Plan;
use super::banks::{self, Mlp, Split};
use crate::device::{Buffer, Context, Handles};
use crate::error::{Fault, Result};

const MIN_ROWS: u32 = 256;
const FINGERPRINT_ROWS: u64 = 64;

pub struct CoreMl {
    splits: BTreeMap<u32, Split>,
    max_rows: u32,
    up_rows: Tensor,
    staged: Tensor,
    other: Tensor,
    next: AtomicU64,
    planned: AtomicU64,
    runner: ane::coreml::Runner,
    _buffers: Vec<Buffer>,
}

impl CoreMl {
    pub fn plan(&self, layer: u32, rows: u32) -> Option<Plan> {
        if rows < MIN_ROWS || rows > self.max_rows || !self.runner.takes(layer) {
            return None;
        }
        let base = self.next.fetch_add(2, Ordering::AcqRel);
        self.planned.fetch_add(1, Ordering::AcqRel);
        Some(Plan {
            split: *self.splits.get(&layer)?,
            up_rows: Tensor {
                rows,
                ..self.up_rows
            },
            layer,
            rows,
            ready: base + 1,
            done: base + 2,
            evaluation: 0,
        })
    }

    pub fn before(
        &self,
        ctx: &kernels_metal::Ctx<'_>,
        plan: &Plan,
        x: Tensor,
    ) -> std::result::Result<(), kernels_metal::Error> {
        kernels::stage(
            ctx,
            x,
            Tensor {
                rows: plan.rows,
                ..self.staged
            },
            plan.ready as u32,
        )?;
        self.runner.submit(ane::coreml::Job {
            layer: plan.layer,
            rows: plan.rows,
            stage: plan.ready,
            done: plan.done,
        });
        Ok(())
    }

    pub fn after(
        &self,
        ctx: &kernels_metal::Ctx<'_>,
        plan: &Plan,
        out: Tensor,
    ) -> std::result::Result<(), kernels_metal::Error> {
        kernels::join(
            ctx,
            out,
            Tensor {
                rows: plan.rows,
                ..self.other
            },
            plan.done as u32,
        )
    }

    pub fn event(&self) -> &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLSharedEvent> {
        self.runner.event()
    }

    pub fn may_split(&self, rows: u32) -> bool {
        (MIN_ROWS..=self.max_rows).contains(&rows)
            && self.splits.keys().any(|&layer| self.runner.takes(layer))
    }

    pub fn planned(&self) -> u64 {
        self.planned.load(Ordering::Acquire)
    }

    pub fn verdict(&self) -> Box<dyn Fn() -> Option<String> + Send> {
        let lost = self.runner.lost();
        let seen = lost();
        Box::new(move || {
            (lost() > seen).then(|| "a Neural Engine layer result was lost".to_string())
        })
    }
}

pub fn load(
    device: &Context,
    handles: &Handles,
    mlps: &[Mlp],
    asked: &str,
) -> Result<Option<CoreMl>> {
    use objc2_metal::MTLDevice;
    let first = &mlps[0].gate_up;
    let row = u64::from(first.codes.width) * u64::from(first.bits) / 8;
    let fingerprint = ane::fingerprint(&handles.read(first.codes.buf, row * FINGERPRINT_ROWS)?);
    let Some(build) = ane::coreml::Build::find(&fingerprint, asked) else {
        return Ok(None);
    };
    let keep = build.intermediate - build.ane;
    let max_rows = build.max_rows();
    let mut buffers = Vec::new();
    let mut splits = BTreeMap::new();
    for mlp in mlps {
        if mlp.intermediate == build.intermediate
            && build.layers.contains(&mlp.layer)
            && mlp.gate_up.codes.width == build.hidden
            && keep.is_multiple_of(mlp.down.group)
        {
            splits.insert(
                mlp.layer,
                banks::split(device, handles, mlp, keep, &mut buffers)?,
            );
        }
    }
    if splits.is_empty() {
        return Ok(None);
    }
    let up_rows = banks::staging(device, handles, max_rows, keep, Dtype::Bf16, &mut buffers)?;
    let staged = banks::staging(
        device,
        handles,
        max_rows,
        build.hidden,
        Dtype::F16,
        &mut buffers,
    )?;
    let other = banks::staging(
        device,
        handles,
        max_rows,
        build.hidden,
        Dtype::F16,
        &mut buffers,
    )?;
    let input = banks::host(&buffers[buffers.len() - 2])?
        .as_ptr()
        .cast::<u16>();
    let output = banks::host(&buffers[buffers.len() - 1])?
        .as_mut_ptr()
        .cast::<u16>();
    let event = device.device().newSharedEvent().ok_or(Fault::Device {
        call: "newSharedEvent",
        why: "the device would not make a shared event".to_string(),
    })?;
    eprintln!(
        "PIE_ANE: CoreML split of {} layers, Neural Engine takes {} columns, from {}",
        splits.len(),
        build.ane,
        build.dir.display()
    );
    let layers: Vec<u32> = splits.keys().copied().collect();
    let runner = unsafe { ane::coreml::Runner::start(build, &layers, input, output, event) }
        .map_err(|e| Fault::Device {
            call: "spawn",
            why: e.to_string(),
        })?;
    Ok(Some(CoreMl {
        splits,
        max_rows,
        up_rows,
        staged,
        other,
        next: AtomicU64::new(0),
        planned: AtomicU64::new(0),
        runner,
        _buffers: buffers,
    }))
}
