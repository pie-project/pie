use eta_compiler::codegen::launch::{LaunchPackage, LaunchValue, ValueOrigin};
use eta_exec::{ExecPlan, StagePlan, StageRunner, Value};
use eta_ir::Dtype;
use eta_ir::op::IntrinsicId;

use super::session::Session;
use super::widen::Widen;
use crate::device::Context;
use crate::device::alloc::Buffer;
use crate::error::{Fault, Result};

#[derive(Clone, Copy)]
pub struct Readout<'a> {
    pub seat: &'a Buffer,

    pub at: u64,

    pub width: u32,
}

pub struct OnDevice<'a> {
    device: &'a Context,
    session: &'a mut Session,
    widen: &'a Widen,
    logits: Option<Readout<'a>>,

    ran: Vec<usize>,
}

impl<'a> OnDevice<'a> {
    pub fn new(
        device: &'a Context,
        session: &'a mut Session,
        widen: &'a Widen,
        logits: Option<Readout<'a>>,
    ) -> OnDevice<'a> {
        OnDevice {
            device,
            session,
            widen,
            logits,
            ran: Vec::new(),
        }
    }

    #[must_use]
    pub fn ran(&self) -> &[usize] {
        &self.ran
    }

    fn stage(
        &mut self,
        package: &LaunchPackage,
        at: usize,
        roots: &[u32],
        wanted: &[u32],
        vals: &mut [Value],
    ) -> Result<()> {
        let plan = &package.plans[at];
        let mut frame = self.device.frame()?;

        for &global in roots {
            let Some(local) = local_of(package, at, global)? else {
                continue;
            };
            let root = &package.values[global as usize];
            if root.source == ValueOrigin::Intrinsic && root.intrinsic == Some(IntrinsicId::Logits)
            {
                let seat = self.logits.ok_or_else(|| Fault::Program {
                    at: "guest::run",
                    why: "this pass reads the logits and no readout row was offered".into(),
                })?;
                let (_, held) = self.session.span(at, local).ok_or_else(|| Fault::Program {
                    at: "guest::run",
                    why: format!("the logits root {local} is not stage {at}'s"),
                })?;
                let lanes = u32::try_from(held / 4).unwrap_or(0);
                if lanes != seat.width {
                    return Err(Fault::Program {
                        at: "guest::run",
                        why: format!(
                            "this pass reads a {lanes}-wide logits row and the fire read out \
                             {} wide; the two are one rectangle or neither",
                            seat.width
                        ),
                    });
                }
                let heap = self.session.heap(at).ok_or_else(|| Fault::Program {
                    at: "guest::run",
                    why: format!("stage {at} has no heap"),
                })?;
                let (into, _) = self
                    .session
                    .span(at, local)
                    .expect("the span was just read");
                self.widen
                    .encode(&frame, seat.seat, seat.at, heap, into, seat.width)?;
                continue;
            }
            let bytes = root_bytes(
                root,
                &vals[global as usize],
                self.session.span(at, local).map_or(0, |(_, l)| l),
            );
            self.session.stage_in(at, local, &bytes)?;
        }

        self.session.flush(&mut frame, at)?;
        self.session.dispatch(&frame, at)?;
        for &global in wanted {
            if let Some(local) = local_of(package, at, global)? {
                self.session.read_back(&mut frame, at, local)?;
            }
        }
        frame.commit()?;

        if let Some(code) = self.session.status()? {
            self.session.clear_status()?;
            return Err(Fault::Program {
                at: "guest::run",
                why: {
                    let named = eta_exec::describe_fault(
                        package,
                        code,
                        u32::try_from(package.channels.len()).unwrap_or(0),
                    );
                    match (named.class, named.channel) {
                        (Some(class), Some(channel)) => format!(
                            "stage {at} faulted on the device: {class} on channel {channel} \
                             (code {code})"
                        ),
                        (Some(class), None) => {
                            format!("stage {at} faulted on the device: {class} (code {code})")
                        }
                        (None, _) => format!(
                            "stage {at} faulted on the device with code {code}, which this \
                             program's fault table does not name"
                        ),
                    }
                },
            });
        }
        for &global in wanted {
            let Some(local) = local_of(package, at, global)? else {
                continue;
            };
            let dtype = plan.value_types[local as usize].dtype;
            let (_, held) = self
                .session
                .span(at, local)
                .expect("a wanted value has a span");
            let raw = self.session.taken(at, local)?;
            vals[global as usize] = from_heap(&raw, dtype, held);
        }
        self.ran.push(at);
        Ok(())
    }
}

impl StageRunner for OnDevice<'_> {
    fn run(
        &mut self,
        plan: &ExecPlan,
        sp: &StagePlan,
        roots: &[u32],
        wanted: &[u32],
        vals: &mut [Value],
    ) -> eta_exec::Result<()> {
        let at = sp.stage_index;
        self.stage(&plan.package, at, roots, wanted, vals)
            .map_err(|fault| eta_exec::Error {
                message: fault.to_string(),
            })
    }

    fn binds(&self, id: Option<IntrinsicId>) -> bool {
        id == Some(IntrinsicId::Logits) && self.logits.is_some()
    }
}

fn local_of(package: &LaunchPackage, at: usize, global: u32) -> Result<Option<u32>> {
    match package.plan_local(global) {
        Some((stage, local)) if stage == at => Ok(local),
        Some((stage, _)) => Err(Fault::Program {
            at: "guest::run",
            why: format!("value {global} is stage {stage}'s and this is stage {at}"),
        }),
        None => Err(Fault::Program {
            at: "guest::run",
            why: format!("value {global} belongs to no stage of this package"),
        }),
    }
}

fn root_bytes(root: &LaunchValue, value: &Value, span: u64) -> Vec<u8> {
    let bytes = to_heap(value);
    if root.source != ValueOrigin::Const {
        return bytes;
    }
    let lane = if matches!(value, Value::Bool(_)) {
        1
    } else {
        4
    };
    let span = usize::try_from(span).unwrap_or(0);
    if bytes.len() != lane || span <= lane || !span.is_multiple_of(lane) {
        return bytes;
    }
    bytes.repeat(span / lane)
}

fn to_heap(value: &Value) -> Vec<u8> {
    match value {
        Value::F32(cells) => cells.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Value::I32(cells) => cells.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Value::U32(cells) => cells.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Value::Bool(cells) => cells.iter().map(|&b| u8::from(b != 0)).collect(),
    }
}

fn from_heap(raw: &[u8], dtype: Dtype, held: u64) -> Value {
    let word = |i: usize| {
        let at = i * 4;
        u32::from_le_bytes([raw[at], raw[at + 1], raw[at + 2], raw[at + 3]])
    };
    match dtype {
        Dtype::Bool => Value::Bool(raw.iter().map(|&b| u8::from(b != 0)).collect()),
        Dtype::I32 => Value::I32((0..(held / 4) as usize).map(|i| word(i) as i32).collect()),
        Dtype::U32 => Value::U32((0..(held / 4) as usize).map(word).collect()),
        _ => Value::F32(
            (0..(held / 4) as usize)
                .map(|i| f32::from_bits(word(i)))
                .collect(),
        ),
    }
}
