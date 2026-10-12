use eta_compiler::codegen::launch::{LaunchPackage, LaunchValue, ValueOrigin};
use eta_exec::{ExecPlan, StagePlan, StageRunner, Value};
use eta_ir::Dtype;
use eta_ir::op::IntrinsicId;

use super::session::Session;
use super::widen::Widen;
use crate::device::Context;
use crate::device::alloc::Buffer;
use crate::device::ctx::Frame;
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

    deferral: Option<Deferral>,
}

/// A pass recorded rather than run: its frame, unsubmitted, and the values
/// the host reads back once it lands. Every put the host does not read
/// stays on the device, so the next pass and the next forward take it from
/// there.
#[derive(Default)]
pub struct Deferral {
    pub frame: Option<Frame>,
    /// The read-back buffer the frame lands into, lent by the session.
    pub landing: Option<Buffer>,
    pub landings: Vec<Landing>,
}

/// A put the host reads, landing in the stage's read-back buffer.
#[derive(Clone, Copy, Debug)]
pub struct Landing {
    pub channel: u32,
    pub at: u64,
    pub dtype: Dtype,
    pub held: u64,
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
            deferral: None,
        }
    }

    /// Records the pass into a frame of its own for the caller to submit,
    /// instead of running each stage to completion.
    #[must_use]
    pub fn deferred(mut self) -> OnDevice<'a> {
        self.deferral = Some(Deferral::default());
        self
    }

    pub fn take_deferral(&mut self) -> Option<Deferral> {
        self.deferral.take()
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
        let deferring = self.deferral.is_some();
        let mut frame = match self.deferral.as_mut().and_then(|d| d.frame.take()) {
            Some(frame) => frame,
            None => self.device.frame()?,
        };

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
            if root.source == ValueOrigin::ChannelTake
                && vals[global as usize].is_empty()
                && self.session.restore(&mut frame, at, local, root.channel)?
            {
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
        // A value is read back when any channel it goes to is one the
        // device does not keep.
        let mut landed: Vec<(u32, u32)> = Vec::new();
        for put in &package.stages[at].puts {
            let Some(local) = local_of(package, at, put.value)? else {
                continue;
            };
            let channel = put.channel;
            let bytes = self.session.span(at, local).map_or(0, |(_, l)| l);
            let on_device = if deferring {
                host_reads_not(package, channel)
            } else {
                keeps(package, channel, bytes)
            };
            if on_device {
                self.session
                    .keep(self.device, &mut frame, at, local, channel)?;
            } else {
                self.session.forget(channel);
                landed.push((put.value, channel));
            }
        }
        let kept = |global: u32| !landed.iter().any(|&(value, _)| value == global);
        let landing = match self.deferral.as_mut() {
            Some(deferral) if deferral.landing.is_none() => {
                if package.plans.len() != 1 {
                    return Err(Fault::Program {
                        at: "guest::run",
                        why: "a pass recorded to land later runs one stage".into(),
                    });
                }
                let lent = self.session.lend_landing(self.device, at)?;
                Some(deferral.landing.insert(lent).clone())
            }
            Some(deferral) => deferral.landing.clone(),
            None => None,
        };
        for &global in wanted {
            if kept(global) {
                continue;
            }
            if let Some(local) = local_of(package, at, global)? {
                match landing.as_ref() {
                    Some(into) => self.session.read_back_into(&mut frame, at, local, into)?,
                    None => self.session.read_back(&mut frame, at, local)?,
                }
            }
        }
        if let Some(deferral) = self.deferral.as_mut() {
            for (index, &global) in wanted.iter().enumerate() {
                if wanted[..index].contains(&global) {
                    continue;
                }
                let Some(local) = local_of(package, at, global)? else {
                    continue;
                };
                let dtype = plan.value_types[local as usize].dtype;
                // The ring takes an empty cell now; the host's own lands
                // after the frame does.
                vals[global as usize] = from_heap(&[], dtype, 0);
                let (from, held) = self
                    .session
                    .span(at, local)
                    .expect("a wanted value has a span");
                for &(_, channel) in landed.iter().filter(|&&(value, _)| value == global) {
                    deferral.landings.push(Landing {
                        channel,
                        at: from,
                        dtype,
                        held,
                    });
                }
            }
            deferral.frame = Some(frame);
            self.ran.push(at);
            return Ok(());
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
            if kept(global) {
                // The device holds the cell; the ring carries an empty one
                // for the next take to restore.
                vals[global as usize] = from_heap(&[], dtype, 0);
                continue;
            }
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
    fn keeps(&self, channel: u32) -> bool {
        self.session.keeps(channel)
    }

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

/// Cells at least this large stay on the device between fires when only the
/// device reads them: a sampler's token histograms, a vocabulary wide each.
const KEEP_BYTES: u64 = 64 << 10;

fn keeps(package: &LaunchPackage, channel: u32, bytes: u64) -> bool {
    package.channels.get(channel as usize).is_some_and(|decl| {
        decl.capacity == 1
            && decl.host_role != eta_ir::container::HostRole::Reader
            && bytes >= KEEP_BYTES
    })
}

/// Whether a deferred pass keeps `channel`'s puts on the device: every one the
/// host does not read.
fn host_reads_not(package: &LaunchPackage, channel: u32) -> bool {
    let read_at_fire = package.ports.iter().any(|binding| {
        binding.channel == channel
            && !binding.is_const
            && binding.port == eta_ir::registry::Port::RsFoldLen
    });
    !read_at_fire
        && package
            .channels
            .get(channel as usize)
            .is_some_and(|decl| decl.host_role != eta_ir::container::HostRole::Reader)
}

/// The value a landed read-back holds.
pub fn landed_value(landed: &Buffer, landing: &Landing) -> Result<Value> {
    let mut raw = vec![0u8; landing.held as usize];
    landed.read(landing.at, &mut raw)?;
    Ok(from_heap(&raw, landing.dtype, landing.held))
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
