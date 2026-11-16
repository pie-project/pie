use eta_compiler::codegen::launch::{LaunchPackage, LaunchValue, ValueOrigin};
use eta_exec::{ExecPlan, StagePlan, StageRunner, Value};
use eta_ir::Dtype;

use super::session::Session;
use super::widen::Widen;
use crate::device::Context;
use crate::device::ctx::Frame;
use engine_gpu::error::{Fault, Result};

pub use engine_gpu::guest::run::{from_heap, local_of, root_bytes, to_heap};

pub type Readout<'a> = engine_gpu::guest::run::Readout<'a, crate::device::Api>;

pub struct OnDevice<'a> {
    device: &'a Context,
    session: &'a mut Session,
    widen: &'a Widen,
    /// What the fire read out for the pass's lane `k`, at `k`.
    seats: Vec<Vec<Readout<'a>>>,

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
    /// The ring rooms the frame lands into, held until it lands.
    pub region: Option<super::session::Region>,
    pub landings: Vec<Landing>,
}

/// A put the host reads, landing in the pass's ring rooms at ring byte
/// `at`.
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
        seats: Vec<Vec<Readout<'a>>>,
    ) -> OnDevice<'a> {
        OnDevice {
            device,
            session,
            widen,
            seats,
            ran: Vec::new(),
            deferral: None,
        }
    }

    /// What this fire read out of output `root` reads, for its lane.
    fn offered(&self, root: &LaunchValue) -> Option<Readout<'a>> {
        let name = root.output.as_deref()?;
        self.seats
            .get(root.lane as usize)?
            .iter()
            .find(|seat| seat.output == name)
            .copied()
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

        self.session.open(at)?;
        for &global in roots {
            let Some(local) = local_of(package, at, global)? else {
                continue;
            };
            let root = &package.values[global as usize];
            if root.source == ValueOrigin::Output {
                let seat = self.offered(root).ok_or_else(|| Fault::Program {
                    at: "guest::run",
                    why: format!(
                        "this pass reads output {:?} of lane {}, which the fire read out none of",
                        root.output, root.lane
                    ),
                })?;
                let (into, into_at, held) = self
                    .session
                    .place(at, local)
                    .map(|(buffer, into_at, held)| (buffer.clone(), into_at, held))
                    .ok_or_else(|| Fault::Program {
                        at: "guest::run",
                        why: format!("the output root {local} is not stage {at}'s"),
                    })?;
                let values = u32::try_from(held / 4).unwrap_or(0);
                if values > seat.width {
                    return Err(Fault::Program {
                        at: "guest::run",
                        why: format!(
                            "this pass reads {values} values of output {:?} and the fire read \
                             out {}",
                            root.output, seat.width
                        ),
                    });
                }
                if seat.dtype != Dtype::Bf16 {
                    frame.copy(seat.seat, seat.at, &into, into_at, u64::from(values) * 4)?;
                } else {
                    self.widen
                        .encode(&frame, seat.seat, seat.at, &into, into_at, values)?;
                }
                continue;
            }
            if root.source == ValueOrigin::ChannelTake
                && vals[global as usize].is_empty()
                && self.session.restore(&mut frame, at, local, root.channel)?
            {
                continue;
            }
            let span = self.session.bytes(at, local).unwrap_or(0);
            if root.source == ValueOrigin::Const
                && let Some(word) = fill_word(&vals[global as usize])
            {
                self.session.fill(at, local, word)?;
                continue;
            }
            let bytes = root_bytes(root, &vals[global as usize], span);
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
            let bytes = self.session.bytes(at, local).unwrap_or(0);
            let on_device = if deferring {
                host_reads_not(package, channel)
            } else {
                keeps(package, channel, bytes)
            };
            if on_device {
                self.session.keep(&mut frame, at, local, channel)?;
            } else {
                self.session.forget(channel);
                landed.push((put.value, channel));
            }
        }
        let kept = |global: u32| !landed.iter().any(|&(value, _)| value == global);
        if deferring && package.plans.len() != 1 {
            return Err(Fault::Program {
                at: "guest::run",
                why: "a pass recorded to land later runs one stage".into(),
            });
        }
        let mut landed_at: Vec<(u32, u64)> = Vec::new();
        for &global in wanted {
            if kept(global) {
                continue;
            }
            if let Some(local) = local_of(package, at, global)? {
                match self.session.read_back(&mut frame, at, local)? {
                    Some(ring_at) => landed_at.push((global, ring_at)),
                    // A put past the rooms lands where it lies, read once the
                    // pass has landed: no deferred pass takes one
                    // (`Plane::defers`).
                    None if deferring => {
                        return Err(Fault::Program {
                            at: "guest::run",
                            why: format!(
                                "stage {at} recorded to land later puts value {local} past \
                                 its lanes' ring rooms"
                            ),
                        });
                    }
                    None => {}
                }
            }
        }
        self.session.land_status(&mut frame)?;
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
                let held = self.session.bytes(at, local).unwrap_or(0);
                let Some(&(_, from)) = landed_at.iter().find(|&&(value, _)| value == global)
                else {
                    continue;
                };
                for &(_, channel) in landed.iter().filter(|&&(value, _)| value == global) {
                    deferral.landings.push(Landing {
                        channel,
                        at: from,
                        dtype,
                        held,
                    });
                }
            }
            deferral.region = self.session.close();
            deferral.frame = Some(frame);
            self.ran.push(at);
            return Ok(());
        }
        frame.commit()?;

        if let Some(code) = self.session.status()? {
            self.session.close();
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
            let held = self.session.bytes(at, local).unwrap_or(0);
            let raw = self.session.taken(at, local)?;
            vals[global as usize] = from_heap(&raw, dtype, held);
        }
        self.session.close();
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

    fn binds(&self, root: &LaunchValue) -> bool {
        self.offered(root).is_some()
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
            && binding.port == eta_ir::registry::Port::Fold
    });
    !read_at_fire
        && package
            .channels
            .get(channel as usize)
            .is_some_and(|decl| decl.host_role != eta_ir::container::HostRole::Reader)
}

/// The one word a constant fills its every lane with on the device: a
/// constant is one literal over its shape (a byte a lane for a bool).
fn fill_word(value: &Value) -> Option<u32> {
    match value {
        Value::F32(cells) => cells.first().map(|x| x.to_bits()),
        Value::I32(cells) => cells.first().map(|&x| x as u32),
        Value::U32(cells) => cells.first().copied(),
        Value::Bool(cells) => cells
            .first()
            .map(|&b| u32::from_le_bytes([u8::from(b != 0); 4])),
    }
    .filter(|_| match value {
        Value::F32(cells) => cells.windows(2).all(|pair| pair[0].to_bits() == pair[1].to_bits()),
        Value::I32(cells) => cells.windows(2).all(|pair| pair[0] == pair[1]),
        Value::U32(cells) => cells.windows(2).all(|pair| pair[0] == pair[1]),
        Value::Bool(cells) => cells.windows(2).all(|pair| (pair[0] != 0) == (pair[1] != 0)),
    })
}

/// The value a landed read-back holds.
pub fn landed_value(landed: &super::session::Region, landing: &Landing) -> Result<Value> {
    let mut raw = vec![0u8; landing.held as usize];
    landed.read(landing.at, &mut raw)?;
    Ok(from_heap(&raw, landing.dtype, landing.held))
}
