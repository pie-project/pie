use eta_compiler::codegen::launch::{LaunchPackage, LaunchValue, ValueOrigin};
use eta_exec::{ExecPlan, StagePlan, StageRunner, Value};
use eta_ir::Dtype;

use super::session::Session;
use super::widen::Widen;
use crate::device::Context;
use crate::device::alloc::Buffer;
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

/// A pass recorded rather than run: the frame it records into (the step's
/// guest frame, which the passes of every attachment of the step share),
/// unsubmitted, and the values the host reads back once it lands. Every put
/// the host does not read stays on the device, so the next stage, the next
/// pass and the next forward take it from there.
#[derive(Default)]
pub struct Deferral {
    pub frame: Option<Frame>,
    /// The read-back buffers the frame lands into, lent by the session, one
    /// for each stage that lands a value, with that stage.
    pub lent: Vec<(usize, Buffer)>,
    pub landings: Vec<Landing>,
}

/// A put the host reads, landing in its stage's read-back buffer, the
/// deferral's `lent[lent]`.
#[derive(Clone, Copy, Debug)]
pub struct Landing {
    pub channel: u32,
    pub lent: usize,
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

    /// Records the pass into `frame` for the caller to submit, instead of
    /// running each stage to completion. The frame comes back with the
    /// deferral ([`OnDevice::take_deferral`]) however the pass ends, with
    /// whatever other passes recorded into it before.
    #[must_use]
    pub fn deferred(mut self, frame: Frame) -> OnDevice<'a> {
        self.deferral = Some(Deferral {
            frame: Some(frame),
            ..Deferral::default()
        });
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
        let Some(deferral) = self.deferral.as_mut() else {
            let mut frame = self.device.frame()?;
            let landed = self.record(package, at, roots, wanted, vals, &mut frame)?;
            return self.land(package, at, wanted, vals, &landed, frame);
        };
        let mut frame = deferral.frame.take().ok_or_else(|| Fault::Program {
            at: "guest::run",
            why: "a deferred pass was given no frame to record into".into(),
        })?;
        // The frame carries other passes of the step: it goes back to the
        // deferral whether or not this stage records.
        let recorded = self.record(package, at, roots, wanted, vals, &mut frame);
        self.deferral
            .as_mut()
            .expect("the deferral was just read")
            .frame = Some(frame);
        recorded?;
        self.ran.push(at);
        Ok(())
    }

    /// Records stage `at` into `frame`: its roots in, its dispatch, its puts
    /// kept on the device or copied out to be read back. Deferred, the
    /// values the host reads land after the frame does; else returns the
    /// (value, channel) pairs it copied out, for [`OnDevice::land`].
    fn record(
        &mut self,
        package: &LaunchPackage,
        at: usize,
        roots: &[u32],
        wanted: &[u32],
        vals: &mut [Value],
        frame: &mut Frame,
    ) -> Result<Vec<(u32, u32)>> {
        let plan = &package.plans[at];
        let deferring = self.deferral.is_some();

        for &global in roots {
            let Some(local) = local_of(package, at, global)? else {
                continue;
            };
            let root = &package.values[global as usize];
            if root.source == ValueOrigin::Output {
                let (_, held) = self.session.span(at, local).ok_or_else(|| Fault::Program {
                    at: "guest::run",
                    why: format!("the output root {local} is not stage {at}'s"),
                })?;
                let values = u32::try_from(held / 4).unwrap_or(0);
                // Row `r` of a root read at lane `k` is lane `k + r`'s
                // readout row, each through its own seat.
                let rows = engine_gpu::guest::run::row_seats(&self.seats, root, values)?;
                let (heap, base) = self.session.heap(at).ok_or_else(|| Fault::Program {
                    at: "guest::run",
                    why: format!("stage {at} has no heap"),
                })?;
                let (into, _) = self
                    .session
                    .span(at, local)
                    .expect("the span was just read");
                let mut into = base + into;
                for (seat, values) in rows {
                    if seat.dtype != Dtype::Bf16 {
                        frame.copy(seat.seat, seat.at, heap, into, u64::from(values) * 4)?;
                    } else {
                        self.widen
                            .encode(frame, seat.seat, seat.at, heap, into, values)?;
                    }
                    into += u64::from(values) * 4;
                }
                continue;
            }
            if root.source == ValueOrigin::ChannelTake
                && vals[global as usize].is_empty()
                && self.session.restore(frame, at, local, root.channel)?
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

        self.session.flush(frame, at)?;
        self.session.dispatch(frame, at)?;
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
                self.session.keep(self.device, frame, at, local, channel)?;
            } else {
                self.session.forget(channel);
                landed.push((put.value, channel));
            }
        }
        let kept = |global: u32| !landed.iter().any(|&(value, _)| value == global);
        let Some(deferral) = self.deferral.as_mut() else {
            for &global in wanted {
                if kept(global) {
                    continue;
                }
                if let Some(local) = local_of(package, at, global)? {
                    self.session.read_back(frame, at, local)?;
                }
            }
            return Ok(landed);
        };
        // The stage's read-back buffer, lent once a value of it lands.
        let mut lent = None;
        for (index, &global) in wanted.iter().enumerate() {
            if kept(global) || wanted[..index].contains(&global) {
                continue;
            }
            let Some(local) = local_of(package, at, global)? else {
                continue;
            };
            let lent = match lent {
                Some(lent) => lent,
                None => {
                    let buffer = self.session.lend_landing(self.device, at)?;
                    deferral.lent.push((at, buffer));
                    *lent.insert(deferral.lent.len() - 1)
                }
            };
            self.session
                .read_back_into(frame, at, local, &deferral.lent[lent].1)?;
            let dtype = plan.value_types[local as usize].dtype;
            let (from, held) = self
                .session
                .span(at, local)
                .expect("a wanted value has a span");
            for &(_, channel) in landed.iter().filter(|&&(value, _)| value == global) {
                deferral.landings.push(Landing {
                    channel,
                    lent,
                    at: from,
                    dtype,
                    held,
                });
            }
        }
        // The ring takes an empty cell now; the host's own lands after the
        // frame does, and a later stage of the pass takes a kept one from
        // the device.
        for &global in wanted {
            if let Some(local) = local_of(package, at, global)? {
                vals[global as usize] = from_heap(&[], plan.value_types[local as usize].dtype, 0);
            }
        }
        Ok(landed)
    }

    /// Runs the frame stage `at` recorded to completion and reads back what
    /// it copied out (`landed`).
    fn land(
        &mut self,
        package: &LaunchPackage,
        at: usize,
        wanted: &[u32],
        vals: &mut [Value],
        landed: &[(u32, u32)],
        frame: Frame,
    ) -> Result<()> {
        let plan = &package.plans[at];
        frame.commit()?;

        if let Some(code) = self.session.status()? {
            self.session.clear_status()?;
            return Err(Fault::Program {
                at: "guest::run",
                why: describe_device_fault(package, at, code),
            });
        }
        let kept = |global: u32| !landed.iter().any(|&(value, _)| value == global);
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

/// What a stage's device fault `code` says, in the words of `package`'s fault
/// table.
pub fn describe_device_fault(package: &LaunchPackage, at: usize, code: u32) -> String {
    let named = eta_exec::describe_fault(
        package,
        code,
        u32::try_from(package.channels.len()).unwrap_or(0),
    );
    match (named.class, named.channel) {
        (Some(class), Some(channel)) => format!(
            "stage {at} faulted on the device: {class} on channel {channel} (code {code})"
        ),
        (Some(class), None) => format!("stage {at} faulted on the device: {class} (code {code})"),
        (None, _) => format!(
            "stage {at} faulted on the device with code {code}, which this program's fault \
             table does not name"
        ),
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
/// host does not read, at the fire (a `fold` count, an `attn_mask`) or after.
fn host_reads_not(package: &LaunchPackage, channel: u32) -> bool {
    let read_at_fire = package.ports.iter().any(|binding| {
        binding.channel == channel
            && !binding.is_const
            && matches!(
                binding.port,
                eta_ir::registry::Port::Fold | eta_ir::registry::Port::AttnMask
            )
    });
    !read_at_fire
        && package
            .channels
            .get(channel as usize)
            .is_some_and(|decl| decl.host_role != eta_ir::container::HostRole::Reader)
}

/// The value a landed read-back holds, of the buffers a deferral lent.
pub fn landed_value(lent: &[Buffer], landing: &Landing) -> Result<Value> {
    let landed = lent.get(landing.lent).ok_or_else(|| Fault::Program {
        at: "guest::run",
        why: format!(
            "a landing names read-back buffer {} of the {} its pass lent",
            landing.lent,
            lent.len()
        ),
    })?;
    let mut raw = vec![0u8; landing.held as usize];
    landed.read(landing.at, &mut raw)?;
    Ok(from_heap(&raw, landing.dtype, landing.held))
}

/// Whether `package`'s epilogue can be recorded whole to land later: no
/// stage takes or reads a channel an earlier stage of the pass puts for
/// the host to read, since that cell reaches the host only once the frame
/// lands. A cell kept on the device a later stage takes from the device.
#[must_use]
pub fn records_whole(package: &LaunchPackage) -> bool {
    let mut landing: Vec<u32> = Vec::new();
    for (at, stage) in package.stages.iter().enumerate() {
        if stage.stage != eta_ir::registry::Stage::Epilogue {
            continue;
        }
        let reads_landing = package.values.iter().enumerate().any(|(global, value)| {
            matches!(
                value.source,
                ValueOrigin::ChannelTake | ValueOrigin::ChannelRead
            ) && landing.contains(&value.channel)
                && package
                    .plan_local(global as u32)
                    .is_some_and(|(stage, local)| stage == at && local.is_some())
        });
        if reads_landing {
            return false;
        }
        landing.extend(
            (stage.puts.iter())
                .map(|put| put.channel)
                .filter(|&channel| !host_reads_not(package, channel)),
        );
    }
    true
}
