use eta_compiler::codegen::launch::{LaunchPackage, LaunchValue, ValueOrigin};
use eta_exec::{ExecPlan, StagePlan, StageRunner, Value};
use eta_ir::Dtype;

use super::session::{Pass, Session};
use super::widen::Widen;
use crate::device::Context;
use crate::device::ctx::Frame;
use engine_gpu::error::{Fault, Result};
pub use engine_gpu::guest::defer::Landing;
use engine_gpu::guest::defer::{host_reads_not, keeps};

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
    /// What the pass reads back, in its frame's readback buffer.
    pub pass: Option<Pass>,
    pub landings: Vec<Landing>,
}

impl engine_gpu::guest::defer::LandedPass for Pass {
    fn status(&self) -> Result<Option<u32>> {
        Pass::status(self)
    }

    fn value(&self, landing: &Landing) -> Result<Value> {
        let raw = Pass::value(self, landing.value, landing.held)?;
        Ok(from_heap(&raw, landing.dtype, landing.held))
    }

    /// The frame's passes share its readback buffer: one map reads them all.
    fn readable(
        passes: Vec<engine_gpu::guest::defer::Recorded<Self>>,
        then: engine_gpu::guest::defer::Then<Self>,
    ) {
        let Some(outbox) = passes
            .first()
            .map(|recorded| std::sync::Arc::clone(recorded.pass().outbox()))
        else {
            then(passes, None);
            return;
        };
        outbox.collect_async(move |unread| then(passes, unread));
    }
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

    /// Records the pass for the caller to submit, instead of running each
    /// stage to completion: into `frame`, which the step's earlier passes
    /// recorded into, or a frame of its own.
    #[must_use]
    pub fn deferred(mut self, frame: Option<Frame>) -> OnDevice<'a> {
        self.deferral = Some(Deferral {
            frame,
            ..Deferral::default()
        });
        self
    }

    pub fn take_deferral(&mut self) -> Option<Deferral> {
        self.deferral.take()
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
        if self
            .deferral
            .as_ref()
            .is_some_and(|deferral| deferral.pass.is_some() || package.plans.len() != 1)
        {
            return Err(Fault::Program {
                at: "guest::run",
                why: "a pass recorded to land later runs one stage".into(),
            });
        }
        let mut frame = match self.deferral.as_mut().and_then(|d| d.frame.take()) {
            Some(frame) => frame,
            None => self.device.frame()?,
        };
        // The stage's bf16 rows, widened together once its roots are seen:
        // a seat, the byte there, the byte into the heap, and the values.
        let mut widened: Vec<(&'a crate::device::Buffer, u64, u64, u32)> = Vec::new();
        let heap_of = self
            .session
            .heap(at)
            .map(|(heap, base)| (heap.clone(), base))
            .ok_or_else(|| Fault::Program {
                at: "guest::run",
                why: format!("stage {at} has no heap"),
            })?;

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
                let (heap, base) = (&heap_of.0, heap_of.1);
                let (into, _) = self
                    .session
                    .span(at, local)
                    .expect("the span was just read");
                let mut into = base + into;
                for (seat, values) in rows {
                    if seat.dtype != Dtype::Bf16 {
                        frame.copy(seat.seat, seat.at, heap, into, u64::from(values) * 4)?;
                    } else {
                        widened.push((seat.seat, seat.at, into, values));
                    }
                    into += u64::from(values) * 4;
                }
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

        let rows: Vec<super::widen::Row<'_>> = widened
            .iter()
            .map(|&(src, src_at, dst_at, lanes)| super::widen::Row {
                src,
                src_at,
                dst: &heap_of.0,
                dst_at,
                lanes,
            })
            .collect();
        self.widen.encode(&frame, &rows)?;
        self.session.flush(&mut frame, at)?;
        self.session.dispatch(&frame, at)?;
        // A value is read back when any channel it goes to is one the device
        // does not keep.
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
                self.session.keep(&mut frame, at, local, channel)?;
            } else {
                self.session.forget(channel);
                landed.push((put.value, channel));
            }
        }
        let kept = |global: u32| !landed.iter().any(|&(value, _)| value == global);
        let mut read: Vec<u32> = Vec::new();
        for &global in wanted {
            if kept(global) {
                continue;
            }
            if let Some(local) = local_of(package, at, global)?
                && !read.contains(&local)
            {
                read.push(local);
            }
        }
        let pass = self.session.read_back(&mut frame, at, &read)?;
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
                let (_, held) = self
                    .session
                    .span(at, local)
                    .expect("a wanted value has a span");
                for &(_, channel) in landed.iter().filter(|&&(value, _)| value == global) {
                    deferral.landings.push(Landing {
                        channel,
                        value: local,
                        dtype,
                        held,
                    });
                }
            }
            deferral.frame = Some(frame);
            deferral.pass = Some(pass);
            self.ran.push(at);
            return Ok(());
        }
        let pending = frame.commit_async(None)?;
        pass.outbox().collect()?;
        pending.wait()?;

        let status = pass.status();
        if status.is_err() {
            self.session.clear_status()?;
        }
        if let Some(code) = status? {
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
            let raw = pass.value(local, held)?;
            tracing::trace!(
                stage = at,
                global,
                ?dtype,
                bytes = raw.len(),
                head = ?&raw[..raw.len().min(16)],
                "guest value taken"
            );
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

    fn keeps(&self, channel: u32) -> bool {
        self.session.keeps(channel)
    }

    fn binds(&self, root: &LaunchValue) -> bool {
        self.offered(root).is_some()
    }
}
