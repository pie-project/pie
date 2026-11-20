use eta_compiler::codegen::launch::{LaunchPackage, LaunchValue, ValueOrigin};
use eta_exec::{ExecPlan, StagePlan, StageRunner, Value};
use eta_ir::Dtype;

use super::session::{Pass, Session};
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
    /// The pass's berth in the guest ring, which the frame lands its status
    /// and the host's puts into.
    pub pass: Option<Pass>,
    pub landings: Vec<Landing>,
}

/// A put the host reads, landing in the pass's berth.
#[derive(Clone, Copy, Debug)]
pub struct Landing {
    pub channel: u32,
    /// Its value in the stage.
    pub value: u32,
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
        // The pass's berth: each root the host may stage, its status, and
        // each value the host may read back.
        let mut staged = Vec::new();
        for &global in roots {
            if package.values[global as usize].source != ValueOrigin::Output
                && let Some(local) = local_of(package, at, global)?
                && !staged.contains(&local)
            {
                staged.push(local);
            }
        }
        let mut read = Vec::new();
        for &global in wanted {
            if let Some(local) = local_of(package, at, global)?
                && !read.contains(&local)
            {
                read.push(local);
            }
        }
        let mut pass = self.session.pass(at, &staged, &read)?;
        let mut frame = match self.deferral.as_mut().and_then(|d| d.frame.take()) {
            Some(frame) => frame,
            None => self.device.frame()?,
        };

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
                            .encode(&frame, seat.seat, seat.at, heap, into, values)?;
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
            self.session.stage_in(&mut pass, at, local, &bytes)?;
        }

        self.session.flush(&mut frame, &pass, at)?;
        self.session.dispatch(&frame, at)?;
        self.session.read_status(&mut frame, &mut pass)?;
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
        for (index, &global) in wanted.iter().enumerate() {
            if kept(global) || wanted[..index].contains(&global) {
                continue;
            }
            if let Some(local) = local_of(package, at, global)? {
                self.session.read_back(&mut frame, &mut pass, at, local)?;
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
        frame.commit()?;

        if let Some(code) = pass.status()? {
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
            let raw = pass.value(local)?;
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

/// The value a landed pass read back for `landing`.
pub fn landed_value(landed: &Pass, landing: &Landing) -> Result<Value> {
    let (at, bytes) = landed
        .read_at(landing.value)
        .ok_or_else(|| Fault::Program {
            at: "guest::run",
            why: format!("value {} was not read back by its pass", landing.value),
        })?;
    let raw = landed.read(at, bytes.min(landing.held))?;
    Ok(from_heap(&raw, landing.dtype, landing.held))
}
