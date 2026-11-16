use eta_compiler::codegen::launch::{LaunchPackage, LaunchValue, ValueOrigin};
use eta_exec::{ExecPlan, StagePlan, StageRunner, Value};
use eta_ir::Dtype;

use super::session::Session;
use super::widen::Widen;
use crate::device::Context;
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
                        "this pass reads output {:?} of lane {} and the fire offered no such row",
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
            let bytes = root_bytes(
                root,
                &vals[global as usize],
                self.session.bytes(at, local).unwrap_or(0),
            );
            self.session.stage_in(at, local, &bytes)?;
        }

        self.session.flush(&mut frame, at)?;
        self.session.dispatch(&frame, at)?;
        let mut locals = Vec::with_capacity(wanted.len());
        for &global in wanted {
            if let Some(local) = local_of(package, at, global)? {
                locals.push(local);
            }
        }
        self.session.read_back(&mut frame, at, &locals)?;
        let pending = frame.commit_async(None)?;
        pending.wait()?;

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
            let held = self.session.bytes(at, local).unwrap_or(0);
            let raw = self.session.taken(at, local)?;
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
        self.session.close();
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

    fn binds(&self, root: &LaunchValue) -> bool {
        self.offered(root).is_some()
    }
}
