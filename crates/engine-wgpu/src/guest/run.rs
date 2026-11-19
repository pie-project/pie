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
        // The set its bind checked against the stage's heap.
        let locals = super::session::device_puts(package, at)?;
        self.session.read_back(&mut frame, at, &locals)?;
        let pending = frame.commit_async(None)?;
        self.session.collect(at)?;
        pending.wait()?;

        let status = self.session.status(at);
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
            let (_, held) = self
                .session
                .span(at, local)
                .expect("a wanted value has a span");
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
