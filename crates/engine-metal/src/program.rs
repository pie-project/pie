pub mod compile;
pub mod launch;
pub mod ports;
pub mod session;
pub mod shared;

use std::collections::BTreeMap;
use std::sync::Arc;

use engine::channel::ChannelId;
use engine::ports::Envelope;
use engine::program::{InstanceId, ProgramId, ProgramRegistration};
use eta_exec::adopt_launch_package;
use eta_exec::{ExecPlan, Extents, Versions};
use eta_ir::registry::{GeometryClass, Stage as Attach};

use crate::device::Context;
use crate::device::ctx::Frame;
use crate::error::{Fault, Result};

pub use compile::{Cache, Compiled, Module, Region, Stage, Target};
pub use launch::{ChannelShape, Cursor, Prepared, Rings};
pub use session::{Blocked, Fired, Launched, Session};
pub use shared::{MAX_ATTACHMENTS, SharedRing};

#[derive(Debug)]
pub struct Program {
    pub id: ProgramId,
    pub hash: u64,
    pub plan: ExecPlan,
    pub compiled: Compiled,
}

#[derive(Debug)]
pub struct Plane {
    cache: Cache,
    pipelines: crate::device::Pipelines,
    programs: BTreeMap<ProgramId, Program>,
    by_hash: BTreeMap<u64, ProgramId>,
    instances: BTreeMap<InstanceId, Bound>,
    channels: BTreeMap<ChannelId, Arc<SharedRing>>,
    batches: BTreeMap<(ProgramId, usize, launch::BatchKey), launch::Batch>,
    next_program: u64,
    next_instance: u64,
}

impl Default for Plane {
    fn default() -> Plane {
        Plane::new()
    }
}

impl Plane {
    #[must_use]
    pub fn new() -> Plane {
        Plane {
            cache: Cache::new(),
            pipelines: crate::device::Pipelines::new(),
            programs: BTreeMap::new(),
            by_hash: BTreeMap::new(),
            instances: BTreeMap::new(),
            channels: BTreeMap::new(),
            batches: BTreeMap::new(),
            next_program: 1,
            next_instance: 1,
        }
    }

    #[must_use]
    pub const fn stats(&self) -> eta_exec::CacheStats {
        self.cache.stats()
    }

    pub fn register(
        &mut self,
        context: &Context,
        registration: &ProgramRegistration,
    ) -> Result<ProgramId> {
        if let Some(&existing) = self.by_hash.get(&registration.program_hash) {
            return Ok(existing);
        }
        let plan = adopt_launch_package(registration.launch.clone())?;
        let compiled = self.cache.compile(
            context,
            registration.program_hash,
            &plan,
            &registration.emitted_kernels,
            Versions::from_compiler(registration.emitter_version),
            Target::of(context)?,
        )?;

        let id = ProgramId(self.next_program);
        self.next_program += 1;
        self.programs.insert(
            id,
            Program {
                id,
                hash: registration.program_hash,
                plan,
                compiled,
            },
        );
        self.by_hash.insert(registration.program_hash, id);
        Ok(id)
    }

    #[must_use]
    pub fn program(&self, id: ProgramId) -> Option<&Program> {
        self.programs.get(&id)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        &mut self,
        context: &Context,
        program_id: ProgramId,
        seeds: &[(u32, Vec<u8>)],
        extents: Extents,
        geometry: GeometryClass,
        device: eta_ir::registry::PortMask,
        channels: &[ChannelId],
    ) -> Result<InstanceId> {
        let program = self
            .programs
            .get(&program_id)
            .ok_or_else(|| Fault::program("program::plane", format!("no program {program_id}")))?;
        let adopted = self.seats_for(&program.plan, channels)?;
        let session = match Session::bind(
            context,
            &program.compiled,
            &program.plan,
            seeds,
            extents,
            &adopted,
        ) {
            Ok(session) => session,
            Err(why) => {
                release_seats(&adopted);
                return Err(why);
            }
        };
        let id = InstanceId(self.next_instance);
        self.next_instance += 1;
        self.instances.insert(
            id,
            Bound {
                program_id,
                session,
                geometry,
                device,
                shared: adopted,
                ids: channels.to_vec(),
            },
        );
        Ok(id)
    }

    pub fn register_channel(
        &mut self,
        context: &Context,
        id: ChannelId,
        shape: ChannelShape,
    ) -> Result<()> {
        if self.channels.contains_key(&id) {
            return Err(Fault::program(
                "program::plane",
                format!(
                    "channel {id} is already registered, and a second ring under one \
                     name would leave the instances bound before it addressing \
                     different cells from the ones bound after"
                ),
            ));
        }
        self.channels
            .insert(id, Arc::new(SharedRing::open(context, shape)?));
        Ok(())
    }

    pub fn close_channel(&mut self, id: ChannelId) -> bool {
        self.channels.remove(&id).is_some()
    }

    #[must_use]
    pub fn channel(&self, id: ChannelId) -> Option<&Arc<SharedRing>> {
        self.channels.get(&id)
    }

    pub fn feed_cell(
        &self,
        instance: InstanceId,
        id: ChannelId,
    ) -> Result<(&crate::device::Buffer, u64, u64, eta_ir::Dtype)> {
        let bound = self.instances.get(&instance).ok_or_else(|| {
            Fault::program(
                "program::plane",
                format!("float-port feed of unbound instance {instance}"),
            )
        })?;
        let dense = bound
            .ids
            .iter()
            .position(|&held| held == id)
            .ok_or_else(|| {
                Fault::program(
                    "program::plane",
                    format!(
                        "float-port feed names channel {id}, which instance {instance} does \
                     not carry"
                    ),
                )
            })?;
        bound.session.feed_cell(dense as u32)?.ok_or_else(|| {
            Fault::program(
                "program::plane",
                format!(
                    "float-port feed channel {id} of instance {instance} holds no \
                         committed cell; a port is fed from the cell the instance's own \
                         `take` would read this fire, so publish one before submitting"
                ),
            )
        })
    }

    #[must_use]
    pub fn cohort(&self, instances: &[InstanceId]) -> Vec<InstanceId> {
        let held: Vec<&Arc<SharedRing>> = instances
            .iter()
            .filter_map(|id| self.instances.get(id))
            .flat_map(|bound| bound.shared.iter().flatten())
            .collect();
        if held.is_empty() {
            return Vec::new();
        }
        let mut cohort = Vec::new();
        for (id, bound) in &self.instances {
            if instances.contains(id) {
                continue;
            }
            if bound
                .shared
                .iter()
                .flatten()
                .any(|mine| held.iter().any(|theirs| Arc::ptr_eq(mine, theirs)))
            {
                cohort.push(*id);
            }
        }
        cohort
    }

    fn seats_for(
        &self,
        plan: &ExecPlan,
        channels: &[ChannelId],
    ) -> Result<Vec<Option<Arc<SharedRing>>>> {
        let mut adopted: Vec<Option<Arc<SharedRing>>> =
            Vec::with_capacity(plan.package.channels.len());
        for dense in 0..plan.package.channels.len() {
            let Some(ring) = channels.get(dense).and_then(|id| self.channels.get(id)) else {
                adopted.push(None);
                continue;
            };
            if let Err(why) = ring.attach() {
                release_seats(&adopted);
                return Err(why);
            }
            adopted.push(Some(Arc::clone(ring)));
        }
        Ok(adopted)
    }

    pub fn envelope(&self, id: InstanceId) -> Result<Option<Envelope>> {
        let bound = self
            .instances
            .get(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        if bound.geometry == GeometryClass::Host {
            return Ok(None);
        }
        let program = self.programs.get(&bound.program_id).ok_or_else(|| {
            Fault::program(
                "program::plane",
                format!(
                    "instance {id} names program {}, which is gone",
                    bound.program_id
                ),
            )
        })?;
        bound
            .session
            .envelope(&program.plan, bound.geometry, bound.device)
            .map(Some)
    }

    #[must_use]
    pub fn geometry_of(&self, id: InstanceId) -> Option<GeometryClass> {
        self.instances.get(&id).map(|bound| bound.geometry)
    }

    #[must_use]
    pub fn instance(&self, id: InstanceId) -> Option<&Session> {
        self.instances.get(&id).map(|bound| &bound.session)
    }

    pub fn instance_mut(&mut self, id: InstanceId) -> Option<&mut Session> {
        self.instances.get_mut(&id).map(|bound| &mut bound.session)
    }

    /// Points instance `id`'s reads of output `name` for its lane `lane` at
    /// `offset` of `base`, stored as `dtype`, rows `stride` elements apart.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_output(
        &mut self,
        id: InstanceId,
        name: &str,
        lane: u32,
        base: &crate::device::Buffer,
        offset: u64,
        dtype: eta_ir::Dtype,
        width: u32,
        stride: u32,
    ) -> Result<()> {
        self.instances
            .get_mut(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?
            .session
            .bind_output(name, lane, base, offset, dtype, width, stride)
    }

    pub fn ready(&self, id: InstanceId) -> Result<Option<session::Blocked>> {
        let bound = self
            .instances
            .get(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        let program = self.programs.get(&bound.program_id).ok_or_else(|| {
            Fault::program(
                "program::plane",
                format!(
                    "instance {id} names program {}, which is gone",
                    bound.program_id
                ),
            )
        })?;
        Ok(bound.session.readiness(&program.plan))
    }

    #[must_use]
    pub fn prologue_runs(&self, id: InstanceId) -> bool {
        self.instances
            .get(&id)
            .and_then(|bound| self.programs.get(&bound.program_id))
            .is_some_and(|program| program.plan.prologue_runs())
    }

    pub fn fire(&mut self, context: &Context, id: InstanceId, at: Attach) -> Result<Fired> {
        let bound = self
            .instances
            .get_mut(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        let program = self.programs.get(&bound.program_id).ok_or_else(|| {
            Fault::program(
                "program::plane",
                format!(
                    "instance {id} names program {}, which is gone",
                    bound.program_id
                ),
            )
        })?;
        bound.session.fire(
            context,
            &self.pipelines,
            &program.compiled,
            &program.plan,
            at,
        )
    }

    pub fn stage_batched(
        &mut self,
        device: &Context,
        frame: &mut Frame,
        ids: &[InstanceId],
    ) -> Result<Vec<(InstanceId, Launched)>> {
        let mut results = Vec::with_capacity(ids.len());
        let mut by_program: BTreeMap<ProgramId, Vec<InstanceId>> = BTreeMap::new();
        let mut refused = false;
        for &id in ids {
            let bound = self
                .instances
                .get_mut(&id)
                .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
            let program = self.programs.get(&bound.program_id).ok_or_else(|| {
                Fault::program(
                    "program::plane",
                    format!(
                        "instance {id} names program {}, which is gone",
                        bound.program_id
                    ),
                )
            })?;
            match bound
                .session
                .prepare_airborne(&program.compiled, &program.plan)?
            {
                Some(fired) => {
                    refused = true;
                    results.push((id, Launched::Refused(fired)));
                }
                None => {
                    by_program.entry(bound.program_id).or_default().push(id);
                    results.push((id, Launched::Airborne));
                }
            }
        }
        if refused {
            return Ok(results);
        }
        for (program_id, members) in by_program {
            let program = self.programs.get(&program_id).ok_or_else(|| {
                Fault::program(
                    "program::plane",
                    format!("program {program_id} vanished between staging and encoding"),
                )
            })?;
            for (stage_index, stage) in program.compiled.stages.iter().enumerate() {
                if stage.regions.is_empty() || !program.plan.runs_at(stage_index, Attach::Epilogue)
                {
                    continue;
                }
                let mut groups: BTreeMap<Option<launch::BatchKey>, Vec<&mut launch::Prepared>> =
                    BTreeMap::new();
                for (id, bound) in self.instances.iter_mut() {
                    if !members.contains(id) {
                        continue;
                    }
                    let shares = bound
                        .shared
                        .iter()
                        .flatten()
                        .any(|ring| ring.attachments() > 1);
                    let Some(prepared) = bound.session.prepared_mut(stage_index) else {
                        continue;
                    };
                    let key = if shares { None } else { prepared.batch_key() };
                    groups.entry(key).or_default().push(prepared);
                }
                for (key, mut group) in groups {
                    let Some(key) = key else {
                        for prepared in group {
                            #[cfg(target_vendor = "apple")]
                            prepared.zero_scratch_on(frame)?;
                            #[cfg(not(target_vendor = "apple"))]
                            prepared.zero_scratch()?;
                            for region in stage.regions.iter() {
                                prepared.encode_into(frame, region)?;
                            }
                        }
                        continue;
                    };
                    let needed = u32::try_from(group.len()).unwrap_or(u32::MAX);
                    let slot = (program_id, stage_index, key);
                    let rebuild = self
                        .batches
                        .get(&slot)
                        .is_none_or(|batch| batch.lanes() < needed);
                    if rebuild {
                        let batch =
                            launch::Batch::build(device, group[0], needed.next_power_of_two())?;
                        self.batches.insert(slot, batch);
                    }
                    let batch = self
                        .batches
                        .get_mut(&slot)
                        .expect("inserted or present one statement ago");
                    batch.encode(frame, &stage.regions, &mut group)?;
                }
            }
        }
        Ok(results)
    }

    pub fn settle_launched(&mut self, id: InstanceId) -> Result<Fired> {
        let bound = self
            .instances
            .get_mut(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        let program = self.programs.get(&bound.program_id).ok_or_else(|| {
            Fault::program(
                "program::plane",
                format!(
                    "instance {id} names program {}, which is gone",
                    bound.program_id
                ),
            )
        })?;
        bound.session.settle_launched(&program.plan)
    }

    pub fn abandon_launched(&mut self, id: InstanceId) {
        if let Some(bound) = self.instances.get_mut(&id) {
            bound.session.abandon_launched();
        }
    }

    /// The outputs instance `id`'s program reads, by name and lane.
    pub fn reads(&self, id: InstanceId) -> Result<Vec<(String, u32)>> {
        let bound = self
            .instances
            .get(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        let program = self.programs.get(&bound.program_id).ok_or_else(|| {
            Fault::program(
                "program::plane",
                format!(
                    "instance {id} names program {}, which is gone",
                    bound.program_id
                ),
            )
        })?;
        Ok(program
            .plan
            .outputs()
            .into_iter()
            .map(|(name, lane)| (name.to_string(), lane))
            .collect())
    }

    pub fn close_instance(&mut self, id: InstanceId) -> Result<()> {
        let bound = self
            .instances
            .remove(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?;
        release_seats(&bound.shared);
        Ok(())
    }

    pub fn close_program(&mut self, id: ProgramId) -> Result<()> {
        let program = self
            .programs
            .get(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no program {id}")))?;
        let hash = program.hash;
        let bound = self
            .instances
            .values()
            .filter(|bound| bound.program_id == id)
            .count();
        if bound != 0 {
            return Err(Fault::program(
                "program::plane",
                format!(
                    "program {id} still has {bound} instance(s) bound; ARC would keep their \
                     pipelines alive, so this is a caller that has lost track of its \
                     instances rather than a launch into freed machine code"
                ),
            ));
        }
        self.cache.forget(hash);
        self.by_hash.remove(&hash);
        self.programs.remove(&id);
        self.batches.retain(|&(program, _, _), _| program != id);
        Ok(())
    }
}

#[derive(Debug)]
struct Bound {
    program_id: ProgramId,
    session: Session,
    geometry: GeometryClass,
    /// The ports the program computes on the device
    /// (`engine::InstanceBinding::device_resolved`).
    device: eta_ir::registry::PortMask,
    shared: Vec<Option<Arc<SharedRing>>>,
    ids: Vec<ChannelId>,
}

fn release_seats(shared: &[Option<Arc<SharedRing>>]) {
    for ring in shared.iter().flatten() {
        ring.detach();
    }
}

#[cfg(all(test, target_vendor = "apple"))]
mod tests {
    use super::*;
    use eta_compiler::codegen::program::{Backend, emit_program};
    use eta_compiler::plan::compile_bound;
    use eta_dsl::prelude::*;
    use eta_ir::registry::{Axis, EngineCaps, OutputDecl, Port, PortMask};

    const VOCAB: u32 = 1024;
    const LANES: u32 = 2;

    /// The registration, the number of channels it binds, and its seeds.
    fn registration() -> (ProgramRegistration, usize, Vec<(u32, Vec<u8>)>) {
        let indptr: &'static Channel = Box::leak(Box::new(Channel::from(vec![0u32, 1])));
        let token: &'static Channel =
            Box::leak(Box::new(Channel::new([1], dtype::i32).capacity(4)));
        let mut b = Builder::new(VOCAB, 16);
        let logits = OutputDecl {
            name: "logits".into(),
            axis: Axis::Row,
            width: VOCAB,
            dtype: eta_ir::Dtype::F32,
            layers: 0,
            readout: true,
        };
        b.outputs(vec![vec![logits.clone()]]);
        b.bind_port(Port::EmbedIndptr, indptr);
        b.stage(Attach::Epilogue, move || {
            let logits = intrinsics::output(0, "logits");
            token.put(reshape(reduce_argmax(&logits), [1]));
        });
        token.note_host_take();
        let traced = b.build().expect("the program traces");
        let bound = eta_ir::validate::bind(
            traced.container().clone(),
            vec![vec![logits]],
            EngineCaps {
                lora: true,
                attn_page_mask: true,
            },
        )
        .expect("the program binds");
        let stages = compile_bound(&bound);
        let backend = Backend::parse("metal").expect("the metal codegen backend");
        let count = bound.container.channels.len();
        let at = traced
            .channel_order()
            .iter()
            .position(|&gid| gid == indptr.gid())
            .expect("the indptr channel is in the trace");
        let seed = [0u32, 1].iter().flat_map(|v| v.to_le_bytes()).collect();
        (
            ProgramRegistration {
                program_hash: bound.hash,
                emitted_kernels: emit_program(backend, &stages, &bound),
                emitter_version: backend.emitter_version(),
                launch: eta_compiler::codegen::launch::build(&bound, &stages),
                ..Default::default()
            },
            count,
            vec![(u32::try_from(at).expect("a channel index"), seed)],
        )
    }

    /// A program that launched its lanes together leaves no batch behind
    /// once it is closed.
    #[test]
    fn closing_a_program_drops_its_batches() {
        let context = Context::bind().expect("a Metal 4 device");
        let mut plane = Plane::new();
        let (registration, count, seeds) = registration();
        let program = plane
            .register(&context, &registration)
            .expect("the program registers");
        let logits =
            crate::device::Buffer::zeroed(&context, u64::from(VOCAB) * 2).expect("a logits plane");
        let lanes: Vec<InstanceId> = (0..LANES)
            .map(|lane| {
                let ids: Vec<ChannelId> = (1..=count as u64)
                    .map(|c| ChannelId(c + 1000 * u64::from(lane)))
                    .collect();
                let id = plane
                    .bind(
                        &context,
                        program,
                        &seeds,
                        Extents {
                            row_count: 1,
                            sampled_rows: 1,
                            query_len: 1,
                            ..Extents::default()
                        },
                        GeometryClass::Host,
                        PortMask::NONE,
                        &ids,
                    )
                    .expect("the lane binds");
                plane
                    .bind_output(
                        id,
                        "logits",
                        0,
                        &logits,
                        0,
                        eta_ir::Dtype::Bf16,
                        VOCAB,
                        VOCAB,
                    )
                    .expect("the logits bind");
                id
            })
            .collect();
        let mut frame = context.frame().expect("a frame");
        let launched = plane
            .stage_batched(&context, &mut frame, &lanes)
            .expect("the lanes stage");
        assert!(
            launched
                .iter()
                .all(|(_, launched)| matches!(launched, Launched::Airborne)),
            "every lane launches: {launched:?}"
        );
        frame.commit().expect("the frame lands");
        for &lane in &lanes {
            plane.settle_launched(lane).expect("the lane settles");
        }
        assert!(
            plane.batches.keys().any(|&(owner, _, _)| owner == program),
            "two lanes of one program launch as a batch"
        );
        for lane in lanes {
            plane.close_instance(lane).expect("the lane closes");
        }
        plane.close_program(program).expect("the program closes");
        assert!(
            plane.batches.is_empty(),
            "the closed program's batches are dropped: {} left",
            plane.batches.len()
        );
    }
}
