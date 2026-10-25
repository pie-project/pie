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
use eta_ir::registry::GeometryClass;

use crate::device::Context;
use crate::device::ctx::Frame;
use crate::error::{Fault, Result};

pub use compile::{Cache, Compiled, Module, Region, Stage, Target};
pub use launch::{ChannelShape, Cursor, Prepared, Rings};
pub use session::{Blocked, Fired, Launched, Session, seeds_of};
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

    pub fn bind(
        &mut self,
        context: &Context,
        program_id: ProgramId,
        seeds: &[(u32, Vec<u8>)],
        extents: Extents,
        geometry: GeometryClass,
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
            .envelope(&program.plan, bound.geometry)
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

    /// Points `intrinsic` of the pass's lane `lane` at `offset` of `base`.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_intrinsic(
        &mut self,
        id: InstanceId,
        intrinsic: eta_ir::op::IntrinsicId,
        lane: u32,
        base: &crate::device::Buffer,
        offset: u64,
        width: u32,
        dtype: eta_ir::Dtype,
    ) -> Result<()> {
        self.instances
            .get_mut(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?
            .session
            .bind_intrinsic(intrinsic, lane, base, offset, width, dtype)
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

    pub fn fire(&mut self, context: &Context, id: InstanceId) -> Result<Fired> {
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
        bound
            .session
            .fire(context, &self.pipelines, &program.compiled, &program.plan)
    }

    pub fn stage_into(&mut self, frame: &Frame, id: InstanceId) -> Result<Launched> {
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
        bound
            .session
            .stage_into(frame, &program.compiled, &program.plan)
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
                if stage.regions.is_empty() {
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

    #[must_use]
    pub fn is_airborne(&self, id: InstanceId) -> bool {
        self.instances
            .get(&id)
            .is_some_and(|bound| bound.session.is_airborne())
    }

    /// Whether instance `id`'s program reads `intrinsic` of its pass's lane `lane`.
    pub fn reads_lane(
        &self,
        id: InstanceId,
        intrinsic: eta_ir::op::IntrinsicId,
        lane: u32,
    ) -> Result<bool> {
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
            .package
            .values
            .iter()
            .any(|value| value.intrinsic == Some(intrinsic) && value.lane == lane))
    }

    pub fn needs_logits(&self, id: InstanceId) -> Result<bool> {
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
            .reads_intrinsic(eta_ir::op::IntrinsicId::Logits))
    }

    pub fn needs_mtp_logits(&self, id: InstanceId) -> Result<bool> {
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
        Ok(program.plan.needs_mtp_logits)
    }

    pub fn needs_mtp_drafts(&self, id: InstanceId) -> Result<bool> {
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
        Ok(program.plan.needs_mtp_drafts)
    }

    pub fn needs_attn_scores(&self, id: InstanceId) -> Result<bool> {
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
        Ok(program.plan.needs_attn_scores)
    }

    #[must_use]
    pub fn declared_score_planes(&self, id: InstanceId) -> Option<u32> {
        let bound = self.instances.get(&id)?;
        let program = self.programs.get(&bound.program_id)?;
        program
            .plan
            .package
            .values
            .iter()
            .filter(|value| value.intrinsic == Some(eta_ir::op::IntrinsicId::AttnScore))
            .filter_map(|value| value.shape.first().copied())
            .max()
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
        Ok(())
    }
}

#[derive(Debug)]
struct Bound {
    program_id: ProgramId,
    session: Session,
    geometry: GeometryClass,
    shared: Vec<Option<Arc<SharedRing>>>,
    ids: Vec<ChannelId>,
}

fn release_seats(shared: &[Option<Arc<SharedRing>>]) {
    for ring in shared.iter().flatten() {
        ring.detach();
    }
}
