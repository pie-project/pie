pub mod compile;
pub mod held;
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

use crate::store::Pools;

/// A ring's facts at `channel_slot`-byte `Channel` slots, `per_chunk` a
/// chunk: cells packed whole a slot, or, where a cell is past a slot, one
/// contiguous run (STATE-ROLES.md A+D). `None` where the run would not fit
/// a chunk: the engine keeps that ring.
#[must_use]
pub fn ring_facts(
    channel: u32,
    shape: ChannelShape,
    channel_slot: u64,
    per_chunk: u32,
) -> Option<engine::RingFacts> {
    let stride = shape.cell_stride() as u64;
    let (slots, contiguous) = match shape.ring_slots(channel_slot) {
        Some(slots) => (slots, false),
        None => (
            held::ring_run(shape.cell_count(), stride, channel_slot, per_chunk)?,
            true,
        ),
    };
    Some(engine::RingFacts {
        channel,
        cell_bytes: shape.cell_bytes() as u64,
        capacity: shape.capacity,
        footprint: engine::Footprint {
            bytes: shape.cell_count() * stride,
            slots,
            contiguous,
        },
    })
}

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
    /// The tables of the lanes one fire launches together, by program,
    /// stage, key and frame in flight: laid when a binding opens them, at
    /// the most lanes a fire flies, never grown at a fire.
    batches: BTreeMap<(ProgramId, usize, launch::BatchKey, usize), launch::Batch>,
    /// The most lanes one fire flies, and the frames in flight.
    lanes: u32,
    arms: usize,
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
            lanes: 64,
            arms: 1,
            next_program: 1,
            next_instance: 1,
        }
    }

    /// The device bytes guest programs hold, as (state, fire): shared
    /// channel rings and each instance's rings and launch tables are kept
    /// between fires; the batches of lanes launched together are a fire's.
    #[must_use]
    pub fn bytes(&self) -> (u64, u64) {
        let state = self
            .instances
            .values()
            .map(|bound| bound.session.bytes())
            .sum::<u64>();
        let fire = self.batches.values().map(launch::Batch::bytes).sum();
        (state, fire)
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
        pooled: session::Pooled<'_>,
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
            pooled,
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
                extents,
                session,
                geometry,
                device,
                shared: adopted,
                ids: channels.to_vec(),
            },
        );
        // Its batch tables are laid now, refused here rather than at a fire.
        if let Err(why) = self.seat_batches(context, id) {
            let _ = self.close_instance(id);
            return Err(why);
        }
        Ok(id)
    }

    /// The most lanes one fire flies and the frames in flight: what the
    /// batch tables a binding opens are laid for.
    pub fn set_flight(&mut self, lanes: u32, arms: usize) {
        self.lanes = lanes.max(1);
        self.arms = arms.max(1);
    }

    /// Lays the batch tables instance `id`'s stages fire with, one set a
    /// frame in flight at the most lanes a fire flies, where none is laid.
    fn seat_batches(&mut self, device: &Context, id: InstanceId) -> Result<()> {
        let Some(bound) = self.instances.get_mut(&id) else {
            return Ok(());
        };
        let program_id = bound.program_id;
        let stages = self
            .programs
            .get(&program_id)
            .map_or(0, |program| program.compiled.stages.len());
        let lanes = self.lanes.next_power_of_two();
        for stage in 0..stages {
            let Some(prepared) = bound.session.prepared_mut(stage) else {
                continue;
            };
            let Some(key) = prepared.batch_key() else {
                continue;
            };
            for arm in 0..self.arms {
                let slot = (program_id, stage, key, arm);
                if !self.batches.contains_key(&slot) {
                    self.batches
                        .insert(slot, launch::Batch::build(device, prepared, lanes)?);
                }
            }
        }
        Ok(())
    }

    pub fn register_channel(
        &mut self,
        id: ChannelId,
        shape: ChannelShape,
        pools: &Pools,
        slots: &[u32],
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
            .insert(id, Arc::new(SharedRing::open(shape, pools, slots)?));
        Ok(())
    }

    /// What an instance of `id` holds in `Instance` and `Channel` slots of
    /// `instance_slot` and `channel_slot` bytes. Tables past a slot leave the
    /// instance's tables the engine's (no instance slots); a ring whose cell
    /// is past a slot is the engine's and not listed.
    pub fn facts(&self, id: ProgramId, pools: &Pools) -> Result<engine::ProgramFacts> {
        use crate::store::layout::{CHANNEL, INSTANCE};
        let instance_slot = pools.slot_bytes(INSTANCE)?;
        let instance_per_chunk = pools.slots_per_chunk(INSTANCE)?;
        let channel_slot = pools.slot_bytes(CHANNEL)?;
        let channel_per_chunk = pools.slots_per_chunk(CHANNEL)?;
        let program = self
            .programs
            .get(&id)
            .ok_or_else(|| Fault::program("program::plane", format!("no program {id}")))?;
        let sizes = Session::table_sizes(&program.compiled);
        let layout = held::TableLayout::of(&sizes, instance_slot);
        if layout.slots > instance_per_chunk {
            return Err(Fault::program(
                "program::plane",
                format!(
                    "program {id}'s launch tables take {} instance slots, more than a chunk \
                     holds ({instance_per_chunk})",
                    layout.slots
                ),
            ));
        }
        let instance = engine::Footprint {
            bytes: sizes.iter().sum(),
            slots: layout.slots,
            contiguous: layout.run,
        };
        let rings = program
            .plan
            .package
            .channels
            .iter()
            .enumerate()
            .filter_map(|(channel, declared)| {
                let shape = ChannelShape::of(declared);
                ring_facts(channel as u32, shape, channel_slot, channel_per_chunk)
            })
            .collect();
        Ok(engine::ProgramFacts { instance, rings })
    }

    /// Points every pooled table and ring at the chunks their slots sit in
    /// now: after an evacuation moved chunks under unchanged refs.
    pub fn rebind(&mut self, pools: &Pools) -> Result<()> {
        for bound in self.instances.values_mut() {
            bound.session.rebind(pools)?;
        }
        for ring in self.channels.values() {
            ring.rebind(pools)?;
        }
        Ok(())
    }

    /// Moves an instance's or a channel's pooled tables and rings to the
    /// refs `rebind` names (`Engine::rebind_slots`).
    pub fn rebind_slots(&mut self, pools: &Pools, rebind: &engine::Rebind) -> Result<()> {
        match rebind {
            engine::Rebind::Instance {
                id,
                instance,
                rings,
            } => self
                .instances
                .get_mut(id)
                .ok_or_else(|| Fault::program("program::plane", format!("no instance {id}")))?
                .session
                .retarget(pools, instance, rings),
            engine::Rebind::Channel { id, slots } => self
                .channels
                .get(id)
                .ok_or_else(|| Fault::program("program::plane", format!("no channel {id}")))?
                .retarget(pools, slots),
        }
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
    ) -> Result<(crate::device::Buffer, u64, u64, eta_ir::Dtype)> {
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

    pub fn fire(
        &mut self,
        context: &Context,
        id: InstanceId,
        at: Attach,
        carve: &GuestCarve,
    ) -> Result<Fired> {
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
            carve,
        )
    }

    /// What a fire holding lanes of `program` bound at `extents` lays in
    /// the span's guest region (`engine::Engine::fire_facts`): a lane's
    /// value scratch; the tables stay the instance's and the batch's own.
    pub fn fire_facts(&self, program: ProgramId, extents: &Extents) -> Result<engine::FireFacts> {
        let program = self
            .programs
            .get(&program)
            .ok_or_else(|| Fault::program("program::plane", format!("no program {program}")))?;
        Ok(engine::FireFacts {
            a_lane: Session::scratch_for(&program.compiled, extents)?,
            fixed: 0,
            max_lanes: 0,
        })
    }

    /// Each live (program, extents) pair's fire term, once a pair.
    pub fn live_terms(&self) -> Result<Vec<engine::FireFacts>> {
        let mut pairs: Vec<(ProgramId, Extents)> = Vec::new();
        for bound in self.instances.values() {
            if !pairs.contains(&(bound.program_id, bound.extents)) {
                pairs.push((bound.program_id, bound.extents));
            }
        }
        pairs
            .iter()
            .map(|(program, extents)| self.fire_facts(*program, extents))
            .collect()
    }

    pub fn stage_batched(
        &mut self,
        frame: &mut Frame,
        ids: &[InstanceId],
        carve: &GuestCarve,
        arm: usize,
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
                // Stages fire one after another: each lays its scratch from
                // the region's top afresh.
                let mut room = carve.clone();
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
                            prepared.seat_scratch(room.take(prepared.scratch_bytes())?);
                            #[cfg(target_vendor = "apple")]
                            prepared.zero_scratch_on(frame)?;
                            for region in stage.regions.iter() {
                                prepared.encode_into(frame, region)?;
                            }
                        }
                        continue;
                    };
                    let needed = u32::try_from(group.len()).unwrap_or(u32::MAX);
                    let slot = (program_id, stage_index, key, arm % self.arms);
                    let batch = self
                        .batches
                        .get_mut(&slot)
                        .filter(|batch| batch.lanes() >= needed)
                        .ok_or(Fault::Ceiling {
                            what: "lanes of a batch laid when its binding opened",
                            need: u64::from(needed),
                            have: u64::from(self.lanes),
                        })?;
                    batch.seat_scratch(room.take(batch.scratch_bytes(needed))?);
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
        self.batches.retain(|&(program, _, _, _), _| program != id);
        Ok(())
    }
}

#[derive(Debug)]
struct Bound {
    program_id: ProgramId,
    /// The extents it was bound at: with the program, the pair whose fire
    /// term the runtime counts (`engine::Engine::fire_facts`).
    extents: Extents,
    session: Session,
    geometry: GeometryClass,
    /// The ports the program computes on the device
    /// (`engine::InstanceBinding::device_resolved`).
    device: eta_ir::registry::PortMask,
    shared: Vec<Option<Arc<SharedRing>>>,
    ids: Vec<ChannelId>,
}

/// One fire's guest region: the span's bytes under the model's carve,
/// handed out from the top down (`next`), never under `floor` (the backed
/// span's bottom). A stage starts again from the top, since stages fire one
/// after another and none reads another's scratch.
#[derive(Debug, Clone)]
pub struct GuestCarve {
    buffer: crate::device::Buffer,
    next: u64,
    floor: u64,
}

impl GuestCarve {
    #[must_use]
    pub fn new(buffer: crate::device::Buffer, top: u64, floor: u64) -> GuestCarve {
        GuestCarve {
            buffer,
            next: top,
            floor,
        }
    }

    /// `bytes` of the region, below what was taken.
    pub fn take(&mut self, bytes: u64) -> Result<launch::ScratchAt> {
        let at = self
            .next
            .checked_sub(bytes)
            .filter(|&at| at >= self.floor)
            .ok_or(Fault::Ceiling {
                what: "bytes of the span's guest region the rung backs",
                need: bytes,
                have: self.next.saturating_sub(self.floor),
            })?;
        self.next = at;
        Ok(launch::ScratchAt {
            buffer: self.buffer.clone(),
            offset: at,
        })
    }
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
    const CHUNK: u64 = 16 << 20;

    /// The slots instance `nth` of `facts` takes: its tables from logical
    /// chunk 1, its rings from chunk 2, each instance's apart.
    /// A guest region of its own for fires with no model carve above
    /// them: a buffer, laid from its top.
    fn guests(context: &Context) -> GuestCarve {
        const BYTES: u64 = 64 << 20;
        GuestCarve::new(
            crate::device::Buffer::zeroed(context, BYTES).expect("a guest region"),
            BYTES,
            0,
        )
    }

    fn slots_for(
        pools: &crate::store::Pools,
        facts: &engine::ProgramFacts,
        nth: u32,
    ) -> (Vec<u32>, Vec<engine::RingSlots>) {
        use crate::store::layout::{CHANNEL, INSTANCE};
        let per_instance = pools.slots_per_chunk(INSTANCE).expect("an instance shape");
        let per_channel = pools.slots_per_chunk(CHANNEL).expect("a channel shape");
        let first = per_instance + nth * facts.instance.slots;
        let instance = (first..first + facts.instance.slots).collect();
        let ring_slots: u32 = facts.rings.iter().map(|ring| ring.footprint.slots).sum();
        let mut next = 2 * per_channel + nth * ring_slots;
        let rings = facts
            .rings
            .iter()
            .map(|ring| {
                let slots = (next..next + ring.footprint.slots).collect();
                next += ring.footprint.slots;
                engine::RingSlots {
                    channel: ring.channel,
                    slots,
                }
            })
            .collect();
        (instance, rings)
    }

    /// The registration, the number of channels it binds, its seeds, and
    /// the token channel's index.
    fn registration() -> (ProgramRegistration, usize, Vec<(u32, Vec<u8>)>, u32) {
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
        let taken = traced
            .channel_order()
            .iter()
            .position(|&gid| gid == token.gid())
            .expect("the token channel is in the trace");
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
            u32::try_from(taken).expect("a channel index"),
        )
    }

    /// A program that launched its lanes together leaves no batch behind
    /// once it is closed.
    #[test]
    fn closing_a_program_drops_its_batches() {
        let context = Context::bind().expect("a Metal 4 device");
        let pools = crate::store::Pools::guests_only(&context, CHUNK, 4).expect("a pool");
        let mut plane = Plane::new();
        let (registration, count, seeds, _) = registration();
        let program = plane
            .register(&context, &registration)
            .expect("the program registers");
        let facts = plane.facts(program, &pools).expect("the program's facts");
        let logits =
            crate::device::Buffer::zeroed(&context, u64::from(VOCAB) * 2).expect("a logits plane");
        let lanes: Vec<InstanceId> = (0..LANES)
            .map(|lane| {
                let ids: Vec<ChannelId> = (1..=count as u64)
                    .map(|c| ChannelId(c + 1000 * u64::from(lane)))
                    .collect();
                let (instance, rings) = slots_for(&pools, &facts, lane);
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
                        session::Pooled {
                            pools: &pools,
                            instance: &instance,
                            rings: &rings,
                        },
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
            .stage_batched(&mut frame, &lanes, &guests(&context), 0)
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
            plane
                .batches
                .keys()
                .any(|&(owner, _, _, _)| owner == program),
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

    /// An instance's tables and rings in pool slots fire as another's in
    /// other slots, after an evacuation moved their chunks and after a park
    /// moved them to other refs.
    #[test]
    fn an_instance_in_pool_slots_fires_through_moves() {
        fires_through_moves(false);
    }

    /// As [`an_instance_in_pool_slots_fires_through_moves`] over four
    /// buffers of a chunk each: the instance's tables, its rings and their
    /// moves lie in buffers past the first.
    #[test]
    fn an_instance_in_pool_slots_past_the_first_buffer_fires_through_moves() {
        fires_through_moves(true);
    }

    fn fires_through_moves(over: bool) {
        use crate::store::Pools;
        use crate::store::layout::{CHANNEL, INSTANCE};
        use engine::chunk::Evacuation;

        let context = Context::bind().expect("a Metal 4 device");
        let mut pools = if over {
            crate::store::pool::cap_parts(CHUNK);
            let pools = Pools::guests_over(&context, CHUNK, 4, 1).expect("a pool");
            crate::store::pool::cap_parts(0);
            assert_eq!(pools.parts(), 4, "a buffer a chunk");
            pools
        } else {
            Pools::guests_only(&context, CHUNK, 4).expect("a pool")
        };
        let mut plane = Plane::new();
        let (registration, _, seeds, token) = registration();
        let program = plane
            .register(&context, &registration)
            .expect("the program registers");
        let facts = plane.facts(program, &pools).expect("the program's facts");
        assert!(facts.instance.slots > 0, "the tables take slots: {facts:?}");
        assert!(!facts.rings.is_empty(), "the rings take slots: {facts:?}");
        let per_instance = pools.slots_per_chunk(INSTANCE).expect("an instance shape");
        let per_channel = pools.slots_per_chunk(CHANNEL).expect("a channel shape");
        let (other_instance, other_rings) = slots_for(&pools, &facts, 1);
        let (instance, rings) = slots_for(&pools, &facts, 0);

        let logits =
            crate::device::Buffer::zeroed(&context, u64::from(VOCAB) * 2).expect("a logits plane");
        let extents = Extents {
            row_count: 1,
            sampled_rows: 1,
            query_len: 1,
            ..Extents::default()
        };
        let bind = |plane: &mut Plane, pooled: session::Pooled<'_>| {
            let id = plane
                .bind(
                    &context,
                    program,
                    &seeds,
                    extents,
                    GeometryClass::Host,
                    PortMask::NONE,
                    &[],
                    pooled,
                )
                .expect("the instance binds");
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
        };
        // Two instances in slots of their own: the second is moved below.
        let own = bind(
            &mut plane,
            session::Pooled {
                pools: &pools,
                instance: &other_instance,
                rings: &other_rings,
            },
        );
        let pooled = bind(
            &mut plane,
            session::Pooled {
                pools: &pools,
                instance: &instance,
                rings: &rings,
            },
        );

        let peak = |plane: &mut Plane, at: u32| {
            let mut bf16 = vec![0u8; VOCAB as usize * 2];
            bf16[at as usize * 2..at as usize * 2 + 2].copy_from_slice(&0x3f80u16.to_le_bytes());
            let mut logits = logits.clone();
            logits.write(0, &bf16).expect("the logits write");
            let mut tokens = Vec::new();
            for id in [own, pooled] {
                let fired = plane
                    .fire(&context, id, Attach::Epilogue, &guests(&context))
                    .expect("the stage fires");
                assert!(matches!(fired, Fired::Committed), "{id} fires: {fired:?}");
                let cell = plane
                    .instance_mut(id)
                    .expect("bound")
                    .take(token)
                    .expect("the take")
                    .expect("a token");
                tokens.push(i32::from_le_bytes(cell[..4].try_into().expect("an i32")));
                // The indptr is consumed a fire: put it back.
                let (channel, seed) = &seeds[0];
                plane
                    .instance_mut(id)
                    .expect("bound")
                    .publish(*channel, seed)
                    .expect("the reseed");
            }
            tokens
        };
        assert_eq!(peak(&mut plane, 777), vec![777, 777]);

        // Move both chunks; the refs stand, the plane rebinds.
        let mut frame = context.frame().expect("a frame");
        pools
            .evacuate(
                &mut frame,
                &Evacuation {
                    moves: vec![(1, 3), (2, 1)],
                },
            )
            .expect("the evacuation");
        frame.commit().expect("the copies land");
        plane.rebind(&pools).expect("the rebind");
        assert_eq!(peak(&mut plane, 123), vec![123, 123]);

        // Park and unpark: the bytes copied to fresh refs, the old refs
        // zeroed (from a slot nothing wrote), the instance rebound.
        let moved = |old: &[u32], by: u32| -> Vec<u32> { old.iter().map(|&r| r + by).collect() };
        let fresh_instance = moved(&instance, 64);
        let fresh_rings: Vec<engine::RingSlots> = rings
            .iter()
            .map(|ring| engine::RingSlots {
                channel: ring.channel,
                slots: moved(&ring.slots, 256),
            })
            .collect();
        let mut frame = context.frame().expect("a frame");
        let pairs = |old: &[u32], new: &[u32]| -> Vec<(u32, u32)> {
            old.iter().copied().zip(new.iter().copied()).collect()
        };
        let blank_instance = 2 * per_instance - 1;
        let blank_channel = 3 * per_channel - 1;
        pools
            .copy_shape(&mut frame, INSTANCE, &pairs(&instance, &fresh_instance))
            .expect("the tables copy");
        for (ring, fresh) in rings.iter().zip(&fresh_rings) {
            pools
                .copy_shape(&mut frame, CHANNEL, &pairs(&ring.slots, &fresh.slots))
                .expect("the ring copies");
        }
        frame.commit().expect("the copies land");
        let mut frame = context.frame().expect("a frame");
        let blanks = |old: &[u32], blank: u32| -> Vec<(u32, u32)> {
            old.iter().map(|&r| (blank, r)).collect()
        };
        pools
            .copy_shape(&mut frame, INSTANCE, &blanks(&instance, blank_instance))
            .expect("the old tables zero");
        for ring in &rings {
            pools
                .copy_shape(&mut frame, CHANNEL, &blanks(&ring.slots, blank_channel))
                .expect("the old ring zeroes");
        }
        frame.commit().expect("the zeroes land");
        plane
            .rebind_slots(
                &pools,
                &engine::Rebind::Instance {
                    id: pooled,
                    instance: fresh_instance.clone(),
                    rings: fresh_rings.clone(),
                },
            )
            .expect("the slots rebind");
        assert_eq!(peak(&mut plane, 456), vec![456, 456]);

        // Park to the host tier and back, as a process parked under pressure
        // is: the Instance and Channel slots copy out whole, the device
        // slots are zeroed, and they copy back into the first refs.
        pools.seat_tier(2 * CHUNK);
        let to_host = |shape, slots: &[u32], first: u32| engine::chunk::SlotCopy {
            shape,
            src: engine::MemoryDomain::MetalShared,
            dst: engine::MemoryDomain::HostPinned,
            slots: slots
                .iter()
                .enumerate()
                .map(|(at, &slot)| (slot, first + at as u32))
                .collect(),
            runs: Vec::new(),
        };
        let from_host = |copy: &engine::chunk::SlotCopy, back: &[u32]| engine::chunk::SlotCopy {
            shape: copy.shape,
            src: engine::MemoryDomain::HostPinned,
            dst: engine::MemoryDomain::MetalShared,
            slots: copy
                .slots
                .iter()
                .zip(back)
                .map(|(&(_, host), &slot)| (host, slot))
                .collect(),
            runs: Vec::new(),
        };
        let mut parked = vec![to_host(INSTANCE, &fresh_instance, 0)];
        let mut back = vec![instance.clone()];
        for (at, (ring, fresh)) in rings.iter().zip(&fresh_rings).enumerate() {
            parked.push(to_host(
                CHANNEL,
                &fresh.slots,
                per_channel + at as u32 * fresh.slots.len() as u32,
            ));
            back.push(ring.slots.clone());
        }
        let mut frame = context.frame().expect("a frame");
        for copy in &parked {
            pools.copy_slots(&mut frame, copy).expect("the slots park");
        }
        frame.commit().expect("nothing in flight");
        let mut frame = context.frame().expect("a frame");
        pools
            .copy_shape(
                &mut frame,
                INSTANCE,
                &blanks(&fresh_instance, blank_instance),
            )
            .expect("the parked tables zero");
        for fresh in &fresh_rings {
            pools
                .copy_shape(&mut frame, CHANNEL, &blanks(&fresh.slots, blank_channel))
                .expect("the parked ring zeroes");
        }
        frame.commit().expect("the zeroes land");
        let mut frame = context.frame().expect("a frame");
        for (copy, back) in parked.iter().zip(&back) {
            pools
                .copy_slots(&mut frame, &from_host(copy, back))
                .expect("the slots unpark");
        }
        frame.commit().expect("nothing in flight");
        plane
            .rebind_slots(
                &pools,
                &engine::Rebind::Instance {
                    id: pooled,
                    instance,
                    rings,
                },
            )
            .expect("the slots rebind");
        assert_eq!(peak(&mut plane, 789), vec![789, 789]);
    }

    /// A ring whose cell is past a Channel slot lies in one run of adjacent
    /// slots: what the host writes, the GPU reads at the cells' addresses,
    /// after an evacuation moves the run's chunk and after a park moves it
    /// to another run.
    #[test]
    fn a_ring_past_a_slot_lies_in_one_run() {
        use crate::store::Pools;
        use crate::store::layout::CHANNEL;
        use engine::chunk::Evacuation;

        const CHUNK: u64 = 16 << 20;
        let context = Context::bind().expect("a Metal 4 device");
        let mut pools = Pools::guests_only(&context, CHUNK, 4).expect("a pool");
        // 1024 f32 lanes a cell: 4 KiB, four Channel slots' worth; capacity
        // 3, so four cells.
        let shape = ChannelShape {
            capacity: 3,
            numel: 1024,
            dtype: eta_ir::Dtype::F32,
        };
        let slot = pools.slot_bytes(CHANNEL).expect("a channel shape");
        let per_chunk = pools.slots_per_chunk(CHANNEL).expect("a channel shape");
        let facts = ring_facts(0, shape, slot, per_chunk).expect("the ring fits a chunk");
        assert!(
            SharedRing::open(shape, &pools, &[]).is_err(),
            "a ring that fits a chunk is placed in slots, never in a buffer of its own"
        );
        assert!(
            facts.footprint.contiguous,
            "a cell past a slot is a run: {facts:?}"
        );
        assert_eq!(facts.footprint.slots, 16);
        assert_eq!(shape.ring_slots(slot), None);

        // A ring whose run passes a chunk is refused by name, never held
        // beside the pool.
        let past = ChannelShape {
            capacity: (CHUNK / 2048) as u32,
            ..shape
        };
        assert!(ring_facts(0, past, slot, per_chunk).is_none());
        let refused =
            SharedRing::open(past, &pools, &[]).expect_err("a ring past a chunk is refused");
        assert!(
            format!("{refused}").contains("pass a chunk"),
            "the refusal says why: {refused}"
        );

        // The run in logical chunk 1, not at its start.
        let first = per_chunk + 100;
        let run: Vec<u32> = (first..first + facts.footprint.slots).collect();
        let ring = SharedRing::open(shape, &pools, &run).expect("the ring opens");
        let cell_bytes = shape.cell_bytes();
        let cells = shape.cell_count();
        let pattern = |cell: u64, salt: u8| -> Vec<u8> {
            (0..cell_bytes)
                .map(|at| (at as u64 * 7 + cell * 31) as u8 ^ salt)
                .collect()
        };
        let gpu_reads = |ring: &SharedRing| -> Vec<Vec<u8>> {
            let out = crate::device::Buffer::zeroed(&context, cells * cell_bytes as u64)
                .expect("a readback");
            let mut frame = context.frame().expect("a frame");
            for cell in 0..cells {
                ring.with_cells(|held| {
                    let (span, at) = held.locate(cell)?;
                    frame.copy_span(
                        span.buffer(),
                        span.offset() + at,
                        &out,
                        cell * cell_bytes as u64,
                        cell_bytes as u64,
                    )
                })
                .expect("the cell copies");
            }
            frame.commit().expect("the copies land");
            (0..cells)
                .map(|cell| {
                    let mut got = vec![0u8; cell_bytes];
                    out.read(cell * cell_bytes as u64, &mut got)
                        .expect("the readback");
                    got
                })
                .collect()
        };
        let write_all = |ring: &SharedRing, salt: u8| {
            for cell in 0..cells {
                ring.write_cell(cell, &pattern(cell, salt))
                    .expect("the cell writes");
            }
        };
        let want =
            |salt: u8| -> Vec<Vec<u8>> { (0..cells).map(|cell| pattern(cell, salt)).collect() };

        write_all(&ring, 1);
        assert_eq!(
            gpu_reads(&ring),
            want(1),
            "the GPU reads what the host wrote"
        );

        // Evacuation: logical chunk 1 moves to physical 3 and the unused
        // chunk 2 (zeros) onto physical 1, so the old place no longer holds
        // the cells; the ref stands.
        let mut frame = context.frame().expect("a frame");
        pools
            .evacuate(
                &mut frame,
                &Evacuation {
                    moves: vec![(1, 3), (2, 1)],
                },
            )
            .expect("the evacuation");
        frame.commit().expect("the copies land");
        ring.rebind(&pools).expect("the rebind");
        assert_eq!(gpu_reads(&ring), want(1), "the moved run reads the same");
        write_all(&ring, 2);
        assert_eq!(
            gpu_reads(&ring),
            want(2),
            "and takes host writes where it is now"
        );

        // Park and unpark into another run of chunk 2, the old run zeroed.
        let fresh: Vec<u32> = run.iter().map(|&r| r + per_chunk).collect();
        let blank = 3 * per_chunk - 1;
        let mut frame = context.frame().expect("a frame");
        let pairs: Vec<(u32, u32)> = run.iter().copied().zip(fresh.iter().copied()).collect();
        pools
            .copy_shape(&mut frame, CHANNEL, &pairs)
            .expect("the run copies");
        frame.commit().expect("the copy lands");
        let mut frame = context.frame().expect("a frame");
        let blanks: Vec<(u32, u32)> = run.iter().map(|&r| (blank, r)).collect();
        pools
            .copy_shape(&mut frame, CHANNEL, &blanks)
            .expect("the old run zeroes");
        frame.commit().expect("the zeroes land");
        let mut gapped = fresh.clone();
        gapped[3] += 1;
        assert!(
            ring.retarget(&pools, &gapped).is_err(),
            "a run is adjacent slots"
        );
        ring.retarget(&pools, &fresh).expect("the run moves");
        assert_eq!(gpu_reads(&ring), want(2), "the unparked run reads the same");
    }
}
