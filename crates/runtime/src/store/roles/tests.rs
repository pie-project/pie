//! Instance and channel slots against a fake engine that states both
//! shapes: binds and closes give their slots back, a shared ring outlives
//! its channel until the last instance on it closes, a park and unpark move
//! an item through the host tier and tell the engine its new refs, and the
//! record math is the same with the shapes as without them. A ring whose
//! cell is past a slot lies in one run of adjacent slots, never split, and
//! host writes into a parked item wait for its rebind. An adapter is keyed
//! by its content: instances that land the same one share one set of
//! `Adapter` slots, freed at the last close, parked only once every holder
//! is, and rebound for each holder where it lands.

use std::sync::{Arc, Mutex};

use ::engine::chunk::{
    ChunkFacts, Growth, Reserved, ReservedRun, SegmentFacts, ShapeId as EngineShape, SlotKind,
    SlotRole, SlotShape,
};
use ::engine::{
    BoundInstance, ChannelId, ChannelRegistration, Engine, Footprint, FrameSubmission, FrameTicket,
    InstanceBinding, InstanceId, LoadRequest, Loaded, ProgramFacts, ProgramId, Rebind,
    RegisteredChannel, RingFacts,
};

use super::{
    Owner, Pump, Roles, RolesError, RolesHandle, bind_instance, learn_program, pump, rebind,
    register_channel,
};
use crate::store::shared::{Shared, SharedPool};

const KIB: u64 = 1 << 10;
const MIB: u64 = 1 << 20;
const CHUNK: u64 = MIB;
const LIST: EngineShape = EngineShape(0);
const RECORD: EngineShape = EngineShape(1);
const INSTANCE: EngineShape = EngineShape(2);
const CHANNEL: EngineShape = EngineShape(3);
const ADAPTER: EngineShape = EngineShape(4);
pub(crate) const PROGRAM: ProgramId = ProgramId(7);

fn shape(id: EngineShape, kind: SlotKind, slot_bytes: u64, role: SlotRole) -> SlotShape {
    SlotShape {
        id,
        kind,
        slot_bytes,
        slots_per_chunk: u32::try_from(CHUNK / slot_bytes).unwrap(),
        sets_per_working_set: 1,
        rows: Vec::new(),
        pinned: None,
        role,
    }
}

/// 32 chunks of 1 MiB, a kv list and a one-piece record, and (`roles`) a
/// 4 KiB `Instance`, a 1 KiB `Channel` and a 64 KiB `Adapter` shape; 8
/// host chunks.
fn facts(roles: bool) -> ChunkFacts {
    facts_with_list(roles, 64 * KIB)
}

/// [`facts`] with a kv list slot of `list` bytes.
fn facts_with_list(roles: bool, list: u64) -> ChunkFacts {
    let mut shapes = vec![
        shape(
            LIST,
            SlotKind::List {
                tokens: 16,
                rate: 1,
                window: None,
            },
            list,
            SlotRole::State,
        ),
        shape(
            RECORD,
            SlotKind::Fixed { pieces: 1 },
            256 * KIB,
            SlotRole::State,
        ),
    ];
    if roles {
        shapes.push(shape(
            INSTANCE,
            SlotKind::Fixed { pieces: 1 },
            4 * KIB,
            SlotRole::Instance,
        ));
        shapes.push(shape(
            CHANNEL,
            SlotKind::Fixed { pieces: 1 },
            KIB,
            SlotRole::Channel,
        ));
        shapes.push(shape(
            ADAPTER,
            SlotKind::Fixed { pieces: 1 },
            64 * KIB,
            SlotRole::Adapter,
        ));
    }
    let facts = ChunkFacts {
        chunk_bytes: CHUNK,
        total_chunks: 32,
        segment_chunks: 32,
        segments: SegmentFacts {
            committed: 1,
            max: 1,
            growth: Growth::Fixed,
        },
        reserved: vec![ReservedRun {
            what: Reserved::Null,
            first: 0,
            chunks: 1,
        }],
        shapes,
        act_chunks: vec![(16, 1), (256, 4)],
        host_chunks: 8,
        ..ChunkFacts::default()
    };
    facts.validate().unwrap();
    facts
}

pub(crate) struct Pools {
    pub(crate) device: Shared,
    pub(crate) host: Shared,
    pub(crate) roles: RolesHandle,
}

pub(crate) fn pools() -> Pools {
    let facts = facts(true);
    let device = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let host = Arc::new(Mutex::new(
        SharedPool::tier(&facts, facts.host_chunks)
            .unwrap()
            .unwrap(),
    ));
    let roles = Roles::new(&device, Some(&host)).expect("the pool states both shapes");
    Pools {
        device,
        host,
        roles: Arc::new(Mutex::new(roles)),
    }
}

/// Slots held in a pool, in bytes.
fn held(pool: &Shared) -> u64 {
    pool.lock().unwrap().pool().demand().live
}

/// The engine half: states an instance of 5000 bytes in three slots (its
/// cells packed whole, so more than the bytes alone take), one ring of its
/// own on channel index 1, and a channel ring of two slots; records what it
/// is bound with and rebound to.
#[derive(Default)]
pub(crate) struct FakeEngine {
    next: u64,
    pub(crate) bindings: Vec<InstanceBinding>,
    registrations: Vec<ChannelRegistration>,
    rebinds: Vec<Rebind>,
    /// What `ring_facts` states in place of the two-slot ring.
    ring: Option<RingFacts>,
    /// The instance's tables must be one run.
    contiguous_tables: bool,
    /// Answers `register_channel` with `Unsupported`, stating no ring for
    /// it: the runtime keeps such a channel alone.
    no_channels: bool,
    /// The adapter an instance lands, where the program carries one.
    adapter: Footprint,
    /// The key of the adapter the next binding lands ([`Programs::adapter_key`]).
    key: Option<u128>,
    /// The instances it closed: as the engine's plane, it holds them no
    /// longer and refuses a rebind of one by name.
    closed: Vec<InstanceId>,
    /// Channels closed while an instance was still bound on them: kept
    /// registered until the last such instance closes
    /// (`Channels::close`).
    closing: Vec<ChannelId>,
    /// Channels released: a rebind of one is refused by name, as an engine
    /// that no longer holds the channel refuses it.
    released: Vec<ChannelId>,
}

impl FakeEngine {
    /// Whether an instance it has not closed is bound on channel `id`
    /// (instance `k + 1` is the `k`th binding).
    fn bound_on(&self, id: ChannelId) -> bool {
        self.bindings.iter().enumerate().any(|(k, binding)| {
            binding.channels.contains(&id) && !self.closed.contains(&InstanceId(k as u64 + 1))
        })
    }
}

impl ::engine::Engine for FakeEngine {
    fn committed_covers(&self) -> &'static [&'static str] {
        &[]
    }

    fn measured_at_load(&self) -> Vec<String> {

        Vec::new()

    }

    fn skips_measuring(&self) -> bool {
        false
    }

    fn kind(&self) -> &'static str {
        "fake-roles"
    }

    fn load(&mut self, _request: LoadRequest) -> ::engine::Result<Loaded> {
        Err(::engine::Error::Load("no model".into()))
    }

    fn submit(&mut self, _frame: &FrameSubmission) -> ::engine::Result<FrameTicket> {
        unreachable!("nothing fires here")
    }

    fn settle_frame(&mut self, _ticket: &mut ::engine::FrameTicket) -> ::engine::Result<()> {
        Ok(())
    }

    fn programs(&mut self) -> Option<&mut dyn ::engine::Programs> {
        Some(self)
    }

    fn programs_ref(&self) -> Option<&dyn ::engine::Programs> {
        Some(self)
    }

    fn channels(&mut self) -> Option<&mut dyn ::engine::Channels> {
        Some(self)
    }

    fn channels_ref(&self) -> Option<&dyn ::engine::Channels> {
        Some(self)
    }

    fn pool(&mut self) -> Option<&mut dyn ::engine::Pool> {
        Some(self)
    }

    fn tiers(&mut self) -> Option<&mut dyn ::engine::Tiers> {
        Some(self)
    }
}

impl ::engine::Programs for FakeEngine {
    fn prepare(
        &mut self,
        _registration: &::engine::program::ProgramRegistration,
    ) -> ::engine::Result<::engine::ProgramCompile> {
        Err(::engine::Error::unsupported(self.kind(), "prepare_program"))
    }

    fn finish(
        &mut self,
        _prepared: ::engine::Prepared,
    ) -> ::engine::Result<::engine::ProgramId> {
        Err(::engine::Error::unsupported(self.kind(), "finish_program"))
    }

    fn close_program(&mut self, _program: ::engine::ProgramId) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "close_program"))
    }

    fn facts(&self, program: ProgramId) -> ::engine::Result<ProgramFacts> {
        assert_eq!(program, PROGRAM);
        Ok(ProgramFacts {
            instance: Footprint {
                bytes: 5000,
                slots: 3,
                contiguous: self.contiguous_tables,
            },
            rings: vec![
                RingFacts {
                    channel: 0,
                    cell_bytes: 64,
                    capacity: 4,
                    footprint: Footprint {
                        bytes: 352,
                        slots: 1,
                        contiguous: false,
                    },
                },
                RingFacts {
                    channel: 1,
                    cell_bytes: 600,
                    capacity: 2,
                    footprint: Footprint {
                        bytes: 1832,
                        slots: 3,
                        contiguous: false,
                    },
                },
            ],
            adapter: self.adapter,
        })
    }

    fn adapter_key(&self, _binding: &InstanceBinding) -> ::engine::Result<Option<u128>> {
        Ok(self.key)
    }

    fn bind(&mut self, binding: &InstanceBinding) -> ::engine::Result<BoundInstance> {
        self.next += 1;
        self.bindings.push(binding.clone());
        Ok(BoundInstance {
            id: InstanceId(self.next),
            program: binding.program,
            geometry: binding.geometry,
            standing_guest_bytes: 0,
        })
    }

    fn rebind(&mut self, rebind: Rebind) -> ::engine::Result<()> {
        if let Rebind::Instance { id, .. } = &rebind
            && self.closed.contains(id)
        {
            return Err(::engine::Error::Program(format!(
                "program::plane: no instance {id}"
            )));
        }
        if let Rebind::Channel { id, .. } = &rebind
            && self.released.contains(id)
        {
            return Err(::engine::Error::Program(format!(
                "channel {id} is not registered on this engine"
            )));
        }
        self.rebinds.push(rebind);
        Ok(())
    }

    fn close(&mut self, id: InstanceId) -> ::engine::Result<u64> {
        self.closed.push(id);
        let closing = std::mem::take(&mut self.closing);
        for channel in closing {
            if self.bound_on(channel) {
                self.closing.push(channel);
            } else {
                self.released.push(channel);
            }
        }
        Ok(0)
    }

    fn register(&mut self, _registration: &::engine::ProgramRegistration) -> ::engine::Result<::engine::ProgramId> {
        Err(::engine::Error::unsupported(self.kind(), "register_program"))
    }

    fn adapter(&mut self, _registration: &::engine::adapter::AdapterRegistration) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "register_adapter"))
    }
}

impl ::engine::Channels for FakeEngine {
    fn ring_facts(
        &self,
        _registration: &ChannelRegistration,
    ) -> ::engine::Result<Option<RingFacts>> {
        if self.no_channels {
            return Ok(None);
        }
        if let Some(ring) = self.ring {
            return Ok(Some(ring));
        }
        Ok(Some(RingFacts {
            channel: 0,
            cell_bytes: 64,
            capacity: 20,
            footprint: Footprint {
                bytes: 1376,
                slots: 2,
                contiguous: false,
            },
        }))
    }

    fn holds(&self, _registration: &ChannelRegistration) -> bool {
        !self.no_channels
    }

    fn open(
        &mut self,
        registration: &ChannelRegistration,
    ) -> ::engine::Result<RegisteredChannel> {
        if self.no_channels {
            return Err(::engine::Error::unsupported(
                self.kind(),
                "register_channel",
            ));
        }
        self.registrations.push(registration.clone());
        Ok(RegisteredChannel {
            id: registration.id,
            reader_wait_id: 0,
            writer_wait_id: 0,
            mirror: None,
        })
    }

    fn close(&mut self, id: ChannelId) -> ::engine::Result<()> {
        if self.bound_on(id) {
            self.closing.push(id);
        } else {
            self.released.push(id);
        }
        Ok(())
    }

    fn publish(&mut self, _instance: ::engine::InstanceId, _channel: u32, _cell: &[u8]) -> ::engine::Result<bool> {
        Err(::engine::Error::unsupported(self.kind(), "publish_channel"))
    }

    fn take(&mut self, _instance: ::engine::InstanceId, _channel: u32) -> ::engine::Result<Option<Vec<u8>>> {
        Err(::engine::Error::unsupported(self.kind(), "take_channel"))
    }
}

impl ::engine::Pool for FakeEngine {
    fn chunk_table(&self) -> ::engine::Result<::engine::chunk::ChunkTable> {
        Err(::engine::Error::unsupported(self.kind(), "chunk_table"))
    }

    fn commit(&mut self, _segments: u32) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "commit_segments"))
    }

    fn release(&mut self, _segments: u32) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "release_segments"))
    }

    fn evacuate(&mut self, _evacuation: &::engine::chunk::Evacuation) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "evacuate"))
    }
}

impl ::engine::Tiers for FakeEngine {
    /// A test engine's copies land inside the call: nothing to wait for.
    fn settle(&mut self) -> ::engine::Result<()> {
        Ok(())
    }

    /// A test engine's copies complete inside the call: their sources are
    /// read by the time it returns.
    fn copy_landing(
        &mut self,
        copy: &::engine::chunk::SlotCopy,
        _token: u64,
    ) -> ::engine::Result<bool> {
        ::engine::Tiers::copy(self, copy)?;
        Ok(true)
    }

    fn copy(&mut self, _copy: &::engine::chunk::SlotCopy) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "copy_slots"))
    }

    fn read(&mut self, _slots: &[(::engine::chunk::ShapeId, Vec<u32>)]) -> ::engine::Result<Vec<u8>> {
        Err(::engine::Error::unsupported(self.kind(), "read_tier_slots"))
    }

    fn write(&mut self, _slots: &[(::engine::chunk::ShapeId, Vec<u32>)], _bytes: &[u8]) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "write_tier_slots"))
    }

    fn copy_kv(&mut self, _copy: &::engine::KvCopy) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "copy_kv"))
    }
}


pub(crate) fn holder(n: u128) -> super::Holder {
    uuid::Uuid::from_u128(n)
}

pub(crate) fn registered(
    roles: &RolesHandle,
    engine: &mut FakeEngine,
    id: u64,
    by: u128,
) -> ChannelId {
    let plan = ChannelRegistration {
        id: ChannelId(id),
        capacity: 20,
        ..ChannelRegistration::default()
    };
    register_channel(
        Some(roles),
        engine,
        &plan,
        Some(holder(by)),
        |engine, plan| Ok(crate::engine::verbs::serve(engine, ::engine::Engine::channels, "register_channel")?.open(plan)?),
    )
    .unwrap();
    plan.id
}

/// Binds an instance of [`PROGRAM`] on `channels` (index 0 a registered
/// channel where given, index 1 the instance's own ring).
fn bind(
    roles: &RolesHandle,
    engine: &mut FakeEngine,
    channels: Vec<ChannelId>,
    by: u128,
) -> InstanceId {
    learn_program(roles, engine, PROGRAM).unwrap();
    let registered: Vec<ChannelId> = engine.registrations.iter().map(|plan| plan.id).collect();
    let binding = InstanceBinding {
        program: PROGRAM,
        channels,
        ..InstanceBinding::default()
    };
    bind_instance(Some(roles), engine, &binding, Some(holder(by)), |id| {
        registered.contains(&id)
    })
    .unwrap()
    .id
}

#[test]
fn a_bind_takes_the_slots_its_facts_count_and_a_close_gives_them_back() {
    let Pools { device, roles, .. } = pools();
    let mut engine = FakeEngine::default();
    let before = held(&device);
    let channel = registered(&roles, &mut engine, 1, 1);
    let ring = engine.registrations[0].slots.clone();
    assert_eq!(ring.len(), 2, "the ring takes the slots ring_facts states");
    let id = bind(&roles, &mut engine, vec![channel, ChannelId(99)], 1);
    let binding = &engine.bindings[0];
    // Three slots, as the engine counts: not ceil(5000 / 4 KiB) = 2.
    assert_eq!(binding.instance_slots.len(), 3);
    // Ring 0 is the registered channel's; ring 1 the instance's own.
    assert_eq!(binding.ring_slots.len(), 1);
    assert_eq!(binding.ring_slots[0].channel, 1);
    assert_eq!(binding.ring_slots[0].slots.len(), 3);
    assert_eq!(held(&device) - before, 3 * 4 * KIB + (3 + 2) * KIB);

    roles.lock().unwrap().close_instance(id);
    assert_eq!(held(&device) - before, 2 * KIB, "the channel's ring stays");
    roles.lock().unwrap().close_channel(channel);
    assert_eq!(held(&device), before, "every slot is back");
}

#[test]
fn a_park_and_unpark_moves_through_the_host_and_rebinds_the_new_refs() {
    let Pools {
        device,
        host,
        roles,
    } = pools();
    let mut engine = FakeEngine::default();
    let before = held(&device);
    let channel = registered(&roles, &mut engine, 1, 1);
    let instance = bind(&roles, &mut engine, vec![channel], 1);
    // Another process's instance on a channel of its own: it stays.
    let other_channel = registered(&roles, &mut engine, 2, 2);
    let other = bind(&roles, &mut engine, vec![other_channel], 2);
    let others = roles.lock().unwrap().slots(Owner::Instance(other));
    let resident = held(&device);
    let was = roles.lock().unwrap().slots(Owner::Instance(instance));

    let park = roles.lock().unwrap().prepare_park(holder(1)).unwrap();
    let copies = park.copies().to_vec();
    assert_eq!(copies.len(), 2, "an instance and a channel shape");
    let instance_copy = copies.iter().find(|copy| copy.shape == INSTANCE).unwrap();
    assert_eq!(&instance_copy.device, &was[..3]);
    roles.lock().unwrap().commit(park);
    let parked = (3 * 4 + 3 + 2) * KIB;
    assert_eq!(
        resident - held(&device),
        parked,
        "its device slots are free"
    );
    assert_eq!(held(&host), parked);
    assert_eq!(roles.lock().unwrap().slots(Owner::Instance(other)), others);
    assert!(roles.lock().unwrap().stale(holder(1)).is_empty());

    // Something else takes the freed slots meanwhile, so the item comes
    // back elsewhere.
    let instance_pool = device.lock().unwrap().roles().instance.unwrap().pool;
    let squatter = device.lock().unwrap().take_refs(instance_pool, 3).unwrap();
    let unpark = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(1))
        .unwrap()
        .unwrap();
    for copy in unpark.copies() {
        assert_eq!(copy.device.len(), copy.host.len());
    }
    roles.lock().unwrap().commit(unpark);
    assert_eq!(held(&host), 0, "the host slots are back");
    let now = roles.lock().unwrap().slots(Owner::Instance(instance));
    assert_ne!(now, was);

    // The restore's rebind: every moved item, its new refs.
    let stale = roles.lock().unwrap().stale(holder(1));
    assert_eq!(stale.len(), 2);
    for rebind in stale {
        let owner = Owner::of(&rebind);
        <FakeEngine as ::engine::Programs>::rebind(&mut engine, rebind).unwrap();
        roles.lock().unwrap().rebound(owner);
    }
    assert!(roles.lock().unwrap().stale(holder(1)).is_empty());
    // Each shape apart: the tables, then its own ring (channel index 1;
    // index 0 is the registered channel's).
    assert!(engine.rebinds.contains(&Rebind::Instance {
        id: instance,
        instance: now[..3].to_vec(),
        rings: vec![::engine::RingSlots {
            channel: 1,
            slots: now[3..].to_vec(),
        }],
        adapter: Vec::new(),
        scratch: ::engine::ScratchPlace::None,
        pair: ::engine::ScratchPlace::None,
        pair_generation: 0,
    }));
    assert!(engine.rebinds.contains(&Rebind::Channel {
        id: channel,
        slots: roles.lock().unwrap().slots(Owner::Channel(channel)),
    }));
    assert!(
        !engine
            .rebinds
            .iter()
            .any(|rebind| Owner::of(rebind) == Owner::Instance(other))
    );

    device.lock().unwrap().give_refs(instance_pool, &squatter);
    for id in [instance, other] {
        roles.lock().unwrap().close_instance(id);
    }
    for id in [channel, other_channel] {
        roles.lock().unwrap().close_channel(id);
    }
    assert_eq!(held(&device), before);
}

/// A ring of `slots` 1 KiB slots (cells past a slot) that must be one run.
fn run_ring(slots: u32) -> RingFacts {
    RingFacts {
        channel: 0,
        cell_bytes: u64::from(slots) * KIB / 2,
        capacity: 1,
        footprint: Footprint {
            bytes: u64::from(slots) * KIB,
            slots,
            contiguous: true,
        },
    }
}

/// 1 KiB `Channel` slots in 1 MiB chunks.
const PER_CHUNK: u32 = 1024;

#[test]
fn a_cell_past_a_chunk_is_refused_by_name() {
    let Pools { device, roles, .. } = pools();
    let mut engine = FakeEngine {
        ring: Some(run_ring(PER_CHUNK + 1)),
        ..FakeEngine::default()
    };
    let before = held(&device);
    let plan = ChannelRegistration {
        id: ChannelId(1),
        ..ChannelRegistration::default()
    };
    let error = register_channel(Some(&roles), &mut engine, &plan, None, |engine, plan| {
        Ok(crate::engine::verbs::serve(engine, ::engine::Engine::channels, "register_channel")?.open(plan)?)
    })
    .unwrap_err();
    assert_eq!(
        error.downcast_ref::<RolesError>(),
        Some(&RolesError::PastChunk {
            role: SlotRole::Channel,
            slots: PER_CHUNK + 1,
            slot_bytes: KIB,
            per_chunk: PER_CHUNK,
        })
    );
    assert!(error.to_string().contains("larger than a chunk"), "{error}");
    assert!(engine.registrations.is_empty(), "the engine is never asked");
    assert_eq!(held(&device), before);
}

#[test]
fn host_writes_into_a_parked_ring_wait_and_replay_in_order_after_the_rebind() {
    let Pools { roles, .. } = pools();
    let mut engine = FakeEngine::default();
    let channel = registered(&roles, &mut engine, 1, 1);
    let instance = bind(&roles, &mut engine, vec![channel], 1);
    let other_channel = registered(&roles, &mut engine, 2, 2);
    let other = bind(&roles, &mut engine, vec![other_channel], 2);
    let written: std::cell::RefCell<Vec<(InstanceId, Pump, u32)>> = Default::default();
    let write = |instance: InstanceId, way: Pump, n: u32| {
        pump(Some(&roles), instance, way, || {
            written.borrow_mut().push((instance, way, n));
            Ok(())
        })
        .unwrap();
    };

    write(instance, Pump::In, 0);
    let park = roles.lock().unwrap().prepare_park(holder(1)).unwrap();
    // Held from the moment the copy starts.
    write(instance, Pump::In, 1);
    roles.lock().unwrap().commit(park);
    write(instance, Pump::Out, 2);
    write(instance, Pump::In, 3);
    // Another process's instance writes on.
    write(other, Pump::In, 4);
    assert_eq!(
        *written.borrow(),
        vec![(instance, Pump::In, 0), (other, Pump::In, 4)]
    );

    let unpark = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(1))
        .unwrap()
        .unwrap();
    roles.lock().unwrap().commit(unpark);
    write(instance, Pump::Out, 5);
    assert_eq!(written.borrow().len(), 2, "back, but not yet rebound");

    let mut replayed = Vec::new();
    let stale = roles.lock().unwrap().stale(holder(1));
    for item in stale {
        rebind(Some(&roles), &mut engine, item, |_, instance, way| {
            replayed.push((instance, way));
            Ok(())
        })
        .unwrap();
    }
    assert_eq!(
        replayed,
        vec![
            (instance, Pump::In),
            (instance, Pump::Out),
            (instance, Pump::In),
            (instance, Pump::Out),
        ],
        "every held write, in the order it was asked, after both rebinds"
    );
    assert_eq!(engine.rebinds.len(), 2);
    write(instance, Pump::In, 6);
    assert_eq!(written.borrow().last(), Some(&(instance, Pump::In, 6)));
}

/// An engine whose program lands an adapter of four planes, a slot each,
/// keyed `key`.
fn adapting(key: u128) -> FakeEngine {
    FakeEngine {
        adapter: Footprint {
            bytes: 4 * 48 * KIB,
            slots: 4,
            contiguous: false,
        },
        key: Some(key),
        ..FakeEngine::default()
    }
}

/// The device bytes an instance of [`PROGRAM`] holds beside its adapter:
/// its tables and its two own rings.
const INSTANCE_BYTES: u64 = 3 * 4 * KIB + (1 + 3) * KIB;
const ADAPTER_BYTES: u64 = 4 * 64 * KIB;

#[test]
fn a_shared_adapter_parks_only_once_every_holder_has() {
    let Pools {
        device,
        host,
        roles,
    } = pools();
    let mut engine = adapting(7);
    let before = held(&device);
    let first = bind(&roles, &mut engine, Vec::new(), 1);
    let second = bind(&roles, &mut engine, Vec::new(), 2);
    let was = engine.bindings[0].adapter_slots.clone();

    // The first holder parks: its tables move, the adapter stays for the
    // second.
    let park = roles.lock().unwrap().prepare_park(holder(1)).unwrap();
    assert!(
        park.copies().iter().all(|copy| copy.shape != ADAPTER),
        "a holder is left on the device"
    );
    roles.lock().unwrap().commit(park);
    assert_eq!(held(&host), INSTANCE_BYTES);
    assert_eq!(roles.lock().unwrap().slots(Owner::Adapter(7)), was);

    // The second parks too: the adapter goes with it, copied, not dropped.
    let park = roles.lock().unwrap().prepare_park(holder(2)).unwrap();
    let copy = park
        .copies()
        .iter()
        .find(|copy| copy.shape == ADAPTER)
        .cloned()
        .expect("the last holder takes the adapter along");
    assert_eq!(copy.device, was);
    roles.lock().unwrap().commit(park);
    assert_eq!(held(&device), before, "parked whole");
    assert_eq!(held(&host), 2 * INSTANCE_BYTES + ADAPTER_BYTES);

    // The first unparks: the adapter comes back with it, elsewhere, and only
    // the holder on the device is rebound now.
    let adapter_pool = device.lock().unwrap().roles().adapter.unwrap().pool;
    let squatter = device.lock().unwrap().take_refs(adapter_pool, 4).unwrap();
    let unpark = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(1))
        .unwrap()
        .unwrap();
    roles.lock().unwrap().commit(unpark);
    let now = roles.lock().unwrap().slots(Owner::Adapter(7));
    assert!(now.iter().all(|slot| !was.contains(slot)));
    let stale = roles.lock().unwrap().stale(holder(1));
    assert_eq!(stale.len(), 1);
    assert!(matches!(
        &stale[0],
        Rebind::Instance { id, adapter, .. } if *id == first && *adapter == now
    ));
    for rebind in stale {
        let owner = Owner::of(&rebind);
        roles.lock().unwrap().rebound(owner);
    }

    // The second unparks later: it is rebound to the same slots.
    let unpark = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(2))
        .unwrap()
        .unwrap();
    assert!(
        unpark.copies().iter().all(|copy| copy.shape != ADAPTER),
        "the adapter is back already"
    );
    roles.lock().unwrap().commit(unpark);
    let stale = roles.lock().unwrap().stale(holder(2));
    assert!(matches!(
        &stale[..],
        [Rebind::Instance { id, adapter, .. }] if *id == second && *adapter == now
    ));
    device.lock().unwrap().give_refs(adapter_pool, &squatter);
    for id in [first, second] {
        roles.lock().unwrap().close_instance(id);
    }
    assert_eq!(held(&device), before);
    assert_eq!(held(&host), 0);
}

/// What the lane does for a `CloseInstance`: the engine closes it, then
/// the roles give its slots back.
fn close_on_lane(roles: &RolesHandle, engine: &mut FakeEngine, id: InstanceId) {
    <FakeEngine as ::engine::Programs>::close(engine, id).unwrap();
    roles.lock().unwrap().close_instance(id);
}

#[test]
fn an_instance_closed_while_evicted_or_coming_back_is_never_rebound() {
    let Pools {
        device,
        host,
        roles,
    } = pools();
    let mut engine = FakeEngine::default();
    let before = held(&device);

    // Closed while parked: its host slots go back, and the restore has
    // nothing of it to rebind.
    let parked = bind(&roles, &mut engine, Vec::new(), 1);
    let park = roles.lock().unwrap().prepare_park(holder(1)).unwrap();
    roles.lock().unwrap().commit(park);
    close_on_lane(&roles, &mut engine, parked);
    assert_eq!(held(&host), 0);
    assert!(
        roles
            .lock()
            .unwrap()
            .prepare_unpark(holder(1))
            .unwrap()
            .is_none()
    );
    assert!(roles.lock().unwrap().stale(holder(1)).is_empty());

    // Closed while its copy back is in flight: once it lands, its slots go
    // back and it is not among the rebinds.
    let moving = bind(&roles, &mut engine, Vec::new(), 2);
    let park = roles.lock().unwrap().prepare_park(holder(2)).unwrap();
    roles.lock().unwrap().commit(park);
    let unpark = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(2))
        .unwrap()
        .unwrap();
    close_on_lane(&roles, &mut engine, moving);
    roles.lock().unwrap().commit(unpark);
    assert!(roles.lock().unwrap().stale(holder(2)).is_empty());
    assert!(engine.rebinds.is_empty());
    assert_eq!(held(&host), 0);
    assert_eq!(held(&device), before);
}
