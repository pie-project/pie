//! An instance's tables, a channel's ring and an instance's adapter in the
//! shared pool (REDESIGN 2.6, `new-memory/STATE-ROLES.md` option A).
//!
//! An engine that states a fixed shape of role [`SlotRole::Instance`],
//! [`SlotRole::Channel`] or [`SlotRole::Adapter`] places a bound instance's
//! launch tables, a ring's cells and the adapter an instance lands in slots
//! of them, which the runtime takes from the pool on demand:
//! [`Engine::program_facts`] and [`Engine::ring_facts`] say how many, and
//! the binding ([`InstanceBinding::instance_slots`],
//! [`InstanceBinding::ring_slots`], [`InstanceBinding::adapter_slots`]) or
//! registration ([`ChannelRegistration::slots`]) names them. They are no
//! part of a record nor of a working set ([`super::shared`]).
//!
//! No such role is replayable ([`SlotRole::replayable`]): no token rebuilds
//! them, so nothing here drops one. Under pressure the policy may park them
//! with their process on the host tier, as it parks the process's kv: a
//! park copies each item to host slots and frees its device slots, an
//! unpark copies it into fresh device slots, and the engine is told the new
//! refs ([`Engine::rebind_slots`]) before the next fire reads them.
//!
//! A ring an instance holds itself is the instance's item. A registered
//! channel's ring is the channel's, shared by every instance bound to it,
//! and freed once the channel is closed and the last of them has closed
//! too. An adapter is keyed by its content ([`Engine::adapter_key`]): every
//! instance whose binding lands the same planes holds the one item, taken
//! at the first bind and freed at the last close; it parks only when every
//! instance holding it parks, and each holder is rebound to where it lands.
//!
//! [`Engine::adapter_key`]: ::engine::Engine::adapter_key
//!
//! An engine that states neither shape has no [`Roles`]: it keeps its
//! tables and rings itself, as before.
//!
//! [`Engine::program_facts`]: ::engine::Engine::program_facts
//! [`Engine::ring_facts`]: ::engine::Engine::ring_facts
//! [`Engine::rebind_slots`]: ::engine::Engine::rebind_slots
//! [`InstanceBinding::instance_slots`]: ::engine::InstanceBinding::instance_slots
//! [`InstanceBinding::ring_slots`]: ::engine::InstanceBinding::ring_slots
//! [`InstanceBinding::adapter_slots`]: ::engine::InstanceBinding::adapter_slots
//! [`ChannelRegistration::slots`]: ::engine::ChannelRegistration::slots

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use ::engine::chunk::{ShapeId as EngineShape, SlotRole};
use ::engine::{
    BoundInstance, ChannelId, ChannelRegistration, Engine, InstanceBinding, InstanceId,
    ProgramFacts, ProgramId, Rebind, RingFacts, RingSlots,
};

use super::shared::{FixedShape, RoleShapes, Shared};
use super::slots::SlotError;

/// Whose slots an item is: an instance's tables (and the rings it holds
/// itself), or a registered channel's ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Owner {
    Instance(InstanceId),
    Channel(ChannelId),
    /// An adapter, by its content key, shared by the instances that land it.
    Adapter(u128),
}

impl Owner {
    /// Whose slots `rebind` moves.
    #[must_use]
    pub fn of(rebind: &Rebind) -> Self {
        match rebind {
            Rebind::Instance { id, .. } => Owner::Instance(*id),
            Rebind::Channel { id, .. } => Owner::Channel(*id),
        }
    }
}

/// A host write the runtime issues into an instance's rings through the
/// engine: cells published into the rings the guest reads
/// ([`::engine::Engine::publish_channel`], `In`), or taken out of the ones
/// it writes ([`::engine::Engine::take_channel`], `Out`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pump {
    In,
    Out,
}

/// The process an item parks and unparks with.
pub type Holder = uuid::Uuid;

/// One engine's instance and channel slots.
pub type RolesHandle = Arc<Mutex<Roles>>;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RolesError {
    #[error("the pool states no {0:?} shape, and the engine asks for its slots")]
    NoShape(SlotRole),
    #[error("the device pool has no room for {slots} {role:?} slots")]
    Short { role: SlotRole, slots: u32 },
    #[error("program {0} was never registered here")]
    Program(ProgramId),
    #[error("channel {0} is parked with its process")]
    Parked(ChannelId),
    #[error("the adapter this binding lands is parked with the instances that hold it")]
    AdapterParked,
    #[error(
        "no chunk has {slots} adjacent free {role:?} slots for an item that must lie in one run"
    )]
    NoRun { role: SlotRole, slots: u32 },
    #[error(
        "a {role:?} item must lie in one run of {slots} adjacent slots of {slot_bytes} bytes, \
         and a chunk holds {per_chunk} of them: its cell is larger than a chunk, which the \
         pool cannot place (new-memory/STATE-ROLES.md option D)"
    )]
    PastChunk {
        role: SlotRole,
        slots: u32,
        slot_bytes: u64,
        per_chunk: u32,
    },
}

/// Where an item's slots lie.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tier {
    Device,
    Host,
    /// A park or unpark is copying it: neither side's slots are free.
    Moving,
}

/// One role's slots of an item, as refs of the tier the item lies on;
/// `contiguous`, adjacent in one chunk ([`::engine::Footprint::contiguous`]),
/// and taken so again on every tier it moves to.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Part {
    role: SlotRole,
    refs: Vec<u32>,
    contiguous: bool,
}

#[derive(Debug)]
struct Item {
    holder: Option<Holder>,
    tier: Tier,
    /// An instance: its tables' part, then each ring it holds itself. A
    /// channel: its ring's part. An adapter: its planes' part.
    parts: Vec<Part>,
    /// An instance: each own ring's channel index, in part order after the
    /// first.
    rings: Vec<u32>,
    /// An instance: the registered channels with slots it is bound to.
    channels: Vec<ChannelId>,
    /// An instance: the adapter it holds, by its key.
    adapter: Option<u128>,
    /// A channel or an adapter: the instances bound to it.
    attached: BTreeSet<InstanceId>,
    /// A channel: still registered. An instance: still bound.
    open: bool,
    /// Its slots moved and the engine has not been told yet.
    stale: bool,
}

impl Item {
    fn new(holder: Option<Holder>, parts: Vec<Part>) -> Self {
        Self {
            holder,
            tier: Tier::Device,
            parts,
            rings: Vec::new(),
            channels: Vec::new(),
            adapter: None,
            attached: BTreeSet::new(),
            open: true,
            stale: false,
        }
    }

    fn refs(&self) -> Vec<u32> {
        self.parts
            .iter()
            .flat_map(|part| part.refs.iter().copied())
            .collect()
    }
}

/// The slots taken for an instance about to bind, before the engine names
/// it: [`Roles::bound`] keeps them, [`Roles::untake`] gives them back.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Taken {
    /// The tables' part, then each own ring's.
    parts: Vec<Part>,
    /// Each own ring's channel index, in part order after the first.
    rings: Vec<u32>,
    /// The registered channels with slots the binding names.
    channels: Vec<ChannelId>,
    /// The adapter the binding lands: its key, its slots, and whether they
    /// were taken for it (the first holder) or are another's.
    adapter: Option<(u128, Part, bool)>,
}

impl Taken {
    /// [`::engine::InstanceBinding::instance_slots`].
    #[must_use]
    pub fn instance_slots(&self) -> Vec<u32> {
        self.parts
            .first()
            .map_or_else(Vec::new, |part| part.refs.clone())
    }

    /// [`::engine::InstanceBinding::ring_slots`].
    #[must_use]
    pub fn ring_slots(&self) -> Vec<RingSlots> {
        own_rings(&self.rings, &self.parts)
    }

    /// [`::engine::InstanceBinding::adapter_slots`].
    #[must_use]
    pub fn adapter_slots(&self) -> Vec<u32> {
        self.adapter
            .as_ref()
            .map_or_else(Vec::new, |(_, part, _)| part.refs.clone())
    }
}

/// The slots taken for a channel about to register:
/// [`Roles::channel_registered`] keeps them, [`Roles::untake_channel`]
/// gives them back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TakenRing(Part);

impl TakenRing {
    /// [`::engine::ChannelRegistration::slots`].
    #[must_use]
    pub fn slots(&self) -> Vec<u32> {
        self.0.refs.clone()
    }
}

/// A park (`out`) or unpark of a process's items: each item's slots on the
/// tier it leaves, paired with the ones taken on the tier it goes to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleMove {
    out: bool,
    items: Vec<(Owner, Vec<Part>)>,
    copies: Vec<RoleCopy>,
}

/// One role shape's slots a move copies: device slot `device[i]` and host
/// slot `host[i]`, whichever way the move goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleCopy {
    pub shape: EngineShape,
    pub device: Vec<u32>,
    pub host: Vec<u32>,
}

impl RoleMove {
    /// The copies the move takes, a role shape each.
    #[must_use]
    pub fn copies(&self) -> &[RoleCopy] {
        &self.copies
    }
}

/// See the module docs.
#[derive(Debug)]
pub struct Roles {
    device: Shared,
    host: Option<Shared>,
    shapes: RoleShapes,
    programs: HashMap<ProgramId, ProgramFacts>,
    items: HashMap<Owner, Item>,
    /// The host writes held while their instance's slots are parked or
    /// not yet rebound, in the order they were asked.
    held: Vec<(InstanceId, Pump)>,
}

impl Roles {
    /// The roles of `device`'s pool, parked on `host`; `None` where the pool
    /// states neither shape.
    #[must_use]
    pub fn new(device: &Shared, host: Option<&Shared>) -> Option<Self> {
        let shapes = device.lock().unwrap().roles();
        if shapes.is_empty() {
            return None;
        }
        Some(Self {
            device: Arc::clone(device),
            host: host.cloned(),
            shapes,
            programs: HashMap::new(),
            items: HashMap::new(),
            held: Vec::new(),
        })
    }

    fn shape(&self, role: SlotRole) -> Result<FixedShape, RolesError> {
        self.shapes.of(role).ok_or(RolesError::NoShape(role))
    }

    /// `slots` of `role` on `on`: any free ones, or (`contiguous`) one run
    /// of adjacent ones in a chunk, never split.
    fn take(
        &self,
        on: &Shared,
        role: SlotRole,
        slots: u32,
        contiguous: bool,
    ) -> Result<Part, RolesError> {
        let part = |refs| Part {
            role,
            refs,
            contiguous,
        };
        if slots == 0 {
            return Ok(part(Vec::new()));
        }
        let shape = self.shape(role)?;
        let mut pool = on.lock().unwrap();
        if !contiguous {
            return pool
                .take_refs(shape.pool, u64::from(slots))
                .map(part)
                .ok_or(RolesError::Short { role, slots });
        }
        match pool.take_run(shape.pool, u64::from(slots)) {
            Ok(refs) => Ok(part(refs)),
            Err(SlotError::RunPastChunk { per_chunk, .. }) => Err(RolesError::PastChunk {
                role,
                slots,
                slot_bytes: shape.slot_bytes,
                per_chunk,
            }),
            Err(_) => Err(RolesError::NoRun { role, slots }),
        }
    }

    fn give(&self, on: &Shared, parts: &[Part]) {
        let mut pool = on.lock().unwrap();
        for part in parts.iter().filter(|part| !part.refs.is_empty()) {
            if let Some(shape) = self.shapes.of(part.role) {
                pool.give_refs(shape.pool, &part.refs);
            }
        }
    }

    /// Takes the slots of every part, all or none.
    fn take_parts(&self, on: &Shared, like: &[Part]) -> Result<Vec<Part>, RolesError> {
        let mut taken = Vec::with_capacity(like.len());
        for part in like {
            match self.take(on, part.role, part.refs.len() as u32, part.contiguous) {
                Ok(part) => taken.push(part),
                Err(error) => {
                    self.give(on, &taken);
                    return Err(error);
                }
            }
        }
        Ok(taken)
    }

    // ---- programs and instances ----

    /// The engine registered `program`, whose instances hold `facts`.
    pub fn program_registered(&mut self, program: ProgramId, facts: ProgramFacts) {
        self.programs.insert(program, facts);
    }

    /// Whether `program`'s facts are known here.
    #[must_use]
    pub fn knows(&self, program: ProgramId) -> bool {
        self.programs.contains_key(&program)
    }

    /// The slots an instance of `program` bound to `channels` takes:
    /// `footprint.slots` of the `Instance` shape (cells are packed whole, so
    /// the engine counts, not the bytes), for each ring of the program no
    /// registered channel covers (`registered`), its slots of the `Channel`
    /// shape, and for the adapter it lands (`adapter`, its key) the slots
    /// of the `Adapter` shape another instance already holds for that key,
    /// else fresh ones.
    pub fn take_instance(
        &mut self,
        program: ProgramId,
        channels: &[ChannelId],
        registered: impl Fn(ChannelId) -> bool,
        adapter: Option<u128>,
    ) -> Result<Taken, RolesError> {
        let facts = self
            .programs
            .get(&program)
            .cloned()
            .ok_or(RolesError::Program(program))?;
        let mut shared = Vec::new();
        for &id in channels {
            if let Some(item) = self.items.get(&Owner::Channel(id)) {
                if item.tier != Tier::Device {
                    return Err(RolesError::Parked(id));
                }
                shared.push(id);
            }
        }
        let device = Arc::clone(&self.device);
        let mut taken = Taken {
            channels: shared,
            ..Taken::default()
        };
        let footprint = facts.instance;
        taken.parts.push(self.take(
            &device,
            SlotRole::Instance,
            footprint.slots,
            footprint.contiguous,
        )?);
        for ring in &facts.rings {
            let covered = channels
                .get(ring.channel as usize)
                .is_some_and(|&id| registered(id));
            if covered {
                continue;
            }
            let footprint = ring.footprint;
            match self.take(
                &device,
                SlotRole::Channel,
                footprint.slots,
                footprint.contiguous,
            ) {
                Ok(part) => {
                    taken.parts.push(part);
                    taken.rings.push(ring.channel);
                }
                Err(error) => {
                    self.give(&device, &taken.parts);
                    return Err(error);
                }
            }
        }
        let footprint = facts.adapter;
        if let Some(key) = adapter.filter(|_| footprint.slots > 0) {
            taken.adapter = match self.items.get(&Owner::Adapter(key)) {
                Some(item) if item.tier == Tier::Device => {
                    Some((key, item.parts[0].clone(), false))
                }
                Some(_) => {
                    self.give(&device, &taken.parts);
                    return Err(RolesError::AdapterParked);
                }
                None => match self.take(
                    &device,
                    SlotRole::Adapter,
                    footprint.slots,
                    footprint.contiguous,
                ) {
                    Ok(part) => Some((key, part, true)),
                    Err(error) => {
                        self.give(&device, &taken.parts);
                        return Err(error);
                    }
                },
            };
        }
        Ok(taken)
    }

    /// The bind of `taken` failed: its slots go back.
    pub fn untake(&mut self, taken: Taken) {
        self.give(&self.device, &taken.parts);
        if let Some((_, part, true)) = taken.adapter {
            self.give(&self.device, &[part]);
        }
    }

    /// The engine bound `taken` as instance `id`, for `holder`.
    pub fn bound(&mut self, id: InstanceId, holder: Option<Holder>, taken: Taken) {
        for &channel in &taken.channels {
            if let Some(item) = self.items.get_mut(&Owner::Channel(channel)) {
                item.attached.insert(id);
            }
        }
        let mut item = Item::new(holder, taken.parts);
        item.rings = taken.rings;
        item.channels = taken.channels;
        if let Some((key, part, _)) = taken.adapter {
            let shared = self
                .items
                .entry(Owner::Adapter(key))
                .or_insert_with(|| {
                    let mut adapter = Item::new(holder, vec![part]);
                    // Held by its instances alone: freed at the last close.
                    adapter.open = false;
                    adapter
                });
            shared.attached.insert(id);
            item.adapter = Some(key);
        }
        if item.parts.iter().all(|part| part.refs.is_empty())
            && item.channels.is_empty()
            && item.adapter.is_none()
        {
            return;
        }
        self.items.insert(Owner::Instance(id), item);
    }

    /// The engine closed instance `id`: its slots go back, and a closed
    /// channel it was the last on with them.
    pub fn close_instance(&mut self, id: InstanceId) {
        let owner = Owner::Instance(id);
        self.held.retain(|&(instance, _)| instance != id);
        let Some(item) = self.items.get_mut(&owner) else {
            return;
        };
        item.open = false;
        let shared: Vec<Owner> = item
            .channels
            .iter()
            .map(|&channel| Owner::Channel(channel))
            .chain(item.adapter.map(Owner::Adapter))
            .collect();
        self.free_if_done(owner);
        for owner in shared {
            if let Some(item) = self.items.get_mut(&owner) {
                item.attached.remove(&id);
            }
            self.free_if_done(owner);
        }
    }

    // ---- channels ----

    /// The slots a channel whose ring `facts` states takes: one run of
    /// adjacent slots where it says `contiguous`.
    pub fn take_channel(&mut self, facts: &RingFacts) -> Result<TakenRing, RolesError> {
        let footprint = facts.footprint;
        self.take(
            &self.device,
            SlotRole::Channel,
            footprint.slots,
            footprint.contiguous,
        )
        .map(TakenRing)
    }

    /// The channel's registration failed: its slots go back.
    pub fn untake_channel(&mut self, taken: TakenRing) {
        self.give(&self.device, &[taken.0]);
    }

    /// The engine registered channel `id` over `taken`, for `holder`.
    pub fn channel_registered(&mut self, id: ChannelId, holder: Option<Holder>, taken: TakenRing) {
        if taken.0.refs.is_empty() {
            return;
        }
        self.items
            .insert(Owner::Channel(id), Item::new(holder, vec![taken.0]));
    }

    /// The engine closed channel `id`: its ring goes back once no bound
    /// instance holds it.
    pub fn close_channel(&mut self, id: ChannelId) {
        let owner = Owner::Channel(id);
        if let Some(item) = self.items.get_mut(&owner) {
            item.open = false;
        }
        self.free_if_done(owner);
    }

    fn free_if_done(&mut self, owner: Owner) {
        let Some(item) = self.items.get(&owner) else {
            return;
        };
        if item.open || !item.attached.is_empty() || item.tier == Tier::Moving {
            return;
        }
        let item = self.items.remove(&owner).expect("present");
        let on = self.tier(item.tier == Tier::Host);
        if let Some(on) = on {
            self.give(&on, &item.parts);
        }
    }

    // ---- parking ----

    /// Whether `owner` parks with `holder`: an instance it holds, a channel
    /// it registered that only its instances are bound to, or an adapter
    /// every holder of which is `holder`'s or parked already.
    fn moves_with(&self, owner: Owner, item: &Item, holder: Holder) -> bool {
        let instance = |id: &InstanceId| self.items.get(&Owner::Instance(*id));
        match owner {
            Owner::Instance(_) => item.holder == Some(holder) && item.open,
            Owner::Channel(_) => {
                item.holder == Some(holder)
                    && item.open
                    && item.attached.iter().all(|id| {
                        instance(id).is_some_and(|instance| instance.holder == Some(holder))
                    })
            }
            Owner::Adapter(_) => {
                item.attached.iter().any(|id| {
                    instance(id).is_some_and(|instance| instance.holder == Some(holder))
                }) && item.attached.iter().all(|id| {
                    instance(id).is_some_and(|instance| {
                        instance.holder == Some(holder) || instance.tier != Tier::Device
                    })
                })
            }
        }
    }

    /// Whether `owner`, parked, unparks with `holder`: what it parked, or
    /// an adapter one of its instances holds.
    fn unparks_with(&self, owner: Owner, item: &Item, holder: Holder) -> bool {
        match owner {
            Owner::Adapter(_) => item.attached.iter().any(|id| {
                self.items
                    .get(&Owner::Instance(*id))
                    .is_some_and(|instance| instance.holder == Some(holder))
            }),
            _ => item.holder == Some(holder),
        }
    }

    /// Parks `holder`'s items on the host: host slots taken for each item
    /// that fits there, to copy into. An item the host lacks room for stays
    /// on the device; nothing is dropped. `None` when nothing moves.
    pub fn prepare_park(&mut self, holder: Holder) -> Option<RoleMove> {
        let host = self.host.clone()?;
        let owners: Vec<Owner> = self
            .items
            .iter()
            .filter(|(owner, item)| {
                item.tier == Tier::Device && self.moves_with(**owner, item, holder)
            })
            .map(|(owner, _)| *owner)
            .collect();
        let mut items = Vec::new();
        for owner in owners {
            let parts = &self.items[&owner].parts;
            match self.take_parts(&host, parts) {
                Ok(to) => {
                    self.items.get_mut(&owner).expect("present").tier = Tier::Moving;
                    items.push((owner, to));
                }
                Err(error) => {
                    tracing::debug!(?owner, %error, "an item stays on the device: host short");
                }
            }
        }
        (!items.is_empty()).then(|| self.moving(true, items))
    }

    /// Unparks `holder`'s parked items: device slots taken for each, all or
    /// none. `None` when nothing is parked.
    pub fn prepare_unpark(&mut self, holder: Holder) -> Result<Option<RoleMove>, RolesError> {
        let owners: Vec<Owner> = self
            .items
            .iter()
            .filter(|(owner, item)| {
                item.tier == Tier::Host && self.unparks_with(**owner, item, holder)
            })
            .map(|(owner, _)| *owner)
            .collect();
        let device = Arc::clone(&self.device);
        let mut items: Vec<(Owner, Vec<Part>)> = Vec::new();
        for owner in owners {
            match self.take_parts(&device, &self.items[&owner].parts) {
                Ok(to) => items.push((owner, to)),
                Err(error) => {
                    for (_, parts) in &items {
                        self.give(&device, parts);
                    }
                    return Err(error);
                }
            }
        }
        for (owner, _) in &items {
            self.items.get_mut(owner).expect("present").tier = Tier::Moving;
        }
        Ok((!items.is_empty()).then(|| self.moving(false, items)))
    }

    /// The move of `items` (each item's slots to take), with the copies
    /// from the slots each holds now.
    fn moving(&self, out: bool, items: Vec<(Owner, Vec<Part>)>) -> RoleMove {
        let mut copies: Vec<RoleCopy> = Vec::new();
        for (owner, to) in &items {
            let from = &self.items[owner].parts;
            for (from, to) in from.iter().zip(to) {
                let Some(shape) = self.shapes.of(from.role) else {
                    continue;
                };
                let (device, host) = if out {
                    (&from.refs, &to.refs)
                } else {
                    (&to.refs, &from.refs)
                };
                let copy = match copies.iter_mut().find(|copy| copy.shape == shape.id) {
                    Some(copy) => copy,
                    None => {
                        copies.push(RoleCopy {
                            shape: shape.id,
                            device: Vec::new(),
                            host: Vec::new(),
                        });
                        copies.last_mut().expect("pushed")
                    }
                };
                copy.device.extend_from_slice(device);
                copy.host.extend_from_slice(host);
            }
        }
        copies.retain(|copy| !copy.device.is_empty());
        RoleMove { out, items, copies }
    }

    /// The copies landed: each item lies at its new slots, and its old ones
    /// go back. An unparked item is stale until [`Self::rebound`].
    pub fn commit(&mut self, txn: RoleMove) {
        let left = self.tier(!txn.out);
        for (owner, parts) in txn.items {
            let Some(item) = self.items.get_mut(&owner) else {
                continue;
            };
            let old = std::mem::replace(&mut item.parts, parts);
            item.tier = if txn.out { Tier::Host } else { Tier::Device };
            item.stale = !txn.out;
            if let Some(left) = &left {
                self.give(left, &old);
            }
            self.free_if_done(owner);
        }
    }

    /// The move failed: each item stays where it was, and the slots taken
    /// for it go back.
    pub fn abort(&mut self, txn: RoleMove) {
        let taken_on = self.tier(txn.out);
        for (owner, parts) in txn.items {
            if let Some(on) = &taken_on {
                self.give(on, &parts);
            }
            if let Some(item) = self.items.get_mut(&owner) {
                item.tier = if txn.out { Tier::Device } else { Tier::Host };
            }
            self.free_if_done(owner);
        }
    }

    /// The host tier (`host`), else the device.
    fn tier(&self, host: bool) -> Option<Shared> {
        if host {
            self.host.clone()
        } else {
            Some(Arc::clone(&self.device))
        }
    }

    /// `holder`'s items whose slots moved since the engine last named them:
    /// what [`::engine::Engine::rebind_slots`] tells it, channels first (an
    /// instance's tables may address their rings).
    #[must_use]
    pub fn stale(&self, holder: Holder) -> Vec<Rebind> {
        // An adapter that moved rebinds every instance holding it.
        let mut instances: BTreeSet<InstanceId> = BTreeSet::new();
        let mut channels: BTreeSet<ChannelId> = BTreeSet::new();
        for (owner, item) in &self.items {
            match *owner {
                Owner::Channel(id) if item.stale && item.holder == Some(holder) => {
                    channels.insert(id);
                }
                Owner::Instance(id) if item.stale && item.holder == Some(holder) => {
                    instances.insert(id);
                }
                // Its holders on the device; a parked one is rebound when it
                // unparks.
                Owner::Adapter(_) if item.stale && self.unparks_with(*owner, item, holder) => {
                    instances.extend(item.attached.iter().copied().filter(|id| {
                        self.items
                            .get(&Owner::Instance(*id))
                            .is_some_and(|instance| instance.tier == Tier::Device)
                    }));
                }
                _ => {}
            }
        }
        channels
            .into_iter()
            .filter_map(|id| {
                let item = self.items.get(&Owner::Channel(id))?;
                Some(Rebind::Channel {
                    id,
                    slots: item.refs(),
                })
            })
            .chain(instances.into_iter().filter_map(|id| {
                let item = self.items.get(&Owner::Instance(id))?;
                Some(Rebind::Instance {
                    id,
                    instance: item.parts.first().map_or_else(Vec::new, |p| p.refs.clone()),
                    rings: own_rings(&item.rings, &item.parts),
                    adapter: self.adapter_refs(item),
                })
            }))
            .collect()
    }

    /// The `Adapter` slots the instance `item` lands its adapter in.
    fn adapter_refs(&self, item: &Item) -> Vec<u32> {
        item.adapter
            .and_then(|key| self.items.get(&Owner::Adapter(key)))
            .map_or_else(Vec::new, Item::refs)
    }

    /// Whether a host write into `instance`'s rings must wait: its own
    /// slots, or a registered channel's it is bound to, are parked, moving,
    /// or back but not yet rebound.
    #[must_use]
    pub fn writes_held(&self, instance: InstanceId) -> bool {
        let away = |owner: Owner| {
            self.items
                .get(&owner)
                .is_some_and(|item| item.tier != Tier::Device || item.stale)
        };
        let Some(item) = self.items.get(&Owner::Instance(instance)) else {
            return false;
        };
        away(Owner::Instance(instance))
            || item
                .channels
                .iter()
                .any(|&channel| away(Owner::Channel(channel)))
    }

    /// Holds `pump` of `instance` until its slots are back and rebound.
    pub fn hold(&mut self, instance: InstanceId, pump: Pump) {
        if self.held.last() != Some(&(instance, pump)) {
            self.held.push((instance, pump));
        }
    }

    /// The held writes whose instances may write again, in the order they
    /// were held; the rest stay held.
    pub fn released(&mut self) -> Vec<(InstanceId, Pump)> {
        let held = std::mem::take(&mut self.held);
        let (keep, go): (Vec<_>, Vec<_>) = held
            .into_iter()
            .partition(|&(instance, _)| self.writes_held(instance));
        self.held = keep;
        go
    }

    /// The engine took `owner`'s new slots.
    pub fn rebound(&mut self, owner: Owner) {
        let adapter = match self.items.get_mut(&owner) {
            Some(item) => {
                item.stale = false;
                item.adapter
            }
            None => None,
        };
        // An instance rebound names its adapter's new slots too.
        if let Some(item) = adapter.and_then(|key| self.items.get_mut(&Owner::Adapter(key))) {
            item.stale = false;
        }
    }

    /// Bytes of the items' slots on the device: state, none of it cache.
    #[must_use]
    pub fn device_bytes(&self) -> u64 {
        self.items
            .values()
            .filter(|item| item.tier != Tier::Host)
            .flat_map(|item| &item.parts)
            .map(|part| {
                part.refs.len() as u64 * self.shapes.of(part.role).map_or(0, |s| s.slot_bytes)
            })
            .sum()
    }

    /// The slots `owner` holds now, in binding order; none where it holds
    /// none.
    #[cfg(test)]
    #[must_use]
    pub fn slots(&self, owner: Owner) -> Vec<u32> {
        self.items.get(&owner).map_or_else(Vec::new, Item::refs)
    }
}

/// Asks `engine` what an instance of `program` holds, where `roles` has
/// not yet: once a program, at its registration.
pub fn learn_program(
    roles: &RolesHandle,
    engine: &dyn Engine,
    program: ProgramId,
) -> ::engine::Result<()> {
    if roles.lock().unwrap().knows(program) {
        return Ok(());
    }
    let facts = engine.program_facts(program)?;
    roles.lock().unwrap().program_registered(program, facts);
    Ok(())
}

/// Registers channel `plan` by `register`, over the `Channel` slots
/// [`Engine::ring_facts`] states its ring takes ([`ChannelRegistration::slots`]),
/// held for `holder` until it closes. Without `roles`, `plan` as it is.
///
/// [`Engine::ring_facts`]: ::engine::Engine::ring_facts
/// [`ChannelRegistration::slots`]: ::engine::ChannelRegistration::slots
pub fn register_channel<T>(
    roles: Option<&RolesHandle>,
    engine: &mut dyn Engine,
    plan: &ChannelRegistration,
    holder: Option<Holder>,
    register: impl FnOnce(&mut dyn Engine, &ChannelRegistration) -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    let Some(roles) = roles else {
        return register(engine, plan);
    };
    let Some(facts) = engine.ring_facts(plan)? else {
        return register(engine, plan);
    };
    let taken = roles.lock().unwrap().take_channel(&facts)?;
    let plan = ChannelRegistration {
        slots: taken.slots(),
        ..plan.clone()
    };
    match register(engine, &plan) {
        Ok(registered) => {
            roles
                .lock()
                .unwrap()
                .channel_registered(plan.id, holder, taken);
            Ok(registered)
        }
        Err(error) => {
            roles.lock().unwrap().untake_channel(taken);
            Err(error)
        }
    }
}

/// Binds `binding` on `engine` over the `Instance` and `Channel` slots its
/// program's facts state, held for `holder` until the instance closes; a
/// ring of a channel `registered` names is the channel's. Without `roles`,
/// `binding` as it is.
pub fn bind_instance(
    roles: Option<&RolesHandle>,
    engine: &mut dyn Engine,
    binding: &InstanceBinding,
    holder: Option<Holder>,
    registered: impl Fn(ChannelId) -> bool,
) -> anyhow::Result<BoundInstance> {
    let Some(roles) = roles else {
        return Ok(engine.bind_instance(binding)?);
    };
    learn_program(roles, engine, binding.program)?;
    let adapter = engine.adapter_key(binding)?;
    let taken = roles.lock().unwrap().take_instance(
        binding.program,
        &binding.channels,
        registered,
        adapter,
    )?;
    let binding = InstanceBinding {
        instance_slots: taken.instance_slots(),
        ring_slots: taken.ring_slots(),
        adapter_slots: taken.adapter_slots(),
        ..binding.clone()
    };
    match engine.bind_instance(&binding) {
        Ok(bound) => {
            roles.lock().unwrap().bound(bound.id, holder, taken);
            Ok(bound)
        }
        Err(error) => {
            roles.lock().unwrap().untake(taken);
            Err(error.into())
        }
    }
}

/// Runs `write`, a host write into `instance`'s rings, unless its slots
/// are away ([`Roles::writes_held`]): then it is held, to be replayed in
/// order once they are rebound ([`rebind`]).
pub fn pump(
    roles: Option<&RolesHandle>,
    instance: InstanceId,
    pump: Pump,
    write: impl FnOnce() -> ::engine::Result<()>,
) -> ::engine::Result<()> {
    if let Some(roles) = roles {
        let mut roles = roles.lock().unwrap();
        if roles.writes_held(instance) {
            roles.hold(instance, pump);
            return Ok(());
        }
    }
    write()
}

/// Tells `engine` where moved slots lie now (`rebind`), then replays, in
/// order, the host writes held for the instances that may write again.
/// Before any later fire: the lane runs it between frames.
pub fn rebind(
    roles: Option<&RolesHandle>,
    engine: &mut dyn Engine,
    rebind: Rebind,
    mut replay: impl FnMut(&mut dyn Engine, InstanceId, Pump) -> ::engine::Result<()>,
) -> ::engine::Result<()> {
    let owner = Owner::of(&rebind);
    engine.rebind_slots(rebind)?;
    let Some(roles) = roles else {
        return Ok(());
    };
    let released = {
        let mut roles = roles.lock().unwrap();
        roles.rebound(owner);
        roles.released()
    };
    for (instance, pump) in released {
        replay(engine, instance, pump)?;
    }
    Ok(())
}

/// An instance's own rings as the engine names them: channel `rings[k]` in
/// part `k + 1` (part 0 the tables).
fn own_rings(rings: &[u32], parts: &[Part]) -> Vec<RingSlots> {
    rings
        .iter()
        .zip(parts.iter().skip(1))
        .map(|(&channel, part)| RingSlots {
            channel,
            slots: part.refs.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests;
