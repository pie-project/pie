//! An instance's tables and a channel's ring in the shared pool (REDESIGN
//! 2.6, `new-memory/STATE-ROLES.md` option A).
//!
//! An engine that states a fixed shape of role [`SlotRole::Instance`] or
//! [`SlotRole::Channel`] places a bound instance's launch tables and a
//! ring's cells in slots of them, which the runtime takes from the pool on
//! demand: [`Engine::program_facts`] and [`Engine::ring_facts`] say how
//! many, and the binding ([`InstanceBinding::instance_slots`],
//! [`InstanceBinding::ring_slots`]) or registration
//! ([`ChannelRegistration::slots`]) names them. They are no part of a
//! record nor of a working set ([`super::shared`]).
//!
//! Neither role is replayable ([`SlotRole::replayable`]): no token rebuilds
//! them, so nothing here drops one. Under pressure the policy may park them
//! with their process on the host tier, as it parks the process's kv: a
//! park copies each item to host slots and frees its device slots, an
//! unpark copies it into fresh device slots, and the engine is told the new
//! refs ([`Engine::rebind_slots`]) before the next fire reads them.
//!
//! A ring an instance holds itself is the instance's item. A registered
//! channel's ring is the channel's, shared by every instance bound to it,
//! and freed once the channel is closed and the last of them has closed
//! too.
//!
//! An engine that states neither shape has no [`Roles`]: it keeps its
//! tables and rings itself, as before.
//!
//! [`Engine::program_facts`]: ::engine::Engine::program_facts
//! [`Engine::ring_facts`]: ::engine::Engine::ring_facts
//! [`Engine::rebind_slots`]: ::engine::Engine::rebind_slots
//! [`InstanceBinding::instance_slots`]: ::engine::InstanceBinding::instance_slots
//! [`InstanceBinding::ring_slots`]: ::engine::InstanceBinding::ring_slots
//! [`ChannelRegistration::slots`]: ::engine::ChannelRegistration::slots

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use ::engine::chunk::{ShapeId as EngineShape, SlotRole};
use ::engine::{
    BoundInstance, ChannelId, ChannelRegistration, Engine, InstanceBinding, InstanceId,
    ProgramFacts, ProgramId, Rebind, RingFacts, RingSlots,
};

use super::shared::{FixedShape, RoleShapes, Shared};

/// Whose slots an item is: an instance's tables (and the rings it holds
/// itself), or a registered channel's ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Owner {
    Instance(InstanceId),
    Channel(ChannelId),
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
}

/// Where an item's slots lie.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tier {
    Device,
    Host,
    /// A park or unpark is copying it: neither side's slots are free.
    Moving,
}

/// A run of one role's slots, as refs of the tier the item lies on.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Part {
    role: SlotRole,
    refs: Vec<u32>,
}

#[derive(Debug)]
struct Item {
    holder: Option<Holder>,
    tier: Tier,
    /// An instance: its tables' part, then each ring it holds itself. A
    /// channel: its ring's part.
    parts: Vec<Part>,
    /// An instance: each own ring's channel index, in part order after the
    /// first.
    rings: Vec<u32>,
    /// An instance: the registered channels with slots it is bound to.
    channels: Vec<ChannelId>,
    /// A channel: the instances bound to it.
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
    pub instance_slots: Vec<u32>,
    pub ring_slots: Vec<RingSlots>,
    /// The registered channels with slots the binding names.
    channels: Vec<ChannelId>,
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
        })
    }

    fn shape(&self, role: SlotRole) -> Result<FixedShape, RolesError> {
        self.shapes.of(role).ok_or(RolesError::NoShape(role))
    }

    fn take(&self, on: &Shared, role: SlotRole, slots: u32) -> Result<Vec<u32>, RolesError> {
        if slots == 0 {
            return Ok(Vec::new());
        }
        let shape = self.shape(role)?;
        on.lock()
            .unwrap()
            .take_refs(shape.pool, u64::from(slots))
            .ok_or(RolesError::Short { role, slots })
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
            match self.take(on, part.role, part.refs.len() as u32) {
                Ok(refs) => taken.push(Part {
                    role: part.role,
                    refs,
                }),
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
    /// the engine counts, not the bytes), and for each ring of the program
    /// no registered channel covers (`registered`), its slots of the
    /// `Channel` shape.
    pub fn take_instance(
        &mut self,
        program: ProgramId,
        channels: &[ChannelId],
        registered: impl Fn(ChannelId) -> bool,
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
        let untake = |roles: &Self, taken: &Taken| roles.give(&device, &parts_of(taken));
        taken.instance_slots = self.take(&device, SlotRole::Instance, facts.instance.slots)?;
        for ring in &facts.rings {
            let covered = channels
                .get(ring.channel as usize)
                .is_some_and(|&id| registered(id));
            if covered {
                continue;
            }
            match self.take(&device, SlotRole::Channel, ring.footprint.slots) {
                Ok(slots) => taken.ring_slots.push(RingSlots {
                    channel: ring.channel,
                    slots,
                }),
                Err(error) => {
                    untake(self, &taken);
                    return Err(error);
                }
            }
        }
        Ok(taken)
    }

    /// The bind of `taken` failed: its slots go back.
    pub fn untake(&mut self, taken: Taken) {
        self.give(&self.device, &parts_of(&taken));
    }

    /// The engine bound `taken` as instance `id`, for `holder`.
    pub fn bound(&mut self, id: InstanceId, holder: Option<Holder>, taken: Taken) {
        for &channel in &taken.channels {
            if let Some(item) = self.items.get_mut(&Owner::Channel(channel)) {
                item.attached.insert(id);
            }
        }
        let mut item = Item::new(holder, parts_of(&taken));
        item.rings = taken.ring_slots.iter().map(|ring| ring.channel).collect();
        item.channels = taken.channels;
        if item.parts.iter().all(|part| part.refs.is_empty()) && item.channels.is_empty() {
            return;
        }
        self.items.insert(Owner::Instance(id), item);
    }

    /// The engine closed instance `id`: its slots go back, and a closed
    /// channel it was the last on with them.
    pub fn close_instance(&mut self, id: InstanceId) {
        let owner = Owner::Instance(id);
        let Some(item) = self.items.get_mut(&owner) else {
            return;
        };
        item.open = false;
        let channels = item.channels.clone();
        self.free_if_done(owner);
        for channel in channels {
            let owner = Owner::Channel(channel);
            if let Some(item) = self.items.get_mut(&owner) {
                item.attached.remove(&id);
            }
            self.free_if_done(owner);
        }
    }

    // ---- channels ----

    /// The slots a channel whose ring `facts` states takes.
    pub fn take_channel(&mut self, facts: &RingFacts) -> Result<Vec<u32>, RolesError> {
        self.take(&self.device, SlotRole::Channel, facts.footprint.slots)
    }

    /// The channel's registration failed: `slots` go back.
    pub fn untake_channel(&mut self, slots: Vec<u32>) {
        self.give(
            &self.device,
            &[Part {
                role: SlotRole::Channel,
                refs: slots,
            }],
        );
    }

    /// The engine registered channel `id` over `slots`, for `holder`.
    pub fn channel_registered(&mut self, id: ChannelId, holder: Option<Holder>, slots: Vec<u32>) {
        if slots.is_empty() {
            return;
        }
        self.items.insert(
            Owner::Channel(id),
            Item::new(
                holder,
                vec![Part {
                    role: SlotRole::Channel,
                    refs: slots,
                }],
            ),
        );
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

    /// Whether `owner` parks and unparks with `holder`: an instance it
    /// holds, or a channel it registered that only its instances are bound
    /// to.
    fn moves_with(&self, owner: Owner, item: &Item, holder: Holder) -> bool {
        if item.holder != Some(holder) || !item.open {
            return false;
        }
        match owner {
            Owner::Instance(_) => true,
            Owner::Channel(_) => item.attached.iter().all(|&id| {
                self.items
                    .get(&Owner::Instance(id))
                    .is_some_and(|instance| instance.holder == Some(holder))
            }),
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
            .filter(|(_, item)| item.tier == Tier::Host && item.holder == Some(holder))
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
        let mut stale: Vec<(Owner, &Item)> = self
            .items
            .iter()
            .filter(|(_, item)| item.stale && item.holder == Some(holder))
            .map(|(owner, item)| (*owner, item))
            .collect();
        stale.sort_by_key(|(owner, _)| match owner {
            Owner::Channel(id) => (0, id.0),
            Owner::Instance(id) => (1, id.0),
        });
        stale
            .into_iter()
            .map(|(owner, item)| match owner {
                Owner::Channel(id) => Rebind::Channel {
                    id,
                    slots: item.refs(),
                },
                Owner::Instance(id) => Rebind::Instance {
                    id,
                    instance: item.parts.first().map_or_else(Vec::new, |p| p.refs.clone()),
                    rings: item
                        .rings
                        .iter()
                        .zip(item.parts.iter().skip(1))
                        .map(|(&channel, part)| RingSlots {
                            channel,
                            slots: part.refs.clone(),
                        })
                        .collect(),
                },
            })
            .collect()
    }

    /// The engine took `owner`'s new slots.
    pub fn rebound(&mut self, owner: Owner) {
        if let Some(item) = self.items.get_mut(&owner) {
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
    let slots = roles.lock().unwrap().take_channel(&facts)?;
    let plan = ChannelRegistration {
        slots: slots.clone(),
        ..plan.clone()
    };
    match register(engine, &plan) {
        Ok(registered) => {
            roles
                .lock()
                .unwrap()
                .channel_registered(plan.id, holder, slots);
            Ok(registered)
        }
        Err(error) => {
            roles.lock().unwrap().untake_channel(slots);
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
    let taken =
        roles
            .lock()
            .unwrap()
            .take_instance(binding.program, &binding.channels, registered)?;
    let binding = InstanceBinding {
        instance_slots: taken.instance_slots.clone(),
        ring_slots: taken.ring_slots.clone(),
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

fn parts_of(taken: &Taken) -> Vec<Part> {
    std::iter::once(Part {
        role: SlotRole::Instance,
        refs: taken.instance_slots.clone(),
    })
    .chain(taken.ring_slots.iter().map(|ring| Part {
        role: SlotRole::Channel,
        refs: ring.slots.clone(),
    }))
    .collect()
}

#[cfg(test)]
mod tests;
