//! What the roles hold, read: a fire's slots, the guest items and chunks a
//! holder holds, the pairs moving, and the room watch.

use super::*;

impl Roles {
    /// The device slots instance `id` names as its fires read them: its
    /// tables, rings and scratch, its pair's state and the channels' rings
    /// and adapter it is bound to, each part with what it is, where the item
    /// lies on the device. What a fire's check at the engine reads.
    pub(crate) fn fire_slots_of(
        &self,
        id: InstanceId,
    ) -> Vec<(String, super::super::slots::ShapeId, super::super::slots::Slot)> {
        let device = self.device.lock().unwrap();
        let mut out = Vec::new();
        let Some(instance) = self.items.get(&Owner::Instance(id)) else {
            return out;
        };
        let mut owners = vec![Owner::Instance(id)];
        owners.extend(instance.pair.map(Owner::Pair));
        owners.extend(instance.channels.iter().map(|&channel| Owner::Channel(channel)));
        owners.extend(instance.adapter.map(Owner::Adapter));
        for owner in owners {
            let Some(item) = self.items.get(&owner) else {
                continue;
            };
            if item.tier != Tier::Device {
                continue;
            }
            for part in item.parts.iter().filter(|part| !part.dead()) {
                let Some(shape) = self.shapes.of(part.role) else {
                    continue;
                };
                for &slot_ref in &part.refs {
                    out.push((
                        format!("{owner:?}'s {:?} slot ref {slot_ref}", part.role),
                        shape.pool,
                        device.slot(shape.pool, slot_ref),
                    ));
                }
            }
        }
        out
    }

    /// Every item `holder` moves with, for a quiet wait's census: its owner,
    /// tier, whether shed, and each part's role, live or dead, and the
    /// physical chunks its refs lie in where it is on the device.
    #[must_use]
    pub fn guest_items_census(&self, holder: Holder) -> String {
        let device = self.device.lock().unwrap();
        let mut out = Vec::new();
        for (owner, item) in &self.items {
            if !self.moves_with(*owner, item, holder) {
                continue;
            }
            let parts: Vec<String> = item
                .parts
                .iter()
                .map(|part| {
                    let chunks: BTreeSet<u32> = match (item.tier == Tier::Device, self.shapes.of(part.role)) {
                        (true, Some(shape)) => part
                            .refs
                            .iter()
                            .filter_map(|&slot_ref| device.pool().physical(device.slot(shape.pool, slot_ref).chunk))
                            .collect(),
                        _ => BTreeSet::new(),
                    };
                    format!(
                        "{:?} {} refs={} chunks={chunks:?}",
                        part.role,
                        if part.dead() { "dead" } else { "live" },
                        part.refs.len()
                    )
                })
                .collect();
            out.push(format!("{owner:?} tier={:?} shed={} [{}]", item.tier, item.shed, parts.join("; ")));
        }
        out.join(" | ")
    }

    /// Per logical chunk, the slots each of `holders` holds there in guest
    /// items bound on the device: dead parts (scratch and pair state, given
    /// back at a park whatever lies below) and live ones (tables, rings),
    /// which a park takes below or, with no room there, leaves parked in
    /// place for a move down a tier to clear. Every one of them goes once
    /// its holder moves, so a chunk they fill is emptiable, as the pool's
    /// census counts it: counting the dead parts alone, a chunk holding a
    /// table never qualified for a run, and a long run (nine chunks, two
    /// of them) found no window, though moving its holders stood it.
    /// Parked items lying on the device are not here: they are the parked
    /// occupants (`parked_chunk_slots`).
    #[must_use]
    pub fn guest_chunk_slots(&self, holders: &[Holder]) -> Vec<(u32, Holder, u32)> {
        let device = self.device.lock().unwrap();
        let mut counts: BTreeMap<(u32, Holder), u32> = BTreeMap::new();
        for (owner, item) in &self.items {
            if item.tier != Tier::Device || item.shed {
                continue;
            }
            let Some(holder) = holders
                .iter()
                .copied()
                .find(|&holder| self.moves_with(*owner, item, holder))
            else {
                continue;
            };
            for part in &item.parts {
                let Some(shape) = self.shapes.of(part.role) else {
                    continue;
                };
                for &slot_ref in &part.refs {
                    let slot = device.slot(shape.pool, slot_ref);
                    *counts.entry((slot.chunk, holder)).or_default() += 1;
                }
            }
        }
        counts
            .into_iter()
            .map(|((chunk, holder), slots)| (chunk, holder, slots))
            .collect()
    }

    /// Whether `holder` holds guest state on the device: a bind of its
    /// short of room would wait on room it holds itself.
    #[must_use]
    pub fn holds_guest_state(&self, holder: Holder) -> bool {
        self.items.values().any(|item| {
            item.holder == Some(holder)
                && item.tier == Tier::Device
                && item
                    .parts
                    .iter()
                    .any(|part| part.dead() && !part.refs.is_empty())
        })
    }

    /// Whether `holder` holds any part of a guest's state on the device,
    /// live, held or dead: what a park of it moves down a tier or gives
    /// back. A process with no kv or state of its own (a prompt not
    /// seated) holding only such parts is a victim whose eviction parks
    /// them ([`crate::planner`]'s eviction): read by dead parts alone, one
    /// holding resident bytes (a held part) "found nothing to move", its
    /// chunks never emptied (seed 3392).
    #[must_use]
    pub fn holds_device_parts(&self, holder: Holder) -> bool {
        self.items.values().any(|item| {
            item.holder == Some(holder)
                && item.tier == Tier::Device
                && item.parts.iter().any(|part| !part.refs.is_empty())
        })
    }

    /// Whether `pair`'s state is on its way to or from the host with the
    /// instances that hold it, and those instances with their holders: what
    /// a restore short of it ([`RolesError::PairParked`]) waits to see end.
    #[must_use]
    pub fn pair_moving(&self, pair: FireKey) -> Option<Vec<(InstanceId, Option<Holder>)>> {
        let item = self.items.get(&Owner::Pair(pair))?;
        (item.tier == Tier::Moving).then(|| {
            item.attached
                .iter()
                .map(|id| {
                    let holder = self
                        .items
                        .get(&Owner::Instance(*id))
                        .and_then(|instance| instance.holder);
                    (*id, holder)
                })
                .collect()
        })
    }

    /// The pair of one of `instances` on its way to or from the device
    /// (another holder's park or restore moving it): what an unpark that
    /// would leave them unbound waits to see end ([`Self::pair_moving`]).
    #[must_use]
    pub fn moving_pair_of(&self, instances: &[InstanceId]) -> Option<FireKey> {
        instances.iter().find_map(|id| {
            let key = self.items.get(&Owner::Instance(*id))?.pair?;
            (self.items.get(&Owner::Pair(key))?.tier == Tier::Moving).then_some(key)
        })
    }

    /// A watch on device room given back from now on.
    #[must_use]
    pub fn room_watch(&self) -> RoomWatch {
        RoomWatch {
            seen: self.freed.gives.load(std::sync::atomic::Ordering::Acquire),
            freed: Arc::clone(&self.freed),
        }
    }
}
