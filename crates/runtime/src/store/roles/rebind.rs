//! After a move: the rebinds the engine is owed, the instances left
//! unbound, and the writes an instance holds.

use super::*;

impl Roles {
    /// Role slots out of service for good: taken by moves whose copies an
    /// engine could not settle ([`Self::abandon`]).
    #[must_use]
    pub fn leaked(&self) -> u64 {
        self.leaked
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
                // Only what lies on the device now: one moved off since its
                // restore (parked again while its rebinds waited) names no
                // slots the engine may claim.
                Owner::Channel(id)
                    if item.stale && item.holder == Some(holder) && item.tier == Tier::Device =>
                {
                    channels.insert(id);
                }
                Owner::Instance(id)
                    if item.stale && item.holder == Some(holder) && item.tier == Tier::Device =>
                {
                    instances.insert(id);
                }
                // Its holders on the device; a parked one is rebound when it
                // unparks. One parked where it lies (no tier below took its
                // live parts, its dead ones given back: [`Item::shed`]) is
                // parked too: rebound now, it named no scratch at all ("a
                // guest item ... and its binding names []" on another
                // process's restore of the pair).
                Owner::Adapter(_) | Owner::Pair(_)
                    if item.stale && self.unparks_with(*owner, item, holder) =>
                {
                    instances.extend(item.attached.iter().copied().filter(|id| {
                        self.items
                            .get(&Owner::Instance(*id))
                            .is_some_and(|instance| instance.tier == Tier::Device && !instance.shed)
                    }));
                }
                _ => {}
            }
        }
        let rebinds: Vec<Rebind> = channels
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
                // A guest part given back (its process parked where it lies,
                // [`Item::shed`]) lies nowhere: never named to the engine;
                // its own restore takes it back first, and rebinds it then.
                if item.shed || item.parts.iter().any(|part| part.dead() && part.refs.is_empty()) {
                    return None;
                }
                // Its pair not on the device yet: rebound once it lands,
                // never told the pair lies nowhere.
                if let Some(key) = item.pair
                    && self
                        .items
                        .get(&Owner::Pair(key))
                        .is_none_or(|pair| pair.tier != Tier::Device)
                {
                    return None;
                }
                Some(Rebind::Instance {
                    id,
                    instance: item.parts.first().map_or_else(Vec::new, |p| p.refs.clone()),
                    rings: own_rings(&item.rings, &item.parts),
                    adapter: self.adapter_refs(item),
                    scratch: scratch(&item.parts)
                        .map_or(ScratchPlace::None, |part| place(part, self.chunk_slots)),
                    pair: item
                        .pair
                        .and_then(|key| self.items.get(&Owner::Pair(key)))
                        .map_or(ScratchPlace::None, |pair| {
                            place(&pair.parts[0], self.chunk_slots)
                        }),
                    pair_generation: item
                        .pair
                        .and_then(|key| self.items.get(&Owner::Pair(key)))
                        .map_or(0, |pair| pair.generation),
                })
            }))
            .collect();
        for rebind in &rebinds {
            tracing::debug!(
                target: "runtime::planner::roles",
                ?holder,
                ?rebind,
                "roles: rebind after the restore"
            );
        }
        rebinds
    }

    /// The `Adapter` slots the instance `item` lands its adapter in.
    pub(super) fn adapter_refs(&self, item: &Item) -> Vec<u32> {
        item.adapter
            .and_then(|key| self.items.get(&Owner::Adapter(key)))
            .map_or_else(Vec::new, Item::refs)
    }

    /// `holder`'s open instances not yet bound where they lie: off the
    /// device, moving, parked where they lie, or back but not rebound. A
    /// restore reports its process resident only once this is empty: a
    /// fire of an instance the engine still binds at the places it gave
    /// back at its park runs on room others hold since.
    #[must_use]
    pub fn unbound(&self, holder: Holder) -> Vec<InstanceId> {
        self.items
            .iter()
            .filter_map(|(owner, item)| match *owner {
                Owner::Instance(id)
                    if item.holder == Some(holder)
                        && item.open
                        && (item.tier != Tier::Device || item.stale || item.takes_fresh()) =>
                {
                    Some(id)
                }
                _ => None,
            })
            .collect()
    }

    /// `holder`'s open instances an unpark of `txn` (none: nothing to
    /// unpark) would leave unbound, its rebinds unable to name them: off
    /// the device or moving and not in it, short of a dead part it does
    /// not take ([`Item::takes_fresh`]), or holding a pair that is not on
    /// the device and not in it. A restore commits only once this is empty
    /// ([`Self::prepare_unpark`] then [`Self::commit`]): committed with
    /// one, it would wait Restoring on a rebind nothing makes.
    #[must_use]
    pub fn unbound_after(&self, holder: Holder, txn: Option<&RoleMove>) -> Vec<InstanceId> {
        let moved = |owner: Owner| {
            txn.is_some_and(|txn| {
                txn.items.iter().chain(&txn.shed).any(|(moved, _)| *moved == owner)
            })
        };
        self.items
            .iter()
            .filter_map(|(owner, item)| {
                let Owner::Instance(id) = *owner else {
                    return None;
                };
                if item.holder != Some(holder) || !item.open {
                    return None;
                }
                let placed = moved(*owner)
                    || (item.tier == Tier::Device && !item.shed && !item.takes_fresh());
                let pair = item.pair.is_none_or(|key| {
                    moved(Owner::Pair(key))
                        || self
                            .items
                            .get(&Owner::Pair(key))
                            .is_some_and(|pair| pair.tier == Tier::Device && !pair.takes_fresh())
                });
                (!placed || !pair).then_some(id)
            })
            .collect()
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

    /// Whether `owner` is still the engine's to rebind: an instance not
    /// closed, a channel some open instance or its own registration still
    /// holds. A restore reads its rebinds off the roles ([`Self::stale`])
    /// and the lane runs them in turn, closes between them: one the lane
    /// closed meanwhile is gone from the engine, and nothing reads its
    /// slots, so its rebind is cancelled, never sent.
    #[must_use]
    pub fn rebinds(&self, owner: Owner) -> bool {
        self.items.get(&owner).is_some_and(|item| match owner {
            Owner::Instance(_) => item.open,
            _ => true,
        })
    }

    /// Whether `rebind`, read off the roles before the lane ran it, still
    /// says where its item lies: a park of an instance not closed since; a
    /// rebind of an item that is on the device, not parked where it lies,
    /// at the slots it names. One whose item has moved since (parked again
    /// by an eviction or a bind's suspension while its restore waited on
    /// its rebinds, its device slots given back at that park's commit) is
    /// cancelled: sent, the engine would claim slots the store has handed
    /// out again, and the next bind there be refused (an instance slot
    /// rebound just after its park, the same refs, then bound to another
    /// instance). The move that
    /// brings it back names it anew.
    #[must_use]
    pub fn rebind_stands(&self, rebind: &Rebind) -> bool {
        let owner = Owner::of(rebind);
        if !self.rebinds(owner) {
            return false;
        }
        let Some(item) = self.items.get(&owner) else {
            return false;
        };
        match rebind {
            Rebind::Parked { .. } => true,
            Rebind::Instance { instance, .. } => {
                item.tier == Tier::Device
                    && !item.shed
                    && item.parts.first().map_or(instance.is_empty(), |part| part.refs == *instance)
            }
            Rebind::Channel { slots, .. } => item.tier == Tier::Device && item.refs() == *slots,
        }
    }

    /// Instances `ids` were said parked to the engine and their park was
    /// taken back: each on the device where it lay, not parked where it
    /// lies, is rebound at its slots ([`Self::stale`]) before its process
    /// fires again, so the engine never believes a park the runtime undid.
    pub fn park_undone(&mut self, ids: &[InstanceId]) {
        for id in ids {
            if let Some(item) = self.items.get_mut(&Owner::Instance(*id))
                && item.tier == Tier::Device
                && !item.shed
            {
                item.stale = true;
            }
        }
    }

    /// The engine took `owner`'s new slots.
    pub fn rebound(&mut self, owner: Owner) {
        let (adapter, pair) = match self.items.get_mut(&owner) {
            Some(item) => {
                item.stale = false;
                (item.adapter, item.pair)
            }
            None => (None, None),
        };
        // An instance rebound names its adapter's and its pair's new slots
        // too.
        if let Some(item) = adapter.and_then(|key| self.items.get_mut(&Owner::Adapter(key))) {
            item.stale = false;
        }
        if let Some(item) = pair.and_then(|key| self.items.get_mut(&Owner::Pair(key))) {
            item.stale = false;
        }
    }

    /// Bytes of the items' slots on the device: state, none of it cache.
    #[must_use]
    pub fn device_bytes(&self) -> u64 {
        self.items
            .values()
            .filter(|item| !matches!(item.tier, Tier::Host | Tier::Disk))
            .flat_map(|item| &item.parts)
            .map(|part| {
                part.refs.len() as u64 * self.shapes.of(part.role).map_or(0, |s| s.slot_bytes)
            })
            .sum()
    }
}
