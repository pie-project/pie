//! Parks and unparks: which items move with a holder, the moves a park, an
//! unpark or a park down plans, and what a part takes on the device.

use super::*;

impl Roles {
    pub(super) fn free_if_done(&mut self, owner: Owner) {
        let Some(item) = self.items.get(&owner) else {
            return;
        };
        if item.open
            || !item.attached.is_empty()
            || item.tier == Tier::Moving
            || item.reads > 0
            || item.binds > 0
        {
            return;
        }
        let item = self.items.remove(&owner).expect("present");
        tracing::debug!(
            target: "runtime::planner::roles",
            ?owner,
            holder = ?item.holder,
            tier = ?item.tier,
            parts = %traced(&item.parts),
            "roles: freed, its slots given back"
        );
        self.give_on(item.tier, &item.parts);
    }

    // ---- parking ----

    /// Whether `owner` parks with `holder`: an instance it holds, a channel
    /// it registered that only its instances are bound to, or an adapter
    /// every holder of which is `holder`'s or parked already.
    pub(super) fn moves_with(&self, owner: Owner, item: &Item, holder: Holder) -> bool {
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
            // A pair's run is its batch's while any member is on the device
            // or on its way to or from it: a member mid-park (its move may
            // yet be taken back, the tier below full) or mid-unpark holds
            // it. Read as parked, the pair went with another holder's park,
            // the member's park was taken back, and the run, given back
            // under a resident member, was another program's fresh room
            // (a pair run bound by another program's instances, a device
            // fault on its next fire). The member's park landing gives it back
            // ([`Self::release_unheld_pairs`]).
            // A bind on its way holds it as a member does.
            Owner::Pair(_) if item.binds > 0 => false,
            Owner::Adapter(_) if item.binds > 0 => false,
            Owner::Pair(_) => {
                item.attached
                    .iter()
                    .any(|id| instance(id).is_some_and(|instance| instance.holder == Some(holder)))
                    && item.attached.iter().all(|id| {
                        instance(id).is_none_or(|instance| {
                            instance.holder == Some(holder) || Self::settled_parked(instance)
                        })
                    })
            }
            Owner::Adapter(_) => {
                item.attached
                    .iter()
                    .any(|id| instance(id).is_some_and(|instance| instance.holder == Some(holder)))
                    && item.attached.iter().all(|id| {
                        // Gone (closed) holds it no longer: read as held, a
                        // beam pair's scratch whose other copies had ended
                        // never moved with the last one, and its chunk stayed
                        // on the device for good (its evicted holder no
                        // candidate, the head one chunk short).
                        // Parked already and settled, as a pair's: a member
                        // on its way holds it (its park may yet be taken
                        // back), and a member's landing park parks it where
                        // it lies ([`Self::release_unheld_pairs`]).
                        instance(id).is_none_or(|instance| {
                            instance.holder == Some(holder) || Self::settled_parked(instance)
                        })
                    })
            }
        }
    }

    /// A member parked and settled: off the device, or parked where it
    /// lies ([`Item::shed`]); never one on its way (`Moving`).
    pub(super) fn settled_parked(instance: &Item) -> bool {
        matches!(instance.tier, Tier::Host | Tier::Disk)
            || (instance.tier == Tier::Device && instance.shed)
    }

    /// Gives back the run of each pair on the device every member of which
    /// is parked and settled, nothing reading it: what a member's park
    /// left behind while another member was on its way
    /// ([`Self::moves_with`]). Its state is dead between fires; it parks
    /// as one with no slots below, and a member's restore takes it fresh.
    pub(super) fn release_unheld_pairs(&mut self) {
        let settled = |item: &Item| {
            item.tier == Tier::Device
                && !item.shed
                && item.reads == 0
                && item.binds == 0
                && !item.attached.is_empty()
                && item.attached.iter().all(|id| {
                    self.items
                        .get(&Owner::Instance(*id))
                        .is_none_or(Self::settled_parked)
                })
        };
        // An adapter holds live parts: parked where it lies, its room a
        // move down a tier takes ([`Self::prepare_park_down`]), its rebind
        // at its first member's unpark.
        let adapters: Vec<Owner> = self
            .items
            .iter()
            .filter(|(owner, item)| matches!(owner, Owner::Adapter(_)) && settled(item))
            .map(|(owner, _)| *owner)
            .collect();
        for owner in adapters {
            self.items.get_mut(&owner).expect("present").shed = true;
        }
        let unheld: Vec<Owner> = self
            .items
            .iter()
            .filter(|(owner, item)| {
                matches!(owner, Owner::Pair(_))
                    && item.tier == Tier::Device
                    && !item.shed
                    && item.reads == 0
                    && item.binds == 0
                    && item.parts.iter().all(Part::dead)
                    && !item.attached.is_empty()
                    && item.attached.iter().all(|id| {
                        self.items
                            .get(&Owner::Instance(*id))
                            .is_none_or(Self::settled_parked)
                    })
            })
            .map(|(owner, _)| *owner)
            .collect();
        for owner in unheld {
            let item = self.items.get_mut(&owner).expect("present");
            let emptied: Vec<Part> = item
                .parts
                .iter()
                .map(|part| Part {
                    refs: Vec::new(),
                    ..part.clone()
                })
                .collect();
            let old = std::mem::replace(&mut item.parts, emptied);
            item.tier = Tier::Host;
            item.stale = false;
            let device = Arc::clone(&self.device);
            self.give_freeing(&device, &old);
        }
    }

    /// Whether `owner`, parked, unparks with `holder`: what it parked, or
    /// an adapter one of its instances holds.
    pub(super) fn unparks_with(&self, owner: Owner, item: &Item, holder: Holder) -> bool {
        match owner {
            Owner::Adapter(_) | Owner::Pair(_) => item.attached.iter().any(|id| {
                self.items
                    .get(&Owner::Instance(*id))
                    .is_some_and(|instance| instance.holder == Some(holder))
            }),
            _ => item.holder == Some(holder),
        }
    }

    /// Parks `holder`'s items: host slots taken for each item that fits
    /// there, to copy into. Dead parts (guest scratch and pair state, dead
    /// between fires) need no slots below and are given back whatever the
    /// tier below holds: an item that is all dead parks with no host slots
    /// at all, and one whose live parts the host lacks room for (or with no
    /// host tier) keeps them where they lie and sheds its dead parts
    /// ([`Item::shed`]). Nothing is dropped. `None` when nothing moves. The
    /// process's fires have landed (the planner quiesced it first): no fire
    /// reads the dead parts given back.
    pub fn prepare_park(&mut self, holder: Holder) -> Option<RoleMove> {
        let owners: Vec<Owner> = self
            .items
            .iter()
            .filter(|(owner, item)| {
                item.tier == Tier::Device && !item.shed && self.moves_with(**owner, item, holder)
            })
            .map(|(owner, _)| *owner)
            .collect();
        let mut items = Vec::new();
        let mut below = Vec::new();
        let mut shed = Vec::new();
        for owner in owners {
            let parts = self.items[&owner].parts.clone();
            let emptied = |parts: &[Part]| -> Vec<Part> {
                parts
                    .iter()
                    .map(|part| {
                        if part.dead() {
                            Part {
                                refs: Vec::new(),
                                ..part.clone()
                            }
                        } else {
                            part.clone()
                        }
                    })
                    .collect()
            };
            // Its live parts to the host where it takes them, else to the
            // disk tier's slot file (no host tier on unified memory, or a
            // host with no room for them): left on the device there, a
            // process's tables and rings would hold their chunks until a
            // move down is asked for them, and none is.
            let moved = if parts.iter().all(Part::dead) {
                Some((emptied(&parts), Tier::Host))
            } else {
                let host = self.host.clone().and_then(|host| {
                    self.take_parts(&host, &parts, false)
                        .map_err(|error| {
                            tracing::debug!(?owner, %error, "an item's live parts find no host room");
                        })
                        .ok()
                });
                host.map(|to| (to, Tier::Host))
                    .or_else(|| self.take_disk(&parts).map(|to| (to, Tier::Disk)))
            };
            match moved {
                Some((to, tier)) => {
                    self.items.get_mut(&owner).expect("present").tier = Tier::Moving;
                    items.push((owner, to));
                    below.push(tier);
                }
                // Its live parts stay where they lie, parked all the same:
                // room a move down a tier makes once the device wants it
                // ([`Self::prepare_park_down`]).
                None => {
                    self.items.get_mut(&owner).expect("present").tier = Tier::Moving;
                    shed.push((owner, emptied(&parts)));
                }
            }
        }
        for (owner, to) in &items {
            tracing::debug!(
                target: "runtime::planner::roles",
                ?holder,
                ?owner,
                from = %traced(&self.items[owner].parts),
                to = %traced(to),
                "roles: park takes host slots"
            );
        }
        for (owner, kept) in &shed {
            tracing::debug!(
                target: "runtime::planner::roles",
                ?holder,
                ?owner,
                from = %traced(&self.items[owner].parts),
                kept = %traced(kept),
                "roles: park sheds dead parts, live ones stay"
            );
        }
        (!items.is_empty() || !shed.is_empty()).then(|| {
            let mut txn = self.moving(true, items, below);
            txn.shed = shed;
            txn
        })
    }

    /// Unparks `holder`'s parked items: device slots taken for each, all or
    /// none. `None` when nothing is parked.
    pub fn prepare_unpark(&mut self, holder: Holder) -> Result<Option<RoleMove>, RolesError> {
        self.claim = self.claims.get(&holder).copied();
        let unparked = self.prepare_unpark_claimed(holder);
        self.claim = None;
        unparked
    }

    pub(super) fn prepare_unpark_claimed(
        &mut self,
        holder: Holder,
    ) -> Result<Option<RoleMove>, RolesError> {
        let (groups, order) = self.unpark_order(holder);
        // An instance whose pair is on its way back with another process's
        // unpark is rebound once that lands: until then its restore is
        // queued again, never told no room for its pair. One parked where
        // it lies (no tier below, [`Item::shed`]) waits the same: let
        // through, it came back with its pair away, so its restore named it
        // no rebind and its next fire ran on the places it gave back.
        if let Some(pair) = groups.iter().find_map(|(owner, _, _)| {
            let key = self.items[owner].pair?;
            (self.items.get(&Owner::Pair(key))?.tier == Tier::Moving).then_some(key)
        }) {
            return Err(RolesError::PairParked(pair));
        }
        let device = Arc::clone(&self.device);
        let mut back: Vec<Vec<Option<Part>>> = groups
            .iter()
            .map(|(_, _, parts)| vec![None; parts.len()])
            .collect();
        for (at, index) in order {
            let part = &groups[at].2[index];
            match self.take_parts(&device, std::slice::from_ref(part), true) {
                Ok(mut taken) => back[at][index] = Some(taken.remove(0)),
                Err(error) => {
                    let taken: Vec<Part> = back.into_iter().flatten().flatten().collect();
                    self.give(&device, &taken);
                    // It held these parts on the device before its park:
                    // the budget holds them, and the restore waits for
                    // room, never failed for it.
                    if error.is_short() {
                        return Err(RolesError::UnparkShort(Box::new(error)));
                    }
                    return Err(error);
                }
            }
        }
        let mut items: Vec<(Owner, Vec<Part>)> = Vec::new();
        let mut shed: Vec<(Owner, Vec<Part>)> = Vec::new();
        for ((owner, is_shed, parts), taken) in groups.into_iter().zip(back) {
            let parts: Vec<Part> = parts
                .into_iter()
                .zip(taken)
                .map(|(part, taken)| taken.unwrap_or(part))
                .collect();
            if is_shed {
                shed.push((owner, parts));
            } else {
                items.push((owner, parts));
            }
        }
        let below: Vec<Tier> = items
            .iter()
            .map(|(owner, _)| self.items[owner].tier)
            .collect();
        for (owner, _) in items.iter().chain(&shed) {
            self.items.get_mut(owner).expect("present").tier = Tier::Moving;
        }
        for (owner, to) in &items {
            tracing::debug!(
                target: "runtime::planner::roles",
                ?holder,
                ?owner,
                from = %traced(&self.items[owner].parts),
                to = %traced(to),
                "roles: unpark takes device slots"
            );
        }
        for (owner, back) in &shed {
            tracing::debug!(
                target: "runtime::planner::roles",
                ?holder,
                ?owner,
                from = %traced(&self.items[owner].parts),
                to = %traced(back),
                "roles: unpark takes fresh dead parts"
            );
        }
        Ok((!items.is_empty() || !shed.is_empty()).then(|| {
            let mut txn = self.moving(false, items, below);
            txn.shed = shed;
            txn
        }))
    }

    /// The process a parked item went with: an instance's or a channel's
    /// holder, an adapter's (or a pair's) first instance's.
    pub(super) fn parked_with(&self, owner: Owner, item: &Item) -> Option<Holder> {
        match owner {
            Owner::Instance(_) | Owner::Channel(_) => item.holder,
            Owner::Adapter(_) | Owner::Pair(_) => item.attached.iter().find_map(|id| {
                self.items
                    .get(&Owner::Instance(*id))
                    .and_then(|instance| instance.holder)
            }),
        }
    }

    /// Parked items whose live parts stayed on the device for want of room
    /// below ([`Item::shed`]): no fire reads them until their process is
    /// restored, so they are room a move down a tier makes.
    pub(super) fn parked_on_device(&self) -> impl Iterator<Item = (Owner, &Item, Holder)> {
        self.items.iter().filter_map(|(owner, item)| {
            (item.tier == Tier::Device && item.shed)
                .then(|| self.parked_with(*owner, item))
                .flatten()
                .map(|holder| (*owner, item, holder))
        })
    }

    /// Per logical chunk, the slots parked items' live parts hold on the
    /// device ([`Item::shed`]), by the process each parked with, `except`'s
    /// left out (its restore keeps them where they lie): the chunks moving
    /// them down a tier ([`Self::prepare_park_down`]) empties.
    #[must_use]
    pub fn parked_chunk_slots(&self, except: Holder) -> Vec<(u32, Holder, u32)> {
        let device = self.device.lock().unwrap();
        let mut counts: BTreeMap<(u32, Holder), u32> = BTreeMap::new();
        for (_, item, holder) in self.parked_on_device() {
            if holder == except {
                continue;
            }
            // Room a move makes only where some tier below takes it.
            if !self.lies_below(&item.parts) {
                continue;
            }
            for part in item.parts.iter().filter(|part| !part.dead()) {
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

    /// Moves the items `holders` parked with whose live parts stayed on the
    /// device (no room below at their park, [`Item::shed`]) down a tier, as
    /// the load's tiers lie: to the host where it has room now, slots taken
    /// there for each that fits, to copy into, as a park takes them; where
    /// the load states no host tier (unified memory: its host is the
    /// device's own memory), or with `disk_too` once the host can make no
    /// room, to the disk tier's slot file ([`DiskSlots`]) for a role whose
    /// slot fits one of its slots. Committed, each lies below as if parked
    /// there, and its restore takes fresh device slots
    /// ([`Self::prepare_unpark`]). What the host has no room for, short of
    /// `disk_too`, is counted in host chunks: the room kv on the host
    /// moving on down to disk makes for it, older below newer.
    pub fn prepare_park_down(&mut self, holders: &[Holder], disk_too: bool) -> ParkDown {
        let mut owners: Vec<Owner> = self
            .parked_on_device()
            .filter(|(_, _, holder)| holders.contains(holder))
            .map(|(owner, ..)| owner)
            .collect();
        // Left behind: a shared item (a pair, an adapter) still on the
        // device that moves with one of these parked holders now, its other
        // holders parked or gone since that holder's park (which left it,
        // shared then). It goes down as a parked item does, its dead parts
        // given back (else a beam pair of dead refs, its last holder
        // evicted, holds the head one chunk short for good).
        let behind: Vec<Owner> = self
            .items
            .iter()
            .filter(|(owner, item)| {
                matches!(owner, Owner::Pair(_) | Owner::Adapter(_))
                    && item.tier == Tier::Device
                    && !item.shed
                    && holders
                        .iter()
                        .any(|&holder| self.moves_with(**owner, item, holder))
            })
            .map(|(owner, _)| *owner)
            .collect();
        owners.extend(behind);
        let mut items = Vec::new();
        let mut below = Vec::new();
        let mut host_chunks = 0u64;
        for owner in owners {
            let parts = self.items[&owner].parts.clone();
            // Nothing live: nothing to copy, its dead parts given back.
            if parts.iter().all(Part::dead) {
                let emptied: Vec<Part> = parts
                    .iter()
                    .map(|part| Part {
                        refs: Vec::new(),
                        ..part.clone()
                    })
                    .collect();
                self.items.get_mut(&owner).expect("present").tier = Tier::Moving;
                items.push((owner, emptied));
                below.push(Tier::Host);
                continue;
            }
            let on_host = self
                .host
                .clone()
                .map(|host| (self.take_parts(&host, &parts, false).ok(), host));
            let to = match on_host {
                Some((Some(to), _)) => Some((to, Tier::Host)),
                Some((None, host)) if !disk_too => {
                    host_chunks += self.chunks_on(&host, &parts);
                    None
                }
                _ => self.take_disk(&parts).map(|to| (to, Tier::Disk)),
            };
            if let Some((to, tier)) = to {
                self.items.get_mut(&owner).expect("present").tier = Tier::Moving;
                items.push((owner, to));
                below.push(tier);
            }
        }
        ParkDown {
            txn: (!items.is_empty()).then(|| self.moving(true, items, below)),
            host_chunks,
        }
    }

    /// The whole chunks `part` takes on the device: a guest part its runs
    /// of chunks (a run of slots, one), any other part its slots in whole
    /// chunks of its shape.
    pub(super) fn part_chunks(
        &self,
        device: &super::super::shared::SharedPool,
        part: &Part,
    ) -> u64 {
        if part.shaped() {
            let (run, pieces) = part.chunk_runs();
            u64::from(run) * u64::from(pieces)
        } else {
            let per_chunk = self
                .shapes
                .of(part.role)
                .and_then(|shape| device.pool().shape(shape.pool))
                .map_or(1, |shape| shape.slots_per_chunk.max(1));
            (part.refs.len() as u64).div_ceil(u64::from(per_chunk))
        }
    }

    /// The bytes `part` holds wherever it lies, one figure on every tier
    /// (a move of it changes no count): a guest part in runs of whole
    /// chunks those chunks, any other part (a run of guest slots, tables,
    /// a ring, an adapter) its slots, dead guest scratch included (given
    /// back at a park, taken fresh at the restore: the process's still).
    pub(super) fn part_bytes(&self, device: &super::super::shared::SharedPool, part: &Part) -> u64 {
        let pool = device.pool();
        if part.pieces > 0 {
            return u64::from(part.chunks) * u64::from(part.pieces) * pool.chunk_bytes();
        }
        let slot_bytes = self
            .shapes
            .of(part.role)
            .and_then(|shape| pool.shape(shape.pool))
            .map_or(0, |shape| shape.slot_bytes);
        let slots = if part.run > 0 {
            part.run as usize
        } else {
            part.refs.len()
        };
        slots as u64 * slot_bytes
    }
}
