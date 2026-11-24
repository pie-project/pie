//! What the roles take of the device and the disk, and the budget a move
//! plans within.

use super::*;

impl Roles {
    /// The bytes every item holds together, wherever each lies
    /// ([`Self::part_bytes`]), and those the binds on their way took: what
    /// the capacity gate counts guest state and tables by
    /// (`planner::PoolPort::capacity_census`). One figure on every tier: a
    /// move of an item changes no count.
    #[must_use]
    pub fn footprint(&self) -> u64 {
        let device = self.device.lock().unwrap();
        self.items
            .values()
            .flat_map(|item| &item.parts)
            .map(|part| self.part_bytes(&device, part))
            .sum::<u64>()
            + self.on_way.1
    }

    /// The bytes the items parked on the disk tier hold
    /// ([`Self::footprint`]'s share of them): what of the disk chunks they
    /// take is theirs.
    #[must_use]
    pub fn footprint_on_disk(&self) -> u64 {
        let device = self.device.lock().unwrap();
        self.items
            .values()
            .filter(|item| matches!(item.tier, Tier::Disk))
            .flat_map(|item| &item.parts)
            .map(|part| self.part_bytes(&device, part))
            .sum()
    }

    /// The bytes the roles' shapes hold open on the device and on the host
    /// tier that no slot uses: a chunk of small slots (tables, rings) or of
    /// runs of one class is room no other shape takes until it empties.
    /// No item's state: the capacity gate counts it beside the state it
    /// holds, as room that is not free (`planner::PoolPort::capacity_census`).
    #[must_use]
    pub fn slack(&self) -> u64 {
        let shapes: Vec<_> = [SlotRole::Instance, SlotRole::Channel, SlotRole::Adapter]
            .into_iter()
            .filter_map(|role| self.shapes.of(role).map(|shape| shape.pool))
            .collect();
        let device = self.device.lock().unwrap().pool().slack_of(&shapes);
        let host = self
            .host
            .as_ref()
            .map_or(0, |host| host.lock().unwrap().pool().slack_of(&shapes));
        device + host
    }

    /// [`Self::footprint`] of what `holder`'s binds took: its items, and
    /// its binds on their way.
    #[must_use]
    pub fn footprint_of(&self, holder: Holder) -> u64 {
        let device = self.device.lock().unwrap();
        self.items
            .values()
            .filter(|item| item.holder == Some(holder))
            .flat_map(|item| &item.parts)
            .map(|part| self.part_bytes(&device, part))
            .sum::<u64>()
            + self.on_way.0.get(&Some(holder)).copied().unwrap_or(0)
    }

    /// `taken` is off its way (bound or given back): its bytes are an
    /// item's now, or free.
    pub(super) fn off_way(&mut self, taken: &Taken) {
        if let Some(bytes) = self.on_way.0.get_mut(&taken.holder) {
            *bytes = bytes.saturating_sub(taken.bytes);
            if *bytes == 0 {
                self.on_way.0.remove(&taken.holder);
            }
        }
        self.on_way.1 = self.on_way.1.saturating_sub(taken.bytes);
    }

    /// The bytes `taken` took for itself: its own parts, and an adapter's
    /// or a pair's taken for it (another's are their item's).
    pub(super) fn taken_bytes(&self, taken: &Taken) -> u64 {
        let device = self.device.lock().unwrap();
        let fresh = |part: &Part, fresh: bool| {
            if fresh {
                self.part_bytes(&device, part)
            } else {
                0
            }
        };
        taken
            .parts
            .iter()
            .map(|part| self.part_bytes(&device, part))
            .sum::<u64>()
            + taken
                .adapter
                .as_ref()
                .map_or(0, |(_, part, own)| fresh(part, *own))
        // A pair taken fresh is an item from its take: counted there.
    }

    /// `taken` is on its way: its bytes are counted until it is bound or
    /// given back.
    pub(super) fn on_way_take(&mut self, taken: &mut Taken) {
        taken.bytes = self.taken_bytes(taken);
        *self.on_way.0.entry(taken.holder).or_default() += taken.bytes;
        self.on_way.1 += taken.bytes;
    }

    /// The whole chunks of `on` the live parts of `parts` take.
    pub(super) fn chunks_on(&self, on: &Shared, parts: &[Part]) -> u64 {
        let pool = on.lock().unwrap();
        parts
            .iter()
            .filter(|part| !part.dead())
            .map(|part| {
                let per_chunk = self
                    .shapes
                    .of(part.role)
                    .and_then(|shape| pool.pool().shape(shape.pool))
                    .map_or(1, |shape| shape.slots_per_chunk.max(1));
                (part.refs.len() as u64).div_ceil(u64::from(per_chunk))
            })
            .sum()
    }

    /// Whether the items of `owners` together fit the device pool's state
    /// chunks at all, whole chunks for each part: room a completion or an
    /// eviction can make.
    pub(super) fn fits_budget(&self, owners: &[Owner]) -> bool {
        let device = self.device.lock().unwrap();
        let chunks: u64 = owners
            .iter()
            .flat_map(|owner| &self.items[owner].parts)
            .map(|part| self.part_chunks(&device, part))
            .sum();
        chunks <= u64::from(device.state_chunks_now())
    }

    /// The move of `items` (each item's slots to take), with the copies
    /// from the slots each holds now: each item's tier off the device is
    /// `below`'s, in item order, the one it goes to (`out`), else the one
    /// it lies on now.
    pub(super) fn moving(
        &self,
        out: bool,
        items: Vec<(Owner, Vec<Part>)>,
        below: Vec<Tier>,
    ) -> RoleMove {
        let mut copies: Vec<RoleCopy> = Vec::new();
        for ((owner, to), tier) in items.iter().zip(&below) {
            let disk = *tier == Tier::Disk;
            let from = &self.items[owner].parts;
            for (from, to) in from.iter().zip(to) {
                // Scratch: dead between fires, nothing to copy.
                if from.dead() {
                    continue;
                }
                let Some(shape) = self.shapes.of(from.role) else {
                    continue;
                };
                let (device, host) = if out {
                    (&from.refs, &to.refs)
                } else {
                    (&to.refs, &from.refs)
                };
                let copy = match copies
                    .iter_mut()
                    .find(|copy| copy.shape == shape.id && copy.disk == disk)
                {
                    Some(copy) => copy,
                    None => {
                        copies.push(RoleCopy {
                            shape: shape.id,
                            device: Vec::new(),
                            host: Vec::new(),
                            disk,
                        });
                        copies.last_mut().expect("pushed")
                    }
                };
                copy.device.extend_from_slice(device);
                copy.host.extend_from_slice(host);
            }
        }
        copies.retain(|copy| !copy.device.is_empty());
        RoleMove {
            out,
            items,
            below,
            copies,
            shed: Vec::new(),
        }
    }
}
