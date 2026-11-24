//! The tiers below the device: which roles the disk tier holds, taking and
//! giving its slots, and giving parts back on whichever tier they lie.

use super::*;

impl Roles {
    /// The slots of `role` a disk chunk holds, laid out as a device chunk,
    /// where the load keeps a disk tier.
    pub(super) fn disk_per_chunk(&self, role: SlotRole) -> Option<u32> {
        self.disk.as_ref()?.per_chunk.get(&role).copied()
    }

    /// Whether the disk tier holds slots of `role`.
    pub(super) fn disk_holds(&self, role: SlotRole) -> bool {
        self.disk_per_chunk(role).is_some()
    }

    /// Whether some tier below the device holds every live part of `parts`.
    pub(super) fn lies_below(&self, parts: &[Part]) -> bool {
        self.host.is_some()
            || parts
                .iter()
                .filter(|part| !part.dead())
                .all(|part| self.disk_holds(part.role))
    }

    /// Disk slots for the live parts of `parts`, all or none: a dead part
    /// holds nothing off the device. Each role's slots come from its own
    /// disk chunks, a chunk opened whole where those are full.
    pub(super) fn take_disk(&self, parts: &[Part]) -> Option<Vec<Part>> {
        let disk = self.disk.as_ref()?;
        if !parts
            .iter()
            .filter(|part| !part.dead())
            .all(|part| self.disk_holds(part.role))
        {
            return None;
        }
        let mut taken: Vec<Part> = Vec::with_capacity(parts.len());
        for part in parts {
            let refs = if part.dead() || part.refs.is_empty() {
                Vec::new()
            } else {
                let per_chunk = self.disk_per_chunk(part.role)?;
                match Self::take_disk_slots(disk, part.role, per_chunk, part.refs.len()) {
                    Some(refs) => refs,
                    None => {
                        self.give_on(Tier::Disk, &taken);
                        return None;
                    }
                }
            };
            taken.push(Part {
                refs,
                ..part.clone()
            });
        }
        Some(taken)
    }

    /// `n` free slots of `role` in its disk chunks, `per_chunk` a chunk,
    /// opening whole chunks for what they lack; all or none.
    pub(super) fn take_disk_slots(disk: &Disk, role: SlotRole, per_chunk: u32, n: usize) -> Option<Vec<u32>> {
        let mut shapes = disk.chunks.lock().unwrap();
        let chunks = shapes.entry(role).or_default();
        let free: usize = chunks.values().map(BTreeSet::len).sum();
        if free < n {
            let opened = (n - free).div_ceil(per_chunk as usize);
            for chunk in disk.slots.take_chunks(opened)? {
                chunks.insert(chunk, (0..per_chunk).collect());
            }
        }
        let mut refs = Vec::with_capacity(n);
        for (&chunk, slots) in chunks.iter_mut() {
            while refs.len() < n {
                let Some(index) = slots.pop_first() else {
                    break;
                };
                refs.push(chunk * per_chunk + index);
            }
            if refs.len() == n {
                break;
            }
        }
        Some(refs)
    }

    /// `refs` of `role` free again in its disk chunks; a chunk wholly free
    /// goes back to the slot file.
    pub(super) fn give_disk_slots(disk: &Disk, role: SlotRole, per_chunk: u32, refs: &[u32]) {
        let mut shapes = disk.chunks.lock().unwrap();
        let chunks = shapes.entry(role).or_default();
        let mut emptied = Vec::new();
        for &slot in refs {
            let (chunk, index) = (slot / per_chunk, slot % per_chunk);
            let slots = chunks.entry(chunk).or_default();
            slots.insert(index);
            if slots.len() == per_chunk as usize {
                emptied.push(chunk);
            }
        }
        for chunk in &emptied {
            chunks.remove(chunk);
        }
        emptied.sort_unstable();
        emptied.dedup();
        if !emptied.is_empty() {
            disk.slots.give_chunks(&emptied);
        }
    }

    /// Gives `parts` back on tier `on` (their refs that tier's).
    pub(super) fn give_on(&self, on: Tier, parts: &[Part]) {
        match on {
            Tier::Disk => {
                let Some(disk) = &self.disk else {
                    return;
                };
                for part in parts.iter().filter(|part| !part.refs.is_empty()) {
                    if let Some(per_chunk) = self.disk_per_chunk(part.role) {
                        Self::give_disk_slots(disk, part.role, per_chunk, &part.refs);
                    }
                }
            }
            Tier::Host => {
                if let Some(host) = self.host.clone() {
                    self.give(&host, parts);
                }
            }
            Tier::Device | Tier::Moving => {
                let device = Arc::clone(&self.device);
                self.give_freeing(&device, parts);
            }
        }
    }
}
