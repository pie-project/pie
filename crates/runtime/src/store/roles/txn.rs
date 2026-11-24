//! A planned move ends: committed, aborted or abandoned.

use super::*;

impl Roles {
    /// The copies landed: each item lies at its new slots, and its old ones
    /// go back. An unparked item is stale until [`Self::rebound`].
    pub fn commit(&mut self, txn: RoleMove) {
        // Back on the device in fresh slots: a taking of their own.
        let generation = self.next_generation();
        for (owner, parts) in txn.shed {
            let Some(item) = self.items.get_mut(&owner) else {
                continue;
            };
            let old = std::mem::replace(&mut item.parts, parts);
            tracing::debug!(
                target: "runtime::planner::roles",
                ?owner,
                out = txn.out,
                old = %traced(&old),
                new = %traced(&item.parts),
                "roles: commit, dead parts shed or taken back"
            );
            item.tier = Tier::Device;
            item.shed = txn.out;
            if !txn.out {
                item.generation = generation;
            }
            // Back in place: the engine is told its fresh dead parts.
            item.stale = !txn.out;
            if txn.out {
                let dead: Vec<Part> = old.into_iter().filter(Part::dead).collect();
                let device = Arc::clone(&self.device);
                self.give_freeing(&device, &dead);
            }
            self.free_if_done(owner);
        }
        for ((owner, parts), below) in txn.items.into_iter().zip(txn.below) {
            let Some(item) = self.items.get_mut(&owner) else {
                continue;
            };
            let old = std::mem::replace(&mut item.parts, parts);
            tracing::debug!(
                target: "runtime::planner::roles",
                ?owner,
                out = txn.out,
                old = %traced(&old),
                new = %traced(&item.parts),
                "roles: commit, moved"
            );
            item.tier = if txn.out { below } else { Tier::Device };
            // Moved whole: nothing of it stayed behind (a parked item whose
            // live parts went on down later, [`Self::prepare_park_down`]).
            item.shed = false;
            if !txn.out {
                item.generation = generation;
            }
            item.stale = !txn.out;
            self.give_on(if txn.out { Tier::Device } else { below }, &old);
            self.free_if_done(owner);
        }
        // A pair a member's park left behind, every member now parked.
        if txn.out {
            self.release_unheld_pairs();
        }
        // A bind waiting on a pair's move looks again.
        self.freed
            .gives
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        self.freed.notify.notify_waiters();
    }

    /// The move failed: each item stays where it was, and the slots taken
    /// for it go back.
    pub fn abort(&mut self, txn: RoleMove) {
        for (owner, parts) in txn.shed {
            if !txn.out {
                // Only the dead parts the unpark took fresh: one the item
                // held in place already ([`Item::takes_fresh`]) stays its.
                let had: Vec<Part> = self
                    .items
                    .get(&owner)
                    .map_or_else(Vec::new, |item| item.parts.clone());
                let dead: Vec<Part> = parts
                    .into_iter()
                    .zip(had.iter().map(Some).chain(std::iter::repeat(None)))
                    .filter(|(part, had)| {
                        part.dead() && had.is_none_or(|had| had.refs.is_empty())
                    })
                    .map(|(part, _)| part)
                    .collect();
                let device = Arc::clone(&self.device);
                self.give(&device, &dead);
            }
            if let Some(item) = self.items.get_mut(&owner) {
                item.tier = Tier::Device;
            }
            self.free_if_done(owner);
        }
        for ((owner, parts), below) in txn.items.into_iter().zip(txn.below) {
            if txn.out {
                self.give_on(below, &parts);
            } else {
                let device = Arc::clone(&self.device);
                self.give(&device, &parts);
            }
            if let Some(item) = self.items.get_mut(&owner) {
                item.tier = if txn.out { Tier::Device } else { below };
            }
            self.free_if_done(owner);
        }
        // What waits on a pair's move (a bind, a restore) looks again: the
        // move ended, here as at a commit.
        self.tell_room();
    }

    /// The move failed and its copies never settled (the engine could not
    /// say they landed): it ends all the same, each item where the move
    /// found it, so nothing waits on it moving, but the slots it took on
    /// the tier it was going to stay out of service for good, leaked by
    /// name: a late copy may still write them, so they are never given
    /// back nor reused. The slots it leaves still hold what they held (a
    /// copy only reads them). Answers the slots leaked.
    pub fn abandon(&mut self, txn: RoleMove) -> u64 {
        let mut leaked = 0u64;
        for (owner, parts) in txn.shed {
            if !txn.out {
                leaked += parts
                    .iter()
                    .filter(|part| part.dead())
                    .map(|part| part.refs.len() as u64)
                    .sum::<u64>();
            }
            if let Some(item) = self.items.get_mut(&owner) {
                item.tier = Tier::Device;
            }
            self.free_if_done(owner);
        }
        for ((owner, parts), below) in txn.items.into_iter().zip(txn.below) {
            leaked += parts.iter().map(|part| part.refs.len() as u64).sum::<u64>();
            if let Some(item) = self.items.get_mut(&owner) {
                item.tier = if txn.out { Tier::Device } else { below };
            }
            self.free_if_done(owner);
        }
        self.leaked += leaked;
        // What waits on the move (a bind, a restore of a pair's other
        // holder) looks again: it no longer moves.
        self.tell_room();
        leaked
    }
}
