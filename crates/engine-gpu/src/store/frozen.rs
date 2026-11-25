//! Where each record's own folds left it. A set whose fold the device
//! counts states its frozen length as a lower bound: the fire that froze
//! further may not have settled when the next one is staged. The record is
//! this engine's, so the engine keeps the slot its last fold left the
//! record at, beside the slot the record stands in, and replays from the
//! further of the two. The position moves with the record wherever
//! [`super::Pools`] copies a slot.
use std::collections::HashMap;

use engine::fire::{FoldLen, Masking, RsVerb};
use engine::rs::LanePlan;
use engine::transfer::MemoryDomain;

/// The tier a slot ref is counted in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tier {
    Device,
    Host,
    Disk,
}

impl Tier {
    /// The tier of `domain`, the device's where `device` says so.
    #[must_use]
    pub fn of(domain: MemoryDomain, device: bool) -> Tier {
        match domain {
            _ if device => Tier::Device,
            MemoryDomain::LocalDisk => Tier::Disk,
            _ => Tier::Host,
        }
    }
}

/// A lane's fold as the device counts it ([`FoldLen::Device`]): the count
/// its program's `fold` port holds at the fire, and the tag its report
/// carries (0: the lane asks none).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceFold {
    pub count: Option<u32>,
    pub reports: u64,
}

/// One lane's fold of one fire.
#[derive(Clone, Debug)]
pub struct LaneFold {
    pub plan: LanePlan,
    /// The slot the fold leaves the record at, where the record keeps its
    /// own position past this fire; `None` where the stated length is the
    /// record's.
    pub left: Option<u32>,
}

impl LaneFold {
    /// The lane's fold: it replays from the further of the frozen length
    /// its verb states and `held`, the slot the record's last fold left it
    /// at. The record keeps its own position while the device counts its
    /// fold, and for as long as it stands past what the verb states.
    pub fn of(
        verb: &RsVerb,
        rows: u32,
        first: u32,
        mask: Option<&Masking>,
        device: Option<u32>,
        held: Option<u32>,
    ) -> engine::Result<LaneFold> {
        let RsVerb::Buffer {
            pages,
            base,
            folded,
            fold,
            lineage,
        } = verb
        else {
            return Ok(LaneFold {
                plan: LanePlan::of(verb, rows, first, mask, device)?,
                left: None,
            });
        };
        let from = held.map_or(*folded, |held| held.max(*folded));
        // The mask's columns start at the stated boundary (`KvDelta::folded`,
        // the verb's `folded`); replayed from `from`, the columns between are
        // committed slots now.
        let shifted = mask
            .filter(|_| from > *folded)
            .map(|mask| after(mask, u64::from(from - *folded)));
        let mask = shifted.as_ref().or(mask);
        let plan = LanePlan::of(
            &RsVerb::Buffer {
                pages: pages.clone(),
                base: *base,
                folded: from,
                fold: *fold,
                lineage: *lineage,
            },
            rows,
            first,
            mask,
            device,
        )?;
        let keeps = *fold == FoldLen::Device || from > *folded;
        let left = keeps.then_some(from + plan.froze);
        Ok(LaneFold { plan, left })
    }

    /// The lane's report on a fire whose fold length is the device's: the
    /// slots its fold froze, and no rows (this engine places none).
    #[must_use]
    pub fn report(&self, reports: u64) -> Option<engine::KvWrites> {
        (reports != 0).then(|| engine::KvWrites {
            tag: reports,
            fold: Some(self.plan.froze),
            ..engine::KvWrites::default()
        })
    }
}

/// `masking` without its first `columns` columns.
fn after(masking: &Masking, columns: u64) -> Masking {
    match masking {
        Masking::Extent(mask) => Masking::Extent(mask.after(columns)),
        Masking::Rows(masks) => Masking::Rows(masks.iter().map(|m| m.after(columns)).collect()),
    }
}

/// Whether `verb` leaves its fold length to the device.
#[must_use]
pub fn counted_on_device(verb: &RsVerb) -> bool {
    matches!(
        verb,
        RsVerb::Buffer {
            fold: FoldLen::Device,
            ..
        }
    )
}

/// Whether a lane takes the count its program's `fold` port holds at the
/// fire: a lane whose recurrent buffer leaves its fold length to the
/// device, and a lane with no recurrent buffer (a set that keeps only kv)
/// that asks a report, for its set's fold is then the count the runtime
/// waits on.
#[must_use]
pub fn reads_count(verb: &RsVerb, reports: u64) -> bool {
    counted_on_device(verb) || (!matches!(verb, RsVerb::Buffer { .. }) && reports != 0)
}

/// What a lane reports of its fold on a fire. A lane with a recurrent
/// buffer reports the slots its fold froze where the fire's fold length is
/// the device's ([`LaneFold::report`]); a lane with none reports the count
/// its program put, as it read it: the runtime clamps it to the fire's end.
#[must_use]
pub fn lane_report(
    verb: &RsVerb,
    fold: &LaneFold,
    device: DeviceFold,
    device_fire: bool,
) -> Option<engine::KvWrites> {
    match verb {
        RsVerb::Buffer { .. } => fold.report(device.reports).filter(|_| device_fire),
        _ => (device.reports != 0)
            .then_some(device.count)
            .flatten()
            .map(|count| engine::KvWrites {
                tag: device.reports,
                fold: Some(count),
                ..engine::KvWrites::default()
            }),
    }
}

/// The slot each record's last fold left it at, by the slot each of its
/// pieces stands in. A record with no entry stands where its verb states.
#[derive(Debug, Default)]
pub struct Frozen {
    at: HashMap<(Tier, u32), u32>,
}

impl Frozen {
    /// The position of the record in `pieces` (device slot refs).
    #[must_use]
    pub fn held(&self, pieces: &[u32]) -> Option<u32> {
        pieces
            .iter()
            .filter_map(|&piece| self.at.get(&(Tier::Device, piece)).copied())
            .max()
    }

    /// The record in `pieces` was left at `at` by a fire; `None` hands its
    /// position back to the verb.
    pub fn leave(&mut self, pieces: &[u32], at: Option<u32>) {
        for &piece in pieces {
            match at {
                Some(at) => self.at.insert((Tier::Device, piece), at),
                None => self.at.remove(&(Tier::Device, piece)),
            };
        }
    }

    /// Slots `(src, dst)` were copied whole, in order: each destination
    /// stands where its source does.
    pub fn copied(&mut self, src: Tier, dst: Tier, moves: &[(u32, u32)]) {
        for &(from, to) in moves {
            if (src, from) == (dst, to) {
                continue;
            }
            match self.at.get(&(src, from)).copied() {
                Some(at) => self.at.insert((dst, to), at),
                None => self.at.remove(&(dst, to)),
            };
        }
    }

    /// Slots of `tier` were written with bytes from outside the pool: what
    /// they hold stands where its verb states.
    pub fn written(&mut self, tier: Tier, slots: &[u32]) {
        for &slot in slots {
            self.at.remove(&(tier, slot));
        }
    }
}

/// The positions a fire's lanes leave their records at, a record once: the
/// furthest any lane of a group sharing it folded.
#[must_use]
pub fn merged(left: Vec<(Vec<u32>, Option<u32>)>) -> Vec<(Vec<u32>, Option<u32>)> {
    let mut out: Vec<(Vec<u32>, Option<u32>)> = Vec::with_capacity(left.len());
    for (pieces, at) in left {
        match out.iter_mut().find(|(seen, _)| *seen == pieces) {
            Some((_, furthest)) => *furthest = (*furthest).max(at),
            None => out.push((pieces, at)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{DeviceFold, Frozen, LaneFold, counted_on_device};
    use engine::fire::{FoldLen, Lineage, RsVerb};

    fn buffer(folded: u32, fold: FoldLen) -> RsVerb {
        RsVerb::Buffer {
            pages: vec![0],
            base: 0,
            folded,
            fold,
            lineage: Lineage::Set,
        }
    }

    /// A lane that asks no report gets none; a device fold length no
    /// program resolved is refused, not counted as zero.
    #[test]
    fn a_lane_reports_only_where_it_asks() {
        let lane = LaneFold::of(&buffer(0, FoldLen::Device), 2, 0, None, Some(1), None).unwrap();
        assert!(lane.report(0).is_none());
        assert!(lane.report(DeviceFold::default().reports).is_none());
        let refused = LaneFold::of(&buffer(0, FoldLen::Device), 2, 0, None, None, None)
            .expect_err("no fold port resolved");
        assert!(refused.to_string().contains("resolved no `fold` port"));
        // A lane of the group whose length the host states reports what it
        // froze, on the same fire.
        let lane = LaneFold::of(&buffer(0, FoldLen::Host(0)), 2, 0, None, None, None).unwrap();
        assert_eq!(lane.report(9).unwrap().fold, Some(0));
        assert_eq!(lane.left, None);
        assert!(counted_on_device(&buffer(0, FoldLen::Device)));
        assert!(!counted_on_device(&buffer(0, FoldLen::Host(3))));
        assert!(!counted_on_device(&RsVerb::Fold));
    }

    /// A record stands where the last fold that went to the device left
    /// it. A step that prepared and whose submit failed fired nothing and
    /// leaves no position: the set's next fire, stating the same frozen
    /// length, replays every slot behind it and not from a slot no fold
    /// reached.
    #[test]
    fn a_step_that_never_fired_leaves_its_record_where_it_was() {
        let pieces = [10, 11];
        let mut frozen = Frozen::default();
        let fold = |rows, first, count, held| {
            LaneFold::of(
                &buffer(0, FoldLen::Device),
                rows,
                first,
                None,
                Some(count),
                held,
            )
            .unwrap()
        };
        // Nothing frozen, 4 rows at 0, the device folds 3: prepared, and
        // its submit fails. The position it would leave is not left.
        let failed = fold(4, 0, 3, frozen.held(&pieces));
        assert_eq!(failed.left, Some(3));
        assert_eq!(frozen.held(&pieces), None);

        // The next fire still states 0 frozen, 2 rows at 4: it replays
        // slots 0..4, all of them unfolded.
        let next = fold(2, 4, 2, frozen.held(&pieces));
        assert_eq!(next.plan.replay, 4);
        // From the position of the step that never fired it would replay
        // slot 3 alone, past three slots no fold reached.
        assert_eq!(fold(2, 4, 2, failed.left).plan.replay, 1);

        // This one goes to the device and leaves its position, and the
        // fire after it replays from there.
        frozen.leave(&pieces, next.left);
        assert_eq!(frozen.held(&pieces), Some(2));
        assert_eq!(fold(1, 6, 1, frozen.held(&pieces)).plan.replay, 4);
    }

}
