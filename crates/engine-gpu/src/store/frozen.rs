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
    use super::{
        DeviceFold, Frozen, LaneFold, Tier, counted_on_device, lane_report, merged, reads_count,
    };
    use engine::fire::{FoldLen, RsVerb};

    fn buffer(folded: u32, fold: FoldLen) -> RsVerb {
        RsVerb::Buffer {
            pages: vec![0],
            base: 0,
            folded,
            fold,
            lineage: Default::default(),
        }
    }

    /// The count a lane reports is this fire's alone, after the clamp to
    /// the fire's end: a device count past the lane's rows freezes only as
    /// far as the fire wrote.
    #[test]
    fn the_reported_count_is_the_fires_own_after_the_clamp() {
        // 4 slots frozen, 2 buffered, 3 rows at slot 6: the port asks 3.
        let lane = LaneFold::of(&buffer(4, FoldLen::Device), 3, 6, None, Some(3), None).unwrap();
        assert_eq!(lane.plan.froze, 3);
        assert_eq!(lane.left, Some(7));
        let report = lane.report(41).expect("the lane asks a report");
        assert_eq!((report.tag, report.fold, report.end), (41, Some(3), None));
        assert!(report.tokens.is_empty() && report.pages.is_empty());

        // The port asks 100: the fold stops at the fire's end, slot 9.
        let lane = LaneFold::of(&buffer(4, FoldLen::Device), 3, 6, None, Some(100), None).unwrap();
        assert_eq!(lane.plan.froze, 5);
        assert_eq!(lane.left, Some(9));
        assert_eq!(lane.report(41).unwrap().fold, Some(5));

        // A fold of none is reported as none, not left unsaid.
        let lane = LaneFold::of(&buffer(4, FoldLen::Device), 3, 6, None, Some(0), None).unwrap();
        assert_eq!(lane.report(41).unwrap().fold, Some(0));
        assert_eq!(lane.left, Some(4));
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

    /// The replay starts from the further of the stated length and the
    /// record's own position: a fire staged before the last fold's report
    /// reached the runtime states the old length, and must not fold the
    /// same slots twice.
    #[test]
    fn the_replay_starts_from_the_further_of_stated_and_own() {
        // Fire 1: nothing frozen, 4 rows at 0, the device folds 3.
        let first = LaneFold::of(&buffer(0, FoldLen::Device), 4, 0, None, Some(3), None).unwrap();
        assert_eq!((first.plan.replay, first.plan.commit), (0, 3));
        assert_eq!(first.left, Some(3));

        // Fire 2 still states 0 frozen: 2 rows at 4, the device folds 2.
        let stale =
            LaneFold::of(&buffer(0, FoldLen::Device), 2, 4, None, Some(2), first.left).unwrap();
        let exact = LaneFold::of(&buffer(3, FoldLen::Device), 2, 4, None, Some(2), None).unwrap();
        assert_eq!(stale.plan.replay, 1, "slot 3 alone is replayed");
        assert_eq!(
            (stale.plan.replay, stale.plan.commit, stale.plan.froze),
            (exact.plan.replay, exact.plan.commit, exact.plan.froze),
        );
        assert_eq!(stale.left, Some(5));
        // Stated alone, the fire would replay slots 0..4 into a record
        // that already holds 0..3.
        let stated = LaneFold::of(&buffer(0, FoldLen::Device), 2, 4, None, Some(2), None).unwrap();
        assert_eq!(stated.plan.replay, 4);

        // A stated length past the record's own wins.
        let ahead =
            LaneFold::of(&buffer(5, FoldLen::Device), 2, 5, None, Some(1), Some(3)).unwrap();
        assert_eq!(ahead.left, Some(6));

        // A host-counted fire on a record still ahead of its verb keeps
        // the position; once the verb has caught up it hands it back.
        let host = LaneFold::of(&buffer(0, FoldLen::Host(1)), 2, 4, None, None, Some(3)).unwrap();
        assert_eq!((host.plan.replay, host.left), (1, Some(4)));
        let caught = LaneFold::of(&buffer(4, FoldLen::Host(1)), 2, 6, None, None, Some(4)).unwrap();
        assert_eq!(caught.left, None);
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

    /// The position travels with the record: a fork copy, a move to the
    /// host tier and back, and a copy over a slot that held another.
    #[test]
    fn the_position_survives_a_slot_copy() {
        let mut frozen = Frozen::default();
        frozen.leave(&[10, 11], Some(7));
        assert_eq!(frozen.held(&[10, 11]), Some(7));
        assert_eq!(frozen.held(&[20, 21]), None);

        // A copy-on-write fork, piece by piece.
        frozen.copied(Tier::Device, Tier::Device, &[(10, 20), (11, 21)]);
        assert_eq!(frozen.held(&[20, 21]), Some(7));
        assert_eq!(frozen.held(&[10, 11]), Some(7));

        // The fork folds on; its source stands.
        frozen.leave(&[20, 21], Some(9));
        assert_eq!(frozen.held(&[10, 11]), Some(7));

        // Suspended to the host tier, its device slots given to a record
        // with no position of its own, then restored elsewhere.
        frozen.copied(Tier::Device, Tier::Host, &[(20, 3), (21, 4)]);
        frozen.copied(Tier::Device, Tier::Device, &[(30, 20), (31, 21)]);
        assert_eq!(frozen.held(&[20, 21]), None);
        frozen.copied(Tier::Host, Tier::Device, &[(3, 40), (4, 41)]);
        assert_eq!(frozen.held(&[40, 41]), Some(9));

        // Bytes written from outside the pool carry no position.
        frozen.written(Tier::Host, &[3, 4]);
        frozen.copied(Tier::Host, Tier::Device, &[(3, 40), (4, 41)]);
        assert_eq!(frozen.held(&[40, 41]), None);

        // A slot copied onto itself keeps what it holds.
        frozen.copied(Tier::Device, Tier::Device, &[(10, 10)]);
        assert_eq!(frozen.held(&[10, 11]), Some(7));
    }

    /// Lanes of a group share a record: it is left where the furthest of
    /// them folded.
    #[test]
    fn a_shared_record_is_left_at_the_furthest_fold() {
        let left = merged(vec![
            (vec![1, 2], Some(5)),
            (vec![1, 2], None),
            (vec![3], None),
            (vec![1, 2], Some(4)),
        ]);
        assert_eq!(left, vec![(vec![1, 2], Some(5)), (vec![3], None)]);
    }

    /// A set that keeps only kv folds by the count its program put on a
    /// fire whose fold length is the device's: the lane has no recurrent
    /// buffer, so no plan states what it froze, and the count is what the
    /// runtime waits on. The lane takes the count and reports it, as read,
    /// under its tag; one that asks no report takes none.
    #[test]
    fn a_lane_with_no_recurrent_buffer_reports_its_device_count() {
        let kv = RsVerb::Fold;
        assert!(reads_count(&kv, 41));
        assert!(!reads_count(&kv, 0));
        let lane = LaneFold::of(&kv, 1, 6, None, Some(1), None).unwrap();
        assert_eq!(lane.plan.froze, 0);
        let device = DeviceFold {
            count: Some(1),
            reports: 41,
        };
        // No lane of the fire has a recurrent buffer: it is still the
        // device's count the runtime waits on.
        let report = lane_report(&kv, &lane, device, false).expect("the lane asks a report");
        assert_eq!((report.tag, report.fold), (41, Some(1)));
        assert!(report.tokens.is_empty() && report.pages.is_empty());
        let device = DeviceFold {
            count: Some(0),
            reports: 41,
        };
        assert_eq!(
            lane_report(&kv, &lane, device, false).unwrap().fold,
            Some(0)
        );
        assert!(lane_report(&kv, &lane, DeviceFold::default(), false).is_none());

        // A lane with a recurrent buffer reads as it did, and reports what
        // its fold froze, on a fire whose fold length is the device's only.
        assert!(reads_count(&buffer(4, FoldLen::Device), 0));
        assert!(!reads_count(&buffer(4, FoldLen::Host(1)), 41));
        let lane = LaneFold::of(&buffer(4, FoldLen::Device), 3, 6, None, Some(3), None).unwrap();
        let device = DeviceFold {
            count: Some(3),
            reports: 41,
        };
        assert_eq!(
            lane_report(&buffer(4, FoldLen::Device), &lane, device, true)
                .unwrap()
                .fold,
            Some(3)
        );
        assert!(lane_report(&buffer(4, FoldLen::Device), &lane, device, false).is_none());
    }
}
