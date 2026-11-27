use crate::store::rs::RsUnits;

/// What the pools hold to the end (`pie/lanes-to-end`): learned from the
/// processes that finished, what each held at most. A
/// fact the runtime reports; it gates no start. Whether starts wait for
/// room to the end is the worker policy's (`schedule/hold`, as the
/// built-in `default` writes it).
#[derive(Default)]
pub(super) struct ToEnd {
    /// The device kv pages and rs units a finished process held at most,
    /// averaged.
    peak: Option<Units>,
}

pub(super) type Units = (u32, RsUnits);

fn fold(old: Units, new: Units) -> Units {
    let f = |a: u32, b: u32| (3 * a + b).div_ceil(4);
    (
        f(old.0, new.0),
        RsUnits {
            slots: f(old.1.slots, new.1.slots),
            pages: f(old.1.pages, new.1.pages),
            ..RsUnits::default()
        },
    )
}

impl ToEnd {
    /// What a finished process held at most.
    pub(super) fn learn(&mut self, peak: Units) {
        self.peak = Some(match self.peak {
            None => peak,
            Some(learned) => fold(learned, peak),
        });
    }

    /// The most lanes the pools hold to their end at once: started together
    /// they reach it together, each holding what a finished one held at
    /// most. `None` until a process has finished. What a policy reads as
    /// `pie/lanes-to-end`.
    pub(super) fn lanes_to_end(&self, kv_total: u32, rs_total: RsUnits) -> Option<u64> {
        let end = self.peak?;
        // A lane holds a kv page at least, and a pool that keeps none of a
        // kind bounds nothing.
        let of = |total: u32, each: u32| (total > 0 && each > 0).then(|| total / each);
        let fit = [
            of(rs_total.pages, end.1.pages),
            of(rs_total.slots, end.1.slots),
        ];
        let kv = kv_total / end.0.max(1);
        Some(u64::from(fit.into_iter().flatten().fold(kv, u32::min)))
    }
}

#[cfg(test)]
mod tests {
    use super::{RsUnits, ToEnd};

    #[test]
    fn the_lanes_to_the_end_are_what_the_pools_hold_at_their_peak() {
        let none = RsUnits::default();
        let mut to_end = ToEnd::default();
        assert_eq!(
            to_end.lanes_to_end(1855, none),
            None,
            "nothing learned: absent"
        );
        to_end.learn((27, none));
        // Started together, each holds 27 at its end: 1855 / 27.
        assert_eq!(to_end.lanes_to_end(1855, none), Some(68));
        assert_eq!(
            to_end.lanes_to_end(26, none),
            Some(0),
            "not one fits to its end"
        );
        // The state pool bounds it where it keeps any.
        let rs = |slots, pages| RsUnits {
            slots,
            pages,
            ..RsUnits::default()
        };
        to_end.learn((27, rs(8, 12)));
        assert_eq!(to_end.lanes_to_end(1855, rs(9, 0)), Some(4));
        assert_eq!(to_end.lanes_to_end(1855, rs(9, 8)), Some(2));
        assert_eq!(to_end.lanes_to_end(1855, none), Some(68));
    }
}
