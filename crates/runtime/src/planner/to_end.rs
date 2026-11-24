use crate::store::rs::RsUnits;

/// What the pools hold to the end (`pie/lanes-to-end`): learned from the
/// processes that finished, what each held at its prompt and at most. A
/// fact the runtime reports; it gates no start. Whether starts wait for
/// room to the end is the worker policy's (`schedule/hold`, as the
/// built-in `default` writes it).
#[derive(Default)]
pub(super) struct ToEnd {
    /// The device kv pages and rs units a finished process held at most,
    /// and at its first grant (its prompt), averaged.
    footprint: Option<(Units, Units)>,
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

/// The most `held` (each lane's pages now, the new lane's prompt among
/// them) reach at once as every lane grows a page a step to `end` and lets
/// all go there.
fn most_at_once(held: &mut [u32], end: u32) -> u64 {
    held.sort_unstable();
    let (mut most, mut sum) = (0u64, 0u64);
    for (i, &h) in held.iter().enumerate() {
        let h = h.min(end);
        sum += u64::from(h);
        if held.get(i + 1).is_some_and(|&next| next.min(end) == h) {
            continue;
        }
        // Just before the lanes holding `h` finish, the lanes up to them
        // have each grown `end - h` pages; the larger ones are gone.
        most = most.max(sum + (i as u64 + 1) * u64::from(end - h));
    }
    most
}

impl ToEnd {
    /// What a finished process held at most and at its prompt.
    pub(super) fn learn(&mut self, peak: Units, first: Units) {
        self.footprint = Some(match self.footprint {
            None => (peak, first),
            Some((p, f)) => (fold(p, peak), fold(f, first)),
        });
    }

    /// The most lanes the pools hold to their end at once, each from the
    /// prompt a finished one held to the pages it held at most
    /// ([`most_at_once`]); `None` until a process has finished. What a
    /// policy reads as `pie/lanes-to-end`.
    pub(super) fn lanes_to_end(&self, kv_total: u32, rs_total: RsUnits) -> Option<u64> {
        let (end, first) = self.footprint?;
        let fits = |n: usize| {
            let mut kv = vec![first.0; n];
            let mut pages = vec![first.1.pages; n];
            most_at_once(&mut kv, end.0) <= u64::from(kv_total)
                && (rs_total.pages == 0
                    || most_at_once(&mut pages, end.1.pages) <= u64::from(rs_total.pages))
                && (rs_total.slots == 0
                    || n as u64 * u64::from(end.1.slots) <= u64::from(rs_total.slots))
        };
        // The pools bound it: each lane holds a page at least.
        let (mut fit, mut over) = (0usize, kv_total as usize + 1);
        while over - fit > 1 {
            let mid = fit + (over - fit) / 2;
            if fits(mid) {
                fit = mid;
            } else {
                over = mid;
            }
        }
        Some(fit as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::{RsUnits, ToEnd, most_at_once};

    #[test]
    fn the_lanes_to_the_end_are_what_the_pools_hold_at_their_peak() {
        let none = RsUnits::default();
        let mut to_end = ToEnd::default();
        assert_eq!(to_end.lanes_to_end(1855, none), None, "nothing learned: absent");
        to_end.learn((27, none), (11, none));
        // Started together, each holds 27 at its end: 1855 / 27.
        assert_eq!(to_end.lanes_to_end(1855, none), Some(68));
        assert_eq!(to_end.lanes_to_end(26, none), Some(0), "not one fits to its end");
    }

    #[test]
    fn lanes_started_apart_need_less_than_each_to_the_end() {
        // In step, each holds its end at once.
        assert_eq!(most_at_once(&mut [20, 20, 20], 27), 81);
        // Apart, the oldest lets go before the youngest is grown.
        assert_eq!(most_at_once(&mut [11, 19, 27], 27), 27 + 19 + 11);
        assert_eq!(most_at_once(&mut [11, 19, 26], 27), 27 + 20 + 12);
        assert_eq!(most_at_once(&mut [11, 11], 27), 54);
        assert_eq!(most_at_once(&mut [30], 27), 27);
    }
}
