use std::collections::{BTreeSet, HashMap, HashSet};

use crate::store::rs::RsUnits;

use super::ProcessId;
use super::proc::Proc;
use super::rank::Rank;

/// Seating to the end (`PoolFacts::seat_to_end`): a process starts only
/// while the pools hold every started one to the pages a finished one
/// held, so lanes seated past what the pools carry do not park at each
/// page boundary and get evicted and replayed, as a queue keeps them out.
/// Lanes started apart finish apart: the pools need only hold the most the
/// live ones will hold at once as they grow and the oldest let go.
#[derive(Default)]
pub(super) struct ToEnd {
    /// The device kv pages and rs units a finished process held at most,
    /// and at its first grant (its prompt), averaged.
    footprint: Option<(Units, Units)>,
    /// Started past the gate, not yet gone.
    pub(super) live: HashSet<ProcessId>,
    /// Waiting at the gate, oldest first.
    pub(super) waiting: BTreeSet<(Rank, ProcessId)>,
}

pub(super) type Units = (u32, RsUnits);

fn fold(old: Units, new: Units) -> Units {
    let f = |a: u32, b: u32| (3 * a + b).div_ceil(4);
    (
        f(old.0, new.0),
        RsUnits {
            slots: f(old.1.slots, new.1.slots),
            pages: f(old.1.pages, new.1.pages),
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

    /// Whether one more process fits beside the live ones to the end.
    pub(super) fn room(
        &self,
        procs: &HashMap<ProcessId, Proc>,
        kv_total: u32,
        rs_total: RsUnits,
    ) -> bool {
        let Some((end, first)) = self.footprint else {
            return true;
        };
        let live: Vec<&Proc> = self
            .live
            .iter()
            .filter_map(|pid| procs.get(pid))
            .filter(|proc| !proc.idle)
            .collect();
        if live.is_empty() {
            return true;
        }
        // A lane not yet granted holds its prompt as soon as it is.
        let mut kv: Vec<u32> = live.iter().map(|proc| proc.peak.0.max(first.0)).collect();
        kv.push(first.0);
        let mut pages: Vec<u32> = live
            .iter()
            .map(|proc| proc.peak.1.pages.max(first.1.pages))
            .collect();
        pages.push(first.1.pages);
        let slots = live
            .iter()
            .map(|proc| u64::from(proc.peak.1.slots))
            .sum::<u64>()
            + u64::from(first.1.slots.max(end.1.slots));
        let young = kv.iter().filter(|&&held| held <= first.0 + 1).count() as u64;
        let kv_most = most_at_once(&mut kv, end.0);
        let pages_most = most_at_once(&mut pages, end.1.pages);
        if kv_most > u64::from(kv_total)
            || pages_most > u64::from(rs_total.pages)
            || slots > u64::from(rs_total.slots)
        {
            return false;
        }
        // Where more wait than the pools hold to the end at once, a page's
        // worth of starts at a time: lanes started in one burst finish in
        // one, and the pools then hold each to its end, where lanes spread
        // over a life hold half as much each on the way.
        let fit = |total: u32, each: u32| u64::from(total) / u64::from(each.max(1));
        let at_once = fit(kv_total, end.0).min(fit(rs_total.pages, end.1.pages));
        if (self.live.len() + self.waiting.len()) as u64 <= at_once {
            return true;
        }
        let spread = fit(kv_total, (end.0 + first.0).div_ceil(2)).min(fit(
            rs_total.pages,
            (end.1.pages + first.1.pages).div_ceil(2),
        ));
        let a_page = spread.div_ceil(u64::from(end.0.saturating_sub(first.0).max(1)));
        young <= a_page
    }
}

#[cfg(test)]
mod tests {
    use super::most_at_once;

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
