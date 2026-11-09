use std::collections::{BTreeSet, HashMap};
use std::time::{Duration, Instant};

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
    /// Started past the gate, not yet gone, and when each passed it.
    pub(super) live: HashMap<ProcessId, Instant>,
    /// Waiting at the gate, oldest first.
    pub(super) waiting: BTreeSet<(Rank, ProcessId)>,
    /// `PoolFacts::seat_spread`: see [`ToEnd::spread_starts`].
    pub(super) spread: bool,
    /// How long a finished process lived past the gate, averaged.
    life: Option<Duration>,
    /// When more wanted to start than the pools hold to the end at once.
    bound: Option<Instant>,
    /// Starts banked by the spread, when it was last topped up, and whether
    /// a group is starting.
    bank: f64,
    topped: Option<Instant>,
    open: bool,
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
    /// What a finished process held at most and at its prompt, and how
    /// long it lived past the gate.
    pub(super) fn learn(&mut self, peak: Units, first: Units, lived: Duration) {
        self.footprint = Some(match self.footprint {
            None => (peak, first),
            Some((p, f)) => (fold(p, peak), fold(f, first)),
        });
        self.life = Some(self.life.map_or(lived, |life| (life * 3 + lived) / 4));
    }

    /// Whether one more process starts now beside the live ones: `Err(None)`
    /// waits for one to leave, `Err(Some(d))` for `d` more of the spread.
    pub(super) fn room(
        &mut self,
        procs: &HashMap<ProcessId, Proc>,
        kv_total: u32,
        rs_total: RsUnits,
        now: Instant,
    ) -> Result<(), Option<Duration>> {
        let Some((end, first)) = self.footprint else {
            return Ok(());
        };
        let live: Vec<&Proc> = self
            .live
            .keys()
            .filter_map(|pid| procs.get(pid))
            .filter(|proc| !proc.idle)
            .collect();
        if live.is_empty() {
            return Ok(());
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
        let kv_most = most_at_once(&mut kv, end.0);
        let pages_most = most_at_once(&mut pages, end.1.pages);
        let over: u64 = std::env::var("PIE_TO_END_OVER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        if kv_most > u64::from(kv_total) * (100 + over) / 100
            || pages_most > u64::from(rs_total.pages)
            || slots > u64::from(rs_total.slots)
        {
            self.bound = Some(now);
            return Err(None);
        }
        if self.spread {
            return self.spread_starts(kv_total, rs_total, end, first, now);
        }
        Ok(())
    }

    /// Where more want to start than the pools hold to the end at once,
    /// starts are spread over a life: the lanes finish apart, and the pools
    /// hold more of them, each midway, than a wave started at once, which
    /// runs to its end together while the next waits for all of it. Spread
    /// while that held within the last two lives, so the wave a finished
    /// one leaves is spread too.
    fn spread_starts(
        &mut self,
        kv_total: u32,
        rs_total: RsUnits,
        end: Units,
        first: Units,
        now: Instant,
    ) -> Result<(), Option<Duration>> {
        let Some(life) = self.life else {
            return Ok(());
        };
        // A pool of no pages (no windowed rings) bounds nothing.
        let fit = |total: u32, end: u32| {
            if total == 0 {
                u64::MAX
            } else {
                u64::from(total) / u64::from(end.max(1))
            }
        };
        let mid = |total: u32, end: u32, first: u32| {
            if total == 0 {
                u64::MAX
            } else {
                2 * u64::from(total) / u64::from((end + first).max(1))
            }
        };
        let at_once = fit(kv_total, end.0).min(fit(rs_total.pages, end.1.pages));
        let steady =
            mid(kv_total, end.0, first.0).min(mid(rs_total.pages, end.1.pages, first.1.pages));
        // Lanes that barely grow past their prompts (a prefill) hold as much
        // started together as apart: spread only for an eighth more lanes.
        if steady.saturating_mul(8) < at_once.saturating_mul(9) {
            return Ok(());
        }
        // More may start at once than that: the starts the admission seats
        // (a burst of them reaches the gate one at a time), or those here.
        let seats = crate::inferlet::process::execution_slot_capacity().unwrap_or(0);
        if seats.max(self.live.len() + self.waiting.len()) as u64 > at_once {
            self.bound = Some(now);
        }
        if self
            .bound
            .is_none_or(|at| now.duration_since(at) >= life * 2)
        {
            self.topped = None;
            return Ok(());
        }
        // The lanes the pools hold at once, each midway, start over a life
        // in eight groups, so a group's prompts share their fires.
        let rate = steady as f64 / life.as_secs_f64().max(1e-3);
        let group = (steady as f64 / 8.0).max(1.0);
        self.bank = match self.topped {
            None => group,
            Some(at) => (self.bank + rate * now.duration_since(at).as_secs_f64()).min(group),
        };
        self.topped = Some(now);
        if self.bank >= if self.open { 1.0 } else { group } {
            self.bank -= 1.0;
            self.open = self.bank >= 1.0;
            Ok(())
        } else {
            self.open = false;
            Err(Some(Duration::from_secs_f64((group - self.bank) / rate)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Duration, HashMap, Instant, Proc, ProcessId, RsUnits, ToEnd, most_at_once};

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

    #[test]
    fn a_wave_the_pools_hold_only_apart_starts_spread_over_a_life() {
        let none = RsUnits { slots: 0, pages: 0 };
        let start = |spread: bool, now: Instant| {
            let mut to_end = ToEnd {
                spread,
                ..ToEnd::default()
            };
            // A finished lane held 27 pages, 11 at its prompt, for 20 s.
            to_end.learn((27, none), (11, none), Duration::from_secs(20));
            let mut procs = HashMap::new();
            // 128 want to start where 1855 pages hold 68 to the end at once.
            for seq in 0..128 {
                let pid = ProcessId::new_v4();
                let proc = Proc::new(seq);
                to_end.waiting.insert((proc.rank(), pid));
                procs.insert(pid, proc);
            }
            let mut started = 0;
            let mut wait = None;
            for pid in procs.keys() {
                match to_end.room(&procs, 1855, none, now) {
                    Ok(()) => {
                        to_end.waiting.retain(|(_, waiting)| waiting != pid);
                        to_end.live.insert(*pid, now);
                        started += 1;
                    }
                    Err(w) => {
                        wait = w;
                        break;
                    }
                }
            }
            (to_end, procs, started, wait)
        };
        let now = Instant::now();
        // Seated to the end alone, the wave the pools hold to the end starts.
        let (_, _, started, wait) = start(false, now);
        assert_eq!((started, wait), (68, None));
        // Spread, a group of an eighth of the 97 the pools hold midway starts
        // beside the first, and the next an eighth of the 20 s life later.
        let (mut to_end, procs, started, wait) = start(true, now);
        assert_eq!(started, 1 + 12);
        let wait = wait.expect("the spread holds the next group back");
        assert!(wait > Duration::from_millis(2400) && wait < Duration::from_millis(2600));
        let later = now + wait + Duration::from_millis(1);
        assert_eq!(to_end.room(&procs, 1855, none, later), Ok(()));
    }
}
