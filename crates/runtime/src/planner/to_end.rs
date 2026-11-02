use std::collections::{BTreeSet, HashMap, HashSet};
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
    /// Started past the gate, not yet gone.
    pub(super) live: HashSet<ProcessId>,
    /// Waiting at the gate, oldest first.
    pub(super) waiting: BTreeSet<(Rank, ProcessId)>,
    /// When each live one passed the gate.
    since: HashMap<ProcessId, Instant>,
    /// How long a process lives past the gate, averaged.
    life: Option<Duration>,
    /// Starts banked by the ramp, and when it was last topped up.
    bank: f64,
    topped: Option<Instant>,
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
    /// Records `pid` passing the gate.
    pub(super) fn pass(&mut self, pid: ProcessId, now: Instant) {
        self.live.insert(pid);
        self.since.insert(pid, now);
    }

    /// Forgets `pid`; what it lived, if it finished, is folded in.
    pub(super) fn leave(&mut self, pid: ProcessId, finished: bool) -> bool {
        let since = self.since.remove(&pid);
        if finished && let Some(since) = since {
            let lived = since.elapsed();
            self.life = Some(self.life.map_or(lived, |life| (life * 3 + lived) / 4));
        }
        self.live.remove(&pid)
    }

    /// What a finished process held at most and at its prompt.
    pub(super) fn learn(&mut self, peak: Units, first: Units) {
        dbg_line(format!(
            "TOEND learn peak={peak:?} first={first:?} fp={:?} live={}",
            self.footprint,
            self.live.len()
        ));
        self.footprint = Some(match self.footprint {
            None => (peak, first),
            Some((p, f)) => (fold(p, peak), fold(f, first)),
        });
    }

    /// Whether one more process starts now: `Err(None)` waits for a process
    /// to leave, `Err(Some(d))` for `d` more of the ramp.
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
            .iter()
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
        if self.live.len() % 16 == 0 {
            dbg_line(format!(
                "TOEND room live={} kv={kv_most}/{kv_total} pages={pages_most}/{} fp={:?} life={:?} bank={:.2}",
                live.len(),
                rs_total.pages,
                self.footprint,
                self.life,
                self.bank
            ));
        }
        if kv_most > u64::from(kv_total)
            || pages_most > u64::from(rs_total.pages)
            || slots > u64::from(rs_total.slots)
        {
            return Err(None);
        }
        // Where more want to run than the pools hold to the end at once,
        // starts are spread over a life, so the lanes finish apart and the
        // pools hold more of them, each midway, than a batch started at once.
        let Some(life) = self.life else {
            return Ok(());
        };
        let fit = |total: u32, end: u32| u64::from(total) / u64::from(end.max(1));
        let at_once = fit(kv_total, end.0).min(fit(rs_total.pages, end.1.pages));
        if (self.live.len() + self.waiting.len()) as u64 <= at_once {
            self.topped = None;
            return Ok(());
        }
        let mid = |total: u32, end: u32, first: u32| {
            2 * u64::from(total) / u64::from((end + first).max(1))
        };
        let steady =
            mid(kv_total, end.0, first.0).min(mid(rs_total.pages, end.1.pages, first.1.pages));
        let knob = |name: &str, default: f64| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(default)
        };
        let rate = knob("PIE_TOEND_RATE", 1.0) * steady as f64 / life.as_secs_f64().max(1e-3);
        let cap = (steady as f64 / knob("PIE_TOEND_BURST", 8.0)).max(1.0);
        self.bank = match self.topped {
            None => cap,
            Some(at) => (self.bank + rate * now.duration_since(at).as_secs_f64()).min(cap),
        };
        self.topped = Some(now);
        if self.bank >= 1.0 {
            self.bank -= 1.0;
            Ok(())
        } else {
            Err(Some(Duration::from_secs_f64((1.0 - self.bank) / rate)))
        }
    }
}

pub(super) fn dbg_line(s: String) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/root/toend.log")
    {
        let _ = writeln!(f, "{s}");
    }
}
