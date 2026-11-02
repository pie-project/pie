use std::collections::{BTreeSet, HashSet};

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

    /// Whether one more process fits beside the live ones to the end.
    pub(super) fn room(
        &self,
        procs: &std::collections::HashMap<ProcessId, Proc>,
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
        let mut kv: Vec<u32> = live.iter().map(|proc| proc.peak.0).collect();
        kv.push(first.0);
        let mut pages: Vec<u32> = live.iter().map(|proc| proc.peak.1.pages).collect();
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
                "TOEND room live={} kv={kv_most}/{kv_total} pages={pages_most}/{} fp={:?}",
                live.len(),
                rs_total.pages,
                self.footprint
            ));
        }
        kv_most <= u64::from(kv_total)
            && pages_most <= u64::from(rs_total.pages)
            && slots <= u64::from(rs_total.slots)
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
