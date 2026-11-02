use std::collections::{BTreeSet, HashSet};

use crate::store::rs::RsUnits;

use super::ProcessId;
use super::proc::Proc;
use super::rank::Rank;

/// Seating to the end (`PoolFacts::seat_to_end`): a process starts only
/// while the pools hold every started one to the pages a finished one
/// held, so lanes seated past what the pools carry do not park at each
/// page boundary and get evicted and replayed, as a queue keeps them out.
#[derive(Default)]
pub(super) struct ToEnd {
    /// The device kv pages and rs units a finished process held, averaged.
    footprint: Option<(u32, RsUnits)>,
    /// Started past the gate, not yet gone.
    pub(super) live: HashSet<ProcessId>,
    /// Waiting at the gate, oldest first.
    pub(super) waiting: BTreeSet<(Rank, ProcessId)>,
}

impl ToEnd {
    /// What a finished process held, folded into the footprint.
    pub(super) fn learn(&mut self, kv: u32, rs: RsUnits) {
        eprintln!("TOEND learn kv={kv} rs={rs:?} fp={:?} live={}", self.footprint, self.live.len());
        self.footprint = Some(match self.footprint {
            None => (kv, rs),
            Some((k, r)) => (
                (3 * k + kv).div_ceil(4),
                RsUnits {
                    slots: (3 * r.slots + rs.slots).div_ceil(4),
                    pages: (3 * r.pages + rs.pages).div_ceil(4),
                },
            ),
        });
    }

    /// Whether one more process fits beside the live ones at the footprint.
    pub(super) fn room(
        &self,
        procs: &std::collections::HashMap<ProcessId, Proc>,
        kv_total: u32,
        rs_total: RsUnits,
    ) -> bool {
        let Some((kv, rs)) = self.footprint else {
            return true;
        };
        let live = self
            .live
            .iter()
            .filter(|pid| procs.get(pid).is_some_and(|proc| !proc.idle))
            .count() as u64;
        if self.live.len() % 16 == 0 {
            eprintln!("TOEND room live={live} kv_total={kv_total} rs_total={rs_total:?} fp={:?}", self.footprint);
        }
        let fits = |each: u32, total: u32| (live + 1) * u64::from(each) <= u64::from(total);
        live == 0
            || (fits(kv, kv_total)
                && fits(rs.slots, rs_total.slots)
                && fits(rs.pages, rs_total.pages))
    }
}
