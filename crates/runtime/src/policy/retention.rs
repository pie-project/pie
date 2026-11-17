//! The retention table: keys on cached KV, the lowest dropped first.
//!
//! An entry keys a working-set's first `upto` tokens, or all of it, and a
//! position takes the highest key among the entries covering it. One no entry
//! covers is at the [`FLOOR`].

use std::collections::HashMap;

use parking_lot::RwLock;

use crate::inferlet::process::ProcessId;

/// The key of KV no entry covers: what every position takes with no policy
/// installed. Idle cache at or below it is the runtime's to drop when its
/// room is wanted; only cache keyed above it is worth a copy down a tier.
pub const FLOOR: i64 = 0;

/// Whether idle cache under `key` is kept above the floor: only such cache
/// moves down a tier (device -> host -> disk) when its room is wanted; the
/// rest is dropped. Suspended or parked working sets move whatever their key.
pub const fn above_floor(key: i64) -> bool {
    key > FLOOR
}

/// Every tier: device, host, disk (bits 0, 1, 2), as `retention/tiers`
/// has them.
pub const ALL_TIERS: u8 = 0b111;
pub const DEVICE: u8 = 0b001;
pub const HOST: u8 = 0b010;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Table {
    steps: HashMap<ProcessId, Vec<(Option<u64>, i64)>>,
    /// The column's default: the key of KV no row covers.
    default: Option<i64>,
    /// `retention/tiers` rows: where KV may sit, by owner and prefix.
    tiers: HashMap<ProcessId, Vec<(Option<u64>, u8)>>,
    tiers_default: Option<u8>,
}

impl Table {
    pub fn new(steps: HashMap<ProcessId, Vec<(Option<u64>, i64)>>, default: Option<i64>) -> Self {
        Self {
            steps,
            default,
            ..Self::default()
        }
    }

    /// With the tiers column: rows by owner and prefix, and its default.
    #[must_use]
    pub fn with_tiers(
        mut self,
        tiers: HashMap<ProcessId, Vec<(Option<u64>, u8)>>,
        default: Option<u8>,
    ) -> Self {
        self.tiers = tiers;
        self.tiers_default = default;
        self
    }

    /// The tiers `owner`'s KV at token `at` may sit on: every row covering
    /// it, together (the most protective), else the default, else all.
    fn tiers_at(&self, owner: ProcessId, at: u64) -> u8 {
        self.tiers
            .get(&crate::policy::schedule::plex_id(owner))
            .and_then(|rows| {
                rows.iter()
                    .filter(|(upto, _)| upto.is_none_or(|upto| at < upto))
                    .map(|&(_, tiers)| tiers)
                    .reduce(|a, b| a | b)
            })
            .or(self.tiers_default)
            .unwrap_or(ALL_TIERS)
    }

    fn key_at(&self, owner: ProcessId, at: u64) -> Option<i64> {
        self.steps
            .get(&crate::policy::schedule::plex_id(owner))
            .and_then(|steps| {
                steps
                    .iter()
                    .filter(|(upto, _)| upto.is_none_or(|upto| at < upto))
                    .map(|&(_, key)| key)
                    .max()
            })
            .or(self.default)
    }
}

/// What the stores read of the retention table.
pub trait Keys {
    /// The key of `owner`'s KV at token `at`.
    fn key_at(&self, owner: Option<ProcessId>, at: u64) -> i64;

    /// Where `owner`'s keys may step down, in tokens.
    fn steps(&self, _owner: ProcessId) -> Vec<u64> {
        Vec::new()
    }

    /// The tiers `owner`'s KV at token `at` may sit on ([`ALL_TIERS`]
    /// unless a row says otherwise).
    fn tiers_at(&self, _owner: Option<ProcessId>, _at: u64) -> u8 {
        ALL_TIERS
    }
}

impl<F: Fn(Option<ProcessId>) -> i64> Keys for F {
    fn key_at(&self, owner: Option<ProcessId>, _at: u64) -> i64 {
        self(owner)
    }
}

/// The installed table.
pub struct Installed;

impl Keys for Installed {
    fn key_at(&self, owner: Option<ProcessId>, at: u64) -> i64 {
        let current = CURRENT.read();
        let Some(table) = current.as_ref() else {
            return FLOOR;
        };
        owner
            .and_then(|owner| table.key_at(owner, at))
            .or(table.default)
            .unwrap_or(FLOOR)
    }

    fn steps(&self, owner: ProcessId) -> Vec<u64> {
        let current = CURRENT.read();
        let Some(table) = current.as_ref() else {
            return Vec::new();
        };
        let id = crate::policy::schedule::plex_id(owner);
        // A key or a tier may change where a row of either ends.
        let keys = table
            .steps
            .get(&id)
            .into_iter()
            .flatten()
            .filter_map(|&(upto, _)| upto);
        let tiers = table
            .tiers
            .get(&id)
            .into_iter()
            .flatten()
            .filter_map(|&(upto, _)| upto);
        keys.chain(tiers).collect()
    }

    fn tiers_at(&self, owner: Option<ProcessId>, at: u64) -> u8 {
        let current = CURRENT.read();
        let Some(table) = current.as_ref() else {
            return ALL_TIERS;
        };
        match owner {
            Some(owner) => table.tiers_at(owner, at),
            None => table.tiers_default.unwrap_or(ALL_TIERS),
        }
    }
}

static CURRENT: RwLock<Option<Table>> = RwLock::new(None);

pub(crate) fn install(retention: Option<Table>) {
    *CURRENT.write() = retention;
    sweep();
}

/// Idle cache appeared since the planners last looked: a tiers row may
/// apply to it.
static IDLE_CACHE_NEW: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The store made idle cache (a working set released, or pages landed).
pub(crate) fn idle_cache_appeared() {
    IDLE_CACHE_NEW.store(true, std::sync::atomic::Ordering::Release);
}

/// Whether idle cache appeared since the last call, where a tiers row is
/// installed to apply to it.
pub(crate) fn take_idle_cache_new() -> bool {
    IDLE_CACHE_NEW.swap(false, std::sync::atomic::Ordering::AcqRel)
        && CURRENT
            .read()
            .as_ref()
            .is_some_and(|table| !table.tiers.is_empty() || table.tiers_default.is_some())
}

/// What the tiers rows keep off a tier leaves it now: on every planner,
/// idle cache allowed nowhere it is is dropped, and cache kept off the
/// device moves down. Also when a process's KV goes idle.
pub(crate) fn sweep() {
    let tiered = CURRENT
        .read()
        .as_ref()
        .is_some_and(|table| !table.tiers.is_empty() || table.tiers_default.is_some());
    if !tiered {
        return;
    }
    idle_cache_appeared();
    for planner in crate::planner::planners() {
        planner.sweep_wanted();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_position_takes_the_highest_key_covering_it_and_none_is_0() {
        let (owner, other) = (ProcessId::from_u128(0xf3), ProcessId::from_u128(0xf4));
        install(Some(Table::new(
            HashMap::from([(owner, vec![(None, -1), (Some(100), 5)])]),
            None,
        )));
        assert_eq!(Installed.key_at(Some(owner), 99), 5);
        assert_eq!(Installed.key_at(Some(owner), 100), -1);
        assert_eq!(Installed.steps(owner), [100]);
        assert_eq!(Installed.key_at(Some(other), 0), FLOOR, "unnamed");
        install(Some(Table::new(
            HashMap::from([(owner, vec![(Some(10), 3)])]),
            None,
        )));
        assert_eq!(Installed.key_at(Some(owner), 10), 0, "past every entry");
        install(None);
    }
}
