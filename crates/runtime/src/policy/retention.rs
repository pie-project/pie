//! The retention table (PLEX 1.1 `retention/priority` and `retention/tiers`):
//! keys on cached KV, the lowest let go first, and the tiers it may sit on.
//!
//! A row keys a working set's first `upto` tokens, or all of it. A position
//! takes the highest key among the rows covering it, and may sit on any tier
//! one of them names. One no row covers takes the column's default: key
//! [`DEFAULT_KEY`] (ties fall to least recently used), every tier.
//!
//! Whether idle cache moves down a tier or is dropped is the tiers column's
//! say alone: cache that may sit on a lower tier with room moves there; what
//! may not is dropped when its room is wanted. The key orders both.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;

use crate::inferlet::process::ProcessId;

/// The column's own default key (`retention/priority`: "0: the engine's
/// order"): what every position takes with no row and no policy default.
pub const DEFAULT_KEY: i64 = 0;

/// Every tier: device, host, disk (bits 0, 1, 2), as `retention/tiers`
/// has them.
pub const ALL_TIERS: u8 = 0b111;
pub const DEVICE: u8 = 0b001;
pub const HOST: u8 = 0b010;
pub const DISK: u8 = 0b100;
/// The tiers below the device.
const BELOW: u8 = HOST | DISK;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Table {
    steps: HashMap<ProcessId, Vec<(Option<u64>, i64)>>,
    /// The column's default: the key of KV no row covers.
    default: Option<i64>,
    /// `retention/tiers` rows: where KV may sit, by owner and prefix.
    tiers: HashMap<ProcessId, Vec<(Option<u64>, u8)>>,
    tiers_default: Option<u8>,
    /// The tiers rows that keep KV off the device.
    off_device: usize,
    /// The tiers rows that let KV sit below the device.
    below_device: usize,
    /// The owners each install named, oldest first, the last at the table's
    /// revision: no more of them than there are rows, past which all the
    /// owners are read again for no more.
    changes: std::collections::VecDeque<(u64, ProcessId)>,
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
        self.off_device = tiers
            .values()
            .flatten()
            .filter(|(_, row)| row & DEVICE == 0)
            .count();
        self.below_device = tiers
            .values()
            .flatten()
            .filter(|(_, row)| row & BELOW != 0)
            .count();
        self.tiers = tiers;
        self.tiers_default = default;
        self
    }

    /// The owners named after `revision`, where all of them are still held.
    fn changed_since(&self, revision: u64) -> Option<Vec<ProcessId>> {
        let after = self.changes.partition_point(|&(at, _)| at <= revision);
        (after > 0).then(|| {
            self.changes
                .range(after..)
                .map(|&(_, owner)| owner)
                .collect()
        })
    }

    /// Install `revision` named `owners`; all of them, where a default moved.
    fn note_changes(
        &mut self,
        revision: u64,
        owners: &std::collections::HashSet<ProcessId>,
        all: bool,
    ) {
        if all {
            self.changes.clear();
        }
        self.changes
            .extend(owners.iter().map(|&owner| (revision, owner)));
        let most = self.steps.len() + self.tiers.len() + owners.len();
        while self.changes.len() > most {
            self.changes.pop_front();
        }
    }

    /// The tiers `owner`'s KV at token `at` may sit on: every row covering
    /// it, together (the most protective), else the default, else all.
    fn tiers_at(&self, owner: ProcessId, at: u64) -> u8 {
        self.rows_at(owner, at)
            .or(self.tiers_default)
            .unwrap_or_else(own_tiers)
    }

    /// What the rows covering `owner`'s token `at` say, together; `None`
    /// where none does.
    fn rows_at(&self, owner: ProcessId, at: u64) -> Option<u8> {
        self.tiers
            .get(&crate::policy::schedule::plex_id(owner))
            .and_then(|rows| {
                rows.iter()
                    .filter(|(upto, _)| upto.is_none_or(|upto| at < upto))
                    .map(|&(_, tiers)| tiers)
                    .reduce(|a, b| a | b)
            })
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

    /// The table's revision: a key read under one revision stands until the
    /// next. `None`: the keys say nothing of when they change, and nothing
    /// read under them may be kept.
    fn revision(&self) -> Option<u64> {
        None
    }

    /// The owners whose rows changed since `revision`, where they say; `None`:
    /// any may have.
    fn changed_since(&self, _revision: u64) -> Option<Vec<ProcessId>> {
        None
    }

    /// The tiers idle cache of `owner`'s at token `at` may sit on: a row's
    /// or the policy's default; with neither, the device alone
    /// ([`Installed`]'s).
    fn cache_tiers_at(&self, owner: Option<ProcessId>, at: u64) -> u8 {
        self.tiers_at(owner, at)
    }

    /// Whether any idle cache may sit below the device
    /// ([`Self::cache_tiers_at`]): `false` where none may, and nothing is
    /// to be moved down for room.
    fn cache_may_go_below(&self) -> bool {
        true
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
            return DEFAULT_KEY;
        };
        owner
            .and_then(|owner| table.key_at(owner, at))
            .or(table.default)
            .unwrap_or(DEFAULT_KEY)
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
            return own_tiers();
        };
        match owner {
            Some(owner) => table.tiers_at(owner, at),
            None => table.tiers_default.unwrap_or_else(own_tiers),
        }
    }

    /// Only what a policy keys is kept: idle cache no row and no default
    /// covers (under `noop`, and from an exit until the policy's round) is
    /// kept on the device alone, and dropped when its room is wanted, never
    /// moved down. This departs from PLEX 1.1, whose tiers column defaults
    /// to every tier the worker has; live, suspended and resting sets keep
    /// that default ([`Self::tiers_at`]).
    fn cache_tiers_at(&self, owner: Option<ProcessId>, at: u64) -> u8 {
        let current = CURRENT.read();
        let Some(table) = current.as_ref() else {
            return DEVICE;
        };
        owner
            .and_then(|owner| table.rows_at(owner, at))
            .or(table.tiers_default)
            .unwrap_or(DEVICE)
    }

    fn cache_may_go_below(&self) -> bool {
        CURRENT.read().as_ref().is_some_and(|table| {
            table.below_device > 0 || table.tiers_default.is_some_and(|row| row & BELOW != 0)
        })
    }

    fn revision(&self) -> Option<u64> {
        Some(REVISION.load(Ordering::Acquire))
    }

    fn changed_since(&self, revision: u64) -> Option<Vec<ProcessId>> {
        let current = CURRENT.read();
        if REVISION.load(Ordering::Acquire) == revision {
            return Some(Vec::new());
        }
        current.as_ref()?.changed_since(revision)
    }
}

static CURRENT: RwLock<Option<Table>> = RwLock::new(None);

/// Bumped by each install, under the table's write lock.
static REVISION: AtomicU64 = AtomicU64::new(0);

/// The tiers column's own default: every tier the worker has.
const fn own_tiers() -> u8 {
    ALL_TIERS
}

/// Forgets the policy's table, for [`crate::bootstrap::reset`].
pub(crate) fn reset() {
    IDLE_CACHE_NEW.store(false, std::sync::atomic::Ordering::Release);
    *CURRENT.write() = None;
    REVISION.fetch_add(1, Ordering::AcqRel);
}

#[cfg(test)]
pub(crate) fn install(retention: Option<Table>) {
    {
        let mut current = CURRENT.write();
        *current = retention;
        REVISION.fetch_add(1, Ordering::AcqRel);
    }
    crate::planner::rekey_holders(None);
    sweep();
}

/// A row: an owner's first `upto` tokens (all of them: `None`).
pub type Prefix = (ProcessId, Option<u64>);

/// What a policy round changed: rows set (`Some`) or deleted (`None`), and
/// defaults changed (`Some(None)`: back to the column's own).
#[derive(Debug, Default)]
pub struct Change {
    pub keys: Vec<(Prefix, Option<i64>)>,
    pub key_default: Option<Option<i64>>,
    pub tiers: Vec<(Prefix, Option<u8>)>,
    pub tiers_default: Option<Option<u8>>,
}

impl Change {
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
            && self.key_default.is_none()
            && self.tiers.is_empty()
            && self.tiers_default.is_none()
    }
}

/// One owner's rows of a column with the row at `upto` set or deleted.
/// Answers the row that stood there.
fn put<V>(
    rows: &mut HashMap<ProcessId, Vec<(Option<u64>, V)>>,
    (owner, upto): Prefix,
    row: Option<V>,
) -> Option<V> {
    let list = rows.entry(owner).or_default();
    let stood = list
        .iter()
        .position(|(at, _)| *at == upto)
        .map(|at| list.remove(at).1);
    if let Some(value) = row {
        list.push((upto, value));
    }
    if list.is_empty() {
        rows.remove(&owner);
    }
    stood
}

/// Enacts a round's changes in place: only the rows named change, and only
/// the holders they name are keyed again (all of them where a default
/// changed). A tiers change asks the planners for a sweep.
pub(crate) fn apply(change: Change) {
    if change.is_empty() {
        return;
    }
    let all = change.key_default.is_some() || change.tiers_default.is_some();
    let touched: std::collections::HashSet<ProcessId> = change
        .keys
        .iter()
        .map(|((owner, _), _)| *owner)
        .chain(change.tiers.iter().map(|((owner, _), _)| *owner))
        .collect();
    let tiered = !change.tiers.is_empty() || change.tiers_default.is_some();
    {
        let mut current = CURRENT.write();
        let table = current.get_or_insert_with(Table::default);
        for (prefix, row) in change.keys {
            put(&mut table.steps, prefix, row);
        }
        for (prefix, row) in change.tiers {
            let off = |row: &Option<u8>| usize::from(row.is_some_and(|row| row & DEVICE == 0));
            table.off_device += off(&row);
            let stood = put(&mut table.tiers, prefix, row);
            table.off_device -= off(&stood);
            let below = |row: &Option<u8>| usize::from(row.is_some_and(|row| row & BELOW != 0));
            table.below_device += below(&row);
            table.below_device -= below(&stood);
        }
        if let Some(default) = change.key_default {
            table.default = default;
        }
        if let Some(default) = change.tiers_default {
            table.tiers_default = default;
        }
        let revision = REVISION.fetch_add(1, Ordering::AcqRel) + 1;
        table.note_changes(revision, &touched, all);
    }
    crate::planner::rekey_holders((!all).then_some(&touched));
    if tiered {
        // A row taken back (every tier again) has nothing to sweep, but
        // one set or a default does.
        idle_cache_appeared();
        for planner in crate::planner::planners() {
            planner.sweep_wanted();
        }
    }
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

/// Whether a tiers row or the default keeps KV off the device: with none,
/// no cache leaves the device but for its room.
pub(crate) fn keeps_off_device() -> bool {
    CURRENT.read().as_ref().is_some_and(|table| {
        table.off_device > 0 || table.tiers_default.is_some_and(|row| row & DEVICE == 0)
    })
}

/// What the tiers rows keep off a tier leaves it now: on every planner,
/// idle cache allowed nowhere it is is dropped, and cache kept off the
/// device moves down. Also when a process's KV goes idle.
#[cfg(test)]
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
        let _lock = crate::policy::schedule::TEST_LOCK.blocking_write();
        let (owner, other) = (ProcessId::from_u128(0xf3), ProcessId::from_u128(0xf4));
        install(Some(Table::new(
            HashMap::from([(owner, vec![(None, -1), (Some(100), 5)])]),
            None,
        )));
        assert_eq!(Installed.key_at(Some(owner), 99), 5);
        assert_eq!(Installed.key_at(Some(owner), 100), -1);
        assert_eq!(Installed.steps(owner), [100]);
        assert_eq!(Installed.key_at(Some(other), 0), DEFAULT_KEY, "unnamed");
        install(Some(Table::new(
            HashMap::from([(owner, vec![(Some(10), 3)])]),
            None,
        )));
        assert_eq!(Installed.key_at(Some(owner), 10), 0, "past every entry");
        install(None);
    }

    /// A table whose tiers default is the device alone (the shipped
    /// default policy's idle cache) holds every owner no row names there.
    #[test]
    fn rows_that_keep_kv_off_the_device_or_let_it_below_are_counted() {
        let _lock = crate::policy::schedule::TEST_LOCK.blocking_write();
        let owner = ProcessId::from_u128(0xf6);
        install(None);
        let row = |tiers| Change {
            tiers: vec![((owner, None), tiers)],
            ..Change::default()
        };
        apply(row(Some(DEVICE)));
        assert!(!keeps_off_device());
        apply(row(Some(HOST)));
        apply(row(Some(HOST | DISK)));
        assert!(keeps_off_device());
        assert!(Installed.cache_may_go_below());
        apply(row(None));
        assert!(!keeps_off_device());
        assert!(!Installed.cache_may_go_below(), "no row lets cache below");
        install(None);
    }

    #[test]
    fn a_device_only_tiers_default_holds_unnamed_owners_on_the_device() {
        let _lock = crate::policy::schedule::TEST_LOCK.blocking_write();
        let (named, unnamed) = (ProcessId::from_u128(0xd1), ProcessId::from_u128(0xd2));
        install(None);
        assert_eq!(
            Installed.tiers_at(Some(unnamed), 0),
            ALL_TIERS,
            "no policy: every tier"
        );
        install(Some(Table::new(HashMap::new(), None).with_tiers(
            HashMap::from([(named, vec![(None, ALL_TIERS)])]),
            Some(DEVICE),
        )));
        assert_eq!(Installed.tiers_at(Some(unnamed), 0), DEVICE);
        assert_eq!(Installed.tiers_at(None, 0), DEVICE);
        assert_eq!(Installed.tiers_at(Some(named), 0), ALL_TIERS, "its row");
        install(None);
    }

    #[test]
    fn a_change_sets_and_deletes_only_the_rows_it_names() {
        let _lock = crate::policy::schedule::TEST_LOCK.blocking_write();
        let (a, b) = (ProcessId::from_u128(0xa1), ProcessId::from_u128(0xb1));
        install(None);
        apply(Change {
            keys: vec![
                ((a, None), Some(5)),
                ((a, Some(10)), Some(9)),
                ((b, None), Some(-3)),
            ],
            tiers: vec![((b, None), Some(HOST))],
            ..Change::default()
        });
        assert_eq!(Installed.key_at(Some(a), 3), 9);
        assert_eq!(Installed.key_at(Some(a), 10), 5);
        assert_eq!(Installed.key_at(Some(b), 0), -3);
        assert_eq!(Installed.tiers_at(Some(b), 0), HOST);
        assert_eq!(
            Installed.tiers_at(Some(a), 0),
            own_tiers(),
            "no row: the column's own"
        );
        apply(Change {
            keys: vec![((a, Some(10)), None)],
            key_default: Some(Some(2)),
            tiers: vec![((b, None), Some(0))],
            ..Change::default()
        });
        assert_eq!(Installed.key_at(Some(a), 3), 5, "the head row deleted");
        assert_eq!(
            Installed.key_at(Some(ProcessId::from_u128(0xc1)), 0),
            2,
            "the default"
        );
        assert_eq!(Installed.tiers_at(Some(b), 0), 0, "nowhere: dropped");
        assert_eq!(Installed.steps(a), Vec::<u64>::new());
        // A lease withdrawal is every row deleted and both defaults reset.
        apply(Change {
            keys: vec![((a, None), None), ((b, None), None)],
            key_default: Some(None),
            tiers: vec![((b, None), None)],
            tiers_default: Some(None),
        });
        let mut left = CURRENT.read().clone().expect("a table");
        left.changes.clear();
        assert_eq!(left, Table::default());
        install(None);
    }
}
