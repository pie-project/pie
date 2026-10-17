//! The retention table: keys on cached KV, the lowest dropped first.
//!
//! An entry keys a working-set's first `upto` tokens, or all of it, and a
//! position takes the highest key among the entries covering it. One no entry
//! covers takes its owner's schedule key, or its last once it exited, so KV is
//! no more exposed just after its owner exits. Below the floor KV is dropped
//! instead of demoted, and no checkpoint is taken.

use std::collections::HashMap;

use parking_lot::RwLock;

use super::schedule;
use crate::inferlet::process::ProcessId;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Table {
    steps: HashMap<ProcessId, Vec<(Option<u64>, i64)>>,
    floor: Option<i64>,
}

impl Table {
    pub fn new(steps: HashMap<ProcessId, Vec<(Option<u64>, i64)>>, floor: Option<i64>) -> Self {
        Self { steps, floor }
    }

    fn key_at(&self, owner: ProcessId, at: u64) -> Option<i64> {
        self.steps
            .get(&owner)?
            .iter()
            .filter(|(upto, _)| upto.is_none_or(|upto| at < upto))
            .map(|&(_, key)| key)
            .max()
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

    fn floor(&self) -> Option<i64> {
        None
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
        let Some(owner) = owner else {
            return 0;
        };
        if let Some(key) = CURRENT
            .read()
            .as_ref()
            .and_then(|table| table.key_at(owner, at))
        {
            return key;
        }
        if let Some(&key) = LAST_KEYS.read().as_ref().and_then(|keys| keys.get(&owner)) {
            return key;
        }
        schedule::key(owner)
    }

    fn steps(&self, owner: ProcessId) -> Vec<u64> {
        CURRENT
            .read()
            .as_ref()
            .and_then(|table| table.steps.get(&owner))
            .map(|steps| steps.iter().filter_map(|&(upto, _)| upto).collect())
            .unwrap_or_default()
    }

    fn floor(&self) -> Option<i64> {
        CURRENT.read().as_ref().and_then(|table| table.floor)
    }
}

static CURRENT: RwLock<Option<Table>> = RwLock::new(None);

static LAST_KEYS: RwLock<Option<HashMap<ProcessId, i64>>> = RwLock::new(None);

pub(crate) fn install(retention: Option<Table>) {
    *CURRENT.write() = retention;
}

pub fn names(owner: ProcessId) -> bool {
    CURRENT
        .read()
        .as_ref()
        .is_some_and(|table| table.steps.contains_key(&owner))
}

pub(crate) fn remember_exit(owner: ProcessId) {
    let key = schedule::key(owner);
    if key != 0 {
        LAST_KEYS
            .write()
            .get_or_insert_with(HashMap::new)
            .insert(owner, key);
    }
}

pub(crate) fn forget(owner: ProcessId) {
    if let Some(keys) = LAST_KEYS.write().as_mut() {
        keys.remove(&owner);
    }
}

pub fn below_floor_at(owner: ProcessId, at: u64) -> bool {
    Installed
        .floor()
        .is_some_and(|floor| Installed.key_at(Some(owner), at) < floor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_position_takes_the_highest_key_covering_it() {
        let (owner, other) = (ProcessId::from_u128(0xf3), ProcessId::from_u128(0xf4));
        install(Some(Table::new(
            HashMap::from([(owner, vec![(None, -1), (Some(100), 5)])]),
            Some(0),
        )));
        assert_eq!(Installed.key_at(Some(owner), 99), 5);
        assert_eq!(Installed.key_at(Some(owner), 100), -1);
        assert_eq!(Installed.steps(owner), [100]);
        assert!(!below_floor_at(owner, 0));
        assert!(below_floor_at(owner, 100));
        assert_eq!(
            Installed.key_at(Some(other), 0),
            0,
            "unnamed: its owner's key"
        );
        install(Some(Table::new(
            HashMap::from([(owner, vec![(Some(10), 3)])]),
            None,
        )));
        assert_eq!(
            Installed.key_at(Some(owner), 10),
            0,
            "past every entry: its owner's key"
        );
        install(None);
    }
}
