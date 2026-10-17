//! The retention table: keys on cached KV, the lowest dropped first.
//!
//! An entry keys a working-set's first `upto` tokens, or all of it, and a
//! position takes the highest key among the entries covering it. One no entry
//! covers is 0.

use std::collections::HashMap;

use parking_lot::RwLock;

use crate::inferlet::process::ProcessId;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Table {
    steps: HashMap<ProcessId, Vec<(Option<u64>, i64)>>,
}

impl Table {
    pub fn new(steps: HashMap<ProcessId, Vec<(Option<u64>, i64)>>) -> Self {
        Self { steps }
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
        owner
            .and_then(|owner| CURRENT.read().as_ref()?.key_at(owner, at))
            .unwrap_or(0)
    }

    fn steps(&self, owner: ProcessId) -> Vec<u64> {
        CURRENT
            .read()
            .as_ref()
            .and_then(|table| table.steps.get(&owner))
            .map(|steps| steps.iter().filter_map(|&(upto, _)| upto).collect())
            .unwrap_or_default()
    }
}

static CURRENT: RwLock<Option<Table>> = RwLock::new(None);

pub(crate) fn install(retention: Option<Table>) {
    *CURRENT.write() = retention;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_position_takes_the_highest_key_covering_it_and_none_is_0() {
        let (owner, other) = (ProcessId::from_u128(0xf3), ProcessId::from_u128(0xf4));
        install(Some(Table::new(HashMap::from([(
            owner,
            vec![(None, -1), (Some(100), 5)],
        )]))));
        assert_eq!(Installed.key_at(Some(owner), 99), 5);
        assert_eq!(Installed.key_at(Some(owner), 100), -1);
        assert_eq!(Installed.steps(owner), [100]);
        assert_eq!(Installed.key_at(Some(other), 0), 0, "unnamed");
        install(Some(Table::new(HashMap::from([(
            owner,
            vec![(Some(10), 3)],
        )]))));
        assert_eq!(Installed.key_at(Some(owner), 10), 0, "past every entry");
        install(None);
    }
}
