//! Cross-tenant share grants (pie's PLEX extension, `pie:plex-ext/share-groups`,
//! `crates/plex-host/EXTENSION.md`): which tenants may share KV under a named
//! group.
//!
//! The cluster policy writes the column at the controller. The controller
//! pushes the whole table, versioned, through every gateway to every worker
//! it links. A worker holds the newest version it was sent while some gateway
//! link stands. The KV hash domain of a `share(group(g))` ask is the model's
//! identity, "group", `g` and the grant's id where [`granted`] says the
//! asker's tenant is in, and is tenant-local otherwise: [`crate::policy::share::granted`]
//! answers from [`granted`].
//!
//! It fails closed: no table yet, a table withdrawn (every gateway link gone),
//! or a policy's rows withdrawn by its lease, is no grant.

use std::collections::{HashMap, HashSet};

use parking_lot::RwLock;

struct Table {
    version: u64,
    groups: HashMap<String, (HashSet<String>, u64)>,
    default: Option<(HashSet<String>, u64)>,
}

static CURRENT: RwLock<Option<Table>> = RwLock::new(None);

/// The id of the grant `tenant` holds for `group`, if any: the group's row,
/// else the column's default. `None` with no table, which is no grant. An id
/// changes with every change to its row, so KV keyed under one never
/// matches under the next. One read lock and two hash lookups: O(1).
pub fn granted(group: &str, tenant: &str) -> Option<u64> {
    let current = CURRENT.read();
    let table = current.as_ref()?;
    let (tenants, id) = table.groups.get(group).or(table.default.as_ref())?;
    tenants.contains(tenant).then_some(*id)
}

/// A group, its tenants and its grant's id.
pub type Grant = (String, Vec<String>, u64);

/// The table as of `version`: each group's grant, and the default's
/// (tenants, id). An older version than the one held is ignored (a
/// gateway's push that raced another's); version 0 is no table.
pub fn install(version: u64, groups: Vec<Grant>, default: Option<(Vec<String>, u64)>) {
    let mut current = CURRENT.write();
    if version == 0 || current.as_ref().is_some_and(|t| t.version >= version) {
        return;
    }
    let set = |tenants: Vec<String>| tenants.into_iter().collect::<HashSet<_>>();
    *current = Some(Table {
        version,
        groups: groups
            .into_iter()
            .map(|(group, tenants, id)| (group, (set(tenants), id)))
            .collect(),
        default: default.map(|(tenants, id)| (set(tenants), id)),
    });
    tracing::debug!(version, "share grants installed");
}

/// No gateway link stands: what the worker holds may be stale, so it holds
/// nothing. The next push installs again whatever its version.
pub fn withdraw() {
    if CURRENT.write().take().is_some() {
        tracing::info!("share grants withdrawn: no gateway link");
    }
}

#[cfg(test)]
pub(crate) static TEST_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    fn grants(version: u64, groups: &[(&str, &[&str], u64)], default: Option<(&[&str], u64)>) {
        let names = |t: &[&str]| t.iter().map(|t| t.to_string()).collect();
        install(
            version,
            groups.iter().map(|(g, t, id)| (g.to_string(), names(t), *id)).collect(),
            default.map(|(t, id)| (names(t), id)),
        );
    }

    #[test]
    fn a_grant_is_the_groups_row_else_the_default_and_fails_closed() {
        let _lock = TEST_LOCK.lock();
        withdraw();
        assert_eq!(granted("g", "acme"), None, "no table: no grant");
        grants(3, &[("g", &["acme", "beta"], 7)], None);
        assert_eq!(granted("g", "acme"), Some(7));
        assert_eq!(granted("g", "beta"), Some(7));
        assert_eq!(granted("g", "gamma"), None, "not named");
        assert_eq!(granted("h", "acme"), None, "no row, no default");
        grants(2, &[("g", &["gamma"], 5)], None);
        assert_eq!(granted("g", "gamma"), None, "an older version is ignored");
        grants(4, &[("g", &["acme"], 8)], Some((&["gamma"], 9)));
        assert_eq!(granted("g", "acme"), Some(8), "a re-grant is a new id");
        assert_eq!(granted("g", "beta"), None, "revoked");
        assert_eq!(granted("h", "gamma"), Some(9), "the default");
        assert_eq!(granted("g", "gamma"), None, "a row stands over the default");
        withdraw();
        assert_eq!(granted("g", "acme"), None, "withdrawn: fails closed");
        grants(1, &[("g", &["acme"], 1)], None);
        assert_eq!(granted("g", "acme"), Some(1), "after a withdrawal any version installs");
        withdraw();
    }
}
