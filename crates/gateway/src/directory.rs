//! Where the cluster's sealed state is, as the controller's directory last
//! told it ([`DirectoryView`]): which worker holds which run, where each
//! serves its pages, what each published. Read-only here: workers report to
//! the controller, and a pool's reports come through this gateway to it.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use controller_api::DirectoryView;
pub use controller_api::Key;
use ids::WorkerId;

/// A snapshot of the controller's directory.
#[derive(Clone, Default)]
pub struct Directory {
    view: Arc<DirectoryView>,
}

/// A run some worker keeps whole, and the workers (pools among them) that
/// hold every page of it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Group {
    pub keys: Arc<Vec<Key>>,
    pub sites: Vec<WorkerId>,
    /// The workers whose persistent disk tier keeps every page of it.
    pub disk_sites: Vec<WorkerId>,
    /// The most leading tokens of it some worker's owner asked kept on
    /// disk.
    pub persist_tokens: Option<u64>,
}

impl From<DirectoryView> for Directory {
    fn from(view: DirectoryView) -> Self {
        Self {
            view: Arc::new(view),
        }
    }
}

impl Directory {
    /// Every run some worker keeps, by its last key, with the workers that
    /// hold all of it.
    pub fn groups(&self) -> HashMap<Key, Group> {
        self.view
            .groups
            .iter()
            .map(|(last, group)| {
                (
                    *last,
                    Group {
                        keys: Arc::new(group.keys.clone()),
                        sites: group.sites.clone(),
                        disk_sites: group.disk_sites.clone(),
                        persist_tokens: group.persist_tokens,
                    },
                )
            })
            .collect()
    }

    pub fn pools(&self) -> Vec<WorkerId> {
        self.view.pools.clone()
    }

    /// What each worker's policy published about the worker itself.
    pub fn published_by_workers(&self) -> HashMap<WorkerId, HashMap<String, serde_json::Value>> {
        self.view
            .published
            .iter()
            .map(|(worker, own)| {
                let own = own
                    .iter()
                    .filter_map(|(name, json)| {
                        Some((name.clone(), serde_json::from_str(json).ok()?))
                    })
                    .collect();
                (*worker, own)
            })
            .collect()
    }

    /// Where `worker` serves its pages' bytes.
    pub fn peer(&self, worker: WorkerId) -> Option<String> {
        self.view
            .peers
            .iter()
            .find(|(w, _)| *w == worker)
            .map(|(_, peer)| peer.clone())
    }

    /// `worker`'s kv page in tokens, and whether its disk tier is kept.
    pub fn disk_tier(&self, worker: WorkerId) -> Option<(u32, bool)> {
        self.view
            .disk_tiers
            .iter()
            .find(|(w, _)| *w == worker)
            .map(|(_, tier)| *tier)
    }

    /// Whether `worker`'s persistent disk tier keeps every page of `keys`.
    pub fn on_disk(&self, worker: WorkerId, keys: &[Key]) -> bool {
        self.view
            .disk
            .iter()
            .find(|(w, _)| *w == worker)
            .is_some_and(|(_, disk)| {
                let disk: HashSet<&Key> = disk.iter().collect();
                keys.iter().all(|key| disk.contains(key))
            })
    }
}

#[cfg(test)]
mod tests {
    use controller_api::GroupView;

    use super::*;

    #[test]
    fn a_view_reads_as_the_directory_it_was() {
        let directory = Directory::from(DirectoryView {
            epoch: 3,
            groups: vec![(
                [2; 32],
                GroupView {
                    keys: vec![[1; 32], [2; 32]],
                    sites: vec![WorkerId(1)],
                    disk_sites: vec![WorkerId(1)],
                    persist_tokens: Some(40),
                },
            )],
            pools: vec![WorkerId(9)],
            peers: vec![(WorkerId(1), "a:1".into())],
            published: vec![(WorkerId(1), vec![("load".into(), "7".into())])],
            disk_tiers: vec![(WorkerId(1), (16, true))],
            disk: vec![(WorkerId(1), vec![[1; 32], [2; 32]])],
        });
        let groups = directory.groups();
        assert_eq!(*groups[&[2; 32]].keys, vec![[1; 32], [2; 32]]);
        assert_eq!(groups[&[2; 32]].persist_tokens, Some(40));
        assert_eq!(directory.pools(), vec![WorkerId(9)]);
        assert_eq!(directory.peer(WorkerId(1)).as_deref(), Some("a:1"));
        assert_eq!(directory.peer(WorkerId(2)), None);
        assert_eq!(
            directory.published_by_workers()[&WorkerId(1)]["load"],
            serde_json::json!(7)
        );
        assert_eq!(directory.disk_tier(WorkerId(1)), Some((16, true)));
        assert!(directory.on_disk(WorkerId(1), &[[1; 32]]));
        assert!(!directory.on_disk(WorkerId(2), &[[1; 32]]));
    }
}
