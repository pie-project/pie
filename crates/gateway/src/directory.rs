//! Where the cluster's sealed state is: which worker holds which page, by
//! key, as each reports it. Soft state: a worker's reconnect resends all it
//! holds, and a worker that goes takes its entries with it. Pages are
//! immutable, so an entry can only be stale by being gone, which a fetch
//! finds out.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use ids::WorkerId;

pub type Key = [u8; 32];

#[derive(Clone, Default)]
pub struct Directory {
    inner: Arc<Mutex<Inner>>,
}

#[derive(Default)]
struct Inner {
    held: HashMap<WorkerId, HashSet<Key>>,
    peers: HashMap<WorkerId, String>,
    /// Where each worker's peer link listens.
    links: HashMap<WorkerId, String>,
    /// Runs workers keep whole from token 0, by their last key: the groups
    /// a cluster policy places.
    runs: HashMap<WorkerId, HashMap<Key, Arc<Vec<Key>>>>,
    pools: HashSet<WorkerId>,
    /// What each worker's policy published, by subject and name.
    published: HashMap<WorkerId, HashMap<(String, String), serde_json::Value>>,
}

/// A run some worker keeps whole, and the workers (pools among them) that
/// hold every page of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Group {
    pub keys: Arc<Vec<Key>>,
    pub sites: Vec<WorkerId>,
}

impl Directory {
    pub fn report(
        &self,
        worker: WorkerId,
        peer: Option<String>,
        added: Vec<Key>,
        removed: Vec<Key>,
        full: bool,
    ) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(peer) = peer {
            inner.peers.insert(worker, peer);
        }
        let held = inner.held.entry(worker).or_default();
        if full {
            held.clear();
        }
        for key in removed {
            held.remove(&key);
        }
        held.extend(added);
    }

    /// The runs `worker` keeps, gained and lost since its last report (all
    /// of them when `full`), and whether it is a pool.
    pub fn report_runs(
        &self,
        worker: WorkerId,
        added: Vec<Vec<Key>>,
        removed: Vec<Key>,
        full: bool,
        pool: bool,
    ) {
        let mut inner = self.inner.lock().unwrap();
        if pool {
            inner.pools.insert(worker);
        }
        let runs = inner.runs.entry(worker).or_default();
        if full {
            runs.clear();
        }
        for last in removed {
            runs.remove(&last);
        }
        for run in added {
            if let Some(last) = run.last().copied() {
                runs.insert(last, Arc::new(run));
            }
        }
    }

    /// Everything `worker`'s policy has published, replacing what it had.
    pub fn report_published(&self, worker: WorkerId, published: Vec<(String, String, String)>) {
        let map = published
            .into_iter()
            .filter_map(|(of, name, json)| Some(((of, name), serde_json::from_str(&json).ok()?)))
            .collect();
        self.inner.lock().unwrap().published.insert(worker, map);
    }

    /// What `worker`'s policy published under `name` about `of` (`worker`,
    /// or `process:<id>`).
    pub fn published(&self, worker: WorkerId, of: &str, name: &str) -> Option<serde_json::Value> {
        self.inner
            .lock()
            .unwrap()
            .published
            .get(&worker)?
            .get(&(of.to_string(), name.to_string()))
            .cloned()
    }

    /// What each worker's policy published about the worker itself.
    pub fn published_by_workers(&self) -> HashMap<WorkerId, HashMap<String, serde_json::Value>> {
        self.inner
            .lock()
            .unwrap()
            .published
            .iter()
            .map(|(worker, map)| {
                let own = map
                    .iter()
                    .filter(|((of, _), _)| of == "worker")
                    .map(|((_, name), value)| (name.clone(), value.clone()))
                    .collect();
                (*worker, own)
            })
            .collect()
    }

    pub fn forget(&self, worker: WorkerId) {
        let mut inner = self.inner.lock().unwrap();
        inner.published.remove(&worker);
        inner.held.remove(&worker);
        inner.peers.remove(&worker);
        inner.links.remove(&worker);
        inner.runs.remove(&worker);
        inner.pools.remove(&worker);
    }

    pub fn pools(&self) -> Vec<WorkerId> {
        let mut pools: Vec<WorkerId> = self.inner.lock().unwrap().pools.iter().copied().collect();
        pools.sort_by_key(|w| w.0);
        pools
    }

    /// Every run some worker keeps, by its last key, with the workers that
    /// hold all of it.
    pub fn groups(&self) -> HashMap<Key, Group> {
        let inner = self.inner.lock().unwrap();
        let mut groups: HashMap<Key, Group> = HashMap::new();
        for runs in inner.runs.values() {
            for (last, keys) in runs {
                groups.entry(*last).or_insert_with(|| Group {
                    keys: keys.clone(),
                    sites: Vec::new(),
                });
            }
        }
        for group in groups.values_mut() {
            let mut sites: Vec<WorkerId> = inner
                .held
                .iter()
                .filter(|(_, held)| group.keys.iter().all(|key| held.contains(key)))
                .map(|(worker, _)| *worker)
                .collect();
            sites.sort_by_key(|w| w.0);
            group.sites = sites;
        }
        groups
    }

    /// Where `worker` serves its pages' bytes.
    pub fn peer(&self, worker: WorkerId) -> Option<String> {
        self.inner.lock().unwrap().peers.get(&worker).cloned()
    }

    pub fn report_link(&self, worker: WorkerId, link: String) {
        self.inner.lock().unwrap().links.insert(worker, link);
    }

    /// Where `worker`'s peer link listens.
    pub fn link(&self, worker: WorkerId) -> Option<String> {
        self.inner.lock().unwrap().links.get(&worker).cloned()
    }

    /// The workers other than `asker`, reachable by their peers, that hold a
    /// run of `keys` from the first: id, address and run length, longest
    /// first.
    pub fn holders(&self, asker: WorkerId, keys: &[Key]) -> Vec<(WorkerId, String, u32)> {
        let inner = self.inner.lock().unwrap();
        let mut holders: Vec<_> = inner
            .held
            .iter()
            .filter(|(worker, _)| **worker != asker)
            .filter_map(|(worker, held)| {
                let addr = inner.peers.get(worker)?;
                let run = keys.iter().take_while(|key| held.contains(*key)).count();
                (run > 0).then(|| (*worker, addr.clone(), run as u32))
            })
            .collect();
        holders.sort_by_key(|(worker, _, run)| (std::cmp::Reverse(*run), worker.0));
        holders
    }

    /// The first of [`Self::holders`].
    pub fn locate(&self, asker: WorkerId, keys: &[Key]) -> Option<(WorkerId, String, u32)> {
        self.holders(asker, keys).into_iter().next()
    }

    /// Pages of `keys`' run from the first each worker holds: what a router
    /// reads as `cached-tokens/<worker>` once it knows a request's keys.
    pub fn runs(&self, keys: &[Key]) -> Vec<(WorkerId, u32)> {
        let inner = self.inner.lock().unwrap();
        inner
            .held
            .iter()
            .map(|(worker, held)| {
                let run = keys.iter().take_while(|key| held.contains(*key)).count();
                (*worker, run as u32)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(n: u8) -> Key {
        [n; 32]
    }

    #[test]
    fn the_longest_run_from_the_first_key_wins_and_the_asker_is_never_named() {
        let directory = Directory::default();
        directory.report(
            WorkerId(1),
            Some("a:1".into()),
            vec![key(1), key(2)],
            vec![],
            true,
        );
        directory.report(
            WorkerId(2),
            Some("b:2".into()),
            vec![key(1), key(2), key(3)],
            vec![],
            true,
        );
        // Holds a later key but not the first: no run.
        directory.report(WorkerId(3), Some("c:3".into()), vec![key(3)], vec![], true);
        let keys = [key(1), key(2), key(3)];
        assert_eq!(
            directory.locate(WorkerId(9), &keys),
            Some((WorkerId(2), "b:2".into(), 3))
        );
        assert_eq!(
            directory.locate(WorkerId(2), &keys),
            Some((WorkerId(1), "a:1".into(), 2))
        );

        directory.report(WorkerId(2), None, vec![], vec![key(2)], false);
        assert_eq!(directory.locate(WorkerId(9), &keys).map(|h| h.2), Some(2));
        directory.forget(WorkerId(1));
        assert_eq!(
            directory.locate(WorkerId(9), &keys),
            Some((WorkerId(2), "b:2".into(), 1))
        );
        // A full report replaces what was said before.
        directory.report(WorkerId(2), None, vec![], vec![], true);
        assert_eq!(directory.locate(WorkerId(9), &keys), None);
    }

    #[test]
    fn a_group_is_a_kept_run_and_its_sites_hold_all_of_it() {
        let directory = Directory::default();
        let run = vec![key(1), key(2)];
        directory.report(WorkerId(1), Some("a:1".into()), run.clone(), vec![], true);
        directory.report_runs(WorkerId(1), vec![run.clone()], vec![], true, false);
        // A pool holding half of it is no site of it.
        directory.report(WorkerId(7), Some("p:7".into()), vec![key(1)], vec![], true);
        directory.report_runs(WorkerId(7), vec![], vec![], true, true);
        let groups = directory.groups();
        assert_eq!(groups[&key(2)].sites, vec![WorkerId(1)]);
        assert_eq!(directory.pools(), vec![WorkerId(7)]);
        directory.report(WorkerId(7), None, vec![key(2)], vec![], false);
        assert_eq!(
            directory.groups()[&key(2)].sites,
            vec![WorkerId(1), WorkerId(7)]
        );
        directory.report_runs(WorkerId(1), vec![], vec![key(2)], false, false);
        assert!(
            directory.groups().is_empty(),
            "no worker keeps the run any more"
        );
    }
}
