//! Where the cluster's sealed state is: which worker holds which page, by
//! key, as each reports it. Soft state: a worker's reconnect resends all it
//! holds, and a worker that goes takes its entries with it. Pages are
//! immutable, so an entry can only be stale by being gone, which a fetch
//! finds out. The gateways read it as a [`DirectoryView`].

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use controller_api::{DirectoryView, GroupView, Key, StateReport};
use ids::WorkerId;

#[derive(Clone, Default)]
pub struct Directory {
    inner: Arc<Mutex<Inner>>,
}

#[derive(Default)]
struct Inner {
    held: HashMap<WorkerId, HashSet<Key>>,
    /// Where each worker serves its pages' bytes (the bulk link).
    peers: HashMap<WorkerId, String>,
    /// Where each worker's peer link listens, which takes others' passes.
    /// Kept apart from `peers`: a pool serves bytes and no passes, and how
    /// far apart a worker puts the two is its own affair.
    links: HashMap<WorkerId, String>,
    /// Runs workers keep whole from token 0, by their last key: the groups
    /// a cluster policy places.
    runs: HashMap<WorkerId, HashMap<Key, Arc<Vec<Key>>>>,
    pools: HashSet<WorkerId>,
    /// What each worker's policy published, by subject and name.
    /// As JSON text.
    published: HashMap<WorkerId, HashMap<(String, String), String>>,
    /// The pages each worker's persistent disk tier keeps a copy of.
    disk: HashMap<WorkerId, HashSet<Key>>,
    /// The runs whose owner asked a prefix kept on disk, by last key: the
    /// tokens asked.
    hints: HashMap<WorkerId, HashMap<Key, u64>>,
    /// Each worker's kv page, in tokens, and whether its disk tier is kept.
    disk_tiers: HashMap<WorkerId, (u32, bool)>,
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

    /// The pages `worker`'s persistent disk tier gained and lost a copy of
    /// (all of them when `full`); the runs whose owner asked a prefix kept on
    /// disk, whole, when they changed; and its page size and whether its
    /// disk tier is kept, when it says.
    pub fn report_disk(
        &self,
        worker: WorkerId,
        added: Vec<Key>,
        removed: Vec<Key>,
        full: bool,
        hints: Option<Vec<(Key, u64)>>,
        tier: Option<(u32, bool)>,
    ) {
        let mut inner = self.inner.lock().unwrap();
        let disk = inner.disk.entry(worker).or_default();
        if full {
            disk.clear();
        }
        for key in removed {
            disk.remove(&key);
        }
        disk.extend(added);
        if let Some(hints) = hints {
            inner.hints.insert(worker, hints.into_iter().collect());
        }
        if let Some(tier) = tier {
            inner.disk_tiers.insert(worker, tier);
        }
    }

    /// Whether `worker`'s persistent disk tier keeps every page of `keys`.
    pub fn on_disk(&self, worker: WorkerId, keys: &[Key]) -> bool {
        self.inner
            .lock()
            .unwrap()
            .disk
            .get(&worker)
            .is_some_and(|disk| keys.iter().all(|key| disk.contains(key)))
    }

    /// `worker`'s kv page in tokens, and whether its disk tier is kept.
    pub fn disk_tier(&self, worker: WorkerId) -> Option<(u32, bool)> {
        self.inner.lock().unwrap().disk_tiers.get(&worker).copied()
    }

    /// Everything `worker`'s policy has published, replacing what it had.
    pub fn report_published(&self, worker: WorkerId, published: Vec<(String, String, String)>) {
        let map = published
            .into_iter()
            .map(|(of, name, json)| ((of, name), json))
            .collect();
        self.inner.lock().unwrap().published.insert(worker, map);
    }

    /// A worker's report, whole: the directory's one input.
    pub fn report_state(&self, worker: WorkerId, report: StateReport) {
        if let Some(link) = report.link {
            self.report_link(worker, link);
        }
        self.report(
            worker,
            report.peer,
            report.added,
            report.removed,
            report.full,
        );
        self.report_runs(
            worker,
            report.runs_added,
            report.runs_removed,
            report.full,
            report.pool,
        );
        if let Some(published) = report.published {
            self.report_published(worker, published);
        }
        self.report_disk(
            worker,
            report.disk_added,
            report.disk_removed,
            report.full,
            report.persist_hints,
            report.disk_tier,
        );
    }

    /// The directory as the gateways read it, numbered `epoch`.
    pub fn view(&self, epoch: u64) -> DirectoryView {
        let mut groups: Vec<(Key, GroupView)> = self
            .groups()
            .into_iter()
            .map(|(last, group)| {
                (
                    last,
                    GroupView {
                        keys: group.keys.as_ref().clone(),
                        sites: group.sites,
                        disk_sites: group.disk_sites,
                        persist_tokens: group.persist_tokens,
                    },
                )
            })
            .collect();
        groups.sort_unstable_by_key(|(last, _)| *last);
        let inner = self.inner.lock().unwrap();
        fn sorted<T>(mut v: Vec<(WorkerId, T)>) -> Vec<(WorkerId, T)> {
            v.sort_unstable_by_key(|(w, _)| w.0);
            v
        }
        DirectoryView {
            epoch,
            groups,
            pools: {
                let mut pools: Vec<WorkerId> = inner.pools.iter().copied().collect();
                pools.sort_unstable_by_key(|w| w.0);
                pools
            },
            peers: sorted(inner.peers.iter().map(|(w, p)| (*w, p.clone())).collect()),
            published: sorted(
                inner
                    .published
                    .iter()
                    .map(|(w, map)| {
                        let mut own: Vec<(String, String)> = map
                            .iter()
                            .filter(|((of, _), _)| of == "worker")
                            .map(|((_, name), json)| (name.clone(), json.clone()))
                            .collect();
                        own.sort_unstable();
                        (*w, own)
                    })
                    .collect(),
            ),
            disk_tiers: sorted(inner.disk_tiers.iter().map(|(w, t)| (*w, *t)).collect()),
            disk: sorted(
                inner
                    .disk
                    .iter()
                    .map(|(w, keys)| {
                        let mut keys: Vec<Key> = keys.iter().copied().collect();
                        keys.sort_unstable();
                        (*w, keys)
                    })
                    .collect(),
            ),
        }
    }

    pub fn forget(&self, worker: WorkerId) {
        let mut inner = self.inner.lock().unwrap();
        inner.published.remove(&worker);
        inner.held.remove(&worker);
        inner.peers.remove(&worker);
        inner.links.remove(&worker);
        inner.runs.remove(&worker);
        inner.pools.remove(&worker);
        inner.disk.remove(&worker);
        inner.hints.remove(&worker);
        inner.disk_tiers.remove(&worker);
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
                    ..Group::default()
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
            let mut disk_sites: Vec<WorkerId> = inner
                .disk
                .iter()
                .filter(|(_, disk)| group.keys.iter().all(|key| disk.contains(key)))
                .map(|(worker, _)| *worker)
                .collect();
            disk_sites.sort_by_key(|w| w.0);
            group.disk_sites = disk_sites;
        }
        for hints in inner.hints.values() {
            for (last, &tokens) in hints {
                if let Some(group) = groups.get_mut(last) {
                    group.persist_tokens = group.persist_tokens.max(Some(tokens));
                }
            }
        }
        groups
    }

    /// Where `worker` serves its pages' bytes.
    pub fn peer(&self, worker: WorkerId) -> Option<String> {
        self.inner.lock().unwrap().peers.get(&worker).cloned()
    }

    /// `worker`'s peer link listens at `link`.
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
            directory.holders(WorkerId(9), &keys).into_iter().next(),
            Some((WorkerId(2), "b:2".into(), 3))
        );
        assert_eq!(
            directory.holders(WorkerId(2), &keys).into_iter().next(),
            Some((WorkerId(1), "a:1".into(), 2))
        );

        directory.report(WorkerId(2), None, vec![], vec![key(2)], false);
        assert_eq!(
            directory
                .holders(WorkerId(9), &keys)
                .into_iter()
                .next()
                .map(|h| h.2),
            Some(2)
        );
        directory.forget(WorkerId(1));
        assert_eq!(
            directory.holders(WorkerId(9), &keys).into_iter().next(),
            Some((WorkerId(2), "b:2".into(), 1))
        );
        // A full report replaces what was said before.
        directory.report(WorkerId(2), None, vec![], vec![], true);
        assert_eq!(
            directory.holders(WorkerId(9), &keys).into_iter().next(),
            None
        );
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

    #[test]
    fn a_pool_s_runs_are_groups_after_every_worker_lets_them_go_and_after_it_restarts() {
        let directory = Directory::default();
        let run = vec![key(1), key(2)];
        directory.report(WorkerId(1), Some("a:1".into()), run.clone(), vec![], true);
        directory.report_runs(WorkerId(1), vec![run.clone()], vec![], true, false);
        directory.report(WorkerId(7), Some("p:7".into()), run.clone(), vec![], true);
        directory.report_runs(WorkerId(7), vec![run.clone()], vec![], true, true);
        assert_eq!(
            directory.groups()[&key(2)].sites,
            vec![WorkerId(1), WorkerId(7)]
        );
        // The worker lets it go: the pool's copy is the group.
        directory.report(WorkerId(1), None, vec![], run.clone(), false);
        directory.report_runs(WorkerId(1), vec![], vec![key(2)], false, false);
        assert_eq!(directory.groups()[&key(2)].sites, vec![WorkerId(7)]);
        // The pool goes, and comes back whole under a new id.
        directory.forget(WorkerId(7));
        assert!(directory.groups().is_empty());
        directory.report(WorkerId(8), Some("p:7".into()), run.clone(), vec![], true);
        directory.report_runs(WorkerId(8), vec![run.clone()], vec![], true, true);
        let groups = directory.groups();
        assert_eq!(groups[&key(2)].sites, vec![WorkerId(8)]);
        assert_eq!(*groups[&key(2)].keys, run);
        assert_eq!(directory.pools(), vec![WorkerId(8)]);
    }

    #[test]
    fn a_group_on_a_worker_s_disk_tier_reads_as_a_disk_site_with_the_tokens_asked() {
        let directory = Directory::default();
        let run = vec![key(1), key(2), key(3)];
        directory.report(WorkerId(1), Some("a:1".into()), run.clone(), vec![], true);
        directory.report_runs(
            WorkerId(1),
            vec![run.clone(), run[..2].to_vec()],
            vec![],
            true,
            false,
        );
        directory.report_disk(
            WorkerId(1),
            run[..2].to_vec(),
            vec![],
            true,
            Some(vec![(key(3), 40), (key(2), 40)]),
            Some((16, true)),
        );
        let groups = directory.groups();
        assert_eq!(
            groups[&key(2)].disk_sites,
            vec![WorkerId(1)],
            "the prefix is on disk"
        );
        assert!(
            groups[&key(3)].disk_sites.is_empty(),
            "the whole run is not"
        );
        assert_eq!(groups[&key(3)].persist_tokens, Some(40));
        assert!(directory.on_disk(WorkerId(1), &run[..2]));
        assert_eq!(directory.disk_tier(WorkerId(1)), Some((16, true)));

        directory.report_disk(WorkerId(1), vec![], vec![key(1)], false, None, None);
        assert!(directory.groups()[&key(2)].disk_sites.is_empty());
        assert_eq!(
            directory.groups()[&key(3)].persist_tokens,
            Some(40),
            "hints stand until the worker sends others"
        );
        directory.forget(WorkerId(1));
        assert_eq!(directory.disk_tier(WorkerId(1)), None);
    }

    #[test]
    fn the_view_is_what_the_reports_left_and_a_quiet_report_changes_none_of_it() {
        let directory = Directory::default();
        let run = vec![key(1), key(2)];
        directory.report_state(
            WorkerId(3),
            StateReport {
                peer: Some("w:3".into()),
                added: run.clone(),
                full: true,
                runs_added: vec![run.clone()],
                published: Some(vec![
                    ("worker".into(), "load".into(), "7".into()),
                    ("process:1".into(), "x".into(), "1".into()),
                ]),
                disk_added: vec![key(1)],
                disk_tier: Some((16, true)),
                ..StateReport::default()
            },
        );
        let view = directory.view(5);
        assert_eq!(view.epoch, 5);
        assert_eq!(view.groups.len(), 1);
        assert_eq!(view.groups[0].0, key(2));
        assert_eq!(view.groups[0].1.sites, vec![WorkerId(3)]);
        assert!(view.groups[0].1.disk_sites.is_empty(), "half of it on disk");
        assert_eq!(view.peers, vec![(WorkerId(3), "w:3".to_string())]);
        assert_eq!(
            view.published,
            vec![(WorkerId(3), vec![("load".to_string(), "7".to_string())])],
            "what a worker published about itself"
        );
        assert_eq!(view.disk_tiers, vec![(WorkerId(3), (16, true))]);
        assert_eq!(view.disk, vec![(WorkerId(3), vec![key(1)])]);

        directory.report_state(WorkerId(3), StateReport::default());
        assert_eq!(directory.view(5), view, "a quiet report");
        directory.forget(WorkerId(3));
        assert!(directory.view(6).groups.is_empty());
    }
}
