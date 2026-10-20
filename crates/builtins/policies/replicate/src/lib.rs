//! Keeps every run some worker caches on two serving workers: the two a
//! rendezvous hash of the run picks among the decode workers not draining,
//! so a run's copies stay where they are as workers come and go, and losing
//! a worker leaves each of its runs on another.

use std::hash::{DefaultHasher, Hash, Hasher};

use policy::{Replicate, Site, Subject, SubjectKind, Tier, attr, candidates, cluster};

/// Workers each run is kept on, besides wherever it was computed.
const COPIES: usize = 2;

#[derive(Default)]
struct KeepTwo;

fn score(group: &str, worker: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    (group, worker).hash(&mut hasher);
    hasher.finish()
}

impl cluster::Policy for KeepTwo {
    fn plan(&mut self, _now_ms: u64) {
        let workers: Vec<String> = candidates(SubjectKind::Worker)
            .into_iter()
            .filter(|worker| {
                let of = Subject::Worker(worker.clone());
                attr(&of, "role").text() == Some("decode")
                    && attr(&of, "draining").boolean() != Some(true)
            })
            .collect();
        if workers.len() < 2 {
            return;
        }
        let entries: Vec<Replicate> = candidates(SubjectKind::Group)
            .into_iter()
            .map(|group| {
                let mut ranked: Vec<&String> = workers.iter().collect();
                ranked.sort_by_key(|worker| std::cmp::Reverse(score(&group, worker)));
                Replicate {
                    at: ranked
                        .into_iter()
                        .take(COPIES)
                        .map(|worker| Site {
                            worker: worker.clone(),
                            tier: Tier::Host,
                        })
                        .collect(),
                    group,
                    upto: None,
                }
            })
            .collect();
        let _ = cluster::install_replication(&entries);
    }
}

policy::cluster!(KeepTwo);
