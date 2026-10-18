//! Keeps every run some worker caches in every pool.

use policy::{Replicate, Site, Subject, SubjectKind, Tier, attr, candidates, cluster};

#[derive(Default)]
struct PoolReplicate;

impl cluster::Policy for PoolReplicate {
    fn plan(&mut self, _now_ms: u64) {
        let pools: Vec<String> = candidates(SubjectKind::Worker)
            .into_iter()
            .filter(|worker| attr(&Subject::Worker(worker.clone()), "pool").boolean() == Some(true))
            .collect();
        if pools.is_empty() {
            return;
        }
        let entries: Vec<Replicate> = candidates(SubjectKind::Group)
            .into_iter()
            .map(|group| Replicate {
                group,
                upto: None,
                at: pools
                    .iter()
                    .map(|worker| Site {
                        worker: worker.clone(),
                        tier: Tier::Pool,
                    })
                    .collect(),
            })
            .collect();
        let _ = cluster::install_replication(&entries);
    }
}

policy::cluster!(PoolReplicate);
