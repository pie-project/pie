//! Retains every run some worker caches, at key 1: the gateway keeps a copy
//! of each in a pool, so the last worker letting it go does not lose it.

use policy::{SubjectKind, candidates, cluster};

#[derive(Default)]
struct RetainAll;

impl cluster::Policy for RetainAll {
    fn plan(&mut self, _now_ms: u64) {
        let entries: Vec<(String, i64)> = candidates(SubjectKind::Group)
            .into_iter()
            .map(|group| (group, 1))
            .collect();
        cluster::install_retention(&entries);
    }
}

policy::cluster!(RetainAll);
