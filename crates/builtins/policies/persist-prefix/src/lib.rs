//! Keeps on disk the prefix a process asked for. The worker that kept the
//! process's run tells the cluster how many leading tokens were asked (group
//! attr `persist-tokens`); each worker holding the run keeps that prefix on
//! its persistent disk tier, cut at its own page. A worker without one is
//! passed over by the host.

use policy::{Replicate, Site, Subject, SubjectKind, Tier, attr, candidates, capability, cluster};

#[derive(Default)]
struct PersistPrefix;

/// The ids in a JSON array of strings, as attr `sites` spells them.
fn ids(json: &str) -> Vec<String> {
    json.trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .map(|id| id.trim().trim_matches('"').to_string())
        .filter(|id| !id.is_empty())
        .collect()
}

impl cluster::Policy for PersistPrefix {
    fn plan(&mut self, _now_ms: u64) {
        // No worker keeps a disk tier across runs: nothing to ask.
        if !capability("kv.persist") {
            cluster::install_replication(&[]);
            return;
        }
        let entries: Vec<Replicate> = candidates(SubjectKind::Group)
            .into_iter()
            .filter_map(|group| {
                let of = Subject::Group(group.clone());
                let tokens = u32::try_from(attr(&of, "persist-tokens").int()?).ok()?;
                let at: Vec<Site> = ids(attr(&of, "sites").json()?)
                    .into_iter()
                    .map(|worker| Site {
                        worker,
                        tier: Tier::Disk,
                    })
                    .collect();
                (!at.is_empty()).then_some(Replicate {
                    group,
                    upto: Some(tokens),
                    at,
                })
            })
            .collect();
        cluster::install_replication(&entries);
    }
}

policy::cluster!(PersistPrefix);
