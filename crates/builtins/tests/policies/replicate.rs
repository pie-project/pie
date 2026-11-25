use plex_host::runtime::wit::Tier;

use crate::defaults;
use crate::rig::{Rig, whole};

fn cached(rig: &Rig, set: &str) {
    let copies = vec![("1".into(), Tier::Device, 64)];
    rig.host
        .put_working_set(set, None, None, None, 64, 0, copies);
}

/// Each run cached is wanted on two decode workers and in every pool, its
/// two whatever joins, and on another where one of them drains; a run
/// cached since is seen at the next second's look.
#[test]
fn each_cached_run_is_kept_on_two_workers_and_every_pool() {
    let mut rig = Rig::new("replicate");
    for id in ["1", "2", "3"] {
        rig.decode(id, false);
    }
    rig.host
        .put_worker("9", false, vec![(Tier::Host, None, 0)], 0, 0, None);
    cached(&rig, "g");
    rig.at(0);
    let row = rig.copies.rows[&whole("g")].clone();
    assert_eq!((row.len(), row[2].as_str()), (3, "9"));
    assert_ne!(row[0], row[1]);

    cached(&rig, "h");
    rig.at(500);
    assert!(
        !rig.copies.rows.contains_key(&whole("h")),
        "no event says so"
    );
    assert_eq!(rig.alarm(), Some(1000));
    rig.at(1000);
    assert_eq!(rig.copies.rows[&whole("h")].len(), 3);
    assert_eq!(rig.alarm(), Some(2000), "it looks again");

    rig.decode(&row[0], true);
    rig.at(1100);
    let next = &rig.copies.rows[&whole("g")];
    assert_eq!(next[0], row[1], "the other stays");
    assert!(next.len() == 3 && !next.contains(&row[0]));

    let one = ["1", "2", "3"]
        .into_iter()
        .find(|id| !next.contains(&id.to_string()));
    assert_eq!(one, Some(row[0].as_str()));
    rig.decode(&row[1], true);
    rig.at(1200);
    assert!(
        rig.copies.rows.is_empty(),
        "one decode worker: the pools alone"
    );
    assert_eq!(rig.copies.default, Some(vec!["9".into()]));
}

#[test]
fn the_cluster_s_defaults_hold() {
    defaults::hold_in_the_cluster("replicate");
}
