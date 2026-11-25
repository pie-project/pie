use crate::{defaults, rig::Rig};

/// The least served tenant runs first, and a tenant that arrives late
/// starts at the least an active one has had, not below it.
#[test]
fn the_least_served_tenant_runs_first_and_idle_time_is_no_credit() {
    let mut rig = Rig::new("fair");
    rig.at_worker("a1", "a", None, &[]);
    rig.host.started("a1");
    rig.at_worker("b1", "b", None, &[]);
    rig.at(0);
    assert_eq!(rig.alarm(), Some(250), "service is counted while a1 runs");

    rig.at(1000);
    assert_eq!(rig.keys.rows["a1"], -1_000_000_000);
    assert!(rig.keys.rows["b1"] > rig.keys.rows["a1"]);

    // Now b runs too: the least an active tenant has had is its second.
    rig.host.started("b1");
    rig.at(2000);
    rig.at_worker("c1", "c", None, &[]);
    rig.at(2000);
    assert_eq!(rig.keys.rows["b1"], -1_000_000_000);
    assert_eq!(rig.keys.rows["c1"], rig.keys.rows["b1"]);
    assert_eq!(rig.keys.rows["a1"], -2_000_000_000);

    // What ran is counted to its end.
    rig.host.set_now(3_000_000_000);
    rig.host.exited("a1", "completed", None, None, 0);
    rig.at_worker("a2", "a", None, &[]);
    rig.at(3000);
    assert_eq!(rig.keys.rows["a2"], -3_000_000_000);
}

/// The early one's tenant is the one that ran.
#[test]
fn the_worker_s_defaults_hold() {
    defaults::hold_on_a_worker("fair", &[], &[], "late");
}
