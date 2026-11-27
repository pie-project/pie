use crate::{defaults, rig::Rig};

/// The soonest due runs first, and one without a deadline after both.
#[test]
fn the_soonest_due_runs_first_and_the_undated_last() {
    let mut rig = Rig::new("deadline");
    rig.at_worker("late", "t", None, &[("deadline-ms", 5000)]);
    rig.at_worker("undated", "t", None, &[]);
    rig.at(0);
    rig.host.set_now(1_000_000_000);
    rig.at_worker("soon", "t", None, &[("deadline-ms", 500)]);
    rig.at(1000);
    assert!(rig.keys.rows["soon"] > rig.keys.rows["late"]);
    assert!(rig.keys.rows["late"] > 0, "above the undated's 0");
    assert!(!rig.keys.rows.contains_key("undated"));
}

#[test]
fn the_worker_s_defaults_hold() {
    let (early, late) = ([("deadline-ms", 9000)], [("deadline-ms", 10)]);
    defaults::hold_on_a_worker("deadline", &early, &late, "late");
}
