use crate::{defaults, rig::Rig};

/// Each process is keyed by the priority its client states, as it changes;
/// one without is not named.
#[test]
fn processes_are_keyed_by_what_their_client_states() {
    let mut rig = Rig::new("priority");
    rig.at_worker("a", "t", None, &[("priority", 5)]);
    rig.at_worker("b", "t", None, &[]);
    rig.at_worker("c", "t", None, &[("priority", -3)]);
    rig.at(0);
    assert_eq!(rig.keys.rows.get("a"), Some(&5));
    assert_eq!(rig.keys.rows.get("c"), Some(&-3));
    assert_eq!(rig.keys.rows.len(), 2, "only ranked processes are named");
    assert_eq!(rig.keys.default, None);

    let seven = plex_host::tables::Value::Int(7);
    rig.host
        .hint(crate::rig::process("b"), "priority", Some(seven));
    rig.at(1);
    assert_eq!(rig.keys.rows.get("b"), Some(&7));
}

#[test]
fn the_worker_s_defaults_hold() {
    let late = [("priority", 1)];
    defaults::hold_on_a_worker("priority", &[], &late, "late");
}
