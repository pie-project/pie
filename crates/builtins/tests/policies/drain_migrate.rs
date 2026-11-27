use plex_host::tables::Value;

use crate::defaults;
use crate::rig::{Rig, process};

/// The turns on a draining worker move to the others in turn; one refused
/// or tied stays, and so does a turn that moved; a launch starts where it
/// names.
#[test]
fn a_drain_moves_its_turns_and_a_refused_one_stays() {
    let mut rig = Rig::new("drain-migrate");
    for id in ["1", "2", "3"] {
        rig.decode(id, false);
    }
    rig.attr(process("0.4"), "pie/tied", Value::Text("1".into()));
    for id in ["0.1", "0.2", "0.3", "0.4"] {
        rig.turn(id, "1");
    }
    rig.launch("0.5", None, &[("worker", 3)]);
    rig.at(0);
    assert_eq!(rig.row("0.5").unwrap()[0], "3");
    assert_eq!(rig.placed.rows.len(), 1, "no drain, no move");

    rig.decode("1", true);
    rig.at(10);
    assert_eq!(rig.row("0.1"), Some(vec!["2", "3"]));
    assert_eq!(rig.row("0.2"), Some(vec!["3", "2"]), "spread");
    assert_eq!(rig.row("0.3"), Some(vec!["2", "3"]));
    assert_eq!(rig.row("0.4"), None, "tied");

    rig.refuse("0.1");
    rig.host.dispatched("0.2", "3");
    rig.at(20);
    assert_eq!(rig.row("0.1"), None, "refused, it is not asked again");
    assert_eq!(rig.row("0.2"), None, "it moved");
    assert_eq!(rig.row("0.3"), Some(vec!["2", "3"]), "its target stands");
}

#[test]
fn the_cluster_s_defaults_hold() {
    defaults::hold_in_the_cluster("drain-migrate");
}
