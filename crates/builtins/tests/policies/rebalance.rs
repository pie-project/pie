use crate::defaults;
use crate::rig::Rig;

/// While one worker runs two more turns than another its youngest moves
/// there, one a call and each at most once; one refused stays; a launch
/// starts where it names.
#[test]
fn the_busiest_s_youngest_turn_moves_to_the_idlest() {
    let mut rig = Rig::new("rebalance");
    rig.decode("1", false);
    rig.decode("2", false);
    for id in ["0.1", "0.2", "0.3", "0.4"] {
        rig.turn(id, "1");
        rig.at(id[2..].parse().unwrap());
        if id == "0.1" {
            assert!(rig.placed.rows.is_empty(), "one more is even");
        }
    }
    // The second made a gap of two, and the youngest then was it; the
    // fourth made the next.
    assert_eq!(rig.row("0.2"), Some(vec!["2"]));
    assert_eq!(rig.row("0.4"), Some(vec!["2"]));
    assert_eq!(rig.placed.rows.len(), 2, "one a call, and then even");

    rig.refuse("0.4");
    rig.host.dispatched("0.2", "2");
    rig.launch("0.5", None, &[("worker", 1)]);
    rig.at(10);
    assert_eq!(rig.row("0.2"), None, "it moved");
    assert_eq!(rig.row("0.4"), None, "refused, it stays");
    assert_eq!(rig.row("0.5"), Some(vec!["1", "2"]));
    assert_eq!(
        rig.row("0.3"),
        Some(vec!["2"]),
        "three to one: the next youngest"
    );
    rig.refuse("0.3");
    rig.at(20);
    rig.at(30);
    for id in ["0.2", "0.3", "0.4"] {
        assert_eq!(rig.row(id), None, "{id} is asked once");
    }
}

#[test]
fn the_cluster_s_defaults_hold() {
    defaults::hold_in_the_cluster("rebalance");
}
