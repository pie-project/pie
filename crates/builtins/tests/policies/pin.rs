use crate::{defaults, rig::Rig};

/// A launch goes first to the worker it names, then to the rest; one that
/// is gone or draining is named for nothing.
#[test]
fn a_launch_goes_first_to_the_worker_it_names() {
    let mut rig = Rig::new("pin");
    for id in ["1", "2", "3"] {
        rig.decode(id, id == "3");
    }
    rig.launch("0.1", None, &[("worker", 2)]);
    rig.launch("0.2", None, &[("worker", 3)]);
    rig.launch("0.3", None, &[("worker", 7)]);
    rig.launch("0.4", Some("t/s"), &[("worker", 1)]);
    let text = plex_host::tables::Value::Text("2".into());
    rig.launch("0.5", None, &[]);
    rig.host
        .hint(crate::rig::process("0.5"), "worker", Some(text));
    rig.at(0);
    assert_eq!(rig.row("0.1"), Some(vec!["2", "1"]));
    for id in ["0.2", "0.3"] {
        assert_eq!(rig.row(id).unwrap().len(), 2, "{id}: as any other");
    }
    assert_eq!(rig.row("0.4"), Some(vec!["1", "2"]), "before its session's");
    assert_eq!(rig.row("0.5"), Some(vec!["2", "1"]), "named in text");
}

#[test]
fn the_cluster_s_defaults_hold() {
    defaults::hold_in_the_cluster("pin");
}
