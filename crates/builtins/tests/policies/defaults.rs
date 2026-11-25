//! What the planes' defaults do, asked of every policy built on them.

use plex_host::runtime::wit::Tier;
use plex_host::tables::{
    COLUMNS, PLACEMENT_HELPERS, PLACEMENT_WORKERS, REPLICATION_WORKERS, RETENTION_PRIORITY,
    RETENTION_TIERS, SCHEDULE_HOLD, SCHEDULE_PRIORITY, Value,
};

use crate::rig::{Rig, process, whole, worker};

/// A plane loads a policy that writes its own columns alone.
fn writes_only(rig: &Rig, plane: &[&str]) {
    for import in &rig.imports {
        let column = COLUMNS.contains(&import.as_str());
        assert!(!column || plane.contains(&import.as_str()), "{import}");
    }
}

/// Under a worker `policy`: past two lanes to the end a start waits, and a
/// lane that ends lets the one of `late`'s hints and tenant start before
/// the older `early`'s (`first` says which goes first); what an exited
/// process left is on the device alone, its tail before its prompt.
pub fn hold_on_a_worker(policy: &str, early: &[(&str, i64)], late: &[(&str, i64)], first: &str) {
    let mut rig = Rig::new(policy);
    writes_only(
        &rig,
        &[
            SCHEDULE_PRIORITY,
            SCHEDULE_HOLD,
            RETENTION_PRIORITY,
            RETENTION_TIERS,
        ],
    );
    rig.host
        .put_worker("self", false, vec![(Tier::Device, None, 0)], 0, 0, None);
    rig.attr(worker("self"), "pie/lanes-to-end", Value::Int(2));
    for id in ["a", "b"] {
        rig.at_worker(id, "busy", None, &[]);
        rig.host.started(id);
    }
    rig.at(0);
    assert!(rig.holds.rows.is_empty() && rig.holds.default.is_none());
    rig.at_worker("early", "busy", None, early);
    rig.at(10);
    rig.at_worker("late", "idle", None, late);
    rig.at(20);
    assert_eq!(rig.holds.default, Some(true), "a start waits");
    assert_eq!(rig.holds.rows.get("a"), Some(&false), "the running run on");
    for id in ["early", "late"] {
        assert!(!rig.holds.rows.contains_key(id), "{id} waits");
    }

    rig.completed("a", None);
    rig.at(1000);
    let other = if first == "late" { "early" } else { "late" };
    assert_eq!(rig.holds.rows.get(first), Some(&false), "{first} is let go");
    assert!(!rig.holds.rows.contains_key(other), "{other} waits on");
    assert_eq!(
        rig.tiers.rows.get(&whole("a")),
        Some(&1),
        "the device alone"
    );
    assert_eq!(rig.priority.rows.get(&whole("a")), Some(&-1), "its tail");
    let prompt = ("a".to_string(), Some(100));
    assert_eq!(rig.priority.rows.get(&prompt), Some(&0), "its prompt");
}

/// Under a cluster `policy`: every pool keeps copies and every worker that
/// serves others' passes helps; a session's turn goes first where its
/// others are; a tied launch goes to its worker, whatever it names; with
/// every worker full a launch is held, and placed once there is room.
pub fn hold_in_the_cluster(policy: &str) {
    let mut rig = Rig::new(policy);
    writes_only(
        &rig,
        &[PLACEMENT_WORKERS, PLACEMENT_HELPERS, REPLICATION_WORKERS],
    );
    rig.decode("1", false);
    rig.decode("2", false);
    rig.host
        .put_worker("3", false, vec![(Tier::Device, None, 0)], 0, 0, None);
    rig.attr(worker("3"), "pie/role", Value::Text("prefill".into()));
    rig.attr(worker("3"), "pie/prefill-listen", Value::Text("at".into()));
    rig.host
        .put_worker("9", false, vec![(Tier::Host, None, 0)], 0, 0, None);
    rig.at(0);
    assert_eq!(rig.copies.default, Some(vec!["9".into()]));
    assert_eq!(rig.helpers.default, Some(vec!["3".into()]));

    for (at, other) in [("1", "2"), ("2", "1")] {
        let session = format!("t/{at}");
        let first = format!("0.1{at}");
        rig.launch(&first, Some(&session), &[]);
        rig.host.dispatched(&first, at);
        rig.host.started(&first);
        let next = format!("0.2{at}");
        rig.launch(&next, Some(&session), &[]);
        rig.at(10);
        assert_eq!(rig.row(&next), Some(vec![at, other]), "where its other is");
        assert_eq!(rig.row(&first), None, "a running turn stays");
        rig.host.exited(&next, "completed", None, None, 0);
    }

    rig.attr(process("0.3"), "pie/tied", Value::Text("2".into()));
    rig.launch("0.3", None, &[("worker", 1)]);
    rig.at(20);
    assert_eq!(rig.row("0.3"), Some(vec!["2"]), "tied");

    for id in ["1", "2"] {
        rig.attr(worker(id), "pie/kv-pressure", Value::Int(255));
    }
    rig.launch("0.4", None, &[("worker", 1)]);
    rig.at(30);
    assert_eq!(rig.row("0.4"), Some(vec![]), "held");
    rig.attr(worker("2"), "pie/kv-pressure", Value::Int(0));
    rig.at(1030);
    let row = rig.row("0.4").expect("a row");
    assert!(row.contains(&"2"), "placed once there is room: {row:?}");
}
