use crate::defaults;
use crate::rig::{Rig, whole};

/// A completed turn's KV is kept above every other until it is due, then
/// dropped; a session keeps its latest turn alone.
#[test]
fn a_session_s_latest_turn_is_kept_until_it_is_due() {
    let mut rig = Rig::new("keep-session");
    let chat = Some("t/chat");
    rig.at_worker("t1", "t", chat, &[("keep-ms", 1000)]);
    rig.host.started("t1");
    rig.at(0);
    rig.completed("t1", chat);
    rig.at(100);
    assert!(rig.priority.rows[&whole("t1")] > 0);
    assert!(rig.tiers.rows.is_empty(), "on any tier");
    assert_eq!(rig.alarm(), Some(1100));

    // The session's next turn takes its place: it is cache as any other.
    rig.at_worker("t2", "t", chat, &[("keep-ms", 1000)]);
    rig.host.started("t2");
    rig.at(500);
    rig.completed("t2", chat);
    rig.at(600);
    assert!(rig.priority.rows[&whole("t2")] > 0);
    assert_eq!(rig.priority.rows[&whole("t1")], -1);
    assert_eq!(rig.tiers.rows.get(&whole("t1")), Some(&1));
    assert_eq!(rig.alarm(), Some(1600));

    // Due: it may sit nowhere, which drops it.
    rig.at(1600);
    assert_eq!(rig.tiers.rows.get(&whole("t2")), Some(&0));
    assert!(!rig.priority.rows.contains_key(&whole("t2")));

    // One that failed is not kept.
    rig.at_worker("t3", "t", chat, &[("keep-ms", 1000)]);
    rig.host.started("t3");
    rig.at(1700);
    rig.host
        .put_working_set("t3", None, None, None, 300, 0, Vec::new());
    rig.host.exited("t3", "failed", None, None, 0);
    rig.host.retained("t3", "self");
    rig.at(1800);
    assert_eq!(rig.priority.rows[&whole("t3")], -1);
}

#[test]
fn the_worker_s_defaults_hold() {
    defaults::hold_on_a_worker("keep-session", &[], &[], "early");
}
