use super::*;

use std::sync::Arc;

use tokio::sync::Notify;

use crate::store::rs::RsUnits;

use super::inner::Inner;
use super::proc::{Proc, Residency};
use super::wait::{EntryKey, WaitKind, Waiter};

use crate::planner::grant::Demand;

fn fleet(spec: &[(u64, Residency, bool)]) -> (Inner, Vec<ProcessId>) {
    let mut inner = Inner::default();
    let mut pids = Vec::new();
    for &(seq, state, admitted) in spec {
        let pid = ProcessId::new_v4();
        let mut proc = Proc::new(seq);
        proc.state = state;
        proc.admitted = admitted;
        inner.procs.insert(pid, proc);
        pids.push(pid);
    }
    (inner, pids)
}

fn park_allocation(inner: &mut Inner, pid: ProcessId, kv_pages: u32) -> EntryKey {
    inner.enqueue(
        pid,
        WaitKind::Allocation {
            demand: Demand {
                kv_pages,
                rs: RsUnits::default(),
            },
            notify: Arc::new(Notify::new()),
            outcome: None,
            yielded: false,
        },
    )
}

fn park_restore(inner: &mut Inner, pid: ProcessId, kv_pages: u32) -> EntryKey {
    inner.enqueue(
        pid,
        WaitKind::Restore {
            demand: Demand {
                kv_pages,
                rs: RsUnits::default(),
            },
        },
    )
}

#[test]
fn planner_every_case() {
    a_restore_yields_the_head_to_a_younger_allocation();
    a_restore_takes_the_head_once_the_fleet_has_stalled();
    a_restore_holds_the_head_when_no_allocation_is_unmet();
    a_younger_restore_still_queues_behind_an_older_allocation();
    returning_host_room_resumes_the_yield();
}

fn a_restore_yields_the_head_to_a_younger_allocation() {
    let (mut inner, pids) = fleet(&[
        (1, Residency::Evicted, true),
        (2, Residency::Resident, true),
        (3, Residency::Resident, true),
    ]);
    park_restore(&mut inner, pids[0], 18);
    park_allocation(&mut inner, pids[2], 1);

    let (key, waiter) = inner.unmet_head().expect("a head");
    assert_eq!(key.0.seq, 3, "the younger allocation must take the head");
    assert!(matches!(waiter.kind, WaitKind::Allocation { .. }));
}

fn a_restore_takes_the_head_once_the_fleet_has_stalled() {
    let (mut inner, pids) = fleet(&[
        (1, Residency::Evicted, true),
        (3, Residency::Resident, true),
    ]);
    park_restore(&mut inner, pids[0], 18);
    park_allocation(&mut inner, pids[1], 1);

    assert!(inner.fleet_stalled(), "every admitted resident is parked");
    let (key, waiter) = inner.unmet_head().expect("a head");
    assert_eq!(key.0.seq, 1, "the older restore must take the head");
    assert!(matches!(waiter.kind, WaitKind::Restore { .. }));
}

fn a_restore_holds_the_head_when_no_allocation_is_unmet() {
    let (mut inner, pids) = fleet(&[
        (1, Residency::Evicted, true),
        (3, Residency::Resident, true),
    ]);
    park_restore(&mut inner, pids[0], 18);
    let parked = park_allocation(&mut inner, pids[1], 1);
    let Some(Waiter {
        kind: WaitKind::Allocation { outcome, .. },
        ..
    }) = inner.queue.get_mut(&parked)
    else {
        unreachable!("just parked an allocation");
    };
    *outcome = Some(Err(PlannerError::Cancelled));

    assert!(!inner.fleet_stalled(), "a served grant is uncollected");
    let (key, waiter) = inner.unmet_head().expect("a head");
    assert_eq!(key.0.seq, 1, "the restore is the only unmet entry");
    assert!(matches!(waiter.kind, WaitKind::Restore { .. }));
}

fn a_younger_restore_still_queues_behind_an_older_allocation() {
    let (mut inner, pids) = fleet(&[
        (1, Residency::Resident, true),
        (2, Residency::Resident, true),
        (3, Residency::Evicted, true),
    ]);
    park_allocation(&mut inner, pids[0], 4);
    park_restore(&mut inner, pids[2], 18);

    let (key, waiter) = inner.unmet_head().expect("a head");
    assert_eq!(key.0.seq, 1);
    assert!(matches!(waiter.kind, WaitKind::Allocation { .. }));
}

fn returning_host_room_resumes_the_yield() {
    let (mut inner, pids) = fleet(&[
        (1, Residency::Evicted, true),
        (2, Residency::Resident, true),
        (3, Residency::Resident, true),
    ]);
    park_restore(&mut inner, pids[0], 18);
    park_allocation(&mut inner, pids[2], 1);
    inner.host_swap_blocked.insert(pids[1]);
    assert_eq!(inner.unmet_head().expect("a head").0.0.seq, 1);

    inner.host_swap_blocked.clear();

    assert!(!inner.eviction_unfundable());
    assert_eq!(
        inner.unmet_head().expect("a head").0.0.seq,
        3,
        "the allocation takes the head again"
    );
}

#[test]
fn a_key_outranks_age_in_the_queue_and_among_victims() {
    let (mut inner, pids) = fleet(&[
        (1, Residency::Resident, true),
        (2, Residency::Resident, true),
    ]);
    let (old, young) = (pids[0], pids[1]);
    let old_key = park_allocation(&mut inner, old, 4);
    let young_key = park_allocation(&mut inner, young, 4);
    assert_eq!(inner.unmet_head().unwrap().1.pid, old, "no schedule: age");

    inner.reschedule(|pid| (if *pid == young { 5 } else { 0 }, false));
    let (head, waiter) = inner.unmet_head().unwrap();
    assert_eq!(waiter.pid, young, "the higher key takes the head");
    assert!(
        inner.procs[&old].rank() > head.0,
        "the older, lower-keyed process may now be preempted for it"
    );
    assert!(inner.procs[&young].rank() <= head.0);
    for handed_out in [old_key, young_key] {
        assert_eq!(
            inner.locate(handed_out).map(|(_, id)| id),
            Some(handed_out.1),
            "a key handed out before the re-sort still finds its entry"
        );
    }

    inner.reschedule(|_| (0, false));
    assert_eq!(
        inner.unmet_head().unwrap().1.pid,
        old,
        "an empty schedule is age again"
    );
}

#[test]
fn the_floor_parks_a_restore_and_lifting_it_queues_the_restore_again() {
    let (mut inner, pids) = fleet(&[
        (1, Residency::Evicted, true),
        (2, Residency::Resident, true),
    ]);
    let (paused, running) = (pids[0], pids[1]);
    park_restore(&mut inner, paused, 3);
    let running_key = park_allocation(&mut inner, running, 2);

    inner.reschedule(|pid| (0, *pid == paused || *pid == running));
    assert!(
        inner.queue.values().all(|w| w.pid != paused),
        "a held process is not restored"
    );
    assert_eq!(inner.procs[&paused].parked.map(|d| d.kv_pages), Some(3));
    assert!(
        inner.preempted_floor().is_none(),
        "nor does it hold others back"
    );
    let head = inner.unmet_head().unwrap().0;
    assert_eq!(head, running_key);
    assert!(inner.procs[&running].held, "held, so a victim at any rank");

    inner.reschedule(|_| (0, false));
    assert!(inner.procs[&paused].parked.is_none());
    assert!(
        inner
            .queue
            .values()
            .any(|w| w.pid == paused && matches!(w.kind, WaitKind::Restore { .. })),
        "the lifted floor queues its restore"
    );
}

#[test]
fn a_preempted_process_holds_back_only_what_it_outranks() {
    let (mut inner, pids) = fleet(&[
        (1, Residency::Evicted, true),
        (2, Residency::Resident, true),
    ]);
    let newcomer = pids[1];
    let floor = inner.preempted_floor().unwrap();
    assert!(
        inner.procs[&newcomer].rank() > floor,
        "no schedule: the newcomer waits for the older preempted process"
    );

    inner.reschedule(|pid| (if *pid == newcomer { 5 } else { 0 }, false));
    let floor = inner.preempted_floor().unwrap();
    assert!(
        inner.procs[&newcomer].rank() <= floor,
        "a higher key starts ahead of a lower-keyed preempted process"
    );
}
