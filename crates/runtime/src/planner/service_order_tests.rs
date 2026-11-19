use super::*;

use std::sync::Arc;

use tokio::sync::Notify;

use crate::store::rs::RsUnits;

use super::inner::{Inner, RestoreGate};
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
            rise: false,
        },
    )
}

fn park_rise(inner: &mut Inner, pid: ProcessId, kv_pages: u32) -> EntryKey {
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
            rise: true,
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
    an_early_evictee_is_restored_before_a_later_arrival_is_admitted();
    a_restore_takes_the_head_once_the_fleet_has_stalled();
    a_restore_holds_the_head_when_no_allocation_is_unmet();
    a_younger_restore_still_queues_behind_an_older_allocation();
    returning_host_room_resumes_the_yield();
    a_restore_short_of_room_for_its_roles_waits_then_evicts_once_stalled();
    a_rise_is_served_in_its_process_s_key_order_beside_kv_waiters();
    victims_go_youngest_resident_first_under_the_policys_keys();
}

/// Victims are picked in one order: the policy's key first, then the
/// process made resident last (admitted or restored) before those running
/// longer. A restore makes an old process the youngest resident; a lower
/// policy key goes before any of them.
fn victims_go_youngest_resident_first_under_the_policys_keys() {
    let (mut inner, pids) = fleet(&[
        (1, Residency::Resident, true),
        (2, Residency::Resident, true),
        (3, Residency::Resident, true),
    ]);
    inner.next_seq = 4;
    let order = |inner: &Inner| {
        let mut by: Vec<_> = pids
            .iter()
            .map(|pid| (inner.procs[pid].victim_order(), *pid))
            .collect();
        by.sort();
        by.into_iter().map(|(_, pid)| pid).collect::<Vec<_>>()
    };
    assert_eq!(
        order(&inner),
        vec![pids[2], pids[1], pids[0]],
        "admitted last, first"
    );
    // The oldest is parked and comes back: now it is the youngest resident.
    inner.came_back(pids[0]);
    assert_eq!(
        order(&inner),
        vec![pids[0], pids[2], pids[1]],
        "restored last, first"
    );
    // Its rank (the queue's and eligibility's order) stays its admission's.
    assert!(inner.procs[&pids[0]].rank() < inner.procs[&pids[1]].rank());
    // A policy key leads whatever the recency.
    inner.procs.get_mut(&pids[1]).unwrap().key = -1;
    assert_eq!(
        order(&inner),
        vec![pids[1], pids[0], pids[2]],
        "the policy's key leads"
    );
}

/// A rise that cannot fire narrower is a keyed demand of its process: beside
/// a kv waiter, the older process's ask is served first, whichever it is;
/// no rise queues anonymously ahead of, or behind, the waiters it starves.
fn a_rise_is_served_in_its_process_s_key_order_beside_kv_waiters() {
    let (mut inner, pids) = fleet(&[
        (1, Residency::Resident, true),
        (3, Residency::Resident, true),
    ]);
    park_rise(&mut inner, pids[1], 8);
    park_allocation(&mut inner, pids[0], 4);
    let (key, waiter) = inner.unmet_head().expect("a head");
    assert_eq!(key.0.seq, 1, "the older kv waiter before the younger rise");
    assert!(matches!(
        waiter.kind,
        WaitKind::Allocation { rise: false, .. }
    ));

    let (mut inner, pids) = fleet(&[
        (1, Residency::Resident, true),
        (3, Residency::Resident, true),
    ]);
    park_allocation(&mut inner, pids[1], 4);
    park_rise(&mut inner, pids[0], 8);
    let (key, waiter) = inner.unmet_head().expect("a head");
    assert_eq!(key.0.seq, 1, "the older rise before the younger kv waiter");
    assert!(matches!(
        waiter.kind,
        WaitKind::Allocation { rise: true, .. }
    ));
}

/// A restore whose role slots found no room is queued again, a waiter: it
/// is not served on spare pages alone (the room it lacks is not pages),
/// goes again on a completion's credit, waits while the fleet decodes (a
/// completion frees room soon; funding it then thrashed, 3x the wall on the
/// M1's beam stress), and has room made for its runs (`Evict`, its need
/// named) once the fleet is stalled, a fleet of processes waiting at binds
/// included (AMD's 7-minute stall).
fn a_restore_short_of_room_for_its_roles_waits_then_evicts_once_stalled() {
    let (mut inner, pids) = fleet(&[
        (1, Residency::Evicted, true),
        (2, Residency::Resident, true),
    ]);
    park_restore(&mut inner, pids[0], 4);
    assert_eq!(
        inner.restore_gate(pids[0], 64),
        RestoreGate::Serve,
        "control"
    );
    inner.procs.get_mut(&pids[0]).unwrap().roles_short = Some(super::proc::PendingBind {
        run: 3,
        pieces: 1,
        pages: 3,
    });
    assert!(!inner.fleet_stalled());
    assert_eq!(
        inner.restore_gate(pids[0], 64),
        RestoreGate::Evict {
            need: (3, 1),
            fund: false
        },
        "spare pages are no room; the fleet decodes: it goes again only on \
         free runs of its need, nothing moved for it"
    );
    inner.completion_credit = 1;
    assert_eq!(
        inner.restore_gate(pids[0], 64),
        RestoreGate::Serve,
        "a completion's credit"
    );
    assert_eq!(inner.completion_credit, 0);
    // The resident waits at a bind for room: it makes no progress, so the
    // fleet is stalled and the restore's runs are funded.
    inner.procs.get_mut(&pids[1]).unwrap().bind_waiting = true;
    assert!(inner.fleet_waits_at_binds(), "the fleet waits at binds");
    assert_eq!(
        inner.restore_gate(pids[0], 0),
        RestoreGate::Evict {
            need: (3, 1),
            fund: true
        }
    );
}

/// Mac Pro's M5 ramp (500 at once): processes evicted in the first minute
/// waited behind every later arrival and were restored only at the end, or
/// never. An older restore heads ahead of a younger admission (a prompt
/// not yet seated), the fleet decoding or not; a seated process's own
/// growth keeps its turn, or the decoders whose completions free the
/// restore's room would wait on it.
fn an_early_evictee_is_restored_before_a_later_arrival_is_admitted() {
    let (mut inner, pids) = fleet(&[
        (1, Residency::Evicted, true),
        (2, Residency::Resident, true),
        (3, Residency::Resident, true),
    ]);
    // pids[1] decodes, seated; pids[2] is a later arrival, not yet seated.
    inner.procs.get_mut(&pids[1]).unwrap().seated = true;
    park_restore(&mut inner, pids[0], 18);
    park_allocation(&mut inner, pids[2], 1);
    assert!(!inner.fleet_stalled(), "the fleet decodes");
    let (key, waiter) = inner.unmet_head().expect("a head");
    assert_eq!(key.0.seq, 1, "the early evictee comes back first");
    assert!(matches!(waiter.kind, WaitKind::Restore { .. }));

    // The seated decoder asking for its next page keeps its turn.
    let (mut inner, pids) = fleet(&[
        (1, Residency::Evicted, true),
        (2, Residency::Resident, true),
    ]);
    let decoder = inner.procs.get_mut(&pids[1]).unwrap();
    decoder.seated = true;
    // On its way: a fire of its own in flight.
    decoder.fires = 1;
    park_restore(&mut inner, pids[0], 18);
    park_allocation(&mut inner, pids[1], 1);
    assert!(!inner.fleet_stalled(), "it decodes");
    let (key, waiter) = inner.unmet_head().expect("a head");
    assert_eq!(key.0.seq, 2, "a seated decoder's growth keeps its turn");
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
    // pids[2] decodes, seated, a fire in flight: its growth is what the
    // older restore yields to (a new admission would wait behind it).
    let decoder = inner.procs.get_mut(&pids[2]).unwrap();
    decoder.seated = true;
    decoder.fires = 1;
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
