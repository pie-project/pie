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
        inner.restore_gate(pids[0]),
        RestoreGate::Serve,
        "control"
    );
    inner.procs.get_mut(&pids[0]).unwrap().roles_short = Some(super::proc::PendingBind {
        singles: 0,
        run: 3,
        pieces: 1,
        pages: 3,
    });
    // The resident decodes: a fire of its in flight.
    {
        let proc = inner.procs.get_mut(&pids[1]).unwrap();
        proc.fires = 1;
        proc.fired_at = crate::rt::Instant::now();
    }
    assert!(!inner.nothing_in_flight());
    assert_eq!(
        inner.restore_gate(pids[0]),
        RestoreGate::Evict {
            need: (3, 1),
            fund: false
        },
        "spare pages are no room; the fleet decodes: it goes again only on \
         free runs of its need, nothing moved for it"
    );
    inner.completion_credit = 1;
    assert_eq!(
        inner.restore_gate(pids[0]),
        RestoreGate::Serve,
        "a completion's credit"
    );
    assert_eq!(inner.completion_credit, 0);
    // Its fire landed and it waits at a bind for room: nothing in flight
    // frees room, and the restore's runs are funded.
    {
        let proc = inner.procs.get_mut(&pids[1]).unwrap();
        proc.fires = 0;
        proc.bind_waiting = true;
    }
    assert!(inner.nothing_in_flight(), "nothing in flight frees room");
    assert_eq!(
        inner.restore_gate(pids[0]),
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

    inner.reschedule(None, |pid| (if *pid == young { 5 } else { 0 }, false));
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

    inner.reschedule(None, |_| (0, false));
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

    inner.reschedule(None, |pid| (0, *pid == paused || *pid == running));
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

    inner.reschedule(None, |_| (0, false));
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

    inner.reschedule(None, |pid| (if *pid == newcomer { 5 } else { 0 }, false));
    let floor = inner.preempted_floor().unwrap();
    assert!(
        inner.procs[&newcomer].rank() <= floor,
        "a higher key starts ahead of a lower-keyed preempted process"
    );
}

/// A roles-short restore at the head behind which another process waits at
/// a bind is funded while a steady decoder keeps firing: that bind goes only
/// once the head is served, so nothing the fleet does frees the head's room,
/// and a decoder that never stops firing never stalls the fleet. Funded only
/// once nothing was in flight, the head starved behind the decoder. Without
/// a bind waiting behind it, a firing fleet still leaves it unfunded.
#[test]
fn a_roles_short_head_with_a_bind_waiting_behind_it_is_funded_while_a_decoder_fires() {
    let (mut inner, pids) = fleet(&[
        (1, Residency::Evicted, true),
        (2, Residency::Resident, true),
        (3, Residency::Resident, true),
    ]);
    let (head, decoder, binder) = (pids[0], pids[1], pids[2]);
    park_restore(&mut inner, head, 4);
    inner.procs.get_mut(&head).unwrap().roles_short = Some(super::proc::PendingBind {
        singles: 0,
        run: 3,
        pieces: 1,
        pages: 3,
    });
    // The decoder fires, and keeps firing.
    {
        let proc = inner.procs.get_mut(&decoder).unwrap();
        proc.fires = 1;
        proc.fired_at = crate::rt::Instant::now();
    }
    assert!(!inner.nothing_in_flight());
    assert_eq!(
        inner.restore_gate(head),
        RestoreGate::Evict {
            need: (3, 1),
            fund: false
        },
        "control: nothing waits behind it, and the fleet decodes"
    );
    // A process waits at a bind behind it.
    inner.procs.get_mut(&binder).unwrap().bind_waiting = true;
    assert!(!inner.nothing_in_flight(), "the decoder has not stopped");
    assert_eq!(
        inner.restore_gate(head),
        RestoreGate::Evict {
            need: (3, 1),
            fund: true
        },
        "a bind behind it: room is made for it while the decoder fires"
    );
    // The head's own bind wait is no reason: only another's.
    inner.procs.get_mut(&binder).unwrap().bind_waiting = false;
    inner.procs.get_mut(&head).unwrap().bind_waiting = true;
    assert_eq!(
        inner.restore_gate(head),
        RestoreGate::Evict {
            need: (3, 1),
            fund: false
        }
    );
}
