//! The slot fence, window by window: a page an in-flight frame names is never
//! freed, reused, moved or copied out until that frame retires. A frame holds
//! its working set's fire lease from the binding of its slots (before its
//! page tables are resolved) until it is finalized after the engine settles
//! it, and its kv epoch from the binding until the engine settles it.
//! (The shared pool's evacuations are pinned in `store::slots`'s tests.)

use std::collections::HashSet;
use std::pin::pin;
use std::task::Poll;

use crate::store::kv::page_table::WorkingSetId;
use crate::store::kv::working_set::{FireLeaseError, KvWorkingSet};
use crate::store::registry::{self, with_kv_lock};

/// A working set of model `model` holding `pages` device pages.
fn backed(model: usize, pages: u64) -> KvWorkingSet {
    let stores = registry::get(model, 0);
    let id = with_kv_lock(&stores.kv, "test", |kv| {
        let id = kv.create_working_set();
        kv.reserve(id, pages).unwrap();
        let mut granted = kv.reserve_device_pages(pages as usize).unwrap();
        kv.ensure_backed_reserved(id, pages, &mut granted).unwrap();
        id
    });
    KvWorkingSet::new(model, 0, id, 16)
}

fn pages_of(model: usize, ws: WorkingSetId) -> HashSet<u32> {
    let stores = registry::get(model, 0);
    with_kv_lock(&stores.kv, "test", |kv| {
        kv.flat_table(ws)
            .unwrap()
            .iter()
            .map(|page| page.0)
            .collect()
    })
}

fn free_pages(model: usize) -> usize {
    let stores = registry::get(model, 0);
    with_kv_lock(&stores.kv, "test", |kv| kv.available_pages())
}

/// Suspect (a): a park (the planner's eviction) raises the set's fence and
/// waits for it to quiesce before it copies a page out and frees it. A frame
/// whose slots were bound before the fence holds the park off until it is
/// finalized; none is bound after it.
#[tokio::test(flavor = "current_thread")]
async fn a_park_waits_for_the_frame_in_flight_and_refuses_new_ones() {
    let model = registry::register_model(16, &[8], &[0]);
    let ws = backed(model, 2);
    let frame = ws.fire_lease().expect("a frame binds its slots");
    let handle = ws.suspend_handle();
    handle.fence();
    assert_eq!(
        ws.fire_lease().err(),
        Some(FireLeaseError::Fenced),
        "a frame was bound after the park's fence went up"
    );
    let mut quiesced = pin!(handle.quiesce());
    assert!(
        matches!(futures::poll!(quiesced.as_mut()), Poll::Pending),
        "the park went on with a frame naming its pages in flight"
    );
    drop(frame);
    assert!(
        matches!(futures::poll!(quiesced.as_mut()), Poll::Ready(())),
        "the park waits on no frame once the last one is finalized"
    );
}

/// Suspects (b), (c): pages let go while a frame that names them is in
/// flight (a set released, a write's old tail, a reader unpinned) wait on
/// the newest epoch handed out: a restore's grant, or any lane's, cannot take
/// them before that frame retires.
#[test]
fn pages_let_go_under_a_frame_are_granted_to_no_one_until_it_retires() {
    let model = registry::register_model(16, &[4], &[0]);
    let ws = backed(model, 4);
    let named = pages_of(model, ws.id);
    let stores = registry::get(model, 0);
    let fire = with_kv_lock(&stores.kv, "test", |kv| {
        let fire = kv.begin_fire();
        let epoch = kv.current_epoch();
        kv.release_working_set(ws.id, epoch);
        kv.retire_idle();
        fire
    });
    assert_eq!(
        free_pages(model),
        0,
        "a page the frame names was freed under it"
    );
    let granted = with_kv_lock(&stores.kv, "test", |kv| kv.reserve_device_pages(1));
    assert!(
        granted.is_none(),
        "a grant took a page a frame in flight names"
    );
    with_kv_lock(&stores.kv, "test", |kv| kv.settle(fire));
    let granted = with_kv_lock(&stores.kv, "test", |kv| kv.reserve_device_pages(4))
        .expect("the frame retired: its pages are back");
    assert_eq!(
        granted.iter().map(|page| page.0).collect::<HashSet<_>>(),
        named
    );
    // Its lifecycle is done with: the store let the set go above.
    std::mem::forget(ws);
}

/// Suspect (c): a process reclaimed and restarted mid-pass gives its sets
/// back while a piece of its pass may still be in flight. The set's pages go
/// back only once that frame's lease drops, which is after the engine has
/// settled it, never at the release.
#[test]
fn a_set_released_mid_pass_keeps_its_pages_until_the_piece_in_flight_is_finalized() {
    let model = registry::register_model(16, &[4], &[0]);
    let ws = backed(model, 3);
    assert_eq!(free_pages(model), 1);
    let piece = ws.fire_lease().expect("a piece of the pass is in flight");
    ws.release();
    drop(ws);
    assert_eq!(
        free_pages(model),
        1,
        "the restart's release freed pages a piece in flight still writes"
    );
    assert_eq!(
        ws_lease_refused_after_release(model),
        Some(FireLeaseError::Released)
    );
    drop(piece);
    assert_eq!(
        free_pages(model),
        4,
        "the piece finalized: its pages are back"
    );
}

/// A set whose release was asked binds no frame after.
fn ws_lease_refused_after_release(model: usize) -> Option<FireLeaseError> {
    let ws = backed(model, 0);
    ws.release();
    ws.fire_lease().err()
}
