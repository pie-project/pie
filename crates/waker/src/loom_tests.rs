#![allow(unsafe_code)]
use super::*;
use ::loom::sync::Arc;
use ::loom::sync::atomic::{AtomicBool, Ordering as O};
use std::task::{RawWaker, RawWakerVTable, Waker};

fn flag_waker(flag: Arc<AtomicBool>) -> Waker {
    fn clone(p: *const ()) -> RawWaker {
        let a = unsafe { Arc::from_raw(p as *const AtomicBool) };
        let b = a.clone();
        std::mem::forget(a);
        RawWaker::new(Arc::into_raw(b) as *const (), &VT)
    }
    fn wake(p: *const ()) {
        let a = unsafe { Arc::from_raw(p as *const AtomicBool) };
        a.store(true, O::SeqCst);
    }
    fn wake_by_ref(p: *const ()) {
        let a = unsafe { Arc::from_raw(p as *const AtomicBool) };
        a.store(true, O::SeqCst);
        std::mem::forget(a);
    }
    fn drop_raw(p: *const ()) {
        unsafe { drop(Arc::from_raw(p as *const AtomicBool)) };
    }
    static VT: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_by_ref, drop_raw);
    unsafe { Waker::from_raw(RawWaker::new(Arc::into_raw(flag) as *const (), &VT)) }
}

fn recycle(table: &WakerTable, old: WakerSlotId) -> WakerSlotId {
    table.free(old);
    let new = table.alloc();
    assert_eq!(old as u32, new as u32, "the model must exercise slot reuse");
    assert_ne!(old, new, "the recycled slot must have a new generation");
    new
}

/// Two tasks wait on one slot while a third settles it (a pipeline's lanes
/// on their front fire): in every interleaving each waiter either sees the
/// settle on its re-check after registering, or is woken by it.
#[test]
fn every_waiter_of_a_slot_sees_the_settle_or_is_woken() {
    ::loom::model(|| {
        let table = Arc::new(WakerTable::new());
        let id = table.alloc();
        let settled = Arc::new(AtomicBool::new(false));
        let waiters: Vec<_> = (0..2)
            .map(|_| {
                let (table, settled) = (table.clone(), settled.clone());
                let woken = Arc::new(AtomicBool::new(false));
                let waker = flag_waker(woken.clone());
                let handle = ::loom::thread::spawn(move || {
                    if settled.load(O::SeqCst) {
                        return true;
                    }
                    assert!(table.register(id, &waker, 0));
                    settled.load(O::SeqCst)
                });
                (handle, woken)
            })
            .collect();
        settled.store(true, O::SeqCst);
        table.wake(id);
        for (handle, woken) in waiters {
            let saw = handle.join().unwrap();
            assert!(
                saw || woken.load(O::SeqCst),
                "a waiter registered before the settle slept through it"
            );
        }
        // Releases what a waiter registered after the settle left behind.
        table.free(id);
    });
}

/// The slot reused after a free carries a new generation: a wake for the
/// old one reaches no waiter of the new.
#[test]
fn a_wake_for_a_freed_slot_misses_its_successor() {
    ::loom::model(|| {
        let table = WakerTable::new();
        let old = table.alloc();
        let new = recycle(&table, old);
        let woken = Arc::new(AtomicBool::new(false));
        assert!(table.register(new, &flag_waker(woken.clone()), 0));
        assert!(!table.register(old, &flag_waker(woken.clone()), 0));
        table.wake(old);
        assert!(!woken.load(O::SeqCst));
        table.free(new);
        assert!(woken.load(O::SeqCst), "a free wakes the slot's waiter");
    });
}
