//! Exhaustive generation/reuse and register-then-recheck models.

use super::*;
use ::loom::sync::Arc;
use ::loom::sync::atomic::{AtomicBool, AtomicU64 as LAtomicU64, Ordering as O};
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

#[test]
fn register_commit_race_no_lost_wake() {
    ::loom::model(|| {
        let table = Arc::new(WakerTable::new());
        let id = table.alloc();
        let head = Arc::new(LAtomicU64::new(0));
        let woken = Arc::new(AtomicBool::new(false));

        let committer = {
            let (table, head) = (table.clone(), head.clone());
            ::loom::thread::spawn(move || {
                head.store(1, O::SeqCst);
                table.wake_past(id, 1);
            })
        };

        let observed = head.load(O::SeqCst);
        if observed == 0 {
            let waker = flag_waker(woken.clone());
            assert!(table.register(id, &waker, observed));
            let ready_now = head.load(O::SeqCst) > observed;
            committer.join().unwrap();
            if !ready_now {
                assert!(
                    woken.load(O::SeqCst),
                    "lost wake: neither re-check nor waker observed commit"
                );
            }
        } else {
            committer.join().unwrap();
        }
        table.free(id);
    });
}

#[test]
fn publish_register_race_no_lost_wake() {
    ::loom::model(|| {
        let table = Arc::new(WakerTable::new());
        let id = table.alloc();
        let woken = Arc::new(AtomicBool::new(false));

        let publisher = {
            let table = table.clone();
            ::loom::thread::spawn(move || {
                table.publish(id, FIRST_COMPLETION_EPOCH);
            })
        };

        let observed = table.published(id).unwrap();
        if observed < FIRST_COMPLETION_EPOCH {
            let waker = flag_waker(woken.clone());
            assert!(table.register(id, &waker, observed));
            let ready_now = table.published(id).unwrap() >= FIRST_COMPLETION_EPOCH;
            publisher.join().unwrap();
            if !ready_now {
                assert!(
                    woken.load(O::SeqCst),
                    "lost completion publication during registration"
                );
            }
        } else {
            publisher.join().unwrap();
        }
        table.free(id);
    });
}

#[test]
fn concurrent_out_of_order_publishers_keep_max_epoch() {
    ::loom::model(|| {
        let table = Arc::new(WakerTable::new());
        let id = table.alloc();
        let low = {
            let table = table.clone();
            ::loom::thread::spawn(move || table.publish(id, 3))
        };
        let high = {
            let table = table.clone();
            ::loom::thread::spawn(move || table.publish(id, 7))
        };
        low.join().unwrap();
        high.join().unwrap();
        assert_eq!(table.published(id), Some(7));
        table.free(id);
    });
}

#[test]
fn stale_publish_cannot_write_recycled_generation() {
    ::loom::model(|| {
        let table = Arc::new(WakerTable::new());
        let old = table.alloc();
        let stale = {
            let table = table.clone();
            ::loom::thread::spawn(move || table.publish(old, 9))
        };

        let new = recycle(&table, old);
        assert_eq!(table.publish(new, 3), WakeOutcome::Empty);
        stale.join().unwrap();
        assert_eq!(table.published(new), Some(3));
        table.free(new);
    });
}

#[test]
fn stale_wake_cannot_take_recycled_waker() {
    ::loom::model(|| {
        let table = Arc::new(WakerTable::new());
        let old = table.alloc();
        let stale = {
            let table = table.clone();
            ::loom::thread::spawn(move || table.wake(old))
        };

        let new = recycle(&table, old);
        let new_woken = Arc::new(AtomicBool::new(false));
        let new_waker = flag_waker(new_woken.clone());
        assert!(table.register(new, &new_waker, 0));
        stale.join().unwrap();
        assert!(!new_woken.load(O::SeqCst));
        assert_eq!(table.wake(new), WakeOutcome::Woken);
        assert!(new_woken.load(O::SeqCst));
        table.free(new);
    });
}

#[test]
fn stale_wake_past_cannot_take_recycled_waker() {
    ::loom::model(|| {
        let table = Arc::new(WakerTable::new());
        let old = table.alloc();
        let stale = {
            let table = table.clone();
            ::loom::thread::spawn(move || table.wake_past(old, 1))
        };

        let new = recycle(&table, old);
        let new_woken = Arc::new(AtomicBool::new(false));
        let new_waker = flag_waker(new_woken.clone());
        assert!(table.register(new, &new_waker, 0));
        stale.join().unwrap();
        assert!(!new_woken.load(O::SeqCst));
        assert_eq!(table.wake_past(new, 1), WakeOutcome::Woken);
        assert!(new_woken.load(O::SeqCst));
        table.free(new);
    });
}

#[test]
fn stale_deregister_cannot_clear_recycled_waker() {
    ::loom::model(|| {
        let table = Arc::new(WakerTable::new());
        let old = table.alloc();
        let stale = {
            let table = table.clone();
            ::loom::thread::spawn(move || table.deregister(old))
        };

        let new = recycle(&table, old);
        let new_woken = Arc::new(AtomicBool::new(false));
        let new_waker = flag_waker(new_woken.clone());
        assert!(table.register(new, &new_waker, 0));
        stale.join().unwrap();
        assert_eq!(table.wake(new), WakeOutcome::Woken);
        assert!(new_woken.load(O::SeqCst));
        table.free(new);
    });
}

#[test]
fn stale_register_cannot_overwrite_recycled_waker() {
    ::loom::model(|| {
        let table = Arc::new(WakerTable::new());
        let old = table.alloc();
        let old_woken = Arc::new(AtomicBool::new(false));
        let stale = {
            let table = table.clone();
            let old_waker = flag_waker(old_woken.clone());
            ::loom::thread::spawn(move || table.register(old, &old_waker, 0))
        };

        let new = recycle(&table, old);
        let new_woken = Arc::new(AtomicBool::new(false));
        let new_waker = flag_waker(new_woken.clone());
        assert!(table.register(new, &new_waker, 0));
        stale.join().unwrap();
        assert!(!new_woken.load(O::SeqCst));
        assert_eq!(table.wake(new), WakeOutcome::Woken);
        assert!(new_woken.load(O::SeqCst));
        table.free(new);
    });
}

#[test]
fn stale_published_read_cannot_observe_recycled_epoch() {
    ::loom::model(|| {
        let table = Arc::new(WakerTable::new());
        let old = table.alloc();
        let stale = {
            let table = table.clone();
            ::loom::thread::spawn(move || table.published(old))
        };

        let new = recycle(&table, old);
        assert_eq!(table.publish(new, 9), WakeOutcome::Empty);
        let old_observation = stale.join().unwrap();
        assert_ne!(old_observation, Some(9));
        assert_eq!(table.published(new), Some(9));
        table.free(new);
    });
}

#[test]
fn free_sweep_and_callback_cannot_touch_recycled_slot() {
    ::loom::model(|| {
        let table = Arc::new(WakerTable::new());
        let old = table.alloc();
        let callback = {
            let table = table.clone();
            ::loom::thread::spawn(move || table.publish(old, 5))
        };
        let sweeper = {
            let table = table.clone();
            ::loom::thread::spawn(move || table.sweep(&[old]))
        };

        let new = recycle(&table, old);
        let new_woken = Arc::new(AtomicBool::new(false));
        let new_waker = flag_waker(new_woken.clone());
        assert!(table.register(new, &new_waker, 0));
        callback.join().unwrap();
        sweeper.join().unwrap();

        assert_eq!(table.published(new), Some(0));
        assert!(!new_woken.load(O::SeqCst));
        assert_eq!(table.wake(new), WakeOutcome::Woken);
        assert!(new_woken.load(O::SeqCst));
        table.free(new);
    });
}

#[test]
fn sweep_vs_register_no_hang() {
    ::loom::model(|| {
        let table = Arc::new(WakerTable::new());
        let id = table.alloc();
        let poisoned = Arc::new(AtomicBool::new(false));
        let woken = Arc::new(AtomicBool::new(false));

        let sweeper = {
            let (table, poisoned) = (table.clone(), poisoned.clone());
            ::loom::thread::spawn(move || {
                poisoned.store(true, O::SeqCst);
                table.sweep(&[id]);
            })
        };

        if !poisoned.load(O::SeqCst) {
            let waker = flag_waker(woken.clone());
            assert!(table.register(id, &waker, 0));
            let saw_poison = poisoned.load(O::SeqCst);
            sweeper.join().unwrap();
            if !saw_poison {
                assert!(woken.load(O::SeqCst), "sweep lost: waiter would hang");
            }
        } else {
            sweeper.join().unwrap();
        }
        table.free(id);
    });
}
