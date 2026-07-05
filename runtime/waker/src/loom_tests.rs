//! X0 loom model-check tests (moved out of lib.rs, Phase 0).

    use super::*;
    use loom::sync::Arc;
    use loom::sync::atomic::{AtomicBool, AtomicU64 as LAtomicU64, Ordering as O};
    use std::task::{RawWaker, RawWakerVTable, Waker};

    /// A loom-friendly waker: sets a flag the waiter thread spins on
    /// (loom's explicit yield makes the spin explorable).
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

    /// B9 exhaustive race: waiter observes e=0 and registers; committer
    /// concurrently bumps the index to 1 and wake_past(1). Assert: the
    /// waiter, following register-then-recheck, ALWAYS either sees the
    /// commit on its re-check or gets the wake — never both missed.
    #[test]
    fn register_commit_race_no_lost_wake() {
        loom::model(|| {
            let table = Arc::new(WakerTable::new());
            let id = table.alloc();
            let head = Arc::new(LAtomicU64::new(0));
            let woken = Arc::new(AtomicBool::new(false));

            let committer = {
                let (table, head) = (table.clone(), head.clone());
                loom::thread::spawn(move || {
                    head.store(1, O::SeqCst);
                    table.wake_past(id, 1);
                })
            };

            // Waiter: observe → register → MANDATORY re-check.
            let e = head.load(O::SeqCst);
            if e == 0 {
                let w = flag_waker(woken.clone());
                assert!(table.register(id, &w, e));
                let ready_now = head.load(O::SeqCst) > e;
                committer.join().unwrap();
                // After the committer finished: if the re-check hadn't seen
                // it, the wake MUST have been delivered.
                if !ready_now {
                    assert!(
                        woken.load(O::SeqCst),
                        "lost wake: neither re-check nor waker fired"
                    );
                }
            } else {
                committer.join().unwrap();
            }
            table.free(id); // drop any parked waker (loom tracks its Arc)
        });
    }

    /// B12 exhaustive: sweep concurrent with registration — the waiter is
    /// always either woken by the sweep or observes the poison flag on its
    /// re-check.
    #[test]
    fn sweep_vs_register_no_hang() {
        loom::model(|| {
            let table = Arc::new(WakerTable::new());
            let id = table.alloc();
            let poisoned = Arc::new(AtomicBool::new(false));
            let woken = Arc::new(AtomicBool::new(false));

            let sweeper = {
                let (table, poisoned) = (table.clone(), poisoned.clone());
                loom::thread::spawn(move || {
                    poisoned.store(true, O::SeqCst);
                    table.sweep(&[id]);
                })
            };

            if !poisoned.load(O::SeqCst) {
                let w = flag_waker(woken.clone());
                assert!(table.register(id, &w, 0));
                let saw_poison = poisoned.load(O::SeqCst);
                sweeper.join().unwrap();
                if !saw_poison {
                    assert!(woken.load(O::SeqCst), "sweep lost: waiter would hang");
                }
            } else {
                sweeper.join().unwrap();
            }
            table.free(id); // drop any parked waker (loom tracks its Arc)
        });
    }
