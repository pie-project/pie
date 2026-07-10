//! X0 waker-table unit tests (moved out of lib.rs, Phase 0).

    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64 as StdAtomicU64};
    use std::task::{Wake, Waker};

    /// A test waker that records wakes and can unpark a spinning poller.
    struct Flag(AtomicBool);
    impl Wake for Flag {
        fn wake(self: Arc<Self>) {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }
    fn flag_waker() -> (Arc<Flag>, Waker) {
        let f = Arc::new(Flag(AtomicBool::new(false)));
        (f.clone(), f.into())
    }

    #[test]
    fn alloc_register_wake_roundtrip() {
        let t = WakerTable::new();
        let id = t.alloc();
        let (f, w) = flag_waker();
        assert!(t.register(id, &w, 0));
        assert_eq!(t.wake(id), WakeOutcome::Woken);
        assert!(f.0.load(std::sync::atomic::Ordering::SeqCst));
        // One-shot: the waker was taken.
        assert_eq!(t.wake(id), WakeOutcome::Empty);
    }

    #[test]
    fn epoch_filter_wakes_only_when_index_passes() {
        let t = WakerTable::new();
        let id = t.alloc();
        let (f, w) = flag_waker();
        assert!(t.register(id, &w, 5));
        assert_eq!(t.wake_past(id, 4), WakeOutcome::Filtered);
        assert_eq!(t.wake_past(id, 5), WakeOutcome::Filtered);
        assert!(!f.0.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(t.wake_past(id, 6), WakeOutcome::Woken);
        assert!(f.0.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn foreign_completion_publish_records_epoch_before_wake() {
        let t = Arc::new(WakerTable::new());
        let id = t.alloc();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        let publisher = {
            let t = Arc::clone(&t);
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(20));
                t.publish(id, 7)
            })
        };
        let observed = rt.block_on(async {
            let t2 = Arc::clone(&t);
            tokio::time::timeout(
                std::time::Duration::from_secs(5),
                WaitFuture::new(&*t, id, move || match t2.published(id) {
                    Some(epoch) if epoch >= 7 => Readiness::Ready(epoch),
                    Some(epoch) => Readiness::Pending { observed_epoch: epoch },
                    None => Readiness::Ready(0),
                }),
            )
            .await
            .expect("foreign completion callback lost its wake")
        });
        let outcome = publisher.join().unwrap();
        assert_eq!(observed, 7);
        assert!(matches!(outcome, WakeOutcome::Woken | WakeOutcome::Empty));
        t.free(id);
    }

    #[test]
    fn published_completion_epoch_never_regresses() {
        let t = WakerTable::new();
        let id = t.alloc();
        assert_eq!(t.publish(id, 7), WakeOutcome::Empty);
        assert_eq!(t.publish(id, 3), WakeOutcome::Empty);
        assert_eq!(t.published(id), Some(7));
        t.free(id);
    }

    #[test]
    fn reserved_epochs_are_rejected_in_release_logic() {
        let t = WakerTable::new();
        let id = t.alloc();
        let (_, w) = flag_waker();
        assert!(!t.register(id, &w, u64::MAX));
        assert_eq!(t.publish(id, 0), WakeOutcome::InvalidEpoch);
        assert_eq!(t.publish(id, u64::MAX), WakeOutcome::InvalidEpoch);
        assert_eq!(t.wake_past(id, u64::MAX), WakeOutcome::InvalidEpoch);
        assert_eq!(t.published(id), Some(0));
        assert_eq!(t.metrics().invalid_epoch, 3);
        t.free(id);
    }

    #[test]
    fn generation_wrap_retires_slot_instead_of_reusing_zero() {
        let t = WakerTable::new();
        let first = t.alloc();
        let wrapped = t
            .force_generation_for_test(first, u32::MAX)
            .expect("live slot");
        let retired_index = wrapped as u32;
        t.free(wrapped);
        assert_eq!(t.wake(wrapped), WakeOutcome::Stale);

        let next = t.alloc();
        assert_ne!(next as u32, retired_index, "retired index must not be reused");
        assert_ne!((next >> 32) as u32, 0, "generation zero is reserved");
        t.free(next);
    }

    #[test]
    fn stale_generation_is_noop_b10() {
        let t = WakerTable::new();
        let id = t.alloc();
        let (_, w) = flag_waker();
        assert!(t.register(id, &w, 0));
        t.free(id);
        // The dead channel's id, still held by "C++": every op is inert.
        assert_eq!(t.wake(id), WakeOutcome::Stale);
        assert_eq!(t.wake_past(id, 99), WakeOutcome::Stale);
        assert_eq!(t.publish(id, 99), WakeOutcome::Stale);
        assert_eq!(t.published(id), None);
        assert!(!t.register(id, &w, 0));
        // The recycled slot gets a NEW generation: old id still stale.
        let id2 = t.alloc();
        assert_eq!(id & 0xFFFF_FFFF, id2 & 0xFFFF_FFFF, "index recycled");
        assert_ne!(id, id2, "generation bumped");
        assert!(t.register(id2, &w, 0));
        assert_eq!(t.wake(id), WakeOutcome::Stale);
        assert_eq!(t.wake(id2), WakeOutcome::Woken);
    }

    #[test]
    fn spurious_wakes_are_harmless() {
        let t = WakerTable::new();
        let id = t.alloc();
        // Nobody parked: empty, not an error, no panic.
        assert_eq!(t.wake(id), WakeOutcome::Empty);
        assert_eq!(t.wake_past(id, 1), WakeOutcome::Empty);
        // Double-wake after a single register: second is empty.
        let (_, w) = flag_waker();
        assert!(t.register(id, &w, 0));
        assert_eq!(t.wake(id), WakeOutcome::Woken);
        assert_eq!(t.wake(id), WakeOutcome::Empty);
        let m = t.metrics();
        assert_eq!(m.woken, 1);
        assert_eq!(m.empty, 3);
    }

    #[test]
    fn sweep_on_abort_resolves_blocked_take_to_err_b12() {
        // A take().await? parked on a channel that will NEVER become full:
        // poison + sweep (from a foreign thread, as the driver would) must
        // resolve it to Err — never hang.
        let t = Arc::new(WakerTable::new());
        let ch = ChannelWakers::alloc(&t);
        let poisoned = Arc::new(AtomicBool::new(false));
        let head = Arc::new(StdAtomicU64::new(0)); // ring index: never bumps

        let sweeper = {
            let (t, poisoned) = (t.clone(), poisoned.clone());
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(20));
                poisoned.store(true, std::sync::atomic::Ordering::SeqCst);
                ch.sweep(&t); // B12: wake both endpoints, epochs ignored
            })
        };

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        let res = rt.block_on(async {
            let (t, poisoned, head) = (t.clone(), poisoned.clone(), head.clone());
            tokio::time::timeout(
                std::time::Duration::from_secs(5),
                WaitFuture::new(&*t, ch.reader, move || {
                    if poisoned.load(std::sync::atomic::Ordering::SeqCst) {
                        return Readiness::Ready(Err::<u64, &str>("poisoned"));
                    }
                    let h = head.load(std::sync::atomic::Ordering::SeqCst);
                    if h > 0 {
                        Readiness::Ready(Ok(h))
                    } else {
                        Readiness::Pending { observed_epoch: h }
                    }
                }),
            )
            .await
            .expect("sweep lost: blocked take hung")
        });
        sweeper.join().unwrap();
        assert_eq!(res, Err("poisoned"));
        let m = t.metrics();
        assert_eq!(m.swept, 2, "both endpoints swept");
    }

    #[test]
    fn ffi_wake_from_foreign_thread_completes_future() {
        // The exported symbol path: a raw std thread (standing in for the
        // C++ commit tail) calls pie_wake on the GLOBAL table.
        let t = WakerTable::global();
        let id = t.alloc();
        let head = Arc::new(StdAtomicU64::new(0));
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        let res = rt.block_on(async {
            let head2 = head.clone();
            let waiter = WaitFuture::new(t, id, move || {
                let h = head2.load(std::sync::atomic::Ordering::SeqCst);
                if h > 0 { Readiness::Ready(h) } else { Readiness::Pending { observed_epoch: h } }
            });
            let committer = {
                let head = head.clone();
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    head.store(1, std::sync::atomic::Ordering::SeqCst);
                    super::pie_wake_past(id, 1)
                })
            };
            let v = tokio::time::timeout(std::time::Duration::from_secs(5), waiter)
                .await
                .expect("no lost wake");
            let woke = committer.join().unwrap();
            (v, woke)
        });
        assert_eq!(res.0, 1);
        // Either the FFI wake delivered it (1) or the fast path won the race
        // and the wake found the slot empty/deregistered (0) — both legal.
        assert!(res.1 <= 1);
        t.free(id);
    }

    #[test]
    fn register_commit_race_stress_no_lost_wake() {
        // Non-loom stress version of the race: many iterations of
        // observe(e) ∥ commit(e+1)+wake_past — a lost wake deadlocks (caught
        // by the timeout).
        let t = WakerTable::global();
        for i in 0..2000 {
            let id = t.alloc();
            let head = Arc::new(StdAtomicU64::new(0));
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_time()
                .build()
                .unwrap();
            let head2 = head.clone();
            let committer = std::thread::spawn(move || {
                if i % 2 == 0 {
                    std::thread::yield_now();
                }
                head2.store(1, std::sync::atomic::Ordering::SeqCst);
                super::pie_wake_past(id, 1);
            });
            let v = rt.block_on(async {
                let head = head.clone();
                tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    WaitFuture::new(t, id, move || {
                        let h = head.load(std::sync::atomic::Ordering::SeqCst);
                        if h > 0 { Readiness::Ready(h) } else { Readiness::Pending { observed_epoch: h } }
                    }),
                )
                .await
                .expect("lost wake (register/commit race)")
            });
            assert_eq!(v, 1);
            committer.join().unwrap();
            t.free(id);
        }
    }

    #[test]
    fn wake_to_poll_latency_probe() {
        // X0 probe: time from pie_wake to the woken future's poll observing
        // readiness. Prints; asserts only a sanity bound (CI-safe).
        let t = WakerTable::global();
        let id = t.alloc();
        let head = Arc::new(StdAtomicU64::new(0));
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        let woke_at = Arc::new(StdAtomicU64::new(0));
        let base = std::time::Instant::now();
        let head2 = head.clone();
        let woke_at2 = woke_at.clone();
        let committer = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            head2.store(1, std::sync::atomic::Ordering::SeqCst);
            let now = base.elapsed().as_nanos() as u64;
            woke_at2.store(now, std::sync::atomic::Ordering::SeqCst);
            super::pie_wake(id);
        });
        let polled_at = rt.block_on(async {
            let head = head.clone();
            WaitFuture::new(t, id, move || {
                let h = head.load(std::sync::atomic::Ordering::SeqCst);
                if h > 0 { Readiness::Ready(base.elapsed().as_nanos() as u64) } else { Readiness::Pending { observed_epoch: h } }
            })
            .await
        });
        committer.join().unwrap();
        let woke = woke_at.load(std::sync::atomic::Ordering::SeqCst);
        let latency_ns = polled_at.saturating_sub(woke);
        println!("wake-to-poll latency: {latency_ns} ns");
        assert!(latency_ns < 100_000_000, "sanity bound: < 100ms");
        t.free(id);
    }

    #[test]
    fn wakes_per_fire_metrics() {
        let t = WakerTable::new();
        let ch = ChannelWakers::alloc(&t);
        let (_, w) = flag_waker();
        // Reader parked at epoch 3; a fire commits index 4 on this channel
        // and blind-fires both endpoints (writer idle): exactly one useful
        // wake, one empty — the probe distinguishes them.
        assert!(t.register(ch.reader, &w, 3));
        assert_eq!(t.wake_past(ch.reader, 4), WakeOutcome::Woken);
        assert_eq!(t.wake_past(ch.writer, 4), WakeOutcome::Empty);
        let m = t.metrics();
        assert_eq!((m.woken, m.empty), (1, 1));
    }
