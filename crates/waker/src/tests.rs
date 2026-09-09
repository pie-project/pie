use super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64 as StdAtomicU64};
use std::task::{Wake, Waker};

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
fn tests_every_case() {
    alloc_register_wake_roundtrip();
    epoch_filter_wakes_only_when_index_passes();
    foreign_completion_publish_records_epoch_before_wake();
    reserved_epochs_are_rejected_in_release_logic();
    stale_generation_is_noop_b10();
    spurious_wakes_are_harmless();
    sweep_on_abort_resolves_blocked_take_to_err_b12();
}

fn alloc_register_wake_roundtrip() {
    let t = WakerTable::new();
    let id = t.alloc();
    let (f, w) = flag_waker();
    assert!(t.register(id, &w, 0));
    assert_eq!(t.wake(id), WakeOutcome::Woken);
    assert!(f.0.load(std::sync::atomic::Ordering::SeqCst));
    assert_eq!(t.wake(id), WakeOutcome::Empty);
}

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
            WaitFuture::new(&t, id, move || match t2.published(id) {
                Some(epoch) if epoch >= 7 => Readiness::Ready(epoch),
                Some(epoch) => Readiness::Pending {
                    observed_epoch: epoch,
                },
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

fn stale_generation_is_noop_b10() {
    let t = WakerTable::new();
    let id = t.alloc();
    let (_, w) = flag_waker();
    assert!(t.register(id, &w, 0));
    t.free(id);
    assert_eq!(t.wake(id), WakeOutcome::Stale);
    assert_eq!(t.wake_past(id, 99), WakeOutcome::Stale);
    assert_eq!(t.publish(id, 99), WakeOutcome::Stale);
    assert_eq!(t.published(id), None);
    assert!(!t.register(id, &w, 0));
    let id2 = t.alloc();
    assert_eq!(id & 0xFFFF_FFFF, id2 & 0xFFFF_FFFF, "index recycled");
    assert_ne!(id, id2, "generation bumped");
    assert!(t.register(id2, &w, 0));
    assert_eq!(t.wake(id), WakeOutcome::Stale);
    assert_eq!(t.wake(id2), WakeOutcome::Woken);
}

fn spurious_wakes_are_harmless() {
    let t = WakerTable::new();
    let id = t.alloc();
    assert_eq!(t.wake(id), WakeOutcome::Empty);
    assert_eq!(t.wake_past(id, 1), WakeOutcome::Empty);
    let (_, w) = flag_waker();
    assert!(t.register(id, &w, 0));
    assert_eq!(t.wake(id), WakeOutcome::Woken);
    assert_eq!(t.wake(id), WakeOutcome::Empty);
    let m = t.metrics();
    assert_eq!(m.woken, 1);
    assert_eq!(m.empty, 3);
}

fn sweep_on_abort_resolves_blocked_take_to_err_b12() {
    let t = Arc::new(WakerTable::new());
    let ch = ChannelWakers::alloc(&t);
    let poisoned = Arc::new(AtomicBool::new(false));
    let head = Arc::new(StdAtomicU64::new(0));

    let sweeper = {
        let (t, poisoned) = (t.clone(), poisoned.clone());
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            poisoned.store(true, std::sync::atomic::Ordering::SeqCst);
            ch.sweep(&t);
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
            WaitFuture::new(&t, ch.reader, move || {
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
