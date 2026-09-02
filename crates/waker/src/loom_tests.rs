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

