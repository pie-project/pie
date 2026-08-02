//! FFI — the only surface the other side of the boundary sees (B10): opaque
//! `u64` slot ids in, `0/1` out, callable from any thread, never unwinds.

#[cfg(not(loom))]
use crate::table::{WakeOutcome, WakerTable};

/// Wake the waiter parked on `slot_id`, unconditionally. Returns `1` if a
/// waker was woken, `0` otherwise (stale id / nobody parked). Callable from
/// any thread; never unwinds.
#[cfg(not(loom))]
#[unsafe(no_mangle)]
pub extern "C" fn pie_wake(slot_id: u64) -> u8 {
    let r = std::panic::catch_unwind(|| WakerTable::global().wake(slot_id));
    matches!(r, Ok(WakeOutcome::Woken)) as u8
}

/// Epoch-filtered wake (B9): wake the waiter parked on `slot_id` iff the
/// committed `ring_index` has passed its registered observation. Callable
/// from any thread; never unwinds.
#[cfg(not(loom))]
#[unsafe(no_mangle)]
pub extern "C" fn pie_wake_past(slot_id: u64, ring_index: u64) -> u8 {
    let r = std::panic::catch_unwind(|| WakerTable::global().wake_past(slot_id, ring_index));
    matches!(r, Ok(WakeOutcome::Woken)) as u8
}
