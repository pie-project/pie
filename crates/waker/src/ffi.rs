#[cfg(all(panic = "abort", not(target_arch = "wasm32")))]
compile_error!(
    "the waker's C ABI contract is `never unwinds, returns 0/1`, which is \
     implemented with `catch_unwind` and so requires `panic = \"unwind\"`"
);

#[cfg(not(loom))]
use crate::table::{WakeOutcome, WakerTable};

#[cfg(not(loom))]
#[unsafe(no_mangle)]
pub extern "C" fn pie_wake(slot_id: u64) -> u8 {
    let r = std::panic::catch_unwind(|| WakerTable::global().wake(slot_id));
    matches!(r, Ok(WakeOutcome::Woken)) as u8
}

#[cfg(not(loom))]
#[unsafe(no_mangle)]
pub extern "C" fn pie_wake_past(slot_id: u64, ring_index: u64) -> u8 {
    let r = std::panic::catch_unwind(|| WakerTable::global().wake_past(slot_id, ring_index));
    matches!(r, Ok(WakeOutcome::Woken)) as u8
}
