#![allow(unsafe_code)]
//! A walked fire dissected by node (`pie tune --check-estimate`): while on,
//! the device is drained before and after each node the walk dispatches and
//! the time between is the node's, summed by its op's name. Only a walk
//! (no capture: `engine.graphs = "off"`) may be dissected, since a drain
//! inside a capture breaks it.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::Instant;

#[derive(Default)]
struct Dissection {
    entered: Option<Instant>,
    /// Each op's nanoseconds and nodes.
    ops: BTreeMap<&'static str, (u64, u32)>,
}

thread_local! {
    static ON: RefCell<Option<Dissection>> = const { RefCell::new(None) };
}

fn drain() {
    #[cfg(feature = "device")]
    // SAFETY: a drain of the current device, which the walk's thread holds.
    unsafe {
        let _ = crate::device::rt::cudaDeviceSynchronize();
    }
}

/// Start dissecting this thread's walks, from nothing.
pub fn start() {
    drain();
    ON.with(|on| *on.borrow_mut() = Some(Dissection::default()));
}

/// Stop, and take each op's nanoseconds and nodes since [`start`].
#[must_use]
pub fn take() -> Vec<(&'static str, u64, u32)> {
    ON.with(|on| on.borrow_mut().take())
        .map(|d| d.ops.into_iter().map(|(op, (ns, n))| (op, ns, n)).collect())
        .unwrap_or_default()
}

pub(crate) fn enter() {
    ON.with(|on| {
        if let Some(d) = on.borrow_mut().as_mut() {
            drain();
            d.entered = Some(Instant::now());
        }
    });
}

pub(crate) fn left(op: &'static str) {
    ON.with(|on| {
        if let Some(d) = on.borrow_mut().as_mut()
            && let Some(at) = d.entered.take()
        {
            drain();
            let ns = u64::try_from(at.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let row = d.ops.entry(op).or_default();
            row.0 += ns;
            row.1 += 1;
        }
    });
}
