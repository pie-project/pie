use std::cell::Cell;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::device::ctx::Frame;
use crate::error::Result;

static DIR: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
static SEQ: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static FRAME: Cell<*const Frame> = const { Cell::new(std::ptr::null()) };
}

fn slot() -> &'static Mutex<Option<PathBuf>> {
    DIR.get_or_init(|| Mutex::new(None))
}

pub fn dump_to(dir: impl Into<PathBuf>) {
    let dir = dir.into();
    let _ = std::fs::create_dir_all(&dir);
    *slot()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(dir);
}

pub fn stop() {
    *slot()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
}

pub(crate) fn dir() -> Option<PathBuf> {
    slot()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

pub(crate) fn next_seq() -> u64 {
    SEQ.fetch_add(1, Ordering::Relaxed)
}

#[cfg(feature = "vulkan")]
pub(crate) fn set_frame(frame: *const Frame) {
    FRAME.with(|f| f.set(frame));
}

pub(crate) fn with_frame<R>(f: impl FnOnce(&Frame) -> R) -> Option<R> {
    let frame = FRAME.with(Cell::get);
    if frame.is_null() {
        return None;
    }

    Some(f(unsafe { &*frame }))
}

pub(crate) fn flush() -> Result<()> {
    let frame = FRAME.with(Cell::get);
    if frame.is_null() {
        return Ok(());
    }

    unsafe { (*frame).flush() }
}
