//! Device time per op, for finding where a step goes: with `PIE_OP_PROFILE`
//! set, each fire issued on a stream that is not capturing is bracketed by
//! two events, and every few seconds the landed spans are summed by op and
//! printed, largest first. Captured fires (graph bodies) are not seen, so a
//! profile wants the load's bodies off (`[engine] bodies_mem = "0MiB"`).

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use crate::hip::rt;

pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("PIE_OP_PROFILE").is_some())
}

struct Span {
    label: String,
    start: rt::Event,
    stop: rt::Event,
}

#[derive(Default)]
struct Book {
    open: Vec<Span>,
    sums: HashMap<String, (u64, f64)>,
    since: Option<Instant>,
}

// SAFETY: the book is only touched under its mutex; events are process-wide.
unsafe impl Send for Book {}

static BOOK: Mutex<Option<Book>> = Mutex::new(None);

/// A start event on `stream`, unless it is capturing.
pub fn begin(stream: *mut core::ffi::c_void) -> Option<rt::Event> {
    let mut status = 0;
    // SAFETY: queries a live stream.
    if unsafe { rt::hipStreamIsCapturing(stream.cast(), &raw mut status) } != 0 || status != 0 {
        return None;
    }
    let mut event: rt::Event = core::ptr::null_mut();
    // SAFETY: creates and records a fresh event on a live stream.
    unsafe {
        if rt::hipEventCreate(&raw mut event) != 0 {
            return None;
        }
        rt::hipEventRecord(event, stream.cast());
    }
    Some(event)
}

/// Closes the span `begin` opened, under `label`.
pub fn end(
    stream: *mut core::ffi::c_void,
    start: Option<rt::Event>,
    label: impl FnOnce() -> String,
) {
    let Some(start) = start else { return };
    let mut stop: rt::Event = core::ptr::null_mut();
    // SAFETY: creates and records a fresh event on a live stream.
    unsafe {
        if rt::hipEventCreate(&raw mut stop) != 0 {
            rt::hipEventDestroy(start);
            return;
        }
        rt::hipEventRecord(stop, stream.cast());
    }
    let mut book = BOOK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let book = book.get_or_insert_with(Book::default);
    book.open.push(Span {
        label: label(),
        start,
        stop,
    });
    book.since.get_or_insert_with(Instant::now);
    if book.open.len() >= 512 {
        settle(book);
    }
    if book.since.is_some_and(|at| at.elapsed().as_secs() >= 5) {
        settle(book);
        report(book);
    }
}

fn settle(book: &mut Book) {
    let mut kept = Vec::new();
    for span in book.open.drain(..) {
        // SAFETY: the span's events are its own.
        let landed = unsafe { rt::hipEventQuery(span.stop) } == 0;
        if !landed {
            kept.push(span);
            continue;
        }
        let mut ms = 0f32;
        // SAFETY: both events have landed (the stop after the start).
        unsafe {
            rt::hipEventElapsedTime(&raw mut ms, span.start, span.stop);
            rt::hipEventDestroy(span.start);
            rt::hipEventDestroy(span.stop);
        }
        let entry = book.sums.entry(span.label).or_default();
        entry.0 += 1;
        entry.1 += f64::from(ms) * 1e3;
    }
    book.open = kept;
}

fn report(book: &mut Book) {
    let mut rows: Vec<(String, (u64, f64))> = book.sums.drain().collect();
    rows.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
    let total: f64 = rows.iter().map(|(_, (_, us))| us).sum();
    let mut out = format!(
        "[op-profile] {:.1} ms device time over {} ops\n",
        total / 1e3,
        rows.len()
    );
    for (label, (count, us)) in rows.iter().take(40) {
        out.push_str(&format!(
            "[op-profile] {:6.2}% {:10.1} us {:7} x {:8.2} us  {label}\n",
            us / total * 100.0,
            us,
            count,
            us / *count as f64
        ));
    }
    eprint!("{out}");
    book.since = Some(Instant::now());
}
