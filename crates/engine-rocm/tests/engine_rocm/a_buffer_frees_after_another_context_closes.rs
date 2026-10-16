//! A buffer freed after another device context of the process has closed:
//! the recycler keeps it behind an event on a live context's compute stream
//! (or frees it outright once none is live), never on the closed context's
//! destroyed stream (a multi-tenant server opens and closes contexts as
//! tenants come and go).

use engine_rocm::device::{Buffer, Context};

#[test]
fn a_buffer_frees_after_another_context_closes() {
    if !engine_rocm::device::present() {
        eprintln!("no device: skipping");
        return;
    }
    let first = Context::bind(0, None).expect("a device context");
    for _ in 0..4 {
        let second = Context::bind(0, None).expect("a second device context");
        let kept = Buffer::zeroed(1 << 20).expect("a buffer");
        drop(second);
        // Freed now: behind the first context's stream, not the second's.
        drop(kept);
        drop(Buffer::zeroed(1 << 20).expect("a buffer"));
    }
    first.synchronize().expect("the first context's stream drains");
    drop(first);
    // No context is live: the buffer is freed outright.
    drop(Buffer::zeroed(1 << 20).expect("a buffer"));
}
