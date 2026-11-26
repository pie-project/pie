#![cfg(feature = "device")]
//! Nothing on the serving path allocates: once a thread's load serves, a
//! device buffer asked on it is refused by its caller's name, never made,
//! and the same thread laying a load again makes its own.

use engine_cuda::device::{Buffer, Context, alloc};

#[test]
fn a_buffer_asked_after_ready_is_refused_by_name() {
    if !engine_cuda::device::present() {
        eprintln!("no device: skipping");
        return;
    }
    let _context = Context::bind(0, None).expect("a device context");
    let free = engine_cuda::device::free_bytes();
    alloc::serving_from_here();
    let refused = Buffer::zeroed(1 << 20).expect_err("a buffer made while serving");
    let why = refused.to_string();
    assert!(why.contains("after the load was ready"), "{why}");
    assert!(why.contains(file!()), "the refusal names its caller: {why}");
    assert_eq!(engine_cuda::device::free_bytes(), free, "the driver was asked");
    alloc::loading_from_here();
    assert!(Buffer::zeroed(1 << 20).is_ok(), "a load lays its own");
}
