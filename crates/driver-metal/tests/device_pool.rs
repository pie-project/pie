//! The transient pool, against a real device.
//!
//! Two claims worth proving on hardware rather than in a map: a pooled buffer
//! is one the GPU can actually write through, and a REUSED buffer is still
//! one the GPU can write through -- residency survives the round trip.

#![allow(clippy::print_stdout)]

use driver_metal::device::SMALLEST_CLASS;
use driver_metal::Error;
use driver_metal::device::{Context, Pool, Stepper, Tables};
use driver_metal::program::Compiler;

const FILL: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void fill(device uint* out [[buffer(0)]],
                 constant uint& tag [[buffer(1)]],
                 uint gid [[thread_position_in_grid]]) {
    out[gid] = tag + gid;
}
";

fn context() -> Option<Context> {
    match Context::new() {
        Ok(c) => Some(c),
        Err(Error::NoDevice) => None,
        Err(e) => panic!("context: {e}"),
    }
}

fn read_u32s(t: &driver_metal::device::Transient, count: usize) -> Vec<u32> {
    // SAFETY: shared storage, wide enough, and the step that wrote it has
    // signalled.
    unsafe { std::slice::from_raw_parts(t.contents().as_ptr().cast::<u32>(), count) }.to_vec()
}

#[test]
fn a_pooled_buffer_is_one_the_gpu_can_write() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let compiler = Compiler::new(&context).expect("compiler");
    let pipeline = compiler.compile(&context, FILL, "fill").expect("fill");
    let pool = Pool::new(1 << 20);

    let count = 256usize;
    let out = pool.acquire(&context, (count * 4) as u64).expect("out");
    let tag = pool.acquire(&context, 4).expect("tag");
    // SAFETY: shared storage, one u32 wide, nothing is running.
    unsafe { tag.contents().as_ptr().cast::<u32>().write(1000) };

    let mut tables = Tables::new();
    tables
        .bind_address(&context, 0, 0, out.gpu_address())
        .expect("0.0");
    tables
        .bind_address(&context, 0, 1, tag.gpu_address())
        .expect("0.1");

    let mut stepper = Stepper::new(&context).expect("stepper");
    stepper
        .run(|step| {
            step.set_pipeline(&pipeline);
            step.set_argument_table_for(&tables, 0)?;
            step.dispatch([count, 1, 1], [64, 1, 1])
        })
        .expect("the step ran");

    let want: Vec<u32> = (0..count as u32).map(|i| 1000 + i).collect();
    assert_eq!(
        read_u32s(&out, count),
        want,
        "the GPU did not write through a pooled buffer; residency did not take"
    );
}

#[test]
fn a_recycled_buffer_comes_back_and_still_works() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let compiler = Compiler::new(&context).expect("compiler");
    let pipeline = compiler.compile(&context, FILL, "fill").expect("fill");
    let pool = Pool::new(1 << 20);
    let count = 256usize;
    let bytes = (count * 4) as u64;

    let first_address = {
        let out = pool.acquire(&context, bytes).expect("out");
        out.gpu_address()
    };
    let after_drop = pool.stats();
    assert_eq!(after_drop.recycles, 1, "the drop did not return the buffer");
    assert_eq!(after_drop.cached_buffers, 1);
    assert_eq!(after_drop.outstanding_buffers(), 0);

    let out = pool.acquire(&context, bytes).expect("out again");
    assert_eq!(
        out.gpu_address(),
        first_address,
        "the second acquisition got a different buffer, so nothing was reused"
    );
    let stats = pool.stats();
    assert_eq!(stats.reuse_hits, 1);
    assert_eq!(stats.allocations, 1, "the pool allocated twice");

    let tag = pool.acquire(&context, 4).expect("tag");
    // SAFETY: shared storage, one u32 wide, nothing is running.
    unsafe { tag.contents().as_ptr().cast::<u32>().write(77) };
    let mut tables = Tables::new();
    tables
        .bind_address(&context, 0, 0, out.gpu_address())
        .expect("0.0");
    tables
        .bind_address(&context, 0, 1, tag.gpu_address())
        .expect("0.1");
    let mut stepper = Stepper::new(&context).expect("stepper");
    stepper
        .run(|step| {
            step.set_pipeline(&pipeline);
            step.set_argument_table_for(&tables, 0)?;
            step.dispatch([count, 1, 1], [64, 1, 1])
        })
        .expect("the step ran");

    let want: Vec<u32> = (0..count as u32).map(|i| 77 + i).collect();
    assert_eq!(
        read_u32s(&out, count),
        want,
        "a REUSED buffer is no longer resident; the round trip lost it"
    );
}

#[test]
fn a_pass_that_repeats_allocates_once_and_reuses_after() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let pool = Pool::new(1 << 22);
    for _ in 0..8 {
        // Every "pass" asks for the same shapes and lets them go at the end,
        // which is the access pattern the pool exists for.
        let _scratch = pool.acquire(&context, 300).expect("scratch");
        let _indices = pool.acquire(&context, 400).expect("indices");
        let _reduce = pool.acquire(&context, 1 << 14).expect("reduce");
    }
    let stats = pool.stats();
    // 300 and 400 share the 512 class, so two of the three live there and the
    // first pass allocates two buffers, not three.
    assert_eq!(
        stats.allocations, 3,
        "eight passes allocated {} buffers; the cache is not being hit",
        stats.allocations
    );
    assert_eq!(stats.reuse_hits, 21);
    assert_eq!(stats.evictions, 0);
    assert_eq!(stats.outstanding_buffers(), 0);
    println!("{stats:?}");
}

#[test]
fn the_budget_evicts_the_largest_cached_class_first() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    // Room for the three below (1792 bytes) but not for them plus a 2048.
    let pool = Pool::new(SMALLEST_CLASS * 12);
    drop(pool.acquire(&context, SMALLEST_CLASS).expect("small"));
    drop(pool.acquire(&context, SMALLEST_CLASS * 2).expect("medium"));
    drop(pool.acquire(&context, SMALLEST_CLASS * 4).expect("large"));
    let before = pool.stats();
    assert_eq!(before.resident_bytes, SMALLEST_CLASS * 7);
    assert_eq!(before.cached_buffers, 3);

    // Its own class is 8x, which nothing is cached in, so room has to be made:
    // 1792 + 2048 is past the 3072 budget.
    let _fresh = pool.acquire(&context, SMALLEST_CLASS * 8).expect("fresh");
    let after = pool.stats();
    assert_eq!(
        after.evictions, 1,
        "more than the largest class was released"
    );
    assert!(
        after.resident_bytes <= after.capacity_bytes,
        "{} bytes resident against a {} budget",
        after.resident_bytes,
        after.capacity_bytes
    );
    // The 4x class went; the 1x and 2x stayed.
    assert!(
        pool.acquire(&context, SMALLEST_CLASS).is_ok(),
        "the smallest class was evicted instead of the largest"
    );
    assert_eq!(
        pool.stats().reuse_hits,
        1,
        "the smallest class was not the one kept"
    );
}

#[test]
fn an_eviction_takes_the_buffer_out_of_the_residency_set() {
    use objc2_metal::MTLResidencySet;

    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    // The pool's own counters cannot see this. A residency set holds its own
    // reference to every allocation it names, so a buffer that is merely
    // dropped is still resident and still costs its bytes -- `evictions`
    // would climb while nothing was actually given back. Only the device's
    // count answers the question, so this test asks the device.
    let resident = || context.residency().allocationCount();

    let pool = Pool::new(SMALLEST_CLASS * 12);
    let before = resident();
    drop(pool.acquire(&context, SMALLEST_CLASS).expect("small"));
    drop(pool.acquire(&context, SMALLEST_CLASS * 2).expect("medium"));
    drop(pool.acquire(&context, SMALLEST_CLASS * 4).expect("large"));
    assert_eq!(
        resident(),
        before + 3,
        "the pool made three buffers; the set should name three more"
    );

    // Room has to be made for this one, so exactly one buffer is released.
    let _fresh = pool.acquire(&context, SMALLEST_CLASS * 8).expect("fresh");
    assert_eq!(pool.stats().evictions, 1, "the setup stopped evicting one");
    assert_eq!(
        resident(),
        before + 3,
        "one buffer was added and one evicted, so the set should be level; \
         if it grew, the evicted buffer was dropped without being removed \
         from the set, which does not free it"
    );
}

#[test]
fn a_pool_that_goes_away_takes_its_buffers_out_of_the_residency_set() {
    use objc2_metal::MTLResidencySet;

    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    // The set outlives every pool that registers in it, so dropping a pool is
    // the one moment where a buffer can be forgotten with nobody left to
    // notice. Two shapes of that: buffers still cached, and a loan still out.
    let resident = || context.residency().allocationCount();
    let before = resident();

    let pool = Pool::new(SMALLEST_CLASS * 64);
    drop(pool.acquire(&context, SMALLEST_CLASS).expect("cached"));
    let outstanding = pool.acquire(&context, SMALLEST_CLASS * 2).expect("loan");
    assert_eq!(resident(), before + 2, "two buffers were made");

    drop(pool);
    assert_eq!(
        resident(),
        before + 1,
        "the pool was dropped holding one cached buffer, which it should have \
         unregistered; only the outstanding loan should still be named"
    );

    drop(outstanding);
    assert_eq!(
        resident(),
        before,
        "a loan that outlives its pool has nowhere to go home to, so it has \
         to unregister itself rather than just drop"
    );
}

#[test]
fn a_request_past_the_budget_is_refused_rather_than_allocated() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let pool = Pool::new(SMALLEST_CLASS * 2);
    let message = pool
        .acquire(&context, 1 << 20)
        .expect_err("a request past the whole budget was served")
        .to_string();
    assert!(
        message.contains("past the pool's"),
        "the error does not say why: {message}"
    );

    let _held = pool.acquire(&context, SMALLEST_CLASS * 2).expect("held");
    let second = pool
        .acquire(&context, SMALLEST_CLASS)
        .expect_err("the budget was exceeded by an outstanding buffer");
    assert!(second.to_string().contains("in a caller's hands"));
    let stats = pool.stats();
    assert_eq!(stats.refusals, 2);
    assert_eq!(stats.allocations, 1, "a refused request allocated anyway");
}

#[test]
fn a_zero_byte_request_is_refused() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let pool = Pool::new(1 << 20);
    let message = pool
        .acquire(&context, 0)
        .expect_err("a zero-byte buffer was handed out")
        .to_string();
    assert!(message.contains("no bytes"), "{message}");
}

#[test]
fn draining_keeps_what_is_outstanding() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let pool = Pool::new(1 << 20);
    let held = pool.acquire(&context, 1024).expect("held");
    drop(pool.acquire(&context, 1024).expect("released"));
    assert_eq!(pool.stats().cached_buffers, 1);

    pool.drain();
    let stats = pool.stats();
    assert_eq!(stats.cached_buffers, 0);
    assert_eq!(stats.cached_bytes, 0);
    assert_eq!(
        stats.outstanding_buffers(),
        1,
        "drain took a buffer a caller is still holding"
    );

    // And the outstanding one is still usable, and still returns home.
    assert_eq!(held.len(), 1024);
    drop(held);
    assert_eq!(pool.stats().cached_buffers, 1);
}

#[test]
fn a_buffer_outliving_its_pool_does_not_fault() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let pool = Pool::new(1 << 20);
    let held = pool.acquire(&context, 4096).expect("held");
    let address = held.gpu_address();
    drop(pool);
    // The pool is gone; the loan still owns its buffer, and dropping it must
    // not reach through a dangling weak reference.
    assert_eq!(held.gpu_address(), address);
    // SAFETY: shared storage the loan still owns.
    unsafe { held.contents().as_ptr().cast::<u32>().write(0x1234) };
    drop(held);
}
