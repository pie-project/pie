//! The pools, against a real device.
//!
//! Two claims worth proving on hardware rather than in a map: a pooled buffer
//! is one the GPU can actually write through, and a REUSED buffer is still
//! one the GPU can write through -- residency survives the round trip.
//!
//! The transient pool is most of the file; the recurrent seats and the PAGED
//! KV pool are here too, because all three answer the same question about the
//! same device. The last two arrived from `device_text_fire.rs`, which was
//! deleted with the by-name walk that its other seven tests fired -- these two
//! never named it.

#![allow(clippy::print_stdout)]

use driver_metal::Error;
use driver_metal::device::SMALLEST_CLASS;
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
        Err(Error::NoDevice) => {
            driver_metal::skip::skipped("no Metal 4 device, so no page was pooled");
            None
        }
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
        driver_metal::skip::skipped("no Metal device");
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
        driver_metal::skip::skipped("no Metal device");
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
        driver_metal::skip::skipped("no Metal device");
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
        driver_metal::skip::skipped("no Metal device");
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
        driver_metal::skip::skipped("no Metal device");
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
        driver_metal::skip::skipped("no Metal device");
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
        driver_metal::skip::skipped("no Metal device");
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
        driver_metal::skip::skipped("no Metal device");
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
        driver_metal::skip::skipped("no Metal device");
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
        driver_metal::skip::skipped("no Metal device");
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

/// **A fork takes the compressed history with it.**
///
/// `Shell::copy_state` refused unconditionally, under prose saying "no model
/// this backend serves has any recurrent state to move ... whose rows this
/// build has no Metal text for". That sentence was true when it was written
/// and stopped being true the day the qwen3.5 forward path landed: both
/// Qwen3.6 checkpoints load, allocate one of these pools, and generate.
///
/// What the refusal cost is a FORK. A branch of a conversation takes its
/// attention prefix through `copy_kv`, and for a linear-attention layer the
/// whole prefix is compressed into a seat -- so a branch that could not copy
/// the seat would attend over the right pages with a history that never saw
/// the prompt. Not a refusal the guest could act on either: the seam answered
/// `Unserved` and named a hole that had been filled.
///
/// Asserted over BOTH conv planes on purpose. `carry_forward` copies the
/// write plane over the read one after the next fire, so a copy that moved
/// only the read plane would be undone one step later -- which is the same
/// argument `clear_slot` makes for zeroing both, and the reason that argument
/// is worth a test is that the undo happens a fire after the mistake.
#[test]
fn a_forked_seat_carries_both_conv_planes_and_the_memory() {
    let Some(context) = context() else {
        driver_metal::skip::skipped("no Metal device");
        return;
    };
    use driver_metal::layout::region::Region as _;
    // Small, and shaped like a gated-DeltaNet layer rather than round: a
    // square slot would hide an offset that multiplied the wrong extent.
    let shape = driver_metal::layout::recurrent::Shape {
        linear_layers: 2,
        conv_dim: 8,
        conv_k: 3,
        v_heads: 2,
        v_dim: 4,
        k_dim: 6,
        slots: 3,
    };
    let pool = driver_metal::pools::recurrent::Pool::allocate(&context, shape).expect("a pool");

    let conv_slot = shape.conv_bytes_per_slot();
    let state_slot = shape.state_bytes_per_slot();
    let mark = |plane: &driver_metal::device::Allocation, offset: u64, bytes: u64, tag: u8| {
        let pattern: Vec<u8> = (0..bytes).map(|i| tag.wrapping_add(i as u8)).collect();
        // SAFETY: nothing has been encoded against this pool.
        unsafe { plane.write(offset, &pattern) }.expect("the pattern fits");
        pattern
    };
    let read = |plane: &driver_metal::device::Allocation, offset: u64, bytes: u64| -> Vec<u8> {
        // SAFETY: shared storage, and no fire exists.
        unsafe {
            std::slice::from_raw_parts(
                plane.contents().cast::<u8>().as_ptr().add(offset as usize),
                bytes as usize,
            )
        }
        .to_vec()
    };

    // Seat 2 is the one being forked FROM, and it is the last seat: a copy
    // that read from the plane's base instead of the slot's offset would
    // pass against seat 0 and fail here.
    let mut want = Vec::new();
    for l in 0..shape.linear_layers {
        let layer = pool.layer(l).expect("a layer");
        want.push((
            mark(&layer.conv, shape.conv_offset(2), conv_slot, 0x10 + l as u8),
            mark(
                &layer.new_conv,
                shape.conv_offset(2),
                conv_slot,
                0x40 + l as u8,
            ),
            mark(
                &layer.state,
                shape.state_offset(2),
                state_slot,
                0x70 + l as u8,
            ),
        ));
    }

    // SAFETY: no fire is reading either seat -- none has ever been encoded.
    unsafe { pool.copy_slot(2, 0) }.expect("the seat forks");

    for l in 0..shape.linear_layers {
        let layer = pool.layer(l).expect("a layer");
        let (conv, new_conv, state) = &want[l as usize];
        assert_eq!(
            &read(&layer.conv, shape.conv_offset(0), conv_slot),
            conv,
            "layer {l}: the read conv plane did not travel"
        );
        assert_eq!(
            &read(&layer.new_conv, shape.conv_offset(0), conv_slot),
            new_conv,
            "layer {l}: the WRITE conv plane did not travel, so `carry_forward` \
             would undo this copy one fire later"
        );
        assert_eq!(
            &read(&layer.state, shape.state_offset(0), state_slot),
            state,
            "layer {l}: the DeltaNet memory did not travel"
        );
        // The seat nobody named is still the initial condition, which is what
        // makes the copy a copy rather than a broadcast.
        assert!(
            read(&layer.conv, shape.conv_offset(1), conv_slot)
                .iter()
                .all(|&b| b == 0),
            "layer {l}: seat 1 was written by a copy that did not name it"
        );
    }
}

/// A seat outside the pool is refused rather than written past the end.
#[test]
fn a_seat_the_pool_does_not_have_is_refused() {
    let Some(context) = context() else {
        driver_metal::skip::skipped("no Metal device");
        return;
    };
    let shape = driver_metal::layout::recurrent::Shape {
        linear_layers: 1,
        conv_dim: 4,
        conv_k: 2,
        v_heads: 1,
        v_dim: 2,
        k_dim: 2,
        slots: 2,
    };
    let pool = driver_metal::pools::recurrent::Pool::allocate(&context, shape).expect("a pool");
    // SAFETY: no fire exists.
    let err = unsafe { pool.copy_slot(0, 2) }.expect_err("seat 2 is past the pool");
    assert!(
        format!("{err}").contains('2'),
        "the refusal does not name the seat: {err}"
    );
}

/// The paged KV pool, allocated at the fire's geometry.
///
/// `metal::stage_decode_storage` has allocated `KvSlots` since the port, but
/// sized from `batch::DecodeGeometry` — a model definition inside the driver.
/// This is the same allocation with its arguments taken from the frame.
#[test]
fn the_kv_pool_allocates_at_the_geometry_the_fire_states() {
    use driver_metal::layout::kv::Shape;
    use driver_metal::pools::kv::{Pool, translate};

    let Some(context) = context() else {
        return;
    };
    let shape = Shape {
        layers: 24,
        // The two numbers `lowering::dispatch::Geometry` used to carry here.
        // Written out because there is no such struct: what a fire is planned
        // over is a `Program`, and what a pool is laid out from is
        // `model::deployment::Deployment::attention` (see
        // `layout::kv::Shape::periodic`).
        kv_heads: 8,
        head_dim: 128,
        page_size: 16,
        pages: 64,
        element_bytes: 2,
        global_head_dim: 0,
        global_kv_heads: 0,
        full_attn_every: 0,
    };
    let pool = Pool::allocate(&context, shape).expect("the pool allocates");

    assert_eq!(pool.pages(), 64);
    assert_eq!(
        pool.bytes(),
        shape.layer_bytes_at(0) * 2 * 24,
        "a K and a V region for every layer"
    );
    let layer = pool.layer(0).expect("layer 0 has pages");
    assert_ne!(
        layer.k.gpu_address(),
        layer.v.gpu_address(),
        "K and V must be distinct regions; one address would make the append \
         to K overwrite V"
    );
    assert!(
        pool.layer(24).is_none(),
        "past the last layer there is none"
    );

    // And the frame's translation reads against it.
    let table = [0u32, 1, 63];
    assert_eq!(
        translate(&pool, &table, &[0, 3], 0).expect("a lane's pages"),
        &[0, 1, 63]
    );
    assert!(
        translate(&pool, &[64], &[0, 1], 0).is_err(),
        "a page past the pool addresses another layer's memory"
    );
}

/// A KV move, run on the pool, checked byte for byte.
///
/// The pages are `StorageModeShared`, so a move is a `memmove` and needs no
/// encoder — and the memmove semantics are not incidental: a compaction slides
/// rows toward the front, so source and destination overlap.
#[test]
fn a_move_plan_slides_rows_without_smearing_them() {
    use driver_metal::layout::kv::Shape;
    use driver_metal::layout::{CellCopy, CellMovePlan};
    use driver_metal::pools::kv::Pool;

    let Some(context) = context() else {
        return;
    };
    // One layer, one head, tiny pages: the arithmetic is the subject, not the
    // size.
    let shape = Shape {
        layers: 1,
        kv_heads: 1,
        head_dim: 4,
        page_size: 2,
        pages: 4,
        element_bytes: 2,
        global_head_dim: 0,
        global_kv_heads: 0,
        full_attn_every: 0,
    };
    let pool = Pool::allocate(&context, shape).expect("the pool allocates");
    let layer = pool.layer(0).expect("layer 0");
    let row = shape.row_bytes().expect("a uniform pool") as usize;

    // Each row is its own byte, so a misplaced one names itself.
    let total = shape.layer_bytes_at(0) as usize;
    let src: Vec<u8> = (0..total).map(|i| (i / row) as u8).collect();
    layer.k.write(0, &src).expect("the pattern fits");
    layer.v.write(0, &src).expect("and into v");

    // Slide page 1 onto page 0 — the overlapping case a compaction makes.
    let page = shape.page_bytes().expect("a uniform pool");
    pool.apply(&CellMovePlan {
        copies: vec![CellCopy {
            src_off: page,
            dst_off: 0,
            bytes: page,
        }],
        pages_touched: 2,
    })
    .expect("the move runs");

    let read = |p: &driver_metal::pools::kv::Pages| -> Vec<u8> {
        let at = p
            .host_span(0, total as u64)
            .expect("the pages are addressable");
        unsafe { std::slice::from_raw_parts(at.as_ptr().cast_const(), total) }.to_vec()
    };
    for (name, got) in [("k", read(&layer.k)), ("v", read(&layer.v))] {
        // Page 0 now holds what page 1 held: rows 2 and 3.
        assert_eq!(got[0], 2, "{name}: page 0 row 0 came from page 1 row 0");
        assert_eq!(got[row], 3, "{name}: page 0 row 1 came from page 1 row 1");
        // And page 1 is untouched — a smear would have overwritten it.
        assert_eq!(got[2 * row], 2, "{name}: page 1 row 0 still its own");
        assert_eq!(got[3 * row], 3, "{name}: page 1 row 1 still its own");
        // Pages 2 and 3 were never named.
        assert_eq!(got[4 * row], 4, "{name}: page 2 untouched");
    }
}
