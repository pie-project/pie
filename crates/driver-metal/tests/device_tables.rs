//! The ordinal-keyed argument table cache, against a real device.
//!
//! The claim worth testing is not that a `BTreeMap` works. It is that a table
//! built once and looked up per step drives a dispatch to the right buffers,
//! and that the three ways a graph walk can be wrong are refused rather than
//! silently run.

#![allow(clippy::print_stdout)]

use driver_metal::{Compiler, Context, Error, Heap, MAX_BINDINGS, Stepper, Tables};

const HEAP_BYTES: u64 = 1 << 20;
const COUNT: usize = 256;

/// Writes `tag` into every element of the buffer at binding 0.
const FILL: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void fill(device uint* out [[buffer(0)]],
                 constant uint& tag [[buffer(1)]],
                 uint gid [[thread_position_in_grid]]) {
    out[gid] = tag;
}
";

struct Fixture {
    context: Context,
    compiler: Compiler,
    heap: Heap,
}

fn fixture() -> Option<Fixture> {
    let context = match Context::new() {
        Ok(c) => c,
        Err(Error::NoDevice) => return None,
        Err(e) => panic!("context: {e}"),
    };
    let compiler = Compiler::new(&context).expect("compiler");
    let heap = Heap::new(&context, HEAP_BYTES).expect("heap");
    Some(Fixture {
        context,
        compiler,
        heap,
    })
}

/// Read `count` u32s out of a shared-storage slot.
fn read_u32s(contents: std::ptr::NonNull<std::ffi::c_void>, count: usize) -> Vec<u32> {
    // SAFETY: the slot is shared storage, at least `count` u32s wide, and the
    // GPU step that wrote it has signalled.
    unsafe { std::slice::from_raw_parts(contents.as_ptr().cast::<u32>(), count) }.to_vec()
}

#[test]
fn two_ordinals_reach_two_different_buffers() {
    let Some(f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let mut heap = f.heap;
    let pipeline = f.compiler.compile(&f.context, FILL, "fill").expect("fill");

    let bytes = (COUNT * 4) as u64;
    let left = heap.alloc(&f.context, bytes, 256).expect("left");
    let left_addr = left.gpu_address();
    let left_contents = left.contents();
    let right = heap.alloc(&f.context, bytes, 256).expect("right");
    let right_addr = right.gpu_address();
    let right_contents = right.contents();
    let tags = heap.alloc(&f.context, 8, 256).expect("tags");
    let tags_addr = tags.gpu_address();
    // SAFETY: shared storage, two u32s wide, no step is running.
    unsafe {
        tags.contents().as_ptr().cast::<u32>().write(0xA1A1_A1A1);
        tags.contents()
            .as_ptr()
            .cast::<u32>()
            .add(1)
            .write(0xB2B2_B2B2);
    }

    // Built ONCE, before any step runs. This is the whole point of the cache.
    let mut tables = Tables::new();
    tables
        .bind_address(&f.context, 7, 0, left_addr)
        .expect("bind 7.0");
    tables
        .bind_address(&f.context, 7, 1, tags_addr)
        .expect("bind 7.1");
    tables
        .bind_address(&f.context, 9, 0, right_addr)
        .expect("bind 9.0");
    tables
        .bind_address(&f.context, 9, 1, tags_addr + 4)
        .expect("bind 9.1");
    assert_eq!(tables.len(), 2);
    assert_eq!(tables.ordinals().collect::<Vec<_>>(), vec![7, 9]);

    let mut stepper = Stepper::new(&f.context).expect("stepper");
    stepper
        .run(|step| {
            step.set_pipeline(&pipeline);
            for ordinal in [7u32, 9] {
                step.set_argument_table_for(&tables, ordinal)?;
                step.dispatch([COUNT, 1, 1], [64, 1, 1])?;
            }
            Ok(())
        })
        .expect("the step ran");

    assert_eq!(
        read_u32s(left_contents, COUNT),
        vec![0xA1A1_A1A1; COUNT],
        "ordinal 7 did not write the buffer it was bound to"
    );
    assert_eq!(
        read_u32s(right_contents, COUNT),
        vec![0xB2B2_B2B2; COUNT],
        "ordinal 9 did not write the buffer it was bound to"
    );
}

#[test]
fn the_same_tables_drive_every_later_step() {
    let Some(f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let mut heap = f.heap;
    let pipeline = f.compiler.compile(&f.context, FILL, "fill").expect("fill");

    let out = heap
        .alloc(&f.context, (COUNT * 4) as u64, 256)
        .expect("out");
    let out_addr = out.gpu_address();
    let out_contents = out.contents();
    let tag = heap.alloc(&f.context, 4, 256).expect("tag");
    let tag_addr = tag.gpu_address();
    let tag_ptr = tag.contents().as_ptr().cast::<u32>();

    let mut tables = Tables::new();
    tables
        .bind_address(&f.context, 0, 0, out_addr)
        .expect("0.0");
    tables
        .bind_address(&f.context, 0, 1, tag_addr)
        .expect("0.1");
    let after_setup = tables.len();

    let mut stepper = Stepper::new(&f.context).expect("stepper");
    for round in 1u32..=4 {
        // SAFETY: shared storage, one u32 wide, the previous step signalled.
        unsafe { tag_ptr.write(round) };
        stepper
            .run(|step| {
                step.set_pipeline(&pipeline);
                step.set_argument_table_for(&tables, 0)?;
                step.dispatch([COUNT, 1, 1], [64, 1, 1])
            })
            .expect("the step ran");
        assert_eq!(
            read_u32s(out_contents, COUNT),
            vec![round; COUNT],
            "round {round} did not land"
        );
    }

    assert_eq!(
        tables.len(),
        after_setup,
        "four steps allocated a table between them; the encode path is not flat"
    );
    assert_eq!(stepper.steps(), 4);
}

#[test]
fn an_ordinal_nobody_bound_is_refused_rather_than_inherited() {
    let Some(f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let mut heap = f.heap;
    let pipeline = f.compiler.compile(&f.context, FILL, "fill").expect("fill");
    let out = heap
        .alloc(&f.context, (COUNT * 4) as u64, 256)
        .expect("out");
    let out_addr = out.gpu_address();
    let tag = heap.alloc(&f.context, 4, 256).expect("tag");
    let tag_addr = tag.gpu_address();

    let mut tables = Tables::new();
    tables
        .bind_address(&f.context, 0, 0, out_addr)
        .expect("0.0");
    tables
        .bind_address(&f.context, 0, 1, tag_addr)
        .expect("0.1");

    let mut stepper = Stepper::new(&f.context).expect("stepper");
    let outcome = stepper.run(|step| {
        step.set_pipeline(&pipeline);
        step.set_argument_table_for(&tables, 0)?;
        step.dispatch([COUNT, 1, 1], [64, 1, 1])?;
        // Ordinal 1 was never built. Metal would leave ordinal 0's table
        // bound and run this dispatch over ITS buffers, reporting success.
        step.set_argument_table_for(&tables, 1)?;
        step.dispatch([COUNT, 1, 1], [64, 1, 1])
    });

    let message = outcome
        .expect_err("the missing ordinal was accepted")
        .to_string();
    assert!(
        message.contains("ordinal 1"),
        "the error does not name the ordinal: {message}"
    );
    assert!(
        !stepper.is_wedged(),
        "a refused encode must not wedge the context"
    );
}

#[test]
fn a_binding_past_the_msl_limit_is_refused() {
    let Some(f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let mut tables = Tables::new();
    let outcome = tables.bind_address(&f.context, 3, MAX_BINDINGS, 0x1000);
    let message = outcome
        .expect_err("a binding past the limit was accepted")
        .to_string();
    assert!(
        message.contains("buffer(0..30)"),
        "the error does not name the MSL limit: {message}"
    );
    assert!(
        tables.is_empty(),
        "a refused binding created a table anyway"
    );
}

#[test]
fn the_cache_knows_which_entries_were_written() {
    let Some(f) = fixture() else {
        println!("no Metal device; skipped");
        return;
    };
    let mut tables = Tables::new();
    assert!(!tables.is_bound(4, 0));

    tables.bind_address(&f.context, 4, 0, 0x1000).expect("4.0");
    tables.bind_address(&f.context, 4, 2, 0x2000).expect("4.2");

    assert!(tables.is_bound(4, 0));
    assert!(
        !tables.is_bound(4, 1),
        "an entry nobody wrote reads as bound"
    );
    assert!(tables.is_bound(4, 2));
    assert_eq!(tables.binding_count(4), 2);
    assert_eq!(tables.binding_count(5), 0);
    assert!(!tables.is_bound(4, MAX_BINDINGS));

    // Rebinding the same entry is not a second binding.
    tables.bind_address(&f.context, 4, 0, 0x3000).expect("4.0");
    assert_eq!(tables.binding_count(4), 2);

    assert!(tables.forget(4));
    assert!(!tables.forget(4));
    assert!(
        !tables.is_bound(4, 0),
        "a forgotten ordinal kept its bindings, so a rebuilt graph inherits them"
    );
    assert!(tables.get(4).is_none());
}

#[test]
fn a_binding_reports_the_address_it_was_given_not_just_that_it_happened() {
    let Ok(context) = Context::new() else {
        println!("no Metal device; skipped");
        return;
    };
    let mut heap = Heap::new(&context, 1 << 20).expect("heap");
    let first = heap
        .alloc(&context, 4096, 256)
        .expect("first")
        .gpu_address();
    let second = heap
        .alloc(&context, 4096, 256)
        .expect("second")
        .gpu_address();
    assert_ne!(first, second);

    let mut tables = Tables::new();
    tables.bind_address(&context, 7, 0, first).expect("bind");
    tables.bind_address(&context, 42, 0, second).expect("bind");

    // The failure this exists to catch: both ordinals report bound, and the
    // one that was supposed to share the first buffer is on another one.
    assert_eq!(tables.address(7, 0), Some(first));
    assert_eq!(tables.address(42, 0), Some(second));
    assert!(tables.is_bound(7, 0) && tables.is_bound(42, 0));

    // Never written is not the same answer as written-with-zero.
    assert_eq!(tables.address(7, 1), None);
    tables.bind_address(&context, 7, 1, 0).expect("bind zero");
    assert_eq!(tables.address(7, 1), Some(0));

    // Rebinding replaces rather than accumulating. The two values here are
    // deliberately not neighbours: two consecutive heap allocations differ in
    // one low bit, so an implementation that OR-ed the new address into the
    // old would still report the right answer for them and only go wrong
    // once the addresses were far apart.
    tables
        .bind_address(&context, 9, 0, 0x0000_FF00_0000_0000)
        .expect("bind");
    tables
        .bind_address(&context, 9, 0, 0x0000_00FF_0000_0000)
        .expect("rebind");
    assert_eq!(
        tables.address(9, 0),
        Some(0x0000_00FF_0000_0000),
        "a rebind was merged with what was there instead of replacing it"
    );

    // And forgetting an ordinal forgets its addresses too, so a later
    // coverage check cannot pass on a table that no longer exists.
    assert!(tables.forget(7));
    assert_eq!(tables.address(7, 0), None);
}
