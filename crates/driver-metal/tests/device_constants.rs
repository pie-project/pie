//! Constant slots: the placement a rebind must not repeat.
//!
//! The bump allocator never takes anything back, which is right for weights
//! and wrong for a constant rewritten every fire. The failure it produces is
//! the reason this cache exists and the reason it is worth a test: nothing
//! goes wrong at the allocation that spends the heap. The model runs, fires,
//! and then fails to set up some LATER sequence with a budget complaint,
//! thousands of fires from the cause.
//!
//! So the test is about the heap's used-bytes, not about correctness of the
//! bytes -- rebinding the same constant a hundred times must cost what
//! binding it once costs.

#![allow(clippy::print_stdout)]

use driver_metal::{Error, Region};
use driver_metal::device::{Context, Heap, Stepper, Tables};
use driver_metal::program::Compiler;

const SPREAD: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void spread(device uint* out [[buffer(0)]],
                   const device uint* src [[buffer(1)]],
                   uint gid [[thread_position_in_grid]]) {
    out[gid] = src[0];
}
";

fn context() -> Option<Context> {
    match Context::new() {
        Ok(c) => Some(c),
        Err(Error::NoDevice) => None,
        Err(e) => panic!("context: {e}"),
    }
}

#[test]
fn rebinding_a_constant_a_hundred_times_costs_one_placement() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let mut heap = Heap::new(&context, 1 << 20).expect("heap");

    let first_address = heap
        .constant(&context, 3, 1, 16)
        .expect("constant")
        .gpu_address();
    let after_first = heap.used();
    assert_eq!(heap.constant_count(), 1);

    for _ in 0..100 {
        let slot = heap.constant(&context, 3, 1, 16).expect("constant");
        assert_eq!(
            slot.gpu_address(),
            first_address,
            "a rebind moved the constant, so the kernel's address is stale"
        );
    }
    assert_eq!(
        heap.used(),
        after_first,
        "a hundred rebinds walked the heap; this is the leak the cache exists to stop"
    );
    assert_eq!(heap.constant_count(), 1);

    // The key is the pair, not either half. Nothing here may collide.
    let mut seen = vec![first_address];
    for (ordinal, index) in [(3u32, 0u8), (0, 1), (4, 1), (3, 2), (0x0001_0000, 1)] {
        let address = heap
            .constant(&context, ordinal, index, 16)
            .expect("constant")
            .gpu_address();
        assert!(
            !seen.contains(&address),
            "({ordinal}, {index}) collided with a slot already placed"
        );
        seen.push(address);
    }
    assert_eq!(heap.constant_count(), 6);
}

/// A smaller later request reuses the placement but is bounded by what it
/// asked for; a larger one is a different constant and gets its own.
#[test]
fn a_size_that_grows_is_a_different_constant() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let mut heap = Heap::new(&context, 1 << 20).expect("heap");

    let big = heap.constant(&context, 7, 0, 4096).expect("big");
    let big_address = big.gpu_address();
    let after_big = heap.used();

    {
        let small = heap.constant(&context, 7, 0, 64).expect("small");
        assert_eq!(
            small.gpu_address(),
            big_address,
            "a smaller fit reallocated"
        );
        assert_eq!(
            small.len(),
            64,
            "the slot reported the first request, so a 64-byte constant could \
             write four kilobytes"
        );
        assert!(unsafe { small.zero(0, 65) }.is_err());
    }
    assert_eq!(heap.used(), after_big);
    assert_eq!(heap.constant_count(), 1);

    let grown = heap.constant(&context, 7, 0, 8192).expect("grown");
    assert_ne!(
        grown.gpu_address(),
        big_address,
        "a constant that outgrew its slot was handed one too small to hold it"
    );
    assert!(heap.used() > after_big);
    assert_eq!(heap.constant_count(), 1, "the cache now names the new one");
}

/// The address a constant slot hands back has to be one the GPU can read
/// after it is rewritten between steps -- which is the entire premise.
#[test]
fn a_constant_rewritten_between_steps_is_read_at_its_new_value() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let compiler = Compiler::new(&context).expect("compiler");
    let pipeline = compiler
        .compile(&context, SPREAD, "spread")
        .expect("spread");
    let mut heap = Heap::new(&context, 1 << 20).expect("heap");

    let out = heap.alloc(&context, 16, 256).expect("out");
    let out_address = out.gpu_address();
    let out_contents = out.contents();

    let mut tables = Tables::new();
    let mut stepper = Stepper::new(&context).expect("stepper");

    for round in 1..=4u32 {
        let tag = heap.constant(&context, 11, 0, 4).expect("tag");
        // SAFETY: the previous step has signalled and the next is not
        // encoded, which is the boundary that makes rewriting a constant safe.
        unsafe { tag.write(0, &round.to_le_bytes()) }.expect("write");
        tables
            .bind_address(&context, 0, 0, out_address)
            .expect("out");
        tables
            .bind_address(&context, 0, 1, tag.gpu_address())
            .expect("tag");

        stepper
            .run(|step| {
                step.set_pipeline(&pipeline);
                step.set_argument_table_for(&tables, 0)?;
                step.dispatch([4, 1, 1], [4, 1, 1])
            })
            .expect("the step ran");

        // SAFETY: shared storage, four u32s, and the step has signalled.
        let got = unsafe { std::slice::from_raw_parts(out_contents.as_ptr().cast::<u32>(), 4) };
        assert_eq!(got, [round; 4], "round {round} read a stale constant");
    }

    assert_eq!(heap.constant_count(), 1);
}
