//! Growing and shrinking a buffer whose address must not move.
//!
//! The claim the whole module exists for is one line long: the GPU address is
//! the same before and after memory is attached and detached. Everything else
//! -- the budget, the pressure clamp, the heaps -- is bookkeeping around it.
//! So that is asserted first, and asserted while a kernel is actually reading
//! through the address, because an address that is merely reported unchanged
//! and no longer works is worse than one that moved.

#![allow(clippy::print_stdout)]

use driver_metal_new::{
    Arena, Context, Elastic, Error, Need, Pressure, Stepper, TILE, Tables, create_elastic,
    pages_for_bytes,
};

/// Writes a known value through the elastic buffer, so a mapping that is not
/// really there faults rather than passing.
const TOUCH: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void touch(device uint* kv [[buffer(0)]],
                  uint gid [[thread_position_in_grid]]) {
    kv[gid] = gid + 7u;
}
";

fn context() -> Option<Context> {
    match Context::new() {
        Ok(c) => Some(c),
        Err(Error::NoDevice) => None,
        Err(e) => panic!("context: {e}"),
    }
}

/// Big enough to need more than one chunk would be 256 MiB; these tests stay
/// small so they can run on a loaded machine, and the multi-chunk path is
/// covered by arithmetic rather than by allocating a gigabyte.
const VIRTUAL: u64 = 8 * 1024 * 1024;

#[test]
fn the_address_survives_growing_and_shrinking() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let arena = Arena::new(64 * 1024 * 1024, 0);
    let mut buffer = create_elastic(&context, &arena, VIRTUAL).expect("sparse buffer");
    let address = buffer.gpu_address();
    assert_ne!(address, 0, "a sparse buffer with no address is not usable");
    assert_eq!(
        buffer.committed(),
        0,
        "creating a sparse buffer must cost address space, not memory"
    );

    let mut stepper = Stepper::new(&context).expect("stepper");
    stepper
        .ensure(&mut buffer, 4 * 1024 * 1024, Pressure::Normal, Need::Step)
        .expect("grow");
    assert_eq!(
        buffer.gpu_address(),
        address,
        "the address moved when memory was attached, which invalidates every \
         argument table and constant block that recorded it"
    );
    assert_eq!(buffer.committed(), 4 * 1024 * 1024);

    stepper.trim(&mut buffer, 0).expect("trim");
    assert_eq!(
        buffer.gpu_address(),
        address,
        "the address moved when memory was detached"
    );
    assert_eq!(buffer.committed(), 0);
}

#[test]
fn a_kernel_writes_through_the_address_after_it_grows() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let arena = Arena::new(64 * 1024 * 1024, 0);
    let mut buffer = create_elastic(&context, &arena, VIRTUAL).expect("sparse buffer");
    let address = buffer.gpu_address();

    // The table is bound to the address BEFORE anything is mapped, which is
    // the situation the type exists for: a binding recorded once and not
    // revisited as the buffer grows.
    let mut tables = Tables::new();
    tables.bind_address(&context, 0, 0, address).expect("bind");

    let compiler = driver_metal_new::Compiler::new(&context).expect("compiler");
    let pipeline = compiler.compile(&context, TOUCH, "touch").expect("touch");

    let mut stepper = Stepper::new(&context).expect("stepper");
    const THREADS: usize = 1024;
    stepper
        .ensure(
            &mut buffer,
            (THREADS * 4) as u64,
            Pressure::Normal,
            Need::Step,
        )
        .expect("grow");

    stepper
        .run(|step| {
            step.set_argument_table_for(&tables, 0)?;
            step.set_pipeline(&pipeline);
            step.dispatch([THREADS, 1, 1], [256, 1, 1])
        })
        .expect("a dispatch through a mapped sparse buffer");

    // Growing again while the binding stands. A second chunk of tiles goes on
    // and the same table must still name the same memory.
    stepper
        .ensure(&mut buffer, 2 * 1024 * 1024, Pressure::Normal, Need::Step)
        .expect("grow again");
    assert_eq!(buffer.gpu_address(), address);
    stepper
        .run(|step| {
            step.set_argument_table_for(&tables, 0)?;
            step.set_pipeline(&pipeline);
            step.dispatch([THREADS, 1, 1], [256, 1, 1])
        })
        .expect("a dispatch after growth, through the unchanged binding");
}

#[test]
fn asking_for_less_than_is_mapped_costs_nothing() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let arena = Arena::new(64 * 1024 * 1024, 0);
    let mut buffer = create_elastic(&context, &arena, VIRTUAL).expect("sparse buffer");
    let mut stepper = Stepper::new(&context).expect("stepper");

    stepper
        .ensure(&mut buffer, 1024 * 1024, Pressure::Normal, Need::Step)
        .expect("grow");
    let steps = stepper.steps();

    // The reason this must be free: a caller asks on every step rather than
    // tracking what it last asked for, and a remap per step is a full queue
    // round-trip per token.
    for _ in 0..8 {
        stepper
            .ensure(&mut buffer, 512 * 1024, Pressure::Normal, Need::Step)
            .expect("no-op");
    }
    assert_eq!(
        stepper.steps(),
        steps,
        "an ask below what is already mapped advanced the timeline, so it \
         issued a remap that was not needed"
    );
    assert_eq!(buffer.committed(), 1024 * 1024);
}

#[test]
fn growth_is_refused_past_the_budget_and_the_buffer_is_untouched() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    // Room for two megabytes, and nothing more.
    let arena = Arena::new(2 * 1024 * 1024, 0);
    let mut buffer = create_elastic(&context, &arena, VIRTUAL).expect("sparse buffer");
    let mut stepper = Stepper::new(&context).expect("stepper");

    stepper
        .ensure(&mut buffer, 2 * 1024 * 1024, Pressure::Normal, Need::Step)
        .expect("the whole budget");
    assert_eq!(arena.budget().reserved, 2 * 1024 * 1024);

    let refused = stepper.ensure(&mut buffer, 3 * 1024 * 1024, Pressure::Normal, Need::Step);
    assert!(
        refused.is_err(),
        "the arena handed out more than it has, so the budget is not a budget"
    );
    assert_eq!(
        buffer.committed(),
        2 * 1024 * 1024,
        "a refused growth left tiles mapped, so the refusal cost memory"
    );
    assert_eq!(
        arena.budget().reserved,
        2 * 1024 * 1024,
        "a refused growth left bytes charged, which shrinks the arena on \
         every failure until nothing fits"
    );
}

#[test]
fn critical_pressure_refuses_growth_and_still_serves_a_step() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let arena = Arena::new(16 * 1024 * 1024, 0);
    let mut buffer = create_elastic(&context, &arena, VIRTUAL).expect("sparse buffer");
    let mut stepper = Stepper::new(&context).expect("stepper");

    let growth = stepper.ensure(&mut buffer, 1024 * 1024, Pressure::Critical, Need::Growth);
    assert!(
        growth.is_err(),
        "critical pressure with a zero floor must stop speculative growth"
    );

    // The same ask, declared as a step requirement, must succeed -- this is
    // the model that loaded and then could not take a step.
    stepper
        .ensure(&mut buffer, 1024 * 1024, Pressure::Critical, Need::Step)
        .expect(
            "critical pressure refused a step requirement, which turns an \
             admitted model into an unusable one without handing back a page",
        );
    assert_eq!(buffer.committed(), 1024 * 1024);
}

#[test]
fn a_batch_that_does_not_fit_is_refused_before_anything_is_mapped() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    // Two megabytes, and the batch below asks for three.
    let arena = Arena::new(2 * 1024 * 1024, 0);
    let mut first = create_elastic(&context, &arena, VIRTUAL).expect("first");
    let mut second = create_elastic(&context, &arena, VIRTUAL).expect("second");
    let mut stepper = Stepper::new(&context).expect("stepper");

    let refused = {
        let mut targets: [(&mut Elastic, u64); 2] =
            [(&mut first, 1024 * 1024), (&mut second, 2 * 1024 * 1024)];
        stepper.ensure_all(&mut targets, Pressure::Normal, Need::Step)
    };
    assert!(
        refused.is_err(),
        "three megabytes fit in a two-megabyte arena"
    );
    // The batch is priced before the first tile goes on, so the refusal costs
    // no GPU work at all -- there is nothing to unwind.
    assert_eq!(
        (first.committed(), second.committed()),
        (0, 0),
        "a refused batch left one buffer grown, so the caller is holding \
         memory it cannot use and did not ask to keep"
    );

    // And a batch that does fit grows both.
    {
        let mut targets: [(&mut Elastic, u64); 2] =
            [(&mut first, 1024 * 1024), (&mut second, 1024 * 1024)];
        stepper
            .ensure_all(&mut targets, Pressure::Normal, Need::Step)
            .expect("two megabytes fit");
    }
    assert_eq!(first.committed(), 1024 * 1024);
    assert_eq!(second.committed(), 1024 * 1024);
}

#[test]
fn a_trimmed_heap_is_given_back_and_the_budget_notices() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let arena = Arena::new(64 * 1024 * 1024, 0);
    let mut buffer = create_elastic(&context, &arena, VIRTUAL).expect("sparse buffer");
    let mut stepper = Stepper::new(&context).expect("stepper");

    stepper
        .ensure(&mut buffer, 4 * 1024 * 1024, Pressure::Normal, Need::Step)
        .expect("grow");
    assert_eq!(arena.budget().committed, 4 * 1024 * 1024);
    assert_eq!(pages_for_bytes(arena.budget().committed), 2);

    stepper.trim(&mut buffer, 0).expect("trim");
    assert_eq!(
        arena.budget().committed,
        0,
        "the bytes were unmapped but still counted, so the arena refuses the \
         next allocation forever"
    );
    assert_eq!(arena.budget().reserved, 0);
    // `trim` waits for its own unmap, so the heap is collectable by the time
    // it returns and nothing should still be pending.
    assert_eq!(
        arena.pending(),
        0,
        "a heap stayed pending after the unmap it was waiting for completed"
    );
}

#[test]
fn dropping_a_buffer_gives_its_bytes_back_to_the_arena() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let arena = Arena::new(8 * 1024 * 1024, 0);
    let mut stepper = Stepper::new(&context).expect("stepper");
    {
        let mut buffer = create_elastic(&context, &arena, VIRTUAL).expect("sparse buffer");
        stepper
            .ensure(&mut buffer, 4 * 1024 * 1024, Pressure::Normal, Need::Step)
            .expect("grow");
        stepper.declare_mandatory(&mut buffer, 4 * 1024 * 1024);
        assert_eq!(arena.budget().mandatory, 4 * 1024 * 1024);
    }
    let budget = arena.budget();
    assert_eq!(
        (budget.reserved, budget.committed, budget.mandatory),
        (0, 0, 0),
        "a dropped buffer is still charged, so an arena leaks its budget one \
         buffer at a time: {budget:?}"
    );

    // And the freed budget is really usable again.
    let mut again = create_elastic(&context, &arena, VIRTUAL).expect("second buffer");
    stepper
        .ensure(&mut again, 8 * 1024 * 1024, Pressure::Normal, Need::Step)
        .expect("the whole budget is free again");
}

#[test]
fn a_zero_length_buffer_is_refused_rather_than_returned_empty() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let arena = Arena::new(1 << 20, 0);
    assert!(
        create_elastic(&context, &arena, 0).is_err(),
        "a zero-length sparse buffer has no address, which is the only thing \
         this type promises"
    );
}

#[test]
fn a_length_that_is_not_a_whole_tile_is_rounded_up_not_down() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let arena = Arena::new(1 << 20, 0);
    let mut buffer = create_elastic(&context, &arena, TILE + 1).expect("odd length");
    assert_eq!(buffer.len(), TILE + 1, "len is what the caller asked for");

    let mut stepper = Stepper::new(&context).expect("stepper");
    stepper
        .ensure(&mut buffer, TILE + 1, Pressure::Normal, Need::Step)
        .expect("grow");
    assert_eq!(
        buffer.committed(),
        2 * TILE,
        "rounding down would leave the last byte of the caller's range \
         unmapped, which faults on access instead of failing at the ask"
    );

    assert!(
        stepper
            .ensure(&mut buffer, TILE * 4, Pressure::Normal, Need::Step)
            .is_err(),
        "asking past the length must be refused rather than served out of \
         the rounding slack"
    );
}

#[test]
fn a_batch_across_two_arenas_is_refused_rather_than_priced_against_one() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    // Each arena has room for the ask made of its own buffer, so anything
    // that reports a failure here is reporting the batch rule and not a
    // shortage.
    let left = Arena::new(4 * 1024 * 1024, 0);
    let right = Arena::new(4 * 1024 * 1024, 0);
    let mut first = create_elastic(&context, &left, VIRTUAL).expect("first");
    let mut second = create_elastic(&context, &right, VIRTUAL).expect("second");
    let mut stepper = Stepper::new(&context).expect("stepper");

    let refused = {
        let mut targets: [(&mut Elastic, u64); 2] = [
            (&mut first, 2 * 1024 * 1024),
            (&mut second, 2 * 1024 * 1024),
        ];
        stepper.ensure_all(&mut targets, Pressure::Normal, Need::Step)
    };
    assert!(
        refused.is_err(),
        "a batch spanning two arenas was priced against one of them, so the \
         other one was never consulted and can be overdrawn without noticing"
    );
    assert_eq!((first.committed(), second.committed()), (0, 0));

    // Each on its own still works, which is the honest way to do it.
    stepper
        .ensure(&mut first, 2 * 1024 * 1024, Pressure::Normal, Need::Step)
        .expect("its own arena has room");
    stepper
        .ensure(&mut second, 2 * 1024 * 1024, Pressure::Normal, Need::Step)
        .expect("its own arena has room");
}

/// A buffer destroyed while its growth is still queued.
///
/// This asserts nothing, and that is not an oversight. The failure it exists
/// for is not a wrong value: `ensure` deliberately does not wait for the map
/// it issues, so at the moment the buffer goes out of scope an
/// `updateBufferMappings` naming its placement heaps can still be queued. A
/// destructor that released those heaps there handed the GPU a mapping into
/// memory that had gone back to the system -- a page fault raised inside
/// AGX, which panics the kernel. The whole machine goes down, so there is no
/// process left to fail an assertion in.
///
/// Three kernel panics on an M1 Max came from exactly this shape, in this
/// file, before the destructor waited. Every step of the loop grows and drops
/// without a step in between, because a step is what would have waited by
/// accident and hidden it.
#[test]
fn a_buffer_dropped_with_its_growth_in_flight_does_not_take_the_gpu_with_it() {
    let Some(context) = context() else {
        println!("no Metal device; skipped");
        return;
    };
    let arena = Arena::new(64 * 1024 * 1024, 0);

    for _ in 0..32 {
        let mut stepper = Stepper::new(&context).expect("stepper");
        let mut buffer = create_elastic(&context, &arena, VIRTUAL).expect("sparse buffer");
        stepper
            .ensure(&mut buffer, 4 * 1024 * 1024, Pressure::Normal, Need::Step)
            .expect("grow");
        // No step, no trim, no wait: straight out of scope with the mapping
        // still on the queue.
        drop(buffer);
    }

    assert_eq!(
        arena.budget().reserved,
        0,
        "every buffer was dropped, so the arena is still counting bytes that \
         belong to nothing and will refuse the next model that fits"
    );
}
