//! Growing and shrinking a buffer whose address must not move.
//!
//! The claim the whole module exists for is one line long: the GPU address is
//! the same before and after memory is attached and detached. Everything else
//! -- the budget, the pressure clamp, the heaps -- is bookkeeping around it.
//! So that is asserted first, and asserted while a kernel is actually reading
//! through the address, because an address that is merely reported unchanged
//! and no longer works is worse than one that moved.

#![allow(clippy::print_stdout)]

use driver_metal::Error;
use driver_metal::device::{
    Arena, Context, Elastic, Need, Pressure, Stepper, TILE, Tables, create_elastic, pages_for_bytes,
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
        Err(Error::NoDevice) => {
            driver_metal::skip::skipped("no Metal 4 device, so no heap grew or shrank");
            None
        }
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
        driver_metal::skip::skipped("no Metal device");
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
        driver_metal::skip::skipped("no Metal device");
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

    let compiler = driver_metal::program::Compiler::new(&context).expect("compiler");
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

/// Writes a pattern and reads it back, so growth can be asked whether the
/// bytes that were already there are still there.
///
/// Two entries rather than one because the sparse buffer is private storage:
/// the host cannot read it, so a kernel has to copy it somewhere shared.
const SURVIVE: &str = r"
#include <metal_stdlib>
using namespace metal;
kernel void fill(device uint* kv [[buffer(0)]],
                 uint gid [[thread_position_in_grid]]) {
    kv[gid] = gid * 2654435761u + 1u;
}
kernel void readback(device const uint* kv [[buffer(0)]],
                     device uint* out [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    out[gid] = kv[gid];
}
";

fn read_u32s(t: &driver_metal::device::Transient, count: usize) -> Vec<u32> {
    // SAFETY: shared storage, wide enough, and the step that wrote it has
    // signalled -- `Stepper::run` waits.
    unsafe { std::slice::from_raw_parts(t.contents().as_ptr().cast::<u32>(), count) }.to_vec()
}

#[test]
fn what_was_written_before_a_growth_is_still_there_after_it() {
    let Some(context) = context() else {
        driver_metal::skip::skipped("no Metal device");
        return;
    };
    // Every other test here asks whether the ADDRESS survives. An address
    // that still resolves while the bytes under it were remapped to fresh
    // heap pages would pass all of them and lose a KV cache. Growth attaches
    // tiles by editing the buffer's page table, so whether the tiles already
    // attached keep their memory is a real question about
    // `updateBufferMappings`, and nothing was asking it.
    let arena = Arena::new(64 * 1024 * 1024, 0);
    let mut buffer = create_elastic(&context, &arena, VIRTUAL).expect("sparse buffer");
    let address = buffer.gpu_address();

    let compiler = driver_metal::program::Compiler::new(&context).expect("compiler");
    let fill = compiler.compile(&context, SURVIVE, "fill").expect("fill");
    let readback = compiler
        .compile(&context, SURVIVE, "readback")
        .expect("readback");

    // Four tiles' worth, so the pattern spans more than the one tile a
    // minimal mapping would attach.
    const WORDS: usize = 16 * 1024;
    const BYTES: u64 = (WORDS * 4) as u64;
    const { assert!(BYTES > TILE, "the pattern has to span more than one tile") };

    let mut tables = Tables::new();
    tables.bind_address(&context, 0, 0, address).expect("kv");

    let mut stepper = Stepper::new(&context).expect("stepper");
    stepper
        .ensure(&mut buffer, BYTES, Pressure::Normal, Need::Step)
        .expect("grow to hold the pattern");
    stepper
        .run(|step| {
            step.set_argument_table_for(&tables, 0)?;
            step.set_pipeline(&fill);
            step.dispatch([WORDS, 1, 1], [256, 1, 1])
        })
        .expect("write the pattern");

    // The growth under test: more tiles on top of the ones holding the
    // pattern.
    stepper
        .ensure(&mut buffer, 2 * 1024 * 1024, Pressure::Normal, Need::Step)
        .expect("grow past the pattern");
    assert_eq!(buffer.gpu_address(), address, "the address moved");

    let pool = driver_metal::device::Pool::new(4 * 1024 * 1024);
    let out = pool.acquire(&context, BYTES).expect("readback buffer");
    tables
        .bind_address(&context, 0, 1, out.gpu_address())
        .expect("out");
    stepper
        .run(|step| {
            step.set_argument_table_for(&tables, 0)?;
            step.set_pipeline(&readback);
            step.dispatch([WORDS, 1, 1], [256, 1, 1])
        })
        .expect("read the pattern back");

    let got = read_u32s(&out, WORDS);
    for (i, &word) in got.iter().enumerate() {
        let want = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(1);
        assert_eq!(
            word, want,
            "word {i} of {WORDS} changed across a growth: the tiles holding it \
             were remapped, so growing an elastic buffer loses what is in it"
        );
    }
}

#[test]
fn the_host_alias_and_the_gpu_address_name_the_same_memory() {
    let Some(context) = context() else {
        driver_metal::skip::skipped("no Metal device");
        return;
    };
    // The assumption the whole design rests on, and nothing was asking it.
    // An elastic buffer has TWO names for its bytes: a private sparse buffer
    // that kernels bind, and the Shared placement heaps underneath it, which
    // `make_chunk` chooses Shared so that "the host can stage into a KV page
    // without a second copy". If those two ever named different memory, the
    // host would stage into bytes no kernel reads -- silently, since both
    // halves would look like they worked.
    let arena = Arena::new(64 * 1024 * 1024, 0);
    let mut buffer = create_elastic(&context, &arena, VIRTUAL).expect("sparse buffer");
    let address = buffer.gpu_address();

    let compiler = driver_metal::program::Compiler::new(&context).expect("compiler");
    let fill = compiler.compile(&context, SURVIVE, "fill").expect("fill");
    let readback = compiler
        .compile(&context, SURVIVE, "readback")
        .expect("readback");

    const WORDS: usize = 4 * 1024;
    const BYTES: u64 = (WORDS * 4) as u64;

    let mut stepper = Stepper::new(&context).expect("stepper");
    stepper
        .ensure(&mut buffer, BYTES, Pressure::Normal, Need::Step)
        .expect("grow");

    let mut tables = Tables::new();
    tables.bind_address(&context, 0, 0, address).expect("kv");
    let pool = driver_metal::device::Pool::new(4 * 1024 * 1024);
    let out = pool.acquire(&context, BYTES).expect("readback buffer");
    tables
        .bind_address(&context, 0, 1, out.gpu_address())
        .expect("out");

    // ── Host writes, GPU reads. ──
    let staged = buffer.host_span(0, BYTES).expect("a host address");
    for i in 0..WORDS {
        // SAFETY: the span is `BYTES` long and `i * 4` stays inside it.
        unsafe {
            staged
                .as_ptr()
                .cast::<u32>()
                .add(i)
                .write(0xF00D_0000 + i as u32)
        };
    }
    stepper
        .run(|step| {
            step.set_argument_table_for(&tables, 0)?;
            step.set_pipeline(&readback);
            step.dispatch([WORDS, 1, 1], [256, 1, 1])
        })
        .expect("read what the host staged");
    let seen = read_u32s(&out, WORDS);
    for (i, &word) in seen.iter().enumerate() {
        assert_eq!(
            word,
            0xF00D_0000 + i as u32,
            "word {i}: the GPU read something other than what the host staged \
             through the heap alias, so the two names are not the same memory"
        );
    }

    // ── GPU writes, host reads. ──
    stepper
        .run(|step| {
            step.set_argument_table_for(&tables, 0)?;
            step.set_pipeline(&fill);
            step.dispatch([WORDS, 1, 1], [256, 1, 1])
        })
        .expect("let the GPU write");
    let staged = buffer.host_span(0, BYTES).expect("a host address");
    for i in 0..WORDS {
        // SAFETY: as above, and the step that wrote it has signalled.
        let word = unsafe { staged.as_ptr().cast::<u32>().add(i).read() };
        let want = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(1);
        assert_eq!(
            word, want,
            "word {i}: the host read something other than what the GPU wrote"
        );
    }
}

#[test]
fn a_host_span_past_what_is_mapped_is_refused_rather_than_returned() {
    let Some(context) = context() else {
        driver_metal::skip::skipped("no Metal device");
        return;
    };
    // Address space is not memory. A pointer into unmapped space would fault
    // on first touch, a long way from the mistake, so it is refused here.
    let arena = Arena::new(64 * 1024 * 1024, 0);
    let mut buffer = create_elastic(&context, &arena, VIRTUAL).expect("sparse buffer");
    assert!(
        buffer.host_span(0, TILE).is_err(),
        "a buffer with nothing mapped handed out an address anyway"
    );

    let mut stepper = Stepper::new(&context).expect("stepper");
    stepper
        .ensure(&mut buffer, TILE, Pressure::Normal, Need::Step)
        .expect("grow");
    assert!(buffer.host_span(0, TILE).is_ok(), "a mapped tile has bytes");
    assert!(
        buffer.host_span(0, TILE + 1).is_err(),
        "one byte past the mapping is still past it"
    );
    assert!(
        buffer.host_span(0, 0).is_err(),
        "a span of no bytes has no address"
    );
}

#[test]
fn asking_for_less_than_is_mapped_costs_nothing() {
    let Some(context) = context() else {
        driver_metal::skip::skipped("no Metal device");
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
        driver_metal::skip::skipped("no Metal device");
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
        driver_metal::skip::skipped("no Metal device");
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
        driver_metal::skip::skipped("no Metal device");
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
        driver_metal::skip::skipped("no Metal device");
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
        driver_metal::skip::skipped("no Metal device");
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
        driver_metal::skip::skipped("no Metal device");
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
        driver_metal::skip::skipped("no Metal device");
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
        driver_metal::skip::skipped("no Metal device");
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
        driver_metal::skip::skipped("no Metal device");
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

#[test]
fn a_host_move_over_the_pages_is_what_the_gpu_reads_afterwards() {
    let Some(context) = context() else {
        driver_metal::skip::skipped("no Metal device");
        return;
    };
    // What `kv::Pool::apply` needs from an elastic-backed pool: a compaction
    // is a memmove over the pages, done on the host between fires, and the
    // next fire has to read the rows where the move put them. `host_span`
    // proved the two names are the same memory; this proves the operations
    // built on it -- `zero` and `copy_within` -- move the bytes the GPU
    // actually binds, rather than a host-side copy of them.
    let arena = Arena::new(64 * 1024 * 1024, 0);
    let mut buffer = create_elastic(&context, &arena, VIRTUAL).expect("sparse buffer");
    let address = buffer.gpu_address();

    let compiler = driver_metal::program::Compiler::new(&context).expect("compiler");
    let readback = compiler
        .compile(&context, SURVIVE, "readback")
        .expect("readback");

    const WORDS: usize = 4 * 1024;
    const BYTES: u64 = (WORDS * 4) as u64;
    /// Rows slide toward the front of the pool, which is the direction a
    /// compaction moves them, and far enough that source and destination
    /// share bytes.
    const ROW: u64 = 1024;

    let mut stepper = Stepper::new(&context).expect("stepper");
    stepper
        .ensure(&mut buffer, BYTES, Pressure::Normal, Need::Step)
        .expect("grow");

    let mut tables = Tables::new();
    tables.bind_address(&context, 0, 0, address).expect("kv");
    let pool = driver_metal::device::Pool::new(4 * 1024 * 1024);
    let out = pool.acquire(&context, BYTES).expect("readback buffer");
    tables
        .bind_address(&context, 0, 1, out.gpu_address())
        .expect("out");

    let staged = buffer.host_span(0, BYTES).expect("a host address");
    for i in 0..WORDS {
        // SAFETY: the span is `BYTES` long and `i * 4` stays inside it.
        unsafe { staged.as_ptr().cast::<u32>().add(i).write(i as u32) };
    }

    // Slide everything from ROW onward down by one row, overlapping, the way
    // a pool closes a gap left by a released page.
    // SAFETY: nothing is encoded against the buffer -- `ensure` is the only
    // thing that has touched it, and `run` has not been called yet.
    unsafe { buffer.copy_within(0, ROW, BYTES - ROW) }.expect("slide the rows down");
    // And clear the row the slide vacated, as a pool does with a page it is
    // handing out fresh.
    // SAFETY: as above.
    unsafe { buffer.zero(BYTES - ROW, ROW) }.expect("clear the tail");

    stepper
        .run(|step| {
            step.set_argument_table_for(&tables, 0)?;
            step.set_pipeline(&readback);
            step.dispatch([WORDS, 1, 1], [256, 1, 1])
        })
        .expect("read what the host moved");

    let seen = read_u32s(&out, WORDS);
    let moved = usize::try_from(ROW / 4).expect("a row of words");
    for (i, &word) in seen.iter().enumerate() {
        let want = if i < WORDS - moved {
            u32::try_from(i + moved).expect("a word index")
        } else {
            0
        };
        assert_eq!(
            word, want,
            "word {i}: the GPU is not reading what the host's move left \
             behind. A move that ran the wrong way smears its first piece \
             down the span; one that wrote through a copy of the pages \
             leaves the originals here"
        );
    }
}

#[test]
fn a_resized_kv_pool_keeps_every_address_it_handed_out() {
    let Some(context) = context() else {
        driver_metal::skip::skipped("no Metal device");
        return;
    };
    // The claim elastic KV exists for. A pool that gave memory back and then
    // took it again must be the SAME pool from every bound address's point of
    // view -- otherwise every argument table staged before the resize is
    // pointing at nothing, and the driver has no way to know which ones.
    use driver_metal::layout::kv::Shape;
    use driver_metal::pools::kv::Pool;

    let shape = Shape {
        layers: 2,
        kv_heads: 8,
        head_dim: 64,
        page_size: 16,
        pages: 64,
        element_bytes: 2,
        global_head_dim: 0,
        global_kv_heads: 0,
        full_attn_every: 0,
    };
    let arena = Arena::new(256 * 1024 * 1024, 0);
    let mut stepper = Stepper::new(&context).expect("stepper");
    let mut pool = Pool::allocate_elastic(&context, &mut stepper, &arena, shape, &[])
        .expect("an elastic pool");
    assert_eq!(pool.pages(), 64);

    let addresses: Vec<(u64, u64)> = (0..shape.layers)
        .map(|l| {
            let layer = pool.layer(l).expect("a layer");
            (layer.k.gpu_address(), layer.v.gpu_address())
        })
        .collect();

    // A row in the last page, written before anything moves. It is inside the
    // span the shrink gives back, so it is the byte that says whether the
    // pool came back as itself or merely as something the same size.
    let page_bytes = shape.page_bytes_at(0);
    let far = page_bytes * 63;
    let mark: Vec<u8> = (0..page_bytes as usize).map(|i| (i % 251) as u8).collect();
    pool.layer(0)
        .expect("layer 0")
        .k
        .write(far, &mark)
        .expect("the last page is addressable");

    pool.resize(&mut stepper, 32).expect("give half of it back");
    assert_eq!(
        pool.pages(),
        32,
        "a pool that reports pages it no longer holds has the scheduler \
         admitting frames onto memory that is not there"
    );
    assert!(
        pool.layer(0).expect("layer 0").k.write(far, &mark).is_err(),
        "a page the pool gave back is one it must refuse to address; \
         serving it would return zeros rather than fault"
    );

    pool.resize(&mut stepper, 64).expect("take it back");
    assert_eq!(pool.pages(), 64);
    for (l, (k, v)) in addresses.iter().enumerate() {
        let layer = pool.layer(l as u32).expect("a layer");
        assert_eq!(
            layer.k.gpu_address(),
            *k,
            "layer {l}'s keys moved across a resize, so every argument table \
             staged before it now points somewhere else"
        );
        assert_eq!(layer.v.gpu_address(), *v, "layer {l}'s values moved");
    }

    // Regrown pages must be CLEAR, not whatever the heap last held. The mark
    // written before the shrink is exactly the kind of leftover an attention
    // would read as keys.
    let back = pool
        .layer(0)
        .expect("layer 0")
        .k
        .host_span(far, page_bytes)
        .expect("the page is back");
    // SAFETY: the span is `page_bytes` long and nothing is encoded here.
    let seen = unsafe { std::slice::from_raw_parts(back.as_ptr().cast_const(), mark.len()) };
    assert!(
        seen.iter().all(|&b| b == 0),
        "a page that came back holding its old bytes is one a frame attends \
         to as if they were keys"
    );
}

#[test]
fn a_fixed_pool_says_it_cannot_be_resized_rather_than_pretending() {
    let Some(context) = context() else {
        driver_metal::skip::skipped("no Metal device");
        return;
    };
    use driver_metal::layout::kv::Shape;
    use driver_metal::pools::kv::Pool;

    let shape = Shape {
        layers: 1,
        kv_heads: 8,
        head_dim: 64,
        page_size: 16,
        pages: 8,
        element_bytes: 2,
        global_head_dim: 0,
        global_kv_heads: 0,
        full_attn_every: 0,
    };
    let mut pool = Pool::allocate(&context, shape, &[]).expect("a fixed pool");
    let mut stepper = Stepper::new(&context).expect("stepper");
    assert!(
        pool.resize(&mut stepper, 4).is_err(),
        "a fixed pool that accepted a resize would either move its addresses \
         or quietly report a size it did not change to"
    );
    assert_eq!(pool.pages(), 8, "a refused resize must change nothing");

    let arena = Arena::new(64 * 1024 * 1024, 0);
    let mut elastic = Pool::allocate_elastic(&context, &mut stepper, &arena, shape, &[])
        .expect("an elastic pool");
    assert!(
        elastic.resize(&mut stepper, 9).is_err(),
        "growing past the count the pool reserved address space for has \
         nothing to attach memory to, and is past what admission weighed"
    );
    assert_eq!(elastic.pages(), 8, "a refused growth must change nothing");
}

#[test]
fn a_pool_that_gave_memory_back_stops_reserving_it() {
    let Some(context) = context() else {
        driver_metal::skip::skipped("no Metal device");
        return;
    };
    // A shrink that moves the bytes but not the accounting is worse than no
    // shrink at all: the memory is gone AND the arena still holds it against
    // the next allocation, so the machine has less to work with and the
    // budget says it has the same. `declare_mandatory` only ever raises --
    // two callers declaring different floors for one buffer must both be
    // honoured -- so the trim is what has to lower it.
    use driver_metal::layout::kv::Shape;
    use driver_metal::pools::kv::Pool;

    let shape = Shape {
        layers: 2,
        kv_heads: 8,
        head_dim: 64,
        page_size: 16,
        pages: 64,
        element_bytes: 2,
        global_head_dim: 0,
        global_kv_heads: 0,
        full_attn_every: 0,
    };
    let arena = Arena::new(256 * 1024 * 1024, 0);
    let mut stepper = Stepper::new(&context).expect("stepper");
    let mut pool = Pool::allocate_elastic(&context, &mut stepper, &arena, shape, &[])
        .expect("an elastic pool");

    let full = arena.budget();
    assert!(
        full.mandatory > 0,
        "a pool that declared no floor is one pressure may unmap under a \
         bound address"
    );

    pool.resize(&mut stepper, 16)
        .expect("give three quarters back");
    let shrunk = arena.budget();
    assert!(
        shrunk.committed < full.committed,
        "the trim did not release anything, so there is nothing to account for"
    );
    assert!(
        shrunk.mandatory < full.mandatory,
        "the arena still reserves {} bytes for a pool that now holds {}; \
         the next model is admitted against a budget that has already been \
         spent on memory nobody has",
        shrunk.mandatory,
        shrunk.committed
    );
    assert!(
        shrunk.mandatory <= shrunk.committed,
        "a floor above what is committed names memory nobody holds"
    );
}
