//! Growing and shrinking a buffer whose address must not move.
//!
//! The claim the whole module exists for is one line long: the GPU address is
//! the same before and after memory is attached and detached. Everything else
//! -- the budget, the heaps, the host alias -- is bookkeeping around it. So
//! that is asserted first, and asserted while a blit is actually writing
//! through the address, because an address that is merely reported unchanged
//! and no longer works is worse than one that moved.
//!
//! Every test here needs a Metal 4 device and skips without one.

#![cfg(target_vendor = "apple")]

use engine_metal::Fault;
use engine_metal::device::elastic::{self, TILE, Target, pages_for_bytes};
use engine_metal::device::{Context, ElasticArena, Sparse};

/// Big enough to need more than one chunk would be 256 MiB; these tests stay
/// small so they can run on a loaded machine, and the multi-chunk path is
/// covered by arithmetic rather than by allocating a gigabyte.
const VIRTUAL: u64 = 8 * 1024 * 1024;

fn sparse() -> Option<(Context, std::sync::Arc<Sparse>)> {
    if !engine_metal::device::present() {
        eprintln!("no Metal device; skipping");
        return None;
    }
    let device = Context::bind().expect("the system device");
    match device
        .sparse()
        .expect("the mapping side opens or is absent")
    {
        Some(sparse) => Some((device, sparse)),
        None => {
            eprintln!("not a Metal 4 device, so no heap grew or shrank; skipping");
            None
        }
    }
}

fn grow(buffer: &mut elastic::Elastic, bytes: u64) -> engine_metal::Result<()> {
    let mut targets = [Target { buffer, bytes }];
    elastic::grow_all(&mut targets).map(|_| ())
}

fn shrink(buffer: &mut elastic::Elastic, bytes: u64) -> engine_metal::Result<()> {
    let mut targets = [Target { buffer, bytes }];
    // SAFETY: nothing is in flight; every frame these tests commit is
    // waited for before the next line.
    unsafe { elastic::shrink_all(&mut targets) }
}

#[test]
fn the_kv_pool_grows_in_place() {
    let Some((device, sparse)) = sparse() else {
        return;
    };
    the_address_survives_growing_and_shrinking(&sparse);
    a_blit_writes_through_the_address_after_it_grows(&device, &sparse);
    what_was_written_before_a_growth_is_still_there_after_it(&sparse);
    a_host_span_past_what_is_mapped_is_refused_rather_than_returned(&sparse);
    asking_for_less_than_is_mapped_costs_nothing(&sparse);
    growth_is_refused_past_the_budget_and_the_buffer_is_untouched(&sparse);
    a_batch_that_does_not_fit_is_refused_before_anything_is_mapped(&sparse);
    a_shrunk_heap_is_given_back_and_the_budget_notices(&sparse);
    dropping_a_buffer_gives_its_bytes_back_to_the_arena(&sparse);
    a_zero_length_buffer_is_refused_rather_than_returned_empty(&sparse);
    a_length_that_is_not_a_whole_tile_is_rounded_up_not_down(&sparse);
    a_batch_across_two_arenas_is_refused_rather_than_priced_against_one(&sparse);
    a_host_move_over_the_pages_lands_where_one_memmove_would(&sparse);
}

fn the_address_survives_growing_and_shrinking(sparse: &std::sync::Arc<Sparse>) {
    let arena = ElasticArena::new(64 * 1024 * 1024);
    let mut buffer = elastic::create(sparse, &arena, VIRTUAL).expect("a sparse buffer");
    let address = buffer.gpu_address();
    assert_ne!(address, 0, "a sparse buffer with no address is not usable");
    assert_eq!(
        buffer.committed(),
        0,
        "creating a sparse buffer must cost address space, not memory"
    );

    grow(&mut buffer, 4 * 1024 * 1024).expect("grow");
    assert_eq!(
        buffer.gpu_address(),
        address,
        "the address moved when memory was attached, which invalidates every \
         argument table and indirect command that recorded it"
    );
    assert_eq!(buffer.committed(), 4 * 1024 * 1024);

    shrink(&mut buffer, 0).expect("shrink");
    assert_eq!(
        buffer.gpu_address(),
        address,
        "the address moved when memory was detached"
    );
    assert_eq!(buffer.committed(), 0);
}

fn a_blit_writes_through_the_address_after_it_grows(
    device: &Context,
    sparse: &std::sync::Arc<Sparse>,
) {
    let arena = ElasticArena::new(64 * 1024 * 1024);
    let mut buffer = elastic::create(sparse, &arena, VIRTUAL).expect("a sparse buffer");
    grow(&mut buffer, 2 * TILE).expect("grow");

    // A known pattern in the second tile, blitted over the first: the blit
    // goes through the sparse buffer's address, the read comes back through
    // the heap's host alias. If the two named different memory, the first
    // tile would still be zero.
    let pattern: Vec<u8> = (0..TILE).map(|at| (at % 253) as u8 + 1).collect();
    // SAFETY: nothing is in flight.
    unsafe { buffer.write_from(TILE, &pattern) }.expect("the host writes tile 1");
    let view = buffer.view();
    let mut frame = device.frame().expect("a frame");
    frame
        .copy_span(&view, TILE, &view, 0, TILE)
        .expect("a blit inside the sparse buffer");
    frame.commit().expect("the blit lands");

    let mut back = vec![0u8; TILE as usize];
    buffer.read_into(0, &mut back).expect("tile 0 reads");
    assert_eq!(
        back, pattern,
        "the GPU wrote through the sparse address and the host alias read it"
    );
}

fn what_was_written_before_a_growth_is_still_there_after_it(sparse: &std::sync::Arc<Sparse>) {
    let arena = ElasticArena::new(64 * 1024 * 1024);
    let mut buffer = elastic::create(sparse, &arena, VIRTUAL).expect("a sparse buffer");
    grow(&mut buffer, TILE).expect("grow");
    let pattern: Vec<u8> = (0..TILE).map(|at| (at % 251) as u8).collect();
    // SAFETY: nothing is in flight.
    unsafe { buffer.write_from(0, &pattern) }.expect("the host writes tile 0");

    grow(&mut buffer, VIRTUAL).expect("grow to the whole buffer");
    let mut back = vec![0u8; TILE as usize];
    buffer.read_into(0, &mut back).expect("tile 0 reads");
    assert_eq!(back, pattern, "a growth must not disturb what was mapped");
    let mut fresh = vec![0xffu8; TILE as usize];
    buffer
        .read_into(VIRTUAL - TILE, &mut fresh)
        .expect("the last tile reads");
    assert!(
        fresh.iter().all(|&b| b == 0),
        "newly mapped tiles are zeroed, whatever the heap held before"
    );
}

fn a_host_span_past_what_is_mapped_is_refused_rather_than_returned(
    sparse: &std::sync::Arc<Sparse>,
) {
    let arena = ElasticArena::new(64 * 1024 * 1024);
    let mut buffer = elastic::create(sparse, &arena, VIRTUAL).expect("a sparse buffer");
    grow(&mut buffer, TILE).expect("grow");
    assert!(buffer.host_span(0, TILE).is_ok());
    let refused = buffer
        .host_span(TILE, 1)
        .expect_err("address space with no memory has nothing to point at");
    assert!(
        matches!(refused, Fault::Ceiling { .. }),
        "past the mapping is a ceiling: {refused:?}"
    );
    assert!(
        buffer.host_span(0, 0).is_err(),
        "a span of no bytes has no address"
    );
}

fn asking_for_less_than_is_mapped_costs_nothing(sparse: &std::sync::Arc<Sparse>) {
    let arena = ElasticArena::new(64 * 1024 * 1024);
    let mut buffer = elastic::create(sparse, &arena, VIRTUAL).expect("a sparse buffer");
    grow(&mut buffer, 4 * TILE).expect("grow");
    let before = arena.budget();
    let mut targets = [Target {
        buffer: &mut buffer,
        bytes: TILE,
    }];
    let grown = elastic::grow_all(&mut targets).expect("no-op");
    assert_eq!(grown, vec![None], "nothing had to change");
    assert_eq!(arena.budget(), before, "and nothing was charged");
}

fn growth_is_refused_past_the_budget_and_the_buffer_is_untouched(sparse: &std::sync::Arc<Sparse>) {
    let arena = ElasticArena::new(2 * TILE);
    let mut buffer = elastic::create(sparse, &arena, VIRTUAL).expect("a sparse buffer");
    grow(&mut buffer, TILE).expect("one tile fits");
    let refused = grow(&mut buffer, 4 * TILE).expect_err("four do not");
    assert!(
        matches!(refused, Fault::Ceiling { .. }),
        "past the budget is a ceiling: {refused:?}"
    );
    assert_eq!(buffer.committed(), TILE, "the refusal mapped nothing");
    assert_eq!(
        arena.budget().reserved,
        TILE,
        "and charged nothing that was not mapped"
    );
    assert!(
        grow(&mut buffer, VIRTUAL + 1).is_err(),
        "past the buffer's own length is refused whatever the budget"
    );
}

fn a_batch_that_does_not_fit_is_refused_before_anything_is_mapped(sparse: &std::sync::Arc<Sparse>) {
    let arena = ElasticArena::new(3 * TILE);
    let mut a = elastic::create(sparse, &arena, VIRTUAL).expect("a");
    let mut b = elastic::create(sparse, &arena, VIRTUAL).expect("b");
    let mut targets = [
        Target {
            buffer: &mut a,
            bytes: 2 * TILE,
        },
        Target {
            buffer: &mut b,
            bytes: 2 * TILE,
        },
    ];
    assert!(
        elastic::grow_all(&mut targets).is_err(),
        "four tiles do not fit in three"
    );
    assert_eq!(a.committed(), 0, "the first buffer was not mapped part-way");
    assert_eq!(b.committed(), 0);
    assert_eq!(arena.budget().reserved, 0, "and nothing stayed charged");
}

fn a_shrunk_heap_is_given_back_and_the_budget_notices(sparse: &std::sync::Arc<Sparse>) {
    let arena = ElasticArena::new(64 * 1024 * 1024);
    let mut buffer = elastic::create(sparse, &arena, VIRTUAL).expect("a sparse buffer");
    grow(&mut buffer, VIRTUAL).expect("grow");
    assert_eq!(arena.budget().committed, VIRTUAL);
    shrink(&mut buffer, 2 * TILE).expect("shrink");
    assert_eq!(buffer.committed(), 2 * TILE);
    let budget = arena.budget();
    assert_eq!(budget.committed, 2 * TILE);
    assert_eq!(budget.reserved, 2 * TILE);
    assert_eq!(budget.high_water, VIRTUAL, "the high water mark stays");
    shrink(&mut buffer, 0).expect("shrink to nothing");
    assert_eq!(
        arena.pending(),
        0,
        "the emptied heap was collected once the unmap landed"
    );
    assert_eq!(pages_for_bytes(arena.budget().committed), 0);
}

fn dropping_a_buffer_gives_its_bytes_back_to_the_arena(sparse: &std::sync::Arc<Sparse>) {
    let arena = ElasticArena::new(64 * 1024 * 1024);
    {
        let mut buffer = elastic::create(sparse, &arena, VIRTUAL).expect("a sparse buffer");
        grow(&mut buffer, VIRTUAL).expect("grow");
        assert_eq!(arena.budget().reserved, VIRTUAL);
    }
    let budget = arena.budget();
    assert_eq!(budget.reserved, 0);
    assert_eq!(budget.committed, 0);
}

fn a_zero_length_buffer_is_refused_rather_than_returned_empty(sparse: &std::sync::Arc<Sparse>) {
    let arena = ElasticArena::new(TILE);
    assert!(
        elastic::create(sparse, &arena, 0).is_err(),
        "a zero-length buffer has no address to promise"
    );
}

fn a_length_that_is_not_a_whole_tile_is_rounded_up_not_down(sparse: &std::sync::Arc<Sparse>) {
    let arena = ElasticArena::new(64 * 1024 * 1024);
    let mut buffer = elastic::create(sparse, &arena, TILE + 1).expect("a sparse buffer");
    assert_eq!(buffer.len(), TILE + 1, "the caller's length is kept");
    grow(&mut buffer, TILE + 1).expect("grow");
    assert_eq!(
        buffer.committed(),
        2 * TILE,
        "the last byte needs a whole second tile"
    );
    assert!(
        buffer.host_span(TILE, 1).is_ok(),
        "and that byte is addressable"
    );
}

fn a_batch_across_two_arenas_is_refused_rather_than_priced_against_one(
    sparse: &std::sync::Arc<Sparse>,
) {
    let one = ElasticArena::new(64 * 1024 * 1024);
    let two = ElasticArena::new(64 * 1024 * 1024);
    let mut a = elastic::create(sparse, &one, VIRTUAL).expect("a");
    let mut b = elastic::create(sparse, &two, VIRTUAL).expect("b");
    let mut targets = [
        Target {
            buffer: &mut a,
            bytes: TILE,
        },
        Target {
            buffer: &mut b,
            bytes: TILE,
        },
    ];
    let refused = elastic::grow_all(&mut targets).expect_err("two budgets cannot be priced as one");
    assert!(matches!(refused, Fault::Device { .. }), "{refused:?}");
    assert_eq!(a.committed(), 0);
    assert_eq!(b.committed(), 0);
}

fn a_host_move_over_the_pages_lands_where_one_memmove_would(sparse: &std::sync::Arc<Sparse>) {
    let arena = ElasticArena::new(64 * 1024 * 1024);
    let mut buffer = elastic::create(sparse, &arena, VIRTUAL).expect("a sparse buffer");
    grow(&mut buffer, 4 * TILE).expect("grow");
    let source: Vec<u8> = (0..4 * TILE).map(|at| (at % 249) as u8).collect();
    // SAFETY: nothing is in flight.
    unsafe { buffer.write_from(0, &source) }.expect("the host writes");
    let (dst, src, len) = (TILE / 2, TILE, 2 * TILE + 100);
    // SAFETY: as above.
    unsafe { buffer.copy_within(dst, src, len) }.expect("an overlapping move");
    let mut want = source.clone();
    want.copy_within(src as usize..(src + len) as usize, dst as usize);
    let mut back = vec![0u8; source.len()];
    buffer.read_into(0, &mut back).expect("the buffer reads");
    assert_eq!(back, want, "the move slid rather than smeared");
}
