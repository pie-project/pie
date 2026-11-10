//! The state pool's one address range: segments backed from the bottom ahead
//! of need, a span held under its end for a fire's activation, a segment
//! given back to the device once its chunks have moved out, and the move
//! itself one batched device copy in the stream's order. No base ever moves.

use engine_cuda::device::elastic::{Commit, PhysicalPool, Range};
use engine_cuda::device::{Context, copy_any, copy_batch_d2d, copy_d2h_on};

const MIB: u64 = 1 << 20;

fn free_bytes() -> u64 {
    engine_cuda::store::device_memory()
        .expect("the device's memory")
        .0
}

#[test]
fn a_range_backs_its_segments_and_holds_its_top() {
    if !engine_cuda::device::present() {
        eprintln!("no device: skipping");
        return;
    }
    let ctx = Context::bind(0, None).expect("a device context");
    let stream = ctx.stream();
    let mut pool = PhysicalPool::open(ctx.ordinal(), 0.9).expect("the physical pool");

    // Chunks of 5 MiB + 256 bytes, eight a segment: neither a chunk nor a
    // segment is a whole number of map units.
    let chunk = 5 * MIB + 256;
    let segment = 8 * chunk;
    let segments = 6u32;
    let bytes = u64::from(segments) * segment;
    let top = 3 * chunk;
    let mut range = Range::reserve(&pool, bytes, segment, "bytes of the state pool")
        .expect("the range's addresses");
    assert_eq!(range.segments(), segments);
    let base = range.base();
    assert!(range.span(0, 1).is_err(), "nothing is backed at first");

    let before = free_bytes();
    assert_eq!(
        range.hold_top(&mut pool, top).expect("the top"),
        Commit::Committed
    );
    assert_eq!(
        range.commit_segments(&mut pool, 2).expect("two segments"),
        Commit::Committed
    );
    assert_eq!(range.base(), base, "backing moves no base");
    assert!(range.span(0, 2 * segment).is_ok());
    assert!(range.span(bytes - top, top).is_ok());
    assert!(
        range.span(3 * segment, chunk).is_err(),
        "a segment not committed is not backed"
    );
    let held = before - free_bytes();
    assert!(
        held >= 2 * segment + top && held <= 2 * segment + top + 4 * 32 * MIB,
        "two segments and the top are what the device gave: {held}"
    );

    // A chunk's bytes survive a move into a lower segment, pair by pair.
    assert_eq!(
        range.commit_segments(&mut pool, 4).expect("four segments"),
        Commit::Committed
    );
    let pattern = |seed: u64| -> Vec<u8> {
        (0..chunk)
            .map(|i| (i.wrapping_mul(2_654_435_761).wrapping_add(seed) >> 7) as u8)
            .collect()
    };
    // Chunks 24..32 (segment 3) hold data; chunks 1, 3, 5.. of the first
    // segments are free.
    let moved: Vec<(u64, u64)> = (0..8).map(|i| (24 + i, 1 + 2 * i)).collect();
    for &(from, _) in &moved {
        let bytes = pattern(from);
        let at = range.span(from * chunk, chunk).expect("a backed chunk");
        copy_any(stream, at, bytes.as_ptr() as u64, bytes.len()).expect("the chunk's bytes");
    }
    let spans: Vec<(u64, u64, usize)> = moved
        .iter()
        .map(|&(from, to)| {
            (
                range
                    .span(to * chunk, chunk)
                    .expect("the chunk it moves to"),
                range
                    .span(from * chunk, chunk)
                    .expect("the chunk it leaves"),
                chunk as usize,
            )
        })
        .collect();
    copy_batch_d2d(stream, &spans).expect("the batched move");
    ctx.synchronize().expect("the stream drains");

    // The top is written before the release and read after it: the release
    // of the segments under it leaves it mapped where it was.
    let top_bytes = pattern(99);
    let top_at = range.span(bytes - chunk, chunk).expect("the top");
    copy_any(stream, top_at, top_bytes.as_ptr() as u64, top_bytes.len()).expect("the top's bytes");
    ctx.synchronize().expect("the stream drains");

    let with_four = free_bytes();
    range
        .release_segments(&mut pool, 3)
        .expect("the fourth segment goes back");
    assert_eq!(range.committed_segments(), 3);
    assert!(
        free_bytes() >= with_four + segment - 2 * 32 * MIB,
        "the device has the segment's memory again"
    );
    assert!(range.span(3 * segment + 32 * MIB, chunk).is_err());
    assert_eq!(range.base(), base);

    let mut read = vec![0u8; chunk as usize];
    for &(from, to) in &moved {
        let at = range.span(to * chunk, chunk).expect("still backed");
        copy_d2h_on(stream, at, &mut read).expect("the moved chunk");
        assert!(read == pattern(from), "chunk {from} is whole at {to}");
    }
    copy_d2h_on(stream, top_at, &mut read).expect("the top");
    assert!(
        read == top_bytes,
        "the top kept its bytes across the release"
    );

    // A segment committed again is backed again, at the address it had.
    assert_eq!(
        range.commit_segments(&mut pool, 6).expect("every segment"),
        Commit::Committed
    );
    assert_eq!(range.span(0, bytes).expect("the whole range"), base);
    range
        .release_segments(&mut pool, 0)
        .expect("all but the top");
    assert!(range.span(bytes - top, top).is_ok(), "the top is held");
    assert!(range.span(0, 1).is_err());
}
