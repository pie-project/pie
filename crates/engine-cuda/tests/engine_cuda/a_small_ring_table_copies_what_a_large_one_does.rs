#![cfg(feature = "device")]

use engine_cuda::device::Buffer;
use engine_cuda::serve::ring_table::{RUN_BYTES, RingTable};

/// Runs past what the table holds go out in several launches over it, in
/// the stream's order, and land the bytes one launch over a large table
/// does: here 40 tables' worth back to back, so host places and events are
/// both reused while earlier copies stand.
#[test]
fn a_small_ring_table_copies_what_a_large_one_does() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    const RUNS: usize = 4001;
    let device = engine_cuda::device::ctx::Context::bind(0, None).expect("the device binds");
    // Run `i` copies `8 + i % 7` words from a place of its own to another,
    // the runs' order scrambled.
    let run_bytes = |i: usize| 8 * (8 + i % 7);
    let slot = 8 * 16;
    let bytes: Vec<u8> = (0..RUNS * slot)
        .map(|at| (at.wrapping_mul(2_654_435_761) >> 11) as u8)
        .collect();
    let mut src = Buffer::zeroed(bytes.len()).expect("a source");
    src.write(0, &bytes).expect("the pattern lands");
    let copied = |table_runs: usize| {
        let dst = Buffer::zeroed(bytes.len()).expect("a destination");
        let runs: Vec<(u64, u64, u64)> = (0..RUNS)
            .map(|i| {
                let to = (i * 1237) % RUNS;
                (
                    dst.ptr() + (to * slot) as u64,
                    src.ptr() + (i * slot) as u64,
                    run_bytes(i) as u64,
                )
            })
            .collect();
        let held = Buffer::zeroed(table_runs * RUN_BYTES).expect("the table");
        let mut table = RingTable::new(table_runs, 1).expect("the host arena");
        table.lay(held.ptr());
        let mut launches = 0;
        table
            .copy(device.stream(), &runs, |at, count| {
                launches += 1;
                kernels_cuda::runtime::channel::copy_runs(device.ctx(), at, count)
                    .map_err(engine_cuda::Fault::from)
            })
            .expect("the runs go out");
        device.synchronize().expect("the stream drains");
        let mut got = vec![0u8; bytes.len()];
        dst.read(0, &mut got).expect("the destination reads");
        (got, launches)
    };
    let (small, launches) = copied(100);
    assert_eq!(launches, RUNS.div_ceil(100));
    let (large, one) = copied(RUNS);
    assert_eq!(one, 1);
    assert!(small == large, "several launches land other bytes");
    for i in 0..RUNS {
        let to = ((i * 1237) % RUNS) * slot;
        assert_eq!(small[to..to + run_bytes(i)], bytes[i * slot..i * slot + run_bytes(i)], "run {i}");
    }
}
