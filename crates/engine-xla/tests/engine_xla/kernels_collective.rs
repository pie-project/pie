//! The collective family on one replica against host references.

use dtype::Dtype;
use engine_xla::bench::{Bench, assert_close, round_bf16};
use kernels_xla::collective;

fn data(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed.wrapping_mul(40503));
            round_bf16(((h >> 8) % 2000) as f32 / 1000.0 - 1.0)
        })
        .collect()
}

#[test]
fn one_replica_reduces_to_itself_and_copies_its_band() {
    let xs = data(3 * 10, 1);
    let mut b = Bench::new();
    let buf = b.bf16(3, 10, &xs);
    let y = b.zeros(Dtype::Bf16, 3, 10);
    let z = b.zeros(Dtype::Bf16, 3, 10);
    let two = b.zeros(Dtype::Bf16, 3, 20);
    if !b
        .run(|ctx| {
            collective::all_reduce(ctx, buf)?;
            collective::all_gather(ctx, buf, y)?;
            collective::reduce_scatter(ctx, buf, z)?;
            // A two-rank gather is refused, not approximated.
            assert!(collective::all_gather(ctx, buf, two).is_err());
            Ok(())
        })
        .unwrap()
    {
        return;
    }
    assert_close(&b.read_f32(buf), &xs, 0.0, 0.0);
    assert_close(&b.read_f32(y), &xs, 0.0, 0.0);
    assert_close(&b.read_f32(z), &xs, 0.0, 0.0);
}
