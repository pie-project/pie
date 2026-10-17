#![cfg(feature = "cuda")]

//! The row pack gathers each row from its source and the row unpack lands
//! the rectangle back.

use crate::common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::layout;
use kernels_cuda::tensor::Tensor;

const ROWS: usize = 33;

fn permutation() -> Vec<i32> {
    (0..ROWS).map(|i| ((i * 7) % ROWS) as i32).collect()
}

fn check(width: usize, dtype: Dtype, elem: usize) {
    let mut lcg = Lcg::seeded(0x9acc);
    let (raw, _) = lcg.row(ROWS * width * elem / 2);
    let perm = permutation();

    let mut gpu = Gpu::open();
    let x_at = gpu.up(&raw);
    let perm_at = gpu.up(&perm);
    let packed = gpu.zeros(ROWS * width * elem);
    let back = gpu.zeros(ROWS * width * elem);
    let ctx = gpu.ctx();

    let rect = |at: u64| Tensor::new(at, ROWS as u32, width as u32, dtype);
    let map = Tensor::new(perm_at, ROWS as u32, 1, Dtype::I32);
    layout::pack_rows(&ctx, rect(x_at), map, &mut rect(packed)).expect("the pack fires");
    layout::unpack_rows(&ctx, rect(packed), map, &mut rect(back)).expect("the unpack fires");
    gpu.sync();

    let want: Vec<u16> = gpu.down(x_at, raw.len());
    let got_packed: Vec<u16> = gpu.down(packed, raw.len());
    let got_back: Vec<u16> = gpu.down(back, raw.len());
    let units = width * elem / 2;
    for row in 0..ROWS {
        let from = perm[row] as usize;
        assert_eq!(
            got_packed[row * units..(row + 1) * units],
            want[from * units..(from + 1) * units],
            "row {row} was not gathered from row {from}"
        );
    }
    assert_eq!(got_back, want, "the round trip did not land the rectangle");
}

#[test]
fn the_row_pack_and_the_row_unpack_round_trip_every_case() {
    // A wide bf16 rectangle, an f32 one, and a row no wide unit covers.
    check(64, Dtype::Bf16, 2);
    check(20, Dtype::F32, 4);
    check(6, Dtype::Bf16, 2);
}
