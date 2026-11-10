#![cfg(feature = "cuda")]

//! A causal conv3d's frame cache read and written through a segment table
//! lands what the slot layout lands, bit for bit: Wan's VAE frame slab cut
//! along its frame-position rows into pieces scattered over a two-segment
//! pool, the garbage between them left alone; a piece offset a few elements
//! off does not.

use crate::common::segments::Cut;
use crate::common::{Gpu, Lcg};
use crate::ssm::segments::{Slab, agree, bits};
use dtype::Dtype;
use kernels_cuda::Snapshots;
use kernels_cuda::conv;
use kernels_cuda::tensor::{RecurrentPool, Tensor};

#[test]
fn the_frame_cache_reads_and_writes_its_slab_through_the_segment_table() {
    const C: usize = 32;
    const FRAMES: u32 = 2;
    // Two clips, `{t, h, w, row_offset}`: 3x4x5 from row 0, 2x3x6 after it.
    let grid: [i32; 8] = [3, 4, 5, 0, 2, 3, 6, 60];
    let rows = 60 + 36;
    let plane = 20;
    let row = FRAMES as usize * plane * C;
    let cache_rows = conv::cache_rows(&grid, FRAMES) as usize;
    for pieces in [2usize, 4] {
        let mut lcg = Lcg::seeded(0xf7a3);
        let mut gpu = Gpu::open();
        let x_at = gpu.up(&lcg.row(rows * C).0);
        let grid_at = gpu.up(&grid);
        let slab = Slab {
            seed: lcg.row(3 * row).0,
            slots: 3,
            stride: row,
            dtype: Dtype::Bf16,
            cut: Cut::runs(pieces, row / pieces),
        };
        let mut fire = |gpu: &mut Gpu, pool: RecurrentPool| {
            let cache_at = gpu.zeros(cache_rows * C * 2);
            let mut cache = Tensor::new(cache_at, cache_rows as u32, C as u32, Dtype::Bf16);
            let grid_t = Tensor::new(grid_at, 2, 4, Dtype::I32);
            let ctx = gpu.ctx();
            conv::cache_gather(&ctx, &pool, pool.slot_ids, grid_t, FRAMES, &mut cache)
                .expect("the gather fires");
            let x = Tensor::new(x_at, rows as u32, C as u32, Dtype::Bf16);
            conv::cache_store(&ctx, x, cache, &pool, pool.slot_ids, grid_t, FRAMES, 0)
                .expect("the store fires");
            gpu.sync();
            bits::<u16>(gpu, cache_at, cache_rows * C)
        };
        agree(
            &mut gpu,
            &format!("frame cache pieces={pieces}"),
            &slab,
            &[2, 0],
            Snapshots::NONE,
            &[],
            &mut fire,
            8,
        );
    }
}
