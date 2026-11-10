//! A causal conv3d's frame cache read and written through a segment table
//! lands what the slot layout lands, bit for bit: Wan's VAE frame slab cut
//! along its frame-position rows into pieces scattered over a two-segment
//! pool, the garbage between them left alone; a piece offset a few elements
//! off does not.

#![cfg(feature = "rocm")]

mod common;
#[path = "common/segments.rs"]
mod segments;

use common::Gpu;
use dtype::Dtype;
use kernels_rocm::conv;
use kernels_rocm::{RecurrentPool, Snapshots};
use segments::{Cut, Lcg, Slab, agree, bits};

#[test]
fn the_frame_cache_reads_and_writes_its_slab_through_the_segment_table() {
    let Some(gpu) = Gpu::open() else { return };
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
        let x = gpu.upload(
            &lcg.bf16s(rows * C, 1.0),
            rows as u32,
            C as u32,
            Dtype::Bf16,
        );
        let grid_t = gpu.ints_shaped(&grid, 2, 4);
        let slab = Slab {
            seed: lcg.bf16s(3 * row, 1.0),
            slots: 3,
            stride: row,
            dtype: Dtype::Bf16,
            cut: Cut::runs(pieces, row / pieces),
        };
        let mut fire = |gpu: &Gpu, pool: RecurrentPool| {
            let mut cache = gpu.zeros(cache_rows as u32, C as u32, Dtype::Bf16);
            let ctx = gpu.ctx();
            conv::cache_gather(&ctx, &pool, pool.slot_ids, grid_t, FRAMES, &mut cache)
                .expect("the gather fires");
            conv::cache_store(&ctx, x, cache, &pool, pool.slot_ids, grid_t, FRAMES, 0)
                .expect("the store fires");
            bits::<u16>(gpu, &cache)
        };
        agree(
            &gpu,
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
