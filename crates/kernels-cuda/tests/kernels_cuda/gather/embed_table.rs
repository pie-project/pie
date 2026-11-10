#![cfg(feature = "cuda")]

//! A token-id gather reads its table through a row table: the rows a block
//! of the table lies wherever its entry says (a slot of the weight cache, or
//! the mapped host copy), each a plain load, and lands the bits the
//! contiguous table lands. Its hit counter counts the rows read a block.

use crate::common::{Gpu, Lcg};
use dtype::Dtype::{Bf16, I32};
use kernels_cuda::gather::embed_tabled;
use kernels_cuda::linear::moe::RowTable;
use kernels_cuda::tensor::Tensor;

#[test]
fn an_embed_lands_the_same_rows_through_any_table() {
    for (width, block_rows) in [(64usize, 1u32), (64, 4), (48, 3), (6, 1)] {
        let (vocab, tokens) = (37usize, 29usize);
        let mut gpu = Gpu::open();
        let mut lcg = Lcg::seeded(width as u64 + u64::from(block_rows));
        let table: Vec<u16> = (0..vocab * width)
            .map(|_| (lcg.bits() >> 20) as u16)
            .collect();
        // Out of range ids read row 0, as over the contiguous table.
        let ids: Vec<i32> = (0..tokens)
            .map(|t| match t {
                5 => -3,
                11 => vocab as i32 + 2,
                _ => ((lcg.bits() >> 33) as usize % vocab) as i32,
            })
            .collect();
        let (dense, ids_at) = (gpu.up(&table), gpu.up(&ids));

        // The table's blocks scattered through another buffer in reverse
        // order, garbage between them; the last block is short.
        let blocks = vocab.div_ceil(block_rows as usize);
        let stride = block_rows as usize * width + 2 * width;
        let mut spread: Vec<u16> = (0..blocks * stride)
            .map(|_| (lcg.bits() >> 20) as u16)
            .collect();
        let place = |block: usize| (blocks - 1 - block) * stride + width;
        for row in 0..vocab {
            let (block, within) = (row / block_rows as usize, row % block_rows as usize);
            let at = place(block) + within * width;
            spread[at..at + width].copy_from_slice(&table[row * width..(row + 1) * width]);
        }
        let spread_at = gpu.up(&spread);
        let scattered: Vec<u64> = (0..blocks)
            .map(|block| spread_at + (place(block) * 2) as u64)
            .collect();
        // A mixed table: even blocks where the contiguous table has them.
        let mixed: Vec<u64> = (0..blocks)
            .map(|block| {
                if block % 2 == 0 {
                    dense + (block * block_rows as usize * width * 2) as u64
                } else {
                    scattered[block]
                }
            })
            .collect();
        let hits = gpu.zeros(blocks * 4);

        let ctx = gpu.ctx();
        let mut landed = Vec::new();
        for rows in [
            RowTable::RESIDENT,
            RowTable::located(gpu.up(&scattered), block_rows),
            RowTable::located(gpu.up(&mixed), block_rows).counting(hits),
        ] {
            let y = gpu.zeros(tokens * width * 2);
            embed_tabled(
                &ctx,
                Tensor::new(ids_at, tokens as u32, 1, I32),
                Tensor::new(dense, vocab as u32, width as u32, Bf16),
                rows,
                vocab as u32,
                &mut Tensor::new(y, tokens as u32, width as u32, Bf16),
            )
            .expect("the gather fires");
            gpu.sync();
            landed.push(gpu.down::<u16>(y, tokens * width));
        }
        let what = format!("width {width}, {block_rows} rows a block");
        for (t, &id) in ids.iter().enumerate() {
            let row = if id >= 0 && (id as usize) < vocab {
                id as usize
            } else {
                0
            };
            assert_eq!(
                landed[0][t * width..(t + 1) * width],
                table[row * width..(row + 1) * width],
                "{what}: token {t} is its row"
            );
        }
        assert!(landed[1] == landed[0], "{what}: scattered blocks");
        assert!(landed[2] == landed[0], "{what}: a mixed table");

        let counted: Vec<u32> = gpu.down(hits, blocks);
        let mut want = vec![0u32; blocks];
        for &id in &ids {
            let row = if id >= 0 && (id as usize) < vocab {
                id as usize
            } else {
                0
            };
            want[row / block_rows as usize] += 1;
        }
        assert_eq!(counted, want, "{what}: a hit a row read, by its block");
    }
}
