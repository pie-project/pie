//! A token-id gather reads its table through a row table: the rows a block
//! of the table lies wherever its entry says (a slot of the weight cache, or
//! the mapped host copy), each a plain load, and lands the bits the
//! contiguous table lands. Its hit counter counts the rows read a block.

#![cfg(feature = "rocm")]

mod common;

use common::Gpu;
use dtype::Dtype::{Bf16, I32, U32};
use kernels_rocm::gather::embed_tabled;
use kernels_rocm::linear::moe::RowTable;
use kernels_rocm::tensor::Tensor;

/// A small LCG, as the CUDA test seeds its tables.
struct Lcg(u64);

impl Lcg {
    fn bits(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        self.0
    }
}

#[test]
fn an_embed_lands_the_same_rows_through_any_table() {
    let Some(gpu) = Gpu::open() else { return };
    // 64 wide takes the 16-byte lanes; 48 and 6 are off them or off a lane.
    for (width, block_rows) in [(64usize, 1u32), (64, 4), (48, 3), (6, 1)] {
        let (vocab, tokens) = (37usize, 29usize);
        let mut lcg = Lcg(width as u64 + u64::from(block_rows));
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
        let dense = gpu.upload(&table, vocab as u32, width as u32, Bf16);
        let ids_at = gpu.column(&ids);

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
        let spread_at = gpu.upload(&spread, 1, spread.len() as u32, Bf16);
        let scattered: Vec<u64> = (0..blocks)
            .map(|block| spread_at.ptr + (place(block) * 2) as u64)
            .collect();
        // A mixed table: even blocks where the contiguous table has them.
        let mixed: Vec<u64> = (0..blocks)
            .map(|block| {
                if block % 2 == 0 {
                    dense.ptr + (block * block_rows as usize * width * 2) as u64
                } else {
                    scattered[block]
                }
            })
            .collect();
        let scattered_t = gpu.upload(&scattered, blocks as u32, 2, U32);
        let mixed_t = gpu.upload(&mixed, blocks as u32, 2, U32);
        let hits = gpu.zeros(blocks as u32, 1, U32);

        let ctx = gpu.ctx();
        let mut landed = Vec::new();
        for rows in [
            RowTable::RESIDENT,
            RowTable::located(scattered_t.ptr, block_rows),
            RowTable::located(mixed_t.ptr, block_rows).counting(hits.ptr),
        ] {
            let mut y = gpu.zeros(tokens as u32, width as u32, Bf16);
            embed_tabled(
                &ctx,
                Tensor::new(ids_at.ptr, tokens as u32, 1, I32),
                dense,
                rows,
                vocab as u32,
                &mut y,
            )
            .expect("the gather fires");
            gpu.sync();
            landed.push(gpu.read_as::<u16>(&y));
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

        let counted: Vec<u32> = gpu.read_as(&hits);
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
