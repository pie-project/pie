#![cfg(feature = "cuda")]

//! A fire seats more lanes than it fills: the unfilled lanes' indptr reads 0
//! and they own no pages. DeepSeek-V4's pool boundaries must not hand such a
//! lane a boundary row, and the gather must not walk pages a lane does not
//! own (it read past the page table, a 700 under compute-sanitizer).

use crate::common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::attn::pool;
use kernels_cuda::tensor::{KvPool, RaggedTensor, Tensor};

const PAGE: usize = 16;

fn pool_of(gpu: &mut Gpu, keys: u64, pages: &[u32], indptr: &[u32], width: usize) -> KvPool {
    KvPool {
        keys: Tensor::new(keys, (pages.len() * PAGE) as u32, width as u32, Dtype::Bf16),
        values: Tensor::ABSENT,
        bf16_keys: Tensor::ABSENT,
        bf16_values: Tensor::ABSENT,
        key_scales: Tensor::ABSENT,
        value_scales: Tensor::ABSENT,
        page_indices: Tensor::new(gpu.up(pages), pages.len() as u32, 1, Dtype::U32),
        page_indptr: Tensor::new(gpu.up(indptr), indptr.len() as u32, 1, Dtype::U32),
        last_page_lens: Tensor::ABSENT,
        row_valid: Tensor::ABSENT,
        env_min: Tensor::ABSENT,
        env_max: Tensor::ABSENT,
        has_envelopes: false,
        page_size: PAGE as i32,
        seq_stride: width as i64,
        head_stride: width as i64,
        layout: 0,
        scheme_byte: 0,
        block_size: 0,
        max_pages_per_request: 1,
        pages_in_batch: pages.len() as i32,
    }
}

#[test]
fn a_padded_lane_closes_no_pool_entry() {
    a_row_past_the_live_lanes_is_no_boundary();
    a_lane_without_pages_gathers_nothing();
}

fn a_row_past_the_live_lanes_is_no_boundary() {
    // Two live requests of 8 rows each, three padded lanes reading 0, and
    // four padded rows whose positions would close entries.
    let ratio = 4usize;
    let positions: Vec<i32> = (0..16).map(|r| r % 8).chain([3, 7, 11, 15]).collect();
    let indptr: Vec<i32> = vec![0, 8, 16, 0, 0, 0];
    let rows = positions.len();
    let mut gpu = Gpu::open();
    let pos_t = Tensor::new(gpu.up(&positions), rows as u32, 1, Dtype::I32);
    let ind_t = Tensor::new(gpu.up(&indptr), indptr.len() as u32, 1, Dtype::I32);
    let (bpos_at, breq_at, brope_at) = (
        gpu.zeros(rows * 4),
        gpu.zeros(rows * 4),
        gpu.zeros(rows * 4),
    );
    let ctx = gpu.ctx();
    pool::boundary_prefill(
        &ctx,
        RaggedTensor {
            data: pos_t,
            indptr: ind_t,
        },
        Tensor::ABSENT,
        ratio as u32,
        &mut Tensor::new(bpos_at, rows as u32, 1, Dtype::I32),
        &mut Tensor::new(breq_at, rows as u32, 1, Dtype::I32),
        &mut Tensor::new(brope_at, rows as u32, 1, Dtype::I32),
    )
    .expect("the boundaries fire");
    gpu.sync();
    let bpos: Vec<i32> = gpu.down(bpos_at, rows);
    let breq: Vec<i32> = gpu.down(breq_at, rows);
    for r in 0..rows {
        let live = r < 16;
        let closes = live && (positions[r] + 1) % ratio as i32 == 0;
        assert_eq!(
            bpos[r],
            if closes { positions[r] } else { -1 },
            "row {r}: boundary position"
        );
        if live {
            assert_eq!(breq[r], (r / 8) as i32, "row {r}: owning request");
        }
    }
}

fn a_lane_without_pages_gathers_nothing() {
    // Request 0 owns page 0; request 2 is a padded lane owning [1, 1). Both
    // rows name position 7 as their boundary. A third row past the staged
    // window holds what an earlier fire left there.
    let (ratio, d) = (4usize, 64usize);
    let width = 2 * d;
    let pages: Vec<u32> = vec![0];
    let indptr: Vec<u32> = vec![0, 1, 1, 1];
    let mut lcg = Lcg::seeded(0x5eed);
    let (kv_bits, _) = lcg.row(PAGE * width);
    let (score_bits, _) = lcg.row(PAGE * width);
    let mut gpu = Gpu::open();
    let kv = gpu.up(&kv_bits);
    let score = gpu.up(&score_bits);
    let state = pool_of(&mut gpu, kv, &pages, &indptr, width);
    let bpos = Tensor::new(gpu.up(&[7i32, 7, 1_000_000_003]), 3, 1, Dtype::I32);
    let breq = Tensor::new(gpu.up(&[0i32, 2, -123_456_789]), 3, 1, Dtype::I32);
    let out_at = gpu.up(&vec![0x3f80u16; 3 * d]);
    let win_at = gpu.up(&[2u32, 0, 3, 0]);
    let ctx = gpu.ctx();
    ctx.arm_stage(win_at);
    pool::gather(
        &ctx,
        bpos,
        breq,
        &state,
        d as u32,
        ratio as u32,
        Tensor::new(kv, PAGE as u32, width as u32, Dtype::Bf16),
        Tensor::new(score, PAGE as u32, width as u32, Dtype::Bf16),
        Tensor::ABSENT,
        &mut Tensor::new(out_at, 3, d as u32, Dtype::Bf16),
    )
    .expect("the gather fires");
    gpu.sync();
    let out: Vec<u16> = gpu.down(out_at, 3 * d);
    assert!(
        out[..d].iter().any(|&b| b != 0),
        "the live lane gathers its entry"
    );
    assert!(
        out[d..2 * d].iter().all(|&b| b == 0),
        "the padded lane gathers nothing"
    );
    assert!(
        out[2 * d..].iter().all(|&b| b == 0x3f80),
        "the row past the window is left alone"
    );
}
