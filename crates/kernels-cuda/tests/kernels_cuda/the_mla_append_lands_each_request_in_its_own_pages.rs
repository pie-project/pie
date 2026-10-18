#![cfg(feature = "cuda")]

//! MLA's append (`attention.mla_kv_append`) lands each token's latent and
//! rotated rows at its own request's next positions, over a query indptr
//! whose entries past the live requests are stale zeros (a fire's unused
//! request slots), on a bf16 pool and on a standard fp8 one.

use crate::common::{Gpu, to_bf16};
use dtype::Dtype;
use kernels_cuda::attn::mla;
use kernels_cuda::tensor::{KvPool, RaggedTensor, Tensor};

const RANK: usize = 512;
const ROPE: usize = 64;
const PAGE: usize = 4;
const ROW: usize = 576;
/// Three live requests of 2, 1 and 3 new tokens, then five unused slots
/// whose indptr entries are zero (enough that a search halving the slots
/// lands on one first).
const QO: [i32; 9] = [0, 2, 3, 6, 0, 0, 0, 0, 0];
const PAGE_INDPTR: [i32; 9] = [0, 2, 3, 5, 5, 5, 5, 5, 5];
const PAGES: [i32; 5] = [4, 1, 3, 0, 2];
/// Keys a request holds after the append: 5, 2 and 7.
const LAST: [i32; 8] = [1, 2, 3, 0, 0, 0, 0, 0];
const KV: [usize; 3] = [5, 2, 7];

fn pool(keys: Tensor, values: Tensor, gpu: &mut Gpu) -> KvPool {
    KvPool {
        keys,
        values,
        bf16_keys: Tensor::ABSENT,
        bf16_values: Tensor::ABSENT,
        key_scales: Tensor::ABSENT,
        value_scales: Tensor::ABSENT,
        page_indices: Tensor::new(gpu.up(&PAGES), PAGES.len() as u32, 1, Dtype::I32),
        page_indptr: Tensor::new(
            gpu.up(&PAGE_INDPTR),
            PAGE_INDPTR.len() as u32,
            1,
            Dtype::I32,
        ),
        last_page_lens: Tensor::new(gpu.up(&LAST), LAST.len() as u32, 1, Dtype::I32),
        row_valid: Tensor::ABSENT,
        env_min: Tensor::ABSENT,
        env_max: Tensor::ABSENT,
        has_envelopes: false,
        page_size: PAGE as i32,
        seq_stride: RANK as i64,
        head_stride: RANK as i64,
        layout: 0,
        scheme_byte: 0,
        block_size: 0,
        max_pages_per_request: 2,
        pages_in_batch: PAGES.len() as i32,
    }
}

/// The pool cell token `t` (of request `r`, its `i`th new row) lands at.
fn cells() -> Vec<usize> {
    let mut out = Vec::new();
    for r in 0..3 {
        let fresh = (QO[r + 1] - QO[r]) as usize;
        for i in 0..fresh {
            let pos = KV[r] - fresh + i;
            let page = PAGES[PAGE_INDPTR[r] as usize + pos / PAGE] as usize;
            out.push(page * PAGE + pos % PAGE);
        }
    }
    out
}

#[test]
fn the_mla_append_lands_each_request_in_its_own_pages() {
    let tokens = 6;
    let ckv: Vec<u16> = (0..tokens * RANK)
        .map(|i| to_bf16(((i / RANK) as f32 + 1.0) * (1.0 + (i % 7) as f32 / 8.0)))
        .collect();
    let kpe: Vec<u16> = (0..tokens * ROPE)
        .map(|i| to_bf16(-((i / ROPE) as f32 + 1.0) - (i % ROPE) as f32 / 64.0))
        .collect();
    let cells_n = PAGES.len() * PAGE;
    for fp8 in [false, true] {
        let mut gpu = Gpu::open();
        let ckv_at = gpu.up(&ckv);
        let kpe_at = gpu.up(&kpe);
        let qo_at = gpu.up(&QO);
        let (keys, values, keys_at, values_at) = if fp8 {
            let at = gpu.zeros(cells_n * ROW);
            let t = Tensor::new(at, cells_n as u32, ROW as u32, Dtype::U8);
            (t, t, at, at)
        } else {
            let k = gpu.zeros(cells_n * RANK * 2);
            let v = gpu.zeros(cells_n * ROPE * 2);
            (
                Tensor::new(k, cells_n as u32, RANK as u32, Dtype::Bf16),
                Tensor::new(v, cells_n as u32, ROPE as u32, Dtype::Bf16),
                k,
                v,
            )
        };
        let pool = pool(keys, values, &mut gpu);
        mla::kv_append(
            &gpu.ctx(),
            RaggedTensor {
                data: Tensor::new(ckv_at, tokens as u32, RANK as u32, Dtype::Bf16),
                indptr: Tensor::new(qo_at, QO.len() as u32, 1, Dtype::I32),
            },
            Tensor::new(kpe_at, tokens as u32, ROPE as u32, Dtype::Bf16),
            &pool,
            Tensor::ABSENT,
            Tensor::ABSENT,
        )
        .expect("the append fires");
        gpu.sync();
        let landed = cells();
        if fp8 {
            let rows: Vec<u8> = gpu.down(keys_at, cells_n * ROW);
            for (t, &cell) in landed.iter().enumerate() {
                let row = &rows[cell * ROW..(cell + 1) * ROW];
                assert!(
                    row[..512].iter().any(|&b| b != 0) && row[512..].iter().any(|&b| b != 0),
                    "fp8: token {t} codes at cell {cell}"
                );
            }
            let used: Vec<usize> = landed.clone();
            for cell in (0..cells_n).filter(|c| !used.contains(c)) {
                assert!(
                    rows[cell * ROW..(cell + 1) * ROW].iter().all(|&b| b == 0),
                    "fp8: cell {cell} written"
                );
            }
        } else {
            let k: Vec<u16> = gpu.down(keys_at, cells_n * RANK);
            let v: Vec<u16> = gpu.down(values_at, cells_n * ROPE);
            for (t, &cell) in landed.iter().enumerate() {
                assert_eq!(
                    &k[cell * RANK..(cell + 1) * RANK],
                    &ckv[t * RANK..(t + 1) * RANK],
                    "bf16: token {t} latent at cell {cell}"
                );
                assert_eq!(
                    &v[cell * ROPE..(cell + 1) * ROPE],
                    &kpe[t * ROPE..(t + 1) * ROPE],
                    "bf16: token {t} rope at cell {cell}"
                );
            }
            for cell in (0..cells_n).filter(|c| !landed.contains(c)) {
                assert!(
                    k[cell * RANK..(cell + 1) * RANK].iter().all(|&x| x == 0),
                    "bf16: cell {cell} written"
                );
            }
        }
    }
}
