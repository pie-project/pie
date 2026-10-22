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
        live: [0, 0],
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

/// The pool's bytes after one append of `rows` rows from `at` of the plane,
/// over `qo`, under `stage` where one is armed.
#[allow(clippy::too_many_arguments)]
fn appended(
    fp8: bool,
    ckv: &[u16],
    kpe: &[u16],
    at: usize,
    rows: usize,
    qo: &[i32],
    row_valid: Option<&[u8]>,
    stage: Option<[u32; 4]>,
) -> (Vec<u8>, Vec<u8>) {
    let cells_n = PAGES.len() * PAGE;
    let mut gpu = Gpu::open();
    let ckv_at = gpu.up(&ckv[at * RANK..]);
    let kpe_at = gpu.up(&kpe[at * ROPE..]);
    let qo_at = gpu.up(qo);
    let (keys, values, keys_at, values_at, key_bytes, value_bytes) = if fp8 {
        let k = gpu.zeros(cells_n * ROW);
        let t = Tensor::new(k, cells_n as u32, ROW as u32, Dtype::U8);
        (t, t, k, k, cells_n * ROW, 0)
    } else {
        let k = gpu.zeros(cells_n * RANK * 2);
        let v = gpu.zeros(cells_n * ROPE * 2);
        (
            Tensor::new(k, cells_n as u32, RANK as u32, Dtype::Bf16),
            Tensor::new(v, cells_n as u32, ROPE as u32, Dtype::Bf16),
            k,
            v,
            cells_n * RANK * 2,
            cells_n * ROPE * 2,
        )
    };
    let mut pool = pool(keys, values, &mut gpu);
    if let Some(valid) = row_valid {
        pool.row_valid = Tensor::new(gpu.up(valid), valid.len() as u32, 1, Dtype::U8);
    }
    let ctx = gpu.ctx();
    if let Some(window) = stage {
        ctx.arm_stage(gpu.up(&window));
    }
    mla::kv_append(
        &ctx,
        RaggedTensor {
            data: Tensor::new(ckv_at, rows as u32, RANK as u32, Dtype::Bf16),
            indptr: Tensor::new(qo_at, qo.len() as u32, 1, Dtype::I32),
        },
        Tensor::new(kpe_at, rows as u32, ROPE as u32, Dtype::Bf16),
        &pool,
        Tensor::ABSENT,
        Tensor::ABSENT,
    )
    .expect("the append fires");
    gpu.sync();
    (
        gpu.down(keys_at, key_bytes),
        if value_bytes > 0 {
            gpu.down(values_at, value_bytes)
        } else {
            Vec::new()
        },
    )
}

/// A body replays the append over a plane whose window starts past row 0
/// and holds fewer rows and requests than the body's ceiling: the rows
/// before it are another window's, the ceiling's trailing rows another's
/// still (valid ones), and the indptr past the live requests is stale and
/// climbing. Under the staged window (`[live rows, first row, live lanes,
/// first lane]`) the pool lands what the window's own rows land alone.
#[test]
fn a_staged_window_appends_only_its_own_rows() {
    const BASE: usize = 2;
    const LIVE: usize = 6;
    const CEILING: usize = 9;
    let plane = BASE + CEILING;
    let ckv: Vec<u16> = (0..plane * RANK)
        .map(|i| to_bf16(((i / RANK) as f32 + 1.0) * (1.0 + (i % 7) as f32 / 8.0)))
        .collect();
    let kpe: Vec<u16> = (0..plane * ROPE)
        .map(|i| to_bf16(-((i / ROPE) as f32 + 1.0) - (i % ROPE) as f32 / 64.0))
        .collect();
    let stale: [i32; 9] = [0, 2, 3, 6, 7, 8, 9, 10, 11];
    let valid = vec![1u8; plane];
    for fp8 in [false, true] {
        let alone = appended(fp8, &ckv, &kpe, BASE, LIVE, &QO, None, None);
        let windowed = appended(
            fp8,
            &ckv,
            &kpe,
            0,
            CEILING,
            &stale,
            Some(&valid),
            Some([LIVE as u32, BASE as u32, 3, 0]),
        );
        assert!(
            alone.0.iter().any(|&b| b != 0),
            "fp8 {fp8}: the window's own rows land"
        );
        assert!(
            alone == windowed,
            "fp8 {fp8}: the windowed append lands what the window's rows land alone"
        );
    }
}
