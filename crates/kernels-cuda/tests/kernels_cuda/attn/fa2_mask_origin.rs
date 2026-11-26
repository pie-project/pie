#![cfg(feature = "cuda")]

//! FA2's custom-mask arm reads a request's committed keys under the causal
//! bound alone and its plane from the boundary on (`maybe_mask_origin`):
//! the same rows land as from a plane over the whole context whose
//! committed columns are all set. A row reads no key after its own, whatever
//! its bits say.

use crate::common::attn_ref::{col, pool, ten};
use crate::common::{Gpu, Lcg};
use dtype::Dtype::{Bf16, I32, I64, U8, U32};
use kernels_cuda::attn::plan::{self, Device, Live, Shape, Workspace};
use kernels_cuda::attn;
use kernels_cuda::tensor::{KvPool, RaggedTensor};
use kernels_cuda::tuning::Tuning;

const PAGE: usize = 16;

/// How a request's plane is laid out.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Plane {
    /// From its committed boundary on, the boundary in its word.
    Origin,
    /// Over the whole context, its committed columns all set.
    Whole,
    /// As `Whole`, with every bit above the causal diagonal set too.
    WholeAboveDiagonal,
}

/// The buffer bit of row `q`, column `c` past the boundary: a pattern that
/// hides some keys of every row and keeps its own.
fn kept(q: usize, c: usize, buffer: usize) -> bool {
    c == buffer + q || (c * 7 + q * 3) % 5 != 0
}

/// The masked prefill over `[q heads, kv heads, head dim]` and requests of
/// `(new, committed, buffered)` rows: the rows it lands, for `plane`.
#[allow(clippy::too_many_lines)]
fn fire(heads: [usize; 3], reqs: &[(usize, usize, usize)], plane: Plane) -> Vec<u16> {
    let ([qh, kvh, d], n) = (heads, reqs.len());
    let (width, qw, scale) = (kvh * d, qh * d, (d as f32).sqrt().recip());
    let (mut qo, mut kv_indptr, mut kv_len, mut last) = (vec![0], vec![0], vec![], vec![]);
    let mut pages: Vec<i32> = Vec::new();
    let mut next = 1usize;
    let mut bits: Vec<u8> = Vec::new();
    let mut mask_indptr = vec![0i64];
    let mut origins: Vec<u32> = Vec::new();
    for (r, &(new, committed, buffered)) in reqs.iter().enumerate() {
        let len = new + committed + buffered;
        let np = len.div_ceil(PAGE);
        pages.extend((0..np).map(|i| (next + i) as i32));
        next += np;
        qo.push(qo[r] + new as i32);
        kv_indptr.push(kv_indptr[r] + np as i32);
        kv_len.push(len as i32);
        last.push(((len - 1) % PAGE + 1) as i32);
        let (origin, columns) = match plane {
            Plane::Origin => (committed, buffered + new),
            _ => (0, len),
        };
        origins.push(origin as u32);
        // Each request's plane starts on a byte, as the engine lays them.
        let mut cells = vec![false; new * columns];
        for q in 0..new {
            for column in 0..columns {
                let key = origin + column;
                cells[q * columns + column] = if key < committed {
                    true
                } else if key > committed + buffered + q {
                    // After the row's own key: the causal bound hides it.
                    plane == Plane::WholeAboveDiagonal
                } else {
                    kept(q, key - committed, buffered)
                };
            }
        }
        let base = bits.len();
        bits.resize(base + (new * columns).div_ceil(8), 0);
        for (cell, &on) in cells.iter().enumerate() {
            if on {
                bits[base + cell / 8] |= 1 << (cell % 8);
            }
        }
        mask_indptr.push(bits.len() as i64);
    }
    bits.resize(bits.len().max(8), 0);
    let (rows, cells) = (qo[n] as usize, next * PAGE);
    let mut lcg = Lcg::seeded(0x0f19 + d as u64);
    let (k, _) = lcg.row(cells * width);
    let (v, _) = lcg.row(cells * width);
    let (q_raw, _) = lcg.row(rows * qw);

    let mut gpu = Gpu::open();
    let ids = col(&mut gpu, &pages, I32);
    let indptr = col(&mut gpu, &kv_indptr, I32);
    let last_page_lens = col(&mut gpu, &last, I32);
    let mask_t = ten(gpu.up(&bits), bits.len(), 1, U8);
    let mask_ip = col(&mut gpu, &mask_indptr, I64);
    let origin_t = col(&mut gpu, &origins, U32);
    let data = ten(gpu.up(&q_raw), rows, qw, Bf16);
    let qt = RaggedTensor {
        data,
        indptr: col(&mut gpu, &qo, I32),
    };
    let (int_ptr, float_ptr, int_bytes) = (gpu.zeros(64 << 20), gpu.zeros(64 << 20), 64 << 20);
    let ws = Workspace {
        int_ptr,
        int_bytes,
        float_ptr,
        float_bytes: int_bytes,
    };
    let ctx = gpu.ctx().with_tuning(Tuning::DEFAULT);
    let device = Device::probe(&ctx).expect("the device answers");
    let ([qh, kvh, d], page_size) = ([qh, kvh, d].map(|x| x as u32), PAGE as u32);
    #[rustfmt::skip]
    let shape = Shape { num_requests: n as u32, lane_offset: 0, num_q_heads: qh, num_kv_heads: kvh, head_dim: d, page_size, hnd_layout: false };
    let live = Live {
        requests: n as u32,
        lane_offset: 0,
        row_offset: 0,
        rows: rows as u32,
    };
    let kvp = KvPool {
        values: ten(gpu.up(&v), cells, width, Bf16),
        last_page_lens,
        head_stride: i64::from(d),
        ..pool(ten(gpu.up(&k), cells, width, Bf16), ids, indptr, PAGE)
    };
    let o_at = gpu.zeros(rows * qw * 2);
    let o = &mut ten(o_at, rows, qw, Bf16);
    let mut plan = plan::plan_prefill(
        &qo,
        &kv_indptr,
        &kv_len,
        rows as u32,
        shape,
        live,
        None,
        true,
        false,
        Some(mask_ip),
        &device,
        ws,
    )
    .expect("the prefill plans");
    plan.mask_origin = Some(origin_t);
    plan.stage(&ctx).expect("the plan uploads");
    attn::masked(&ctx, qt, &plan, mask_t, &kvp, None, d, scale, o)
        .unwrap_or_else(|e| panic!("{plane:?} over {qh}/{kvh} heads of {d}: {e}"));
    gpu.sync();
    gpu.down::<u16>(o_at, rows * qw)
}

#[test]
fn a_masked_prefill_reads_its_committed_keys_and_no_key_after_its_row() {
    // (new, committed, buffered): committed parts that span whole kv tiles
    // and parts of one, a buffer and none, a decode row and prompt chunks.
    let reqs: &[(usize, usize, usize)] = &[(1, 300, 7), (33, 257, 0), (70, 129, 40), (17, 0, 9)];
    for heads in [[16, 4, 128], [8, 8, 64], [8, 2, 256]] {
        let from_origin = fire(heads, reqs, Plane::Origin);
        let whole = fire(heads, reqs, Plane::Whole);
        assert_eq!(
            from_origin, whole,
            "{heads:?}: a plane from the boundary lands what the whole-context plane lands"
        );
        let above = fire(heads, reqs, Plane::WholeAboveDiagonal);
        assert_eq!(
            above, whole,
            "{heads:?}: a row reads no key after its own, whatever its bits say"
        );
    }
}
