#![cfg(feature = "cuda")]

//! Qwen3.5's attention prologue in one launch (`qg_qknorm_rope_append`)
//! lands what its five launches land: the cut, the two per-head norms, the
//! partial rotation and the cache append. The gate and the cached values
//! match bit for bit; q and the cached keys within a bf16 rounding (the
//! norm's sum runs in another order).

use crate::common;

use common::{Gpu, Lcg, from_bf16};
use dtype::Dtype;
use kernels_cuda::attn;
use kernels_cuda::elemwise::{norm, rope};
use kernels_cuda::layout;
use kernels_cuda::tensor::{KvPool, RaggedTensor, Tensor};

const PAGE: usize = 16;
const PAGES: usize = 12;

/// Qwen3.8-27B's attention head: 24 q heads (with their gates), 4 kv heads
/// at 256, the first 64 dimensions turned, one row's cell invalid.
#[test]
fn the_fused_qwen_prologue_lands_what_its_five_launches_land() {
    let (q_heads, kv_heads, d, rotary) = (24u32, 4u32, 256u32, 64u32);
    let span = rotary / 2;
    let theta = 10_000_000.0f32;
    let rows = 7usize;
    let mut gpu = Gpu::open();
    let mut lcg = Lcg::seeded(0x51c);
    let (qg_bits, _) = lcg.row(rows * (2 * q_heads * d) as usize);
    let (k_bits, _) = lcg.row(rows * (kv_heads * d) as usize);
    let (v_bits, _) = lcg.row(rows * (kv_heads * d) as usize);
    let (qn_bits, _) = lcg.row(d as usize);
    let (kn_bits, _) = lcg.row(d as usize);
    let positions: Vec<i32> = (0..rows as i32).map(|r| 211 * r + 3).collect();
    let pages: Vec<u32> = vec![3, 9, 0, 11, 5, 3, 7];
    let offsets: Vec<u32> = vec![0, 15, 4, 2, 9, 1, 16];
    let valid: Vec<u8> = vec![1, 1, 1, 1, 0, 1, 1];
    let indptr: Vec<i32> = vec![0, rows as i32];
    let width = (kv_heads * d) as usize;
    let cells = PAGES * PAGE;
    let qw = (q_heads * d) as usize;
    let run = |gpu: &mut Gpu, fused: bool| -> [Vec<u16>; 4] {
        let qg = gpu.up(&qg_bits);
        let k = gpu.up(&k_bits);
        let v = gpu.up(&v_bits);
        let q_norm = Tensor::new(gpu.up(&qn_bits), d, 1, Dtype::Bf16);
        let k_norm = Tensor::new(gpu.up(&kn_bits), d, 1, Dtype::Bf16);
        let q_split = gpu.zeros(rows * qw * 2);
        let gate = gpu.zeros(rows * qw * 2);
        let q_out = gpu.zeros(rows * qw * 2);
        let k_out = gpu.zeros(rows * width * 2);
        let keys = gpu.zeros(cells * width * 2);
        let values = gpu.zeros(cells * width * 2);
        let pool = KvPool {
            keys: Tensor::new(keys, cells as u32, width as u32, Dtype::Bf16),
            values: Tensor::new(values, cells as u32, width as u32, Dtype::Bf16),
            bf16_keys: Tensor::ABSENT,
            bf16_values: Tensor::ABSENT,
            key_scales: Tensor::ABSENT,
            value_scales: Tensor::ABSENT,
            page_indices: Tensor::ABSENT,
            page_indptr: Tensor::ABSENT,
            last_page_lens: Tensor::ABSENT,
            row_valid: Tensor::new(gpu.up(&valid), rows as u32, 1, Dtype::U8),
            env_min: Tensor::ABSENT,
            env_max: Tensor::ABSENT,
            has_envelopes: false,
            page_size: PAGE as i32,
            seq_stride: width as i64,
            head_stride: i64::from(d),
            layout: 0,
            scheme_byte: 0,
            block_size: 0,
            max_pages_per_request: PAGES as i32,
            pages_in_batch: PAGES as i32,
        };
        let t = |at: u64, w: usize| Tensor::new(at, rows as u32, w as u32, Dtype::Bf16);
        let pos = Tensor::new(gpu.up(&positions), rows as u32, 1, Dtype::I32);
        let wp = Tensor::new(gpu.up(&pages), rows as u32, 1, Dtype::U32);
        let wo = Tensor::new(gpu.up(&offsets), rows as u32, 1, Dtype::U32);
        let ctx = gpu.ctx();
        let (q_eps, k_eps) = (1e-6f32, 1e-6f32);
        if fused {
            let append = rope::YarnAppend {
                v: t(v, width),
                pool: &pool,
                write_page: wp,
                write_offset: wo,
            };
            assert!(append.fits(), "a plain bf16 pool takes the fused write");
            rope::qg_qknorm_rope_append(
                &ctx,
                t(qg, 2 * qw),
                t(k, width),
                q_norm,
                k_norm,
                pos,
                d,
                rotary,
                span,
                theta,
                q_eps,
                k_eps,
                &mut t(q_out, qw),
                &mut t(gate, qw),
                append,
            )
            .expect("the fused prologue fires");
        } else {
            layout::split_q_gate(
                &ctx,
                t(qg, 2 * qw),
                d,
                &mut t(q_split, qw),
                &mut t(gate, qw),
            )
            .expect("the cut fires");
            norm::rmsnorm_per_head_plus_one(
                &ctx,
                t(q_split, qw),
                q_norm,
                d,
                q_eps,
                &mut t(q_out, qw),
            )
            .expect("the q norm fires");
            norm::rmsnorm_per_head_plus_one(
                &ctx,
                t(k, width),
                k_norm,
                d,
                k_eps,
                &mut t(k_out, width),
            )
            .expect("the k norm fires");
            rope::partial(
                &ctx,
                &mut t(q_out, qw),
                &mut t(k_out, width),
                pos,
                rotary,
                d,
                theta,
                span,
            )
            .expect("the rotation fires");
            attn::kv_append(
                &ctx,
                RaggedTensor {
                    data: t(k_out, width),
                    indptr: Tensor::new(gpu.up(&indptr), 2, 1, Dtype::I32),
                },
                t(v, width),
                &pool,
                wp,
                wo,
            )
            .expect("the append fires");
        }
        gpu.sync();
        [
            gpu.down(q_out, rows * qw),
            gpu.down(gate, rows * qw),
            gpu.down(keys, cells * width),
            gpu.down(values, cells * width),
        ]
    };
    let apart = run(&mut gpu, false);
    let fused = run(&mut gpu, true);
    assert!(
        apart[2].iter().any(|&b| b != 0),
        "the five launches landed keys"
    );
    assert_eq!(fused[1], apart[1], "the gate");
    assert_eq!(fused[3], apart[3], "the cached values");
    for (what, i) in [("q", 0usize), ("the cached keys", 2)] {
        for (at, (&g, &w)) in fused[i].iter().zip(&apart[i]).enumerate() {
            let (g, w) = (from_bf16(g), from_bf16(w));
            assert!(
                (g - w).abs() <= w.abs() * (1.0 / 64.0) + 1e-3,
                "{what} element {at}: fused {g} against {w}"
            );
        }
    }
}
