#![cfg(feature = "cuda")]

use crate::common;

use common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::attn;
use kernels_cuda::elemwise::rope;
use kernels_cuda::tensor::{KvPool, RaggedTensor, Tensor};

const PAGE: usize = 16;
const PAGES: usize = 12;

/// gpt-oss's rotation (64 q heads, 8 kv heads at 64, YaRN at factor 32) over
/// a few rows, each row's cell somewhere in the pool, one row's cell invalid.
#[test]
fn the_fused_yarn_write_lands_what_the_rotation_and_the_append_land() {
    for interleaved in [false, true] {
        check(interleaved);
    }
}

fn check(interleaved: bool) {
    let (q_heads, kv_heads, d) = (64u32, 8u32, 64u32);
    let rows = 7usize;
    let mut gpu = Gpu::open();
    let mut lcg = Lcg::seeded(0x7a2);
    let (q_bits, _) = lcg.row(rows * (q_heads * d) as usize);
    let (k_bits, _) = lcg.row(rows * (kv_heads * d) as usize);
    let (v_bits, _) = lcg.row(rows * (kv_heads * d) as usize);
    let positions: Vec<i32> = (0..rows as i32).map(|r| 37 * r + 5).collect();
    let pages: Vec<u32> = vec![3, 9, 0, 11, 5, 3, 7];
    let offsets: Vec<u32> = vec![0, 15, 4, 2, 9, 1, 16];
    let valid: Vec<u8> = vec![1, 1, 1, 1, 0, 1, 1];
    let indptr: Vec<i32> = vec![0, rows as i32];
    let width = (kv_heads * d) as usize;
    let cells = PAGES * PAGE;
    let run = |gpu: &mut Gpu, fused: bool| -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
        let q = gpu.up(&q_bits);
        let k = gpu.up(&k_bits);
        let v = gpu.up(&v_bits);
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
        let mut qt = Tensor::new(q, rows as u32, q_heads * d, Dtype::Bf16);
        let mut kt = Tensor::new(k, rows as u32, kv_heads * d, Dtype::Bf16);
        let vt = Tensor::new(v, rows as u32, kv_heads * d, Dtype::Bf16);
        let pos = Tensor::new(gpu.up(&positions), rows as u32, 1, Dtype::I32);
        let wp = Tensor::new(gpu.up(&pages), rows as u32, 1, Dtype::U32);
        let wo = Tensor::new(gpu.up(&offsets), rows as u32, 1, Dtype::U32);
        let ctx = gpu.ctx();
        let params = (d, 150_000.0, 32.0, 32.0, 1.0, 1.347, 4096u32, interleaved);
        if fused {
            let append = rope::YarnAppend {
                v: vt,
                pool: &pool,
                write_page: wp,
                write_offset: wo,
            };
            assert!(append.fits(), "a plain bf16 pool takes the fused write");
            rope::yarn_append(
                &ctx, &mut qt, &mut kt, pos, params.0, params.1, params.2, params.3, params.4,
                params.5, params.6, params.7, append,
            )
            .expect("the fused rotation fires");
        } else {
            rope::yarn(
                &ctx, &mut qt, &mut kt, pos, params.0, params.1, params.2, params.3, params.4,
                params.5, params.6, params.7,
            )
            .expect("the rotation fires");
            attn::kv_append(
                &ctx,
                RaggedTensor {
                    data: kt,
                    indptr: Tensor::new(gpu.up(&indptr), 2, 1, Dtype::I32),
                },
                vt,
                &pool,
                wp,
                wo,
            )
            .expect("the append fires");
        }
        gpu.sync();
        (
            gpu.down(q, rows * (q_heads * d) as usize),
            gpu.down(k, rows * width),
            gpu.down(keys, cells * width),
            gpu.down(values, cells * width),
        )
    };
    let apart = run(&mut gpu, false);
    let fused = run(&mut gpu, true);
    assert!(apart.2.iter().any(|&b| b != 0), "the pair landed keys");
    assert_eq!(fused.0, apart.0, "interleaved {interleaved}: q");
    assert_eq!(fused.1, apart.1, "interleaved {interleaved}: k");
    assert_eq!(fused.2, apart.2, "interleaved {interleaved}: the cached keys");
    assert_eq!(fused.3, apart.3, "interleaved {interleaved}: the cached values");
}
