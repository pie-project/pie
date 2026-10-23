//! The paged KV append on the first wgpu adapter, beside the hand-written kernels,
//! against a host reference: keys and values written into their pages, and
//! the shared form that writes the keys to both.

use kernels_wgpu::{KvPool, Tensor, attn};
use poem_ir::Dtype;

use crate::gpu::{Gpu, both, noise};

fn st(t: Tensor) -> kernels_slang::Tensor {
    kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype)
}

fn sp(p: &KvPool) -> kernels_slang::KvPool {
    kernels_slang::KvPool {
        keys: st(p.keys),
        values: st(p.values),
        page_indices: st(p.page_indices),
        page_indptr: st(p.page_indptr),
        page_size: p.page_size,
        seq_stride: p.seq_stride,
        head_stride: p.head_stride,
    }
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|x| f64::from(*x)).collect()
}

#[test]
fn the_kv_append_answers_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let hd = 64usize;
    // kv append: 3 rows of 2 kv heads into 2 pages of 4 slots.
    let (kvh, page) = (2usize, 4usize);
    let row = kvh * hd;
    let (k, kr) = gpu.up(&noise(8, 3 * row));
    let (v, vr) = gpu.up(&noise(9, 3 * row));
    let (kr, vr) = (f64s(&kr), f64s(&vr));
    let pages = [1u32, 0, 1];
    let offs = [3u32, 0, 1];
    let (wp, wo) = (gpu.up_u32(&pages), gpu.up_u32(&offs));
    let u = |buf: u32| Tensor::new(buf, 3, 1, Dtype::U32);
    let slots = 2 * page;
    let mut want_k = vec![0.0; slots * row];
    let mut want_v = vec![0.0; slots * row];
    for i in 0..3 {
        let slot = pages[i] as usize * page + offs[i] as usize;
        want_k[slot * row..][..row].copy_from_slice(&kr[i * row..][..row]);
        want_v[slot * row..][..row].copy_from_slice(&vr[i * row..][..row]);
    }
    let pool = |keys: u32, values: u32| KvPool {
        keys: Tensor::new(keys, slots as u32, row as u32, Dtype::Bf16),
        values: Tensor::new(values, slots as u32, row as u32, Dtype::Bf16),
        page_indices: u(wp),
        page_indptr: u(wo),
        page_size: page as i32,
        seq_stride: row as u64,
        head_stride: hd as u64,
    };
    let kt = Tensor::new(k, 3, row as u32, Dtype::Bf16);
    let vt = Tensor::new(v, 3, row as u32, Dtype::Bf16);
    both(
        &mut gpu,
        "kv append",
        &|g| {
            vec![
                (g.zeros(slots * row), slots * row, false),
                (g.zeros(slots * row), slots * row, false),
            ]
        },
        &[want_k.clone(), want_v],
        &|g, o| g.run(|c| attn::kv_append(c, kt, vt, &pool(o[0].0, o[1].0), u(wp), u(wo))),
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::attn::kv_append::kv_append(
                    c,
                    st(kt),
                    st(vt),
                    &sp(&pool(o[0].0, o[1].0)),
                    st(u(wp)),
                    st(u(wo)),
                )
            })
        },
    );
    both(
        &mut gpu,
        "kv append shared",
        &|g| {
            vec![
                (g.zeros(slots * row), slots * row, false),
                (g.zeros(slots * row), slots * row, false),
            ]
        },
        &[want_k.clone(), want_k],
        &|g, o| g.run(|c| attn::kv_append_shared(c, kt, &pool(o[0].0, o[1].0), u(wp), u(wo))),
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::attn::kv_append::kv_append_shared(
                    c,
                    st(kt),
                    &sp(&pool(o[0].0, o[1].0)),
                    st(u(wp)),
                    st(u(wo)),
                )
            })
        },
    );
}
