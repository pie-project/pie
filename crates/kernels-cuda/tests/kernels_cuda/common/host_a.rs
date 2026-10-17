//! Host references and fixtures several kernel tests share.
#![allow(dead_code)]

use super::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::cudarc::runtime::sys as rt;
use kernels_cuda::tensor::Tensor;

/// Asserts a runtime call answered success.
pub fn cuda_ok(code: rt::cudaError, call: &str) {
    assert_eq!(
        code,
        rt::cudaError::cudaSuccess,
        "`{call}` answered {code:?}"
    );
}

/// Copies `values` over the device bytes at `at`.
pub fn upload_to<T: Copy>(at: u64, values: &[T]) {
    let _legacy = super::legacy();
    let code = unsafe {
        rt::cudaMemcpy(
            at as *mut core::ffi::c_void,
            values.as_ptr().cast(),
            core::mem::size_of_val(values),
            rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
        )
    };
    cuda_ok(code, "cudaMemcpy H2D");
}

/// `tokens` rows of `top` experts each, spread over `experts`.
pub fn routes_of(tokens: usize, experts: usize, top: usize) -> Vec<i32> {
    (0..tokens)
        .flat_map(|t| (0..top).map(move |j| ((t * 5 + j * 7 + t / 3) % experts) as i32))
        .collect()
}

/// Spatial attention over `c`-wide rows, each box's voxels attending within
/// runs of `seg` frames (`0`: the whole box), in f64.
pub fn spatial_attention_ref(
    [q, k, v]: [&[f32]; 3],
    boxes: &[super::spatial::Box3],
    c: usize,
    seg: usize,
    scale: f32,
) -> Vec<f32> {
    let rows = q.len() / c;
    let mut y = vec![0f32; rows * c];
    let mut off = 0usize;
    for b in boxes {
        let plane = b.plane();
        let per = if seg == 0 { b.t } else { seg };
        let mut frame = 0usize;
        while frame < b.t {
            let last = (frame + per).min(b.t);
            let (begin, end) = (off + frame * plane, off + last * plane);
            for i in begin..end {
                let s: Vec<f64> = (begin..end)
                    .map(|j| {
                        let dot: f64 = (0..c)
                            .map(|e| f64::from(q[i * c + e]) * f64::from(k[j * c + e]))
                            .sum();
                        dot * f64::from(scale)
                    })
                    .collect();
                let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let p: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
                let l: f64 = p.iter().sum();
                for e in 0..c {
                    let acc: f64 = (0..end - begin)
                        .map(|jj| p[jj] * f64::from(v[(begin + jj) * c + e]))
                        .sum();
                    y[i * c + e] = (acc / l) as f32;
                }
            }
            frame = last;
        }
        off += b.voxels();
    }
    y
}

/// A zeroed bf16 `[rows][width]` tensor.
pub fn bf16_zeros(gpu: &mut Gpu, rows: usize, width: usize) -> Tensor {
    Tensor::new(
        gpu.zeros(rows * width * 2),
        rows as u32,
        width as u32,
        Dtype::Bf16,
    )
}

/// A random bf16 `[rows][width]` tensor, with its values.
pub fn bf16_rows(gpu: &mut Gpu, lcg: &mut Lcg, rows: usize, width: usize) -> (Tensor, Vec<f32>) {
    let (raw, exact) = lcg.row(rows * width);
    (
        Tensor::new(gpu.up(&raw), rows as u32, width as u32, Dtype::Bf16),
        exact,
    )
}
