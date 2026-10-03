#![cfg(feature = "cuda")]

//! The M1b oracle: the CUDA PTQ1_0 decode-in-dot must read back each decoded
//! weight BIT-EXACT against the host decoder `checkpoint::codec::ptq1_0::
//! decode_block` (which is itself bit-exact against the fork's own dequant).
//!
//! We fire the `bf16 -> f32` qmv with a K-wide IDENTITY of activations: row `e`
//! is one-hot at column `e`, so `y[e][r] = sum_c W[r][c] * I[e][c] = W[r][e]`.
//! One fire reads back the whole decoded weight matrix. A decoded weight is an
//! integer in {-1, 0, +1} times a finite fp16 scale, and the surviving one-hot
//! term is that weight times exactly 1.0, so the read-back is bit-exact.

use core::ffi::c_void;

use checkpoint::codec::ptq1_0::{BLOCK_BYTES, QK_PTQ1_0, decode_block};
use kernels_cuda::cudarc::runtime::sys as rt;
use kernels_cuda::linear::ptq1_0;
use kernels_cuda::tensor::Tensor;
use kernels_cuda::{Ctx, Slabs};
use model_ir::Dtype;

/// Output rows (columns this projection lands); 6 deliberately straddles the
/// 8-row thread-block tile so the `r >= out_vec_size` guard is exercised.
const N: usize = 6;

/// Blocks per weight row; two blocks exercise the block boundary and the `qh`
/// tail twice.
const BLOCKS: usize = 2;

/// Contracted width: a whole number of 128-weight ternary blocks.
const K: usize = BLOCKS * QK_PTQ1_0;

fn check(code: rt::cudaError, call: &str) {
    assert_eq!(
        code,
        rt::cudaError::cudaSuccess,
        "`{call}` answered {code:?}"
    );
}

struct Gpu {
    stream: rt::cudaStream_t,
    slabs: Slabs,
    device: Vec<*mut c_void>,
}

impl Gpu {
    fn open() -> Self {
        kernels_cuda::disk::install(Some(std::path::Path::new(concat!(
            env!("CARGO_TARGET_TMPDIR"),
            "/kernel-cache"
        ))));
        unsafe {
            check(rt::cudaSetDevice(0), "cudaSetDevice");
            let mut stream: rt::cudaStream_t = core::ptr::null_mut();
            check(rt::cudaStreamCreate(&raw mut stream), "cudaStreamCreate");
            let slabs = Slabs::open();
            slabs.attach(stream.cast());
            Self {
                stream,
                slabs,
                device: Vec::new(),
            }
        }
    }

    fn ctx(&self) -> Ctx {
        // SAFETY: the stream outlives every fire in the test, and `Gpu`'s drop
        // synchronizes before destroying it.
        unsafe { Ctx::on(self.stream.cast()).with_slabs(self.slabs) }
    }

    fn zeros(&mut self, bytes: usize) -> u64 {
        unsafe {
            let mut at: *mut c_void = core::ptr::null_mut();
            check(rt::cudaMalloc(&raw mut at, bytes.max(1)), "cudaMalloc");
            check(rt::cudaMemset(at, 0, bytes.max(1)), "cudaMemset");
            self.device.push(at);
            at as u64
        }
    }

    fn up<T: Copy>(&mut self, values: &[T]) -> u64 {
        let bytes = core::mem::size_of_val(values);
        let at = self.zeros(bytes.max(1));
        if bytes > 0 {
            unsafe {
                check(
                    rt::cudaMemcpy(
                        at as *mut c_void,
                        values.as_ptr().cast(),
                        bytes,
                        rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                    ),
                    "cudaMemcpy H2D",
                );
            }
        }
        at
    }

    fn down<T: Copy + Default>(&self, at: u64, count: usize) -> Vec<T> {
        let mut out = vec![T::default(); count];
        unsafe {
            check(
                rt::cudaMemcpy(
                    out.as_mut_ptr().cast(),
                    at as *const c_void,
                    core::mem::size_of_val(out.as_slice()),
                    rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                ),
                "cudaMemcpy D2H",
            );
        }
        out
    }

    fn sync(&self) {
        unsafe {
            check(
                rt::cudaStreamSynchronize(self.stream),
                "cudaStreamSynchronize",
            );
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            rt::cudaStreamSynchronize(self.stream);
            for at in self.device.drain(..) {
                rt::cudaFree(at);
            }
            rt::cudaStreamDestroy(self.stream);
        }
    }
}

fn bf16_bits(value: f32) -> u16 {
    let b = value.to_bits();
    if (b & 0x7fff_ffff) > 0x7f80_0000 {
        return ((b >> 16) | 0x0040) as u16;
    }
    let rounding = 0x7fff + ((b >> 16) & 1);
    (b.wrapping_add(rounding) >> 16) as u16
}

/// Positive, finite fp16 scales (`half::f16::from_bits` and the kernel's
/// `f16_to_f32` agree exactly on every normal value). Positive scales keep the
/// zero weights at `+0.0`, so the one-hot read-back matches the host's
/// `0.0 * d` bit-for-bit.
const SCALE_BITS: [u16; 6] = [
    0x3c00, // 1.0
    0x3800, // 0.5
    0x4000, // 2.0
    0x3400, // 0.25
    0x4200, // 3.0
    0x3000, // 0.125
];

/// Synthesize `N` weight rows of `BLOCKS` 28-byte ternary blocks. The `qs`/`qh`
/// bytes are pseudo-random (so the decoded trits span {-1, 0, +1} across the
/// {16,8} chunks and the `qh` tail); the inline fp16 scale cycles through
/// `SCALE_BITS`. The host `decode_block` is the oracle for the bytes we land.
fn synthesize() -> (Vec<u8>, Vec<[f32; QK_PTQ1_0]>) {
    let row_bytes = BLOCKS * BLOCK_BYTES;
    let mut bytes = vec![0u8; N * row_bytes];
    let mut expect = Vec::with_capacity(N * BLOCKS);
    let mut lcg: u64 = 0x1234_5678_9abc_def0;
    let mut next = || {
        lcg = lcg
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (lcg >> 56) as u8
    };
    for row in 0..N {
        for blk in 0..BLOCKS {
            let at = row * row_bytes + blk * BLOCK_BYTES;
            for b in 0..(BLOCK_BYTES - 2) {
                bytes[at + b] = next();
            }
            let scale = SCALE_BITS[(row * BLOCKS + blk) % SCALE_BITS.len()];
            bytes[at + 26] = (scale & 0xff) as u8;
            bytes[at + 27] = (scale >> 8) as u8;
            let block: [u8; BLOCK_BYTES] = bytes[at..at + BLOCK_BYTES]
                .try_into()
                .expect("a 28-byte block");
            expect.push(decode_block(&block));
        }
    }
    (bytes, expect)
}

#[test]
fn the_ptq1_0_cuda_decode_in_dot_is_bit_exact() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }

    let (weight_bytes, expect) = synthesize();

    // A K-wide identity of activations: row `e` is one-hot at column `e`.
    let one = bf16_bits(1.0);
    let mut act = vec![0u16; K * K];
    for e in 0..K {
        act[e * K + e] = one;
    }

    let mut gpu = Gpu::open();
    let w = gpu.up(&weight_bytes);
    let x = gpu.up(&act);
    let y = gpu.zeros(K * N * core::mem::size_of::<f32>());

    let w_tensor = Tensor::new(w, N as u32, K as u32, Dtype::Ptq1_0);
    let x_tensor = Tensor::new(x, K as u32, K as u32, Dtype::Bf16);
    let mut y_tensor = Tensor::new(y, K as u32, N as u32, Dtype::F32);

    ptq1_0::matmul(&gpu.ctx(), x_tensor, w_tensor, &mut y_tensor)
        .expect("the bf16 -> f32 PTQ1_0 qmv fires");
    gpu.sync();

    let got: Vec<f32> = gpu.down(y, K * N);

    for row in 0..N {
        for e in 0..K {
            let want = expect[row * BLOCKS + e / QK_PTQ1_0][e % QK_PTQ1_0];
            let read = got[e * N + row];
            assert_eq!(
                read.to_bits(),
                want.to_bits(),
                "row {row} element {e}: CUDA read {read:?} ({:#010x}) != host decode {want:?} \
                 ({:#010x})",
                read.to_bits(),
                want.to_bits(),
            );
        }
    }
}
