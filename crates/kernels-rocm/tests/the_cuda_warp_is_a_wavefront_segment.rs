//! The compat layer every carried unit stands on (`shim/pie_hip_compat.h`):
//! a CUDA warp is one 32-lane segment of a 64-lane wavefront, and the PTX
//! the units wrapped is answered lane for lane. Each case runs on the device
//! and is held against a host reference. Skips on a machine with no HIP
//! device.

#![cfg(feature = "rocm")]

use core::ffi::c_void;

use kernels_rocm::hip;

const SOURCE: &str = r#"
#include "prelude/device.cuh"

// Each lane writes what the segment reduction and shuffles answered.
__global__ void warp_ops(int* sum_down, int* xor_bfly, int* idx_bcast, unsigned* ballot,
                         int* up, int* any_all) {
    const int t = threadIdx.x;
    const int lane = t & 31;
    int v = t;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    sum_down[t] = v;
    int x = t;
    for (int m = 1; m < 32; m <<= 1) x += __shfl_xor_sync(0xffffffffu, x, m);
    xor_bfly[t] = x;
    idx_bcast[t] = __shfl_sync(0xffffffffu, t * 10, 5);
    ballot[t] = __ballot_sync(0xffffffffu, (lane % 3) == 0 || t == 40);
    up[t] = __shfl_up_sync(0xffffffffu, t, 1);
    any_all[t] = __any_sync(0xffffffffu, t == 33) * 2 + __all_sync(0xffffffffu, t < 32);
}

// D = A·B (16x8x16, bf16 in, f32 out) through the m16n8k16 fragment map.
__global__ void mma(const unsigned short* A, const unsigned short* B, float* D) {
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    auto pair = [](unsigned short lo, unsigned short hi) {
        return (unsigned)lo | ((unsigned)hi << 16);
    };
    const unsigned a[4] = {
        pair(A[g * 16 + 2 * t], A[g * 16 + 2 * t + 1]),
        pair(A[(g + 8) * 16 + 2 * t], A[(g + 8) * 16 + 2 * t + 1]),
        pair(A[g * 16 + 2 * t + 8], A[g * 16 + 2 * t + 9]),
        pair(A[(g + 8) * 16 + 2 * t + 8], A[(g + 8) * 16 + 2 * t + 9]),
    };
    // B is K x N row-major (16 x 8).
    const unsigned b[2] = {
        pair(B[(2 * t) * 8 + g], B[(2 * t + 1) * 8 + g]),
        pair(B[(2 * t + 8) * 8 + g], B[(2 * t + 9) * 8 + g]),
    };
    const float c[4] = {0.f, 0.f, 0.f, 0.f};
    float d[4];
    pie_hip::mma_m16n8k16_bf16(d, a, b, c);
    if (threadIdx.x < 32) {
        D[g * 8 + 2 * t] = d[0];
        D[g * 8 + 2 * t + 1] = d[1];
        D[(g + 8) * 8 + 2 * t] = d[2];
        D[(g + 8) * 8 + 2 * t + 1] = d[3];
    }
}

// Four 8x8 b16 matrices in LDS, rows 16 bytes apart; lane l names row l%8
// of matrix l/8 and gets back its fragment words.
__global__ void ldmatrix(const unsigned short* src, unsigned* out) {
    __shared__ unsigned short tile[4 * 8 * 8];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) tile[i] = src[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    unsigned r[4];
    pie_hip::ldmatrix_x4(r, &tile[(lane >> 3) * 64 + (lane & 7) * 8]);
    for (int i = 0; i < 4; ++i) out[threadIdx.x * 4 + i] = r[i];
}

__global__ void bits(const unsigned* in, unsigned* out) {
    const unsigned x = in[0], y = in[1];
    out[0] = __byte_perm(x, y, 0x5410);
    out[1] = __byte_perm(x, y, 0x0123);
    out[2] = __byte_perm(x, y, 0x7362);
    out[3] = (unsigned)pie_hip::lop3<(0xf0 & 0xcc) | 0xaa>((int)x, (int)y, (int)in[2]);
    out[4] = (unsigned)pie_hip::lop3<0x96>((int)x, (int)y, (int)in[2]);
}
"#;

struct Device {
    stream: hip::Stream,
}

fn device() -> Option<Device> {
    if hip::device_count().ok()? == 0 {
        eprintln!("no HIP device; skipping");
        return None;
    }
    hip::set_device(0).ok()?;
    Some(Device {
        stream: hip::Stream::new().ok()?,
    })
}

fn launch(dev: &Device, f: hip::Function, block: u32, ptrs: &[u64]) {
    let mut cells: Vec<u64> = ptrs.to_vec();
    let mut params: Vec<*mut c_void> = cells.iter_mut().map(|c| (c as *mut u64).cast()).collect();
    // SAFETY: every kernel here takes only pointers, one per entry of `ptrs`,
    // each a live buffer the caller sized for `block` lanes.
    unsafe { f.launch(&dev.stream, [1, 1, 1], [block, 1, 1], 0, &mut params) }.unwrap();
    dev.stream.synchronize().unwrap();
}

fn bf16(v: f32) -> u16 {
    // round to nearest even, as the device does
    let b = v.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

fn from_bf16(v: u16) -> f32 {
    f32::from_bits(u32::from(v) << 16)
}

#[test]
fn the_cuda_warp_is_a_wavefront_segment() {
    let Some(dev) = device() else { return };
    let (module, names) = kernels_rocm::jit::compile_with_shims(
        "compat_test",
        SOURCE,
        &["warp_ops", "mma", "ldmatrix", "bits"],
    )
    .unwrap();

    a_shuffle_stays_in_its_32_lane_segment(&dev, module.function(&names[0]).unwrap());
    the_mma_fragment_map_multiplies(&dev, module.function(&names[1]).unwrap());
    ldmatrix_hands_each_lane_its_fragment(&dev, module.function(&names[2]).unwrap());
    byte_perm_and_lop3_match_ptx(&dev, module.function(&names[3]).unwrap());
}

fn a_shuffle_stays_in_its_32_lane_segment(dev: &Device, f: hip::Function) {
    const N: usize = 64;
    let bufs: Vec<hip::Buffer> = (0..6).map(|_| hip::Buffer::alloc(N * 4).unwrap()).collect();
    launch(dev, f, N as u32, &bufs.iter().map(hip::Buffer::device_ptr).collect::<Vec<_>>());
    let [sum, bfly, bcast, ballot, up, anyall]: [Vec<i32>; 6] =
        core::array::from_fn(|i| bufs[i].read().unwrap());
    for seg in 0..2 {
        let base = (seg * 32) as i32;
        let total: i32 = (base..base + 32).sum();
        assert_eq!(sum[seg * 32], total, "a down-reduction sums its own segment");
        for lane in 0..32 {
            let t = seg * 32 + lane;
            assert_eq!(bfly[t], total, "a butterfly gives every lane its segment's sum");
            assert_eq!(bcast[t], (base + 5) * 10, "lane 5 of the segment, not of the wavefront");
            let want_up = if lane == 0 { t as i32 } else { t as i32 - 1 };
            assert_eq!(up[t], want_up, "shfl_up keeps lane 0 of a segment in place");
        }
        let mut want: u32 = 0;
        for lane in 0..32 {
            let t = seg * 32 + lane;
            if lane % 3 == 0 || t == 40 {
                want |= 1 << lane;
            }
        }
        for lane in 0..32 {
            assert_eq!(ballot[seg * 32 + lane] as u32, want, "the ballot is the segment's 32 bits");
            let any = i32::from(seg == 1);
            let all = i32::from(seg == 0);
            assert_eq!(anyall[seg * 32 + lane], any * 2 + all, "any/all see only their segment");
        }
    }
}

fn the_mma_fragment_map_multiplies(dev: &Device, f: hip::Function) {
    let a: Vec<u16> = (0..256).map(|i| bf16(((i * 37 % 23) as f32 - 11.0) / 8.0)).collect();
    let b: Vec<u16> = (0..128).map(|i| bf16(((i * 53 % 19) as f32 - 9.0) / 4.0)).collect();
    let da = hip::Buffer::from_slice(&a).unwrap();
    let db = hip::Buffer::from_slice(&b).unwrap();
    let dd = hip::Buffer::alloc(128 * 4).unwrap();
    // Both segments run the same mma; the first writes the result.
    launch(dev, f, 64, &[da.device_ptr(), db.device_ptr(), dd.device_ptr()]);
    let got: Vec<f32> = dd.read().unwrap();
    for m in 0..16 {
        for n in 0..8 {
            let want: f32 = (0..16).map(|k| from_bf16(a[m * 16 + k]) * from_bf16(b[k * 8 + n])).sum();
            let have = got[m * 8 + n];
            assert!((have - want).abs() <= 1e-4 * want.abs().max(1.0), "D[{m}][{n}] = {have}, want {want}");
        }
    }
}

fn ldmatrix_hands_each_lane_its_fragment(dev: &Device, f: hip::Function) {
    let src: Vec<u16> = (0..256).collect();
    let ds = hip::Buffer::from_slice(&src).unwrap();
    let dout = hip::Buffer::alloc(64 * 4 * 4).unwrap();
    launch(dev, f, 64, &[ds.device_ptr(), dout.device_ptr()]);
    let got: Vec<u32> = dout.read().unwrap();
    for t in 0..64 {
        let lane = t % 32;
        for i in 0..4 {
            let row = lane / 4;
            let col = 2 * (lane % 4);
            let at = i * 64 + row * 8 + col;
            let want = u32::from(src[at]) | (u32::from(src[at + 1]) << 16);
            assert_eq!(got[t * 4 + i], want, "lane {t}, matrix {i}");
        }
    }
}

fn byte_perm_and_lop3_match_ptx(dev: &Device, f: hip::Function) {
    let (x, y, z) = (0x4433_2211u32, 0x8877_6655u32, 0x0f0f_00ffu32);
    let din = hip::Buffer::from_slice(&[x, y, z]).unwrap();
    let dout = hip::Buffer::alloc(5 * 4).unwrap();
    launch(dev, f, 1, &[din.device_ptr(), dout.device_ptr()]);
    let got: Vec<u32> = dout.read().unwrap();
    let bytes = [x.to_le_bytes(), y.to_le_bytes()].concat();
    let perm = |s: u32| -> u32 {
        (0..4).map(|i| u32::from(bytes[((s >> (4 * i)) & 7) as usize]) << (8 * i)).sum()
    };
    assert_eq!(got[0], perm(0x5410));
    assert_eq!(got[1], perm(0x0123));
    assert_eq!(got[2], perm(0x7362));
    assert_eq!(got[3], (x & y) | z, "lut (a & b) | c");
    assert_eq!(got[4], x ^ y ^ z, "lut 0x96 is a three-way xor");
}
