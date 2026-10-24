//! The foundation every other kernel stands on: hiprtc compiles a unit for
//! the device's architecture, the module loads, and a launch writes what the
//! host expects. Skips (passes vacuously) on a machine with no HIP device.
#![allow(unsafe_code)]
#![cfg(feature = "rocm")]

use core::ffi::c_void;

use kernels_rocm::hip;

// hiprtc puts the HIP device runtime in scope itself; an explicit
// `#include <hip/hip_runtime.h>` is an error under it.
const SOURCE: &str = r#"
template <typename T>
__global__ void axpy(T a, const T* x, T* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

// A wavefront-wide sum: each lane adds its neighbours down to lane 0, so the
// answer is only right if the shuffle spans the whole wavefront.
__global__ void wave_sum(const float* x, float* out) {
    float v = x[threadIdx.x];
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        v += __shfl_down(v, offset);
    }
    if (threadIdx.x == 0) out[0] = v;
}
"#;

#[test]
fn a_jit_kernel_runs_on_the_device() {
    let Ok(count) = hip::device_count() else {
        eprintln!("no HIP runtime; skipping");
        return;
    };
    if count == 0 {
        eprintln!("no HIP device; skipping");
        return;
    }
    hip::set_device(0).unwrap();
    let wave = hip::warp_size(0).unwrap();

    let built = hip::compile(
        "smoke",
        SOURCE,
        kernels_rocm::jit::cache::arch(0).expect("the device names its architecture"),
        &["-O3", "-std=c++17"],
        &["axpy<float>", "wave_sum"],
    )
    .unwrap();
    assert!(!built.code.is_empty());
    let module = hip::Module::load(&built.code).unwrap();
    let axpy = module.function(&built.lowered[0]).unwrap();
    let wave_sum = module.function(&built.lowered[1]).unwrap();
    let stream = hip::Stream::new().unwrap();

    let n = 10_000i32;
    let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let y: Vec<f32> = (0..n).map(|i| 1.0 - i as f32).collect();
    let dx = hip::Buffer::from_slice(&x).unwrap();
    let dy = hip::Buffer::from_slice(&y).unwrap();
    let mut a = 2.0f32;
    let (mut px, mut py, mut pn) = (dx.device_ptr(), dy.device_ptr(), n);
    let mut params: [*mut c_void; 4] = [
        (&raw mut a).cast(),
        (&raw mut px).cast(),
        (&raw mut py).cast(),
        (&raw mut pn).cast(),
    ];
    let threads = 256u32;
    let blocks = (n as u32).div_ceil(threads);
    // SAFETY: the params match `axpy<float>(float, const float*, float*, int)`
    // and both buffers outlive the synchronize below.
    unsafe { axpy.launch(&stream, [blocks, 1, 1], [threads, 1, 1], 0, &mut params) }.unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = dy.read().unwrap();
    for (i, v) in got.iter().enumerate() {
        assert_eq!(*v, 2.0 * i as f32 + 1.0 - i as f32, "axpy lane {i}");
    }

    let lanes: Vec<f32> = (0..wave).map(|i| i as f32).collect();
    let dl = hip::Buffer::from_slice(&lanes).unwrap();
    let dout = hip::Buffer::alloc(4).unwrap();
    let (mut pl, mut po) = (dl.device_ptr(), dout.device_ptr());
    let mut params: [*mut c_void; 2] = [(&raw mut pl).cast(), (&raw mut po).cast()];
    // SAFETY: the params match `wave_sum(const float*, float*)`.
    unsafe { wave_sum.launch(&stream, [1, 1, 1], [wave, 1, 1], 0, &mut params) }.unwrap();
    stream.synchronize().unwrap();
    let sum: Vec<f32> = dout.read().unwrap();
    let expect = (wave * (wave - 1) / 2) as f32;
    assert_eq!(sum[0], expect, "a {wave}-lane wavefront sum");
    eprintln!(
        "gfx ok: wavefront {wave}, {} CUs, code object {} bytes",
        hip::multiprocessors(0).unwrap(),
        built.code.len()
    );
}
