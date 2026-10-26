#![cfg(feature = "cuda")]

//! A kernel's resolution (its thread's context bound, its unit compiled
//! and its module loaded) beside an open capture leaves the capture whole:
//! the engine compiles a load's generated kernels while serving another
//! model's captures, a body's first capture may be a kernel's first fire,
//! and a test binary resolves kernels on one thread while another captures.
//! Each is ordered here so the resolution lands inside the capture every
//! run.

use std::sync::mpsc;

use crate::common::attn_ref::ten;
use crate::common::{Gpu, captured, forget, replay};
use dtype::Dtype::Bf16;
use kernels_cuda::synth::{self, In, Plane, Region, Synthesized};
use kernels_ir::{Binary, Literal, Op, Shape};

const WIDTH: usize = 256;
const ROWS: usize = 4;

/// `x * c`, a kernel of its own for each `c`.
fn scaled(c: f32) -> Synthesized {
    let region = Region {
        width: WIDTH as u32,
        params: vec![Shape::Row],
        ops: vec![Op::Const(Literal::F32(c)), Op::Binary(Binary::Mul, 0, 1)],
        results: vec![2],
        rounds: vec![],
    };
    synth::synth(&region, &[In::Rows(Bf16)], 1).expect("synthesized")
}

/// A constant no earlier run compiled: the kernel misses every cache.
fn fresh() -> f32 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("after the epoch")
        .subsec_nanos();
    1.0 + (nanos % 1_000_000) as f32 / 1_000_000.0
}

#[test]
#[allow(unsafe_code)]
fn a_cold_load_on_another_thread_leaves_an_open_capture_whole() {
    let mut gpu = Gpu::open();
    let ordinal = gpu.ordinal();
    let x = gpu.zeros(ROWS * WIDTH * 2);
    let y = gpu.zeros(ROWS * WIDTH * 2);
    let ctx = gpu.ctx();
    let warm = scaled(2.0);
    warm.load()
        .expect("the captured kernel loads before the capture");
    let fire = || {
        warm.fire(
            &ctx,
            &[Plane::rows(&ten(x, ROWS, WIDTH, Bf16))],
            &[Plane::rows(&ten(y, ROWS, WIDTH, Bf16))],
            ROWS as u32,
        )
        .expect("the captured kernel fires");
    };
    let cold = scaled(fresh());
    let (go, start) = mpsc::channel::<()>();
    let (loaded, done) = mpsc::channel::<()>();
    let loader = std::thread::spawn(move || {
        // SAFETY: binds this thread to the device the capture runs on.
        unsafe { kernels_cuda::cudarc::runtime::sys::cudaSetDevice(ordinal) };
        start.recv().expect("the capture opens");
        cold.load().expect("the cold kernel compiles and loads");
        loaded.send(()).expect("the capture waits");
    });
    let exec = captured(&ctx, || {
        fire();
        go.send(()).expect("the loader waits");
        done.recv().expect("the loader loads");
        fire();
    });
    loader.join().expect("the loader joins");
    replay(&ctx, exec);
    gpu.sync();
    forget(exec);
    assert!(cold.kernel().is_loaded(), "the cold kernel is loaded");
}

/// A generated kernel loaded before the capture (as a load loads every one)
/// and first launched inside it: its function is loaded with its module,
/// not lazily at that first launch.
#[test]
fn a_kernel_first_fired_inside_a_capture_leaves_it_whole() {
    let mut gpu = Gpu::open();
    let x = gpu.zeros(ROWS * WIDTH * 2);
    let y = gpu.zeros(ROWS * WIDTH * 2);
    let ctx = gpu.ctx();
    let cold = scaled(fresh());
    let fire = |k: Synthesized| {
        k.fire(
            &ctx,
            &[Plane::rows(&ten(x, ROWS, WIDTH, Bf16))],
            &[Plane::rows(&ten(y, ROWS, WIDTH, Bf16))],
            ROWS as u32,
        )
    };
    // a generated kernel refuses a fire before its load; load it, then
    // fire it first inside the capture
    cold.load().expect("the cold kernel loads");
    let exec = captured(&ctx, || {
        fire(cold).expect("the kernel fires inside the capture");
    });
    replay(&ctx, exec);
    gpu.sync();
    forget(exec);
}

/// A carried kernel first fired inside a capture, on a thread no kernel was
/// resolved on before: its resolution binds the thread's context and loads
/// its module there, mid-capture. The binding was a `cudaFree(nullptr)`,
/// which a capture refuses (`cudaErrorStreamCaptureUnsupported`, 900) and
/// is invalidated by; it is a query of the context the capture runs in.
#[test]
fn a_carried_kernel_first_fired_inside_a_capture_leaves_it_whole() {
    let mut gpu = Gpu::open();
    let x = gpu.zeros(ROWS * WIDTH * 2);
    let ctx = gpu.ctx();
    let exec = captured(&ctx, || {
        kernels_cuda::elemwise::pointwise::mul_scalar(
            &ctx,
            fresh(),
            &mut ten(x, ROWS, WIDTH, Bf16),
        )
        .expect("the carried kernel fires inside the capture");
    });
    replay(&ctx, exec);
    gpu.sync();
    forget(exec);
}

/// A cuBLAS handle let go on another thread (a test's `Gpu` dropped) while
/// a capture is open: the handle is kept, not destroyed, so the capture
/// (the GEMM tuner's, timing a shape its cache does not hold) stays whole.
/// `cublasDestroy` synchronizes the whole device, which invalidates every
/// capture open on it, whatever its mode or stream.
#[test]
#[allow(unsafe_code)]
fn a_gpu_dropped_on_another_thread_leaves_an_open_capture_whole() {
    use kernels_cuda::cudarc::runtime::sys as rt;
    let mut gpu = Gpu::open();
    let buf = gpu.zeros(1 << 16);
    let mut stream: rt::cudaStream_t = std::ptr::null_mut();
    // SAFETY: a local out-pointer; the stream is destroyed below.
    unsafe { rt::cudaStreamCreateWithFlags(&raw mut stream, rt::cudaStreamNonBlocking) };
    let (ready, opened) = mpsc::channel::<()>();
    let (go, start) = mpsc::channel::<()>();
    let (fin, done) = mpsc::channel::<()>();
    let dropper = std::thread::spawn(move || {
        let other = Gpu::open();
        ready.send(()).expect("the capture waits");
        start.recv().expect("the capture opens");
        drop(other);
        fin.send(()).expect("the capture waits");
    });
    opened.recv().expect("the other gpu opens");
    // SAFETY: `stream` is live and `buf` a live allocation of 64 KiB.
    let status = unsafe {
        rt::cudaStreamBeginCapture(
            stream,
            rt::cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal,
        );
        rt::cudaMemsetAsync(buf as *mut _, 1, 64, stream);
        go.send(()).expect("the dropper waits");
        done.recv().expect("the dropper drops");
        let mut status = rt::cudaStreamCaptureStatus::cudaStreamCaptureStatusNone;
        rt::cudaStreamIsCapturing(stream, &raw mut status);
        let mut graph: rt::cudaGraph_t = std::ptr::null_mut();
        let end = rt::cudaStreamEndCapture(stream, &raw mut graph);
        assert_eq!(end, rt::cudaError::cudaSuccess, "the capture ends");
        rt::cudaGraphDestroy(graph);
        rt::cudaStreamDestroy(stream);
        status
    };
    dropper.join().expect("the dropper joins");
    assert_eq!(
        status,
        rt::cudaStreamCaptureStatus::cudaStreamCaptureStatusActive
    );
}
