#![cfg(feature = "cuda")]

use core::ffi::c_void;

use crate::common::Gpu;
use kernels_cuda::Ctx;
use kernels_cuda::cudarc::runtime::sys as rt;

const DECODED_WEIGHT: &str = "linear.quant.decoded_weight";
const TILE: u64 = 4 << 20;

fn check(code: rt::cudaError, call: &str) {
    assert_eq!(
        code,
        rt::cudaError::cudaSuccess,
        "`{call}` answered {code:?}"
    );
}

/// A conditional body is captured on a stream of its own and replayed in
/// the main stream's order. The decoded-weight tile a load warms on the
/// main stream must serve it there: a capture cannot grow a slab, so a body
/// that looked for its own tile found none and every prefill body that
/// decodes a plane was refused at load.
#[test]
fn a_body_captured_on_its_own_stream_takes_its_owners_tile() {
    let gpu = Gpu::open();
    let main = gpu.ctx();
    kernels_cuda::linear::quant::warm_decoded_weight(&main, TILE).expect("the main tile warms");
    let warmed = main
        .scratch_stream("test", DECODED_WEIGHT, TILE as usize)
        .expect("the warmed tile is there");

    let mut body: rt::cudaStream_t = core::ptr::null_mut();
    unsafe {
        check(rt::cudaStreamCreate(&raw mut body), "cudaStreamCreate");
    }
    let body = body.cast::<c_void>();
    // SAFETY: the stream lives until the end of this test.
    let own = unsafe { Ctx::on(body).with_slabs(main.slabs()) };
    let in_order = unsafe { Ctx::on(body).with_slabs(main.slabs()) }.in_order_of(main.stream());

    unsafe {
        check(
            rt::cudaStreamBeginCapture(
                body.cast(),
                rt::cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal,
            ),
            "cudaStreamBeginCapture",
        );
    }
    let shared = in_order.scratch_stream("test", DECODED_WEIGHT, TILE as usize / 2);
    let unwarmed = own.scratch_stream("test", DECODED_WEIGHT, TILE as usize / 2);
    let mut graph: rt::cudaGraph_t = core::ptr::null_mut();
    unsafe {
        check(
            rt::cudaStreamEndCapture(body.cast(), &raw mut graph),
            "cudaStreamEndCapture",
        );
        check(rt::cudaGraphDestroy(graph), "cudaGraphDestroy");
        check(rt::cudaStreamDestroy(body.cast()), "cudaStreamDestroy");
    }

    assert_eq!(
        shared.expect("a body in the main stream's order takes the main tile"),
        warmed,
    );
    assert!(
        unwarmed.is_err(),
        "a stream nothing warmed has no tile for a capture to take"
    );
}
