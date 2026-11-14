#![cfg(feature = "cuda")]
#![allow(unsafe_code)]

use core::ffi::c_void;

use crate::common::Gpu;
use crate::common::host::cuda_ok as check;
use kernels_cuda::Ctx;
use kernels_cuda::cudarc::runtime::sys as rt;

const DECODED_WEIGHT: &str = "linear.quant.decoded_weight";
const TILE: usize = 4 << 20;

/// A conditional body is captured on a stream of its own and replayed in
/// the main stream's order. The decoded-weight tile laid for the main
/// stream must serve it there; a body that looked for a tile of its own
/// stream would find none laid and be refused.
#[test]
fn a_body_on_its_own_stream_takes_its_owners_tile() {
    let gpu = Gpu::open();
    let main = gpu.ctx();
    let laid = gpu
        .laid(&main, |ctx| ctx.scratch_stream("test", DECODED_WEIGHT, TILE))
        .expect("the main tile is laid");

    let mut body: rt::cudaStream_t = core::ptr::null_mut();
    // SAFETY: `body` is a local out-pointer for the new stream.
    unsafe {
        check(rt::cudaStreamCreate(&raw mut body), "cudaStreamCreate");
    }
    let body = body.cast::<c_void>();
    // SAFETY: the stream lives until the end of this test.
    let own = unsafe { Ctx::on(body).with_slabs(main.slabs()) };
    // SAFETY: as `own`: the same stream, live until the end of this test.
    let in_order = unsafe { Ctx::on(body).with_slabs(main.slabs()) }.in_order_of(main.stream());

    let shared = in_order.scratch_stream("test", DECODED_WEIGHT, TILE / 2);
    let unlaid = own.scratch_stream("test", DECODED_WEIGHT, TILE / 2);
    // SAFETY: the stream created above; nothing was enqueued on it.
    unsafe {
        check(rt::cudaStreamDestroy(body.cast()), "cudaStreamDestroy");
    }

    assert_eq!(
        shared.expect("a body in the main stream's order takes the main tile"),
        laid,
    );
    assert!(
        unlaid.is_err(),
        "a stream nothing was laid for has no tile to take"
    );
}
