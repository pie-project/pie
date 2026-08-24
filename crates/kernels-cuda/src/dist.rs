use crate::jit::Ctx;
use crate::jit::abi::Tensor;
use kernels::Refusal;
use kernels::routine::InOut;

fn no_nccl(what: &'static str) -> Refusal {
    let _ = what;
    Refusal::Absent {
        what: "NCCL: `cudarc` is built without its `nccl` feature, so no \
               communicator binding is generated and nothing in this \
               workspace calls one. This is the ABOVE-CROSSOVER arm; below \
               the crossover `comm::all_reduce_bf16` is the one that runs",
    }
}

/// The `Dist` family, claimed. The point quantifies over `T: Scalar` and a
/// communicator takes an address and a count, so nothing here reads an
/// element: the rectangle is checked and the above-crossover arm refuses.
#[kernels_macros::claims]
impl kernels::points::Dist for Ctx<'_> {
    fn all_reduce<T: kernels::points::Scalar>(&self, buf: InOut<Tensor<T>>) -> Result<(), Refusal> {
        buf.all("out_width(0)")?;
        Err(no_nccl("all_reduce"))
    }
}

/// The out-of-place arm [`crate::comm`] falls back to when its one-shot
/// plane declines.
pub(crate) fn all_reduce_out_of_place(
    _ctx: &Ctx<'_>,
    _src: *const core::ffi::c_void,
    _dst: *mut core::ffi::c_void,
    _elems: i64,
) -> Result<(), Refusal> {
    Err(no_nccl("all_reduce_out"))
}
