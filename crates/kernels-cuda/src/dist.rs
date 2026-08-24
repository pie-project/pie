use crate::jit::Ctx;
use kernels::Refusal;

#[kernels_macros::claims]
impl kernels::points::Dist for Ctx<'_> {}

pub(crate) fn all_reduce_out_of_place(
    _ctx: &Ctx<'_>,
    _src: *const core::ffi::c_void,
    _dst: *mut core::ffi::c_void,
    _elems: i64,
) -> Result<(), Refusal> {
    Err(Refusal::Absent {
        what: "NCCL: `cudarc` is built without its `nccl` feature, so no \
               communicator binding is generated and nothing in this \
               workspace calls one. This is the ABOVE-CROSSOVER arm; below \
               the crossover `comm::all_reduce_bf16` is the one that runs",
    })
}
