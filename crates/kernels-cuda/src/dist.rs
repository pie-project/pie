use crate::jit::Ctx;
use crate::jit::abi::Tensor;
use crate::jit::abi::bf16;
use kernels::Refusal;
use kernels::routine::{In, InOut, Out};
use kernels_macros::routine;

fn no_nccl(what: &'static str) -> Refusal {
    let _ = what;
    Refusal::Absent {
        what: "NCCL: `cudarc` is built without its `nccl` feature, so no \
               communicator binding is generated and nothing in this \
               workspace calls one. This is the ABOVE-CROSSOVER arm; below \
               the crossover `comm::all_reduce_bf16` is the one that runs",
    }
}

#[routine(whole, out(buf = like(buf)))]
pub fn all_reduce_bf16(ctx: &Ctx<'_>, buf: InOut<Tensor<bf16>>) -> Result<(), Refusal> {
    let r = buf.all("out_width(0)")?;
    all_reduce_in_place(ctx, r.ptr.cast(), i64::from(r.elements()))
}

pub fn all_reduce_in_place(
    _ctx: &Ctx<'_>,
    _buf: *mut core::ffi::c_void,
    _elems: i64,
) -> Result<(), Refusal> {
    Err(no_nccl("all_reduce"))
}

#[routine(whole, out(dst = like(src)))]
pub fn all_reduce_bf16_out(
    ctx: &Ctx<'_>,
    src: In<Tensor<bf16>>,
    dst: Out<Tensor<bf16>>,
) -> Result<(), Refusal> {
    let d = dst.all("out_width(0)")?;
    all_reduce_out_of_place(ctx, src.ptr.cast(), d.ptr.cast(), i64::from(d.elements()))
}

pub fn all_reduce_out_of_place(
    _ctx: &Ctx<'_>,
    _src: *const core::ffi::c_void,
    _dst: *mut core::ffi::c_void,
    _elems: i64,
) -> Result<(), Refusal> {
    Err(no_nccl("all_reduce_out"))
}

#[routine(whole)]
pub fn all_gather_bf16(
    ctx: &Ctx<'_>,
    src: In<Tensor<bf16>>,
    dst: Out<Tensor<bf16>>,
) -> Result<(), Refusal> {
    let s = src.all("in_width(0)")?;
    all_gather(ctx, s.ptr.cast(), dst.ptr.cast(), i64::from(s.elements()))
}

pub fn all_gather(
    _ctx: &Ctx<'_>,
    _src: *const core::ffi::c_void,
    _dst: *mut core::ffi::c_void,
    _elems_per_rank: i64,
) -> Result<(), Refusal> {
    Err(no_nccl("all_gather"))
}
