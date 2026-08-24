use crate::jit::Ctx;
use crate::jit::abi::Tensor;
use crate::jit::abi::bf16;
use crate::views::GemmGroups;
use kernels::Refusal;
use kernels::raises::Struct;
use kernels::routine::{Const, In, InOut, Out};

use core::ffi::c_void;

pub mod absorb;

#[cfg(feature = "_cuda")]
pub mod dense;

// GATED WITH ITS CALLER. `dense`'s autotuner is the only thing that picks
// this launch, and that module is `_cuda`'s.
#[cfg(feature = "_cuda")]
pub mod gemv;

pub mod lora;

#[cfg(feature = "_cuda")]
pub mod quant;

/// The plane's payload, forgotten down to an address. cuBLAS's routine below
/// takes `Tensor<c_void>` because the dtype it multiplies in is the handle's
/// business, not the operand's; a point quantifies over `T: Scalar`, so the
/// three bodies hand the pointer across and keep nothing else.
fn opaque_in<T: kernels::points::Scalar>(x: In<Tensor<T>>) -> In<Tensor<c_void>> {
    In {
        ptr: x.ptr.cast(),
        rows: x.rows,
        width: x.width,
    }
}

fn opaque_const<T: kernels::points::Scalar>(w: Const<Tensor<T>>) -> Const<Tensor<c_void>> {
    Const { v: w.v.cast() }
}

fn opaque_out<T: kernels::points::Scalar>(y: Out<Tensor<T>>) -> Out<Tensor<c_void>> {
    Out {
        ptr: y.ptr.cast(),
        rows: y.rows,
        width: y.width,
    }
}

/// The `Gemm` family, claimed. `matmul` delegates to the cuBLAS routine that
/// already fires it; the two purpose-wearing points claim by calling this
/// plane's own `matmul`, which is what the retired `canon::DEFAULTS` table
/// used to say from a distance.
///
/// `attention_landing`'s `layer` is stated and unread here: cuBLAS needs no
/// layer, and the driver finds the attention output by the statement's own
/// layer tag, which the DSL records beside the op rather than in its params.
#[kernels_macros::claims]
impl kernels::points::Gemm for Ctx<'_> {
    fn matmul<T: kernels::points::Scalar>(
        &self,
        act: In<Tensor<T>>,
        w: Const<Tensor<T>>,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        act_x_wt_bf16(self, opaque_in(act), opaque_const(w), opaque_out(y))
    }

    fn lm_head<T: kernels::points::Scalar>(
        &self,
        act: In<Tensor<T>>,
        w: Const<Tensor<T>>,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        self.matmul(act, w, y)
    }

    fn attention_landing<T: kernels::points::Scalar>(
        &self,
        act: In<Tensor<T>>,
        w: Const<Tensor<T>>,
        layer: u32,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = layer;
        self.matmul(act, w, y)
    }
}

pub(crate) fn act_x_wt_bf16(
    ctx: &Ctx<'_>,
    act: In<Tensor<c_void>>,
    w: Const<Tensor<c_void>>,
    y: Out<Tensor<c_void>>,
) -> Result<(), Refusal> {
    let beta = 0.0f32;
    act_x_wt_bf16_beta(ctx, act, w, y, beta)
}

fn act_x_wt_bf16_beta(
    ctx: &Ctx<'_>,
    act: In<Tensor<c_void>>,
    w: Const<Tensor<c_void>>,
    y: Out<Tensor<c_void>>,
    beta: f32,
) -> Result<(), Refusal> {
    let dst = crate::layout::stated(y.all("n or k"))?;
    let src = crate::layout::stated(act.all("n or k"))?;
    let m = dst.rows;
    let n = dst.stride;
    let k = src.stride;

    #[cfg(feature = "_cuda")]
    unsafe {
        dense::act_x_wt_bf16(ctx.cublas()?, act.ptr, w.v, y.ptr, m, n.0, k.0, beta);
    }
    #[cfg(not(feature = "_cuda"))]
    let _ = (ctx.cublas()?, act.ptr, w.v, y.ptr, m, n.0, k.0, beta);
    Ok(())
}

pub fn act_x_wt_bf16_out_fp32(
    ctx: &Ctx<'_>,
    act: In<Tensor<c_void>>,
    w: Const<Tensor<c_void>>,
    y: Out<Tensor<f32>>,
) -> Result<(), Refusal> {
    let dst = crate::layout::stated(y.all("n or k"))?;
    let src = crate::layout::stated(act.all("n or k"))?;
    let m = dst.rows;
    let n = dst.stride;
    let k = src.stride;

    #[cfg(feature = "_cuda")]
    unsafe {
        dense::act_x_wt_bf16_out_fp32(ctx.cublas()?, act.ptr, w.v, y.ptr, m, n.0, k.0);
    }
    #[cfg(not(feature = "_cuda"))]
    let _ = (ctx.cublas()?, act.ptr, w.v, y.ptr, m, n.0, k.0);
    Ok(())
}

pub fn grouped_act_x_wt_bf16(
    ctx: &Ctx<'_>,
    group_count: Const<i32>,
    beta: Const<f32>,
    n: Const<i32>,
    k: Const<i32>,
    groups: In<Struct<GemmGroups>>,
) -> Result<(), Refusal> {
    if groups.ptr.is_null() {
        return Err(Refusal::Null {
            what: "the grouped-GEMM view this statement names",
        });
    }
    let groups = unsafe { &*groups.ptr };
    let group_count = *group_count;
    let beta = *beta;
    let act_ptrs_dev = groups.act_ptrs;
    let w_ptrs_dev = groups.weight_ptrs;
    let y_ptrs_dev = groups.out_ptrs;
    let m_array_host = groups.m_array_host;
    let n = *n;
    let k = *k;
    let handle = ctx.cublas()?;

    #[cfg(feature = "_cuda")]
    unsafe {
        dense::grouped_act_x_wt_bf16(
            handle,
            act_ptrs_dev,
            w_ptrs_dev,
            y_ptrs_dev,
            m_array_host,
            group_count,
            n,
            k,
            beta,
        );
    }
    #[cfg(not(feature = "_cuda"))]
    let _ = (
        handle,
        act_ptrs_dev,
        w_ptrs_dev,
        y_ptrs_dev,
        m_array_host,
        group_count,
        n,
        k,
        beta,
    );
    Ok(())
}

pub fn act_x_wt_bias_bf16(
    ctx: &Ctx<'_>,
    act: In<Tensor<c_void>>,
    w: Const<Tensor<c_void>>,
    bias: Const<Tensor<c_void>>,
    y: Out<Tensor<c_void>>,
) -> Result<(), Refusal> {
    let beta = 0.0f32;

    let dst = crate::layout::stated(y.all("n or k"))?;
    let src = crate::layout::stated(act.all("n or k"))?;
    let m = dst.rows;
    let n = dst.stride;
    let k = src.stride;

    #[cfg(feature = "_cuda")]
    unsafe {
        dense::act_x_wt_bf16(ctx.cublas()?, act.ptr, w.v, y.ptr, m, n.0, k.0, beta);
    }
    #[cfg(not(feature = "_cuda"))]
    let _ = (ctx.cublas()?, act.ptr, w.v, y.ptr, m, n.0, k.0, beta);
    if bias.v.is_null() {
        return Ok(());
    }

    if m <= 0 {
        return Ok(());
    }

    crate::norm::add_bias::<bf16>(
        ctx,
        InOut {
            ptr: dst.ptr.cast::<bf16>(),
            rows: dst.rows,
            width: dst.width,
        },
        Const {
            v: bias.v.cast::<bf16>(),
        },
    )
}



