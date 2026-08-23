use crate::jit::Ctx;
use crate::jit::abi::Tensor;
use crate::jit::abi::bf16;
use crate::views::GemmGroups;
use kernels::Refusal;
use kernels::raises::Struct;
use kernels::routine::{Const, In, InOut, Out};
use kernels_macros::routine;

use core::ffi::c_void;

pub mod absorb;

#[cfg(feature = "_cuda")]
pub mod dense;

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

#[routine(canon = "gemm.matmul", out(y = rows(act) x weight(w)))]
pub fn act_x_wt_bf16(
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

#[routine(out(y = rows(act) x weight(w)))]
pub fn act_x_w(
    ctx: &Ctx<'_>,
    act: In<Tensor<c_void>>,
    w: Const<Tensor<c_void>>,
    y: Out<Tensor<c_void>>,
) -> Result<(), Refusal> {
    let beta = 0.0f32;
    act_x_wt_bf16_beta(ctx, act, w, y, beta)
}

#[routine(canon = "gemm.matmul_acc", out(y = like(y)))]
pub fn act_x_w_acc(
    ctx: &Ctx<'_>,
    act: In<Tensor<c_void>>,
    w: Const<Tensor<c_void>>,
    y: InOut<Tensor<c_void>>,
) -> Result<(), Refusal> {
    let beta = 1.0f32;

    let dst = Out {
        ptr: y.ptr,
        rows: y.rows,
        width: y.width,
    };
    act_x_wt_bf16_beta(ctx, act, w, dst, beta)
}

#[routine]
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

#[routine(whole)]
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

#[routine(out(y = rows(act) x weight(w)))]
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

#[routine(untraced)]
pub fn act_x_wt_channel_scaled(
    ctx: &Ctx<'_>,
    act: In<Tensor<c_void>>,
    w: Const<Tensor<c_void>>,
    w_dtype: i32,
    w_nbytes: usize,
    scale: Const<Tensor<c_void>>,
    scale_dtype: i32,
    scale_numel: usize,
    zero_point: Const<Tensor<c_void>>,
    channel_axis: i32,
    y: Out<Tensor<c_void>>,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) -> Result<(), Refusal> {
    let handle = ctx.cublas()?;
    let (act, w, scale, zero_point, y) = (act.ptr, w.v, scale.v, zero_point.v, y.ptr);

    #[cfg(feature = "_cuda")]
    unsafe {
        quant::act_x_wt_channel_scaled(
            handle,
            act,
            w,
            w_dtype,
            w_nbytes,
            scale,
            scale_dtype,
            scale_numel,
            zero_point,
            channel_axis,
            y,
            m,
            n,
            k,
            beta,
        );
    }
    #[cfg(not(feature = "_cuda"))]
    let _ = (
        handle,
        act,
        w,
        w_dtype,
        w_nbytes,
        scale,
        scale_dtype,
        scale_numel,
        zero_point,
        channel_axis,
        y,
        m,
        n,
        k,
        beta,
    );
    Ok(())
}

#[routine(untraced)]
pub fn act_x_wt_grouped_scaled(
    ctx: &Ctx<'_>,
    act: In<Tensor<c_void>>,
    w: Const<Tensor<c_void>>,
    w_dtype: i32,
    w_nbytes: usize,
    scale: Const<Tensor<c_void>>,
    scale_dtype: i32,
    scale_numel: usize,
    zero_point: Const<Tensor<c_void>>,
    group_size: i32,
    y: Out<Tensor<c_void>>,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) -> Result<(), Refusal> {
    let handle = ctx.cublas()?;
    let (act, w, scale, zero_point, y) = (act.ptr, w.v, scale.v, zero_point.v, y.ptr);

    #[cfg(feature = "_cuda")]
    unsafe {
        quant::act_x_wt_grouped_scaled(
            handle,
            act,
            w,
            w_dtype,
            w_nbytes,
            scale,
            scale_dtype,
            scale_numel,
            zero_point,
            group_size,
            y,
            m,
            n,
            k,
            beta,
        );
    }
    #[cfg(not(feature = "_cuda"))]
    let _ = (
        handle,
        act,
        w,
        w_dtype,
        w_nbytes,
        scale,
        scale_dtype,
        scale_numel,
        zero_point,
        group_size,
        y,
        m,
        n,
        k,
        beta,
    );
    Ok(())
}

#[routine(untraced)]
pub fn act_x_wt_mxfp4_marlin(
    ctx: &Ctx<'_>,
    act: In<Tensor<c_void>>,
    w: Const<Tensor<c_void>>,
    w_nbytes: usize,
    scale: Const<Tensor<c_void>>,
    scale_numel: usize,
    y: Out<Tensor<c_void>>,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) -> Result<(), Refusal> {
    let handle = ctx.cublas()?;
    let (act, w, scale, y) = (act.ptr, w.v, scale.v, y.ptr);

    #[cfg(feature = "_cuda")]
    unsafe {
        quant::act_x_wt_mxfp4_marlin(
            handle,
            act,
            w,
            w_nbytes,
            scale,
            scale_numel,
            y,
            m,
            n,
            k,
            beta,
        );
    }
    #[cfg(not(feature = "_cuda"))]
    let _ = (
        handle,
        act,
        w,
        w_nbytes,
        scale,
        scale_numel,
        y,
        m,
        n,
        k,
        beta,
    );
    Ok(())
}

pub use gemv::gemv_bf16;

pub use absorb::{mla_absorb_latent_to_v_bf16, mla_absorb_q_to_latent_bf16};

pub use lora::lora_qkv_correction;
