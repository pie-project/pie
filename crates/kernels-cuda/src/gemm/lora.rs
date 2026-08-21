use core::ffi::c_void;
use kernels_macros::routine;

use crate::jit::Ctx;
use crate::jit::abi::Tensor;
use crate::jit::abi::bf16;
use kernels::Refusal;

use kernels::routine::{Const, In, InOut, Out};

pub const LORA_SITE_Q: u64 = 1 << 0;

pub const LORA_SITE_K: u64 = 1 << 1;

pub const LORA_SITE_V: u64 = 1 << 2;

pub const LORA_SITE_O: u64 = 1 << 3;

pub const LORA_SITE_GATE_UP: u64 = 1 << 4;

pub const LORA_SITE_DOWN: u64 = 1 << 5;

pub const LORA_SITES_KNOWN: u64 =
    LORA_SITE_Q | LORA_SITE_K | LORA_SITE_V | LORA_SITE_O | LORA_SITE_GATE_UP | LORA_SITE_DOWN;

pub const LORA_SITES_CONSUMED: u64 = LORA_SITE_Q | LORA_SITE_V;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum LoraForm {
    #[default]
    LowRank = 0,
    Scale = 1,
}

#[derive(Debug, Clone, Copy)]
pub struct LoraLaneView {
    pub a: *const c_void,
    pub b: *const c_void,
    pub sites_bits: u64,
    pub token_start: u32,
    pub token_count: u32,
    pub num_layers: u32,
    pub rank: u32,
    pub d_in: u32,
    pub d_out: u32,
    pub form: LoraForm,
}

#[derive(Debug, Clone, Copy)]
pub struct Lane {
    pub view: LoraLaneView,
    pub a_bf16: *mut c_void,
    pub b_bf16: *mut c_void,
    pub xa_offset: usize,
    pub grouped: bool,
}

#[derive(Debug, Clone, Default)]
pub struct Group {
    pub rank: i32,
    pub d_in: i32,
    pub d_out: i32,
    pub members: Vec<usize>,
    pub nq: i32,
    pub nv: i32,
    pub m: Vec<i32>,
    pub mq: Vec<i32>,
    pub mv: Vec<i32>,
    pub slab_off: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Staged<'a> {
    pub lanes: &'a [Lane],
    pub groups: &'a [Group],
    pub ptr_slab: *mut c_void,
    pub slab_stride: usize,
}

pub fn bf16_row(base: *const c_void, row: u32, width: i32) -> *const c_void {
    let off = row as usize * usize::try_from(width.max(0)).unwrap_or(0) * 2;

    unsafe { base.cast::<u8>().add(off).cast() }
}

#[allow(clippy::too_many_arguments)]
unsafe fn grouped(
    ctx: &Ctx<'_>,
    act_ptrs_dev: *const *const c_void,
    w_ptrs_dev: *const *const c_void,
    y_ptrs_dev: *const *mut c_void,
    m_array_host: *const i32,
    group_count: i32,
    n: i32,
    k: i32,
    beta: f32,
) -> Result<(), Refusal> {
    let handle = ctx.cublas()?;

    #[cfg(feature = "_cuda")]
    unsafe {
        super::dense::grouped_act_x_wt_bf16(
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

#[routine(untraced, driver)]
pub fn lora_qkv_correction(
    ctx: &Ctx<'_>,
    staged: Staged<'_>,
    layer: i32,
    qkv_in: In<Tensor<c_void>>,
    h: i32,
    hq: i32,
    hk: i32,
    q_out: Out<Tensor<c_void>>,
    v_out: Out<Tensor<c_void>>,
    // THE DRIVER'S OWN CARVE, HANDED OVER RATHER THAN ASKED FOR. A `driver`
    // row fires through `Ctx::on(stream)`, which carries no `Facts` — so a
    // `keys::` ask here could never be answered, whoever published it. The
    // staging that allocated this slab is the caller, and `Out` is how it
    // already hands over `q_out` and `v_out`.
    xa_scratch: Out<Tensor<c_void>>,
) -> Result<(), Refusal> {
    let (qkv_in, q_out, v_out, xa_scratch) = (qkv_in.ptr, q_out.ptr, v_out.ptr, xa_scratch.ptr);
    let layer_u = usize::try_from(layer).unwrap_or(0);

    for lane in staged.lanes {
        if lane.grouped {
            continue;
        }
        let v = &lane.view;
        if v.form == LoraForm::Scale {
            continue;
        }
        let t = i32::try_from(v.token_count).unwrap_or(0);
        let r = i32::try_from(v.rank).unwrap_or(0);
        let a_l = bf16_row(
            lane.a_bf16.cast_const(),
            u32::try_from(layer_u * v.rank as usize).unwrap_or(0),
            i32::try_from(v.d_in).unwrap_or(0),
        );
        let b_l = bf16_row(
            lane.b_bf16.cast_const(),
            u32::try_from(layer_u * v.d_out as usize).unwrap_or(0),
            r,
        );
        let x = bf16_row(qkv_in, v.token_start, h);

        super::act_x_wt_bf16_beta(
            ctx,
            In {
                ptr: x,
                rows: t,
                width: h,
            },
            Const { v: a_l },
            Out {
                ptr: xa_scratch,
                rows: t,
                width: r,
            },
            0.0,
        )?;
        let d_out = i32::try_from(v.d_out).unwrap_or(0);
        if v.sites_bits & LORA_SITE_Q != 0 {
            super::act_x_wt_bf16_beta(
                ctx,
                In {
                    ptr: xa_scratch.cast_const(),
                    rows: t,
                    width: r,
                },
                Const { v: b_l },
                Out {
                    ptr: bf16_row(q_out.cast_const(), v.token_start, hq).cast_mut(),
                    rows: t,
                    width: d_out,
                },
                1.0,
            )?;
        }
        if v.sites_bits & LORA_SITE_V != 0 {
            super::act_x_wt_bf16_beta(
                ctx,
                In {
                    ptr: xa_scratch.cast_const(),
                    rows: t,
                    width: r,
                },
                Const { v: b_l },
                Out {
                    ptr: bf16_row(v_out.cast_const(), v.token_start, hk).cast_mut(),
                    rows: t,
                    width: d_out,
                },
                1.0,
            )?;
        }
    }

    for g in staged.groups {
        let n = g.members.len();

        let slot = unsafe {
            staged
                .ptr_slab
                .cast::<*const c_void>()
                .add(layer_u * staged.slab_stride + g.slab_off)
        };
        let x_ptrs: *const *const c_void = slot;

        unsafe {
            let a_ptrs = x_ptrs.add(n);
            let xa_ptrs = x_ptrs.add(2 * n);

            grouped(
                ctx,
                x_ptrs,
                a_ptrs,
                xa_ptrs.cast::<*mut c_void>().cast_mut(),
                g.m.as_ptr(),
                i32::try_from(n).unwrap_or(0),
                g.rank,
                g.d_in,
                0.0,
            )?;
            if g.nq > 0 {
                let base = x_ptrs.add(3 * n);
                grouped(
                    ctx,
                    base,
                    base.add(g.nq as usize),
                    base.add(2 * g.nq as usize).cast::<*mut c_void>().cast_mut(),
                    g.mq.as_ptr(),
                    g.nq,
                    g.d_out,
                    g.rank,
                    1.0,
                )?;
            }
            if g.nv > 0 {
                let base = x_ptrs.add(3 * n + 3 * g.nq as usize);
                grouped(
                    ctx,
                    base,
                    base.add(g.nv as usize),
                    base.add(2 * g.nv as usize).cast::<*mut c_void>().cast_mut(),
                    g.mv.as_ptr(),
                    g.nv,
                    g.d_out,
                    g.rank,
                    1.0,
                )?;
            }
        }
    }

    for lane in staged.lanes {
        let v = &lane.view;
        if v.form != LoraForm::Scale {
            continue;
        }
        let t = i32::try_from(v.token_count).unwrap_or(0);
        let d_out = i32::try_from(v.d_out).unwrap_or(0);
        let l_l = bf16_row(
            lane.a_bf16.cast_const(),
            u32::try_from(layer_u).unwrap_or(0),
            d_out,
        );
        if v.sites_bits & LORA_SITE_Q != 0 {
            crate::quant::scale_rows::<bf16>(
                ctx,
                InOut {
                    ptr: bf16_row(q_out.cast_const(), v.token_start, hq)
                        .cast_mut()
                        .cast::<crate::jit::abi::bf16>(),
                    rows: t,
                    width: d_out,
                },
                In {
                    ptr: l_l.cast::<crate::jit::abi::bf16>(),
                    rows: t,
                    width: d_out,
                },
            )?;
        }
        if v.sites_bits & LORA_SITE_V != 0 {
            crate::quant::scale_rows::<bf16>(
                ctx,
                InOut {
                    ptr: bf16_row(v_out.cast_const(), v.token_start, hk)
                        .cast_mut()
                        .cast::<crate::jit::abi::bf16>(),
                    rows: t,
                    width: d_out,
                },
                In {
                    ptr: l_l.cast::<crate::jit::abi::bf16>(),
                    rows: t,
                    width: d_out,
                },
            )?;
        }
    }
    Ok(())
}
