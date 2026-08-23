use core::ffi::c_void;
use core::ptr::NonNull;
use kernels::{Bind, Fire};
use kernels_macros::routine;

use crate::jit::abi::Tensor;
use crate::jit::abi::bf16;
use crate::jit::{Ctx, Launch};
use crate::{norm, quant};
use kernels::Refusal;
use kernels::routine::{Const, In, InOut, Out};

const BLOCK: u32 = 256;

#[routine(namespace = "attn", canon = "layout.split_qkv")]
pub fn split_qkv_bf16(
    ctx: &Ctx<'_>,
    packed: In<Tensor<bf16>>,
    q_out: Out<Tensor<bf16>>,
    k_out: Out<Tensor<bf16>>,
    v_out: Out<Tensor<bf16>>,
) -> Result<(), Refusal> {
    let q = q_out.all("the q half")?;
    let k = k_out.all("the k half")?;
    let (q_dim, kv_dim) = (q.width, k.width);
    if q_dim <= 0 && kv_dim <= 0 {
        return Err(Refusal::Empty {
            what: "q_dim and kv_dim",
        });
    }
    let n_tokens = q.rows;
    let width = q_dim.max(kv_dim).unsigned_abs();
    ctx.fire(
        Fire::at(
            "attn/split_packed.cuh",
            "::pie::attn::split_qkv<::pie::bf16>",
        )
        .apply(Launch::grid(
            [width.div_ceil(BLOCK), n_tokens.unsigned_abs(), 1],
            [BLOCK, 1, 1],
        )),
        &[
            packed.arg(),
            q_out.arg(),
            k_out.arg(),
            v_out.arg(),
            q_dim.arg(),
            kv_dim.arg(),
        ],
    )
}

pub fn add_bias_bf16(
    ctx: &Ctx<'_>,
    out: *mut c_void,
    bias: *const c_void,
    num_rows: i32,
    dim: i32,
) -> Result<(), Refusal> {
    norm::add_bias::<bf16>(
        ctx,
        InOut {
            ptr: out.cast::<bf16>(),
            rows: num_rows,
            width: dim,
        },
        Const {
            v: bias.cast::<bf16>(),
        },
    )
}

#[routine(namespace = "ssm", canon = "ssm.gdn_prep")]
pub fn qwen_gdn_post_conv_prep_bf16(
    ctx: &Ctx<'_>,
    qkv_post: In<Tensor<bf16>>,
    a: In<Tensor<bf16>>,
    b: In<Tensor<bf16>>,
    a_log: Const<Tensor<f32>>,
    dt_bias: Const<Tensor<bf16>>,
    q_norm_kh: Out<Tensor<f32>>,
    k_norm_kh: Out<Tensor<f32>>,
    v_fp32: Out<Tensor<f32>>,
    g_log_out: Out<Tensor<f32>>,
    beta_out: Out<Tensor<f32>>,
    k_h: Const<i32>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>,
    conv_dim: Const<i32>,
) -> Result<(), Refusal> {
    let k_h = *k_h;
    let v_h = *v_h;
    let k_d = *k_d;
    let v_d = *v_d;
    let conv_dim = *conv_dim;

    const PREP_BLOCK: u32 = 128;
    let n = qkv_post.all("the post-convolution qkv")?.rows;
    #[allow(clippy::cast_precision_loss)]
    let q_scale = (k_d as f32).sqrt().recip();

    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::qwen_gdn_qk_norm<::pie::bf16, 128>",
        )
        .apply(Launch::grid(
            [n.unsigned_abs(), k_h.unsigned_abs(), 1],
            [PREP_BLOCK, 1, 1],
        )),
        &[
            qkv_post.arg(),
            q_norm_kh.arg(),
            k_norm_kh.arg(),
            k_h.arg(),
            k_d.arg(),
            conv_dim.arg(),
            q_scale.arg(),
        ],
    )?;

    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::qwen_gdn_v_g_beta<::pie::bf16, 128>",
        )
        .apply(Launch::grid(
            [n.unsigned_abs(), v_h.unsigned_abs(), 1],
            [PREP_BLOCK, 1, 1],
        )),
        &[
            qkv_post.arg(),
            a.arg(),
            b.arg(),
            a_log.arg(),
            dt_bias.arg(),
            v_fp32.arg(),
            g_log_out.arg(),
            beta_out.arg(),
            k_h.arg(),
            v_h.arg(),
            k_d.arg(),
            v_d.arg(),
            conv_dim.arg(),
        ],
    )
}

#[routine(namespace = "layout", canon = "layout.split_q_gate")]
pub fn split_q_gate_bf16(
    ctx: &Ctx<'_>,
    packed: In<Tensor<bf16>>,
    q_out: Out<Tensor<bf16>>,
    gate_out: Out<Tensor<bf16>>,
    head_dim: Const<i32>,
) -> Result<(), Refusal> {
    let head_dim = *head_dim;
    let q = q_out.all("the query half")?;
    if head_dim <= 0 {
        return Err(Refusal::Unstated {
            what: "the head pitch a q/gate split grids by",
        });
    }
    if q.width % head_dim != 0 {
        return Err(Refusal::Unstated {
            what: "a q/gate half whose width is not whole heads",
        });
    }
    let (n, num_heads) = (q.rows, q.width / head_dim);
    let block = if head_dim < 128 { 64 } else { 128 };
    ctx.fire(
        Fire::at(
            "layout/deinterleave.cuh",
            "::pie::layout::split_q_gate<::pie::bf16>",
        )
        .apply(Launch::grid(
            [n.unsigned_abs(), num_heads.unsigned_abs(), 1],
            [block, 1, 1],
        )),
        &[
            packed.arg(),
            q_out.arg(),
            gate_out.arg(),
            n.arg(),
            num_heads.arg(),
            head_dim.arg(),
        ],
    )
}

#[routine(namespace = "mlp", canon = "gate.sigmoid_mul", out(x = like(x)))]
pub fn sigmoid_gate_inplace_bf16(
    ctx: &Ctx<'_>,
    x: InOut<Tensor<bf16>>,
    gate: In<Tensor<bf16>>,
) -> Result<(), Refusal> {
    let num_elements = x.all("the gated rectangle")?.elements();
    ctx.fire(
        Fire::at(
            "mlp/swiglu.cuh",
            "::pie::mlp::sigmoid_gate_inplace<::pie::bf16>",
        )
        .apply(Launch::flat(num_elements.unsigned_abs(), BLOCK)),
        &[x.arg(), gate.arg(), num_elements.arg()],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_gated_fp32_in_bf16(
    ctx: &Ctx<'_>,
    x: *const c_void,
    gate: *const c_void,
    weight: *const c_void,
    y: *mut c_void,
    num_rows: i32,
    hidden: i32,
    eps: f32,
    per_head_dim: i32,
) -> Result<(), Refusal> {
    let shape = |p: *mut bf16| Out {
        ptr: p,
        rows: num_rows,
        width: hidden,
    };
    norm::rmsnorm_gated_fp32_in::<bf16>(
        ctx,
        In {
            ptr: x.cast::<f32>(),
            rows: num_rows,
            width: hidden,
        },
        In {
            ptr: gate.cast::<bf16>(),
            rows: num_rows,
            width: hidden,
        },
        Const {
            v: weight.cast::<f32>(),
        },
        shape(y.cast::<bf16>()),
        Const { v: eps },
        Const { v: per_head_dim },
    )
}

fn addr<T>(p: *const T) -> crate::jit::abi::DevicePtr {
    p as usize as u64
}

pub fn derive_kv_len(
    ctx: &Ctx<'_>,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    page_size: u32,
    num_requests: u32,
    kv_len: *mut u32,
) -> Result<(), Refusal> {
    ctx.fire(
        Fire::at("layout/geometry.cuh", "::pie::layout::derive_kv_len")
            .apply(Launch::flat(num_requests, BLOCK)),
        &[
            kv_page_indptr.arg(),
            kv_last_page_lens.arg(),
            page_size.arg(),
            num_requests.arg(),
            kv_len.arg(),
        ],
    )
}

pub fn resolve_slot_to_block(
    ctx: &Ctx<'_>,
    pages: *const u32,
    slot_to_block: *const u32,
    num_slots: u32,
    count: u32,
    page_indices: *mut u32,
) -> Result<(), Refusal> {
    ctx.fire(
        Fire::at(
            "layout/geometry.cuh",
            "::pie::layout::resolve_slot_to_block",
        )
        .apply(Launch::flat(count, BLOCK)),
        &[
            pages.arg(),
            slot_to_block.arg(),
            num_slots.arg(),
            count.arg(),
            page_indices.arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn compose_envelope_csr(
    ctx: &Ctx<'_>,
    members: *const c_void,
    traced_page_indptr: *const u32,
    traced_pages: *const u32,
    traced_kv_len: *const u32,
    traced_w_slot: *const u32,
    token_ids: *const u32,
    member_count: u32,
    page_size: u32,
    kv_page_indptr: *mut u32,
    kv_page_indices: *mut u32,
    kv_last_page_lens: *mut u32,
    w_slot_out: *mut u32,
    row_valid: *mut u8,
    kills: Option<NonNull<u32>>,
) -> Result<(), Refusal> {
    if member_count > MAX_BLOCK.unsigned_abs() {
        return Err(Refusal::Wide {
            what: "member_count, as one block's threads",
            at: i64::from(member_count),
            max: i64::from(MAX_BLOCK),
        });
    }

    let smem = member_count * 4;
    ctx.fire(
        Fire::at("layout/geometry.cuh", "::pie::layout::compose_envelope_csr")
            .apply(Launch::grid([1, 1, 1], [member_count, 1, 1]).smem(smem)),
        &[
            members.arg(),
            traced_page_indptr.arg(),
            traced_pages.arg(),
            traced_kv_len.arg(),
            traced_w_slot.arg(),
            token_ids.arg(),
            member_count.arg(),
            page_size.arg(),
            kv_page_indptr.arg(),
            kv_page_indices.arg(),
            kv_last_page_lens.arg(),
            w_slot_out.arg(),
            row_valid.arg(),
            kills.arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gather_tokens(
    ctx: &Ctx<'_>,
    k_pages: *mut u16,
    v_pages: *mut u16,
    ops: *const c_void,
    num_ops: i32,
    num_layers: i32,
    layer_stride_elems: i64,
    page_size: i32,
    num_kv_heads: i32,
    head_dim: i32,
) -> Result<(), Refusal> {
    let token_stride = i64::from(num_kv_heads) * i64::from(head_dim);
    let page_stride = token_stride * i64::from(page_size);
    let grid = Launch::grid(
        [num_ops.unsigned_abs(), 1, num_layers.unsigned_abs()],
        [BLOCK, 1, 1],
    );

    if token_stride % 8 == 0 && layer_stride_elems % 8 == 0 {
        return ctx.fire(
            Fire::at("layout/gather_tokens.cuh", "::pie::layout::gather_i4").apply(grid),
            &[
                k_pages.cast::<c_void>().arg(),
                v_pages.cast::<c_void>().arg(),
                ops.arg(),
                (token_stride / 8).arg(),
                (page_stride / 8).arg(),
                (layer_stride_elems / 8).arg(),
            ],
        );
    }
    ctx.fire(
        Fire::at("layout/gather_tokens.cuh", "::pie::layout::gather_u16").apply(grid),
        &[
            k_pages.arg(),
            v_pages.arg(),
            ops.arg(),
            token_stride.arg(),
            page_stride.arg(),
            layer_stride_elems.arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn graph_pad_rows(
    ctx: &Ctx<'_>,
    qo_indptr: *mut u32,
    kv_page_indptr: *mut u32,
    kv_page_indices: *mut u32,
    kv_last_page_lens: *mut u32,
    tokens: *mut u32,
    positions: *mut u32,
    row_valid: *mut u8,
    custom_mask: Option<NonNull<u8>>,
    custom_mask_indptr: Option<NonNull<i32>>,
    real_mask_bytes: i32,
    real_requests: i32,
    real_tokens: i32,
    padding: i32,
    pad_tokens: i32,
    pad_page: u32,
) -> Result<(), Refusal> {
    if pad_tokens < padding {
        return Err(Refusal::Narrow {
            what: "pad_tokens, in lanes",
            at: i64::from(pad_tokens),
        });
    }

    if padding > MAX_BLOCK {
        return Err(Refusal::Wide {
            what: "padding, as one block's threads",
            at: i64::from(padding),
            max: i64::from(MAX_BLOCK),
        });
    }
    ctx.fire(
        Fire::at("layout/graph_pad.cuh", "::pie::layout::graph_pad_rows")
            .apply(Launch::grid([1, 1, 1], [padding.unsigned_abs(), 1, 1])),
        &[
            qo_indptr.arg(),
            kv_page_indptr.arg(),
            kv_page_indices.arg(),
            kv_last_page_lens.arg(),
            tokens.arg(),
            positions.arg(),
            row_valid.arg(),
            custom_mask.arg(),
            custom_mask_indptr.arg(),
            real_mask_bytes.arg(),
            real_requests.arg(),
            real_tokens.arg(),
            padding.arg(),
            pad_tokens.arg(),
            pad_page.arg(),
        ],
    )
}

const MAX_BLOCK: i32 = 1024;

pub fn split_gate_up_bf16(
    ctx: &Ctx<'_>,
    packed: *const c_void,
    gate_out: *mut c_void,
    up_out: *mut c_void,
    n_tokens: i32,
    inter: i32,
) -> Result<(), Refusal> {
    ctx.fire(
        Fire::at(
            "layout/split_gate_up.cuh",
            "::pie::layout::split_gate_up<::pie::bf16>",
        )
        .apply(Launch::grid(
            [
                inter.unsigned_abs().div_ceil(BLOCK),
                n_tokens.unsigned_abs(),
                1,
            ],
            [BLOCK, 1, 1],
        )),
        &[
            packed.cast::<bf16>().arg(),
            gate_out.cast::<bf16>().arg(),
            up_out.cast::<bf16>().arg(),
            inter.arg(),
        ],
    )
}

const MXFP4_GROUP: i32 = 32;

pub fn transcode_bf16_to_mxfp4(
    ctx: &Ctx<'_>,
    src: *const bf16,
    packed: *mut u8,
    scales: *mut u8,
    rows: i32,
    cols: i32,
) -> Result<(), Refusal> {
    if cols % MXFP4_GROUP != 0 {
        return Err(Refusal::Narrow {
            what: "cols, in whole 32-element blocks",
            at: i64::from(cols),
        });
    }
    let decode = quant::transcode::DecodeBf16 {
        src: addr(src),
        cols,
    };
    let encode = quant::transcode::EncodeMxfp4 {
        packed: addr(packed),
        scales: addr(scales),
        cols,
    };

    ctx.fire(Fire::at("quant/transcode.cuh", "::pie::transcode::transcode_rowmajor_kernel<\
                 ::pie::transcode::EncodeMxfp4::kGroup,::pie::transcode::DecodeBf16,::pie::transcode::EncodeMxfp4>").apply(Launch::per_row(rows.unsigned_abs(), BLOCK)), &[decode.arg(), encode.arg(), cols.arg()])
}

#[allow(clippy::too_many_arguments)]
pub fn transcode_fp8_e4m3_per_group_to_mxfp4(
    ctx: &Ctx<'_>,
    src: *const u8,
    src_scales: *const f32,
    packed: *mut u8,
    scales: *mut u8,
    rows: i32,
    cols: i32,
    group_size: i32,
) -> Result<(), Refusal> {
    if cols % MXFP4_GROUP != 0 {
        return Err(Refusal::Narrow {
            what: "cols, in whole 32-element blocks",
            at: i64::from(cols),
        });
    }
    let decode = quant::transcode::DecodeFp8E4m3PerGroup {
        src: addr(src),
        scales: addr(src_scales),
        cols,
        scale_cols: (cols + group_size - 1) / group_size,
        group_size,
    };
    let encode = quant::transcode::EncodeMxfp4 {
        packed: addr(packed),
        scales: addr(scales),
        cols,
    };
    ctx.fire(Fire::at("quant/transcode.cuh", "::pie::transcode::transcode_rowmajor_kernel<\
                 ::pie::transcode::EncodeMxfp4::kGroup,::pie::transcode::DecodeFp8E4m3PerGroup,::pie::transcode::EncodeMxfp4>").apply(Launch::per_row(rows.unsigned_abs(), BLOCK)), &[decode.arg(), encode.arg(), cols.arg()])
}
