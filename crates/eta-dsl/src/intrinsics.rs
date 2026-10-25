use eta_ir::op::IntrinsicId;
use eta_ir::types::{Dtype, Shape};

use crate::context::current_rows;
use crate::model;
use crate::value::{Tensor, intrinsic_val};

pub fn vocab() -> u32 {
    model::vocab()
}
pub fn page_size() -> u32 {
    model::page_size()
}
#[allow(non_upper_case_globals)]
pub const activation_type: Dtype = Dtype::F32;

fn logits_shape() -> Shape {
    let rows = current_rows();
    let v = vocab();
    Shape::matrix(rows.max(1), v)
}

pub fn logits() -> Tensor {
    let t = intrinsic_val(IntrinsicId::Logits, 0, logits_shape(), Dtype::F32);
    single_row_reshape(t)
}
pub fn mtp_logits(k: u32) -> Tensor {
    intrinsic_val(
        IntrinsicId::MtpLogits,
        0,
        Shape::matrix(k, vocab()),
        Dtype::F32,
    )
}
pub fn mtp_drafts(n: u32) -> Tensor {
    intrinsic_val(
        IntrinsicId::MtpDrafts,
        0,
        Shape::vector(n.max(1)),
        Dtype::I32,
    )
}
/// Lane `lane`'s `rows` hidden rows, `width` wide.
pub fn hidden(lane: u32, rows: u32, width: u32) -> Tensor {
    intrinsic_val(
        IntrinsicId::Hidden,
        lane,
        Shape::matrix(rows.max(1), width.max(1)),
        activation_type,
    )
}
/// Lane `lane`'s velocity, `rows` rows `width` wide.
pub fn velocity(lane: u32, rows: u32, width: u32) -> Tensor {
    intrinsic_val(
        IntrinsicId::Velocity,
        lane,
        Shape::matrix(rows.max(1), width.max(1)),
        activation_type,
    )
}
/// Lane `lane`'s `rows` pixel rows, `width` wide.
pub fn pixels(lane: u32, rows: u32, width: u32) -> Tensor {
    intrinsic_val(
        IntrinsicId::Pixels,
        lane,
        Shape::matrix(rows.max(1), width.max(1)),
        activation_type,
    )
}
pub fn query(width: u32) -> Tensor {
    intrinsic_val(
        IntrinsicId::Query,
        0,
        Shape::vector(width.max(1)),
        activation_type,
    )
}
pub fn layer() -> Tensor {
    intrinsic_val(IntrinsicId::Layer, 0, Shape::SCALAR, Dtype::U32)
}
pub fn attn_score(planes: u32) -> Tensor {
    intrinsic_val(
        IntrinsicId::AttnScore,
        0,
        Shape::matrix(planes.max(1), eta_ir::registry::ATTN_SCORE_KV_MAX),
        Dtype::F32,
    )
}

pub const fn attn_score_kv_max() -> u32 {
    eta_ir::registry::ATTN_SCORE_KV_MAX
}

fn single_row_reshape(t: Tensor) -> Tensor {
    let s = t.shape();
    if s.rank() == 2 && s.dims()[0] == 1 {
        crate::value::reshape(t, [s.dims()[1]])
    } else {
        t
    }
}

pub mod kernel {
    use crate::context::{emit, intern_name, record_sink};
    use crate::error::Span;
    use crate::value::{AsTensor, Tensor};
    use alloc::string::String;
    use alloc::vec;
    use eta_ir::op::{IntrinsicId, Op};
    use eta_ir::registry::SinkScope;
    use eta_ir::types::{Dtype, Shape, ValueType};

    #[track_caller]
    pub fn envelope_dot(p_max: u32) -> Tensor {
        let query_ty = ValueType::new(Shape::vector(1), super::activation_type);
        let query = emit(
            Op::IntrinsicVal {
                intr: IntrinsicId::Query,
                lane: 0,
                shape: query_ty.shape,
                dtype: query_ty.dtype,
            },
            &[query_ty],
        );
        let score_ty = ValueType::new(Shape::vector(p_max), Dtype::F32);
        let name = intern_name("envelope_dot");
        Tensor::node(
            emit(
                Op::KernelCall {
                    name,
                    args: vec![query],
                    shape: score_ty.shape,
                    dtype: score_ty.dtype,
                },
                &[score_ty],
            ),
            score_ty,
        )
    }

    #[track_caller]
    pub fn attn_page_mask(mask: impl AsTensor) {
        let span = Span::here();
        let (mask, _) = mask.to_arg().materialize();
        let name = intern_name("attn_page_mask");
        emit(
            Op::SinkCall {
                name,
                args: vec![mask],
            },
            &[],
        );
        record_sink(String::from("attn_page_mask"), span, SinkScope::Attention);
    }

    #[track_caller]
    pub fn lora(a: impl AsTensor, b: impl AsTensor, sites: impl AsTensor) {
        let span = Span::here();
        let (a, _) = a.to_arg().materialize();
        let (b, _) = b.to_arg().materialize();
        let (sites, _) = sites.to_arg().materialize();
        let name = intern_name("lora");
        emit(
            Op::SinkCall {
                name,
                args: vec![a, b, sites],
            },
            &[],
        );
        record_sink(String::from("lora"), span, SinkScope::PassWide);
    }

    #[track_caller]
    pub fn adapter_scale(l: impl AsTensor, sites: impl AsTensor) {
        let span = Span::here();
        let (l, _) = l.to_arg().materialize();
        let (sites, _) = sites.to_arg().materialize();
        let name = intern_name("lora");
        emit(
            Op::SinkCall {
                name,
                args: vec![l, sites],
            },
            &[],
        );
        record_sink(String::from("lora"), span, SinkScope::PassWide);
    }
}
