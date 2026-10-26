use eta_ir::registry::{Axis, OutputDecl};
use eta_ir::types::{Dtype, Shape};

use crate::context::{current_rows, intern_name};
use crate::model;
use crate::value::{Tensor, output_val};

pub fn vocab() -> u32 {
    model::vocab()
}
pub fn page_size() -> u32 {
    model::page_size()
}
#[allow(non_upper_case_globals)]
pub const activation_type: Dtype = Dtype::F32;

/// Lane `lane`'s output `name` as the model declares it: its readout rows (a
/// lone row read as its width alone), or its layers stacked.
#[track_caller]
pub fn output(lane: u32, name: &str) -> Tensor {
    let decl = declared(lane, name);
    let rows = match decl.axis {
        Axis::Lane => decl.layers.max(1),
        Axis::Row | Axis::Voxel if decl.readout => current_rows().max(1),
        Axis::Row | Axis::Voxel => panic!(
            "output `{name}` is one row a token or voxel of lane {lane}: \
             `output_rows` reads the rows it names"
        ),
    };
    let t = output_val(
        intern_name(name),
        lane,
        Shape::matrix(rows, decl.width),
        decl.dtype,
    );
    if decl.readout {
        single_row_reshape(t)
    } else {
        t
    }
}

/// `rows` rows of lane `lane`'s output `name`, one a token or voxel.
#[track_caller]
pub fn output_rows(lane: u32, name: &str, rows: u32) -> Tensor {
    let decl = declared(lane, name);
    output_val(
        intern_name(name),
        lane,
        Shape::matrix(rows.max(1), decl.width),
        decl.dtype,
    )
}

#[track_caller]
fn declared(lane: u32, name: &str) -> OutputDecl {
    model::output(lane, name)
        .unwrap_or_else(|| panic!("lane {lane} of this pass reads out no output `{name}`"))
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
    use crate::value::AsTensor;
    use alloc::string::String;
    use alloc::vec;
    use eta_ir::op::Op;

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
        record_sink(String::from("attn_page_mask"), span);
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
        record_sink(String::from("lora"), span);
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
        record_sink(String::from("lora"), span);
    }
}
