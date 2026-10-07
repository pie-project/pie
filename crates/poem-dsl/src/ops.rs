use crate::declare::Weight;
use crate::record::Value;
use poem_ir::{
    Attention, Collective, CustomCuda, Dim, Dtype, Elementwise, GateActivation, Layout, Linear,
    ModulateForm, MropeForm, RaggedMask, RopeForm, StructKind, Ty, ValueId,
};

pub mod attn;
pub mod collective;
pub mod custom;
pub mod elemwise;
pub mod layout;
pub mod linear;
pub mod spatial;

fn tensor(rows: Dim, width: impl Into<u64>, dtype: Dtype) -> Ty {
    Ty::Tensor {
        shape: vec![rows, Dim::Const(width.into())],
        dtype,
    }
}
