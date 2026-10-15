use crate::declare::Weight;
use crate::record::{Recorder, Value};
use poem_ir::{
    ActQuant, Attention, Collective, Dim, Dtype, Elementwise, GateActivation, IndexScore, Layout,
    Linear, ModulateForm, MropeForm, PixelOrder, RaggedMask, RopeForm, StructKind, TapKind, Ty,
    ValueId,
};

pub mod attn;
pub mod collective;
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
