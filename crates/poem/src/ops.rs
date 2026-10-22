use crate::declare::Weight;
use crate::record::{Recorder, Value};
use poem_ir::{
    ActQuant, Attention, Collective, Dim, Dtype, Elementwise, GateActivation, Gather, IndexScore,
    Linear, ModulateForm, MropeForm, PixelOrder, RaggedMask, RopeForm, Sampling, Ssm, StructKind,
    TapKind, Ty, ValueId,
};

pub mod attn;
pub mod collective;
pub mod conv;
pub mod elemwise;
pub mod gather;
pub mod linear;
pub mod sampling;
pub mod ssm;

fn tensor(rows: Dim, width: impl Into<u64>, dtype: Dtype) -> Ty {
    Ty::Tensor {
        shape: vec![rows, Dim::Const(width.into())],
        dtype,
    }
}
