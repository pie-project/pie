//! Model IR: traced ops, symbolic shapes, seams and kernel checks.
//! plan only names symbols they cover.

pub mod facts;
pub mod kernels;
pub mod seam;
pub mod trace;

pub use facts::{NormPlacement, QkNorm};
pub use trace::{
    DType, Dim, DynAxis, FireClass, ForwardPlan, HookStage, Op, OpKind, Shape, StateRef,
    StateStore, TraceBuilder, ValueId,
};
