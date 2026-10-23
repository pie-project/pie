extern crate self as poem_ir;

pub mod canon;
pub mod check;
pub mod fact;
pub mod guard;
pub mod operands;
pub mod ops;
pub mod request;
pub mod trace;
pub mod value;

pub use check::classes::{
    Class, ClassSet, ClassTable, Fault as ClassFault, fact_width, resolve_classes,
};
pub use check::{Fault, check, checked};
pub use fact::{Builtin, Fact, Facts};
pub use guard::Guard;
pub use operands::{Operands, Part, Slot};
pub use ops::{
    ActFn, ActQuant, Attn, BinaryOp, Blocks, Bounds, Collective, Conv, Elemwise, Fused,
    GateActivation, GateHeads, Gather, GluAct, GridRule, IndexScore, Linear, ModulateForm,
    MropeForm, NormAxis, NormKind, NormNext, NormWeight, Operation, PixelOrder, RaggedMask,
    Relative, ResidualGate, RopeForm, RopeFreqs, RouteScore, Sampling, Ssm, TapKind, TimePad, Turn,
    VoxelSegment, Yarn,
};
pub use poem_ir_macros::Op;
pub use request::{Request, Stream};
pub use trace::{
    BlockDrafter, CacheRow, Node, Param, ParamLayout, ParamPlacement, ParamSource, Platform, Seam,
    Shard, Trace,
};
pub use value::{
    BIASES, Def, Dim, Dtype, GLOBAL_SCALES, GeomKind, INPUT_SCALES, PerAxis, RowAxis, RuntimeInput,
    SCALES, Selection, StructKind, TILED_BAND, TILED_STEP, Ty, ValueDecl, ValueId,
};
