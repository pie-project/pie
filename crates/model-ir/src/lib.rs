//! The canonical IR of a traced forward pass: typed op enums, `Def` × `Ty`
//! value declarations, guard conditions, and the serializable `Trace` that
//! carries them. Backend-free by construction — a leaf of the workspace, pure
//! data plus serde, so every crate that reads a plan reads only this one.

pub mod check;
pub mod guard;
pub mod operands;
pub mod ops;
pub mod trace;
pub mod value;

// The class sweep asks a different question than the validator, and the
// validator had the name `Fault` first — at the root, the sweep's is
// `ClassFault`; inside `check::classes` it keeps the design's spelling.
pub use check::classes::{
    Class, ClassSet, ClassTable, Fault as ClassFault, fact_width, resolve_classes,
};
pub use check::{Fault, check, checked};
pub use guard::Guard;
pub use operands::Operands;
pub use ops::{
    Attention, Collective, CustomCuda, Elementwise, GateActivation, Layout, Linear, MropeForm,
    Operation,
};
pub use trace::{CacheRow, Node, Param, ParamSource, Platform, Seam, Shard, Trace};
pub use value::{
    Def, Dim, Dtype, GeomKind, PerAxis, RowAxis, RuntimeInput, StructKind, TILED_BAND,
    TILED_STEP, Ty, ValueDecl, ValueId,
};
