#![allow(clippy::too_many_arguments)]

pub mod declare;
pub mod fact;
pub mod forward;
pub mod generative;
pub mod import;
pub mod numpy;
pub mod ops;
pub mod pattern;
mod record;
pub mod rope;
#[allow(
    unsafe_code,
    reason = "starlark 0.14's `ProvidesStaticType` derive emits an `unsafe impl`"
)]
pub mod star;

pub use declare::*;
pub use fact::Predicate;
pub use forward::*;
pub use poem_ir::{
    Attn, BlockDrafter, CacheRow, Collective, Def, Dim, Dtype, Elemwise, GateActivation, Gather,
    GeomKind, Guard, IndexScore, InputAxis, Linear, ModulateForm, MropeForm, Operands, Operation,
    Param, ParamSource, PixelOrder, Platform, RaggedMask, RopeForm, RuntimeInput, Sampling,
    Selection, Shard, TapKind, Trace, Ty, ValueId, VoxelSegment, resolve_classes,
};
pub use record::{Arm, Primitive, Recorder, Refine, Switch, Value, restate, switch};

/// Reads `value` out to the guest as `name`; a voxel `value` laid out by
/// clips of its own names their `grid`.
pub fn output(name: &str, value: &Value, grid: Option<&Value>) {
    value.rec().output(name, value, grid);
}

/// [`output`], each entry's width laid out as `planes`, outermost first
/// (`[q_heads, kv_max]` for attention scores): what the guest reads the
/// entry's runs by, not a count it guesses.
pub fn output_in_planes(name: &str, value: &Value, grid: Option<&Value>, planes: &[u32]) {
    value.rec().output_in_planes(name, value, grid, planes);
}
