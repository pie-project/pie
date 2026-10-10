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
#[allow(
    unsafe_code,
    reason = "starlark 0.14's `ProvidesStaticType` derive emits an `unsafe impl`"
)]
pub mod star;

pub use declare::*;
pub use fact::Predicate;
pub use forward::*;
pub use poem_ir::{
    Attention, BlockDrafter, CacheRow, Collective, Def, Dim, Dtype, Elementwise, GateActivation,
    GeomKind, Guard, Layout, Linear, ModulateForm, MropeForm, Operands, Operation, Param,
    ParamSource, PixelOrder, Platform, RaggedMask, RopeForm, RuntimeInput, Selection, Shard,
    Stream, TapKind, Trace, Ty, ValueId, VoxelSegment, resolve_classes,
};
pub use record::{Arm, Primitive, Recorder, Refine, Switch, Value, switch};

pub mod seam {

    use crate::record::Value;

    pub struct Def {
        pub name: &'static str,
    }

    pub const ATTN_Q: Def = Def { name: "attn.q" };

    pub const ATTN_OUT: Def = Def { name: "attn.out" };

    pub const ATTN_QV: Def = Def { name: "attn.qv" };

    pub const RECURRENT: Def = Def { name: "recurrent" };

    pub const IN: Def = Def { name: "in" };

    pub const OUT: Def = Def { name: "out" };

    pub const MTP: Def = Def { name: "mtp" };

    pub const MTP_DRAFTS: Def = Def { name: "mtp.drafts" };

    pub const SCORES: Def = Def {
        name: "attn.scores",
    };

    pub const VELOCITY: Def = Def { name: "velocity" };

    pub const HIDDEN: Def = Def { name: "hidden" };

    pub const PIXELS: Def = Def { name: "pixels" };

    pub const FLOAT_READOUTS: [&str; 3] = [VELOCITY.name, HIDDEN.name, PIXELS.name];

    pub fn at(def: Def, values: &[&Value]) {
        let first = values
            .first()
            .unwrap_or_else(|| panic!("seam `{}` names no value", def.name));
        first.rec().seam(def.name, values);
    }
}
