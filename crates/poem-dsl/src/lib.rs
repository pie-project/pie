#![allow(clippy::too_many_arguments)]

pub mod declare;
pub mod facts;
pub mod forward;
pub mod ops;
mod record;

pub use declare::*;
pub use facts::*;
pub use forward::*;
pub use poem_ir::{
    Attention, BlockDrafter, CacheRow, Collective, Def, Dim, Dtype, Elementwise, GateActivation,
    GeomKind, Guard, Layout, Linear, ModulateForm, MropeForm, Operands, Operation, Param,
    ParamSource, Platform, RaggedMask, RopeForm, RuntimeInput, Selection, Shard, Stream, Trace, Ty,
    ValueId, VoxelSegment, resolve_classes,
};
pub use record::{Recorder, Refine, SplitSpec, Value};

pub type TraceFn = fn(Platform) -> Trace;

pub use poem_ir::ClassifyFn;

#[macro_export]
macro_rules! catalog {
    ($( ($name:literal, $tp:literal, $trace:path, $m:expr $(,)?) ),+ $(,)?) => {
        &[ $( (
            $name,
            $tp,
            (|platform| {
                let model = $m;
                $trace($name, &model, platform)
            }) as _,
            (|request: &$crate::Request| $crate::word_of(|| $m, request)) as _,
        ) ),+ ]
    };
}

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
