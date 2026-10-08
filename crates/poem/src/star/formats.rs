//! What a format may use: the names and attributes of the checkpoint it
//! recognizes, and the reads that land its tensors on a deployment's weights.
//!
//! A read is stated, not performed: `formats.star` names what lands where,
//! and the builder the contract is built by performs it, refusing a missing or
//! misshapen tensor exactly as a Rust import's reads do.

use std::cell::RefCell;
use std::fmt;

use crate::Weight;
use crate::import::format::{Stated, attribute, config};
use allocative::Allocative;
use starlark::any::ProvidesStaticType;
use starlark::environment::{GlobalsBuilder, Methods, MethodsBuilder};
use starlark::starlark_simple_value;
use starlark::values::float::UnpackFloat;
use starlark::values::list::UnpackList;
use starlark::values::none::{NoneOr, NoneType};
use starlark::values::structs::AllocStruct;
use starlark::values::{Heap, NoSerialize, StarlarkPagableUnsupported, StarlarkValue, Value};
use starlark_derive::{starlark_module, starlark_value};

use crate::star::values::weight;

/// One read a format states.
pub(crate) enum Read {
    One(Weight, String),
    Concat(Weight, Vec<String>),
}

/// What a format's test may read of a checkpoint: its names and attributes.
pub(crate) struct Snapshot {
    names: std::collections::HashSet<String>,
    attributes: Option<ztensor::format::cbor::Value>,
}

impl Snapshot {
    pub(crate) fn of(src: &ztensor::Source) -> Snapshot {
        Snapshot {
            names: src.names().map(str::to_string).collect(),
            attributes: src.attributes().cloned(),
        }
    }
}

thread_local! {
    /// The reads the format reading a checkpoint states.
    pub(crate) static READS: RefCell<Vec<Read>> = const { RefCell::new(Vec::new()) };
    /// The checkpoint a format is recognizing, while its test runs.
    pub(crate) static SOURCE: RefCell<Option<Snapshot>> = const { RefCell::new(None) };
}

fn source<R>(f: impl FnOnce(&Snapshot) -> R) -> anyhow::Result<R> {
    SOURCE.with(|s| match s.borrow().as_ref() {
        Some(snapshot) => Ok(f(snapshot)),
        None => anyhow::bail!("a checkpoint is read only while a format recognizes one"),
    })
}

/// The checkpoint a format's test reads.
#[derive(
    Debug, Clone, Copy, ProvidesStaticType, NoSerialize, StarlarkPagableUnsupported, Allocative,
)]
pub struct SourceHandle;

starlark_simple_value!(SourceHandle);

impl fmt::Display for SourceHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "checkpoint")
    }
}

#[starlark_value(type = "checkpoint")]
impl<'v> StarlarkValue<'v> for SourceHandle {
    fn get_methods() -> Option<&'static Methods> {
        Some(SOURCE_METHODS_STATICS.methods())
    }
}

#[starlark_module]
fn source_methods(builder: &mut MethodsBuilder) {
    /// Whether the checkpoint holds a tensor named `name`.
    fn has(
        #[starlark(this)] _this: &SourceHandle,
        #[starlark(require = pos)] name: &str,
    ) -> anyhow::Result<bool> {
        source(|src| src.names.contains(name))
    }

    /// Whether the checkpoint holds a tensor whose name ends in `suffix`.
    fn has_suffix(
        #[starlark(this)] _this: &SourceHandle,
        #[starlark(require = pos)] suffix: &str,
    ) -> anyhow::Result<bool> {
        source(|src| src.names.iter().any(|name| name.ends_with(suffix)))
    }

    /// The container attribute `key`, as text, or `None`.
    fn attribute(
        #[starlark(this)] _this: &SourceHandle,
        #[starlark(require = pos)] key: &str,
    ) -> anyhow::Result<NoneOr<String>> {
        source(|src| {
            match src
                .attributes
                .as_ref()
                .and_then(|a| a.get(key))
                .and_then(|v| v.as_text())
            {
                Some(text) => NoneOr::Other(text.to_string()),
                None => NoneOr::None,
            }
        })
    }
}

/// The reads of the format reading a checkpoint.
#[derive(
    Debug, Clone, Copy, ProvidesStaticType, NoSerialize, StarlarkPagableUnsupported, Allocative,
)]
pub struct ReadsHandle;

starlark_simple_value!(ReadsHandle);

impl fmt::Display for ReadsHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "reads")
    }
}

#[starlark_value(type = "reads")]
impl<'v> StarlarkValue<'v> for ReadsHandle {
    fn get_methods() -> Option<&'static Methods> {
        Some(READS_METHODS_STATICS.methods())
    }
}

#[starlark_module]
fn reads_methods(builder: &mut MethodsBuilder) {
    /// `w` lands the checkpoint's tensor `name`.
    fn read<'v>(
        #[starlark(this)] _this: &ReadsHandle,
        #[starlark(require = pos)] w: Value<'v>,
        #[starlark(require = pos)] name: String,
    ) -> anyhow::Result<NoneType> {
        let w = weight(w)?;
        READS.with(|r| r.borrow_mut().push(Read::One(w, name)));
        Ok(NoneType)
    }

    /// `w` lands the checkpoint's tensors `names`, joined along its first
    /// axis.
    fn read_concat<'v>(
        #[starlark(this)] _this: &ReadsHandle,
        #[starlark(require = pos)] w: Value<'v>,
        #[starlark(require = pos)] names: UnpackList<String>,
    ) -> anyhow::Result<NoneType> {
        let w = weight(w)?;
        READS.with(|r| r.borrow_mut().push(Read::Concat(w, names.items)));
        Ok(NoneType)
    }
}

/// A value a checkpoint states of the model it holds.
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize, StarlarkPagableUnsupported, Allocative)]
pub struct StatedValue(#[allocative(skip)] pub(crate) Stated);

starlark_simple_value!(StatedValue);

impl fmt::Display for StatedValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

#[starlark_value(type = "stated")]
impl<'v> StarlarkValue<'v> for StatedValue {}

#[starlark_module]
pub(crate) fn formats(builder: &mut GlobalsBuilder) {
    /// A format named `name`: `read(reads)` states what lands where, and
    /// `recognizes(checkpoint)`, if given, tells a checkpoint of it apart
    /// (without one, reading is the test); `states` is what a checkpoint of
    /// it states of the model.
    fn format<'v>(
        #[starlark(require = pos)] name: &str,
        #[starlark(require = named)] read: Value<'v>,
        #[starlark(require = named, default = NoneOr::None)] recognizes: NoneOr<Value<'v>>,
        #[starlark(require = named, default = UnpackList::default())] states: UnpackList<Value<'v>>,
        heap: Heap<'v>,
    ) -> anyhow::Result<Value<'v>> {
        let recognizes = match recognizes {
            NoneOr::Other(f) => f,
            NoneOr::None => Value::new_none(),
        };
        Ok(heap.alloc(AllocStruct([
            ("name", heap.alloc(name)),
            ("read", read),
            ("recognizes", recognizes),
            ("states", heap.alloc(states.items)),
        ])))
    }

    /// The configuration value at `path` is `want`, or at least `want`.
    fn config(
        #[starlark(require = pos)] path: &str,
        #[starlark(require = pos)] want: UnpackFloat,
        #[starlark(require = named, default = false)] or_deeper: bool,
    ) -> anyhow::Result<StatedValue> {
        let stated = config(path, want.0);
        Ok(StatedValue(if or_deeper {
            stated.or_deeper()
        } else {
            stated
        }))
    }

    /// The container attribute `key` is `want`, or at least `want`.
    fn attribute(
        #[starlark(require = pos)] key: &str,
        #[starlark(require = pos)] want: UnpackFloat,
        #[starlark(require = named, default = false)] or_deeper: bool,
    ) -> anyhow::Result<StatedValue> {
        let stated = attribute(key, want.0);
        Ok(StatedValue(if or_deeper {
            stated.or_deeper()
        } else {
            stated
        }))
    }
}
