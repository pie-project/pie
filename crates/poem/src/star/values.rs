//! The Rust values a model package handles: dtypes, weights, the facts a
//! forward branches on, and handles to the DSL's values, which live with the
//! trace being recorded rather than on the Starlark heap.

use std::fmt;

use crate::{Dtype, Predicate, Weight};
use allocative::Allocative;
use starlark::any::ProvidesStaticType;
use starlark::environment::{Methods, MethodsBuilder};
use starlark::starlark_simple_value;
use starlark::values::list::UnpackList;
use starlark::values::none::NoneOr;
use starlark::values::{Heap, NoSerialize, StarlarkPagableUnsupported, StarlarkValue, Value};
use starlark_derive::{starlark_module, starlark_value};

/// A dtype, spelled as the catalog's names spell it (`bf16`, `u4g64`).
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    ProvidesStaticType,
    NoSerialize,
    StarlarkPagableUnsupported,
    Allocative,
)]
pub struct DtypeValue(#[allocative(skip)] pub Dtype);

starlark_simple_value!(DtypeValue);

impl fmt::Display for DtypeValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dtype.{}", word(self.0))
    }
}

#[starlark_value(type = "dtype")]
impl<'v> StarlarkValue<'v> for DtypeValue {
    fn equals(&self, other: Value<'v>) -> starlark::Result<bool> {
        Ok(other
            .downcast_ref::<DtypeValue>()
            .is_some_and(|other| other.0 == self.0))
    }
}

/// The name a dtype is spelled by.
#[must_use]
pub fn word(dtype: Dtype) -> String {
    format!("{dtype:?}").to_lowercase()
}

/// A weight a layout declares.
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize, StarlarkPagableUnsupported, Allocative)]
pub struct WeightValue(#[allocative(skip)] pub Weight);

starlark_simple_value!(WeightValue);

impl fmt::Display for WeightValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "weight({:?}, {:?}, {})",
            self.0.name,
            self.0.shape,
            word(self.0.dtype)
        )
    }
}

#[starlark_value(type = "weight")]
impl<'v> StarlarkValue<'v> for WeightValue {
    fn get_methods() -> Option<&'static Methods> {
        Some(WEIGHT_METHODS_STATICS.methods())
    }
}

#[starlark_module]
fn weight_methods(builder: &mut MethodsBuilder) {
    /// The weight's rows are banks a rank holds a whole of, cut into
    /// `segments` along them.
    fn packed(this: &WeightValue, segments: UnpackList<u64>) -> anyhow::Result<WeightValue> {
        Ok(WeightValue(this.0.clone().packed(segments.items)))
    }

    /// The weight is cut along its rows between ranks.
    fn columns(this: &WeightValue) -> anyhow::Result<WeightValue> {
        Ok(WeightValue(this.0.clone().columns()))
    }

    /// The weight is cut along its last axis between ranks.
    fn rows(this: &WeightValue) -> anyhow::Result<WeightValue> {
        Ok(WeightValue(this.0.clone().rows()))
    }

    /// The weight is a bank the host registers, not one a checkpoint holds.
    fn registered(this: &WeightValue) -> anyhow::Result<WeightValue> {
        Ok(WeightValue(this.0.clone().registered()))
    }

    #[starlark(attribute)]
    fn name(this: &WeightValue) -> anyhow::Result<String> {
        Ok(this.0.name.clone())
    }

    #[starlark(attribute)]
    fn dtype(this: &WeightValue) -> anyhow::Result<DtypeValue> {
        Ok(DtypeValue(this.0.dtype))
    }
}

/// The rows a forward branches onto.
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize, StarlarkPagableUnsupported, Allocative)]
pub struct PredicateValue(#[allocative(skip)] pub Predicate);

starlark_simple_value!(PredicateValue);

impl fmt::Display for PredicateValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

#[starlark_value(type = "fact")]
impl<'v> StarlarkValue<'v> for PredicateValue {
    fn bit_and(&self, other: Value<'v>, heap: Heap<'v>) -> starlark::Result<Value<'v>> {
        let Some(other) = other.downcast_ref::<PredicateValue>() else {
            return Err(starlark::Error::new_other(anyhow::anyhow!(
                "a fact joins only another fact, not {}",
                other.get_type()
            )));
        };
        Ok(heap.alloc(PredicateValue(self.0.clone() & other.0.clone())))
    }

    fn bit_not(&self, heap: Heap<'v>) -> starlark::Result<Value<'v>> {
        Ok(heap.alloc(PredicateValue(!self.0.clone())))
    }
}

/// Unpacks an optional window: `None`, or a count of positions.
pub fn window(value: NoneOr<u32>) -> Option<u32> {
    value.into_option()
}

/// The value `value` is, if it is a [`PredicateValue`].
pub fn predicate<'v>(value: Value<'v>) -> anyhow::Result<Predicate> {
    value
        .downcast_ref::<PredicateValue>()
        .map(|p| p.0.clone())
        .ok_or_else(|| anyhow::anyhow!("a fact was wanted, not {}", value.get_type()))
}

/// The weight `value` is.
pub fn weight<'v>(value: Value<'v>) -> anyhow::Result<Weight> {
    value
        .downcast_ref::<WeightValue>()
        .map(|w| w.0.clone())
        .ok_or_else(|| anyhow::anyhow!("a weight was wanted, not {}", value.get_type()))
}
