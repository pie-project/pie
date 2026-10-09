//! The Rust values a model package handles: dtypes, weights, the facts a
//! forward branches on, and handles to the DSL's values, which live with the
//! trace being recorded rather than on the Starlark heap.

use std::fmt;

use crate::{Dtype, Predicate, Weight};
use allocative::Allocative;
use starlark::any::ProvidesStaticType;
use starlark::environment::{Methods, MethodsBuilder};
use starlark::starlark_simple_value;
use starlark::values::ValueLike;
use starlark::values::list::UnpackList;
use starlark::values::none::NoneOr;
use starlark::values::{Heap, NoSerialize, StarlarkValue, Value};
use starlark_derive::{starlark_module, starlark_value};

/// A dtype, spelled as the catalog's names spell it (`bf16`, `u4g64`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ProvidesStaticType, NoSerialize, Allocative)]
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
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize, Allocative)]
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
    /// `segments` along them; `heads` are the heads each segment holds.
    fn packed(
        this: &WeightValue,
        #[starlark(require = pos)] segments: UnpackList<u64>,
        #[starlark(require = named, default = NoneOr::None)] heads: NoneOr<UnpackList<u64>>,
    ) -> anyhow::Result<WeightValue> {
        crate::star::forward::dsl("packed", || {
            let w = this.0.clone().packed(segments.items);
            match heads.into_option() {
                Some(heads) => WeightValue(w.heads(heads.items)),
                None => WeightValue(w),
            }
        })
    }

    /// The weight is a bank of experts, each one's rows cut into
    /// `segments`, a rank holding a share of every expert.
    fn bank(this: &WeightValue, segments: UnpackList<u64>) -> anyhow::Result<WeightValue> {
        crate::star::forward::dsl("bank", || WeightValue(this.0.clone().bank(segments.items)))
    }

    /// A conv's weight, its taps major and `c_in` inputs minor.
    fn conv_taps_major(this: &WeightValue, c_in: u32, taps: u32) -> anyhow::Result<WeightValue> {
        crate::star::forward::dsl("conv_taps_major", || {
            WeightValue(this.0.clone().conv_taps_major(c_in, taps))
        })
    }

    /// The weight is cut along its rows between ranks; `heads` are the
    /// heads its rows hold.
    fn columns(
        this: &WeightValue,
        #[starlark(require = named, default = NoneOr::None)] heads: NoneOr<u64>,
    ) -> anyhow::Result<WeightValue> {
        crate::star::forward::dsl("columns", || {
            let w = this.0.clone().columns();
            match heads.into_option() {
                Some(heads) => WeightValue(w.heads([heads])),
                None => WeightValue(w),
            }
        })
    }

    /// The weight is cut along its last axis between ranks.
    fn rows(this: &WeightValue) -> anyhow::Result<WeightValue> {
        crate::star::forward::dsl("rows", || WeightValue(this.0.clone().rows()))
    }

    /// The weight is a bank the host registers, not one a checkpoint holds.
    fn registered(this: &WeightValue) -> anyhow::Result<WeightValue> {
        Ok(WeightValue(this.0.clone().registered()))
    }

    #[starlark(attribute)]
    fn name(this: &WeightValue) -> anyhow::Result<String> {
        Ok(this.0.name.clone())
    }

    /// The axis tensor-parallel ranks cut the weight along, or `None` if
    /// every rank holds it whole.
    #[starlark(attribute)]
    fn cut_axis(this: &WeightValue) -> anyhow::Result<NoneOr<u32>> {
        Ok(match &this.0.shard {
            crate::Shard::Cut { axis, .. } => NoneOr::Other(*axis),
            crate::Shard::Replicated => NoneOr::None,
        })
    }

    #[starlark(attribute)]
    fn shape(this: &WeightValue) -> anyhow::Result<Vec<u64>> {
        Ok(this.0.shape.clone())
    }

    #[starlark(attribute)]
    fn dtype(this: &WeightValue) -> anyhow::Result<DtypeValue> {
        Ok(DtypeValue(this.0.dtype))
    }
}

/// The rows a forward branches onto.
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize, Allocative)]
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
