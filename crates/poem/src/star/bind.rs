//! How an op is bound: the arguments a call passes, by place or by the
//! op's own parameter names, unpacked into what the op takes, and its result
//! held for the forward.

use std::fmt;

use allocative::Allocative;
use starlark::any::ProvidesStaticType;
use starlark::eval::{Arguments, Evaluator};
use starlark::starlark_simple_value;
use starlark::values::float::StarlarkFloat;
use starlark::values::list::ListRef;
use starlark::values::tuple::TupleRef;
use starlark::values::{
    Heap, NoSerialize, StarlarkPagableUnsupported, StarlarkValue, Value as Star,
};
use starlark_derive::{starlark_module, starlark_value};

use crate::star::forward::{CacheHandle, InputHandle, ValueHandle, held, hold, input_of};
use crate::star::values::{DtypeValue, weight};
use crate::{ValueId, Weight};
use spelled::*;

/// The types an op's parameters are spelled in.
pub(crate) mod spelled {
    pub(crate) use crate::ops::elemwise::Yarn;
    pub(crate) use crate::ops::spatial::Conv;
    pub(crate) use crate::{
        Dtype, GateActivation, Input, ModulateForm, MropeForm, RaggedMask, RopeForm, Value,
        VoxelSegment, Weight,
    };
    pub(crate) use poem_ir::GridRule;
}

/// One op of the DSL, as a forward calls it.
pub(crate) struct Op {
    pub(crate) module: &'static str,
    pub(crate) name: &'static str,
    pub(crate) params: &'static [&'static str],
    pub(crate) call: for<'v> fn(&mut Args<'v>, Heap<'v>) -> anyhow::Result<Star<'v>>,
}

/// The arguments one call passes, in the op's parameter order.
pub(crate) struct Args<'v> {
    op: &'static Op,
    given: Vec<Option<Star<'v>>>,
    at: usize,
}

impl<'v> Args<'v> {
    /// The next parameter's argument, unpacked as `T`.
    pub(crate) fn next<T: Arg<'v>>(&mut self) -> anyhow::Result<T> {
        let at = self.at;
        self.at += 1;
        T::arg(self.given[at]).map_err(|why| {
            anyhow::anyhow!(
                "`ops.{}.{}`'s `{}`: {why}",
                self.op.module,
                self.op.name,
                self.op.params[at]
            )
        })
    }
}

/// A cache row an op reads or writes.
pub(crate) struct Cache(pub(crate) ValueId);

/// What a parameter of an op is unpacked as.
pub(crate) trait Arg<'v>: Sized {
    fn arg(given: Option<Star<'v>>) -> anyhow::Result<Self>;
}

fn given(v: Option<Star<'_>>) -> anyhow::Result<Star<'_>> {
    v.filter(|v| !v.is_none())
        .ok_or_else(|| anyhow::anyhow!("is not given"))
}

fn wanted<T>(what: &str, v: Star<'_>) -> anyhow::Result<T> {
    Err(anyhow::anyhow!("{what} was wanted, not {}", v.get_type()))
}

impl<'v> Arg<'v> for Value {
    fn arg(v: Option<Star<'v>>) -> anyhow::Result<Self> {
        let v = given(v)?;
        match v.downcast_ref::<ValueHandle>() {
            Some(h) => Ok(held(*h)),
            None => wanted("a value", v),
        }
    }
}

impl<'v> Arg<'v> for Weight {
    fn arg(v: Option<Star<'v>>) -> anyhow::Result<Self> {
        weight(given(v)?)
    }
}

impl<'v> Arg<'v> for Cache {
    fn arg(v: Option<Star<'v>>) -> anyhow::Result<Self> {
        let v = given(v)?;
        match v.downcast_ref::<CacheHandle>() {
            Some(h) => Ok(Cache(h.0)),
            None => wanted("a cache", v),
        }
    }
}

impl<'v> Arg<'v> for Input {
    fn arg(v: Option<Star<'v>>) -> anyhow::Result<Self> {
        let v = given(v)?;
        match v.downcast_ref::<InputHandle>() {
            Some(h) => Ok(input_of(*h)),
            None => wanted("inputs", v),
        }
    }
}

impl<'v> Arg<'v> for u32 {
    fn arg(v: Option<Star<'v>>) -> anyhow::Result<Self> {
        let v = given(v)?;
        match v.unpack_i32() {
            Some(n) => u32::try_from(n).map_err(|_| anyhow::anyhow!("{n} is negative")),
            None => wanted("a count", v),
        }
    }
}

impl<'v> Arg<'v> for u8 {
    fn arg(v: Option<Star<'v>>) -> anyhow::Result<Self> {
        let n = u32::arg(v)?;
        u8::try_from(n).map_err(|_| anyhow::anyhow!("{n} is past a u8"))
    }
}

impl<'v> Arg<'v> for String {
    fn arg(v: Option<Star<'v>>) -> anyhow::Result<Self> {
        let v = given(v)?;
        match v.unpack_str() {
            Some(s) => Ok(s.to_string()),
            None => wanted("text", v),
        }
    }
}

impl<'v> Arg<'v> for u64 {
    fn arg(v: Option<Star<'v>>) -> anyhow::Result<Self> {
        let v = given(v)?;
        match <i64 as starlark::values::UnpackValue>::unpack_value(v) {
            Ok(Some(n)) => u64::try_from(n).map_err(|_| anyhow::anyhow!("{n} is negative")),
            _ => wanted("a count", v),
        }
    }
}

impl<'v> Arg<'v> for f32 {
    fn arg(v: Option<Star<'v>>) -> anyhow::Result<Self> {
        let v = given(v)?;
        if let Some(n) = v.unpack_i32() {
            return Ok(n as f32);
        }
        match v.downcast_ref::<StarlarkFloat>() {
            Some(x) => Ok(x.0 as f32),
            None => wanted("a number", v),
        }
    }
}

impl<'v> Arg<'v> for bool {
    fn arg(v: Option<Star<'v>>) -> anyhow::Result<Self> {
        let v = given(v)?;
        match v.unpack_bool() {
            Some(b) => Ok(b),
            None => wanted("a bool", v),
        }
    }
}

impl<'v> Arg<'v> for Dtype {
    fn arg(v: Option<Star<'v>>) -> anyhow::Result<Self> {
        let v = given(v)?;
        match v.downcast_ref::<DtypeValue>() {
            Some(d) => Ok(d.0),
            None => wanted("a dtype", v),
        }
    }
}

impl<'v, T: Arg<'v>> Arg<'v> for Option<T> {
    fn arg(v: Option<Star<'v>>) -> anyhow::Result<Self> {
        match v.filter(|v| !v.is_none()) {
            None => Ok(None),
            some => T::arg(some).map(Some),
        }
    }
}

fn items(v: Star<'_>) -> anyhow::Result<Vec<Star<'_>>> {
    if let Some(list) = ListRef::from_value(v) {
        return Ok(list.iter().collect());
    }
    match TupleRef::from_value(v) {
        Some(tuple) => Ok(tuple.iter().collect()),
        None => wanted("a list", v),
    }
}

impl<'v, T: Arg<'v>> Arg<'v> for Vec<T> {
    fn arg(v: Option<Star<'v>>) -> anyhow::Result<Self> {
        items(given(v)?)?
            .into_iter()
            .map(|item| T::arg(Some(item)))
            .collect()
    }
}

impl<'v, T: Arg<'v>, const N: usize> Arg<'v> for [T; N] {
    fn arg(v: Option<Star<'v>>) -> anyhow::Result<Self> {
        let items: Vec<T> = Vec::arg(v)?;
        let n = items.len();
        items
            .try_into()
            .map_err(|_| anyhow::anyhow!("{N} items were wanted, not {n}"))
    }
}

impl<'v, A: Arg<'v>, B: Arg<'v>> Arg<'v> for (A, B) {
    fn arg(v: Option<Star<'v>>) -> anyhow::Result<Self> {
        match items(given(v)?)?[..] {
            [a, b] => Ok((A::arg(Some(a))?, B::arg(Some(b))?)),
            ref other => Err(anyhow::anyhow!(
                "a pair was wanted, not {} items",
                other.len()
            )),
        }
    }
}

/// An enum a package spells by name.
fn spelled<T: Copy>(v: Option<Star<'_>>, what: &str, words: &[(&str, T)]) -> anyhow::Result<T> {
    let v = given(v)?;
    let Some(word) = v.unpack_str() else {
        return wanted(what, v);
    };
    words
        .iter()
        .find(|(w, _)| *w == word)
        .map(|(_, t)| *t)
        .ok_or_else(|| {
            let known: Vec<&str> = words.iter().map(|(w, _)| *w).collect();
            anyhow::anyhow!("{what} is one of {known:?}, not {word:?}")
        })
}

impl<'v> Arg<'v> for GateActivation {
    fn arg(v: Option<Star<'v>>) -> anyhow::Result<Self> {
        use GateActivation::*;
        spelled(v, "an activation", &[("silu", Silu), ("sigmoid", Sigmoid)])
    }
}

impl<'v> Arg<'v> for RopeForm {
    fn arg(v: Option<Star<'v>>) -> anyhow::Result<Self> {
        use RopeForm::*;
        spelled(
            v,
            "a rope form",
            &[
                ("interleaved", Interleaved),
                ("neox", Neox),
                ("split", Split),
                ("split_ladder", SplitLadder),
            ],
        )
    }
}

impl<'v> Arg<'v> for MropeForm {
    fn arg(v: Option<Star<'v>>) -> anyhow::Result<Self> {
        use MropeForm::*;
        spelled(
            v,
            "an mrope form",
            &[
                ("interleaved", Interleaved),
                ("blocked", Blocked),
                ("split", Split),
            ],
        )
    }
}

impl<'v> Arg<'v> for ModulateForm {
    fn arg(v: Option<Star<'v>>) -> anyhow::Result<Self> {
        use ModulateForm::*;
        spelled(
            v,
            "a modulation",
            &[
                ("scale_shift", ScaleShift),
                ("scale", Scale),
                ("tanh_gate", TanhGate),
            ],
        )
    }
}

/// A value of the DSL a package holds whole: a conv's geometry, a grid
/// rule, a rope's yarn scaling, a ragged mask.
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize, StarlarkPagableUnsupported, Allocative)]
pub(crate) struct Whole(#[allocative(skip)] pub(crate) Held);

#[derive(Debug, Clone)]
pub(crate) enum Held {
    Conv(Conv),
    Grid(GridRule),
    Yarn(Yarn),
    Mask(RaggedMask),
    Segment(VoxelSegment),
}

starlark_simple_value!(Whole);

impl fmt::Display for Whole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

#[starlark_value(type = "whole")]
impl<'v> StarlarkValue<'v> for Whole {}

macro_rules! whole {
    ($t:ty, $variant:ident, $what:literal) => {
        impl<'v> Arg<'v> for $t {
            fn arg(v: Option<Star<'v>>) -> anyhow::Result<Self> {
                let v = given(v)?;
                match v.downcast_ref::<Whole>() {
                    Some(Whole(Held::$variant(t))) => Ok(t.clone()),
                    _ => wanted($what, v),
                }
            }
        }
    };
}

whole!(Conv, Conv, "a conv");
whole!(GridRule, Grid, "a grid rule");
whole!(Yarn, Yarn, "a yarn scaling");
whole!(RaggedMask, Mask, "a ragged mask");
whole!(VoxelSegment, Segment, "a voxel segment");

/// What an op's result is held as.
pub(crate) trait Result {
    fn star(self, heap: Heap<'_>) -> Star<'_>;
}

impl Result for Value {
    fn star(self, heap: Heap<'_>) -> Star<'_> {
        heap.alloc(hold(self))
    }
}

impl Result for (Value, Value) {
    fn star(self, heap: Heap<'_>) -> Star<'_> {
        heap.alloc((hold(self.0), hold(self.1)))
    }
}

impl Result for (Value, Value, Value) {
    fn star(self, heap: Heap<'_>) -> Star<'_> {
        heap.alloc((hold(self.0), hold(self.1), hold(self.2)))
    }
}

impl Result for () {
    fn star(self, _heap: Heap<'_>) -> Star<'_> {
        Star::new_none()
    }
}

impl Result for ValueId {
    fn star(self, heap: Heap<'_>) -> Star<'_> {
        heap.alloc(CacheHandle(self))
    }
}

impl Result for RaggedMask {
    fn star(self, heap: Heap<'_>) -> Star<'_> {
        heap.alloc(Whole(Held::Mask(self)))
    }
}

pub(crate) fn result<R: Result>(heap: Heap<'_>, r: R) -> anyhow::Result<Star<'_>> {
    Ok(r.star(heap))
}

pub(crate) use crate::star::forward::dsl;

/// An op as the value a forward calls.
#[derive(Clone, Copy, ProvidesStaticType, NoSerialize, StarlarkPagableUnsupported, Allocative)]
pub(crate) struct OpValue(#[allocative(skip)] pub(crate) &'static Op);

impl fmt::Debug for OpValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ops.{}.{}", self.0.module, self.0.name)
    }
}

impl fmt::Display for OpValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ops.{}.{}({})",
            self.0.module,
            self.0.name,
            self.0.params.join(", ")
        )
    }
}

starlark_simple_value!(OpValue);

/// An `Input` method, bound to the rows it is called on.
#[derive(Clone, Copy, ProvidesStaticType, NoSerialize, StarlarkPagableUnsupported, Allocative)]
pub(crate) struct Bound(
    #[allocative(skip)] pub(crate) &'static Op,
    pub(crate) InputHandle,
);

impl fmt::Debug for Bound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "inputs.{}", self.0.name)
    }
}

impl fmt::Display for Bound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "inputs.{}({})",
            self.0.name,
            self.0.params[1..].join(", ")
        )
    }
}

starlark_simple_value!(Bound);

#[starlark_value(type = "bound_method")]
impl<'v> StarlarkValue<'v> for Bound {
    fn invoke(
        &self,
        _me: Star<'v>,
        args: &Arguments<'v, '_>,
        eval: &mut Evaluator<'v, '_, '_>,
    ) -> starlark::Result<Star<'v>> {
        let this = eval.heap().alloc(self.1);
        call(self.0, &self.to_string(), Some(this), args, eval)
    }
}

/// The `Input` method `name`, bound to `this`.
pub(crate) fn input_method<'v>(this: InputHandle, name: &str, heap: Heap<'v>) -> Option<Star<'v>> {
    crate::star::ops::INPUTS
        .iter()
        .find(|op| op.name == name)
        .map(|op| heap.alloc(Bound(op, this)))
}

/// Calls `op` with `args`, after `this` if it is a method.
fn call<'v>(
    op: &'static Op,
    spelled: &str,
    this: Option<Star<'v>>,
    args: &Arguments<'v, '_>,
    eval: &mut Evaluator<'v, '_, '_>,
) -> starlark::Result<Star<'v>> {
    let heap = eval.heap();
    let fail = |why: String| starlark::Error::new_other(anyhow::anyhow!("{why}"));
    let mut given: Vec<Option<Star<'v>>> = vec![None; op.params.len()];
    let skip = usize::from(this.is_some());
    given[..skip].copy_from_slice(&this.into_iter().map(Some).collect::<Vec<_>>());
    for (at, v) in args.positions(heap)?.enumerate() {
        let at = at + skip;
        if at >= op.params.len() {
            return Err(fail(format!(
                "{spelled} takes {} arguments",
                op.params.len() - skip
            )));
        }
        given[at] = Some(v);
    }
    for (name, v) in args.names_map()? {
        let name = name.as_str();
        let at = op.params[skip..]
            .iter()
            .position(|p| *p == name)
            .map(|at| at + skip)
            .ok_or_else(|| fail(format!("{spelled} takes no `{name}`")))?;
        if given[at].is_some() {
            return Err(fail(format!("{spelled} is given `{name}` twice")));
        }
        given[at] = Some(v);
    }
    let mut args = Args { op, given, at: 0 };
    (op.call)(&mut args, heap).map_err(starlark::Error::new_other)
}

#[starlark_value(type = "op")]
impl<'v> StarlarkValue<'v> for OpValue {
    fn invoke(
        &self,
        _me: Star<'v>,
        args: &Arguments<'v, '_>,
        eval: &mut Evaluator<'v, '_, '_>,
    ) -> starlark::Result<Star<'v>> {
        call(self.0, &self.to_string(), None, args, eval)
    }
}

fn axes(what: &str, v: Vec<u32>) -> anyhow::Result<[u32; 3]> {
    match v[..] {
        [h, w] => Ok([1, h, w]),
        [t, h, w] => Ok([t, h, w]),
        _ => Err(anyhow::anyhow!(
            "a conv's {what} is 2 or 3 axes, not {}",
            v.len()
        )),
    }
}

/// The builtins that make the whole values ops take.
#[starlark_module]
pub(crate) fn wholes(builder: &mut starlark::environment::GlobalsBuilder) {
    /// A conv of kernel `k`, `stride` and `pad` over 2 (h, w) or 3 (t, h, w)
    /// axes; `pad_back` pads the far side otherwise, `causal` ("zero" or
    /// "replicate") pads time on the near side only, as wide as the kernel.
    fn conv<'v>(
        #[starlark(require = pos)] k: Star<'v>,
        #[starlark(require = pos)] stride: Star<'v>,
        #[starlark(require = pos)] pad: Star<'v>,
        #[starlark(require = named, default = starlark::values::none::NoneOr::None)]
        pad_back: starlark::values::none::NoneOr<Star<'v>>,
        #[starlark(require = named, default = starlark::values::none::NoneOr::None)]
        causal: starlark::values::none::NoneOr<&str>,
    ) -> anyhow::Result<Whole> {
        let two = Vec::<u32>::arg(Some(k))?.len() == 2;
        let (k, stride, pad) = (
            axes("kernel", Vec::arg(Some(k))?)?,
            axes("stride", Vec::arg(Some(stride))?)?,
            axes("pad", Vec::arg(Some(pad))?)?,
        );
        let mut conv = if two {
            Conv::conv2d([k[1], k[2]], [stride[1], stride[2]], [pad[1], pad[2]])
        } else {
            Conv::conv3d(k, stride, pad)
        };
        if let Some(back) = pad_back.into_option() {
            conv = conv.pad_back(axes("pad_back", Vec::arg(Some(back))?)?);
        }
        conv = match causal.into_option() {
            None => conv,
            Some("zero") => conv.causal(poem_ir::ops::spatial::TimePad::Zero),
            Some("replicate") => conv.causal(poem_ir::ops::spatial::TimePad::Replicate),
            Some(other) => anyhow::bail!("a causal conv pads with zero or replicate, not {other}"),
        };
        Ok(Whole(Held::Conv(conv)))
    }

    /// A ragged attention's mask that lets each row attend only rows of its
    /// own group.
    fn group_block_diagonal() -> anyhow::Result<Whole> {
        Ok(Whole(Held::Mask(RaggedMask::GroupBlockDiagonal)))
    }

    /// A ragged attention's mask that masks nothing.
    fn unmasked() -> anyhow::Result<Whole> {
        Ok(Whole(Held::Mask(RaggedMask::None)))
    }

    /// A ragged attention's mask under which a reference row attends only
    /// the rows of its own reference, by the tags `q_tags` and `kv_tags`.
    fn reference_self_only<'v>(
        #[starlark(require = pos)] q_tags: Star<'v>,
        #[starlark(require = pos)] kv_tags: Star<'v>,
    ) -> anyhow::Result<Whole> {
        Ok(Whole(Held::Mask(RaggedMask::ReferenceSelfOnly {
            q_tags: Value::arg(Some(q_tags))?.id(),
            kv_tags: Value::arg(Some(kv_tags))?.id(),
        })))
    }

    /// The grid rule a conv steps its grid by.
    fn grid_rule(#[starlark(require = pos)] conv: Star<'_>) -> anyhow::Result<Whole> {
        Ok(Whole(Held::Grid(Conv::arg(Some(conv))?.rule())))
    }

    /// Yarn's rope scaling.
    fn yarn(
        #[starlark(require = named)] factor: Star<'_>,
        #[starlark(require = named)] beta_fast: Star<'_>,
        #[starlark(require = named)] beta_slow: Star<'_>,
        #[starlark(require = named)] original_max_position: u32,
    ) -> anyhow::Result<Whole> {
        Ok(Whole(Held::Yarn(Yarn {
            factor: f32::arg(Some(factor))?,
            beta_fast: f32::arg(Some(beta_fast))?,
            beta_slow: f32::arg(Some(beta_slow))?,
            original_max_position,
        })))
    }

    /// The voxels an attention reads: the whole clip, or runs of `frames`.
    fn segment(
        #[starlark(require = named, default = starlark::values::none::NoneOr::None)]
        frames: starlark::values::none::NoneOr<u32>,
    ) -> anyhow::Result<Whole> {
        Ok(Whole(Held::Segment(match frames.into_option() {
            None => VoxelSegment::Clip,
            Some(n) => VoxelSegment::Frames(n),
        })))
    }
}

#[cfg(test)]
mod tests {
    /// Every op of `src/ops` is bound, but for the ones `tools/bind_ops.py`
    /// names as not spellable: a new op is one run of it away.
    #[test]
    fn every_op_a_package_can_spell_is_bound() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let generated = std::fs::read_to_string(root.join("src/star/ops.rs")).unwrap();
        let unbound: Vec<&str> = generated
            .lines()
            .take_while(|l| l.starts_with("//!"))
            .filter_map(|l| l.strip_prefix("//! `")?.strip_suffix('`'))
            .collect();
        let mut missing = Vec::new();
        for module in [
            "attn",
            "collective",
            "elemwise",
            "layout",
            "linear",
            "spatial",
        ] {
            let text = std::fs::read_to_string(root.join(format!("src/ops/{module}.rs"))).unwrap();
            for line in text.lines() {
                let Some(rest) = line.strip_prefix("pub fn ") else {
                    continue;
                };
                let name = &rest[..rest.find(['(', '<']).unwrap()];
                let bound = super::super::ops::OPS
                    .iter()
                    .any(|op| op.module == module && op.name == name);
                if !bound && !unbound.contains(&format!("{module}::{name}").as_str()) {
                    missing.push(format!("{module}::{name}"));
                }
            }
        }
        let forward = std::fs::read_to_string(root.join("src/forward.rs")).unwrap();
        let methods = &forward[forward.find("impl Input {").unwrap()..];
        let methods = &methods[..methods.find("\n}\n").unwrap()];
        let by_hand = ["recorder", "on", "partition", "reading", "walk_layers"];
        for line in methods.lines() {
            let Some(rest) = line.strip_prefix("    pub fn ") else {
                continue;
            };
            let name = &rest[..rest.find(['(', '<']).unwrap()];
            let bound = super::super::ops::INPUTS.iter().any(|op| op.name == name);
            if !bound
                && !by_hand.contains(&name)
                && !unbound.contains(&format!("inputs::{name}").as_str())
            {
                missing.push(format!("Input::{name}"));
            }
        }
        assert!(
            missing.is_empty(),
            "run crates/poem/tools/bind_ops.py: {missing:?} are not bound"
        );
    }
}
