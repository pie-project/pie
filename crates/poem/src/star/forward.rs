//! What a forward may use: the rows it reads, the facts it branches them on,
//! the DSL's ops over them and the seams it plants.
//!
//! The DSL's values hold the trace being recorded, which no Starlark heap
//! may: a forward handles them by place in the trail this thread records,
//! and the trail is emptied before the trace finishes.

use std::cell::RefCell;
use std::fmt;

use crate::record::FirstMatch;
use crate::{HybridSpec, Input, Predicate, Refine as _, Value, ValueId, fact};
use allocative::Allocative;
use starlark::any::ProvidesStaticType;
use starlark::environment::{GlobalsBuilder, Methods, MethodsBuilder};
use starlark::eval::Evaluator;
use starlark::starlark_simple_value;
use starlark::values::ValueLike;
use starlark::values::list::UnpackList;
use starlark::values::none::{NoneOr, NoneType};
use starlark::values::{Heap, NoSerialize, StarlarkValue, Value as Star};
use starlark_derive::{starlark_module, starlark_value};

use crate::star::values::{DtypeValue, PredicateValue, predicate};

/// The DSL values a forward handles, by place.
#[derive(Default)]
pub(crate) struct Trail {
    pub(crate) values: Vec<Value>,
    pub(crate) inputs: Vec<Input>,
    pub(crate) spec: Option<HybridSpec>,
}

thread_local! {
    pub(crate) static TRAIL: RefCell<Trail> = RefCell::default();
}

fn with<R>(f: impl FnOnce(&mut Trail) -> R) -> R {
    TRAIL.with(|trail| f(&mut trail.borrow_mut()))
}

/// Runs `op`, a call into the DSL, turning the panic a misuse of it raises
/// into an error the script's caller is told of.
pub(crate) fn dsl<R>(what: &str, op: impl FnOnce() -> R) -> anyhow::Result<R> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(op))
        .map_err(|panic| anyhow::anyhow!("{what}: {}", panic_message(&*panic)))
}

/// What a panic said.
pub(crate) fn panic_message(panic: &(dyn std::any::Any + Send)) -> String {
    panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap_or_else(|| "a panic with no message".to_string())
}

/// A DSL value a forward holds.
#[derive(Debug, Clone, Copy, ProvidesStaticType, NoSerialize, Allocative)]
pub struct ValueHandle(pub(crate) u32);

starlark_simple_value!(ValueHandle);

impl fmt::Display for ValueHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "value#{}", self.0)
    }
}

pub(crate) fn hold(value: Value) -> ValueHandle {
    with(|t| {
        t.values.push(value);
        ValueHandle((t.values.len() - 1) as u32)
    })
}

pub(crate) fn held(handle: ValueHandle) -> Value {
    with(|t| t.values[handle.0 as usize].clone())
}

#[starlark_value(type = "value")]
impl<'v> StarlarkValue<'v> for ValueHandle {
    fn get_methods() -> Option<&'static Methods> {
        Some(VALUE_METHODS_STATICS.methods())
    }

    fn mul(&self, rhs: Star<'v>, heap: Heap<'v>) -> Option<starlark::Result<Star<'v>>> {
        let by = rhs.unpack_i32().map(f64::from).or_else(|| {
            rhs.downcast_ref::<starlark::values::float::StarlarkFloat>()
                .map(|f| f.0)
        })?;
        let x = held(*self);
        Some(
            dsl("a value scaled", move || x * by as f32)
                .map(|y| heap.alloc(hold(y)))
                .map_err(starlark::Error::new_other),
        )
    }
}

#[starlark_module]
fn value_methods(builder: &mut MethodsBuilder) {
    /// This value over every row, the rows it was not computed on held as
    /// they were.
    fn everywhere(this: &ValueHandle) -> anyhow::Result<ValueHandle> {
        let x = held(*this);
        dsl("everywhere", || x.everywhere()).map(hold)
    }

    /// The width of this value's rows.
    fn width(this: &ValueHandle) -> anyhow::Result<u64> {
        let x = held(*this);
        dsl("width", || x.width())
    }

    /// The rows of this value `fact` holds for.
    fn on(
        this: &ValueHandle,
        #[starlark(require = pos)] fact: Star<'_>,
    ) -> anyhow::Result<ValueHandle> {
        let (x, p) = (held(*this), predicate(fact)?);
        dsl("on", || x.on(p)).map(hold)
    }

    /// This value's rows parted by `cases`: each arm the rows the first case
    /// to hold names, and the rest.
    fn partition<'v>(
        this: &ValueHandle,
        #[starlark(require = pos)] cases: UnpackList<Star<'v>>,
        heap: Heap<'v>,
    ) -> anyhow::Result<Star<'v>> {
        let x = held(*this);
        let cases: Vec<Predicate> = cases
            .items
            .into_iter()
            .map(predicate)
            .collect::<anyhow::Result<_>>()?;
        let (arms, rest) = dsl("partition", || parted(&x, cases))?;
        let arms: Vec<Star<'v>> = arms.into_iter().map(|a| heap.alloc(hold(a))).collect();
        Ok(heap.alloc((heap.alloc(arms), hold(rest))))
    }
}

fn parted(x: &Value, cases: Vec<Predicate>) -> (Vec<Value>, Value) {
    let rec = x.rec().clone();
    let mut chain = FirstMatch::default();
    let arms = cases
        .iter()
        .map(|case| x.refined(chain.case(rec.guard_of(case))))
        .collect();
    (arms, x.refined(chain.rest()))
}

/// The rows a forward reads.
#[derive(Debug, Clone, Copy, ProvidesStaticType, NoSerialize, Allocative)]
pub struct InputHandle(pub(crate) u32);

starlark_simple_value!(InputHandle);

impl fmt::Display for InputHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "inputs#{}", self.0)
    }
}

pub(crate) fn hold_input(input: Input) -> InputHandle {
    with(|t| {
        t.inputs.push(input);
        InputHandle((t.inputs.len() - 1) as u32)
    })
}

pub(crate) fn input_of(handle: InputHandle) -> Input {
    with(|t| t.inputs[handle.0 as usize].clone())
}

#[starlark_value(type = "inputs")]
impl<'v> StarlarkValue<'v> for InputHandle {
    fn get_methods() -> Option<&'static Methods> {
        Some(INPUT_METHODS_STATICS.methods())
    }

    fn get_attr(&self, attribute: &str, heap: Heap<'v>) -> Option<Star<'v>> {
        crate::star::bind::input_method(*self, attribute, heap)
    }

    fn has_attr(&self, attribute: &str, _heap: Heap<'v>) -> bool {
        crate::star::ops::INPUTS
            .iter()
            .any(|op| op.name == attribute)
    }
}

/// A cache row a forward reads and writes.
#[derive(Debug, Clone, Copy, ProvidesStaticType, NoSerialize, Allocative)]
pub struct CacheHandle(#[allocative(skip)] pub(crate) ValueId);

starlark_simple_value!(CacheHandle);

impl fmt::Display for CacheHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cache#{}", self.0.0)
    }
}

#[starlark_value(type = "cache")]
impl<'v> StarlarkValue<'v> for CacheHandle {}

#[starlark_module]
fn input_methods(builder: &mut MethodsBuilder) {
    /// The rows `fact` holds for.
    fn on(
        this: &InputHandle,
        #[starlark(require = pos)] fact: Star<'_>,
    ) -> anyhow::Result<InputHandle> {
        let (i, p) = (input_of(*this), predicate(fact)?);
        dsl("on", || i.on(p)).map(hold_input)
    }

    /// The rows parted by `cases`, as a value's are.
    fn partition<'v>(
        this: &InputHandle,
        #[starlark(require = pos)] cases: UnpackList<Star<'v>>,
        heap: Heap<'v>,
    ) -> anyhow::Result<Star<'v>> {
        let i = input_of(*this);
        let cases: Vec<Predicate> = cases
            .items
            .into_iter()
            .map(predicate)
            .collect::<anyhow::Result<_>>()?;
        let rec = i.recorder().clone();
        let mut chain = FirstMatch::default();
        let arms: Vec<Star<'v>> = cases
            .iter()
            .map(|case| heap.alloc(hold_input(i.refined(chain.case(rec.guard_of(case))))))
            .collect();
        let rest = hold_input(i.refined(chain.rest()));
        Ok(heap.alloc((heap.alloc(arms), rest)))
    }

    /// `block(l, layer, carried)` over every layer in turn, each traced as
    /// that layer's, its result carried to the next; the last result.
    fn fold_layers<'v>(
        this: &InputHandle,
        #[starlark(require = pos)] layers: UnpackList<Star<'v>>,
        #[starlark(require = pos)] carried: Star<'v>,
        #[starlark(require = pos)] block: Star<'v>,
        eval: &mut Evaluator<'v, '_, '_>,
    ) -> anyhow::Result<Star<'v>> {
        let rec = input_of(*this).recorder().clone();
        let heap = eval.heap();
        let mut carried = carried;
        for (l, layer) in layers.items.into_iter().enumerate() {
            rec.enter(l as u32);
            let out = eval.eval_function(block, &[heap.alloc(l as u32), layer, carried], &[]);
            rec.leave();
            carried = out.map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        Ok(carried)
    }

    /// What `read` computes over the rows of passes that run the reading
    /// `name`.
    fn reading<'v>(
        this: &InputHandle,
        #[starlark(require = pos)] name: &str,
        #[starlark(require = pos)] read: Star<'v>,
        eval: &mut Evaluator<'v, '_, '_>,
    ) -> anyhow::Result<Star<'v>> {
        let i = input_of(*this);
        let rows = dsl("reading", || i.on(fact::reading(name))).map(hold_input)?;
        let heap = eval.heap();
        eval.eval_function(read, &[heap.alloc(rows)], &[])
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

/// The caches a forward declares, while its `caches` runs.
#[derive(Debug, Clone, Copy, ProvidesStaticType, NoSerialize, Allocative)]
pub struct SpecHandle;

starlark_simple_value!(SpecHandle);

impl fmt::Display for SpecHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "caches")
    }
}

#[starlark_value(type = "caches")]
impl<'v> StarlarkValue<'v> for SpecHandle {
    fn get_methods() -> Option<&'static Methods> {
        Some(SPEC_METHODS_STATICS.methods())
    }
}

#[starlark_module]
fn spec_methods(builder: &mut MethodsBuilder) {
    /// A kv space of `dtype`, its rows read through a window of `window`
    /// tokens if one is given.
    fn kv_space(
        #[starlark(this)] _this: &SpecHandle,
        #[starlark(require = pos)] dtype: &DtypeValue,
        #[starlark(default = NoneOr::None)] window: NoneOr<u32>,
    ) -> anyhow::Result<u32> {
        with(|t| {
            let spec = t.spec.get_or_insert_with(HybridSpec::new);
            Ok(match window.into_option() {
                Some(w) => spec.windowed_kv_space(dtype.0, w),
                None => spec.kv_space(dtype.0),
            }
            .0)
        })
    }

    /// The state row `name`, a slab of `slab` of `dtype`; `split` cuts it
    /// between ranks along that axis.
    fn state(
        #[starlark(this)] _this: &SpecHandle,
        #[starlark(require = pos)] name: &str,
        #[starlark(require = pos)] slab: UnpackList<u64>,
        #[starlark(require = pos)] dtype: &DtypeValue,
        #[starlark(require = named, default = NoneOr::None)] split: NoneOr<u32>,
    ) -> anyhow::Result<NoneType> {
        with(|t| {
            let spec = t.spec.get_or_insert_with(HybridSpec::new);
            dsl("state", || {
                let row = spec.state(name, slab.items, dtype.0);
                if let Some(axis) = split.into_option() {
                    row.split(axis);
                }
            })
        })?;
        Ok(NoneType)
    }

    /// The kv row `name` in `space`, its planes of `planes` and heads of
    /// `head_dim`; `heads` cuts it between ranks by head.
    fn kv(
        #[starlark(this)] _this: &SpecHandle,
        #[starlark(require = pos)] space: u32,
        #[starlark(require = pos)] name: &str,
        #[starlark(require = pos)] planes: UnpackList<u64>,
        #[starlark(require = pos)] head_dim: u32,
        #[starlark(require = named, default = false)] heads: bool,
    ) -> anyhow::Result<NoneType> {
        with(|t| {
            let spec = t.spec.get_or_insert_with(HybridSpec::new);
            let row = spec.kv(crate::KvSpace(space), name, planes.items, head_dim);
            if heads {
                row.heads();
            }
        });
        Ok(NoneType)
    }
}

/// A `(fact, arm)` pair, as a list or a tuple.
fn pair<'v>(v: Star<'v>) -> anyhow::Result<(Star<'v>, Star<'v>)> {
    let items: Vec<Star<'v>> = if let Some(list) = starlark::values::list::ListRef::from_value(v) {
        list.iter().collect()
    } else if let Some(tuple) = starlark::values::tuple::TupleRef::from_value(v) {
        tuple.iter().collect()
    } else {
        anyhow::bail!("a switch case is a (fact, arm) pair, not {}", v.get_type());
    };
    match items[..] {
        [fact, arm] => Ok((fact, arm)),
        _ => anyhow::bail!(
            "a switch case is a (fact, arm) pair, not {} items",
            items.len()
        ),
    }
}

/// A switch arm's result: a value, or a tuple of them.
fn arm_of(v: Star<'_>) -> anyhow::Result<Vec<Value>> {
    if let Some(h) = v.downcast_ref::<ValueHandle>() {
        return Ok(vec![held(*h)]);
    }
    let Some(tuple) = starlark::values::tuple::TupleRef::from_value(v) else {
        anyhow::bail!(
            "a switch arm computes a value or a tuple of them, not {}",
            v.get_type()
        );
    };
    tuple
        .iter()
        .map(|item| {
            item.downcast_ref::<ValueHandle>()
                .map(|h| held(*h))
                .ok_or_else(|| anyhow::anyhow!("a switch arm's tuple holds {}", item.get_type()))
        })
        .collect()
}

#[starlark_module]
pub(crate) fn forward(builder: &mut GlobalsBuilder) {
    /// Branches the rows of `x`: each `(fact, arm)` of `cases` computes
    /// `arm(rows)` over the rows it is the first to hold for, `otherwise` over
    /// the rows none holds for, and the arms' results (a value, or a tuple of
    /// them each) are joined over all of them.
    fn switch<'v>(
        #[starlark(require = pos)] x: Star<'v>,
        #[starlark(require = pos)] cases: UnpackList<Star<'v>>,
        #[starlark(require = named, default = NoneOr::None)] otherwise: NoneOr<Star<'v>>,
        eval: &mut Evaluator<'v, '_, '_>,
    ) -> anyhow::Result<Star<'v>> {
        let x = <Value as crate::star::bind::Arg>::arg(Some(x))?;
        let rec = x.rec().clone();
        let mut chain = FirstMatch::default();
        let mut arms: Vec<Vec<Value>> = Vec::new();
        let call = |rows: Value, arm: Star<'v>, eval: &mut Evaluator<'v, '_, '_>| {
            let rows = eval.heap().alloc(hold(rows));
            let out = eval
                .eval_function(arm, &[rows], &[])
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            arm_of(out)
        };
        for case in cases.items {
            let (fact, arm) = pair(case)?;
            let rows = x.refined(chain.case(rec.guard_of(&predicate(fact)?)));
            arms.push(call(rows, arm, eval)?);
        }
        if let NoneOr::Other(arm) = otherwise {
            let rows = x.refined(chain.rest());
            arms.push(call(rows, arm, eval)?);
        }
        let width = arms.first().map_or(1, Vec::len);
        if arms.iter().any(|a| a.len() != width) {
            anyhow::bail!("a switch's arms compute the same number of values");
        }
        let joined: Vec<Value> = (0..width)
            .map(|i| {
                let column: Vec<Value> = arms.iter().map(|a| a[i].clone()).collect();
                dsl("switch", || <Value as crate::Arm>::join(column))
            })
            .collect::<anyhow::Result<_>>()?;
        let heap = eval.heap();
        Ok(match &joined[..] {
            [one] => heap.alloc(hold(one.clone())),
            many => heap.alloc(starlark::values::tuple::AllocTuple(
                many.iter().map(|v| hold(v.clone())).collect::<Vec<_>>(),
            )),
        })
    }

    /// States the block drafter whose proposals `logits` reads out: blocks of
    /// `rows`, masked with `mask_token`, attending both ways if
    /// `bidirectional`, proposing from row `proposals_from` on.
    fn block_drafter<'v>(
        #[starlark(require = pos)] logits: Star<'v>,
        #[starlark(require = named)] rows: u32,
        #[starlark(require = named)] mask_token: u32,
        #[starlark(require = named)] bidirectional: bool,
        #[starlark(require = named)] proposals_from: u32,
    ) -> anyhow::Result<NoneType> {
        let logits = <Value as crate::star::bind::Arg>::arg(Some(logits))?;
        dsl("block_drafter", || {
            logits.rec().block_drafter(crate::BlockDrafter {
                rows,
                mask_token,
                bidirectional,
                proposals_from,
            });
        })?;
        Ok(NoneType)
    }

    /// Joins arms computed over disjoint rows into one value over all of them.
    fn merge<'v>(
        #[starlark(require = pos)] arms: UnpackList<Star<'v>>,
    ) -> anyhow::Result<ValueHandle> {
        let arms: Vec<Value> = arms
            .items
            .into_iter()
            .map(|v| <Value as crate::star::bind::Arg>::arg(Some(v)))
            .collect::<anyhow::Result<_>>()?;
        dsl("merge", || Value::merge(arms)).map(hold)
    }
}

/// The `ops` namespace: the DSL's ops a forward writes.
pub(crate) fn ops(builder: &mut GlobalsBuilder) {
    builder.namespace("ops", |ops| {
        for module in crate::star::ops::MODULES {
            ops.namespace(module, |ns| {
                for op in crate::star::ops::OPS
                    .iter()
                    .filter(|op| op.module == *module)
                {
                    ns.set(op.name, crate::star::bind::OpValue(op));
                }
            });
        }
    });
    builder.namespace("fact", facts);
    crate::star::bind::wholes(builder);
    builder.namespace("seam", seams);
}

#[starlark_module]
fn facts(builder: &mut GlobalsBuilder) {
    const Mask: &str = "mask";
    const Adapter: &str = "adapter";
    const Media: &str = "media";

    /// The rows whose lane carries `what` (`fact.Mask`, `fact.Adapter`,
    /// `fact.Media`).
    fn has(#[starlark(require = pos)] what: &str) -> anyhow::Result<PredicateValue> {
        let carried = match what {
            "mask" => fact::Mask,
            "adapter" => fact::Adapter,
            "media" => fact::Media,
            other => anyhow::bail!("a lane carries a mask, adapter routes or media, not `{other}`"),
        };
        Ok(PredicateValue(fact::has(carried)))
    }

    fn single_token() -> anyhow::Result<PredicateValue> {
        Ok(PredicateValue(fact::single_token()))
    }

    fn drafts() -> anyhow::Result<PredicateValue> {
        Ok(PredicateValue(fact::drafts()))
    }

    fn block_draft() -> anyhow::Result<PredicateValue> {
        Ok(PredicateValue(fact::block_draft()))
    }

    fn scores() -> anyhow::Result<PredicateValue> {
        Ok(PredicateValue(fact::scores()))
    }

    fn bidirectional() -> anyhow::Result<PredicateValue> {
        Ok(PredicateValue(fact::bidirectional()))
    }

    /// The rows of the stream `name` (text, image, video, audio, context,
    /// reference).
    fn stream(#[starlark(require = pos)] name: &str) -> anyhow::Result<PredicateValue> {
        let stream = crate::star::generative::stream(name)?;
        Ok(PredicateValue(fact::stream(stream)))
    }

    /// The rows of the reading `name`.
    fn reading(#[starlark(require = pos)] name: &str) -> anyhow::Result<PredicateValue> {
        Ok(PredicateValue(fact::reading(name)))
    }
}

#[starlark_module]
fn seams(builder: &mut GlobalsBuilder) {
    const ATTN_Q: &str = crate::seam::ATTN_Q.name;
    const ATTN_OUT: &str = crate::seam::ATTN_OUT.name;
    const ATTN_QV: &str = crate::seam::ATTN_QV.name;
    const RECURRENT: &str = crate::seam::RECURRENT.name;
    const IN: &str = crate::seam::IN.name;
    const OUT: &str = crate::seam::OUT.name;
    const MTP: &str = crate::seam::MTP.name;
    const MTP_DRAFTS: &str = crate::seam::MTP_DRAFTS.name;
    const SCORES: &str = crate::seam::SCORES.name;
    const VELOCITY: &str = crate::seam::VELOCITY.name;
    const HIDDEN: &str = crate::seam::HIDDEN.name;
    const PIXELS: &str = crate::seam::PIXELS.name;

    /// Plants the seam `name` over `values`.
    fn at<'v>(
        #[starlark(require = pos)] name: &str,
        #[starlark(require = pos)] values: UnpackList<Star<'v>>,
    ) -> anyhow::Result<NoneType> {
        let values: Vec<Value> = values
            .items
            .into_iter()
            .map(|v| <Value as crate::star::bind::Arg>::arg(Some(v)))
            .collect::<anyhow::Result<_>>()?;
        let refs: Vec<&Value> = values.iter().collect();
        let first = refs
            .first()
            .ok_or_else(|| anyhow::anyhow!("seam `{name}` names no value"))?;
        dsl("seam", || first.rec().seam(name, &refs))?;
        Ok(NoneType)
    }
}
