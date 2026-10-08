//! What a forward may use: the rows it reads, the facts it branches them on,
//! the DSL's ops over them and the seams it plants.
//!
//! The DSL's values hold the trace being recorded, which no Starlark heap
//! may: a forward handles them by place in the trail this thread records,
//! and the trail is emptied before the trace finishes.

use std::cell::RefCell;
use std::fmt;

use allocative::Allocative;
use poem_dsl::{HybridSpec, Input, Predicate, Value, ValueId, fact, ops};
use starlark::any::ProvidesStaticType;
use starlark::environment::{GlobalsBuilder, Methods, MethodsBuilder};
use starlark::eval::Evaluator;
use starlark::starlark_simple_value;
use starlark::values::float::UnpackFloat;
use starlark::values::list::UnpackList;
use starlark::values::none::{NoneOr, NoneType};
use starlark::values::{
    Heap, NoSerialize, StarlarkPagableUnsupported, StarlarkValue, Value as Star,
};
use starlark_derive::{starlark_module, starlark_value};

use crate::values::{DtypeValue, PredicateValue, predicate, weight};

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
fn dsl<R>(what: &str, op: impl FnOnce() -> R) -> anyhow::Result<R> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(op)).map_err(|panic| {
        let why = panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .unwrap_or_else(|| "a panic with no message".to_string());
        anyhow::anyhow!("{what}: {why}")
    })
}

/// A DSL value a forward holds.
#[derive(
    Debug, Clone, Copy, ProvidesStaticType, NoSerialize, StarlarkPagableUnsupported, Allocative,
)]
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

fn value_of(value: Star<'_>) -> anyhow::Result<Value> {
    value
        .downcast_ref::<ValueHandle>()
        .map(|h| held(*h))
        .ok_or_else(|| anyhow::anyhow!("a value was wanted, not {}", value.get_type()))
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
    let mut rest = poem_dsl::Guard::Always;
    let rec = x.rec().clone();
    let mut arms = Vec::new();
    for case in cases {
        let holds = rec.guard_of(&case);
        arms.push(<Value as poem_dsl::Refine>::refined(
            x,
            poem_dsl::Guard::and(rest.clone(), holds.clone()),
        ));
        rest = poem_dsl::Guard::and(rest, poem_dsl::Guard::not(holds));
    }
    (arms, <Value as poem_dsl::Refine>::refined(x, rest))
}

/// The rows a forward reads.
#[derive(
    Debug, Clone, Copy, ProvidesStaticType, NoSerialize, StarlarkPagableUnsupported, Allocative,
)]
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

fn input_of(handle: InputHandle) -> Input {
    with(|t| t.inputs[handle.0 as usize].clone())
}

fn inputs_of(value: Star<'_>) -> anyhow::Result<Input> {
    value
        .downcast_ref::<InputHandle>()
        .map(|h| input_of(*h))
        .ok_or_else(|| anyhow::anyhow!("inputs were wanted, not {}", value.get_type()))
}

#[starlark_value(type = "inputs")]
impl<'v> StarlarkValue<'v> for InputHandle {
    fn get_methods() -> Option<&'static Methods> {
        Some(INPUT_METHODS_STATICS.methods())
    }
}

/// A cache row a forward reads and writes.
#[derive(
    Debug, Clone, Copy, ProvidesStaticType, NoSerialize, StarlarkPagableUnsupported, Allocative,
)]
pub struct CacheHandle(#[allocative(skip)] pub(crate) ValueId);

starlark_simple_value!(CacheHandle);

impl fmt::Display for CacheHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cache#{}", self.0.0)
    }
}

#[starlark_value(type = "cache")]
impl<'v> StarlarkValue<'v> for CacheHandle {}

fn cache_of(value: Star<'_>) -> anyhow::Result<ValueId> {
    value
        .downcast_ref::<CacheHandle>()
        .map(|h| h.0)
        .ok_or_else(|| anyhow::anyhow!("a cache was wanted, not {}", value.get_type()))
}

#[starlark_module]
fn input_methods(builder: &mut MethodsBuilder) {
    fn tokens(this: &InputHandle) -> anyhow::Result<ValueHandle> {
        let i = input_of(*this);
        dsl("tokens", || i.tokens()).map(hold)
    }

    fn positions(this: &InputHandle) -> anyhow::Result<ValueHandle> {
        let i = input_of(*this);
        dsl("positions", || i.positions()).map(hold)
    }

    fn mask(this: &InputHandle) -> anyhow::Result<ValueHandle> {
        let i = input_of(*this);
        dsl("mask", || i.mask()).map(hold)
    }

    fn adapter_routes(this: &InputHandle) -> anyhow::Result<ValueHandle> {
        let i = input_of(*this);
        dsl("adapter_routes", || i.adapter_routes()).map(hold)
    }

    fn readout_rows(this: &InputHandle) -> anyhow::Result<ValueHandle> {
        let i = input_of(*this);
        dsl("readout_rows", || i.readout_rows()).map(hold)
    }

    /// The kv cache row `name` the caches declare.
    fn kv(
        this: &InputHandle,
        #[starlark(require = pos)] name: &str,
    ) -> anyhow::Result<CacheHandle> {
        let i = input_of(*this);
        dsl("kv", || i.kv(name)).map(CacheHandle)
    }

    fn write_page(
        this: &InputHandle,
        #[starlark(require = pos)] row: &str,
    ) -> anyhow::Result<ValueHandle> {
        let i = input_of(*this);
        dsl("write_page", || i.write_page(row)).map(hold)
    }

    fn write_offset(
        this: &InputHandle,
        #[starlark(require = pos)] row: &str,
    ) -> anyhow::Result<ValueHandle> {
        let i = input_of(*this);
        dsl("write_offset", || i.write_offset(row)).map(hold)
    }

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
        let mut rest = poem_dsl::Guard::Always;
        let rec = i.recorder().clone();
        let mut arms = Vec::new();
        for case in cases {
            let holds = rec.guard_of(&case);
            arms.push(heap.alloc(hold_input(<Input as poem_dsl::Refine>::refined(
                &i,
                poem_dsl::Guard::and(rest.clone(), holds.clone()),
            ))));
            rest = poem_dsl::Guard::and(rest, poem_dsl::Guard::not(holds));
        }
        let rest = hold_input(<Input as poem_dsl::Refine>::refined(&i, rest));
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
#[derive(
    Debug, Clone, Copy, ProvidesStaticType, NoSerialize, StarlarkPagableUnsupported, Allocative,
)]
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
            let row = spec.kv(poem_dsl::KvSpace(space), name, planes.items, head_dim);
            if heads {
                row.heads();
            }
        });
        Ok(NoneType)
    }
}

fn f(x: UnpackFloat) -> f32 {
    x.0 as f32
}

fn tuple2<'v>(heap: Heap<'v>, (a, b): (Value, Value)) -> Star<'v> {
    heap.alloc((hold(a), hold(b)))
}

#[starlark_module]
pub(crate) fn forward(builder: &mut GlobalsBuilder) {
    /// Joins arms computed over disjoint rows into one value over all of them.
    fn merge<'v>(
        #[starlark(require = pos)] arms: UnpackList<Star<'v>>,
    ) -> anyhow::Result<ValueHandle> {
        let arms: Vec<Value> = arms
            .items
            .into_iter()
            .map(value_of)
            .collect::<anyhow::Result<_>>()?;
        dsl("merge", || Value::merge(arms)).map(hold)
    }
}

/// The `ops` namespace: the DSL's ops a forward writes.
pub(crate) fn ops(builder: &mut GlobalsBuilder) {
    builder.namespace("ops", |ops| {
        ops.namespace("attn", attn);
        ops.namespace("elemwise", elemwise);
        ops.namespace("layout", layout);
        ops.namespace("linear", linear);
    });
    builder.namespace("fact", facts);
    builder.namespace("seam", seams);
}

#[starlark_module]
fn attn(builder: &mut GlobalsBuilder) {
    fn plan_decode<'v>(
        #[starlark(require = pos)] inputs: Star<'v>,
        #[starlark(require = pos)] q_heads: u32,
        #[starlark(require = pos)] kv_heads: u32,
        #[starlark(require = pos)] head_dim: u32,
        #[starlark(require = pos)] window: NoneOr<u32>,
    ) -> anyhow::Result<ValueHandle> {
        let i = inputs_of(inputs)?;
        dsl("plan_decode", || {
            ops::attn::plan_decode(&i, q_heads, kv_heads, head_dim, window.into_option())
        })
        .map(hold)
    }

    fn plan_prefill<'v>(
        #[starlark(require = pos)] inputs: Star<'v>,
        #[starlark(require = pos)] q_heads: u32,
        #[starlark(require = pos)] kv_heads: u32,
        #[starlark(require = pos)] head_dim: u32,
        #[starlark(require = pos)] window: NoneOr<u32>,
    ) -> anyhow::Result<ValueHandle> {
        let i = inputs_of(inputs)?;
        dsl("plan_prefill", || {
            ops::attn::plan_prefill(&i, q_heads, kv_heads, head_dim, window.into_option())
        })
        .map(hold)
    }

    fn prefill<'v>(
        #[starlark(require = pos)] q: Star<'v>,
        #[starlark(require = pos)] plan: Star<'v>,
        #[starlark(require = pos)] pages: Star<'v>,
        #[starlark(require = pos)] window: NoneOr<u32>,
        #[starlark(require = pos)] head_dim: u32,
        #[starlark(require = pos)] kv_heads: u32,
        #[starlark(require = pos)] sm_scale: UnpackFloat,
    ) -> anyhow::Result<ValueHandle> {
        let (q, plan, pages) = (value_of(q)?, value_of(plan)?, cache_of(pages)?);
        dsl("prefill", || {
            ops::attn::prefill(
                &q,
                &plan,
                pages,
                window.into_option(),
                head_dim,
                kv_heads,
                f(sm_scale),
            )
        })
        .map(hold)
    }

    fn prefill_lse<'v>(
        #[starlark(require = pos)] q: Star<'v>,
        #[starlark(require = pos)] plan: Star<'v>,
        #[starlark(require = pos)] pages: Star<'v>,
        #[starlark(require = pos)] window: NoneOr<u32>,
        #[starlark(require = pos)] head_dim: u32,
        #[starlark(require = pos)] kv_heads: u32,
        #[starlark(require = pos)] sm_scale: UnpackFloat,
        heap: Heap<'v>,
    ) -> anyhow::Result<Star<'v>> {
        let (q, plan, pages) = (value_of(q)?, value_of(plan)?, cache_of(pages)?);
        dsl("prefill_lse", || {
            ops::attn::prefill_lse(
                &q,
                &plan,
                pages,
                window.into_option(),
                head_dim,
                kv_heads,
                f(sm_scale),
            )
        })
        .map(|o| tuple2(heap, o))
    }

    fn masked<'v>(
        #[starlark(require = pos)] q: Star<'v>,
        #[starlark(require = pos)] plan: Star<'v>,
        #[starlark(require = pos)] mask: Star<'v>,
        #[starlark(require = pos)] pages: Star<'v>,
        #[starlark(require = pos)] window: NoneOr<u32>,
        #[starlark(require = pos)] head_dim: u32,
        #[starlark(require = pos)] kv_heads: u32,
        #[starlark(require = pos)] causal: bool,
        #[starlark(require = pos)] sm_scale: UnpackFloat,
    ) -> anyhow::Result<ValueHandle> {
        let (q, plan, mask, pages) = (
            value_of(q)?,
            value_of(plan)?,
            value_of(mask)?,
            cache_of(pages)?,
        );
        dsl("masked", || {
            ops::attn::masked(
                &q,
                &plan,
                &mask,
                pages,
                window.into_option(),
                head_dim,
                kv_heads,
                causal,
                f(sm_scale),
            )
        })
        .map(hold)
    }

    fn decode<'v>(
        #[starlark(require = pos)] q: Star<'v>,
        #[starlark(require = pos)] plan: Star<'v>,
        #[starlark(require = pos)] pages: Star<'v>,
        #[starlark(require = pos)] window: NoneOr<u32>,
        #[starlark(require = pos)] head_dim: u32,
        #[starlark(require = pos)] sm_scale: UnpackFloat,
    ) -> anyhow::Result<ValueHandle> {
        let (q, plan, pages) = (value_of(q)?, value_of(plan)?, cache_of(pages)?);
        dsl("decode", || {
            ops::attn::decode(
                &q,
                &plan,
                pages,
                window.into_option(),
                head_dim,
                f(sm_scale),
            )
        })
        .map(hold)
    }

    fn kv_append<'v>(
        #[starlark(require = pos)] k: Star<'v>,
        #[starlark(require = pos)] v: Star<'v>,
        #[starlark(require = pos)] pages: Star<'v>,
        #[starlark(require = pos)] write_page: Star<'v>,
        #[starlark(require = pos)] write_offset: Star<'v>,
    ) -> anyhow::Result<NoneType> {
        let (k, v, pages, page, offset) = (
            value_of(k)?,
            value_of(v)?,
            cache_of(pages)?,
            value_of(write_page)?,
            value_of(write_offset)?,
        );
        dsl("kv_append", || {
            ops::attn::kv_append(&k, &v, pages, &page, &offset)
        })?;
        Ok(NoneType)
    }

    fn logit_softcap<'v>(
        #[starlark(require = pos)] x: Star<'v>,
        #[starlark(require = pos)] cap: UnpackFloat,
    ) -> anyhow::Result<ValueHandle> {
        let x = value_of(x)?;
        dsl("logit_softcap", || ops::attn::logit_softcap(&x, f(cap))).map(hold)
    }
}

#[starlark_module]
fn elemwise(builder: &mut GlobalsBuilder) {
    fn rmsnorm_no_scale<'v>(
        #[starlark(require = pos)] x: Star<'v>,
        #[starlark(require = pos)] width: u32,
        #[starlark(require = pos)] eps: UnpackFloat,
    ) -> anyhow::Result<ValueHandle> {
        let x = value_of(x)?;
        dsl("rmsnorm_no_scale", || {
            ops::elemwise::rmsnorm_no_scale(&x, width, f(eps))
        })
        .map(hold)
    }

    fn rmsnorm_plus_one<'v>(
        #[starlark(require = pos)] x: Star<'v>,
        #[starlark(require = pos)] w: Star<'v>,
        #[starlark(require = pos)] eps: UnpackFloat,
    ) -> anyhow::Result<ValueHandle> {
        let (x, w) = (value_of(x)?, weight(w)?);
        dsl("rmsnorm_plus_one", || {
            ops::elemwise::rmsnorm_plus_one(&x, &w, f(eps))
        })
        .map(hold)
    }

    fn rmsnorm<'v>(
        #[starlark(require = pos)] x: Star<'v>,
        #[starlark(require = pos)] w: Star<'v>,
        #[starlark(require = pos)] eps: UnpackFloat,
    ) -> anyhow::Result<ValueHandle> {
        let (x, w) = (value_of(x)?, weight(w)?);
        dsl("rmsnorm", || ops::elemwise::rmsnorm(&x, &w, f(eps))).map(hold)
    }

    fn rope_full<'v>(
        #[starlark(require = pos)] q: Star<'v>,
        #[starlark(require = pos)] k: Star<'v>,
        #[starlark(require = pos)] positions: Star<'v>,
        #[starlark(require = pos)] head_dim: u32,
        #[starlark(require = pos)] theta: UnpackFloat,
        #[starlark(require = pos)] interleaved: bool,
        heap: Heap<'v>,
    ) -> anyhow::Result<Star<'v>> {
        let (q, k, p) = (value_of(q)?, value_of(k)?, value_of(positions)?);
        dsl("rope_full", || {
            ops::elemwise::rope_full(&q, &k, &p, head_dim, f(theta), interleaved)
        })
        .map(|o| tuple2(heap, o))
    }

    fn residual_add<'v>(
        #[starlark(require = pos)] x: Star<'v>,
        #[starlark(require = pos)] y: Star<'v>,
    ) -> anyhow::Result<ValueHandle> {
        let (x, y) = (value_of(x)?, value_of(y)?);
        dsl("residual_add", || ops::elemwise::residual_add(&x, &y)).map(hold)
    }

    fn gate_sigmoid_mul<'v>(
        #[starlark(require = pos)] x: Star<'v>,
        #[starlark(require = pos)] gate: Star<'v>,
    ) -> anyhow::Result<ValueHandle> {
        let (x, gate) = (value_of(x)?, value_of(gate)?);
        dsl("gate_sigmoid_mul", || {
            ops::elemwise::gate_sigmoid_mul(&x, &gate)
        })
        .map(hold)
    }
}

#[starlark_module]
fn layout(builder: &mut GlobalsBuilder) {
    fn embed<'v>(
        #[starlark(require = pos)] ids: Star<'v>,
        #[starlark(require = pos)] table: Star<'v>,
        #[starlark(require = pos)] vocab: u32,
    ) -> anyhow::Result<ValueHandle> {
        let (ids, table) = (value_of(ids)?, weight(table)?);
        dsl("embed", || ops::layout::embed(&ids, &table, vocab)).map(hold)
    }

    fn split_qkv<'v>(
        #[starlark(require = pos)] packed: Star<'v>,
        #[starlark(require = pos)] q_width: u32,
        #[starlark(require = pos)] kv_width: u32,
        heap: Heap<'v>,
    ) -> anyhow::Result<Star<'v>> {
        let packed = value_of(packed)?;
        let (q, k, v) = dsl("split_qkv", || {
            ops::layout::split_qkv(&packed, q_width, kv_width)
        })?;
        Ok(heap.alloc((hold(q), hold(k), hold(v))))
    }

    fn gather_rows<'v>(
        #[starlark(require = pos)] x: Star<'v>,
        #[starlark(require = pos)] rows: Star<'v>,
    ) -> anyhow::Result<ValueHandle> {
        let (x, rows) = (value_of(x)?, value_of(rows)?);
        dsl("gather_rows", || ops::layout::gather_rows(&x, &rows)).map(hold)
    }
}

#[starlark_module]
fn linear(builder: &mut GlobalsBuilder) {
    fn matmul<'v>(
        #[starlark(require = pos)] x: Star<'v>,
        #[starlark(require = pos)] w: Star<'v>,
    ) -> anyhow::Result<ValueHandle> {
        let (x, w) = (value_of(x)?, weight(w)?);
        dsl("matmul", || ops::linear::matmul(&x, &w)).map(hold)
    }

    fn lora_correct<'v>(
        #[starlark(require = pos)] x: Star<'v>,
        #[starlark(require = pos)] bank_a: Star<'v>,
        #[starlark(require = pos)] bank_b: Star<'v>,
        #[starlark(require = pos)] routes: Star<'v>,
        #[starlark(require = pos)] y: Star<'v>,
    ) -> anyhow::Result<ValueHandle> {
        let (x, a, b, routes, y) = (
            value_of(x)?,
            weight(bank_a)?,
            weight(bank_b)?,
            value_of(routes)?,
            value_of(y)?,
        );
        dsl("lora_correct", || {
            ops::linear::lora_correct(&x, &a, &b, &routes, &y)
        })
        .map(hold)
    }

    fn mlp_swiglu<'v>(
        #[starlark(require = pos)] packed: Star<'v>,
        #[starlark(require = pos)] intermediate: u32,
    ) -> anyhow::Result<ValueHandle> {
        let packed = value_of(packed)?;
        dsl("mlp_swiglu", || {
            ops::linear::mlp_swiglu(&packed, intermediate)
        })
        .map(hold)
    }

    fn lm_head<'v>(
        #[starlark(require = pos)] x: Star<'v>,
        #[starlark(require = pos)] w: Star<'v>,
    ) -> anyhow::Result<ValueHandle> {
        let (x, w) = (value_of(x)?, weight(w)?);
        dsl("lm_head", || ops::linear::lm_head(&x, &w)).map(hold)
    }
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

    /// The rows of the reading `name`.
    fn reading(#[starlark(require = pos)] name: &str) -> anyhow::Result<PredicateValue> {
        Ok(PredicateValue(fact::reading(name)))
    }

    /// The rows whose inferlet turned the flag `name` on.
    fn flag(#[starlark(require = pos)] name: &str) -> anyhow::Result<PredicateValue> {
        Ok(PredicateValue(fact::flag(name)))
    }
}

#[starlark_module]
fn seams(builder: &mut GlobalsBuilder) {
    const ATTN_Q: &str = "attn.q";
    const ATTN_OUT: &str = "attn.out";
    const SCORES: &str = "attn.scores";
    const HIDDEN: &str = "hidden";
    const VELOCITY: &str = "velocity";

    /// Plants the seam `name` over `values`.
    fn at<'v>(
        #[starlark(require = pos)] name: &str,
        #[starlark(require = pos)] values: UnpackList<Star<'v>>,
    ) -> anyhow::Result<NoneType> {
        let values: Vec<Value> = values
            .items
            .into_iter()
            .map(value_of)
            .collect::<anyhow::Result<_>>()?;
        let refs: Vec<&Value> = values.iter().collect();
        let first = refs
            .first()
            .ok_or_else(|| anyhow::anyhow!("seam `{name}` names no value"))?;
        dsl("seam", || first.rec().seam(name, &refs))?;
        Ok(NoneType)
    }
}
