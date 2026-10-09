//! What a format may use: the names and attributes of the checkpoint it
//! recognizes, and the reads that land its tensors on a deployment's weights.
//!
//! A read is stated, not performed: `formats.poem` names what lands where,
//! and the builder the contract is built by performs it, refusing a missing or
//! misshapen tensor.

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

use crate::star::values::{DtypeValue, weight};
use checkpoint::contract::{Expr, TensorContract, TensorType};
use checkpoint::types::Encoding;

/// One read a format states.
pub(crate) enum Read {
    One(Weight, String),
    Concat(Weight, Vec<String>),
    Stack(Weight, Vec<Vec<String>>),
    Over(Weight, String, Expr),
    Expr(Weight, Expr),
    Own(Weight),
    Push(TensorContract),
    Extend(Vec<TensorContract>),
}

impl Read {
    /// Lands this read on `b`.
    pub(crate) fn onto(
        self,
        b: &mut crate::import::Builder<'_>,
    ) -> Result<(), crate::import::Error> {
        match self {
            Read::One(w, from) => b.read(&w, from),
            Read::Concat(w, from) => b.read_concat(&w, from),
            Read::Stack(w, rows) => b.read_stack(&w, rows),
            Read::Over(w, from, over) => b.read_over(&w, from, |_| over),
            Read::Expr(w, expr) => b.read_expr(&w, expr),
            Read::Own(w) => b.read_own(&w),
            Read::Push(t) => {
                b.push(t);
                Ok(())
            }
            Read::Extend(ts) => {
                b.extend(ts);
                Ok(())
            }
        }
    }
}

/// A tensor a read computes from the checkpoint's.
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize, StarlarkPagableUnsupported, Allocative)]
pub struct ExprValue(#[allocative(skip)] pub(crate) Expr);

starlark_simple_value!(ExprValue);

impl fmt::Display for ExprValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "expr({})", self.0.node_name())
    }
}

#[starlark_value(type = "expr")]
impl<'v> StarlarkValue<'v> for ExprValue {
    fn get_methods() -> Option<&'static Methods> {
        Some(EXPR_METHODS_STATICS.methods())
    }
}

fn expr_of(v: Value<'_>) -> anyhow::Result<Expr> {
    v.downcast_ref::<ExprValue>()
        .map(|e| e.0.clone())
        .ok_or_else(|| anyhow::anyhow!("an expr was wanted, not {}", v.get_type()))
}

/// A tensor a contract states whole: its name, what computes it, its shape
/// and encoding, and what scales it.
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize, StarlarkPagableUnsupported, Allocative)]
pub struct TensorValue(#[allocative(skip)] pub(crate) TensorContract);

starlark_simple_value!(TensorValue);

impl fmt::Display for TensorValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tensor({:?})", self.0.name)
    }
}

#[starlark_value(type = "tensor")]
impl<'v> StarlarkValue<'v> for TensorValue {}

/// How a tensor's bytes are stored.
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize, StarlarkPagableUnsupported, Allocative)]
pub struct EncodingValue(#[allocative(skip)] pub(crate) Encoding);

starlark_simple_value!(EncodingValue);

impl fmt::Display for EncodingValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

#[starlark_value(type = "encoding")]
impl<'v> StarlarkValue<'v> for EncodingValue {
    fn get_methods() -> Option<&'static Methods> {
        Some(ENCODING_METHODS_STATICS.methods())
    }

    fn equals(&self, other: Value<'v>) -> starlark::Result<bool> {
        Ok(other
            .downcast_ref::<EncodingValue>()
            .is_some_and(|other| other.0 == self.0))
    }
}

/// A checkpoint attribute as Starlark spells it.
fn cbor<'v>(v: &ztensor::format::cbor::Value, heap: Heap<'v>) -> anyhow::Result<Value<'v>> {
    use ztensor::format::cbor::Value as C;
    Ok(match v {
        C::Bool(b) => Value::new_bool(*b),
        C::Uint(n) => heap.alloc(*n),
        C::Nint(n) => heap.alloc(-1 - i64::try_from(*n)?),
        C::Float(x) => heap.alloc(*x),
        C::Text(t) => heap.alloc(t.as_str()),
        C::Array(items) => {
            let items = items
                .iter()
                .map(|item| cbor(item, heap))
                .collect::<anyhow::Result<Vec<_>>>()?;
            heap.alloc(items)
        }
        other => anyhow::bail!("an attribute of {other:?} has no Starlark spelling here"),
    })
}

#[starlark_module]
fn encoding_methods(builder: &mut MethodsBuilder) {
    /// The dtype of plain values, or `None` for a quantized encoding.
    #[starlark(attribute)]
    fn raw(this: &EncodingValue) -> anyhow::Result<NoneOr<DtypeValue>> {
        Ok(match this.0 {
            Encoding::Raw(dtype) => NoneOr::Other(DtypeValue(dtype)),
            _ => NoneOr::None,
        })
    }
}

fn tensor_of(v: Value<'_>) -> anyhow::Result<TensorContract> {
    v.downcast_ref::<TensorValue>()
        .map(|t| t.0.clone())
        .ok_or_else(|| anyhow::anyhow!("a tensor was wanted, not {}", v.get_type()))
}

fn axis(axis: u32) -> anyhow::Result<u8> {
    u8::try_from(axis).map_err(|_| anyhow::anyhow!("axis {axis} is past any tensor's"))
}

#[starlark_module]
fn expr_methods(builder: &mut MethodsBuilder) {
    /// `len` items of `axis` from `start`.
    fn slice(
        this: &ExprValue,
        #[starlark(require = pos)] axis: u32,
        #[starlark(require = pos)] start: i64,
        #[starlark(require = pos)] len: i64,
    ) -> anyhow::Result<ExprValue> {
        Ok(ExprValue(this.0.clone().slice(
            self::axis(axis)?,
            start,
            len,
        )))
    }

    /// Every `stride`-th run of `len` along `axis`.
    fn select(
        this: &ExprValue,
        #[starlark(require = pos)] axis: u32,
        #[starlark(require = pos)] stride: i64,
        #[starlark(require = pos)] len: i64,
    ) -> anyhow::Result<ExprValue> {
        Ok(ExprValue(this.0.clone().select(
            self::axis(axis)?,
            stride,
            len,
        )))
    }

    /// `len` items of `axis` from `start`, every `step`-th.
    fn stride(
        this: &ExprValue,
        #[starlark(require = pos)] axis: u32,
        #[starlark(require = pos)] start: i64,
        #[starlark(require = pos)] len: i64,
        #[starlark(require = pos)] step: i64,
    ) -> anyhow::Result<ExprValue> {
        Ok(ExprValue(this.0.clone().stride(
            self::axis(axis)?,
            start,
            len,
            step,
        )))
    }

    /// The items of `axis` at `indices`.
    fn gather(
        this: &ExprValue,
        #[starlark(require = pos)] axis: u32,
        #[starlark(require = pos)] indices: UnpackList<i64>,
    ) -> anyhow::Result<ExprValue> {
        Ok(ExprValue(
            this.0.clone().gather(self::axis(axis)?, indices.items),
        ))
    }

    /// The same bytes read as `shape` (`-1` the one extent left to infer)
    /// of `encoding`.
    fn transmute(
        this: &ExprValue,
        #[starlark(require = pos)] shape: UnpackList<i64>,
        #[starlark(require = pos)] encoding: &EncodingValue,
    ) -> anyhow::Result<ExprValue> {
        Ok(ExprValue(this.0.clone().transmute(TensorType::new(
            shape.items,
            encoding.0.clone(),
        ))))
    }

    /// The values re-encoded as `encoding`.
    fn cast(
        this: &ExprValue,
        #[starlark(require = pos)] encoding: &EncodingValue,
    ) -> anyhow::Result<ExprValue> {
        Ok(ExprValue(this.0.clone().cast(encoding.0.clone())))
    }

    /// The values times `factor`.
    fn scale(
        this: &ExprValue,
        #[starlark(require = pos)] factor: UnpackFloat,
    ) -> anyhow::Result<ExprValue> {
        Ok(ExprValue(this.0.clone().scale(factor.0 as f32)))
    }

    /// The values plus `by`.
    fn bias(
        this: &ExprValue,
        #[starlark(require = pos)] by: UnpackFloat,
    ) -> anyhow::Result<ExprValue> {
        Ok(ExprValue(this.0.clone().bias(by.0 as f32)))
    }

    /// The values put through `op`: "neg_ln" (`ln(-x)`), "sqrt" or "rsqrt".
    fn unary(this: &ExprValue, #[starlark(require = pos)] op: &str) -> anyhow::Result<ExprValue> {
        use checkpoint::contract::UnaryOp;
        let op = match op {
            "neg_ln" => UnaryOp::NegLn,
            "sqrt" => UnaryOp::Sqrt,
            "rsqrt" => UnaryOp::Rsqrt,
            other => anyhow::bail!("a unary op is neg_ln, sqrt or rsqrt, not {other:?}"),
        };
        Ok(ExprValue(this.0.clone().unary(op)))
    }

    /// This rank's share along `axis`.
    fn shard(this: &ExprValue, #[starlark(require = pos)] axis: u32) -> anyhow::Result<ExprValue> {
        Ok(ExprValue(this.0.clone().shard(self::axis(axis)?)))
    }
}

/// What a format may read of a checkpoint without reading its bytes: its
/// tensors' names, shapes and stored encodings, and its attributes.
pub(crate) struct Snapshot {
    names: std::collections::HashSet<String>,
    tensors: std::collections::HashMap<String, (Vec<u64>, Result<Encoding, String>)>,
    attributes: Option<ztensor::format::cbor::Value>,
}

impl Snapshot {
    pub(crate) fn of(src: &ztensor::Source) -> Snapshot {
        let tensors = src
            .names()
            .filter_map(|name| {
                let tensor = src.get(name)?;
                let encoding =
                    checkpoint::file::encoding_of(&tensor).map_err(|why| why.to_string());
                Some((name.to_string(), (tensor.shape().to_vec(), encoding)))
            })
            .collect();
        Snapshot {
            names: src.names().map(str::to_string).collect(),
            tensors,
            attributes: src.attributes().cloned(),
        }
    }
}

/// A tensor a format asked of a checkpoint that does not hold it.
pub(crate) fn missed() -> Option<String> {
    MISSED.with(|m| m.borrow_mut().take())
}

fn tensor<R>(
    name: &str,
    f: impl FnOnce(&(Vec<u64>, Result<Encoding, String>)) -> R,
) -> anyhow::Result<R> {
    source(|src| src.tensors.get(name).map(f))?.ok_or_else(|| {
        MISSED.with(|m| *m.borrow_mut() = Some(name.to_string()));
        anyhow::anyhow!("the checkpoint holds no tensor `{name}`")
    })
}

thread_local! {
    /// The reads the format reading a checkpoint states.
    pub(crate) static READS: RefCell<Vec<Read>> = const { RefCell::new(Vec::new()) };
    /// The checkpoint a format is recognizing or reading.
    pub(crate) static SOURCE: RefCell<Option<Snapshot>> = const { RefCell::new(None) };
    /// The tensor a format asked for and the checkpoint does not hold.
    static MISSED: RefCell<Option<String>> = const { RefCell::new(None) };
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
impl<'v> StarlarkValue<'v> for SourceHandle {}

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

    /// `w` lands what `expr` computes.
    fn read_expr<'v>(
        #[starlark(this)] _this: &ReadsHandle,
        #[starlark(require = pos)] w: Value<'v>,
        #[starlark(require = pos)] expr: Value<'v>,
    ) -> anyhow::Result<NoneType> {
        let (w, expr) = (weight(w)?, expr_of(expr)?);
        READS.with(|r| r.borrow_mut().push(Read::Expr(w, expr)));
        Ok(NoneType)
    }

    /// `w` lands what `over(src(name))` computes of the tensor `name`, as
    /// stored, then brought to `w`'s dtype.
    fn read_over<'v>(
        #[starlark(this)] _this: &ReadsHandle,
        #[starlark(require = pos)] w: Value<'v>,
        #[starlark(require = pos)] name: String,
        #[starlark(require = pos)] over: Value<'v>,
        eval: &mut starlark::eval::Evaluator<'v, '_, '_>,
    ) -> anyhow::Result<NoneType> {
        let w = weight(w)?;
        let from = eval.heap().alloc(ExprValue(Expr::src(name.clone())));
        let expr = eval
            .eval_function(over, &[from], &[])
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let expr = expr_of(expr)?;
        READS.with(|r| r.borrow_mut().push(Read::Over(w, name, expr)));
        Ok(NoneType)
    }

    /// `w` lands its rows from `rows`, each row's tensors joined.
    fn read_stack<'v>(
        #[starlark(this)] _this: &ReadsHandle,
        #[starlark(require = pos)] w: Value<'v>,
        #[starlark(require = pos)] rows: UnpackList<UnpackList<String>>,
    ) -> anyhow::Result<NoneType> {
        let w = weight(w)?;
        let rows = rows.items.into_iter().map(|r| r.items).collect();
        READS.with(|r| r.borrow_mut().push(Read::Stack(w, rows)));
        Ok(NoneType)
    }

    /// The contract states `tensor` as it is.
    fn push<'v>(
        #[starlark(this)] _this: &ReadsHandle,
        #[starlark(require = pos)] tensor: Value<'v>,
    ) -> anyhow::Result<NoneType> {
        let t = tensor_of(tensor)?;
        READS.with(|r| r.borrow_mut().push(Read::Push(t)));
        Ok(NoneType)
    }

    /// The contract states each of `tensors` as it is.
    fn extend<'v>(
        #[starlark(this)] _this: &ReadsHandle,
        #[starlark(require = pos)] tensors: UnpackList<Value<'v>>,
    ) -> anyhow::Result<NoneType> {
        let ts = tensors
            .items
            .into_iter()
            .map(tensor_of)
            .collect::<anyhow::Result<_>>()?;
        READS.with(|r| r.borrow_mut().push(Read::Extend(ts)));
        Ok(NoneType)
    }

    /// `w` lands as an artifact of it holds it: this rank's share, by name.
    fn read_own<'v>(
        #[starlark(this)] _this: &ReadsHandle,
        #[starlark(require = pos)] w: Value<'v>,
    ) -> anyhow::Result<NoneType> {
        let w = weight(w)?;
        READS.with(|r| r.borrow_mut().push(Read::Own(w)));
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

    /// The checkpoint's attribute `key` as Starlark spells it (a bool,
    /// number, text, or a list of them), or `None`.
    fn attribute_value<'v>(
        #[starlark(require = pos)] key: &str,
        heap: Heap<'v>,
    ) -> anyhow::Result<Value<'v>> {
        let found = source(|src| src.attributes.as_ref().and_then(|a| a.get(key)).cloned())?;
        Ok(match found {
            Some(v) => cbor(&v, heap)?,
            None => Value::new_none(),
        })
    }

    /// A tensor the contract states itself: `values` in `shape`, stored as
    /// raw `dtype`, each value rounded to it; `name` names it in a refusal.
    fn constant(
        #[starlark(require = pos)] name: &str,
        #[starlark(require = pos)] values: UnpackList<UnpackFloat>,
        #[starlark(require = pos)] shape: UnpackList<i64>,
        #[starlark(require = pos)] dtype: &DtypeValue,
    ) -> anyhow::Result<ExprValue> {
        let values: Vec<f32> = values.items.iter().map(|v| v.0 as f32).collect();
        crate::import::constant(name, &values, shape.items, dtype.0)
            .map(ExprValue)
            .map_err(|why| anyhow::anyhow!("{why}"))
    }

    /// The checkpoint's attribute `key`, as text, or `None`.
    fn text_attribute(#[starlark(require = pos)] key: &str) -> anyhow::Result<NoneOr<String>> {
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

    /// Whether the checkpoint holds a tensor named `name`.
    fn has(#[starlark(require = pos)] name: &str) -> anyhow::Result<bool> {
        source(|src| src.names.contains(name))
    }

    /// Whether the checkpoint holds a tensor whose name begins with `prefix`.
    fn has_prefix(#[starlark(require = pos)] prefix: &str) -> anyhow::Result<bool> {
        source(|src| src.names.iter().any(|name| name.starts_with(prefix)))
    }

    /// The shape the checkpoint stores its tensor `name` in.
    fn shape(#[starlark(require = pos)] name: &str) -> anyhow::Result<Vec<u64>> {
        tensor(name, |(shape, _)| shape.clone())
    }

    /// How the checkpoint stores its tensor `name`.
    fn stored(#[starlark(require = pos)] name: &str) -> anyhow::Result<EncodingValue> {
        tensor(name, |(_, encoding)| encoding.clone())?
            .map(EncodingValue)
            .map_err(|why| {
                anyhow::anyhow!("`{name}` is stored in terms no reader here can name ({why})")
            })
    }

    /// The checkpoint's tensor `name`.
    fn src(#[starlark(require = pos)] name: String) -> anyhow::Result<ExprValue> {
        Ok(ExprValue(Expr::src(name)))
    }

    /// The contract's own tensor `name`, which another read computes.
    fn out(#[starlark(require = pos)] name: String) -> anyhow::Result<ExprValue> {
        Ok(ExprValue(Expr::out(name)))
    }

    /// `parts` joined along `axis`.
    fn concat<'v>(
        #[starlark(require = pos)] axis: u32,
        #[starlark(require = pos)] parts: UnpackList<Value<'v>>,
    ) -> anyhow::Result<ExprValue> {
        let parts = parts
            .items
            .into_iter()
            .map(expr_of)
            .collect::<anyhow::Result<_>>()?;
        Ok(ExprValue(Expr::concat(self::axis(axis)?, parts)))
    }

    /// A tensor of `shape` and `encoding` holding `value` throughout.
    fn fill(
        #[starlark(require = pos)] value: UnpackFloat,
        #[starlark(require = pos)] shape: UnpackList<i64>,
        #[starlark(require = pos)] encoding: &EncodingValue,
    ) -> anyhow::Result<ExprValue> {
        Ok(ExprValue(Expr::fill(
            value.0 as f32,
            TensorType::new(shape.items, encoding.0.clone()),
        )))
    }

    /// The tensor `name` of a contract, which `expr` computes, of `encoding`
    /// and `shape` (inferred from `expr` if not given); `scaling` names the
    /// quantized weight it holds the scales of, `internal` keeps it from the
    /// artifact.
    fn tensor<'v>(
        #[starlark(require = pos)] name: String,
        #[starlark(require = pos)] expr: Value<'v>,
        #[starlark(require = pos)] encoding: &EncodingValue,
        #[starlark(require = named, default = NoneOr::None)] shape: NoneOr<UnpackList<i64>>,
        #[starlark(require = named, default = NoneOr::None)] scaling: NoneOr<Value<'v>>,
        #[starlark(require = named, default = false)] internal: bool,
    ) -> anyhow::Result<TensorValue> {
        let expr = expr_of(expr)?;
        let mut t = match shape.into_option() {
            Some(shape) => TensorContract::new(name, expr, shape.items, encoding.0.clone()),
            None => TensorContract::inferred(name, expr, encoding.0.clone()),
        };
        if let Some(w) = scaling.into_option() {
            let w = weight(w)?;
            t = t.scaling(crate::star::forward::dsl("scaling", || {
                crate::import::scaling(&w)
            })?);
        }
        if internal {
            t = t.internal();
        }
        Ok(TensorValue(t))
    }

    /// How a quantized bank `w` is stored, its blocked axis named.
    fn grouped<'v>(#[starlark(require = pos)] w: Value<'v>) -> anyhow::Result<EncodingValue> {
        let w = weight(w)?;
        crate::star::forward::dsl("grouped", || EncodingValue(crate::import::grouped(&w)))
    }

    /// The shape of `w`'s scales: its own, its blocked axis counted in
    /// blocks.
    fn scales_shape<'v>(#[starlark(require = pos)] w: Value<'v>) -> anyhow::Result<Vec<i64>> {
        let w = weight(w)?;
        crate::star::forward::dsl("scales_shape", || {
            let pairing = crate::import::scaling(&w);
            crate::import::divided(
                &crate::import::extents(&w),
                pairing.channel_axis,
                pairing.group_size,
                &w.name,
            )
        })
    }

    /// The name the scales of the weight `name` are held under.
    fn scales_name(#[starlark(require = pos)] name: &str) -> anyhow::Result<String> {
        Ok(crate::scales_name(name))
    }

    /// The name the biases of the weight `name` are held under.
    fn biases_name(#[starlark(require = pos)] name: &str) -> anyhow::Result<String> {
        Ok(crate::biases_name(name))
    }

    /// How a weight of `dtype` is stored.
    fn encoding(#[starlark(require = pos)] dtype: &DtypeValue) -> anyhow::Result<EncodingValue> {
        Ok(EncodingValue(crate::import::encoding(dtype.0)))
    }

    /// The dtype the values of `encoding` are, quantized or not.
    fn logical(#[starlark(require = pos)] encoding: &EncodingValue) -> anyhow::Result<DtypeValue> {
        Ok(DtypeValue(match &encoding.0 {
            Encoding::Raw(dtype) => *dtype,
            Encoding::Quant(spec) => spec.logical_dtype,
        }))
    }

    /// Plain values of `dtype`, one after another.
    fn raw(#[starlark(require = pos)] dtype: &DtypeValue) -> anyhow::Result<EncodingValue> {
        Ok(EncodingValue(Encoding::Raw(dtype.0)))
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
