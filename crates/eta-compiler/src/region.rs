//! A fused region: a function of one row, in eta ops. poem-compiler forms one
//! from a chain of elementwise and row ops and lowers it here; kernels-*
//! `synth` (and the XLA and Cerebras engines) take it.
//!
//! A region is pure math over its parameters. It never goes through the plan
//! pipeline (`TraceContainer → bind → normalize → CompiledStage`), which is
//! shaped for guest programs; it goes straight to the stream core
//! ([`Region::body`]) and to the tier-0 interpreter ([`Region::eval`]) that
//! holds a generated kernel to it.
//!
//! **One row.** Every value is a row (rank 1, the region's one width) or a
//! scalar (rank 0, one per row: a mean, an `rsqrt`). How a parameter is
//! stored — each row its own, one row every row reads (a norm weight), rows
//! picked by id — is the kernels' `In`/`Out`, not the region's: a norm
//! weight is a row parameter like any other. Whether the region reduces its
//! row is read from its ops; no reach is passed.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use eta_ir::infer::{BodyCtx, BodyError, body_types_after};
use eta_ir::op::Op;
use eta_ir::types::{Shape, ValueId, ValueType};

use crate::codegen::op_view::{OpView, result_bases};
use crate::codegen::simt::hooks;
use crate::codegen::simt::stream::{Streams, View};
use crate::eval::interp::{PassInputs, Value};
use crate::plan::{Dimension, SymbolicType};

pub use crate::codegen::simt::hooks::Uncovered;

/// A function of one row: parameters in, eta ops over them, results out.
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    /// Values `0..n` are the parameters.
    pub params: Vec<SymbolicType>,
    /// Their results are numbered from `n`, in order.
    pub ops: Vec<Op>,
    /// What leaves the kernel; `results[k]` is written through output `k`.
    pub results: Vec<ValueId>,
}

/// Why a region is not a function of one row.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum RegionError {
    /// A parameter whose extent is not stated; a region is typed for one
    /// width (`Broadcast` and `Iota` state theirs).
    Symbolic { param: u32 },
    /// A parameter of a dtype eta does not compute in.
    Dtype { param: u32 },
    /// An op that is not pure math: a channel, a call, a draw, an intrinsic.
    Impure { op: u32 },
    /// The ops do not type over the parameters.
    Untyped(BodyError),
    /// A value that is neither a scalar nor a row of the region's width.
    NotARow { value: ValueId },
    /// A result that names no value.
    NoSuchResult { result: u32, value: ValueId },
    /// Arguments that are not the parameters' (`eval`).
    Arguments { param: u32 },
    /// The interpreter faulted (`eval`).
    Fault(String),
}

impl fmt::Display for RegionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegionError::Symbolic { param } => {
                write!(
                    f,
                    "parameter {param} has an extent the region does not state"
                )
            }
            RegionError::Dtype { param } => {
                write!(f, "parameter {param} is of a dtype eta does not compute in")
            }
            RegionError::Impure { op } => {
                write!(f, "op {op} is not pure math over the parameters")
            }
            RegionError::Untyped(why) => write!(f, "the ops do not type: {why}"),
            RegionError::NotARow { value } => {
                write!(
                    f,
                    "value {value} is neither a scalar nor a row of the region's width"
                )
            }
            RegionError::NoSuchResult { result, value } => {
                write!(
                    f,
                    "result {result} names value {value}, which nothing defines"
                )
            }
            RegionError::Arguments { param } => {
                write!(f, "argument {param} is not of its parameter's type")
            }
            RegionError::Fault(why) => write!(f, "the interpreter faulted: {why}"),
        }
    }
}

impl std::error::Error for RegionError {}

/// The kernel-facing contract: what the stream core hands a kernels-* frame,
/// and all it hands one. It targets the hook ABI of kernels-cuda's
/// `RowFrame` (`kernels/elemwise/frame.cuh`, the Phase 1 pilot), whose hand
/// `ResidualAddRmsnorm` body has the same shape.
///
/// `source` defines one struct, pure computation:
///
/// ```cpp
/// struct Body {
///   template <class F> __device__ __forceinline__ void operator()(F& f) const;
/// };
/// ```
///
/// fired as `::pie::elemwise::row_frame<Body, BLOCK, PER, IN, OUT>(planes,
/// Body{}, win)`, `IN` the region's parameters and `OUT` its results, every
/// one a row plane. The struct holds no fields: the region's constants (a
/// norm's eps, its width) are literals in the source, so `hash` keys them.
/// Inside `operator()`, `F` provides:
///
/// | member | meaning |
/// |---|---|
/// | `F::N` | the elements of the row a thread holds (`PER * 8`) |
/// | `f.ld(k, e)` | element `e` of parameter `k`, as f32 |
/// | `f.st(k, e, v)` | write element `e` of result `k`; returns `v` as stored (rounded to the plane's dtype), which every later read of that value uses |
/// | `f.valid(e)` | element `e` lies inside the row; elements past it read garbage, and the body keeps them out of its reductions |
/// | `f.reduce_sum(v)`, `f.reduce_max(v)` | `v` combined over the block, the total returned to every thread |
/// | `f.template flush<K>()` | result `K` is complete; every thread calls it at the same point |
///
/// - **Indices are constant.** Every `k` and `e` the body passes is a
///   constant once its loops unroll, so the frame's arrays stay in
///   registers.
/// - **Rows only.** Parameters and results are rows of the region's width;
///   a region with a scalar parameter or result has no body.
/// - **Math.** f32, each op as the region states it, in its order (the JIT
///   compiles with `fmad=false`, so a product and a sum stay two roundings),
///   through `expf`, `logf`, `sinf`, `cosf`, `sqrtf`, `rsqrtf`, `fmaxf`,
///   `fminf` and `fmodf`; a reduction sums one element at a time, and a
///   minimum is `-reduce_max(-v)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Body {
    /// The function's C++ source.
    pub source: String,
    /// FNV-1a over `source`: the body's part of a generated kernel's key.
    pub hash: u64,
    /// The passes the body makes over the row (one, plus one per reduction
    /// a later op reads).
    pub passes: u32,
}

impl Region {
    /// The type of every value, the parameters first, or why the region is
    /// not a function of one row.
    pub fn types(&self) -> Result<Vec<SymbolicType>, RegionError> {
        Ok(self
            .value_types()?
            .iter()
            .map(|ty| SymbolicType {
                dtype: ty.dtype,
                dims: ty
                    .shape
                    .dims()
                    .iter()
                    .copied()
                    .map(Dimension::Static)
                    .collect(),
            })
            .collect())
    }

    /// The width of the region's rows; `None` where every value is a scalar.
    pub fn width(&self) -> Result<Option<u32>, RegionError> {
        Ok(self
            .value_types()?
            .iter()
            .find_map(|ty| ty.shape.dims().first().copied()))
    }

    fn value_types(&self) -> Result<Vec<ValueType>, RegionError> {
        let mut given = Vec::with_capacity(self.params.len());
        for (param, ty) in (0u32..).zip(&self.params) {
            let mut dims = Vec::with_capacity(ty.dims.len());
            for dim in &ty.dims {
                match dim {
                    Dimension::Static(n) => dims.push(*n),
                    Dimension::Symbolic(_) => return Err(RegionError::Symbolic { param }),
                }
            }
            if !eta_ir::types::supports(ty.dtype) {
                return Err(RegionError::Dtype { param });
            }
            let shape = Shape::new(&dims).ok_or(RegionError::NotARow { value: param })?;
            given.push(ValueType::new(shape, ty.dtype));
        }
        for (op, at) in self.ops.iter().zip(0u32..) {
            let pure = !op.is_effectful()
                && !matches!(
                    op,
                    Op::IntrinsicVal { .. } | Op::Rng { .. } | Op::RngKeyed { .. }
                );
            if !pure {
                return Err(RegionError::Impure { op: at });
            }
        }
        let ctx = BodyCtx {
            channel_types: &[],
            n_names: 0,
        };
        let types = body_types_after(&given, &self.ops, &ctx).map_err(RegionError::Untyped)?;
        let mut width: Option<u32> = None;
        for (value, ty) in (0u32..).zip(&types) {
            match ty.shape.dims() {
                [] => {}
                [n] if width.is_none_or(|w| w == *n) => width = Some(*n),
                _ => return Err(RegionError::NotARow { value }),
            }
        }
        for (result, &value) in (0u32..).zip(&self.results) {
            if value as usize >= types.len() {
                return Err(RegionError::NoSuchResult { result, value });
            }
        }
        Ok(types)
    }

    /// The region at one row, by the tier-0 interpreter: `args[k]` is
    /// parameter `k`'s row (or scalar), and `results[k]`'s value comes back
    /// at `k`. Every generated kernel is held to this.
    pub fn eval(&self, args: &[Value]) -> Result<Vec<Value>, RegionError> {
        use crate::eval::interp::Evaled;
        let types = self.value_types()?;
        if args.len() != self.params.len() {
            return Err(RegionError::Arguments {
                param: u32::try_from(args.len().min(self.params.len())).unwrap_or(u32::MAX),
            });
        }
        for (param, (arg, ty)) in (0u32..).zip(args.iter().zip(&types)) {
            if !crate::eval::interp::value_matches(arg, *ty) {
                return Err(RegionError::Arguments { param });
            }
        }
        let ty_of = |v: ValueId| types[v as usize];
        let inputs = PassInputs::default();
        let mut vals: Vec<Value> = args.to_vec();
        for op in &self.ops {
            match crate::eval::interp::eval_op(op, &vals, &ty_of, &inputs, 0) {
                Ok(Evaled::One(v)) => vals.push(v),
                Ok(Evaled::Two(a, b)) => {
                    vals.push(a);
                    vals.push(b);
                }
                Ok(_) => return Err(RegionError::Fault("an effect in a pure region".into())),
                Err(why) => return Err(RegionError::Fault(format!("{why}"))),
            }
        }
        Ok(self
            .results
            .iter()
            .map(|&r| vals[r as usize].clone())
            .collect())
    }

    /// The region's body in hook form, written by the stream core, or the op
    /// it does not cover. A region the core covers whole is one kernels-*
    /// `synth` may take; one it does not runs as the ops it came from.
    pub fn body(&self) -> Result<Body, Uncovered> {
        let types = self.types().map_err(|_| Uncovered {
            op: None,
            why: "the region is not a function of one row",
        })?;
        let ops: Vec<OpView> = OpView::of_all(&self.ops);
        let params = u32::try_from(self.params.len()).unwrap_or(u32::MAX);
        let bases: Vec<u32> = result_bases(&ops).iter().map(|b| b + params).collect();
        let kinds: Vec<u8> = types
            .iter()
            .map(|ty| match ty.dims.len() {
                0 => SCALAR_KIND,
                _ => ROW_KIND,
            })
            .collect();
        let unclaimed = alloc::vec![u16::MAX; ops.len()];
        let kept = alloc::vec![0u8; ops.len()];
        let no_topk = alloc::vec![None; ops.len()];
        let streams = Streams::new(
            View {
                value_types: &types,
                nodes: (0..ops.len()).collect(),
                outputs: &self.results,
            },
            &ops,
            &bases,
            &kinds,
            &unclaimed,
            &kept,
            &no_topk,
        );
        let (source, passes) = hooks::emit_body(&streams, params, &self.results)?;
        Ok(Body {
            hash: eta_ir::fnv1a64(source.as_bytes()),
            source,
            passes,
        })
    }
}

/// The stream core's row kinds (`codegen::simt::fused::stream_rows`): a
/// value as wide as the row, and one a row.
const ROW_KIND: u8 = 1;
const SCALAR_KIND: u8 = 2;
