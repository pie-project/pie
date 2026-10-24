//! Generated row kernels: a fused region's body, as eta's stream core writes
//! it ([`Body`]), compiled with the row frame (`elemwise/frame.cuh`) into a
//! unit of its own and fired over the region's planes.
//!
//! - **What the frame takes** ([`supports`]): a region of pure math that
//!   the stream core writes whole, over one or more parameters and to one
//!   or more results, every one a bf16 row (a parameter may be one row
//!   every row reads), the row whole vectors of 8 and at most
//!   [`MAX_WIDTH`] wide.
//! - **Identity.** One generated unit a body, `synth/<hash>.cu`: the
//!   frame's header and the body's source. One kernel a shape of it: the
//!   vectors a thread holds (from the width) and the plane counts. The disk
//!   cache keys the unit's text, the headers and the instantiation, as it
//!   does a carried unit's, so a warm start compiles nothing.
//! - **Loading.** [`synth`] names the kernel; [`Synthesized::load`] compiles
//!   and loads it, and a fire of one not loaded on its device is refused: a
//!   module load mid-fire waits on whatever the device runs, an
//!   expert-parallel peer's barrier included, so every generated kernel is
//!   loaded with the model, before a body is armed or captured.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use dtype::Dtype;
use eta_ir::op::Op;

pub use crate::elemwise::frame::{MAX_WIDTH, Plane};
pub use eta_compiler::region::{Body, Region};

use crate::elemwise::frame;
use crate::error::Error;
use crate::jit::{ArgValue, Ctx, Kernel, Root, refuse};

const OP: &str = "fused.region";

/// The struct every generated body defines (`Body`'s contract), in the
/// namespace its unit puts it in.
const BODY: &str = "::pie::synth::Body";

/// A generated body holds no fields; C++ gives the empty struct one byte.
const NO_FIELDS: u8 = 0;

/// How a region's parameter is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum In {
    /// Each row its own.
    Rows(Dtype),
    /// One row every row reads (a norm's weight).
    Shared(Dtype),
}

/// How a region's result is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Out {
    /// Each row its own.
    Rows(Dtype),
}

/// Why the frame does not take a region: the condition it breaks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal(String);

impl core::fmt::Display for Refusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Refusal {}

fn refusal(why: impl Into<String>) -> Refusal {
    Refusal(why.into())
}

/// Whether the row frame takes `region`, its parameters stored as `ins`
/// and its results as `outs`; the condition it breaks where it does not.
pub fn supports(region: &Region, ins: &[In], outs: &[Out]) -> Result<(), Refusal> {
    if ins.len() != region.params.len() || outs.len() != region.results.len() {
        return Err(refusal(format!(
            "the region has {} parameter(s) and {} result(s), and {} and {} planes were given",
            region.params.len(),
            region.results.len(),
            ins.len(),
            outs.len()
        )));
    }
    if ins.is_empty() || outs.is_empty() {
        return Err(refusal(
            "a region of no parameter or no result: the frame reads and writes planes",
        ));
    }
    let dtypes = ins
        .iter()
        .map(|form| match form {
            In::Rows(dtype) | In::Shared(dtype) => ("parameter", *dtype),
        })
        .chain(outs.iter().map(|Out::Rows(dtype)| ("result", *dtype)));
    for (k, (what, dtype)) in dtypes.enumerate() {
        if dtype != Dtype::Bf16 {
            let k = if k < ins.len() { k } else { k - ins.len() };
            return Err(refusal(format!(
                "{what} {k} is stored {dtype:?}, and the frame's planes are bf16"
            )));
        }
    }
    for (at, op) in region.ops.iter().enumerate() {
        if let Some(what) = refused(op) {
            return Err(refusal(format!("op {at} is {what}")));
        }
    }
    let width = match region.width() {
        Ok(Some(width)) => width,
        Ok(None) => return Err(refusal("every value is a scalar: the region has no row")),
        Err(why) => return Err(refusal(format!("the region is not one row: {why}"))),
    };
    frame::per(OP, width).map_err(|why| refusal(why.to_string()))?;
    region
        .body()
        .map(|_| ())
        .map_err(|uncovered| match uncovered.op {
            Some(at) => refusal(format!(
                "the stream core does not write op {at}: {}",
                uncovered.why
            )),
            None => refusal(format!("the stream core writes no body: {}", uncovered.why)),
        })
}

/// What `op` is, where no generated kernel computes it.
fn refused(op: &Op) -> Option<&'static str> {
    Some(match op {
        Op::ChanTake(..) | Op::ChanRead(..) | Op::ChanPut { .. } => "a channel op",
        Op::Rng { .. } | Op::RngKeyed { .. } => "a draw",
        Op::IntrinsicVal { .. } => "an intrinsic",
        Op::MatMul(..) => "a matmul",
        Op::TopK { .. } => "a top-k",
        Op::SortDesc(..) => "a sort",
        Op::KernelCall { .. } => "a kernel call",
        Op::SinkCall { .. } => "a sink call",
        _ => return None,
    })
}

/// A generated kernel: `body` in the row frame over `ins` parameters and
/// `outs` results, rows `width` wide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Synthesized {
    kernel: &'static Kernel,
    ins: usize,
    outs: usize,
    width: u32,
}

/// The kernel for `body` (the body of a region [`supports`] takes) over
/// `ins` parameters and `outs` results, rows `width` wide: the same one for
/// every call with the same body and shape. Nothing compiles here;
/// [`Synthesized::load`] does.
pub fn synth(body: &Body, ins: usize, outs: usize, width: u32) -> Result<Synthesized, Error> {
    if ins == 0 || outs == 0 {
        return Err(refuse(OP, "a region of no parameter or no result"));
    }
    let per = frame::per(OP, width)?;
    let root = unit(body)?;
    Ok(Synthesized {
        kernel: crate::jit::generated(root, &frame::entrypoint(BODY, per, ins, outs)),
        ins,
        outs,
        width,
    })
}

/// The generated unit of `body`: one a body hash, for the life of the
/// process.
fn unit(body: &Body) -> Result<Root, Error> {
    static UNITS: OnceLock<Mutex<HashMap<u64, Root>>> = OnceLock::new();
    let text = format!(
        "// A generated row kernel: eta region body {:016x}.\n\
         #include \"elemwise/frame.cuh\"\n\n\
         namespace pie::synth {{\n\n{}\n}}\n",
        body.hash, body.source
    );
    let mut units = UNITS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("generated unit table lock poisoned");
    if let Some(root) = units.get(&body.hash) {
        return if root.text == text {
            Ok(*root)
        } else {
            Err(refuse(
                OP,
                format!(
                    "two bodies of different source share the hash {:016x}",
                    body.hash
                ),
            ))
        };
    }
    let file: &'static str = Box::leak(format!("synth/{:016x}.cu", body.hash).into_boxed_str());
    let root = Root::generated(file, Box::leak(text.into_boxed_str()));
    units.insert(body.hash, root);
    Ok(root)
}

impl Synthesized {
    #[must_use]
    pub const fn kernel(self) -> &'static Kernel {
        self.kernel
    }

    /// Compiles the kernel (or reads it from the disk cache) and loads it on
    /// the current device: what a fire of it needs done first.
    pub fn load(self) -> Result<(), Error> {
        self.kernel.load()
    }

    /// Fires the kernel over `rows` rows, the stage window's live ones
    /// (`ctx.stage()`) where one is armed: `ins[k]` is the region's
    /// parameter `k` and `outs[k]` its result `k`, each a 16-byte aligned
    /// bf16 plane. Refused unless [`Synthesized::load`] loaded it on this
    /// device.
    pub fn fire(self, ctx: &Ctx, ins: &[Plane], outs: &[Plane], rows: u32) -> Result<(), Error> {
        if ins.len() != self.ins || outs.len() != self.outs {
            return Err(refuse(
                OP,
                format!(
                    "the kernel takes {} parameter and {} result planes and the fire passes \
                     {} and {}",
                    self.ins,
                    self.outs,
                    ins.len(),
                    outs.len()
                ),
            ));
        }
        if !self.kernel.is_loaded() {
            return Err(refuse(
                OP,
                format!(
                    "`{}` is not loaded on this device: a generated kernel is loaded with its \
                     model, never by a fire",
                    self.kernel.file
                ),
            ));
        }
        frame::fire(
            ctx,
            OP,
            self.kernel,
            ArgValue::of(&NO_FIELDS),
            ins,
            outs,
            rows,
            self.width,
        )
    }
}

#[cfg(test)]
mod tests {
    use eta_compiler::plan::{Dimension, SymbolicType};
    use eta_ir::types::Literal;

    use super::*;

    /// `rmsnorm(y + x) * w` and the stream, as poem lowers a residual norm.
    fn residual_norm(width: u32) -> Region {
        let row = SymbolicType {
            dtype: eta_ir::Dtype::F32,
            dims: vec![Dimension::Static(width)],
        };
        Region {
            params: vec![row; 3],
            ops: vec![
                Op::Add(1, 0),
                Op::Mul(3, 3),
                Op::ReduceSum(4),
                Op::Const(Literal::F32(width as f32)),
                Op::Div(5, 6),
                Op::Const(Literal::F32(1e-6)),
                Op::Add(7, 8),
                Op::Rsqrt(9),
                Op::Mul(3, 10),
                Op::Mul(11, 2),
            ],
            results: vec![3, 12],
            rounds: vec![],
        }
    }

    const INS: [In; 3] = [
        In::Rows(Dtype::Bf16),
        In::Rows(Dtype::Bf16),
        In::Shared(Dtype::Bf16),
    ];
    const OUTS: [Out; 2] = [Out::Rows(Dtype::Bf16), Out::Rows(Dtype::Bf16)];

    fn refused(region: &Region, ins: &[In], outs: &[Out]) -> String {
        supports(region, ins, outs)
            .expect_err("the frame refuses it")
            .to_string()
    }

    #[test]
    fn the_frame_takes_a_bf16_residual_norm_up_to_its_widest_row() {
        for width in [8, 1544, 4096, MAX_WIDTH] {
            assert_eq!(
                supports(&residual_norm(width), &INS, &OUTS),
                Ok(()),
                "{width}"
            );
        }
    }

    #[test]
    fn a_plane_of_another_dtype_is_refused_by_name() {
        let ins = [In::Rows(Dtype::Bf16), In::Rows(Dtype::F32), INS[2]];
        let why = refused(&residual_norm(4096), &ins, &OUTS);
        assert!(why.contains("parameter 1 is stored F32"), "{why}");
        let outs = [OUTS[0], Out::Rows(Dtype::F16)];
        let why = refused(&residual_norm(4096), &INS, &outs);
        assert!(why.contains("result 1 is stored F16"), "{why}");
    }

    #[test]
    fn a_row_the_frame_does_not_hold_is_refused() {
        let why = refused(&residual_norm(MAX_WIDTH + 8), &INS, &OUTS);
        assert!(why.contains("wider than the frame holds"), "{why}");
        let why = refused(&residual_norm(12), &INS, &OUTS);
        assert!(why.contains("not whole vectors of 8"), "{why}");
    }

    #[test]
    fn an_op_no_generated_kernel_computes_is_refused() {
        let mut region = residual_norm(64);
        region.ops.push(Op::MatMul(12, 2));
        let why = refused(&region, &INS, &OUTS);
        assert!(why.contains("op 10 is a matmul"), "{why}");
    }

    #[test]
    fn planes_that_are_not_the_regions_are_refused() {
        let why = refused(&residual_norm(64), &INS[..2], &OUTS);
        assert!(why.contains("3 parameter(s) and 2 result(s)"), "{why}");
    }
}
