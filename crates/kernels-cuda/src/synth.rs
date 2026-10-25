//! Generated row kernels: a fused region (kernels-ir's [`Region`]),
//! scheduled ([`kernels_ir::schedule`]) and printed as CUDA C++ ([`print`],
//! a [`Body`]), compiled with the row frame (`elemwise/frame.cuh`) into a
//! unit of its own and fired over the region's planes.
//!
//! - **What the frame takes** ([`supports`]): a region of pure math that
//!   the row core schedules (`kernels_ir::schedule`), over one or more
//!   parameters and to one or more results, every result a bf16 row and
//!   every parameter a bf16 or f32 row (its own, one every row reads, or
//!   one picked by id) or one element a row (bf16, f32 or i32), the row
//!   whole vectors of 8 and at most [`MAX_WIDTH`] wide, each head it
//!   reduces over a power-of-two run of a row's threads.
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

pub use crate::elemwise::frame::{MAX_WIDTH, Plane, lanes};
pub use kernels_ir::Region;
use kernels_ir::{Op, Shape};

/// A region's body printed in the row frame's hook form ([`print`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Body {
    /// The struct's C++ source.
    pub source: String,
    /// FNV-1a over `source`: the body's part of a generated kernel's key.
    pub hash: u64,
    /// The passes the body makes over the row.
    pub passes: u32,
    /// The floats it tables what a column alone computes in.
    pub table: u32,
    /// Whether it shares its row (a gather within the row).
    pub shares: bool,
}

impl Body {
    /// `region` scheduled and printed, or why the row core or the printer
    /// does not take it.
    pub fn of(region: &Region) -> Result<Body, Refusal> {
        let kernel = kernels_ir::schedule(region)
            .map_err(|why| refusal(format!("the row core does not schedule the region: {why}")))?;
        let source = print::body(&kernel).map_err(|print::Unprinted(dtype)| {
            refusal(format!(
                "a value rounds to {dtype:?}, which no hook rounds to"
            ))
        })?;
        Ok(Body {
            hash: kernels_ir::fnv1a64(source.as_bytes()),
            source,
            passes: kernel.passes(),
            table: kernel.table,
            shares: kernel.shares,
        })
    }
}

pub use crate::elemwise::frame::Schedule;
use crate::elemwise::frame::{self, Shared, Stored};
use crate::error::Error;
use crate::jit::{ArgValue, Ctx, Kernel, Root, refuse};

pub mod print;

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
    /// One row every row reads (a norm's weight; a head's, repeated across
    /// the row's heads, where it is one head wide).
    Shared(Dtype),
    /// Each row a row of a table, picked by id ([`Plane::gathered`]): an
    /// embedding's rows, a lane's modulation.
    Gathered(Dtype),
    /// One element every row reads (a layer's scale): a scalar parameter.
    One(Dtype),
    /// One element a row, its own (a position): a scalar parameter.
    Each(Dtype),
}

impl In {
    const fn dtype(self) -> Dtype {
        match self {
            In::Rows(d) | In::Shared(d) | In::Gathered(d) | In::One(d) | In::Each(d) => d,
        }
    }

    const fn scalar(self) -> bool {
        matches!(self, In::One(_) | In::Each(_))
    }

    /// How the frame reads it.
    fn stored(self) -> Option<Stored> {
        match (self.scalar(), self.dtype()) {
            (_, Dtype::Bf16) => Some(Stored::Bf16),
            (_, Dtype::F32) => Some(Stored::F32),
            (true, Dtype::I32) => Some(Stored::I32),
            _ => None,
        }
    }
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
    for (k, form) in ins.iter().enumerate() {
        if form.stored().is_none() {
            return Err(refusal(format!(
                "parameter {k} is stored {:?}, and the frame's rows are bf16 or f32 (a scalar's \
                 also i32)",
                form.dtype()
            )));
        }
        let scalar = region.params[k] == Shape::Scalar;
        if form.scalar() != scalar {
            return Err(refusal(format!(
                "parameter {k} is stored as {form:?}, and the region's is {}",
                if scalar { "a scalar" } else { "a row" }
            )));
        }
    }
    for (k, Out::Rows(dtype)) in outs.iter().enumerate() {
        if *dtype != Dtype::Bf16 {
            return Err(refusal(format!(
                "result {k} is stored {dtype:?}, and the frame's results are bf16"
            )));
        }
    }
    if let Err(why) = region.types() {
        return Err(refusal(format!("the region is not one row: {why}")));
    }
    frame::per(OP, region.width).map_err(|why| refusal(why.to_string()))?;
    heads(region)?;
    Body::of(region).map(|_| ())
}

/// Whether each head `region` reduces over is a power-of-two run of the
/// threads a row of its width takes (`reduce_sum_heads`): its vectors of 8
/// divide the row's lanes.
fn heads(region: &Region) -> Result<(), Refusal> {
    let lanes = frame::lanes(region.width);
    for (at, op) in region.ops.iter().enumerate() {
        let Op::ReduceHeads(_, _, head) = *op else {
            continue;
        };
        let seg = head / 8;
        if !seg.is_power_of_two() || lanes % seg != 0 {
            return Err(refusal(format!(
                "op {at} reduces heads {head} wide, which the {lanes} threads of a {}-wide row \
                 do not split evenly",
                region.width
            )));
        }
    }
    Ok(())
}

/// A generated kernel: `body` in the row frame over `ins` parameters and
/// `outs` results, rows `width` wide, packed in blocks and (where a row is
/// narrower than a block) spread a row a block, each fire taking the one
/// its rows want ([`Schedule::of`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Synthesized {
    kernel: &'static Kernel,
    spread: Option<&'static Kernel>,
    ins: usize,
    outs: usize,
    width: u32,
    /// The parameters (by bit) the frame reads one element of.
    ones: u32,
    /// How the frame reads each parameter.
    stored: [Stored; MAX_PARAMS],
    /// The shared memory its body takes beside the frame.
    shared: Shared,
}

/// The most parameters one generated kernel reads.
const MAX_PARAMS: usize = 32;

/// The kernel for `region` (one [`supports`] takes) over parameters stored
/// as `ins` and `outs` results: its body printed ([`Body::of`]), then
/// [`synth_body`]. Nothing compiles here; [`Synthesized::load`] does.
pub fn synth(region: &Region, ins: &[In], outs: usize) -> Result<Synthesized, Error> {
    let body = Body::of(region).map_err(|why| refuse(OP, why.to_string()))?;
    synth_body(&body, ins, outs, region.width)
}

/// The kernel for `body` (a body in the row frame's hook form) over
/// parameters stored as `ins` and `outs` results, rows `width` wide: the
/// same one for every call with the same body and shape. Nothing compiles
/// here; [`Synthesized::load`] does.
pub fn synth_body(body: &Body, ins: &[In], outs: usize, width: u32) -> Result<Synthesized, Error> {
    if ins.is_empty() || outs == 0 {
        return Err(refuse(OP, "a region of no parameter or no result"));
    }
    if ins.len() > MAX_PARAMS {
        return Err(refuse(OP, "a region of more than 32 parameters"));
    }
    let ones = (0u32..)
        .zip(ins)
        .filter(|(_, form)| form.scalar())
        .fold(0u32, |mask, (k, _)| mask | 1 << k);
    let mut stored = [Stored::Bf16; MAX_PARAMS];
    for (k, form) in ins.iter().enumerate() {
        stored[k] = form.stored().ok_or_else(|| {
            refuse(
                OP,
                format!(
                    "parameter {k} is stored {:?}, which the frame does not read",
                    form.dtype()
                ),
            )
        })?;
    }
    let per = frame::per(OP, width)?;
    let root = unit(body)?;
    let laid = |schedule| {
        crate::jit::generated(
            root,
            &frame::entrypoint(
                BODY,
                width,
                per,
                &stored[..ins.len()],
                outs,
                ones,
                true,
                schedule,
            ),
        )
    };
    let spread = (frame::block(width, Schedule::Spread) < frame::block(width, Schedule::Packed))
        .then(|| laid(Schedule::Spread));
    Ok(Synthesized {
        kernel: laid(Schedule::Packed),
        spread,
        ins: ins.len(),
        outs,
        width,
        ones,
        stored,
        shared: Shared {
            table: body.table,
            shares: body.shares,
        },
    })
}

/// The generated unit of `body`: one a body hash, for the life of the
/// process.
fn unit(body: &Body) -> Result<Root, Error> {
    static UNITS: OnceLock<Mutex<HashMap<u64, Root>>> = OnceLock::new();
    let text = format!(
        "// A generated row kernel: region body {:016x}.\n\
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
        if let Some(spread) = self.spread {
            spread.load()?;
        }
        self.kernel.load()
    }

    /// Fires the kernel over `rows` rows, the stage window's live ones
    /// (`ctx.stage()`) where one is armed: `ins[k]` is the region's
    /// parameter `k` and `outs[k]` its result `k`, each a 16-byte aligned
    /// plane of the dtype the kernel was made for (a scalar parameter's,
    /// [`In::One`] or [`In::Each`], its element, any alignment), no result
    /// over a parameter. Refused unless [`Synthesized::load`] loaded it on
    /// this device.
    pub fn fire(self, ctx: &Ctx, ins: &[Plane], outs: &[Plane], rows: u32) -> Result<(), Error> {
        self.fire_parts(ctx, ins, outs, rows, 1)
    }

    /// [`Synthesized::fire`] over `rows` rows of planes each `parts` of the
    /// kernel's rows wide (a row of heads, the kernel's row one head): a
    /// row's planes cut into `parts` rows of the kernel's, a scalar
    /// parameter's element (a position) read by every part of its row.
    pub fn fire_parts(
        self,
        ctx: &Ctx,
        ins: &[Plane],
        outs: &[Plane],
        rows: u32,
        parts: u32,
    ) -> Result<(), Error> {
        let schedule = Schedule::of(
            self.width,
            rows.saturating_mul(parts),
            ctx.multiprocessors().unwrap_or(1),
        );
        self.fire_laid(ctx, ins, outs, (rows, parts), schedule)
    }

    /// [`Synthesized::fire_parts`] under `schedule` (packed where the
    /// kernel has no spread form): what a bench sets each against.
    pub fn fire_laid(
        self,
        ctx: &Ctx,
        ins: &[Plane],
        outs: &[Plane],
        (rows, parts): (u32, u32),
        schedule: Schedule,
    ) -> Result<(), Error> {
        let (kernel, schedule) = match (schedule, self.spread) {
            (Schedule::Spread, Some(spread)) => (spread, Schedule::Spread),
            _ => (self.kernel, Schedule::Packed),
        };
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
        // The frame reads its parameters through the read-only path, so no
        // result may land over one.
        if outs
            .iter()
            .any(|out| ins.iter().any(|input| input.base() == out.base()))
        {
            return Err(refuse(
                OP,
                "a result lands over a parameter, and a generated kernel reads its parameters \
                 read-only",
            ));
        }
        if !kernel.is_loaded() {
            return Err(refuse(
                OP,
                format!(
                    "`{}` is not loaded on this device: a generated kernel is loaded with its \
                     model, never by a fire",
                    kernel.file
                ),
            ));
        }
        frame::fire(
            ctx,
            OP,
            kernel,
            ArgValue::of(&NO_FIELDS),
            (ins, &self.stored[..self.ins]),
            outs,
            (rows, parts),
            self.width,
            self.ones,
            self.shared,
            schedule,
        )
    }
}

#[cfg(test)]
mod tests {
    use kernels_ir::{Binary, Literal, Op, Reduce, Shape, Unary};

    use super::*;

    /// `rmsnorm(y + x) * w` and the stream, as poem lowers a residual norm.
    fn residual_norm(width: u32) -> Region {
        Region {
            width,
            params: vec![Shape::Row; 3],
            ops: vec![
                Op::Binary(Binary::Add, 1, 0),
                Op::Binary(Binary::Mul, 3, 3),
                Op::Reduce(Reduce::Sum, 4),
                Op::Const(Literal::F32(width as f32)),
                Op::Binary(Binary::Div, 5, 6),
                Op::Const(Literal::F32(1e-6)),
                Op::Binary(Binary::Add, 7, 8),
                Op::Unary(Unary::Rsqrt, 9),
                Op::Binary(Binary::Mul, 3, 10),
                Op::Binary(Binary::Mul, 11, 2),
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
    fn a_region_the_row_core_does_not_schedule_is_refused() {
        let mut region = residual_norm(64);
        region.results = vec![9];
        let why = refused(&region, &INS, &OUTS[..1]);
        assert!(why.contains("does not schedule"), "{why}");
    }

    #[test]
    fn planes_that_are_not_the_regions_are_refused() {
        let why = refused(&residual_norm(64), &INS[..2], &OUTS);
        assert!(why.contains("3 parameter(s) and 2 result(s)"), "{why}");
    }
}
