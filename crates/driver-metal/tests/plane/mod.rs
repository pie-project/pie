//! Firing ONE entry point by hand, and sabotaging the shader so the
//! comparison has to prove it can go red.
//!
//! # Why this module exists
//!
//! `device_attention` and `device_gdn` each build their `Dispatch`es inline,
//! and each carries its own `alloc_bf16`, `read_f32`, `kernels_dir` and
//! `to_bf16` at the tail. Two copies of forty lines is a style; the eight
//! files that arrived with this module would have made ten, and the ten
//! would have had to be kept agreeing about a `ParamSlot` layout that is a
//! fact about `driver-metal` rather than about any one shader. So the
//! staging is here, once, and the two older files are left as they are:
//! rewriting a test that has already measured a device is how a measurement
//! gets lost.
//!
//! # What a caller states, and what it deliberately does not
//!
//! [`fire`] takes the argument list IN THE CLAIM BODY'S ORDER, because that
//! order IS the argument-table numbering -- `baker::encode::lay_out` walks
//! the list once and an argument's position is its slot. A buffer binds its
//! address there and a scalar binds the address of its own four bytes inside
//! the dispatch's staged run, which is what [`Arg`] is for: one enum, four
//! variants, and no caller anywhere has to know that the second kind goes
//! through `Params::stage`.
//!
//! Nothing here computes a grid. Every caller states the grid its POINT
//! states -- `kernels_metal::gemm::vector_grid`, `moe::route_rows`,
//! `ssm::recurrence_grid` -- because a harness that derived one would be a
//! second implementation of the thing under test, and the two would agree
//! with each other rather than with the shader.
//!
//! # The mutation, which is the part that makes any of this evidence
//!
//! A comparison that cannot fail is worth nothing, and this tree has found
//! three of them in a week. [`mutant`] answers that: it reads a shader the
//! way the driver reads it (`layout::shader::read_source`, includes spliced),
//! edits the resolved text, and writes it into a scratch tree that a second
//! [`Pipelines`] compiles from. The kernel under test is then the tree's own
//! kernel with one term wrong, and a check that still passes against it is a
//! check that was never reading the device.
//!
//! `read_source` resolves every quoted `#include` before the edit, so the
//! scratch tree needs no headers beside the file -- and an edit whose target
//! text lives in an included header is reached just the same.
//!
//! # The archive is off
//!
//! `Archives::new(None)`, not `Archives::discover()`. A mutated source keys
//! differently from the real one so a shared cache would be correct here,
//! but "would be correct" is the sentence this crate does not accept about a
//! thing it can simply not do: nothing in a test run should be able to serve
//! a pipeline the run did not compile, and nothing in a test run should write
//! to the cache the developer's own serving uses.

#![allow(
    dead_code,
    reason = "one harness, eight test binaries, and each uses the subset its \
              own point needs -- an allocator for an element no file in this \
              sweep binds is still the allocator the next one will"
)]

pub mod vectors;

use std::path::{Path, PathBuf};

use driver_metal::baker::dispatch::{Dispatch, ParamSlot, Touches};
use driver_metal::baker::{BoundRegion as BoundArg, Slice};
use driver_metal::bind::encode::{Params, Pipelines, encode};
use driver_metal::device::{Allocation, Archives, ArgumentTable, Context, Stepper};
use driver_metal::layout::region::Region as _;
use driver_metal::program::Compiler;

/// One operand, in the claim body's order.
///
/// The three scalar variants are three variants and not one `u32`, because
/// the bits a kernel reads at `const constant float&` are not the bits it
/// reads at `const constant int&` and a test that spelled a float as its
/// integer value would bind a denormal. `f32::to_bits` is the one conversion,
/// and it happens here rather than at every call site.
#[derive(Clone, Copy)]
pub enum Arg<'a> {
    /// A device buffer, bound at its address.
    Buf(&'a Allocation),
    /// `const constant uint&`.
    U32(u32),
    /// `const constant int&`.
    I32(i32),
    /// `const constant float&`.
    F32(f32),
}

impl Arg<'_> {
    /// The four bytes a scalar binds, or `None` for a buffer.
    fn word(self) -> Option<u32> {
        match self {
            Self::Buf(_) => None,
            Self::U32(v) => Some(v),
            Self::I32(v) => Some(v as u32),
            Self::F32(v) => Some(v.to_bits()),
        }
    }
}

/// A device and a shader compiler, or nothing on a machine without one.
pub struct Rig {
    pub context: Context,
    pub compiler: Compiler,
}

impl Rig {
    /// Open the device, or answer `None` where there is not one.
    ///
    /// The caller reports the skip through `driver_metal::skip::skipped`, so
    /// that `PIE_METAL_NO_SKIP` turns a run with no device into a failure
    /// rather than into thirty silent passes.
    pub fn open() -> Option<Self> {
        let context = Context::new().ok()?;
        let compiler = Compiler::with_archives(&context, Archives::new(None))
            .expect("a device that exists has a compiler");
        Some(Self { context, compiler })
    }
}

/// Where the shader tree is.
pub fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .join("kernels-metal/kernels")
}

/// Compile `symbol` out of `root`'s copy of `file` and run one dispatch.
///
/// `root` is the shader tree to compile from: [`kernels_dir`] for the honest
/// run and a [`mutant`]'s path for the run that has to fail.
pub fn fire(
    rig: &Rig,
    root: &Path,
    file: &'static str,
    symbol: &'static str,
    grid: [u32; 3],
    threadgroup: [u32; 3],
    args: &[Arg<'_>],
) {
    // Every slot needs SOME address before `encode_one` walks the list, and
    // a scalar's is overwritten a few lines later by the staged run's. The
    // first buffer is the filler because a zero address is not a thing this
    // driver's argument table accepts.
    let filler = args
        .iter()
        .find_map(|a| match a {
            Arg::Buf(b) => Some(b.gpu_address()),
            _ => None,
        })
        .expect("a dispatch binds at least one buffer");

    let mut bound = Vec::with_capacity(args.len());
    let mut params = Vec::new();
    let mut param_slots = Vec::new();
    for (slot, arg) in args.iter().enumerate() {
        let address = match arg {
            Arg::Buf(b) => b.gpu_address(),
            _ => filler,
        };
        bound.push(BoundArg {
            slice: Slice {
                address,
                bytes: match arg {
                    Arg::Buf(b) => b.len(),
                    _ => 0,
                },
            },
            width: 0,
        });
        if let Some(word) = arg.word() {
            param_slots.push(ParamSlot {
                slot,
                at: (params.len() as u32) * 4,
                bytes: 4,
                value: params.len() as u8,
            });
            params.push(word);
        }
    }

    let dispatch = Dispatch {
        symbol,
        file,
        stamp: "",
        grid,
        threadgroup,
        touches: Touches::everything(&bound),
        args: bound,
        params,
        param_slots,
        layers: 0..1,
        op: 0,
    };

    let mut pipelines = Pipelines::new(root);
    pipelines
        .ensure(&rig.context, &rig.compiler, std::slice::from_ref(&dispatch))
        .unwrap_or_else(|why| panic!("`{symbol}` builds a pipeline out of {root:?}: {why}"));
    let staged =
        Params::stage(&rig.context, std::slice::from_ref(&dispatch)).expect("the scalars stage");
    let table = ArgumentTable::new(&rig.context, args.len().max(1))
        .expect("a table as wide as the argument list");
    let mut stepper = Stepper::new(&rig.context).expect("a stepper");
    stepper
        .run(|encoder| {
            encode(
                encoder,
                &table,
                &pipelines,
                &staged,
                std::slice::from_ref(&dispatch),
            )
        })
        .unwrap_or_else(|why| panic!("`{symbol}` fires: {why}"));
}

/// The tree's own `file`, with `from` replaced by `to`, in a scratch root.
///
/// # Panics
///
/// If `from` is not in the resolved source. That is the whole point of the
/// assertion: a mutation whose target text has been edited away silently
/// stops mutating anything, and a tripwire that has quietly become a copy of
/// the kernel is exactly the defect these mutations exist to catch.
pub fn mutant(file: &str, from: &str, to: &str) -> tempfile::TempDir {
    let source = driver_metal::layout::shader::read_source(kernels_dir().join(file))
        .unwrap_or_else(|why| panic!("`{file}` splices: {why}"));
    let hits = source.matches(from).count();
    assert!(
        hits > 0,
        "`{file}` no longer contains `{from}`, so this mutation would compile \
         the kernel unchanged and the check below would pass for the wrong \
         reason"
    );
    let root = tempfile::tempdir().expect("a scratch tree");
    let at = root.path().join(file);
    std::fs::create_dir_all(at.parent().expect("a shader sits in a directory"))
        .expect("the scratch tree's directories");
    std::fs::write(&at, source.replace(from, to)).expect("the mutated source");
    root
}

/// The widest relative disagreement, against a floor that keeps a near-zero
/// reference from turning a rounding into an infinity.
pub fn worst(got: &[f32], want: &[f32], floor: f32) -> f32 {
    assert_eq!(
        got.len(),
        want.len(),
        "the two slabs are the same rectangle"
    );
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs() / w.abs().max(floor))
        .fold(0.0, f32::max)
}

/// Where the worst disagreement is, for a message that names an element
/// rather than a number.
pub fn worst_at(got: &[f32], want: &[f32], floor: f32) -> usize {
    let mut at = 0;
    let mut seen = f32::NEG_INFINITY;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let e = (g - w).abs() / w.abs().max(floor);
        if e > seen {
            seen = e;
            at = i;
        }
    }
    at
}

/// How many representable bf16 values lie between two of them.
///
/// # Why an integer distance and not a relative bound
///
/// A kernel whose result lands in bf16 rounds ONCE, at the store, out of a
/// float computation. So the only thing a reference written in Rust can
/// disagree with it about is the last few bits of that float -- Metal's
/// `exp` is not `f32::exp` and its `fast::exp` is neither -- and after
/// rounding to eight mantissa bits those disagreements vanish except where
/// the exact answer sits within a transcendental's error of a rounding
/// boundary. That happens, and it happens at a rate a fixture of a few
/// hundred elements sees a handful of times.
///
/// The honest bound for that is "one bf16 step", and it is worth saying as
/// an integer rather than as `2^-8` of something: a relative bound of one
/// ulp is a number a later hand can double without the sentence reading
/// differently, and `<= 1` is not. It is also the tightest bound there is --
/// a defect that moves an answer by less than one bf16 step did not move the
/// answer, because the answer is a bf16.
///
/// The mapping is monotone through zero, so `+0` and `-0` are one value
/// apart from nothing and a sign flip is the whole scale of the exponent
/// range rather than a small number.
pub fn bf16_ulps(a: f32, b: f32) -> i32 {
    let ordered = |x: f32| -> i32 {
        let w = to_bf16(x);
        let magnitude = i32::from(w & 0x7fff);
        if w & 0x8000 != 0 {
            -magnitude
        } else {
            magnitude
        }
    };
    (ordered(a) - ordered(b)).abs()
}

/// The widest [`bf16_ulps`] over two slabs, where it is, and how many
/// elements are not exact.
pub fn ulp_spread(got: &[f32], want: &[f32]) -> (i32, usize, usize) {
    assert_eq!(
        got.len(),
        want.len(),
        "the two slabs are the same rectangle"
    );
    let mut widest = 0;
    let mut at = 0;
    let mut inexact = 0;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = bf16_ulps(*g, *w);
        if d > widest {
            widest = d;
            at = i;
        }
        if d != 0 {
            inexact += 1;
        }
    }
    (widest, at, inexact)
}

/// `exp` and `tanh` CORRECTLY ROUNDED to f32: the real answer, rounded once.
///
/// A model that spells these with Rust's own `f32::exp` is comparing one
/// platform's libm against another's, and the difference between two
/// implementations that are each within an ulp is two ulps of noise the
/// comparison then has to be widened to accept. Taking the double's answer
/// and rounding it once removes that side of the question entirely: what is
/// left over the comparison is whether METAL's function is the function,
/// which is the only half a Metal test can answer and the only half worth
/// asking.
///
/// It matters more than an ulp where an expression cancels. `1 + tanh(x)` at
/// `x = -8` is three ulps of one, so a one-ulp disagreement about `tanh`
/// there is a third of the answer -- see `device_packed`'s GELU, which is
/// where this pair of functions came from.
pub fn exp32(x: f32) -> f32 {
    f64::from(x).exp() as f32
}

pub fn tanh32(x: f32) -> f32 {
    f64::from(x).tanh() as f32
}

/// The bound is measuring THIS device, rather than being wide enough to
/// accept anything.
///
/// `device_gdn` states the reasoning and paid for it: a bound reasoned from
/// bf16's half-ulp rather than measured let a one-percent error in the
/// central gate of a kernel pass. Above 1 the assertion did not hold; below
/// an eighth the bound is more than eight times what the device delivers,
/// which is a bound a broken kernel fits inside. Exact agreement is let
/// through, because the absence of error is not a loose bound.
pub fn tolerance_holds(worst: f32, bound: f32, what: &str) {
    if worst == 0.0 {
        return;
    }
    let ratio = worst / bound;
    assert!(
        (0.125..=1.0).contains(&ratio),
        "{what}: the widest comparison took {ratio} of its bound ({worst} \
         against {bound}). Under 1/8 the bound is not measuring this device; \
         over 1 the assertion did not hold"
    );
}

/// Say what the device actually delivered.
///
/// A test that only asserts leaves its numbers in the assertion it did not
/// trip, and "the bound held" is not a measurement. Everything this sweep
/// learned about the device is printed through here, so that
/// `cargo test -- --ignored --nocapture` IS the report.
#[allow(
    clippy::print_stderr,
    reason = "the number the device produced is the output of these tests"
)]
pub fn measured(what: &str, line: &str) {
    eprintln!("MEASURED {what}: {line}");
}

/// A value generator whose period shares no factor with any stride a fixture
/// here uses, so that a mis-indexed read lands on a DIFFERENT number rather
/// than on the one it should have had.
///
/// Seventeen is prime and divides none of the widths below; the values stay
/// inside `[-1, 1)`, which is where a checkpoint's activations are and where
/// an exponential is neither saturated nor flat.
pub fn spread(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = ((i * 7 + seed * 13) % 17) as f32;
            (t - 8.0) / 8.5
        })
        .collect()
}

/// [`spread`] scaled, for a plane whose useful range is not `[-1, 1)`.
pub fn spread_by(n: usize, seed: usize, gain: f32) -> Vec<f32> {
    spread(n, seed).into_iter().map(|v| v * gain).collect()
}

pub fn to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let round = ((bits >> 16) & 1) + 0x7fff;
    ((bits.wrapping_add(round)) >> 16) as u16
}

pub fn from_bf16(x: u16) -> f32 {
    f32::from_bits(u32::from(x) << 16)
}

/// `x` as this plane's element sees it: what a bf16 buffer hands back.
pub fn narrowed(x: f32) -> f32 {
    from_bf16(to_bf16(x))
}

pub fn alloc_bf16(context: &Context, values: &[f32], what: &'static str) -> Allocation {
    let narrow: Vec<u16> = values.iter().copied().map(to_bf16).collect();
    alloc_bytes(context, cast(&narrow), what)
}

pub fn alloc_f32(context: &Context, values: &[f32], what: &'static str) -> Allocation {
    alloc_bytes(context, cast(values), what)
}

pub fn alloc_i32(context: &Context, values: &[i32], what: &'static str) -> Allocation {
    alloc_bytes(context, cast(values), what)
}

pub fn alloc_u32(context: &Context, values: &[u32], what: &'static str) -> Allocation {
    alloc_bytes(context, cast(values), what)
}

fn alloc_bytes(context: &Context, bytes: &[u8], what: &'static str) -> Allocation {
    let len = bytes.len() as u64;
    let a = Allocation::new(context, len.max(4), what).expect("an allocation");
    unsafe {
        a.write(0, bytes).expect("the bytes fit");
    }
    a
}

pub fn read_bf16(a: &Allocation, n: usize) -> Vec<f32> {
    let words = unsafe { core::slice::from_raw_parts(a.contents().as_ptr().cast::<u16>(), n) };
    words.iter().copied().map(from_bf16).collect()
}

pub fn read_f32(a: &Allocation, n: usize) -> Vec<f32> {
    unsafe { core::slice::from_raw_parts(a.contents().as_ptr().cast::<f32>(), n) }.to_vec()
}

pub fn read_i32(a: &Allocation, n: usize) -> Vec<i32> {
    unsafe { core::slice::from_raw_parts(a.contents().as_ptr().cast::<i32>(), n) }.to_vec()
}

fn cast<T>(v: &[T]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}
