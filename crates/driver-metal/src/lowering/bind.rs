//! What the FIRE answers, on this backend. The rest of binding is shared.
//!
//! `arm.rs` was ninety-one functions that each said, in Rust, where every
//! argument of one routine comes from. STAGES 2 through 6 of
//! `.wiki/kilimanjaro4.md` moved that same knowledge into the signatures
//! themselves, as the `sources` column every `KernelFn` derives, and gated
//! each move against the arm it was copied from. Reading that column is
//! [`kernels::bind`], and it is SHARED: the column is the same column on all
//! three shader planes -- `shader_backends_agree` holds them identical --
//! so a per-driver reading of it would be three transcriptions of one
//! decision, which is the defect class this tree has now paid for five
//! times.
//!
//! What is left here is the half that is honestly per-driver: which handle a
//! FACT names. `keys::KvKeys` is an offset into this backend's paged pool
//! and a bind-group entry on wgpu; `keys::Rows` is a number on both but
//! comes off a different struct. That is [`Held::fact`], and the slot
//! accessors around it are this backend's [`Handles`].

use kernels::bind::Holds;
use kernels::routine::Refusal;
use kernels::{Source, Ty};
use kernels_metal::routine::ArgValue;

use crate::lowering::hold::{Facts, Handles};

/// Bind one launch's arguments from the row the signature derived.
///
/// [`kernels::bind::one`] per argument, paired with this backend's
/// [`Handles`] and the launch's [`Facts`] — plus the one carrier the shared
/// reader cannot answer: a `Ty::Raised` operand becomes a driver-built view
/// through `views`.
///
/// # Errors
///
/// [`Refusal::Unstated`] when an argument has no source, or has one this
/// backend cannot answer. Otherwise whatever the statement's own absences
/// produce: [`Refusal::Absent`] for a slot or scalar the trace does not
/// carry.
pub fn bind(
    args: &[Ty],
    sources: &[Option<Source>],
    o: &mut Handles<'_>,
    f: Facts,
    views: &mut super::views::Views<'_>,
) -> Result<Vec<ArgValue>, Refusal> {
    // Argument by argument rather than the shared list reader, because ONE
    // carrier is this plane's to answer before the reader sees it: a
    // `Ty::Raised` operand is a HOST view the driver builds
    // (`super::views`), and the shared binder has no door for a value that
    // is neither a handle nor a scalar. Everything else goes through
    // `kernels::bind::one` exactly as the list reader would have sent it —
    // same order, same slot numbering, same refusals.
    let mut out = Vec::with_capacity(args.len());
    for (at, ty) in args.iter().enumerate() {
        let source = sources.get(at).copied().flatten().ok_or(Refusal::Unstated {
            what: "an argument whose signature does not say where it comes from",
        })?;
        if matches!(ty, Ty::Raised) {
            out.push(views.raise(source, o, f)?);
            continue;
        }
        out.push(kernels::bind::one::<ArgValue, _>(*ty, source, &mut Held { o, f })?);
    }
    Ok(out)
}

/// ONE value, for a body that ASKS rather than a column that declares.
///
/// The same resolver, entered at one argument instead of a list. Nothing new
/// answers — what changed is only where the question is asked from.
///
/// # Errors
///
/// [`Refusal::Unstated`] for a fact this backend does not answer, and whatever
/// the fact's own absence means otherwise.
pub fn one(
    ty: Ty,
    source: Source,
    o: &mut Handles<'_>,
    f: Facts,
) -> Result<ArgValue, Refusal> {
    kernels::bind::one::<ArgValue, _>(ty, source, &mut Held { o, f })
}

/// This backend's answers, for the shared reader.
///
/// It borrows rather than owns the [`Handles`] because binding MUTATES them
/// -- every `input` numbers a handle -- and the caller keeps them afterwards
/// to build the encoder's binding list.
struct Held<'a, 'h> {
    o: &'h mut Handles<'a>,
    f: Facts,
}

impl Holds for Held<'_, '_> {
    fn input(&mut self, n: usize) -> Result<u32, Refusal> {
        self.o.input(n)
    }

    fn output(&mut self, n: usize) -> Result<u32, Refusal> {
        self.o.output(n)
    }

    fn output_read(&mut self, n: usize) -> Result<u32, Refusal> {
        self.o.output_read(n)
    }

    fn weight(&mut self, n: usize) -> Result<u32, Refusal> {
        self.o.weight(n)
    }

    // THE RECTANGLE, WHICH THE MARK NOW CARRIES. `shaped` asks for both
    // halves of an operand -- the handle and its row width -- and the default
    // here answers `Unstated`, which `bind` reads as a width of ZERO. Every
    // body that takes an `In<Tensor<_>>` and reads `x.width` then refuses
    // `Empty`, which is what left this backend unable to dispatch a rotation
    // or a strided activation.
    fn in_width(&self, n: usize) -> Result<i32, Refusal> {
        self.o.in_width(n)
    }

    fn out_width(&self, n: usize) -> Result<i32, Refusal> {
        self.o.out_width(n)
    }


    fn param(&self, n: usize) -> Result<i32, Refusal> {
        self.o.param(n)
    }

    fn param_f32(&self, n: usize) -> Result<f32, Refusal> {
        self.o.param_f32(n)
    }

    fn null(&mut self) -> u32 {
        self.o.state(None)
    }

    fn rows(&mut self) -> i32 {
        // The launch rectangle's — what `keys::Rows` used to answer.
        self.f.rows.cast_signed()
    }
}


