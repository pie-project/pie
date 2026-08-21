//! Binding a launch from the SIGNATURE, for any backend.
//!
//! A routine's `sources` column says where each argument comes from, derived
//! from the typed signature rather than written down. Every driver reads the
//! SAME column, so reading it per-driver is three transcriptions of one
//! decision.
//!
//! Shared: everything the column can say on its own -- slots, chains,
//! arithmetic, literals, and the carrier each lands at. That is [`bind`].
//!
//! Per-driver: what a FACT answers to. `keys::KvKeys` is a handle into metal's
//! paged pool and a bind-group entry on wgpu. A driver supplies that through
//! [`Holds::fact`] and its slots through the rest of the trait.
//!
//! The trait returns HANDLES rather than finished values because the carrier
//! is the signature's business -- `InSlot<0, I32s>` and `InSlot<0, Buf>` are
//! one handle at two spellings -- and a driver choosing it would be reading
//! the column a second time.

use crate::routine::Refusal;
use crate::shader::ShaderValue;
use crate::{Kind, Lit, Source, Ty};

/// What a driver answers a [`Source::Named`] with. Which kind a key is, is the
/// driver's knowledge; what carrier it lands at is [`bind`]'s.
///
/// `Eq` is deliberately absent: [`Answer::Float`] carries a real number, and
/// an epsilon compared for exact equality answers according to how it was
/// computed rather than to what it is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Answer {
    /// A staged buffer, by the handle the driver numbered it with.
    Handle(u32),
    /// One of the fire's numbers.
    Number(i32),
    /// A number already unsigned and possibly wider than an `i32`, which the
    /// pool's strides are on a large deployment.
    Wide(u64),
    /// One of the fire's REAL numbers -- an epsilon, a rotary base, a
    /// routing scale.
    ///
    /// Its own variant rather than a number, because the bits of an `f32`
    /// read as an `i32` are a different value and nothing downstream could
    /// tell which reading was meant.
    Float(f32),
    /// One of the fire's flags -- a layout, a normalisation switch.
    ///
    /// A flag is not a one-bit number: the three shader planes pack it into
    /// a scalar and CUDA passes a one-byte `bool` by value, and only the
    /// backend's own [`ShaderValue::bool`] knows which.
    Truth(bool),
}

/// The driver's side of a binding: the handles a statement carries and the
/// facts the fire answers.
pub trait Holds {
    /// The statement's `n`th input.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] when the statement carries no such operand.
    fn input(&mut self, n: usize) -> Result<u32, Refusal>;

    /// The statement's `n`th result, bound for WRITING.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] when the statement carries no such operand.
    fn output(&mut self, n: usize) -> Result<u32, Refusal>;

    /// The statement's `n`th result, bound for READING.
    ///
    /// Not cosmetic on a backend that tracks hazards: binding a read as a
    /// write serialises launches that could have run together. Defaults to
    /// [`Self::output`].
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] when the statement carries no such operand.
    fn output_read(&mut self, n: usize) -> Result<u32, Refusal> {
        self.output(n)
    }

    /// The statement's `n`th weight.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] when the statement carries no such operand.
    fn weight(&mut self, n: usize) -> Result<u32, Refusal>;


    /// The statement's `n`th scalar.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] when the statement carries no such scalar.
    fn param(&self, n: usize) -> Result<i32, Refusal>;

    /// The statement's `n`th float scalar.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] when the statement carries no such scalar.
    fn param_f32(&self, n: usize) -> Result<f32, Refusal>;

    /// A handle standing for an operand the statement does not carry. It still
    /// takes one: an argument list is positional, so an absence binding
    /// NOTHING would shift every argument after it.
    fn null(&mut self) -> u32;

    /// Elements per row of the statement's `n`th INPUT.
    ///
    /// A width is a fact ABOUT an operand and not an operand, so asking mints
    /// no handle and occupies no argument position -- hence `&self` where the
    /// slot accessors take `&mut self`.
    ///
    /// Defaults to a refusal: a shader statement's rectangle is a uniform the
    /// kernel reads, not a property the binder can be asked for.
    ///
    /// # Errors
    ///
    /// [`Refusal::Unstated`] on a driver that does not carry row widths;
    /// [`Refusal::Absent`] for an input the statement does not carry or
    /// states no width for.
    fn in_width(&self, n: usize) -> Result<i32, Refusal> {
        let _ = n;
        Err(Refusal::Unstated { what: "an input's row width, which this backend does not carry" })
    }

    /// Elements per row of the statement's `n`th RESULT. See
    /// [`Self::in_width`].
    ///
    /// # Errors
    ///
    /// As [`Self::in_width`].
    fn out_width(&self, n: usize) -> Result<i32, Refusal> {
        let _ = n;
        Err(Refusal::Unstated { what: "a result's row width, which this backend does not carry" })
    }

    /// How many elements the statement's `n`th RESULT holds in total.
    ///
    /// Not `rows * out_width` computed here: the result is the only thing that
    /// knows its own size, so asking IT cannot disagree with anything.
    ///
    /// # Errors
    ///
    /// As [`Self::in_width`].
    fn out_elements(&self, n: usize) -> Result<i32, Refusal> {
        let _ = n;
        Err(Refusal::Unstated { what: "a result's element count, which this backend does not carry" })
    }

    /// What the fire answers for a fact. `None` for a key this driver does not
    /// answer, which [`bind`] turns into [`Refusal::Unstated`].
    ///
    /// # Errors
    ///
    /// Whatever the fact's own absence means -- a tile on a device that
    /// states none, a page size on a pool that has none.
    fn fact(&mut self, key: &'static str) -> Option<Result<Answer, Refusal>>;
}

/// Bind one launch's arguments from the row the signature derived.
///
/// `args` is the routine's [`Ty`] column and `sources` is its
/// `Source` column; they are the same length and the same order as the
/// dispatch the kernel expects.
///
/// # Errors
///
/// [`Refusal::Unstated`] when an argument has no source, or has one this
/// backend cannot answer. Otherwise whatever the statement's own absences
/// produce: [`Refusal::Absent`] for a slot or scalar the trace does not
/// carry.
pub fn bind<V: ShaderValue, H: Holds + ?Sized>(
    args: &[Ty],
    sources: &[Option<Source>],
    h: &mut H,
) -> Result<Vec<V>, Refusal> {
    let mut out = Vec::with_capacity(args.len());
    for (at, ty) in args.iter().enumerate() {
        let source = sources.get(at).copied().flatten().ok_or(Refusal::Unstated {
            what: "an argument whose signature does not say where it comes from",
        })?;
        out.push(one(*ty, source, h)?);
    }
    Ok(out)
}

/// One argument. See [`bind`].
pub fn one<V: ShaderValue, H: Holds + ?Sized>(
    ty: Ty,
    source: Source,
    h: &mut H,
) -> Result<V, Refusal> {
    match source {
        // A SLOT. The CARRIER picks the accessor, not the kind: an
        // `OutSlot<0, Buf>` is a read, and `output_read` is what keeps an
        // encoder's hazard tracking honest.
        // A SLOT, AND THE RECTANGLE THAT CAME WITH IT. The CARRIER picks the
        // accessor, not the kind: an `Out<Tensor<u32>>` is a read, and
        // `output_read` is what keeps an encoder's hazard tracking honest.
        //
        // THE SHAPE RIDES ALONG because the mark is where a body reads it now.
        // `Holds::in_width` and `out_width` are the same accessors a
        // `Kind::InWidth` slot already went through -- the only change is that
        // the answer reaches the operand instead of a parameter beside it.
        Source::Slot(Kind::In, n) => {
            let at = h.input(n.into())?;
            shaped(ty, at, rows(h), h.in_width(n.into()).unwrap_or(0))
        }
        Source::Slot(Kind::Out, n) => {
            let at = if matches!(ty, Ty::Buf | Ty::I32s | Ty::U32s | Ty::F32s) {
                h.output_read(n.into())?
            } else {
                h.output(n.into())?
            };
            shaped(ty, at, rows(h), h.out_width(n.into()).unwrap_or(0))
        }
        // A WEIGHT HAS NO LAUNCH RECTANGLE. Its extents are the checkpoint
        // tensor's, which the fire's row count says nothing about, and a
        // reader taking `rows` off a weight would get this batch's token
        // count for a bank that has none.
        Source::Slot(Kind::Weight, n) => handle(ty, h.weight(n.into())?),
        // THE CARRIER SAYS WHICH READING. `Param<2, f32>` and `ParamF32<2>`
        // name the same scalar and differ only in how its bits are read.
        // Refusing the first spelling was a real defect, not a hypothetical.
        Source::Slot(Kind::Param, n) if matches!(ty, Ty::F32) => {
            Ok(V::f32(h.param_f32(n.into())?))
        }
        Source::Slot(Kind::Param, n) => number(ty, h.param(n.into())?),
        Source::Slot(Kind::ParamF32, n) => Ok(V::f32(h.param_f32(n.into())?)),
        // A RECTANGLE'S OWN EXTENTS, which are read rather than reckoned.
        // `Kind::InWidth` asks to be TOLD a width, so a statement stating
        // none refuses here where a region carrying none would take a zero:
        // the two are different questions and only one has a neutral answer.
        Source::Slot(Kind::InWidth, n) => number(ty, h.in_width(n.into())?),
        Source::Slot(Kind::OutWidth, n) => number(ty, h.out_width(n.into())?),
        Source::Slot(Kind::OutElements, n) => number(ty, h.out_elements(n.into())?),
        Source::Slot(kind, _) => Err(unstated(kind)),
        // ONE ADDRESS IN TWO SLOTS, resolved as the INPUT: that is the address
        // the statement placed, and the result wears the same one by
        // construction. NOT a chain — there is no second thing to try.
        Source::Alias(n, _) => {
            let at = h.input(n.into())?;
            shaped(ty, at, rows(h), h.in_width(n.into()).unwrap_or(0))
        }
        // A FACT the driver answers, by name.
        Source::Named(key) => named(ty, key, h),
        // A CHAIN, and only for scalars: `param` mints nothing, while `input`
        // numbers a handle whether or not the caller keeps it, so a discarded
        // attempt at a buffer would shift every handle after it.
        //
        // ZERO IS ABSENT: a grid axis of zero launches nothing.
        Source::Or(&Source::Slot(Kind::Param, n), fallback) => {
            match h.param(n.into()).ok().filter(|n| *n > 0) {
                Some(stated) => number(ty, stated),
                None => one(ty, *fallback, h),
            }
        }
        // THE WEIGHT CHAIN: the named bank first and the positional one after,
        // which is what `Const<Tensor<E>>` derives for every weight a routine
        // takes (`Or(Named("weight"), Slot(Kind::Weight, 0))`).
        //
        // It is safe where the scalar chain above needed a note: `named`
        // resolves a fact and mints NOTHING when it fails, so a discarded
        // attempt shifts no handle. `weight(n)` is the one that numbers, and
        // it is only reached once, on the fallback.
        //
        // Without this arm every shader-plane weight refused. The five marks
        // spelled the two halves apart -- `Weight<N, _>` took the named bank
        // and `Bank<N, _>` the positional one -- so no chain reached here at
        // all, and the four marks resolve one mark to both.
        Source::Or(&Source::Named(key), fallback) => match named(ty, key, h) {
            Ok(v) => Ok(v),
            Err(_) => one(ty, *fallback, h),
        },
        Source::Or(..) => Err(Refusal::Unstated {
            what: "a chain whose first half is neither one of the statement's \
                   scalars nor a fact the driver answers by name",
        }),
        // ARITHMETIC ON WHAT IS KNOWN. Both halves are themselves sources,
        // because a divisor may be a chain.
        Source::Times(a, b) => number(
            ty,
            count::<V, H>(*a, h)?.saturating_mul(count::<V, H>(*b, h)?),
        ),
        Source::Over(a, b) => {
            let d = count::<V, H>(*b, h)?;
            if d == 0 {
                return Err(Refusal::Empty { what: "a divisor" });
            }
            number(ty, count::<V, H>(*a, h)? / d)
        }
        // THE ABSENCE IS THE ANSWER, and it still takes a handle.
        Source::Lit(Lit::Null) => handle(ty, h.null()),
        Source::Lit(Lit::I32(n)) => number(ty, n),
        Source::Lit(_) => Err(Refusal::Unstated {
            what: "a literal argument at a carrier this binder has not met",
        }),
    }
}

/// One side of an arithmetic source, as a number. It goes through [`one`] so a
/// half is anything a whole argument may be; the carrier is fixed at
/// [`Ty::I32`] because arithmetic on a handle means nothing.
fn count<V: ShaderValue, H: Holds + ?Sized>(source: Source, h: &mut H) -> Result<i32, Refusal> {
    one::<V, H>(Ty::I32, source, h)?
        .as_i32()
        .ok_or(Refusal::Unstated {
            what: "a side of an arithmetic source that is not a number",
        })
}

/// This fire's row count, or zero where the driver does not answer for one.
///
/// Zero is already this crate's word for *"the statement gave no extent"*, so
/// a plane that answers no `rows` lands on the value every reader checks
/// rather than refusing an operand that is otherwise perfectly bound.
fn rows<H: Holds + ?Sized>(h: &mut H) -> i32 {
    match h.fact(<crate::keys::Rows as crate::keys::Fact>::KEY) {
        Some(Ok(Answer::Number(n))) => n,
        _ => 0,
    }
}

/// A bound buffer CARRYING ITS RECTANGLE, where the value has room for one.
///
/// [`ShaderValue::buffer_at`] defaults to dropping the shape, so a plane that
/// has nowhere to put it is unchanged and its marks answer zero -- which is
/// what they answered before this existed.
fn shaped<V: ShaderValue>(ty: Ty, at: u32, rows: i32, width: i32) -> Result<V, Refusal> {
    let whole: V = handle(ty, at)?;
    Ok(match whole.as_buffer() {
        // The direction is already decided by `handle`; re-deciding it here
        // from the `Ty` is one place too many for one fact.
        Some(_) if ty.binds() == crate::Binds::Writes => V::buffer_mut_at(at, rows, width),
        Some(_) => V::buffer_at(at, rows, width),
        None => whole,
    })
}

/// A bound buffer, at whatever carrier the signature spells.
///
/// The list is the whole POINTER half of [`Ty`], and it is long because CUDA
/// declares the element width where a shader plane declares only "a storage
/// buffer": `const int64_t*` and `const void*` are the same address and
/// different compile errors, which is the point of spelling both.
///
/// The split is by whether the LAUNCHER MAY WRITE, which is what a hazard
/// tracker needs -- see [`ShaderValue::buffer_mut`]. For the pointer arrays
/// the question is about the ENTRIES rather than the array: `void* const*`
/// is an array the launch reads, of buffers it writes, and it is the writes
/// that can collide.
fn handle<V: ShaderValue>(ty: Ty, at: u32) -> Result<V, Refusal> {
    Ok(match ty {
        Ty::Buf
        | Ty::I32s
        | Ty::U32s
        | Ty::F32s
        | Ty::U8s
        | Ty::I64s
        | Ty::U16s
        | Ty::I8s
        | Ty::Bf16s
        | Ty::F16s
        | Ty::U8Array
        | Ty::I32Array
        | Ty::BufArray => V::buffer(at),
        Ty::BufMut
        | Ty::F32sMut
        | Ty::I32sMut
        | Ty::U32sMut
        | Ty::U8sMut
        | Ty::U16sMut
        | Ty::I8sMut
        | Ty::Bf16sMut
        | Ty::F16sMut
        | Ty::BufArrayMut
        | Ty::BufArrayOut
        | Ty::BufArrayOutMut => V::buffer_mut(at),
        _ => {
            return Err(Refusal::Unstated {
                what: "a buffer argument at a carrier this binder has not met",
            });
        }
    })
}

/// A number, at whatever width the signature spells. The width is the point: a
/// backend packing scalars into a run of stated byte widths writes over a
/// neighbour when a [`Ty::Usize`] lands in a four-byte slot.
fn number<V: ShaderValue>(ty: Ty, n: i32) -> Result<V, Refusal> {
    Ok(match ty {
        Ty::I32 => V::i32(n),
        // A count riding in the struct rather than its own slot.
        Ty::U32 | Ty::InPacked => V::u32(n.cast_unsigned()),
        Ty::Usize => V::usize(u64::from(n.cast_unsigned())),
        // CUDA'S OWN WIDTHS. `long long` is a real parameter width there and
        // nowhere else, and a flag is one byte rather than four.
        Ty::I64 => V::i64(i64::from(n)),
        Ty::Bool => V::bool(n != 0),
        _ => {
            return Err(Refusal::Unstated {
                what: "a scalar argument at a carrier this binder has not met",
            });
        }
    })
}

/// A fact, by the key the signature names. See [`Holds::fact`].
fn named<V: ShaderValue, H: Holds + ?Sized>(
    ty: Ty,
    key: &'static str,
    h: &mut H,
) -> Result<V, Refusal> {
    let answer = h.fact(key).ok_or(Refusal::Unstated {
        what: "a fact this backend does not answer",
    })??;
    match answer {
        Answer::Handle(at) => handle(ty, at),
        Answer::Number(n) => number(ty, n),
        // NOT THROUGH `number`: an `f32`'s bits read as an `i32` are a
        // different value, and `Ty::F32` asks for the reading, not the bits.
        Answer::Float(x) => match ty {
            Ty::F32 => Ok(V::f32(x)),
            _ => Err(Refusal::Unstated {
                what: "a real fact at a carrier this binder has not met",
            }),
        },
        Answer::Truth(b) => match ty {
            Ty::Bool => Ok(V::bool(b)),
            _ => number(ty, i32::from(b)),
        },
        // ALREADY WIDE, so no sign round trip: the pool's strides fit an
        // `i32` on every deployment in the suite and would stop on a big one.
        Answer::Wide(n) => Ok(match ty {
            Ty::Usize => V::usize(n),
            Ty::U32 | Ty::InPacked => V::u32(u32::try_from(n).map_err(|_| too_wide(n, u64::from(u32::MAX)))?),
            Ty::I32 => V::i32(i32::try_from(n).map_err(|_| too_wide(n, i32::MAX as u64))?),
            _ => {
                return Err(Refusal::Unstated {
                    what: "a scalar argument at a carrier this binder has not met",
                });
            }
        }),
    }
}

/// A fact that does not fit the width its signature states.
fn too_wide(at: u64, max: u64) -> Refusal {
    Refusal::Wide {
        what: "a fact at a carrier narrower than the number the fire answers",
        at: i64::try_from(at).unwrap_or(i64::MAX),
        max: i64::try_from(max).unwrap_or(i64::MAX),
    }
}

/// The refusal for a slot kind no signature states.
fn unstated(kind: Kind) -> Refusal {
    Refusal::Unstated {
        what: match kind {
            Kind::Aux => "an `Aux` slot, which this backend does not stage",
            _ => "an operand kind this binder has not met",
        },
    }
}
