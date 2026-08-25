//! One statement of this fire, bound: the half of the point path that a
//! generator CANNOT write.
//!
//! A plane's `points_dispatch` is generated off `kernels::points`' slot lists
//! and that plane's claim tables, and it says -- for every point the plane
//! claims -- which column each slot reads, what element the axis rides, and
//! which trait method to call. What it cannot say is where a column LIVES. That
//! is the executor's, and it differs by executor: the same generated file
//! serves a driver with a batched arena and real pools and would serve a
//! one-row smoke, so the file is generated per plane and this one is written
//! once for all of them.
//!
//! # What is a payload here, and what is not
//!
//! EVERY ACCESSOR BELOW IS `driver-cuda/src/baker/bound.rs`'s WITH THE PAYLOAD
//! LIFTED OUT, and the lifting cost nothing because the floor had already done
//! it. `kernels::points::Plane` says what a `Tensor<T>`, a `Bank<R>`, a
//! `Recurrent` and a `Pages` are per plane; this file names all four through
//! that projection and never spells one. What it adds is the fifth -- the
//! REGION -- which the floor has no opinion about because no kernel ever sees
//! one.
//!
//! Two places the difference between the shader planes actually shows, and
//! neither of them is in this file:
//!
//! * **`tconst` and `bank` read a REGION, not a pointer.** Cuda's `Bank`
//!   carries `*mut c_void`; metal's carries an address and an extent, because a
//!   Metal buffer is bound with an extent and a binding past its end is an
//!   error rather than a fault; wgpu's names its BUFFER, because a
//!   `wgpu::BufferBinding` is an object and two offsets. All three arrive here
//!   as [`Plane::Slice`] and go straight back out through
//!   [`Fires::wconst`]/[`Fires::wbank`].
//! * **`tinout` SCHEDULES a copy rather than issuing one.** Cuda calls
//!   `cudaMemcpyAsync` inside the accessor; neither shader plane has a command
//!   encoder here, so `Fire::inout` records the pair and the device half stages
//!   it in walk order. See [`crate::walk::Blit`].
//!
//! The dtype check is [`rides`] (private, below), and it is owed once for every
//! slot of every point -- which is what a `Scalar` bound on the accessor is for.
//! A `tin::<f32>` against a bf16 rectangle is a REINTERPRETATION, not a cast:
//! it halves every stride inside the kernel and returns a plausible wrong
//! answer.
//!
//! [`Plane::Slice`]: crate::Plane::Slice
//! [`Fires::wconst`]: crate::Fires::wconst
//! [`Fires::wbank`]: crate::Fires::wbank

use kernels::bound::{BoundOp, Site};
use kernels::plane::{Cache, Const, In, InOut, Out, Refusal};
use kernels::points::{Form, Repr, Scalar, ScalarKind};
use model::produce::Dtype;
use model_compiler::program::Dt;
use model_ir::plan::Op;

use crate::walk::Fire;
use crate::walk::{BankPlanes, Fires, Pages, Recurrent, Tensor};

/// One bound statement: the fire it runs in, the statement it is, and the point
/// the lane resolved it to.
pub(crate) struct Bound<'f, 'a, 'c, P: Fires<'c>> {
    pub fire: &'f Fire<'a, P>,
    pub op: &'f Op,
    /// Index into `plan.ops`, for the one thing a statement cannot carry: which
    /// step a scheduled copy belongs to.
    pub at: u32,
    pub point: &'f str,
    /// THE ENCODER'S LIFETIME, WHICH THIS STATEMENT DOES NOT HOLD.
    ///
    /// `BoundOp::Plane` must be the very `Ctx<'c>` the generated dispatch was
    /// handed, and `'c` reaches this type only through that impl -- which makes
    /// it unconstrained without a field naming it. It is a marker rather than a
    /// borrow because a bound statement genuinely holds nothing of the
    /// encoder's: `Fire::step` mints one, hands it over and drops it inside a
    /// single call.
    pub fires: core::marker::PhantomData<&'c ()>,
}

/// What a rectangle the walk sized rides, as the floor names it.
fn axis(dt: Dt) -> ScalarKind {
    match dt {
        Dt::Bf16 => ScalarKind::Bf16,
        Dt::F32 => ScalarKind::F32,
        Dt::I32 => ScalarKind::I32,
        Dt::U32 => ScalarKind::U32,
        Dt::U8 => ScalarKind::U8,
    }
}

/// What a BANK rides, which is the CHECKPOINT's storage axis and not the plan's
/// repr column. `None` for a dtype no point can be instantiated at, which reads
/// as a refusal rather than as a match.
fn bank_axis(d: Dtype) -> Option<ScalarKind> {
    match d {
        Dtype::Bf16 => Some(ScalarKind::Bf16),
        Dtype::F16 => Some(ScalarKind::F16),
        Dtype::F32 => Some(ScalarKind::F32),
        Dtype::I32 => Some(ScalarKind::I32),
        Dtype::U32 => Some(ScalarKind::U32),
        Dtype::U8 => Some(ScalarKind::U8),
        _ => None,
    }
}

/// THE CHECK EVERY SLOT OF EVERY POINT IS OWED.
///
/// Private because it is the implementation of one accessor's obligation and
/// not a surface: what a caller sees is a refusal naming the slot.
///
/// A dispatch arm picks the element off ONE witness slot and asks every other
/// slot for the element its declaration pins. `norm.rmsnorm_gated` states an
/// f32 core and an f32 weight beside a bf16 gate; reading a bf16 rectangle as
/// f32 halves every stride inside the kernel.
fn rides<T: Scalar>(what: &'static str, have: ScalarKind) -> Result<(), Refusal> {
    if T::KIND == have {
        return Ok(());
    }
    Err(Refusal::Absent { what })
}

impl<'c, P: Fires<'c>> BoundOp for Bound<'_, '_, 'c, P> {
    type Plane = P::Ctx;

    fn point(&self) -> &str {
        self.point
    }

    fn dtype(&self, at: Site) -> Result<ScalarKind, Refusal> {
        Ok(match at {
            Site::In(i) => axis(self.fire.input(self.op, i)?.dt),
            Site::Out(i) => axis(self.fire.output(self.op, i)?.dt),
            Site::Const(i) => {
                bank_axis(self.fire.weight(self.op, i)?.dtype).ok_or(Refusal::Absent {
                    what: "a bank at an element no point is instantiated at",
                })?
            }
        })
    }

    fn tin<T: Scalar>(&self, at: usize) -> Result<In<Tensor<'c, P, T>>, Refusal> {
        let r = self.fire.input(self.op, at)?;
        rides::<T>(
            "an operand at an element the point does not state",
            axis(r.dt),
        )?;
        Ok(P::rin(&mut self.fire.bindings.borrow_mut(), r))
    }

    fn tout<T: Scalar>(&self, at: usize) -> Result<Out<Tensor<'c, P, T>>, Refusal> {
        let r = self.fire.output(self.op, at)?;
        rides::<T>(
            "a result at an element the point does not state",
            axis(r.dt),
        )?;
        Ok(P::rout(&mut self.fire.bindings.borrow_mut(), r))
    }

    fn tinout<T: Scalar>(
        &self,
        from: usize,
        to: usize,
    ) -> Result<InOut<Tensor<'c, P, T>>, Refusal> {
        // The copy that makes an `InOut` honest here: the walk mints a FRESH
        // rectangle for every result, so the operand's bytes have to be in the
        // result's region before the kernel writes through it. See
        // `Fire::inout`.
        let r = self.fire.inout(
            self.fire.input(self.op, from)?,
            self.fire.output(self.op, to)?,
            self.at,
        )?;
        rides::<T>(
            "an in-place operand at an element the point does not state",
            axis(r.dt),
        )?;
        Ok(P::rio(&mut self.fire.bindings.borrow_mut(), r))
    }

    fn tconst<T: Scalar>(&self, at: usize) -> Result<Const<Tensor<'c, P, T>>, Refusal> {
        let bank = self.fire.weight(self.op, at)?;
        let have = bank_axis(bank.dtype).ok_or(Refusal::Absent {
            what: "a bank at an element no point is instantiated at",
        })?;
        rides::<T>("a bank at an element the point does not state", have)?;
        Ok(P::wconst(&mut self.fire.bindings.borrow_mut(), bank.slice))
    }

    fn form(&self, at: usize) -> Result<Form, Refusal> {
        Form::from_name(&self.fire.weight(self.op, at)?.repr).ok_or(Refusal::Absent {
            what: "a bank at a repr no point is instantiated at",
        })
    }

    fn bank<R: Repr>(&self, at: usize) -> Result<Const<BankPlanes<'c, P, R>>, Refusal> {
        // THE PLANES ARE COLUMNS, and this is the only accessor that reads more
        // than one. The model text registered them in the repr's own order --
        // codes, then scales -- and the DSL's `Stmt::bank` is what put them in
        // the statement that way, so this reads them positionally exactly as
        // every other accessor reads its column.
        let planes: Vec<P::Slice> = (0..R::PLANES)
            .map(|p| {
                let bank = self.fire.weight(self.op, at + p)?;
                // Both planes are BYTES on every repr this executor binds, and
                // a plane that is not is a bank read as something it is not.
                if bank.dtype != Dtype::U8 {
                    return Err(Refusal::Absent {
                        what: "a quantised bank plane stored at an element that is not `u8`",
                    });
                }
                Ok(bank.slice)
            })
            .collect::<Result<_, _>>()?;
        let [codes, scales] = planes.as_slice() else {
            return Err(Refusal::Absent {
                what: "a bank whose repr stores a plane count this executor cannot bind",
            });
        };
        Ok(P::wbank(
            &mut self.fire.bindings.borrow_mut(),
            *codes,
            *scales,
        ))
    }

    fn recurrent(&self) -> Result<Cache<Recurrent<'c, P>>, Refusal> {
        self.fire.recurrent(self.op)
    }

    fn pages(&self) -> Result<Cache<Pages<'c, P>>, Refusal> {
        self.fire.pages(self.op)
    }

    fn u32(&self, at: usize) -> Result<u32, Refusal> {
        Fire::<P>::p32(self.op, at)
    }

    fn f32(&self, at: usize) -> Result<f32, Refusal> {
        Fire::<P>::pf32(self.op, at)
    }

    fn bool(&self, at: usize) -> Result<bool, Refusal> {
        self.u32(at).map(|w| w != 0)
    }

    fn layer(&self) -> Result<u32, Refusal> {
        self.op.layer.ok_or(Refusal::Unstated {
            what: "the layer tag this statement is read at",
        })
    }
}
