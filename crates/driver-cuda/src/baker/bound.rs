//! One statement of this fire, bound: the half of the point path that a
//! generator CANNOT write.
//!
//! `kernels_cuda::points_dispatch` is generated off `kernels::points`' slot
//! lists and this plane's claim tables, and it says — for every point cuda
//! claims — which column each slot reads, what element the axis rides, and
//! which trait method to call. What it cannot say is where a column LIVES.
//! That is the executor's, and it differs: `baker-smoke` has one row and one
//! `cudaMalloc`ed page, this driver has a batched arena and real pools. So
//! the generated file is shared and this file is not.
//!
//! # What stood here instead
//!
//! `points_shim.rs`: a hand-written `match` over point names, seventeen arms
//! deep, whose own header called itself *"a placeholder for a generator"* and
//! named W5 as the generator. W5 landed — `baker-smoke` has fired the
//! generated dispatch since — and the driver kept the placeholder because
//! nothing had forced the crossing. R4b forced it: the fa2 points are claim
//! BODIES now, and a hand-written arm would have had to grow a door onto the
//! fire's staging that the generated arm already has (`Ctx::raised`). Every
//! arm the shim carried is emitted, and the twelve methods below are what
//! each of those arms used to open with.
//!
//! The dtype check the shim owed is [`rides`], and it is now owed once for
//! every slot of every point rather than at the two the shim remembered.

use kernels::bound::{Axis, BoundOp, Rides, Site};
use kernels::plane::{Cache, Const, In, InOut, Out, Refusal};
use kernels::points::{Form, Repr};
use kernels::raises::Struct;
use kernels_cuda::jit::Ctx;
use kernels_cuda::jit::abi::{Bank as CudaBank, Planes, Tensor};
use kernels_cuda::views::{KvCache, RecurrentState};
use model::produce::Dtype;
use model_compiler::program::Dt;
use model_ir::plan::Op;

use super::Bank;
use super::fire::Fire;
use super::marks::{rin, rio, rout, wconst};

/// One bound statement: the fire it runs in, the statement it is, and the
/// point the lane resolved it to.
pub(crate) struct Bound<'f, 'a> {
    pub fire: &'f Fire<'a>,
    pub op: &'f Op,
    pub point: &'f str,
}

/// What a rectangle the walk sized rides, as the floor names it.
fn axis(dt: Dt) -> Axis {
    match dt {
        Dt::Bf16 => Axis::Bf16,
        Dt::F32 => Axis::F32,
        Dt::I32 => Axis::I32,
        Dt::U32 => Axis::U32,
        Dt::U8 => Axis::U8,
    }
}

/// What a BANK rides, which is the checkpoint's storage axis and not the
/// plan's repr column. `None` for a dtype no point can be instantiated at,
/// which reads as a refusal rather than as a match.
fn bank_axis(d: Dtype) -> Option<Axis> {
    match d {
        Dtype::Bf16 => Some(Axis::Bf16),
        Dtype::F16 => Some(Axis::F16),
        Dtype::F32 => Some(Axis::F32),
        Dtype::I32 => Some(Axis::I32),
        Dtype::U32 => Some(Axis::U32),
        Dtype::U8 => Some(Axis::U8),
        _ => None,
    }
}

/// THE CHECK THE HAND SHIM MADE TWICE AND OWED EVERYWHERE ELSE.
///
/// A dispatch arm picks the element off ONE witness slot and asks every other
/// slot for the element its declaration pins. `norm.rmsnorm_gated` states an
/// f32 core and an f32 weight beside a bf16 gate, and reading a bf16
/// rectangle as f32 is a reinterpretation, not a cast — it halves every
/// stride inside the kernel and returns a plausible wrong answer.
fn rides<T: Rides>(what: &'static str, have: Axis) -> Result<(), Refusal> {
    if T::AXIS == have {
        return Ok(());
    }
    Err(Refusal::Absent { what })
}

impl<'a> BoundOp for Bound<'_, 'a> {
    type Plane = Ctx<'a>;

    fn point(&self) -> &str {
        self.point
    }

    fn dtype(&self, at: Site) -> Result<Axis, Refusal> {
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

    fn tin<T: Rides>(&self, at: usize) -> Result<In<Tensor<T>>, Refusal> {
        let r = self.fire.input(self.op, at)?;
        rides::<T>(
            "an operand at an element the point does not state",
            axis(r.dt),
        )?;
        Ok(rin(r))
    }

    fn tout<T: Rides>(&self, at: usize) -> Result<Out<Tensor<T>>, Refusal> {
        let r = self.fire.output(self.op, at)?;
        rides::<T>(
            "a result at an element the point does not state",
            axis(r.dt),
        )?;
        Ok(rout(r))
    }

    fn tinout<T: Rides>(&self, from: usize, to: usize) -> Result<InOut<Tensor<T>>, Refusal> {
        // The D2D that makes an `InOut` honest here: the walk mints a FRESH
        // rectangle for every result, so the operand's bytes have to be in
        // the result's rectangle before the kernel writes through it. See
        // `Fire::inout`.
        let r = self.fire.inout(
            self.fire.input(self.op, from)?,
            self.fire.output(self.op, to)?,
        )?;
        rides::<T>(
            "an in-place operand at an element the point does not state",
            axis(r.dt),
        )?;
        Ok(rio(r))
    }

    fn tconst<T: Rides>(&self, at: usize) -> Result<Const<Tensor<T>>, Refusal> {
        let bank = self.fire.weight(self.op, at)?;
        let have = bank_axis(bank.dtype).ok_or(Refusal::Absent {
            what: "a bank at an element no point is instantiated at",
        })?;
        rides::<T>("a bank at an element the point does not state", have)?;
        Ok(wconst(bank.ptr))
    }

    fn form(&self, at: usize) -> Result<Form, Refusal> {
        Form::from_name(&self.fire.weight(self.op, at)?.repr).ok_or(Refusal::Absent {
            what: "a bank at a repr no point is instantiated at",
        })
    }

    fn bank<R: Repr>(&self, at: usize) -> Result<Const<CudaBank<R>>, Refusal> {
        // THE PLANES ARE COLUMNS, and this is the only accessor that reads
        // more than one. The model text registered them in the repr's own
        // order — codes, then scales — and the DSL's `Stmt::bank` is what put
        // them in the statement that way, so this reads them positionally
        // exactly as every other accessor reads its column.
        let planes: Vec<&Bank> = (0..R::PLANES)
            .map(|p| self.fire.weight(self.op, at + p))
            .collect::<Result<_, _>>()?;
        let [codes, scales] = planes.as_slice() else {
            return Err(Refusal::Absent {
                what: "a bank whose repr stores a plane count this executor cannot bind",
            });
        };
        // Both planes are BYTES on every repr this executor binds, and a
        // plane that is not is a bank read as something it is not.
        for plane in [codes, scales] {
            if plane.dtype != Dtype::U8 {
                return Err(Refusal::Absent {
                    what: "a quantised bank plane stored at an element that is not `u8`",
                });
            }
        }
        Ok(Const::new(Planes {
            codes: codes.ptr.cast_const().cast::<u8>(),
            scales: scales.ptr.cast_const().cast::<u8>(),
        }))
    }

    fn recurrent(&self) -> Result<Cache<Struct<RecurrentState>>, Refusal> {
        self.fire.recurrent(self.op)
    }

    fn pages(&self) -> Result<Cache<Struct<KvCache>>, Refusal> {
        self.fire.pages(self.op)
    }

    fn u32(&self, at: usize) -> Result<u32, Refusal> {
        Fire::p32(self.op, at)
    }

    fn f32(&self, at: usize) -> Result<f32, Refusal> {
        Fire::pf32(self.op, at)
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
