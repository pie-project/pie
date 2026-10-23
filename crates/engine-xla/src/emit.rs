//! The sink a lowering emits into.
//!
//! A lowering reads operand handles as SSA values, lets a kernel compute with
//! the [`Func`] builder, and writes the results back through their handles.
//! The [`Env`] decides what a handle is: a function parameter, a slice of a
//! value traced earlier in the same fire, a pool the executable donates.

use std::ops::{Deref, DerefMut};

use crate::handle::Tensor;
use kernels_xla::hlo::{Elem, Func, Val};
use kernels_xla::{Error, elem_of};

/// What a lowering emits into: the engine's sink for one node (or one
/// fire). A lowering hands its body to [`Emit::emit`], which lends it a [`Cx`]
/// for the duration; `&Ctx` is shared, like kernels-slang's `&Ctx`, so an
/// engine dispatch resolves its operands from `&self` while it holds one.
pub trait Emit {
    fn emit(&self, body: &mut dyn FnMut(&mut Cx<'_>) -> Result<(), Error>) -> Result<(), Error>;

    /// Names what is emitted next (the op a dispatch is lowering), for
    /// profiles; a sink may ignore it.
    fn scope(&self, name: &str) {
        let _ = name;
    }
}

pub type Ctx<'a> = dyn Emit + 'a;

/// What resolves the handles a lowering reads and writes.
pub trait Env {
    /// The value `t` holds now, as `[t.rows, t.width]` of `elem_of(t.dtype)`.
    fn read(&mut self, f: &mut Func, t: Tensor) -> Result<Val, Error>;

    /// Makes `v` (shaped `[t.rows, t.width]`) what `t` holds from here on.
    fn write(&mut self, f: &mut Func, t: Tensor, v: Val) -> Result<(), Error>;
}

/// A lowering's view: the function under construction plus the handle
/// resolution. Derefs to [`Func`], so a lowering builds with `cx.add(..)`.
pub struct Cx<'a> {
    f: &'a mut Func,
    env: &'a mut dyn Env,
    tuning: kernels_xla::tuning::Tuning,
}

impl<'a> Cx<'a> {
    pub fn new(f: &'a mut Func, env: &'a mut dyn Env) -> Self {
        Self {
            f,
            env,
            tuning: kernels_xla::tuning::Tuning::DEFAULT,
        }
    }

    /// The kernel decisions of the sink's load; a nested view takes its
    /// parent's.
    #[must_use]
    pub fn with_tuning(mut self, tuning: kernels_xla::tuning::Tuning) -> Self {
        self.tuning = tuning;
        self
    }

    #[must_use]
    pub fn tuning(&self) -> kernels_xla::tuning::Tuning {
        self.tuning
    }

    /// `t`'s current value, `[rows, width]`.
    pub fn read(&mut self, t: Tensor) -> Result<Val, Error> {
        self.env.read(self.f, t)
    }

    /// `t`'s current value converted to `elem`.
    pub fn read_as(&mut self, t: Tensor, elem: Elem) -> Result<Val, Error> {
        let v = self.read(t)?;
        Ok(self.f.convert(v, elem))
    }

    /// `t`'s current value in f32, the precision every reduction runs in.
    pub fn read_f32(&mut self, t: Tensor) -> Result<Val, Error> {
        self.read_as(t, Elem::F32)
    }

    /// Stores `v` into `t`, converting to `t`'s storage type and reshaping to
    /// `[rows, width]` when `v` holds the same elements in another shape.
    pub fn write(&mut self, t: Tensor, v: Val) -> Result<(), Error> {
        let elem = elem_of("write", t.dtype)?;
        let v = self.f.convert(v, elem);
        let v = self
            .f
            .reshape(v, &[i64::from(t.rows), i64::from(t.width)])?;
        self.env.write(self.f, t, v)
    }

    #[must_use]
    pub fn func(&mut self) -> &mut Func {
        self.f
    }
}

impl Deref for Cx<'_> {
    type Target = Func;
    fn deref(&self) -> &Func {
        self.f
    }
}

impl DerefMut for Cx<'_> {
    fn deref_mut(&mut self) -> &mut Func {
        self.f
    }
}
