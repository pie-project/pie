//! The sink kernels emit into.
//!
//! A kernel entry reads its operand handles as SSA values, computes with the
//! [`Func`] builder, and writes its results back through their handles. The
//! engine's [`Env`] decides what a handle is: a function parameter, a slice of
//! a value traced earlier in the same fire, a pool the executable donates.

use std::ops::{Deref, DerefMut};

use dtype::Dtype;

use crate::error::{Error, refuse};
use crate::hlo::{Elem, Func, Val};
use crate::tensor::Tensor;

/// What a kernel entry emits into: the engine's sink for one node (or one
/// fire). An entry hands its body to [`Emit::emit`], which lends it a [`Cx`]
/// for the duration; `&Ctx` is shared, like kernels-wgpu's `&Ctx`, so an
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

/// What an engine supplies so kernels can name its handles.
pub trait Env {
    /// The value `t` holds now, as `[t.rows, t.width]` of `elem_of(t.dtype)`.
    fn read(&mut self, f: &mut Func, t: Tensor) -> Result<Val, Error>;

    /// Makes `v` (shaped `[t.rows, t.width]`) what `t` holds from here on.
    fn write(&mut self, f: &mut Func, t: Tensor, v: Val) -> Result<(), Error>;
}

/// A kernel's view: the function under construction plus the engine's
/// handle resolution. Derefs to [`Func`], so a kernel builds with `cx.add(..)`.
pub struct Cx<'a> {
    f: &'a mut Func,
    env: &'a mut dyn Env,
}

impl<'a> Cx<'a> {
    pub fn new(f: &'a mut Func, env: &'a mut dyn Env) -> Self {
        Self { f, env }
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
        let v = self.f.reshape(v, &[i64::from(t.rows), i64::from(t.width)])?;
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

/// The element type a plain dtype stores as.
pub fn elem_of(op: &'static str, dtype: Dtype) -> Result<Elem, Error> {
    Ok(match dtype {
        Dtype::F32 => Elem::F32,
        Dtype::F16 => Elem::F16,
        Dtype::Bf16 => Elem::Bf16,
        Dtype::E4m3 => Elem::F8E4m3fn,
        Dtype::E5m2 => Elem::F8E5m2,
        Dtype::E8m0 => Elem::F8E8m0fnu,
        Dtype::I64 => Elem::I64,
        Dtype::I32 => Elem::I32,
        Dtype::I16 => Elem::I16,
        Dtype::I8 => Elem::I8,
        Dtype::U64 => Elem::U64,
        Dtype::U32 => Elem::U32,
        Dtype::U16 => Elem::U16,
        Dtype::U8 | Dtype::Bool => Elem::U8,
        other => {
            return Err(refuse(
                op,
                format!("{other:?} is a packed format; it is read through its bank"),
            ));
        }
    })
}

/// Refuses unless `t` is one of `dtypes`.
pub fn expect(op: &'static str, t: Tensor, dtypes: &[Dtype]) -> Result<(), Error> {
    if dtypes.contains(&t.dtype) {
        Ok(())
    } else {
        Err(Error::DtypeUnsupported { op, dtype: t.dtype })
    }
}

/// Refuses unless `t` is `[rows, width]`.
pub fn shaped(op: &'static str, what: &str, t: Tensor, rows: u32, width: u32) -> Result<(), Error> {
    if t.rows == rows && t.width == width {
        Ok(())
    } else {
        Err(refuse(
            op,
            format!(
                "the {what} is {}x{}, and this op wants {rows}x{width}",
                t.rows, t.width
            ),
        ))
    }
}
