//! Lowering one kernel call: the engine reads the operand handles as values,
//! the kernel (a pure function of them) computes, and the engine lands what
//! it returns in the result handles, converted to their dtype and reshaped
//! to their rows. Where data lives is the engine's; the kernel only emits.

use kernels_xla::hlo::{Func, Val};
use kernels_xla::tensor::{Operand, RaggedTensor, Tensor};
use kernels_xla::tuning::Tuning;
use kernels_xla::{Ctx, Cx, Error};

/// A kernel call under construction.
pub struct Lower<'a, 'b> {
    cx: &'a mut Cx<'b>,
}

impl Lower<'_, '_> {
    /// The function the kernel emits into.
    pub fn f(&mut self) -> &mut Func {
        self.cx.func()
    }

    /// The kernel decisions of this load.
    #[must_use]
    pub fn tuning(&self) -> Tuning {
        self.cx.tuning()
    }

    /// `t`'s current value, as the operand a kernel reads.
    pub fn read(&mut self, t: Tensor) -> Result<Operand, Error> {
        Ok(Operand {
            val: self.cx.read(t)?,
            rows: t.rows,
            width: t.width,
            dtype: t.dtype,
        })
    }

    /// `t`'s current value, when there is a `t`.
    pub fn read_opt(&mut self, t: Option<Tensor>) -> Result<Option<Operand>, Error> {
        t.map(|t| self.read(t)).transpose()
    }

    /// A ragged tensor's data, then its indptr.
    pub fn ragged(&mut self, r: RaggedTensor) -> Result<RaggedTensor<Operand>, Error> {
        r.try_map(|t| self.read(t))
    }

    /// Makes `v` what `t` holds from here on, converted to `t`'s storage
    /// element and reshaped to `[rows, width]`.
    pub fn land(&mut self, t: Tensor, v: Val) -> Result<(), Error> {
        self.cx.write(t, v)
    }
}

/// Lowers one kernel call into `ctx`'s function.
pub fn lower(
    ctx: &Ctx<'_>,
    mut body: impl FnMut(&mut Lower<'_, '_>) -> Result<(), Error>,
) -> Result<(), Error> {
    ctx.emit(&mut |cx| body(&mut Lower { cx }))
}
