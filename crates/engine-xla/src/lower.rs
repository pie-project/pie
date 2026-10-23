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

    /// A bank, codes, then scales, then biases. A bank of pre-scaled weights
    /// (`E5m2` codes, `kernels_xla::pack`) reads only its codes: its factor
    /// operands keep their planes' shapes over the codes' value, which no
    /// kernel reads for such a bank.
    pub fn bank(
        &mut self,
        w: kernels_xla::tensor::Bank,
    ) -> Result<kernels_xla::tensor::Bank<Operand>, Error> {
        if w.codes.dtype != dtype::Dtype::E5m2 {
            return w.try_map(|t| self.read(t));
        }
        let codes = self.read(w.codes)?;
        let stand_in = |t: Tensor| Operand {
            val: codes.val,
            rows: t.rows,
            width: t.width,
            dtype: t.dtype,
        };
        Ok(kernels_xla::tensor::Bank {
            codes,
            scales: stand_in(w.scales),
            biases: w.biases.map(stand_in),
            group: w.group,
            bits: w.bits,
        })
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

    /// Lands the `[rows, 1]` value `v` in column `column` of `t`, the other
    /// columns kept as they were.
    pub fn land_column(&mut self, t: Tensor, column: u32, v: Val) -> Result<(), Error> {
        if t.width == 1 {
            return self.cx.write(t, v);
        }
        let old = self.cx.read(t)?;
        let f = self.cx.func();
        let r = f.const_i(kernels_xla::hlo::Elem::I32, 0, &[]);
        let c = f.const_i(kernels_xla::hlo::Elem::I32, i64::from(column), &[]);
        let merged = f.dynamic_update_slice(old, v, &[r, c])?;
        self.cx.write(t, merged)
    }

    /// Lands `v` over the first rows of `t`, the rest of `t` kept as it was.
    pub fn land_head(&mut self, t: Tensor, v: Val) -> Result<(), Error> {
        let f = self.cx.func();
        let n = f.dims(v)[0];
        if n == i64::from(t.rows) {
            return self.cx.write(t, v);
        }
        let elem = kernels_xla::elem_of("layout", t.dtype)?;
        let v = self.cx.func().convert(v, elem);
        let old = self.cx.read(t)?;
        let f = self.cx.func();
        let zero = f.const_i(kernels_xla::hlo::Elem::I32, 0, &[]);
        let merged = f.dynamic_update_slice(old, v, &[zero, zero])?;
        self.cx.write(t, merged)
    }
}

/// Lowers one kernel call into `ctx`'s function.
pub fn lower(
    ctx: &Ctx<'_>,
    mut body: impl FnMut(&mut Lower<'_, '_>) -> Result<(), Error>,
) -> Result<(), Error> {
    ctx.emit(&mut |cx| body(&mut Lower { cx }))
}

/// The quantized projections lowered from handles: each plane read as its
/// kernel's view says (`bank_view`, `kquant::view`, `nvfp4::views`), in the
/// order the kernel consumes them.
pub mod linear {
    use kernels_xla::linear::{decode, kquant, nvfp4, quant};
    use kernels_xla::tensor::{Bank, Tensor};
    use kernels_xla::{Ctx, Error};

    use super::lower;

    /// `y = act · wᵀ` (or the lm head) for a quantized bank.
    pub fn quant(ctx: &Ctx<'_>, head: bool, act: Tensor, w: Bank, y: Tensor) -> Result<(), Error> {
        lower(ctx, |l| {
            let a = l.read(act)?;
            let w = l.bank(quant::bank_view(w))?;
            let tuning = l.tuning();
            let v = if head {
                quant::lm_head(l.f(), a, w, y.landing(), tuning)?
            } else {
                quant::matmul(l.f(), a, w, y.landing(), tuning)?
            };
            match v {
                Some(v) => l.land(y, v),
                None => Ok(()),
            }
        })
    }

    /// `y = act · wᵀ` (or the lm head) for a K-quant plane.
    pub fn kquant(
        ctx: &Ctx<'_>,
        head: bool,
        act: Tensor,
        w: Tensor,
        y: Tensor,
    ) -> Result<(), Error> {
        lower(ctx, |l| {
            let w = l.read(kquant::view(w))?;
            let a = l.read(act)?;
            let v = if head {
                kquant::lm_head(l.f(), a, w, y.landing())?
            } else {
                kquant::matmul(l.f(), a, w, y.landing())?
            };
            match v {
                Some(v) => l.land(y, v),
                None => Ok(()),
            }
        })
    }

    /// `y = act · wᵀ` (or the lm head) for nvfp4 planes.
    #[allow(clippy::too_many_arguments)]
    pub fn nvfp4(
        ctx: &Ctx<'_>,
        head: bool,
        act: Tensor,
        codes: Tensor,
        scales: Tensor,
        tensor_scale: f32,
        y: Tensor,
    ) -> Result<(), Error> {
        lower(ctx, |l| {
            let (codes, scales) = nvfp4::views(codes, scales, act.width);
            let a = l.read(act)?;
            let c = l.read(codes)?;
            let s = l.read(scales)?;
            let tuning = l.tuning();
            let v = if head {
                nvfp4::lm_head(l.f(), a, c, s, tensor_scale, y.landing(), tuning)?
            } else {
                nvfp4::matmul(l.f(), a, c, s, tensor_scale, y.landing(), tuning)?
            };
            match v {
                Some(v) => l.land(y, v),
                None => Ok(()),
            }
        })
    }

    /// `out = decode(w)`, bf16.
    pub fn decoded_plane(
        ctx: &Ctx<'_>,
        op: &'static str,
        w: Bank,
        out: Tensor,
    ) -> Result<(), Error> {
        lower(ctx, |l| {
            let w = l.bank(quant::bank_view(w))?;
            let v = decode::decoded_plane(l.f(), op, w, out.landing())?;
            l.land(out, v)
        })
    }

    /// A K-quant plane repacked into an affine bank's three planes.
    pub fn kquant_to_affine(
        ctx: &Ctx<'_>,
        w: Tensor,
        k: u32,
        codes: Tensor,
        scales: Tensor,
        biases: Tensor,
    ) -> Result<kquant::Scheme, Error> {
        let mut scheme = None;
        lower(ctx, |l| {
            let w = l.read(kquant::view(w))?;
            let (s, [q, sc, b]) = kquant::to_affine(
                l.f(),
                w,
                k,
                codes.landing(),
                scales.landing(),
                biases.landing(),
            )?;
            l.land(codes, q)?;
            l.land(scales, sc)?;
            l.land(biases, b)?;
            scheme = Some(s);
            Ok(())
        })?;
        Ok(scheme.expect("the lowering ran"))
    }
}
