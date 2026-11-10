//! Lowering one kernel call: the engine reads the operand handles as values,
//! the kernel (a pure function of them) computes, and the engine lands what
//! it returns in the result handles, converted to their dtype and reshaped
//! to their rows. Where data lives is the engine's; the kernel only emits.

use crate::emit::{Ctx, Cx};
use crate::handle::{RaggedTensor, Tensor};
use kernels_xla::Error;
use kernels_xla::hlo::{Func, Val};
use kernels_xla::tensor::Operand;
use kernels_xla::tuning::Tuning;

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

    /// A slot table `t` of a recurrent pool whose slots lie where `map`
    /// puts them ([`kernels_xla::tensor::PageMap`]): each slot as its block
    /// of the pool's plane, the `-(s) - 1` encoding kept. The identity reads
    /// `t` as it is.
    pub fn read_slots(
        &mut self,
        t: Tensor,
        map: kernels_xla::tensor::PageMap,
    ) -> Result<Operand, Error> {
        let o = self.read(t)?;
        if map.is_identity() {
            return Ok(o);
        }
        let f = self.cx.func();
        let v = f.convert(o.val, kernels_xla::hlo::Elem::I32);
        let v = map.apply(f, v)?;
        Ok(Operand {
            val: v,
            dtype: dtype::Dtype::I32,
            ..o
        })
    }

    /// A recurrent pool's slot table, mapped ([`Lower::read_slots`]).
    pub fn slots(&mut self, pool: &crate::handle::RecurrentPool) -> Result<Operand, Error> {
        self.read_slots(pool.slots, pool.map)
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
        w: crate::handle::Bank,
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

    /// A streamed bank (`crate::store::streamed`) as the dense
    /// `[experts · rows, width]` value the base build reads: expert `e`'s
    /// rows gathered from the pool view at `loc[e]`.
    pub fn streamed_bank(
        &mut self,
        bank: crate::store::streamed::Bank,
        width: u32,
    ) -> Result<Operand, Error> {
        use kernels_xla::hlo::Elem;
        let view = self.read(bank.view)?;
        let loc = self.read(bank.loc)?;
        let (e, r) = (i64::from(bank.experts), i64::from(bank.view_rows));
        let f = self.cx.func();
        let loc = f.reshape(loc.val, &[e])?;
        let loc = f.broadcast(loc, &[e, r], &[0])?;
        let iota = f.iota(Elem::I32, &[e, r], 1);
        let idx = f.add(loc, iota)?;
        let idx = f.reshape(idx, &[e * r])?;
        let val = f.take_rows(view.val, idx)?;
        let val = f.reshape(
            val,
            &[i64::from(bank.experts * bank.rows), i64::from(width)],
        )?;
        Ok(Operand {
            val,
            rows: bank.experts * bank.rows,
            width,
            dtype: dtype::Dtype::Bf16,
        })
    }

    /// A ragged tensor's data, then its indptr.
    pub fn ragged(
        &mut self,
        r: RaggedTensor,
    ) -> Result<kernels_xla::tensor::RaggedTensor<Operand>, Error> {
        r.try_map(|t| self.read(t))
    }

    /// Makes `v` what `t` holds from here on, converted to `t`'s storage
    /// element and reshaped to `[rows, width]`.
    pub fn land(&mut self, t: Tensor, v: Val) -> Result<(), Error> {
        self.cx.write(t, v)
    }

    /// [`Self::land`], keeping `sums`, the per-head sums of squares the
    /// kernel landed beside `v`, for the norm that reads `t` next.
    pub fn land_sums(&mut self, t: Tensor, v: Val, sums: Val) -> Result<(), Error> {
        self.cx.write(t, v)?;
        self.cx.keep_sums(t, sums);
        Ok(())
    }

    /// The per-head sums of squares landed beside `t`'s current value.
    pub fn sums(&mut self, t: Tensor) -> Option<Val> {
        self.cx.sums(t)
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

pub mod attn;

/// Lowers a kernel call that lands several pool planes: once a plane, each
/// time reading its operands afresh and landing only that plane (the others'
/// results are dead and dropped), since every plane is a view of one pool
/// and a whole-plane result read before another plane's write would land
/// without it. Planes that are one handle land in one pass, chained as the
/// kernel chains them. `body` answers one value a plane, `None` where the
/// call lands nothing there.
pub fn lower_planes(
    ctx: &Ctx<'_>,
    planes: &[Tensor],
    mut body: impl FnMut(&mut Lower<'_, '_>) -> Result<Vec<Option<Val>>, Error>,
) -> Result<(), Error> {
    let one = planes.windows(2).all(|w| w[0].buf == w[1].buf);
    if one {
        return lower(ctx, |l| {
            for (plane, v) in planes.iter().zip(body(l)?) {
                if let Some(v) = v {
                    l.land(*plane, v)?;
                }
            }
            Ok(())
        });
    }
    for (at, plane) in planes.iter().enumerate() {
        lower(ctx, |l| match body(l)?.get(at).copied().flatten() {
            Some(v) => l.land(*plane, v),
            None => Ok(()),
        })?;
    }
    Ok(())
}

/// [`lower_planes`] for a kernel that writes its planes in turn, each onto
/// what the one before left (planes of one pool: the kv append): one body,
/// every plane landed from it.
pub fn lower_planes_chained(
    ctx: &Ctx<'_>,
    planes: &[Tensor],
    mut body: impl FnMut(&mut Lower<'_, '_>) -> Result<Vec<Option<Val>>, Error>,
) -> Result<(), Error> {
    lower(ctx, |l| {
        for (plane, v) in planes.iter().zip(body(l)?) {
            if let Some(v) = v {
                l.land(*plane, v)?;
            }
        }
        Ok(())
    })
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
    use crate::emit::Ctx;
    use crate::handle::{Bank, Tensor};
    use kernels_xla::Error;
    use kernels_xla::linear::{decode, kquant, nvfp4, quant};

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

    /// [`quant`] (not the lm head) with `sums` the width of the heads a norm
    /// of `y` takes next: where the kernel can, it lands the heads' sums of
    /// squares beside the rows, kept for that norm.
    pub fn quant_sums(
        ctx: &Ctx<'_>,
        act: Tensor,
        w: Bank,
        y: Tensor,
        sums: Option<i64>,
    ) -> Result<(), Error> {
        lower(ctx, |l| {
            let a = l.read(act)?;
            let w = l.bank(quant::bank_view(w))?;
            let tuning = l.tuning();
            match quant::matmul_sums(l.f(), a, w, y.landing(), sums, tuning)? {
                Some((v, Some(s))) => l.land_sums(y, v, s),
                Some((v, None)) => l.land(y, v),
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
