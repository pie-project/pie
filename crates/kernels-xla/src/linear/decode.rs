//! A split-plane bank decoded to a dense bf16 plane, for the ops that read a
//! weight as a plain array rather than contracting it (MLA's absorbed
//! `kv_b`). Reference: kernels-cuda `linear::quant::decoded_plane` with
//! `OffsetKind::Post` into bf16, which engine-cuda's `dense_or_decoded`
//! stages for `attention.mla_absorb_q/out`: `w = s · c + b` (affine) or
//! `w = e2m1(c) · 2^(e − 127)` (mxfp4), each weight rounded once to bf16.

use dtype::Dtype;

use crate::cx::{Ctx, Cx};
use crate::error::{Error, refuse};
use crate::hlo::{Elem, Val};
use crate::linear::quant::{Codes, Form, code_values, e8m0, planes, read_codes};
use crate::tensor::{Bank, Tensor};

/// `out = decode(w)`, bf16 `[n, k]`.
pub fn decoded_plane(ctx: &Ctx<'_>, op: &'static str, w: Bank, out: Tensor) -> Result<(), Error> {
    if out.dtype != Dtype::Bf16 {
        return Err(Error::DtypeUnsupported { op, dtype: out.dtype });
    }
    let (n, k) = (out.rows, out.width);
    let p = planes(op, &w, n, k)?;
    if n == 0 || k == 0 {
        return Err(refuse(op, "an empty plane decodes to nothing"));
    }
    ctx.emit(&mut |cx| {
        let v = decode(cx, &p, i64::from(n), i64::from(k))?;
        cx.write(out, v)
    })
}

/// The bank's weights, f32 `[n, k]`.
pub(crate) fn decode(
    cx: &mut Cx<'_>,
    p: &crate::linear::quant::Planes,
    n: i64,
    k: i64,
) -> Result<Val, Error> {
    let g = i64::from(p.groups);
    let mxfp4 = matches!(p.form, Form::Mxfp4);
    let codes = read_codes(cx, p.codes, p.bits, mxfp4, n, k)?;
    if let Codes::Weights(_) = codes {
        return code_values(cx, codes, Elem::F32);
    }
    let c = code_values(cx, codes, Elem::F32)?;
    let c = cx.reshape(c, &[n, g, k / g])?;
    let d = [n, g, k / g];
    let w = match p.form {
        Form::Affine { biases } => {
            let s = cx.read_f32(p.scales)?;
            let b = cx.read_f32(biases)?;
            let s = cx.broadcast(s, &d, &[0, 1])?;
            let b = cx.broadcast(b, &d, &[0, 1])?;
            let w = cx.mul(c, s)?;
            cx.add(w, b)?
        }
        Form::Mxfp4 => {
            let e = cx.read(p.scales)?;
            let e = cx.convert(e, Elem::U32);
            let s = e8m0(cx, e)?;
            let s = cx.broadcast(s, &d, &[0, 1])?;
            cx.mul(c, s)?
        }
    };
    Ok(cx.reshape(w, &[n, k])?)
}
