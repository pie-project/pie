//! 3-D convolution over the voxels of each clip's lane.

use crate::conv::{
    Conv3d, TimePad, add_k, all, boxes, cmp_k, exclusive, k, lane_pair, mul_k, nonzero, place,
    ravel, spread, take1, unit_where_invalid, within, zero_invalid,
};
use crate::error::{Error, refuse};
use crate::hlo::{Cmp, Elem, Func, Ty, Val};
use crate::tensor::{Landing, Operand, Stored};
use dtype::Dtype;

fn conv_checks(
    op: &'static str,
    x: impl Stored,
    w: impl Stored,
    bias: Option<impl Stored>,
    conv: &Conv3d,
    cache: Option<impl Stored>,
    y: impl Stored,
) -> Result<(), Error> {
    if x.dtype() != Dtype::Bf16 {
        return Err(Error::DtypeUnsupported {
            op,
            dtype: x.dtype(),
        });
    }
    if w.dtype() != Dtype::Bf16 || y.dtype() != Dtype::Bf16 {
        return Err(refuse(op, "the weight and the landing are bf16"));
    }
    nonzero(op, "the input channels", x.width())?;
    nonzero(op, "the output channels", y.width())?;
    for (i, a) in ["kt", "kh", "kw"].into_iter().enumerate() {
        nonzero(op, a, conv.k[i])?;
        nonzero(op, ["st", "sh", "sw"][i], conv.stride[i])?;
    }
    let taps = u64::from(conv.taps());
    if w.rows() != y.width() || u64::from(w.width()) != taps * u64::from(x.width()) {
        return Err(refuse(
            op,
            format!(
                "the weight is {}x{}; this convolution reads [{}, {}] = [C_out, kt*kh*kw*C_in]",
                w.rows(),
                w.width(),
                y.width(),
                taps * u64::from(x.width())
            ),
        ));
    }
    if let Some(b) = bias
        && (b.dtype() != Dtype::F32 || b.elements() != u64::from(y.width()))
    {
        return Err(refuse(
            op,
            format!(
                "the bias is {}x{} {:?}; expected {} f32",
                b.rows(),
                b.width(),
                b.dtype(),
                y.width()
            ),
        ));
    }
    if let Some(c) = cache {
        if !conv.causal_t || conv.pad[0] == 0 {
            return Err(refuse(
                op,
                "a frame cache is read only under `causal_t` with a front pad",
            ));
        }
        if c.dtype() != x.dtype() || c.width() != x.width() {
            return Err(refuse(
                op,
                format!(
                    "the cache is {}x{} {:?}; it holds `[frames, C_in]` bf16",
                    c.rows(),
                    c.width(),
                    c.dtype()
                ),
            ));
        }
    }
    Ok(())
}

/// Bias and zeroed dead rows on an f32 accumulator, then the bf16 landing.
/// The accumulator with the bias added and the rows no output lane holds
/// zeroed: the convolution's result.
fn conv_land(
    f: &mut Func,
    acc: Val,
    bias: Option<Operand>,
    valid: Option<Val>,
) -> Result<Val, Error> {
    let d = f.dims(acc).to_vec();
    let mut acc = acc;
    if let Some(b) = bias {
        let bv = b.f32(f);
        let bv = f.reshape(bv, &[d[1]])?;
        let bv = spread(f, bv, &d, &[1])?;
        acc = f.add(acc, bv)?;
    }
    if let Some(valid) = valid {
        acc = zero_invalid(f, acc, valid)?;
    }
    Ok(acc)
}

/// Bytes of gathered taps one step of the implicit GEMM may hold.
const TAP_BUDGET: i64 = 256 << 20;

/// A 3-d convolution over each clip of `grid`, landing clip `l` of `y_grid`
/// (kernels-cuda `conv::conv3d`). Padding: height and
/// width pad with zeros; time pads front by `pad[0]` and back by
/// `pad_back[0]` (none when `causal_t`) with zeros, or with the edge frame
/// under `TimePad::Replicate` (never past the end when causal); under
/// `causal_t` a `cache` of `pad[0]` frames per clip (clip `l`'s at row
/// `pad[0] * Σ_{j<l} h_j*w_j`) stands in front of frame 0. f32 accumulation,
/// f32 bias, one bf16 rounding; a row no output lane holds lands zero.
///
/// The table is read as data (see the module note); [`conv3d_boxed`] is the
/// `stablehlo.convolution` form for a table the host knows.
pub fn conv3d(
    f: &mut Func,
    x: Operand,
    grid: Operand,
    w: Operand,
    bias: Option<Operand>,
    conv: Conv3d,
    cache: Option<Operand>,
    y: Landing,
    y_grid: Operand,
    bases: [u32; 2],
) -> Result<Val, Error> {
    let grid = crate::conv::rebased(f, grid, bases[0])?;
    let y_grid = crate::conv::rebased(f, y_grid, bases[1])?;
    const OP: &str = "conv.conv3d";
    lane_pair(OP, grid, y_grid)?;
    conv_checks(OP, x, w, bias, &conv, cache, y)?;
    let taps = conv.taps() as usize;
    let c_in = i64::from(x.width);
    let rows = i64::from(y.rows);
    let per = (TAP_BUDGET / (rows.max(1) * c_in * 2)).clamp(1, taps as i64) as usize;
    let og = y_grid.val;
    let o = place(f, og, rows)?;
    let ig = grid.val;
    let ib = boxes(f, ig)?;
    let cache_base = {
        let planes = ib.plane(f)?;
        let before = exclusive(f, planes)?;
        let base = mul_k(f, before, i64::from(conv.pad[0]))?;
        take1(f, base, o.lane)?
    };
    let inb = ib.at(f, o.lane)?;
    let inb = unit_where_invalid(f, inb, o.valid)?;

    // One table: the plane, the cache behind it, a zero row last.
    let xv = x.val;
    let mut parts = vec![xv];
    let cache_at = i64::from(x.rows);
    if let Some(c) = cache {
        parts.push(c.val);
    }
    let dead = cache_at + cache.map_or(0, |c| i64::from(c.rows));
    let zero = f.const_i(Elem::Bf16, 0, &[1, c_in]);
    parts.push(zero);
    let table = f.concat(&parts, 0)?;

    let [_, kh, kw] = conv.k.map(i64::from);
    let [st, sh, sw] = conv.stride.map(i64::from);
    let [pt, ph, pw] = conv.pad.map(i64::from);
    let replicate = conv.time_pad == TimePad::Replicate;
    let mut index = Vec::with_capacity(taps);
    for tap in 0..taps as i64 {
        let (it, rest) = (tap / (kh * kw), tap % (kh * kw));
        let (ih, iw) = (rest / kw, rest % kw);
        let hi = mul_k(f, o.v.h, sh)?;
        let hi = add_k(f, hi, ih - ph)?;
        let wi = mul_k(f, o.v.w, sw)?;
        let wi = add_k(f, wi, iw - pw)?;
        let h_ok = within(f, hi, inb.h)?;
        let w_ok = within(f, wi, inb.w)?;
        let ti = mul_k(f, o.v.t, st)?;
        let ti = add_k(f, ti, it - pt)?;
        let before = cmp_k(f, Cmp::Lt, ti, 0)?;
        let after = f.compare(Cmp::Ge, ti, inb.t)?;
        let mut ok = all(f, &[o.valid, h_ok, w_ok])?;
        // Frames before the clip.
        let mut from_cache = None;
        let ti = if conv.causal_t && cache.is_some() {
            let front = add_k(f, ti, pt)?;
            let r = f.mul(front, inb.h)?;
            let r = f.add(r, hi)?;
            let r = f.mul(r, inb.w)?;
            let r = f.add(r, wi)?;
            let r = f.add(r, cache_base)?;
            let r = add_k(f, r, cache_at)?;
            from_cache = Some((before, r));
            ti
        } else if replicate {
            let z = k(f, ti, 0);
            f.select(before, z, ti)?
        } else {
            let keep = f.not(before);
            ok = f.and(ok, keep)?;
            ti
        };
        // Frames after it.
        let ti = if conv.causal_t || !replicate {
            let keep = f.not(after);
            ok = f.and(ok, keep)?;
            ti
        } else {
            let last = add_k(f, inb.t, -1)?;
            f.select(after, last, ti)?
        };
        let zero = k(f, ti, 0);
        let ti_safe = f.max(ti, zero)?;
        let hi_safe = f.max(hi, zero)?;
        let wi_safe = f.max(wi, zero)?;
        let mut src = ravel(f, &inb, ti_safe, hi_safe, wi_safe)?;
        if let Some((before, r)) = from_cache {
            src = f.select(before, r, src)?;
        }
        let far = k(f, src, dead);
        let src = f.select(ok, src, far)?;
        index.push(f.reshape(src, &[rows, 1])?);
    }
    let index = f.concat(&index, 1)?;
    let wv = w.val;
    let c_out = i64::from(y.width);
    let mut acc = f.const_f(Elem::F32, 0.0, &[rows, c_out]);
    let mut t0 = 0usize;
    while t0 < taps {
        let n = per.min(taps - t0) as i64;
        let idx = f.slice_axis(index, 1, t0 as i64, t0 as i64 + n)?;
        let idx = f.reshape(idx, &[rows * n])?;
        let a = f.take_rows(table, idx)?;
        let a = f.reshape(a, &[rows, n * c_in])?;
        let wt = f.slice_axis(wv, 1, t0 as i64 * c_in, (t0 as i64 + n) * c_in)?;
        let part = f.matmul_nt(a, wt, Elem::F32)?;
        acc = f.add(acc, part)?;
        t0 += n as usize;
    }
    conv_land(f, acc, bias, Some(o.valid))
}

/// [`conv3d`] over clip boxes the host knows: `clips` is the input lane
/// table's contents (`{t, h, w, row_offset}` per lane), `y_clips` the output
/// one. Each clip runs as one `stablehlo.convolution` of a `[1, t, h, w,
/// C_in]` volume (time padding and the cache concatenated in front first),
/// so the boxes are static attributes: an executable built here is keyed by
/// them. Rows outside every output box land zero.
pub fn conv3d_boxed(
    f: &mut Func,
    x: Operand,
    clips: &[[u32; 4]],
    w: Operand,
    bias: Option<Operand>,
    conv: Conv3d,
    cache: Option<Operand>,
    y: Landing,
    y_clips: &[[u32; 4]],
) -> Result<Val, Error> {
    const OP: &str = "conv.conv3d";
    conv_checks(OP, x, w, bias, &conv, cache, y)?;
    if clips.len() != y_clips.len() || clips.is_empty() {
        return Err(refuse(
            OP,
            format!(
                "{} input boxes against {} output boxes",
                clips.len(),
                y_clips.len()
            ),
        ));
    }
    let pt = u64::from(conv.pad[0]);
    let mut cache_at = 0u64;
    let mut plan = Vec::with_capacity(clips.len());
    for (l, (c, o)) in clips.iter().zip(y_clips).enumerate() {
        let [t, h, wd, off] = *c;
        let plane = u64::from(h) * u64::from(wd);
        let base = cache_at;
        cache_at += pt * plane;
        let want = conv.out_extent([t, h, wd]);
        let vox = plane * u64::from(t);
        if vox == 0 || want.is_none() {
            continue;
        }
        let want = want.unwrap_or_default();
        if [o[0], o[1], o[2]] != want {
            return Err(refuse(
                OP,
                format!(
                    "clip {l}'s box {:?} convolves to {want:?}, and its output box is {:?}",
                    &c[..3],
                    &o[..3]
                ),
            ));
        }
        let o_vox = want.iter().map(|&n| u64::from(n)).product::<u64>();
        if u64::from(off) + vox > u64::from(x.rows) || u64::from(o[3]) + o_vox > u64::from(y.rows) {
            return Err(refuse(OP, format!("clip {l}'s rows run past the plane")));
        }
        if let Some(cc) = cache
            && conv.causal_t
            && base + pt * plane > u64::from(cc.rows)
        {
            return Err(refuse(
                OP,
                format!("clip {l}'s cached frames run past the cache"),
            ));
        }
        plan.push((*c, *o, base));
    }
    let c_in = i64::from(x.width);
    let c_out = i64::from(y.width);
    let [kt, kh, kw] = conv.k.map(i64::from);
    let wv = w.val;
    let kernel = f.reshape(wv, &[c_out, kt, kh, kw, c_in])?;
    let xv = x.val;
    let cv = cache.map(|c| c.val);
    let mut out = f.const_f(Elem::F32, 0.0, &[i64::from(y.rows), c_out]);
    for &([t, h, wd, off], [ot, oh, ow, o_off], base) in &plan {
        let (t, h, wd) = (i64::from(t), i64::from(h), i64::from(wd));
        let plane = h * wd;
        let clip = f.slice_axis(xv, 0, i64::from(off), i64::from(off) + t * plane)?;
        let clip = f.reshape(clip, &[1, t, h, wd, c_in])?;
        let frame = |f: &mut Func, at: i64| -> Result<Val, Error> {
            Ok(f.slice_axis(clip, 1, at, at + 1)?)
        };
        let fill = |f: &mut Func, src: Option<Val>, n: i64| -> Result<Val, Error> {
            Ok(match src {
                Some(fq) => f.broadcast(fq, &[1, n, h, wd, c_in], &[0, 1, 2, 3, 4])?,
                None => f.const_i(Elem::Bf16, 0, &[1, n, h, wd, c_in]),
            })
        };
        let replicate = conv.time_pad == TimePad::Replicate;
        let pt = i64::from(conv.pad[0]);
        let mut vol5 = Vec::new();
        if pt > 0 {
            vol5.push(match (&cv, conv.causal_t) {
                (Some(cv), true) => {
                    let rows = f.slice_axis(*cv, 0, base as i64, base as i64 + pt * plane)?;
                    f.reshape(rows, &[1, pt, h, wd, c_in])?
                }
                _ if replicate => {
                    let f0 = frame(f, 0)?;
                    fill(f, Some(f0), pt)?
                }
                _ => fill(f, None, pt)?,
            });
        }
        vol5.push(clip);
        let back = i64::from(conv.back_t());
        if back > 0 {
            vol5.push(if replicate {
                let fl = frame(f, t - 1)?;
                fill(f, Some(fl), back)?
            } else {
                fill(f, None, back)?
            });
        }
        let vol5 = f.concat(&vol5, 1)?;
        let [_, sh, sw] = conv.stride.map(i64::from);
        let st = i64::from(conv.stride[0]);
        let attrs = format!(
            "window_strides = array<i64: {st}, {sh}, {sw}>, padding = dense<[[0, 0], [{}, {}], [{}, {}]]> : tensor<3x2xi64>, lhs_dilation = array<i64: 1, 1, 1>, rhs_dilation = array<i64: 1, 1, 1>, dimension_numbers = #stablehlo.conv<[b, 0, 1, 2, fq]x[o, 0, 1, 2, i]->[b, 0, 1, 2, fq]>, feature_group_count = 1 : i64, batch_group_count = 1 : i64",
            conv.pad[1], conv.pad_back[1], conv.pad[2], conv.pad_back[2]
        );
        let (ot, oh, ow) = (i64::from(ot), i64::from(oh), i64::from(ow));
        let y5 = f.op_n(
            "convolution",
            &[vol5, kernel],
            &attrs,
            vec![Ty::new(Elem::F32, &[1, ot, oh, ow, c_out])],
        )[0];
        let rows = f.reshape(y5, &[ot * oh * ow, c_out])?;
        let r0 = f.const_i(Elem::I32, i64::from(o_off), &[]);
        let c0 = f.const_i(Elem::I32, 0, &[]);
        out = f.dynamic_update_slice(out, rows, &[r0, c0])?;
    }
    // Rows no box covers stay zero, bias included.
    let valid = if bias.is_some() {
        let row = f.iota(Elem::I32, &[i64::from(y.rows)], 0);
        let mut hit = f.const_i(Elem::Pred, 0, &[i64::from(y.rows)]);
        for &(_, [ot, oh, ow, o_off], _) in &plan {
            let n = i64::from(ot) * i64::from(oh) * i64::from(ow);
            let lo = cmp_k(f, Cmp::Ge, row, i64::from(o_off))?;
            let hi = cmp_k(f, Cmp::Lt, row, i64::from(o_off) + n)?;
            let inside = f.and(lo, hi)?;
            hit = f.or(hit, inside)?;
        }
        Some(hit)
    } else {
        None
    };
    conv_land(f, out, bias, valid)
}
