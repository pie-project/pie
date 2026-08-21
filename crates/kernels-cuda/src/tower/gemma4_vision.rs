
use std::ffi::c_void;

use crate::jit::abi::bf16;
use crate::{norm, vision};
use kernels::routine::{Const, In, InOut, Out};

use super::{Refused, Result, Scratch, Stream, call, read_raw_span};

const WHO: &str = "gemma4_vision";

const PATCH_DIM: usize = 3 * 16 * 16;

#[derive(Clone, Copy, Debug)]
pub struct Clip {

    pub w: *const c_void,
    pub imin: *const c_void,
    pub imax: *const c_void,
    pub omin: *const c_void,
    pub omax: *const c_void,
}

impl Clip {

    fn of(t: &[*const c_void]) -> Self {
        Self {
            w: t[0],
            imin: t[1],
            imax: t[2],
            omin: t[3],
            omax: t[4],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Layer {

    pub in_ln: *const c_void,
    pub post_attn_ln: *const c_void,
    pub pre_ff_ln: *const c_void,
    pub post_ff_ln: *const c_void,
    pub q_norm: *const c_void,
    pub k_norm: *const c_void,
    pub q: Clip,
    pub k: Clip,
    pub v: Clip,
    pub o: Clip,
    pub gate: Clip,
    pub up: Clip,
    pub down: Clip,
}

#[derive(Clone, Debug)]
pub struct Weights {

    pub patch_w: *const c_void,
    pub pos_table: *const c_void,
    pub embed_proj: *const c_void,
    pub layers: Vec<Layer>,
    pub hidden: i32,
    pub heads: i32,
    pub intermediate: i32,
    pub pos_table_size: i32,
    pub text_hidden: i32,
    pub pool_kernel: i32,
    pub eps: f32,
    pub theta: f32,
}

impl Weights {

    pub fn from_flat(
        patch_w: *const c_void,
        pos_table: *const c_void,
        embed_proj: *const c_void,
        layer_w: &[*const c_void],
        depth: usize,
        slots_per_layer: usize,
        hidden: i32,
        heads: i32,
        intermediate: i32,
        pos_table_size: i32,
        text_hidden: i32,
        pool_kernel: i32,
        eps: f32,
        theta: f32,
    ) -> Result<Self> {
        let want = depth
            .checked_mul(slots_per_layer)
            .ok_or_else(|| Refused::new(WHO, "layer table length overflowed"))?;
        if layer_w.len() < want {
            return Err(Refused::new(
                WHO,
                format!(
                    "layer table holds {} pointers for {depth} layers of \
                     {slots_per_layer}, which needs {want}",
                    layer_w.len()
                ),
            ));
        }
        let mut layers = Vec::with_capacity(depth);
        for i in 0..depth {
            let t = &layer_w[i * slots_per_layer..(i + 1) * slots_per_layer];
            layers.push(Layer {
                in_ln: t[0],
                post_attn_ln: t[1],
                pre_ff_ln: t[2],
                post_ff_ln: t[3],
                q_norm: t[4],
                k_norm: t[5],
                q: Clip::of(&t[6..]),
                k: Clip::of(&t[11..]),
                v: Clip::of(&t[16..]),
                o: Clip::of(&t[21..]),
                gate: Clip::of(&t[26..]),
                up: Clip::of(&t[31..]),
                down: Clip::of(&t[36..]),
            });
        }
        Ok(Self {
            patch_w,
            pos_table,
            embed_proj,
            layers,
            hidden,
            heads,
            intermediate,
            pos_table_size,
            text_hidden,
            pool_kernel,
            eps,
            theta,
        })
    }
}

fn rms(
    x: *const c_void,
    weight: *const c_void,
    y: *mut c_void,
    rows: i32,
    hidden: i32,
    eps: f32,
    stream: Stream<'_>,
) -> Result<()> {
    call("norm::rmsnorm_strided_bf16", stream, |ctx| {
        norm::rmsnorm_strided_bf16_at(
            ctx,
            In { ptr: x.cast(), rows, width: hidden },
            Const { v: weight.cast() },
            Out { ptr: y.cast(), rows, width: hidden },
            eps,
        )
    })
}

fn extent(what: &'static str, value: i32) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Refused::new(WHO, format!("{what}: {value} is not an extent")))
}

#[allow(clippy::too_many_arguments)]
fn run(
    w: &Weights,
    pixel: *const c_void,
    pos: *mut c_void,
    grp: *mut c_void,
    n: i32,
    out_len: i32,
    out_proj: *mut c_void,
    cublas: *mut c_void,
    stream: Stream<'_>,
) -> Result<()> {
    let (hd, nh, im) = (w.hidden, w.heads, w.intermediate);
    let (txt, pt) = (w.text_hidden, w.pos_table_size);
    let (eps, theta) = (w.eps, w.theta);

    if hd != 768 || nh != 12 {
        return Err(Refused::new(
            WHO,
            format!("unexpected dims (expected hidden=768, heads=12, got hidden={hd}, heads={nh})"),
        ));
    }
    let n_u = extent("patches", n)?;
    let hd_u = extent("hidden", hd)?;

    extent("intermediate", im)?;

    extent("pooled rows", out_len)?;
    let nz = usize::try_from(n).unwrap_or(0);
    let hidden_elems = nz * hd as usize;
    let inter_elems = nz * im as usize;

    let mut scratch = Scratch::new();
    let h = scratch.bf16(hidden_elems)?;
    let hn = scratch.bf16(hidden_elems)?;
    let xc = scratch.bf16(inter_elems)?;
    let q = scratch.bf16(hidden_elems)?;
    let k = scratch.bf16(hidden_elems)?;
    let v = scratch.bf16(hidden_elems)?;
    let attn = scratch.bf16(hidden_elems)?;
    let gate = scratch.bf16(inter_elems)?;
    let up = scratch.bf16(inter_elems)?;
    let act = scratch.bf16(inter_elems)?;
    let tmp = scratch.bf16(hidden_elems)?;
    let scr = scratch.f32s(nz * nz)?;

    call("vision::k_scale_bf16", stream, |ctx| {
        vision::k_scale_bf16(ctx, pixel, hn, hidden_elems)
    })?;

    gemm(cublas, hn.cast_const(), w.patch_w, h, n, hd, hd);

    call("vision::k_addpos_grid2d_bf16", stream, |ctx| {
        vision::k_addpos_grid2d_bf16(ctx, h, w.pos_table, pos.cast_const(), n, hd, pt)
    })?;

    for layer in &w.layers {
        rms(h.cast_const(), layer.in_ln, hn, n, hd, eps, stream)?;
        clin(cublas, hn.cast_const(), q, xc, &layer.q, n, hd, hd, stream)?;
        clin(cublas, hn.cast_const(), k, xc, &layer.k, n, hd, hd, stream)?;
        clin(cublas, hn.cast_const(), v, xc, &layer.v, n, hd, hd, stream)?;

        let head_rows = n
            .checked_mul(nh)
            .ok_or_else(|| Refused::new(WHO, "N * NH overflowed"))?;
        let head_dim = hd / nh;
        rms(
            q.cast_const(),
            layer.q_norm,
            q,
            head_rows,
            head_dim,
            eps,
            stream,
        )?;
        rms(
            k.cast_const(),
            layer.k_norm,
            k,
            head_rows,
            head_dim,
            eps,
            stream,
        )?;

        call("norm::rmsnorm_no_scale_bf16", stream, |ctx| {
            norm::rmsnorm_no_scale_at(
                ctx,
                In { ptr: v.cast_const().cast(), rows: head_rows, width: head_dim },
                Out { ptr: v.cast(), rows: head_rows, width: head_dim },
                0,
                eps,
            )
        })?;

        for tensor in [q, k] {
            call("vision::k_rope_axial2d_bf16", stream, |ctx| {
                vision::k_rope_axial2d_bf16(ctx, tensor, pos.cast_const(), n, nh, theta)
            })?;
        }

        for head in 0..nh {
            call("vision::k_qk_bf16", stream, |ctx| {
                vision::k_qk_bf16(ctx, q.cast_const(), k.cast_const(), scr, n, nh, head, 1.0)
            })?;
            call("vision::k_softmax_bf16", stream, |ctx| {
                vision::k_softmax_bf16(ctx, scr, n)
            })?;
            call("vision::k_av_bf16", stream, |ctx| {
                vision::k_av_bf16(ctx, scr.cast_const(), v.cast_const(), attn, n, nh, head)
            })?;
        }
        clin(
            cublas,
            attn.cast_const(),
            tmp,
            xc,
            &layer.o,
            n,
            hd,
            hd,
            stream,
        )?;
        rms(
            tmp.cast_const(),
            layer.post_attn_ln,
            tmp,
            n,
            hd,
            eps,
            stream,
        )?;
        residual_add(h, tmp.cast_const(), hidden_elems, n_u, hd_u, stream)?;
        rms(h.cast_const(), layer.pre_ff_ln, hn, n, hd, eps, stream)?;
        clin(
            cublas,
            hn.cast_const(),
            gate,
            xc,
            &layer.gate,
            n,
            hd,
            im,
            stream,
        )?;
        clin(
            cublas,
            hn.cast_const(),
            up,
            xc,
            &layer.up,
            n,
            hd,
            im,
            stream,
        )?;

        let ctx = unsafe { crate::jit::Ctx::on(stream.as_raw()) };
        crate::mlp::geglu_tanh::<bf16>(
            &ctx,
            In { ptr: gate.cast_const().cast(), rows: n, width: im },
            In { ptr: up.cast_const().cast(), rows: n, width: im },
            Out { ptr: act.cast(), rows: n, width: im },
        )
        .map_err(|why| Refused::new("mlp::geglu_tanh_bf16", format!("{why:?}")))?;
        clin(
            cublas,
            act.cast_const(),
            tmp,
            xc,
            &layer.down,
            n,
            im,
            hd,
            stream,
        )?;
        rms(tmp.cast_const(), layer.post_ff_ln, tmp, n, hd, eps, stream)?;
        residual_add(h, tmp.cast_const(), hidden_elems, n_u, hd_u, stream)?;
    }

    let pooled_elems = usize::try_from(out_len).unwrap_or(0) * hd as usize;

    let pf = scratch.zeroed_f32s(pooled_elems, stream)?;

    call("vision::k_pool_bf16", stream, |ctx| {
        vision::k_pool_bf16(ctx, h.cast_const(), grp.cast_const(), pf, n, hd, 9.0)
    })?;
    let pooled = scratch.bf16(pooled_elems)?;

    #[allow(clippy::cast_precision_loss)]
    let scale = (hd as f32).sqrt();
    call("vision::k_pool_finish_bf16", stream, |ctx| {
        vision::k_pool_finish_bf16(ctx, pf.cast_const(), pooled, scale, pooled_elems)
    })?;
    let pn = scratch.bf16(pooled_elems)?;

    call("norm::rmsnorm_no_scale_bf16", stream, |ctx| {
        norm::rmsnorm_no_scale_at(
            ctx,
            In { ptr: pooled.cast_const().cast(), rows: out_len, width: hd },
            Out { ptr: pn.cast(), rows: out_len, width: hd },
            0,
            eps,
        )
    })?;

    gemm(
        cublas,
        pn.cast_const(),
        w.embed_proj,
        out_proj,
        out_len,
        txt,
        hd,
    );

    stream.synchronize()?;
    drop(scratch);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn clin(
    cublas: *mut c_void,
    x: *const c_void,
    out: *mut c_void,
    xc: *mut c_void,
    c: &Clip,
    n: i32,
    k_in: i32,
    out_width: i32,
    stream: Stream<'_>,
) -> Result<()> {

    extent("clin rows", n)?;
    extent("clin in", k_in)?;
    extent("clin out", out_width)?;
    let in_elems = usize::try_from(n).unwrap_or(0) * k_in as usize;
    let out_elems = usize::try_from(n).unwrap_or(0) * out_width as usize;
    call("vision::k_clamp_bf16", stream, |ctx| {
        vision::k_clamp_bf16(ctx, x, xc, c.imin, c.imax, in_elems)
    })?;
    gemm(cublas, xc.cast_const(), c.w, out, n, out_width, k_in);
    call("vision::k_clamp_bf16", stream, |ctx| {
        vision::k_clamp_bf16(ctx, out.cast_const(), out, c.omin, c.omax, out_elems)
    })
}

fn gemm(
    handle: *mut c_void,
    act: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
) {

    unsafe {
        crate::gemm::dense::act_x_wt_bf16(handle, act, w, y, m, n, k, 0.0);
    }
}

fn residual_add(
    y: *mut c_void,
    x: *const c_void,
    elems: usize,
    rows: u32,
    width: u32,
    stream: Stream<'_>,
) -> Result<()> {
    if elems == 0 {
        return Ok(());
    }

    let rows = i32::try_from(rows)
        .map_err(|_| Refused::new(WHO, "the residual rectangle's row count exceeds i32"))?;
    let width = i32::try_from(width)
        .map_err(|_| Refused::new(WHO, "the residual rectangle's row width exceeds i32"))?;
    call("norm::residual_add_bf16", stream, |ctx| {
        norm::residual_add::<bf16>(
            ctx,
            InOut { ptr: y.cast(), rows, width },
            In { ptr: x.cast(), rows, width },
        )
    })
}

pub fn encode(
    w: &Weights,
    pixels: &[u8],
    pixel_byte_indptr: &[u32],
    patch_positions: &[u32],
    output_rows: &mut [u8],
    output_row_indptr: &mut [u32],
    cublas: *mut c_void,
    stream: Stream<'_>,
) -> Result<()> {
    let num_images = output_row_indptr.len().saturating_sub(1);
    if num_images == 0 || pixel_byte_indptr.len() < num_images + 1 {
        return Err(Refused::new(WHO, "invalid standalone encode inputs"));
    }
    let pk = usize::try_from(w.pool_kernel)
        .map_err(|_| Refused::new(WHO, "pool_kernel is negative"))?;
    let pk2 = pk
        .checked_mul(pk)
        .filter(|v| *v != 0)
        .ok_or_else(|| Refused::new(WHO, "pool_kernel is zero"))?;
    let row_bytes = usize::try_from(w.text_hidden).unwrap_or(0) * 2;
    let mut rows_written = 0usize;
    let mut patch_off = 0usize;
    output_row_indptr[0] = 0;
    for image in 0..num_images {
        let blo = pixel_byte_indptr[image] as usize;
        let bhi = pixel_byte_indptr[image + 1] as usize;
        if bhi < blo || bhi > pixels.len() {
            return Err(Refused::new(
                WHO,
                format!("image {image}'s pixel span [{blo}, {bhi}) leaves the payload"),
            ));
        }
        let n_floats = (bhi - blo) / 4;
        let n_patch = n_floats / PATCH_DIM;
        if n_patch == 0 || !n_patch.is_multiple_of(pk2) {
            return Err(Refused::new(
                WHO,
                format!("invalid patch count ({n_patch} for a {pk}x{pk} pool)"),
            ));
        }
        let out_len = n_patch / pk2;
        let want = rows_written
            .checked_add(out_len)
            .and_then(|r| r.checked_mul(row_bytes))
            .ok_or_else(|| Refused::new(WHO, "output row count overflowed"))?;
        if want > output_rows.len() {
            return Err(Refused::new(WHO, "encode output buffer too small"));
        }
        if patch_positions.len() < (patch_off + n_patch) * 2 {
            return Err(Refused::new(
                WHO,
                "patch position table is shorter than the patches it describes",
            ));
        }
        let pos_h = &patch_positions[patch_off * 2..(patch_off + n_patch) * 2];

        let mut scratch = Scratch::new();
        let pix_f32 = scratch.upload_bytes(&pixels[blo..bhi], stream)?;
        let pix_bf = scratch.bf16(n_floats)?;

        call("vision::k_f32_to_bf16_bf16", stream, |ctx| {
            vision::k_f32_to_bf16_bf16(ctx, pix_f32.cast_const(), pix_bf, n_floats)
        })?;

        let mut posf = vec![0.0f32; n_patch * 2];
        let mut grp = vec![0i32; n_patch];
        let mut maxx = 0u32;
        for patch in 0..n_patch {
            maxx = maxx.max(pos_h[2 * patch]);
        }
        let gx = i32::try_from((maxx as usize + 1) / pk)
            .map_err(|_| Refused::new(WHO, "pooling grid width overflowed"))?;
        for patch in 0..n_patch {
            let (px, py) = (pos_h[2 * patch], pos_h[2 * patch + 1]);
            #[allow(clippy::cast_precision_loss)]
            {
                posf[2 * patch] = px as f32;
                posf[2 * patch + 1] = py as f32;
            }
            let (cx, cy) = (
                i32::try_from(px as usize / pk).unwrap_or(0),
                i32::try_from(py as usize / pk).unwrap_or(0),
            );
            grp[patch] = cx + gx * cy;
        }
        let pos_d = scratch.upload_f32s(&posf, stream)?;
        let grp_d = scratch.upload_i32s(&grp, stream)?;
        let proj_d = scratch.bf16(out_len * usize::try_from(w.text_hidden).unwrap_or(0))?;
        run(
            w,
            pix_bf.cast_const(),
            pos_d,
            grp_d,
            i32::try_from(n_patch)
                .map_err(|_| Refused::new(WHO, "patch count overflowed an int"))?,
            i32::try_from(out_len)
                .map_err(|_| Refused::new(WHO, "pooled row count overflowed"))?,
            proj_d,
            cublas,
            stream,
        )?;

        let begin = rows_written * row_bytes;
        let end = begin + out_len * row_bytes;

        unsafe { read_raw_span(proj_d.cast_const(), &mut output_rows[begin..end], stream)? };
        stream.synchronize()?;
        drop(scratch);
        rows_written += out_len;
        output_row_indptr[image + 1] = u32::try_from(rows_written)
            .map_err(|_| Refused::new(WHO, "encoded row count overflowed u32"))?;
        patch_off += n_patch;
    }
    Ok(())
}
