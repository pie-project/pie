
use std::ffi::c_void;

use crate::jit::abi::bf16;
use crate::{ssm, vision};

use super::{Refused, Result, Scratch, Stream, call, read_raw_span};

const WHO: &str = "gemma4_audio";

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
pub struct Ffn {

    pub pre_ln: *const c_void,
    pub post_ln: *const c_void,
    pub fc1: Clip,
    pub fc2: Clip,
}

impl Ffn {

    fn of(t: &[*const c_void]) -> Self {
        Self {
            pre_ln: t[0],
            post_ln: t[1],
            fc1: Clip::of(&t[2..]),
            fc2: Clip::of(&t[7..]),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Layer {

    pub ff1: Ffn,
    pub ff2: Ffn,
    pub norm_pre_attn: *const c_void,
    pub norm_post_attn: *const c_void,
    pub q: Clip,
    pub k: Clip,
    pub v: Clip,
    pub post: Clip,
    pub relative_k: *const c_void,
    pub per_dim_scale: *const c_void,
    pub lconv_pre_ln: *const c_void,
    pub lconv_conv_norm: *const c_void,
    pub lconv_start: Clip,
    pub lconv_end: Clip,
    pub depthwise_conv: *const c_void,
    pub norm_out: *const c_void,
}

#[derive(Clone, Debug)]
pub struct Weights {

    pub sscp0_conv: *const c_void,
    pub sscp0_norm: *const c_void,
    pub sscp1_conv: *const c_void,
    pub sscp1_norm: *const c_void,
    pub sscp_input_proj: *const c_void,
    pub layers: Vec<Layer>,
    pub output_proj_w: *const c_void,
    pub output_proj_b: *const c_void,
    pub embed_proj: *const c_void,
    pub hidden: i32,
    pub heads: i32,
    pub conv_kernel: i32,
    pub n_mel: i32,
    pub sscp_ch0: i32,
    pub sscp_ch1: i32,
    pub out_proj_dims: i32,
    pub text_hidden: i32,
    pub chunk_size: i32,
    pub context_left: i32,
    pub context_right: i32,
    pub logit_cap: f32,
    pub residual_weight: f32,
    pub eps: f32,
}

impl Weights {

    #[allow(clippy::too_many_arguments)]
    pub fn from_flat(
        head: [*const c_void; 8],
        layer_w: &[*const c_void],
        depth: usize,
        slots_per_layer: usize,
        dims: [i32; 9],
        logit_cap: f32,
        residual_weight: f32,
        eps: f32,
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
                ff1: Ffn::of(t),
                ff2: Ffn::of(&t[12..]),
                norm_pre_attn: t[24],
                norm_post_attn: t[25],
                q: Clip::of(&t[26..]),
                k: Clip::of(&t[31..]),
                v: Clip::of(&t[36..]),
                post: Clip::of(&t[41..]),
                relative_k: t[46],
                per_dim_scale: t[47],
                lconv_pre_ln: t[48],
                lconv_conv_norm: t[49],
                lconv_start: Clip::of(&t[50..]),
                lconv_end: Clip::of(&t[55..]),
                depthwise_conv: t[60],
                norm_out: t[61],
            });
        }
        let [
            sscp0_conv,
            sscp0_norm,
            sscp1_conv,
            sscp1_norm,
            sscp_input_proj,
            output_proj_w,
            output_proj_b,
            embed_proj,
        ] = head;
        let [
            hidden,
            heads,
            conv_kernel,
            n_mel,
            sscp_ch0,
            sscp_ch1,
            out_proj_dims,
            text_hidden,
            chunk_size,
        ] = dims;
        Ok(Self {
            sscp0_conv,
            sscp0_norm,
            sscp1_conv,
            sscp1_norm,
            sscp_input_proj,
            layers,
            output_proj_w,
            output_proj_b,
            embed_proj,
            hidden,
            heads,
            conv_kernel,
            n_mel,
            sscp_ch0,
            sscp_ch1,
            out_proj_dims,
            text_hidden,
            chunk_size,
            context_left: 13,
            context_right: 0,
            logit_cap,
            residual_weight,
            eps,
        })
    }

    #[must_use]
    pub fn with_context(mut self, left: i32, right: i32) -> Self {
        self.context_left = left;
        self.context_right = right;
        self
    }
}

struct Walk {

    hd: i32,
    nh: i32,
    head_dim: i32,
    im: i32,
    n: i32,
    t1: i32,
    f1: i32,
    f2: i32,
    q_scale: f32,
    k_scale: f32,
    pp: i32,
}

impl Walk {

    fn new(w: &Weights, n_frames: i32, n_mel: i32, out_len: i32) -> Result<Self> {
        let (hd, nh) = (w.hidden, w.heads);
        if hd != 1024 || nh != 8 {
            return Err(Refused::new(
                WHO,
                format!("unexpected dims (expected hidden=1024, heads=8, got {hd}/{nh})"),
            ));
        }
        let head_dim = hd / nh;

        let cdim = |n: i32| (n - 1) / 2 + 1;
        if n_frames <= 0 || n_mel <= 0 {
            return Err(Refused::new(
                WHO,
                format!("invalid feature shape ({n_frames} frames of {n_mel})"),
            ));
        }
        let (t1, f1) = (cdim(n_frames), cdim(n_mel));
        let (t2, f2) = (cdim(t1), cdim(f1));
        if t2 != out_len {
            return Err(Refused::new(
                WHO,
                format!("out_len != subsampled frames ({out_len} vs {t2})"),
            ));
        }
        #[allow(clippy::cast_precision_loss)]
        let q_scale = (head_dim as f32).powf(-0.5) / core::f32::consts::LN_2;
        let k_scale = (1.0f32 + core::f32::consts::E).ln() / core::f32::consts::LN_2;
        Ok(Self {
            hd,
            nh,
            head_dim,
            im: 4 * hd,
            n: t2,
            t1,
            f1,
            f2,
            q_scale,
            k_scale,
            pp: w.context_left,
        })
    }
}

fn elems(what: &str, rows: i32, width: i32) -> Result<usize> {
    let r = usize::try_from(rows).map_err(|_| Refused::new(WHO, format!("{what}: rows")))?;
    let c = usize::try_from(width).map_err(|_| Refused::new(WHO, format!("{what}: width")))?;
    r.checked_mul(c)
        .ok_or_else(|| Refused::new(WHO, format!("{what}: element count overflowed")))
}

fn rms(
    x: *const c_void,
    weight: *const c_void,
    o: *mut c_void,
    rows: i32,
    width: i32,
    eps: f32,
    stream: Stream<'_>,
) -> Result<()> {
    call("vision::k_rms_bf16", stream, |ctx| {
        vision::k_rms_bf16(ctx, x, weight, o, rows, width, eps)
    })
}

fn matmul(
    x: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    n: i32,
    kin: i32,
    out: i32,
    stream: Stream<'_>,
) -> Result<()> {
    call("vision::k_matmul_bf16", stream, |ctx| {
        vision::k_matmul_bf16(ctx, x, w, y, n, kin, out)
    })
}

fn clamp(
    x: *const c_void,
    o: *mut c_void,
    lo: *const c_void,
    hi: *const c_void,
    n: usize,
    stream: Stream<'_>,
) -> Result<()> {
    call("vision::k_clamp_bf16", stream, |ctx| {
        vision::k_clamp_bf16(ctx, x, o, lo, hi, n)
    })
}

#[allow(clippy::too_many_arguments)]
fn clin(
    x: *const c_void,
    out: *mut c_void,
    xc: *mut c_void,
    c: &Clip,
    n: i32,
    kin: i32,
    o: i32,
    stream: Stream<'_>,
) -> Result<()> {
    clamp(x, xc, c.imin, c.imax, elems("clin in", n, kin)?, stream)?;
    matmul(xc.cast_const(), c.w, out, n, kin, o, stream)?;
    clamp(
        out.cast_const(),
        out,
        c.omin,
        c.omax,
        elems("clin out", n, o)?,
        stream,
    )
}

struct Arena {

    h: *mut c_void,
    hn: *mut c_void,
    xc: *mut c_void,
    ffmid: *mut c_void,
    ffout: *mut c_void,
    q: *mut c_void,
    k: *mut c_void,
    v: *mut c_void,
    attn: *mut c_void,
    glu: *mut c_void,
    conv: *mut c_void,
    tmp: *mut c_void,
    start: *mut c_void,
    pe: *mut c_void,
    relk: *mut c_void,
}

#[allow(clippy::too_many_lines)]
pub fn run(
    w: &Weights,
    features: &[u8],
    n_frames: i32,
    n_mel: i32,
    out_len: i32,
    out_proj: *mut c_void,
    stream: Stream<'_>,
) -> Result<()> {
    let g = Walk::new(w, n_frames, n_mel, out_len)?;
    let (hd, nh, im, n) = (g.hd, g.nh, g.im, g.n);
    let (opd, txt, eps) = (w.out_proj_dims, w.text_hidden, w.eps);
    let mut scratch = Scratch::new();

    let (t0, f0) = (n_frames, n_mel);
    let mel = elems("mel plane", t0, f0)?;
    if features.len() < mel * 4 {
        return Err(Refused::new(
            WHO,
            format!(
                "clip features hold {} bytes for a [{t0}, {f0}] f32 plane, which needs {}",
                features.len(),
                mel * 4
            ),
        ));
    }
    let f32d = scratch.upload_bytes(&features[..mel * 4], stream)?;
    let feat = scratch.bf16(mel)?;
    call("vision::k_f32_to_bf16_bf16", stream, |ctx| {
        vision::k_f32_to_bf16_bf16(ctx, f32d.cast_const(), feat, mel)
    })?;

    let (c0ch, c1ch) = (w.sscp_ch0, w.sscp_ch1);
    let (t1, f1, t2, f2) = (g.t1, g.f1, n, g.f2);
    let c0 = scratch.bf16(elems("sscp0", c0ch, t1 * f1)?)?;
    let c0cl = scratch.bf16(elems("sscp0 (channels-last)", c0ch, t1 * f1)?)?;
    let c1 = scratch.bf16(elems("sscp1", c1ch, t2 * f2)?)?;
    let c1cl = scratch.bf16(elems("sscp1 (channels-last)", c1ch, t2 * f2)?)?;

    sscp(
        SscpStage {
            src: feat.cast_const(),
            conv_w: w.sscp0_conv,
            norm_w: w.sscp0_norm,
            chw: c0,
            chlast: c0cl,
            in_ch: 1,
            t_in: t0,
            f_in: f0,
            out_ch: c0ch,
            t_out: t1,
            f_out: f1,
            eps,
        },
        stream,
    )?;
    sscp(
        SscpStage {
            src: c0.cast_const(),
            conv_w: w.sscp1_conv,
            norm_w: w.sscp1_norm,
            chw: c1,
            chlast: c1cl,
            in_ch: c0ch,
            t_in: t1,
            f_in: f1,
            out_ch: c1ch,
            t_out: t2,
            f_out: f2,
            eps,
        },
        stream,
    )?;

    let flat_w = f2
        .checked_mul(c1ch)
        .ok_or_else(|| Refused::new(WHO, "flattened SSCP width overflowed"))?;
    let flat = scratch.bf16(elems("sscp flat", n, flat_w)?)?;
    call("vision::k_sscp_flatten_bf16", stream, |ctx| {
        vision::k_sscp_flatten_bf16(ctx, c1.cast_const(), flat, c1ch, t2, f2)
    })?;

    let a = Arena {
        h: scratch.bf16(elems("hidden", n, hd)?)?,
        hn: scratch.bf16(elems("hidden (normed)", n, hd)?)?,
        xc: scratch.bf16(elems("clip staging", n, im)?)?,
        ffmid: scratch.bf16(elems("ffn inner", n, im)?)?,
        ffout: scratch.bf16(elems("ffn out", n, hd)?)?,
        q: scratch.bf16(elems("q", n, hd)?)?,
        k: scratch.bf16(elems("k", n, hd)?)?,
        v: scratch.bf16(elems("v", n, hd)?)?,
        attn: scratch.bf16(elems("attn", n, hd)?)?,
        glu: scratch.bf16(elems("glu", n, hd)?)?,
        conv: scratch.bf16(elems("conv", n, hd)?)?,
        tmp: scratch.bf16(elems("tmp", n, hd)?)?,
        start: scratch.bf16(elems("lconv start", n, 2 * hd)?)?,
        pe: scratch.bf16(elems("pos enc", g.pp, hd)?)?,
        relk: scratch.bf16(elems("relative k", g.pp, hd)?)?,
    };
    matmul(
        flat.cast_const(),
        w.sscp_input_proj,
        a.h,
        n,
        flat_w,
        hd,
        stream,
    )?;

    call("vision::k_rel_pos_enc_bf16", stream, |ctx| {
        vision::k_rel_pos_enc_bf16(ctx, a.pe, g.pp, hd)
    })?;

    for layer in &w.layers {
        ffn(&g, &a, &layer.ff1, w.residual_weight, eps, stream)?;

        rms(
            a.h.cast_const(),
            layer.norm_pre_attn,
            a.hn,
            n,
            hd,
            eps,
            stream,
        )?;
        clin(a.hn.cast_const(), a.q, a.xc, &layer.q, n, hd, hd, stream)?;
        clin(a.hn.cast_const(), a.k, a.xc, &layer.k, n, hd, hd, stream)?;
        clin(a.hn.cast_const(), a.v, a.xc, &layer.v, n, hd, hd, stream)?;

        call("vision::k_qkv_scale_bf16", stream, |ctx| {
            vision::k_qkv_scale_bf16(
                ctx,
                a.q,
                a.k,
                layer.per_dim_scale,
                n,
                nh,
                g.head_dim,
                g.q_scale,
                g.k_scale,
            )
        })?;

        matmul(
            a.pe.cast_const(),
            layer.relative_k,
            a.relk,
            g.pp,
            hd,
            hd,
            stream,
        )?;
        call("vision::k_local_attn_bf16", stream, |ctx| {
            vision::k_local_attn_bf16(
                ctx,
                a.q.cast_const(),
                a.k.cast_const(),
                a.v.cast_const(),
                a.relk.cast_const(),
                a.attn,
                n,
                nh,
                g.head_dim,
                g.pp,
                w.logit_cap,
            )
        })?;
        clin(
            a.attn.cast_const(),
            a.tmp,
            a.xc,
            &layer.post,
            n,
            hd,
            hd,
            stream,
        )?;
        rms(
            a.tmp.cast_const(),
            layer.norm_post_attn,
            a.tmp,
            n,
            hd,
            eps,
            stream,
        )?;
        add(
            a.h,
            a.tmp.cast_const(),
            elems("attn residual", n, hd)?,
            stream,
        )?;

        rms(
            a.h.cast_const(),
            layer.lconv_pre_ln,
            a.hn,
            n,
            hd,
            eps,
            stream,
        )?;
        clin(
            a.hn.cast_const(),
            a.start,
            a.xc,
            &layer.lconv_start,
            n,
            hd,
            2 * hd,
            stream,
        )?;

        call("vision::k_glu_bf16", stream, |ctx| {
            vision::k_glu_bf16(ctx, a.start.cast_const(), a.glu, n, hd)
        })?;

        let k = w.conv_kernel;
        if n > 0 && hd > 0 && k > 0 {
            call("ssm::causal_conv1d_prefill_noact_bf16", stream, |ctx| {
                ssm::causal_conv1d_prefill_noact::<bf16>(
                    ctx,
                    a.glu.cast_const().cast(),
                    layer.depthwise_conv.cast(),
                    crate::jit::abi::MaybeConst::none(),
                    a.conv.cast(),
                    core::ptr::null_mut(),
                    n,
                    hd,
                    k,
                )
            })?;
        }

        rms(
            a.conv.cast_const(),
            layer.lconv_conv_norm,
            a.conv,
            n,
            hd,
            eps,
            stream,
        )?;
        silu(a.conv, elems("conv silu", n, hd)?, stream)?;
        clin(
            a.conv.cast_const(),
            a.tmp,
            a.xc,
            &layer.lconv_end,
            n,
            hd,
            hd,
            stream,
        )?;
        add(
            a.h,
            a.tmp.cast_const(),
            elems("conv residual", n, hd)?,
            stream,
        )?;

        ffn(&g, &a, &layer.ff2, w.residual_weight, eps, stream)?;
        rms(a.h.cast_const(), layer.norm_out, a.h, n, hd, eps, stream)?;
    }

    let enc = scratch.bf16(elems("encoder out", n, opd)?)?;
    call("vision::k_matmul_bias_bf16", stream, |ctx| {
        vision::k_matmul_bias_bf16(
            ctx,
            a.h.cast_const(),
            w.output_proj_w,
            w.output_proj_b,
            enc,
            n,
            hd,
            opd,
        )
    })?;

    let en = scratch.bf16(elems("embedder", n, opd)?)?;
    rms(enc.cast_const(), core::ptr::null(), en, n, opd, eps, stream)?;
    matmul(en.cast_const(), w.embed_proj, out_proj, n, opd, txt, stream)?;

    stream.synchronize()?;
    drop(scratch);
    Ok(())
}

struct SscpStage {

    src: *const c_void,
    conv_w: *const c_void,
    norm_w: *const c_void,
    chw: *mut c_void,
    chlast: *mut c_void,
    in_ch: i32,
    t_in: i32,
    f_in: i32,
    out_ch: i32,
    t_out: i32,
    f_out: i32,
    eps: f32,
}

fn sscp(s: SscpStage, stream: Stream<'_>) -> Result<()> {
    call("vision::k_conv2d_s2_bf16", stream, |ctx| {
        vision::k_conv2d_s2_bf16(
            ctx, s.src, s.conv_w, s.chw, s.in_ch, s.t_in, s.f_in, s.out_ch, s.t_out, s.f_out,
        )
    })?;
    call("vision::k_chlast_bf16", stream, |ctx| {
        vision::k_chlast_bf16(
            ctx,
            s.chw.cast_const(),
            s.chlast,
            s.out_ch,
            s.t_out,
            s.f_out,
        )
    })?;

    let rows = s
        .t_out
        .checked_mul(s.f_out)
        .ok_or_else(|| Refused::new(WHO, "SSCP row count overflowed"))?;
    call("vision::k_layernorm_relu_bf16", stream, |ctx| {
        vision::k_layernorm_relu_bf16(
            ctx,
            s.chlast.cast_const(),
            s.norm_w,
            s.chlast,
            rows,
            s.out_ch,
            s.eps,
        )
    })?;
    call("vision::k_chfirst_bf16", stream, |ctx| {
        vision::k_chfirst_bf16(
            ctx,
            s.chlast.cast_const(),
            s.chw,
            s.out_ch,
            s.t_out,
            s.f_out,
        )
    })
}

fn add(y: *mut c_void, x: *const c_void, n: usize, stream: Stream<'_>) -> Result<()> {
    call("vision::k_add_bf16", stream, |ctx| {
        vision::k_add_bf16(ctx, y, x, n)
    })
}

fn silu(x: *mut c_void, n: usize, stream: Stream<'_>) -> Result<()> {
    call("vision::k_silu_bf16", stream, |ctx| {
        vision::k_silu_bf16(ctx, x.cast_const(), x, n)
    })
}

fn ffn(
    g: &Walk,
    a: &Arena,
    f: &Ffn,
    residual_weight: f32,
    eps: f32,
    stream: Stream<'_>,
) -> Result<()> {
    let (n, hd, im) = (g.n, g.hd, g.im);
    rms(a.h.cast_const(), f.pre_ln, a.hn, n, hd, eps, stream)?;
    clin(a.hn.cast_const(), a.ffmid, a.xc, &f.fc1, n, hd, im, stream)?;
    silu(a.ffmid, elems("ffn silu", n, im)?, stream)?;
    clin(
        a.ffmid.cast_const(),
        a.ffout,
        a.xc,
        &f.fc2,
        n,
        im,
        hd,
        stream,
    )?;
    rms(a.ffout.cast_const(), f.post_ln, a.ffout, n, hd, eps, stream)?;

    let count = elems("ffn residual", n, hd)?;
    call("vision::k_axpy_bf16", stream, |ctx| {
        vision::k_axpy_bf16(ctx, a.h, a.ffout.cast_const(), residual_weight, count)
    })
}

#[must_use]
pub fn subsampled_len(n_frames: i32) -> i32 {
    let conv = |n: i32| (n + 2 - 3) / 2 + 1;
    conv(conv(n_frames))
}

pub fn encode(
    w: &Weights,
    features: &[u8],
    feature_byte_indptr: &[u32],
    output_rows: &mut [u8],
    output_row_indptr: &mut [u32],
    stream: Stream<'_>,
) -> Result<()> {
    let num_clips = output_row_indptr.len().saturating_sub(1);
    if num_clips == 0 || feature_byte_indptr.len() < num_clips + 1 {
        return Err(Refused::new(WHO, "invalid standalone encode inputs"));
    }
    let n_mel = w.n_mel;
    let mel = usize::try_from(n_mel)
        .ok()
        .filter(|v| *v != 0)
        .ok_or_else(|| Refused::new(WHO, "the tower states no mel bin count"))?;
    let row_bytes = usize::try_from(w.text_hidden).unwrap_or(0) * 2;
    let mut rows_written = 0usize;
    output_row_indptr[0] = 0;
    for clip in 0..num_clips {
        let blo = feature_byte_indptr[clip] as usize;
        let bhi = feature_byte_indptr[clip + 1] as usize;
        if bhi < blo || bhi > features.len() {
            return Err(Refused::new(
                WHO,
                format!("clip {clip}'s feature span [{blo}, {bhi}) leaves the payload"),
            ));
        }
        let floats = (bhi - blo) / 4;
        let frames = floats / mel;
        if frames == 0 || !floats.is_multiple_of(mel) {
            return Err(Refused::new(
                WHO,
                format!("invalid feature shape ({floats} floats over {mel} mel bins)"),
            ));
        }
        let frames_i = i32::try_from(frames)
            .map_err(|_| Refused::new(WHO, "mel frame count overflowed an int"))?;
        let rows = subsampled_len(frames_i);
        let rows_u = usize::try_from(rows)
            .map_err(|_| Refused::new(WHO, "subsampled row count is negative"))?;
        let want = rows_written
            .checked_add(rows_u)
            .and_then(|r| r.checked_mul(row_bytes))
            .ok_or_else(|| Refused::new(WHO, "output row count overflowed"))?;
        if want > output_rows.len() {
            return Err(Refused::new(WHO, "encode output buffer too small"));
        }

        let mut scratch = Scratch::new();
        let projected = scratch.bf16(
            rows_u
                .checked_mul(usize::try_from(w.text_hidden).unwrap_or(0))
                .ok_or_else(|| Refused::new(WHO, "projected row buffer overflowed"))?,
        )?;
        run(
            w,
            &features[blo..bhi],
            frames_i,
            n_mel,
            rows,
            projected,
            stream,
        )?;
        let begin = rows_written * row_bytes;
        let end = begin + rows_u * row_bytes;

        unsafe { read_raw_span(projected.cast_const(), &mut output_rows[begin..end], stream)? };
        stream.synchronize()?;
        drop(scratch);
        rows_written += rows_u;
        output_row_indptr[clip + 1] = u32::try_from(rows_written)
            .map_err(|_| Refused::new(WHO, "encoded row count overflowed u32"))?;
    }
    Ok(())
}
