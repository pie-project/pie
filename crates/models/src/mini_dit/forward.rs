use poem_dsl::fact;
use poem_dsl::{
    Dtype, ForwardHybrid, HybridSpec, Input, ModulateForm, RaggedMask, RopeForm, Stream, Value,
    Weight, ops, seam,
};

use crate::{
    AxisRole, Generative, LatentSpace, PortFact, PortKind, PositionConvention, ReadingFact,
    ReadoutKind, ScheduleFact, ScheduleKind,
};

use super::model::{
    CrossAttn, HEAD_DIM, INTER, LN_EPS, Linear, MOD_SLICES, Model, RMS_EPS, ROPE_AXES, ROPE_DIMS,
    ROPE_THETA, SM_SCALE, SelfAttn, Side, Swiglu, TIMESTEP_DIM, TIMESTEP_FLIP_SIN_COS,
    TIMESTEP_MAX_PERIOD, TIMESTEP_SCALE, port,
};

pub const STREAM_BASE: u8 = 0;

pub const DENOISE_BIT: u8 = 6;

pub const DENOISE_READING: u8 = 0;

pub struct Tap;

impl Tap {
    pub const ENV: &'static str = "PIE_MINI_DIT_TAP";

    #[must_use]
    pub fn from_env() -> Option<String> {
        std::env::var(Self::ENV).ok().filter(|key| !key.is_empty())
    }

    #[must_use]
    pub fn width(key: Option<&str>) -> u32 {
        match key {
            None | Some("final.tokens") => super::model::PATCH_FEATURES,
            Some(_) => super::model::HIDDEN,
        }
    }
}

#[must_use]
pub fn generative(tap: Option<&str>) -> Generative {
    Generative {
        readings: vec![denoise_reading(tap)],
        latent: Some(LatentSpace {
            channels: super::model::CHANNELS,
            patch_t: 1,
            patch_h: super::model::PATCH,
            patch_w: super::model::PATCH,
            spatial_compression: 1,
            temporal_compression: 1,
        }),
        schedule: Some(ScheduleFact {
            kind: ScheduleKind::Flow,
            shift: 1.0,
            train_steps: 1000,
            boundary: None,
            pinned_sigmas: vec![1.0, 0.75, 0.5, 0.25],
            stream_shifts: vec![],
        }),
        max_rows: 4096,
    }
}

fn denoise_reading(tap: Option<&str>) -> ReadingFact {
    let port = |name, kind, width, streams: &[Stream]| PortFact {
        name,
        kind,
        width,
        streams: streams.to_vec(),
        at: None,
        rows: None,
    };
    ReadingFact {
        name: "denoise",
        index: DENOISE_READING,
        has_kv: false,
        takes_tokens: false,
        streams: vec![Stream::Text, Stream::Image, Stream::Context],
        ports: vec![
            port(
                "latents",
                PortKind::Latents,
                super::model::PATCH_FEATURES,
                &[Stream::Image],
            ),
            port(
                "text",
                PortKind::Context,
                super::model::TEXT_WIDTH,
                &[Stream::Text],
            ),
            port(
                "context",
                PortKind::Context,
                super::model::CONTEXT_WIDTH,
                &[Stream::Context],
            ),
            port(
                "timestep",
                PortKind::LaneVector,
                1,
                &[Stream::Text, Stream::Image],
            ),
            port(
                "positions",
                PortKind::AxisPositions,
                u32::from(ROPE_AXES),
                &[Stream::Text, Stream::Image],
            ),
        ],
        positions: Some(PositionConvention {
            axes: vec![AxisRole::Time, AxisRole::Height, AxisRole::Width],
            text_axis: 0,
            text_origin: 0,
            image_follows_text: false,
            reference_stride: None,
        }),
        readout: ReadoutKind::Velocity,
        readout_width: Tap::width(tap),
    }
}

macro_rules! tap {
    ($m:expr, $key:literal, $v:expr) => {
        if let Some(v) = tapped($m, "", $key, $v) {
            return v;
        }
    };
}

macro_rules! unwrap_tap {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(tapped) => return tapped,
        }
    };
}

impl ForwardHybrid for Model {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }

    fn forward(&self, inputs: Input) -> Value {
        let m = self;

        let (ctx, joint) = (
            inputs.on(fact::stream(Stream::Context)),
            inputs.on(!fact::stream(Stream::Context)),
        );
        let (txt_in, img_in) = (
            joint.on(fact::stream(Stream::Text)),
            joint.on(!fact::stream(Stream::Text)),
        );

        let lanes = inputs.request_of_token();
        let positions = inputs.axis_positions(port::POSITIONS, ROPE_AXES);

        let t = inputs.lane_vector(port::TIMESTEP, 1);
        let temb = ops::elemwise::sinusoid(
            &t,
            TIMESTEP_DIM,
            TIMESTEP_MAX_PERIOD,
            TIMESTEP_FLIP_SIN_COS,
            TIMESTEP_SCALE,
        );
        let temb = ops::elemwise::silu(&temb);

        let joint_perm = joint.row_permutation();
        let joint_csr = joint.group_indptr();
        let img_perm = img_in.row_permutation();
        let img_csr = img_in.group_indptr();

        let txt = txt_in.context(port::TEXT, super::model::TEXT_WIDTH);
        let patches = img_in.latents(port::LATENTS, super::model::PATCH_FEATURES, Dtype::Bf16);
        let ctx_rows = ctx.context(port::CONTEXT, super::model::CONTEXT_WIDTH);
        tap!(
            m,
            "in.text",
            &ops::elemwise::layernorm_no_scale(&txt, LN_EPS)
        );
        let img = linear(&m.x_embed, &patches);
        tap!(m, "x_embed", &img);

        let x = Value::merge(vec![txt, img]);
        tap!(m, "b0.in", &x);
        let (msa, gate_a, mmlp, gate_m) = adaln6(&linear(&m.single.ada, &temb));
        let x = unwrap_tap!(attn_sublayer(
            m,
            "b0",
            &x,
            &m.single.attn,
            &msa,
            &gate_a,
            &lanes,
            &positions,
            &joint_perm,
            &joint_csr,
        ));
        tap!(m, "b0.x_after_attn", &x);
        let x = unwrap_tap!(mlp_sublayer(
            m,
            "b0",
            &x,
            &m.single.mlp,
            &mmlp,
            &gate_m,
            &lanes
        ));
        tap!(m, "b0.out", &x);

        let (txt, img) = (
            x.on(fact::stream(Stream::Text)),
            x.on(!fact::stream(Stream::Text)),
        );
        let (txt_mod, img_mod) = (
            adaln6(&linear(&m.double.txt.ada, &temb)),
            adaln6(&linear(&m.double.img.ada, &temb)),
        );
        let (tq, tk, tv) = unwrap_tap!(qkv(
            m,
            "b1.txt_",
            &txt,
            &m.double.txt,
            &txt_mod.0,
            &lanes,
            &positions
        ));
        let (iq, ik, iv) = unwrap_tap!(qkv(
            m,
            "b1.img_",
            &img,
            &m.double.img,
            &img_mod.0,
            &lanes,
            &positions
        ));
        let o = joint_attention(
            &Value::merge(vec![tq, iq]),
            &Value::merge(vec![tk, ik]),
            &Value::merge(vec![tv, iv]),
            &joint_perm,
            &joint_csr,
        );
        tap!(m, "b1.joint_attn_heads", &o);
        let (o_txt, o_img) = (
            o.on(fact::stream(Stream::Text)),
            o.on(!fact::stream(Stream::Text)),
        );
        let ta = linear(&m.double.txt.attn.out, &o_txt);
        tap!(m, "b1.txt_attn_out", &ta);
        let ia = linear(&m.double.img.attn.out, &o_img);
        tap!(m, "b1.img_attn_out", &ia);
        let txt = ops::elemwise::gated_residual_add(&txt, &txt_mod.1, &ta, Some(&lanes));
        tap!(m, "b1.txt_after_attn", &txt);
        let img = ops::elemwise::gated_residual_add(&img, &img_mod.1, &ia, Some(&lanes));
        tap!(m, "b1.img_after_attn", &img);
        let txt = unwrap_tap!(mlp_sublayer(
            m,
            "b1.txt_",
            &txt,
            &m.double.txt.mlp,
            &txt_mod.2,
            &txt_mod.3,
            &lanes
        ));
        tap!(m, "b1.out_txt", &txt);
        let x = unwrap_tap!(mlp_sublayer(
            m,
            "b1.img_",
            &img,
            &m.double.img.mlp,
            &img_mod.2,
            &img_mod.3,
            &lanes
        ));
        tap!(m, "b1.out_img", &x);

        let mod2 = ops::elemwise::add_bias(&m.cross.mod_table, &linear(&m.cross.ada, &temb));
        let (msa, gate_a, mffn, gate_f) = adaln6(&mod2);
        let x = unwrap_tap!(attn_sublayer(
            m,
            "b2",
            &x,
            &m.cross.self_attn,
            &msa,
            &gate_a,
            &lanes,
            &positions,
            &img_perm,
            &img_csr,
        ));
        tap!(m, "b2.x_after_self", &x);

        let hc = ops::elemwise::layernorm(&x, &m.cross.norm, &m.cross.norm_bias, LN_EPS);
        tap!(m, "b2.cross_norm_out", &hc);
        let cq = ops::elemwise::rmsnorm_per_head(
            &linear(&m.cross.cross.q, &hc),
            &m.cross.cross.q_norm,
            HEAD_DIM,
            RMS_EPS,
        );
        tap!(m, "b2.cross_q", &cq);
        let (ck, cv) = context_kv(&ctx_rows, &m.cross.cross);
        let ctx_perm = ctx.row_permutation();
        let ctx_csr = ctx.group_indptr();
        let ca = ops::attn::ragged(
            &ops::layout::pack_rows(&cq, &img_perm),
            &ops::layout::pack_rows(&ck, &ctx_perm),
            &ops::layout::pack_rows(&cv, &ctx_perm),
            &img_csr,
            &ctx_csr,
            HEAD_DIM,
            SM_SCALE,
            RaggedMask::GroupBlockDiagonal,
        );
        let ca = ops::layout::unpack_rows(&ca, &img_perm);
        tap!(m, "b2.cross_attn_heads", &ca);
        let ca = linear(&m.cross.cross.out, &ca);
        tap!(m, "b2.cross_attn_out", &ca);
        let x = ops::elemwise::residual_add(&ca, &x);
        tap!(m, "b2.x_after_cross", &x);
        let x = unwrap_tap!(mlp_sublayer(
            m,
            "b2",
            &x,
            &m.cross.mlp,
            &mffn,
            &gate_f,
            &lanes
        ));
        tap!(m, "b2.out", &x);

        let scale_shift = linear(&m.final_ada, &temb);
        let hf = ops::elemwise::modulate(
            &ops::elemwise::layernorm_no_scale(&x, LN_EPS),
            &scale_shift,
            Some(&lanes),
            ModulateForm::ScaleShift,
        );
        tap!(m, "final.norm_out", &hf);
        let velocity = linear(&m.final_proj, &hf);
        seam::at(seam::VELOCITY, &[&velocity]);
        velocity
    }
}

fn linear(w: &Linear, x: &Value) -> Value {
    ops::elemwise::add_bias(&w.bias, &ops::linear::matmul(x, &w.w))
}

fn adaln6(m: &Value) -> (Value, Value, Value, Value) {
    let width = super::model::HIDDEN;
    debug_assert_eq!(m.width(), u64::from(MOD_SLICES * width));
    let (msa, rest) = ops::layout::split_rows(m, 2 * width);
    let (gate_a, rest) = ops::layout::split_rows(&rest, width);
    let (mmlp, gate_m) = ops::layout::split_rows(&rest, 2 * width);
    (msa, gate_a, mmlp, gate_m)
}

#[allow(clippy::too_many_arguments)]
fn qkv(
    model: &Model,
    stem: &str,
    x: &Value,
    side: &Side,
    m: &Value,
    lanes: &Value,
    positions: &Value,
) -> Result<(Value, Value, Value), Value> {
    heads(model, stem, x, &side.attn, m, lanes, positions)
}

fn tapped(model: &Model, stem: &str, tail: &str, v: &Value) -> Option<Value> {
    let key = if stem.is_empty() || stem.ends_with('_') {
        format!("{stem}{tail}")
    } else {
        format!("{stem}.{tail}")
    };
    if model.tap.as_deref() == Some(key.as_str()) {
        seam::at(seam::VELOCITY, &[v]);
        Some(v.clone())
    } else {
        None
    }
}

fn tap_at(model: &Model, stem: &str, tail: &str, v: &Value) -> Result<(), Value> {
    match tapped(model, stem, tail, v) {
        Some(v) => Err(v),
        None => Ok(()),
    }
}

#[allow(clippy::too_many_arguments)]
fn heads(
    model: &Model,
    stem: &str,
    x: &Value,
    attn: &SelfAttn,
    m: &Value,
    lanes: &Value,
    positions: &Value,
) -> Result<(Value, Value, Value), Value> {
    let h = ops::elemwise::modulate(
        &ops::elemwise::layernorm_no_scale(x, LN_EPS),
        m,
        Some(lanes),
        ModulateForm::ScaleShift,
    );
    tap_at(model, stem, "norm1_out", &h)?;
    let (q, k, v) = ops::layout::split_qkv(
        &linear(&attn.qkv, &h),
        super::model::HIDDEN,
        super::model::HIDDEN,
    );
    let hp = if stem == "b2" { "self_" } else { "" };
    tap_at(model, stem, &format!("{hp}q_raw"), &q)?;
    tap_at(model, stem, &format!("{hp}k_raw"), &k)?;
    tap_at(model, stem, &format!("{hp}v"), &v)?;
    let turn = |x: &Value, gain: &Weight, tail: &str| -> Result<Value, Value> {
        let n = ops::elemwise::rmsnorm_per_head(x, gain, HEAD_DIM, RMS_EPS);
        tap_at(model, stem, &format!("{hp}{tail}_qknorm"), &n)?;
        let r = ops::elemwise::rope_axes(
            &n,
            positions,
            ROPE_DIMS,
            [ROPE_THETA; 4],
            RopeForm::Interleaved,
            HEAD_DIM,
            HEAD_DIM,
        );
        tap_at(model, stem, &format!("{hp}{tail}_rope"), &r)?;
        Ok(r)
    };
    Ok((
        turn(&q, &attn.q_norm, "q")?,
        turn(&k, &attn.k_norm, "k")?,
        v,
    ))
}

fn joint_attention(q: &Value, k: &Value, v: &Value, perm: &Value, csr: &Value) -> Value {
    let o = ops::attn::ragged(
        &ops::layout::pack_rows(q, perm),
        &ops::layout::pack_rows(k, perm),
        &ops::layout::pack_rows(v, perm),
        csr,
        csr,
        HEAD_DIM,
        SM_SCALE,
        RaggedMask::GroupBlockDiagonal,
    );
    ops::layout::unpack_rows(&o, perm)
}

#[allow(clippy::too_many_arguments)]
fn attn_sublayer(
    model: &Model,
    stem: &str,
    x: &Value,
    attn: &SelfAttn,
    m: &Value,
    gate: &Value,
    lanes: &Value,
    positions: &Value,
    perm: &Value,
    csr: &Value,
) -> Result<Value, Value> {
    let (q, k, v) = heads(model, stem, x, attn, m, lanes, positions)?;
    let o = joint_attention(&q, &k, &v, perm, csr);
    let (heads_key, out_key) = if stem == "b2" {
        ("self_attn_heads", "self_attn_out")
    } else {
        ("attn_heads", "attn_out")
    };
    tap_at(model, stem, heads_key, &o)?;
    let o = linear(&attn.out, &o);
    tap_at(model, stem, out_key, &o)?;
    Ok(ops::elemwise::gated_residual_add(x, gate, &o, Some(lanes)))
}

#[allow(clippy::too_many_arguments)]
fn mlp_sublayer(
    model: &Model,
    stem: &str,
    x: &Value,
    mlp: &Swiglu,
    m: &Value,
    gate: &Value,
    lanes: &Value,
) -> Result<Value, Value> {
    let h = ops::elemwise::modulate(
        &ops::elemwise::layernorm_no_scale(x, LN_EPS),
        m,
        Some(lanes),
        ModulateForm::ScaleShift,
    );
    let norm_key = if stem == "b2" {
        "norm3_out"
    } else {
        "norm2_out"
    };
    tap_at(model, stem, norm_key, &h)?;
    let h = ops::linear::mlp_swiglu(&linear(&mlp.gate_up, &h), INTER);
    let y = linear(&mlp.down, &h);
    tap_at(model, stem, "mlp_out", &y)?;
    Ok(ops::elemwise::gated_residual_add(x, gate, &y, Some(lanes)))
}

fn context_kv(c: &Value, cross: &CrossAttn) -> (Value, Value) {
    let (k, v) = ops::layout::split_rows(&linear(&cross.kv, c), super::model::HIDDEN);
    (
        ops::elemwise::rmsnorm_per_head(&k, &cross.k_norm, HEAD_DIM, RMS_EPS),
        v,
    )
}
