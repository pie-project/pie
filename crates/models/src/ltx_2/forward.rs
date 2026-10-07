use poem_dsl::ops::spatial;
use poem_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, ModulateForm, Predicate, RaggedMask,
    Request, RopeForm, Stream, Value, Weight, ops, seam,
};

use crate::{
    Generative, LatentSpace, PortFact, PortKind, ReadingFact, ReadoutKind, ScheduleFact,
    ScheduleKind,
};

use super::model::{
    AV_GATE_TIMESTEP_SCALE, AV_SS_SLICES, AdaLn, Attn, Block, Connector, DISTILLED_SIGMAS, Dims,
    Dit, Ffn, GATE_SCALE, Linear, MOD_SLICES, Model, NORM_EPS, PATCH_H, PATCH_T, PATCH_W,
    ROPE_AXES, ROPE_THETA, Side, Stream as StreamHeads, T_FLIP_SIN_COS, T_FREQ_DIM, T_MAX_PERIOD,
    T_SCALE, TEXT_LEN, TRAIN_STEPS, VAE_EPS, VAE_PATCH, VAE_RGB, VAE_SPATIAL_COMPRESSION,
    VAE_TEMPORAL_COMPRESSION, VAE_Z, Vae, VaeConv, VaeResnet, port,
};

pub const STREAM_BASE: u8 = 0;

pub const READING_LO: u8 = 6;
pub const READING_HI: u8 = 7;

pub const DENOISE: u8 = 0;
pub const REFINE_VIDEO: u8 = 1;
pub const REFINE_AUDIO: u8 = 2;
pub const VAE_DECODE: u8 = 3;

impl Model {
    #[must_use]
    pub fn generative(&self) -> Generative {
        let d = &self.dims;
        let port = |name, kind, width, streams: &[Stream], at| PortFact {
            name,
            kind,
            width,
            streams: streams.to_vec(),
            at,
            rows: None,
        };
        let refine_ports = || {
            vec![
                port(
                    "text",
                    PortKind::Latents,
                    d.text_in(),
                    &[Stream::Text],
                    Some(port::TEXT),
                ),
                port(
                    "text_positions",
                    PortKind::AxisPositions,
                    1,
                    &[Stream::Text],
                    Some(port::TIME_POSITIONS),
                ),
            ]
        };
        let mut readings = vec![
            ReadingFact {
                name: "denoise",
                index: DENOISE,
                has_kv: false,
                takes_tokens: false,
                streams: vec![
                    Stream::Video,
                    Stream::Audio,
                    Stream::Context,
                    Stream::Reference,
                ],
                ports: vec![
                    port(
                        "latents",
                        PortKind::Latents,
                        d.channels,
                        &[Stream::Video, Stream::Audio],
                        Some(port::LATENTS),
                    ),
                    port(
                        "context",
                        PortKind::Context,
                        d.cross_dim,
                        &[Stream::Context],
                        Some(port::CONTEXT),
                    ),
                    port(
                        "audio_context",
                        PortKind::Context,
                        d.audio_cross_dim,
                        &[Stream::Reference],
                        Some(port::AUDIO_CONTEXT),
                    ),
                    port(
                        "timestep",
                        PortKind::LaneVector,
                        1,
                        &[],
                        Some(port::TIMESTEP),
                    ),
                    port(
                        "positions",
                        PortKind::AxisPositions,
                        u32::from(ROPE_AXES),
                        &[Stream::Video],
                        Some(port::POSITIONS),
                    ),
                    port(
                        "audio_positions",
                        PortKind::AxisPositions,
                        1,
                        &[Stream::Audio],
                        Some(port::TIME_POSITIONS),
                    ),
                ],
                positions: None,
                readout: ReadoutKind::Velocity,
                readout_width: d.channels,
            },
            ReadingFact {
                name: "refine.video",
                index: REFINE_VIDEO,
                has_kv: false,
                takes_tokens: false,
                streams: vec![Stream::Text],
                ports: refine_ports(),
                positions: None,
                readout: ReadoutKind::Hidden,
                readout_width: d.cross_dim,
            },
            ReadingFact {
                name: "refine.audio",
                index: REFINE_AUDIO,
                has_kv: false,
                takes_tokens: false,
                streams: vec![Stream::Text],
                ports: refine_ports(),
                positions: None,
                readout: ReadoutKind::Hidden,
                readout_width: d.audio_cross_dim,
            },
        ];
        if self.vae.is_some() {
            readings.push(ReadingFact {
                name: "vae.decode",
                index: VAE_DECODE,
                has_kv: false,
                takes_tokens: false,
                streams: vec![Stream::Video],
                ports: vec![port(
                    "latent",
                    PortKind::Voxels,
                    VAE_Z,
                    &[Stream::Video],
                    Some(port::VOXELS),
                )],
                positions: None,
                readout: ReadoutKind::Pixels,
                readout_width: VAE_RGB,
            });
        }
        Generative {
            readings,
            latent: Some(LatentSpace {
                channels: d.channels,
                patch_t: PATCH_T,
                patch_h: PATCH_H,
                patch_w: PATCH_W,
                spatial_compression: VAE_SPATIAL_COMPRESSION,
                temporal_compression: VAE_TEMPORAL_COMPRESSION,
            }),
            schedule: Some(ScheduleFact {
                kind: ScheduleKind::Flow,
                shift: 1.0,
                train_steps: TRAIN_STEPS,
                boundary: None,
                pinned_sigmas: DISTILLED_SIGMAS.to_vec(),
                stream_shifts: Vec::new(),
            }),
            max_rows: match d.layers {
                48 => 32_768 + 2 * TEXT_LEN,
                _ => 4096,
            },
        }
    }
}

pub struct Facts {
    pub stream: Stream,
    pub reading: u8,
}

impl Facts {
    #[must_use]
    pub fn video() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Video)
    }

    #[must_use]
    pub fn audio() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Audio)
    }

    #[must_use]
    pub fn context() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Context)
    }

    #[must_use]
    pub fn audio_context() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Reference)
    }

    #[must_use]
    pub fn text() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Text)
    }

    #[must_use]
    pub fn reading_lo() -> Predicate {
        Predicate::fact(READING_LO)
    }

    #[must_use]
    pub fn reading_hi() -> Predicate {
        Predicate::fact(READING_HI)
    }
}

impl Classify for Facts {
    fn of(r: &Request) -> Facts {
        Facts {
            stream: r.stream(),
            reading: r.reading() & 3,
        }
    }

    fn word(&self) -> u64 {
        self.stream.word(STREAM_BASE) | (u64::from(self.reading & 3) << READING_LO)
    }
}

impl ForwardHybrid for Model {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let (hi, lo) = inputs.split(&Facts::reading_hi());
        let (c3, c2) = hi.split(&Facts::reading_lo());
        let (c1, c0) = lo.split(&Facts::reading_lo());
        let arms = [c0, c1, c2, c3];
        let arm = |code: u8| &arms[usize::from(code)];

        let velocity = denoise(arm(DENOISE), self);
        let (text_in, caption) = (self.dims.text_in(), self.dims.caption);
        refine(
            arm(REFINE_VIDEO),
            &self.connectors.0,
            text_in,
            self.connectors.0.rescale(caption),
        );
        refine(
            arm(REFINE_AUDIO),
            &self.connectors.1,
            text_in,
            self.connectors.1.rescale(caption),
        );
        if let Some(vae) = &self.vae {
            let _ = vae_decode(arm(VAE_DECODE), vae);
        }
        velocity
    }
}

fn linear(w: &Linear, x: &Value) -> Value {
    let y = ops::linear::matmul(x, &w.w);
    match &w.bias {
        Some(bias) => ops::elemwise::add_bias(bias, &y),
        None => y,
    }
}

fn rms(x: &Value) -> Value {
    let width = u32::try_from(x.width()).expect("a row narrower than 4 G");
    ops::elemwise::rmsnorm_no_scale(x, width, NORM_EPS)
}

fn modulate(x: &Value, scale_shift: &Value, lanes: &Value) -> Value {
    ops::elemwise::modulate(x, scale_shift, Some(lanes), ModulateForm::ScaleShift)
}

fn norm_modulate(x: &Value, scale_shift: &Value, lanes: &Value) -> Value {
    modulate(&rms(x), scale_shift, lanes)
}

struct Geom {
    lanes: Value,
    positions: Value,
    perm: Value,
    csr: Value,
}

struct KeyGeom {
    lanes: Value,
    perm: Value,
    csr: Value,
}

struct Mods {
    msa_ss: Value,
    msa_gate: Value,
    mlp_ss: Value,
    mlp_gate: Value,
    q_ss: Value,
    q_gate: Value,
}

fn adaln9(e: &Value, dim: u32) -> Mods {
    debug_assert_eq!(e.width(), u64::from(MOD_SLICES * dim));
    let (msa_ss, rest) = ops::layout::split_rows(e, 2 * dim);
    let (msa_gate, rest) = ops::layout::split_rows(&rest, dim);
    let (mlp_ss, rest) = ops::layout::split_rows(&rest, 2 * dim);
    let (mlp_gate, rest) = ops::layout::split_rows(&rest, dim);
    let (q_ss, q_gate) = ops::layout::split_rows(&rest, 2 * dim);
    Mods {
        msa_ss,
        msa_gate,
        mlp_ss,
        mlp_gate,
        q_ss,
        q_gate,
    }
}

struct AvMods {
    a2v_ss: Value,
    v2a_ss: Value,
}

fn adaln_av(e: &Value, dim: u32) -> AvMods {
    debug_assert_eq!(e.width(), u64::from(AV_SS_SLICES * dim));
    let (a2v_ss, v2a_ss) = ops::layout::split_rows(e, 2 * dim);
    AvMods { a2v_ss, v2a_ss }
}

fn adaln(head: &AdaLn, sinusoid: &Value) -> (Value, Value) {
    let h = ops::elemwise::silu(&linear(&head.embed.linear_1, sinusoid));
    let emb = linear(&head.embed.linear_2, &h);
    (linear(&head.proj, &ops::elemwise::silu(&emb)), h)
}

fn sinusoid(t: &Value) -> Value {
    ops::elemwise::sinusoid(t, T_FREQ_DIM, T_MAX_PERIOD, T_FLIP_SIN_COS, T_SCALE)
}

struct StreamMods {
    proj9: Value,
    av_ss: Value,
    av_gate: Value,
    head: Value,
}

fn stream_mods(heads: &StreamHeads, t: &Value) -> StreamMods {
    let sin = sinusoid(t);
    let (proj9, hidden) = adaln(&heads.adaln, &sin);
    let (av_ss, _) = adaln(&heads.av_ss, &sin);
    debug_assert_eq!(AV_GATE_TIMESTEP_SCALE, 1.0);
    let (av_gate, _) = adaln(&heads.av_gate, &sin);
    let head = ops::elemwise::add_bias(&heads.head_table, &linear(&heads.head_proj, &hidden));
    StreamMods {
        proj9,
        av_ss,
        av_gate,
        head,
    }
}

fn qk_norm(x: &Value, gain: &Weight) -> Value {
    ops::elemwise::rmsnorm(x, gain, NORM_EPS)
}

fn turn(x: &Value, positions: &Value, dims: [u32; 4], head_dim: u32) -> Value {
    ops::elemwise::rope_axes(
        x,
        positions,
        dims,
        [ROPE_THETA; 4],
        RopeForm::SplitLadder,
        head_dim,
        head_dim,
    )
}

fn gate_out(o: &Value, h: &Value, a: &Attn) -> Value {
    let logits = linear(&a.gate, h);
    let gated = ops::elemwise::gate_sigmoid_mul_heads(o, &logits, a.head_dim, GATE_SCALE);
    linear(&a.out, &gated)
}

fn self_attention(h: &Value, a: &Attn, dims: [u32; 4], g: &Geom) -> Value {
    let inner = a.inner();
    let (q, k, v) = ops::layout::split_qkv(&linear(&a.qkv, h), inner, inner);
    let q = turn(&qk_norm(&q, &a.q_norm), &g.positions, dims, a.head_dim);
    let k = turn(&qk_norm(&k, &a.k_norm), &g.positions, dims, a.head_dim);
    let o = ops::attn::ragged(
        &ops::layout::pack_rows(&q, &g.perm),
        &ops::layout::pack_rows(&k, &g.perm),
        &ops::layout::pack_rows(&v, &g.perm),
        &g.csr,
        &g.csr,
        a.head_dim,
        a.sm_scale(),
        RaggedMask::GroupBlockDiagonal,
    );
    gate_out(&ops::layout::unpack_rows(&o, &g.perm), h, a)
}

#[allow(clippy::too_many_arguments)]
fn cross_attention(
    h: &Value,
    ctx: &Value,
    a: &Attn,
    rope: Option<(&Value, &Value, [u32; 4])>,
    qg: &Geom,
    kg: &KeyGeom,
) -> Value {
    let inner = a.inner();
    let kv =
        a.kv.as_ref()
            .expect("a cross-attention keeps its k|v apart");
    let q = qk_norm(&linear(&a.qkv, h), &a.q_norm);
    let (k, v) = ops::layout::split_rows(&linear(kv, ctx), inner);
    let k = qk_norm(&k, &a.k_norm);
    let (q, k) = match rope {
        Some((q_pos, k_pos, dims)) => (
            turn(&q, q_pos, dims, a.head_dim),
            turn(&k, k_pos, dims, a.head_dim),
        ),
        None => (q, k),
    };
    let o = ops::attn::ragged(
        &ops::layout::pack_rows(&q, &qg.perm),
        &ops::layout::pack_rows(&k, &kg.perm),
        &ops::layout::pack_rows(&v, &kg.perm),
        &qg.csr,
        &kg.csr,
        a.head_dim,
        a.sm_scale(),
        RaggedMask::GroupBlockDiagonal,
    );
    gate_out(&ops::layout::unpack_rows(&o, &qg.perm), h, a)
}

fn ff_sublayer(x: &Value, ff: &Ffn, ss: &Value, gate: &Value, lanes: &Value) -> Value {
    let h = norm_modulate(x, ss, lanes);
    let f = linear(&ff.down, &ops::elemwise::gelu(&linear(&ff.up, &h), true));
    ops::elemwise::gated_residual_add(x, gate, &f, Some(lanes))
}

struct BlockMods {
    m: Mods,
    av: AvMods,
    av_gate: Value,
    prompt_ss: Value,
}

fn table_add(table: &Weight, v: &Value) -> Value {
    ops::elemwise::add_bias(table, &ops::elemwise::copy(v))
}

fn block_mods(side: &Side, s: &StreamMods, prompt: &Value, dim: u32) -> BlockMods {
    BlockMods {
        m: adaln9(&table_add(&side.table, &s.proj9), dim),
        av: adaln_av(&table_add(&side.av_ss_table, &s.av_ss), dim),
        av_gate: table_add(&side.av_gate_table, &s.av_gate),
        prompt_ss: table_add(&side.prompt_table, prompt),
    }
}

#[allow(clippy::too_many_arguments)]
fn block(
    xv: &Value,
    xa: &Value,
    b: &Block,
    d: &Dims,
    mv: &BlockMods,
    ma: &BlockMods,
    ctx: &Value,
    actx: &Value,
    vg: &Geom,
    ag: &Geom,
    v_time: &Value,
    cg: &KeyGeom,
    acg: &KeyGeom,
) -> (Value, Value) {
    let hv = norm_modulate(xv, &mv.m.msa_ss, &vg.lanes);
    let ov = self_attention(&hv, &b.video.self_attn, d.rope_dims(), vg);
    let xv = ops::elemwise::gated_residual_add(xv, &mv.m.msa_gate, &ov, Some(&vg.lanes));

    let ha = norm_modulate(xa, &ma.m.msa_ss, &ag.lanes);
    let oa = self_attention(&ha, &b.audio.self_attn, d.audio_rope_dims(), ag);
    let xa = ops::elemwise::gated_residual_add(xa, &ma.m.msa_gate, &oa, Some(&ag.lanes));

    let hv = norm_modulate(&xv, &mv.m.q_ss, &vg.lanes);
    let c = modulate(ctx, &mv.prompt_ss, &cg.lanes);
    let ov = cross_attention(&hv, &c, &b.video.cross, None, vg, cg);
    let xv = ops::elemwise::gated_residual_add(&xv, &mv.m.q_gate, &ov, Some(&vg.lanes));

    let ha = norm_modulate(&xa, &ma.m.q_ss, &ag.lanes);
    let ac = modulate(actx, &ma.prompt_ss, &acg.lanes);
    let oa = cross_attention(&ha, &ac, &b.audio.cross, None, ag, acg);
    let xa = ops::elemwise::gated_residual_add(&xa, &ma.m.q_gate, &oa, Some(&ag.lanes));

    let nv = rms(&xv);
    let na = rms(&xa);
    let av_dims = d.av_rope_dims();

    let q_in = modulate(&nv, &mv.av.a2v_ss, &vg.lanes);
    let kv_in = modulate(&na, &ma.av.a2v_ss, &ag.lanes);
    let o = cross_attention(
        &q_in,
        &kv_in,
        &b.a2v,
        Some((v_time, &ag.positions, av_dims)),
        vg,
        &KeyGeom {
            lanes: ag.lanes.clone(),
            perm: ag.perm.clone(),
            csr: ag.csr.clone(),
        },
    );
    let xv = ops::elemwise::gated_residual_add(&xv, &mv.av_gate, &o, Some(&vg.lanes));

    let q_in = modulate(&na, &ma.av.v2a_ss, &ag.lanes);
    let kv_in = modulate(&nv, &mv.av.v2a_ss, &vg.lanes);
    let o = cross_attention(
        &q_in,
        &kv_in,
        &b.v2a,
        Some((&ag.positions, v_time, av_dims)),
        ag,
        &KeyGeom {
            lanes: vg.lanes.clone(),
            perm: vg.perm.clone(),
            csr: vg.csr.clone(),
        },
    );
    let xa = ops::elemwise::gated_residual_add(&xa, &ma.av_gate, &o, Some(&ag.lanes));

    let xv = ff_sublayer(&xv, &b.video.ffn, &mv.m.mlp_ss, &mv.m.mlp_gate, &vg.lanes);
    let xa = ff_sublayer(&xa, &b.audio.ffn, &ma.m.mlp_ss, &ma.m.mlp_gate, &ag.lanes);
    (xv, xa)
}

fn denoise(arm: &Input<Facts>, m: &Model) -> Value {
    let d = &m.dims;
    let dit: &Dit = &m.dit;

    let (ctx, rest) = arm.split(&Facts::context());
    let (actx, media) = rest.split(&Facts::audio_context());
    let (vid, aud) = media.split(&Facts::video());

    let vg = Geom {
        lanes: vid.request_of_token(),
        positions: vid.axis_positions(port::POSITIONS, ROPE_AXES),
        perm: vid.row_permutation(),
        csr: vid.group_indptr(),
    };
    let ag = Geom {
        lanes: aud.request_of_token(),
        positions: aud.axis_positions(port::TIME_POSITIONS, 1),
        perm: aud.row_permutation(),
        csr: aud.group_indptr(),
    };
    let cg = KeyGeom {
        lanes: ctx.request_of_token(),
        perm: ctx.row_permutation(),
        csr: ctx.group_indptr(),
    };
    let acg = KeyGeom {
        lanes: actx.request_of_token(),
        perm: actx.row_permutation(),
        csr: actx.group_indptr(),
    };
    let (v_time, _) = ops::layout::split_rows(&vg.positions, 1);

    let vt = vid.lane_vector(port::TIMESTEP, 1);
    let at = aud.lane_vector(port::TIMESTEP, 1);
    let mods_v = stream_mods(&dit.video, &vt);
    let mods_a = stream_mods(&dit.audio, &at);
    let (prompt_v, _) = adaln(&dit.prompt, &sinusoid(&ctx.lane_vector(port::TIMESTEP, 1)));
    let (prompt_a, _) = adaln(
        &dit.audio_prompt,
        &sinusoid(&actx.lane_vector(port::TIMESTEP, 1)),
    );

    let mut xv = linear(
        &dit.video.patchify,
        &vid.latents(port::LATENTS, d.channels, Dtype::Bf16),
    );
    let mut xa = linear(
        &dit.audio.patchify,
        &aud.latents(port::LATENTS, d.channels, Dtype::Bf16),
    );
    let ctx_rows = ctx.context(port::CONTEXT, d.cross_dim);
    let actx_rows = actx.context(port::AUDIO_CONTEXT, d.audio_cross_dim);

    for (_, b) in arm.walk_layers(&dit.blocks) {
        let mv = block_mods(&b.video, &mods_v, &prompt_v, d.dim());
        let ma = block_mods(&b.audio, &mods_a, &prompt_a, d.audio_dim());
        let (v, a) = block(
            &xv, &xa, b, d, &mv, &ma, &ctx_rows, &actx_rows, &vg, &ag, &v_time, &cg, &acg,
        );
        xv = v;
        xa = a;
    }

    let head = |x: &Value, s: &StreamHeads, mods: &StreamMods, g: &Geom| {
        let h = ops::elemwise::modulate(
            &ops::elemwise::layernorm_no_scale(x, NORM_EPS),
            &mods.head,
            Some(&g.lanes),
            ModulateForm::ScaleShift,
        );
        linear(&s.proj_out, &h)
    };
    let vv = head(&xv, &dit.video, &mods_v, &vg);
    let va = head(&xa, &dit.audio, &mods_a, &ag);
    let velocity = Value::merge(vec![vv, va]);
    seam::at(seam::VELOCITY, &[&velocity]);
    velocity
}

fn refine(arm: &Input<Facts>, conn: &Connector, text_in: u32, rescale: f32) {
    let g = Geom {
        lanes: arm.request_of_token(),
        positions: arm.axis_positions(port::TIME_POSITIONS, 1),
        perm: arm.row_permutation(),
        csr: arm.lane_indptr(),
    };
    let x = arm.latents(port::TEXT, text_in, Dtype::Bf16);
    let mut h = ops::elemwise::mul_scalar(rescale, &ops::linear::matmul(&x, &conn.aggregate.w));
    if let Some(bias) = &conn.aggregate.bias {
        h = ops::elemwise::add_bias(bias, &h);
    }
    let dims = conn.rope_dims();
    for (_, b) in arm.walk_layers(&conn.blocks) {
        let n = rms(&h);
        let o = self_attention(&n, &b.attn, dims, &g);
        h = ops::elemwise::residual_add(&o, &h);
        let n = rms(&h);
        let f = linear(
            &b.ffn.down,
            &ops::elemwise::gelu(&linear(&b.ffn.up, &n), true),
        );
        h = ops::elemwise::residual_add(&f, &h);
    }
    let out = rms(&h);
    seam::at(seam::HIDDEN, &[&out]);
}

fn conv(x: &Value, g: &Value, c: &VaeConv) -> (Value, Value) {
    spatial::conv3d(
        x,
        g,
        &c.w,
        Some(&c.bias),
        spatial::Conv::same3().replicate_time(),
        None,
    )
}

fn norm_silu(x: &Value) -> Value {
    let width = u32::try_from(x.width()).expect("a VAE row is narrower than 2^32");
    ops::elemwise::silu(&ops::elemwise::rmsnorm_no_scale(x, width, VAE_EPS))
}

fn resnet(x: &Value, g: &Value, r: &VaeResnet) -> Value {
    let (h, _) = conv(&norm_silu(x), g, &r.conv1);
    let (h, _) = conv(&norm_silu(&h), g, &r.conv2);
    ops::elemwise::add(x, &h)
}

pub fn vae_decode(arm: &Input<Facts>, vae: &Vae) -> Value {
    let mut g = arm.grid();
    let z = arm.voxels(port::VOXELS, VAE_Z, Dtype::Bf16);
    let z = ops::elemwise::mul_scalar(0.5, &ops::elemwise::add(&z, &z));
    let z = ops::elemwise::standardize(&z, &vae.zero, &vae.latents_std);
    let z = ops::elemwise::add_bias(&vae.latents_mean, &z);

    let (mut x, _) = conv(&z, &g, &vae.conv_in);
    for r in &vae.mid {
        x = resnet(&x, &g, r);
    }
    for up in &vae.up {
        let (y, gy) = conv(&x, &g, &up.upsampler);
        let (y, gy) = spatial::pixel_shuffle_trimming(&y, &gy, up.stride, up.stride[0] - 1);
        x = y;
        g = gy;
        for r in &up.resnets {
            x = resnet(&x, &g, r);
        }
    }

    let x = norm_silu(&x);
    let (y, gy) = conv(&x, &g, &vae.conv_out);
    let (pixels, gp) = spatial::pixel_shuffle(&y, &gy, [1, VAE_PATCH, VAE_PATCH]);
    seam::at(seam::PIXELS, &[&pixels, &gp]);
    pixels
}
