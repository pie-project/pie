use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, ModulateForm, Predicate, RaggedMask,
    Request, RopeForm, Stream, Value, Weight, ops, seam,
};

use crate::{
    AxisRole, Generative, LatentSpace, PortFact, PortKind, PositionConvention, ReadingFact,
    ReadoutKind, ScheduleFact, ScheduleKind,
};

use super::model::{
    Attn, DOUBLE_MOD_SLICES, Dit, Embedder, GUIDANCE_SCALE, HEAD_DIM, IN_CHANNELS, Model, NORM_EPS,
    REFERENCE_TIME_STRIDE, ROPE_AXES, ROPE_DIMS, ROPE_THETA, SINGLE_MOD_SLICES, SM_SCALE, Swiglu,
    T_FLIP_SIN_COS, T_FREQ_DIM, T_MAX_PERIOD, T_SCALE, TE_LAYERS, TE_MAX_TOKENS, TE_TAPS,
    TOKEN_COMPRESSION, TRAIN_STEPS, TextEncoder, port,
};

pub const STREAM_BASE: u8 = 0;

pub const READING_LO: u8 = 6;
pub const READING_HI: u8 = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Readings {
    pub text: Option<u8>,
    pub denoise: u8,
    pub vae_decode: Option<u8>,
    pub vae_encode: Option<u8>,
}

impl Model {
    #[must_use]
    pub fn readings(&self) -> Readings {
        let mut next = 0u8;
        let mut take = || {
            let code = next;
            next += 1;
            code
        };
        let text = self.te.as_ref().map(|_| take());
        let denoise = take();
        let vae_decode = self.vae.as_ref().map(|_| take());
        let vae_encode = self.vae.as_ref().map(|_| take());
        Readings {
            text,
            denoise,
            vae_decode,
            vae_encode,
        }
    }

    #[must_use]
    pub fn generative(&self) -> Generative {
        let d = &self.dims;
        let codes = self.readings();
        let port = |name, kind, width, streams: &[Stream]| PortFact {
            name,
            kind,
            width,
            streams: streams.to_vec(),
            at: None,
            rows: None,
        };
        let mut readings = Vec::new();
        if let (Some(index), Some(_)) = (codes.text, &self.te) {
            readings.push(ReadingFact {
                name: "text",
                index,
                has_kv: true,
                takes_tokens: true,
                streams: vec![Stream::Text],
                ports: vec![],
                positions: None,
                readout: ReadoutKind::Hidden,
                readout_width: if te_tap().is_some() {
                    super::model::TE_HIDDEN
                } else {
                    d.dim
                },
            });
        }
        let image_side = [Stream::Image, Stream::Reference];
        let every = [Stream::Text, Stream::Image, Stream::Reference];
        let mut ports = vec![
            port("latents", PortKind::Latents, IN_CHANNELS, &image_side),
            port(
                "context",
                PortKind::Context,
                if self.te.is_some() {
                    d.dim
                } else {
                    d.context_in
                },
                &[Stream::Text],
            ),
            port("timestep", PortKind::LaneVector, 1, &every),
        ];
        if d.guidance_embeds {
            ports.push(port("guidance", PortKind::LaneVector, 1, &every));
        }
        ports.push(port(
            "positions",
            PortKind::AxisPositions,
            u32::from(ROPE_AXES),
            &every,
        ));
        readings.push(ReadingFact {
            name: "denoise",
            index: codes.denoise,
            has_kv: false,
            takes_tokens: false,
            streams: every.to_vec(),
            ports,
            positions: Some(PositionConvention {
                axes: vec![
                    AxisRole::Time,
                    AxisRole::Height,
                    AxisRole::Width,
                    AxisRole::Index,
                ],
                text_axis: 3,
                text_origin: 0,
                image_follows_text: false,
                reference_stride: Some(REFERENCE_TIME_STRIDE),
            }),
            readout: ReadoutKind::Velocity,
            readout_width: IN_CHANNELS,
        });
        if let (Some(decode), Some(encode), Some(_)) =
            (codes.vae_decode, codes.vae_encode, &self.vae)
        {
            readings.push(ReadingFact {
                name: "vae.decode",
                index: decode,
                has_kv: false,
                takes_tokens: false,
                streams: vec![Stream::Image],
                ports: vec![port(
                    "latent",
                    PortKind::Voxels,
                    IN_CHANNELS,
                    &[Stream::Image],
                )],
                positions: None,
                readout: ReadoutKind::Pixels,
                readout_width: super::vae::RGB,
            });
            readings.push(ReadingFact {
                name: "vae.encode",
                index: encode,
                has_kv: false,
                takes_tokens: false,
                streams: vec![Stream::Image],
                ports: vec![PortFact {
                    name: "pixels",
                    kind: PortKind::Voxels,
                    width: super::vae::RGB,
                    streams: vec![Stream::Image],
                    at: Some(port::PIXEL_VOXELS),
                    rows: None,
                }],
                positions: None,
                readout: ReadoutKind::Pixels,
                readout_width: IN_CHANNELS,
            });
        }
        Generative {
            readings,
            latent: Some(LatentSpace {
                channels: IN_CHANNELS,
                patch_t: 1,
                patch_h: 1,
                patch_w: 1,
                spatial_compression: TOKEN_COMPRESSION,
                temporal_compression: 1,
            }),
            schedule: Some(ScheduleFact {
                kind: ScheduleKind::Flow,
                shift: empirical_mu(4096, 4).exp(),
                train_steps: TRAIN_STEPS,
                boundary: None,
                pinned_sigmas: sigmas(4096, 4),
                stream_shifts: vec![],
            }),
            max_rows: match self.te {
                Some(_) => 5 * 4096 + TE_MAX_TOKENS,
                None => 4096,
            },
        }
    }
}

#[must_use]
pub fn empirical_mu(image_rows: u32, steps: u32) -> f32 {
    const A1: f64 = 8.738_095_24e-5;
    const B1: f64 = 1.898_333_33;
    const A2: f64 = 0.000_169_27;
    const B2: f64 = 0.456_666_66;
    let rows = f64::from(image_rows);
    if image_rows > 4300 {
        return (A2 * rows + B2) as f32;
    }
    let m_200 = A2 * rows + B2;
    let m_10 = A1 * rows + B1;
    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    (a * f64::from(steps) + b) as f32
}

#[must_use]
pub fn sigmas(image_rows: u32, steps: u32) -> Vec<f32> {
    let steps = steps.max(1);
    let shift = f64::from(empirical_mu(image_rows, steps)).exp();
    (0..steps)
        .map(|i| {
            let n = f64::from(steps);
            let sigma = 1.0 - f64::from(i) * (1.0 - 1.0 / n) / (n - 1.0).max(1.0);
            (shift / (shift + 1.0 / sigma - 1.0)) as f32
        })
        .collect()
}

pub struct Facts {
    pub stream: Stream,
    pub reading: u8,
}

impl Facts {
    #[must_use]
    pub fn text() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Text)
    }

    #[must_use]
    pub fn image() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Image)
    }

    #[must_use]
    pub fn reference() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Reference)
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
        let mut c = HybridSpec::new();
        if let Some(te) = &self.te {
            let kv = c.kv_space(self.kv);
            let plane = u64::from(te.kv_heads) * u64::from(te.head_dim);
            for layer in &te.layers {
                c.kv(kv, layer.kv.clone(), [plane, plane]);
            }
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let codes = self.readings();
        let (hi, lo) = inputs.split(&Facts::reading_hi());
        let (c3, c2) = hi.split(&Facts::reading_lo());
        let (c1, c0) = lo.split(&Facts::reading_lo());
        let arms = [c0, c1, c2, c3];
        let arm = |code: u8| &arms[usize::from(code)];

        if let (Some(code), Some(te)) = (codes.text, &self.te) {
            text_encode(arm(code), te);
        }
        let velocity = denoise(arm(codes.denoise), self);
        if let (Some(decode), Some(encode), Some(vae)) =
            (codes.vae_decode, codes.vae_encode, &self.vae)
        {
            super::vae::decode(arm(decode), vae);
            super::vae::encode(arm(encode), vae);
        }
        velocity
    }
}

/// `PIE_FLUX2_TE_TAP=<n>`: export the encoder's residual stream after block
/// `n` (0 = the embedding) on the hidden seam instead of the projected taps.
#[must_use]
pub fn te_tap() -> Option<u32> {
    std::env::var("PIE_FLUX2_TE_TAP").ok()?.parse().ok()
}

fn text_encode(arm: &Input<Facts>, te: &TextEncoder) {
    let plan = ops::attn::plan_prefill(arm, te.q_heads, te.kv_heads, te.head_dim, None);
    let ids = arm.tokens();
    let positions = arm.positions();
    let mut y = ops::layout::embed(&ids, &te.embed, te.vocab);
    let tap_at = te_tap();
    if tap_at == Some(0) {
        seam::at(seam::HIDDEN, &[&y]);
        return;
    }
    let mut ctx: Option<Value> = None;
    debug_assert_eq!(te.layers.len(), TE_LAYERS as usize);
    for (l, w) in arm.walk_layers(&te.layers) {
        let pages = arm.kv(&w.kv);
        let x = ops::elemwise::rmsnorm(&y, &w.attn_norm, te.eps);
        let q = ops::linear::matmul(&x, &w.q);
        let k = ops::linear::matmul(&x, &w.k);
        let v = ops::linear::matmul(&x, &w.v);
        let q = ops::elemwise::rmsnorm_per_head(&q, &w.q_norm, te.head_dim, te.eps);
        let k = ops::elemwise::rmsnorm_per_head(&k, &w.k_norm, te.head_dim, te.eps);
        let (q, k) = ops::elemwise::rope_full(&q, &k, &positions, te.head_dim, te.theta, false);
        ops::attn::kv_append(
            &k,
            &v,
            pages,
            &arm.write_page(&w.kv),
            &arm.write_offset(&w.kv),
        );
        let o = ops::attn::prefill(
            &q,
            &plan,
            pages,
            None,
            te.head_dim,
            te.kv_heads,
            te.sm_scale,
        );
        let o = ops::linear::matmul(&o, &w.o);
        y = ops::elemwise::residual_add(&o, &y);

        let x = ops::elemwise::rmsnorm(&y, &w.mlp_norm, te.eps);
        let f = ops::linear::matmul(
            &ops::linear::mlp_swiglu(&ops::linear::matmul(&x, &w.gate_up), te.inter),
            &w.down,
        );
        y = ops::elemwise::residual_add(&f, &y);

        if tap_at == Some(l + 1) {
            seam::at(seam::HIDDEN, &[&y]);
            return;
        }
        if let Some(tap) = TE_TAPS.iter().position(|&k| k == l + 1) {
            let part = ops::linear::matmul(&y, &te.context_embed[tap]);
            let sum = match ctx.take() {
                None => part,
                Some(acc) => ops::elemwise::residual_add(&part, &acc),
            };
            if tap + 1 == TE_TAPS.len() {
                seam::at(seam::HIDDEN, &[&sum]);
            }
            ctx = Some(sum);
        }
    }
}

struct Mod {
    scale_shift: Value,
    gate: Value,
}

struct DoubleMod {
    attn: Mod,
    mlp: Mod,
}

fn adaln6(m: &Value, dim: u32) -> DoubleMod {
    debug_assert_eq!(m.width(), u64::from(DOUBLE_MOD_SLICES * dim));
    let (a_ss, rest) = ops::layout::split_rows(m, 2 * dim);
    let (a_gate, rest) = ops::layout::split_rows(&rest, dim);
    let (m_ss, m_gate) = ops::layout::split_rows(&rest, 2 * dim);
    DoubleMod {
        attn: Mod {
            scale_shift: a_ss,
            gate: a_gate,
        },
        mlp: Mod {
            scale_shift: m_ss,
            gate: m_gate,
        },
    }
}

fn adaln3(m: &Value, dim: u32) -> Mod {
    debug_assert_eq!(m.width(), u64::from(SINGLE_MOD_SLICES * dim));
    let (scale_shift, gate) = ops::layout::split_rows(m, 2 * dim);
    Mod { scale_shift, gate }
}

struct Joint {
    perm: Value,
    csr: Value,
}

fn embed(e: &Embedder, x: &Value) -> Value {
    let h = ops::elemwise::silu(&ops::linear::matmul(x, &e.linear_1));
    ops::linear::matmul(&h, &e.linear_2)
}

fn norm_modulate(x: &Value, scale_shift: &Value, lanes: &Value) -> Value {
    ops::elemwise::modulate(
        &ops::elemwise::layernorm_no_scale(x, NORM_EPS),
        scale_shift,
        Some(lanes),
        ModulateForm::ScaleShift,
    )
}

fn turn(x: &Value, gain: &Weight, positions: &Value) -> Value {
    ops::elemwise::rope_axes(
        &ops::elemwise::rmsnorm_per_head(x, gain, HEAD_DIM, NORM_EPS),
        positions,
        ROPE_DIMS,
        [ROPE_THETA; 4],
        RopeForm::Interleaved,
        HEAD_DIM,
        HEAD_DIM,
    )
}

fn heads(
    x: &Value,
    attn: &Attn,
    m: &Mod,
    dim: u32,
    lanes: &Value,
    positions: &Value,
) -> (Value, Value, Value) {
    let h = norm_modulate(x, &m.scale_shift, lanes);
    let (q, k, v) = ops::layout::split_qkv(&ops::linear::matmul(&h, &attn.qkv), dim, dim);
    (
        turn(&q, &attn.q_norm, positions),
        turn(&k, &attn.k_norm, positions),
        v,
    )
}

fn joint_attention(q: &Value, k: &Value, v: &Value, j: &Joint) -> Value {
    let o = ops::attn::ragged(
        &ops::layout::pack_rows(q, &j.perm),
        &ops::layout::pack_rows(k, &j.perm),
        &ops::layout::pack_rows(v, &j.perm),
        &j.csr,
        &j.csr,
        HEAD_DIM,
        SM_SCALE,
        RaggedMask::GroupBlockDiagonal,
    );
    ops::layout::unpack_rows(&o, &j.perm)
}

fn ff_sublayer(x: &Value, ff: &Swiglu, m: &Mod, inter: u32, lanes: &Value) -> Value {
    let h = norm_modulate(x, &m.scale_shift, lanes);
    let h = ops::linear::mlp_swiglu(&ops::linear::matmul(&h, &ff.linear_in), inter);
    ops::elemwise::gated_residual_add(
        x,
        &m.gate,
        &ops::linear::matmul(&h, &ff.linear_out),
        Some(lanes),
    )
}

fn denoise(arm: &Input<Facts>, m: &Model) -> Value {
    let d = &m.dims;
    let dit: &Dit = &m.dit;
    let dim = d.dim;

    let (txt_in, img_in) = arm.split(&Facts::text());

    let lanes = arm.request_of_token();
    let positions = arm.axis_positions(port::POSITIONS, ROPE_AXES);
    let joint = Joint {
        perm: arm.row_permutation(),
        csr: arm.group_indptr(),
    };

    let t = arm.lane_vector(port::TIMESTEP, 1);
    let temb = embed(
        &dit.t_embed,
        &ops::elemwise::sinusoid(&t, T_FREQ_DIM, T_MAX_PERIOD, T_FLIP_SIN_COS, T_SCALE),
    );
    let temb = match &dit.g_embed {
        Some(g_embed) => {
            let g = arm.lane_vector(port::GUIDANCE, 1);
            let gemb = embed(
                g_embed,
                &ops::elemwise::sinusoid(
                    &g,
                    T_FREQ_DIM,
                    T_MAX_PERIOD,
                    T_FLIP_SIN_COS,
                    GUIDANCE_SCALE,
                ),
            );
            ops::elemwise::add(&temb, &gemb)
        }
        None => temb,
    };
    let stemb = ops::elemwise::silu(&temb);
    let (mod_txt, _) = ops::linear::matmul(&stemb, &dit.mod_txt).split(&Facts::text());
    let (_, mod_img) = ops::linear::matmul(&stemb, &dit.mod_img).split(&Facts::text());
    let mod_txt = adaln6(&mod_txt, dim);
    let mod_img = adaln6(&mod_img, dim);
    let mod_single = adaln3(&ops::linear::matmul(&stemb, &dit.mod_single), dim);
    let mod_out = ops::linear::matmul(&stemb, &dit.norm_out);
    let (lanes_txt, lanes_img) = lanes.split(&Facts::text());
    let (pos_txt, pos_img) = positions.split(&Facts::text());

    let mut txt = match &dit.context_embed {
        Some(w) => ops::linear::matmul(&txt_in.context(port::CONTEXT, d.context_in), w),
        None => {
            let c = txt_in.context(port::CONTEXT, dim);
            let perm = txt_in.row_permutation();
            ops::layout::unpack_rows(&ops::layout::pack_rows(&c, &perm), &perm)
        }
    };
    let mut img = ops::linear::matmul(
        &img_in.latents(port::LATENTS, IN_CHANNELS, Dtype::Bf16),
        &dit.x_embed,
    );

    for (_, block) in arm.walk_layers(&dit.double) {
        let (tq, tk, tv) = heads(
            &txt,
            &block.txt.attn,
            &mod_txt.attn,
            dim,
            &lanes_txt,
            &pos_txt,
        );
        let (iq, ik, iv) = heads(
            &img,
            &block.img.attn,
            &mod_img.attn,
            dim,
            &lanes_img,
            &pos_img,
        );
        let o = joint_attention(
            &Value::merge(vec![tq, iq]),
            &Value::merge(vec![tk, ik]),
            &Value::merge(vec![tv, iv]),
            &joint,
        );
        let (o_txt, o_img) = o.split(&Facts::text());
        txt = ops::elemwise::gated_residual_add(
            &txt,
            &mod_txt.attn.gate,
            &ops::linear::matmul(&o_txt, &block.txt.attn.out),
            Some(&lanes_txt),
        );
        img = ops::elemwise::gated_residual_add(
            &img,
            &mod_img.attn.gate,
            &ops::linear::matmul(&o_img, &block.img.attn.out),
            Some(&lanes_img),
        );
        txt = ff_sublayer(&txt, &block.txt.ff, &mod_txt.mlp, d.inter, &lanes_txt);
        img = ff_sublayer(&img, &block.img.ff, &mod_img.mlp, d.inter, &lanes_img);
    }

    let mut x = Value::merge(vec![txt, img]);
    for (_, block) in arm.walk_layers(&dit.single) {
        let h = norm_modulate(&x, &mod_single.scale_shift, &lanes);
        let proj = ops::linear::matmul(&h, &block.in_proj);
        let (qkv, mlp) = ops::layout::split_rows(&proj, 3 * dim);
        let (q, k, v) = ops::layout::split_qkv(&qkv, dim, dim);
        let o = joint_attention(
            &turn(&q, &block.q_norm, &positions),
            &turn(&k, &block.k_norm, &positions),
            &v,
            &joint,
        );
        let a = ops::linear::matmul(&o, &block.out_attn);
        let f = ops::linear::matmul(&ops::linear::mlp_swiglu(&mlp, d.inter), &block.out_mlp);
        x = ops::elemwise::gated_residual_add(
            &x,
            &mod_single.gate,
            &ops::elemwise::residual_add(&a, &f),
            Some(&lanes),
        );
    }

    let (_, img_all) = x.split(&Facts::text());
    let (target, _refs) = img_all.split(&Facts::image());
    let (mod_out, _) = mod_out.split(&Facts::text()).1.split(&Facts::image());
    let (lanes_target, _) = lanes_img.split(&Facts::image());
    let h = norm_modulate(&target, &mod_out, &lanes_target);
    let velocity = ops::linear::matmul(&h, &dit.proj_out);
    seam::at(seam::VELOCITY, &[&velocity]);
    velocity
}
