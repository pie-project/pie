use model_dsl::ops::spatial;
use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, ModulateForm, Platform, Predicate, Request,
    RopeForm, Stream, Value, Weight, ops, seam,
};

use crate::{
    Generative, LatentSpace, PortFact, PortKind, ReadingFact, ReadoutKind, ScheduleFact,
    ScheduleKind,
};

use super::model::{
    Conv, Dims, Embedder, GN_EPS, GN_GROUPS, LATENT_CHANNELS, Linear, Model, NORM_EPS, PATCH,
    ROPE_AXES, ROPE_THETA, ResBlock, SPATIAL_COMPRESSION, T_FLIP_SIN_COS, T_FREQ_DIM, T_MAX_PERIOD,
    T_SCALE, TRAIN_STEPS, port,
};

pub const STREAM_BASE: u8 = 0;

pub const READING_LO: u8 = 6;
pub const READING_HI: u8 = 7;
pub const QO_ONE: u8 = 8;

pub const ENCODE: u8 = 0;
pub const DENOISE: u8 = 1;
pub const IMAGE_IN: u8 = 2;
pub const IMAGE_OUT: u8 = 3;

impl Model {
    #[must_use]
    pub fn generative(&self) -> Generative {
        let d = &self.dims;
        let port = |name, kind, width, at| PortFact {
            name,
            kind,
            width,
            streams: vec![],
            at,
            rows: None,
        };
        let readings = vec![
            ReadingFact {
                name: "encode",
                index: ENCODE,
                has_kv: true,
                takes_tokens: true,
                streams: vec![Stream::Text],
                ports: vec![port(
                    "positions",
                    PortKind::AxisPositions,
                    u32::from(ROPE_AXES),
                    None,
                )],
                positions: None,
                readout: ReadoutKind::Logits,
                readout_width: d.vocab,
            },
            ReadingFact {
                name: "denoise",
                index: DENOISE,
                has_kv: true,
                takes_tokens: true,
                streams: vec![Stream::Image],
                ports: vec![
                    port("latents", PortKind::Latents, d.hidden, None),
                    port("special", PortKind::Latents, 1, None),
                    port("timestep", PortKind::LaneVector, 1, None),
                    port(
                        "positions",
                        PortKind::AxisPositions,
                        u32::from(ROPE_AXES),
                        None,
                    ),
                ],
                positions: None,
                readout: ReadoutKind::Hidden,
                readout_width: d.hidden,
            },
            ReadingFact {
                name: "image.in",
                index: IMAGE_IN,
                has_kv: false,
                takes_tokens: false,
                streams: vec![Stream::Image],
                ports: vec![port(
                    "latent",
                    PortKind::Voxels,
                    LATENT_CHANNELS + T_FREQ_DIM,
                    Some(port::LATENT_VOXELS),
                )],
                positions: None,
                readout: ReadoutKind::Pixels,
                readout_width: d.hidden,
            },
            ReadingFact {
                name: "image.out",
                index: IMAGE_OUT,
                has_kv: false,
                takes_tokens: false,
                streams: vec![Stream::Image],
                ports: vec![port(
                    "rows",
                    PortKind::Voxels,
                    d.hidden + T_FREQ_DIM,
                    Some(port::ROW_VOXELS),
                )],
                positions: None,
                readout: ReadoutKind::Pixels,
                readout_width: LATENT_CHANNELS,
            },
        ];
        Generative {
            readings,
            latent: Some(LatentSpace {
                channels: LATENT_CHANNELS,
                patch_t: 1,
                patch_h: PATCH,
                patch_w: PATCH,
                spatial_compression: SPATIAL_COMPRESSION,
                temporal_compression: 1,
            }),
            schedule: Some(ScheduleFact {
                kind: ScheduleKind::Flow,
                shift: super::model::FLOW_SHIFT,
                train_steps: TRAIN_STEPS,
                boundary: None,
                pinned_sigmas: vec![],
                stream_shifts: vec![],
            }),
            max_rows: if d.layers > 4 { 8192 } else { 1024 },
        }
    }
}

pub struct Facts {
    pub stream: Stream,
    pub reading: u8,
    pub qo_one: bool,
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
    pub fn reading_lo() -> Predicate {
        Predicate::fact(READING_LO)
    }

    #[must_use]
    pub fn reading_hi() -> Predicate {
        Predicate::fact(READING_HI)
    }

    #[must_use]
    pub fn qo_one() -> Predicate {
        Predicate::fact(QO_ONE)
    }
}

impl Classify for Facts {
    fn of(r: &Request) -> Facts {
        Facts {
            stream: r.stream(),
            reading: r.reading() & 3,
            qo_one: r.query_len() == 1,
        }
    }

    fn word(&self) -> u64 {
        self.stream.word(STREAM_BASE)
            | (u64::from(self.reading & 3) << READING_LO)
            | (u64::from(self.qo_one) << QO_ONE)
    }
}

impl ForwardHybrid for Model {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();
        let kv = c.kv_space(self.kv_dtype);
        let plane = u64::from(self.kv_width());
        for w in &self.layers {
            c.kv(kv, w.kv.clone(), [plane, plane]);
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let (hi, lo) = inputs.split(&Facts::reading_hi());
        let (image_out_arm, image_in_arm) = hi.split(&Facts::reading_lo());
        let (denoise_arm, encode_arm) = lo.split(&Facts::reading_lo());

        image_in(&image_in_arm, self);
        image_out(&image_out_arm, self);

        let _ = &encode_arm;
        trunk(&lo, &denoise_arm, self)
    }
}

fn linear(w: &Linear, x: &Value) -> Value {
    ops::elemwise::add_bias(&w.bias, &ops::linear::matmul(x, &w.w))
}

fn embedder(e: &Embedder, t_freq: &Value) -> Value {
    linear(
        &e.mlp_out,
        &ops::elemwise::gelu(&linear(&e.mlp_in, t_freq), true),
    )
}

fn trunk(all: &Input<Facts>, den: &Input<Facts>, m: &Model) -> Value {
    let d: &Dims = &m.dims;
    let hd = d.head_dim;
    let sm = d.sm_scale();
    let classes = [Facts::reading_lo(), Facts::qo_one(), Predicate::rest()];
    let [canvas_in, ar_decode, ar_prefill] = all.split(classes.clone());

    let plan_den = ops::attn::plan_prefill(&canvas_in, m.q_heads, m.kv_heads, hd, None);
    let plan_dec = ops::attn::plan_decode(&ar_decode, m.q_heads, m.kv_heads, hd, None);
    let plan_pre = ops::attn::plan_prefill(&ar_prefill, m.q_heads, m.kv_heads, hd, None);
    let mask = canvas_in.mask();

    let positions = all.axis_positions(port::POSITIONS, ROPE_AXES);
    let ids = all.tokens();
    let y = ops::layout::embed(&ids, &m.embed, d.vocab);
    let (_, encoded) = y.split(&Facts::reading_lo());
    let mut y = Value::merge(vec![canvas_rows(den, m), encoded]);

    for (l, w) in all.walk_layers(&m.layers) {
        let n = ops::elemwise::rmsnorm(&y, &w.attn_norm, NORM_EPS);
        let (q, k, v) =
            ops::layout::split_qkv(&ops::linear::matmul(&n, &w.qkv), m.q_width(), m.kv_width());
        let turn = |x: &Value| {
            ops::elemwise::rope_axes(
                x,
                &positions,
                d.rope_dims(),
                [ROPE_THETA; 4],
                RopeForm::Split,
                hd,
                hd,
            )
        };
        let q = ops::elemwise::rmsnorm_per_head(&turn(&q), &w.q_norm, hd, NORM_EPS);
        let k = ops::elemwise::rmsnorm_per_head(&turn(&k), &w.k_norm, hd, NORM_EPS);

        let pages = all.kv(&w.kv);
        ops::attn::kv_append(
            &k,
            &v,
            pages,
            &all.write_page(&w.kv),
            &all.write_offset(&w.kv),
        );

        let [dq, aq, pq] = q.split(classes.clone());
        let a = Value::merge(vec![
            ops::attn::masked(
                &dq, &plan_den, &mask, pages, None, hd, m.kv_heads, false, sm,
            ),
            ops::attn::decode(&aq, &plan_dec, pages, None, hd, sm),
            ops::attn::prefill(&pq, &plan_pre, pages, None, hd, m.kv_heads, sm),
        ]);
        let o = ops::linear::matmul(&a, &w.o_proj);
        let o = if m.tp > 1 {
            ops::collective::all_reduce(&o)
        } else {
            o
        };
        y = if l == 0 {
            ops::elemwise::add(&o, &y)
        } else {
            ops::elemwise::residual_add(&o, &y)
        };

        let n = ops::elemwise::rmsnorm(&y, &w.mlp_norm, NORM_EPS);
        let f = moe(&n, w, m);
        let f = if m.tp > 1 {
            ops::collective::all_reduce(&f)
        } else {
            f
        };
        y = ops::elemwise::residual_add(&f, &y);
    }

    let (canvas, text) = y.split(&Facts::reading_lo());
    seam::at(seam::HIDDEN, &[&canvas]);
    let (_, text_in) = all.split(&Facts::reading_lo());
    let x = ops::elemwise::rmsnorm(&text, &m.final_norm, NORM_EPS);
    let x = ops::layout::gather_rows(&x, &text_in.readout_rows());
    let logits = ops::linear::lm_head(&x, &m.head);
    seam::at(seam::OUT, &[&logits]);
    canvas
}

fn moe(x: &Value, w: &super::model::Layer, m: &Model) -> Value {
    let d = &m.dims;
    let (routes, weights) =
        ops::linear::moe_topk_softmax(&ops::linear::matmul(x, &w.router), d.experts, d.top_k);
    let select = |act: &Value, bank: &Weight| {
        if matches!(bank.dtype, Dtype::Bf16 | Dtype::F16 | Dtype::F32) {
            ops::linear::moe_matmul_select(act, bank, &routes, d.top_k)
        } else {
            ops::linear::moe_matmul_select_quant(act, bank, &routes, d.top_k)
        }
    };
    let act = ops::linear::mlp_swiglu(&select(x, &w.experts_gate_up), m.moe_inter);
    let routed = ops::linear::moe_weighted_sum(&select(&act, &w.experts_down), &weights);
    let shared = ops::linear::matmul(
        &ops::linear::mlp_swiglu(&ops::linear::matmul(x, &w.shared_gate_up), m.shared_inter),
        &w.shared_down,
    );
    ops::elemwise::residual_add(&shared, &routed)
}

fn canvas_rows(arm: &Input<Facts>, m: &Model) -> Value {
    let d = &m.dims;
    let u = arm.latents(port::ROWS, d.hidden, Dtype::Bf16);
    let flag = arm.latents(port::SPECIAL, 1, Dtype::Bf16);
    let lanes = arm.request_of_token();

    let t = arm.lane_vector(port::TIMESTEP, 1);
    let freqs = ops::elemwise::sinusoid(&t, T_FREQ_DIM, T_MAX_PERIOD, T_FLIP_SIN_COS, T_SCALE);
    let doubled = embedder(&m.timestep_emb, &freqs);

    let spread = ops::linear::matmul(&flag, &m.ones);
    let (pos, neg) = ops::layout::split_rows(&spread, d.hidden);
    let zero = ops::elemwise::add(&pos, &neg);
    let special = ops::elemwise::modulate(&zero, &doubled, Some(&lanes), ModulateForm::ScaleShift);

    let kept = ops::elemwise::modulate(&u, &neg, None, ModulateForm::Scale);
    ops::elemwise::add(&kept, &ops::elemwise::mul(&special, &pos))
}

fn conv(x: &Value, g: &Value, c: &Conv) -> (Value, Value) {
    let shape = spatial::Conv::conv3d(c.k, [1, 1, 1], [0, c.k[1] / 2, c.k[2] / 2]);
    spatial::conv3d(x, g, &c.w, Some(&c.bias), shape, None)
}

fn resblock(x: &Value, g: &Value, r: &ResBlock, temb: &Value) -> Value {
    let h = spatial::group_norm(
        x,
        g,
        GN_GROUPS,
        &r.norm_in.weight,
        &r.norm_in.bias,
        GN_EPS,
        true,
    );
    let (h, g1) = conv(&h, g, &r.conv_in);
    let m = linear(&r.emb, &ops::elemwise::silu(temb));
    let h = spatial::group_norm(
        &h,
        &g1,
        GN_GROUPS,
        &r.norm_out.weight,
        &r.norm_out.bias,
        GN_EPS,
        false,
    );
    let h = ops::elemwise::modulate(&h, &m, None, ModulateForm::ScaleShift);
    let h = ops::elemwise::silu(&h);
    let (h, _) = conv(&h, &g1, &r.conv_out);
    let skip = match &r.skip {
        Some(c) => conv(x, g, c).0,
        None => x.clone(),
    };
    ops::elemwise::add(&skip, &h)
}

fn image_in(arm: &Input<Facts>, m: &Model) {
    let g = arm.grid();
    let clip = arm.voxels(
        port::LATENT_VOXELS,
        LATENT_CHANNELS + T_FREQ_DIM,
        Dtype::Bf16,
    );
    let (z, freqs) = ops::layout::split_rows(&clip, LATENT_CHANNELS);
    let temb = embedder(&m.time_embed, &freqs);
    let (h, g1) = conv(&z, &g, &m.patch_embed.conv_in);
    let x = resblock(&h, &g1, &m.patch_embed.res, &temb);
    seam::at(seam::PIXELS, &[&x, &g1]);
}

fn image_out(arm: &Input<Facts>, m: &Model) {
    let g = arm.grid();
    let clip = arm.voxels(port::ROW_VOXELS, m.dims.hidden + T_FREQ_DIM, Dtype::Bf16);
    let (rows, freqs) = ops::layout::split_rows(&clip, m.dims.hidden);
    let temb = embedder(&m.time_embed_2, &freqs);
    let x = resblock(&rows, &g, &m.final_layer.res, &temb);
    let h = spatial::group_norm(
        &x,
        &g,
        GN_GROUPS,
        &m.final_layer.norm_out.weight,
        &m.final_layer.norm_out.bias,
        GN_EPS,
        true,
    );
    let (v, gv) = conv(&h, &g, &m.final_layer.conv_out);
    seam::at(seam::PIXELS, &[&v, &gv]);
}

#[allow(dead_code)]
fn platform_is_stated(_: Platform) {}
