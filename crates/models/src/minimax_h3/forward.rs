use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, ModulateForm, Predicate, RaggedMask,
    Request, RopeForm, Stream, Value, Weight, ops, seam,
};

use crate::{
    Generative, LatentSpace, PortFact, PortKind, ReadingFact, ReadoutKind, ScheduleFact,
    ScheduleKind,
};

use super::model::{
    ADALN_SLICES, AUDIO_CHANNELS, AUDIO_SHIFT, Attn, Block, CONDITION_TIMESTEP, Dims, Dit,
    FINAL_SLICES, Linear, Mlp, Model, NORM_EPS, PATCH_H, PATCH_T, PATCH_W, ROPE_AXES, ROPE_THETA,
    Refiner, SPATIAL_COMPRESSION, STEPS, T_FLIP_SIN_COS, T_MAX_PERIOD, T_SCALE, TE_HIDDEN,
    TE_LAYERS, TE_MAX_TOKENS, TEMPORAL_COMPRESSION, TIMESTEP_SLOTS, TRAIN_STEPS, TextEncoder,
    VIDEO_FEATURES, VIDEO_SHIFT, modality, port, timestep_slot,
};

pub const STREAM_BASE: u8 = 0;

pub const READING_LO: u8 = 6;
pub const READING_HI: u8 = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Readings {
    pub text: Option<u8>,
    pub refine: u8,
    pub denoise: u8,
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
        let refine = take();
        let denoise = take();
        Readings {
            text,
            refine,
            denoise,
        }
    }

    #[must_use]
    pub fn generative(&self) -> Generative {
        let d = &self.dims;
        let codes = self.readings();
        let port = |name, kind, at, width, streams: &[Stream]| PortFact {
            name,
            kind,
            width,
            streams: streams.to_vec(),
            at: Some(at),
            rows: None,
        };
        let mut readings = Vec::new();
        if let (Some(index), Some(te)) = (codes.text, &self.te) {
            readings.push(ReadingFact {
                name: "text",
                index,
                has_kv: true,
                takes_tokens: true,
                streams: vec![Stream::Text],
                ports: vec![],
                positions: None,
                readout: ReadoutKind::Hidden,
                readout_width: te.hidden,
            });
        }
        readings.push(ReadingFact {
            name: "refine",
            index: codes.refine,
            has_kv: false,
            takes_tokens: false,
            streams: vec![Stream::Text],
            ports: vec![port(
                "caption",
                PortKind::Context,
                port::CAPTION,
                d.text_dim,
                &[Stream::Text],
            )],
            positions: None,
            readout: ReadoutKind::Hidden,
            readout_width: d.dim,
        });
        let every = [
            Stream::Text,
            Stream::Video,
            Stream::Audio,
            Stream::Reference,
        ];
        readings.push(ReadingFact {
            name: "denoise",
            index: codes.denoise,
            has_kv: false,
            takes_tokens: false,
            streams: every.to_vec(),
            ports: vec![
                port(
                    "latents",
                    PortKind::Latents,
                    port::LATENTS,
                    VIDEO_FEATURES,
                    &[Stream::Video],
                ),
                port(
                    "reference",
                    PortKind::Latents,
                    port::REFERENCE,
                    VIDEO_FEATURES,
                    &[Stream::Reference],
                ),
                port(
                    "audio",
                    PortKind::Latents,
                    port::AUDIO,
                    AUDIO_CHANNELS,
                    &[Stream::Audio],
                ),
                port(
                    "context",
                    PortKind::Latents,
                    port::CONTEXT,
                    d.dim,
                    &[Stream::Text],
                ),
                port(
                    "timestep",
                    PortKind::LaneVector,
                    port::TIMESTEP,
                    TIMESTEP_SLOTS,
                    &every,
                ),
                port(
                    "positions",
                    PortKind::AxisPositions,
                    port::POSITIONS,
                    u32::from(ROPE_AXES),
                    &every,
                ),
            ],
            positions: None,
            readout: ReadoutKind::Velocity,
            readout_width: VIDEO_FEATURES,
        });
        Generative {
            readings,
            latent: Some(LatentSpace {
                channels: super::model::LATENT_CHANNELS,
                patch_t: PATCH_T,
                patch_h: PATCH_H,
                patch_w: PATCH_W,
                spatial_compression: SPATIAL_COMPRESSION,
                temporal_compression: TEMPORAL_COMPRESSION,
            }),
            schedule: Some(ScheduleFact {
                kind: ScheduleKind::Flow,
                shift: VIDEO_SHIFT,
                train_steps: TRAIN_STEPS,
                boundary: None,
                pinned_sigmas: shifted_sigmas(VIDEO_SHIFT, STEPS),
                stream_shifts: vec![
                    (Stream::Video, VIDEO_SHIFT),
                    (Stream::Audio, AUDIO_SHIFT),
                    (Stream::Reference, 1.0),
                ],
            }),
            max_rows: match self.te {
                Some(_) => 131_072,
                None => 4096,
            },
        }
    }
}

#[must_use]
pub fn shifted_sigmas(shift: f32, steps: u32) -> Vec<f32> {
    let n = steps.max(2);
    (0..n - 1)
        .map(|i| {
            let base = 1.0 - f64::from(i) / f64::from(n - 1);
            let s = f64::from(shift);
            (s * base / (1.0 + (s - 1.0) * base)) as f32
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
    pub fn video() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Video)
    }

    #[must_use]
    pub fn audio() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Audio)
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
                c.kv(kv, layer.kv.clone(), [plane, plane], te.head_dim);
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
        refine(arm(codes.refine), &self.dims, &self.dit);
        denoise(arm(codes.denoise), &self.dims, &self.dit)
    }
}

fn text_encode(arm: &Input<Facts>, te: &TextEncoder) {
    let plan = ops::attn::plan_prefill(arm, te.q_heads, te.kv_heads, te.head_dim, None);
    let ids = arm.tokens();
    let positions = arm.positions();
    let mut y = ops::layout::embed(&ids, &te.embed, te.vocab);
    let last = te.layers.len() - 1;
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
        if l as usize == last {
            seam::at(seam::HIDDEN, &[&y]);
        }
    }
}

struct Geom {
    perm: Value,
    csr: Value,
    mask: RaggedMask,
}

fn linear(w: &Linear, x: &Value) -> Value {
    ops::elemwise::add_bias(&w.bias, &ops::linear::matmul(x, &w.w))
}

fn turn(x: &Value, gain: &Weight, positions: &Value, d: &Dims) -> Value {
    ops::elemwise::rope_axes(
        &ops::elemwise::rmsnorm_per_head(x, gain, d.head_dim, NORM_EPS),
        positions,
        d.rope_dims(),
        [ROPE_THETA; 4],
        RopeForm::Neox,
        d.rotary_dim(),
        d.head_dim,
    )
}

fn attention(x: &Value, a: &Attn, d: &Dims, g: &Geom, positions: Option<&Value>) -> Value {
    let inner = d.inner();
    let (q, k, v) = ops::layout::split_qkv(&ops::linear::matmul(x, &a.qkv), inner, inner);
    let (q, k) = match positions {
        Some(p) => (turn(&q, &a.q_norm, p, d), turn(&k, &a.k_norm, p, d)),
        None => (
            ops::elemwise::rmsnorm_per_head(&q, &a.q_norm, d.head_dim, NORM_EPS),
            ops::elemwise::rmsnorm_per_head(&k, &a.k_norm, d.head_dim, NORM_EPS),
        ),
    };
    let o = ops::attn::ragged(
        &ops::layout::pack_rows(&q, &g.perm),
        &ops::layout::pack_rows(&k, &g.perm),
        &ops::layout::pack_rows(&v, &g.perm),
        &g.csr,
        &g.csr,
        d.head_dim,
        d.sm_scale(),
        g.mask,
    );
    ops::linear::matmul(&ops::layout::unpack_rows(&o, &g.perm), &a.out)
}

fn mlp(x: &Value, m: &Mlp, d: &Dims) -> Value {
    ops::linear::matmul(
        &ops::linear::mlp_swiglu(&ops::linear::matmul(x, &m.fc1), d.inter),
        &m.fc2,
    )
}

fn refine(arm: &Input<Facts>, d: &Dims, m: &Dit) {
    let mut x = linear(&m.condition, &arm.context(port::CAPTION, d.text_dim));
    let geom = Geom {
        perm: arm.row_permutation(),
        csr: arm.lane_indptr(),
        mask: RaggedMask::None,
    };
    let last = m.refine.len() - 1;
    for (l, block) in arm.walk_layers(&m.refine) {
        x = refiner_block(&x, block, d, &geom);
        if l as usize == last {
            let y = ops::elemwise::rmsnorm(&x, &m.refine_norm, NORM_EPS);
            seam::at(seam::HIDDEN, &[&y]);
        }
    }
}

fn refiner_block(x: &Value, b: &Refiner, d: &Dims, g: &Geom) -> Value {
    let h = ops::elemwise::rmsnorm(x, &b.norm1, NORM_EPS);
    let x = ops::elemwise::residual_add(&attention(&h, &b.attn, d, g, None), x);
    let h = ops::elemwise::rmsnorm(&x, &b.norm2, NORM_EPS);
    ops::elemwise::residual_add(&mlp(&h, &b.mlp, d), &x)
}

struct Mods {
    attn: Value,
    attn_gate: Value,
    mlp: Value,
    mlp_gate: Value,
}

fn adaln6(m: &Value, dim: u32) -> Mods {
    debug_assert_eq!(m.width(), u64::from(ADALN_SLICES * dim));
    let (attn, rest) = ops::layout::split_rows(m, 2 * dim);
    let (attn_gate, rest) = ops::layout::split_rows(&rest, dim);
    let (mlp, mlp_gate) = ops::layout::split_rows(&rest, 2 * dim);
    Mods {
        attn,
        attn_gate,
        mlp,
        mlp_gate,
    }
}

fn column(v: &Value, slot: u32, total: u32) -> Value {
    debug_assert!(slot < total);
    let tail = if slot == 0 {
        v.clone()
    } else {
        ops::layout::split_rows(v, slot).1
    };
    if slot + 1 == total {
        tail
    } else {
        ops::layout::split_rows(&tail, 1).0
    }
}

struct Side {
    arm: Input<Facts>,
    stream: Stream,
    stemb: Value,
}

impl Side {
    fn modulation(&self, block: &Block) -> Value {
        linear(&block.adaln[modality(self.stream)], &self.stemb)
    }
}

fn denoise(arm: &Input<Facts>, d: &Dims, m: &Dit) -> Value {
    let (text, rest) = arm.split(&Facts::text());
    let (video, rest) = rest.split(&Facts::video());
    let (audio, reference) = rest.split(&Facts::audio());

    let lanes = arm.request_of_token();
    let positions = arm.axis_positions(port::POSITIONS, ROPE_AXES);
    let joint = Geom {
        perm: arm.row_permutation(),
        csr: arm.group_indptr(),
        mask: RaggedMask::GroupBlockDiagonal,
    };

    let side = |arm: Input<Facts>, stream: Stream| {
        let t = column(
            &arm.lane_vector(port::TIMESTEP, TIMESTEP_SLOTS),
            timestep_slot(stream),
            TIMESTEP_SLOTS,
        );
        let e = ops::elemwise::sinusoid(&t, d.t_freq, T_MAX_PERIOD, T_FLIP_SIN_COS, T_SCALE);
        let e = linear(&m.t_out, &ops::elemwise::silu(&linear(&m.t_in, &e)));
        Side {
            arm,
            stream,
            stemb: ops::elemwise::silu(&e),
        }
    };
    let sides = [
        side(text, Stream::Text),
        side(video, Stream::Video),
        side(audio, Stream::Audio),
        side(reference, Stream::Reference),
    ];

    let ctx = sides[0].arm.latents(port::CONTEXT, d.dim, Dtype::Bf16);
    let ctx_perm = sides[0].arm.row_permutation();
    let text_rows = ops::layout::unpack_rows(&ops::layout::pack_rows(&ctx, &ctx_perm), &ctx_perm);
    let video_rows = linear(
        &m.video_patch,
        &sides[1]
            .arm
            .latents(port::LATENTS, VIDEO_FEATURES, Dtype::Bf16),
    );
    let audio_rows = linear(
        &m.audio_patch,
        &sides[2]
            .arm
            .latents(port::AUDIO, AUDIO_CHANNELS, Dtype::Bf16),
    );
    let reference_rows = linear(
        &m.video_patch,
        &sides[3]
            .arm
            .latents(port::REFERENCE, VIDEO_FEATURES, Dtype::Bf16),
    );
    let mut x = Value::merge(vec![text_rows, video_rows, audio_rows, reference_rows]);

    for (_, block) in arm.walk_layers(&m.blocks) {
        let mods = adaln6(
            &Value::merge(sides.iter().map(|s| s.modulation(block)).collect()),
            d.dim,
        );
        let h = ops::elemwise::modulate(
            &ops::elemwise::rmsnorm(&x, &block.norm1, NORM_EPS),
            &mods.attn,
            Some(&lanes),
            ModulateForm::ScaleShift,
        );
        x = ops::elemwise::gated_residual_add(
            &x,
            &mods.attn_gate,
            &attention(&h, &block.attn, d, &joint, Some(&positions)),
            Some(&lanes),
        );
        let h = ops::elemwise::modulate(
            &ops::elemwise::rmsnorm(&x, &block.norm2, NORM_EPS),
            &mods.mlp,
            Some(&lanes),
            ModulateForm::ScaleShift,
        );
        x = ops::elemwise::gated_residual_add(
            &x,
            &mods.mlp_gate,
            &mlp(&h, &block.mlp, d),
            Some(&lanes),
        );
    }

    let final_mod = Value::merge(
        sides
            .iter()
            .map(|s| linear(&m.final_adaln, &s.stemb))
            .collect(),
    );
    debug_assert_eq!(final_mod.width(), u64::from(FINAL_SLICES * d.dim));
    let h = ops::elemwise::modulate(
        &ops::elemwise::rmsnorm(&x, &m.final_norm, NORM_EPS),
        &final_mod,
        Some(&lanes),
        ModulateForm::ScaleShift,
    );
    let (_, rest) = h.split(&Facts::text());
    let (h_video, rest) = rest.split(&Facts::video());
    let (h_audio, _) = rest.split(&Facts::audio());
    let velocity = linear(&m.video_out, &h_video);
    seam::at(seam::VELOCITY, &[&velocity]);
    let audio_velocity = linear(&m.audio_out, &h_audio);
    seam::at(seam::HIDDEN, &[&audio_velocity]);
    velocity
}

const _: () = assert!(Dims::h3(1).text_dim == TE_HIDDEN);
const _: () = assert!(TE_MAX_TOKENS > 0);
const _: () = assert!(CONDITION_TIMESTEP > 0.0);
