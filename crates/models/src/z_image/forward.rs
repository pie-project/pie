use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, ModulateForm, Predicate, RaggedMask,
    Request, RopeForm, Stream, Value, Weight, ops, seam,
};

use crate::{
    AxisRole, Generative, LatentSpace, PortFact, PortKind, PositionConvention, ReadingFact,
    ReadoutKind, ScheduleFact, ScheduleKind,
};

use super::model::{
    ADALN_DIM, Block, Dims, Dit, FINAL_LN_EPS, Linear, MOD_SLICES, Model, NORM_EPS, PATCH_FEATURES,
    ROPE_AXES, ROPE_THETA, T_FLIP_SIN_COS, T_FREQ_DIM, T_MAX_PERIOD, TE_HIDDEN, TE_LAYERS,
    TE_MAX_TOKENS, TRAIN_STEPS, TextEncoder, port,
};
use super::model::{CHANNELS, PATCH, SPATIAL_COMPRESSION};

pub const STREAM_BASE: u8 = 0;

pub const READING_LO: u8 = 6;
pub const READING_MID: u8 = 7;
pub const READING_HI: u8 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Readings {
    pub text: Option<u8>,
    pub refine: u8,
    pub denoise: u8,
    pub vae_decode: Option<u8>,
    pub vae_encode: Option<u8>,
}

impl Model {
    #[must_use]
    pub fn readings(&self) -> Readings {
        let mut next = 0u8;
        let mut take = |present: bool| {
            present.then(|| {
                next += 1;
                next - 1
            })
        };
        let text = take(self.te.is_some());
        let refine = take(true).unwrap_or(0);
        let denoise = take(true).unwrap_or(0);
        let vae_decode = take(self.vae.is_some());
        let vae_encode = take(self.vae.is_some());
        Readings {
            text,
            refine,
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
        let axes = u32::from(ROPE_AXES);
        readings.push(ReadingFact {
            name: "refine",
            index: codes.refine,
            has_kv: false,
            takes_tokens: false,
            streams: vec![Stream::Context],
            ports: vec![
                port("pad", PortKind::Latents, 1, &[Stream::Context]),
                port(
                    "caption",
                    PortKind::Context,
                    d.cap_width,
                    &[Stream::Context],
                ),
                port(
                    "positions",
                    PortKind::AxisPositions,
                    axes,
                    &[Stream::Context],
                ),
            ],
            positions: Some(PositionConvention {
                axes: vec![AxisRole::Time, AxisRole::Height, AxisRole::Width],
                text_axis: 0,
                text_origin: 1,
                image_follows_text: false,
                reference_stride: None,
            }),
            readout: ReadoutKind::Hidden,
            readout_width: d.dim,
        });
        readings.push(ReadingFact {
            name: "denoise",
            index: codes.denoise,
            has_kv: false,
            takes_tokens: false,
            streams: vec![Stream::Image, Stream::Context],
            ports: vec![
                port("pad", PortKind::Latents, 1, &[Stream::Image]),
                port(
                    "latents",
                    PortKind::Latents,
                    PATCH_FEATURES,
                    &[Stream::Image],
                ),
                port("context", PortKind::Latents, d.dim, &[Stream::Context]),
                port(
                    "timestep",
                    PortKind::LaneVector,
                    1,
                    &[Stream::Image, Stream::Context],
                ),
                port(
                    "positions",
                    PortKind::AxisPositions,
                    axes,
                    &[Stream::Image, Stream::Context],
                ),
            ],
            positions: Some(PositionConvention {
                axes: vec![AxisRole::Time, AxisRole::Height, AxisRole::Width],
                text_axis: 0,
                text_origin: 1,
                image_follows_text: true,
                reference_stride: None,
            }),
            readout: ReadoutKind::Velocity,
            readout_width: Tap::width(Tap::from_env().as_deref(), d),
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
                ports: vec![port("latent", PortKind::Voxels, CHANNELS, &[Stream::Image])],
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
                    at: Some(super::model::port::PIXEL_VOXELS),
                    rows: None,
                }],
                positions: None,
                readout: ReadoutKind::Pixels,
                readout_width: CHANNELS,
            });
        }
        Generative {
            readings,
            latent: Some(LatentSpace {
                channels: CHANNELS,
                patch_t: 1,
                patch_h: PATCH,
                patch_w: PATCH,
                spatial_compression: SPATIAL_COMPRESSION,
                temporal_compression: 1,
            }),
            schedule: Some(ScheduleFact {
                kind: ScheduleKind::Flow,
                shift: self.shift,
                train_steps: TRAIN_STEPS,
                boundary: None,
                pinned_sigmas: turbo_sigmas(self.shift),
                stream_shifts: vec![],
            }),
            max_rows: match self.te {
                Some(_) => 16_384 + TE_MAX_TOKENS,
                None => 4096,
            },
        }
    }
}

#[must_use]
pub fn turbo_sigmas(shift: f32) -> Vec<f32> {
    (0..8)
        .map(|i| 1.0 - i as f32 / 8.0)
        .map(|sigma| shift * sigma / (1.0 + (shift - 1.0) * sigma))
        .collect()
}

pub struct Tap;

impl Tap {
    pub const ENV: &'static str = "PIE_Z_IMAGE_TAP";

    #[must_use]
    pub fn from_env() -> Option<String> {
        std::env::var(Self::ENV).ok().filter(|key| !key.is_empty())
    }

    #[must_use]
    pub fn width(key: Option<&str>, d: &Dims) -> u32 {
        match key {
            None | Some("latents") | Some("final_linear") => PATCH_FEATURES,
            Some(_) => d.dim,
        }
    }
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
    pub fn context() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Context)
    }

    #[must_use]
    pub fn reading_lo() -> Predicate {
        Predicate::fact(READING_LO)
    }

    #[must_use]
    pub fn reading_mid() -> Predicate {
        Predicate::fact(READING_MID)
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
            reading: r.reading() & 7,
        }
    }

    fn word(&self) -> u64 {
        self.stream.word(STREAM_BASE) | (u64::from(self.reading & 7) << READING_LO)
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
        let (hi_mid, hi_low) = hi.split(&Facts::reading_mid());
        let (lo_mid, lo_low) = lo.split(&Facts::reading_mid());
        let (c7, c6) = hi_mid.split(&Facts::reading_lo());
        let (c5, c4) = hi_low.split(&Facts::reading_lo());
        let (c3, c2) = lo_mid.split(&Facts::reading_lo());
        let (c1, c0) = lo_low.split(&Facts::reading_lo());
        let arms = [c0, c1, c2, c3, c4, c5, c6, c7];
        let arm = |code: u8| &arms[usize::from(code)];

        if let (Some(code), Some(te)) = (codes.text, &self.te) {
            text_encode(arm(code), te);
        }
        refine(arm(codes.refine), &self.dims, &self.dit);
        let velocity = denoise(arm(codes.denoise), &self.dims, &self.dit);
        if let (Some(decode), Some(encode), Some(vae)) =
            (codes.vae_decode, codes.vae_encode, &self.vae)
        {
            super::vae::decode(arm(decode), vae);
            super::vae::encode(arm(encode), vae);
        }
        velocity
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

fn refine(arm: &Input<Facts>, d: &Dims, m: &Dit) {
    let c = arm.context(port::CAPTION, d.cap_width);
    let c = ops::elemwise::rmsnorm(&c, &m.cap_norm, NORM_EPS);
    let c = linear(&m.cap_embed, &c);
    let c = pad_rows(
        &c,
        &arm.latents(port::PAD_CAPTION, 1, Dtype::Bf16),
        &m.cap_pad_mod,
    );
    let geom = Geom {
        positions: arm.axis_positions(port::POSITIONS, ROPE_AXES),
        perm: arm.row_permutation(),
        csr: arm.lane_indptr(),
        mask: RaggedMask::None,
    };
    let mut c = c;
    let last = m.context_refiner.len() - 1;
    for (l, block) in arm.walk_layers(&m.context_refiner) {
        c = run_block(&c, block, None, d, &geom);
        if l as usize == last {
            seam::at(seam::HIDDEN, &[&c]);
        }
    }
}

fn denoise(arm: &Input<Facts>, d: &Dims, m: &Dit) -> Value {
    let (img, ctx) = arm.split(&Facts::image());

    let lanes = arm.request_of_token();
    let positions = arm.axis_positions(port::POSITIONS, ROPE_AXES);
    let joint = Geom {
        positions: positions.clone(),
        perm: arm.row_permutation(),
        csr: arm.group_indptr(),
        mask: RaggedMask::GroupBlockDiagonal,
    };

    let t = arm.lane_vector(port::TIMESTEP, 1);
    let u = ops::elemwise::mul_scalar(-0.5, &ops::elemwise::add(&t, &t));
    let u = ops::elemwise::add_bias(&m.t_flip, &u);
    let temb = ops::elemwise::sinusoid(&u, T_FREQ_DIM, T_MAX_PERIOD, T_FLIP_SIN_COS, 1.0);
    let temb = linear(&m.t_mlp1, &ops::elemwise::silu(&linear(&m.t_mlp0, &temb)));
    debug_assert_eq!(temb.width(), u64::from(ADALN_DIM));

    let (img_lanes, _) = lanes.split(&Facts::image());
    let (img_positions, _) = positions.split(&Facts::image());
    let (temb_img, _) = temb.split(&Facts::image());
    let own = Geom {
        positions: img_positions,
        perm: img.row_permutation(),
        csr: img.lane_indptr(),
        mask: RaggedMask::None,
    };
    let x = img.latents(port::LATENTS, PATCH_FEATURES, Dtype::Bf16);
    let tap = Tap::from_env().unwrap_or_default();
    let c_early = (!tap.is_empty()).then(|| ctx.latents(port::CONTEXT_REFINED, d.dim, Dtype::Bf16));
    macro_rules! tap {
        ($name:expr, $v:expr) => {
            if tap == $name {
                return seam_tapped($v, c_early.expect("a tapped arm reads the context port"));
            }
        };
    }
    tap!("latents", x.clone());
    let x = linear(&m.x_embed, &x);
    tap!("x_linear", x.clone());
    let flag = img.latents(port::PAD_IMAGE, 1, Dtype::Bf16);
    let mut x = pad_rows(&x, &flag, &m.x_pad_mod);
    tap!("x_embed", x.clone());
    for (l, block) in img.walk_layers(&m.noise_refiner) {
        let mods = adaln4(
            &linear(
                block.ada.as_ref().expect("a noise refiner is modulated"),
                &temb_img,
            ),
            d.dim,
        );
        if l == 0 {
            tap!(
                "normed0",
                ops::elemwise::rmsnorm(&x, &block.attn_norm1, NORM_EPS)
            );
            tap!("scaled0", {
                let h = ops::elemwise::rmsnorm(&x, &block.attn_norm1, NORM_EPS);
                ops::elemwise::modulate(&h, &mods.scale_msa, Some(&img_lanes), ModulateForm::Scale)
            });
        }
        let want = if l == 0 { tap.as_str() } else { "" };
        let (next, hit) = run_block_tapped(&x, block, Some((&mods, &img_lanes)), d, &own, want);
        if let Some(hit) = hit {
            return seam_tapped(hit, c_early.expect("a tapped arm reads the context port"));
        }
        x = next;
        tap!(format!("refiner{l}"), x.clone());
    }

    tap!("x_refined", x.clone());
    let c = c_early
        .clone()
        .unwrap_or_else(|| ctx.latents(port::CONTEXT_REFINED, d.dim, Dtype::Bf16));

    let mut u = Value::merge(vec![x, c]);
    for (l, block) in arm.walk_layers(&m.layers) {
        let mods = adaln4(
            &linear(
                block.ada.as_ref().expect("a joint block is modulated"),
                &temb,
            ),
            d.dim,
        );
        u = run_block(&u, block, Some((&mods, &lanes)), d, &joint);
        if tap == format!("layer{l}") {
            let (ui, ci) = u.split(&Facts::image());
            let both = Value::merge(vec![ui, ci]);
            seam::at(seam::VELOCITY, &[&both]);
            return both;
        }
    }

    let scale = linear(&m.final_ada, &ops::elemwise::silu(&temb));
    let (scale_img, _) = scale.split(&Facts::image());
    let (ui, _) = u.split(&Facts::image());

    let h = ops::elemwise::modulate(
        &ops::elemwise::layernorm_no_scale(&ui, FINAL_LN_EPS),
        &scale_img,
        Some(&img_lanes),
        ModulateForm::Scale,
    );
    tap!("final_norm", h.clone());
    let v = linear(&m.final_linear, &h);
    tap!("final_linear", v.clone());
    let velocity = ops::elemwise::mul_scalar(-1.0, &v);
    seam::at(seam::VELOCITY, &[&velocity]);
    velocity
}

struct Geom {
    positions: Value,
    perm: Value,
    csr: Value,
    mask: RaggedMask,
}

fn linear(w: &Linear, x: &Value) -> Value {
    ops::elemwise::add_bias(&w.bias, &ops::linear::matmul(x, &w.w))
}

fn seam_tapped(image: Value, context: Value) -> Value {
    let width = u32::try_from(image.width()).unwrap_or(u32::MAX);
    let context = if u64::from(width) < context.width() {
        ops::layout::split_rows(&context, width).0
    } else {
        context
    };
    let both = Value::merge(vec![image, context]);
    seam::at(seam::VELOCITY, &[&both]);
    both
}

fn pad_rows(x: &Value, flag: &Value, bank: &Weight) -> Value {
    let m = ops::linear::matmul(flag, bank);
    ops::elemwise::modulate(x, &m, None, ModulateForm::ScaleShift)
}

struct Mods {
    scale_msa: Value,
    gate_msa: Value,
    scale_mlp: Value,
    gate_mlp: Value,
}

fn adaln4(m: &Value, dim: u32) -> Mods {
    debug_assert_eq!(m.width(), u64::from(MOD_SLICES * dim));
    let (scale_msa, rest) = ops::layout::split_rows(m, dim);
    let (gate_msa, rest) = ops::layout::split_rows(&rest, dim);
    let (scale_mlp, gate_mlp) = ops::layout::split_rows(&rest, dim);
    Mods {
        scale_msa,
        gate_msa: ops::elemwise::tanh(&gate_msa),
        scale_mlp,
        gate_mlp: ops::elemwise::tanh(&gate_mlp),
    }
}

fn run_block(x: &Value, b: &Block, mods: Option<(&Mods, &Value)>, d: &Dims, g: &Geom) -> Value {
    run_block_tapped(x, b, mods, d, g, "").0
}

fn run_block_tapped(
    x: &Value,
    b: &Block,
    mods: Option<(&Mods, &Value)>,
    d: &Dims,
    g: &Geom,
    tap: &str,
) -> (Value, Option<Value>) {
    let mut hit: Option<Value> = None;
    macro_rules! tap {
        ($name:expr, $v:expr) => {
            if tap == $name {
                hit = Some($v);
            }
        };
    }
    let dim = d.dim;
    let hd = d.head_dim;
    let scaled = |normed: &Value, s: &Value, lanes: &Value| {
        ops::elemwise::modulate(normed, s, Some(lanes), ModulateForm::Scale)
    };

    let h = ops::elemwise::rmsnorm(x, &b.attn_norm1, NORM_EPS);
    let h = match mods {
        Some((m, lanes)) => scaled(&h, &m.scale_msa, lanes),
        None => h,
    };
    let (q, k, v) = ops::layout::split_qkv(&ops::linear::matmul(&h, &b.attn.qkv), dim, dim);
    let turn = |x: &Value, gain: &Weight| {
        ops::elemwise::rope_axes(
            &ops::elemwise::rmsnorm_per_head(x, gain, hd, NORM_EPS),
            &g.positions,
            d.rope_dims,
            [ROPE_THETA; 4],
            RopeForm::Interleaved,
            hd,
            hd,
        )
    };
    let (q, k) = (turn(&q, &b.attn.q_norm), turn(&k, &b.attn.k_norm));
    tap!("b0.q", q.clone());
    tap!("b0.k", k.clone());
    tap!("b0.v", v.clone());
    let o = ops::attn::ragged(
        &ops::layout::pack_rows(&q, &g.perm),
        &ops::layout::pack_rows(&k, &g.perm),
        &ops::layout::pack_rows(&v, &g.perm),
        &g.csr,
        &g.csr,
        hd,
        d.sm_scale(),
        g.mask,
    );
    let o = ops::layout::unpack_rows(&o, &g.perm);
    tap!("b0.attn", o.clone());
    let o = ops::linear::matmul(&o, &b.attn.out);
    tap!("b0.out", o.clone());
    let o = ops::elemwise::rmsnorm(&o, &b.attn_norm2, NORM_EPS);
    tap!("b0.norm2", o.clone());
    let x = match mods {
        Some((m, lanes)) => ops::elemwise::gated_residual_add(x, &m.gate_msa, &o, Some(lanes)),
        None => ops::elemwise::residual_add(&o, x),
    };
    tap!("b0.res1", x.clone());

    let h = ops::elemwise::rmsnorm(&x, &b.ffn_norm1, NORM_EPS);
    let h = match mods {
        Some((m, lanes)) => scaled(&h, &m.scale_mlp, lanes),
        None => h,
    };
    let f = ops::linear::matmul(
        &ops::linear::mlp_swiglu(&ops::linear::matmul(&h, &b.mlp.gate_up), d.inter),
        &b.mlp.down,
    );
    tap!("b0.ffn", f.clone());
    let f = ops::elemwise::rmsnorm(&f, &b.ffn_norm2, NORM_EPS);
    tap!("b0.ffn_norm2", f.clone());
    let out = match mods {
        Some((m, lanes)) => ops::elemwise::gated_residual_add(&x, &m.gate_mlp, &f, Some(lanes)),
        None => ops::elemwise::residual_add(&f, &x),
    };
    tap!("b0.res2", out.clone());
    (out, hit)
}

const _: () = assert!(Dims::turbo().cap_width == TE_HIDDEN);
