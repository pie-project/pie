use poem_dsl::ops::spatial;
use poem_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, ModulateForm, Predicate, RaggedMask,
    Request, RopeForm, Stream, Value, Weight, ops, seam,
};
use poem_ir::{TimePad, VoxelSegment};

use crate::{
    AxisRole, Generative, LatentSpace, PortFact, PortKind, PositionConvention, ReadingFact,
    ReadoutKind, ScheduleFact, ScheduleKind,
};

use super::model::{
    Block, CONTEXT_LEN, Conv, Dims, Dit, Linear, MOD_SLICES, MidAttention, Model, NORM_EPS,
    PATCH_H, PATCH_T, PATCH_W, ROPE_AXES, ROPE_THETA, Resnet, Shortcut, T_FLIP_SIN_COS,
    T_MAX_PERIOD, T_SCALE, TE_BUCKETS, TE_EPS, TE_HEAD_DIM, TE_HIDDEN, TE_MAX_DISTANCE,
    TE_MAX_TOKENS, TE_VOCAB, TRAIN_STEPS, TextEncoder, VAE_EPS, VAE_PATCH, VAE_RGB,
    VAE_SPATIAL_COMPRESSION, VAE_TEMPORAL_COMPRESSION, VAE_Z, Vae, VaeEncoder, port,
};

pub const STREAM_BASE: u8 = 0;

pub const READING_LO: u8 = 6;
pub const READING_HI: u8 = 7;
pub const READING_TOP: u8 = 8;
const READING_MASK: u8 = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Readings {
    pub text: Option<u8>,
    pub denoise: u8,
    pub vae_decode_head: Option<u8>,
    pub vae_decode: Option<u8>,
    pub vae_encode_head: Option<u8>,
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
        let vae_decode_head = self.vae.as_ref().map(|_| take());
        let vae_decode = self.vae.as_ref().map(|_| take());
        let vae_encode_head = self.vae.as_ref().map(|_| take());
        let vae_encode = self.vae.as_ref().map(|_| take());
        Readings {
            text,
            denoise,
            vae_decode_head,
            vae_decode,
            vae_encode_head,
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
                has_kv: false,
                takes_tokens: true,
                streams: vec![Stream::Text],
                ports: vec![],
                positions: None,
                readout: ReadoutKind::Hidden,
                readout_width: TE_HIDDEN,
            });
        }
        readings.push(ReadingFact {
            name: "denoise",
            index: codes.denoise,
            has_kv: false,
            takes_tokens: false,
            streams: vec![Stream::Video, Stream::Context],
            ports: vec![
                port("latents", PortKind::Latents, d.patch_in(), &[Stream::Video]),
                PortFact {
                    name: "context",
                    kind: PortKind::Context,
                    width: d.text_dim,
                    streams: vec![Stream::Context],
                    at: None,
                    rows: self.te.as_ref().map(|_| CONTEXT_LEN),
                },
                port("timestep", PortKind::LaneVector, 1, &[Stream::Video]),
                port(
                    "positions",
                    PortKind::AxisPositions,
                    u32::from(ROPE_AXES),
                    &[Stream::Video],
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
            readout_width: d.patch_out(),
        });
        if let (Some(head), Some(rest), Some(_)) =
            (codes.vae_decode_head, codes.vae_decode, &self.vae)
        {
            for (name, index) in [("vae.decode.head", head), ("vae.decode", rest)] {
                readings.push(ReadingFact {
                    name,
                    index,
                    has_kv: false,
                    takes_tokens: false,
                    streams: vec![Stream::Video],
                    ports: vec![port("latent", PortKind::Voxels, VAE_Z, &[Stream::Video])],
                    positions: None,
                    readout: ReadoutKind::Pixels,
                    readout_width: VAE_RGB,
                });
            }
        }
        if let (Some(head), Some(rest), Some(_)) =
            (codes.vae_encode_head, codes.vae_encode, &self.vae)
        {
            for (name, index) in [("vae.encode.head", head), ("vae.encode", rest)] {
                readings.push(ReadingFact {
                    name,
                    index,
                    has_kv: false,
                    takes_tokens: false,
                    streams: vec![Stream::Video],
                    ports: vec![PortFact {
                        name: "pixels",
                        kind: PortKind::Voxels,
                        width: VAE_RGB,
                        streams: vec![Stream::Video],
                        at: Some(port::PIXEL_VOXELS),
                        rows: None,
                    }],
                    positions: None,
                    readout: ReadoutKind::Pixels,
                    readout_width: VAE_Z,
                });
            }
        }
        Generative {
            readings,
            latent: Some(LatentSpace {
                channels: d.in_channels,
                patch_t: PATCH_T,
                patch_h: PATCH_H,
                patch_w: PATCH_W,
                spatial_compression: VAE_SPATIAL_COMPRESSION,
                temporal_compression: VAE_TEMPORAL_COMPRESSION,
            }),
            schedule: Some(ScheduleFact {
                kind: ScheduleKind::Flow,
                shift: self.shift,
                train_steps: TRAIN_STEPS,
                boundary: None,
                pinned_sigmas: vec![],
                stream_shifts: vec![],
            }),
            max_rows: match self.te {
                Some(_) => 32_768 + CONTEXT_LEN,
                None => 4096,
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
    pub fn text() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Text)
    }

    #[must_use]
    pub fn video() -> Predicate {
        Predicate::stream(STREAM_BASE, Stream::Video)
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
    pub fn reading_hi() -> Predicate {
        Predicate::fact(READING_HI)
    }

    #[must_use]
    pub fn reading_top() -> Predicate {
        Predicate::fact(READING_TOP)
    }
}

impl Classify for Facts {
    fn of(r: &Request) -> Facts {
        Facts {
            stream: r.stream(),
            reading: r.reading() & READING_MASK,
        }
    }

    fn word(&self) -> u64 {
        self.stream.word(STREAM_BASE) | (u64::from(self.reading & READING_MASK) << READING_LO)
    }
}

impl ForwardHybrid for Model {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();
        if let Some(vae) = &self.vae {
            for conv in vae.cached_convs().chain(vae.enc.cached_convs()) {
                let name = conv.cache.as_ref().expect("a cached conv names its slab");
                c.state(name.clone(), conv.slab(), Dtype::Bf16);
            }
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let codes = self.readings();
        let (top, bot) = inputs.split(&Facts::reading_top());
        let (t_hi, t_lo) = top.split(&Facts::reading_hi());
        let (b_hi, b_lo) = bot.split(&Facts::reading_hi());
        let (c7, c6) = t_hi.split(&Facts::reading_lo());
        let (c5, c4) = t_lo.split(&Facts::reading_lo());
        let (c3, c2) = b_hi.split(&Facts::reading_lo());
        let (c1, c0) = b_lo.split(&Facts::reading_lo());
        let arms = [c0, c1, c2, c3, c4, c5, c6, c7];
        let arm = |code: u8| &arms[usize::from(code)];

        if let (Some(code), Some(te)) = (codes.text, &self.te) {
            text_encode(arm(code), te);
        }
        let velocity = denoise(arm(codes.denoise), self);
        if let (Some(head), Some(rest), Some(vae)) =
            (codes.vae_decode_head, codes.vae_decode, &self.vae)
        {
            let _ = vae_decode(arm(head), vae, true);
            let _ = vae_decode(arm(rest), vae, false);
        }
        if let (Some(head), Some(rest), Some(vae)) =
            (codes.vae_encode_head, codes.vae_encode, &self.vae)
        {
            let _ = vae_encode(arm(head), &vae.enc, true);
            let _ = vae_encode(arm(rest), &vae.enc, false);
        }
        velocity
    }
}

fn text_encode(arm: &Input<Facts>, te: &TextEncoder) {
    let ids = arm.tokens();
    let perm = arm.row_permutation();
    let csr = arm.lane_indptr();
    let mut y = ops::layout::embed(&ids, &te.embed, TE_VOCAB);
    for (_, w) in arm.walk_layers(&te.layers) {
        let table = ops::elemwise::relative_bucket_bias(
            arm.recorder(),
            &w.rel_bias,
            TE_MAX_TOKENS,
            TE_BUCKETS,
            TE_MAX_DISTANCE,
            true,
        );
        let x = ops::elemwise::rmsnorm(&y, &w.attn_norm, TE_EPS);
        let q = ops::linear::matmul(&x, &w.q);
        let k = ops::linear::matmul(&x, &w.k);
        let v = ops::linear::matmul(&x, &w.v);
        let o = ops::attn::ragged(
            &ops::layout::pack_rows(&q, &perm),
            &ops::layout::pack_rows(&k, &perm),
            &ops::layout::pack_rows(&v, &perm),
            &csr,
            &csr,
            TE_HEAD_DIM,
            1.0,
            ops::attn::relative_bias(&table, TE_MAX_TOKENS),
        );
        let o = ops::layout::unpack_rows(&o, &perm);
        y = ops::elemwise::residual_add(&ops::linear::matmul(&o, &w.o), &y);

        let x = ops::elemwise::rmsnorm(&y, &w.ffn_norm, TE_EPS);
        let gate = ops::linear::matmul(&x, &w.wi_0);
        let up = ops::linear::matmul(&x, &w.wi_1);
        let act = ops::linear::mlp_geglu_tanh(&gate, &up);
        y = ops::elemwise::residual_add(&ops::linear::matmul(&act, &w.wo), &y);
    }
    let out = ops::elemwise::rmsnorm(&y, &te.final_norm, TE_EPS);
    seam::at(seam::HIDDEN, &[&out]);
}

fn linear(w: &Linear, x: &Value) -> Value {
    ops::elemwise::add_bias(&w.bias, &ops::linear::matmul(x, &w.w))
}

struct VideoGeom {
    lanes: Value,
    positions: Value,
    perm: Value,
    csr: Value,
}

struct ContextGeom {
    perm: Value,
    csr: Value,
}

struct Mods {
    attn_ss: Value,
    attn_gate: Value,
    ffn_ss: Value,
    ffn_gate: Value,
}

fn adaln6(e: &Value, dim: u32) -> Mods {
    debug_assert_eq!(e.width(), u64::from(MOD_SLICES * dim));
    let (attn_ss, rest) = ops::layout::split_rows(e, 2 * dim);
    let (attn_gate, rest) = ops::layout::split_rows(&rest, dim);
    let (ffn_ss, ffn_gate) = ops::layout::split_rows(&rest, 2 * dim);
    Mods {
        attn_ss,
        attn_gate,
        ffn_ss,
        ffn_gate,
    }
}

fn norm_modulate(x: &Value, scale_shift: &Value, lanes: &Value) -> Value {
    ops::elemwise::modulate(
        &ops::elemwise::layernorm_no_scale(x, NORM_EPS),
        scale_shift,
        Some(lanes),
        ModulateForm::ScaleShift,
    )
}

fn block(
    x: &Value,
    c: &Value,
    b: &Block,
    e: &Value,
    d: &Dims,
    vg: &VideoGeom,
    cg: &ContextGeom,
) -> Value {
    let dim = d.dim;
    let hd = d.head_dim;
    let m = adaln6(e, dim);

    let h = norm_modulate(x, &m.attn_ss, &vg.lanes);
    let (q, k, v) = ops::layout::split_qkv(&linear(&b.self_attn.qkv, &h), dim, dim);
    let turn = |x: &Value, gain: &Weight| {
        ops::elemwise::rope_axes(
            &ops::elemwise::rmsnorm(x, gain, NORM_EPS),
            &vg.positions,
            d.rope_dims(),
            [ROPE_THETA; 4],
            RopeForm::Interleaved,
            hd,
            hd,
        )
    };
    let o = ops::attn::ragged(
        &ops::layout::pack_rows(&turn(&q, &b.self_attn.norm_q), &vg.perm),
        &ops::layout::pack_rows(&turn(&k, &b.self_attn.norm_k), &vg.perm),
        &ops::layout::pack_rows(&v, &vg.perm),
        &vg.csr,
        &vg.csr,
        hd,
        d.sm_scale(),
        RaggedMask::GroupBlockDiagonal,
    );
    let o = ops::layout::unpack_rows(&o, &vg.perm);
    let x = ops::elemwise::gated_residual_add(
        x,
        &m.attn_gate,
        &linear(&b.self_attn.out, &o),
        Some(&vg.lanes),
    );

    let hc = ops::elemwise::layernorm(&x, &b.norm2, &b.norm2_bias, NORM_EPS);
    let cq = ops::elemwise::rmsnorm(&linear(&b.cross.q, &hc), &b.cross.norm_q, NORM_EPS);
    let (ck, cv) = ops::layout::split_rows(&linear(&b.cross.kv, c), dim);
    let ck = ops::elemwise::rmsnorm(&ck, &b.cross.norm_k, NORM_EPS);
    let ca = ops::attn::ragged(
        &ops::layout::pack_rows(&cq, &vg.perm),
        &ops::layout::pack_rows(&ck, &cg.perm),
        &ops::layout::pack_rows(&cv, &cg.perm),
        &vg.csr,
        &cg.csr,
        hd,
        d.sm_scale(),
        RaggedMask::GroupBlockDiagonal,
    );
    let ca = ops::layout::unpack_rows(&ca, &vg.perm);
    let x = ops::elemwise::residual_add(&linear(&b.cross.out, &ca), &x);

    let hf = norm_modulate(&x, &m.ffn_ss, &vg.lanes);
    let f = linear(
        &b.ffn.down,
        &ops::elemwise::gelu(&linear(&b.ffn.up, &hf), true),
    );
    ops::elemwise::gated_residual_add(&x, &m.ffn_gate, &f, Some(&vg.lanes))
}

fn denoise(arm: &Input<Facts>, m: &Model) -> Value {
    let d = &m.dims;
    let dit: &Dit = &m.dit;

    let (ctx, vid) = arm.split(&Facts::context());
    let vg = VideoGeom {
        lanes: vid.request_of_token(),
        positions: vid.axis_positions(port::POSITIONS, ROPE_AXES),
        perm: vid.row_permutation(),
        csr: vid.group_indptr(),
    };
    let cg = ContextGeom {
        perm: ctx.row_permutation(),
        csr: ctx.group_indptr(),
    };

    let t = vid.lane_vector(port::TIMESTEP, 1);
    let h_t = ops::elemwise::silu(&linear(
        &dit.time_embed.linear_1,
        &ops::elemwise::sinusoid(&t, d.freq_dim, T_MAX_PERIOD, T_FLIP_SIN_COS, T_SCALE),
    ));
    let temb = linear(&dit.time_embed.linear_2, &h_t);
    let proj = linear(&dit.time_proj, &ops::elemwise::silu(&temb));
    let head_mod = ops::elemwise::add_bias(&dit.head_table, &linear(&dit.head_proj, &h_t));

    let c = ctx.context(port::CONTEXT, d.text_dim);
    let c = linear(
        &dit.text_embed.linear_2,
        &ops::elemwise::gelu(&linear(&dit.text_embed.linear_1, &c), true),
    );

    let mut x = linear(
        &dit.patch_embed,
        &vid.latents(port::LATENTS, d.patch_in(), Dtype::Bf16),
    );

    for (_, b) in arm.walk_layers(&dit.blocks) {
        let e = ops::elemwise::add_bias(&b.table, &ops::elemwise::copy(&proj));
        x = block(&x, &c, b, &e, d, &vg, &cg);
    }

    let h = norm_modulate(&x, &head_mod, &vg.lanes);
    let velocity = linear(&dit.proj_out, &h);
    seam::at(seam::VELOCITY, &[&velocity]);
    velocity
}

fn conv(x: &Value, g: &Value, c: &Conv, arm: &Input<Facts>) -> (Value, Value) {
    let shape = spatial::Conv::conv3d(c.k, [1, 1, 1], [0, c.k[1] / 2, c.k[2] / 2]);
    let (shape, cache) = match &c.cache {
        Some(name) => (shape.causal(TimePad::Zero), Some(arm.state(name))),
        None => (shape, None),
    };
    spatial::conv3d(x, g, &c.w, Some(&c.bias), shape, cache)
}

fn norm_silu(x: &Value, gain: &Weight) -> Value {
    ops::elemwise::silu(&ops::elemwise::rmsnorm(x, gain, VAE_EPS))
}

fn resnet(x: &Value, g: &Value, r: &Resnet, arm: &Input<Facts>) -> Value {
    let h = norm_silu(x, &r.norm1);
    let (h, _) = conv(&h, g, &r.conv1, arm);
    let h = norm_silu(&h, &r.norm2);
    let (h, _) = conv(&h, g, &r.conv2, arm);
    let skip = match &r.shortcut {
        Some(c) => conv(x, g, c, arm).0,
        None => x.clone(),
    };
    ops::elemwise::add(&skip, &h)
}

fn mid_attention(x: &Value, g: &Value, a: &MidAttention) -> Value {
    let width = u32::try_from(a.proj.w.shape[0]).expect("a VAE row is narrower than 2^32");
    let h = ops::elemwise::rmsnorm(x, &a.norm, VAE_EPS);
    let (q, k, v) = ops::layout::split_qkv(&linear(&a.qkv, &h), width, width);
    let o = spatial::attention_over(
        &q,
        &k,
        &v,
        g,
        VoxelSegment::Frames(1),
        (width as f32).sqrt().recip(),
    );
    ops::elemwise::add(x, &linear(&a.proj, &o))
}

pub fn vae_decode(arm: &Input<Facts>, vae: &Vae, first: bool) -> Value {
    let g0 = arm.grid();
    let z = arm.voxels(port::VOXELS, VAE_Z, Dtype::Bf16);
    let z = ops::elemwise::mul_scalar(0.5, &ops::elemwise::add(&z, &z));
    let z = ops::elemwise::standardize(&z, &vae.denorm_bias, &vae.denorm_scale);
    let (z, g) = conv(&z, &g0, &vae.post_quant, arm);
    let (mut x, _) = conv(&z, &g, &vae.conv_in, arm);
    let mut g = g;

    x = resnet(&x, &g, &vae.mid_res0, arm);
    x = mid_attention(&x, &g, &vae.mid_attn);
    x = resnet(&x, &g, &vae.mid_res1, arm);

    for up in &vae.up {
        let (x_in, g_in) = (x.clone(), g.clone());
        for r in &up.resnets {
            x = resnet(&x, &g, r, arm);
        }
        if let Some(u) = &up.upsampler {
            if let (Some(tc), false) = (&u.time_conv, first) {
                let (y, gy) = conv(&x, &g, tc, arm);
                let (y, gy) = spatial::pixel_shuffle(&y, &gy, [2, 1, 1]);
                x = y;
                g = gy;
            }
            let (y, gy) = spatial::upsample_nearest(&x, &g, [1, 2, 2], false);
            let (y, gy) = conv(&y, &gy, &u.resample, arm);
            x = y;
            g = gy;
        }
        if let Some(shortcut) = up.shortcut {
            let s = match shortcut {
                Shortcut::Nearest222 => {
                    let factor = if first { [1, 2, 2] } else { [2, 2, 2] };
                    spatial::upsample_nearest(&x_in, &g_in, factor, false).0
                }
                Shortcut::ShuffleH => {
                    let (s, gs) = spatial::pixel_shuffle(&x_in, &g_in, [1, 2, 1]);
                    spatial::upsample_nearest(&s, &gs, [1, 1, 2], false).0
                }
            };
            x = ops::elemwise::add(&x, &s);
        }
    }

    let x = norm_silu(&x, &vae.norm_out);
    let (y, gy) = conv(&x, &g, &vae.conv_out, arm);
    let (pixels, gp) = spatial::pixel_shuffle(&y, &gy, [1, VAE_PATCH, VAE_PATCH]);
    let pixels = ops::elemwise::clamp(&pixels, -1.0, 1.0);
    seam::at(seam::PIXELS, &[&pixels, &gp]);
    pixels
}

fn downsample2d(x: &Value, g: &Value, c: &Conv) -> (Value, Value) {
    let shape = spatial::Conv::conv2d([3, 3], [2, 2], [0, 0]).pad_back([0, 1, 1]);
    spatial::conv3d(x, g, &c.w, Some(&c.bias), shape, None)
}

fn downsample_time(x: &Value, g: &Value, c: &Conv, arm: &Input<Facts>) -> (Value, Value) {
    let name = c.cache.as_ref().expect("the time conv keeps a frame cache");
    let shape = spatial::Conv {
        k: c.k,
        stride: [2, 1, 1],
        pad: [c.front, 0, 0],
        pad_back: [0, 0, 0],
        causal_t: true,
        time_pad: TimePad::Zero,
    };
    spatial::conv3d(x, g, &c.w, Some(&c.bias), shape, Some(arm.state(name)))
}

pub fn vae_encode(arm: &Input<Facts>, e: &VaeEncoder, first: bool) -> Value {
    let g0 = arm.grid();
    let px = arm.voxels(port::PIXEL_VOXELS, VAE_RGB, Dtype::Bf16);
    let (x, g) = spatial::pixel_unshuffle(&px, &g0, [1, VAE_PATCH, VAE_PATCH]);
    let (mut x, mut g) = conv(&x, &g, &e.conv_in, arm);

    for block in &e.down {
        let (x_in, g_in) = (x.clone(), g.clone());
        for r in &block.resnets {
            x = resnet(&x, &g, r, arm);
        }
        if let Some(d) = &block.downsampler {
            let (y, gy) = downsample2d(&x, &g, &d.resample);
            x = y;
            g = gy;
            if let Some(tc) = &d.time_conv {
                if first {
                    let name = tc.cache.as_ref().expect("the time conv keeps a cache");
                    x = spatial::store_frames(&x, &g, arm.state(name), tc.front);
                } else {
                    let (y, gy) = downsample_time(&x, &g, tc, arm);
                    x = y;
                    g = gy;
                }
            }
        }
        let (s, _) = spatial::avg_down(&x_in, &g_in, block.shortcut.factor, block.shortcut.group);
        x = ops::elemwise::add(&x, &s);
    }

    x = resnet(&x, &g, &e.mid_res0, arm);
    x = mid_attention(&x, &g, &e.mid_attn);
    x = resnet(&x, &g, &e.mid_res1, arm);

    let x = norm_silu(&x, &e.norm_out);
    let (h, gh) = conv(&x, &g, &e.conv_out, arm);
    let (z, gz) = conv(&h, &gh, &e.quant, arm);
    let z = ops::elemwise::standardize(&z, &e.norm_bias, &e.norm_scale);
    seam::at(seam::PIXELS, &[&z, &gz]);
    z
}
