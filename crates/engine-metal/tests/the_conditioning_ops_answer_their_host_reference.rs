#![cfg(target_vendor = "apple")]

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::elemwise::{gate, modulate, norm, pointwise, rope_axes, sinusoid};
use kernels_metal::layout;
use model_ir::Dtype;

fn noise(at: u64) -> u32 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 32) as u32
}

fn unit(at: u64) -> f32 {
    (noise(at) as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn bf16_round(v: f32) -> f32 {
    let bits = v.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(((bits + rounding) >> 16) << 16)
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .map(|f| ((f.to_bits() + 0x7fff + ((f.to_bits() >> 16) & 1)) >> 16) as u16)
        .flat_map(u16::to_le_bytes)
        .collect()
}

fn bf16_floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn f32_floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|i| i.to_le_bytes()).collect()
}

struct Rig {
    device: Context,
    handles: Handles,
    pipelines: Pipelines,
}

impl Rig {
    fn open() -> Option<Self> {
        let device = Context::bind().ok()?;
        Some(Self {
            device,
            handles: Handles::new(),
            pipelines: Pipelines::new(),
        })
    }

    fn plane(&self, data: &[f32], dtype: Dtype) -> (Buffer, u32) {
        let bytes = match dtype {
            Dtype::Bf16 => bf16_bytes(data),
            Dtype::F32 => f32_bytes(data),
            other => panic!("no test plane for {other:?}"),
        };
        let mut buf = Buffer::zeroed(&self.device, bytes.len() as u64).expect("a plane");
        buf.write(0, &bytes).expect("write");
        let handle = self.handles.bind(&buf, 0, buf.bytes()).expect("a handle");
        (buf, handle)
    }

    fn i32_plane(&self, data: &[i32]) -> (Buffer, u32) {
        let bytes = i32_bytes(data);
        let mut buf = Buffer::zeroed(&self.device, bytes.len() as u64).expect("a plane");
        buf.write(0, &bytes).expect("write");
        let handle = self.handles.bind(&buf, 0, buf.bytes()).expect("a handle");
        (buf, handle)
    }

    fn empty(&self, elements: usize, dtype: Dtype) -> (Buffer, u32) {
        let width = match dtype {
            Dtype::Bf16 => 2,
            Dtype::F32 => 4,
            other => panic!("no test plane for {other:?}"),
        };
        let buf = Buffer::zeroed(&self.device, (elements * width) as u64).expect("a plane");
        let handle = self.handles.bind(&buf, 0, buf.bytes()).expect("a handle");
        (buf, handle)
    }

    fn read(&self, handle: u32, elements: usize, dtype: Dtype) -> Vec<f32> {
        match dtype {
            Dtype::Bf16 => bf16_floats(
                &self
                    .handles
                    .read(handle, (elements * 2) as u64)
                    .expect("read"),
            ),
            Dtype::F32 => f32_floats(
                &self
                    .handles
                    .read(handle, (elements * 4) as u64)
                    .expect("read"),
            ),
            other => panic!("no test plane for {other:?}"),
        }
    }

    fn fire(&self, f: impl FnOnce(&Sink<'_>)) {
        let frame = self.device.frame().expect("a frame");
        let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
        f(&sink);
        frame.commit().expect("the commit");
    }
}

fn within_a_bf16_ulp(what: &str, want: &[f32], got: &[f32]) {
    assert_eq!(want.len(), got.len(), "{what}: length");
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        let ulp = (w.abs() * (1.0 / 128.0)).max(1.0 / 2560.0);
        let d = (w - g).abs() / ulp;
        if d > worst {
            worst = d;
            at = i;
        }
    }
    eprintln!("{what}: worst {worst:.3} ulp at {at}");
    assert!(
        worst <= 1.0,
        "{what}: parts from the reference by {worst:.2} bf16 ulp at {at} \
         (want {}, got {})",
        want[at],
        got[at]
    );
}

fn within_relative(what: &str, want: &[f32], got: &[f32], bar: f32) {
    assert_eq!(want.len(), got.len(), "{what}: length");
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        let d = (w - g).abs() / w.abs().max(1e-3);
        if d > worst {
            worst = d;
            at = i;
        }
    }
    eprintln!("{what}: worst relative {worst:.3e} at {at}");
    assert!(
        worst <= bar,
        "{what}: parts from the reference by {worst:.3e} at {at} (want {}, got {})",
        want[at],
        got[at]
    );
}

const ROWS: u32 = 7;
const WIDTH: u32 = 96;
const LANES: u32 = 3;

const LANE_OF_ROW: [i32; ROWS as usize] = [0, 0, 1, 2, 1, 2, 2];

#[test]
fn the_conditioning_ops_answer_their_host_reference_every_case() {
    the_pointwise_arms_answer_the_reference_at_both_elements();
    a_modulation_reads_its_vector_per_lane_and_per_token();
    the_gated_residual_folds_what_the_reference_folds();
    the_timestep_embedding_matches_the_diffusers_formula();
    the_centring_answers_the_two_pass_reference();
    every_rope_pairing_turns_the_channels_its_form_names();
    the_relative_bias_table_reads_the_bucket_torch_lands();
    the_head_gate_scales_by_one_logit_a_head();
    a_pack_and_its_unpack_round_trip_the_rectangle();
    the_residual_blend_scores_normalized_and_blends_raw();
}

fn the_pointwise_arms_answer_the_reference_at_both_elements() {
    let Some(rig) = Rig::open() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    eprintln!("device: {}", rig.device.name());

    for dtype in [Dtype::Bf16, Dtype::F32] {
        let n = (ROWS * WIDTH) as usize;
        let round = |v: f32| {
            if dtype == Dtype::Bf16 {
                bf16_round(v)
            } else {
                v
            }
        };
        let x: Vec<f32> = (0..n as u64).map(|at| round(2.0 * unit(at))).collect();
        let y: Vec<f32> = (0..n as u64)
            .map(|at| round(2.0 * unit(at ^ 0x77)))
            .collect();

        let (_xb, hx) = rig.plane(&x, dtype);
        let (_yb, hy) = rig.plane(&y, dtype);
        let (_zb, hz) = rig.empty(n, dtype);
        let xt = Tensor::new(hx, ROWS, WIDTH, dtype);
        let yt = Tensor::new(hy, ROWS, WIDTH, dtype);
        let zt = Tensor::new(hz, ROWS, WIDTH, dtype);

        rig.fire(|s| pointwise::add(s, xt, yt, zt).expect("add"));
        let want: Vec<f32> = x.iter().zip(&y).map(|(a, b)| round(a + b)).collect();
        check(
            &format!("add {dtype:?}"),
            &want,
            &rig.read(hz, n, dtype),
            dtype,
        );

        rig.fire(|s| pointwise::mul(s, xt, yt, zt).expect("mul"));
        let want: Vec<f32> = x.iter().zip(&y).map(|(a, b)| round(a * b)).collect();
        check(
            &format!("mul {dtype:?}"),
            &want,
            &rig.read(hz, n, dtype),
            dtype,
        );

        rig.fire(|s| pointwise::silu(s, xt, zt).expect("silu"));
        let want: Vec<f32> = x.iter().map(|v| round(v / (1.0 + (-v).exp()))).collect();
        check(
            &format!("silu {dtype:?}"),
            &want,
            &rig.read(hz, n, dtype),
            dtype,
        );

        rig.fire(|s| pointwise::tanh(s, xt, zt).expect("tanh"));
        let want: Vec<f32> = x.iter().map(|v| round(v.tanh())).collect();
        check(
            &format!("tanh {dtype:?}"),
            &want,
            &rig.read(hz, n, dtype),
            dtype,
        );

        rig.fire(|s| pointwise::gelu_tanh(s, xt, zt).expect("gelu"));
        let want: Vec<f32> = x
            .iter()
            .map(|v| {
                const K: f32 = 0.797_884_6;
                round(0.5 * v * (1.0 + (K * (v + 0.044_715 * v * v * v)).tanh()))
            })
            .collect();
        check(
            &format!("gelu {dtype:?}"),
            &want,
            &rig.read(hz, n, dtype),
            dtype,
        );

        let (_cb, hc) = rig.plane(&x, dtype);
        let ct = Tensor::new(hc, ROWS, WIDTH, dtype);
        rig.fire(|s| pointwise::clamp(s, -0.5, 0.75, ct).expect("clamp"));
        let want: Vec<f32> = x.iter().map(|v| round(v.clamp(-0.5, 0.75))).collect();
        check(
            &format!("clamp {dtype:?}"),
            &want,
            &rig.read(hc, n, dtype),
            dtype,
        );

        let bounds = [round(-0.5), round(0.75)];
        let (_lb, hlo) = rig.plane(&bounds[..1], dtype);
        let (_hb, hhi) = rig.plane(&bounds[1..], dtype);
        let (_db, hd) = rig.plane(&x, dtype);
        rig.fire(|s| {
            pointwise::clamp_learned(
                s,
                Tensor::new(hlo, 1, 1, dtype),
                Tensor::new(hhi, 1, 1, dtype),
                Tensor::new(hd, ROWS, WIDTH, dtype),
            )
            .expect("clamp_learned")
        });
        let want: Vec<f32> = x
            .iter()
            .map(|v| round(v.clamp(bounds[0], bounds[1])))
            .collect();
        check(
            &format!("clamp_learned {dtype:?}"),
            &want,
            &rig.read(hd, n, dtype),
            dtype,
        );
    }
}

fn check(what: &str, want: &[f32], got: &[f32], dtype: Dtype) {
    if dtype == Dtype::Bf16 {
        within_a_bf16_ulp(what, want, got);
    } else {
        within_relative(what, want, got, 5e-6);
    }
}

fn a_modulation_reads_its_vector_per_lane_and_per_token() {
    let Some(rig) = Rig::open() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let n = (ROWS * WIDTH) as usize;
    let x: Vec<f32> = (0..n as u64).map(|at| bf16_round(unit(at))).collect();
    let (_xb, hx) = rig.plane(&x, Dtype::Bf16);
    let xt = Tensor::new(hx, ROWS, WIDTH, Dtype::Bf16);
    let (_ob, ho) = rig.empty(n, Dtype::Bf16);
    let ot = Tensor::new(ho, ROWS, WIDTH, Dtype::Bf16);
    let (_lb, hl) = rig.i32_plane(&LANE_OF_ROW);
    let lanes = Tensor::new(hl, ROWS, 1, Dtype::I32);

    for (form, vectors, name) in [
        (modulate::Form::ScaleShift, 2u32, "scale_shift"),
        (modulate::Form::Scale, 1, "scale"),
        (modulate::Form::TanhGate, 1, "tanh_gate"),
    ] {
        for m_dtype in [Dtype::Bf16, Dtype::F32] {
            let m_rows = LANES;
            let m_width = vectors * WIDTH;
            let m: Vec<f32> = (0..(m_rows * m_width) as u64)
                .map(|at| {
                    let v = 0.5 * unit(at ^ 0xBEEF);
                    if m_dtype == Dtype::Bf16 {
                        bf16_round(v)
                    } else {
                        v
                    }
                })
                .collect();
            let (_mb, hm) = rig.plane(&m, m_dtype);
            let mt = Tensor::new(hm, m_rows, m_width, m_dtype);
            rig.fire(|s| {
                modulate::modulate(s, form, xt, mt, Some(lanes), ot).expect("modulate per lane")
            });
            let want = modulation_reference(&x, &m, form, m_width, |row| LANE_OF_ROW[row] as usize);
            within_a_bf16_ulp(
                &format!("modulate {name} per lane, m {m_dtype:?}"),
                &want,
                &rig.read(ho, n, Dtype::Bf16),
            );

            let m: Vec<f32> = (0..(ROWS * m_width) as u64)
                .map(|at| {
                    let v = 0.5 * unit(at ^ 0xF00D);
                    if m_dtype == Dtype::Bf16 {
                        bf16_round(v)
                    } else {
                        v
                    }
                })
                .collect();
            let (_mb, hm) = rig.plane(&m, m_dtype);
            let mt = Tensor::new(hm, ROWS, m_width, m_dtype);
            rig.fire(|s| {
                modulate::modulate(s, form, xt, mt, None, ot).expect("modulate per token")
            });
            let want = modulation_reference(&x, &m, form, m_width, |row| row);
            within_a_bf16_ulp(
                &format!("modulate {name} per token, m {m_dtype:?}"),
                &want,
                &rig.read(ho, n, Dtype::Bf16),
            );
        }
    }
}

fn modulation_reference(
    x: &[f32],
    m: &[f32],
    form: modulate::Form,
    m_width: u32,
    row_of: impl Fn(usize) -> usize,
) -> Vec<f32> {
    let w = WIDTH as usize;
    let mw = m_width as usize;
    let mut out = Vec::with_capacity(ROWS as usize * w);
    for row in 0..ROWS as usize {
        let base = row_of(row) * mw;
        for i in 0..w {
            let xv = x[row * w + i];
            out.push(bf16_round(match form {
                modulate::Form::ScaleShift => xv.mul_add(1.0 + m[base + i], m[base + i + w]),
                modulate::Form::Scale => xv * (1.0 + m[base + i]),
                modulate::Form::TanhGate => m[base + i].tanh() * xv,
            }));
        }
    }
    out
}

fn the_gated_residual_folds_what_the_reference_folds() {
    let Some(rig) = Rig::open() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let n = (ROWS * WIDTH) as usize;
    let r: Vec<f32> = (0..n as u64).map(|at| bf16_round(unit(at))).collect();
    let y: Vec<f32> = (0..n as u64)
        .map(|at| bf16_round(unit(at ^ 0x99)))
        .collect();
    let g: Vec<f32> = (0..(LANES * WIDTH) as u64)
        .map(|at| bf16_round(0.3 * unit(at ^ 0x1357)))
        .collect();

    let (_rb, hr) = rig.plane(&r, Dtype::Bf16);
    let (_yb, hy) = rig.plane(&y, Dtype::Bf16);
    let (_gb, hg) = rig.plane(&g, Dtype::Bf16);
    let (_lb, hl) = rig.i32_plane(&LANE_OF_ROW);
    let rt = Tensor::new(hr, ROWS, WIDTH, Dtype::Bf16);
    let yt = Tensor::new(hy, ROWS, WIDTH, Dtype::Bf16);
    let gt = Tensor::new(hg, LANES, WIDTH, Dtype::Bf16);
    let lanes = Tensor::new(hl, ROWS, 1, Dtype::I32);

    rig.fire(|s| {
        modulate::gated_residual_add(s, rt, gt, yt, Some(lanes), rt).expect("gated residual")
    });

    let w = WIDTH as usize;
    let mut want = Vec::with_capacity(n);
    for row in 0..ROWS as usize {
        let lane = LANE_OF_ROW[row] as usize;
        for i in 0..w {
            want.push(bf16_round(
                g[lane * w + i].mul_add(y[row * w + i], r[row * w + i]),
            ));
        }
    }
    within_a_bf16_ulp("gated_residual_add", &want, &rig.read(hr, n, Dtype::Bf16));
}

fn the_timestep_embedding_matches_the_diffusers_formula() {
    let Some(rig) = Rig::open() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    for dim in [64u32, 65, 256] {
        for flip in [false, true] {
            let t: Vec<f32> = (0..LANES as u64)
                .map(|at| 1000.0 * (0.1 + 0.3 * unit(at)))
                .collect();
            let (_tb, ht) = rig.plane(&t, Dtype::F32);
            let (_yb, hy) = rig.empty((LANES * dim) as usize, Dtype::F32);
            let tt = Tensor::new(ht, LANES, 1, Dtype::F32);
            let yt = Tensor::new(hy, LANES, dim, Dtype::F32);
            rig.fire(|s| {
                sinusoid::sinusoid(s, tt, dim, 10_000.0, flip, 1.0, yt).expect("sinusoid")
            });

            let half = (dim / 2) as usize;
            let mut want = vec![0.0f32; (LANES * dim) as usize];
            for row in 0..LANES as usize {
                for i in 0..half {
                    let freq = (-(10_000.0f32).ln() * i as f32 / half as f32).exp();
                    let angle = t[row] * freq;
                    let (s, c) = (angle.sin(), angle.cos());
                    want[row * dim as usize + i] = if flip { c } else { s };
                    want[row * dim as usize + i + half] = if flip { s } else { c };
                }
            }
            let got = rig.read(hy, (LANES * dim) as usize, Dtype::F32);
            let worst = want
                .iter()
                .zip(&got)
                .map(|(w, g)| (w - g).abs())
                .fold(0.0f32, f32::max);
            eprintln!("sinusoid dim {dim} flip {flip}: worst absolute {worst:.3e}");
            assert!(worst < 5e-5, "sinusoid dim {dim} flip {flip}: {worst:.3e}");
            if dim % 2 == 1 {
                for row in 0..LANES as usize {
                    assert_eq!(
                        got[row * dim as usize + dim as usize - 1],
                        0.0,
                        "an odd width's last column is the zero pad"
                    );
                }
            }
        }
    }
}

fn the_centring_answers_the_two_pass_reference() {
    let Some(rig) = Rig::open() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let n = (ROWS * WIDTH) as usize;
    let w = WIDTH as usize;
    let x: Vec<f32> = (0..n as u64)
        .map(|at| bf16_round(50.0 + 4.0 * unit(at)))
        .collect();
    let (_xb, hx) = rig.plane(&x, Dtype::Bf16);
    let (_yb, hy) = rig.empty(n, Dtype::Bf16);
    let xt = Tensor::new(hx, ROWS, WIDTH, Dtype::Bf16);
    let yt = Tensor::new(hy, ROWS, WIDTH, Dtype::Bf16);
    rig.fire(|s| norm::layernorm_no_scale(s, xt, 1e-5, yt).expect("layernorm_no_scale"));

    let mut want = Vec::with_capacity(n);
    for row in 0..ROWS as usize {
        let r = &x[row * w..(row + 1) * w];
        let mean = r.iter().sum::<f32>() / w as f32;
        let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / w as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        want.extend(r.iter().map(|v| bf16_round((v - mean) * inv)));
    }
    within_a_bf16_ulp("layernorm_no_scale", &want, &rig.read(hy, n, Dtype::Bf16));

    let x: Vec<f32> = (0..n as u64)
        .map(|at| 5000.0 + 4.0 * unit(at ^ 0x321))
        .collect();
    let (_xb, hx) = rig.plane(&x, Dtype::F32);
    let (_yb, hy) = rig.empty(n, Dtype::F32);
    let xt = Tensor::new(hx, ROWS, WIDTH, Dtype::F32);
    let yt = Tensor::new(hy, ROWS, WIDTH, Dtype::F32);
    rig.fire(|s| norm::layernorm_no_scale(s, xt, 1e-5, yt).expect("layernorm_no_scale f32"));
    let mut want = Vec::with_capacity(n);
    for row in 0..ROWS as usize {
        let r = &x[row * w..(row + 1) * w];
        let mean = r.iter().sum::<f32>() / w as f32;
        let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / w as f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        want.extend(r.iter().map(|v| (v - mean) * inv));
    }
    let got = rig.read(hy, n, Dtype::F32);
    let worst = want
        .iter()
        .zip(&got)
        .map(|(w, g)| (w - g).abs())
        .fold(0.0f32, f32::max);
    eprintln!("layernorm_no_scale f32: worst absolute {worst:.3e}");
    assert!(
        worst < 1e-3,
        "layernorm_no_scale f32: worst absolute {worst:.3e}"
    );
}

const HEADS: u32 = 3;
const HEAD_DIM: u32 = 32;

fn every_rope_pairing_turns_the_channels_its_form_names() {
    let Some(rig) = Rig::open() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let width = HEADS * HEAD_DIM;
    let n = (ROWS * width) as usize;
    let dims = [8u32, 8, 8, 0];
    let thetas = [64.0f32, 32.0, 16.0, 1.0];
    let rotary: u32 = dims.iter().sum();
    let axes = 3usize;

    let positions: Vec<f32> = (0..(ROWS as usize * axes) as u64)
        .map(|at| (noise(at) % 8) as f32 + 0.25)
        .collect();
    let (_pb, hp) = rig.plane(&positions, Dtype::F32);
    let pt = Tensor::new(hp, ROWS, axes as u32, Dtype::F32);

    for (form, name) in [
        (rope_axes::RopeForm::Interleaved, "interleaved"),
        (rope_axes::RopeForm::Neox, "neox"),
        (rope_axes::RopeForm::Split, "split"),
        (rope_axes::RopeForm::SplitLadder, "split_ladder"),
    ] {
        let x: Vec<f32> = (0..n as u64)
            .map(|at| bf16_round(unit(at ^ 0x2468)))
            .collect();
        let (_xb, hx) = rig.plane(&x, Dtype::Bf16);
        let (_ob, ho) = rig.empty(n, Dtype::Bf16);
        let xt = Tensor::new(hx, ROWS, width, Dtype::Bf16);
        let ot = Tensor::new(ho, ROWS, width, Dtype::Bf16);
        rig.fire(|s| {
            rope_axes::rope_axes(s, xt, pt, dims, thetas, form, rotary, HEAD_DIM, ot)
                .expect("rope_axes")
        });

        let want = rope_reference(&x, &positions, dims, thetas, form, rotary, axes);
        within_a_bf16_ulp(
            &format!("rope_axes {name}"),
            &want,
            &rig.read(ho, n, Dtype::Bf16),
        );
    }
}

fn rope_reference(
    x: &[f32],
    positions: &[f32],
    dims: [u32; 4],
    thetas: [f32; 4],
    form: rope_axes::RopeForm,
    rotary: u32,
    axes: usize,
) -> Vec<f32> {
    let width = (HEADS * HEAD_DIM) as usize;
    let angles = (rotary / 2) as usize;
    let hd = HEAD_DIM as usize;
    let mut out = x.to_vec();
    for row in 0..ROWS as usize {
        let pos = &positions[row * axes..(row + 1) * axes];
        for head in 0..HEADS as usize {
            for angle in 0..angles {
                let idx = head * angles + angle;
                let (lo, hi, cos_v, sin_v);
                if form == rope_axes::RopeForm::SplitLadder {
                    let span: u32 = dims[..axes].iter().sum();
                    let pad = ((HEADS * rotary - span) / 2) as usize;
                    let (c, s) = if idx >= pad {
                        let slot = idx - pad;
                        let axis = slot % axes;
                        let f = slot / axes;
                        let ladder = (dims[axis] / 2) as usize;
                        let e = if ladder > 1 {
                            f as f32 / (ladder - 1) as f32
                        } else {
                            0.0
                        };
                        let a = pos[axis] * thetas[axis].powf(e);
                        (a.cos(), a.sin())
                    } else {
                        (1.0, 0.0)
                    };
                    (lo, hi, cos_v, sin_v) = (angle, angle + angles, c, s);
                } else {
                    let mut axis = 0usize;
                    let mut first_angle = 0usize;
                    let mut first_channel = 0usize;
                    while axis < axes && angle >= first_angle + (dims[axis] / 2) as usize {
                        first_angle += (dims[axis] / 2) as usize;
                        first_channel += dims[axis] as usize;
                        axis += 1;
                    }
                    if axis >= axes {
                        continue;
                    }
                    let within = angle - first_angle;
                    let freq = thetas[axis].powf(-2.0 * within as f32 / dims[axis] as f32);
                    let a = pos[axis] * freq;
                    let (l, h) = match form {
                        rope_axes::RopeForm::Interleaved => {
                            (first_channel + 2 * within, first_channel + 2 * within + 1)
                        }
                        rope_axes::RopeForm::Neox => (angle, angle + angles),
                        _ => (
                            first_channel + within,
                            first_channel + (dims[axis] / 2) as usize + within,
                        ),
                    };
                    (lo, hi, cos_v, sin_v) = (l, h, a.cos(), a.sin());
                }
                let at = row * width + head * hd;
                let a = x[at + lo];
                let b = x[at + hi];
                out[at + lo] = bf16_round(a * cos_v - b * sin_v);
                out[at + hi] = bf16_round(b * cos_v + a * sin_v);
            }
        }
    }
    out
}

fn the_relative_bias_table_reads_the_bucket_torch_lands() {
    let Some(rig) = Rig::open() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    const HEADS: u32 = 6;
    const BUCKETS: u32 = 32;
    const MAX_LEN: u32 = 40;
    const MAX_DISTANCE: f32 = 128.0;
    let span = 2 * MAX_LEN - 1;
    let embedding: Vec<f32> = (0..(BUCKETS * HEADS) as u64)
        .map(|at| bf16_round(unit(at ^ 0xB0B0)))
        .collect();
    let (_eb, he) = rig.plane(&embedding, Dtype::Bf16);
    let (_yb, hy) = rig.empty((HEADS * span) as usize, Dtype::F32);
    rig.fire(|s| {
        pointwise::relative_bucket_bias(
            s,
            Tensor::new(he, BUCKETS, HEADS, Dtype::Bf16),
            MAX_LEN,
            BUCKETS,
            MAX_DISTANCE,
            true,
            Tensor::new(hy, HEADS, span, Dtype::F32),
        )
        .expect("the launch")
    });

    let bucket = |d: i32| -> usize {
        let mut b = 0i32;
        let n_buckets = (BUCKETS / 2) as i32;
        if d > 0 {
            b += n_buckets;
        }
        let n = d.abs();
        let max_exact = n_buckets / 2;
        if n < max_exact {
            return (b + n) as usize;
        }
        let log_ratio = (f64::from(MAX_DISTANCE) / f64::from(max_exact)).ln() as f32;
        let x = ((n as f32) / (max_exact as f32)).ln();
        let scaled = x / log_ratio * (n_buckets - max_exact) as f32;
        let large = (max_exact + scaled as i32).min(n_buckets - 1);
        (b + large) as usize
    };
    let mut want = Vec::with_capacity((HEADS * span) as usize);
    for h in 0..HEADS as usize {
        for c in 0..span as i32 {
            want.push(embedding[bucket(c - (MAX_LEN as i32 - 1)) * HEADS as usize + h]);
        }
    }
    let got = rig.read(hy, (HEADS * span) as usize, Dtype::F32);
    for (i, (w, g)) in want.iter().zip(&got).enumerate() {
        assert!(
            (w - g).abs() < 1e-6,
            "cell {i} (head {}, distance {}) reads {g} and the reference {w}",
            i / span as usize,
            (i % span as usize) as i32 - (MAX_LEN as i32 - 1)
        );
    }
    eprintln!("relative_bucket_bias: {} cells exact", want.len());
}

fn the_head_gate_scales_by_one_logit_a_head() {
    let Some(rig) = Rig::open() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    const HEADS: u32 = 4;
    const HEAD_DIM: u32 = 24;
    let width = HEADS * HEAD_DIM;
    let n = (ROWS * width) as usize;
    let scale = 0.75f32;
    let x: Vec<f32> = (0..n as u64)
        .map(|at| bf16_round(unit(at ^ 0x6161)))
        .collect();
    let g: Vec<f32> = (0..(ROWS * HEADS) as u64)
        .map(|at| bf16_round(2.0 * unit(at ^ 0x2727)))
        .collect();
    let (_xb, hx) = rig.plane(&x, Dtype::Bf16);
    let (_gb, hg) = rig.plane(&g, Dtype::Bf16);
    rig.fire(|s| {
        gate::sigmoid_mul_heads(
            s,
            Tensor::new(hg, ROWS, HEADS, Dtype::Bf16),
            HEAD_DIM,
            scale,
            Tensor::new(hx, ROWS, width, Dtype::Bf16),
        )
        .expect("the launch")
    });
    let mut want = Vec::with_capacity(n);
    for row in 0..ROWS as usize {
        for col in 0..width as usize {
            let logit = g[row * HEADS as usize + col / HEAD_DIM as usize];
            let s = scale / (1.0 + (-logit).exp());
            want.push(bf16_round(x[row * width as usize + col] * s));
        }
    }
    within_a_bf16_ulp(
        "gate_sigmoid_mul_heads",
        &want,
        &rig.read(hx, n, Dtype::Bf16),
    );
}

fn a_pack_and_its_unpack_round_trip_the_rectangle() {
    let Some(rig) = Rig::open() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    const PACKED: u32 = 4;
    let perm: [i32; ROWS as usize] = [5, 1, 6, 2, -1, -1, -1];
    let n = (ROWS * WIDTH) as usize;
    let x: Vec<f32> = (0..n as u64)
        .map(|at| bf16_round(unit(at ^ 0xDEAD)))
        .collect();

    let (_xb, hx) = rig.plane(&x, Dtype::Bf16);
    let (_pb, hp) = rig.i32_plane(&perm);
    let (_tb, ht) = rig.empty((PACKED * WIDTH) as usize, Dtype::Bf16);
    let xt = Tensor::new(hx, ROWS, WIDTH, Dtype::Bf16);
    let pt = Tensor::new(hp, ROWS, 1, Dtype::I32);
    let tight = Tensor::new(ht, PACKED, WIDTH, Dtype::Bf16);

    rig.fire(|s| layout::pack_rows(s, xt, pt, tight).expect("pack"));
    let packed = rig.read(ht, (PACKED * WIDTH) as usize, Dtype::Bf16);
    let w = WIDTH as usize;
    for (i, &from) in perm.iter().take(PACKED as usize).enumerate() {
        let from = from as usize;
        assert_eq!(
            &packed[i * w..(i + 1) * w],
            &x[from * w..(from + 1) * w],
            "packed row {i} is fire row {from}"
        );
    }

    let sentinel = vec![bf16_round(7.5); n];
    let (_ub, hu) = rig.plane(&sentinel, Dtype::Bf16);
    let wide = Tensor::new(hu, ROWS, WIDTH, Dtype::Bf16);
    rig.fire(|s| layout::unpack_rows(s, tight, pt, wide).expect("unpack"));
    let back = rig.read(hu, n, Dtype::Bf16);
    for row in 0..ROWS as usize {
        let named = perm
            .iter()
            .take(PACKED as usize)
            .any(|p| *p as usize == row);
        if named {
            assert_eq!(
                &back[row * w..(row + 1) * w],
                &x[row * w..(row + 1) * w],
                "row {row} came back"
            );
        } else {
            assert!(
                back[row * w..(row + 1) * w].iter().all(|v| *v == 7.5),
                "row {row} is not named by the permutation and must be unwritten"
            );
        }
    }
}

fn the_residual_blend_scores_normalized_and_blends_raw() {
    let Some(rig) = Rig::open() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    const ROWS: u32 = 5;
    const H: u32 = 96;
    const B: usize = 3;
    const EPS: f32 = 1e-6;
    let n = (ROWS * H) as usize;

    for dtype in [Dtype::Bf16, Dtype::F32] {
        let round = |v: f32| {
            if dtype == Dtype::Bf16 {
                bf16_round(v)
            } else {
                v
            }
        };
        let blocks: Vec<f32> = (0..B * n)
            .map(|i| round(unit(i as u64) * (1.0 + (i / n) as f32)))
            .collect();
        let prefix: Vec<f32> = (0..n).map(|i| round(unit(i as u64 + 4242))).collect();
        let wn: Vec<f32> = (0..H as usize)
            .map(|i| round(unit(i as u64 + 11) * 0.5))
            .collect();
        let wp: Vec<f32> = (0..H as usize)
            .map(|i| round(unit(i as u64 + 97) * 0.5))
            .collect();

        let (_pb, hp) = rig.plane(&prefix, dtype);
        let (_bb, hb) = rig.plane(&blocks, dtype);
        let (_nb, hn) = rig.plane(&wn, dtype);
        let (_jb, hj) = rig.plane(&wp, dtype);
        let (_yb, hy) = rig.empty(n, dtype);
        rig.fire(|s| {
            norm::res_blend(
                s,
                Tensor::new(hp, ROWS, H, dtype),
                Tensor::new(hb, ROWS * B as u32, H, dtype),
                B as u32,
                Tensor::new(hn, 1, H, dtype),
                EPS,
                Tensor::new(hj, 1, H, dtype),
                Tensor::new(hy, ROWS, H, dtype),
            )
            .expect("res_blend")
        });

        let mut want = vec![0.0f32; n];
        for t in 0..ROWS as usize {
            let row = |j: usize| -> &[f32] {
                if j < B {
                    &blocks[(j * ROWS as usize + t) * H as usize..][..H as usize]
                } else {
                    &prefix[t * H as usize..][..H as usize]
                }
            };
            let scores: Vec<f32> = (0..=B)
                .map(|j| {
                    let v = row(j);
                    let ss: f32 = v.iter().map(|x| x * x).sum();
                    let scale = (ss / H as f32 + EPS).sqrt().recip();
                    (0..H as usize).map(|h| v[h] * scale * wn[h] * wp[h]).sum()
                })
                .collect();
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
            let sum: f32 = exp.iter().sum();
            for h in 0..H as usize {
                let acc: f32 = (0..=B).map(|j| exp[j] / sum * row(j)[h]).sum();
                want[t * H as usize + h] = round(acc);
            }
        }
        let got = rig.read(hy, n, dtype);
        let bar = if dtype == Dtype::Bf16 { 8e-3 } else { 5e-6 };
        let mut worst = 0.0f32;
        let mut at = 0usize;
        for (i, (w, g)) in want.iter().zip(&got).enumerate() {
            let d = (w - g).abs();
            if d > worst {
                worst = d;
                at = i;
            }
        }
        eprintln!("res_blend {dtype:?}: worst absolute {worst:.3e} at {at}");
        assert!(
            worst <= bar,
            "res_blend {dtype:?}: parts from the reference by {worst:.3e} at {at} \
             (want {}, got {})",
            want[at],
            got[at]
        );
    }
}
