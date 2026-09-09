#![cfg(target_vendor = "apple")]

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::attn::ragged::{self, Arm, RaggedMask};
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

const Q_LENS: [i32; 3] = [40, 9, 20];
const KV_LENS: [i32; 3] = [24, 5, 33];

const Q_HEADS: u32 = 4;
const KV_HEADS: u32 = 2;

fn indptr(lens: &[i32]) -> Vec<i32> {
    let mut v = vec![0i32];
    for l in lens {
        v.push(v.last().unwrap() + l);
    }
    v
}

struct Rig {
    device: Context,
    handles: Handles,
    pipelines: Pipelines,
}

impl Rig {
    fn open() -> Option<Self> {
        Some(Self {
            device: Context::bind().ok()?,
            handles: Handles::new(),
            pipelines: Pipelines::new(),
        })
    }

    fn bf16(&self, data: &[f32]) -> (Buffer, u32) {
        let bytes = bf16_bytes(data);
        let mut b = Buffer::zeroed(&self.device, bytes.len() as u64).expect("a plane");
        b.write(0, &bytes).expect("write");
        let h = self.handles.bind(&b, 0, b.bytes()).expect("a handle");
        (b, h)
    }

    fn i32s(&self, data: &[i32]) -> (Buffer, u32) {
        let bytes: Vec<u8> = data.iter().flat_map(|i| i.to_le_bytes()).collect();
        let mut b = Buffer::zeroed(&self.device, bytes.len() as u64).expect("a plane");
        b.write(0, &bytes).expect("write");
        let h = self.handles.bind(&b, 0, b.bytes()).expect("a handle");
        (b, h)
    }

    fn f32s(&self, data: &[f32]) -> (Buffer, u32) {
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let mut b = Buffer::zeroed(&self.device, bytes.len() as u64).expect("a plane");
        b.write(0, &bytes).expect("write");
        let h = self.handles.bind(&b, 0, b.bytes()).expect("a handle");
        (b, h)
    }

    fn empty_bf16(&self, elements: usize) -> (Buffer, u32) {
        let b = Buffer::zeroed(&self.device, (elements * 2) as u64).expect("a plane");
        let h = self.handles.bind(&b, 0, b.bytes()).expect("a handle");
        (b, h)
    }

    fn read_bf16(&self, handle: u32, elements: usize) -> Vec<f32> {
        bf16_floats(
            &self
                .handles
                .read(handle, (elements * 2) as u64)
                .expect("read"),
        )
    }

    fn fire(&self, f: impl FnOnce(&Sink<'_>)) {
        let frame = self.device.frame().expect("a frame");
        let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
        f(&sink);
        frame.commit().expect("the commit");
    }
}

#[allow(clippy::too_many_arguments)]
fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    qp: &[i32],
    kp: &[i32],
    head_dim: usize,
    sm_scale: f32,
    keep: &dyn Fn(usize, usize, usize) -> bool,
    bias: &dyn Fn(usize, usize, usize, usize) -> f32,
) -> Vec<f32> {
    let qh = Q_HEADS as usize;
    let kh = KV_HEADS as usize;
    let rows = *qp.last().unwrap() as usize;
    let group = qh / kh;
    let mut out = vec![0.0f32; rows * qh * head_dim];
    for s in 0..qp.len() - 1 {
        let (qb, qe) = (qp[s] as usize, qp[s + 1] as usize);
        let (kb, ke) = (kp[s] as usize, kp[s + 1] as usize);
        for row in qb..qe {
            for head in 0..qh {
                let kvh = head / group;
                let mut scores = Vec::with_capacity(ke - kb);
                for j in kb..ke {
                    if !keep(s, row, j) {
                        scores.push(f32::NEG_INFINITY);
                        continue;
                    }
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot +=
                            q[(row * qh + head) * head_dim + d] * k[(j * kh + kvh) * head_dim + d];
                    }
                    scores.push(dot * sm_scale + bias(s, row, j, head));
                }
                let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if !m.is_finite() {
                    continue;
                }
                let mut denom = 0.0f32;
                let mut acc = vec![0.0f32; head_dim];
                for (n, j) in (kb..ke).enumerate() {
                    if !scores[n].is_finite() {
                        continue;
                    }
                    let w = (scores[n] - m).exp();
                    denom += w;
                    for d in 0..head_dim {
                        acc[d] += w * v[(j * kh + kvh) * head_dim + d];
                    }
                }
                for d in 0..head_dim {
                    out[(row * qh + head) * head_dim + d] = acc[d] / denom;
                }
            }
        }
    }
    out
}

fn compare(what: &str, want: &[f32], got: &[f32], bar: f32) {
    assert_eq!(want.len(), got.len(), "{what}: length");
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        let d = (w - g).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    eprintln!("{what}: worst absolute {worst:.3e} at {at}");
    assert!(
        worst <= bar,
        "{what}: parts from the reference by {worst:.3e} at {at} (want {}, got {})",
        want[at],
        got[at]
    );
}

fn run_case(rig: &Rig, head_dim: usize, mask_kind: &str) {
    let qp = indptr(&Q_LENS);
    let kp = indptr(&KV_LENS);
    let q_rows = *qp.last().unwrap() as usize;
    let kv_rows = *kp.last().unwrap() as usize;
    let qh = Q_HEADS as usize;
    let kh = KV_HEADS as usize;
    let sm_scale = (head_dim as f32).sqrt().recip();

    let q: Vec<f32> = (0..(q_rows * qh * head_dim) as u64)
        .map(|at| bf16_round(unit(at)))
        .collect();
    let k: Vec<f32> = (0..(kv_rows * kh * head_dim) as u64)
        .map(|at| bf16_round(unit(at ^ 0xA5A5)))
        .collect();
    let v: Vec<f32> = (0..(kv_rows * kh * head_dim) as u64)
        .map(|at| bf16_round(unit(at ^ 0x5A5A)))
        .collect();

    let (_qb, hq) = rig.bf16(&q);
    let (_kb, hk) = rig.bf16(&k);
    let (_vb, hv) = rig.bf16(&v);
    let (_qpb, hqp) = rig.i32s(&qp);
    let (_kpb, hkp) = rig.i32s(&kp);
    let (_ob, ho) = rig.empty_bf16(q_rows * qh * head_dim);

    let qt = Tensor::new(hq, q_rows as u32, (qh * head_dim) as u32, Dtype::Bf16);
    let kt = Tensor::new(hk, kv_rows as u32, (kh * head_dim) as u32, Dtype::Bf16);
    let vt = Tensor::new(hv, kv_rows as u32, (kh * head_dim) as u32, Dtype::Bf16);
    let qpt = Tensor::new(hqp, qp.len() as u32, 1, Dtype::I32);
    let kpt = Tensor::new(hkp, kp.len() as u32, 1, Dtype::I32);
    let ot = Tensor::new(ho, q_rows as u32, (qh * head_dim) as u32, Dtype::Bf16);

    let max_len: u32 = 24;
    let span = (2 * max_len - 1) as usize;
    let q_tags: Vec<i32> = (0..q_rows)
        .map(|r| if r % 5 == 0 { (r % 3) as i32 } else { -1 })
        .collect();
    let kv_tags: Vec<i32> = (0..kv_rows)
        .map(|r| if r % 4 == 0 { (r % 3) as i32 } else { -1 })
        .collect();
    let table: Vec<f32> = (0..(qh * span) as u64)
        .map(|at| 0.5 * unit(at ^ 0xB1A5))
        .collect();

    let (_qtb, hqt) = rig.i32s(&q_tags);
    let (_ktb, hkt) = rig.i32s(&kv_tags);
    let (_tb, htab) = rig.f32s(&table);

    let mask = match mask_kind {
        "segments" => RaggedMask::Segments,
        "tags" => RaggedMask::ReferenceTags {
            q_tags: Tensor::new(hqt, q_rows as u32, 1, Dtype::I32),
            kv_tags: Tensor::new(hkt, kv_rows as u32, 1, Dtype::I32),
        },
        _ => RaggedMask::RelativeBias {
            table: Tensor::new(htab, qh as u32, span as u32, Dtype::F32),
            max_len,
        },
    };

    let keep = |_s: usize, row: usize, j: usize| -> bool {
        if mask_kind != "tags" {
            return true;
        }
        let t = q_tags[row];
        t < 0 || kv_tags[j] == t
    };
    let bias = |s: usize, row: usize, j: usize, head: usize| -> f32 {
        if mask_kind != "bias" {
            return 0.0;
        }
        let qi = row as i64 - qp[s] as i64;
        let kj = j as i64 - kp[s] as i64;
        let at = (kj - qi + max_len as i64 - 1).clamp(0, span as i64 - 1) as usize;
        table[head * span + at]
    };
    let want = reference(&q, &k, &v, &qp, &kp, head_dim, sm_scale, &keep, &bias);

    for (arm, bar) in [(Arm::Scalar, 3e-3f32), (Arm::Tiled, 4e-3f32)] {
        if arm == Arm::Tiled && !(head_dim == 64 || head_dim == 128) {
            continue;
        }
        rig.fire(|s| {
            ragged::fire(
                s,
                arm,
                qt,
                kt,
                vt,
                qpt,
                kpt,
                head_dim as u32,
                KV_HEADS,
                sm_scale,
                &mask,
                ot,
            )
            .expect("the launch")
        });
        let got = rig.read_bf16(ho, q_rows * qh * head_dim);
        compare(
            &format!("ragged {arm:?} d{head_dim} {mask_kind}"),
            &want,
            &got,
            bar,
        );
    }
}

#[test]
fn the_ragged_attention_answers_its_host_reference_every_case() {
    both_arms_answer_the_reference_under_every_mask();
    a_head_width_the_matrix_unit_cannot_tile_still_answers();
}

fn both_arms_answer_the_reference_under_every_mask() {
    let Some(rig) = Rig::open() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    eprintln!("device: {}", rig.device.name());
    for head_dim in [64usize, 128] {
        for mask in ["segments", "tags", "bias"] {
            run_case(&rig, head_dim, mask);
        }
    }
}

fn a_head_width_the_matrix_unit_cannot_tile_still_answers() {
    let Some(rig) = Rig::open() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    run_case(&rig, 32, "segments");
}
