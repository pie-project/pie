#![cfg(feature = "cuda")]

use crate::common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::ssm;
use kernels_cuda::tensor::{RaggedTensor, RecurrentPool, Tensor};

const HEADS: usize = 16;
const GROUPS: usize = 2;
const P: usize = 64;
const N: usize = 128;
const INNER: usize = HEADS * P;
const WIDTH: usize = INNER + 2 * GROUPS * N;
const STRIDE: usize = HEADS * P * N;
const LENS: [usize; 2] = [37, 5];
const SLOT_OF: [i32; 2] = [1, 0];
const EPS: f32 = 1e-5;

struct Rows {
    xbc: Vec<f32>,
    dt: Vec<f32>,
    z: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    d: Vec<f32>,
    norm: Vec<f32>,
}

/// One row's step of every head over `slab`'s slot, the slot folded in
/// place; answers the gated, group-normed output.
fn reference(r: &Rows, row: usize, slab: &mut [f32], slot: usize) -> Vec<f32> {
    let at = &r.xbc[row * WIDTH..(row + 1) * WIDTH];
    let mut y = vec![0f32; INNER];
    for h in 0..HEADS {
        let g = h / (HEADS / GROUPS);
        let raw = r.dt[row * HEADS + h] + r.dt_bias[h];
        let dt = if raw > 20.0 { raw } else { raw.exp().ln_1p() };
        let decay = (-r.a_log[h].exp() * dt).exp();
        for p in 0..P {
            let x = at[h * P + p];
            let s = &mut slab[slot * STRIDE + (h * P + p) * N..][..N];
            let mut acc = 0f32;
            for n in 0..N {
                s[n] = s[n] * decay + dt * x * at[INNER + g * N + n];
                acc += s[n] * at[INNER + (GROUPS + g) * N + n];
            }
            let z = r.z[row * INNER + h * P + p];
            y[h * P + p] = (acc + r.d[h] * x) * z / (1.0 + (-z).exp());
        }
    }
    let group = INNER / GROUPS;
    for g in 0..GROUPS {
        let run = &mut y[g * group..(g + 1) * group];
        let rms = (run.iter().map(|v| v * v).sum::<f32>() / group as f32 + EPS).sqrt();
        for (i, v) in run.iter_mut().enumerate() {
            *v = *v / rms * r.norm[g * group + i];
        }
    }
    y
}

fn worst(what: &str, got: &[u16], want: &[f32]) {
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let off = got
        .iter()
        .zip(want)
        .fold(0f32, |m, (g, w)| m.max((from_bf16(*g) - w).abs()));
    assert!(off <= 3e-2 * scale, "{what}: off by {off} of {scale}");
}

/// The chunked walk over two prompts (one crossing two chunk boundaries)
/// lands each row of the serial recurrence and folds each slot to its last
/// row's state; a step from those slots lands the next row likewise.
#[test]
fn the_mamba2_scan_and_step_answer_the_serial_recurrence() {
    let mut gpu = Gpu::open();
    let total: usize = LENS.iter().sum::<usize>() + LENS.len();
    let mut lcg = Lcg::seeded(0x3a2);
    let (xbc_raw, xbc) = lcg.row(total * WIDTH);
    let (_, dt) = lcg.row(total * HEADS);
    let (z_raw, z) = lcg.row(total * INNER);
    let (norm_raw, norm) = lcg.row(INNER);
    let rows = Rows {
        xbc,
        dt: dt.iter().map(|v| from_bf16(to_bf16(v - 2.0))).collect(),
        z,
        a_log: (0..HEADS).map(|_| 0.5 * lcg.unit()).collect(),
        dt_bias: (0..HEADS).map(|_| 0.5 * lcg.unit()).collect(),
        d: (0..HEADS).map(|_| lcg.unit()).collect(),
        norm,
    };
    let dt_raw: Vec<u16> = rows.dt.iter().map(|v| to_bf16(*v)).collect();
    let seed: Vec<f32> = (0..2 * STRIDE).map(|_| 0.1 * lcg.unit()).collect();

    let prompt: usize = LENS.iter().sum();
    let mut want_slab = seed.clone();
    let mut want = Vec::new();
    let mut row = 0;
    for (lane, len) in LENS.iter().enumerate() {
        for _ in 0..*len {
            want.extend(reference(
                &rows,
                row,
                &mut want_slab,
                SLOT_OF[lane] as usize,
            ));
            row += 1;
        }
    }
    let mut want_step = Vec::new();
    for (lane, &slot) in SLOT_OF.iter().enumerate().take(LENS.len()) {
        want_step.extend(reference(
            &rows,
            prompt + lane,
            &mut want_slab,
            slot as usize,
        ));
    }

    let xbc_at = gpu.up(&xbc_raw);
    let dt_at = gpu.up(&dt_raw);
    let z_at = gpu.up(&z_raw);
    let m = ssm::Mamba2 {
        a_log: Tensor::new(gpu.up(&rows.a_log), 1, HEADS as u32, Dtype::F32),
        dt_bias: Tensor::new(gpu.up(&rows.dt_bias), 1, HEADS as u32, Dtype::F32),
        d: Tensor::new(gpu.up(&rows.d), 1, HEADS as u32, Dtype::F32),
        norm: Tensor::new(gpu.up(&norm_raw), 1, INNER as u32, Dtype::Bf16),
        eps: EPS,
        heads: HEADS as u32,
        groups: GROUPS as u32,
    };
    let slab_at = gpu.up(&seed);
    let slots_at = gpu.up(&SLOT_OF);
    let pool = RecurrentPool {
        slab: Tensor::new(slab_at, 2, STRIDE as u32, Dtype::F32),
        slot_ids: Tensor::new(slots_at, LENS.len() as u32, 1, Dtype::I32),
        slot_stride_elems: STRIDE as i64,
        conv_slab: Tensor::ABSENT,
        conv_stride: 0,
        write_state: true,
        write_state_mask: Tensor::ABSENT,
        commit_len: Tensor::ABSENT,
        begin_at: Tensor::ABSENT,
        fused_decay: false,
        snapshots: kernels_cuda::Snapshots::NONE,
    };
    let at = |base: u64, row: usize, width: usize| base + (row * width * 2) as u64;
    let indptr: Vec<i32> = vec![0, LENS[0] as i32, prompt as i32];
    let indptr_at = gpu.up(&indptr);
    let y_at = gpu.zeros(total * INNER * 2);
    let n = prompt as u32;
    ssm::mamba2_chunked(
        &gpu.ctx(),
        RaggedTensor {
            data: Tensor::new(xbc_at, n, WIDTH as u32, Dtype::Bf16),
            indptr: Tensor::new(indptr_at, 3, 1, Dtype::I32),
        },
        Tensor::new(dt_at, n, HEADS as u32, Dtype::Bf16),
        Tensor::new(z_at, n, INNER as u32, Dtype::Bf16),
        m,
        &pool,
        N as u32,
        &mut Tensor::new(y_at, n, INNER as u32, Dtype::Bf16),
    )
    .expect("the Mamba-2 scan fires");
    gpu.sync();
    worst(
        "the scan's rows",
        &gpu.down::<u16>(y_at, prompt * INNER),
        &want[..],
    );

    let steps = LENS.len() as u32;
    ssm::mamba2_step(
        &gpu.ctx(),
        Tensor::new(at(xbc_at, prompt, WIDTH), steps, WIDTH as u32, Dtype::Bf16),
        Tensor::new(at(dt_at, prompt, HEADS), steps, HEADS as u32, Dtype::Bf16),
        Tensor::new(at(z_at, prompt, INNER), steps, INNER as u32, Dtype::Bf16),
        m,
        &pool,
        N as u32,
        &mut Tensor::new(at(y_at, prompt, INNER), steps, INNER as u32, Dtype::Bf16),
    )
    .expect("the Mamba-2 step fires");
    gpu.sync();
    worst(
        "the step's rows",
        &gpu.down::<u16>(at(y_at, prompt, INNER), steps as usize * INNER),
        &want_step,
    );
    let slab: Vec<f32> = gpu.down(slab_at, 2 * STRIDE);
    let scale = want_slab.iter().fold(0f32, |m, v| m.max(v.abs()));
    let off = slab
        .iter()
        .zip(&want_slab)
        .fold(0f32, |m, (g, w)| m.max((g - w).abs()));
    assert!(
        off <= 2e-2 * scale,
        "the folded slots: off by {off} of {scale}"
    );
}
