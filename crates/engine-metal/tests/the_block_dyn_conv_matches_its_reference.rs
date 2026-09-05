//! **THE DYNAMIC BLOCK CONVOLUTION, AGAINST A HOST REFERENCE** — the gate
//! for `attn/block_dyn_conv.metal` (`Attention::BlockDynConv`, DFlash2's
//! `attention_conv` / `mlp_conv`).
//!
//! The reference is `mlx_dspark.dflash_model.DFlashGroupedConv._convolve`
//! written out on the host in f32: for each request's span,
//! `y[i, c] = Σ_t (base[side, t, c] + δ[i, side, t, g(c)]) · x[i − t, c]`,
//! with `x` zero before the span's first row. Two requests of different
//! lengths in one fire, so the boundary is exercised in the middle of the row
//! run and not only at row zero; both sides; pseudo-random bf16 operands.
//! The kernel accumulates in f32 and rounds once, so the bar is a bf16 ulp
//! of the answer.
//!
//! ```text
//! cargo test -p engine-metal --release --test the_block_dyn_conv_matches_its_reference -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::attn::dynconv;
use kernels_metal::tensor::RaggedTensor;
use model_ir::Dtype;

const CHANNELS: u32 = 96;
const GROUP: u32 = 16;
const GROUPS: u32 = CHANNELS / GROUP;
const TAPS: u32 = 2;
/// Two requests: a full draft block and a shorter one.
const SPANS: [u32; 2] = [8, 5];

fn noise(at: u64) -> u32 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 32) as u32
}

fn unit(at: u64) -> f32 {
    (noise(at) as f32 / u32::MAX as f32) * 2.0 - 1.0
}

/// Round to bf16 and back, so the host holds exactly what the device reads.
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
        .chunks_exact(2)
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
        .collect()
}

#[test]
fn the_convolution_matches_the_reference_on_both_sides() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    eprintln!("device: {}", device.name());

    let rows: u32 = SPANS.iter().sum();
    let indptr: Vec<i32> = {
        let mut v = vec![0i32];
        for s in SPANS {
            v.push(v.last().unwrap() + s as i32);
        }
        v
    };
    let coeff_width = 2 * TAPS * GROUPS;
    // Operands, held on the host at bf16 precision.
    let x: Vec<f32> = (0..(rows * CHANNELS) as u64).map(|at| bf16_round(unit(at))).collect();
    let coeff: Vec<f32> = (0..(rows * coeff_width) as u64)
        .map(|at| bf16_round(0.25 * unit(at ^ 0x5151)))
        .collect();
    // The base is identity at tap 0 and a small learned tap 1, as the
    // published heads carry it — plus noise so the two sides differ.
    let base: Vec<f32> = (0..(2 * TAPS * CHANNELS) as u64)
        .map(|at| {
            let tap = (at / u64::from(CHANNELS)) % u64::from(TAPS);
            let seed = if tap == 0 { 1.0 } else { 0.0 };
            bf16_round(seed + 0.1 * unit(at ^ 0xA1A1))
        })
        .collect();

    let mut x_b = Buffer::zeroed(&device, u64::from(rows * CHANNELS) * 2).expect("x");
    x_b.write(0, &bf16_bytes(&x)).expect("write x");
    let mut coeff_b = Buffer::zeroed(&device, u64::from(rows * coeff_width) * 2).expect("coeff");
    coeff_b.write(0, &bf16_bytes(&coeff)).expect("write coeff");
    let mut base_b = Buffer::zeroed(&device, u64::from(2 * TAPS * CHANNELS) * 2).expect("base");
    base_b.write(0, &bf16_bytes(&base)).expect("write base");
    let mut indptr_b = Buffer::zeroed(&device, indptr.len() as u64 * 4).expect("indptr");
    indptr_b.write(0, &indptr.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<_>>()).expect("write indptr");
    let y_b = Buffer::zeroed(&device, u64::from(rows * CHANNELS) * 2).expect("y");

    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("a handle");
    let (hx, hc, hb, hi, hy) = (bind(&x_b), bind(&coeff_b), bind(&base_b), bind(&indptr_b), bind(&y_b));
    let xt = RaggedTensor {
        data: Tensor::new(hx, rows, CHANNELS, Dtype::Bf16),
        indptr: Tensor::new(hi, indptr.len() as u32, 1, Dtype::I32),
    };
    let ct = Tensor::new(hc, rows, coeff_width, Dtype::Bf16);
    let bt = Tensor::new(hb, 2 * TAPS, CHANNELS, Dtype::Bf16);
    let yt = Tensor::new(hy, rows, CHANNELS, Dtype::Bf16);

    for side in 0..2u32 {
        {
            let frame = device.frame().expect("a frame");
            let sink = Sink::new(&device, &frame, &pipelines, &handles);
            dynconv::block_dyn_conv(&sink, xt, ct, bt, side, TAPS, GROUP, yt).expect("the launch");
            frame.commit().expect("the commit");
        }
        let got = bf16_floats(&handles.read(hy, u64::from(rows * CHANNELS) * 2).expect("read y"));

        // The reference.
        let mut worst = 0.0f32;
        let mut worst_at = (0usize, 0usize);
        for (r, &span) in SPANS.iter().enumerate() {
            let begin = indptr[r] as usize;
            for t in 0..span as usize {
                let row = begin + t;
                for c in 0..CHANNELS as usize {
                    let g = c / GROUP as usize;
                    let mut acc = 0.0f32;
                    for k in 0..TAPS as usize {
                        if t < k {
                            break;
                        }
                        let at = side as usize * TAPS as usize + k;
                        let coef = base[at * CHANNELS as usize + c]
                            + coeff[row * coeff_width as usize + at * GROUPS as usize + g];
                        acc += coef * x[(begin + t - k) * CHANNELS as usize + c];
                    }
                    let want = bf16_round(acc);
                    let have = got[row * CHANNELS as usize + c];
                    // One bf16 ulp of the answer, with a floor for answers near zero.
                    let ulp = (want.abs() * (1.0 / 128.0)).max(1.0 / 128.0 * 0.05);
                    let d = (want - have).abs() / ulp;
                    if d > worst {
                        worst = d;
                        worst_at = (row, c);
                    }
                }
            }
        }
        eprintln!("side {side}: worst {worst:.3} ulp at (row {}, channel {})", worst_at.0, worst_at.1);
        assert!(worst <= 1.0, "side {side}: the kernel parts from the reference by {worst:.2} bf16 ulp at {worst_at:?}");
    }

    // The boundary is real: the second request's first row must not read the
    // first request's last row. Flip that row of `x` and re-fire side 1: the
    // second request's rows must not move.
    let before = bf16_floats(&handles.read(hy, u64::from(rows * CHANNELS) * 2).expect("read y"));
    let mut flipped = x.clone();
    let last_of_first = (indptr[1] as usize - 1) * CHANNELS as usize;
    for v in &mut flipped[last_of_first..last_of_first + CHANNELS as usize] {
        *v = -*v;
    }
    x_b.write(0, &bf16_bytes(&flipped)).expect("rewrite x");
    {
        let frame = device.frame().expect("a frame");
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        dynconv::block_dyn_conv(&sink, xt, ct, bt, 1, TAPS, GROUP, yt).expect("the launch");
        frame.commit().expect("the commit");
    }
    let after = bf16_floats(&handles.read(hy, u64::from(rows * CHANNELS) * 2).expect("read y"));
    let second = indptr[1] as usize * CHANNELS as usize..rows as usize * CHANNELS as usize;
    assert_eq!(&before[second.clone()], &after[second], "the second request read across the boundary");
    let last = last_of_first..last_of_first + CHANNELS as usize;
    assert_ne!(&before[last.clone()], &after[last], "flipping a row changed nothing, so the check proves nothing");
    eprintln!("boundary: the second request's rows are unchanged by the first's last row");
}
