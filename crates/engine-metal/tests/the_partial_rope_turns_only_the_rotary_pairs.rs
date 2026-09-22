#![cfg(target_vendor = "apple")]

//! `elementwise.rope_partial` on Metal rotates the first `rotary_dim`
//! entries of a head as pairs `(i, i + rotary_dim/2)` at
//! `pos * theta^(-2i/rotary_dim)` and leaves the rest alone, the way HF
//! transformers, mlx-lm and llama.cpp do. Positions sit far from zero: the
//! head-width convention this replaced agrees with the rotary-width one at
//! position 0 and drifts with distance.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::elemwise::rope;
use model_ir::Dtype;

fn noise(at: u64) -> u32 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5E5E_1234_9ABC_DEF0;
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

/// The reference: pairs `(i, i + rotary_dim/2)` turned by
/// `pos * theta^(-2i/rotary_dim)`, everything past `rotary_dim` untouched.
fn turn(head: &mut [f32], pos: i32, rotary_dim: usize, theta: f32) {
    let rope_half = rotary_dim / 2;
    for dp in 0..rope_half {
        let (a, b) = (head[dp], head[dp + rope_half]);
        let freq = theta.powf(-2.0 * dp as f32 / rotary_dim as f32);
        let (s, c) = (pos as f32 * freq).sin_cos();
        head[dp] = a * c - b * s;
        head[dp + rope_half] = b * c + a * s;
    }
}

fn check(
    device: &Context,
    handles: &Handles,
    pipelines: &Pipelines,
    head_dim: u32,
    q_heads: u32,
    kv_heads: u32,
    rotary_dim: u32,
) {
    let rows: u32 = 6;
    let (q_width, k_width) = (q_heads * head_dim, kv_heads * head_dim);
    let salt = u64::from(head_dim) ^ (u64::from(rotary_dim) << 8);
    let q_in: Vec<f32> = (0..u64::from(rows * q_width))
        .map(|at| bf16_round(unit(at ^ salt)))
        .collect();
    let k_in: Vec<f32> = (0..u64::from(rows * k_width))
        .map(|at| bf16_round(unit((at ^ salt) | 1 << 40)))
        .collect();
    let positions: Vec<i32> = (0..rows as i32).map(|r| 700 + 331 * r).collect();
    let theta = 1.0e7f32;

    let mut q_b = Buffer::zeroed(device, u64::from(rows * q_width) * 2).expect("q");
    q_b.write(0, &bf16_bytes(&q_in)).expect("write q");
    let mut k_b = Buffer::zeroed(device, u64::from(rows * k_width) * 2).expect("k");
    k_b.write(0, &bf16_bytes(&k_in)).expect("write k");
    let mut p_b = Buffer::zeroed(device, u64::from(rows) * 4).expect("positions");
    p_b.write(
        0,
        &positions
            .iter()
            .flat_map(|p| p.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .expect("write positions");
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("a handle");
    let (hq, hk, hp) = (bind(&q_b), bind(&k_b), bind(&p_b));
    let q = Tensor::new(hq, rows, q_width, Dtype::Bf16);
    let k = Tensor::new(hk, rows, k_width, Dtype::Bf16);
    let p = Tensor::new(hp, rows, 1, Dtype::I32);

    {
        let frame = device.frame().expect("a frame");
        let sink = Sink::new(device, &frame, pipelines, handles);
        rope::partial(&sink, q, k, p, rotary_dim, head_dim, theta).expect("the launch");
        frame.commit().expect("the commit");
    }
    let got_q = bf16_floats(
        &handles
            .read(hq, u64::from(rows * q_width) * 2)
            .expect("read q"),
    );
    let got_k = bf16_floats(
        &handles
            .read(hk, u64::from(rows * k_width) * 2)
            .expect("read k"),
    );

    for (label, src, got, heads, width) in [
        ("q", &q_in, &got_q, q_heads, q_width),
        ("k", &k_in, &got_k, kv_heads, k_width),
    ] {
        for (r, &position) in positions.iter().enumerate() {
            for h in 0..heads as usize {
                let at = r * width as usize + h * head_dim as usize;
                let mut want = src[at..at + head_dim as usize].to_vec();
                turn(&mut want, position, rotary_dim as usize, theta);
                for (i, want) in want.into_iter().enumerate() {
                    let g = got[at + i];
                    assert!(
                        (g - want).abs() <= want.abs() * (1.0 / 64.0) + 1.5e-2,
                        "{label} head_dim {head_dim} rotary {rotary_dim}: [{r}][{h}][{i}] = {g}, want {want}"
                    );
                }
            }
        }
    }
}

#[test]
fn the_partial_rope_turns_only_the_rotary_pairs() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    // a full rotary head turns end to end
    check(&device, &handles, &pipelines, 128, 4, 2, 128);
    // the shape Qwen3 A3B asks for: head_dim 256, rotary_dim 64
    check(&device, &handles, &pipelines, 256, 16, 2, 64);
    check(&device, &handles, &pipelines, 128, 8, 1, 64);
}
