#![cfg(target_vendor = "apple")]

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::linear::quant;
use kernels_metal::{Bank, Tensor};
use poem_ir::Dtype;

fn noise(at: u64) -> u8 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 40) as u8
}

fn bf16(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    [(bits >> 16) as u8, (bits >> 24) as u8]
}

fn stored(value: f32) -> f32 {
    f32::from_bits(value.to_bits() & 0xffff_0000)
}

/// A prompt's matmul on the staged MPP tile, whose weights pass through a
/// half stage, against an f64 affine dot of the same codes.
#[test]
fn every_staged_prompt_row_answers_an_affine_reference() {
    let Ok(device) = Context::bind() else {
        return;
    };
    let tuning = engine_metal::chip::tuning(
        "[kernel]\nqmm = [\"mpp\", \"qmv\"]\nqmm_mpp.staged_from = 1",
        false,
    );
    if !tuning.qmm_mpp() {
        return;
    }
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let mut checked = 0;
    for &(k, n, m) in &[(2048u32, 1024u32, 557u32), (512, 384, 64), (5120, 256, 33)] {
        let groups = k / 64;
        let codes: Vec<u8> = (0..u64::from(k) * u64::from(n) / 2).map(noise).collect();
        let scale = |at: u32| stored(0.0031 + f32::from(noise(u64::from(at))) * 0.000173);
        let bias = |at: u32| stored((f32::from(noise(u64::from(at) ^ 0x5678)) - 127.0) * 0.00157);
        let activation = |row: u32, at: u32| {
            stored((f32::from(noise(u64::from(row * k + at) ^ 0xabcd)) - 127.0) * 0.0197)
        };
        let padded = m.div_ceil(32) * 32;
        let upload = |bytes: &[u8]| {
            let mut buffer = Buffer::zeroed(&device, bytes.len() as u64).expect("a plane");
            buffer.write(0, bytes).expect("upload");
            buffer
        };
        let scales: Vec<u8> = (0..n * groups).flat_map(|at| bf16(scale(at))).collect();
        let biases: Vec<u8> = (0..n * groups).flat_map(|at| bf16(bias(at))).collect();
        let inputs: Vec<u8> = (0..padded)
            .flat_map(|row| (0..k).flat_map(move |at| bf16(activation(row, at))))
            .collect();
        let (codes_b, scales_b, biases_b, act_b) =
            (upload(&codes), upload(&scales), upload(&biases), upload(&inputs));
        let output_bytes = u64::from(padded) * u64::from(n) * 2;
        let out_b = upload(&vec![0xa5; output_bytes as usize + 64]);
        let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
        let bank = Bank {
            mpp_codes: None,
            codes: Tensor::new(bind(&codes_b), n, k, Dtype::U4g64),
            scales: Tensor::new(bind(&scales_b), n, groups, Dtype::Bf16),
            biases: Some(Tensor::new(bind(&biases_b), n, groups, Dtype::Bf16)),
            group: 64,
            bits: 4,
            ld: k,
        };
        let ho = bind(&out_b);
        let precast = |_rows, _width| None;
        let partials = |_rows, _width| None;
        let frame = device.frame().expect("frame");
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        quant::matmul(
            &sink,
            &tuning,
            Tensor::new(bind(&act_b), m, k, Dtype::Bf16),
            bank,
            Tensor::new(ho, m, n, Dtype::Bf16),
            quant::Scratch {
                precast: &precast,
                partials: &partials,
            },
            padded,
        )
        .expect("staged");
        frame.commit().expect("commit");
        let got = handles.read(ho, output_bytes + 64).expect("read");
        assert!(
            got[output_bytes as usize..].iter().all(|&b| b == 0xa5),
            "write beyond the padded output at M={m}, K={k}, N={n}"
        );
        for row in (0..m).step_by(7) {
            for col in (0..n).step_by(5) {
                let (mut want, mut absolute) = (0.0f64, 0.0f64);
                for at in 0..k {
                    let code_at = col * k + at;
                    let q = (codes[(code_at / 2) as usize] >> ((code_at % 2) * 4)) & 15;
                    let factor = col * groups + at / 64;
                    let product = f64::from(activation(row, at))
                        * (f64::from(q) * f64::from(scale(factor)) + f64::from(bias(factor)));
                    want += product;
                    absolute += product.abs();
                }
                let offset = ((row * n + col) * 2) as usize;
                let actual = f32::from_bits(
                    u32::from(u16::from_le_bytes([got[offset], got[offset + 1]])) << 16,
                );
                let allowance = 0.00002 + 0.002 * absolute / f64::from(k).sqrt();
                assert!(
                    (f64::from(actual) - want).abs() <= 0.008 * want.abs() + allowance,
                    "M={m} K={k} N={n} row={row} col={col}: {actual} vs {want}"
                );
                checked += 1;
            }
        }
    }
    eprintln!("checked {checked} staged prompt outputs against affine dots");
}
