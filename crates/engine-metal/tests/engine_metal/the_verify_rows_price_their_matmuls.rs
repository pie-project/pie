#![cfg(target_vendor = "apple")]
//! Microbench (not a test): the dense projections of a Qwen3.6-35B-A3B speculative verify (8 rows) and of a
//! decode (1 row), u4g64 affine, on each 5-to-8-row matmul the profiles can pick: the register MMA tile
//! (Apple9's default), the multi-row qmv (one read of the weights for every row), and the tiled qmm, at the
//! M4 Pro's split-k. A memory-bound 8-row matmul costs what one row does; how far each path is from that.
//! Run: cargo test --release -p engine-metal --test engine_metal \
//!        the_verify_rows_price_their_matmuls -- --ignored --nocapture

use std::cell::RefCell;

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::linear::quant;
use kernels_metal::{Bank, Tensor};
use poem_ir::Dtype;

const ITERS: usize = 30;
const WARM: usize = 20;

fn noise(at: u64) -> u8 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 40) as u8
}

fn bf16(v: f32) -> [u8; 2] {
    let b = v.to_bits();
    [(b >> 16) as u8, (b >> 24) as u8]
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
#[ignore = "a microbenchmark"]
fn the_verify_rows_price_their_matmuls() {
    let Ok(device) = Context::bind() else { return };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    // the M4 Pro's split-k, under each 5-to-8-row choice
    let splitk = "qmm_splitk = { use = \"fill\", fill_tg = 896, max = 8 }\n";
    // Apple9's own rungs, with the 5-to-8-row rung swapped for each candidate
    let rest = |packs: u32| {
        format!(
            "{{ when = \"rows >= 5 and (bits == 2 or format == fp16)\", use = \"tiled\", crossover_tg = 96 }}, \
         {{ when = \"rows >= 12\", use = \"tiled\", crossover_tg = 96 }}, \
         {{ use = \"qmv\", rows_max = 8, packs = {packs}, crossover_tg = 96 }}"
        )
    };
    let paths: [(&str, String); 4] = [
        (
            "register",
            format!(
                "[kernel]\n{splitk}qmm = [{{ when = \"rows >= 5 and rows <= 8\", use = \"register\" }}, {}]\n",
                rest(1)
            ),
        ),
        ("qmv p1", format!("[kernel]\n{splitk}qmm = [{}]\n", rest(1))),
        ("qmv p2", format!("[kernel]\n{splitk}qmm = [{}]\n", rest(2))),
        (
            "tiled",
            format!(
                "[kernel]\n{splitk}qmm = [{{ when = \"rows >= 5\", use = \"tiled\", crossover_tg = 96 }}, {{ use = \"qmv\", rows_max = 8, packs = 1, crossover_tg = 96 }}]\n"
            ),
        ),
    ];
    let tunings: Vec<(&str, kernels_metal::Tuning)> = paths
        .iter()
        .map(|(n, t)| (*n, engine_metal::chip::tuning(t, false)))
        .collect();
    let keep: RefCell<Vec<Buffer>> = RefCell::new(Vec::new());
    let plane = |rows: u32, width: u32, dtype: Dtype| -> Option<Tensor> {
        let bytes = (u64::from(rows) * u64::from(width) * dtype.bits() / 8).max(16384);
        let buffer = Buffer::zeroed(&device, bytes).ok()?;
        let handle = handles.bind(&buffer, 0, bytes).ok()?;
        keep.borrow_mut().push(buffer);
        Some(Tensor::new(handle, rows, width, dtype))
    };
    let upload = |bytes: &[u8]| {
        let mut b = Buffer::zeroed(&device, bytes.len() as u64).expect("plane");
        b.write(0, bytes).expect("upload");
        b
    };
    let mut total = [0f64; 5];
    // (what, K, N): the verify's dense projections, from the 8-row fire's kernel profile
    for (what, k, n) in [
        ("lm_head", 2048u32, 248_320u32),
        ("gdn in_proj", 2048, 12_288),
        ("attn/gdn out", 4096, 2048),
        ("q/kv proj", 2048, 8192),
        ("shared gate_up", 2048, 1024),
        ("shared down", 512, 2048),
    ] {
        let groups = k / 64;
        let codes: Vec<u8> = (0..u64::from(k) * u64::from(n) / 2).map(noise).collect();
        let scales: Vec<u8> = (0..n * groups)
            .flat_map(|a| bf16(0.003 + (a % 7) as f32 * 1e-4))
            .collect();
        let biases: Vec<u8> = (0..n * groups)
            .flat_map(|a| bf16(((a % 9) as f32 - 4.0) / 64.0))
            .collect();
        let (cb, sb, bb) = (upload(&codes), upload(&scales), upload(&biases));
        let t = |b: &Buffer, rows: u32, width: u32, dtype: Dtype| {
            Tensor::new(
                handles.bind(b, 0, b.bytes()).expect("bind"),
                rows,
                width,
                dtype,
            )
        };
        let bank = Bank {
            mpp_codes: None,
            codes: t(&cb, n, k, Dtype::U4g64),
            scales: t(&sb, n, groups, Dtype::Bf16),
            biases: Some(t(&bb, n, groups, Dtype::Bf16)),
            group: 64,
            bits: 4,
            ld: k,
        };
        let mut line = format!(
            "{what:15} K={k:5} N={n:6} MB={:6.1}",
            f64::from(k) * f64::from(n) * 0.5625 / 1e6
        );
        for (col, (name, tuning)) in tunings.iter().enumerate() {
            for rows in [1u32, 8] {
                if rows == 1 && col > 0 {
                    continue; // one row takes the qmv whatever the 5-to-8-row rule says
                }
                let x: Vec<u8> = (0..rows * k)
                    .flat_map(|a| bf16((f32::from(noise(u64::from(a))) - 127.0) / 256.0))
                    .collect();
                let xb = upload(&x);
                let yb = Buffer::zeroed(&device, (u64::from(rows) * u64::from(n) * 2).max(16384))
                    .expect("y");
                let (xt, yt) = (t(&xb, rows, k, Dtype::Bf16), t(&yb, rows, n, Dtype::Bf16));
                let mut v = Vec::new();
                let mut ok = true;
                for i in 0..(ITERS + WARM) {
                    let frame = device.frame().expect("frame");
                    let sink = Sink::new(&device, &frame, &pipelines, &handles);
                    let scratch = quant::Scratch {
                        precast: &|r, c| plane(r, c, Dtype::F16),
                        partials: &|r, w| plane(r, w, Dtype::F32),
                        table: None,
                    };
                    if quant::matmul(&sink, tuning, xt, bank, yt, scratch, 32).is_err() {
                        ok = false;
                        break;
                    }
                    let s = frame.commit_timed().expect("commit");
                    if i >= WARM {
                        v.push(s * 1e6);
                    }
                    keep.borrow_mut().clear();
                }
                if !ok {
                    line += &format!("  {name}x{rows} n/a");
                    continue;
                }
                let us = median(v);
                if rows == 1 {
                    total[0] += us;
                    line += &format!("  qmv x1 {us:8.1}");
                } else {
                    total[col + 1] += us;
                    line += &format!("  {name} x8 {us:8.1}");
                }
            }
        }
        eprintln!("{line}");
    }
    eprintln!(
        "sum over the six (one layer's worth is not this; the ratio is what matters): x1 {:.1}  register {:.1}  qmv p1 {:.1}  qmv p2 {:.1}  tiled {:.1} us",
        total[0], total[1], total[2], total[3], total[4]
    );
}
