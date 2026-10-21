#![cfg(target_vendor = "apple")]
//! The routed MLP on the register tile (`moe_sg.metal`: expert tiles, the
//! gate-up landing its SwiGLU as the down's table, the down into sorted rows)
//! against the per-route path (`matmul_select_quant`, `swiglu`, again, and
//! the fold), with and without the shared expert riding along as tile 0.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::linear::{mlp, moe};
use kernels_metal::{Bank, Tensor};
use poem_ir::Dtype;

const EXPERTS: u32 = 24;
const TOP_K: u32 = 8;
const HIDDEN: u32 = 2048;
const INTER: u32 = 512;

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

fn floats(raw: &[u8]) -> Vec<f32> {
    raw.chunks(2)
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
        .collect()
}

/// A 4-bit, 64-group affine bank of `n` rows of `k` codes, seeded by `seed`.
fn bank_of(device: &Context, handles: &Handles, k: u32, n: u32, seed: u64) -> (Bank, [Buffer; 3]) {
    let groups = k / 64;
    let upload = |bytes: &[u8]| {
        let mut buffer = Buffer::zeroed(device, bytes.len() as u64).expect("a plane");
        buffer.write(0, bytes).expect("upload");
        buffer
    };
    let codes: Vec<u8> = (0..u64::from(k) * u64::from(n) / 2)
        .map(|at| noise(at ^ seed))
        .collect();
    let factor = |at: u32, salt: u64, base: f32, step: f32| {
        bf16(base + f32::from(noise(u64::from(at) ^ seed ^ salt)) * step)
    };
    let scales: Vec<u8> = (0..n * groups)
        .flat_map(|at| factor(at, 0x11, 0.0031, 0.000173))
        .collect();
    let biases: Vec<u8> = (0..n * groups)
        .flat_map(|at| factor(at, 0x5678, -0.2, 0.00157))
        .collect();
    let planes = [upload(&codes), upload(&scales), upload(&biases)];
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
    let bank = Bank {
        mpp_codes: None,
        codes: Tensor::new(bind(&planes[0]), n, k, Dtype::U4g64),
        scales: Tensor::new(bind(&planes[1]), n, groups, Dtype::Bf16),
        biases: Some(Tensor::new(bind(&planes[2]), n, groups, Dtype::Bf16)),
        group: 64,
        bits: 4,
        ld: k,
    };
    (bank, planes)
}

/// `tokens` rows of `TOP_K` distinct experts, consecutive rows sharing half.
fn routes(tokens: u32, seed: u64) -> Vec<i32> {
    let mut out: Vec<i32> = Vec::new();
    let mut n = seed;
    for t in 0..tokens {
        let mut row: Vec<i32> = Vec::new();
        if t > 0 {
            row.extend_from_slice(&out[((t - 1) * TOP_K) as usize..((t - 1) * TOP_K + TOP_K / 2) as usize]);
        }
        while row.len() < TOP_K as usize {
            n += 1;
            let e = i32::from(noise(n)) % EXPERTS as i32;
            if !row.contains(&e) {
                row.push(e);
            }
        }
        out.extend(row);
    }
    out
}

/// The largest difference of `got` from `want`, relative to `want`'s largest magnitude.
fn spread(want: &[f32], got: &[f32]) -> f32 {
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    want.iter().zip(got).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / scale
}

#[test]
fn the_register_routed_mlp_answers_the_per_route_path() {
    let Ok(device) = Context::bind() else {
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let (gate_up, _g) = bank_of(&device, &handles, HIDDEN, EXPERTS * 2 * INTER, 0x1);
    let (down, _d) = bank_of(&device, &handles, INTER, EXPERTS * HIDDEN, 0x2);
    let (shared_gate_up, _sg) = bank_of(&device, &handles, HIDDEN, 2 * INTER, 0x3);
    let (shared_down, _sd) = bank_of(&device, &handles, INTER, HIDDEN, 0x4);
    let plane = |bytes: u64| Buffer::zeroed(&device, bytes.max(4)).expect("a plane");
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
    let upload = |bytes: &[u8]| {
        let mut buffer = plane(bytes.len() as u64);
        buffer.write(0, bytes).expect("upload");
        buffer
    };
    let mut checked = 0;
    for tokens in [2u32, 5, 8] {
        for fragments in [1u32, 2] {
            for shared in [false, true] {
                let pairs = tokens * TOP_K;
                let r = routes(tokens, u64::from(tokens) * 31);
                let x: Vec<u8> = (0..tokens * HIDDEN)
                    .flat_map(|at| bf16((f32::from(noise(u64::from(at) ^ 0x77)) - 127.0) / 64.0))
                    .collect();
                let weights: Vec<u8> = (0..pairs)
                    .flat_map(|at| (f32::from(noise(u64::from(at) ^ 0x99)) / 2048.0).to_le_bytes())
                    .collect();
                let xb = upload(&x);
                let rb = upload(&r.iter().flat_map(|e| e.to_le_bytes()).collect::<Vec<u8>>());
                let wb = upload(&weights);
                let xt = Tensor::new(bind(&xb), tokens, HIDDEN, Dtype::Bf16);
                let rt = Tensor::new(bind(&rb), tokens, TOP_K, Dtype::I32);
                let wt = Tensor::new(bind(&wb), tokens, TOP_K, Dtype::F32);

                // The per-route path.
                let packed = plane(u64::from(pairs) * u64::from(2 * INTER) * 2);
                let hidden = plane(u64::from(pairs) * u64::from(INTER) * 2);
                let routed = plane(u64::from(pairs) * u64::from(HIDDEN) * 2);
                let folded = plane(u64::from(tokens) * u64::from(HIDDEN) * 2);
                let own_packed = plane(u64::from(tokens) * u64::from(2 * INTER) * 2);
                let own_hidden = plane(u64::from(tokens) * u64::from(INTER) * 2);
                let own = plane(u64::from(tokens) * u64::from(HIDDEN) * 2);
                let zero = upload(&vec![0u8; tokens as usize * 4]);
                let (pt, ht, yt, ft) = (
                    Tensor::new(bind(&packed), pairs, 2 * INTER, Dtype::Bf16),
                    Tensor::new(bind(&hidden), pairs, INTER, Dtype::Bf16),
                    Tensor::new(bind(&routed), pairs, HIDDEN, Dtype::Bf16),
                    Tensor::new(bind(&folded), tokens, HIDDEN, Dtype::Bf16),
                );
                let (opt, oht, ot, zt) = (
                    Tensor::new(bind(&own_packed), tokens, 2 * INTER, Dtype::Bf16),
                    Tensor::new(bind(&own_hidden), tokens, INTER, Dtype::Bf16),
                    Tensor::new(bind(&own), tokens, HIDDEN, Dtype::Bf16),
                    Tensor::new(bind(&zero), tokens, 1, Dtype::I32),
                );

                // The register path's rooms.
                let tiles = moe::register_tiles(tokens, TOP_K, EXPERTS, shared);
                let room = |width: u64, dtype: Dtype, bytes: u64| {
                    let b = plane(bytes);
                    let t = Tensor::new(bind(&b), 1, u32::try_from(width).unwrap(), dtype);
                    (b, t)
                };
                let (_a, tiles_t) = room(u64::from(tiles) + 1, Dtype::I32, (u64::from(tiles) + 1) * 4);
                let (_b, rows_t) = room(u64::from(tiles) * 8, Dtype::I32, u64::from(tiles) * 32);
                let (_c, inv_t) = room(u64::from(pairs), Dtype::I32, u64::from(pairs) * 4);
                let in_width = u64::from(tiles) * moe::register_table_width(HIDDEN);
                let (_e, table_t) = room(in_width, Dtype::Bf16, in_width * 2);
                let mid_width = u64::from(tiles) * moe::register_table_width(INTER);
                let (_f, hidden_t) = room(mid_width, Dtype::Bf16, mid_width * 2);
                let out_bytes = u64::from(tiles) * 8 * u64::from(HIDDEN) * 2;
                let (yb, y_t) = room(u64::from(tiles) * 8 * u64::from(HIDDEN), Dtype::Bf16, out_bytes);
                let rooms = moe::RegisterRooms {
                    tiles: tiles_t,
                    rows: rows_t,
                    inv: inv_t,
                    table: table_t,
                    hidden: hidden_t,
                    y: y_t,
                };
                let sorted_fold = plane(u64::from(tokens) * u64::from(HIDDEN) * 2);
                let sft = Tensor::new(bind(&sorted_fold), tokens, HIDDEN, Dtype::Bf16);

                let frame = device.frame().expect("frame");
                let sink = Sink::new(&device, &frame, &pipelines, &handles);
                moe::matmul_select_quant(&sink, xt, gate_up, rt, pt).expect("gate-up");
                mlp::swiglu(&sink, pt, INTER, ht).expect("swiglu");
                moe::matmul_select_quant(&sink, ht, down, rt, yt).expect("down");
                moe::weighted_sum(&sink, yt, wt, ft).expect("fold");
                moe::matmul_select_quant(&sink, xt, shared_gate_up, zt, opt).expect("own gate-up");
                mlp::swiglu(&sink, opt, INTER, oht).expect("own swiglu");
                moe::matmul_select_quant(&sink, oht, shared_down, zt, ot).expect("own down");

                let own_banks = |bank| shared.then_some(bank);
                moe::register_group(&sink, xt, rt, EXPERTS, shared, rooms).expect("group");
                moe::register_glu(
                    &sink,
                    gate_up,
                    own_banks(shared_gate_up),
                    HIDDEN,
                    INTER,
                    tiles,
                    EXPERTS,
                    rooms,
                    fragments,
                )
                .expect("register gate-up");
                moe::register_down(&sink, down, own_banks(shared_down), INTER, HIDDEN, tiles, EXPERTS, rooms)
                    .expect("register down");
                let sorted = Tensor::new(y_t.buf, tiles * 8, HIDDEN, Dtype::Bf16);
                moe::weighted_sum_sorted(&sink, sorted, inv_t, wt, TOP_K, sft).expect("sorted fold");
                frame.commit().expect("commit");

                let read = |b: &Buffer, bytes: u64| {
                    let mut raw = vec![0u8; bytes as usize];
                    b.read(0, &mut raw).expect("read");
                    floats(&raw)
                };
                let mut inv = vec![0u8; pairs as usize * 4];
                _c.read(0, &mut inv).expect("inv");
                let inv: Vec<usize> = inv
                    .chunks(4)
                    .map(|c| usize::try_from(i32::from_le_bytes([c[0], c[1], c[2], c[3]])).expect("a row"))
                    .collect();
                let want = read(&routed, u64::from(pairs) * u64::from(HIDDEN) * 2);
                let got = read(&yb, out_bytes);
                let width = HIDDEN as usize;
                for (route, &row) in inv.iter().enumerate() {
                    let w = &want[route * width..(route + 1) * width];
                    let g = &got[row * width..(row + 1) * width];
                    let off = spread(w, g);
                    assert!(
                        off < 0.02,
                        "tokens {tokens} fragments {fragments} shared {shared}: route {route}'s \
                         down row {row} is {off:.2e} off the per-route path's"
                    );
                    checked += 1;
                }
                let fold_want = read(&folded, u64::from(tokens) * u64::from(HIDDEN) * 2);
                let fold_got = read(&sorted_fold, u64::from(tokens) * u64::from(HIDDEN) * 2);
                let off = spread(&fold_want, &fold_got);
                assert!(off < 0.02, "tokens {tokens}: the sorted fold is {off:.2e} off the per-route fold");
                if shared {
                    let own_want = read(&own, u64::from(tokens) * u64::from(HIDDEN) * 2);
                    for t in 0..tokens as usize {
                        let off = spread(&own_want[t * width..(t + 1) * width], &got[t * width..(t + 1) * width]);
                        assert!(off < 0.02, "tokens {tokens}: shared row {t} is {off:.2e} off");
                    }
                }
                eprintln!("tokens {tokens} fragments {fragments} shared {shared}: fold {off:.2e} off");
            }
        }
    }
    eprintln!("checked {checked} register routes against the per-route path");
}
