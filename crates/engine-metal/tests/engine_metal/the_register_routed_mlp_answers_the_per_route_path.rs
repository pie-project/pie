#![cfg(target_vendor = "apple")]
//! The routed MLP on the register tile (`linear/moe/select_sg.metal`: expert tiles, the
//! gate-up landing its SwiGLU as the down's table, the down into sorted rows)
//! against the per-route path (`matmul_select_quant`, `swiglu`, again, and
//! the fold), with and without the shared expert riding along as tile 0; and
//! where it rides along, the tiles' combine against the fold's gated add of
//! the shared expert, and both against f64.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::linear::{mlp, moe, qmm};
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

fn bf16_value(v: f32) -> f32 {
    f32::from_bits(v.to_bits() & 0xffff_0000)
}

/// Rounds to the nearest bf16, ties to even, as the GPU lands a bf16.
fn to_bf16(v: f64) -> f64 {
    let bits = (v as f32).to_bits();
    f64::from(f32::from_bits(
        ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) << 16,
    ))
}

/// A bank's planes on the host: row `row`'s dot with `x` in f64.
struct Host {
    k: usize,
    bits: u32,
    codes: Vec<u8>,
    scales: Vec<f32>,
    biases: Vec<f32>,
}

impl Host {
    fn dot(&self, row: usize, x: &[f64]) -> f64 {
        let groups = self.k / 64;
        (0..self.k)
            .map(|at| {
                let code = row * self.k + at;
                let q = f64::from(if self.bits == 8 {
                    self.codes[code]
                } else {
                    (self.codes[code / 2] >> ((code % 2) * 4)) & 15
                });
                let g = row * groups + at / 64;
                x[at] * (q * f64::from(self.scales[g]) + f64::from(self.biases[g]))
            })
            .sum()
    }
}

/// A route's (or the shared expert's) down row in f64 from token row `x`: the
/// gate-up and its SwiGLU landed in bf16, as both paths land them.
fn reference(
    gate_up: &Host,
    down: &Host,
    expert: usize,
    inter: usize,
    width: usize,
    x: &[f64],
) -> Vec<f64> {
    let hidden: Vec<f64> = (0..inter)
        .map(|c| {
            let g = to_bf16(gate_up.dot(expert * 2 * inter + c, x));
            let u = to_bf16(gate_up.dot(expert * 2 * inter + inter + c, x));
            to_bf16(g / (1.0 + (-g).exp()) * u)
        })
        .collect();
    (0..width)
        .map(|c| down.dot(expert * width + c, &hidden))
        .collect()
}

/// The largest difference of `got` from the f64 `want`, relative to `want`'s largest magnitude.
fn spread64(want: &[f64], got: &[f32]) -> f64 {
    let scale = want.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-9);
    want.iter()
        .zip(got)
        .fold(0f64, |m, (a, b)| m.max((a - f64::from(*b)).abs()))
        / scale
}

/// A 4-bit, 64-group affine bank of `n` rows of `k` codes, seeded by `seed`.
fn bank_of(
    device: &Context,
    handles: &Handles,
    k: u32,
    n: u32,
    seed: u64,
) -> (Bank, [Buffer; 3], Host) {
    bits_bank_of(device, handles, k, n, seed, 4)
}

/// A `bits`-bit (4 or 8), 64-group affine bank of `n` rows of `k` codes.
fn bits_bank_of(
    device: &Context,
    handles: &Handles,
    k: u32,
    n: u32,
    seed: u64,
    bits: u32,
) -> (Bank, [Buffer; 3], Host) {
    let groups = k / 64;
    let upload = |bytes: &[u8]| {
        let mut buffer = Buffer::zeroed(device, bytes.len() as u64).expect("a plane");
        buffer.write(0, bytes).expect("upload");
        buffer
    };
    let codes: Vec<u8> = (0..u64::from(k) * u64::from(n) * u64::from(bits) / 8)
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
    let values = |bytes: &[u8]| -> Vec<f32> { floats(bytes).into_iter().map(bf16_value).collect() };
    let host = Host {
        k: k as usize,
        bits,
        scales: values(&scales),
        biases: values(&biases),
        codes,
    };
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
    let bank = Bank {
        mpp_codes: None,
        codes: Tensor::new(
            bind(&planes[0]),
            n,
            k,
            if bits == 8 {
                Dtype::U8g64
            } else {
                Dtype::U4g64
            },
        ),
        scales: Tensor::new(bind(&planes[1]), n, groups, Dtype::Bf16),
        biases: Some(Tensor::new(bind(&planes[2]), n, groups, Dtype::Bf16)),
        group: 64,
        bits,
        ld: k,
    };
    (bank, planes, host)
}

/// `tokens` rows of `TOP_K` distinct experts, consecutive rows sharing half.
fn routes(tokens: u32, seed: u64) -> Vec<i32> {
    let mut out: Vec<i32> = Vec::new();
    let mut n = seed;
    for t in 0..tokens {
        let mut row: Vec<i32> = Vec::new();
        if t > 0 {
            row.extend_from_slice(
                &out[((t - 1) * TOP_K) as usize..((t - 1) * TOP_K + TOP_K / 2) as usize],
            );
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
    want.iter()
        .zip(got)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
        / scale
}

#[test]
fn the_register_routed_mlp_answers_the_per_route_path() {
    let Ok(device) = Context::bind() else {
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let tuning = engine_metal::chip::tuning("", false);
    // The routed banks as a checkpoint's experts land: an expert a row.
    let expert_rows = |bank: Bank, columns: u32| Bank {
        codes: Tensor::new(
            bank.codes.buf,
            EXPERTS,
            columns * bank.codes.width,
            bank.codes.dtype,
        ),
        ld: columns * bank.codes.width,
        ..bank
    };
    let (gate_up, _g, gate_up_host) = bank_of(&device, &handles, HIDDEN, EXPERTS * 2 * INTER, 0x1);
    let (down, _d, down_host) = bank_of(&device, &handles, INTER, EXPERTS * HIDDEN, 0x2);
    let (gate_up, down) = (expert_rows(gate_up, 2 * INTER), expert_rows(down, HIDDEN));
    let (shared_gate_up, _sg, shared_gate_up_host) =
        bank_of(&device, &handles, HIDDEN, 2 * INTER, 0x3);
    let (shared_down, _sd, shared_down_host) = bank_of(&device, &handles, INTER, HIDDEN, 0x4);
    let (gate, _gb, gate_host) = bits_bank_of(&device, &handles, HIDDEN, 1, 0x5, 8);
    // Each path's largest spread from the f64 reference.
    let (mut route_off, mut tile_off) = (0f64, 0f64);
    // And of the gated sums, the per-route path's and the tiles' combine.
    let (mut sum_off, mut combine_off) = (0f64, 0f64);
    let plane = |bytes: u64| Buffer::zeroed(&device, bytes.max(4)).expect("a plane");
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
    let upload = |bytes: &[u8]| {
        let mut buffer = plane(bytes.len() as u64);
        buffer.write(0, bytes).expect("upload");
        buffer
    };
    let mut checked = 0;
    for tokens in [2u32, 5, 8] {
        for cut in [
            moe::REGISTER_CUT,
            moe::RegisterCut {
                fragments: 2,
                glu_simdgroups: 8,
                down_simdgroups: 16,
                slots: 3,
            },
            moe::RegisterCut {
                fragments: 1,
                glu_simdgroups: 16,
                down_simdgroups: 8,
                slots: 16,
            },
        ] {
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
                let gates = plane(u64::from(tokens) * 2);
                let gated = plane(u64::from(tokens) * u64::from(HIDDEN) * 2);
                let combined = plane(u64::from(tokens) * u64::from(HIDDEN) * 2);
                let (gt, gdt, ct) = (
                    Tensor::new(bind(&gates), tokens, 1, Dtype::Bf16),
                    Tensor::new(bind(&gated), tokens, HIDDEN, Dtype::Bf16),
                    Tensor::new(bind(&combined), tokens, HIDDEN, Dtype::Bf16),
                );
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
                let (_a, tiles_t) =
                    room(u64::from(tiles) + 1, Dtype::I32, (u64::from(tiles) + 1) * 4);
                let (_b, rows_t) = room(u64::from(tiles) * 8, Dtype::I32, u64::from(tiles) * 32);
                let (_c, inv_t) = room(u64::from(pairs), Dtype::I32, u64::from(pairs) * 4);
                let in_width = u64::from(tiles) * moe::register_table_width(HIDDEN);
                let (_e, table_t) = room(in_width, Dtype::Bf16, in_width * 2);
                let mid_width = u64::from(tiles) * moe::register_table_width(INTER);
                let (_f, hidden_t) = room(mid_width, Dtype::Bf16, mid_width * 2);
                let out_bytes = u64::from(tiles) * 8 * u64::from(HIDDEN) * 2;
                let (yb, y_t) = room(
                    u64::from(tiles) * 8 * u64::from(HIDDEN),
                    Dtype::Bf16,
                    out_bytes,
                );
                let (_m, mix_t) = room(
                    u64::from(moe::REGISTER_MIX),
                    Dtype::F32,
                    u64::from(moe::REGISTER_MIX) * 4,
                );
                let (_n, grids_t) = room(
                    u64::from(moe::REGISTER_GRIDS),
                    Dtype::I32,
                    u64::from(moe::REGISTER_GRIDS) * 4,
                );
                let rooms = moe::RegisterRooms {
                    tiles: tiles_t,
                    rows: rows_t,
                    inv: inv_t,
                    table: table_t,
                    hidden: hidden_t,
                    y: Tensor::new(y_t.buf, tiles * 8, HIDDEN, Dtype::Bf16),
                    mix: mix_t,
                    grids: grids_t,
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
                qmm::matmul(
                    &sink,
                    &tuning,
                    xt,
                    gate,
                    gt,
                    qmm::Scratch {
                        precast: &|_, _| None,
                        partials: &|_, _| None,
                        table: None,
                    },
                    tokens,
                )
                .expect("gate");
                moe::sigmoid_gate_add(&sink, ft, ot, gt, gdt).expect("gated add");

                let own_banks = |bank| shared.then_some(bank);
                let landed = shared.then_some(moe::RegisterGate { weights: wt, gate });
                let groups = cut.groups(INTER, HIDDEN);
                moe::register_group(&sink, xt, rt, EXPERTS, shared, landed, groups, rooms)
                    .expect("group");
                moe::register_glu(
                    &sink,
                    gate_up,
                    own_banks(shared_gate_up),
                    HIDDEN,
                    INTER,
                    tiles,
                    EXPERTS,
                    rooms,
                    cut,
                )
                .expect("register gate-up");
                moe::register_down(
                    &sink,
                    down,
                    own_banks(shared_down),
                    INTER,
                    HIDDEN,
                    tiles,
                    EXPERTS,
                    rooms,
                    cut,
                )
                .expect("register down");
                let sorted = Tensor::new(y_t.buf, tiles * 8, HIDDEN, Dtype::Bf16);
                moe::weighted_sum_sorted(&sink, sorted, inv_t, wt, TOP_K, sft)
                    .expect("sorted fold");
                if shared {
                    moe::register_combine(&sink, rooms, tiles, TOP_K, ct).expect("combine");
                }
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
                    .map(|c| {
                        usize::try_from(i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .expect("a row")
                    })
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
                        "tokens {tokens} cut {cut:?} shared {shared}: route {route}'s \
                         down row {row} is {off:.2e} off the per-route path's"
                    );
                    checked += 1;
                }
                // Against f64, for a few routes and the shared rows: the register path is
                // no further off than the per-route path.
                let token_row = |t: usize| -> Vec<f64> {
                    floats(&x[t * width * 2..(t + 1) * width * 2])
                        .into_iter()
                        .map(f64::from)
                        .collect()
                };
                for (route, &row) in inv.iter().enumerate().step_by(9) {
                    let x = token_row(route / TOP_K as usize);
                    let exact = reference(
                        &gate_up_host,
                        &down_host,
                        r[route] as usize,
                        INTER as usize,
                        width,
                        &x,
                    );
                    route_off =
                        route_off.max(spread64(&exact, &want[route * width..(route + 1) * width]));
                    tile_off = tile_off.max(spread64(&exact, &got[row * width..(row + 1) * width]));
                }
                if shared {
                    for t in [0usize, tokens as usize - 1] {
                        let exact = reference(
                            &shared_gate_up_host,
                            &shared_down_host,
                            0,
                            INTER as usize,
                            width,
                            &token_row(t),
                        );
                        tile_off = tile_off.max(spread64(&exact, &got[t * width..(t + 1) * width]));
                    }
                }
                let fold_want = read(&folded, u64::from(tokens) * u64::from(HIDDEN) * 2);
                let fold_got = read(&sorted_fold, u64::from(tokens) * u64::from(HIDDEN) * 2);
                let off = spread(&fold_want, &fold_got);
                assert!(
                    off < 0.02,
                    "tokens {tokens}: the sorted fold is {off:.2e} off the per-route fold"
                );
                if shared {
                    let own_want = read(&own, u64::from(tokens) * u64::from(HIDDEN) * 2);
                    for t in 0..tokens as usize {
                        let off = spread(
                            &own_want[t * width..(t + 1) * width],
                            &got[t * width..(t + 1) * width],
                        );
                        assert!(
                            off < 0.02,
                            "tokens {tokens}: shared row {t} is {off:.2e} off"
                        );
                    }
                }
                if shared {
                    let rows = u64::from(tokens) * u64::from(HIDDEN) * 2;
                    let (want, got) = (read(&gated, rows), read(&combined, rows));
                    let off = spread(&want, &got);
                    assert!(
                        off < 0.02,
                        "tokens {tokens} cut {cut:?}: the combine is {off:.2e} off the gated add"
                    );
                    // Each token's gated sum in f64: its routes' rows weighed, and the
                    // shared expert's by the sigmoid of its gate (landed in bf16).
                    let mut raw = vec![0u8; pairs as usize * 4];
                    wb.read(0, &mut raw).expect("weights");
                    let w: Vec<f64> = raw
                        .chunks(4)
                        .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                        .collect();
                    for t in [0usize, tokens as usize - 1] {
                        let x = token_row(t);
                        let g = to_bf16(gate_host.dot(0, &x));
                        let g = 1.0 / (1.0 + (-g).exp());
                        let own = reference(
                            &shared_gate_up_host,
                            &shared_down_host,
                            0,
                            INTER as usize,
                            width,
                            &x,
                        );
                        let mut exact: Vec<f64> = own.iter().map(|v| g * v).collect();
                        for k in 0..TOP_K as usize {
                            let route = t * TOP_K as usize + k;
                            let row = reference(
                                &gate_up_host,
                                &down_host,
                                r[route] as usize,
                                INTER as usize,
                                width,
                                &x,
                            );
                            for (e, v) in exact.iter_mut().zip(row) {
                                *e += w[route] * v;
                            }
                        }
                        sum_off = sum_off.max(spread64(&exact, &want[t * width..(t + 1) * width]));
                        combine_off =
                            combine_off.max(spread64(&exact, &got[t * width..(t + 1) * width]));
                    }
                }
                eprintln!("tokens {tokens} cut {cut:?} shared {shared}: fold {off:.2e} off");
            }
        }
    }
    eprintln!(
        "checked {checked} register routes against the per-route path; against f64 the per-route \
         path is {route_off:.2e} off and the register tiles {tile_off:.2e}"
    );
    eprintln!(
        "against f64 the gated sums are {sum_off:.2e} off on the per-route path and \
         {combine_off:.2e} from the tiles' combine"
    );
    assert!(
        tile_off <= 1.5 * route_off.max(2e-3),
        "the register tiles are {tile_off:.2e} off f64, the per-route path {route_off:.2e}"
    );
    assert!(
        combine_off <= sum_off.max(2e-3),
        "the tiles' combine is {combine_off:.2e} off f64, the per-route gated add {sum_off:.2e}"
    );
}
