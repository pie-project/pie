#![cfg(target_vendor = "apple")]
//! Microbench (not a test): one Qwen3.6-35B-A3B routed projection
//! (256 experts, top-8, u4g64 affine, bf16 scales/biases) at 1 and 9 rows,
//! on the per-route qmv path (what a verify of < 64 rows takes today) and on
//! the sorted/batched path (forced via `moe.tile_rows`), plus the qmv at a
//! single row routed to U distinct experts (the dedup floor).
//! Run: cargo test --release -p engine-metal --test engine_metal \
//!        the_routed_mlp_prices_its_rows -- --ignored --nocapture

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::linear::moe::{self, RoutedScratch};
use kernels_metal::{Bank, Tensor};
use poem_ir::Dtype;

const EXPERTS: u32 = 256;
const TOP_K: u32 = 8;
const ITERS: usize = 30;

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

/// `tokens` rows of `TOP_K` distinct experts; `share` of each row's experts
/// repeat the previous row's (consecutive verify tokens route alike).
fn routes(tokens: u32, share: u32, seed: u64) -> Vec<i32> {
    let mut out: Vec<i32> = Vec::new();
    let mut n = seed;
    for t in 0..tokens {
        let mut row: Vec<i32> = Vec::new();
        if t > 0 {
            let prev = &out[((t - 1) * TOP_K) as usize..(t * TOP_K) as usize];
            row.extend_from_slice(&prev[..share.min(TOP_K) as usize]);
        }
        while row.len() < TOP_K as usize {
            n += 1;
            let e = (u32::from(noise(n)) | (u32::from(noise(n ^ 0xfeed)) << 8)) % EXPERTS;
            if !row.contains(&(e as i32)) {
                row.push(e as i32);
            }
        }
        out.extend(row);
    }
    out
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
#[ignore = "a microbenchmark"]
fn the_routed_mlp_prices_its_rows() {
    let Ok(device) = Context::bind() else { return };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    // Per-route path: the shipped default (per_expert < 2 -> 1).
    // Sorted path: force a tile for any pair count. On an M4 Pro (Apple9) the
    // routed path is the steel qmm ("tiled"); on M5 (Apple10) add moe.path = "mpp".
    let sorted16 = engine_metal::chip::tuning("[kernel]\nmoe.tile_rows = [{ value = 16 }]\n", false);
    let sorted8 = engine_metal::chip::tuning("[kernel]\nmoe.tile_rows = [{ value = 8 }]\n", false);
    let upload = |bytes: &[u8]| {
        let mut b = Buffer::zeroed(&device, bytes.len() as u64).expect("plane");
        b.write(0, bytes).expect("upload");
        b
    };
    // (name, K, N): gate_up (hidden 2048 -> 2*512) and down (512 -> 2048).
    for (proj, k, n) in [("gate_up", 2048u32, 1024u32), ("down", 512, 2048)] {
        let groups = k / 64;
        let columns = EXPERTS * n;
        let codes: Vec<u8> = (0..u64::from(k) * u64::from(columns) / 2).map(noise).collect();
        let scales: Vec<u8> = (0..columns * groups).flat_map(|a| bf16(0.003 + (a % 7) as f32 * 1e-4)).collect();
        let biases: Vec<u8> = (0..columns * groups).flat_map(|a| bf16(((a % 9) as f32 - 4.0) / 64.0)).collect();
        let (codes, scales, biases) = (upload(&codes), upload(&scales), upload(&biases));
        let t = |b: &Buffer, rows: u32, width: u32, dtype: Dtype| {
            Tensor::new(handles.bind(b, 0, b.bytes()).expect("bind"), rows, width, dtype)
        };
        let bank = Bank {
            mpp_codes: None,
            codes: t(&codes, columns, k, Dtype::U4g64),
            scales: t(&scales, columns, groups, Dtype::Bf16),
            biases: Some(t(&biases, columns, groups, Dtype::Bf16)),
            group: 64,
            bits: 4,
            ld: k,
        };
        // (label, tokens, fan, share)
        for (tokens, share) in [(1u32, 0u32), (9, 0), (9, 4)] {
            let fan = TOP_K;
            let r = routes(tokens, share, 7);
            let unique = { let mut u = r.clone(); u.sort(); u.dedup(); u.len() as u32 };
            let pairs = tokens * fan;
            let x: Vec<u8> = (0..tokens * k).flat_map(|a| bf16((f32::from(noise(u64::from(a))) - 127.0) / 256.0)).collect();
            let rbytes: Vec<u8> = r.iter().flat_map(|e| e.to_le_bytes()).collect();
            let (xb, rb) = (upload(&x), upload(&rbytes));
            let yb = Buffer::zeroed(&device, u64::from(pairs) * u64::from(n) * 2).expect("y");
            let (xt, rt, yt) = (t(&xb, tokens, k, Dtype::Bf16), t(&rb, tokens, fan, Dtype::I32), t(&yb, pairs, n, Dtype::Bf16));

            let time = |f: &dyn Fn(&Sink) -> bool| -> Option<f64> {
                let mut v = Vec::new();
                for i in 0..(ITERS + 40) {
                    let frame = device.frame().expect("frame");
                    if !f(&Sink::new(&device, &frame, &pipelines, &handles)) {
                        return None;
                    }
                    let s = frame.commit_timed().expect("commit");
                    if i >= 40 { v.push(s * 1e6); }
                }
                Some(median(v))
            };
            let qmv = time(&|s| { moe::matmul_select_quant_as(s, xt, bank, rt, yt, 0).expect("qmv"); true });
            let mut line = format!("{proj:8} rows={tokens} unique={unique:2}  per-route qmv {:7.1} us", qmv.unwrap());
            // the expert-major kernel, against the per-route one's output
            let read = |b: &Buffer| -> Vec<f32> {
                let mut raw = vec![0u8; (pairs * n * 2) as usize];
                b.read(0, &mut raw).expect("read");
                raw.chunks(2).map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)).collect()
            };
            let want = read(&yb);
            for group in [8u32, 16] {
                let us = time(&|s| { moe::matmul_select_quant_as(s, xt, bank, rt, yt, group).expect("em"); true });
                let got = read(&yb);
                let err = want.iter().zip(&got).map(|(a, b)| (a - b).abs() / a.abs().max(1e-2)).fold(0f32, f32::max);
                line += &format!("  em{group} {:7.1} us (rel err {err:.1e})", us.unwrap());
            }
            for (label, tuning) in [("sorted16", &sorted16), ("sorted8", &sorted8)] {
                let sorted = moe::sorted_rows(pairs, EXPERTS, tuning);
                let z = |rows: u32, width: u32, dt: Dtype, bytes: u64| {
                    let b = Buffer::zeroed(&device, bytes.max(4)).expect("scratch");
                    let tt = t(&b, rows, width, dt);
                    (b, tt)
                };
                let (_a, perm) = z(1, sorted, Dtype::I32, u64::from(sorted) * 4);
                let (_b, rowe) = z(1, sorted, Dtype::I32, u64::from(sorted) * 4);
                let (_c, tile) = z(1, sorted, Dtype::I32, u64::from(sorted) * 4);
                let (_d, inv) = z(1, pairs, Dtype::I32, u64::from(pairs) * 4);
                let (_e, sx) = z(sorted, k, Dtype::Bf16, u64::from(sorted) * u64::from(k) * 2);
                let (_f, sy) = z(sorted, n, Dtype::Bf16, u64::from(sorted) * u64::from(n) * 2);
                let (_g, sums) = z(sorted, groups, Dtype::F32, u64::from(sorted) * u64::from(groups) * 4);
                let scratch = RoutedScratch { perm, row_expert: rowe, tile_expert: tile, inv, x: sx, y: sy, sums };
                let us = time(&|s| {
                    moe::matmul_select_batched(s, "bench", xt, bank, None, rt, EXPERTS, scratch, yt, tuning, moe::Sorted::default())
                        .expect("batched")
                });
                line += &match us { Some(us) => format!("  {label} {us:7.1} us (stack {sorted})"), None => format!("  {label} n/a") };
            }
            // Dedup floor: one row routed to the `unique` experts it touches.
            let u: Vec<i32> = { let mut u = r.clone(); u.sort(); u.dedup(); u };
            let ub = upload(&u.iter().flat_map(|e| e.to_le_bytes()).collect::<Vec<u8>>());
            let uy = Buffer::zeroed(&device, u64::from(unique) * u64::from(n) * 2).expect("uy");
            let (ut, uyt) = (t(&ub, 1, unique, Dtype::I32), t(&uy, unique, n, Dtype::Bf16));
            let x1 = t(&xb, 1, k, Dtype::Bf16);
            let floor = time(&|s| { moe::matmul_select_quant(s, x1, bank, ut, uyt).expect("floor"); true });
            line += &format!("  dedup-floor(1 row x {unique}) {:7.1} us", floor.unwrap());
            eprintln!("{line}");
        }
    }
}

/// A Qwen3.6-35B-A3B routed MLP (gate-up, SwiGLU, down, fold, and its shared
/// expert) over a verify's rows: today's per-route path, its shared expert on the
/// dense register tile, against the register expert tiles, the shared expert
/// riding along as a tile or not. `LAYERS` layers' banks in turn, so no layer's
/// codes sit in the cache from the last.
/// Run: cargo test --release -p engine-metal --test engine_metal \
///        the_register_routed_mlp_prices -- --ignored --nocapture
#[test]
#[ignore = "a microbenchmark"]
fn the_register_routed_mlp_prices_its_tiles() {
    use kernels_metal::linear::{mlp, quant};
    const LAYERS: usize = 3;
    const PASSES: usize = 12;
    const HIDDEN: u32 = 2048;
    const INTER: u32 = 512;
    let Ok(device) = Context::bind() else { return };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let register = engine_metal::chip::tuning("[kernel]\nqmm = [\"register\", \"qmv\"]\n", false);
    let upload = |bytes: &[u8]| {
        let mut b = Buffer::zeroed(&device, bytes.len().max(4) as u64).expect("plane");
        b.write(0, bytes).expect("upload");
        b
    };
    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
    let mut held: Vec<Buffer> = Vec::new();
    let mut bank = |k: u32, n: u32, seed: u64| {
        let groups = k / 64;
        let codes: Vec<u8> = (0..u64::from(k) * u64::from(n) / 2).map(|a| noise(a ^ seed)).collect();
        let scales: Vec<u8> = (0..n * groups).flat_map(|a| bf16(0.003 + (a % 7) as f32 * 1e-4)).collect();
        let biases: Vec<u8> = (0..n * groups).flat_map(|a| bf16(((a % 9) as f32 - 4.0) / 64.0)).collect();
        let (c, s, b) = (upload(&codes), upload(&scales), upload(&biases));
        let bank = Bank {
            mpp_codes: None,
            codes: Tensor::new(bind(&c), n, k, Dtype::U4g64),
            scales: Tensor::new(bind(&s), n, groups, Dtype::Bf16),
            biases: Some(Tensor::new(bind(&b), n, groups, Dtype::Bf16)),
            group: 64,
            bits: 4,
            ld: k,
        };
        held.extend([c, s, b]);
        bank
    };
    // (gate-up, down, shared gate-up, shared down) a layer.
    let layers: Vec<[Bank; 4]> = (0..LAYERS as u64)
        .map(|l| {
            [
                bank(HIDDEN, EXPERTS * 2 * INTER, 4 * l + 1),
                bank(INTER, EXPERTS * HIDDEN, 4 * l + 2),
                bank(HIDDEN, 2 * INTER, 4 * l + 3),
                bank(INTER, HIDDEN, 4 * l + 4),
            ]
        })
        .collect();
    let zeroed = |bytes: u64| Buffer::zeroed(&device, bytes.max(4)).expect("scratch");
    for (tokens, share) in [(8u32, 0u32), (8, 2), (8, 4), (8, 6), (6, 4), (1, 0)] {
        let pairs = tokens * TOP_K;
        let r = routes(tokens, share, 11);
        let unique = { let mut u = r.clone(); u.sort(); u.dedup(); u.len() };
        let x: Vec<u8> = (0..tokens * HIDDEN).flat_map(|a| bf16((f32::from(noise(u64::from(a))) - 127.0) / 256.0)).collect();
        let w: Vec<u8> = (0..pairs).flat_map(|a| (1.0f32 / TOP_K as f32 + a as f32 * 1e-4).to_le_bytes()).collect();
        let (xb, rb, wb) = (upload(&x), upload(&r.iter().flat_map(|e| e.to_le_bytes()).collect::<Vec<u8>>()), upload(&w));
        let xt = Tensor::new(bind(&xb), tokens, HIDDEN, Dtype::Bf16);
        let rt = Tensor::new(bind(&rb), tokens, TOP_K, Dtype::I32);
        let wt = Tensor::new(bind(&wb), tokens, TOP_K, Dtype::F32);
        let plane = |rows: u32, width: u32, dtype: Dtype| {
            let b = zeroed(u64::from(rows) * u64::from(width) * dtype.bytes_ceil());
            let t = Tensor::new(bind(&b), rows, width, dtype);
            (b, t)
        };
        let (_p, pt) = plane(pairs, 2 * INTER, Dtype::Bf16);
        let (_h, ht) = plane(pairs, INTER, Dtype::Bf16);
        let (_y, yt) = plane(pairs, HIDDEN, Dtype::Bf16);
        let (_f, ft) = plane(tokens, HIDDEN, Dtype::Bf16);
        let (_sp, spt) = plane(tokens, 2 * INTER, Dtype::Bf16);
        let (_sh, sht) = plane(tokens, INTER, Dtype::Bf16);
        let (_sy, syt) = plane(tokens, HIDDEN, Dtype::Bf16);
        let (_t, table) = plane(64, HIDDEN, Dtype::F16);
        let (_q, partial) = plane(8 * 8, HIDDEN, Dtype::F32);
        let precast = |rows: u32, width: u32| (rows <= 64 && width <= HIDDEN).then(|| Tensor::new(table.buf, rows, width, Dtype::F16));
        let partials = |rows: u32, width: u32| (rows <= 64 && width <= HIDDEN).then(|| Tensor::new(partial.buf, rows, width, Dtype::F32));
        let scratch = || quant::Scratch { precast: &precast, partials: &partials, table: None };
        let tiles = moe::register_tiles(tokens, TOP_K, EXPERTS, true);
        let (_a, tiles_t) = plane(1, tiles + 1, Dtype::I32);
        let (_b, rows_t) = plane(1, tiles * 8, Dtype::I32);
        let (_c, inv_t) = plane(1, pairs, Dtype::I32);
        let (_d, table_t) = plane(tiles, u32::try_from(moe::register_table_width(HIDDEN)).unwrap(), Dtype::Bf16);
        let (_e, hidden_t) = plane(tiles, u32::try_from(moe::register_table_width(INTER)).unwrap(), Dtype::Bf16);
        let (_g, y_t) = plane(tiles * 8, HIDDEN, Dtype::Bf16);
        let rooms = moe::RegisterRooms { tiles: tiles_t, rows: rows_t, inv: inv_t, table: table_t, hidden: hidden_t, y: y_t };

        let time = |f: &dyn Fn(&Sink, &[Bank; 4])| -> f64 {
            let mut v = Vec::new();
            for i in 0..(ITERS + 5) {
                let frame = device.frame().expect("frame");
                let sink = Sink::new(&device, &frame, &pipelines, &handles);
                for pass in 0..PASSES {
                    f(&sink, &layers[pass % LAYERS]);
                }
                let s = frame.commit_timed().expect("commit");
                if i >= 5 {
                    v.push(s * 1e6 / PASSES as f64);
                }
            }
            median(v)
        };
        let route_gate_up = |s: &Sink, l: &[Bank; 4]| {
            moe::matmul_select_quant(s, xt, l[0], rt, pt).expect("gate-up");
            mlp::swiglu(s, pt, INTER, ht).expect("swiglu");
        };
        let route_down = |s: &Sink, l: &[Bank; 4]| {
            moe::matmul_select_quant(s, ht, l[1], rt, yt).expect("down");
            moe::weighted_sum(s, yt, wt, ft).expect("fold");
        };
        let own = |s: &Sink, l: &[Bank; 4]| {
            quant::matmul(s, &register, xt, l[2], spt, scratch(), 8).expect("shared gate-up");
            mlp::swiglu(s, spt, INTER, sht).expect("shared swiglu");
            quant::matmul(s, &register, sht, l[3], syt, scratch(), 8).expect("shared down");
        };
        let tiled = |s: &Sink, l: &[Bank; 4], shared: bool, fragments: u32| {
            let tiles = moe::register_tiles(tokens, TOP_K, EXPERTS, shared);
            let own = |b: Bank| shared.then_some(b);
            moe::register_group(s, xt, rt, EXPERTS, shared, rooms).expect("group");
            moe::register_glu(s, l[0], own(l[2]), HIDDEN, INTER, tiles, EXPERTS, rooms, fragments).expect("glu");
            moe::register_down(s, l[1], own(l[3]), INTER, HIDDEN, tiles, EXPERTS, rooms).expect("down");
            let sorted = Tensor::new(rooms.y.buf, tiles * 8, HIDDEN, Dtype::Bf16);
            moe::weighted_sum_sorted(s, sorted, rooms.inv, wt, TOP_K, ft).expect("fold");
        };
        let per_route = time(&|s, l| { route_gate_up(s, l); route_down(s, l); });
        let per_route_gate_up = time(&|s, l| route_gate_up(s, l));
        let shared_dense = time(&|s, l| own(s, l));
        let routed_h1 = time(&|s, l| tiled(s, l, false, 1));
        let routed_h2 = time(&|s, l| tiled(s, l, false, 2));
        let folded_h1 = time(&|s, l| tiled(s, l, true, 1));
        let folded_h2 = time(&|s, l| tiled(s, l, true, 2));
        let glu_h1 = time(&|s, l| {
            moe::register_group(s, xt, rt, EXPERTS, true, rooms).expect("group");
            moe::register_glu(s, l[0], Some(l[2]), HIDDEN, INTER, tiles, EXPERTS, rooms, 1).expect("glu");
        });
        let group_only = time(&|s, _l| {
            moe::register_group(s, xt, rt, EXPERTS, true, rooms).expect("group");
        });
        let down_only = time(&|s, l| {
            moe::register_down(s, l[1], Some(l[3]), INTER, HIDDEN, tiles, EXPERTS, rooms).expect("down");
        });
        // The same passes launched over the live tiles alone, as an indirect grid would.
        let live = u32::try_from(unique).unwrap() + 1;
        let glu_live = time(&|s, l| {
            moe::register_glu(s, l[0], Some(l[2]), HIDDEN, INTER, live, EXPERTS, rooms, 1).expect("glu");
        });
        let glu_max = time(&|s, l| {
            moe::register_glu(s, l[0], Some(l[2]), HIDDEN, INTER, tiles, EXPERTS, rooms, 1).expect("glu");
        });
        let down_live = time(&|s, l| {
            moe::register_down(s, l[1], Some(l[3]), INTER, HIDDEN, live, EXPERTS, rooms).expect("down");
        });
        let fold_only = time(&|s, _l| {
            let sorted = Tensor::new(rooms.y.buf, tiles * 8, HIDDEN, Dtype::Bf16);
            moe::weighted_sum_sorted(s, sorted, rooms.inv, wt, TOP_K, ft).expect("fold");
        });
        eprintln!(
            "rows={tokens} unique={unique:2}: glu {glu_max:6.1} us ({glu_live:6.1} over the live tiles), \
             down {down_only:6.1} us ({down_live:6.1} live), fold {fold_only:6.1} us"
        );
        let route_down_only = time(&|s, l| {
            moe::matmul_select_quant(s, ht, l[1], rt, yt).expect("down");
        });
        eprintln!(
            "rows={tokens} unique={unique:2}: group+gather {group_only:6.1} us, register down {down_only:6.1} us, \
             per-route down {route_down_only:6.1} us"
        );
        eprintln!(
            "rows={tokens} unique={unique:2}: per-route {per_route:6.1} us (gate-up+swiglu {per_route_gate_up:6.1}) \
             + dense shared {shared_dense:6.1} = {:6.1} | register h1 {routed_h1:6.1} h2 {routed_h2:6.1} | \
             with shared tile h1 {folded_h1:6.1} h2 {folded_h2:6.1} (group+glu h1 {glu_h1:6.1}) us a layer",
            per_route + shared_dense
        );
    }
}
