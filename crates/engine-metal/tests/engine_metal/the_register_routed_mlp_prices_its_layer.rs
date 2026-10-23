#![cfg(target_vendor = "apple")]
//! Microbench (not a test): a Qwen3.6-35B-A3B layer's routed MLP and its gated
//! shared expert (256 experts, top-8, 2048 -> 2 x 512 -> 2048, u4g64 affine,
//! an 8-bit gate) over a verify's eight rows: the per-route path as a verify
//! runs it (the routed gate-up, SwiGLU, down and fold; the shared expert on
//! the register tile; its gate; the gated add) against the register expert
//! tiles (the grouping with the gate, the gate-up and the down on the live
//! tiles' grids, the combine), each pass alone and the whole. Three layers' banks
//! in turn, so no layer's codes sit in the cache from the last.
//! Run: cargo test --release -p engine-metal --test engine_metal \
//!        the_register_routed_mlp_prices_its_layer -- --ignored --nocapture

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::linear::{mlp, moe, qmm};
use kernels_metal::{Bank, Tensor};
use poem_ir::Dtype;

const EXPERTS: u32 = 256;
const TOP_K: u32 = 8;
const HIDDEN: u32 = 2048;
const INTER: u32 = 512;
const LAYERS: usize = 3;
const PASSES: usize = 12;
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
fn the_register_routed_mlp_prices_its_layer() {
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
    let mut bank = |k: u32, n: u32, seed: u64, bits: u32, expert: u32| {
        let groups = k / 64;
        let codes: Vec<u8> = (0..u64::from(k) * u64::from(n) * u64::from(bits) / 8)
            .map(|a| noise(a ^ seed))
            .collect();
        let scales: Vec<u8> = (0..n * groups)
            .flat_map(|a| bf16(0.003 + (a % 7) as f32 * 1e-4))
            .collect();
        let biases: Vec<u8> = (0..n * groups)
            .flat_map(|a| bf16(((a % 9) as f32 - 4.0) / 64.0))
            .collect();
        let (c, s, b) = (upload(&codes), upload(&scales), upload(&biases));
        // A routed bank lands an expert a row, as a checkpoint's experts do.
        let rows = n / expert;
        let bank = Bank {
            mpp_codes: None,
            codes: Tensor::new(
                bind(&c),
                rows,
                expert * k,
                if bits == 8 {
                    Dtype::U8g64
                } else {
                    Dtype::U4g64
                },
            ),
            scales: Tensor::new(bind(&s), n, groups, Dtype::Bf16),
            biases: Some(Tensor::new(bind(&b), n, groups, Dtype::Bf16)),
            group: 64,
            bits,
            ld: expert * k,
        };
        held.extend([c, s, b]);
        bank
    };
    // (gate-up, down, shared gate-up, shared down, gate) a layer.
    let layers: Vec<[Bank; 5]> = (0..LAYERS as u64)
        .map(|l| {
            [
                bank(HIDDEN, EXPERTS * 2 * INTER, 5 * l + 1, 4, 2 * INTER),
                bank(INTER, EXPERTS * HIDDEN, 5 * l + 2, 4, HIDDEN),
                bank(HIDDEN, 2 * INTER, 5 * l + 3, 4, 1),
                bank(INTER, HIDDEN, 5 * l + 4, 4, 1),
                bank(HIDDEN, 1, 5 * l + 5, 8, 1),
            ]
        })
        .collect();
    let zeroed = |bytes: u64| Buffer::zeroed(&device, bytes.max(4)).expect("scratch");
    for (tokens, share) in [(8u32, 0u32), (8, 3), (8, 4), (8, 5), (8, 6)] {
        let pairs = tokens * TOP_K;
        let r = routes(tokens, share, 11);
        let unique = {
            let mut u = r.clone();
            u.sort_unstable();
            u.dedup();
            u.len()
        };
        let x: Vec<u8> = (0..tokens * HIDDEN)
            .flat_map(|a| bf16((f32::from(noise(u64::from(a))) - 127.0) / 256.0))
            .collect();
        let w: Vec<u8> = (0..pairs)
            .flat_map(|a| (1.0f32 / TOP_K as f32 + a as f32 * 1e-4).to_le_bytes())
            .collect();
        let (xb, rb, wb) = (
            upload(&x),
            upload(&r.iter().flat_map(|e| e.to_le_bytes()).collect::<Vec<u8>>()),
            upload(&w),
        );
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
        let (_g, gt) = plane(tokens, 1, Dtype::Bf16);
        let (_o, ot) = plane(tokens, HIDDEN, Dtype::Bf16);
        let (_t, table) = plane(64, HIDDEN, Dtype::F16);
        let (_q, partial) = plane(8 * 8, HIDDEN, Dtype::F32);
        let precast = |rows: u32, width: u32| {
            (rows <= 64 && width <= HIDDEN).then(|| Tensor::new(table.buf, rows, width, Dtype::F16))
        };
        let partials = |rows: u32, width: u32| {
            (rows <= 64 && width <= HIDDEN)
                .then(|| Tensor::new(partial.buf, rows, width, Dtype::F32))
        };
        let scratch = || qmm::Scratch {
            precast: &precast,
            partials: &partials,
            table: None,
        };
        let tiles = moe::register_tiles(tokens, TOP_K, EXPERTS, true);
        let (_a, tiles_t) = plane(1, tiles + 1, Dtype::I32);
        let (_b, rows_t) = plane(1, tiles * 8, Dtype::I32);
        let (_c, inv_t) = plane(1, pairs, Dtype::I32);
        let wide = |w| u32::try_from(moe::register_table_width(w)).expect("a width");
        let (_d, table_t) = plane(tiles, wide(HIDDEN), Dtype::Bf16);
        let (_e, hidden_t) = plane(tiles, wide(INTER), Dtype::Bf16);
        let (_r, y_t) = plane(tiles * 8, HIDDEN, Dtype::Bf16);
        let (_m, mix_t) = plane(1, moe::REGISTER_MIX, Dtype::F32);
        let (_n, grids_t) = plane(1, moe::REGISTER_GRIDS, Dtype::I32);
        let rooms = moe::RegisterRooms {
            tiles: tiles_t,
            rows: rows_t,
            inv: inv_t,
            table: table_t,
            hidden: hidden_t,
            y: y_t,
            mix: mix_t,
            grids: grids_t,
        };

        let time = |f: &dyn Fn(&Sink, &[Bank; 5])| -> f64 {
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
        let routed = |s: &Sink, l: &[Bank; 5]| {
            moe::matmul_select_quant(s, xt, l[0], rt, pt).expect("gate-up");
            mlp::swiglu(s, pt, INTER, ht).expect("swiglu");
            moe::matmul_select_quant(s, ht, l[1], rt, yt).expect("down");
            moe::weighted_sum(s, yt, wt, ft).expect("fold");
        };
        let own = |s: &Sink, l: &[Bank; 5]| {
            qmm::matmul(s, &register, xt, l[2], spt, scratch(), 8).expect("shared gate-up");
            mlp::swiglu(s, spt, INTER, sht).expect("shared swiglu");
            qmm::matmul(s, &register, sht, l[3], syt, scratch(), 8).expect("shared down");
        };
        let gated = |s: &Sink, l: &[Bank; 5]| {
            qmm::matmul(s, &register, xt, l[4], gt, scratch(), 8).expect("gate");
            moe::sigmoid_gate_add(s, ft, syt, gt, ot).expect("gated add");
        };
        let cut = moe::REGISTER_CUT;
        let groups = cut.groups(INTER, HIDDEN);
        let group = |s: &Sink, l: &[Bank; 5]| {
            let gate = moe::RegisterGate {
                weights: wt,
                gate: l[4],
            };
            moe::register_group(s, xt, rt, EXPERTS, true, Some(gate), groups, rooms)
                .expect("group");
        };
        let glu = |s: &Sink, l: &[Bank; 5]| {
            moe::register_glu(
                s,
                l[0],
                Some(l[2]),
                HIDDEN,
                INTER,
                tiles,
                EXPERTS,
                rooms,
                cut,
            )
            .expect("glu");
        };
        let down = |s: &Sink, l: &[Bank; 5]| {
            moe::register_down(
                s,
                l[1],
                Some(l[3]),
                INTER,
                HIDDEN,
                tiles,
                EXPERTS,
                rooms,
                cut,
            )
            .expect("down");
        };
        let combine = |s: &Sink| {
            moe::register_combine(s, rooms, tiles, TOP_K, ot).expect("combine");
        };
        let per_route = time(&|s, l| {
            routed(s, l);
            own(s, l);
            gated(s, l);
        });
        let line = format!(
            "rows={tokens} unique={unique:2}: per-route {per_route:6.1} us (routed {:6.1}, shared {:5.1}, \
             gate+add {:5.1})\n  register {:6.1} us: group {:5.1}, glu {:6.1}, down {:6.1}, combine {:5.1}",
            time(&|s, l| routed(s, l)),
            time(&|s, l| own(s, l)),
            time(&|s, l| gated(s, l)),
            time(&|s, l| {
                group(s, l);
                glu(s, l);
                down(s, l);
                combine(s);
            }),
            time(&|s, l| group(s, l)),
            time(&|s, l| glu(s, l)),
            time(&|s, l| down(s, l)),
            time(&|s, _| combine(s)),
        );
        eprintln!("{line}");
    }
}
