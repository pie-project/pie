//! The mega-MoE (`linear::moe::mega`): a routed MLP in one persistent
//! launch, against the unfused composition the engine runs apart (route
//! order, the selects, the GLU, the combine).

use crate::common::SM100;
use crate::common::linear_ref::{Bench, t, tab};
use crate::common::tc5::random;
use dtype::Dtype::{Bf16, F32, I32, U8};
use kernels_cuda::jit::Ctx;
use kernels_cuda::linear::act_quant::{self, Bank, Format, Source};
use kernels_cuda::linear::gemm_sm100::{Elem, Glu, atom_bytes, supported};
use kernels_cuda::linear::moe;
use kernels_cuda::tensor::Tensor;

/// A routed MLP's shape: hidden, inter, experts, top-k.
#[derive(Clone, Copy, Debug)]
struct Shape {
    hidden: usize,
    inter: usize,
    experts: usize,
    top: usize,
}

/// DeepSeek-V4-Flash's routed experts.
const V4_FLASH: Shape = Shape {
    hidden: 4096,
    inter: 2048,
    experts: 256,
    top: 6,
};

const LIMIT: f32 = 10.0;

/// A W4A8 routed MLP's banks on the device: MXFP4 codes `[e][rows][k / 2]`
/// and their ue8m0 scales laid in atoms.
struct W4a8 {
    up: (u64, u64),
    down: (u64, u64),
}

/// Bytes drawn from splitmix64, one word at a time.
fn bytes(state: &mut u64, n: usize, f: impl Fn(u32) -> u8) -> Vec<u8> {
    tab(n, |_| f(random(state)))
}

/// `n` bytes at a fresh device allocation, a random `chunk` repeated.
fn tiled(b: &mut Bench, n: usize, chunk: &[u8]) -> u64 {
    let at = b.zeros(n);
    let mut off = 0usize;
    while off < n {
        let len = chunk.len().min(n - off);
        crate::common::host::upload_to(at + off as u64, &chunk[..len]);
        off += len;
    }
    at
}

/// The two banks of `s`: whole random codes and scales where `full`, else a
/// random 32 MiB tile repeated (a rate's operands).
fn w4a8(b: &mut Bench, s: Shape, full: bool) -> W4a8 {
    let mut bank = |rows: usize, k: usize| {
        let (codes, atoms) = (
            s.experts * rows * k / 2,
            atom_bytes(s.experts * rows, k, 32),
        );
        let mut bits = b.bits ^ (rows * 31 + k) as u64;
        let code = |r: u32| (r & 0xff) as u8;
        // ue8m0 2^-6 .. 2^-3: the products stay well inside bf16.
        let scale = |r: u32| (121 + r % 4) as u8;
        if full {
            let c = bytes(&mut bits, codes, code);
            let a = bytes(&mut bits, atoms, scale);
            (b.up(&c), b.up(&a))
        } else {
            let chunk = 32 << 20;
            let c = bytes(&mut bits, chunk.min(codes), code);
            let a = bytes(&mut bits, chunk.min(atoms), scale);
            (tiled(b, codes, &c), tiled(b, atoms, &a))
        }
    };
    let up = bank(2 * s.inter, s.hidden);
    let down = bank(s.hidden, s.inter);
    W4a8 { up, down }
}

/// The routed MLP's operands for `tokens` rows: activations, routes,
/// weights, the shared row, the output.
struct Fire {
    x: Tensor,
    routes: Tensor,
    weights: Tensor,
    shared: Tensor,
    y: Tensor,
}

fn fire_of(b: &mut Bench, s: Shape, tokens: usize, seed: u64) -> Fire {
    let mut bits = seed ^ 0x77;
    // Each token's `top` distinct experts.
    let mut routes = Vec::with_capacity(tokens * s.top);
    for _ in 0..tokens {
        let mut row: Vec<i32> = Vec::new();
        while row.len() < s.top {
            let e = (random(&mut bits) % s.experts as u32) as i32;
            if !row.contains(&e) {
                row.push(e);
            }
        }
        routes.extend(row);
    }
    let weights = tab(tokens * s.top, |i| 0.05 + (i % 9) as f32 * 0.07);
    let (x, _) = b.bf16(tokens * s.hidden);
    let (shared, _) = b.bf16(tokens * s.hidden);
    Fire {
        x: t(x, tokens, s.hidden, Bf16),
        routes: t(b.up(&routes), tokens, s.top, I32),
        weights: t(b.up(&weights), tokens, s.top, F32),
        shared: t(shared, tokens, s.hidden, Bf16),
        y: b.out(tokens, s.hidden),
    }
}

/// One W4A8 select as the engine runs it apart: a decode's few routes on the
/// register-dequant kernel over the rows read back from e4m3 (`read`: the
/// rows are read back already), the rest on the grouped tcgen05 GEMM.
#[allow(clippy::too_many_arguments)]
fn select_w4a8(
    ctx: &Ctx,
    x: Tensor,
    read: bool,
    (codes, atoms): (u64, u64),
    [experts, n, k]: [u32; 3],
    routes: Tensor,
    back: u64,
    y: &mut Tensor,
) {
    let fmt = Format::E4m3 { gran: 128 };
    if moe::select_mxfp4_atoms_takes(ctx, routes, x) {
        let back = if read {
            x
        } else {
            act_quant::read_back(ctx, "test", fmt, Source::rows(x, None), back).expect("reads")
        };
        let c = Tensor::new(codes, experts, n * k / 2, U8);
        let took = moe::select_mxfp4_atoms(ctx, back, c, atoms, routes, y).expect("fires");
        assert!(took, "the register-dequant select takes the fire");
        return;
    }
    #[rustfmt::skip]
    let bank = Bank { codes, atoms, experts, n, k, elem: Elem::E2m1, gran: 32, alpha_e: 0, alpha_from: None };
    act_quant::select(ctx, "test", fmt, x, routes, None, bank, y).expect("fires");
}

/// Scratch planes the apart composition lands between its fires.
struct Apart {
    back: u64,
    packed: Tensor,
    hidden: Tensor,
    routed: Tensor,
}

fn apart_of(b: &mut Bench, s: Shape, tokens: usize) -> Apart {
    let count = tokens * s.top;
    Apart {
        back: b.zeros(tokens * s.hidden * 2),
        packed: b.out(count, 2 * s.inter),
        hidden: b.out(count, s.inter),
        routed: b.out(count, s.hidden),
    }
}

/// DeepSeek-V4's routed MLP as the engine runs it apart: the gate-up
/// select, the clamped SwiGLU under each route's weight landing its rows
/// read back through e4m3, the down select, and the plain sum with the
/// shared row added.
fn apart(ctx: &Ctx, s: Shape, w: &W4a8, f: &mut Fire, a: &mut Apart) {
    let (e, h, i) = (s.experts as u32, s.hidden as u32, s.inter as u32);
    select_w4a8(
        ctx,
        f.x,
        false,
        w.up,
        [e, 2 * i, h],
        f.routes,
        a.back,
        &mut a.packed,
    );
    let fan = s.top as u32;
    let landed = kernels_cuda::linear::glu::swiglu_clamp_routed_e4m3(
        ctx,
        a.packed,
        i,
        f.weights,
        fan,
        LIMIT,
        &mut a.hidden,
    )
    .expect("fires");
    assert!(landed, "the e4m3 GLU lands the rows");
    select_w4a8(
        ctx,
        a.hidden,
        true,
        w.down,
        [e, h, i],
        f.routes,
        a.back,
        &mut a.routed,
    );
    moe::sum_add(ctx, a.routed, f.shared, &mut f.y).expect("fires");
}

/// The microseconds of `body` captured into a graph: the least of many
/// replays each timed by events, a card shared with other work slowing
/// some.
#[allow(unsafe_code)]
fn timed(b: &Bench, ctx: &Ctx, reps: u32, mut body: impl FnMut()) -> f64 {
    use kernels_cuda::cudarc::runtime::sys as rt;
    body();
    b.sync();
    let exec = crate::common::captured(ctx, &mut body);
    let stream = ctx.stream().cast();
    let mut best = f32::MAX;
    // SAFETY: the events are locals created and destroyed here, recorded on
    // the test's live stream around a replay of `exec`, which outlives them.
    unsafe {
        let (mut start, mut stop): (rt::cudaEvent_t, rt::cudaEvent_t) =
            (core::ptr::null_mut(), core::ptr::null_mut());
        rt::cudaEventCreate(&raw mut start);
        rt::cudaEventCreate(&raw mut stop);
        for _ in 0..10 * reps {
            rt::cudaEventRecord(start, stream);
            crate::common::replay(ctx, exec);
            rt::cudaEventRecord(stop, stream);
            rt::cudaEventSynchronize(stop);
            let mut ms = 0f32;
            rt::cudaEventElapsedTime(&raw mut ms, start, stop);
            best = best.min(ms);
        }
        rt::cudaEventDestroy(start);
        rt::cudaEventDestroy(stop);
    }
    crate::common::forget(exec);
    f64::from(best) * 1e3
}

/// The mega-MoE's rate at DeepSeek-V4-Flash's decode shapes, against the
/// composition apart: `PIE_TOKENS` a comma list of decode widths.
#[test]
#[ignore = "a rate bench: cargo test ... -- --ignored mega_moe_rate --nocapture"]
fn mega_moe_rate() {
    if !supported() {
        eprintln!("skipped: no tcgen05 on this device");
        return;
    }
    let s = V4_FLASH;
    let widths: Vec<usize> = std::env::var("PIE_TOKENS")
        .unwrap_or_else(|_| "1,2,4,8,16,32,64,128,256".to_string())
        .split(',')
        .map(|v| v.trim().parse().expect("a count"))
        .collect();
    let mut b = Bench::new(0x4f1a);
    let w = w4a8(&mut b, s, false);
    let ctx = b.ctx().with_tuning(SM100);
    for tokens in widths {
        let mut f = fire_of(&mut b, s, tokens, tokens as u64);
        let mut a = apart_of(&mut b, s, tokens);
        let us = timed(&b, &ctx, 20, || apart(&ctx, s, &w, &mut f, &mut a));
        let mut y = b.out(tokens, s.hidden);
        let m = mlp_of(s, &w, &f);
        let fold = moe::Fold::First(Some(f.shared));
        let mega = timed(&b, &ctx, 20, || {
            assert!(
                moe::mega(&ctx, m, fold, &mut y).expect("fires"),
                "takes the fire"
            );
        });
        let speedup = us / mega;
        eprintln!("v4-flash {tokens} tokens: apart {us:.1} us, mega {mega:.1} us ({speedup:.2}x)");
    }
}

/// B200's tuning, the mega-MoE taking every fire it can.
const WIDE: kernels_cuda::tuning::Tuning = kernels_cuda::tuning::Tuning {
    moe_mega_until: 65_536,
    ..SM100
};

/// The routed MLP's banks as the mega-MoE reads them.
fn mlp_of(s: Shape, w: &W4a8, f: &Fire) -> moe::RoutedMlp {
    let (e, h, i) = (s.experts, s.hidden, s.inter);
    let bank = |(codes, atoms): (u64, u64), rows: usize, k: usize| moe::GluBank::Mxfp4A8 {
        codes: t(codes, e, rows * k / 2, U8),
        atoms: t(atoms, e, rows * k / 32, U8),
    };
    moe::RoutedMlp {
        x: f.x,
        up: bank(w.up, 2 * i, h),
        glu: kernels_cuda::linear::gemm_sm100::Glu::SwigluClamp { limit: LIMIT },
        down: bank(w.down, h, i),
        routes: f.routes,
        weights: f.weights,
        inter: i as u32,
    }
}

/// The mean and largest distance of `got` from `want`, each over want's
/// mean magnitude.
fn spread(got: &[f64], want: &[f64]) -> (f64, f64) {
    let scale = want.iter().map(|w| w.abs()).sum::<f64>() / want.len() as f64;
    let far: Vec<f64> = got.iter().zip(want).map(|(g, w)| (g - w).abs()).collect();
    let mean = far.iter().sum::<f64>() / far.len() as f64;
    (
        mean / scale,
        far.iter().copied().fold(0.0, f64::max) / scale,
    )
}

#[test]
fn the_w4a8_mega_moe_answers_the_composition_apart() {
    if !supported() {
        eprintln!("skipped: no tcgen05 on this device");
        return;
    }
    // V4-Flash's experts (fewer of them), and a narrower GLU of whole 256s
    // (768 wide, Qwen3-30B-A3B's) under eight routes a token.
    let narrow = Shape {
        hidden: 2048,
        inter: 768,
        experts: 24,
        top: 8,
    };
    for (s, widths) in [
        (
            Shape {
                experts: 32,
                ..V4_FLASH
            },
            // 400 tokens: 2400 routes, laid out in global memory.
            &[1usize, 3, 8, 33, 100, 400][..],
        ),
        (narrow, &[2, 40, 300][..]),
    ] {
        let mut b = Bench::new(0x3e9a ^ s.inter as u64);
        let w = w4a8(&mut b, s, true);
        // Every width the kernel takes, B200's span of it aside.
        let ctx = b.ctx().with_tuning(WIDE);
        for &tokens in widths {
            let mut f = fire_of(&mut b, s, tokens, 0x51 + tokens as u64);
            let mut a = apart_of(&mut b, s, tokens);
            apart(&ctx, s, &w, &mut f, &mut a);
            let want = b.read(f.y.ptr, tokens * s.hidden);
            let mut y = b.out(tokens, s.hidden);
            let m = mlp_of(s, &w, &f);
            let took = moe::mega(&ctx, m, moe::Fold::First(Some(f.shared)), &mut y).expect("fires");
            assert!(took, "{tokens}: the mega-MoE takes the fire");
            let got = b.read(y.ptr, tokens * s.hidden);
            let (mean, max) = spread(&got, &want);
            let same = got
                .iter()
                .zip(&want)
                .filter(|(g, w)| g.to_bits() == w.to_bits())
                .count();
            eprintln!(
                "{tokens} tokens: mean {mean:.2e} max {max:.2e}, {same} of {} bit for bit",
                got.len()
            );
            assert!(
                mean < 1e-3 && max < 0.05,
                "{tokens}: {mean:.3e} mean, {max:.3e} at most"
            );
        }
    }
}

/// A bf16 bank `[experts][rows][k]` of small random values.
fn bf16_bank(b: &mut Bench, experts: usize, rows: usize, k: usize) -> Tensor {
    let mut bits = b.bits ^ (rows * 131 + k) as u64;
    let words = tab(experts * rows * k, |_| {
        let r = random(&mut bits);
        // Values in (-1/16, 1/16): a 4096-long dot stays near one.
        let v = ((r >> 8) as f32 / 16_777_216.0 - 0.5) / 8.0;
        crate::common::to_bf16(v)
    });
    t(b.up(&words), experts, rows * k, Bf16)
}

/// A bf16 routed MLP as the engine runs it apart: the gate-up select, the
/// GLU (clamped and under each route's weight where `first`), the down
/// select, and the fold (the plain sum plus the shared row where `first`,
/// else each route under its weight).
#[allow(clippy::too_many_arguments)]
fn apart_bf16(
    ctx: &Ctx,
    s: Shape,
    (up, down): (Tensor, Tensor),
    first: bool,
    f: &mut Fire,
    a: &mut Apart,
) {
    use kernels_cuda::linear::glu;
    let (fan, i) = (s.top as u32, s.inter as u32);
    let res = moe::ExpertTable::RESIDENT;
    moe::matmul_select(ctx, f.x, up, f.routes, &mut a.packed, res).expect("fires");
    if first {
        glu::swiglu_clamp_routed(ctx, a.packed, i, f.weights, fan, LIMIT, &mut a.hidden)
            .expect("fires");
    } else {
        glu::swiglu(ctx, a.packed, i, fan, &mut a.hidden).expect("fires");
    }
    moe::matmul_select(ctx, a.hidden, down, f.routes, &mut a.routed, res).expect("fires");
    if first {
        moe::sum_add(ctx, a.routed, f.shared, &mut f.y).expect("fires");
    } else {
        moe::weighted_sum(ctx, a.routed, f.weights, &mut f.y).expect("fires");
    }
}

#[test]
fn the_bf16_mega_moe_answers_the_composition_apart() {
    if !supported() {
        eprintln!("skipped: no tcgen05 on this device");
        return;
    }
    let s = Shape {
        hidden: 2048,
        inter: 1024,
        experts: 16,
        top: 4,
    };
    let mut b = Bench::new(0xbf16);
    let up = bf16_bank(&mut b, s.experts, 2 * s.inter, s.hidden);
    let down = bf16_bank(&mut b, s.experts, s.hidden, s.inter);
    // Every width the kernel takes, B200's floor for bf16 banks aside.
    let ctx = b.ctx().with_tuning(kernels_cuda::tuning::Tuning {
        moe_mega_bf16_from: 0,
        ..SM100
    });
    for first in [false, true] {
        // 600 tokens: 2400 routes, laid out in global memory.
        for tokens in [1usize, 5, 40, 600] {
            let what = format!("{tokens} tokens, weights first {first}");
            let mut f = fire_of(&mut b, s, tokens, 0x77 + tokens as u64);
            let mut a = apart_of(&mut b, s, tokens);
            apart_bf16(&ctx, s, (up, down), first, &mut f, &mut a);
            let want = b.read(f.y.ptr, tokens * s.hidden);
            let mut y = b.out(tokens, s.hidden);
            let glu = if first {
                Glu::SwigluClamp { limit: LIMIT }
            } else {
                Glu::Swiglu
            };
            let m = moe::RoutedMlp {
                x: f.x,
                up: moe::GluBank::Bf16(up),
                glu,
                down: moe::GluBank::Bf16(down),
                routes: f.routes,
                weights: f.weights,
                inter: s.inter as u32,
            };
            let fold = if first {
                moe::Fold::First(Some(f.shared))
            } else {
                moe::Fold::Weighted
            };
            let took = moe::mega(&ctx, m, fold, &mut y).expect("fires");
            assert!(took, "{what}: the mega-MoE takes the fire");
            let got = b.read(y.ptr, tokens * s.hidden);
            let (mean, max) = spread(&got, &want);
            eprintln!("bf16 {what}: mean {mean:.2e} max {max:.2e}");
            assert!(
                mean < 2e-3 && max < 5e-2,
                "{what}: {mean:.3e} mean, {max:.3e} at most"
            );
        }
    }
}

/// A fire under a live window (a graph's replay at fewer rows than it was
/// captured for) lands those rows as a fire of them alone would, bit for
/// bit, and leaves the rest.
#[test]
fn the_mega_moe_lands_its_live_window_alone() {
    if !supported() {
        eprintln!("skipped: no tcgen05 on this device");
        return;
    }
    let s = Shape {
        experts: 16,
        ..V4_FLASH
    };
    let mut b = Bench::new(0x3a1d);
    let w = w4a8(&mut b, s, true);
    // As B200 launches it: a programmatic dependent, not a cooperative launch.
    let ctx = b
        .ctx()
        .with_tuning(kernels_cuda::tuning::Tuning { pdl: true, ..SM100 });
    let tokens = 24usize;
    let (live, first) = (9usize, 5usize);
    let f = fire_of(&mut b, s, tokens, 0x91);
    let mut whole = b.out(tokens, s.hidden);
    let m = mlp_of(s, &w, &f);
    let fold = moe::Fold::First(Some(f.shared));
    assert!(
        moe::mega(&ctx, m, fold, &mut whole).expect("fires"),
        "takes the fire"
    );
    let want = b.read(whole.ptr, tokens * s.hidden);
    let mut y = t(
        b.up(&vec![0x7f7fu16; tokens * s.hidden]),
        tokens,
        s.hidden,
        Bf16,
    );
    ctx.arm_stage(b.up(&[live as u32, first as u32]));
    let took = moe::mega(&ctx, m, fold, &mut y).expect("fires");
    ctx.disarm_stage();
    assert!(took, "the windowed fire is taken");
    let got = b.read(y.ptr, tokens * s.hidden);
    let span = first * s.hidden..(first + live) * s.hidden;
    let same = got[span.clone()]
        .iter()
        .zip(&want[span.clone()])
        .all(|(g, w)| g.to_bits() == w.to_bits());
    assert!(same, "the live rows are the whole fire's, bit for bit");
    let sentinel = f64::from(crate::common::from_bf16(0x7f7f));
    let rest = got
        .iter()
        .enumerate()
        .filter(|(i, _)| !span.contains(i))
        .all(|(_, &v)| v == sentinel);
    assert!(rest, "rows past the window are untouched");
}

/// The bf16 mega-MoE's rate at Qwen3-30B-A3B's routed experts (128 of them,
/// top 8, 2048 by 768), against the composition apart.
#[test]
#[ignore = "a rate bench: cargo test ... -- --ignored mega_moe_bf16_rate --nocapture"]
fn mega_moe_bf16_rate() {
    if !supported() {
        eprintln!("skipped: no tcgen05 on this device");
        return;
    }
    let s = Shape {
        hidden: 2048,
        inter: 768,
        experts: 128,
        top: 8,
    };
    let widths: Vec<usize> = std::env::var("PIE_TOKENS")
        .unwrap_or_else(|_| "1,4,16,64,128,256".to_string())
        .split(',')
        .map(|v| v.trim().parse().expect("a count"))
        .collect();
    let mut b = Bench::new(0xb1f6);
    let up = bf16_bank(&mut b, s.experts, 2 * s.inter, s.hidden);
    let down = bf16_bank(&mut b, s.experts, s.hidden, s.inter);
    // Every width timed, B200's floor for bf16 banks (where it falls) aside.
    let ctx = b.ctx().with_tuning(kernels_cuda::tuning::Tuning {
        moe_mega_bf16_from: 0,
        ..SM100
    });
    for tokens in widths {
        let mut f = fire_of(&mut b, s, tokens, tokens as u64);
        let mut a = apart_of(&mut b, s, tokens);
        let us = timed(&b, &ctx, 20, || {
            apart_bf16(&ctx, s, (up, down), false, &mut f, &mut a)
        });
        let mut y = b.out(tokens, s.hidden);
        let m = moe::RoutedMlp {
            x: f.x,
            up: moe::GluBank::Bf16(up),
            glu: Glu::Swiglu,
            down: moe::GluBank::Bf16(down),
            routes: f.routes,
            weights: f.weights,
            inter: s.inter as u32,
        };
        let mega = timed(&b, &ctx, 20, || {
            assert!(
                moe::mega(&ctx, m, moe::Fold::Weighted, &mut y).expect("fires"),
                "takes the fire"
            );
        });
        let speedup = us / mega;
        eprintln!(
            "qwen3-30b bf16 {tokens} tokens: apart {us:.1} us, mega {mega:.1} us ({speedup:.2}x)"
        );
    }
}

/// The mega-MoE's cost in a decode's chain of launches, against the
/// composition apart's: each timed after the same programmatic predecessor
/// (a quantizer's read-back) under B200's launch decisions (`kernel.pdl`),
/// the predecessor's own time taken off. `PIE_TOKENS` a comma list.
#[test]
#[ignore = "a rate bench: cargo test ... -- --ignored mega_moe_chain_rate --nocapture"]
fn mega_moe_chain_rate() {
    if !supported() {
        eprintln!("skipped: no tcgen05 on this device");
        return;
    }
    let s = V4_FLASH;
    let widths: Vec<usize> = std::env::var("PIE_TOKENS")
        .unwrap_or_else(|_| "1,2,4,8,16,64".to_string())
        .split(',')
        .map(|v| v.trim().parse().expect("a count"))
        .collect();
    let mut b = Bench::new(0x4f1a);
    let w = w4a8(&mut b, s, false);
    let ctx = b
        .ctx()
        .with_tuning(kernels_cuda::tuning::Tuning { pdl: true, ..SM100 });
    for tokens in widths {
        let mut f = fire_of(&mut b, s, tokens, tokens as u64);
        let mut a = apart_of(&mut b, s, tokens);
        let lead = b.zeros(tokens * s.hidden * 2);
        let fmt = Format::E4m3 { gran: 128 };
        let x = f.x;
        let pred = || {
            act_quant::read_back(&ctx, "test", fmt, Source::rows(x, None), lead).expect("reads");
        };
        let alone = timed(&b, &ctx, 20, pred);
        let apart_us = timed(&b, &ctx, 20, || {
            pred();
            apart(&ctx, s, &w, &mut f, &mut a);
        }) - alone;
        let mut y = b.out(tokens, s.hidden);
        let m = mlp_of(s, &w, &f);
        let fold = moe::Fold::First(Some(f.shared));
        let mega = timed(&b, &ctx, 20, || {
            pred();
            assert!(
                moe::mega(&ctx, m, fold, &mut y).expect("fires"),
                "takes the fire"
            );
        }) - alone;
        let speedup = apart_us / mega;
        eprintln!(
            "v4-flash chained {tokens} tokens: apart {apart_us:.1} us, mega {mega:.1} us ({speedup:.2}x)"
        );
    }
}

/// The mega-MoE's rate at DeepSeek-V4-Flash's prefill widths, against the
/// path the engine runs there (the fused routed MLP from
/// `kernel.moe.w4a8_fused_from` tokens, the four apart below it), both
/// chained after the same programmatic producer. `PIE_TOKENS` a comma list.
#[test]
#[ignore = "a rate bench: cargo test ... -- --ignored mega_moe_prefill_rate --nocapture"]
fn mega_moe_prefill_rate() {
    if !supported() {
        eprintln!("skipped: no tcgen05 on this device");
        return;
    }
    let s = V4_FLASH;
    let widths: Vec<usize> = std::env::var("PIE_TOKENS")
        .unwrap_or_else(|_| "256,384,512,1024,2048,4096,8192".to_string())
        .split(',')
        .map(|v| v.trim().parse().expect("a count"))
        .collect();
    let mut b = Bench::new(0x4f1a);
    let w = w4a8(&mut b, s, false);
    let pdl = kernels_cuda::tuning::Tuning { pdl: true, ..WIDE };
    let ctx = b.ctx().with_tuning(pdl);
    for tokens in widths {
        let mut f = fire_of(&mut b, s, tokens, tokens as u64);
        let mut a = apart_of(&mut b, s, tokens);
        let lead = b.zeros(tokens * s.hidden * 2);
        let x = f.x;
        let fmt = Format::E4m3 { gran: 128 };
        let pred = || {
            act_quant::read_back(&ctx, "test", fmt, Source::rows(x, None), lead).expect("reads");
        };
        let alone = timed(&b, &ctx, 5, pred);
        let m = mlp_of(s, &w, &f);
        let mut fused_y = b.out(tokens, s.hidden);
        let mut fused = false;
        let today = timed(&b, &ctx, 5, || {
            pred();
            fused = moe::experts_weighted_first(&ctx, m, f.shared, &mut fused_y).expect("fires");
            if !fused {
                apart(&ctx, s, &w, &mut f, &mut a);
            }
        }) - alone;
        let mut y = b.out(tokens, s.hidden);
        let fold = moe::Fold::First(Some(f.shared));
        let mega = timed(&b, &ctx, 5, || {
            pred();
            assert!(
                moe::mega(&ctx, m, fold, &mut y).expect("fires"),
                "takes the fire"
            );
        }) - alone;
        let path = if fused { "fused" } else { "apart" };
        let speedup = today / mega;
        eprintln!(
            "v4-flash prefill {tokens} tokens: {path} {today:.1} us, mega {mega:.1} us ({speedup:.2}x)"
        );
    }
}

/// A kernel holding a CTA a multiprocessor (200 KiB of shared memory each,
/// too much beside it for a mega-MoE CTA), sleeping out `ns` nanoseconds.
const HOG: &str = r#"
extern "C" __global__ void hog(unsigned long long ns) {
    extern __shared__ unsigned char held[];
    held[threadIdx.x] = 0;
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        __nanosleep(1000);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}
"#;
const HOG_SMEM: u32 = 200 * 1024;

/// The hog compiled and loaded: its module and function.
#[allow(unsafe_code)]
fn hog_kernel() -> (
    kernels_cuda::cudarc::driver::sys::CUmodule,
    kernels_cuda::cudarc::driver::sys::CUfunction,
) {
    use kernels_cuda::cudarc::driver::sys as dr;
    let opts = kernels_cuda::cudarc::nvrtc::CompileOptions {
        arch: Some("compute_100"),
        ..Default::default()
    };
    let ptx = kernels_cuda::cudarc::nvrtc::compile_ptx_with_opts(HOG, opts).expect("compiles");
    let text = std::ffi::CString::new(ptx.to_src()).expect("ptx text");
    // SAFETY: the out-pointers are locals; the module image is the
    // NUL-terminated PTX just compiled.
    unsafe {
        let mut module: dr::CUmodule = core::ptr::null_mut();
        let code = dr::cuModuleLoadData(&raw mut module, text.as_ptr().cast());
        assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "loads");
        let mut f: dr::CUfunction = core::ptr::null_mut();
        let code = dr::cuModuleGetFunction(&raw mut f, module, c"hog".as_ptr());
        assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "finds the hog");
        let attr = dr::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
        let code = dr::cuFuncSetAttribute(f, attr, HOG_SMEM as i32);
        assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "takes its shared memory");
        (module, f)
    }
}

/// The mega-MoE's launch, cooperative and a programmatic dependent in one
/// config, as the driver takes it on this card: a grid a CTA a
/// multiprocessor at a mega-MoE CTA's shared memory launches, and one that
/// cannot all be resident at once is refused rather than seated in waves.
#[test]
#[allow(unsafe_code)]
fn a_cooperative_programmatic_launch_seats_its_grid_whole_or_not_at_all() {
    use kernels_cuda::cudarc::driver::sys as dr;
    if !supported() {
        eprintln!("skipped: no tcgen05 on this device");
        return;
    }
    let b = Bench::new(1);
    let ctx = b.ctx();
    let sms = ctx.multiprocessors().expect("a device");
    let (module, hog) = hog_kernel();
    let launch = |grid: u32| {
        let mut ns: u64 = 1000;
        let mut args = [(&raw mut ns).cast::<core::ffi::c_void>()];
        // SAFETY: plain C data, valid zeroed; the attributes and the one
        // argument are locals outliving the call on the test's live stream.
        unsafe {
            let mut attrs: [dr::CUlaunchAttribute; 2] = core::mem::zeroed();
            attrs[0].id = dr::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
            attrs[0].value.cooperative = 1;
            attrs[1].id =
                dr::CUlaunchAttributeID::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
            attrs[1].value.programmaticStreamSerializationAllowed = 1;
            let config = dr::CUlaunchConfig {
                gridDimX: grid,
                gridDimY: 1,
                gridDimZ: 1,
                blockDimX: 32,
                blockDimY: 1,
                blockDimZ: 1,
                sharedMemBytes: HOG_SMEM,
                hStream: ctx.stream().cast(),
                attrs: attrs.as_mut_ptr(),
                numAttrs: 2,
            };
            dr::cuLaunchKernelEx(
                &raw const config,
                hog,
                args.as_mut_ptr(),
                core::ptr::null_mut(),
            )
        }
    };
    assert_eq!(
        launch(sms),
        dr::CUresult::CUDA_SUCCESS,
        "a grid the card seats at once launches"
    );
    assert_eq!(
        launch(sms + 1),
        dr::CUresult::CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE,
        "a grid the card cannot seat at once is refused"
    );
    b.sync();
    // SAFETY: the module is this test's, its launches drained above.
    unsafe {
        dr::cuModuleUnload(module);
    }
}

/// A mega-MoE fire beside a long kernel on another stream holding most SMs
/// lands (its cooperative launch seated once they free), bit for bit what
/// it lands alone.
#[test]
#[allow(unsafe_code)]
fn the_mega_moe_lands_beside_another_streams_long_kernel() {
    use kernels_cuda::cudarc::driver::sys as dr;
    use kernels_cuda::cudarc::runtime::sys as rt;
    if !supported() {
        eprintln!("skipped: no tcgen05 on this device");
        return;
    }
    let s = Shape {
        experts: 32,
        ..V4_FLASH
    };
    let mut b = Bench::new(0x6a11);
    let w = w4a8(&mut b, s, true);
    let tuning = kernels_cuda::tuning::Tuning { pdl: true, ..SM100 };
    let ctx = b.ctx().with_tuning(tuning);
    let (module, hog) = hog_kernel();
    // A decode (its fold in the down epilogue) and a wider fire (folded at
    // the end), the hog leaving 8 SMs free.
    let held = ctx.multiprocessors().expect("a device") - 8;
    for tokens in [3usize, 40] {
        let f = fire_of(&mut b, s, tokens, 0x5150 + tokens as u64);
        let m = mlp_of(s, &w, &f);
        let fold = moe::Fold::First(Some(f.shared));
        let mut alone = b.out(tokens, s.hidden);
        assert!(
            moe::mega(&ctx, m, fold, &mut alone).expect("fires"),
            "takes the fire"
        );
        let want = b.read(alone.ptr, tokens * s.hidden);
        let mut y = b.out(tokens, s.hidden);
        b.sync();
        // SAFETY: every out-pointer is a local; the stream and events live
        // to the end of the block, and the hog's one argument outlives its
        // launch on that stream.
        unsafe {
            let mut side: rt::cudaStream_t = core::ptr::null_mut();
            rt::cudaStreamCreateWithFlags(&raw mut side, rt::cudaStreamNonBlocking);
            let mut done: rt::cudaEvent_t = core::ptr::null_mut();
            rt::cudaEventCreate(&raw mut done);
            let mut ns: u64 = 300_000_000;
            let mut args = [(&raw mut ns).cast::<core::ffi::c_void>()];
            let code = dr::cuLaunchKernel(
                hog,
                held,
                1,
                1,
                32,
                1,
                1,
                HOG_SMEM,
                side.cast(),
                args.as_mut_ptr(),
                core::ptr::null_mut(),
            );
            assert_eq!(code, dr::CUresult::CUDA_SUCCESS, "the hog launches");
            std::thread::sleep(std::time::Duration::from_millis(20));
            assert!(
                moe::mega(&ctx, m, fold, &mut y).expect("fires"),
                "takes the fire"
            );
            rt::cudaEventRecord(done, ctx.stream().cast());
            let start = std::time::Instant::now();
            while rt::cudaEventQuery(done) == rt::cudaError::cudaErrorNotReady {
                let late = start.elapsed() > std::time::Duration::from_secs(10);
                assert!(
                    !late,
                    "{tokens} tokens: the fire never lands beside the hog"
                );
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
            rt::cudaStreamSynchronize(side);
            rt::cudaEventDestroy(done);
            rt::cudaStreamDestroy(side);
        }
        let got = b.read(y.ptr, tokens * s.hidden);
        let same = got
            .iter()
            .zip(&want)
            .all(|(g, w)| g.to_bits() == w.to_bits());
        assert!(
            same,
            "{tokens} tokens: the fire beside the hog lands what it lands alone"
        );
    }
    // SAFETY: the module is this test's, its launches drained above.
    unsafe {
        dr::cuModuleUnload(module);
    }
}
