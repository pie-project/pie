//! The mega-MoE (`linear::moe::mega`): a routed MLP in one persistent
//! launch, against the unfused composition the engine runs apart (route
//! order, the selects, the GLU, the combine).

use crate::common::SM100;
use crate::common::linear_ref::{Bench, t, tab};
use crate::common::tc5::random;
use dtype::Dtype::{Bf16, F32, I32, U8};
use kernels_cuda::jit::Ctx;
use kernels_cuda::linear::act_quant::{self, Bank, Format, Source};
use kernels_cuda::linear::gemm_sm100::{Elem, atom_bytes, supported};
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

/// The mean microseconds of `body`, `reps` of it captured into one graph
/// and replayed (a graph launch's own cost spread over them).
fn timed(b: &Bench, ctx: &Ctx, reps: u32, mut body: impl FnMut()) -> f64 {
    body();
    b.sync();
    let exec = crate::common::captured(ctx, || {
        for _ in 0..reps {
            body();
        }
    });
    crate::common::replay(ctx, exec);
    b.sync();
    let start = std::time::Instant::now();
    crate::common::replay(ctx, exec);
    crate::common::replay(ctx, exec);
    b.sync();
    let us = start.elapsed().as_secs_f64() * 1e6 / f64::from(2 * reps);
    crate::common::forget(exec);
    us
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
    let s = Shape {
        experts: 32,
        ..V4_FLASH
    };
    let mut b = Bench::new(0x3e9a);
    let w = w4a8(&mut b, s, true);
    let ctx = b.ctx().with_tuning(SM100);
    for tokens in [1usize, 3, 8, 33, 100] {
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
            mean < 1e-2 && max < 0.2,
            "{tokens}: {mean:.3e} mean, {max:.3e} at most"
        );
    }
}
