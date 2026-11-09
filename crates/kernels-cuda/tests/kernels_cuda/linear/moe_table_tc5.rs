//! The tcgen05 grouped selects, routed MLPs and mega-MoE through expert
//! tables: the contiguous bank, its own table, and its experts scattered
//! across a pool (each plane a whole number of its rows past the pool's
//! base, which the launch maps as the table's window) land the same bits; a
//! table with two experts swapped does not. B200 only (they skip elsewhere).

use crate::common::SM100;
use crate::common::expert_pool::{Plane, degenerate, garble, scatter};
use crate::common::linear_ref::{t, tab};
use crate::common::{Gpu, Lcg};
use crate::linear::moe_table::{routes, same_bits};
use dtype::Dtype::{Bf16, F32, I32, U8};
use kernels_cuda::linear::act_quant::{self, Bank, Format};
use kernels_cuda::linear::gemm_sm100::{Elem, Glu, atom_bytes, supported};
use kernels_cuda::linear::moe::{self, ExpertTable, GluBank};
use kernels_cuda::tuning::Tuning;

/// B200's tuning with every select on the grouped GEMM: tcgen05 from one
/// token, the register-dequant kernels off.
const TC5: Tuning = Tuning {
    moe_tc5_from: 1,
    mxfp4_mma_from: u32::MAX,
    nvfp4_mma_from: u32::MAX,
    mxfp4_a8_from: u32::MAX,
    moe_w4a8_fused_from: 1,
    moe_mega_until: 65_536,
    moe_mega_bf16_from: 1,
    ..SM100
};

fn here() -> bool {
    if supported() {
        return true;
    }
    eprintln!("skipped: no tcgen05 on this device");
    false
}

/// `n` random bytes drawn by `f` from a 32-bit word.
fn bytes(lcg: &mut Lcg, n: usize, f: impl Fn(u32) -> u8) -> Vec<u8> {
    tab(n, |_| f((lcg.unit() * 4_294_967_000.0) as u32))
}

/// ue8m0 2^-6 .. 2^-3: products stay well inside bf16.
fn ue8m0(r: u32) -> u8 {
    (121 + r % 4) as u8
}

/// A bf16 plane of `n` random values.
fn bf16(g: &mut Gpu, lcg: &mut Lcg, n: usize) -> u64 {
    let (raw, _) = lcg.row(n);
    g.up(&raw)
}

/// [`same_bits`] for a routed MLP's two banks, one pool and window: `fire`
/// under no tables, the banks' own, the scattered ones (the banks garbled
/// after), and with experts 0 and 1 swapped in the gate-up bank's.
fn same_bits_mlp(
    g: &mut Gpu,
    what: &str,
    experts: usize,
    [up, down]: [&[Plane]; 2],
    (y, y_bytes): (u64, usize),
    mut fire: impl FnMut([ExpertTable; 2]),
) {
    let mut run = |g: &mut Gpu, tables: [ExpertTable; 2]| -> Vec<u8> {
        g.fill(y, 0x7f, y_bytes);
        fire(tables);
        g.sync();
        g.down::<u8>(y, y_bytes)
    };
    let want = run(g, [ExpertTable::RESIDENT; 2]);
    let own = [degenerate(g, experts, up), degenerate(g, experts, down)];
    let got = run(g, own.map(|at| ExpertTable::located(at, (0, 0))));
    assert!(got == want, "{what}: the banks' own tables moved a bit");
    let all: Vec<Plane> = up.iter().chain(down).copied().collect();
    let pool = scatter(g, experts, &all, 0x7c5 ^ experts as u64);
    let (count, nu) = (all.len(), up.len());
    let split = |lo: usize, hi: usize, swap: bool| -> Vec<u64> {
        let order = |e: usize| match (swap, e) {
            (true, 0) => 1,
            (true, 1) => 0,
            _ => e,
        };
        (0..experts)
            .flat_map(|e| (lo..hi).map(move |p| (order(e), p)))
            .map(|(e, p)| pool.entries[e * count + p])
            .collect()
    };
    let (up_at, down_at) = (g.up(&split(0, nu, false)), g.up(&split(nu, count, false)));
    let wrong = g.up(&split(0, nu, true));
    garble(g, experts, up);
    garble(g, experts, down);
    let window = pool.window();
    let got = run(
        g,
        [up_at, down_at].map(|at| ExpertTable::located(at, window)),
    );
    assert!(got == want, "{what}: the scattered tables moved a bit");
    let got = run(
        g,
        [wrong, down_at].map(|at| ExpertTable::located(at, window)),
    );
    assert!(
        got != want,
        "{what}: a gate-up table with two experts swapped landed the same bits"
    );
}

#[test]
fn a_bf16_grouped_select_lands_the_same_bits_through_any_table() {
    if !here() {
        return;
    }
    let mut g = Gpu::open();
    let mut lcg = Lcg::seeded(0x7c51);
    let (experts, tokens, top, n, k) = (8usize, 48usize, 2usize, 256usize, 512usize);
    let bank = bf16(&mut g, &mut lcg, experts * n * k);
    let x = t(bf16(&mut g, &mut lcg, tokens * k), tokens, k, Bf16);
    let r = t(g.up(&routes(tokens, top, experts)), tokens, top, I32);
    let y = g.zeros(tokens * top * n * 2);
    let ctx = g.ctx().with_tuning(TC5);
    let plane = Plane::of(bank, (n * k * 2) as u64).aligned((k * 2) as u64);
    same_bits(
        &mut g,
        "bf16 tc5 select",
        experts,
        &[plane],
        (y, tokens * top * n * 2),
        |table| {
            let mut yt = t(y, tokens * top, n, Bf16);
            let bank = t(bank, experts, n * k, Bf16);
            moe::matmul_select(&ctx, x, bank, r, &mut yt, table).expect("fires");
        },
    );
}

/// A GLU select's bank of `kind` (bf16, MXFP4 biased, NVFP4) and its planes.
fn glu_bank(
    g: &mut Gpu,
    lcg: &mut Lcg,
    kind: u8,
    [experts, rows, k]: [usize; 3],
) -> (GluBank, Vec<Plane>) {
    match kind {
        0 => {
            let w = bf16(g, lcg, experts * rows * k);
            let plane = Plane::of(w, (rows * k * 2) as u64).aligned((k * 2) as u64);
            (GluBank::Bf16(t(w, experts, rows * k, Bf16)), vec![plane])
        }
        1 => {
            let codes = g.up(&bytes(lcg, experts * rows * k / 2, |r| r as u8));
            let scales = g.up(&bytes(lcg, experts * rows * k / 32, ue8m0));
            let bias = bf16(g, lcg, experts * rows);
            let planes = vec![
                Plane::of(codes, (rows * k / 2) as u64).aligned((k / 2) as u64),
                Plane::of(scales, (rows * k / 32) as u64).aligned((k / 32).max(16) as u64),
                Plane::of(bias, (rows * 2) as u64).aligned(16),
            ];
            #[rustfmt::skip]
            let bank = GluBank::Mxfp4 {
                codes: t(codes, experts, rows * k / 2, U8),
                scales: t(scales, experts, rows * k / 32, U8),
                bias: Some(t(bias, experts, rows, Bf16)),
            };
            (bank, planes)
        }
        _ => {
            let codes = g.up(&bytes(lcg, experts * rows * k / 2, |r| r as u8));
            // e4m3 0.25 .. 1.75: finite scales.
            let scales = g.up(&bytes(lcg, experts * rows * k / 16, |r| {
                (0x30 + r % 0x10) as u8
            }));
            let ts = g.up(&tab(experts * rows, |i| 0.5 + (i % 7) as f32 / 8.0));
            let planes = vec![
                Plane::of(codes, (rows * k / 2) as u64).aligned((k / 2) as u64),
                Plane::of(scales, (rows * k / 16) as u64).aligned((k / 16).max(16) as u64),
                Plane::of(ts, (rows * 4) as u64).aligned(16),
            ];
            #[rustfmt::skip]
            let bank = GluBank::Nvfp4 {
                codes: t(codes, experts, rows * k / 2, U8),
                scales: t(scales, experts, rows * k / 16, U8),
                tensor_scales: t(ts, experts, rows * 4, U8),
            };
            (bank, planes)
        }
    }
}

#[test]
fn a_grouped_glu_select_lands_the_same_bits_through_any_table() {
    if !here() {
        return;
    }
    let mut g = Gpu::open();
    let mut lcg = Lcg::seeded(0x7c52);
    let (experts, tokens, top, inter, k) = (8usize, 48usize, 2usize, 128usize, 512usize);
    for kind in [0u8, 1, 2] {
        let (bank, planes) = glu_bank(&mut g, &mut lcg, kind, [experts, 2 * inter, k]);
        let x = t(bf16(&mut g, &mut lcg, tokens * k), tokens, k, Bf16);
        let r = t(g.up(&routes(tokens, top, experts)), tokens, top, I32);
        let y = g.zeros(tokens * top * inter * 2);
        let ctx = g.ctx().with_tuning(TC5);
        let what = format!("tc5 GLU select, bank kind {kind}");
        same_bits(
            &mut g,
            &what,
            experts,
            &planes,
            (y, tokens * top * inter * 2),
            |table| {
                let mut yt = t(y, tokens * top, inter, Bf16);
                let took =
                    moe::select_glu(&ctx, x, bank, r, &mut yt, Glu::Swiglu, table).expect("fires");
                assert!(took, "{what}: the grouped GEMM takes the fire");
            },
        );
    }
}

/// An MXFP4 bank `[experts][rows][k]` under atom-laid ue8m0 scales.
fn mx_atoms(
    g: &mut Gpu,
    lcg: &mut Lcg,
    [experts, rows, k]: [usize; 3],
) -> ((u64, u64), [Plane; 2]) {
    let codes = g.up(&bytes(lcg, experts * rows * k / 2, |r| r as u8));
    let atoms = g.up(&bytes(lcg, atom_bytes(experts * rows, k, 32), ue8m0));
    let atom = atom_bytes(rows, k, 32) as u64;
    let along = atom / (rows as u64 / 128);
    let planes = [
        Plane::of(codes, (rows * k / 2) as u64).aligned((k / 2) as u64),
        Plane::of(atoms, atom).aligned(along),
    ];
    ((codes, atoms), planes)
}

#[test]
fn a_w4a8_grouped_select_lands_the_same_bits_through_any_table() {
    if !here() {
        return;
    }
    let mut g = Gpu::open();
    let mut lcg = Lcg::seeded(0x7c53);
    let (experts, tokens, top, n, k) = (8usize, 64usize, 4usize, 256usize, 512usize);
    let ((codes, atoms), planes) = mx_atoms(&mut g, &mut lcg, [experts, n, k]);
    let x = t(bf16(&mut g, &mut lcg, tokens * k), tokens, k, Bf16);
    let r = t(g.up(&routes(tokens, top, experts)), tokens, top, I32);
    let y = g.zeros(tokens * top * n * 2);
    let ctx = g.ctx().with_tuning(TC5);
    #[rustfmt::skip]
    let bank = Bank {
        codes, atoms, experts: experts as u32, n: n as u32, k: k as u32,
        elem: Elem::E2m1, gran: 32, alpha_e: 0, alpha_from: None,
    };
    same_bits(
        &mut g,
        "W4A8 tc5 select",
        experts,
        &planes,
        (y, tokens * top * n * 2),
        |table| {
            let mut yt = t(y, tokens * top, n, Bf16);
            let fmt = Format::E4m3 { gran: 128 };
            act_quant::select_with(&ctx, "test", fmt, x, r, None, bank, &mut yt, table)
                .expect("fires");
        },
    );
}

/// A routed MLP's fire: activations, routes, route weights, the shared row
/// and its gate.
struct Fire {
    x: kernels_cuda::tensor::Tensor,
    routes: kernels_cuda::tensor::Tensor,
    weights: kernels_cuda::tensor::Tensor,
    shared: kernels_cuda::tensor::Tensor,
    y: u64,
}

fn fire_of(g: &mut Gpu, lcg: &mut Lcg, [experts, tokens, top, hidden]: [usize; 4]) -> Fire {
    let weights = tab(tokens * top, |i| 0.05 + (i % 9) as f32 * 0.07);
    // Each token's routes distinct.
    let routes = tab(tokens * top, |r| ((r / top * 3 + r % top) % experts) as i32);
    Fire {
        x: t(bf16(g, lcg, tokens * hidden), tokens, hidden, Bf16),
        routes: t(g.up(&routes), tokens, top, I32),
        weights: t(g.up(&weights), tokens, top, F32),
        shared: t(bf16(g, lcg, tokens * hidden), tokens, hidden, Bf16),
        y: g.zeros(tokens * hidden * 2),
    }
}

#[test]
fn a_w4a8_routed_mlp_lands_the_same_bits_through_any_tables() {
    if !here() {
        return;
    }
    let mut g = Gpu::open();
    let mut lcg = Lcg::seeded(0x7c54);
    let (experts, tokens, top, hidden, inter) = (8usize, 64usize, 4usize, 512usize, 256usize);
    let (up, up_planes) = mx_atoms(&mut g, &mut lcg, [experts, 2 * inter, hidden]);
    let (down, down_planes) = mx_atoms(&mut g, &mut lcg, [experts, hidden, inter]);
    let f = fire_of(&mut g, &mut lcg, [experts, tokens, top, hidden]);
    let ctx = g.ctx().with_tuning(TC5);
    let bank = |(codes, atoms): (u64, u64), rows: usize, k: usize| GluBank::Mxfp4A8 {
        codes: t(codes, experts, rows * k / 2, U8),
        atoms: t(atoms, experts, rows * k / 32, U8),
    };
    let m = moe::RoutedMlp {
        x: f.x,
        up: bank(up, 2 * inter, hidden),
        glu: Glu::SwigluClamp { limit: 10.0 },
        down: bank(down, hidden, inter),
        routes: f.routes,
        weights: f.weights,
        inter: inter as u32,
    };
    same_bits_mlp(
        &mut g,
        "W4A8 routed MLP",
        experts,
        [&up_planes, &down_planes],
        (f.y, tokens * hidden * 2),
        |[u, d]| {
            let mut y = t(f.y, tokens, hidden, Bf16);
            let took = moe::experts(&ctx, m, &mut y, u, d, None, None).expect("fires");
            assert!(took, "the grouped routed MLP takes the fire");
        },
    );
}

#[test]
fn the_mega_moe_lands_the_same_bits_through_any_tables() {
    if !here() {
        return;
    }
    let mut g = Gpu::open();
    let mut lcg = Lcg::seeded(0x7c55);
    let (experts, tokens, top, hidden, inter) = (8usize, 32usize, 4usize, 2048usize, 768usize);
    for w4 in [true, false] {
        let f = fire_of(&mut g, &mut lcg, [experts, tokens, top, hidden]);
        // The gate-up bank's codes map in halves of `inter` rows: each
        // expert's copy a whole number of halves past the pool's base.
        let (up, down, planes): (GluBank, GluBank, [Vec<Plane>; 2]) = if w4 {
            let ((uc, ua), mut up) = mx_atoms(&mut g, &mut lcg, [experts, 2 * inter, hidden]);
            let ((dc, da), down) = mx_atoms(&mut g, &mut lcg, [experts, hidden, inter]);
            up[0] = up[0].aligned((inter * hidden / 2) as u64);
            let bank = |c: u64, a: u64, rows: usize, k: usize| GluBank::Mxfp4A8 {
                codes: t(c, experts, rows * k / 2, U8),
                atoms: t(a, experts, rows * k / 32, U8),
            };
            (
                bank(uc, ua, 2 * inter, hidden),
                bank(dc, da, hidden, inter),
                [up.to_vec(), down.to_vec()],
            )
        } else {
            let uc = bf16(&mut g, &mut lcg, experts * 2 * inter * hidden);
            let dc = bf16(&mut g, &mut lcg, experts * hidden * inter);
            let up =
                Plane::of(uc, (2 * inter * hidden * 2) as u64).aligned((inter * hidden * 2) as u64);
            let down = Plane::of(dc, (hidden * inter * 2) as u64).aligned((inter * 2) as u64);
            (
                GluBank::Bf16(t(uc, experts, 2 * inter * hidden, Bf16)),
                GluBank::Bf16(t(dc, experts, hidden * inter, Bf16)),
                [vec![up], vec![down]],
            )
        };
        let ctx = g.ctx().with_tuning(TC5);
        let glu = if w4 {
            Glu::SwigluClamp { limit: 10.0 }
        } else {
            Glu::Swiglu
        };
        #[rustfmt::skip]
        let m = moe::RoutedMlp {
            x: f.x, up, glu, down, routes: f.routes, weights: f.weights, inter: inter as u32,
        };
        let fold = if w4 {
            moe::Fold::First(Some(f.shared))
        } else {
            moe::Fold::Weighted
        };
        let what = format!("mega-MoE w4 {w4}");
        same_bits_mlp(
            &mut g,
            &what,
            experts,
            [&planes[0], &planes[1]],
            (f.y, tokens * hidden * 2),
            |tables| {
                let mut y = t(f.y, tokens, hidden, Bf16);
                let took = moe::mega(&ctx, m, fold, &mut y, tables).expect("fires");
                assert!(took, "{what}: the mega-MoE takes the fire");
            },
        );
    }
}
