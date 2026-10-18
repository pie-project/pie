//! The gated activations past swiglu: GEGLU (split and packed), GELU, SiTU
//! and the clamped swiglus, against host references, over a ragged width,
//! a fan-out and a staged window.

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, close_bf16, gelu_tanh, round_bf16, sigmoid, values};
use dtype::Dtype;
use kernels_rocm::linear::mlp;

/// Bf16 values in [-scale, scale).
fn bf16s(n: u32, seed: u64, scale: f32) -> Vec<f32> {
    values(n as usize, seed, scale)
        .into_iter()
        .map(round_bf16)
        .collect()
}

/// The plane rows a launch over `rows` launch rows (`fan` a stated row)
/// lands under the window `{live, offset}`, or every row with none.
fn landed(rows: u32, fan: u32, win: Option<(u32, u32)>) -> Vec<bool> {
    (0..rows)
        .map(|r| match win {
            None => true,
            Some((live, offset)) => (offset * fan..(offset + live) * fan).contains(&r),
        })
        .collect()
}

/// Rows the window lands match `want`; the rest stay zero.
fn check_rows(what: &str, have: &[f32], want: &[f32], width: usize, land: &[bool]) {
    for (r, &on) in land.iter().enumerate() {
        let (h, w) = (
            &have[r * width..(r + 1) * width],
            &want[r * width..(r + 1) * width],
        );
        if on {
            close_bf16(&format!("{what} row {r}"), h, w, 1.0);
        } else {
            assert!(
                h.iter().all(|v| *v == 0.0),
                "{what}: row {r} is outside the window"
            );
        }
    }
}

/// (rows, width, fan, window): a ragged width, a fan-out of two, and a
/// window of stated rows 1..3 over a fan of two.
/// A launch: rows, width, fan-out, and the window {live, offset} if staged.
type Shape = (u32, u32, u32, Option<(u32, u32)>);

const SHAPES: [Shape; 3] = [
    (5, 1000, 1, None),
    (6, 384, 2, None),
    (8, 260, 2, Some((2, 1))),
];

#[test]
fn linear_glu_packed() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    type Act = fn(f32, f32) -> f32;
    let (limit, alpha, beta, cap) = (7.0f32, 1.702f32, 2.5f32, 3.0f32);
    let acts: [(&str, Act); 6] = [
        ("geglu_tanh_packed", |g, u| gelu_tanh(g) * u),
        ("situ", |g, u| 2.5 * (g / 2.5).tanh() * sigmoid(g) * u),
        ("situ_capped", |g, u| {
            2.5 * (g / 2.5).tanh() * sigmoid(g) * (3.0 * (u / 3.0).tanh())
        }),
        ("swiglu_clamp", |g, u| {
            let (g, u) = (g.min(7.0), u.clamp(-7.0, 7.0));
            g * sigmoid(g) * u
        }),
        ("swiglu_clamp_alpha", |g, u| {
            let (g, u) = (g.min(7.0), u.clamp(-7.0, 7.0));
            (u + 1.0) * g * sigmoid(1.702 * g)
        }),
        ("swiglu", |g, u| g * sigmoid(g) * u),
    ];
    for (rows, width, fan, win) in SHAPES {
        // Past the clamp's limit on both sides.
        let packed = bf16s(rows * 2 * width, 1, 9.0);
        let x = gpu.tensor(&packed, rows, 2 * width, Dtype::Bf16);
        for (name, act) in acts {
            let mut y = gpu.zeros(rows, width, Dtype::Bf16);
            let stage = win.map(|(live, offset)| gpu.ints(&[live as i32, offset as i32, 0, 0]));
            if let Some(s) = &stage {
                ctx.arm_stage(s.ptr);
            }
            match name {
                "geglu_tanh_packed" => mlp::geglu_tanh_packed(&ctx, x, width, fan, &mut y),
                "situ" => mlp::situ(&ctx, x, width, fan, beta, None, &mut y),
                "situ_capped" => mlp::situ(&ctx, x, width, fan, beta, Some(cap), &mut y),
                "swiglu_clamp" => mlp::swiglu_clamp(&ctx, x, width, fan, limit, &mut y),
                "swiglu_clamp_alpha" => {
                    mlp::swiglu_clamp_alpha(&ctx, x, width, fan, limit, alpha, &mut y)
                }
                _ => mlp::swiglu(&ctx, x, width, fan, &mut y),
            }
            .unwrap();
            ctx.disarm_stage();
            let w = width as usize;
            let want: Vec<f32> = (0..(rows * width) as usize)
                .map(|i| {
                    let (r, c) = (i / w, i % w);
                    act(packed[r * 2 * w + c], packed[r * 2 * w + w + c])
                })
                .collect();
            check_rows(
                &format!("{name} {rows}x{width} fan {fan} {win:?}"),
                &gpu.read(&y),
                &want,
                w,
                &landed(rows, fan, win),
            );
        }
    }
}

#[test]
fn linear_glu_flat() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    for (rows, width, fan, win) in SHAPES {
        let n = rows * width;
        let (gs, us) = (bf16s(n, 2, 4.0), bf16s(n, 3, 2.0));
        let gate = gpu.tensor(&gs, rows, width, Dtype::Bf16);
        let up = gpu.tensor(&us, rows, width, Dtype::Bf16);
        let stage = win.map(|(live, offset)| gpu.ints(&[live as i32, offset as i32, 0, 0]));
        let land = landed(rows, fan, win);
        let what = format!("{rows}x{width} fan {fan} {win:?}");

        let mut y = gpu.zeros(rows, width, Dtype::Bf16);
        if let Some(s) = &stage {
            ctx.arm_stage(s.ptr);
        }
        mlp::geglu_tanh(&ctx, gate, up, fan, &mut y).unwrap();
        ctx.disarm_stage();
        let want: Vec<f32> = gs
            .iter()
            .zip(&us)
            .map(|(&g, &u)| gelu_tanh(g) * u)
            .collect();
        check_rows(
            &format!("geglu_tanh {what}"),
            &gpu.read(&y),
            &want,
            width as usize,
            &land,
        );

        let mut y = gpu.zeros(rows, width, Dtype::Bf16);
        if let Some(s) = &stage {
            ctx.arm_stage(s.ptr);
        }
        mlp::gelu_tanh(&ctx, gate, fan, &mut y).unwrap();
        ctx.disarm_stage();
        let want: Vec<f32> = gs.iter().map(|&g| gelu_tanh(g)).collect();
        check_rows(
            &format!("gelu_tanh {what}"),
            &gpu.read(&y),
            &want,
            width as usize,
            &land,
        );
    }

    // The f32 instantiation of GELU.
    let xs = values(3000, 4, 4.0);
    let x = gpu.tensor(&xs, 3, 1000, Dtype::F32);
    let mut y = gpu.zeros(3, 1000, Dtype::F32);
    mlp::gelu_tanh(&ctx, x, 1, &mut y).unwrap();
    let have = gpu.read(&y);
    for (i, (&h, &v)) in have.iter().zip(&xs).enumerate() {
        let want = gelu_tanh(v);
        assert!(
            (h - want).abs() <= 1e-5 * want.abs().max(1.0),
            "gelu f32[{i}] = {h}, want {want}"
        );
    }
}

/// The decode projection with its epilogue fused: GEGLU over a packed
/// `[gate | up]` bank and a soft-capped head, against the unfused host
/// arithmetic (each product rounded to bf16 before the epilogue).
#[test]
fn linear_skinny_epilogues() {
    use kernels_rocm::linear::skinny::{self, Epilogue};
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    // The diagonal kernel's shapes, then banks of at least 8192 rows (whole
    // products on the matrix core): a K split across wavefronts with the
    // activation read from L2, and one staged in LDS.
    for (n, k, m) in [
        (1000u32, 1024u32, 1u32),
        (1000, 1024, 3),
        (1000, 1024, 8),
        (4200, 12800, 3),
        (4200, 12800, 8),
        (8200, 2048, 4),
        (8200, 2048, 7),
    ] {
        let x = bf16s(m * k, 5, 1.0);
        let w = bf16s(2 * n * k, 6, 0.08);
        let at = gpu.tensor(&x, m, k, Dtype::Bf16);
        let wt = gpu.tensor(&w, 2 * n, k, Dtype::Bf16);
        let dot = |r: usize, row: usize| -> f32 {
            let k = k as usize;
            let v: f64 = (0..k)
                .map(|i| f64::from(x[r * k + i]) * f64::from(w[row * k + i]))
                .sum();
            round_bf16(v as f32)
        };
        for epilogue in [Epilogue::Geglu, Epilogue::Softcap(2.0), Epilogue::Store] {
            // The wide bank's GLU halves would double its reference cost.
            if n == 8200 && epilogue == Epilogue::Geglu {
                continue;
            }
            // Past two rows a long K over a short bank is left unfused.
            let reread = m > 2 && m * k * 2 > 64 << 10 && n < 8192 && epilogue != Epilogue::Geglu;
            assert_eq!(
                skinny::covers(m as i32, n as i32, k as i32, epilogue, 64),
                !reread
            );
            if reread {
                continue;
            }
            let y = gpu.zeros(m, n, Dtype::Bf16);
            skinny::skinny_bf16(
                &ctx, wt.ptr, at.ptr, y.ptr, m as i32, n as i32, k as i32, epilogue,
            )
            .unwrap();
            let want: Vec<f32> = (0..(m * n) as usize)
                .map(|i| {
                    let (r, c) = (i / n as usize, i % n as usize);
                    match epilogue {
                        Epilogue::Geglu => gelu_tanh(dot(r, c)) * dot(r, n as usize + c),
                        Epilogue::Softcap(cap) => cap * (dot(r, c) / cap).tanh(),
                        Epilogue::Store => dot(r, c),
                    }
                })
                .collect();
            close_bf16(
                &format!("skinny {epilogue:?} m={m} n={n} k={k}"),
                &gpu.read(&y),
                &want,
                2.0,
            );
        }
    }
    // Past its rows the kernel declines, and the engine takes the unfused path.
    assert!(!skinny::covers(9, 1000, 1024, Epilogue::Geglu, 64));
}

/// The split swiglu-clamp: the gate clamped from above, the up half both
/// ways, against the host.
#[test]
fn linear_swiglu_clamp_split() {
    let Some(gpu) = Gpu::open() else { return };
    let ctx = gpu.ctx();
    let (rows, width, limit) = (5u32, 2304u32, 1.5f32);
    let g = bf16s(rows * width, 31, 3.0);
    let u = bf16s(rows * width, 32, 3.0);
    let gt = gpu.tensor(&g, rows, width, Dtype::Bf16);
    let ut = gpu.tensor(&u, rows, width, Dtype::Bf16);
    let mut y = gpu.zeros(rows, width, Dtype::Bf16);
    mlp::swiglu_clamp_split(&ctx, gt, ut, limit, &mut y).unwrap();
    let want: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(&g, &u)| {
            let g = g.min(limit);
            let u = u.clamp(-limit, limit);
            g * sigmoid(g) * u
        })
        .collect();
    close_bf16("swiglu_clamp_split", &gpu.read(&y), &want, 1.0);
}
