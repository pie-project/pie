//! The routed matmuls a route at a time on the first Vulkan device, beside
//! the hand-written kernels, against an f64 host reference: a dense bank,
//! affine banks with and without an expert bias, and an MXFP4 bank, over
//! three tokens of two routes each (one route empty), the activation a row
//! a token and a row a route.

use kernels_slang::Tier;
use kernels_slang::linear::moe::select;
use kernels_vulkan::{Arg, ArgValue, Bank, Encode, Fire, Tensor};
use poem_ir::Dtype;

use crate::gpu::{Gpu, both, noise};

const HAND: &str = "moe/qmv_routed.slang";

fn st(t: Tensor) -> kernels_slang::Tensor {
    kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype)
}

fn t(b: u32, r: usize, w: usize, d: Dtype) -> Tensor {
    Tensor::new(b, r as u32, w as u32, d)
}

fn sb(b: Bank) -> kernels_slang::Bank {
    kernels_slang::Bank {
        codes: st(b.codes),
        scales: st(b.scales),
        biases: b.biases.map(st),
        group: b.group,
        bits: b.bits,
    }
}

fn words(seed: u64, n: usize) -> Vec<u32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s as u32
        })
        .collect()
}

fn e2m1(nibble: u32) -> f64 {
    let e = (nibble >> 1) & 3;
    let mant = f64::from(nibble & 1);
    let mag = if e == 0 {
        0.5 * mant
    } else {
        (1.0 + 0.5 * mant) * 2f64.powi(e as i32 - 1)
    };
    if nibble & 8 != 0 { -mag } else { mag }
}

const TOKENS: usize = 3;
const TOP_K: usize = 2;
const EXPERTS: usize = 3;
const K: usize = 128;
const N: usize = 20;
const ROUTES: [i32; 6] = [0, 2, 1, -1, 2, 0];

/// The reference: route `r`'s row against its expert's `weight(e, n, k)`,
/// plus `bias(e, n)`; an empty route lands zero.
fn reference(
    x: &[f32],
    per_route: bool,
    weight: &dyn Fn(usize, usize, usize) -> f64,
    bias: &dyn Fn(usize, usize) -> f64,
) -> Vec<f64> {
    (0..TOKENS * TOP_K * N)
        .map(|i| {
            let (route, n) = (i / N, i % N);
            let Ok(e) = usize::try_from(ROUTES[route]) else {
                return 0.0;
            };
            let row = if per_route { route } else { route / TOP_K };
            (0..K)
                .map(|k| f64::from(x[row * K + k]) * weight(e, n, k))
                .sum::<f64>()
                + bias(e, n)
        })
        .collect()
}

fn push(per_route: bool) -> [ArgValue; 5] {
    let (row, slot) = if per_route { (K * TOP_K, K) } else { (K, 0) };
    [
        (K as i32).arg(),
        (N as i32).arg(),
        (slot as i32).arg(),
        (row as i32).arg(),
        (TOP_K as i32).arg(),
    ]
}

#[test]
fn the_routed_selects_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let routes = t(gpu.up_i32(&ROUTES), TOKENS, TOP_K, Dtype::I32);
    let groups = [TOKENS as u32, (N as u32).div_ceil(8), TOP_K as u32];
    let out = TOKENS * TOP_K * N;
    for per_route in [false, true] {
        let rows = if per_route { TOKENS * TOP_K } else { TOKENS };
        let (x, xr) = gpu.up(&noise(1 + rows as u64, rows * K));
        let xt = t(x, rows, K, Dtype::Bf16);
        let p = push(per_route);

        // A dense bank.
        let (bank, br) = gpu.up(&noise(2, EXPERTS * N * K));
        let bt = t(bank, EXPERTS * N, K, Dtype::Bf16);
        let want = reference(
            &xr,
            per_route,
            &|e, n, k| f64::from(br[(e * N + n) * K + k]),
            &|_, _| 0.0,
        );
        both(
            &mut gpu,
            &format!(
                "select dense, a row a {}",
                if per_route { "route" } else { "token" }
            ),
            &|g| vec![(g.zeros(out), out, false)],
            std::slice::from_ref(&want),
            &|g, o| {
                let y = t(o[0].0, TOKENS * TOP_K, N, Dtype::Bf16);
                g.run(|c| {
                    c.fire(
                        Fire::at(HAND, "select_gemv_bfloat16")
                            .groups(groups)
                            .group([32, 8, 1]),
                        &[
                            bt.arg(),
                            xt.arg(),
                            y.arg_mut(),
                            routes.arg(),
                            p[0],
                            p[1],
                            p[2],
                            p[3],
                            p[4],
                        ],
                    )
                });
            },
            &|g, o| {
                let y = st(t(o[0].0, TOKENS * TOP_K, N, Dtype::Bf16));
                g.run_slang(|c| {
                    select::matmul_select(c, Tier::Baseline, st(xt), st(bt), st(routes), y)
                });
            },
        );

        // Affine banks.
        let (bias, biasr) = gpu.up(&noise(3, EXPERTS * N));
        let biast = t(bias, EXPERTS, N, Dtype::Bf16);
        for (group, bits) in [(64usize, 4u32), (32, 2)] {
            let per = 32 / bits as usize;
            let cw = words(u64::from(bits), EXPERTS * N * K / per);
            let w = gpu.up_u32(&cw);
            let gcount = EXPERTS * N * K / group;
            let (sc, scr) = gpu.up(&noise(4, gcount)
                .iter()
                .map(|v| v * 0.05)
                .collect::<Vec<_>>());
            let (zp, zpr) = gpu.up(&noise(5, gcount).iter().map(|v| v * 0.1).collect::<Vec<_>>());
            let bank = Bank {
                codes: t(w, EXPERTS * N, K * bits as usize / 32, Dtype::U32),
                scales: t(sc, EXPERTS * N, K / group, Dtype::Bf16),
                biases: Some(t(zp, EXPERTS * N, K / group, Dtype::Bf16)),
                group: group as u32,
                bits,
            };
            let weight = |e: usize, n: usize, k: usize| {
                let i = (e * N + n) * K + k;
                let code = (cw[i / per] >> ((k % per) as u32 * bits)) & ((1 << bits) - 1);
                let g = (e * N + n) * (K / group) + k / group;
                f64::from(scr[g]) * f64::from(code) + f64::from(zpr[g])
            };
            for biased in [false, true] {
                let want = reference(&xr, per_route, &weight, &|e, n| {
                    if biased {
                        f64::from(biasr[e * N + n])
                    } else {
                        0.0
                    }
                });
                let hand: &'static str = Box::leak(
                    format!(
                        "affine_qmv_routed{}_bfloat16_gs_{group}_b_{bits}",
                        if biased { "_bias" } else { "" }
                    )
                    .into_boxed_str(),
                );
                both(
                    &mut gpu,
                    &format!(
                        "select affine {group}/{bits}, biased {biased}, a row a route {per_route}"
                    ),
                    &|g| vec![(g.zeros(out), out, false)],
                    std::slice::from_ref(&want),
                    &|g, o| {
                        let y = t(o[0].0, TOKENS * TOP_K, N, Dtype::Bf16);
                        g.run(|c| {
                            let absent = c.absent()?;
                            c.fire(
                                Fire::at(HAND, hand).groups(groups).group([32, 8, 1]),
                                &[
                                    bank.codes.arg(),
                                    bank.scales.arg(),
                                    bank.biases.expect("affine").arg(),
                                    xt.arg(),
                                    y.arg_mut(),
                                    if biased { biast.arg() } else { absent },
                                    routes.arg(),
                                    p[0],
                                    p[1],
                                    p[2],
                                    p[3],
                                    p[4],
                                ],
                            )
                        });
                    },
                    &|g, o| {
                        let y = st(t(o[0].0, TOKENS * TOP_K, N, Dtype::Bf16));
                        g.run_slang(|c| {
                            if biased {
                                select::matmul_select_bias(
                                    c,
                                    Tier::Baseline,
                                    st(xt),
                                    sb(bank),
                                    st(biast),
                                    st(routes),
                                    y,
                                )
                            } else {
                                select::matmul_select_quant(
                                    c,
                                    Tier::Baseline,
                                    st(xt),
                                    sb(bank),
                                    st(routes),
                                    y,
                                )
                            }
                        });
                    },
                );
            }
        }

        // An MXFP4 bank: eight codes a word, an exponent byte a block.
        let cw = words(9, EXPERTS * N * K / 8);
        let w = gpu.up_u32(&cw);
        let blocks = EXPERTS * N * K / 32;
        let exps: Vec<u8> = words(10, blocks)
            .iter()
            .map(|v| 120 + (v % 10) as u8)
            .collect();
        let mut packed = exps.clone();
        packed.resize(blocks.next_multiple_of(4), 0);
        let sc = gpu.up_u32(
            &packed
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect::<Vec<_>>(),
        );
        let bank = Bank {
            codes: t(w, EXPERTS * N, K / 8, Dtype::U32),
            scales: t(sc, EXPERTS * N, K / 32, Dtype::U8),
            biases: None,
            group: 32,
            bits: 4,
        };
        let weight = |e: usize, n: usize, k: usize| {
            let i = (e * N + n) * K + k;
            let nibble = (cw[i / 8] >> ((k % 8) as u32 * 4)) & 0xf;
            e2m1(nibble) * 2f64.powi(i32::from(exps[i / 32]) - 127)
        };
        let want = reference(&xr, per_route, &weight, &|_, _| 0.0);
        both(
            &mut gpu,
            &format!("select mxfp4, a row a route {per_route}"),
            &|g| vec![(g.zeros(out), out, false)],
            std::slice::from_ref(&want),
            &|g, o| {
                let y = t(o[0].0, TOKENS * TOP_K, N, Dtype::Bf16);
                g.run(|c| {
                    let absent = c.absent()?;
                    c.fire(
                        Fire::at(HAND, "mxfp4_qmv_routed_bfloat16_gs_32_b_4")
                            .groups(groups)
                            .group([32, 8, 1]),
                        &[
                            bank.codes.arg(),
                            bank.scales.arg(),
                            absent,
                            xt.arg(),
                            y.arg_mut(),
                            absent,
                            routes.arg(),
                            p[0],
                            p[1],
                            p[2],
                            p[3],
                            p[4],
                        ],
                    )
                });
            },
            &|g, o| {
                let y = st(t(o[0].0, TOKENS * TOP_K, N, Dtype::Bf16));
                g.run_slang(|c| {
                    select::matmul_select_quant(c, Tier::Baseline, st(xt), sb(bank), st(routes), y)
                });
            },
        );
    }
}

/// The batched routed matmul on the first Vulkan device: routes sorted into
/// expert tiles, gathered, multiplied a tile an expert and scattered back,
/// beside the hand-written route, against an f64 host reference, for an
/// affine bank with an expert bias and an MXFP4 bank without, at the tile
/// rows the device's profile picks (and the cooperative-matrix tier where
/// the device has one).
mod batched {
    use kernels_slang::Tier;
    use kernels_slang::linear::moe::select::{self, Batch};
    use kernels_vulkan::linear::moe::{self, RoutedScratch};
    use kernels_vulkan::{Bank, Tensor};
    use poem_ir::Dtype;

    use crate::gpu::{Gpu, both, close, noise};

    fn st(t: Tensor) -> kernels_slang::Tensor {
        kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype)
    }

    fn t(b: u32, r: usize, w: usize, d: Dtype) -> Tensor {
        Tensor::new(b, r as u32, w as u32, d)
    }

    fn words(seed: u64, n: usize) -> Vec<u32> {
        let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s as u32
            })
            .collect()
    }

    fn e2m1(nibble: u32) -> f64 {
        let e = (nibble >> 1) & 3;
        let mant = f64::from(nibble & 1);
        let mag = if e == 0 {
            0.5 * mant
        } else {
            (1.0 + 0.5 * mant) * 2f64.powi(e as i32 - 1)
        };
        if nibble & 8 != 0 { -mag } else { mag }
    }

    /// A bank, its expert bias, and its weights to f64.
    type Case<'a> = (
        &'a str,
        Bank,
        Option<Tensor>,
        &'a dyn Fn(usize, usize, usize) -> f64,
    );

    const TOKENS: usize = 40;
    const TOP_K: usize = 2;
    const EXPERTS: usize = 3;
    const K: usize = 64;
    const N: usize = 40;

    #[test]
    fn the_batched_route_answers_the_reference() {
        let Some(mut gpu) = Gpu::open() else {
            eprintln!("not asked: no device");
            return;
        };
        let chip = gpu.device.chip();
        let (profile, _) = engine_vulkan::chip::resolve(&chip, None);
        let tuning = kernels_vulkan::Tuning::new(profile, kernels_vulkan::tuning::Limits::ASSUMED);
        let pairs = TOKENS * TOP_K;
        let tile = moe::tile_rows(pairs as u32, EXPERTS as u32, &tuning);
        let padded = moe::sorted_rows(pairs as u32, EXPERTS as u32, &tuning) as usize;
        assert!(
            tile > 1,
            "the profile sorts {pairs} routes over {EXPERTS} experts"
        );
        let ids: Vec<i32> = (0..pairs)
            .map(|i| {
                if i % 7 == 3 {
                    -1
                } else {
                    ((i * 5 + i / 3) % EXPERTS) as i32
                }
            })
            .collect();
        let routes = t(gpu.up_i32(&ids), TOKENS, TOP_K, Dtype::I32);
        let (x, xr) = gpu.up(&noise(1, TOKENS * K));
        let xt = t(x, TOKENS, K, Dtype::Bf16);
        let (bias, biasr) = gpu.up(&noise(2, EXPERTS * N));
        let biast = t(bias, EXPERTS, N, Dtype::Bf16);

        // An affine 64/4 bank and an MXFP4 one.
        let cw = words(3, EXPERTS * N * K / 8);
        let w4 = gpu.up_u32(&cw);
        let groups = EXPERTS * N * K / 64;
        let (sc, scr) = gpu.up(&noise(4, groups)
            .iter()
            .map(|v| v * 0.05)
            .collect::<Vec<_>>());
        let (zp, zpr) = gpu.up(&noise(5, groups).iter().map(|v| v * 0.1).collect::<Vec<_>>());
        let affine = Bank {
            codes: t(w4, EXPERTS * N, K / 8, Dtype::U32),
            scales: t(sc, EXPERTS * N, K / 64, Dtype::Bf16),
            biases: Some(t(zp, EXPERTS * N, K / 64, Dtype::Bf16)),
            group: 64,
            bits: 4,
        };
        let affine_w = |e: usize, n: usize, k: usize| {
            let i = (e * N + n) * K + k;
            let code = (cw[i / 8] >> ((k % 8) as u32 * 4)) & 0xf;
            let g = (e * N + n) * (K / 64) + k / 64;
            f64::from(scr[g]) * f64::from(code) + f64::from(zpr[g])
        };
        let blocks = EXPERTS * N * K / 32;
        let exps: Vec<u8> = words(6, blocks)
            .iter()
            .map(|v| 122 + (v % 6) as u8)
            .collect();
        let mut packed = exps.clone();
        packed.resize(blocks.next_multiple_of(4), 0);
        let mx_sc = gpu.up_u32(
            &packed
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect::<Vec<_>>(),
        );
        let mxfp4 = Bank {
            codes: t(w4, EXPERTS * N, K / 8, Dtype::U32),
            scales: t(mx_sc, EXPERTS * N, K / 32, Dtype::U8),
            biases: None,
            group: 32,
            bits: 4,
        };
        let mxfp4_w = |e: usize, n: usize, k: usize| {
            let i = (e * N + n) * K + k;
            e2m1((cw[i / 8] >> ((k % 8) as u32 * 4)) & 0xf)
                * 2f64.powi(i32::from(exps[i / 32]) - 127)
        };

        let scratch = RoutedScratch {
            perm: t(gpu.zeros(2 * padded), padded, 1, Dtype::I32),
            row_expert: t(gpu.zeros(2 * padded), padded, 1, Dtype::I32),
            tile_expert: t(gpu.zeros(2 * padded), padded, 1, Dtype::I32),
            inv: t(gpu.zeros(2 * pairs), pairs, 1, Dtype::I32),
            x: t(gpu.zeros(padded * K), padded, K, Dtype::Bf16),
            y: t(gpu.zeros(padded * N), padded, N, Dtype::Bf16),
        };
        let sscratch = select::RoutedScratch {
            perm: st(scratch.perm),
            row_expert: st(scratch.row_expert),
            tile_expert: st(scratch.tile_expert),
            inv: st(scratch.inv),
            x: st(scratch.x),
            y: st(scratch.y),
        };
        let mut tiers = vec![Tier::Baseline];
        if chip.coopmat {
            tiers.push(Tier::Coopmat);
        }
        let cases: [Case<'_>; 2] = [
            ("affine 64/4 biased", affine, Some(biast), &affine_w),
            ("mxfp4", mxfp4, None, &mxfp4_w),
        ];
        for (what, bank, bias, weight) in cases {
            let coopmat = |tier: Tier| tier == Tier::Coopmat;
            for &tier in &tiers {
                let half = |v: f64| {
                    if v == 0.0 || !coopmat(tier) {
                        return v;
                    }
                    let quantum = 2f64.powi((v.abs().log2().floor() as i32).max(-14) - 10);
                    (v / quantum).round_ties_even() * quantum
                };
                let want: Vec<f64> = (0..pairs * N)
                    .map(|i| {
                        let (route, n) = (i / N, i % N);
                        let Ok(e) = usize::try_from(ids[route]) else {
                            return 0.0;
                        };
                        let row = route / TOP_K;
                        (0..K)
                            .map(|k| half(f64::from(xr[row * K + k])) * half(weight(e, n, k)))
                            .sum::<f64>()
                            + bias.map_or(0.0, |_| f64::from(biasr[e * N + n]))
                    })
                    .collect();
                let label = format!("batched {what} at {}, {tile}-row tiles", tier.name());
                let slang = |g: &Gpu, o: &[crate::gpu::Out]| {
                    let y = st(t(o[0].0, pairs, N, Dtype::Bf16));
                    g.run_slang(|c| {
                        let sbank = kernels_slang::Bank {
                            codes: st(bank.codes),
                            scales: st(bank.scales),
                            biases: bank.biases.map(st),
                            group: bank.group,
                            bits: bank.bits,
                        };
                        let ran = select::matmul_select_batched(
                            c,
                            "linear.moe_select",
                            st(xt),
                            sbank,
                            bias.map(st),
                            st(routes),
                            EXPERTS as u32,
                            sscratch,
                            y,
                            Batch {
                                tile,
                                padded: padded as u32,
                                tier,
                                sort_lanes: tuning.sort_lanes(),
                            },
                        )?;
                        assert!(ran, "the Slang route serves the shape");
                        Ok(())
                    });
                };
                // The hand-written route runs the tier the profile picks; the
                // other tier is checked against the reference alone.
                let hand_tier = if tuning.moe_coopmat() {
                    Tier::Coopmat
                } else {
                    Tier::Baseline
                };
                if tier != hand_tier {
                    let y = gpu.zeros(pairs * N);
                    slang(&gpu, &[(y, pairs * N, false)]);
                    close(&gpu.down(y, pairs * N), &want, &format!("slang {label}"));
                    continue;
                }
                both(
                    &mut gpu,
                    &label,
                    &|g| vec![(g.zeros(pairs * N), pairs * N, false)],
                    std::slice::from_ref(&want),
                    &|g, o| {
                        let y = t(o[0].0, pairs, N, Dtype::Bf16);
                        g.run(|c| {
                            let ran = moe::matmul_select_batched(
                                c,
                                "linear.moe_select",
                                xt,
                                bank,
                                bias,
                                routes,
                                EXPERTS as u32,
                                scratch,
                                y,
                                &tuning,
                            )?;
                            assert!(ran, "the hand-written route serves the shape");
                            Ok(())
                        });
                    },
                    &slang,
                );
            }
        }
    }
}
