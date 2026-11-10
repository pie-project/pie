//! The routed kernels through an expert table (the weight cache's experts)
//! against the same kernels over the contiguous bank: an expert's planes
//! laid back to back in a slot of a two-segment pool (one expert in each
//! segment) or staged in a seat of the stage, the layer's atoms after
//! another layer's. Every scheme the engine serves (dense bf16, affine
//! 64/8, 64/4, 32/2, MXFP4; with and without an expert bias), a route at a
//! time (at the subgroup tier too) and in sorted tiles, and the expert bias sum: the same bits as the
//! contiguous bank, and a use counted a route.

use kernels_slang::linear::moe::combine;
use kernels_slang::linear::moe::select::{self, Batch, ExpertTable, RoutedScratch};
use kernels_slang::{ABSENT, Bank, Segments, Tensor, Tier};
use poem_ir::Dtype;

use crate::gpu::{Gpu, noise};

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

fn bytes_of(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| crate::gpu::bf16(*v).to_le_bytes())
        .collect()
}

fn up_bytes(gpu: &mut Gpu, bytes: &[u8]) -> u32 {
    let mut padded = bytes.to_vec();
    padded.resize(bytes.len().next_multiple_of(4).max(4), 0);
    gpu.up_u32(
        &padded
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect::<Vec<_>>(),
    )
}

const EXPERTS: usize = 3;
/// The layer's first atom: a layer of three experts before it.
const FIRST: u32 = 3;

/// The expert table of `planes` (each every expert's bytes, expert-major),
/// four origins `[codes or dense, scales, zero points, bias]` by `roles`:
/// expert 0 in segment 0's second slot, expert 2 in segment 1's first,
/// expert 1 staged in the stage's second seat. Returns the table and its
/// hit counter.
fn table(gpu: &mut Gpu, planes: &[(usize, &[u8])]) -> (ExpertTable, u32) {
    let mut origins = [0u32; 4];
    let mut at = 0usize;
    let mut laid = Vec::new();
    for &(role, plane) in planes {
        let per = plane.len() / EXPERTS;
        origins[role] = (at / 4) as u32;
        laid.push((at, per, plane));
        at = (at + per).next_multiple_of(4);
    }
    let atom = at.next_multiple_of(256);
    let image = |e: usize| {
        let mut out = vec![0u8; atom];
        for &(at, per, plane) in &laid {
            out[at..at + per].copy_from_slice(&plane[e * per..(e + 1) * per]);
        }
        out
    };
    // Two slots a segment; unit = slot.
    let mut seg0 = vec![0xa5u8; 2 * atom];
    seg0[atom..].copy_from_slice(&image(0));
    let mut seg1 = vec![0x5au8; 2 * atom];
    seg1[..atom].copy_from_slice(&image(2));
    let mut stage = vec![0x3cu8; 2 * atom];
    stage[atom..].copy_from_slice(&image(1));
    let none = u32::MAX;
    let units = [none, none, none, 1, none, 2];
    let staged = [0, 0, 0, none, 1, none];
    let seg0 = up_bytes(gpu, &seg0);
    let seg1 = up_bytes(gpu, &seg1);
    let stage_h = up_bytes(gpu, &stage);
    let hits = gpu.up_u32(&[0; 6]);
    let atom_words = (atom / 4) as u32;
    (
        ExpertTable {
            units: t(gpu.up_u32(&units), 6, 1, Dtype::U32),
            staged: t(gpu.up_u32(&staged), 6, 1, Dtype::U32),
            stage: t(stage_h, stage.len() / 4, 1, Dtype::U32),
            hits: t(hits, 6, 1, Dtype::U32),
            pool: Segments {
                bufs: [seg0, seg1, ABSENT, ABSENT],
                words: 2 * atom_words,
            },
            unit_words: atom_words,
            atom_words,
            atom_first: FIRST,
            origins,
            count: true,
        },
        hits,
    )
}

/// The bits `fire` lands in a fresh `n`-value bf16 output.
fn landed_bits(gpu: &mut Gpu, n: usize, fire: &dyn Fn(&Gpu, Tensor)) -> Vec<u32> {
    let y = gpu.zeros(n);
    fire(gpu, t(y, n, 1, Dtype::Bf16));
    gpu.down(y, n).iter().map(|v| v.to_bits()).collect()
}

fn same(what: &str, got: &[u32], want: &[u32]) {
    let differ = got.iter().zip(want).filter(|(a, b)| a != b).count();
    assert!(
        differ == 0 && got.len() == want.len(),
        "{what}: {differ} of {} values differ",
        want.len()
    );
    assert!(want.iter().any(|&b| b != 0), "{what}: something landed");
    eprintln!("{what}: tabled = contiguous, bit for bit");
}

/// The uses a run counted, expert by expert of the layer.
fn uses(gpu: &Gpu, hits: u32) -> Vec<u32> {
    let raw = gpu.down_f32(hits, 6);
    raw[FIRST as usize..].iter().map(|v| v.to_bits()).collect()
}

/// A scheme's bank: its planes by role and the bank the contiguous kernels
/// read.
struct Scheme {
    name: String,
    planes: Vec<(usize, Vec<u8>)>,
    bank: Option<Bank>,
    dense: Option<Tensor>,
}

fn schemes(gpu: &mut Gpu, n: usize, k: usize) -> Vec<Scheme> {
    let mut out = Vec::new();
    let dense: Vec<f32> = noise(2, EXPERTS * n * k);
    let bytes = bf16_bytes(&dense);
    let h = up_bytes(gpu, &bytes);
    out.push(Scheme {
        name: "dense".into(),
        planes: vec![(0, bytes)],
        bank: None,
        dense: Some(t(h, EXPERTS * n, k, Dtype::Bf16)),
    });
    for (group, bits) in [(64usize, 8u32), (64, 4), (32, 2)] {
        let per = 32 / bits as usize;
        let codes = bytes_of(&words(u64::from(bits), EXPERTS * n * k / per));
        let gcount = EXPERTS * n * k / group;
        let scales = bf16_bytes(
            &noise(4, gcount)
                .iter()
                .map(|v| v * 0.05)
                .collect::<Vec<_>>(),
        );
        let zeros = bf16_bytes(&noise(5, gcount).iter().map(|v| v * 0.1).collect::<Vec<_>>());
        let bank = Bank {
            codes: t(
                up_bytes(gpu, &codes),
                EXPERTS * n,
                k * bits as usize / 32,
                Dtype::U32,
            ),
            scales: t(up_bytes(gpu, &scales), EXPERTS * n, k / group, Dtype::Bf16),
            biases: Some(t(
                up_bytes(gpu, &zeros),
                EXPERTS * n,
                k / group,
                Dtype::Bf16,
            )),
            group: group as u32,
            bits,
        };
        out.push(Scheme {
            name: format!("affine {group}/{bits}"),
            planes: vec![(0, codes), (1, scales), (2, zeros)],
            bank: Some(bank),
            dense: None,
        });
    }
    let codes = bytes_of(&words(9, EXPERTS * n * k / 8));
    let blocks = EXPERTS * n * k / 32;
    let exps: Vec<u8> = words(10, blocks)
        .iter()
        .map(|v| 120 + (v % 10) as u8)
        .collect();
    let bank = Bank {
        codes: t(up_bytes(gpu, &codes), EXPERTS * n, k / 8, Dtype::U32),
        scales: t(up_bytes(gpu, &exps), EXPERTS * n, k / 32, Dtype::U8),
        biases: None,
        group: 32,
        bits: 4,
    };
    out.push(Scheme {
        name: "mxfp4".into(),
        planes: vec![(0, codes), (1, exps)],
        bank: Some(bank),
        dense: None,
    });
    out
}

#[test]
fn a_tabled_select_lands_the_contiguous_banks_bits() {
    const TOKENS: usize = 3;
    const TOP_K: usize = 2;
    const K: usize = 128;
    const N: usize = 20;
    const ROUTES: [i32; 6] = [0, 2, 1, -1, 2, 0];
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let routes = t(gpu.up_i32(&ROUTES), TOKENS, TOP_K, Dtype::I32);
    let bias_bytes = bf16_bytes(&noise(3, EXPERTS * N));
    let bias = t(up_bytes(&mut gpu, &bias_bytes), EXPERTS, N, Dtype::Bf16);
    let out = TOKENS * TOP_K * N;
    for scheme in schemes(&mut gpu, N, K) {
        for per_route in [false, true] {
            let rows = if per_route { TOKENS * TOP_K } else { TOKENS };
            let (x, _) = gpu.up(&noise(1 + rows as u64, rows * K));
            let x = t(x, rows, K, Dtype::Bf16);
            for tier in [Tier::Baseline, Tier::Subgroup] {
                for biased in [false, true] {
                    if scheme.dense.is_some() && biased {
                        continue;
                    }
                    let mut planes: Vec<(usize, &[u8])> = scheme
                        .planes
                        .iter()
                        .map(|(r, b)| (*r, b.as_slice()))
                        .collect();
                    if biased {
                        planes.push((3, &bias_bytes));
                    }
                    let (tab, hits) = table(&mut gpu, &planes);
                    let what = format!(
                        "select {}{}, a row a {}, {}",
                        scheme.name,
                        if biased { " biased" } else { "" },
                        if per_route { "route" } else { "token" },
                        tier.name()
                    );
                    let want = landed_bits(&mut gpu, out, &|g, y| {
                        let y = Tensor::new(y.buf, (TOKENS * TOP_K) as u32, N as u32, y.dtype);
                        g.run(|c| match (scheme.dense, scheme.bank) {
                            (Some(bank), _) => select::matmul_select(c, tier, x, bank, routes, y),
                            (None, Some(bank)) if biased => {
                                select::matmul_select_bias(c, tier, x, bank, bias, routes, y)
                            }
                            (None, Some(bank)) => {
                                select::matmul_select_quant(c, tier, x, bank, routes, y)
                            }
                            (None, None) => unreachable!(),
                        });
                    });
                    let got = landed_bits(&mut gpu, out, &|g, y| {
                        let y = Tensor::new(y.buf, (TOKENS * TOP_K) as u32, N as u32, y.dtype);
                        g.run(|c| match (scheme.dense, scheme.bank) {
                            (Some(_), _) => {
                                select::matmul_select_tabled(c, tier, x, &tab, routes, y)
                            }
                            (None, Some(bank)) => select::matmul_select_quant_tabled(
                                c, tier, x, bank, biased, &tab, routes, y,
                            ),
                            (None, None) => unreachable!(),
                        });
                    });
                    same(&what, &got, &want);
                    assert_eq!(uses(&gpu, hits), vec![2, 1, 2], "{what}: a use a route");
                }
            }
        }
    }

    // The expert bias sum.
    let (x, _) = gpu.up(&noise(7, TOKENS * N));
    let x = t(x, TOKENS, N, Dtype::Bf16);
    let weights = t(
        gpu.up_f32(&noise(8, TOKENS * TOP_K)),
        TOKENS,
        TOP_K,
        Dtype::F32,
    );
    let (tab, _) = table(&mut gpu, &[(3, &bias_bytes)]);
    let want = landed_bits(&mut gpu, TOKENS * N, &|g, y| {
        let y = Tensor::new(y.buf, TOKENS as u32, N as u32, y.dtype);
        g.run(|c| combine::bias_sum(c, x, bias, routes, weights, y));
    });
    let got = landed_bits(&mut gpu, TOKENS * N, &|g, y| {
        let y = Tensor::new(y.buf, TOKENS as u32, N as u32, y.dtype);
        g.run(|c| combine::bias_sum_tabled(c, x, &tab, routes, weights, y));
    });
    same("bias sum", &got, &want);
}

#[test]
fn a_tabled_batched_route_lands_the_contiguous_banks_bits() {
    const TOKENS: usize = 40;
    const TOP_K: usize = 2;
    const K: usize = 64;
    const N: usize = 40;
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let chip = gpu.device.chip();
    let (profile, _) = engine_wgpu::chip::resolve(&chip, None);
    let tuning =
        engine_wgpu::tuning::Tuning::new(profile, gpu.device.cores(), gpu.device.storage_buffers());
    let pairs = TOKENS * TOP_K;
    let tile = tuning.moe_tile_rows(pairs as u32, EXPERTS as u32);
    let padded = tuning.moe_sorted_rows(pairs as u32, EXPERTS as u32) as usize;
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
    let (x, _) = gpu.up(&noise(1, TOKENS * K));
    let x = t(x, TOKENS, K, Dtype::Bf16);
    let bias_bytes = bf16_bytes(&noise(2, EXPERTS * N));
    let bias = t(up_bytes(&mut gpu, &bias_bytes), EXPERTS, N, Dtype::Bf16);
    let scratch = RoutedScratch {
        perm: t(gpu.zeros(2 * padded), padded, 1, Dtype::I32),
        row_expert: t(gpu.zeros(2 * padded), padded, 1, Dtype::I32),
        tile_expert: t(gpu.zeros(2 * padded), padded, 1, Dtype::I32),
        inv: t(gpu.zeros(2 * pairs), pairs, 1, Dtype::I32),
        x: t(gpu.zeros(padded * K), padded, K, Dtype::Bf16),
        y: t(gpu.zeros(padded * N), padded, N, Dtype::Bf16),
    };
    let tiers = vec![Tier::Baseline];
    let mut ran_any = false;
    for scheme in schemes(&mut gpu, N, K) {
        let Some(bank) = scheme.bank else {
            continue;
        };
        for &tier in &tiers {
            for biased in [false, true] {
                let batch = Batch {
                    tile,
                    padded: padded as u32,
                    tier,
                    sort_lanes: 256,
                };
                let mut planes: Vec<(usize, &[u8])> = scheme
                    .planes
                    .iter()
                    .map(|(r, b)| (*r, b.as_slice()))
                    .collect();
                if biased {
                    planes.push((3, &bias_bytes));
                }
                let (tab, _) = table(&mut gpu, &planes);
                let ran = std::cell::Cell::new(true);
                let want = landed_bits(&mut gpu, pairs * N, &|g, y| {
                    let y = Tensor::new(y.buf, pairs as u32, N as u32, y.dtype);
                    g.run(|c| {
                        ran.set(select::matmul_select_batched(
                            c,
                            "linear.moe_select",
                            x,
                            bank,
                            biased.then_some(bias),
                            routes,
                            EXPERTS as u32,
                            scratch,
                            y,
                            batch,
                        )?);
                        Ok(())
                    });
                });
                if !ran.get() {
                    continue;
                }
                let got = landed_bits(&mut gpu, pairs * N, &|g, y| {
                    let y = Tensor::new(y.buf, pairs as u32, N as u32, y.dtype);
                    g.run(|c| {
                        let ran = select::matmul_select_batched_tabled(
                            c,
                            "linear.moe_select",
                            x,
                            bank,
                            biased,
                            &tab,
                            routes,
                            EXPERTS as u32,
                            scratch,
                            y,
                            batch,
                        )?;
                        assert!(ran, "the tabled route serves what the contiguous one does");
                        Ok(())
                    });
                });
                ran_any = true;
                same(
                    &format!(
                        "batched {}{} at {}, {tile}-row tiles",
                        scheme.name,
                        if biased { " biased" } else { "" },
                        tier.name()
                    ),
                    &got,
                    &want,
                );
            }
        }
    }
    assert!(ran_any, "some scheme ran batched");
}
