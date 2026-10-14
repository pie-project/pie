#![cfg(feature = "cuda")]

//! The grouped tcgen05 GEMM's rate at real MoE layers' expert projections:
//! DeepSeek-V4-Flash (256 experts, top 6, fp8 1x128 and W4A8) and
//! Qwen3.8-Flash-Next (512 experts, top 10, NVFP4), uniform and skewed
//! routing. Each batch cycles 16 routings, so each launch reads mostly
//! other experts' weights than the last. Ignored: a measurement. Run alone
//! on an idle GPU:
//!
//!     cargo test -p kernels-cuda --features cuda --release --test kernels_cuda \
//!         the_tcgen05_grouped_gemm_paces -- --ignored --nocapture
//!
//! `PIE_TC5_LAYER` keeps the layers whose name holds it, `PIE_TC5_TOKENS`
//! one token count; `PIE_TC5_SPLITS` overrides the plan's K splits;
//! `PIE_TC5_PAIR=0|1` runs contiguous launches on lone CTAs or CTA pairs;
//! `PIE_TC5_FUSED_BF16` lands the fused gate_up's GLU as bf16 instead.

use crate::common::Gpu;
use crate::common::tc5::random;
use crate::the_tcgen05_gemm_lands_and_paces_what_cublaslt_does::{BATCH, FLUSH, Timer, noise};
use kernels_cuda::cudarc::runtime::sys as rt;
use kernels_cuda::linear::tc5::{
    Elem, Epilogue, GROUP_ALIGN, Glu, Grouped, Groups, Operand, Out, Plan, atom_bytes,
    grouped_gemm_epi_with, grouped_gemm_with, masked_atom_bytes, supported,
};

/// Each token's `top` distinct experts: uniform, or skewed so the first
/// tenth of the experts take about half the routes.
fn routing(tokens: usize, experts: usize, top: usize, skewed: bool, seed: u64) -> Vec<usize> {
    let mut bits = seed;
    let mut counts = vec![0usize; experts];
    for _ in 0..tokens {
        let mut picked: Vec<usize> = Vec::with_capacity(top);
        while picked.len() < top {
            let u = f64::from(random(&mut bits)) / 4_294_967_296.0;
            let e = if skewed {
                ((u * u * u) * experts as f64) as usize
            } else {
                (u * experts as f64) as usize
            }
            .min(experts - 1);
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        for e in picked {
            counts[e] += 1;
        }
    }
    counts
}

#[derive(Clone, Copy)]
struct Kind {
    name: &'static str,
    act: (Elem, Option<u32>),
    w: (Elem, Option<u32>),
}

impl Kind {
    fn bytes(e: Elem, n: usize) -> usize {
        match e {
            Elem::Bf16 => n * 2,
            Elem::E4m3 => n,
            Elem::E2m1 | Elem::U4 => n / 2,
        }
    }

    /// A fill whose every reading is finite, and its scale bytes'.
    fn fill(e: Elem) -> (u8, u8) {
        match e {
            Elem::Bf16 => (0xbf, 0),
            Elem::E4m3 => (0xb7, 0),
            Elem::E2m1 | Elem::U4 => (0xff, 0),
        }
    }

    fn scale_fill(gran: u32) -> (u8, u8) {
        if gran == 16 {
            (0x07, 0x30)
        } else {
            (0x03, 0x7c)
        }
    }
}

fn operand(
    gpu: &mut Gpu,
    (elem, gran): (Elem, Option<u32>),
    rows: usize,
    k: usize,
    atoms: usize,
    seed: u64,
) -> Operand {
    let (mask, base) = Kind::fill(elem);
    let ptr = gpu.up(&noise(Kind::bytes(elem, rows * k), mask, base, seed));
    match gran {
        Some(gran) => {
            let (m, b) = Kind::scale_fill(gran);
            Operand::scaled(ptr, elem, gpu.up(&noise(atoms, m, b, seed + 1)), gran)
        }
        None => Operand::plain(ptr, elem),
    }
}

#[allow(clippy::too_many_lines)]
fn layer(name: &str, experts: usize, top: usize, [n, k]: [usize; 2], kind: Kind) {
    if std::env::var("PIE_TC5_LAYER").is_ok_and(|only| !name.contains(only.as_str())) {
        return;
    }
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let stream: rt::cudaStream_t = ctx.stream().cast();
    let timer = Timer {
        flush: gpu.zeros(FLUSH),
        stream,
    };
    // A W4A16 bank's scales lie row-major: as many bytes, rounded up.
    let bank_atoms = kind.w.1.map_or(0, |g| {
        atom_bytes(experts * n, k, g as usize).max(experts * n * k / g as usize)
    });
    let w = operand(&mut gpu, kind.w, experts * n, k, bank_atoms, 3);
    let alpha_e = gpu.up(&vec![1f32; experts]);
    for skewed in [false, true] {
        let only: Option<usize> = std::env::var("PIE_TC5_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok());
        for tokens in [1usize, 16, 128, 1024, 8192] {
            if only.is_some_and(|t| t != tokens) {
                continue;
            }
            let routings: Vec<Vec<usize>> = (0..BATCH)
                .map(|r| {
                    routing(
                        tokens,
                        experts,
                        top,
                        skewed,
                        0x51 + r as u64 * 7919 + tokens as u64,
                    )
                })
                .collect();
            let routed = tokens * top;
            let flops = 2.0 * (routed * n * k) as f64;
            let mut cells = Vec::new();
            for masked in [true, false] {
                let busiest = routings
                    .iter()
                    .flat_map(|c| c.iter().copied())
                    .max()
                    .unwrap_or(0);
                if masked && (tokens > 1024 || busiest > 256) || !masked && tokens < 128 {
                    continue;
                }
                let (groups, rows, act_atoms): (Vec<Groups>, usize, usize) = if masked {
                    let m_max = routings
                        .iter()
                        .flat_map(|c| c.iter().copied())
                        .max()
                        .unwrap_or(1)
                        .next_multiple_of(16);
                    let groups = routings
                        .iter()
                        .map(|c| {
                            let counts: Vec<i32> = c.iter().map(|&x| x as i32).collect();
                            Groups::Masked {
                                counts: gpu.up(&counts),
                                m_max: m_max as u32,
                            }
                        })
                        .collect();
                    let atoms = kind
                        .act
                        .1
                        .map_or(0, |g| masked_atom_bytes(experts, m_max, k, g as usize));
                    (groups, experts * m_max, atoms)
                } else {
                    let align = GROUP_ALIGN as usize;
                    let mut cap = 0;
                    let groups = routings
                        .iter()
                        .map(|c| {
                            let mut offsets = vec![0i32];
                            for &x in c {
                                offsets.push(
                                    offsets.last().unwrap() + x.next_multiple_of(align) as i32,
                                );
                            }
                            cap = cap.max(*offsets.last().unwrap() as usize);
                            Groups::Contiguous {
                                offsets: gpu.up(&offsets),
                                rows: 0,
                            }
                        })
                        .collect::<Vec<_>>();
                    let cap = cap.max(align);
                    let groups = groups
                        .into_iter()
                        .map(|g| match g {
                            Groups::Contiguous { offsets, .. } => Groups::Contiguous {
                                offsets,
                                rows: cap as u32,
                            },
                            other => other,
                        })
                        .collect();
                    let atoms = kind.act.1.map_or(0, |g| atom_bytes(cap, k, g as usize));
                    (groups, cap, atoms)
                };
                let act = operand(&mut gpu, kind.act, rows, k, act_atoms, 9);
                let y = gpu.zeros(rows * n * 2);
                let grouped = |i: usize| Grouped {
                    experts: experts as u32,
                    groups: groups[i % groups.len()],
                    alpha_e,
                };
                let shape = [n as u32, k as u32];
                let us = match run(&ctx, act, w, y, grouped(0), shape, 1.0, routed as u32) {
                    Ok(()) => timer.median(|i| {
                        run(&ctx, act, w, y, grouped(i), shape, 1.0, routed as u32)
                            .expect("fires again");
                    }),
                    Err(why) => panic!("{name} {}: {why}", kind.name),
                };
                cells.push(format!(
                    "{} {:.1}us {:.0}TF",
                    if masked { "masked" } else { "contiguous" },
                    us,
                    flops / us * 1e-6
                ));
                // A gate_up's fused form: the SwiGLU of the halves, landed as
                // the down projection's quantized operand.
                if name.contains("gate_up") {
                    let inter = n / 2;
                    let scale_rows = match groups[0] {
                        Groups::Masked { m_max, .. } => {
                            experts * (m_max as usize).next_multiple_of(128)
                        }
                        Groups::Contiguous { .. } => rows,
                    };
                    let out = if std::env::var("PIE_TC5_FUSED_BF16").is_ok()
                        || kind.act.0 == Elem::Bf16
                    {
                        Out::Bf16
                    } else if kind.act.0 == Elem::E2m1 {
                        Out::Nvfp4 {
                            data: gpu.zeros(rows * inter / 2),
                            scales: gpu.zeros(atom_bytes(scale_rows, inter, 16)),
                            gscale: 1.0,
                        }
                    } else {
                        Out::Fp8Block {
                            data: gpu.zeros(rows * inter),
                            scales: gpu.zeros(atom_bytes(scale_rows, inter, 128)),
                        }
                    };
                    let epi = Epilogue {
                        glu: Some(Glu::Swiglu),
                        out,
                        row_scale: 0,
                        bias: 0,
                        weight_scale: 0,
                    };
                    let fused = |i: usize| {
                        let g = grouped(i);
                        let shape = [inter as u32, k as u32];
                        grouped_gemm_epi_with(
                            &ctx,
                            act,
                            w,
                            y,
                            g,
                            shape,
                            1.0,
                            epi,
                            plan_for(&g, shape, routed as u32),
                        )
                    };
                    let us = match fused(0) {
                        Ok(()) => timer.median(|i| fused(i).expect("fires again")),
                        Err(why) => panic!("{name} {} fused: {why}", kind.name),
                    };
                    cells.push(format!("+swiglu+quant {us:.1}us"));
                }
            }
            let busiest = routings
                .iter()
                .flat_map(|c| c.iter().copied())
                .max()
                .unwrap_or(0);
            println!(
                "| {name} | {} | {} | {tokens} | {busiest} | {} |",
                kind.name,
                if skewed { "skewed" } else { "uniform" },
                cells.join(" / ")
            );
        }
    }
}

/// The planned grouped GEMM, its K splits overridden by `PIE_TC5_SPLITS`.
#[allow(clippy::too_many_arguments)]
fn run(
    ctx: &kernels_cuda::Ctx,
    act: Operand,
    w: Operand,
    y: u64,
    g: Grouped,
    shape: [u32; 2],
    alpha: f32,
    routed: u32,
) -> Result<(), kernels_cuda::Error> {
    grouped_gemm_with(ctx, act, w, y, g, shape, alpha, plan_for(&g, shape, routed))
}

/// The grouped plan, its K splits overridden by `PIE_TC5_SPLITS` and a
/// contiguous launch's CTA pairs by `PIE_TC5_PAIR` (0 or 1).
fn plan_for(g: &Grouped, shape: [u32; 2], routed: u32) -> Plan {
    let mut plan = Plan::grouped(g, shape[0], shape[1], routed, 148);
    if let Some(s) = std::env::var("PIE_TC5_SPLITS")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        plan.splits = s;
    }
    if let (Groups::Contiguous { .. }, Ok(pair)) = (g.groups, std::env::var("PIE_TC5_PAIR")) {
        plan.pair = pair == "1";
        if plan.pair {
            plan.bn = 256;
        }
    }
    plan
}

#[test]
#[ignore = "a measurement: run alone, on an idle GPU"]
fn the_tcgen05_grouped_gemm_paces_a_moe_layer() {
    if !supported() {
        eprintln!("skipped: no tcgen05 on this device");
        return;
    }
    let fp8 = Kind {
        name: "fp8 1x128",
        act: (Elem::E4m3, Some(128)),
        w: (Elem::E4m3, Some(128)),
    };
    let w4a8 = Kind {
        name: "w4a8",
        act: (Elem::E4m3, Some(128)),
        w: (Elem::E2m1, Some(32)),
    };
    let nvfp4 = Kind {
        name: "nvfp4",
        act: (Elem::E2m1, Some(16)),
        w: (Elem::E2m1, Some(16)),
    };
    println!("| layer | kind | routing | tokens | busiest expert | time, rate |");
    println!("|---|---|---|---|---|---|");
    for kind in [fp8, w4a8] {
        layer("dsv4 gate_up", 256, 6, [4096, 4096], kind);
        layer("dsv4 down", 256, 6, [4096, 2048], kind);
    }
    layer("qwen gate_up", 512, 10, [1280, 2560], nvfp4);
    layer("qwen down", 512, 10, [2560, 640], nvfp4);
    let w4a16 = Kind {
        name: "w4a16 mxfp4",
        act: (Elem::Bf16, None),
        w: (Elem::E2m1, Some(32)),
    };
    layer("gpt-oss gate_up", 128, 4, [5760, 2880], w4a16);
    layer("gpt-oss down", 128, 4, [2880, 2880], w4a16);
    let bf16 = Kind {
        name: "bf16",
        act: (Elem::Bf16, None),
        w: (Elem::Bf16, None),
    };
    layer("gpt-oss-bf16 gate_up", 128, 4, [5760, 2880], bf16);
}
