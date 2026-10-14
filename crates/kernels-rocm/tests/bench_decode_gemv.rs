//! The dense models' decode projections (Gemma-4 31B, Qwen3.8-27B) at 1-8
//! rows through the dispatch the engine runs, with cold weights: the
//! captured launches cycle copies that together outgrow the 256 MiB
//! last-level cache, filled with random values (a constant buffer draws
//! less power and streams 10-15% faster).
//! `cargo test --release --features rocm --test bench_decode_gemv -- --ignored --nocapture`
//! (`GEMV_ONLY="gemma"` or a shape name narrows it, `GEMV_M="1,2"` the rows).

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, values};
use dtype::Dtype;
use kernels_rocm::Tensor;
use kernels_rocm::hip::rt;
use kernels_rocm::linear::{gemm, skinny};

const COLD_BYTES: usize = 768 << 20;
const BLOCK_BYTES: usize = 16 << 20;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Matmul,
    Geglu,
    Swiglu,
    Softcap,
}

/// (name, kind, weight rows, k)
const SHAPES: [(&str, Kind, u32, u32); 12] = [
    ("gemma.gate_up", Kind::Geglu, 43008, 5376),
    ("gemma.down", Kind::Matmul, 5376, 21504),
    ("gemma.qkv", Kind::Matmul, 16384, 5376),
    ("gemma.o", Kind::Matmul, 5376, 8192),
    ("gemma.qkv_glb", Kind::Matmul, 20480, 5376),
    ("gemma.lm_head", Kind::Softcap, 262_144, 5376),
    ("qwen.gate_up", Kind::Swiglu, 34816, 5120),
    ("qwen.down", Kind::Matmul, 5120, 17408),
    ("qwen.qkv", Kind::Matmul, 16384, 5120),
    ("qwen.o", Kind::Matmul, 5120, 6144),
    ("qwen.lm_head", Kind::Matmul, 248_320, 5120),
    ("qwen.attn_qkv", Kind::Matmul, 14336, 5120),
];

#[test]
#[ignore = "benchmark"]
fn bench_decode_gemv() {
    let Some(gpu) = Gpu::open() else { return };
    let only = std::env::var("GEMV_ONLY").unwrap_or_default();
    let ms: Vec<u32> = std::env::var("GEMV_M").ok().map_or(vec![1, 2, 4, 8], |v| {
        v.split(',').map(|x| x.parse().unwrap()).collect()
    });
    let block_elems = BLOCK_BYTES / 2;
    let block = gpu.tensor(&values(block_elems, 7, 0.05), 1, block_elems as u32, Dtype::Bf16);
    // `GEMV_NK="n,k;n,k"`: plain projections at these shapes instead.
    let custom: Vec<(&str, Kind, u32, u32)> = std::env::var("GEMV_NK").ok().map_or(Vec::new(), |v| {
        v.split(';')
            .map(|p| {
                let mut it = p.split(',').map(|x| x.parse().unwrap());
                ("custom", Kind::Matmul, it.next().unwrap(), it.next().unwrap())
            })
            .collect()
    });
    let shapes: Vec<(&str, Kind, u32, u32)> = if custom.is_empty() { SHAPES.to_vec() } else { custom };
    for (name, kind, n, k) in shapes {
        if !only.is_empty() && !only.split(',').any(|o| name.contains(o)) {
            continue;
        }
        let bytes = n as usize * k as usize * 2;
        let copies: Vec<Tensor> = (0..(COLD_BYTES / bytes + 1).max(2))
            .map(|c| {
                let w = gpu.zeros(n, k, Dtype::Bf16);
                for at in (0..bytes).step_by(BLOCK_BYTES) {
                    // Each copy starts the block at its own offset.
                    let skew = (c * 4096 + at / 7) % (BLOCK_BYTES / 2) & !15;
                    let len = (BLOCK_BYTES - skew).min(bytes - at);
                    // SAFETY: both ranges are live device allocations.
                    unsafe {
                        rt::hipMemcpy(
                            (w.ptr as usize + at) as *mut _,
                            (block.ptr as usize + skew) as *const _,
                            len,
                            rt::hipMemcpyDeviceToDevice,
                        );
                    }
                    if len < (BLOCK_BYTES).min(bytes - at) {
                        let rest = BLOCK_BYTES.min(bytes - at) - len;
                        unsafe {
                            rt::hipMemcpy(
                                (w.ptr as usize + at + len) as *mut _,
                                block.ptr as usize as *const _,
                                rest,
                                rt::hipMemcpyDeviceToDevice,
                            );
                        }
                    }
                }
                w
            })
            .collect();
        gpu.sync();
        for &m in &ms {
            let a = gpu.tensor(&values((m * k) as usize, 1, 1.0), m, k, Dtype::Bf16);
            let out_n = if matches!(kind, Kind::Geglu | Kind::Swiglu) { n / 2 } else { n };
            let y = gpu.zeros(m, n, Dtype::Bf16);
            let reps = (4 * copies.len()).max(24) as u32;
            let mut i = 0usize;
            let ctx = gpu.ctx();
            // The median of GEMV_RUNS replays (after one to warm the clocks).
            let runs: usize = std::env::var("GEMV_RUNS").ok().map_or(5, |v| v.parse().unwrap());
            let mut samples = Vec::new();
            for _ in 0..=runs {
                let us = gpu.time_graph(reps, || {
                let w = copies[i % copies.len()];
                i += 1;
                let mut yt = Tensor::new(y.ptr, m, out_n, Dtype::Bf16);
                match kind {
                    Kind::Matmul => gemm::matmul(&ctx, a, w, &mut yt).unwrap(),
                    Kind::Geglu => {
                        if skinny::covers(m as i32, out_n as i32, k as i32, skinny::Epilogue::Geglu, 64) {
                            skinny::skinny_bf16(&ctx, w.ptr, a.ptr, y.ptr, m as i32, out_n as i32, k as i32, skinny::Epilogue::Geglu).unwrap();
                        } else {
                            let mut yp = Tensor::new(y.ptr, m, n, Dtype::Bf16);
                            gemm::matmul(&ctx, a, w, &mut yp).unwrap();
                        }
                    }
                    Kind::Swiglu => {
                        if !gemm::matmul_swiglu(&ctx, a, w, &mut yt).unwrap() {
                            let mut yp = Tensor::new(y.ptr, m, n, Dtype::Bf16);
                            gemm::matmul(&ctx, a, w, &mut yp).unwrap();
                        }
                    }
                    Kind::Softcap => {
                        if skinny::covers(m as i32, n as i32, k as i32, skinny::Epilogue::Softcap(30.0), 64) {
                            skinny::skinny_bf16(&ctx, w.ptr, a.ptr, y.ptr, m as i32, n as i32, k as i32, skinny::Epilogue::Softcap(30.0)).unwrap();
                        } else {
                            gemm::lm_head(&ctx, a, w, &mut yt).unwrap();
                        }
                    }
                }
            });
                samples.push(us);
            }
            samples.remove(0);
            samples.sort_by(f64::total_cmp);
            let us = samples[samples.len() / 2];
            eprintln!(
                "{name:14} m={m} n={n:6} k={k:5}: {us:8.1} us {:5.2} TB/s",
                bytes as f64 / us / 1e6
            );
            // `GEMV_PICK="q,yt,unrl,waves,gridx,ks;..."` (q = 0 diagonal, 1
            // whole-product): each walk timed GEMV_REPS times, interleaved;
            // the default walk first.
            if let Ok(pick) = std::env::var("GEMV_PICK") {
                let epi = match kind {
                    Kind::Matmul => skinny::Epilogue::Store,
                    Kind::Geglu | Kind::Swiglu => skinny::Epilogue::Geglu,
                    Kind::Softcap => skinny::Epilogue::Softcap(30.0),
                };
                let cus = skinny::skinny_walk(1, 1 << 20, 4096, skinny::Epilogue::Store).4;
                let def = skinny::skinny_walk(m, out_n, k, epi);
                let mut walks = vec![((def.0, def.1, def.2, def.3, def.4, def.5), def.7, def.6)];
                for w in pick.split(';').filter(|w| !w.is_empty()) {
                    let v: Vec<u32> = w.split(',').map(|x| x.parse().unwrap()).collect();
                    walks.push(((v[1], v[2], v[3], v[3], v[4] * cus, false), v[0] == 1, v[5]));
                }
                let n_reps: usize = std::env::var("GEMV_REPS").ok().map_or(3, |v| v.parse().unwrap());
                let mut times = vec![Vec::new(); walks.len()];
                for _ in 0..n_reps {
                    for (j, (wk, quad, ks)) in walks.iter().enumerate() {
                        let mut i = 0usize;
                        let us = gpu.time_graph(reps, || {
                            let w = copies[i % copies.len()];
                            i += 1;
                            skinny::skinny_probe(&ctx, w.ptr, a.ptr, y.ptr, m as i32, out_n as i32, k as i32, epi,
                                (wk.0, wk.1, wk.2, wk.3, wk.4, wk.5), *quad, *ks).unwrap();
                        });
                        times[j].push(us);
                    }
                }
                for (j, (wk, quad, ks)) in walks.iter().enumerate() {
                    let mut t = times[j].clone();
                    t.sort_by(f64::total_cmp);
                    let med = t[t.len() / 2];
                    eprintln!(
                        "   {} q={} yt={} unrl={} waves={} grid={} ks={}: med {med:7.1} us min {:7.1} us {:5.2} TB/s",
                        if j == 0 { "default" } else { "pick   " },
                        u8::from(*quad), wk.0, wk.1, wk.2, wk.4, ks, t[0], bytes as f64 / med / 1e6
                    );
                }
            }
            if std::env::var_os("GEMV_BATCH").is_some() {
                let glu = matches!(kind, Kind::Geglu | Kind::Swiglu);
                let mut i = 0usize;
                let mut ok = true;
                let us = gpu.time_graph(reps, || {
                    let w = copies[i % copies.len()];
                    i += 1;
                    ok = ok && skinny::dense_batch_probe(&ctx, w.ptr, a.ptr, y.ptr, m as i32, out_n as i32, k as i32, glu).unwrap();
                });
                if ok {
                    eprintln!("   batch   {us:8.1} us {:5.2} TB/s", bytes as f64 / us / 1e6);
                }
            }
            if let Ok(mode) = std::env::var("GEMV_SWEEP") {
                let epi = match kind {
                    Kind::Matmul => skinny::Epilogue::Store,
                    Kind::Geglu | Kind::Swiglu => skinny::Epilogue::Geglu,
                    Kind::Softcap => skinny::Epilogue::Softcap(30.0),
                };
                let glu = epi == skinny::Epilogue::Geglu;
                let cus = skinny::skinny_walk(1, 1 << 20, 4096, skinny::Epilogue::Store).4;
                let d8 = skinny::skinny_walk(m, out_n, k, epi);
                let def = (d8.0, d8.1, d8.2, d8.3, d8.4, d8.5);
                let mut walks = vec![(def, d8.7, d8.6)];
                let confs = [(16u32, cus), (8, cus), (8, 2 * cus), (4, 4 * cus), (12, cus)];
                for quad in [false, true] {
                    if (quad && mode == "diag") || (!quad && mode == "quad") {
                        continue;
                    }
                    for yt in [1u32, 2, 4] {
                        for unrl in [1u32, 2, 4] {
                            if !quad && (m > 4 || glu) && yt * unrl > 4 {
                                continue;
                            }
                            if quad && glu && yt * unrl > 8 {
                                continue;
                            }
                            for ks in [1u32, 2, 4, 8] {
                                if !quad && ks > 1 {
                                    continue;
                                }
                                for (bw, g) in confs {
                                    if bw % ks != 0 {
                                        continue;
                                    }
                                    for ag in [false, true] {
                                        walks.push(((yt, unrl, bw, bw, g, ag), quad, ks));
                                    }
                                }
                            }
                        }
                    }
                }
                // The whole-product kernel lands what the diagonal one does.
                let check = gpu.zeros(m, n, Dtype::Bf16);
                let w0 = copies[0];
                skinny::skinny_probe(&ctx, w0.ptr, a.ptr, check.ptr, m as i32, out_n as i32, k as i32, epi, def, d8.7, d8.6).unwrap();
                let want = gpu.read(&check);
                let mut res: Vec<(f64, ((u32, u32, u32, u32, u32, bool), bool, u32))> = Vec::new();
                for (wk, quad, ks) in walks {
                    if quad {
                        let have = gpu.zeros(m, n, Dtype::Bf16);
                        skinny::skinny_probe(&ctx, w0.ptr, a.ptr, have.ptr, m as i32, out_n as i32, k as i32, epi, wk, true, ks).unwrap();
                        let got = gpu.read(&have);
                        let bad = got.iter().zip(&want).take((m * out_n) as usize)
                            .filter(|(h, w)| (*h - *w).abs() > 0.02 * w.abs().max(0.05)).count();
                        assert!(bad == 0, "{name} m={m} {wk:?} ks={ks}: {bad} cells off");
                    }
                    let mut i = 0usize;
                    let us = gpu.time_graph(reps, || {
                        let w = copies[i % copies.len()];
                        i += 1;
                        skinny::skinny_probe(&ctx, w.ptr, a.ptr, y.ptr, m as i32, out_n as i32, k as i32, epi, wk, quad, ks).unwrap();
                    });
                    res.push((us, (wk, quad, ks)));
                }
                let d = res[0];
                res.sort_by(|a, b| a.0.total_cmp(&b.0));
                eprintln!("   default {:?} {:8.1} us {:5.2} TB/s", d.1, d.0, bytes as f64 / d.0 / 1e6);
                for r in res.iter().take(5) {
                    eprintln!("   best    {:?} {:8.1} us {:5.2} TB/s", r.1, r.0, bytes as f64 / r.0 / 1e6);
                }
            }
        }
    }
}

/// The one-row chain kernels a Gemma-4 31B layer fires besides its
/// projections and attention, each replayed from a graph.
#[test]
#[ignore = "benchmark"]
fn bench_gemma_small() {
    use kernels_rocm::elemwise::norm;
    let Some(gpu) = Gpu::open() else { return };
    let h = 5376u32;
    let bf = |rows: u32, width: u32, seed: u64| {
        gpu.tensor(&values((rows * width) as usize, seed, 0.5), rows, width, Dtype::Bf16)
    };
    let x = bf(1, h, 1);
    let w0 = bf(1, h, 2);
    let w1 = bf(1, h, 3);
    let s = bf(1, 1, 4);
    let mut t = gpu.zeros(1, h, Dtype::Bf16);
    let mut y = bf(1, h, 5);
    let mut scaled = gpu.zeros(1, h, Dtype::Bf16);
    let mut out = gpu.zeros(1, h, Dtype::Bf16);
    let floor = gpu.time_graph(100, || norm::mul_scalar(&gpu.ctx(), 1.0, &mut out).unwrap());
    eprintln!("small mul_scalar (floor)          : {floor:6.2} us");
    let us = gpu.time_graph(100, || {
        norm::rmsnorm_residual_add(
            &gpu.ctx(),
            x,
            w0,
            1e-6,
            &mut t,
            &mut y,
            None,
            Some(norm::PostNorm { weight: w1, plus_one: false, eps: 1e-6, out: &mut out }),
        )
        .unwrap()
    });
    eprintln!("small rmsnorm_residual_add post   : {us:6.2} us");
    let us = gpu.time_graph(100, || {
        norm::rmsnorm_residual_add(
            &gpu.ctx(),
            x,
            w0,
            1e-6,
            &mut t,
            &mut y,
            Some((s, &mut scaled)),
            Some(norm::PostNorm { weight: w1, plus_one: false, eps: 1e-6, out: &mut out }),
        )
        .unwrap()
    });
    eprintln!("small rmsnorm_residual_add scaled : {us:6.2} us");
}
