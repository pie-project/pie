//! A decode's dense bf16 products (`y = act . w^T`, one to 64 rows) on the
//! vendor's library against pie's own kernels: what each lands against
//! cuBLAS, and how fast against the weight-streaming roofline, at the
//! projections of the catalog's dense models.
//!
//! The kernels:
//!
//! - `matmul`: `linear::gemm::matmul`, the engine's path for an unquantized
//!   product: cuBLAS and cuBLASLt tactics timed per shape (and, at one row,
//!   the gemv), under the default (L40S) profile.
//! - `gemmex`: `cublasGemmEx` alone at its default tensor-op algorithm, the
//!   vendor's own heuristic with no tuning and no kernel of ours.
//! - `skinny`: `linear::skinny` with the plain store, where it `covers`.
//! - `rows`: `linear::gemv::gemm_rows_bf16`, one to eight rows.
//! - `gemv`: `linear::gemv::gemv_bf16`, one row.
//! - `lane`: `linear::lane_gemm`, a warp a dot, at one and two rows.

use crate::common::linear_ref::t;
use crate::common::{Gpu, Lcg, captured, copy_rate, forget, from_bf16, replay};
use dtype::Dtype::Bf16;
use kernels_cuda::cudarc::cublas::sys as blas;
use kernels_cuda::jit::Ctx;
use kernels_cuda::linear::skinny::{self, Epilogue};
use kernels_cuda::linear::{gemm, gemv, lane_gemm};

/// (model, projection, n, k): each distinct product of a decode's layer.
#[rustfmt::skip]
const SHAPES: [(&str, &str, usize, usize); 19] = [
    ("qwen3-1.7b", "qkv", 4096, 2048),
    ("qwen3-1.7b", "o", 2048, 2048),
    ("qwen3-1.7b", "gate_up", 12288, 2048),
    ("qwen3-1.7b", "down", 2048, 6144),
    ("qwen3-8b/llama3-8b", "qkv", 6144, 4096),
    ("qwen3-8b/llama3-8b", "o", 4096, 4096),
    ("qwen3-8b", "gate_up", 24576, 4096),
    ("qwen3-8b", "down", 4096, 12288),
    ("llama3-8b", "gate_up", 28672, 4096),
    ("llama3-8b", "down", 4096, 14336),
    ("qwen3.6-27b", "qkv+gate", 14336, 5120),
    ("qwen3.6-27b", "gdn qkvz", 16384, 5120),
    ("qwen3.6-27b", "o", 5120, 6144),
    ("qwen3.6-27b", "gate_up", 34816, 5120),
    ("qwen3.6-27b", "down", 5120, 17408),
    ("gemma4-31b", "qkv", 16384, 5376),
    ("gemma4-31b", "o", 5376, 8192),
    ("gemma4-31b", "gate_up", 43008, 5376),
    ("gemma4-31b", "down", 5376, 21504),
];

const ROWS: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kernel {
    Matmul,
    GemmEx,
    Skinny,
    Rows,
    Gemv,
    Lane,
}

impl Kernel {
    const ALL: [Kernel; 6] = [
        Kernel::Matmul,
        Kernel::GemmEx,
        Kernel::Skinny,
        Kernel::Rows,
        Kernel::Gemv,
        Kernel::Lane,
    ];

    fn name(self) -> &'static str {
        match self {
            Kernel::Matmul => "matmul",
            Kernel::GemmEx => "gemmex",
            Kernel::Skinny => "skinny",
            Kernel::Rows => "rows",
            Kernel::Gemv => "gemv",
            Kernel::Lane => "lane",
        }
    }

    fn applies(self, m: usize, n: usize, k: usize) -> bool {
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        match self {
            Kernel::Matmul | Kernel::GemmEx => true,
            Kernel::Skinny => skinny::covers(mi, ni, ki, Epilogue::Store, skinny::ROWS),
            Kernel::Rows => m <= 8 && k.is_multiple_of(8),
            Kernel::Gemv => m == 1 && k.is_multiple_of(8),
            Kernel::Lane => m <= 2,
        }
    }

    /// `y = act . w^T` over `m` rows on this kernel.
    fn fire(self, ctx: &Ctx, w: u64, act: u64, y: u64, [m, n, k]: [usize; 3]) {
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        let (a, wt) = (t(act, m, k, Bf16), t(w, n, k, Bf16));
        let mut yt = t(y, m, n, Bf16);
        let fired = match self {
            Kernel::Matmul => gemm::matmul(ctx, a, wt, &mut yt),
            Kernel::GemmEx => {
                gemm_ex(ctx, w, act, y, [mi, ni, ki]);
                Ok(())
            }
            Kernel::Skinny => skinny::skinny_bf16(ctx, w, act, y, mi, ni, ki, Epilogue::Store),
            Kernel::Rows => gemv::gemm_rows_bf16(ctx, w, act, 0, y, mi, ni, ki, false),
            Kernel::Gemv => gemv::gemv_bf16(ctx, w, act, 0, y, ni, ki),
            Kernel::Lane => lane_gemm::act_x_wt(ctx, "linear.matmul", a, wt, &mut yt),
        };
        fired.unwrap_or_else(|e| panic!("{} at {m}x{n}x{k} fires: {e:?}", self.name()));
    }
}

/// `cublasGemmEx` on `ctx`'s handle and stream, its default tensor-op
/// algorithm, fp32 accumulation: what `matmul` calls a `GemmEx` tactic.
#[allow(unsafe_code)]
fn gemm_ex(ctx: &Ctx, w: u64, act: u64, y: u64, [m, n, k]: [i32; 3]) {
    let handle: blas::cublasHandle_t = ctx.cublas("bench.gemm_ex").expect("a handle").cast();
    let (alpha, beta) = (1.0f32, 0.0f32);
    // SAFETY: the handle is the test's, bound to `ctx`'s stream; `w` is an
    // `n x k`, `act` an `m x k` and `y` an `m x n` bf16 allocation the
    // bench holds, and the scalars are host locals (the default pointer mode).
    let status = unsafe {
        blas::cublasGemmEx(
            handle,
            blas::cublasOperation_t::CUBLAS_OP_T,
            blas::cublasOperation_t::CUBLAS_OP_N,
            n,
            m,
            k,
            std::ptr::from_ref(&alpha).cast(),
            w as usize as *const _,
            blas::cudaDataType::CUDA_R_16BF,
            k,
            act as usize as *const _,
            blas::cudaDataType::CUDA_R_16BF,
            k,
            std::ptr::from_ref(&beta).cast(),
            y as usize as *mut _,
            blas::cudaDataType::CUDA_R_16BF,
            n,
            blas::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            blas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    };
    assert_eq!(
        status,
        blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS,
        "`cublasGemmEx` at {m}x{n}x{k}"
    );
}

/// The worst distance of `got` from cuBLAS's `want`, in units of a bf16
/// product's tolerance: each element's distance over 1/64 of its value
/// plus the plane's RMS (a near-zero sum's terms are the RMS's size).
fn off(got: &[u16], want: &[u16]) -> f64 {
    let f = |v: u16| f64::from(from_bf16(v));
    let rms = (want.iter().map(|&v| f(v) * f(v)).sum::<f64>() / want.len() as f64).sqrt();
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (f(g) - f(w)).abs() / ((f(w).abs() + rms) / 64.0))
        .fold(0.0, f64::max)
}

fn median(mut us: Vec<f64>) -> f64 {
    us.sort_by(f64::total_cmp);
    us[us.len() / 2]
}

/// Each own kernel against cuBLAS at a decode's rows over the catalog's
/// dense projections: each lands cuBLAS's product within a bf16 tolerance,
/// and its time against the weight's bytes streamed at the device's copy
/// rate, the roofline: `cargo test --release -p kernels-cuda --features cuda
/// --test kernels_cuda -- --ignored decode_gemm_rate --nocapture`
/// (`PIE_DECODE_GEMM=<words>` for the shapes whose model and projection
/// read so).
///
/// A fire is a graph of its fires over weight planes rotating through at
/// least 1 GiB, so every fire streams its weight from DRAM, not the L2 (the
/// activation, a few rows, stays there, as a decode's does). Each time is
/// the median over rounds, every kernel's graph in turn, of a round's
/// median replay; the minimum over rounds rides beside it, and their
/// spread says how quiet a shared card was.
#[test]
#[ignore = "a rate bench: cargo test --release ... -- --ignored decode_gemm_rate --nocapture"]
fn decode_gemm_rate() {
    const FOOTPRINT: usize = 1 << 30;
    const FIRES: usize = 24;
    const REPLAYS: usize = 11;
    const ROUNDS: usize = 7;
    // The roofline at the fastest copy of one before each shape: another
    // process's work only ever slows one.
    let mut copy = 0f64;
    let mut lines = Vec::new();
    let mut lcg = Lcg::seeded(0xdec0de);
    // `PIE_DECODE_GEMM=<words>` times only the shapes whose "model proj"
    // holds them (a rerun of the rows a burst on a shared card spoiled).
    let only = std::env::var("PIE_DECODE_GEMM").unwrap_or_default();
    for (model, proj, n, k) in SHAPES {
        if !format!("{model} {proj}").contains(only.as_str()) {
            continue;
        }
        copy = copy.max(copy_rate());
        let mut gpu = Gpu::open();
        let ctx = gpu.ctx();
        let plane = n * k * 2;
        let copies = FOOTPRINT.div_ceil(plane).max(2);
        let fires = FIRES.max(copies).div_ceil(copies) * copies;
        // The first plane random, for the check; the rest one bf16 (0x3c3c,
        // 0.0115) throughout, streamed as alike.
        let weights: Vec<u64> = (0..copies)
            .map(|i| {
                if i == 0 {
                    gpu.up(&lcg.row(n * k).0)
                } else {
                    let at = gpu.zeros(plane);
                    gpu.fill(at, 0x3c, plane);
                    at
                }
            })
            .collect();
        let m_most = *ROWS.last().expect("rows");
        let act = gpu.up(&lcg.row(m_most * k).0);
        let outs: Vec<u64> = Kernel::ALL
            .iter()
            .map(|_| gpu.zeros(m_most * n * 2))
            .collect();
        gpu.sync();
        for m in ROWS {
            let kernels: Vec<(usize, Kernel)> = Kernel::ALL
                .into_iter()
                .enumerate()
                .filter(|(_, kernel)| kernel.applies(m, n, k))
                .collect();
            // The check: each on the random plane against cuBLAS (`matmul`
            // twice, its first call tuning the shape's tactic).
            let shape = [m, n, k];
            Kernel::Matmul.fire(&ctx, weights[0], act, outs[0], shape);
            for &(i, kernel) in &kernels {
                kernel.fire(&ctx, weights[0], act, outs[i], shape);
            }
            gpu.sync();
            let want: Vec<u16> = gpu.down(outs[0], m * n);
            let mut worst = 0f64;
            for &(i, kernel) in &kernels[1..] {
                let got: Vec<u16> = gpu.down(outs[i], m * n);
                let apart = off(&got, &want);
                assert!(
                    apart <= 1.0,
                    "{} at {m}x{n}x{k} is {apart:.2} tolerances off cuBLAS",
                    kernel.name()
                );
                worst = worst.max(apart);
            }
            let execs: Vec<_> = kernels
                .iter()
                .map(|&(i, kernel)| {
                    let all = || {
                        for f in 0..fires {
                            kernel.fire(&ctx, weights[f % copies], act, outs[i], shape);
                        }
                    };
                    all();
                    gpu.sync();
                    let exec = captured(&ctx, all);
                    replay(&ctx, exec);
                    gpu.sync();
                    exec
                })
                .collect();
            // rounds of each graph in turn, so a burst of another process's
            // work on a shared card lands in a round of each
            let mut rounds = vec![Vec::new(); kernels.len()];
            let mut least = vec![f64::INFINITY; kernels.len()];
            for _ in 0..ROUNDS {
                for (r, exec) in execs.iter().enumerate() {
                    let us: Vec<f64> = (0..REPLAYS)
                        .map(|_| {
                            let start = std::time::Instant::now();
                            replay(&ctx, *exec);
                            gpu.sync();
                            start.elapsed().as_secs_f64() * 1e6 / fires as f64
                        })
                        .collect();
                    least[r] = us.iter().copied().fold(least[r], f64::min);
                    rounds[r].push(median(us));
                }
            }
            for exec in execs {
                forget(exec);
            }
            let times: Vec<Option<(f64, f64)>> = Kernel::ALL
                .iter()
                .map(|&kernel| {
                    let r = kernels.iter().position(|&(_, x)| x == kernel)?;
                    Some((median(rounds[r].clone()), least[r]))
                })
                .collect();
            let line = Line {
                model,
                proj,
                n,
                k,
                m,
                times,
                worst,
            };
            eprintln!("{}", line.raw());
            lines.push(line);
        }
    }
    eprintln!("\ndevice copy (bytes read and written): {copy:.0} GB/s");
    eprintln!(
        "{:<18} {:<8} {:>5} {:>5} {:>2} |{}| best own  own/matmul (med, min)  spread  off",
        "model",
        "proj",
        "n",
        "k",
        "m",
        Kernel::ALL
            .map(|k| format!(" {:>7} us   min %roof |", k.name()))
            .concat()
    );
    for line in &lines {
        eprintln!("{}", line.table(copy));
    }
}

/// One shape's times: each kernel's (median, minimum) microseconds a fire,
/// where it serves the shape, and the worst own kernel's distance from
/// cuBLAS in tolerances.
struct Line {
    model: &'static str,
    proj: &'static str,
    n: usize,
    k: usize,
    m: usize,
    times: Vec<Option<(f64, f64)>>,
    worst: f64,
}

impl Line {
    fn raw(&self) -> String {
        let cells: String = Kernel::ALL
            .iter()
            .zip(&self.times)
            .filter_map(|(kernel, time)| {
                time.map(|(us, least)| format!(" {} {us:.2}/{least:.2}", kernel.name()))
            })
            .collect();
        format!(
            "{} {} {}x{}x{}:{cells}",
            self.model, self.proj, self.m, self.n, self.k
        )
    }

    fn table(&self, copy: f64) -> String {
        let bytes = (self.n * self.k * 2) as f64;
        let mut cells = String::new();
        for time in &self.times {
            match time {
                Some((us, least)) => {
                    let roof = 100.0 * bytes / us / 1e3 / copy;
                    cells.push_str(&format!(" {us:>10.2} {least:>8.2} {roof:>4.0}% |"));
                }
                None => cells.push_str(&format!(" {:>10} {:>8} {:>5} |", "-", "-", "-")),
            }
        }
        let own = Kernel::ALL
            .iter()
            .zip(&self.times)
            .filter(|(kernel, _)| !matches!(kernel, Kernel::Matmul | Kernel::GemmEx))
            .filter_map(|(kernel, time)| time.map(|t| (*kernel, t)))
            .min_by(|a, b| a.1.0.total_cmp(&b.1.0))
            .expect("skinny serves every shape here");
        let (matmul, matmul_least) = self.times[0].expect("matmul serves every shape");
        let spread = self
            .times
            .iter()
            .flatten()
            .map(|(us, least)| us / least - 1.0)
            .fold(0.0, f64::max);
        format!(
            "{:<18} {:<8} {:>5} {:>5} {:>2} |{cells} {:<8} {:>6.3} {:>6.3}           {:>5.1}%  {:.2}",
            self.model,
            self.proj,
            self.n,
            self.k,
            self.m,
            own.0.name(),
            own.1.0 / matmul,
            own.1.1 / matmul_least,
            100.0 * spread,
            self.worst,
        )
    }
}
