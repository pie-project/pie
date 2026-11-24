//! The bf16 routed select's pick against every leg that could take its fire,
//! each timed on this device as serving runs it: captured in a graph, the
//! bank rotated through copies past the last-level cache. Each leg's line
//! states its median and spread in us and its price on the tests' roofline
//! (the RTX Spark's rates: `common::roof`); `*` marks the leg priced
//! cheapest, the one `matmul_select` takes on a device of those rates. A
//! probe run by hand (`--ignored --nocapture`).

use crate::common::linear_ref::t;
use crate::common::{Gpu, Lcg, captured, forget, replay};
use dtype::Dtype::{Bf16, I32};
use kernels_cuda::jit::Ctx;
use kernels_cuda::linear::moe::{self, ExpertTable};
use kernels_cuda::tuning::Tuning;

/// A routed projection: `[experts, top, n, k]`.
struct Shape {
    name: &'static str,
    experts: usize,
    top: usize,
    n: usize,
    k: usize,
}

const SHAPES: [Shape; 4] = [
    Shape {
        name: "a3b gate-up",
        experts: 128,
        top: 8,
        n: 1536,
        k: 2048,
    },
    Shape {
        name: "a3b down",
        experts: 128,
        top: 8,
        n: 2048,
        k: 768,
    },
    Shape {
        name: "uncached gate-up",
        experts: 32,
        top: 4,
        n: 1024,
        k: 2048,
    },
    Shape {
        name: "uncached down",
        experts: 32,
        top: 4,
        n: 2048,
        k: 512,
    },
];

/// Routes an expert averages: a decode's one up to a prefill's.
const BANDS: [usize; 6] = [1, 4, 8, 12, 32, 256];

/// Launches a graph holds, so one replay's launch is noise beside them.
const REPS: usize = 16;

/// Timed replays; the median is kept.
const ROUNDS: usize = 7;

/// The bytes the bank copies span: past any last-level cache.
const SPAN: usize = 256 << 20;

fn times(gpu: &Gpu, ctx: &Ctx, launch: &dyn Fn(&Ctx, usize)) -> Vec<f64> {
    gpu.lay(ctx, |c| launch(c, 0));
    launch(ctx, 0);
    gpu.sync();
    let exec = captured(ctx, || (0..REPS).for_each(|i| launch(ctx, i)));
    replay(ctx, exec);
    gpu.sync();
    let mut out: Vec<f64> = (0..ROUNDS)
        .map(|_| {
            let at = std::time::Instant::now();
            replay(ctx, exec);
            gpu.sync();
            at.elapsed().as_secs_f64() * 1e6 / REPS as f64
        })
        .collect();
    forget(exec);
    out.sort_by(f64::total_cmp);
    out
}

#[test]
#[ignore = "a probe run by hand"]
fn each_legs_time_by_band() {
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let roof = crate::common::roof();
    let tuning = ctx.tuning();
    let ctx = ctx.with_tuning(Tuning { roof, ..tuning });
    let rates = roof.rates().expect("the tests' roofline states every rate");
    let cores = moe::Cores::of(&ctx, ExpertTable::RESIDENT).expect("the device answers");
    let mut lcg = Lcg::seeded(0x005e_1ec7);
    for s in &SHAPES {
        let plane = s.n * s.k;
        let bank_bytes = s.experts * plane * 2;
        let copies = SPAN.div_ceil(bank_bytes).max(1);
        let noise: Vec<u16> = (0..s.experts * plane)
            .map(|_| crate::common::to_bf16((lcg.unit() - 0.5) / 16.0))
            .collect();
        let banks: Vec<u64> = (0..copies).map(|_| gpu.up(&noise)).collect();
        for &per in &BANDS {
            let count = per * s.experts;
            let tokens = count / s.top;
            let routes: Vec<i32> = (0..count).map(|r| (r % s.experts) as i32).collect();
            let r = t(gpu.up(&routes), tokens, s.top, I32);
            let xs: Vec<u16> = (0..tokens * s.k)
                .map(|_| crate::common::to_bf16(lcg.unit() - 0.5))
                .collect();
            let x = t(gpu.up(&xs), tokens, s.k, Bf16);
            let y_at = gpu.zeros(count * s.n * 2);
            let bank = |i: usize| t(banks[i % copies], s.experts, plane, Bf16);
            let y = t(y_at, count, s.n, Bf16);
            let legs = moe::legs(x, bank(0), &y, ExpertTable::RESIDENT);
            #[rustfmt::skip]
            let shape = moe::Shape { routes: count as u32, experts: s.experts as u32, n: s.n as u32, k: s.k as u32 };
            let pick =
                moe::cheapest(&ctx, x, bank(0), r, &y, ExpertTable::RESIDENT).expect("prices");
            let mut line = format!("{} r/e {per} ({count} routes):", s.name);
            for leg in legs {
                let run = |c: &Ctx, i: usize| {
                    let mut y = y;
                    moe::matmul_select_on(c, leg, x, bank(i), r, &mut y, ExpertTable::RESIDENT)
                        .unwrap_or_else(|e| panic!("{leg:?}: {e}"));
                };
                let ts = times(&gpu, &ctx, &run);
                let mark = if leg == pick { "*" } else { "" };
                line += &format!(
                    " {leg:?}{mark} {:.1} [{:.1}..{:.1}] priced {:.1}",
                    ts[ROUNDS / 2],
                    ts[0],
                    ts[ROUNDS - 1],
                    moe::price(leg, shape, cores, rates) / 1e3
                );
            }
            eprintln!("{line}");
        }
    }
}
