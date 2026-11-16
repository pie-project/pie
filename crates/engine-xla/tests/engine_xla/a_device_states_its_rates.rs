//! PROBE (`PIE_XLA_RATES=1`): the device's rates through XLA's own path,
//! what the top rung's estimate reads from the device file's `[rates]`:
//! a 4096^3 bf16 dot (`matmul_nt`), a 256 MiB elementwise pass counted
//! read plus write, and a one-tile program launched back to back. Median
//! of 5 after a warm one. Skipped without a device.

use dtype::Dtype;
use engine_xla::bench::Bench;
use kernels_xla::hlo::Elem;

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[xs.len() / 2]
}

/// Six timings of `one`, the first a warm one dropped: their median.
fn timed(mut one: impl FnMut() -> Option<f64>) -> Option<f64> {
    let mut xs = Vec::new();
    for at in 0..6 {
        let s = one()?;
        if at > 0 {
            xs.push(s);
        }
    }
    Some(median(xs))
}

#[test]
fn a_device_states_its_rates() {
    if std::env::var_os("PIE_XLA_RATES").is_none() {
        return;
    }
    const N: u32 = 4096;
    let Some(dot) = timed(|| {
        let mut b = Bench::new();
        let x = b.zeros(Dtype::Bf16, N, N);
        let w = b.zeros(Dtype::Bf16, N, N);
        let y = b.zeros(Dtype::Bf16, N, N);
        b.time_pipelined(50, |ctx| {
            engine_xla::lower::lower(ctx, |l| {
                let xv = l.read(x)?.val;
                let wv = l.read(w)?.val;
                let v = l.f().matmul_nt(xv, wv, Elem::Bf16)?;
                l.land(y, v)
            })
        })
        .expect("times")
    }) else {
        return;
    };
    // 256 MiB of bf16: 8192 rows of 16384.
    let (rows, width) = (8192u32, 16384u32);
    let bytes = 2.0 * f64::from(rows) * f64::from(width) * 2.0;
    let copy = timed(|| {
        let mut b = Bench::new();
        let x = b.zeros(Dtype::Bf16, rows, width);
        let y = b.zeros(Dtype::Bf16, rows, width);
        b.time_pipelined(50, |ctx| {
            engine_xla::lower::lower(ctx, |l| {
                let xv = l.read(x)?.val;
                let v = l.f().neg(xv);
                l.land(y, v)
            })
        })
        .expect("times")
    })
    .expect("a device");
    let tiny = |pipelined: bool| {
        timed(|| {
            let mut b = Bench::new();
            let x = b.zeros(Dtype::Bf16, 8, 128);
            let y = b.zeros(Dtype::Bf16, 8, 128);
            let body = |ctx: &engine_xla::emit::Ctx<'_>| {
                engine_xla::lower::lower(ctx, |l| {
                    let xv = l.read(x)?.val;
                    let v = l.f().neg(xv);
                    l.land(y, v)
                })
            };
            if pipelined {
                b.time_pipelined(1000, body)
            } else {
                b.time(1000, body)
            }
            .expect("times")
        })
        .expect("a device")
    };
    let launch = tiny(true);
    let waited = tiny(false);
    let flops = 2.0 * f64::from(N).powi(3);
    eprintln!(
        "RATES dot 4096^3 bf16 {:.3} ms = {:.0} flops/ns; copy 256 MiB {:.3} ms = {:.0} bytes/ns \
         (read plus write); launch back to back {:.0} ns, waited {:.0} ns",
        dot * 1e3,
        flops / (dot * 1e9),
        copy * 1e3,
        bytes / (copy * 1e9),
        launch * 1e9,
        waited * 1e9
    );
}
