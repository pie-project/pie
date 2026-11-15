//! The browser's rates path, run natively on the first adapter: the
//! micro-fires state rates, and the ruling picks a top rung by them.

use engine_gpu::top::{Counts, choose, line_of, rung_facts};
use engine_wgpu::api::DeviceBoot;
use engine_wgpu::device::Context;

#[test]
fn the_micro_fires_state_rates_and_the_ruling_picks_by_them() {
    let Ok(device) = Context::bind(&DeviceBoot::default(), crate::backend::ALLOWANCE) else {
        eprintln!("no wgpu adapter answers; skipped");
        return;
    };
    let (chip, _) = engine_wgpu::chip::resolve(&device.chip(), None);
    let tuning = engine_wgpu::tuning::Tuning::new(chip.clone(), device.cores());
    let rates = engine_wgpu::rates::measure(&device, &tuning).unwrap();
    let (flops, bytes, launch) = (
        rates.bf16_flops_per_ns.unwrap(),
        rates.bytes_per_ns.unwrap(),
        rates.launch_ns.unwrap(),
    );
    eprintln!(
        "{}: {:.1} TFLOP/s bf16, {bytes:.0} GB/s, launch {launch} ns",
        device.chip().name,
        flops / 1e3
    );
    // Any GPU: between a GFLOP/s and a PFLOP/s, a GB/s and 10 TB/s.
    assert!((1.0..1e6).contains(&flops), "{flops}");
    assert!((1.0..1e4).contains(&bytes), "{bytes}");
    assert!(launch > 0);

    let mut tuned = chip.tuned().clone();
    tuned.rates = rates;
    tuned.measured_this_load = true;
    let profile = chip.with_tuned(tuned);
    const MIB: u64 = 1 << 20;
    let rungs = rung_facts(
        &[(1024, 100 * MIB), (2048, 200 * MIB), (4096, 400 * MIB)],
        512,
        1 << 30,
    );
    let counts = Counts {
        weight_bytes: 1 << 30,
        flops_per_token: 1 << 30,
        attn_flops_per_row_key: 1 << 16,
    };
    let top = choose(&rungs, 8 << 30, &profile, "adapter", "m", &counts, 1 << 30).unwrap();
    let line = line_of(
        &top,
        top.tokens,
        top.span_bytes,
        8 << 30,
        "adapter",
        "m",
        profile.tuned(),
    );
    assert!(line.contains("by the estimate"), "{line}");
    assert!(line.contains("rates measured this load"), "{line}");
    // Never written: a dump of the profile states no rates.
    assert!(!profile::render::dump(&profile).contains("[rates]"));
}
