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
        rates.peak_flops().unwrap(),
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
    // The grid at every row and depth of the lattice, the attention and
    // scan rates.
    let grid = rates.bf16_flops_per_ns_by_rows_and_depth.clone().unwrap();
    assert_eq!(
        grid.len(),
        engine::rates::LATTICE_ROWS.len() * engine::rates::DEPTHS.len()
    );
    assert_eq!(profile::fire::grid(grid.clone()), Some(grid.clone()));
    eprintln!("grid {grid:?}");
    for rate in [
        rates.attn_at(2048, 128).unwrap(),
        rates.scan_elements_per_ns.unwrap(),
    ] {
        assert!((1.0..1e6).contains(&rate), "{rate}");
    }

    let mut tuned = chip.tuned().clone();
    tuned.rates = rates;
    tuned.measured_this_load = true;
    let profile = chip.with_tuned(tuned);
    const MIB: u64 = 1 << 20;
    let rungs = rung_facts(
        &[(1024, 100 * MIB), (2048, 200 * MIB), (4096, 400 * MIB)],
        512,
        1 << 30,
        300,
        &|rows| rows.div_ceil(64) * 64,
    );
    let counts = Counts {
        weight_bytes: 1 << 30,
        expert_bytes: 0,
        experts: None,
        gemm_shapes: vec![engine::load::GemmShape::dense(4096, 4096, 1 << 30, 1 << 30)],
        head_bytes: 0,
        attn_flops_per_row_key: 1 << 16,
        kv_bytes_per_key: 1 << 16,
        attn_head_dim: 128,
        readout_flops: 1 << 28,
        readout_depth: 4096,
        state_elements_per_row: 1 << 20,
        elementwise_ops: vec![1 << 16],
        attn_dispatches: 28,
        scan_dispatches: 0,
        width: 4096,
    };
    let (top, _) = choose(&rungs, 8 << 30, &profile, "adapter", &counts, 1 << 30).unwrap();
    let line = line_of(&top, top.tokens, top.span_bytes, 8 << 30, profile.tuned());
    assert!(line.contains("by the estimate"), "{line}");
    assert!(line.contains("rates measured this load"), "{line}");
    // Never written: a dump of the profile states no rates.
    assert!(!profile::render::dump(&profile).contains("[rates]"));
}
