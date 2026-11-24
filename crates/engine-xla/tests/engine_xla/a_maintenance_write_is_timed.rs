//! `PIE_XLA_MAINT_BENCH=1`: what a slot copy (a fork's) and a host tier
//! restore of n slots cost, for n of 1, 16 and a chunk's worth, at the
//! `Channel` slot and the kv list's slot sizes; the first run (its
//! compile) apart. Each timed op ends on a one-slot read off the device,
//! whose own time is taken off.

use std::time::Instant;

use engine::chunk::{ShapeId, SlotCopy, SlotKind, SlotRole};
use engine::transfer::MemoryDomain;
use engine_xla::store::layout::CHANNEL;

use crate::common_llm::{Model, Refs};

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[xs.len() / 2]
}

#[test]
fn a_maintenance_write_is_timed() {
    if std::env::var_os("PIE_XLA_MAINT_BENCH").is_none() {
        return;
    }
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let t = std::time::Instant::now();
    let mut model = Model::load_with("qwen3-0.6b", Some(2), 512, 7, None);
    eprintln!("MAINT load {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    if let Some(facts) = model.engine.shell().map(|s| s.staging_facts().clone()) {
        eprintln!(
            "MAINT warmed {} maintenance program(s) at load in {:.1} ms; bulk ring {} B",
            facts.maintenance.0, facts.maintenance.1, facts.bulk_bytes
        );
    }
    let device = model.loaded.caps.device.domain;
    let facts = model.loaded.caps.chunks.clone();
    let mut refs = Refs::of(&model);
    let kv = refs
        .shapes(|s| matches!(s.kind, SlotKind::List { window: None, .. }) && s.role == SlotRole::State)[0]
        .id;
    let copy = |model: &mut Model, shape: ShapeId, src, dst, slots: Vec<(u32, u32)>| {
        engine::Tiers::copy(&mut model.engine, &SlotCopy {
                shape,
                src,
                dst,
                slots,
                runs: Vec::new(),
            })
            .expect("the slots move");
    };
    // The barrier: one `Channel` slot off the device to the host tier.
    let sink = refs.take(CHANNEL);
    let barrier = |model: &mut Model| {
        copy(model, CHANNEL, device, MemoryDomain::HostPinned, vec![(sink, 0)]);
    };
    barrier(&mut model);
    let mut lone = Vec::new();
    for _ in 0..9 {
        let t = Instant::now();
        barrier(&mut model);
        lone.push(t.elapsed().as_secs_f64());
    }
    let lone = median(lone);
    eprintln!("MAINT barrier {:.3} ms", lone * 1e3);
    for shape in [CHANNEL, kv] {
        let spec = facts.shape(shape).expect("a stated shape").clone();
        let per_chunk = spec.slots_per_chunk;
        let host_room = (facts.host_chunks * per_chunk).max(1);
        let most = (2 * per_chunk).min(host_room).max(1);
        let counts: Vec<u32> = [1, 16, per_chunk, 2 * per_chunk]
            .into_iter()
            .filter(|&n| n <= most)
            .collect();
        let pool: Vec<u32> = (0..2 * most).map(|_| refs.take(shape)).collect();
        for &n in &counts {
            let pairs: Vec<(u32, u32)> = (0..n as usize)
                .map(|i| (pool[i], pool[most as usize + i]))
                .collect();
            let host: Vec<(u32, u32)> = (0..n).map(|i| (i + 1, pool[i as usize])).collect();
            for (what, src, dst, slots) in [
                ("device copy", device, device, pairs.clone()),
                ("host restore", MemoryDomain::HostPinned, device, host.clone()),
            ] {
                let t = Instant::now();
                copy(&mut model, shape, src, dst, slots.clone());
                barrier(&mut model);
                let first = t.elapsed().as_secs_f64() - lone;
                let mut runs = Vec::new();
                for _ in 0..7 {
                    let t = Instant::now();
                    copy(&mut model, shape, src, dst, slots.clone());
                    barrier(&mut model);
                    runs.push(t.elapsed().as_secs_f64() - lone);
                }
                eprintln!(
                    "MAINT shape {} ({} B a slot) {what} of {n}: {:.3} ms (first run {:.1} ms)",
                    shape.0,
                    spec.slot_bytes,
                    median(runs) * 1e3,
                    first * 1e3
                );
            }
        }
    }
    let t = Instant::now();
    let shell = model.engine.shell().expect("loaded");
    eprintln!(
        "MAINT programs held after the bench: {} ({:.1} ms to ask)",
        shell.device().compiled(),
        t.elapsed().as_secs_f64() * 1e3
    );
}
