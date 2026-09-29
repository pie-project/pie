//! Prefill fire timing on a whole model (temporary bench). `PIE_XLA_ARTIFACT`
//! names the model; `PF_LENS` the prompt lengths (default 128,1024,2048),
//! `PF_CHUNK` the most tokens a fire carries (default 2048), `PF_LANES` how
//! many lanes prefill together, `PF_CTX` the context.

use std::time::Instant;

mod common;

use engine_xla::{Boot, DeviceBoot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Platform, Request};

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[test]
fn prefill_fires_are_timed() {
    let Some(m) = common::model() else {
        return;
    };
    let (checkpoint, sku, contract) = (m.checkpoint.clone(), m.sku, &m.contract);
    let trace = (sku.trace)(Platform::Xla);
    let word = |query_len: u32| (sku.classify)(&Request::new(query_len, false));
    let chunk = env_u32("PF_CHUNK", 2048);
    let lanes = env_u32("PF_LANES", 1);
    let context = env_u32("PF_CTX", 20480);
    let lens: Vec<u32> = std::env::var("PF_LENS")
        .unwrap_or_else(|_| "128,1024,2048".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let _device = engine_xla::bench::lock_device();
    let mut shell = Shell::load(Boot {
        trace,
        contract,
        checkpoint: &checkpoint,
        budget: Budget::new(64, chunk),
        page_size: 16,
        context,
        slots: 64,
        pages: (lanes.max(2) * context / 16) + 64,
        device: &DeviceBoot::default(),
        patches: None,
    })
    .expect("the shell loads");
    let mut rng = 12345u64;
    let mut next = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        1000 + ((rng >> 33) % 199000) as u32
    };
    for &len in &lens {
        for rep in 0..3 {
            let prompts: Vec<Vec<u32>> = (0..lanes).map(|_| (0..len).map(|_| next()).collect()).collect();
            for slot in 0..lanes {
                shell.open(slot).unwrap();
            }
            let per = (chunk / lanes).max(1);
            let started = Instant::now();
            let mut at = 0u32;
            let mut fires = 0;
            while at < len {
                let end = (at + per).min(len);
                let ls: Vec<Lane<'_>> = prompts
                    .iter()
                    .enumerate()
                    .map(|(slot, p)| Lane {
                        slot: slot as u32,
                        word: word(end - at),
                        tokens: &p[at as usize..end as usize],
                    })
                    .collect();
                let t = Instant::now();
                shell.fire(&ls).expect("fires");
                fires += 1;
                if rep == 2 {
                    eprintln!("   fire {fires} rows {} : {:.2} ms", (end - at) * lanes, t.elapsed().as_secs_f64() * 1e3);
                }
                at = end;
            }
            let dt = started.elapsed().as_secs_f64();
            eprintln!(
                "PF len {len} lanes {lanes} rep {rep}: {:.1} ms, {:.0} tok/s",
                dt * 1e3,
                f64::from(len * lanes) / dt
            );
        }
    }
}

/// `PF_DEC="lanes:ctx,..."`: prefill each lane to `ctx` then time decode
/// steps (median of 24).
#[test]
fn decode_steps_are_timed() {
    let Ok(spec) = std::env::var("PF_DEC") else {
        return;
    };
    let Some(m) = common::model() else {
        return;
    };
    let (checkpoint, sku, contract) = (m.checkpoint.clone(), m.sku, &m.contract);
    let trace = (sku.trace)(Platform::Xla);
    let word = |query_len: u32| (sku.classify)(&Request::new(query_len, false));
    let context = env_u32("PF_CTX", 20480);
    let _device = engine_xla::bench::lock_device();
    let mut shell = Shell::load(Boot {
        trace,
        contract,
        checkpoint: &checkpoint,
        budget: Budget::new(64, 2048),
        page_size: 16,
        context,
        slots: 64,
        pages: 16 * context / 16 + 64,
        device: &DeviceBoot::default(),
        patches: None,
    })
    .expect("the shell loads");
    for item in spec.split(',') {
        let (lanes, ctx) = item.split_once(':').unwrap();
        let (lanes, ctx): (u32, u32) = (lanes.parse().unwrap(), ctx.parse().unwrap());
        for slot in 0..lanes {
            shell.open(slot).unwrap();
            let mut at = 0;
            while at < ctx {
                let end = (at + 2048).min(ctx);
                let toks: Vec<u32> = (at..end).map(|i| 1000 + (i * 7919 + slot * 31) % 150000).collect();
                shell
                    .fire(&[Lane { slot, word: word(end - at), tokens: &toks }])
                    .expect("prefill");
                at = end;
            }
        }
        let toks: Vec<[u32; 1]> = (0..lanes).map(|l| [5000 + l]).collect();
        let mut times = Vec::new();
        for step in 0..28 {
            let ls: Vec<Lane<'_>> = (0..lanes)
                .map(|slot| Lane { slot, word: word(1), tokens: &toks[slot as usize] })
                .collect();
            let t = Instant::now();
            shell.fire(&ls).expect("decode");
            if step >= 4 {
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!("DEC lanes {lanes} ctx {ctx}: median {:.2} ms/step", times[times.len() / 2]);
    }
}
