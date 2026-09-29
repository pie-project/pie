//! TEMP bench (moe agent): decode step time by width. `PIE_XLA_BENCH_MODE`:
//! `fixed` (lane i fed token 100+i every step, as a_batch_decodes) or
//! `vllm` (lane i prompted "{i}: The capital of France is" and decoding its
//! own greedy tokens, as the vLLM bench script: a first pass records the
//! tokens, a second replays them with the logits kept on the device).

mod common;

use std::time::Instant;

use engine_xla::{Boot, DeviceBoot, Lane, Seated, Shell};
use model_compiler::Budget;
use model_dsl::{Platform, Request};

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best as u32
}

#[test]
fn gptoss_decode_bench_tmp() {
    if std::env::var("PIE_XLA_BENCH").is_err() {
        return;
    }
    let Some(m) = common::model() else { return };
    let sku = m.sku;
    let word = |query_len: u32| (sku.classify)(&Request::new(query_len, false));
    let widths: Vec<u32> = std::env::var("PIE_XLA_BENCH_WIDTHS")
        .ok()
        .map(|s| s.split(',').filter_map(|w| w.parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 8, 32, 64]);
    let mode = std::env::var("PIE_XLA_BENCH_MODE").unwrap_or_else(|_| "fixed".into());
    let most = *widths.iter().max().unwrap_or(&1);
    let context = 1024;
    let tokenizer = common::tokenizer(&m);
    let _device = engine_xla::bench::lock_device();
    let mut shell = Shell::load(Boot {
        trace: (sku.trace)(Platform::Xla),
        contract: &m.contract,
        checkpoint: &m.checkpoint,
        budget: Budget::new(most, 2048),
        page_size: 16,
        context,
        slots: most,
        pages: 2 * most * context / 16,
        device: &DeviceBoot::default(),
        patches: None,
    })
    .expect("the shell loads");
    let kept_step = |shell: &mut Shell, seated: &[Seated<'_>]| {
        let fired = shell.fire_kept(seated).expect("a decode fires");
        if let Some(kept) = fired.kept {
            kept.logits
                .ready()
                .and_then(engine_xla::pjrt::Event::wait)
                .expect("the logits compute");
        }
    };
    for &width in &widths {
        if mode == "vllm" {
            let steps_n = 128usize;
            let prompts: Vec<Vec<u32>> = (0..width)
                .map(|i| tokenizer.encode(&format!("{i}: The capital of France is")))
                .collect();
            let mut tokens: Vec<Vec<u32>> = vec![Vec::new(); width as usize];
            for pass in 0..2 {
                for slot in 0..width {
                    shell.open(slot).expect("a slot opens");
                    let p = &prompts[slot as usize];
                    let rows = shell
                        .fire(&[Lane { slot, word: word(p.len() as u32), tokens: p }])
                        .expect("a prefill fires");
                    if pass == 0 {
                        let last = &rows[0];
                        let w = 201_088;
                        assert_eq!(last.len() % w, 0, "gpt-oss rows");
                        tokens[slot as usize].push(argmax(&last[last.len() - w..]));
                    }
                }
                let started = Instant::now();
                for step in 0..steps_n - 1 {
                    let fed: Vec<[u32; 1]> =
                        (0..width as usize).map(|i| [tokens[i][step]]).collect();
                    let lanes: Vec<Lane<'_>> = fed
                        .iter()
                        .enumerate()
                        .map(|(slot, t)| Lane { slot: slot as u32, word: word(1), tokens: t })
                        .collect();
                    if pass == 0 {
                        let rows = shell.fire(&lanes).expect("a decode fires");
                        for (i, row) in rows.iter().enumerate() {
                            let t = argmax(row);
                            tokens[i].push(t);
                        }
                    } else {
                        let seated: Vec<Seated<'_>> = lanes.iter().copied().map(Seated::of).collect();
                        kept_step(&mut shell, &seated);
                    }
                }
                let per = started.elapsed().as_secs_f64() / (steps_n - 1) as f64;
                if pass == 1 {
                    let distinct: std::collections::BTreeSet<&Vec<u32>> = tokens.iter().collect();
                    eprintln!(
                        "vllm width {width:>3}: {:.2} ms/step (kept, synced; {} distinct continuations; lane 0: {:?})",
                        per * 1e3,
                        distinct.len(),
                        tokenizer.decode(&tokens[0][..16.min(tokens[0].len())], true)
                    );
                } else {
                    eprintln!("vllm width {width:>3}: first pass (host logits) {:.2} ms/step", per * 1e3);
                }
            }
            continue;
        }
        let prompt: Vec<u32> = (0..32).map(|i| 1000 + i).collect();
        let fed: Vec<[u32; 1]> = (0..width).map(|i| [100 + i]).collect();
        let lanes: Vec<Lane<'_>> = fed
            .iter()
            .enumerate()
            .map(|(slot, t)| Lane { slot: slot as u32, word: word(1), tokens: t })
            .collect();
        let seated: Vec<Seated<'_>> = lanes.iter().copied().map(Seated::of).collect();
        let steps = 20;
        // Two passes over the same steps: the first traces (and compiles)
        // every shape the steps reach (a new page bound is a new program),
        // the second is timed.
        let mut per = 0.0;
        let mut kept = 0.0;
        for _pass in 0..2 {
            for slot in 0..width {
                shell.open(slot).expect("a slot opens");
                shell
                    .fire(&[Lane { slot, word: word(prompt.len() as u32), tokens: &prompt }])
                    .expect("a prefill fires");
            }
            shell.fire(&lanes).expect("a decode fires");
            let started = Instant::now();
            for _ in 0..steps {
                shell.fire(&lanes).expect("a decode fires");
            }
            per = started.elapsed().as_secs_f64() / f64::from(steps);
            kept_step(&mut shell, &seated);
            let started = Instant::now();
            for _ in 0..steps {
                kept_step(&mut shell, &seated);
            }
            kept = started.elapsed().as_secs_f64() / f64::from(steps);
        }
        eprintln!(
            "width {width:>3}: {:.2} ms/step, {:.0} tok/s; logits kept on the device {:.2} ms/step",
            per * 1e3,
            f64::from(width) / per,
            kept * 1e3
        );
    }
}
