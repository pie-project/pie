//! A whole model on the device: load a snapshot, prefill a prompt, decode
//! greedily, and check a decode walked token by token reads what the
//! prefill read. Asked for with `PIE_XLA_SNAPSHOT` (a Hugging Face snapshot
//! directory) and `PIE_XLA_SKU` (the catalog row that reads it).

use std::time::Instant;

mod common;

use engine_xla::{Boot, DeviceBoot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Platform, Request};

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

#[test]
fn a_prompt_is_answered_and_its_decode_agrees_with_its_prefill() {
    let Some(m) = common::model() else {
        eprintln!("not asked: set PIE_XLA_SNAPSHOT + PIE_XLA_SKU or PIE_XLA_ARTIFACT");
        return;
    };
    let (checkpoint, sku, contract) = (m.checkpoint.clone(), m.sku, &m.contract);
    let trace = (sku.trace)(Platform::Xla);
    let word = |query_len: u32| (sku.classify)(&Request::new(query_len, false));
    let context = 512;

    let _device = engine_xla::bench::lock_device();
    let booted = Instant::now();
    let mut shell = Shell::load(Boot {
        trace,
        contract,
        checkpoint: &checkpoint,
        budget: Budget::new(4, context),
        page_size: 16,
        context,
        slots: 4,
        pages: 4 * context / 16,
        device: &DeviceBoot::default(),
        patches: None,
    })
    .expect("the shell loads");
    eprintln!(
        "loaded {} on {} in {:.1}s",
        sku.name,
        shell.device().kind(),
        booted.elapsed().as_secs_f64()
    );

    let tokenizer = common::tokenizer(&m);
    let prompt = tokenizer.encode("The capital of France is");
    eprintln!("prompt ids {prompt:?}");

    shell.open(0).expect("slot 0 opens");
    let started = Instant::now();
    let rows = shell
        .fire(&[Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: &prompt,
        }])
        .expect("the prefill fires");
    eprintln!(
        "prefill of {} tokens: {:.1}s (compiles included)",
        prompt.len(),
        started.elapsed().as_secs_f64()
    );
    let first = rows[0].clone();
    assert!(
        first.iter().all(|v| v.is_finite()),
        "the prefill's logits are finite"
    );
    let mut produced = vec![argmax(&first)];
    let mut step_times = Vec::new();
    for _ in 0..15 {
        let fed = [*produced.last().expect("a token")];
        let at = Instant::now();
        let rows = shell
            .fire(&[Lane {
                slot: 0,
                word: word(1),
                tokens: &fed,
            }])
            .expect("a decode fires");
        step_times.push(at.elapsed().as_secs_f64());
        produced.push(argmax(&rows[0]));
    }
    let text = tokenizer.decode(&produced, true);
    eprintln!("gen {produced:?}\n  => {text:?}");
    eprintln!(
        "decode steps (s): first {:.3}, then median {:.4}",
        step_times[0],
        {
            let mut t = step_times[1..].to_vec();
            t.sort_by(f32_order);
            t[t.len() / 2]
        }
    );

    // The same prompt walked one token at a time answers what the prefill did.
    shell.open(1).expect("slot 1 opens");
    let mut last = Vec::new();
    for id in &prompt {
        let fed = [*id];
        last = shell
            .fire(&[Lane {
                slot: 1,
                word: word(1),
                tokens: &fed,
            }])
            .expect("a teacher-forced fire returns")
            .remove(0);
    }
    let worst = first
        .iter()
        .zip(&last)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!(
        "prefill vs token-by-token: argmax {} vs {}, max |Δlogit| {worst}",
        argmax(&first),
        argmax(&last)
    );
    assert_eq!(
        argmax(&first),
        argmax(&last),
        "the two walks pick the same token"
    );

    // Three prompts in one fire (prefill, then batched decode) answer what
    // each answers alone: padding lanes and batching leave the real rows be.
    let prompts: Vec<Vec<u32>> = [
        "Water boils at",
        "The largest planet in the solar system is",
        "1, 2, 3, 4,",
    ]
    .iter()
    .map(|p| tokenizer.encode(p))
    .collect();
    let mut alone: Vec<Vec<u32>> = Vec::new();
    for prompt in &prompts {
        shell.open(2).expect("slot 2 opens");
        let mut row = shell
            .fire(&[Lane {
                slot: 2,
                word: word(prompt.len() as u32),
                tokens: prompt,
            }])
            .expect("a lone prefill fires")
            .remove(0);
        let mut produced = Vec::new();
        for _ in 0..6 {
            let next = argmax(&row);
            produced.push(next);
            row = shell
                .fire(&[Lane {
                    slot: 2,
                    word: word(1),
                    tokens: &[next],
                }])
                .expect("a lone decode fires")
                .remove(0);
        }
        alone.push(produced);
    }
    for slot in 0..3 {
        shell.open(slot).expect("a slot opens");
    }
    let rows = shell
        .fire(
            &prompts
                .iter()
                .enumerate()
                .map(|(slot, p)| Lane {
                    slot: slot as u32,
                    word: word(p.len() as u32),
                    tokens: p,
                })
                .collect::<Vec<_>>(),
        )
        .expect("a batched prefill fires");
    let mut next: Vec<u32> = rows.iter().map(|r| argmax(r)).collect();
    let mut together: Vec<Vec<u32>> = vec![Vec::new(); 3];
    for _ in 0..6 {
        for (lane, token) in next.iter().enumerate() {
            together[lane].push(*token);
        }
        let fed: Vec<[u32; 1]> = next.iter().map(|t| [*t]).collect();
        let rows = shell
            .fire(
                &fed.iter()
                    .enumerate()
                    .map(|(slot, t)| Lane {
                        slot: slot as u32,
                        word: word(1),
                        tokens: t,
                    })
                    .collect::<Vec<_>>(),
            )
            .expect("a batched decode fires");
        next = rows.iter().map(|r| argmax(r)).collect();
    }
    for (lane, (a, b)) in alone.iter().zip(&together).enumerate() {
        eprintln!(
            "lane {lane}: alone {:?} / together {:?}",
            tokenizer.decode(a, true),
            tokenizer.decode(b, true)
        );
    }
    assert_eq!(
        alone, together,
        "a batch answers what each lane answers alone"
    );
}

fn f32_order(a: &f64, b: &f64) -> std::cmp::Ordering {
    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
}
