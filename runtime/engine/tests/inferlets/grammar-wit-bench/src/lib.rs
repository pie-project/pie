//! Measures grammar resource calls through the actual WIT component boundary.

use std::hint::black_box;
use std::time::Instant;

use inferlet::grammar::{Grammar, Matcher};
use inferlet::serde_json;
use inferlet::{Context, Result};

const GRAMMAR: &str = "root ::= [a-z]*";
const ACCEPTED_TOKEN_ID: u32 = b'a' as u32;

fn default_iterations() -> usize {
    1_000
}

fn default_rounds() -> usize {
    7
}

fn median(mut samples: Vec<u64>) -> u64 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn per_call(elapsed_ns: u128, iterations: usize) -> u64 {
    (elapsed_ns / iterations.max(1) as u128) as u64
}

#[inferlet::main]
async fn main(input: String) -> Result<String> {
    let input: serde_json::Value = serde_json::from_str(&input).unwrap_or(serde_json::Value::Null);
    let iterations = input
        .get("iterations")
        .and_then(|value| value.as_u64())
        .map(|value| value as usize)
        .unwrap_or_else(default_iterations)
        .max(1);
    let rounds = input
        .get("rounds")
        .and_then(|value| value.as_u64())
        .map(|value| value as usize)
        .unwrap_or_else(default_rounds)
        .max(1);

    let rejection_probe = Matcher::new(&Grammar::from_ebnf(r#"root ::= "a""#)?);
    if rejection_probe.accept_tokens(&[3]).is_ok() {
        return Err("Matcher::accept_tokens swallowed a rejected token".to_string());
    }

    let grammar = Grammar::from_ebnf(GRAMMAR)?;
    let matcher = Matcher::new(&grammar);
    let mask_words = matcher.mask().len();

    let mut setup_samples = Vec::with_capacity(rounds);
    let mut mask_samples = Vec::with_capacity(rounds);
    let mut accept_samples = Vec::with_capacity(rounds);
    let mut combined_samples = Vec::with_capacity(rounds);

    for _ in 0..rounds {
        let setup_iterations = iterations.min(100);
        let start = Instant::now();
        for _ in 0..setup_iterations {
            let grammar = Grammar::from_ebnf(black_box(GRAMMAR))?;
            let matcher = Matcher::new(&grammar);
            black_box(matcher.is_terminated());
        }
        setup_samples.push(per_call(start.elapsed().as_nanos(), setup_iterations));

        let start = Instant::now();
        for _ in 0..iterations {
            black_box(matcher.mask());
        }
        mask_samples.push(per_call(start.elapsed().as_nanos(), iterations));

        matcher.reset();
        let start = Instant::now();
        for _ in 0..iterations {
            matcher
                .accept_tokens(black_box(&[ACCEPTED_TOKEN_ID]))
                .context("accept-tokens")?;
        }
        accept_samples.push(per_call(start.elapsed().as_nanos(), iterations));

        matcher.reset();
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(matcher.mask());
            matcher
                .accept_tokens(black_box(&[ACCEPTED_TOKEN_ID]))
                .context("mask+accept")?;
        }
        combined_samples.push(per_call(start.elapsed().as_nanos(), iterations));
    }

    Ok(serde_json::json!({
        "iterations": iterations,
        "rounds": rounds,
        "mask_words": mask_words,
        "warm_setup_ns": median(setup_samples),
        "mask_ns": median(mask_samples),
        "accept_ns": median(accept_samples),
        "mask_accept_ns": median(combined_samples),
    })
    .to_string())
}
