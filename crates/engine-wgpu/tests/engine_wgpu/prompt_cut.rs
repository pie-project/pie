//! A prompt lands the same logits, bit for bit, however its rows are cut
//! into passes: a repeated prompt whose prefix the cache holds prefills only
//! the tail past its last whole page, and greedy decoding answers it as it
//! answered the whole prompt.

use crate::backend::{Lane, Seated, Shell};
use crate::model_logits::{CONTEXT, argmax, fixture, load, same, word};
use poem_ir::Trace;

/// Prefills `prompt` in slot `slot` in passes ending at each of `cuts`, then
/// decodes greedily `steps` times: the last prefill row, then every
/// decode's.
fn served(
    shell: &mut Shell,
    trace: &Trace,
    slot: u32,
    prompt: &[u32],
    cuts: &[usize],
    steps: usize,
) -> Vec<Vec<f32>> {
    let mut held = 0u32;
    let mut last = Vec::new();
    let mut from = 0usize;
    for &to in cuts.iter().chain(std::iter::once(&prompt.len())) {
        let tokens = &prompt[from..to];
        let seated = Seated {
            held: Some(held),
            ..Seated::of(Lane {
                slot,
                word: word(trace, tokens.len() as u32),
                tokens,
            })
        };
        last = shell.fire_seated(&[seated]).expect("the prefill lands");
        held += tokens.len() as u32;
        from = to;
    }
    let mut rows = last.clone();
    let mut fed = vec![argmax(&last[0])];
    for _ in 0..steps {
        let seated = Seated {
            held: Some(held),
            ..Seated::of(Lane {
                slot,
                word: word(trace, 1),
                tokens: &fed,
            })
        };
        let got = shell.fire_seated(&[seated]).expect("the decode lands");
        held += 1;
        fed = vec![argmax(&got[0])];
        rows.extend(got);
    }
    rows
}

/// Qwen3-0.6B's widths: the whole prompt's projections take the 64-row
/// tile and its attention the tiled walk; the tails past a page (16 tokens)
/// take the 16-row tile, two or three rows among them, and a tail of fewer
/// rows than a request's tile.
#[test]
fn a_prompt_cut_at_a_page_lands_the_bits_it_lands_whole() {
    let fixture = fixture("qwen3-0.6b", Some(2));
    let Some(mut shell) = load(&fixture, CONTEXT) else {
        return;
    };
    let trace = &fixture.trace;
    let prompt: Vec<u32> = (0..58u32).map(|at| (at * 7919 + 13) % 150_000).collect();
    let whole = served(&mut shell, trace, 0, &prompt, &[], 6);
    for (slot, cuts) in [(1, &[48usize][..]), (2, &[56]), (3, &[16, 32, 48])] {
        let cut = served(&mut shell, trace, slot, &prompt, cuts, 6);
        same(&format!("the prompt cut at {cuts:?}"), &cut, &whole);
    }
}
