//! The ids an MTP head is fed on a prompt, as Qwen3.5's text feeds them:
//! the lane's next token at each row (`inputs.next_tokens()`), and at the
//! row the pass samples, the token the trunk picks there. That is vLLM's
//! speculator's shifted ids (`target_input_ids[1:]`, then the sampled
//! token): row i of the head's input is the embedding of token i + 1, the
//! last row the sample's. The host forward runs it over a prompt whose last
//! row alone is read out, and its rows are held to the reference.

use engine_cpu::Forward;
use engine_cpu::forward::{Output, Piece, Values};
use poem::{Dtype, ForwardHybrid, Input, Platform, Request, StateRows, Weight, ops};

const VOCAB: u32 = 32;
const WIDTH: u32 = 16;

struct HeadFeed;

impl ForwardHybrid for HeadFeed {
    fn states(&self) -> StateRows {
        StateRows::new()
    }
    fn forward(&self, inputs: Input) {
        let table = Weight::sym("embed", [u64::from(VOCAB), u64::from(WIDTH)], Dtype::F32);
        let head = Weight::sym("head", [u64::from(VOCAB), u64::from(WIDTH)], Dtype::F32);
        let hidden = ops::gather::embed(&inputs.tokens(), &table, VOCAB);
        let logits = ops::linear::lm_head(
            &ops::gather::gather_rows(&hidden, &inputs.readout_rows()),
            &head,
        );
        let sampled = ops::gather::embed(&ops::sampling::argmax(&[&logits]), &table, VOCAB);
        let nexts = ops::gather::embed(&inputs.next_tokens(), &table, VOCAB);
        let fed = ops::gather::scatter_rows(&sampled, &inputs.readout_rows(), &nexts);
        poem::output("logits", &logits, None);
        poem::output("fed", &fed, None);
    }
}

fn plane(name: &str, count: usize) -> Vec<f32> {
    let mut seed = name.bytes().fold(0x9e37_79b9_u64, |h, b| h.wrapping_mul(31) ^ u64::from(b));
    (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    f32::from_bits((bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) << 16)
}

fn f32s(outs: &[Output], name: &str) -> Vec<f32> {
    match outs.iter().find(|out| out.name == name) {
        Some(Output {
            values: Values::F32(v),
            ..
        }) => v.clone(),
        _ => panic!("no f32 output {name}"),
    }
}

#[test]
fn a_draft_head_is_fed_the_shifted_ids() {
    let trace = poem::trace_hybrid("head-feed", &HeadFeed, Platform::Cuda);
    assert!(trace.reads_next_tokens());
    let prompt: [u32; 7] = [5, 17, 2, 30, 9, 9, 21];
    let mut forward = Forward::new(trace.clone(), None, &plane).expect("the miniature loads");
    let word = trace.facts.word(&Request::new(prompt.len() as u32, false));
    let outs = forward.step(&prompt, 0, word).expect("the prompt runs");

    let (v, w) = (VOCAB as usize, WIDTH as usize);
    // The forward holds the table as bf16, as a checkpoint stores it.
    let table: Vec<f32> = plane("embed", v * w).into_iter().map(bf16).collect();
    let logits = f32s(&outs, "logits");
    assert_eq!(logits.len(), v, "the last row alone is read out");
    let sampled = (0..v).fold(0, |best, c| if logits[c] > logits[best] { c } else { best });
    // vLLM's draft ids for this prompt: the target's shifted by one, the
    // sampled token last.
    let ids: Vec<usize> = prompt
        .iter()
        .skip(1)
        .map(|&t| t as usize)
        .chain([sampled])
        .collect();
    let fed = f32s(&outs, "fed");
    assert_eq!(fed.len(), prompt.len() * w, "a fed row a prompt row");
    for (row, &id) in ids.iter().enumerate() {
        assert_eq!(
            &fed[row * w..][..w],
            &table[id * w..][..w],
            "row {row}: the head is fed token {id}, as vLLM's shifted ids feed it"
        );
    }
}

/// A prompt the runtime cuts into pieces at the rung feeds the head what
/// one whole-prompt fire feeds it: each piece's last row takes the next
/// piece's first token (the runtime's lookahead, `Lane::lookahead`), and only
/// the last piece, which reads its row out, takes the sample there.
#[test]
fn a_prompt_cut_into_pieces_feeds_the_head_what_one_fire_does() {
    let trace = poem::trace_hybrid("head-feed", &HeadFeed, Platform::Cuda);
    let prompt: [u32; 11] = [5, 17, 2, 30, 9, 9, 21, 4, 13, 1, 28];
    let word = |rows: usize| trace.facts.word(&Request::new(rows as u32, false));
    let mut whole = Forward::new(trace.clone(), None, &plane).expect("the miniature loads");
    let want = f32s(
        &whole.step(&prompt, 0, word(prompt.len())).expect("the whole prompt runs"),
        "fed",
    );
    for cuts in [vec![4, 11], vec![1, 6, 11], vec![3, 7, 10, 11]] {
        let mut cut = Forward::new(trace.clone(), None, &plane).expect("the miniature loads");
        let mut fed = Vec::new();
        let mut from = 0;
        for &to in &cuts {
            let last = to == prompt.len();
            let piece = Piece {
                lookahead: prompt.get(to).copied(),
                reads_out: last,
            };
            let outs = cut
                .step_piece(&prompt[from..to], from as u32, word(to - from), piece)
                .expect("the piece runs");
            fed.extend(f32s(&outs, "fed"));
            from = to;
        }
        assert_eq!(fed, want, "pieces {cuts:?} feed the head the whole prompt's rows");
    }
}
