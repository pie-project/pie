//! A windowed lane that leaves rows unfolded in window-buffer slots reads
//! what a lane folding every row at once reads, bit for bit: its buffered
//! rows from the slots, the folded ones from its ring, and, where its
//! buffer starts mid-page below the folded index (a prompt of no whole
//! number of pages folded by its own fire, then a decode buffering), the
//! page's folded rows from the slot its first writer filled from the ring.
//! A fold of the buffered rows (and the last fire of a prompt past the
//! ring leaving a tail unfolded) carries on as the eager lane does.
//! Folding every row at once answers what the window lists answered
//! (`a_windowed_model_answers_bit_for_bit_past_its_window`). Skipped
//! without a PJRT plugin.

use crate::common_llm::{Model, Refs, Seq, bits, tokens};

/// The scenario: each step's tokens and, for the lagged lane, its fold.
fn steps(vocab: u32) -> Vec<(Vec<u32>, u32)> {
    let mut out = vec![(tokens(1, 37, vocab), 37)];
    // three decodes buffering, then one folding them and its own row
    for t in 0..3u64 {
        out.push((tokens(10 + t, 1, vocab), 0));
    }
    out.push((tokens(13, 1, vocab), 4));
    out.push((tokens(14, 1, vocab), 0));
    out.push((tokens(15, 1, vocab), 2));
    // a prompt past the ring in fires of 100 rows (each past the ring
    // with the window behind it: own-row pages), the last leaving ten rows
    // of it unfolded
    for piece in 0..7u64 {
        out.push((
            tokens(2 + piece, 100, vocab),
            if piece == 6 { 90 } else { 100 },
        ));
    }
    for t in 0..3u64 {
        out.push((tokens(20 + t, 1, vocab), 1));
    }
    out
}

fn served(lagged: bool) -> Vec<Vec<u32>> {
    let mut model = Model::load("gemma4-e4b-mini-l1", None, 1024, 7);
    let mut refs = Refs::of(&model);
    let mut seq = Seq::new(0);
    let mut out = Vec::new();
    for (tokens, fold) in steps(model.vocab) {
        let rows = tokens.len() as u32;
        let lane = if lagged {
            seq.lane_fold(&model, &mut refs, tokens, fold)
        } else {
            seq.lane(&model, &mut refs, tokens)
        };
        out.extend(model.fire(vec![lane]).iter().map(|v| bits(v)));
        seq.held += rows;
    }
    out
}

#[test]
fn a_lagged_window_fold_reads_what_folding_reads() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let eager = served(false);
    let lagged = served(true);
    assert_eq!(eager.len(), lagged.len());
    for (step, (e, l)) in eager.iter().zip(&lagged).enumerate() {
        assert!(
            e == l,
            "step {step}: the lagged lane reads what the eager lane reads"
        );
    }
}
