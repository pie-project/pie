//! A load lays each arm's readout seat for what a fire reads out (its
//! readout ceiling's rows, not every token's), and lowers its top rung
//! until every reservation a rung needs in one buffer fits the device's
//! largest: a device that binds less loads a narrower top and serves.

use crate::backend::{Lane, Seated, Shell};
use crate::model_logits::{CONTEXT, SEATS, argmax, fixture, load, load_capped, word};

const MODEL: &str = "qwen3-micro-text";

/// The top rung's tokens, as the pool states its rungs.
fn top(shell: &Shell) -> u32 {
    shell
        .pool_chunk_facts()
        .act_chunks
        .last()
        .map(|&(tokens, _)| tokens)
        .expect("the load states its rungs")
}

/// The most bytes one of the load's reservations takes.
fn widest(shell: &Shell) -> u64 {
    shell.span_bytes().max(shell.readout_bytes())
}

/// A dense model's readout seat holds what a fire reads out, its readout
/// ceiling's rows of each output, not every token's: wide enough for the
/// logits of the ceiling's rows, and short of a seat at the fire's tokens.
#[test]
fn the_readout_seat_is_the_readout_ceilings_rows_not_the_tokens() {
    let fixture = fixture(MODEL, None);
    let Some(shell) = load(&fixture, CONTEXT) else {
        return;
    };
    let tokens = u64::from(top(&shell));
    // A logit row of the readout's width, at its narrowest element (bf16)
    // and its widest (f32).
    let width = shell.out_width().expect("a readout width");
    let ceiling = shell.readouts_ceiling();
    assert!(2 * ceiling <= tokens, "the ceiling is short of the fire's tokens");
    let seat = shell.readout_bytes();
    assert!(
        seat >= ceiling * width * 2,
        "the seat of {seat} bytes holds the ceiling's {ceiling} rows of {width} logits"
    );
    assert!(
        seat < tokens * width * 2,
        "the seat of {seat} bytes is laid at the ceiling, not at every one of {tokens} tokens"
    );
}

/// A device whose largest buffer is short of the load's widest reservation
/// gets a narrower top rung whose every reservation fits that buffer, and
/// the load serves at it.
#[test]
fn a_device_that_binds_less_gets_a_top_whose_every_reservation_fits() {
    let fixture = fixture(MODEL, None);
    let Some(open) = load(&fixture, CONTEXT) else {
        return;
    };
    let (asked, cap) = (top(&open), widest(&open) - 1);
    drop(open);
    // A pool of three such buffers beside the span, in chunks a sixty-fourth
    // of one: no more segments than the kernels bind, each of many chunks.
    let Some(mut shell) = load_capped(&fixture, CONTEXT, cap, 3 * cap, cap / 64) else {
        return;
    };
    let cut = top(&shell);
    assert!(cut < asked, "the top {cut} is cut below the {asked} the device took whole");
    assert!(
        shell.span_bytes() <= cap && shell.readout_bytes() <= cap,
        "every reservation fits the {cap}-byte buffer: span {}, readout seat {}",
        shell.span_bytes(),
        shell.readout_bytes()
    );
    // A fire outside the runtime commits no segment: commit the pool's.
    let segments = shell.pool_chunk_facts().segments.max;
    shell.commit_segments(segments).expect("the segments commit");
    // Two lanes prefill within the cut top, then decode together.
    let rows = cut.min(16) / 2;
    let prompts: Vec<Vec<u32>> = (0..2u32)
        .map(|lane| (0..rows).map(|at| (at * 37 + lane * 11 + 5) % 256).collect())
        .collect();
    let seated: Vec<Seated<'_>> = (0..2u32)
        .map(|lane| Seated {
            held: Some(0),
            ..Seated::of(Lane {
                slot: lane,
                word: word(&fixture.trace, rows),
                tokens: &prompts[lane as usize],
            })
        })
        .collect();
    let got = shell.fire_seated(&seated).expect("the prefill lands");
    let fed: Vec<Vec<u32>> = got.iter().map(|row| vec![argmax(row)]).collect();
    let seated: Vec<Seated<'_>> = (0..2u32)
        .map(|lane| Seated {
            held: Some(rows),
            ..Seated::of(Lane {
                slot: lane,
                word: word(&fixture.trace, 1),
                tokens: &fed[lane as usize],
            })
        })
        .collect();
    let got = shell.fire_seated(&seated).expect("the decode lands");
    assert!(got.iter().all(|row| row.iter().all(|v| v.is_finite())));
    assert!(SEATS >= 2);
}
