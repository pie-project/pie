//! Prompts of row counts the arm walk lays no composition of (past the
//! rows it walks one a count, short of the next power of two it walks):
//! each serving fire of such a length replays its armed family with a fill
//! a lay-only pass laid on the host, once a length, never walked for at
//! serving, and every logit is the eager walk's.

use crate::backend::{Lane, Seated, Shell};
use crate::model_logits::{CONTEXT, argmax, fixture, load, same, word};

/// The lengths: past the vector GEMM's widest row tile, short of the
/// powers of two the walk lays.
const LENGTHS: [u32; 4] = [17, 19, 23, 27];

/// Each length's prompt in its own seat, then one decode of each: every
/// logit row, in order.
pub fn serve(shell: &mut Shell, trace: &poem_ir::Trace) -> Vec<Vec<f32>> {
    let mut rows = Vec::new();
    for (seat, &length) in LENGTHS.iter().enumerate() {
        let tokens: Vec<u32> = (0..length).map(|at| (at * 37 + 11 + seat as u32) % 256).collect();
        let got = shell
            .fire_seated(&[Seated {
                held: Some(0),
                ..Seated::of(Lane {
                    slot: seat as u32,
                    word: word(trace, length),
                    tokens: &tokens,
                })
            }])
            .expect("the prompt lands");
        let next = [argmax(&got[0])];
        rows.extend(got);
        let got = shell
            .fire_seated(&[Seated {
                held: Some(length),
                ..Seated::of(Lane {
                    slot: seat as u32,
                    word: word(trace, 1),
                    tokens: &next,
                })
            }])
            .expect("the decode lands");
        rows.extend(got);
    }
    rows
}

#[test]
fn an_unwalked_length_replays_its_family_with_a_laid_fill() {
    assert!(LENGTHS.iter().all(|&length| length < CONTEXT));
    let fixture = fixture("qwen3-micro-text", None);
    let Some(mut eager) = load(&fixture, CONTEXT) else {
        return;
    };
    eager.walk_every_fire();
    let want = serve(&mut eager, &fixture.trace);
    drop(eager);

    let Some(mut shell) = load(&fixture, CONTEXT) else {
        return;
    };
    shell.record_bodies(None).expect("the load arms its bodies");
    shell.recorded_serving();
    shell.unkept_serving();
    shell.eager_serving();
    shell.filled_serving();
    shell.laid_serving();
    shell.serving = true;
    let got = serve(&mut shell, &fixture.trace);
    assert_eq!(shell.recorded_serving(), 0, "a fire after ready recorded a body");
    if <crate::backend::Api as engine_gpu::serve::Serve>::RECORDS {
        assert_eq!(
            shell.unkept_serving(),
            0,
            "a fire after ready walked with no armed body and was not stated eager"
        );
        assert_eq!(
            shell.filled_serving(),
            0,
            "a serving fire walked for its fill (filled_in_fire)"
        );
        assert!(
            shell.laid_serving() >= 1,
            "no serving fire of an unwalked length laid its fill (laid_at_first_fire)"
        );
    }
    same("the laid fills against the eager walk", &got, &want);
}
