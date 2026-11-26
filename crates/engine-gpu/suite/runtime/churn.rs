//! The pool's governor commits and releases segments while a load serves:
//! the bodies the load armed before ready replay across every commit and
//! release (a backend that keeps recordings rebinds what they bind of the
//! pool, never records one again), no fire after ready goes unarmed, and
//! every logit is the eager walk's.

use crate::backend::{Lane, Seated, Shell};
use crate::model_logits::{CONTEXT, argmax, fixture, load, prompts, same, word};

/// Prefills two seats, then decodes both `steps` times; before each decode
/// `churn` may move the pool's segments. Every logit row, in order.
fn serve(
    shell: &mut Shell,
    trace: &poem_ir::Trace,
    steps: usize,
    mut churn: impl FnMut(&mut Shell, usize),
) -> Vec<Vec<f32>> {
    let mut fed: [Vec<u32>; 2] = prompts();
    let mut held = [0u32; 2];
    let mut rows = Vec::new();
    for lane in 0..2usize {
        let seated = Seated {
            held: Some(held[lane]),
            ..Seated::of(Lane {
                slot: lane as u32,
                word: word(trace, fed[lane].len() as u32),
                tokens: &fed[lane],
            })
        };
        let got = shell.fire_seated(&[seated]).expect("the prefill lands");
        held[lane] += fed[lane].len() as u32;
        fed[lane] = vec![argmax(&got[0])];
        rows.extend(got);
    }
    for step in 0..steps {
        churn(shell, step);
        let seated: Vec<Seated<'_>> = (0..2usize)
            .map(|lane| Seated {
                held: Some(held[lane]),
                ..Seated::of(Lane {
                    slot: lane as u32,
                    word: word(trace, 1),
                    tokens: &fed[lane],
                })
            })
            .collect();
        let got = shell.fire_seated(&seated).expect("the decode lands");
        for lane in 0..2 {
            held[lane] += 1;
            fed[lane] = vec![argmax(&got[lane])];
        }
        rows.extend(got);
    }
    rows
}

/// Segment commits and releases while serving: the armed bodies replay
/// across them, none is recorded after ready and no fire goes unarmed, and
/// the logits are the eager walk's.
#[test]
fn armed_bodies_replay_across_segment_churn() {
    const STEPS: usize = 8;
    let fixture = fixture("qwen3-micro-text", None);
    let Some(mut eager) = load(&fixture, CONTEXT) else {
        return;
    };
    eager.walk_every_fire();
    let want = serve(&mut eager, &fixture.trace, STEPS, |_, _| {});
    drop(eager);

    let Some(mut shell) = load(&fixture, CONTEXT) else {
        return;
    };
    shell.record_bodies(None).expect("the load arms its bodies");
    // The load is done: serving counts from here.
    shell.recorded_serving();
    shell.unkept_serving();
    shell.eager_serving();
    shell.serving = true;
    let segments = shell
        .chunk_facts()
        .expect("the pool states its chunks")
        .segments;
    let resizes = segments.growth != engine::chunk::Growth::Fixed;
    if !resizes {
        eprintln!(
            "the pool is laid whole on this device: no segment commits or releases while it serves"
        );
    }
    let (least, most) = (segments.committed, segments.max);
    let got = serve(&mut shell, &fixture.trace, STEPS, |shell, step| {
        if !resizes {
            return;
        }
        // Every other step backs every segment, the steps between give back
        // all the load did not back (no lane's page lies past them).
        if step % 2 == 0 {
            shell.commit_segments(most).expect("the segments commit");
        } else {
            shell
                .release_segments(least)
                .expect("the segments are given back");
        }
    });
    assert_eq!(
        shell.recorded_serving(),
        0,
        "a fire after ready recorded a body: the load arms every one before ready"
    );
    if <crate::backend::Api as engine_gpu::serve::Serve>::RECORDS {
        assert_eq!(
            shell.unkept_serving(),
            0,
            "a fire after ready walked with no armed body and was not stated eager"
        );
    }
    same("the churned load against the eager walk", &got, &want);
}
