#![cfg(target_vendor = "apple")]

//! The load's walk takes the step set (CONTRACT-BC 24) shape by shape,
//! smallest first, and fires each at the rows and readout tiers its spans
//! hold: a step of any row count its set states, reading out any count of
//! its rows, meets only pipelines built before ready. On an M1 a c64 step
//! met `dense_gemv_t_rows8_bfloat16` (the 5-8-row gemv) unbuilt after ready
//! and was refused.

use engine_metal::serve::Seated;
use engine_metal::{Lane, Shell};

use super::a_scattered_pool_serves_the_same_logits::{fixture_of, load_with, word};

#[test]
fn a_load_walks_every_readout_tier_its_set_states() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);
    let mut shell: Shell = load_with(&fixture, 256);
    let steps = shell.step_shapes();
    assert!(!steps.is_empty(), "the load states a step set");
    let most = steps
        .shapes
        .iter()
        .map(|shape| shape.readout_rows.hi)
        .max()
        .expect("a shape");
    shell.warm_pipelines_over(&steps).expect("the walk runs");
    shell.seal_pipelines().expect("the pipelines seal");
    shell.open(0).expect("a slot opens");
    // Every row count to 16 (each few-rows matmul and the bands past it),
    // every row read out.
    let top = most.min(16);
    for rows in 1..=top {
        let tokens: Vec<u32> = (0..rows).map(|at| (11 + at * 37) % 256).collect();
        let every: Vec<u32> = (0..rows).collect();
        shell
            .fire_seated(&[Seated {
                held: Some(0),
                readout: Some(&every),
                ..Seated::of(Lane {
                    slot: 0,
                    word: word(&fixture.trace, rows),
                    tokens: &tokens,
                })
            }])
            .unwrap_or_else(|why| panic!("a prompt of {rows} rows reading out each is served: {why}"));
        assert_eq!(
            shell.take_unbuilt_pipelines(),
            0,
            "a prompt of {rows} rows reading out each met a pipeline the walk did not build"
        );
    }
}
