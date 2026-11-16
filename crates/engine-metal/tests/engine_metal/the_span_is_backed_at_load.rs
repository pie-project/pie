#![cfg(target_vendor = "apple")]

//! The activation span is backed whole at the load, before ready: the top
//! rung's carve and every segment a lowered rung lends to state. So a rung
//! raise from the smallest to the top mid-serve, a lend and its release
//! make no placement heap, which is a device allocation, after ready. They
//! made one on each first raise before: the pool started at the smallest
//! rung's carve and backed each wider one as it came.

use engine_metal::serve::Seated;
use engine_metal::{Boot, Lane, Shell};
use poem_compiler::Budget;

use super::a_scattered_pool_serves_the_same_logits::{Fixture, fixture_of, word};

/// A load whose top rung is wide (4096 tokens a fire), so its span lies
/// over several placement heaps and the smallest rung's carve over one.
fn load(fixture: &Fixture) -> Shell {
    Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(4, 4096),
        patches: None,
        voxels: None,
        profile: None,
        page_size: 16,
        context: 4096,
        slots: 4,
        kv_budget: None,
        allowance: engine::Allowance::WHOLE,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
        profiles: None,
        // Every rung to 4096, so the span lies over several heaps.
        top: engine_metal::Top::EveryRung,
    })
    .unwrap_or_else(|why| panic!("the shell loads: {why}"))
}

#[test]
fn the_span_is_backed_at_load() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);
    let mut shell = load(&fixture);
    let made = shell.span_heaps_made();
    assert!(made > 0, "the load backed the span's heaps before ready");
    let device = shell.audit().device;
    eprintln!("ready: {made} span heaps, {} MiB on the device", device >> 20);
    let tokens: Vec<u32> = (0..9u32).map(|at| (7 + at * 37) % 256).collect();
    let fire = |shell: &mut engine_metal::Shell, slot: u32, rows: &[u32], held: u32| {
        shell
            .fire_seated(&[Seated {
                held: Some(held),
                ..Seated::of(Lane {
                    slot,
                    word: word(&fixture.trace, rows.len() as u32),
                    tokens: rows,
                })
            }])
            .expect("the fire lands")
    };
    // The shell stands at the smallest rung, as a load leaves it.
    shell.open(0).expect("slot 0 opens");
    shell.set_rung(0, 1).expect("the smallest rung");
    fire(&mut shell, 0, &tokens[..1], 0);
    shell.set_rung(1, u32::MAX).expect("the rung raises to the top");
    fire(&mut shell, 0, &tokens[1..], 1);
    assert!(
        shell.span_heaps_made() > 1,
        "the top rung's carve lies over several heaps, so a raise could make one"
    );
    shell.set_rung(u32::MAX, 1).expect("the rung lowers to the smallest");
    fire(&mut shell, 0, &[3], tokens.len() as u32);
    shell.set_rung(1, u32::MAX).expect("the rung raises to the top mid-serve");
    fire(&mut shell, 0, &[5], tokens.len() as u32 + 1);
    shell.commit_span(1).expect("a span segment lends to state");
    shell.release_span(0).expect("and goes back");
    shell.set_rung(u32::MAX, 1).expect("the rung lowers again");
    shell.set_rung(1, u32::MAX).expect("and raises again");
    fire(&mut shell, 0, &[9], tokens.len() as u32 + 2);
    assert_eq!(
        shell.span_heaps_made(),
        made,
        "a raise, a lower or a lend after ready made a placement heap"
    );
    assert_eq!(
        shell.audit().device,
        device,
        "the device holds what it held at ready"
    );
}
