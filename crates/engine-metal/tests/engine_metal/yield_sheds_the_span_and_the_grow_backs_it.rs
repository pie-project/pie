#![cfg(target_vendor = "apple")]

//! Yield's last shed gives the activation span's heaps back to the system
//! (`Engine::release_span`): the span is backed whole from the load, and
//! the shed un-maps what the carve and the kept segments do not take. A
//! raise into what was shed is refused by name until the governor's grow
//! (`Engine::commit_span`) backs the whole span again; after it a raise to
//! the top serves. No heap is made after ready but on that grow.

use engine_metal::serve::Seated;
use engine_metal::{Boot, Lane, Shell};
use poem_compiler::Budget;

use super::a_scattered_pool_serves_the_same_logits::{Fixture, argmax, fixture_of, word};

/// A top rung wide enough (4096 tokens) that the span lies over several
/// heaps and the smallest rung's carve over one.
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
        top: engine_metal::Top::EveryRung,
    })
    .unwrap_or_else(|why| panic!("the shell loads: {why}"))
}

fn fire(shell: &mut Shell, fixture: &Fixture, rows: &[u32], held: u32) -> u32 {
    let got = shell
        .fire_seated(&[Seated {
            held: Some(held),
            ..Seated::of(Lane {
                slot: 0,
                word: word(&fixture.trace, rows.len() as u32),
                tokens: rows,
            })
        }])
        .expect("the fire lands");
    argmax(&got[0])
}

#[test]
fn yield_sheds_the_span_and_the_grow_backs_it() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);
    let tokens: Vec<u32> = (0..300u32).map(|at| (7 + at * 37) % 256).collect();

    // What an unshed shell answers: a wide prompt at the top, then a decode.
    let mut never = load(&fixture);
    never.open(0).expect("slot 0 opens");
    never.set_rung(1, u32::MAX).expect("the top");
    let want_prompt = fire(&mut never, &fixture, &tokens, 0);
    let want_next = fire(&mut never, &fixture, &[want_prompt], tokens.len() as u32);
    drop(never);

    let mut shell = load(&fixture);
    let ready_heaps = shell.span_heaps_made();
    let ready_device = shell.device_allocated();
    assert!(ready_heaps > 1, "the span lies over several heaps");
    shell.open(0).expect("slot 0 opens");
    shell.set_rung(1, 1).expect("the smallest rung");

    // Yield's last shed: no span segment kept, the smallest carve in use.
    shell.release_span(0).expect("yield sheds the span");
    let shed_device = shell.device_allocated();
    eprintln!(
        "ready: {ready_heaps} span heaps, {} MiB; shed: {} MiB",
        ready_device >> 20,
        shed_device >> 20
    );
    assert!(
        shed_device < ready_device,
        "the shed gave nothing back: {shed_device} bytes on the device, {ready_device} at ready"
    );
    // The smallest rung still serves inside what the shed kept.
    fire(&mut shell, &fixture, &tokens[..1], 0);
    let refused = shell
        .set_rung(1, u32::MAX)
        .expect_err("a raise into the shed span waits on the grow");
    assert!(
        refused.to_string().contains("since yield shed the rest"),
        "{refused}"
    );
    assert_eq!(
        shell.span_heaps_made(),
        ready_heaps,
        "the refused raise made a heap"
    );

    // The governor's grow backs the whole span again, then the raise serves.
    let segments = shell
        .pool_chunk_facts()
        .act_span_chunks
        .unwrap_or(0)
        .div_ceil(shell.pool_chunk_facts().segment_chunks.max(1));
    shell.commit_span(segments).expect("the governor's grow");
    assert_eq!(
        shell.device_allocated(),
        ready_device,
        "the grow backs what the load backed"
    );
    let grown = shell.span_heaps_made();
    assert!(grown > ready_heaps, "the grow made the heaps the shed gave back");
    shell.set_rung(1, u32::MAX).expect("the raise to the top after the grow");
    // A fresh slot, so the prompt answers as the unshed shell's did.
    shell.open(1).expect("slot 1 opens");
    let got = shell
        .fire_seated(&[Seated {
            held: Some(0),
            ..Seated::of(Lane {
                slot: 1,
                word: word(&fixture.trace, tokens.len() as u32),
                tokens: &tokens,
            })
        }])
        .expect("a prompt at the top after the grow");
    let prompt = argmax(&got[0]);
    assert_eq!(prompt, want_prompt, "the grown span answers another prompt");
    let next = shell
        .fire_seated(&[Seated {
            held: Some(tokens.len() as u32),
            ..Seated::of(Lane {
                slot: 1,
                word: word(&fixture.trace, 1),
                tokens: &[prompt],
            })
        }])
        .expect("a decode after the grow");
    assert_eq!(argmax(&next[0]), want_next, "the grown span answers another decode");
    shell.set_rung(u32::MAX, 1).expect("a lower");
    shell.set_rung(1, u32::MAX).expect("and a raise");
    assert_eq!(
        shell.span_heaps_made(),
        grown,
        "a raise or a lower after the grow made a heap"
    );
}
