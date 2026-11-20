//! A load states the mixed fires it lays (`Capabilities::mixed_lanes`, the
//! runtime's cutter composing no other): every prompt-lane rung with the
//! decode lanes beside it. Each composition the statement admits, at each
//! rung's extremes (the most prompts the rung holds beside the most
//! decodes the rung and the fire's lanes allow, the fewest prompts that
//! take the rung, one decode), fires on
//! the warm ladder's programs and builds nothing after ready; one decode
//! past a rung is a composition the statement refuses. Skipped without a
//! PJRT plugin.

use engine::Engine;
use engine::live::Tally;
use engine::{MixedLanes, MixedRung};

use crate::common_llm::{Model, Refs, Seq, set_knob, tokens};

#[test]
fn a_mixed_fire_the_load_states_laid_finds_its_program() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    set_knob("LANES", 48);
    set_knob("SLOTS", 64);
    set_knob("TOKENS", 256);
    let mut model = Model::load_with("qwen3-micro-text", None, 256, 7, None);
    let live = model.engine.live();
    let vocab = model.vocab;
    let caps = model.loaded.caps.clone();
    let MixedLanes::Laid(rungs) = &caps.mixed_lanes else {
        panic!("XLA states the mixed fires it lays");
    };
    assert!(!rungs.is_empty(), "the load lays mixed fires");
    let limits = caps.limits;
    let mut refs = Refs::of(&model);
    let mut next_slot = 0u32;
    // Decodes holding a few rows each, ready to take a row a fire.
    let most = rungs.iter().map(|r| r.decode_lanes).max().unwrap_or(0);
    let mut decodes: Vec<Seq> = Vec::new();
    for _ in 0..most {
        let mut seq = Seq::new(next_slot);
        next_slot += 1;
        let lane = seq.lane(&model, &mut refs, tokens(u64::from(next_slot), 4, vocab));
        model.fire(vec![lane]);
        seq.held += 4;
        decodes.push(seq);
    }
    let prompt_slots = next_slot;
    let mut fired = 0;
    let mut below = 1u32;
    for &MixedRung {
        prompt_lanes,
        decode_lanes,
    } in rungs
    {
        let fewest = below;
        // as many prompts and decodes as the rung and the fire's own lanes
        // allow (a rung no fire within the lanes reaches is not tried)
        let most = (prompt_lanes - 1).min(limits.max_lanes - 1);
        if most < fewest {
            below = prompt_lanes;
            continue;
        }
        let most_beside = |prompts: u32| decode_lanes.min(limits.max_lanes - prompts);
        for (prompts, n_decodes) in [
            (most, most_beside(most)),
            (fewest, most_beside(fewest)),
            (most, 1),
        ] {
            let rows = 2 * prompts;
            assert!(
                caps.mixed_lanes.fits(&limits, prompts, rows, n_decodes),
                "{prompts} prompts of 2 rows beside {n_decodes} decodes: within the statement"
            );
            let mut lanes = Vec::new();
            for i in 0..prompts {
                let mut seq = Seq::new(prompt_slots + i);
                lanes.push(seq.lane(&model, &mut refs, tokens(u64::from(1000 + i), 2, vocab)));
            }
            for seq in decodes.iter_mut().take(n_decodes as usize) {
                lanes.push(seq.lane(&model, &mut refs, vec![7]));
                seq.held += 1;
            }
            model.fire(lanes);
            fired += 1;
        }
        assert!(
            !caps.mixed_lanes.fits(
                &limits,
                prompt_lanes - 1,
                2 * (prompt_lanes - 1),
                decode_lanes + 1
            ),
            "one decode past the rung beside {} prompts is refused",
            prompt_lanes - 1
        );
        below = prompt_lanes;
    }
    assert!(fired >= 3);
    assert_eq!(
        live.tallied(Tally::BuiltInFire),
        0,
        "every mixed fire the statement admits found its program on the warm ladder"
    );
}
