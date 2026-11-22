#![cfg(target_vendor = "apple")]

//! A prompt's attention reads none of its pages' cells past its last key. A
//! key tile runs to a page's end whatever a prompt's length, and the cells past
//! the prompt's keys were never written: a kernel that keeps them out by a zero
//! weight multiplies whatever they hold by zero, which is not zero where that is
//! a NaN. The kv pool's memory is recycled, so a cell past a lane's keys can hold
//! one; under `test-slot-fill` the pool's new bytes are NaN, so a read of an
//! unwritten cell fails here rather than finding a zero. Several lanes of the
//! same prompt, of lengths off the tile, prefilled in one fire, each read
//! finite logits, and the same ones: 57 rows end in a tile's second page at
//! a page of 16 (25 keys held of its 32), past the first.

use engine_metal::serve::Seated;
use engine_metal::{Boot, Lane, Shell};
use poem_compiler::Budget;

use super::a_scattered_pool_serves_the_same_logits::{Fixture, fixture_of, word};

const SEATS: u32 = 8;

fn load(fixture: &Fixture, page_size: u32) -> Shell {
    Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(SEATS, 512),
        patches: None,
        voxels: None,
        profile: None,
        page_size,
        context: 256,
        slots: SEATS,
        kv_budget: None,
        allowance: engine::Allowance::WHOLE,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
        profiles: None,
        top: engine_metal::Top::EveryRung,
        measure: engine::rates::Measuring::Time,
    })
    .and_then(|mut shell| shell.raise_whole().map(|()| shell))
    .unwrap_or_else(|why| panic!("the shell loads: {why}"))
}

#[test]
fn a_prefill_reads_no_kv_cell_past_its_keys() {
    if !engine_metal::device::present() {
        return;
    }
    let mut wrong = Vec::new();
    for model in ["qwen3-micro-text", "qwen3-micro-text-hd128"] {
        let fixture = fixture_of(model, None);
        for page_size in [16u32, 32] {
            for (lanes, rows) in [(1u32, 35u32), (4, 35), (4, 33), (2, 47), (8, 35), (1, 57), (4, 57)] {
                let mut shell = load(&fixture, page_size);
                let prompt: Vec<u32> = (0..rows).map(|t| (t * 7 + 3) % 200).collect();
                for slot in 0..lanes {
                    shell.open(slot).expect("the slot opens");
                }
                let seated: Vec<Seated<'_>> = (0..lanes)
                    .map(|slot| Seated {
                        held: Some(0),
                        ..Seated::of(Lane { slot, word: word(&fixture.trace, rows), tokens: &prompt })
                    })
                    .collect();
                let out = shell.fire_seated(&seated).expect("the fire runs");
                let case = format!("{model} page {page_size}, {lanes} lane(s) of {rows} rows");
                for (lane, row) in out.iter().enumerate() {
                    let bad = row.iter().filter(|x| !x.is_finite()).count();
                    if bad > 0 {
                        wrong.push(format!("{case}: lane {lane} read {bad} non-finite logits"));
                    } else if row.iter().zip(&out[0]).any(|(a, b)| a.to_bits() != b.to_bits()) {
                        wrong.push(format!("{case}: lane {lane} read other logits than lane 0"));
                    }
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "a prefill read a kv cell past its keys (poisoned pool: {}): {wrong:#?}",
        cfg!(feature = "test-slot-fill")
    );
}
