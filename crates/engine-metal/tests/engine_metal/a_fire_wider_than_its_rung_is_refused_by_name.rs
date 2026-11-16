#![cfg(target_vendor = "apple")]

//! A fire wider than the rung the runtime raised the shell to is refused by
//! name before its rows are staged (the staged regions are the held rung's)
//! and writes nothing: the next fire at that rung answers as a shell that
//! never saw the wide one.

use engine_metal::serve::Seated;
use engine_metal::{Boot, Lane, Shell};
use poem_compiler::Budget;

use super::a_scattered_pool_serves_the_same_logits::{Fixture, argmax, fixture_of, word};

fn load(fixture: &Fixture) -> Shell {
    Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(4, 256),
        patches: None,
        voxels: None,
        profile: None,
        page_size: 16,
        context: 1024,
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

fn fire(
    shell: &mut Shell,
    fixture: &Fixture,
    rows: &[u32],
    held: u32,
) -> engine_metal::Result<Vec<Vec<f32>>> {
    shell.fire_seated(&[Seated {
        held: Some(held),
        ..Seated::of(Lane {
            slot: 0,
            word: word(&fixture.trace, rows.len() as u32),
            tokens: rows,
        })
    }])
}

#[test]
fn a_fire_wider_than_its_rung_is_refused_by_name() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);
    let prompt: Vec<u32> = (0..4u32).map(|at| (11 + at * 37) % 256).collect();
    // Wide enough that its tokens alone pass the 4-token rung's field and
    // its alignment, so a staged write would run into the next field.
    let wide: Vec<u32> = (0..100u32).map(|at| (5 + at * 53) % 256).collect();

    let mut never = load(&fixture);
    assert!(
        never.rung_tokens().contains(&4),
        "the load lays a 4-token rung: {:?}",
        never.rung_tokens()
    );
    never.open(0).expect("slot 0 opens");
    never.set_rung(0, 4).expect("the 4-token rung");
    let want = argmax(&fire(&mut never, &fixture, &prompt, 0).expect("the fire lands")[0]);

    let mut shell = load(&fixture);
    shell.open(0).expect("slot 0 opens");
    shell.set_rung(0, 4).expect("the 4-token rung");
    let refused = fire(&mut shell, &fixture, &wide, 0).expect_err("a 100-token fire at rung 4");
    assert!(
        refused.to_string().contains("is wider than the rung the runtime raised to"),
        "{refused}"
    );
    let got = argmax(&fire(&mut shell, &fixture, &prompt, 0).expect("the fire lands")[0]);
    assert_eq!(got, want, "the refused fire left something behind");
}
