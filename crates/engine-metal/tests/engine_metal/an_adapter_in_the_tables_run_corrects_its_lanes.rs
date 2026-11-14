#![cfg(target_vendor = "apple")]

//! The registered adapter banks lie in the pool's tables run
//! (`Reserved::Tables`): an adapter landed into a seat there is what the
//! correction reads. A lane naming a seat of zero planes serves the
//! unadapted logits bit for bit; a seat of other planes moves them; the
//! same planes landed in two seats serve the same logits.

use engine_metal::serve::Seated;
use engine_metal::weights::AdapterPlane;
use engine_metal::{Lane, Shell};

use super::a_scattered_pool_serves_the_same_logits::{Fixture, fixture_of, load_with, word};

const CONTEXT: u32 = 256;
const PROMPT: u32 = 24;

/// Bf16 values of a small spread, a seed apart: an adapter's plane.
fn plane(bytes: u64, seed: u64) -> Vec<u8> {
    (0..bytes / 2)
        .flat_map(|at| {
            let x = ((at.wrapping_mul(2_654_435_761).wrapping_add(seed * 97)) % 2001) as f32;
            let value = (x - 1000.0) / 20_000.0;
            ((value.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}

/// Lands every bank's plane `seed` (0: zeros) in adapter seat `seat`.
fn land(shell: &mut Shell, seat: u32, seed: u64) {
    let banks: Vec<(String, u64)> = shell
        .banks()
        .into_iter()
        .map(|(name, _, slot)| (name.to_string(), slot))
        .collect();
    assert!(!banks.is_empty(), "the model registers adapter banks");
    let planes: Vec<(String, Vec<u8>)> = banks
        .iter()
        .map(|(name, slot)| {
            let bytes = if seed == 0 {
                vec![0u8; *slot as usize]
            } else {
                plane(*slot, seed + name.len() as u64)
            };
            (name.clone(), bytes)
        })
        .collect();
    let planes: Vec<AdapterPlane<'_>> = planes
        .iter()
        .map(|(bank, bytes)| AdapterPlane { bank, bytes })
        .collect();
    shell
        .register_adapter(seat, &planes)
        .expect("the adapter lands in the tables run");
}

/// The prompt's logits on seat `slot`, read through adapter seat
/// `adapter`.
fn logits(shell: &mut Shell, fixture: &Fixture, slot: u32, adapter: Option<u32>) -> Vec<f32> {
    let tokens: Vec<u32> = (0..PROMPT).map(|at| (at * 37 + 11) % 256).collect();
    let mut lane_word = word(&fixture.trace, PROMPT);
    if adapter.is_some() {
        lane_word = shell
            .adapted_word(lane_word)
            .expect("the model has an adapted arm");
    }
    let seated = Seated {
        held: Some(0),
        adapter,
        ..Seated::of(Lane {
            slot,
            word: lane_word,
            tokens: &tokens,
        })
    };
    let mut rows = shell.fire_seated(&[seated]).expect("the fire lands");
    rows.remove(0)
}

#[test]
fn an_adapter_in_the_tables_run_corrects_only_its_lanes() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);
    let mut shell = load_with(&fixture, CONTEXT);
    assert!(
        shell.pool_layout().tables > 0,
        "the registered banks lie in the tables run"
    );
    land(&mut shell, 1, 0);
    land(&mut shell, 2, 7);
    land(&mut shell, 5, 7);
    let plain = logits(&mut shell, &fixture, 0, None);
    let zero = logits(&mut shell, &fixture, 1, Some(1));
    assert_eq!(zero, plain, "an adapter of zero planes corrects nothing");
    let adapted = logits(&mut shell, &fixture, 2, Some(2));
    assert_ne!(adapted, plain, "an adapter's planes move its lane's logits");
    let again = logits(&mut shell, &fixture, 3, Some(5));
    assert_eq!(
        again, adapted,
        "the same planes in another seat of the run serve the same logits"
    );
    eprintln!("an adapter in the tables run corrects only its lanes");
}
