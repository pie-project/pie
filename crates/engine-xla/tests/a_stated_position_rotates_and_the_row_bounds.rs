//! A lane's stated positions rotate its rows; its rows' indices in the
//! cache bound its attention (engine-cuda's `kv_len - qo_len + i`). An
//! attention sink shifted past, a StreamingLLM rewind or M-RoPE text after
//! an image all state positions that are not rows.
//!
//! Rotary attention reads only position differences, so a prompt stated 100
//! ahead of its rows reads the same logits as at its rows. Bounded at the
//! stated position instead, every prefill row would see every key of the
//! fire (the causal end past the lane's capacity), and a decode stated
//! behind its row would stop short of its own key.
//!
//! Asked for with `PIE_XLA_ARTIFACT` (or `PIE_XLA_SNAPSHOT` + `PIE_XLA_DEPLOYMENT`)
//! naming a text model.

mod common;

use engine_xla::{Boot, DeviceBoot, Lane, Seated, Shell};
use poem::{Platform, Request};
use poem_compiler::{Budget, PatchLadder};

const SHIFT: u32 = 100;

fn worst(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

#[test]
fn a_prompt_stated_ahead_of_its_rows_reads_what_it_reads_at_them() {
    let Some(m) = common::model() else {
        eprintln!("not asked: set PIE_XLA_ARTIFACT or PIE_XLA_SNAPSHOT + PIE_XLA_DEPLOYMENT");
        return;
    };
    let trace = m.deployment.trace(Platform::Xla);
    let facts = m.deployment.trace(models::Platform::Xla).facts;
    let word = |len: u32| facts.word(&Request::new(len, false));
    let prompt =
        common::tokenizer(&m).encode("The capital of France is Paris, and the capital of Italy is");

    let context = 256;
    let _device = engine_xla::bench::lock_device();
    let mut shell = Shell::load(Boot {
        trace,
        contract: &m.contract,
        checkpoint: &m.checkpoint,
        budget: Budget::new(2, context),
        page_size: 16,
        context,
        slots: 2,
        pages: 2 * context / 16,
        device: &DeviceBoot::default(),
        patches: m
            .deployment
            .name
            .contains("vision")
            .then(|| PatchLadder::new(256, 1)),
    })
    .expect("the shell loads");

    // Each walk: the prompt in one prefill, then one decode of a fixed token.
    let mut walk = |slot: u32, shift: u32| -> (Vec<f32>, Vec<f32>) {
        shell.open(slot).expect("a slot opens");
        let positions: Vec<u32> = (0..prompt.len() as u32).map(|p| p + shift).collect();
        let mut seated = Seated::of(Lane {
            slot,
            word: word(prompt.len() as u32),
            tokens: &prompt,
        });
        seated.positions = &positions;
        let prefill = shell
            .fire_seated(&[seated])
            .expect("the prefill fires")
            .remove(0);
        let at = [prompt.len() as u32 + shift];
        let fed = [prompt[0]];
        let mut seated = Seated::of(Lane {
            slot,
            word: word(1),
            tokens: &fed,
        });
        seated.positions = &at;
        let decode = shell
            .fire_seated(&[seated])
            .expect("the decode fires")
            .remove(0);
        (prefill, decode)
    };
    let (prefill, decode) = walk(0, 0);
    let (shifted_prefill, shifted_decode) = walk(1, SHIFT);
    drop(shell);

    let (p, d) = (
        worst(&prefill, &shifted_prefill),
        worst(&decode, &shifted_decode),
    );
    eprintln!(
        "{}: stated {SHIFT} ahead, prefill max |Δlogit| {p}, decode {d}",
        m.deployment.name
    );
    // Rounding only: the rotations differ in their f32 angles.
    assert!(
        p < 0.5,
        "a prompt stated {SHIFT} ahead of its rows reads logits {p} away"
    );
    assert!(
        d < 0.5,
        "a decode stated {SHIFT} ahead of its row reads logits {d} away"
    );
}
