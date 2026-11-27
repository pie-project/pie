//! A lane's attention mask is data, not a class: a masked beam lane and
//! plain decode lanes share one fire and one program, and each answers what
//! it answers alone (contract A: a token parts only on a near-tie). The
//! beam's mask hides a stretch of its prompt, so it answers otherwise than
//! the same lane read without one. XLA's form of the CUDA suite's test of
//! the same name. Skipped without a PJRT plugin.

use engine::fire::{Mask, Masking};

use crate::common_llm::{Model, Refs, Seq, tokens};

const PROMPT: usize = 40;
/// Plain lanes beside the beam.
const PLAIN: u32 = 3;
const LANES: u32 = PLAIN + 1;
const STEPS: u32 = 6;

/// The beam's mask over `columns` keys: every key but a stretch of its
/// prompt (`8..16`).
fn beam_mask(columns: u32) -> Masking {
    Masking::Extent(Mask::new(vec![0, 8, 8, columns - 16], u64::from(columns)))
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(at, _)| at)
}

/// Whether `got` answers `want`'s token, or parts from it only on a
/// near-tie of `want`'s two best.
fn agrees(want: &[f32], got: &[f32]) -> bool {
    let (a, b) = (argmax(want), argmax(got));
    if a == b {
        return true;
    }
    let scale = want.iter().fold(0f32, |m, &x| m.max(x.abs())).max(1.0);
    (want[a] - want[b]).abs() <= 2e-2 * scale
}

#[test]
fn a_masked_beam_beside_plain_decodes_answers_as_each_alone() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let mut model = Model::load_with("qwen3-micro-text", None, 256, 7, None);
    let vocab = model.vocab;
    let mut refs = Refs::of(&model);
    // Lane `l` batched in slot `l`, alone in slot `LANES + l`; the beam's
    // unmasked twin in slot `2 * LANES`.
    let mut batched: Vec<Seq> = (0..LANES).map(Seq::new).collect();
    let mut alone: Vec<Seq> = (0..LANES).map(|l| Seq::new(LANES + l)).collect();
    let mut twin = Seq::new(2 * LANES);
    let prompt = |l: u32| tokens(u64::from(l) + 3, PROMPT, vocab);
    for l in 0..LANES {
        for seq in [&mut batched[l as usize], &mut alone[l as usize]] {
            let lane = seq.lane(&model, &mut refs, prompt(l));
            model.fire(vec![lane]);
            seq.held += PROMPT as u32;
        }
    }
    let lane = twin.lane(&model, &mut refs, prompt(PLAIN));
    model.fire(vec![lane]);
    twin.held += PROMPT as u32;

    let beam = PLAIN as usize;
    let mut parted = 0f32;
    let mut beam_alone = Vec::new();
    for step in 0..STEPS {
        let token = |l: usize| (step * 7 + 5 + l as u32 * 11) % vocab;
        let mask = beam_mask(batched[beam].held + 1);
        let lanes: Vec<_> = (0..LANES as usize)
            .map(|l| {
                let mut lane = batched[l].lane(&model, &mut refs, vec![token(l)]);
                if l == beam {
                    lane.mask = Some(mask.clone());
                }
                lane
            })
            .collect();
        let together = model.fire(lanes);
        for l in 0..LANES as usize {
            let mut lane = alone[l].lane(&model, &mut refs, vec![token(l)]);
            if l == beam {
                lane.mask = Some(mask.clone());
            }
            let by_itself = model.fire(vec![lane]);
            if l == beam {
                beam_alone = by_itself[0].clone();
            }
            assert!(
                agrees(&by_itself[0], &together[l]),
                "step {step}: lane {l} answers {} beside the others and {} alone",
                argmax(&together[l]),
                argmax(&by_itself[0])
            );
        }
        let lane = twin.lane(&model, &mut refs, vec![token(beam)]);
        let unmasked = model.fire(vec![lane]);
        // the masked beam alone against its unmasked twin alone: the mask,
        // not the batch, parts them
        let scale = unmasked[0]
            .iter()
            .fold(0f32, |m, &x| m.max(x.abs()))
            .max(1.0);
        let gap = (beam_alone.iter().zip(&unmasked[0]))
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        parted = parted.max(gap / scale);
        for seq in batched.iter_mut().chain(alone.iter_mut()) {
            seq.held += 1;
        }
        twin.held += 1;
    }
    eprintln!("the mask moved the beam's logits by {parted:.3} of their scale at most");
    assert!(
        parted > 1e-2,
        "the beam's mask hid nothing: it answered as its unmasked twin every step ({parted})"
    );
}
