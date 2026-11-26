//! A lane's mask speaks of its keys from its set's committed boundary on
//! (`KvDelta::folded`, `Seated::folded`): column `c` is key `folded + c`,
//! and the committed keys are read under the causal bound and no bit. Two
//! rows past six held keys, committed through 3, hiding buffer column 1
//! (key 4), read out bit for bit what the same rows read with nothing
//! committed and the three committed columns spelled out as admitted; and
//! not what they read with no mask (the hidden key is hidden). Skipped
//! without a PJRT plugin.

use engine::fire::{Mask, Masking};
use engine_xla::{Lane, Seated, Shell};
use poem::Request;

use crate::common_llm::{Model, Refs};

fn fire(
    shell: &mut Shell,
    pages: &[u32],
    tokens: &[u32],
    held: u32,
    folded: u32,
    mask: Option<&Masking>,
) -> Vec<f32> {
    let facts = shell.trace().facts.clone();
    let word = facts.word(&Request::new(tokens.len() as u32, mask.is_some()));
    let mut s = Seated::of(Lane {
        slot: 0,
        word,
        tokens,
    });
    s.pages = &pages[..(held as usize + tokens.len()).div_ceil(16)];
    s.held = Some(held);
    s.folded = folded;
    s.mask = mask;
    shell.open(0).expect("opens");
    let fired = shell.fire_kept_with(&[s], &[]).expect("fires");
    let kept = fired.kept.as_ref().expect("the readout is kept");
    kept.lane_rows(0)
        .expect("the lane's rows read")
        .map(|(_, _, values)| values)
        .expect("a row read out")
}

#[test]
fn a_mask_reads_from_the_committed_boundary() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let mut model = Model::load_with("qwen3-micro-text", None, 256, 7, None);
    let facts = model.loaded.caps.chunks.clone();
    let shell = model.engine.shell_mut().expect("loaded");
    let mut refs = Refs::of_facts(facts);
    let kv = refs.shapes(|s| s.role == engine::chunk::SlotRole::State)[0].id;
    let pages: Vec<u32> = (0..4).map(|_| refs.take(kv)).collect();
    let prompt: Vec<u32> = (0..6).map(|i| 11 + 7 * i).collect();
    let rows = [301u32, 302];
    let _ = fire(shell, &pages, &prompt, 0, 0, None);
    // Committed through 3: columns are keys 3..8, the buffer 3, 4, 5, then
    // the two rows; column 1 (key 4) hidden.
    let from_boundary = Masking::Extent(Mask {
        runs: vec![0, 1, 1, 3],
        total: 5,
    });
    let got = fire(shell, &pages, &rows, 6, 3, Some(&from_boundary));
    // Nothing committed: the same keys, the committed three admitted.
    let from_zero = Masking::Extent(Mask {
        runs: vec![0, 4, 1, 3],
        total: 8,
    });
    let want = fire(shell, &pages, &rows, 6, 0, Some(&from_zero));
    assert_eq!(
        got, want,
        "the boundary's columns read what the whole mask reads"
    );
    let open = fire(shell, &pages, &rows, 6, 0, None);
    assert_ne!(got, open, "key 4 is hidden");
    eprintln!("a mask from the committed boundary reads what the whole mask reads, bit for bit");
}
