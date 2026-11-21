//! A decode fire padded to its rung (padding lanes on the sink slot) leaves
//! every other sequence's recurrent state as it was: a hybrid sequence that
//! decodes, sits out a fire of two other sequences padded to four lanes,
//! then decodes again answers bit for bit what it answers with no fire
//! between. qwen35-tiny (gated-delta layers). Skipped without a PJRT plugin.

use crate::common_llm::{Model, Refs, Seq, tokens};

fn served(between: bool) -> Vec<f32> {
    let mut model = Model::load_with("qwen35-tiny", None, 256, 7, None);
    let mut refs = Refs::of(&model);
    let vocab = model.vocab;
    let mut a = Seq::new(0);
    let lane = a.lane(&model, &mut refs, tokens(3, 12, vocab));
    model.fire(vec![lane]);
    a.held += 12;
    let lane = a.lane(&model, &mut refs, vec![5]);
    model.fire(vec![lane]);
    a.held += 1;
    if between {
        // two other sequences on slots 1 and 2, then their decode together
        // (three lanes' worth of rung: padding lanes fill it)
        let mut others = [Seq::new(1), Seq::new(2), Seq::new(3)];
        for (n, seq) in others.iter_mut().enumerate() {
            let lane = seq.lane(&model, &mut refs, tokens(10 + n as u64, 8, vocab));
            model.fire(vec![lane]);
            seq.held += 8;
        }
        let lanes = others
            .iter_mut()
            .map(|seq| seq.lane(&model, &mut refs, vec![7]))
            .collect();
        model.fire(lanes);
    }
    let lane = a.lane(&model, &mut refs, vec![9]);
    model.fire(vec![lane]).pop().expect("a row")
}

#[test]
fn a_padded_decode_leaves_another_lanes_state() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let alone = served(false);
    let beside = served(true);
    let bits = |row: &[f32]| row.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&beside),
        bits(&alone),
        "a sequence's next logits after another sequences' padded decode are the ones it gives alone"
    );
}
