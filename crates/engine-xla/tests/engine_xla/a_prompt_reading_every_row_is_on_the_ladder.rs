//! A prompt reading every row of itself (prompt logprobs) puts its fire's
//! readouts at the load's store rows, a count of the load's and not of the
//! fire's, and the warm ladder lays such fires: one alone and one beside
//! decodes build nothing after ready, and each reads out a row a prompt row.
//! Skipped without a PJRT plugin.

use engine::Engine;
use engine::fire::Readout;
use engine::live::Tally;

use crate::common_llm::{Model, Refs, Seq, tokens};

#[test]
fn a_prompt_reading_every_row_is_on_the_ladder() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let mut model = Model::load_with("qwen3-micro-text", None, 256, 7, None);
    let live = model.engine.live();
    let vocab = model.vocab;
    let mut refs = Refs::of(&model);
    let logprobs = |model: &mut Model, refs: &mut Refs, slot: u32, rows: usize| {
        let mut seq = Seq::new(slot);
        let mut lane = seq.lane(model, refs, tokens(u64::from(slot) + 1, rows, vocab));
        lane.readout = Readout::Rows((0..rows as u32).collect());
        lane
    };
    // Alone.
    let lane = logprobs(&mut model, &mut refs, 0, 12);
    let read = model.fire(vec![lane]);
    assert_eq!(read[0].len(), 12 * vocab as usize, "a row a prompt row");
    // Its last row is the row a plain fire of the same prompt reads out
    // (contract A: the argmax, or a near-tie of the plain row's two best).
    let mut seq = Seq::new(4);
    let plain = seq.lane(&model, &mut refs, tokens(1, 12, vocab));
    let want = model.fire(vec![plain]).remove(0);
    let got = &read[0][11 * vocab as usize..];
    let argmax = |row: &[f32]| {
        (row.iter().enumerate())
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(at, _)| at)
    };
    let (a, b) = (argmax(&want), argmax(got));
    let scale = want.iter().fold(0f32, |m, &x| m.max(x.abs())).max(1.0);
    assert!(
        a == b || (want[a] - want[b]).abs() <= 2e-2 * scale,
        "the every-row lane's last row answers {b} and the plain fire's {a}"
    );
    // Beside two decodes.
    let mut decodes = Vec::new();
    for slot in [1u32, 2] {
        let mut seq = Seq::new(slot);
        let lane = seq.lane(&model, &mut refs, tokens(u64::from(slot), 8, vocab));
        model.fire(vec![lane]);
        seq.held += 8;
        decodes.push(seq);
    }
    let mut lanes = vec![logprobs(&mut model, &mut refs, 3, 20)];
    for seq in &mut decodes {
        lanes.push(seq.lane(&model, &mut refs, vec![5]));
    }
    let read = model.fire(lanes);
    assert_eq!(
        read[0].len(),
        20 * vocab as usize,
        "a row a prompt row beside decodes"
    );
    assert_eq!(
        live.tallied(Tally::BuiltInFire),
        0,
        "a prompt reading every row found its program on the warm ladder"
    );
}
