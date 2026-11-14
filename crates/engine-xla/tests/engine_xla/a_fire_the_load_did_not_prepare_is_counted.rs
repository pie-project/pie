//! What a served model meets worse than its load prepared is counted where
//! the runtime reads it (`engine::live::Tally`): a fire whose program the
//! load did not hold, once a shape; a frame the engine refuses.
//! The shipped config, no prewarm. Skipped without a PJRT plugin.

use engine::Engine;
use engine::live::Tally;

use crate::common_llm::{Model, Refs, Seq, tokens};

#[test]
fn a_fire_the_load_did_not_prepare_is_counted() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let mut model = Model::load_with("qwen3-micro-text", None, 256, 7, None);
    let live = model.engine.live();
    assert_eq!(live.tallied(Tally::Compiled), 0, "nothing served yet");
    let vocab = model.vocab;
    let mut refs = Refs::of(&model);
    let mut fire = |model: &mut Model, slot: u32| {
        let mut seq = Seq::new(slot);
        let lane = seq.lane(model, &mut refs, tokens(u64::from(slot), 12, vocab));
        model.fire(vec![lane]);
    };
    fire(&mut model, 0);
    let first = live.tallied(Tally::Compiled);
    assert!(first > 0, "the first fire's programs were not held");
    fire(&mut model, 1);
    assert_eq!(
        live.tallied(Tally::Compiled),
        first,
        "a shape is counted once"
    );
    // A lane naming a slot past the load's.
    let refused = model
        .engine
        .submit(&engine::fire::FrameSubmission::of(engine::fire::Step {
            lanes: vec![engine::fire::Lane {
                slot: u32::MAX,
                tokens: vec![1],
                positions: vec![0],
                ..engine::fire::Lane::default()
            }],
            attachments: Vec::new(),
            media: Vec::new(),
            voxels: Vec::new(),
        }));
    assert!(refused.is_err(), "a lane past the load's slots is refused");
    assert_eq!(live.tallied(Tally::Refused), 1);
}
