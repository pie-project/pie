//! What a served model meets worse than its load prepared is counted where
//! the runtime reads it (`engine::live::Tally`): a fire whose program the
//! load did not hold, once a shape; a frame the engine refuses. The load
//! builds its warm ladder before it is ready: a plain prompt is on it and
//! compiles nothing; one reading two rows of itself (neither its last alone
//! nor every row, a lane the ladder does not lay) is a miss, counted once. Skipped without a PJRT plugin.

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
    let plain = |model: &mut Model, refs: &mut Refs, slot: u32| {
        let mut seq = Seq::new(slot);
        let lane = seq.lane(model, refs, tokens(u64::from(slot), 12, vocab));
        model.fire(vec![lane]);
    };
    plain(&mut model, &mut refs, 0);
    assert_eq!(
        live.tallied(Tally::Compiled),
        0,
        "a plain prompt's programs were built before ready"
    );
    let two_rows = |model: &mut Model, refs: &mut Refs, slot: u32| {
        let mut seq = Seq::new(slot);
        let mut lane = seq.lane(model, refs, tokens(u64::from(slot), 8, vocab));
        lane.readout = engine::fire::Readout::Rows(vec![2, 7]);
        model.fire(vec![lane]);
    };
    two_rows(&mut model, &mut refs, 1);
    let first = live.tallied(Tally::Compiled);
    assert!(
        first > 0,
        "a prompt reading two rows is off the ladder, and counted"
    );
    assert_eq!(
        live.tallied(Tally::BuiltInFire),
        first,
        "each program built after ready on a fire's path is built in a fire"
    );
    assert_eq!(
        live.tallied(Tally::BuiltAtRegister),
        0,
        "nothing registered"
    );
    two_rows(&mut model, &mut refs, 2);
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
            shape: None,
        }));
    assert!(refused.is_err(), "a lane past the load's slots is refused");
    assert_eq!(live.tallied(Tally::Refused), 1);
}
