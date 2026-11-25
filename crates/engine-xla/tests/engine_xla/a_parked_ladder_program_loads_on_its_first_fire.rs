//! A warm program the host holds no room for is parked in the disk cache
//! and loaded on the first fire of its shape, not refused as off the
//! ladder and not built: a load holding no programs in host memory
//! (`Residency::host_programs` 0) parks its whole ladder, then serves a
//! prompt and its decode with the logits a load holding every program
//! gives, counting each parked program loaded once (`loaded_in_fire`) and
//! nothing compiled after ready. Skipped without a PJRT plugin.

use engine::Engine;
use engine::live::Tally;

use crate::common_llm::{Model, Refs, Seq, clear_knob, set_knob, tokens};

fn served(parked: bool) -> (Vec<Vec<f32>>, u64, u64, u64) {
    if parked {
        set_knob("HOST_PROGRAMS_MIB", 0);
    }
    let mut model = Model::load_with("qwen3-micro-text", None, 256, 7, None);
    clear_knob("HOST_PROGRAMS_MIB");
    // what the host holds loaded behind serving before the first fire, so
    // a fire loading a program is one the host held no room for
    model
        .engine
        .shell_mut()
        .expect("a loaded model has its shell")
        .load_parked_programs();
    let live = model.engine.live();
    let vocab = model.vocab;
    let mut refs = Refs::of(&model);
    let mut seq = Seq::new(0);
    let mut out = Vec::new();
    let lane = seq.lane(&model, &mut refs, tokens(3, 12, vocab));
    out.extend(model.fire(vec![lane]));
    seq.held += 12;
    for t in 0..3u32 {
        let lane = seq.lane(&model, &mut refs, vec![5 + t]);
        out.extend(model.fire(vec![lane]));
        seq.held += 1;
    }
    let loaded = live.tallied(Tally::LoadedInFire);
    // the same shapes again: held now, nothing loaded
    let lane = seq.lane(&model, &mut refs, vec![9]);
    model.fire(vec![lane]);
    let again = live.tallied(Tally::LoadedInFire) - loaded;
    (out, loaded, again, live.tallied(Tally::Compiled) + live.tallied(Tally::BuiltInFire))
}

#[test]
fn a_parked_ladder_program_loads_on_its_first_fire() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let (want, loaded, _, built) = served(false);
    assert_eq!(loaded, 0, "a load holding every program parks none");
    assert_eq!(built, 0, "the ladder holds a prompt and its decodes");
    let (got, loaded, again, built) = served(true);
    assert!(loaded > 0, "a load holding none parks its ladder and loads each shape on its first fire");
    assert_eq!(again, 0, "a loaded program is held: its shape loads once");
    assert_eq!(built, 0, "a parked program is loaded, never built after ready");
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(&want) {
        assert_eq!(g, w, "a parked program's logits are the laid one's");
    }
}
