//! A fire's readout plane stays what it wrote while it is read: the fires
//! after it write the ring's other planes (the fires that can be
//! outstanding, the frames in flight of `STEPS_MAX` fires each and one,
//! `staging_depth`), and one that would come round to a plane still read is
//! refused by name. Two frames of four fires outstanding fit a load of two
//! frames in flight; on a load of one they are refused, as is a fire reading
//! out past the store. Skipped without a PJRT plugin.

use engine_xla::{Lane, Seated, Shell};
use poem::Request;

use crate::common_llm::{Model, Refs};

/// The fires a load of `frames` in flight holds outstanding before the
/// next is refused: each fire's readouts held, as frames in flight hold them.
fn outstanding(frames: u32) -> (usize, Option<usize>) {
    crate::common_llm::set_knob("FRAMES", frames);
    let mut model = Model::load_with("qwen3-micro-text", None, 256, 7, None);
    let facts = model.loaded.caps.chunks.clone();
    let shell = model.engine.shell_mut().expect("loaded");
    let depth = shell.pools().readout_depth();
    let mut refs = Refs::of_facts(facts);
    let kv = refs.shapes(|s| s.role == engine::chunk::SlotRole::State)[0].id;
    let pages: Vec<u32> = (0..4).map(|_| refs.take(kv)).collect();
    let steps = usize::from(engine::runahead::Runahead::STEPS_MAX);
    let mut held = Vec::new();
    for k in 0..2 * steps + 1 {
        let facts = shell.trace().facts.clone();
        let tokens = vec![3u32 + k as u32; 4];
        let word = facts.word(&Request::new(4, false));
        let mut s = Seated::of(Lane { slot: 0, word, tokens: &tokens });
        s.pages = &pages[..1];
        s.held = Some(0);
        shell.open(0).expect("opens");
        match shell.fire_kept_with(&[s], &[]) {
            Ok(fired) => held.push(fired),
            Err(_) => return (depth, Some(k)),
        }
    }
    crate::common_llm::set_knob("FRAMES", 1);
    (depth, None)
}

#[test]
fn two_frames_of_four_fires_outstanding_fit_two_frames_in_flight() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    // Two frames of four fires outstanding and the next being written.
    let (depth, refused) = outstanding(2);
    assert_eq!(depth, engine::runahead::Runahead::of(2).staging_depth());
    assert_eq!(refused, None, "two frames in flight hold {depth} fires");
    // The same outstanding on a load of one frame (a ring five deep) is
    // refused at its sixth fire: the depth is what holds them.
    let (narrow, refused) = outstanding(1);
    crate::common_llm::set_knob("FRAMES", 1);
    assert_eq!(refused, Some(narrow), "a ring {narrow} deep refuses fire {narrow}");
    eprintln!(
        "two frames of four fires outstanding: held at depth {depth}, refused at depth {narrow}"
    );
}

/// A fire whose sequence lanes read out more rows than the load's store
/// holds (`ReadoutFacts::store_rows`, the ring's rows) is refused by name,
/// nothing laid per fire: the runtime cuts fires to the store, so none is
/// sent. The store's rows themselves read out.
#[test]
fn a_fire_reading_out_past_the_store_is_refused() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let mut model = Model::load_with("qwen3-micro-text", None, 256, 7, None);
    let facts = model.loaded.caps.chunks.clone();
    let shell = model.engine.shell_mut().expect("loaded");
    let store = shell.readout_store_rows().expect("a sequence store");
    let mut refs = Refs::of_facts(facts);
    let kv = refs.shapes(|s| s.role == engine::chunk::SlotRole::State)[0].id;
    let pages: Vec<u32> = (0..16).map(|_| refs.take(kv)).collect();
    let read = |shell: &mut Shell, count: u32| {
        let facts = shell.trace().facts.clone();
        let tokens = vec![3u32; count as usize];
        let rows: Vec<u32> = (0..count).collect();
        let word = facts.word(&Request::new(count, false));
        let mut s = Seated::of(Lane { slot: 0, word, tokens: &tokens });
        s.pages = &pages[..(count as usize).div_ceil(16)];
        s.held = Some(0);
        s.readout = Some(&rows);
        shell.open(0).expect("opens");
        shell.fire_kept_with(&[s], &[]).map(|_| ())
    };
    read(shell, store).expect("the store's rows read out");
    let refused = read(shell, store + 1).expect_err("one row past the store");
    assert!(
        refused.to_string().contains("serve::readout_plane"),
        "refused by name: {refused}"
    );
    eprintln!("a store of {store} rows reads out {store}; {} refused: {refused}", store + 1);
}
