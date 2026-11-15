//! Every fire's readout lands in one plane shape: a ring of planes laid at
//! load (`Pools::lay_readouts`, a readout row a lane at the most lanes a
//! fire holds), which a fire writes its readout rows into in place, so a
//! fire of one lane and a fire of three hand back planes of one shape, and
//! the guest stages and host gathers reading them take one shape too. A
//! fire's plane stays what it wrote while it is read: the fires after it
//! write the ring's other planes (the fires that can be outstanding, the
//! frames in flight of `STEPS_MAX` fires each and one, `staging_depth`),
//! and a fire that would come round to a plane still read is refused by
//! name, nothing laid. Two frames of four fires outstanding fit a load of
//! two frames in flight; on a load of one they are refused. Skipped without
//! a PJRT plugin.

use engine_xla::{Lane, Seated, Shell};
use poem::Request;

use crate::common_llm::{Model, Refs};

/// Fires one prefill lane a `rows` entry (fresh slots), the readout kept.
fn fire(shell: &mut Shell, pages: &[u32], rows: &[u32], seed: u32) -> engine_xla::serve::Fired {
    let facts = shell.trace().facts.clone();
    let tokens: Vec<Vec<u32>> = rows
        .iter()
        .enumerate()
        .map(|(lane, &n)| (0..n).map(|i| (i * 31 + lane as u32 * 7 + seed) % 500 + 3).collect())
        .collect();
    let words: Vec<u64> = rows.iter().map(|&n| facts.word(&Request::new(n, false))).collect();
    let seated: Vec<Seated<'_>> = (0..rows.len())
        .map(|at| {
            let mut s = Seated::of(Lane {
                slot: at as u32,
                word: words[at],
                tokens: &tokens[at],
            });
            s.pages = &pages[at..at + 1];
            s.held = Some(0);
            s
        })
        .collect();
    for at in 0..rows.len() as u32 {
        shell.open(at).expect("opens");
    }
    shell.fire_kept_with(&seated, &[]).expect("fires")
}

fn lane(fired: &engine_xla::serve::Fired, at: usize) -> Vec<f32> {
    let kept = fired.kept.as_ref().expect("the readout is kept");
    kept.lane_rows(at)
        .expect("the lane's rows read")
        .map(|(_, _, values)| values)
        .expect("the lane reads a row out")
}

#[test]
fn a_readout_lands_in_one_plane_shape() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let mut model = Model::load_with("qwen3-micro-text", None, 256, 7, None);
    let facts = model.loaded.caps.chunks.clone();
    let shell = model.engine.shell_mut().expect("loaded");
    let rings: Vec<(u32, u32)> =
        (shell.pools().readouts().iter()).map(|r| (r.rows, r.width)).collect();
    assert!(!rings.is_empty(), "the load lays a readout ring");
    let mut refs = Refs::of_facts(facts.clone());
    let kv = refs.shapes(|s| s.role == engine::chunk::SlotRole::State)[0].id;
    let pages: Vec<u32> = (0..4).map(|_| refs.take(kv)).collect();

    let one = fire(shell, &pages, &[5], 1);
    let three = fire(shell, &pages, &[3, 4, 6], 2);
    let shape = |fired: &engine_xla::serve::Fired| -> Vec<(u32, u32)> {
        let kept = fired.kept.as_ref().expect("kept");
        (kept.planes.iter())
            .filter(|p| p.readout)
            .map(|p| (p.rows * p.layers.max(1), p.width))
            .collect()
    };
    assert_eq!(shape(&one), shape(&three), "a fire of one lane and of three: one plane shape");
    for s in shape(&one) {
        assert!(rings.contains(&s), "a readout plane is a ring's: {s:?} of {rings:?}");
    }

    // Held across the fires a ring holds (`staging_depth`; the three-lane
    // fire was one of them): still what it wrote.
    let first = lane(&one, 0);
    drop(three);
    let depth = shell.pools().readout_depth();
    assert_eq!(depth, engine::runahead::Runahead::of(1).staging_depth(), "a load of one frame");
    let mut held = Vec::new();
    for k in 0..depth - 2 {
        held.push(fire(shell, &pages, &[2 + k as u32, 7], 10 + k as u32));
    }
    assert_eq!(lane(&one, 0), first, "a held plane is not written over");
    // One more would come round to it: refused by name, nothing laid.
    let refused = {
        let facts = shell.trace().facts.clone();
        let tokens = vec![3u32; 4];
        let word = facts.word(&Request::new(4, false));
        let mut s = Seated::of(Lane { slot: 0, word, tokens: &tokens });
        s.pages = &pages[..1];
        s.held = Some(0);
        shell.open(0).expect("opens");
        shell.fire_kept_with(&[s], &[]).err().expect("a fire coming round to a held plane")
    };
    assert!(refused.to_string().contains("still read"), "refused by name: {refused}");
    // Read, released: the ring serves again.
    drop(one);
    drop(held);
    let again = fire(shell, &pages, &[5], 1);
    assert_eq!(lane(&again, 0), first, "the same lane fired again reads the same row");
    eprintln!("readout planes {rings:?}, {depth} deep; a fire past a held plane: {refused}");
}

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
    eprintln!("two frames of four fires outstanding: held at depth {depth}, refused at depth {narrow}");
}
