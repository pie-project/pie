#![cfg(feature = "device")]
//! A lane of a kv-less bidirectional layer states which of its group's rows
//! each row attends: the engine stages the fire's group masks (bits and a
//! byte offset a group) and the ragged kernel's custom arm reads them. The
//! hidden rows land the host reference with the same rows hidden, so the
//! staged offsets' type and the kernel's are checked together. The masks are
//! data: the same lanes stating none replay the same body and read every row.

use crate::common_encoder;

use common_encoder::{
    Lcg, Rig, WIDTH, Weights, assert_close, attach, bf, frame, lane, reference_with,
};
use engine::Engine;
use engine::fire::{Mask, Masking};
use engine_cuda::Graphs;

const ROWS: [usize; 3] = [5, 13, 29];

/// Row `i` of lane `at` attends key row `j`: itself always, else a pattern
/// that differs lane by lane, so a lane reading another's bits shows.
fn admits(at: usize, i: usize, j: usize) -> bool {
    i == j || (i * 7 + j * 3 + at) % 4 != 0
}

/// Row `i`'s mask over the lane's `rows` key rows, as runs: hidden,
/// admitted, hidden, ...
fn row_mask(at: usize, i: usize, rows: usize) -> Mask {
    let mut runs = Vec::new();
    let (mut admitted, mut run) = (false, 0u32);
    for j in 0..rows {
        if admits(at, i, j) == admitted {
            run += 1;
        } else {
            runs.push(run);
            admitted = !admitted;
            run = 1;
        }
    }
    runs.push(run);
    Mask::new(runs, rows as u64)
}

/// Fires the three lanes, each stating its mask or (`masked` false) none,
/// and compares their rows with the host's under the same masks; returns the
/// body the armed fire replayed.
fn fire(
    rig: &mut Rig,
    weights: &Weights,
    inputs: &[Vec<f32>],
    graphs: Graphs,
    masked: bool,
) -> Option<String> {
    let want: Vec<Vec<f32>> = ROWS
        .iter()
        .zip(inputs)
        .enumerate()
        .map(|(at, (&n, x))| {
            reference_with(weights, x, n, None, &|i, j| !masked || admits(at, i, j))
        })
        .collect();
    let mut lanes = Vec::new();
    let mut attachments = Vec::new();
    for (at, (&n, x)) in ROWS.iter().zip(inputs).enumerate() {
        let handles = rig.lane(n as u32);
        rig.publish(handles.instance, 0, x);
        let mut stated = lane(rig, at as u32, &handles, at as u32);
        // The runtime states a kv-less lane so: a group mask is a kv-less
        // lane's.
        stated.kv_less = true;
        stated.mask = masked.then(|| Masking::Rows((0..n).map(|i| row_mask(at, i, n)).collect()));
        lanes.push(stated);
        attachments.push(attach(at as u32, &handles));
    }
    let mut ticket = rig
        .engine
        .submit(&frame(lanes, attachments))
        .expect("the frame fires");
    rig.engine
        .settle_frame(&mut ticket)
        .expect("the frame settles");
    let replayed = matches!(graphs, Graphs::On).then(|| {
        let shell = rig.engine.shell().expect("the load holds a shell");
        shell
            .replayed_body()
            .expect("an armed load replays a body for the fire")
            .to_string()
    });
    let readouts = &ticket.steps[0].readouts;
    assert_eq!(readouts.len(), 3);
    for (at, readout) in readouts.iter().enumerate() {
        assert_eq!(readout.output, "hidden");
        assert_close(
            &readout.values,
            &want[at],
            &format!("lane {at} hidden rows, masked: {masked}"),
        );
    }
    replayed
}

fn fire_and_compare(graphs: Graphs) {
    let trace = common_encoder::unbiased_trace();
    let weights = Weights::random(&trace, 13);
    let mut rig = Rig::load_trace(trace, &weights, 64, vec![16, 32, 64], graphs);

    let mut rng = Lcg::seeded(17);
    let inputs: Vec<Vec<f32>> = ROWS
        .iter()
        .map(|&n| (0..n * WIDTH as usize).map(|_| bf(rng.unit())).collect())
        .collect();
    // The masks are data: the masked fire and the maskless one replay one
    // body, each reading its own masks (a group stating none, every row).
    let masked = fire(&mut rig, &weights, &inputs, graphs, true);
    let maskless = fire(&mut rig, &weights, &inputs, graphs, false);
    assert_eq!(
        masked, maskless,
        "a fire stating masks and one stating none replay one body"
    );
}

#[test]
fn a_stated_group_mask_reads_only_the_rows_it_admits_every_case() {
    if !engine_cuda::device::present() {
        eprintln!("no device: skipping");
        return;
    }
    fire_and_compare(Graphs::Shaped);
    fire_and_compare(Graphs::On);
}

/// The same layer whose model states no lane's mask on its attention: a
/// lane's mask is refused by name, and the load lays no mask stretch.
#[test]
fn a_mask_over_an_attention_stating_none_is_refused() {
    if !engine_cuda::device::present() {
        eprintln!("no device: skipping");
        return;
    }
    let trace = common_encoder::unstated_trace();
    let weights = Weights::random(&trace, 13);
    let mut rig = Rig::load_trace(trace, &weights, 64, vec![16, 32, 64], Graphs::On);
    let n = ROWS[0];
    let mut rng = Lcg::seeded(17);
    let x: Vec<f32> = (0..n * WIDTH as usize).map(|_| bf(rng.unit())).collect();
    let handles = rig.lane(n as u32);
    rig.publish(handles.instance, 0, &x);
    let mut stated = lane(&rig, 0, &handles, 0);
    stated.kv_less = true;
    stated.mask = Some(Masking::Rows((0..n).map(|i| row_mask(0, i, n)).collect()));
    let refused = rig
        .engine
        .submit(&frame(vec![stated], vec![attach(0, &handles)]))
        .and_then(|mut ticket| rig.engine.settle_frame(&mut ticket).map(|()| ticket));
    let why = match refused {
        Ok(_) => panic!("a lane's mask over an attention stating none fired"),
        Err(why) => why.to_string(),
    };
    assert!(
        why.contains("stated_group_mask()"),
        "the refusal names what the model must state: {why}"
    );
}
