#![cfg(feature = "cuda")]

mod common_encoder;

use common_encoder::{Lcg, Rig, WIDTH, Weights, assert_close, attach, bf, frame, lane, reference};
use engine::Engine;
use engine::fire::ReadoutSeam;
use engine_cuda::Graphs;

const ROWS: [usize; 3] = [5, 13, 29];

fn fire_and_compare(graphs: Graphs) {
    let weights = Weights::random(&common_encoder::trace(), 13);
    let mut rig = Rig::load(&weights, 64, vec![16, 32, 64], graphs);

    let mut rng = Lcg::seeded(17);
    let inputs: Vec<Vec<f32>> = ROWS
        .iter()
        .map(|&n| (0..n * WIDTH as usize).map(|_| bf(rng.unit())).collect())
        .collect();
    let want: Vec<Vec<f32>> = ROWS
        .iter()
        .zip(&inputs)
        .map(|(&n, x)| reference(&weights, x, n))
        .collect();

    let mut lanes = Vec::new();
    let mut attachments = Vec::new();
    for (at, (&n, x)) in ROWS.iter().zip(&inputs).enumerate() {
        let handles = rig.lane(n as u32);
        rig.publish(handles.instance, 0, x);
        lanes.push(lane(at as u32, &handles, at as u32));
        attachments.push(attach(at as u32, &handles));
    }
    let mut ticket = rig
        .engine
        .submit(&frame(lanes, attachments))
        .expect("the frame fires");
    rig.engine
        .settle_frame(&mut ticket)
        .expect("the frame settles");
    let readouts = &ticket.steps[0].readouts;
    assert_eq!(readouts.len(), 3);
    for (at, readout) in readouts.iter().enumerate() {
        assert_eq!(readout.seam, ReadoutSeam::Hidden);
        assert_eq!(readout.width, WIDTH);
        assert_close(
            &readout.values,
            &want[at],
            &format!("lane {at} hidden rows"),
        );
    }
}

#[test]
fn a_bidirectional_attention_adds_its_relative_bias_through_the_engine_every_case() {
    three_lanes_of_unequal_length_land_the_host_reference();
    the_same_fire_lands_the_host_reference_from_an_armed_body();
}

fn three_lanes_of_unequal_length_land_the_host_reference() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    fire_and_compare(Graphs::Shaped);
}

fn the_same_fire_lands_the_host_reference_from_an_armed_body() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    fire_and_compare(Graphs::On);
}
