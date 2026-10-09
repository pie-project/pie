#![cfg(feature = "cuda")]

mod common_dit;

use std::time::Instant;

use common_dit::{
    HostRequest, Lcg, Rig, WIDTH, Weights, assert_close, attach, bf, frame, lane, reference,
};
use engine::Engine;
use engine::fire::LaneStream;

#[test]
fn the_plan_loads_at_the_ceiling_and_fires_below_the_first_rung() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let weights = Weights::random(&common_dit::trace(), 17);
    let began = Instant::now();
    let mut rig = Rig::load(&weights, 65536, vec![8192, 32768]);
    let loaded = began.elapsed();
    let facts = &rig.loaded.facts;
    eprintln!(
        "loaded at max_tokens=65536, buckets=[8192, 32768] in {:.2?}: arena {} MiB, inputs {} MiB",
        loaded,
        facts.arena_bytes >> 20,
        facts.input_bytes >> 20,
    );
    assert!(
        facts.input_bytes < 64 << 20,
        "the inputs store is {} bytes for a plan with no masked arm",
        facts.input_bytes
    );

    let mut rng = Lcg::seeded(9);
    let (text_rows, image_rows) = (5usize, 200usize);
    let w = WIDTH as usize;
    let request = HostRequest {
        text: (0..text_rows * w).map(|_| bf(rng.unit())).collect(),
        image: (0..image_rows * w).map(|_| bf(rng.unit())).collect(),
        text_rows,
        image_rows,
        timestep: 0.25,
        positions: (0..text_rows + image_rows)
            .map(|r| [r as f32, 1.0])
            .collect(),
    };
    let want = reference(&weights, &request);
    let text = rig.lane(text_rows as u32);
    let image = rig.lane(image_rows as u32);
    rig.publish(text.instance, 0, &request.text);
    rig.publish(text.instance, 1, &[request.timestep]);
    rig.publish(
        text.instance,
        2,
        &request.positions[..text_rows]
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<f32>>(),
    );
    rig.publish(image.instance, 0, &request.image);
    rig.publish(image.instance, 1, &[request.timestep]);
    rig.publish(
        image.instance,
        2,
        &request.positions[text_rows..]
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<f32>>(),
    );
    let began = Instant::now();
    let mut ticket = rig
        .engine
        .submit(&frame(
            vec![
                lane(&rig, 0, &text, LaneStream::Text, 0),
                lane(&rig, 1, &image, LaneStream::Image, 0),
            ],
            vec![attach(0, &text), attach(1, &image)],
        ))
        .expect("the frame fires");
    rig.engine
        .settle_frame(&mut ticket)
        .expect("the frame settles");
    eprintln!(
        "a {}-row fire under the 8192 rung took {:.2?} to submit and settle",
        text_rows + image_rows,
        began.elapsed()
    );
    assert_close(
        &ticket.steps[0].readouts[0].values,
        &want.0,
        "text velocity",
    );
    assert_close(
        &ticket.steps[0].readouts[1].values,
        &want.1,
        "image velocity",
    );
}
