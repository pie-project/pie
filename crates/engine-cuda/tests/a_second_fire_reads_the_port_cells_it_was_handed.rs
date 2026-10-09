#![cfg(feature = "cuda")]

mod common_dit;

use common_dit::{
    HostRequest, Lcg, Rig, WIDTH, Weights, assert_close, attach, bf, frame, lane, reference,
};
use engine::Engine;
use engine::fire::LaneStream;

fn request(rng: &mut Lcg, text_rows: usize, image_rows: usize) -> HostRequest {
    let w = WIDTH as usize;
    HostRequest {
        text: (0..text_rows * w).map(|_| bf(rng.unit())).collect(),
        image: (0..image_rows * w).map(|_| bf(rng.unit())).collect(),
        text_rows,
        image_rows,
        timestep: 0.7,
        positions: (0..text_rows + image_rows)
            .map(|r| [r as f32, 0.5 * r as f32])
            .collect(),
    }
}

#[test]
fn the_second_fire_lands_the_second_cells() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let weights = Weights::random(&common_dit::trace(), 11);
    let mut rig = Rig::load(&weights, 32, vec![16, 32]);
    let mut rng = Lcg::seeded(5);
    let (text_rows, image_rows) = (2, 4);
    let first = request(&mut rng, text_rows, image_rows);
    let second = HostRequest {
        text: (0..text_rows * WIDTH as usize)
            .map(|_| bf(rng.unit()))
            .collect(),
        image: (0..image_rows * WIDTH as usize)
            .map(|_| bf(rng.unit()))
            .collect(),
        ..request(&mut rng, text_rows, image_rows)
    };
    let want_first = reference(&weights, &first);
    let want_second = reference(&weights, &second);

    let text = rig.lane(text_rows as u32);
    let image = rig.lane(image_rows as u32);
    for handles in [&text, &image] {
        rig.publish(handles.instance, 1, &[first.timestep]);
    }
    rig.publish(
        text.instance,
        2,
        &first.positions[..text_rows]
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<f32>>(),
    );
    rig.publish(
        image.instance,
        2,
        &first.positions[text_rows..]
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<f32>>(),
    );

    let submission = |rig: &mut Rig, req: &HostRequest| -> Vec<Vec<f32>> {
        rig.publish(text.instance, 0, &req.text);
        rig.publish(image.instance, 0, &req.image);
        let lanes = vec![
            lane(&rig, 0, &text, LaneStream::Text, 0),
            lane(&rig, 1, &image, LaneStream::Image, 0),
        ];
        let attachments = vec![attach(0, &text), attach(1, &image)];
        let mut ticket = rig
            .engine
            .submit(&frame(lanes, attachments))
            .expect("the frame fires");
        rig.engine
            .settle_frame(&mut ticket)
            .expect("the frame settles");
        ticket.steps[0]
            .readouts
            .iter()
            .map(|readout| readout.values.clone())
            .collect()
    };

    let got_first = submission(&mut rig, &first);
    assert_close(&got_first[0], &want_first.0, "first fire, text");
    assert_close(&got_first[1], &want_first.1, "first fire, image");

    let got_second = submission(&mut rig, &second);
    assert_close(&got_second[0], &want_second.0, "second fire, text");
    assert_close(&got_second[1], &want_second.1, "second fire, image");

    let moved = got_first[1]
        .iter()
        .zip(&got_second[1])
        .any(|(a, b)| (a - b).abs() > 1e-2);
    assert!(moved, "the second fire re-read the first fire's cells");
}
