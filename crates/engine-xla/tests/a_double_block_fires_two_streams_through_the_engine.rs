//! engine-cuda's `a_double_block_fires_two_streams_through_the_engine` and
//! `a_second_fire_reads_the_port_cells_it_was_handed`, on the xla engine:
//! two requests of a text lane and an image lane each, fed latents, a
//! timestep and axis positions off their channels, attend within their
//! group through a row permutation, and read back the velocity seam; a
//! second fire lands the second cells.

mod common_dit;

use common_dit::{
    HostRequest, Lcg, Rig, WIDTH, Weights, assert_close, attach, bf, lane, reference,
    reference_with, trace,
};
use engine::fire::{LaneStream, ReadoutSeam};

fn request(rng: &mut Lcg, text_rows: usize, image_rows: usize) -> HostRequest {
    let w = WIDTH as usize;
    let rows = text_rows + image_rows;
    HostRequest {
        text: (0..text_rows * w).map(|_| bf(rng.unit())).collect(),
        image: (0..image_rows * w).map(|_| bf(rng.unit())).collect(),
        text_rows,
        image_rows,
        timestep: 0.5 + 0.5 * rng.unit(),
        positions: (0..rows)
            .map(|r| [r as f32, (r % 3) as f32 + 0.25 * rng.unit()])
            .collect(),
    }
}

fn publish(
    rig: &mut Rig,
    handles: &common_dit::LaneHandles,
    latent: &[f32],
    t: f32,
    pos: &[[f32; 2]],
) {
    rig.publish(handles.instance, 0, latent);
    rig.publish(handles.instance, 1, &[t]);
    rig.publish(
        handles.instance,
        2,
        &pos.iter().flatten().copied().collect::<Vec<f32>>(),
    );
}

#[test]
fn the_double_block_lands_the_host_reference_on_four_lanes() {
    let Some(_device) = common_dit::device() else {
        return;
    };
    let weights = Weights::random(&trace(), 7);
    let mut rig = Rig::load(trace(), &weights, 64, vec![16, 32, 64], None);
    assert!(
        rig.profile().has_velocity,
        "the plan plants a velocity seam"
    );
    assert_eq!(rig.profile().velocity_width, WIDTH);
    assert_eq!(rig.profile().vocab, 0, "a denoiser has no vocabulary");

    let mut rng = Lcg::seeded(3);
    let requests = [request(&mut rng, 3, 5), request(&mut rng, 4, 6)];
    let seconds = [request(&mut rng, 3, 5), request(&mut rng, 4, 6)];
    let want: Vec<(Vec<f32>, Vec<f32>)> = requests.iter().map(|r| reference(&weights, r)).collect();
    let want_second: Vec<(Vec<f32>, Vec<f32>)> = requests
        .iter()
        .zip(&seconds)
        .map(|(first, second)| {
            // The second cells: new latents, the first request's timestep
            // and positions published again.
            reference(
                &weights,
                &HostRequest {
                    text: second.text.clone(),
                    image: second.image.clone(),
                    text_rows: first.text_rows,
                    image_rows: first.image_rows,
                    timestep: first.timestep,
                    positions: first.positions.clone(),
                },
            )
        })
        .collect();

    let mut lanes = Vec::new();
    let mut attachments = Vec::new();
    let mut handles = Vec::new();
    for (at, req) in requests.iter().enumerate() {
        let text = rig.lane(req.text_rows as u32);
        let image = rig.lane(req.image_rows as u32);
        publish(
            &mut rig,
            &text,
            &req.text,
            req.timestep,
            &req.positions[..req.text_rows],
        );
        publish(
            &mut rig,
            &image,
            &req.image,
            req.timestep,
            &req.positions[req.text_rows..],
        );
        let slot = (2 * at) as u32;
        // Request 0 submits its image lane first: the packing, not the
        // submission order, puts a group's rows together.
        if at == 0 {
            lanes.push(lane(slot, &image, LaneStream::Image, at as u32));
            lanes.push(lane(slot + 1, &text, LaneStream::Text, at as u32));
            attachments.push(attach(lanes.len() as u32 - 2, image.instance));
            attachments.push(attach(lanes.len() as u32 - 1, text.instance));
        } else {
            lanes.push(lane(slot, &text, LaneStream::Text, at as u32));
            lanes.push(lane(slot + 1, &image, LaneStream::Image, at as u32));
            attachments.push(attach(lanes.len() as u32 - 2, text.instance));
            attachments.push(attach(lanes.len() as u32 - 1, image.instance));
        }
        handles.push((text, image));
    }
    let readouts = rig.fire(lanes.clone(), attachments.clone(), Vec::new());
    assert_eq!(readouts.len(), 4);
    for readout in &readouts {
        assert_eq!(readout.seam, ReadoutSeam::Velocity);
        assert_eq!(readout.width, WIDTH);
    }
    assert_close(&readouts[0].values, &want[0].1, "request 0 image velocity");
    assert_close(&readouts[1].values, &want[0].0, "request 0 text velocity");
    assert_close(&readouts[2].values, &want[1].0, "request 1 text velocity");
    assert_close(&readouts[3].values, &want[1].1, "request 1 image velocity");

    // The carrier took the first cells; the second fire reads the second.
    for ((text, image), (first, second)) in handles.iter().zip(requests.iter().zip(&seconds)) {
        publish(
            &mut rig,
            text,
            &second.text,
            first.timestep,
            &first.positions[..first.text_rows],
        );
        publish(
            &mut rig,
            image,
            &second.image,
            first.timestep,
            &first.positions[first.text_rows..],
        );
    }
    let readouts = rig.fire(lanes.clone(), attachments.clone(), Vec::new());
    assert_close(
        &readouts[0].values,
        &want_second[0].1,
        "second fire, request 0 image",
    );
    assert_close(
        &readouts[1].values,
        &want_second[0].0,
        "second fire, request 0 text",
    );
    assert_close(
        &readouts[2].values,
        &want_second[1].0,
        "second fire, request 1 text",
    );
    assert_close(
        &readouts[3].values,
        &want_second[1].1,
        "second fire, request 1 image",
    );

    // Attention classes: text rows class 0, image rows class 1, and a table
    // that keeps each class to itself, so each stream attends only within
    // itself (engine-cuda's ragged `ClassTable`).
    let apart = [request(&mut rng, 3, 5), request(&mut rng, 4, 6)];
    let want_apart: Vec<(Vec<f32>, Vec<f32>)> = requests
        .iter()
        .zip(&apart)
        .map(|(first, next)| {
            reference_with(
                &weights,
                &HostRequest {
                    text: next.text.clone(),
                    image: next.image.clone(),
                    text_rows: first.text_rows,
                    image_rows: first.image_rows,
                    timestep: first.timestep,
                    positions: first.positions.clone(),
                },
                false,
            )
        })
        .collect();
    for ((text, image), (first, next)) in handles.iter().zip(requests.iter().zip(&apart)) {
        publish(
            &mut rig,
            text,
            &next.text,
            first.timestep,
            &first.positions[..first.text_rows],
        );
        publish(
            &mut rig,
            image,
            &next.image,
            first.timestep,
            &first.positions[first.text_rows..],
        );
    }
    let mut classed = lanes;
    for lane in &mut classed {
        let class = i32::from(lane.stream != LaneStream::Text);
        lane.attn_classes = Some(engine::fire::AttnClasses {
            classes: vec![class; lane.tokens.len()],
            table: vec![1, 0, 0, 1],
            count: 2,
        });
    }
    let readouts = rig.fire(classed, attachments, Vec::new());
    assert_close(
        &readouts[0].values,
        &want_apart[0].1,
        "classed, request 0 image",
    );
    assert_close(
        &readouts[1].values,
        &want_apart[0].0,
        "classed, request 0 text",
    );
    assert_close(
        &readouts[2].values,
        &want_apart[1].0,
        "classed, request 1 text",
    );
    assert_close(
        &readouts[3].values,
        &want_apart[1].1,
        "classed, request 1 image",
    );
}
