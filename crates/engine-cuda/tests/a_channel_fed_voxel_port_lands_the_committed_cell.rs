#![cfg(feature = "cuda")]

mod common_two_axis;

use common_two_axis::{
    C_IN, C_OUT, Lcg, Rig, Weights, assert_close, attach, bf, conv_reference, frame,
    pixel_epilogue, trace,
};
use engine::Engine;
use engine::fire::{ReadoutSeam, StepVoxels};

const CLIP: [u32; 3] = [1, 5, 7];

const fn voxels() -> u32 {
    CLIP[0] * CLIP[1] * CLIP[2]
}

#[test]
fn the_cell_the_channel_holds_is_the_clip_the_convolution_reads() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let weights = Weights::random(&trace(), 0x5a);
    let mut rig = Rig::load(&weights, 32, vec![16, 32], voxels() + 8);
    assert!(
        rig.profile().has_pixels,
        "a plan planting `seam::PIXELS` states the `pixels()` gate"
    );

    let program = rig.register(pixel_epilogue(CLIP, voxels()), 1);
    let cell = rig.channel(
        vec![CLIP[1], CLIP[2], C_IN],
        eta_ir::container::HostRole::Writer,
    );
    let back = rig.channel(vec![voxels(), C_OUT], eta_ir::container::HostRole::Reader);
    let instance = rig.bind(program, vec![cell, back], voxels());

    let mut rng = Lcg::seeded(3);
    let mut draw = || -> Vec<f32> {
        (0..voxels() as usize * C_IN as usize)
            .map(|_| bf(rng.unit()))
            .collect()
    };
    let first = draw();
    let second = draw();

    let fire = |rig: &mut Rig, clip: &[f32]| -> (Vec<f32>, Vec<[u32; 3]>, Vec<f32>) {
        rig.publish(instance, 0, clip);
        let mut ticket = rig
            .engine
            .submit(&frame(
                vec![rig.vae_lane(0, cell)],
                vec![attach(0, instance)],
                vec![StepVoxels {
                    lane: 0,
                    clips: vec![CLIP],
                    payload: Vec::new(),
                }],
            ))
            .expect("the frame fires");
        rig.engine
            .settle_frame(&mut ticket)
            .expect("the frame settles");
        let readout = ticket.steps[0].readouts[0].clone();
        assert_eq!(
            readout.seam,
            ReadoutSeam::Pixels,
            "a VAE lane answers pixels"
        );
        (readout.values, readout.clips, rig.take(instance, 1))
    };

    let (seam_first, boxes, guest_first) = fire(&mut rig, &first);
    assert_eq!(boxes, vec![CLIP], "a `same3` convolution keeps the box");
    let want_first = conv_reference(&weights, CLIP, &first);
    assert_close(&seam_first, &want_first, "the first fire's pixels seam");
    assert_close(&guest_first, &want_first, "the first fire's `pixels()`");

    let (seam_second, _, guest_second) = fire(&mut rig, &second);
    let want_second = conv_reference(&weights, CLIP, &second);
    assert_close(&seam_second, &want_second, "the second fire's pixels seam");
    assert_close(&guest_second, &want_second, "the second fire's `pixels()`");

    let moved = guest_first
        .iter()
        .zip(&guest_second)
        .any(|(a, b)| (a - b).abs() > 1e-3);
    assert!(moved, "the second fire re-read the first fire's cell");
}
