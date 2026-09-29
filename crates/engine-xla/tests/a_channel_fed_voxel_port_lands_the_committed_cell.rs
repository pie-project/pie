//! engine-cuda's `a_channel_fed_voxel_port_lands_the_committed_cell`, on the
//! xla engine: a two-axis plan whose VAE reading convolves a voxel port into
//! a pixels seam. The port is fed off a channel (and, beside it, off the
//! step's own payload); each fire reads the cell committed for it, and the
//! plan's token reading still answers its velocity.

mod common_dit;

use common_dit::{
    C_IN, C_OUT, Lcg, Rig, VAE_READING, WIDTH, Weights, assert_close, attach, bf, carrier,
    condition, conv_reference, matmul, modulation, two_axis, vae_word,
};
use engine::fire::{Lane, LaneStream, PortFeed, PortKind, Readout, ReadoutSeam, StepVoxels};
use eta_ir::container::HostRole;
use eta_ir::types::Shape;

const CLIP: [u32; 3] = [1, 5, 7];

const fn voxels() -> u32 {
    CLIP[0] * CLIP[1] * CLIP[2]
}

fn vae_lane(slot: u32, channel: Option<u64>) -> Lane {
    Lane {
        slot,
        word: vae_word(VAE_READING),
        tokens: vec![0],
        readout: Readout::None,
        stream: LaneStream::Image,
        reading: VAE_READING,
        ports: channel
            .map(|channel| PortFeed {
                kind: PortKind::Voxels,
                port: 0,
                channel,
            })
            .into_iter()
            .collect(),
        ..Lane::default()
    }
}

#[test]
fn the_cell_the_channel_holds_is_the_clip_the_convolution_reads() {
    let Some(_device) = common_dit::device() else {
        return;
    };
    let weights = Weights::random(&two_axis(), 0x5a);
    let mut rig = Rig::load(two_axis(), &weights, 32, vec![16, 32], Some(voxels() + 8));
    assert!(
        rig.profile().has_pixels,
        "a plan planting `seam::PIXELS` states the `pixels()` gate"
    );
    assert_eq!(rig.profile().pixels_width, C_OUT);
    assert!(rig.profile().has_velocity);

    let program = rig.register(carrier(&[Shape::new(&[CLIP[1], CLIP[2], C_IN]).expect("a box")], 1));
    let cell = rig.channel(vec![CLIP[1], CLIP[2], C_IN], HostRole::Writer);
    let instance = rig.bind(program, vec![cell], voxels());

    let mut rng = Lcg::seeded(3);
    let mut draw = || -> Vec<f32> {
        (0..voxels() as usize * C_IN as usize)
            .map(|_| bf(rng.unit()))
            .collect()
    };
    let first = draw();
    let second = draw();
    let host = draw();

    let fire = |rig: &mut Rig, clip: &[f32]| -> (Vec<f32>, Vec<[u32; 3]>) {
        rig.publish(instance, 0, clip);
        let readouts = rig.fire(
            vec![vae_lane(0, Some(cell))],
            vec![attach(0, instance)],
            vec![StepVoxels {
                lane: 0,
                clips: vec![CLIP],
                payload: Vec::new(),
            }],
        );
        let readout = readouts[0].clone();
        assert_eq!(readout.seam, ReadoutSeam::Pixels, "a VAE lane answers pixels");
        assert_eq!(readout.width, C_OUT);
        (readout.values, readout.clips)
    };

    let (seam_first, boxes) = fire(&mut rig, &first);
    assert_eq!(boxes, vec![CLIP], "a `same3` convolution keeps the box");
    assert_close(&seam_first, &conv_reference(&weights, CLIP, &first), "the first fire's pixels");

    let (seam_second, _) = fire(&mut rig, &second);
    assert_close(&seam_second, &conv_reference(&weights, CLIP, &second), "the second fire's pixels");
    assert!(
        seam_first.iter().zip(&seam_second).any(|(a, b)| (a - b).abs() > 1e-3),
        "the second fire re-read the first fire's cell"
    );

    // The step's own payload feeds a lane that names no channel.
    let readouts = rig.fire(
        vec![vae_lane(1, None)],
        Vec::new(),
        vec![StepVoxels {
            lane: 0,
            clips: vec![CLIP],
            payload: host.clone(),
        }],
    );
    assert_close(&readouts[0].values, &conv_reference(&weights, CLIP, &host), "the host payload's pixels");

    // The plan's token reading: a velocity off its latent and timestep ports.
    let rows = 5u32;
    let shapes = [Shape::matrix(rows, WIDTH), Shape::matrix(1, 1)];
    let dit = rig.register(carrier(&shapes, 2));
    let latent_ch = rig.channel(vec![rows, WIDTH], HostRole::Writer);
    let time_ch = rig.channel(vec![1, 1], HostRole::Writer);
    let dit_instance = rig.bind(dit, vec![latent_ch, time_ch], rows);
    let latent: Vec<f32> = (0..rows as usize * WIDTH as usize).map(|_| bf(rng.unit())).collect();
    let timestep = 0.4;
    rig.publish(dit_instance, 0, &latent);
    rig.publish(dit_instance, 1, &[timestep]);
    let lane = Lane {
        slot: 2,
        word: vae_word(0),
        tokens: vec![0; rows as usize],
        readout: Readout::Rows((0..rows).collect()),
        stream: LaneStream::Image,
        group: Some(0),
        ports: vec![
            PortFeed {
                kind: PortKind::Latents,
                port: 0,
                channel: latent_ch,
            },
            PortFeed {
                kind: PortKind::LaneVector,
                port: 0,
                channel: time_ch,
            },
        ],
        ..Lane::default()
    };
    let readouts = rig.fire(vec![lane], vec![attach(0, dit_instance)], Vec::new());
    assert_eq!(readouts[0].seam, ReadoutSeam::Velocity);
    let w = WIDTH as usize;
    let m = modulation(&weights, timestep);
    let h = condition(&latent, rows as usize, &m);
    let y = matmul(&h, rows as usize, w, weights.get("o"), w, true);
    let want: Vec<f32> = y.iter().zip(&latent).map(|(a, b)| bf(a + b)).collect();
    assert_close(&readouts[0].values, &want, "the token reading's velocity");
}
