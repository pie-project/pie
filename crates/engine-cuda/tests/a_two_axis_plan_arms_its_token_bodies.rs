#![cfg(feature = "cuda")]

mod common_two_axis;

use common_two_axis::{
    C_IN, C_OUT, Lcg, Rig, WIDTH, Weights, assert_close, attach, bf, conv_reference, frame,
    pixel_epilogue, trace,
};
use engine::Engine;
use engine::fire::{ReadoutSeam, StepVoxels};
use eta_ir::container::HostRole;

const CLIP: [u32; 3] = [1, 4, 6];
const ROWS: u32 = 8;

const fn voxels() -> u32 {
    CLIP[0] * CLIP[1] * CLIP[2]
}

#[test]
fn the_token_regions_arm_while_the_voxel_regions_stay_eager() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let weights = Weights::random(&trace(), 0x2b);
    let mut rig = Rig::load(&weights, 32, vec![16, 32], voxels() + 8);

    let stats = rig
        .engine
        .shell()
        .expect("the load holds a shell")
        .body_stats();
    println!("[two-axis] {stats}");
    assert!(
        stats.census.bodies > 0,
        "a plan stating voxel rows armed no token body: {stats}"
    );
    assert!(
        stats.tally.armed_at_load > 0,
        "the arming pass recorded nothing at load: {stats}"
    );

    let program = rig.register(pixel_epilogue(CLIP, voxels()), 1);
    let cell = rig.channel(vec![CLIP[1], CLIP[2], C_IN], HostRole::Writer);
    let back = rig.channel(vec![voxels(), C_OUT], HostRole::Reader);
    let vae = rig.bind(program, vec![cell, back], voxels());

    let mut rng = Lcg::seeded(9);
    let clip: Vec<f32> = (0..voxels() as usize * C_IN as usize)
        .map(|_| bf(rng.unit()))
        .collect();
    rig.publish(vae, 0, &clip);
    let mut ticket = rig
        .engine
        .submit(&frame(
            vec![rig.vae_lane(0, cell)],
            vec![attach(0, vae)],
            vec![StepVoxels {
                lane: 0,
                clips: vec![CLIP],
                payload: Vec::new(),
            }],
        ))
        .expect("the voxel arm fires");
    rig.engine
        .settle_frame(&mut ticket)
        .expect("the voxel fire settles");
    assert_eq!(ticket.steps[0].readouts[0].seam, ReadoutSeam::Pixels);
    assert_close(
        &rig.take(vae, 1),
        &conv_reference(&weights, CLIP, &clip),
        "the voxel arm's pixels",
    );

    let velocity = rig.register(common_two_axis::velocity_epilogue(ROWS), 2);
    let latent = rig.channel(vec![ROWS, WIDTH], HostRole::Writer);
    let timestep = rig.channel(vec![1, 1], HostRole::Writer);
    let positions = rig.channel(vec![ROWS, 2], HostRole::Writer);
    let out = rig.channel(vec![ROWS, WIDTH], HostRole::Reader);
    let dit = rig.bind(velocity, vec![latent, timestep, positions, out], ROWS);
    rig.publish(
        dit,
        0,
        &(0..ROWS as usize * WIDTH as usize)
            .map(|_| bf(rng.unit()))
            .collect::<Vec<f32>>(),
    );
    rig.publish(dit, 1, &[0.7]);
    rig.publish(
        dit,
        2,
        &(0..ROWS)
            .flat_map(|r| [r as f32, 0.5 * r as f32])
            .collect::<Vec<f32>>(),
    );
    let mut ticket = rig
        .engine
        .submit(&frame(
            vec![rig.dit_lane(0, ROWS, latent, timestep, positions)],
            vec![attach(0, dit)],
            Vec::new(),
        ))
        .expect("the token arm fires");
    rig.engine
        .settle_frame(&mut ticket)
        .expect("the token fire settles");
    let answered = rig.take(dit, 3);
    assert_eq!(answered.len(), (ROWS * WIDTH) as usize);
    assert!(
        answered.iter().all(|v| v.is_finite()) && answered.iter().any(|v| v.abs() > 1e-6),
        "the token arm answered nothing"
    );

    let after = rig
        .engine
        .shell()
        .expect("the load holds a shell")
        .body_stats();
    println!("[two-axis, after] {after}");
    assert!(
        after.tally.hits > stats.tally.hits,
        "no fire replayed a body: {after}"
    );
}
