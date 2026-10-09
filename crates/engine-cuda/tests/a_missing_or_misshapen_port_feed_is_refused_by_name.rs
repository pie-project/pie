#![cfg(feature = "cuda")]

mod common_dit;

use common_dit::{Rig, WIDTH, Weights, attach, frame, lane};
use engine::Engine;
use engine::fire::{LaneStream, PortKind};

#[test]
fn a_missing_feed_and_a_wrong_cell_are_refused_by_name() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let weights = Weights::random(&common_dit::trace(), 13);
    let mut rig = Rig::load(&weights, 32, vec![32]);
    let rows = 3u32;
    let handles = rig.lane(rows);
    let w = WIDTH as usize;
    rig.publish(handles.instance, 1, &[0.3]);
    rig.publish(handles.instance, 2, &vec![0.0; rows as usize * 2]);

    let mut bare = lane(&rig, 0, &handles, LaneStream::Text, 0);
    bare.ports.retain(|feed| feed.kind != PortKind::Latents);
    let refusal = rig
        .engine
        .submit(&frame(vec![bare], vec![attach(0, &handles)]))
        .expect_err("a lane feeding no latents is refused")
        .to_string();
    assert!(
        refusal.contains("Latents port 0") && refusal.contains("feeds it no channel"),
        "the refusal names the port: {refusal}"
    );

    let wide = rig.lane(rows + 1);
    rig.publish(wide.instance, 0, &vec![0.0; (rows as usize + 1) * w]);
    rig.publish(wide.instance, 1, &[0.3]);
    rig.publish(wide.instance, 2, &vec![0.0; (rows as usize + 1) * 2]);
    let mut short = lane(&rig, 0, &wide, LaneStream::Text, 0);
    short.tokens = vec![0; rows as usize];
    short.readout = engine::fire::Readout::Rows((0..rows).collect());
    let refusal = rig
        .engine
        .submit(&frame(vec![short], vec![attach(0, &wide)]))
        .expect_err("a cell of the wrong shape is refused")
        .to_string();
    assert!(
        refusal.contains("Latents port 0")
            && refusal.contains("whose cell is")
            && refusal.contains(&format!("{rows} row(s) x {WIDTH}")),
        "the refusal names the port and the shapes: {refusal}"
    );

    let mut stray = lane(&rig, 0, &handles, LaneStream::Text, 0);
    rig.publish(handles.instance, 0, &vec![0.0; rows as usize * w]);
    stray.ports.push(engine::fire::PortFeed {
        kind: PortKind::Context,
        port: 0,
        channel: handles.latent,
    });
    let refusal = rig
        .engine
        .submit(&frame(vec![stray], vec![attach(0, &handles)]))
        .expect_err("a feed for an undeclared port is refused")
        .to_string();
    assert!(
        refusal.contains("Context port 0") && refusal.contains("declares no such port"),
        "the refusal names the port: {refusal}"
    );
}
