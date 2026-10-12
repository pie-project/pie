#![cfg(feature = "vulkan")]

//! A device whose subgroups run narrower than 32 lanes (lavapipe: 8) serves
//! the same readouts as one pinned to 32 (an NVIDIA GPU): decode attention's
//! per-lane registers once held HEAD_DIM / 32 dims whatever the width, so a
//! narrow device scored half the head and read uninitialised workgroup
//! memory for the rest.
//!
//! Needs both devices in one process (`VK_ICD_FILENAMES=<lavapipe>:<gpu>`)
//! and a Vulkan-stamped artifact of a small model in `~/.pie/artifacts`;
//! skips without them.

use std::path::PathBuf;

use engine_vulkan::api::DeviceBoot;
use engine_vulkan::device::Context;
use engine_vulkan::{Boot, Lane, Shell};
use poem::{Platform, Request};

const CONTEXT: u32 = 512;

fn artifact() -> Option<(PathBuf, String)> {
    let store = PathBuf::from(std::env::var_os("HOME")?).join(".pie/artifacts");
    engine_vulkan::tune::artifacts(&store).into_iter().next()
}

/// The prompt's readout and four decode steps' readouts on device `index`.
fn serve(index: u32, artifact: &PathBuf, deployment: &str) -> Vec<Vec<f32>> {
    let row = poem_compiler::catalog::deployment(deployment).expect("a catalog row");
    let trace = row.trace(Platform::Vulkan);
    let source = ztensor_compat::index(artifact).expect("the artifact opens");
    let contract = poem::import::own_contract(&source, &trace.params, row.deploy.tp, Platform::Vulkan)
        .expect("the artifact holds the row's planes");
    drop(source);
    let boot = DeviceBoot {
        device_index: index,
        ..DeviceBoot::default()
    };
    let patches = engine_vulkan::api::patch_ladder(
        &trace,
        &engine::load::Budgets {
            max_lanes: 1,
            max_tokens: CONTEXT,
            ..engine::load::Budgets::default()
        },
    );
    let mut shell = Shell::load(Boot {
        trace: trace.clone(),
        contract: &contract,
        checkpoint: artifact,
        budget: poem_compiler::Budget::new(1, CONTEXT),
        patches,
        profile: None,
        page_size: 16,
        context: CONTEXT,
        slots: 1,
        pages: CONTEXT / 16,
        runahead: engine::runahead::Runahead::F1,
        device: &boot,
        residency: engine::load::Residency::default(),
    })
    .expect("the shell loads");
    let prompt: Vec<u32> = (0..40u32).map(|i| 100 + (i * 7919) % 5000).collect();
    shell.open(0).expect("the slot opens");
    let mut fire = |tokens: &[u32]| {
        shell
            .fire(&[Lane {
                slot: 0,
                word: trace.facts.word(&Request::new(tokens.len() as u32, false)),
                tokens,
            }])
            .expect("the fire lands")
            .into_iter()
            .next()
            .expect("one readout")
    };
    let mut rows = vec![fire(&prompt)];
    for step in 0..4u32 {
        rows.push(fire(&[200 + step]));
    }
    rows
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += f64::from(x) * f64::from(y);
        aa += f64::from(x) * f64::from(x);
        bb += f64::from(y) * f64::from(y);
    }
    dot / (aa * bb).sqrt()
}

#[test]
fn a_narrow_device_reads_what_a_wide_one_does() {
    let lanes: Vec<(u32, u32, String)> = (0..4u32)
        .filter_map(|i| {
            let ctx = Context::bind(&DeviceBoot {
                device_index: i,
                ..DeviceBoot::default()
            })
            .ok()?;
            Some((i, ctx.lanes(), ctx.chip().name))
        })
        .collect();
    let narrow = lanes.iter().find(|(_, l, _)| *l < 32);
    let wide = lanes.iter().find(|(_, l, _)| *l >= 32);
    let (Some(narrow), Some(wide)) = (narrow, wide) else {
        eprintln!("not asked: no narrow and wide device side by side: {lanes:?}");
        return;
    };
    let Some((artifact, deployment)) = artifact() else {
        eprintln!("not asked: no Vulkan-stamped artifact in ~/.pie/artifacts");
        return;
    };
    eprintln!("{deployment}: {} ({} lanes) against {} ({} lanes)", narrow.2, narrow.1, wide.2, wide.1);
    let a = serve(narrow.0, &artifact, &deployment);
    let b = serve(wide.0, &artifact, &deployment);
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        let c = cosine(x, y);
        eprintln!("readout {i}: cosine {c:.5}");
        assert!(c > 0.995, "readout {i} of {deployment} differs between widths: cosine {c}");
    }
}
