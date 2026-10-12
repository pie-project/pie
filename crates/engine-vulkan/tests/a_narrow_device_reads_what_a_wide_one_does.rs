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

/// The prompt's readout and four decode steps' readouts on device `index`,
/// under the user profile `kernel` (a `[kernel]` body) where one is given.
fn serve(index: u32, artifact: &PathBuf, deployment: &str, kernel: Option<&str>) -> Vec<Vec<f32>> {
    let row = poem_compiler::catalog::deployment(deployment).expect("a catalog row");
    let trace = row.trace(Platform::Vulkan);
    let source = ztensor_compat::index(artifact).expect("the artifact opens");
    let contract = poem::import::own_contract(&source, &trace.params, row.deploy.tp, Platform::Vulkan)
        .expect("the artifact holds the row's planes");
    drop(source);
    let dir = std::env::temp_dir().join(format!("pie-vk-widths-{}-{index}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    if let Some(kernel) = kernel {
        let name = Context::bind(&DeviceBoot {
            device_index: index,
            ..DeviceBoot::default()
        })
        .expect("the device binds")
        .chip()
        .name;
        let path = engine_vulkan::chip::SHELF.user_path(&dir, &name);
        std::fs::create_dir_all(path.parent().expect("a parent")).expect("the shelf");
        std::fs::write(&path, format!("api = 1\n[kernel]\n{kernel}\n")).expect("the file");
    }
    let boot = DeviceBoot {
        device_index: index,
        profiles: kernel.map(|_| dir.clone()),
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
    if kernel.is_some() {
        assert!(
            matches!(shell.tuning().profile().source(), profile::Source::User(_)),
            "the user file was refused: {:?}",
            shell.tuning().profile().source()
        );
    }
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
    engine_vulkan::encode::reset_kernel_profile();
    engine_vulkan::encode::profile_kernels(true);
    let mut rows = vec![fire(&prompt)];
    for step in 0..4u32 {
        rows.push(fire(&[200 + step]));
    }
    engine_vulkan::encode::profile_kernels(false);
    let ran: Vec<String> = engine_vulkan::encode::kernel_profile()
        .into_iter()
        .map(|(name, ..)| name)
        .filter(|name| name.starts_with("sdpa_paged_"))
        .collect();
    eprintln!("{kernel:?}: {ran:?}");
    if let Some(kernel) = kernel {
        let t128 = ran.iter().any(|name| name.contains("_t128"));
        assert_eq!(t128, kernel.contains("threads = 128"), "{kernel}: ran {ran:?}");
        if kernel.contains("groups = 64") {
            assert!(ran.iter().any(|name| name.contains("_split_")), "{kernel}: ran {ran:?}");
        }
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
    let a = serve(narrow.0, &artifact, &deployment, None);
    let b = serve(wide.0, &artifact, &deployment, None);
    agree(&a, &b, &deployment, "between widths");
}

fn agree(a: &[Vec<f32>], b: &[Vec<f32>], deployment: &str, what: &str) {
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let c = cosine(x, y);
        eprintln!("readout {i}: cosine {c:.5}");
        assert!(c > 0.995, "readout {i} of {deployment} differs {what}: cosine {c}");
    }
}

/// Decode attention at 128 invocations (`kernel.sdpa.decode.threads`),
/// whole and split across workgroups, reads what it does at 256, on every
/// device in the process.
#[test]
fn a_decode_at_128_invocations_reads_what_one_at_256_does() {
    let Some((artifact, deployment)) = artifact() else {
        eprintln!("not asked: no Vulkan-stamped artifact in ~/.pie/artifacts");
        return;
    };
    let split = "sdpa.decode.split = { use = \"fixed\", groups = 64 }\nsdpa.decode.split_min_keys = 8";
    for index in 0..4u32 {
        if Context::bind(&DeviceBoot {
            device_index: index,
            ..DeviceBoot::default()
        })
        .is_err()
        {
            break;
        }
        let wide = serve(index, &artifact, &deployment, Some("sdpa.decode.threads = 256"));
        let narrow = serve(index, &artifact, &deployment, Some("sdpa.decode.threads = 128"));
        agree(&wide, &narrow, &deployment, "at 128 invocations");
        let split_wide = serve(
            index,
            &artifact,
            &deployment,
            Some(&format!("sdpa.decode.threads = 256\n{split}")),
        );
        let split_narrow = serve(
            index,
            &artifact,
            &deployment,
            Some(&format!("sdpa.decode.threads = 128\n{split}")),
        );
        agree(&wide, &split_wide, &deployment, "split");
        agree(&wide, &split_narrow, &deployment, "split at 128 invocations");
    }
}
