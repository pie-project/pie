//! The Vulkan seam, against a real artifact.
//!
//! Every other test of this crate's Vulkan path stops at the door: the
//! backend opens a device, states its facts, and refuses what it cannot
//! serve. What none of them do is put a MODEL through it, because doing so
//! needs three things at once -- a Vulkan device, the compiled SPIR-V, and a
//! checkpoint the catalog identifies whose weights are the quantized layout
//! this driver binds.
//!
//! All three exist now. `pie model build --backend vulkan` authors the third
//! from an `mlx-community/*-4bit` snapshot, so this is the first test in the
//! workspace where the seam's `load_model` reads real bytes off a disk and
//! the capabilities it answers with are the checkpoint's own numbers rather
//! than a fixture's.
//!
//! # Why it skips
//!
//! `PIE_VULKAN_ARTIFACT` names `.zt` artifacts built for this backend,
//! colon-separated, and `PIE_KERNELS_VULKAN_SPV_DIR` names the module
//! directory. Without either,
//! this prints and returns: a test that passed on a machine with no artifact
//! would report the absence of the checkpoint as the presence of a working
//! load.

#![cfg(feature = "driver-vulkan")]

use engine::driver::backend::open;

/// The boot TOML `worker::embedded_driver::write_vulkan_startup_toml` writes.
fn boot(modules: &str) -> Vec<u8> {
    format!("[model]\nkernels = \"{modules}\"\nkv_pages = 64\n").into_bytes()
}

fn env(key: &str) -> Option<String> {
    match std::env::var(key) {
        Ok(value) if !value.is_empty() => Some(value),
        _ => None,
    }
}

/// What each row this seam has been measured against answers with.
///
/// A table rather than one model's numbers, because an artifact is named by
/// a path and identified by its TENSORS: the test cannot know which row it
/// was handed until the load says so, and asserting whatever the load said
/// would assert nothing. So the row picks its own row of expectations, and a
/// row that is not in this table is a refusal.
const MEASURED: &[(&str, &str, u32, u32)] = &[
    ("qwen3-0.6b", "qwen3", 151_936, 1024),
    ("qwen2.5-1.5b", "qwen2", 151_936, 1536),
];

/// A real checkpoint, loaded, answering with its own shape.
///
/// The numbers asserted are the identified row's own and are checked as
/// equalities: a load that quietly identified another row would have to
/// answer that row's four.
#[test]
fn the_seam_loads_an_artifact_and_answers_with_the_checkpoint_s_own_shape() {
    let (Some(modules), Some(artifacts)) = (
        env("PIE_KERNELS_VULKAN_SPV_DIR"),
        env("PIE_VULKAN_ARTIFACT"),
    ) else {
        eprintln!("SKIP: PIE_KERNELS_VULKAN_SPV_DIR and PIE_VULKAN_ARTIFACT name the inputs");
        return;
    };
    let mut seen: Vec<String> = Vec::new();
    for artifact in artifacts.split(':').filter(|a| !a.is_empty()) {
        let Ok(mut backend) = open::vulkan(&boot(&modules)) else {
            eprintln!("SKIP: no Vulkan device");
            return;
        };
        assert_eq!(
            backend
                .device_facts()
                .expect("a local driver knows its device")
                .backend,
            "vulkan"
        );

        let caps = backend
            .load_model(vec![driver_api::ModelLoadDesc {
                snapshot_dir: std::path::PathBuf::from(artifact),
                // "Whatever the checkpoint is": the artifact was authored
                // with its quantization already baked in, so a second request
                // here would be a requantization of a requantization.
                runtime_quant: String::new(),
                mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
                component: driver_api::ModelComponent::Text,
            }])
            .unwrap_or_else(|e| panic!("{artifact} loads: {e}"));

        let Some((_, arch, vocab, hidden)) = MEASURED.iter().find(|(id, ..)| *id == caps.model_id)
        else {
            panic!(
                "{artifact} identified as `{}`, which this test has no measured \
                 numbers for",
                caps.model_id
            );
        };
        assert_eq!(&caps.arch_name, arch);
        assert_eq!(caps.vocab_size, *vocab);
        assert_eq!(caps.hidden_size, *hidden);
        // These two ARE the boot file's, and they are the pair that proves
        // the seam read it: 64 pages was asked for above.
        assert_eq!(caps.total_pages, 64);
        assert!(caps.kv_page_size > 0, "a page with no rows holds nothing");
        assert_eq!(caps.activation_dtype, "bf16");
        seen.push(caps.model_id.clone());
    }
    // The same artifact twice would run this twice and prove it once.
    let mut once = seen.clone();
    once.sort();
    once.dedup();
    assert_eq!(
        once.len(),
        seen.len(),
        "the same row was served twice: {seen:?}"
    );
    assert!(!seen.is_empty(), "PIE_VULKAN_ARTIFACT named nothing");
}
