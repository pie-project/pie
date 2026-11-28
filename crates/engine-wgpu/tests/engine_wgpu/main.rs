//! The shared GPU suite (`crates/engine-gpu/suite`) on the first wgpu
//! adapter (lavapipe through wgpu's Vulkan backend will do).

#![cfg(feature = "wgpu")]

/// What the suite takes from this backend.
mod backend {
    pub use engine_gpu::serve::{Clips, Lane};
    pub use engine_wgpu::api::DeviceBoot;
    pub use engine_wgpu::device::{Api, Buffer, Context, Pipelines};
    pub use engine_wgpu::encode::Sink;

    pub type Handles = engine_gpu::Handles<Api>;
    pub type Shell = engine_gpu::serve::Shell<Api>;
    pub type Boot<'a> = engine_gpu::serve::Boot<'a, Api>;
    pub type Seated<'a> = engine_gpu::serve::Seated<'a, Api>;
    /// The backend's engine.
    pub type Engine = engine_gpu::api::Gpu<Api>;

    /// The backend's boot as a test run states it: the tab's forms this
    /// run takes on a native device (`DeviceBoot::test_forms`), from the
    /// harness's environment, which the shell itself never reads:
    /// `PIE_TEST_STORAGE_BUFFERS` (a stage's storage buffers, below the
    /// adapter's), `PIE_TEST_ALIASING=log` (each overlap said, none
    /// refused), `PIE_TEST_GUEST_COMPACT` and `PIE_TEST_GUEST_ROLLED`.
    pub fn default_boot() -> DeviceBoot {
        DeviceBoot {
            test_forms: engine_wgpu::api::TestForms {
                storage_buffers: std::env::var("PIE_TEST_STORAGE_BUFFERS")
                    .ok()
                    .and_then(|v| v.parse().ok()),
                aliasing_said: std::env::var("PIE_TEST_ALIASING").as_deref() == Ok("log"),
                guest_compact: std::env::var_os("PIE_TEST_GUEST_COMPACT").is_some(),
                guest_rolled: std::env::var_os("PIE_TEST_GUEST_ROLLED").is_some(),
            },
            ..DeviceBoot::default()
        }
    }

    /// wgpu times no frame apart from its own fires here: nothing to watch.
    pub fn watch_frames(_on: bool) {}

    /// `None`: wgpu keeps no longest frame span.
    pub fn longest_frame_ns() -> Option<u64> {
        None
    }

    pub const NAME: &str = "wgpu";
    pub const PLATFORM: poem::Platform = poem::Platform::Wgpu;
    /// Kernels compile to WGSL: no register-tiled or cooperative-matrix
    /// rung, and the subgroup tiers only where every subgroup is 32 lanes.
    pub const SPIRV: bool = false;

    /// Where the logits gates write their rows, keep their fixtures, and
    /// read the pool's chunk size.
    pub const LOGITS: &str = "PIE_WGPU_LOGITS";
    pub const FIXTURES: &str = "PIE_WGPU_FIXTURES";
    pub const CHUNK: &str = "PIE_WGPU_CHUNK";

    /// What a test's load takes: `[engine.memory] budget_device` is
    /// stated, so a test takes no more than this of a device it shares.
    pub const ALLOWANCE: engine::Allowance = engine::Allowance {
        budget: Some(8 << 30),
    };

    /// The narrowest subgroup the adapter runs (0 without subgroups).
    pub fn subgroup(device: &Context) -> u32 {
        device.chip().subgroup_min.unwrap_or(0)
    }

    pub fn coopmat(_: &Context) -> bool {
        false
    }

    /// The adapter profile's moe tile rows and sorted rows for `pairs`
    /// routes over `experts`, and its sort's lanes.
    pub fn moe_rows(device: &Context, pairs: u32, experts: u32) -> (u32, u32, u32) {
        use engine_gpu::backend::Tactics as _;
        let (profile, _) = engine_wgpu::chip::resolve(&device.chip(), None);
        let tuning = engine_wgpu::tuning::Tuning::new(profile);
        (
            tuning.moe_tile_rows(pairs, experts),
            tuning.moe_sorted_rows(pairs, experts),
            256,
        )
    }
}

/// A test of a SPIR-V-only form: reported skipped here.
macro_rules! spirv_only {
    ($test:item) => {
        #[ignore = "SPIR-V only: it folds through subgroups WGSL does not have"]
        $test
    };
}

#[path = "../../../engine-gpu/suite/attn/mod.rs"]
mod attn;
mod built_at_load;
#[path = "../../../engine-gpu/suite/collective/mod.rs"]
mod collective;
#[path = "../../../engine-gpu/suite/conv/mod.rs"]
mod conv;
#[path = "../../../engine-gpu/suite/elemwise/mod.rs"]
mod elemwise;
#[path = "../../../engine-gpu/suite/gather/mod.rs"]
mod gather;
#[path = "../../../engine-gpu/suite/gpu.rs"]
mod gpu;
#[path = "../../../engine-gpu/suite/linear/mod.rs"]
mod linear;
#[path = "../../../engine-gpu/suite/model_logits.rs"]
mod model_logits;
#[path = "../../../engine-gpu/suite/runtime/mod.rs"]
mod runtime;
#[path = "../../../engine-gpu/suite/sampling/mod.rs"]
mod sampling;
#[path = "../../../engine-gpu/suite/ssm/mod.rs"]
mod ssm;
