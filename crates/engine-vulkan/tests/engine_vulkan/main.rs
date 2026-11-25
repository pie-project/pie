//! The shared GPU suite (`crates/engine-gpu/suite`) on the first Vulkan
//! device (a software one such as lavapipe will do).

#![cfg(feature = "vulkan")]

/// What the suite takes from this backend.
mod backend {
    pub use engine_gpu::serve::{Clips, Lane};
    pub use engine_vulkan::api::DeviceBoot;
    pub use engine_vulkan::device::{Api, Buffer, Context, Pipelines};
    pub use engine_vulkan::encode::Sink;

    pub type Handles = engine_gpu::Handles<Api>;
    pub type Shell = engine_gpu::serve::Shell<Api>;
    pub type Boot<'a> = engine_gpu::serve::Boot<'a, Api>;
    pub type Seated<'a> = engine_gpu::serve::Seated<'a, Api>;
    /// The backend's engine.
    pub type Engine = engine_gpu::api::Gpu<Api>;

    /// A timed step's frame committed as a serving step is (kept and
    /// replayed where `keep`), and its device time read as the harvest reads
    /// it, from its timestamps once its fence was waited.
    pub fn fired_time(frame: engine_vulkan::device::Frame, keep: bool) -> Option<u64> {
        let pending = if keep {
            frame
                .into_kept()
                .and_then(|kept| kept.replay(None))
                .expect("the kept frame replays")
        } else {
            frame.commit_async(None).expect("the frame is submitted")
        };
        pending.wait().expect("the frame lands");
        pending.span_ns().expect("its timestamps read")
    }

    /// Every frame this thread waits on from here timed by the device's
    /// timestamps (`on`), or no longer.
    pub fn watch_frames(on: bool) {
        engine_vulkan::encode::profile_kernels(on);
        let _ = engine_vulkan::encode::take_longest_frame_span_ns();
    }

    /// The longest device span of a frame this thread waited on since
    /// [`watch_frames`]; `None` where the backend times none.
    pub fn longest_frame_ns() -> Option<u64> {
        Some(engine_vulkan::encode::take_longest_frame_span_ns())
    }

    /// The backend's profile registry: what a user's or a committed file is
    /// read against.
    pub fn registry() -> &'static profile::Registry {
        engine_vulkan::chip::registry()
    }

    pub const NAME: &str = "vulkan";
    pub const PLATFORM: poem::Platform = poem::Platform::Vulkan;
    /// Kernels compile to SPIR-V: the register-tiled and cooperative-matrix
    /// rungs, and the subgroup tiers everywhere, are its own.
    pub const SPIRV: bool = true;

    /// Where the logits gates write their rows, keep their fixtures, and
    /// read the pool's chunk size.
    pub const LOGITS: &str = "PIE_VK_LOGITS";
    pub const FIXTURES: &str = "PIE_VK_FIXTURES";
    pub const CHUNK: &str = "PIE_VK_CHUNK";

    /// What a test's load takes: the whole device.
    pub const ALLOWANCE: engine::Allowance = engine::Allowance::WHOLE;

    /// The narrowest subgroup the device runs.
    pub fn subgroup(device: &Context) -> u32 {
        device.chip().subgroup
    }

    pub fn coopmat(device: &Context) -> bool {
        device.chip().coopmat
    }

    /// The device profile's moe tile rows and sorted rows for `pairs` routes
    /// over `experts`, and its sort's lanes.
    pub fn moe_rows(device: &Context, pairs: u32, experts: u32) -> (u32, u32, u32) {
        use engine_gpu::backend::Tactics as _;
        let (profile, _) = engine_vulkan::chip::resolve(&device.chip(), None);
        let tuning =
            engine_vulkan::tuning::Tuning::new(profile, engine_vulkan::tuning::Limits::ASSUMED);
        (
            tuning.moe_tile_rows(pairs, experts),
            tuning.moe_sorted_rows(pairs, experts),
            tuning.sort_lanes(),
        )
    }
}

/// A test of a SPIR-V-only form: it runs here.
macro_rules! spirv_only {
    ($test:item) => {
        $test
    };
}

#[path = "../../../engine-gpu/suite/attn/mod.rs"]
mod attn;
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
mod fills_verified;
mod rates_bound;
mod replay;
#[path = "../../../engine-gpu/suite/model_logits.rs"]
mod model_logits;
#[path = "../../../engine-gpu/suite/runtime/mod.rs"]
mod runtime;
#[path = "../../../engine-gpu/suite/sampling/mod.rs"]
mod sampling;
#[path = "../../../engine-gpu/suite/ssm/mod.rs"]
mod ssm;
