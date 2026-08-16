//! Default `config.toml` emitted by `pie config init`.
//!
//! The highest-priority compiled flavor (cuda → metal → vulkan → wgpu) picks
//! the `[driver]` block, so the generated file works without a follow-up edit.
//! A binary carrying none of them has no block to write and says so, rather
//! than writing a file it knows will not parse.
//!
//! Five live sections, plus a commented `[cluster]` for distributed
//! deployments. What the file no longer has is as much the point as what it
//! does: no `[controller]` / `[gateway]` (empty in every single-node config,
//! present only because three role libs each parsed their own), no `worker.`
//! prefixing all 44 keys, and no `[model.driver.options]` four levels down.

use anyhow::{Result, bail};
use worker::driver_ffi;

/// Render the default `config.toml`.
///
/// Fallible for one reason: `[driver]` is a required section and every
/// `DriverKind` the schema accepts names a driver that has to be compiled in.
/// A binary built with no `driver-*` feature therefore has nothing true to
/// put there. It used to write `type = "dummy"`, which was honest while an
/// always-present interpreter flavor existed; that flavor was deleted for
/// being a paper-over (see `worker::driver_ffi::Flavor`) and this template
/// kept writing its name, so `pie config init` produced a config that failed
/// to parse on the next command. Refusing says the same thing one step
/// earlier and names the rebuild that fixes it.
pub fn default_config_content() -> Result<String> {
    let flavor = driver_ffi::default_flavor();
    let driver_block: Option<&str> = match flavor {
        #[cfg(feature = "_driver-cuda")]
        Some(driver_ffi::Flavor::Cuda) => Some(CUDA_DRIVER_BLOCK),
        #[cfg(feature = "driver-metal")]
        Some(driver_ffi::Flavor::Metal) => Some(METAL_DRIVER_BLOCK),
        #[cfg(feature = "driver-vulkan")]
        Some(driver_ffi::Flavor::Vulkan) => Some(VULKAN_DRIVER_BLOCK),
        #[cfg(feature = "driver-wgpu")]
        Some(driver_ffi::Flavor::Wgpu) => Some(WGPU_DRIVER_BLOCK),
        // `default_flavor` answers `None` when this binary carries no driver,
        // and `worker` may compile `Flavor::Cuda`/`Metal` while this crate's
        // matching `driver-*` arm is cfg'd off (workspace feature-unification
        // can desync the two). Both land here, and both mean the same thing to
        // the operator: this binary cannot serve, so a config naming a driver
        // would be a guess.
        _ => None,
    };
    let Some(driver_block) = driver_block else {
        bail!(
            "this pie binary carries no driver, so there is no `[driver]` \
             section to write and the config would not parse. Rebuild with \
             --features set to one of driver-cuda-13, driver-cuda-12, \
             driver-metal, driver-vulkan or driver-wgpu."
        );
    };
    let model_block: &str = match flavor {
        // Metal's llama path binds `.weight`/`.scales`/`.biases` for every
        // matvec and has no unquantized kernel, so the stock bf16 default
        // imports fine and then cannot bind at load. A default has to run.
        #[cfg(feature = "driver-metal")]
        Some(driver_ffi::Flavor::Metal) => METAL_MODEL_BLOCK,
        // Both portable shells load through `Binding::MLX_IN_PLACE`, so both
        // want the quantized default for one reason. See `MLX_MODEL_BLOCK`.
        #[cfg(feature = "driver-vulkan")]
        Some(driver_ffi::Flavor::Vulkan) => MLX_MODEL_BLOCK,
        #[cfg(feature = "driver-wgpu")]
        Some(driver_ffi::Flavor::Wgpu) => MLX_MODEL_BLOCK,
        _ => DEFAULT_MODEL_BLOCK,
    };
    Ok(format!("{HEADER}{model_block}{driver_block}{TAIL}"))
}

/// The same template rendered against CUDA whatever this binary carries.
///
/// Tests of everything *around* the driver block — the schema accepting the
/// document, the retired sections staying gone, the section budget — need a
/// document, not this binary's driver. Asking for the compiled flavor would
/// make them pass or fail on which features the test run happened to enable,
/// which is how the parseability test came to be red under `cargo test -p
/// pie --lib`: no feature, no flavor, no config.
#[cfg(test)]
pub(crate) fn config_content_with_any_driver() -> String {
    format!("{HEADER}{DEFAULT_MODEL_BLOCK}{CUDA_DRIVER_BLOCK}{TAIL}")
}

const HEADER: &str = r#"# Pie configuration, written by `pie config init`. Edit freely.
#
# Every key has a default — delete a line to get it back. `pie config list`
# prints all of them with their current values and what they mean.

[server]
host = "127.0.0.1"          # loopback. Exposing the port is an edit here.
port = 8080
registry = "https://registry.pie-project.org/"
verbose = false
telemetry = false
# otlp_endpoint   = "http://localhost:4317"
# service_name    = "pie"
# worker_threads  = 16        # derived from visible CPUs, capped at 64
# max_upload      = "256MiB"
# python_snapshot = true

"#;

const DEFAULT_MODEL_BLOCK: &str = r#"[model]
name = "default"
model = "Qwen/Qwen3-0.6B"
# weight_cache_dir = ""       # empty derives $PIE_HOME/models
"#;

/// Metal's llama path is 4-bit-only, so the default names a quantized repo.
#[cfg(feature = "driver-metal")]
const METAL_MODEL_BLOCK: &str = r#"[model]
name = "default"
# Metal's llama path is 4-bit-only: every matvec binds `.weight`/`.scales`/
# `.biases`, and there is no unquantized kernel to fall back to. So the default
# is an MLX-quantized checkpoint -- a raw bf16 repo (e.g. `Qwen/Qwen3-0.6B`)
# imports fine and then fails to bind at load.
model = "mlx-community/Qwen3-0.6B-4bit"
# weight_cache_dir = ""       # empty derives $PIE_HOME/models
"#;

/// The default both portable shells need, and they need it for ONE reason
/// rather than two: each seam compiles its load plan through
/// `model::boot::Binding::MLX_IN_PLACE`, so every projection binds
/// `.weight`/`.scales`/`.biases` and a raw bf16 repo is refused at load with
/// the remedy named in the refusal.
///
/// Shared between them and NOT with `METAL_MODEL_BLOCK` above, which names the
/// same repo for a different reason -- that driver has no unquantized matvec
/// at all. Two reasons, two blocks; one reason, one block.
#[cfg(any(feature = "driver-vulkan", feature = "driver-wgpu"))]
const MLX_MODEL_BLOCK: &str = r#"[model]
name = "default"
# This driver loads through MLX in-place binding: every projection wants
# `.weight`/`.scales`/`.biases`, so the default is an MLX-quantized checkpoint.
# A raw bf16 repo (e.g. `Qwen/Qwen3-0.6B`) imports fine and is then refused at
# load, saying it needs quantized weights.
model = "mlx-community/Qwen3-0.6B-4bit"
# weight_cache_dir = ""       # empty derives $PIE_HOME/models
"#;

const TAIL: &str = r#"
[runtime]
# Batching and timeouts. Every default here is measured; `pie config list`
# carries the reasoning.
request_timeout = "120s"
# submit_deadline          = "50ms"
# silence_timeout          = "30s"
# frame_size               = 2     # guest contract: moving it means
# frame_submit_depth       = 3     # re-measuring frame_submit_depth
# frame_dispatch_depth     = 2
# max_concurrent_processes = 64    # omit: from the driver's max_forward_requests

[sandbox]
# The box an inferlet runs in: its walls, and its size.
allow_fs = false
allow_network = true
network_allowed_hosts = ["*"]  # wasi:sockets only — wasi:http resolves names
                               # in the host stack and bypasses this list
# fs_scratch_dir = "/tmp/pie"
# max_memory     = "4GiB"
# max_instances  = 1000
# warm_memory    = "0B"
# warm_slots     = 100

# [cluster]
# Distributed serving only. A single-node config omits this section entirely.
# controller = "tcp://10.0.0.1:9102"
# role = "decode"
# gateways = ["tcp://10.0.0.2:8081"]
# max_clients = 4
# offload = true
# transfer = "auto"
# prefill_min_suffix_tokens = 0
# max_outstanding_per_partner = 4
"#;

// `test` as well as the feature: `config_content_with_any_driver` renders
// this block regardless of what the test run compiled.
#[cfg(any(feature = "_driver-cuda", test))]
const CUDA_DRIVER_BLOCK: &str = r#"
[driver]
# Which keys are valid here depends on `type`: the common ones below, plus
# whatever the named driver accepts. A key it does not know is a parse error
# naming the driver that rejected it.
type = "cuda_native"
device = ["cuda:0"]
tensor_parallel_size = 1
activation_dtype = "bfloat16"
gpu_mem_utilization = 0.90
# kv_cache_dtype  = "auto"
# kv_page_size    = 32      # omit: the memory planner derives one
# max_total_pages = 4096    # omit: derived from gpu_mem_utilization
# random_seed     = 42
"#;

#[cfg(feature = "driver-metal")]
const METAL_DRIVER_BLOCK: &str = r#"
[driver]
# Which keys are valid here depends on `type`: the common ones below, plus
# whatever the named driver accepts. A key it does not know is a parse error
# naming the driver that rejected it.
type = "metal"
device = ["metal:0"]
activation_dtype = "bfloat16"
kv_page_size = 32
total_pages = 1024
# max_forward_tokens      = 10240
# max_forward_requests    = 512
# cpu_pages               = 0      # 0 disables KV swapping to host memory
# kv_cache_dtype          = "auto"
# stream_routed_experts   = false  # page MoE experts from the checkpoint
"#;

#[cfg(feature = "driver-vulkan")]
const VULKAN_DRIVER_BLOCK: &str = r#"
[driver]
# Which keys are valid here depends on `type`: the common ones below, plus
# whatever the named driver accepts. A key it does not know is a parse error
# naming the driver that rejected it.
type = "vulkan"
# Not a selector this driver reads: it opens the first Vulkan device the
# loader reports. Written because `device` is required of every driver.
device = ["vulkan:0"]
activation_dtype = "bfloat16"
kv_pages = 1024
"#;

#[cfg(feature = "driver-wgpu")]
const WGPU_DRIVER_BLOCK: &str = r#"
[driver]
# Which keys are valid here depends on `type`: the common ones below, plus
# whatever the named driver accepts. A key it does not know is a parse error
# naming the driver that rejected it.
type = "wgpu"
# Not a selector this driver reads either, and for a different reason than the
# Vulkan block's: `wgpu` asks the platform for an adapter itself. Written
# because `device` is required of every driver.
device = ["gpu:0"]
activation_dtype = "bfloat16"
# The one knob this backend reads. No `kv_page_size` (16, fixed by the
# kernels) and no `kv_cache_dtype` (bf16, the only one it stores). The shaders
# are inside the binary, as the Vulkan block's SPIR-V now is too.
kv_pages = 1024
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_parseable() {
        // The template is the first config almost anyone has, so one that does
        // not parse is the worst possible first impression.
        let content = config_content_with_any_driver();
        worker::Config::parse(&content).expect("generated config must parse");
    }

    #[test]
    fn a_binary_with_a_driver_writes_a_config_that_parses() {
        // The test above pins the template's shape against a driver block it
        // chooses. This one pins the block this binary would actually write,
        // which is the only one an operator ever sees -- and it is a no-op
        // exactly when there is no driver to write about.
        let Ok(content) = default_config_content() else {
            return;
        };
        worker::Config::parse(&content).expect("generated config must parse");
    }

    #[test]
    fn a_binary_without_a_driver_refuses_instead_of_writing_one() {
        // The negative control for the arm above, and the bug it fixes: a
        // driverless build used to write `type = "dummy"` -- a name the schema
        // stopped accepting when the dummy driver was deleted -- so `pie
        // config init` succeeded and every command after it failed to parse
        // the file it had just written.
        if driver_ffi::default_flavor().is_some() {
            return;
        }
        let err = default_config_content().expect_err("driverless must refuse");
        let msg = err.to_string();
        assert!(
            msg.contains("carries no driver"),
            "unhelpful refusal: {msg}"
        );
        assert!(msg.contains("driver-cuda-13"), "no fix named: {msg}");
    }

    #[test]
    fn it_names_the_driver_this_binary_actually_has() {
        // The whole promise of picking a block per flavor is that `pie config
        // init` produces a file that runs. A missing match arm does not fail to
        // compile -- it falls into the catch-all, which is how Metal once came
        // to be handed the dummy driver's block: it parsed perfectly and then
        // generated random tokens. The catch-all refuses now, so the same slip
        // costs a refusal rather than nonsense, and the invariant is still
        // checked rather than assumed: the template's `type` is this binary's.
        let expected: Option<&str> = match driver_ffi::default_flavor() {
            #[cfg(feature = "_driver-cuda")]
            Some(driver_ffi::Flavor::Cuda) => Some("cuda_native"),
            #[cfg(feature = "driver-metal")]
            Some(driver_ffi::Flavor::Metal) => Some("metal"),
            #[cfg(feature = "driver-vulkan")]
            Some(driver_ffi::Flavor::Vulkan) => Some("vulkan"),
            #[cfg(feature = "driver-wgpu")]
            Some(driver_ffi::Flavor::Wgpu) => Some("wgpu"),
            // No flavor, no claim to check -- the refusal is pinned by
            // `a_binary_without_a_driver_refuses_instead_of_writing_one`.
            _ => None,
        };
        let Some(expected) = expected else { return };
        let content = default_config_content().expect("a flavor means a config");
        assert!(
            content.contains(&format!("type = \"{expected}\"")),
            "template does not select the compiled flavor {expected:?}"
        );
    }

    #[test]
    fn the_retired_sections_are_gone_from_it() {
        let content = config_content_with_any_driver();
        for retired in ["[controller]", "[gateway]", "[worker", "[model.driver"] {
            assert!(
                !content.contains(retired),
                "template still writes {retired}"
            );
        }
    }

    #[test]
    fn it_stays_within_the_section_budget() {
        // Five live, plus a commented `[cluster]`. Section count is what this
        // format was redesigned to bring down; a test is cheaper than noticing
        // later that it crept back.
        let live = content_sections(&config_content_with_any_driver());
        assert!(live <= 5, "template has {live} live sections");
    }

    fn content_sections(content: &str) -> usize {
        content
            .lines()
            .filter(|l| l.starts_with('[') && !l.starts_with("[["))
            .count()
    }
}
