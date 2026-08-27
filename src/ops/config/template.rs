//! Default `config.toml` emitted by `pie config init`.
//!
//! The compiled flavor picks the `[driver]` block, so the generated file
//! works without a follow-up edit. There is ONE candidate — it was a
//! cuda → metal → vulkan → wgpu priority order while the shader shells were
//! in the workspace — and a binary carrying none has no block to write and
//! says so, rather than writing a file it knows will not parse.
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
        // `default_flavor` answers `None` when this binary carries no driver,
        // and `worker` may compile `Flavor::Cuda` while this crate's matching
        // `driver-cuda-*` arm is cfg'd off (workspace feature-unification can
        // desync the two). Both land here, and both mean the same thing to the
        // operator: this binary cannot serve, so a config naming a driver
        // would be a guess.
        _ => None,
    };
    let Some(driver_block) = driver_block else {
        bail!(
            "this pie binary carries no driver, so there is no `[driver]` \
             section to write and the config would not parse. Rebuild with \
             --features set to driver-cuda-13 or driver-cuda-12."
        );
    };
    // ONE FLAVOR, so one block. It was a `match` on the flavor while Metal
    // wanted a 4-bit default its llama path could bind; that driver is out of
    // the workspace until P5.
    let model_block: &str = DEFAULT_MODEL_BLOCK;
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

// THE DEFAULT NAMES A MODEL THIS BUILD CAN ACTUALLY SERVE.
//
// It was `Qwen/Qwen3-0.6B` and nothing here ships a trace for it: the catalog
// (`model::catalog`) is qwen35 / gemma4 / gptoss / glm5 / kimik3 / dsv4, and
// the palo rewrite made the SKU the load's own identity — a checkpoint is
// matched against every import contract in the build and refused by name when
// none fits. So `pie config init && pie serve` answered
//
//     "/root/.pie/models/Qwen--Qwen3-0.6B.zt" matches no SKU this build ships
//
// with ten candidate refusals under it, out of the box, on a machine where
// everything worked. A generated default that cannot boot is worse than no
// default: the operator's first move is to debug a file they did not write.
//
// `Qwen/Qwen3.5-0.8B` is the smallest catalog row (`qwen35-d0.8b-bf16-kv-bf16`,
// 1.6 GiB) and is what every gate from `driver-cuda/tests/serve_smoke` up to
// `tests/gpu/tests/cuda_serve_round_trip` is pinned against. `pie model list`
// prints the SKU beside each snapshot, which is the door for choosing another.
const DEFAULT_MODEL_BLOCK: &str = r#"[model]
name = "default"
model = "Qwen/Qwen3.5-0.8B"
# weight_cache_dir = ""       # empty derives $PIE_HOME/models
"#;

// `METAL_MODEL_BLOCK` STOOD HERE, beside `MLX_MODEL_BLOCK` below: Metal's
// llama path is 4-bit-only, so its default named an MLX-quantized repo. Its
// driver left the workspace at R3 and returns at P5.

// `MLX_MODEL_BLOCK` STOOD HERE — the quantized default both portable shells
// wanted, because each loaded through `model_legacy::boot::Binding::MLX_IN_PLACE`
// and every projection bound `.weight`/`.scales`/`.biases`. Its two readers
// left the workspace at R3 and return at P5, when their baker executors say
// for themselves what they bind.

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

// `METAL_DRIVER_BLOCK`, `VULKAN_DRIVER_BLOCK` and `WGPU_DRIVER_BLOCK` STOOD
// HERE, and went with their drivers at R3. A `[driver] type = "vulkan"`
// config still PARSES — `DriverKind` keeps all three names — and is refused
// at boot with what happened, which is `driver_ffi::retired_msg`.

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
