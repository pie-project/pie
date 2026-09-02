//! Default `config.toml` emitted by `pie config init`.
//!
//! The compiled flavor picks the `[engine]` block, so the generated file
//! works without a follow-up edit. There are TWO candidates — CUDA and, on
//! Apple hardware, Metal — and a binary carrying neither has no block to
//! write and says so, rather than writing a file it knows will not parse.
//!
//! Five live sections, plus a commented `[cluster]` for distributed
//! deployments. The file states only what a single-node deployment decides:
//! no `[controller]` / `[gateway]` sections, no `worker.` prefix on every key,
//! and no options table four levels down.

use anyhow::{Result, bail};
use worker::backend::flavor;

/// Render the default `config.toml`.
///
/// Fallible for one reason: `[engine]` is a required section and every
/// `EngineKind` the schema accepts names an engine that has to be compiled in.
/// A binary built with no `engine-*` feature therefore has nothing true to
/// put there, so it refuses rather than writing a file that fails to parse on
/// the next command, and names the rebuild that fixes it.
pub fn default_config_content() -> Result<String> {
    let flavor = flavor::default_flavor();
    let engine_block: Option<&str> = match flavor {
        #[cfg(feature = "cuda")]
        Some(flavor::Flavor::Cuda) => Some(CUDA_ENGINE_BLOCK),
        // `Flavor::Metal` exists only on an Apple target with the feature on,
        // and this arm carries the same cfg pair so the block and the flavor
        // cannot disagree about which builds have it.
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        Some(flavor::Flavor::Metal) => Some(METAL_ENGINE_BLOCK),
        // `default_flavor` answers `None` when this binary carries no engine,
        // and `worker` may compile a flavor while this crate's matching arm is
        // cfg'd off (workspace feature-unification can desync the two). Both
        // land here, and both mean the same thing to the operator: this binary
        // cannot serve, so a config naming an engine would be a guess.
        #[allow(unreachable_patterns)]
        _ => None,
    };
    let Some(engine_block) = engine_block else {
        bail!(
            "this pie binary carries no engine, so there is no `[engine]` \
             section to write and the config would not parse. Rebuild with \
             `--features cuda` or, on Apple hardware, `--features metal`."
        );
    };
    // ONE MODEL BLOCK for both flavors: the catalog SKU below loads on either,
    // so nothing here depends on which engine the block above named.
    let model_block: &str = DEFAULT_MODEL_BLOCK;
    Ok(format!("{HEADER}{model_block}{engine_block}{TAIL}"))
}

/// The same template rendered against CUDA whatever this binary carries.
///
/// Tests of everything *around* the engine block — the schema accepting the
/// document, the retired sections staying gone, the section budget — need a
/// document, not this binary's engine. Asking for the compiled flavor would
/// make them pass or fail on which features the test run happened to enable,
/// which is how the parseability test came to be red under `cargo test -p
/// pie --lib`: no feature, no flavor, no config.
#[cfg(test)]
pub(crate) fn config_content_with_any_engine() -> String {
    format!("{HEADER}{DEFAULT_MODEL_BLOCK}{CUDA_ENGINE_BLOCK}{TAIL}")
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

"#;

// THE DEFAULT NAMES A MODEL THIS BUILD CAN ACTUALLY SERVE. A checkpoint is
// matched against every import contract in the build and refused by name when
// none fits, so a default the catalog does not ship would make the operator's
// first move debugging a file they did not write.
//
// `Qwen/Qwen3.5-0.8B` is the smallest catalog row (`qwen35-d0.8b-bf16-kv-bf16`,
// 1.6 GiB) and is what every gate from `engine-cuda/tests/serve_smoke` up to
// `tests/gpu/tests/cuda_serve_round_trip` is pinned against. `pie model list`
// prints the SKU beside each snapshot, which is the door for choosing another.
const DEFAULT_MODEL_BLOCK: &str = r#"[model]
name = "default"
model = "Qwen/Qwen3.5-0.8B"
# weight_cache_dir = ""       # empty derives $PIE_HOME/models
# weight_dtype     = "bfloat16"  # what the CHECKPOINT holds. activation_dtype
                                 # is what the engine computes in, so that is
                                 # an engine key and this is a model one.
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
# max_concurrent_processes = 64    # omit: from the engine's max_forward_requests

[sandbox]
# The box an inferlet runs in: its walls, and its size.
allow_fs = false
allow_network = true
network_allowed_hosts = ["*"]  # wasi:sockets only — wasi:http resolves names
                               # in the host stack and bypasses this list
# fs_scratch_dir  = "/tmp/pie"
# max_memory      = "4GiB"
# max_instances   = 1000
# warm_memory     = "0B"
# warm_slots      = 100
# python_snapshot = true
# python_runtime  = true

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

// `test` as well as the feature: `config_content_with_any_engine` renders
// this block regardless of what the test run compiled.
#[cfg(any(feature = "cuda", test))]
const CUDA_ENGINE_BLOCK: &str = r#"
[engine]
# Which keys are valid here depends on `type`: the common ones below, plus
# whatever the named engine accepts. A key it does not know is a parse error
# naming the engine that rejected it.
type = "cuda_native"
device = ["cuda:0"]
tensor_parallel_size = 1
activation_dtype = "bfloat16"
gpu_mem_utilization = 0.90
# kv_page_size    = 32      # omit: the engine derives one
# max_total_pages = 4096    # omit: derived from gpu_mem_utilization
"#;

// `test` as well as the cfg pair, for the same reason the CUDA block carries
// it: `config_content_with_metal_engine` renders this block on a host that
// hosts no Metal device.
//
// EVERY KEY BELOW IS ONE `MetalEngineOptions` DECLARES
// (`worker::config::backend`), with that struct's own default as its value:
// the Metal arm of `EngineConfig::validate` reads the options table raw
// rather than through a `deny_unknown_fields` deserialize, so a stray key
// here would be silently ignored rather than refused -- which makes writing
// only real ones this template's job.
#[cfg(all(feature = "metal", target_vendor = "apple"))]
const METAL_ENGINE_BLOCK: &str = r#"
[engine]
# Which keys are valid here depends on `type`: the common ones below, plus
# whatever the named engine accepts. `tensor_parallel_size` is not among them
# for this engine — the Metal shell serves one device.
type = "metal"
device = ["metal:0"]
activation_dtype = "bfloat16"
gpu_mem_utilization = 0.90  # of the device's recommended working set. A
                            # GPU-touched shared page is WIRED on Apple
                            # silicon, so this ceiling is hard, not a hint.
kv_page_size = 32           # used as given: this engine has no planner to
total_pages  = 1024         # derive a geometry, so these two ARE the pool
# max_forward_tokens   = 10240  # omit for the engine's own defaults
# max_forward_requests = 512    # (max_concurrent_processes derives from this)
# max_model_len        = 8192   # omit to keep the engine's KV-ring ceiling;
                                # setting it only ever shrinks the ring
"#;

// There is no Vulkan or WGPU engine block: no build hosts those engines. A
// `[engine] type = "vulkan"` config still PARSES — `EngineKind` keeps both
// names — and is refused at boot by `flavor::retired_msg`.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_parseable() {
        // The template is the first config almost anyone has, so one that does
        // not parse is the worst possible first impression.
        let content = config_content_with_any_engine();
        worker::Config::parse(&content).expect("generated config must parse");
    }

    #[test]
    fn a_binary_with_an_engine_writes_a_config_that_parses() {
        // The test above pins the template's shape against an engine block it
        // chooses. This one pins the block this binary would actually write,
        // which is the only one an operator ever sees -- and it is a no-op
        // exactly when there is no engine to write about.
        let Ok(content) = default_config_content() else {
            return;
        };
        worker::Config::parse(&content).expect("generated config must parse");
    }

    #[test]
    fn it_names_the_engine_this_binary_actually_has() {
        // The whole promise of picking a block per flavor is that `pie config
        // init` produces a file that runs. A missing match arm does not fail to
        // compile -- it falls into the catch-all, which is how Metal once came
        // to be handed the dummy engine's block: it parsed perfectly and then
        // generated random tokens. The catch-all refuses now, so the same slip
        // costs a refusal rather than nonsense, and the invariant is still
        // checked rather than assumed: the template's `type` is this binary's.
        let expected: Option<&str> = match flavor::default_flavor() {
            #[cfg(feature = "cuda")]
            Some(flavor::Flavor::Cuda) => Some("cuda_native"),
            #[cfg(all(feature = "metal", target_vendor = "apple"))]
            Some(flavor::Flavor::Metal) => Some("metal"),
            // No flavor, no claim to check -- the refusal is pinned by
            // `a_binary_without_an_engine_refuses_instead_of_writing_one`.
            #[allow(unreachable_patterns)]
            _ => None,
        };
        let Some(expected) = expected else { return };
        let content = default_config_content().expect("a flavor means a config");
        assert!(
            content.contains(&format!("type = \"{expected}\"")),
            "template does not select the compiled flavor {expected:?}"
        );
    }

}
