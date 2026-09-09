use anyhow::{Result, bail};
use worker::backend::flavor;

pub fn default_config_content() -> Result<String> {
    let flavor = flavor::default_flavor();
    let engine_block: Option<&str> = match flavor {
        #[cfg(feature = "cuda")]
        Some(flavor::Flavor::Cuda) => Some(CUDA_ENGINE_BLOCK),
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        Some(flavor::Flavor::Metal) => Some(METAL_ENGINE_BLOCK),
        #[cfg(feature = "vulkan")]
        Some(flavor::Flavor::Vulkan) => Some(VULKAN_ENGINE_BLOCK),
        #[cfg(feature = "wgpu")]
        Some(flavor::Flavor::Wgpu) => Some(WGPU_ENGINE_BLOCK),
        #[allow(unreachable_patterns)]
        _ => None,
    };
    let Some(engine_block) = engine_block else {
        bail!(
            "this pie binary carries no engine, so there is no `[engine]` \
             section to write and the config would not parse. Rebuild with \
             `--features cuda`, `--features vulkan`, `--features wgpu`, or, \
             on Apple hardware, `--features metal`."
        );
    };
    let model_block: &str = DEFAULT_MODEL_BLOCK;
    Ok(format!("{HEADER}{model_block}{engine_block}{TAIL}"))
}

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
# frame_size               = 2     # guest contract: the submit depth is
# frame_dispatch_depth     = 2     # derived from it, not a key of its own
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
# max_state_slots = 256     # recurrent-state seats (hybrid models); omit for 256
# max_model_len   = 4096    # the most tokens one sequence may hold; omit for
#                           # 4096. State it at what you serve: a decode body
#                           # for n lanes is armed only when the pool holds
#                           # n x max_model_len tokens
"#;

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
# max_state_slots      = 256    # recurrent-state seats (hybrid models)
"#;

#[cfg(any(feature = "vulkan", test))]
const VULKAN_ENGINE_BLOCK: &str = r#"
[engine]
# Which keys are valid here depends on `type`: the common ones below, plus
# whatever the named engine accepts. `tensor_parallel_size` is not among them
# for this engine — the Vulkan shell serves one device.
type = "vulkan"
device = ["vulkan:0"]
activation_dtype = "bfloat16"
device_index = 0            # vkEnumeratePhysicalDevices order; 0 is the first
gpu_mem_utilization = 0.90  # of the device-local heap: weights, kv pool, scratch
# max_total_pages      = 4096   # omit: derived from gpu_mem_utilization
# max_forward_tokens   = 10240  # omit for the engine's own defaults
# max_forward_requests = 512    # (max_concurrent_processes derives from this)
# max_state_slots      = 256    # recurrent-state seats (hybrid models)
# validation           = false  # VK_LAYER_KHRONOS_validation; diagnostic only
# pipeline_cache       = "/path/to/pipeline.bin"  # omit for in-memory only
"#;

#[cfg(any(feature = "wgpu", test))]
const WGPU_ENGINE_BLOCK: &str = r#"
[engine]
# Which keys are valid here depends on `type`: the common ones below, plus
# whatever the named engine accepts. `tensor_parallel_size` is not among them
# for this engine — the wgpu shell serves one adapter.
type = "wgpu"
device = ["wgpu:0"]
activation_dtype = "bfloat16"
adapter_index = 0           # enumeration order among the reachable adapters
# backends = "vulkan"           # omit for every backend this build carries
gpu_mem_utilization = 0.90  # of the device-local heap: weights, kv pool, scratch
# power_preference     = "high-performance"  # or "low-power", "none"
# max_total_pages      = 4096   # omit: derived from gpu_mem_utilization
# max_forward_tokens   = 10240  # omit for the engine's own defaults
# max_forward_requests = 512    # (max_concurrent_processes derives from this)
# max_state_slots      = 256    # recurrent-state seats (hybrid models)
# device_memory        = "16GiB"  # omit unless the backend reports none
# pipeline_cache       = "/path/to/pipeline.bin"  # omit for in-process only
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_every_case() {
        default_config_is_parseable();
        the_vulkan_block_states_only_keys_the_engine_declares();
        the_wgpu_block_states_only_keys_the_engine_declares();
        a_binary_with_an_engine_writes_a_config_that_parses();
        it_names_the_engine_this_binary_actually_has();
    }

    fn default_config_is_parseable() {
        let content = config_content_with_any_engine();
        worker::Config::parse(&content).expect("generated config must parse");
    }

    fn the_vulkan_block_states_only_keys_the_engine_declares() {
        let content = format!("{HEADER}{DEFAULT_MODEL_BLOCK}{VULKAN_ENGINE_BLOCK}{TAIL}");
        worker::Config::parse(&content).expect("the vulkan template must parse");
    }

    fn the_wgpu_block_states_only_keys_the_engine_declares() {
        let content = format!("{HEADER}{DEFAULT_MODEL_BLOCK}{WGPU_ENGINE_BLOCK}{TAIL}");
        worker::Config::parse(&content).expect("the wgpu template must parse");
    }

    fn a_binary_with_an_engine_writes_a_config_that_parses() {
        let Ok(content) = default_config_content() else {
            return;
        };
        worker::Config::parse(&content).expect("generated config must parse");
    }

    fn it_names_the_engine_this_binary_actually_has() {
        let expected: Option<&str> = match flavor::default_flavor() {
            #[cfg(feature = "cuda")]
            Some(flavor::Flavor::Cuda) => Some("cuda_native"),
            #[cfg(all(feature = "metal", target_vendor = "apple"))]
            Some(flavor::Flavor::Metal) => Some("metal"),
            #[cfg(feature = "vulkan")]
            Some(flavor::Flavor::Vulkan) => Some("vulkan"),
            #[cfg(feature = "wgpu")]
            Some(flavor::Flavor::Wgpu) => Some("wgpu"),
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
