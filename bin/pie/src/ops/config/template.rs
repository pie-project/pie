//! Default `config.toml` emitted by `pie config init`.
//!
//! The highest-priority compiled flavor (cuda → metal → dummy) picks the
//! `[driver]` block, so the generated file works without a follow-up edit.
//!
//! Five live sections, plus a commented `[cluster]` for distributed
//! deployments. What the file no longer has is as much the point as what it
//! does: no `[controller]` / `[gateway]` (empty in every single-node config,
//! present only because three role libs each parsed their own), no `worker.`
//! prefixing all 44 keys, and no `[model.driver.options]` four levels down.

use pie_worker::driver_ffi::{self, Flavor};

/// Render the default `config.toml`.
pub fn default_config_content() -> String {
    let flavor = driver_ffi::default_flavor();
    let driver_block = match flavor {
        #[cfg(feature = "driver-cuda")]
        Some(Flavor::Cuda) => CUDA_DRIVER_BLOCK,
        #[cfg(feature = "driver-metal")]
        Some(Flavor::Metal) => METAL_DRIVER_BLOCK,
        Some(Flavor::Dummy) => DUMMY_DRIVER_BLOCK,
        // Fallback for exhaustiveness: `default_flavor` always returns `Some`
        // (dummy is linked unconditionally), and `pie-worker` may compile
        // `Flavor::Cuda`/`Metal` while this crate's matching `driver-*` arm is
        // cfg'd off (workspace feature-unification can desync the two) — those
        // land here → the dummy block.
        _ => DUMMY_DRIVER_BLOCK,
    };
    let model_block = match flavor {
        // Metal's llama path binds `.weight`/`.scales`/`.biases` for every
        // matvec and has no unquantized kernel, so the stock bf16 default
        // imports fine and then cannot bind at load. A default has to run.
        #[cfg(feature = "driver-metal")]
        Some(Flavor::Metal) => METAL_MODEL_BLOCK,
        _ => DEFAULT_MODEL_BLOCK,
    };
    format!("{HEADER}{model_block}{driver_block}{TAIL}")
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

#[cfg(feature = "driver-cuda")]
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

const DUMMY_DRIVER_BLOCK: &str = r#"
[driver]
# Which keys are valid here depends on `type`: the common ones below, plus
# whatever the named driver accepts. A key it does not know is a parse error
# naming the driver that rejected it.
type = "dummy"
device = ["cpu"]
activation_dtype = "bfloat16"
vocab_size = 151936
arch_name = "qwen3"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_parseable() {
        // The template is the first config almost anyone has, so one that does
        // not parse is the worst possible first impression.
        let content = default_config_content();
        pie_worker::Config::parse(&content).expect("generated config must parse");
    }

    #[test]
    fn it_names_the_driver_this_binary_actually_has() {
        // The whole promise of picking a block per flavor is that `pie config
        // init` produces a file that runs. A missing match arm does not fail to
        // compile -- it falls into the catch-all and writes the dummy driver,
        // which parses perfectly and then generates random tokens. That is
        // exactly what happened to Metal, so the invariant is checked rather
        // than assumed: the template's `type` is the flavor this binary has.
        let expected = match driver_ffi::default_flavor() {
            #[cfg(feature = "driver-cuda")]
            Some(Flavor::Cuda) => "cuda_native",
            #[cfg(feature = "driver-metal")]
            Some(Flavor::Metal) => "metal",
            _ => "dummy",
        };
        let content = default_config_content();
        assert!(
            content.contains(&format!("type = \"{expected}\"")),
            "template does not select the compiled flavor {expected:?}"
        );
    }

    #[test]
    fn the_retired_sections_are_gone_from_it() {
        let content = default_config_content();
        for retired in ["[controller]", "[gateway]", "[worker", "[model.driver"] {
            assert!(!content.contains(retired), "template still writes {retired}");
        }
    }

    #[test]
    fn it_stays_within_the_section_budget() {
        // Five live, plus a commented `[cluster]`. Section count is what this
        // format was redesigned to bring down; a test is cheaper than noticing
        // later that it crept back.
        let live = content_sections(&default_config_content());
        assert!(live <= 5, "template has {live} live sections");
    }

    fn content_sections(content: &str) -> usize {
        content
            .lines()
            .filter(|l| l.starts_with('[') && !l.starts_with("[["))
            .count()
    }
}
