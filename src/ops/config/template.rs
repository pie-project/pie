use anyhow::{Result, bail};
use worker::backend::flavor;

pub fn default_config_content() -> Result<String> {
    let flavor = flavor::default_flavor();
    let engine_block: Option<&str> = match flavor {
        #[cfg(feature = "engine-cuda")]
        Some(flavor::Flavor::Cuda) => Some(CUDA_ENGINE_BLOCK),
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        Some(flavor::Flavor::Metal) => Some(METAL_ENGINE_BLOCK),
        #[cfg(feature = "vulkan")]
        Some(flavor::Flavor::Vulkan) => Some(VULKAN_ENGINE_BLOCK),
        #[cfg(feature = "wgpu")]
        Some(flavor::Flavor::Wgpu) => Some(WGPU_ENGINE_BLOCK),
        #[cfg(feature = "xla")]
        Some(flavor::Flavor::Xla) => Some(XLA_ENGINE_BLOCK),
        #[cfg(feature = "cerebras")]
        Some(flavor::Flavor::Cerebras) => Some(CEREBRAS_ENGINE_BLOCK),
        #[allow(unreachable_patterns)]
        _ => None,
    };
    let Some(engine_block) = engine_block else {
        bail!(
            "this pie binary carries no engine, so there is no `[engine]` \
             section to write and the config would not parse. Rebuild with \
             `--features cuda`, `--features vulkan`, `--features wgpu`, `--features xla`, \
             `--features cerebras`, or, on Apple hardware, `--features metal`."
        );
    };
    let model_block: &str = DEFAULT_MODEL_BLOCK;
    let frames: &str = match flavor {
        #[cfg(feature = "cerebras")]
        Some(flavor::Flavor::Cerebras) => CEREBRAS_FRAMES,
        #[allow(unreachable_patterns)]
        _ => DEFAULT_FRAMES,
    };
    Ok(format!(
        "{HEADER}{model_block}{engine_block}{MEMORY}{RUNTIME}{frames}{TAIL}"
    ))
}

#[cfg(test)]
pub(crate) fn config_content_with_any_engine() -> String {
    format!(
        "{HEADER}{DEFAULT_MODEL_BLOCK}{CUDA_ENGINE_BLOCK}{MEMORY}{RUNTIME}{DEFAULT_FRAMES}{TAIL}"
    )
}

const HEADER: &str = r#"# Pie configuration, written by `pie config init`. Edit freely.
#
# Every key has a default — delete a line to get it back. `pie config list`
# prints all of them with their current values and what they mean.

[server]
listen = "127.0.0.1:8080"   # loopback. Exposing the port is an edit here.
verbose = false
# worker_threads  = 16        # derived from visible CPUs, capped at 64
# max_upload      = "256MiB"
# allow_upload    = true      # false: clients run only the built-ins and `pie inferlet install`s

# [telemetry]
# endpoint     = "http://localhost:4317"  # an OTLP collector; omit to export none
# service_name = "pie"

"#;

const DEFAULT_MODEL_BLOCK: &str = r#"[model]
name = "default"
source = "Qwen/Qwen3.5-0.8B"
# weight_cache_dir = ""       # empty derives $PIE_HOME/cache/weights
# weight_dtype     = "bfloat16"  # what the CHECKPOINT holds. activation_dtype
                                 # is what the engine computes in, so that is
                                 # an engine key and this is a model one.
"#;

const MEMORY: &str = r#"
[engine.memory]
# Nothing to set on a dedicated machine: the engine measures the device and
# uses all of it, and the split between weights, kv and fires follows demand.
# Each key is a cap for sharing the machine.
# budget_device         = "20GiB"   # per device; omit for the whole device
# budget_host           = "32GiB"   # suspended kv goes to the host, then the disk
# budget_disk           = "200GiB"  # the tier is always on
# disk_path             = "/nvme/pie"  # omit for $PIE_HOME/cache/kv/<digest>
# disk_persistent       = false     # keep the disk tier as cache for the next run
# governor_backoff      = 1.0       # share kept each congested stretch; 1.0 never yields
# governor_release_idle = false     # give back memory demand stopped asking for
"#;

const RUNTIME: &str = r#"
[runtime]
# Batching and timeouts. Every default here is measured; `pie config list`
# carries the reasoning.
# submit_deadline          = "50ms"
# silence_timeout          = "30s"
"#;

const DEFAULT_FRAMES: &str = r#"# frame_size               = 2     # guest contract: the submit depth is
# frame_dispatch_depth     = 2     # derived from it, not a key of its own
"#;

// The wafer's token loops answer each lane as far as its guest has posted:
// at the runtime's most (4 steps a frame, 15 frames) every lane keeps posted
// past a loop's end, so each continuation carries every lane (the 8B pipeline
// ring's 12 streams, 12 lanes a launch; at 2 and 2 or 2 and 15, fewer).
#[cfg(any(feature = "cerebras", test))]
const CEREBRAS_FRAMES: &str = r#"frame_size               = 4     # the most: each lane posts this far
frame_dispatch_depth     = 15    # ahead, so every lane rides each wafer loop
"#;

const TAIL: &str = r#"# max_concurrent_processes = 64    # omit: from the engine's fire lanes
# worker_policy            = "noop"  # a PLEX worker policy: built-in name or .wasm path
# snapshot_budget          = "64GiB"  # what handle snapshots may take on disk

[sandbox]
# The box an inferlet runs in: its walls, and its size.
allow_fs = false
allow_network = true
network_allowed_hosts = ["*"]  # wasi:sockets only — wasi:http resolves names
                               # in the host stack and bypasses this list
# scratch_dir     = "/tmp/pie"
# max_memory      = "4GiB"
# max_instances   = 1000
# warm_memory     = "0B"
# warm_slots      = 100

# [cluster]
# cluster_policies = []        # cluster policies: built-in names or .wasm paths,
#                              # no two granted the same write (`publish`
#                              # excepted: each writes under its own name).
#                              # SIGHUP reloads this and `[runtime] worker_policy`.
#                              # Single-node only: a distributed controller
#                              # states its own.
# Distributed serving only:
# controller = "tcp://10.0.0.1:9102"
# role = "decode"                # or "prefill": takes peers' passes, no client turns
# gateways = ["tcp://10.0.0.2:8081"]

# [offload]
# Other workers' part in this one's passes, and its in theirs. Omit to serve
# and send none.
# listen = "10.0.0.1:7100"     # where this worker serves peers' passes; their
#                              # pages' bytes ride one port above
# peers  = ["10.0.0.2:7100"]   # workers to prefill this worker's passes on
# pools  = ["10.0.0.3:7201"]   # pool workers (`pie-pool`) to keep its runs on,
#                              # by their bulk address
# transfer = "auto"            # the bulk link; "nixl" reads peers' kv in place
# prefill_min_suffix_tokens = 0  # shortest uncached suffix worth a peer's
#                                # prefill; 0: 2048 tokens
"#;

#[cfg(any(feature = "engine-cuda", test))]
const CUDA_ENGINE_BLOCK: &str = r#"
[engine]
# Which keys are valid here depends on `type`: the common ones below, plus
# whatever the named engine accepts. A key it does not know is a parse error
# naming the engine that rejected it.
type = "cuda"
device = ["cuda:0"]
tensor_parallel_size = 1
activation_dtype = "bfloat16"
# max_context     = 32768   # the most tokens one sequence may hold; omit for
#                           # the context the model's config.json declares,
#                           # shortened where one sequence at it would take
#                           # more than half the pool (4096 if it declares
#                           # none)
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
# max_context          = 8192   # omit to keep the engine's KV-ring ceiling;
                                # setting it only ever shrinks the ring
"#;

#[cfg(any(feature = "vulkan", test))]
const VULKAN_ENGINE_BLOCK: &str = r#"
[engine]
# Which keys are valid here depends on `type`: the common ones below, plus
# whatever the named engine accepts. `tensor_parallel_size` is not among them
# for this engine — the Vulkan shell serves one device.
type = "vulkan"
device = ["vulkan:0"]       # vkEnumeratePhysicalDevices order; 0 is the first
activation_dtype = "bfloat16"
# validation           = false  # VK_LAYER_KHRONOS_validation; diagnostic only
# pipeline_cache       = "/path/to/pipeline.bin"  # omit for $PIE_HOME/cache/vulkan-pipelines.bin
"#;

#[cfg(any(feature = "wgpu", test))]
const WGPU_ENGINE_BLOCK: &str = r#"
[engine]
# Which keys are valid here depends on `type`: the common ones below, plus
# whatever the named engine accepts. `tensor_parallel_size` is not among them
# for this engine — the wgpu shell serves one adapter.
type = "wgpu"
device = ["wgpu:0"]         # enumeration order among the reachable adapters
activation_dtype = "bfloat16"
# backends = "vulkan"           # omit for every backend this build carries
# power_preference     = "high-performance"  # or "low-power", "none"
# pipeline_cache       = "/path/to/pipeline.bin"  # omit for in-process only
"#;

#[cfg(any(feature = "xla", test))]
const XLA_ENGINE_BLOCK: &str = r#"
[engine]
# Which keys are valid here depends on `type`: the common ones below, plus
# whatever the named engine accepts. The xla shell drives one PJRT device.
type = "xla"
device = ["tpu:0"]          # which of the plugin's addressable devices
activation_dtype = "bfloat16"
# plugin               = "/path/to/libtpu.so"  # omit: PIE_XLA_PLUGIN, TPU_LIBRARY_PATH, libtpu.so
"#;

#[cfg(any(feature = "cerebras", test))]
const CEREBRAS_ENGINE_BLOCK: &str = r#"
[engine]
# Which keys are valid here depends on `type`: the common ones below, plus
# whatever the named engine accepts. The cerebras shell drives one wafer:
# the fabric simulator, or a CS system named by cmaddr.
type = "cerebras"
device = ["cerebras:0"]
activation_dtype = "bfloat16"
target = "wse3"             # the wafer generation: "wse2" or "wse3"
num_threads = 16            # host threads for the simulator and staging
# cmaddr               = "10.0.0.1:9000"  # omit to run the fabric simulator
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_every_case() {
        default_config_is_parseable();
        every_memory_line_uncommented_parses();
        the_vulkan_block_states_only_keys_the_engine_declares();
        the_wgpu_block_states_only_keys_the_engine_declares();
        the_xla_block_states_only_keys_the_engine_declares();
        the_cerebras_block_states_only_keys_the_engine_declares();
        a_binary_with_an_engine_writes_a_config_that_parses();
        it_names_the_engine_this_binary_actually_has();
        no_block_names_a_removed_key();
    }

    /// Not even commented out: an operator who uncomments a line the
    /// template offers gets a file that parses.
    fn no_block_names_a_removed_key() {
        #[allow(unused_mut)]
        let mut blocks = vec![
            HEADER,
            DEFAULT_MODEL_BLOCK,
            CUDA_ENGINE_BLOCK,
            VULKAN_ENGINE_BLOCK,
            WGPU_ENGINE_BLOCK,
            XLA_ENGINE_BLOCK,
            CEREBRAS_ENGINE_BLOCK,
            MEMORY,
            RUNTIME,
            DEFAULT_FRAMES,
            CEREBRAS_FRAMES,
            TAIL,
        ];
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        blocks.push(METAL_ENGINE_BLOCK);
        for block in blocks {
            for line in block.lines() {
                let stated = line.trim_start_matches(['#', ' ']);
                let Some((key, _)) = stated.split_once('=') else {
                    continue;
                };
                let key = key.trim();
                assert!(
                    worker::config::removed::refusal_for(key).is_none(),
                    "the template offers the removed key {key}: {line}"
                );
            }
        }
    }

    /// The `[engine.memory]` block, every line uncommented, is a config.
    fn every_memory_line_uncommented_parses() {
        let memory: String = MEMORY
            .lines()
            .map(|line| line.strip_prefix("# ").unwrap_or(line))
            .filter(|line| line.starts_with('[') || line.contains(" = "))
            .map(|line| format!("{line}\n"))
            .collect();
        let content = format!(
            "{HEADER}{DEFAULT_MODEL_BLOCK}{CUDA_ENGINE_BLOCK}{memory}{RUNTIME}{DEFAULT_FRAMES}{TAIL}"
        );
        let config = worker::Config::parse(&content).expect("the memory block must parse");
        assert_eq!(
            config.engine.memory.budget_device.map(|b| b.as_bytes()),
            Some(20 << 30)
        );
    }

    fn default_config_is_parseable() {
        let content = config_content_with_any_engine();
        worker::Config::parse(&content).expect("generated config must parse");
    }

    fn the_vulkan_block_states_only_keys_the_engine_declares() {
        let content = format!(
            "{HEADER}{DEFAULT_MODEL_BLOCK}{VULKAN_ENGINE_BLOCK}{MEMORY}{RUNTIME}{DEFAULT_FRAMES}{TAIL}"
        );
        worker::Config::parse(&content).expect("the vulkan template must parse");
    }

    fn the_wgpu_block_states_only_keys_the_engine_declares() {
        let content = format!(
            "{HEADER}{DEFAULT_MODEL_BLOCK}{WGPU_ENGINE_BLOCK}{MEMORY}{RUNTIME}{DEFAULT_FRAMES}{TAIL}"
        );
        worker::Config::parse(&content).expect("the wgpu template must parse");
    }

    fn the_xla_block_states_only_keys_the_engine_declares() {
        let content = format!(
            "{HEADER}{DEFAULT_MODEL_BLOCK}{XLA_ENGINE_BLOCK}{MEMORY}{RUNTIME}{DEFAULT_FRAMES}{TAIL}"
        );
        worker::Config::parse(&content).expect("the xla template must parse");
    }

    fn the_cerebras_block_states_only_keys_the_engine_declares() {
        let content = format!(
            "{HEADER}{DEFAULT_MODEL_BLOCK}{CEREBRAS_ENGINE_BLOCK}{MEMORY}{RUNTIME}{CEREBRAS_FRAMES}{TAIL}"
        );
        let config = worker::Config::parse(&content).expect("the cerebras template must parse");
        // The runtime's most (`Runahead::STEPS_MAX`, `MAX_FRAMES`), which the
        // worker's boot advises for cerebras.
        assert_eq!(
            (
                config.runtime.frame_size,
                config.runtime.frame_dispatch_depth
            ),
            (4, 15)
        );
    }

    fn a_binary_with_an_engine_writes_a_config_that_parses() {
        let Ok(content) = default_config_content() else {
            return;
        };
        worker::Config::parse(&content).expect("generated config must parse");
    }

    fn it_names_the_engine_this_binary_actually_has() {
        let expected: Option<&str> = match flavor::default_flavor() {
            #[cfg(feature = "engine-cuda")]
            Some(flavor::Flavor::Cuda) => Some("cuda"),
            #[cfg(all(feature = "metal", target_vendor = "apple"))]
            Some(flavor::Flavor::Metal) => Some("metal"),
            #[cfg(feature = "vulkan")]
            Some(flavor::Flavor::Vulkan) => Some("vulkan"),
            #[cfg(feature = "wgpu")]
            Some(flavor::Flavor::Wgpu) => Some("wgpu"),
            #[cfg(feature = "xla")]
            Some(flavor::Flavor::Xla) => Some("xla"),
            #[cfg(feature = "cerebras")]
            Some(flavor::Flavor::Cerebras) => Some("cerebras"),
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
