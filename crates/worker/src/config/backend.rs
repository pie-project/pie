//! Backend-specific option structs — typed views over `EngineConfig::options`.
//!
//! Split out of `config.rs` for size, not meaning: these are the `[model.
//! engine.options]` tables, one struct per hostable flavor, and only
//! feature-gated code ever reads them (the config layer parses them on every
//! build so a foreign flavor's file still validates). `config/schema.rs`
//! scrapes doc comments out of this file too — its `SOURCE` concatenation
//! names it — so a field's description keeps living on the field.

use std::path::PathBuf;

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

use super::units::ByteSize;

/// `[model.engine.options]` for `type = "cuda_native"`.
/// Mirrors `pie/src/pie_driver_cuda_native/config.py::CudaNativeDriverConfig`.
///
/// **THIRTEEN KEYS RETIRED** -- eleven with the boot document, then
/// `memory_profile` and `mtp_assistant_snapshot_dir` once their last readers
/// were measured (the tune command owns its objective as a flag; no boot ever
/// seated the assistant dir). `model_id`,
/// `kv_cache_dtype`, `swap_pool_size`, `runtime_quant`, `mxfp4_moe`,
/// `mtp_num_drafts`, `stream_routed_experts`, `expert_cache`,
/// `expert_host_cache`, `enable_system_speculation` and the `#[serde(skip)]`
/// `calibrate_planner` were declared, defaulted, validated and schema'd here
/// and written into a TOML only the deleted C++ engine could have read — the
/// Rust shell's `DeviceBoot`/`Budgets` never had a seat for any of them. Their
/// live successors, where one exists: naming a row is `[model] sku`; the KV
/// dtype is the loaded SKU's own (`-kv-bf16` in its name); expert residency is
/// `[model] device_weight_budget` / `host_weight_budget`
/// (`engine::Residency`); the MoE lowering is the dispatch arm's decision. A
/// config still stating one is refused by name (`deny_unknown_fields`) rather
/// than silently ignored.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct CudaNativeEngineOptions {
    /// Fraction of each GPU's memory pie may use, weights included.
    ///
    /// What is left after the weights becomes the KV pool, so this is the
    /// knob that sizes it -- `max_total_pages` only caps the result.
    ///
    /// **AND IT REACHES AN ENGINE NOW** (alto streaming §3 item 5,
    /// `next.md` B1). This key was written into the boot document's
    /// `[batching]` table, which the C++ engine read and the Rust shell never
    /// did, so for four waves it was declared, defaulted, validated and
    /// schema'd here and read by nothing: the elastic pool took ~100% of
    /// whatever the card had free. It crosses as a typed `Knobs` field on the
    /// shell's `DeviceBoot` now, and
    /// `engine_cuda::device::elastic::budget_bytes` is the one arithmetic that
    /// reads it -- `total x utilization - (total - free) - floor`, which at
    /// `1.0` is byte for byte the pool this deployment had before.
    pub gpu_mem_utilization: f64,
    /// KV page size in tokens. **Omit to let the engine's memory planner
    /// derive one** by scoring candidates against the serving profile, which
    /// is what every deployment has been getting: this field reached the
    /// engine but the planner never read it, so only the (now deleted)
    /// `PIE_CUDA_KV_PAGE_SIZE` could pin it. Setting it pins it, and the
    /// planner searches a single-candidate lattice.
    pub kv_page_size: Option<u32>,
    /// HARD cap on the runtime KV page count. **Omit to derive it from
    /// `gpu_mem_utilization`.** Setting it forces a tiny deterministic pool
    /// for contention/preempt tests + CI, independent of the forward-layout
    /// floor.
    ///
    /// Named for what it is rather than for what Metal calls its own field.
    /// Both were `total_pages` and they are not the same quantity: there the
    /// value IS the pool and 1024 is a real default; here it is a ceiling over
    /// a number derived from `gpu_mem_utilization`, and its absence is the
    /// normal case. One name for two meanings is a question a reader cannot
    /// answer from the file.
    pub max_total_pages: Option<u32>,
    /// Recurrent-state seats (KDA/GDN and conv state), one sequence in flight
    /// each; absent takes 256. A model with no state rows seats by pages alone.
    pub max_state_slots: Option<u32>,
    /// HARD pin on the prefill token budget the forward step is built for.
    ///
    /// **Omit to let the memory planner choose.** Setting it collapses that
    /// axis of the planner's lattice to a single candidate, the same way
    /// `kv_page_size` does — which is the point: a value measured on this
    /// machine beats one scored by a model of it.
    ///
    /// Named for the same quantity Metal calls `max_forward_tokens`, because
    /// here they ARE the same quantity — unlike `total_pages`, where one name
    /// covered two different things and this engine's field had to be renamed
    /// `max_total_pages` to break the collision.
    pub max_forward_tokens: Option<u32>,
    /// HARD pin on the decode width — how many requests one forward step may
    /// carry. **Omit to let the memory planner choose.**
    ///
    /// Pinning this moves `[runtime] max_concurrent_processes` underneath you
    /// when that key is absent: admission derives from the engine's request
    /// cap, so the two are one decision. Set both, or neither.
    pub max_forward_requests: Option<u32>,
    /// CUDA device string, e.g. `"cuda:0"`. Populated by the caller
    /// from `model.engine.device`; set on the C++ side via
    /// `cudaSetDevice` (see `crates/engine-cuda/csrc/src/engine.cpp`).
    #[serde(skip)]
    pub device: String,
    /// Engine-side verbose logging. Populated from `server.verbose` rather
    /// than written here.
    #[serde(skip)]
    pub verbose: bool,
    /// **HOW MUCH OF A FIRE THE SHELL RECORDS**: `"off"` (the golden eager
    /// path), `"shaped"` (eager, with graph-shaped padded schedules — the
    /// attribution arm) or `"on"` (bodies: captured at load, replayed after).
    /// **Omit for the shell's own default**, which is `"on"` — the other two
    /// are diagnostic arms and the shell prints a line when it is asked to
    /// serve one, because an uncaptured decode pays ~470 kernel launches of
    /// host time per token-step.
    ///
    /// Crosses as `DeviceBoot::graphs`, parsed into the shell's own `Graphs`
    /// at boot construction — a spelling it does not speak refuses there.
    pub graphs: Option<String>,
    /// **WHAT THE SHELL RECORDS OF A FIRE**: `"off"` (no pad, no bodies —
    /// every launch at its live extent), `"shaped"` (the pad armed, the eager
    /// walk at graph-shaped extents) or `"bodies"` (the pad armed and bodies
    /// served — captured at load, replayed after). **Omit for `"bodies"`.**
    ///
    /// Crosses as `Knobs::recording`, parsed into the shell's own `Recording`
    /// at boot construction — a spelling it does not speak refuses there.
    pub recording: Option<String>,
    /// **DEPRECATED — write `recording` instead.** `false` is
    /// `recording = "off"`; `true` lifts an `"off"` to `"shaped"`.
    pub pad: Option<bool>,
    /// **DEPRECATED — write `recording` instead.** `false` is
    /// `recording = "shaped"`; `true` is `recording = "bodies"`.
    pub bodies: Option<bool>,
    /// **DEPRECATED — write `recording` instead.** Under `"bodies"`, whether
    /// the arming pass diffs each armed body against its own eager walk and
    /// fails the load on a difference. Ignored under any other mode.
    pub golden: Option<bool>,
    /// **DEPRECATED — write `recording` instead.** Under `"bodies"`, how many
    /// megabytes of graph exec the arming pass may take off the card
    /// (the shell's own default otherwise; `0` arms nothing and warms bodies
    /// from traffic). Ignored under any other mode.
    pub bodies_mem: Option<u32>,
    /// **`Fallback::Copy` WHERE THE COMPILER'S TABLE ASKS FOR ONE.** Below the
    /// copy/split crossover — every bucket a decode fire lands in — a copy
    /// measured 1.07x the ideal against a split's 1.82x. **Omit for on.**
    /// `false` is the A/B arm and the free oracle: a copy computes the same
    /// bytes over the same rows, and only a byte-for-byte diff against a split
    /// can settle that.
    pub fallback_copy: Option<bool>,
    /// **THE GROUPED ARM**: the ops whose kernels walk a segment list are named
    /// to the compiler, so a withdrawn consumer is served as ONE launch over
    /// that list instead of one per rectangle. **Omit for on.** `false` names
    /// none of them, which is the off arm of a Grouped-versus-Split
    /// measurement — the same row order, a different answer on it.
    pub grouped: Option<bool>,
    /// **HOW MANY SIDE STREAMS THE COMPILER MAY HAND OUT.** `0` bakes an
    /// artifact with no fork group, no event point and stream 0 on every
    /// region — the artifact this compiler produced before concurrency
    /// existed, which is the only honest off arm for a streams measurement.
    /// **Omit for the device profile's own figure.**
    pub side_streams: Option<u32>,
}

// `CudaMemoryProfile` STOOD HERE -- `auto`/`latency`/`throughput`, "which
// serving objective the engine's memory planner optimises for". Measured
// before deletion: no planner input exists on `DeviceBoot`, engine-cuda
// names no such policy anywhere, and the one reader left in the workspace
// was `pie config tune`, which used the key to remember its own `--for`
// flag between runs. A tuner's memory is not an engine knob: the flag is
// stated per run now, and the key refuses by name below.

impl Default for CudaNativeEngineOptions {
    fn default() -> Self {
        Self {
            gpu_mem_utilization: 0.90,
            kv_page_size: None,
            max_total_pages: None,
            max_state_slots: None,
            max_forward_tokens: None,
            max_forward_requests: None,
            device: String::new(),
            verbose: false,
            graphs: None,
            recording: None,
            pad: None,
            bodies: None,
            golden: None,
            bodies_mem: None,
            fallback_copy: None,
            grouped: None,
            side_streams: None,
        }
    }
}

/// `[model.engine.options]` for `type = "metal"` (Apple Silicon MLX/Metal
/// engine) — page geometry and forward limits; the metal engine speaks the
/// embedded in-process ABI. `device` is the `metal:N` selector filled from
/// `model.engine.device`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct MetalEngineOptions {
    /// KV page size in tokens. Used as given -- unlike the CUDA engine, the
    /// Metal engine has no planner to derive one.
    pub kv_page_size: u32,
    /// KV pages to allocate. Used directly, and 1024 is a real default.
    ///
    /// The CUDA engine's nearest equivalent is `max_total_pages`, which is a
    /// different quantity with a name that now says so: a ceiling over a
    /// derived number, usually absent.
    pub total_pages: u32,
    /// Tokens one forward pass may carry, across all requests in the batch.
    pub max_forward_tokens: u32,
    /// Requests one forward pass may carry. Also what `max_concurrent_processes`
    /// derives from when the operator leaves it unset.
    pub max_forward_requests: u32,
    /// Tokens the KV ring holds across the whole resident fleet. Absent -- the
    /// default -- keeps the engine's own constant, which is what a `pie serve`
    /// fleet wants and what every run got before this existed.
    ///
    /// The one knob that shrinks the KV, and it only shrinks: the engine
    /// clamps to its own ceiling, so this cannot ask for a ring it will not
    /// build. `total_pages` is NOT that knob and never was -- the simple
    /// families derive their pool from this context and discard it.
    pub max_model_len: Option<u32>,
    /// Recurrent-state seats, one sequence in flight each; absent takes 256.
    pub max_state_slots: Option<u32>,
    // FIVE KEYS RETIRED FROM THIS TABLE -- `model_id`, `cpu_pages`,
    // `kv_cache_dtype`, `stream_routed_experts`, `expert_slab_bytes`. Every
    // one was parsed for a C++ Metal engine that read the raw config bytes;
    // that engine is deleted, the Rust boot arm reads only the pool geometry
    // (`kv_page_size`, `total_pages`, `max_forward_*`, `max_model_len`) and
    // `gpu_mem_utilization`, and two of the docs here asserted a C++ reader
    // in the present tense. `expert_slab_bytes`'s live successor is
    // `[model] device_weight_budget` / `host_weight_budget`
    // (`engine::Residency`); the rest have none. A config still stating one
    // is refused by name.
    /// **Fraction of the device's recommended working set pie may hold
    /// resident** — weights, kv pool and scratch. The Metal twin of the CUDA
    /// engine's `gpu_mem_utilization`, and 0.90 by default exactly as that one
    /// is.
    ///
    /// It exists because on Apple Silicon a GPU-touched `StorageModeShared`
    /// page is WIRED and the pager never evicts it, so `recommendedMaxWorking-
    /// SetSize` is a HARD ceiling and not a hint: a load whose resident weights
    /// plus kv pool cross it does not page, it resets the box. This fraction is
    /// how much of that ceiling pie claims, and the engine refuses — or streams
    /// the weight tier down through `device_weight_budget` — a load that would
    /// exceed it (`engine_metal::store::accounting`).
    ///
    /// Unlike the CUDA key it is `[model.engine.options]` rather than
    /// `[engine]`, because the Metal engine has no separate `[engine]` table on
    /// the wire — its device knobs ride the options block the runtime already
    /// writes for it, reaching the shell as `[metal] gpu_mem_utilization` in an
    /// in-memory boot document — the one key of the old bootstrap file the
    /// shell ever read.
    pub gpu_mem_utilization: f64,
    /// Kernel-selection overrides for the Metal shell, `[engine.tuning]`,
    /// handed through verbatim as the boot document's `[metal.tuning]`
    /// (`kernels_metal::tuning::Overrides` names the keys: `qmv_rows_packs`,
    /// `qmm_min_batch`, `sdpa_mma`, …). Empty by default; an unknown key is
    /// dropped there, not refused here.
    pub tuning: toml::Table,
    /// Metal device string, e.g. `"metal:0"`. Populated from
    /// `model.engine.device` rather than written here.
    #[serde(skip)]
    pub device: String,
    /// Engine-side verbose logging. Populated from `server.verbose` rather
    /// than written here.
    #[serde(skip)]
    pub verbose: bool,
}

impl Default for MetalEngineOptions {
    fn default() -> Self {
        Self {
            kv_page_size: 32,
            total_pages: 1024,
            max_forward_tokens: 10240,
            max_forward_requests: 512,
            max_model_len: None,
            max_state_slots: None,
            gpu_mem_utilization: 0.90,
            tuning: toml::Table::new(),
            device: "metal:0".to_string(),
            verbose: false,
        }
    }
}

/// `[model.engine.options]` for `type = "vulkan"` — the portable shell, on
/// whatever Vulkan device the machine exposes.
///
/// **THE DEVICE IS A NUMBER HERE**, not a `"vulkan:0"` string like the two
/// engines above: Vulkan enumerates its physical devices and the boot document
/// carries the index straight through as `[vulkan] device_index`, so there is
/// no selector to parse and no prefix to strip.
///
/// The pool half mirrors [`MetalEngineOptions`] knob for knob (the forward
/// limits, the seat/page split) because both shells size their pools from what
/// the operator states rather than from a planner. It differs in one place:
/// `max_total_pages` is CUDA's `Option<u32>` ceiling rather than Metal's
/// `total_pages`, since this shell derives its pool from
/// `gpu_mem_utilization` and only caps the result.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct VulkanEngineOptions {
    /// Which physical device to bind, in `vkEnumeratePhysicalDevices` order.
    /// `0` is the first, and a machine with one GPU never states this.
    pub device_index: u32,
    /// **Fraction of the device-local heap pie may hold resident** — weights,
    /// kv pool and scratch. The Vulkan twin of the CUDA engine's
    /// `gpu_mem_utilization`, 0.90 by default exactly as that one is.
    pub gpu_mem_utilization: f64,
    /// HARD cap on the runtime KV page count. **Omit to derive it from
    /// `gpu_mem_utilization`.** Named for the same quantity CUDA's
    /// `max_total_pages` is — a ceiling over a derived number, usually absent
    /// — and not for Metal's `total_pages`, which IS the pool.
    pub max_total_pages: Option<u32>,
    /// Tokens one forward pass may carry, across all requests in the batch.
    pub max_forward_tokens: u32,
    /// Requests one forward pass may carry. Also what
    /// `max_concurrent_processes` derives from when the operator leaves it
    /// unset.
    pub max_forward_requests: u32,
    /// Recurrent-state seats (KDA/GDN and conv state), one sequence in flight
    /// each; absent takes 256. A model with no state rows seats by pages alone.
    pub max_state_slots: Option<u32>,
    /// Load `VK_LAYER_KHRONOS_validation` when the loader can find it.
    /// Diagnostic only, and off by default: the layer costs a large multiple
    /// of the dispatch time it validates.
    pub validation: bool,
    /// Where to persist the `VkPipelineCache` between runs. **Omit for an
    /// in-memory cache**, which is correct and pays the pipeline compiles
    /// again on every boot.
    pub pipeline_cache: Option<PathBuf>,
}

impl Default for VulkanEngineOptions {
    fn default() -> Self {
        Self {
            device_index: 0,
            gpu_mem_utilization: 0.90,
            max_total_pages: None,
            max_forward_tokens: 10240,
            max_forward_requests: 512,
            max_state_slots: None,
            validation: false,
            pipeline_cache: None,
        }
    }
}

impl VulkanEngineOptions {
    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            self.gpu_mem_utilization.is_finite()
                && self.gpu_mem_utilization > 0.0
                && self.gpu_mem_utilization <= 1.0,
            "engine.gpu_mem_utilization must be finite and in (0.0, 1.0]"
        );
        // Present means the operator chose a size, so a present zero is a
        // contradiction rather than a way to say "derive".
        if let Some(pages) = self.max_total_pages {
            ensure!(
                pages > 0,
                "engine.max_total_pages must be > 0; \
                 omit it to derive from gpu_mem_utilization"
            );
        }
        ensure!(
            self.max_forward_tokens > 0,
            "engine.max_forward_tokens must be > 0"
        );
        ensure!(
            self.max_forward_requests > 0,
            "engine.max_forward_requests must be > 0"
        );
        Ok(())
    }
}

/// `[model.engine.options]` for `type = "wgpu"` — the portable shell that
/// reaches a device through whichever backend wgpu finds (Vulkan on Linux,
/// Metal on a Mac, DX12 on Windows).
///
/// **THE DEVICE IS A NUMBER HERE**, as it is for [`VulkanEngineOptions`] and
/// unlike the `"cuda:0"`/`"metal:0"` selector strings: wgpu enumerates its
/// adapters and the boot document carries the index straight through as
/// `[wgpu] adapter_index`, so there is no selector to parse.
///
/// The pool half is [`VulkanEngineOptions`] knob for knob. The two knobs that
/// differ are the ones wgpu has and Vulkan does not: `backends` narrows which
/// backends the instance may enumerate at all, and `power_preference` ranks
/// the adapters inside that set.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct WgpuEngineOptions {
    /// Which adapter to bind, in enumeration order among those the instance
    /// reaches. `0` is the first, and a machine with one GPU never states it.
    pub adapter_index: u32,
    /// Which wgpu backends the instance may enumerate, as the comma-separated
    /// list wgpu itself spells (`"vulkan"`, `"vulkan,metal,dx12"`). **Omit for
    /// every backend this build carries** — narrowing it is how an operator
    /// pins a machine that exposes two.
    pub backends: Option<String>,
    /// How adapters are ranked inside that set: `"high-performance"`,
    /// `"low-power"`, or `"none"`. A preference ranks; it never reaches a
    /// software adapter.
    pub power_preference: String,
    /// **Fraction of the adapter's device-local heap pie may hold resident** —
    /// weights, kv pool and scratch. 0.90 by default, exactly as the CUDA and
    /// Vulkan engines are.
    pub gpu_mem_utilization: f64,
    /// HARD cap on the runtime KV page count. **Omit to derive it from
    /// `gpu_mem_utilization`.** A ceiling over a derived number, like CUDA's
    /// and Vulkan's `max_total_pages` and unlike Metal's `total_pages`, which
    /// IS the pool.
    pub max_total_pages: Option<u32>,
    /// Tokens one forward pass may carry, across all requests in the batch.
    pub max_forward_tokens: u32,
    /// Requests one forward pass may carry. Also what
    /// `max_concurrent_processes` derives from when the operator leaves it
    /// unset.
    pub max_forward_requests: u32,
    /// Recurrent-state seats (KDA/GDN and conv state), one sequence in flight
    /// each; absent takes 256. A model with no state rows seats by pages alone.
    pub max_state_slots: Option<u32>,
    /// Where to persist the pipeline cache between runs. **Omit for an
    /// in-process cache**, which is correct and pays the pipeline compiles
    /// again on every boot.
    pub pipeline_cache: Option<PathBuf>,
    /// How much device memory the adapter has (`"16GiB"`), for a backend that
    /// publishes none. **Omit on Vulkan**, where the shell reads the
    /// device-local heap itself; a backend that reports nothing otherwise
    /// assumes 8GiB, which is what this key is for overriding.
    pub device_memory: Option<ByteSize>,
}

impl Default for WgpuEngineOptions {
    fn default() -> Self {
        Self {
            adapter_index: 0,
            backends: None,
            power_preference: "high-performance".to_string(),
            gpu_mem_utilization: 0.90,
            max_total_pages: None,
            max_forward_tokens: 10240,
            max_forward_requests: 512,
            max_state_slots: None,
            pipeline_cache: None,
            device_memory: None,
        }
    }
}

impl WgpuEngineOptions {
    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            self.gpu_mem_utilization.is_finite()
                && self.gpu_mem_utilization > 0.0
                && self.gpu_mem_utilization <= 1.0,
            "engine.gpu_mem_utilization must be finite and in (0.0, 1.0]"
        );
        // The three wgpu spells. Checked here rather than at boot so a typo is
        // a config refusal naming the alternatives, not an adapter request
        // that quietly ranked nothing.
        ensure!(
            matches!(
                self.power_preference.as_str(),
                "high-performance" | "low-power" | "none"
            ),
            "engine.power_preference must be one of \"high-performance\", \
             \"low-power\", \"none\"; got {:?}",
            self.power_preference
        );
        // Present means the operator chose a size, so a present zero is a
        // contradiction rather than a way to say "derive".
        if let Some(pages) = self.max_total_pages {
            ensure!(
                pages > 0,
                "engine.max_total_pages must be > 0; \
                 omit it to derive from gpu_mem_utilization"
            );
        }
        ensure!(
            self.max_forward_tokens > 0,
            "engine.max_forward_tokens must be > 0"
        );
        ensure!(
            self.max_forward_requests > 0,
            "engine.max_forward_requests must be > 0"
        );
        // Stated means the operator is correcting a backend that reports
        // nothing, so a stated zero is a contradiction, not a way to say
        // "ask the backend".
        if let Some(memory) = self.device_memory {
            ensure!(
                memory.as_bytes() > 0,
                "engine.device_memory must be > 0; \
                 omit it to read the adapter's own answer"
            );
        }
        Ok(())
    }
}

impl CudaNativeEngineOptions {
    pub(super) fn validate(&self) -> Result<()> {
        ensure!(
            self.gpu_mem_utilization.is_finite()
                && self.gpu_mem_utilization > 0.0
                && self.gpu_mem_utilization <= 1.0,
            "engine.gpu_mem_utilization must be finite and in (0.0, 1.0]"
        );
        // Present means the operator chose a size, so a present zero is a
        // contradiction rather than a way to say "derive" -- that is what
        // omitting the key is for.
        if let Some(pages) = self.max_total_pages {
            ensure!(
                pages > 0,
                "engine.max_total_pages must be > 0; \
                 omit it to derive from gpu_mem_utilization"
            );
        }
        if let Some(size) = self.kv_page_size {
            ensure!(
                size > 0,
                "engine.kv_page_size must be > 0; \
                 omit it to let the memory planner derive one"
            );
        }
        Ok(())
    }
}
