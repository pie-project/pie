//! Engine-backend bootstrap helpers for pie-worker.
//!
//! Builds a runtime-owned [`runtime::engine::EngineBox`] and its
//! [`EngineCapabilities`] from the operator's config before
//! `runtime::bootstrap`, mapping options onto the per-backend boot struct
//! (e.g. cuda's [`DeviceBoot`]) that crosses the seam.

/// Which engine flavor this binary can host, and the refusals for the rest.
pub mod flavor;

use std::path::Path;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};

#[cfg(feature = "cuda")]
use runtime::engine::backend::{DeviceBoot, Graphs, Knobs, Recording, ordinal_of};

use crate::config;
#[cfg(any(feature = "cuda", test))]
use crate::config::CudaNativeEngineOptions;
#[cfg(all(feature = "metal", target_vendor = "apple"))]
use crate::config::MetalEngineOptions;
use crate::backend::flavor::Flavor;

/// Per-flavor engine options, passed to native-engine creation helpers. `Clone`
/// exists so `serve.rs` can rebuild a per-group variant.
#[derive(Clone)]
pub enum EngineOptions {
    #[cfg(feature = "cuda")]
    CudaNative(CudaNativeEngineOptions),
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    Metal(MetalEngineOptions),
}

impl EngineOptions {
    /// Which compiled flavor this options bundle targets.
    pub fn flavor(&self) -> Flavor {
        match self {
            #[cfg(feature = "cuda")]
            EngineOptions::CudaNative(_) => Flavor::Cuda,
            #[cfg(all(feature = "metal", target_vendor = "apple"))]
            EngineOptions::Metal(_) => Flavor::Metal,
            #[cfg(not(any(
                feature = "cuda",
                all(feature = "metal", target_vendor = "apple")
            )))]
            _ => unreachable!("`EngineOptions` has no variants in this build"),
        }
    }
}

/// What a load answered about itself: device, pools, limits, and a
/// `ModelProfile`. `snapshot_dir`/`model_id`/`arch_name` live on
/// [`GroupEngine`] instead — they name the caller's checkpoint, not the engine's.
pub use engine::Capabilities as EngineCapabilities;

/// The pool ceilings a load is baked against, out of what the operator stated.
///
/// `slots` seats recurrent state only (`max_state_slots`); the KV page pool
/// is shared by every live sequence and seats nothing.
#[cfg(any(feature = "cuda", test))]
fn cuda_budgets(
    opts: &CudaNativeEngineOptions,
    adapter_seats: u32,
    patch_ceilings: (Option<u32>, Option<u32>),
) -> engine::Budgets {
    let page_size = opts.kv_page_size.unwrap_or(16).max(1);
    // No CUDA knob states a context ceiling, so this is the contract's default.
    let max_context = engine::Budgets::default().max_context;
    let pages_per_slot = max_context.div_ceil(page_size).max(1);
    engine::Budgets {
        max_lanes: opts.max_forward_requests.unwrap_or(256).max(1),
        max_tokens: opts.max_forward_tokens.unwrap_or(8192).max(1),
        // Empty defers to `engine_cuda::api::lattice`'s `default_lattice` rungs.
        buckets: Vec::new(),
        // How many banks the deployment intends to register; a load whose
        // intent exceeds what the text seats is refused.
        max_adapters: adapter_seats,
        page_size,
        max_context,
        slots: opts.max_state_slots.unwrap_or(256).max(1),
        pages: opts
            .max_total_pages
            .unwrap_or_else(|| pages_per_slot.saturating_mul(256))
            .max(1),
        // Both absent: the shell derives a ladder from the loaded text.
        max_patches: patch_ceilings.0,
        max_images: patch_ceilings.1,
    }
}

/// The cuda boot, as the shell's own type: device, cache directories and
/// `[engine]` knobs. An absent knob takes `Knobs::default()`.
///
/// # Errors
///
/// A `[engine] graphs` or `recording` spelling the shell does not speak.
#[cfg(feature = "cuda")]
fn device_boot(
    opts: &CudaNativeEngineOptions,
    cache_dir: &Path,
    adapter_dir: Option<&Path>,
) -> Result<DeviceBoot> {
    let graphs = match opts.graphs.as_deref() {
        None => Graphs::default(),
        Some(word) => word
            .parse::<Graphs>()
            .map_err(|error| anyhow!("[engine] graphs: {error}"))?,
    };
    let mut knobs = Knobs {
        gpu_mem_utilization: opts.gpu_mem_utilization,
        ..Knobs::default()
    };
    if let Some(word) = opts.recording.as_deref() {
        knobs.recording = word
            .parse::<Recording>()
            .map_err(|error| anyhow!("[engine] recording: {error}"))?;
    }
    // Deprecated keys map onto `recording`; `pad` applies last since the
    // bodies route requires it.
    match opts.bodies {
        Some(true) if !knobs.bodies() => knobs.recording = Recording::default(),
        Some(false) if knobs.bodies() => knobs.recording = Recording::Shaped,
        _ => {}
    }
    if let Recording::Bodies {
        golden,
        mem_megabytes,
    } = &mut knobs.recording
    {
        if let Some(stated) = opts.golden {
            *golden = stated;
        }
        if let Some(megabytes) = opts.bodies_mem {
            *mem_megabytes = megabytes;
        }
    }
    match opts.pad {
        Some(false) => knobs.recording = Recording::Off,
        Some(true) if !knobs.pad() => knobs.recording = Recording::Shaped,
        _ => {}
    }
    if let Some(copies) = opts.fallback_copy {
        knobs.copies = copies;
    }
    if let Some(grouped) = opts.grouped {
        knobs.grouped = grouped;
    }
    if let Some(streams) = opts.side_streams {
        knobs.side_streams = Some(streams);
    }
    Ok(DeviceBoot {
        ordinal: ordinal_of(&opts.device),
        graphs,
        knobs,
        cache_dir: Some(cache_dir.to_path_buf()),
        adapter_dir: adapter_dir.map(Path::to_path_buf),
    })
}

/// A one-way, human-readable record of what a boot asked for, under
/// `$PIE_HOME/logs/`. Nothing reads this back; write failures are logged
/// and swallowed rather than failing the boot.
#[cfg(feature = "cuda")]
fn dump_device_boot(boot: &DeviceBoot, group_id: usize, rank: Option<usize>) {
    let dir = bootstrap::paths::pie_home().join("logs");
    let name = match rank {
        Some(rank) => format!("engine-boot-g{group_id}-r{rank}.txt"),
        None => format!("engine-boot-g{group_id}.txt"),
    };
    if let Err(error) = std::fs::create_dir_all(&dir)
        .and_then(|()| std::fs::write(dir.join(&name), format!("{boot:#?}\n")))
    {
        tracing::warn!(%error, name, "could not write the engine boot dump");
    }
}

/// Hand an engine its model: trace the plan runtime-side, state the
/// ceilings, and land the checkpoint. Reaching the tracer through
/// `runtime::engine::load` keeps `model` out of this crate's dependency
/// graph.
fn land(
    backend: &mut runtime::engine::EngineBox,
    snapshot_dir: &Path,
    budgets: engine::Budgets,
    residency: engine::Residency,
    platform: model_ir::Platform,
    component: crate::executor::ModelComponent,
    frames_in_flight: u8,
    // `[model] sku`, or `None` to identify one (first fit wins).
    sku: Option<&str>,
) -> Result<engine::Loaded> {
    if component != crate::executor::ModelComponent::Full {
        // The catalog ships no encoder trace, so it is refused by name.
        return Err(anyhow!(
            "this build loads only the full model; {component:?} needs a traced plan the catalog \
             does not ship"
        ));
    }
    let request = runtime::engine::load::request_of(
        sku,
        snapshot_dir,
        platform,
        budgets,
        // `None` is uncapped; a budget the shell cannot meet refuses the load.
        residency,
        -1,
        frames_in_flight,
    )?;
    backend.load(request).map_err(anyhow::Error::from)
}

/// Write the operator's adapters into the banks the load just reserved. A
/// plane file is one bank's slot, verbatim.
///
/// # Errors
///
/// A plane file that will not read, or whatever the engine refused.
fn register_operator_adapters(
    backend: &mut runtime::engine::EngineBox,
    adapters: &crate::config::AdapterConfig,
) -> Result<()> {
    for adapter in &adapters.registered {
        let mut planes = Vec::with_capacity(adapter.planes.len());
        for (bank, path) in &adapter.planes {
            let bytes = std::fs::read(path).with_context(|| {
                format!(
                    "read the plane for bank {bank:?} of adapter {} from {path:?}",
                    adapter.id
                )
            })?;
            planes.push(engine::AdapterPlane {
                bank: bank.clone(),
                bytes,
            });
        }
        let registration = engine::AdapterRegistration {
            id: adapter.id,
            planes,
        };
        runtime::engine::verbs::register_adapter(backend, &registration).with_context(|| {
            format!("registering adapter {} into this model's banks", adapter.id)
        })?;
        tracing::info!(
            id = adapter.id,
            planes = adapter.planes.len(),
            "registered an operator-declared adapter into this model's banks"
        );
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Native engine creation helpers.
// -----------------------------------------------------------------------------

/// What an engine may be pointed at: a `.zt` artifact, or a snapshot directory.
fn validate_snapshot_dir(snapshot_dir: &Path) -> Result<()> {
    if snapshot_dir.is_dir()
        || (snapshot_dir.is_file() && crate::weights::is_artifact_path(snapshot_dir))
    {
        return Ok(());
    }
    Err(anyhow!(
        "model {snapshot_dir:?} is neither a .zt artifact nor a snapshot directory; \
         `pie model import` writes the former"
    ))
}

#[cfg(feature = "cuda")]
pub(crate) fn create_engine_backend_group(
    rank_options: &[EngineOptions],
    snapshot_dir: &Path,
    cache_dir: &Path,
    // The shared-adapter mount, or `None` for the feature off.
    adapter_dir: Option<&Path>,
    group_id: usize,
    component: crate::executor::ModelComponent,
    frames_in_flight: u8,
    adapters: &crate::config::AdapterConfig,
    residency: engine::Residency,
    // `[model] max_patches` / `[model] max_images`, or both `None` to derive them.
    patch_ceilings: (Option<u32>, Option<u32>),
    // `[model] sku`, or `None` to identify one — see `land`.
    sku: Option<&str>,
) -> Result<GroupEngine> {
    validate_snapshot_dir(snapshot_dir)?;
    if rank_options.is_empty() {
        return Err(anyhow!("cuda group requires at least one rank"));
    }

    let mut boots = Vec::with_capacity(rank_options.len());
    for (rank, rank_options) in rank_options.iter().enumerate() {
        // Irrefutable in a CUDA-only build; load-bearing once another feature widens the enum.
        #[allow(
            irrefutable_let_patterns,
            reason = "`EngineOptions` has one variant in a CUDA-only build"
        )]
        let EngineOptions::CudaNative(opts) = rank_options else {
            return Err(anyhow!(
                "cuda group creation requires cuda-native rank options"
            ));
        };
        let boot = device_boot(opts, cache_dir, adapter_dir)?;
        if opts.verbose {
            dump_device_boot(&boot, group_id, Some(rank));
        }
        boots.push(boot);
    }

    let ranks = boots.len();
    let (mut backend, opened) = runtime::engine::backend::open::cuda_group(boots)?;
    if opened != ranks {
        return Err(anyhow!(
            "cuda group opened {opened} ranks for {ranks} rank configs"
        ));
    }
    // One load, not one per rank: a rank is not a load.
    #[allow(
        irrefutable_let_patterns,
        reason = "`EngineOptions` has one variant in a CUDA-only build"
    )]
    let EngineOptions::CudaNative(opts) = &rank_options[0] else {
        unreachable!("validated cuda options above");
    };
    let loaded = land(
        &mut backend,
        snapshot_dir,
        cuda_budgets(opts, adapters.seats(), patch_ceilings),
        residency,
        model_ir::Platform::Cuda,
        component,
        frames_in_flight,
        sku,
    )?;
    // The banks are reserved by the load; this is the write.
    register_operator_adapters(&mut backend, adapters)?;

    Ok(GroupEngine {
        caps: loaded.caps,
        facts: loaded.facts,
        snapshot_dir: snapshot_dir.to_path_buf(),
        backend,
    })
}

#[cfg_attr(
    not(feature = "cuda"),
    allow(
        unused_variables,
        unreachable_code,
        reason = "with no `engine-*` feature `EngineOptions` is uninhabited, so \
                  every path that takes one diverges"
    )
)]
pub(crate) fn create_engine_backend(
    options: &EngineOptions,
    snapshot_dir: &Path,
    cache_dir: &Path,
    // The shared-adapter mount, or `None` for the feature off.
    adapter_dir: Option<&Path>,
    group_id: usize,
    component: crate::executor::ModelComponent,
    frames_in_flight: u8,
    adapters: &crate::config::AdapterConfig,
    residency: engine::Residency,
    // `[model] max_patches` / `[model] max_images`, or both `None` to derive them.
    patch_ceilings: (Option<u32>, Option<u32>),
    // `[model] sku`, or `None` to identify one — see `land`.
    sku: Option<&str>,
) -> Result<GroupEngine> {
    // Each is used only inside a `#[cfg(feature = "engine-…")]` arm below.
    let _ = (group_id, cache_dir, adapter_dir);
    validate_snapshot_dir(snapshot_dir)?;

    // Typed: with no `engine-*` feature `EngineOptions` has no variants.
    let (mut backend, budgets, platform): (
        runtime::engine::EngineBox,
        engine::Budgets,
        model_ir::Platform,
    ) = match options {
        #[cfg(not(any(
            feature = "cuda",
            all(feature = "metal", target_vendor = "apple")
        )))]
        _ => unreachable!("`EngineOptions` has no variants in this build"),
        #[cfg(feature = "cuda")]
        EngineOptions::CudaNative(opts) => {
            let boot = device_boot(opts, cache_dir, adapter_dir)?;
            if opts.verbose {
                dump_device_boot(&boot, group_id, None);
            }
            let backend = runtime::engine::backend::open::cuda(boot)?;
            (
                backend,
                cuda_budgets(opts, adapters.seats(), patch_ceilings),
                model_ir::Platform::Cuda,
            )
        }
        // The shell reads exactly one key from this in-memory document.
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        EngineOptions::Metal(opts) => {
            // `{:?}` keeps the decimal point; `{}` on `1.0` reads as an integer in TOML.
            let mut boot_doc = format!(
                "[metal]\ngpu_mem_utilization = {:?}\n",
                opts.gpu_mem_utilization
            );
            if !opts.tuning.is_empty() {
                boot_doc.push_str("\n[metal.tuning]\n");
                boot_doc.push_str(&opts.tuning.to_string());
            }
            // Quoted through `toml::Value` so an unusual path cannot break the document.
            if let Some(mount) = adapter_dir {
                boot_doc.push_str(&format!(
                    "\n[model]\nadapter_dir = {}\n",
                    toml::Value::String(mount.display().to_string())
                ));
            }
            let backend = runtime::engine::backend::open::metal(boot_doc.as_bytes())?;
            let page_size = opts.kv_page_size.max(1);
            let max_context = opts
                .max_model_len
                .unwrap_or_else(|| engine::Budgets::default().max_context);
            (
                backend,
                engine::Budgets {
                    max_lanes: opts.max_forward_requests.max(1),
                    max_tokens: opts.max_forward_tokens.max(1),
                    buckets: Vec::new(),
                    max_adapters: adapters.seats(),
                    page_size,
                    max_context,
                    slots: opts.max_state_slots.unwrap_or(256).max(1),
                    pages: opts.total_pages.max(1),
                    // `[model] max_patches` / `max_images` when stated; absent,
                    // the shell derives a ladder from the loaded text
                    // (`engine_metal::api::patch_ladder`: the token ceiling,
                    // capped at two native-grid images) — which for a qwen
                    // tower (256 patches at its smallest image) is under one
                    // picture at a 128-token fire, so a vision deployment
                    // states it.
                    max_patches: patch_ceilings.0,
                    max_images: patch_ceilings.1,
                },
                model_ir::Platform::Metal,
            )
        }
    };
    // Unreachable in a build with no `engine-*` feature.
    #[cfg_attr(
        not(feature = "cuda"),
        allow(
            unreachable_code,
            reason = "`EngineOptions` has no variants in this build"
        )
    )]
    let loaded = land(
        &mut backend,
        snapshot_dir,
        budgets,
        residency,
        platform,
        component,
        frames_in_flight,
        sku,
    )?;

    // The banks are reserved by the load; this is the write.
    register_operator_adapters(&mut backend, adapters)?;

    Ok(GroupEngine {
        caps: loaded.caps,
        facts: loaded.facts,
        snapshot_dir: snapshot_dir.to_path_buf(),
        backend,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The seat count is `max_state_slots`, not the page pool.
    #[test]
    fn the_pool_budget_seats_by_state_slots_not_pages() {
        let mut opts = CudaNativeEngineOptions {
            kv_page_size: Some(16),
            max_total_pages: Some(1024),
            ..Default::default()
        };
        let budgets = cuda_budgets(&opts, 0, (None, None));
        assert_eq!(budgets.page_size, 16);
        assert_eq!(budgets.max_context, 4096);
        // No seat count stated: the contract's own default, whatever the pool.
        assert_eq!(budgets.slots, 256);
        assert_eq!(budgets.pages, 1024);
        opts.max_total_pages = None;
        assert_eq!(cuda_budgets(&opts, 0, (None, None)).pages, 65536);

        opts.max_state_slots = Some(4);
        assert_eq!(cuda_budgets(&opts, 0, (None, None)).slots, 4);

        // Zero still seats one.
        opts.max_state_slots = Some(0);
        assert_eq!(cuda_budgets(&opts, 0, (None, None)).slots, 1);
    }

    /// What an engine may be handed: an artifact, or a snapshot directory.
    #[test]
    fn an_engine_takes_an_artifact_or_a_snapshot_and_nothing_else() {
        let tmp = tempfile::tempdir().unwrap();

        let artifact = tmp.path().join("model.zt");
        std::fs::write(&artifact, b"stand-in").unwrap();
        validate_snapshot_dir(&artifact).unwrap();

        let snapshot = tmp.path().join("snap");
        std::fs::create_dir(&snapshot).unwrap();
        validate_snapshot_dir(&snapshot).unwrap();

        let gguf = tmp.path().join("model.gguf");
        std::fs::write(&gguf, b"GGUF").unwrap();
        let error = validate_snapshot_dir(&gguf).unwrap_err().to_string();
        assert!(error.contains("pie model import"), "{error}");

        let error = validate_snapshot_dir(&tmp.path().join("nope"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("neither a .zt artifact"), "{error}");
    }

    // The tests below need `DeviceBoot` (`cargo test -p worker --features cuda`).
}

/// Per-engine bundle created before bootstrap.
pub struct GroupEngine {
    /// What the load can do — device, pools, ceilings, guest-visible profile.
    pub caps: EngineCapabilities,
    /// What the load came out as: its plan's name and the bytes it landed.
    pub facts: engine::LoadFacts,
    /// Where this worker resolved the checkpoint.
    pub snapshot_dir: PathBuf,
    /// The device behind it.
    pub backend: runtime::engine::EngineBox,
}

/// Per-model bundle of concrete engine backends: one entry per DP replica.
pub struct ModelEngines {
    pub groups: Vec<GroupEngine>,
}

/// Partition `world_size` ranks into one tensor-parallel group, e.g.
/// `world_size=2, tp_degree=2 → [[0, 1]]`. Refuses >1 group: a worker
/// serves exactly one replica.
pub fn calculate_topology(world_size: usize, tp_degree: usize) -> Result<Vec<Vec<usize>>> {
    if tp_degree == 0 {
        anyhow::bail!("tensor_parallel_size must be > 0");
    }
    if !world_size.is_multiple_of(tp_degree) {
        anyhow::bail!(
            "world_size ({world_size}) must be divisible by \
             tensor_parallel_size ({tp_degree})"
        );
    }
    let num_groups = world_size / tp_degree;
    if num_groups > 1 {
        anyhow::bail!(
            "model.engine.device lists {world_size} devices with \
             tensor_parallel_size = {tp_degree}, which asks for \
             {num_groups} data-parallel replicas in one engine. A worker \
             serves one replica: run {num_groups} workers, each with \
             {tp_degree} device(s), and let the gateway spread requests \
             over them."
        );
    }
    Ok((0..num_groups)
        .map(|g| (g * tp_degree..(g + 1) * tp_degree).collect())
        .collect())
}

/// Project a [`config::ModelConfig`] into the typed [`EngineOptions`] the
/// engine expects. The cuda variant's `device` is a placeholder the
/// per-group spawn loop overwrites.
#[cfg_attr(
    not(feature = "cuda"),
    allow(
        unused_variables,
        unreachable_code,
        reason = "with no `engine-*` feature `EngineOptions` is uninhabited, so \
                  every path that produces one diverges"
    )
)]
pub(crate) fn build_options(m: &config::ModelConfig, flavor: Flavor) -> Result<EngineOptions> {
    match flavor {
        #[cfg(feature = "cuda")]
        Flavor::Cuda => {
            let mut c: CudaNativeEngineOptions = m
                .engine
                .options
                .clone()
                .try_into()
                .map_err(|e| anyhow!("[engine] options for {:?}: {e}", m.name))?;
            let device = m.engine.device.first().ok_or_else(|| {
                anyhow!(
                    "model {:?}: cuda_native requires at least one device",
                    m.name
                )
            })?;
            c.device = device.clone();
            Ok(EngineOptions::CudaNative(c))
        }
        // No device selector: `Shell::open` always takes the default Metal device.
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        Flavor::Metal => {
            let p: MetalEngineOptions = m
                .engine
                .options
                .clone()
                .try_into()
                .map_err(|e| anyhow!("[engine] options for {:?}: {e}", m.name))?;
            Ok(EngineOptions::Metal(p))
        }
    }
}

#[cfg(test)]
mod topology_tests {
    use super::*;

    #[test]
    fn topology_rejects_dp_two() {
        let err = calculate_topology(2, 1).unwrap_err().to_string();
        assert!(err.contains("run 2 workers"), "got: {err}");
    }

    #[test]
    fn topology_rejects_indivisible() {
        let err = calculate_topology(3, 2).unwrap_err().to_string();
        assert!(err.contains("must be divisible"), "got: {err}");
    }

    #[test]
    fn topology_rejects_zero_tp() {
        let err = calculate_topology(4, 0).unwrap_err().to_string();
        assert!(err.contains("must be > 0"), "got: {err}");
    }
}
