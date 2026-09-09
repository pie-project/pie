pub mod flavor;

use std::path::Path;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};

#[cfg(feature = "cuda")]
use runtime::engine::backend::{
    DeviceBoot, Diagnostics, Graphs, Knobs, Recording, Transport, World, ordinal_of,
};

use crate::backend::flavor::Flavor;
use crate::config;
#[cfg(any(feature = "cuda", test))]
use crate::config::CudaNativeEngineOptions;
#[cfg(all(feature = "metal", target_vendor = "apple"))]
use crate::config::MetalEngineOptions;
#[cfg(feature = "vulkan")]
use crate::config::VulkanEngineOptions;
#[cfg(feature = "wgpu")]
use crate::config::WgpuEngineOptions;

#[derive(Clone)]
pub enum EngineOptions {
    #[cfg(feature = "cuda")]
    CudaNative(CudaNativeEngineOptions),
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    Metal(MetalEngineOptions),
    #[cfg(feature = "vulkan")]
    Vulkan(VulkanEngineOptions),
    #[cfg(feature = "wgpu")]
    Wgpu(WgpuEngineOptions),
}

impl EngineOptions {
    pub fn flavor(&self) -> Flavor {
        match self {
            #[cfg(feature = "cuda")]
            EngineOptions::CudaNative(_) => Flavor::Cuda,
            #[cfg(all(feature = "metal", target_vendor = "apple"))]
            EngineOptions::Metal(_) => Flavor::Metal,
            #[cfg(feature = "vulkan")]
            EngineOptions::Vulkan(_) => Flavor::Vulkan,
            #[cfg(feature = "wgpu")]
            EngineOptions::Wgpu(_) => Flavor::Wgpu,
            #[cfg(not(any(
                feature = "cuda",
                feature = "vulkan",
                feature = "wgpu",
                all(feature = "metal", target_vendor = "apple")
            )))]
            _ => unreachable!("`EngineOptions` has no variants in this build"),
        }
    }
}

pub use engine::Capabilities as EngineCapabilities;

#[cfg(all(feature = "metal", target_vendor = "apple"))]
fn metal_geometry_is_stated(opts: &MetalEngineOptions) -> Result<()> {
    for (key, value) in [
        ("total_pages", opts.total_pages),
        ("max_forward_tokens", opts.max_forward_tokens),
        ("max_forward_requests", opts.max_forward_requests),
        ("kv_page_size", opts.kv_page_size),
    ] {
        if value == 0 {
            anyhow::bail!("[engine] {key} must be > 0");
        }
    }
    for (key, value) in [
        ("max_model_len", opts.max_model_len),
        ("max_state_slots", opts.max_state_slots),
    ] {
        if value == Some(0) {
            anyhow::bail!("[engine] {key} must be > 0 when stated");
        }
    }
    Ok(())
}

#[cfg(any(feature = "cuda", test))]
fn cuda_budgets(
    opts: &CudaNativeEngineOptions,
    adapter_seats: u32,
    patch_ceilings: (Option<u32>, Option<u32>),
    voxel_ceilings: (Option<u32>, Option<u32>),
) -> engine::Budgets {
    let page_size = opts.kv_page_size.unwrap_or(16).max(1);
    let max_context = opts
        .max_model_len
        .filter(|&len| len > 0)
        .unwrap_or_else(|| engine::Budgets::default().max_context)
        .max(page_size);
    let pages_per_slot = max_context.div_ceil(page_size).max(1);
    engine::Budgets {
        max_lanes: opts.max_forward_requests.unwrap_or(256).max(1),
        max_tokens: opts.max_forward_tokens.unwrap_or(8192).max(1),
        buckets: Vec::new(),
        max_adapters: adapter_seats,
        page_size,
        max_context,
        slots: opts.max_state_slots.unwrap_or(256).max(1),
        pages: opts
            .max_total_pages
            .unwrap_or_else(|| pages_per_slot.saturating_mul(256))
            .max(1),
        max_patches: patch_ceilings.0,
        max_images: patch_ceilings.1,
        max_voxels: voxel_ceilings.0,
        max_clips: voxel_ceilings.1,
    }
}

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
    if let Some(words) = opts.diagnostics.as_deref() {
        knobs.diagnostics = words
            .parse::<Diagnostics>()
            .map_err(|error| anyhow!("[engine] diagnostics: {error}"))?;
    }
    if let Some(word) = opts.nccl_transport.as_deref() {
        knobs.nccl_transport = word
            .parse::<Transport>()
            .map_err(|error| anyhow!("[engine] nccl_transport: {error}"))?;
    }
    Ok(DeviceBoot {
        ordinal: ordinal_of(&opts.device),
        world: World::default(),
        comm: None,
        graphs,
        knobs,
        cache_dir: Some(cache_dir.to_path_buf()),
        adapter_dir: adapter_dir.map(Path::to_path_buf),
    })
}

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

#[allow(clippy::too_many_arguments)]
fn land(
    backend: &mut runtime::engine::EngineBox,
    snapshot_dir: &Path,
    budgets: engine::Budgets,
    residency: engine::Residency,
    platform: model_ir::Platform,
    component: crate::executor::ModelComponent,
    frames_in_flight: u8,
    sku: Option<&str>,
) -> Result<engine::Loaded> {
    if component != crate::executor::ModelComponent::Full {
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
        residency,
        -1,
        frames_in_flight,
    )?;
    backend.load(request).map_err(anyhow::Error::from)
}

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
    adapter_dir: Option<&Path>,
    group_id: usize,
    component: crate::executor::ModelComponent,
    frames_in_flight: u8,
    adapters: &crate::config::AdapterConfig,
    residency: engine::Residency,
    patch_ceilings: (Option<u32>, Option<u32>),
    voxel_ceilings: (Option<u32>, Option<u32>),
    sku: Option<&str>,
) -> Result<GroupEngine> {
    validate_snapshot_dir(snapshot_dir)?;
    if rank_options.is_empty() {
        return Err(anyhow!("cuda group requires at least one rank"));
    }

    let mut boots = Vec::with_capacity(rank_options.len());
    for (rank, rank_options) in rank_options.iter().enumerate() {
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
    let widened;
    let sku = match sku {
        Some(named) => Some(named),
        None if ranks > 1 => {
            let base = runtime::engine::load::identify(snapshot_dir, model_ir::Platform::Cuda)?;
            widened = format!("{base}-tp{ranks}");
            runtime::engine::load::trace(&widened, model_ir::Platform::Cuda).with_context(
                || {
                    format!(
                        "{snapshot_dir:?} is `{base}`, and this build ships no {ranks}-rank \
                         row for it (`{widened}`); add one to the catalog or serve it on \
                         one device"
                    )
                },
            )?;
            Some(widened.as_str())
        }
        None => None,
    };
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
        cuda_budgets(opts, adapters.seats(), patch_ceilings, voxel_ceilings),
        residency,
        model_ir::Platform::Cuda,
        component,
        frames_in_flight,
        sku,
    )?;
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn create_engine_backend(
    options: &EngineOptions,
    snapshot_dir: &Path,
    cache_dir: &Path,
    adapter_dir: Option<&Path>,
    group_id: usize,
    component: crate::executor::ModelComponent,
    frames_in_flight: u8,
    adapters: &crate::config::AdapterConfig,
    residency: engine::Residency,
    patch_ceilings: (Option<u32>, Option<u32>),
    voxel_ceilings: (Option<u32>, Option<u32>),
    sku: Option<&str>,
) -> Result<GroupEngine> {
    let _ = (group_id, cache_dir, adapter_dir);
    validate_snapshot_dir(snapshot_dir)?;

    let (mut backend, budgets, platform): (
        runtime::engine::EngineBox,
        engine::Budgets,
        model_ir::Platform,
    ) = match options {
        #[cfg(not(any(
            feature = "cuda",
            feature = "vulkan",
            feature = "wgpu",
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
                cuda_budgets(opts, adapters.seats(), patch_ceilings, voxel_ceilings),
                model_ir::Platform::Cuda,
            )
        }
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        EngineOptions::Metal(opts) => {
            let mut boot_doc = format!(
                "[metal]\ngpu_mem_utilization = {:?}\n",
                opts.gpu_mem_utilization
            );
            if let Some(words) = opts.diagnostics.as_deref() {
                boot_doc.push_str(&format!(
                    "diagnostics = {}\n",
                    toml::Value::String(words.to_string())
                ));
            }
            if !opts.tuning.is_empty() {
                boot_doc.push_str("\n[metal.tuning]\n");
                boot_doc.push_str(&opts.tuning.to_string());
            }
            if let Some(mount) = adapter_dir {
                boot_doc.push_str(&format!(
                    "\n[model]\nadapter_dir = {}\n",
                    toml::Value::String(mount.display().to_string())
                ));
            }
            metal_geometry_is_stated(opts)?;
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
                    max_patches: patch_ceilings.0,
                    max_images: patch_ceilings.1,
                    max_voxels: None,
                    max_clips: None,
                },
                model_ir::Platform::Metal,
            )
        }
        #[cfg(feature = "vulkan")]
        EngineOptions::Vulkan(opts) => {
            let mut boot_doc = format!(
                "[vulkan]\ndevice_index = {}\ngpu_mem_utilization = {:?}\nvalidation = {}\n",
                opts.device_index, opts.gpu_mem_utilization, opts.validation
            );
            if let Some(cache) = &opts.pipeline_cache {
                boot_doc.push_str(&format!(
                    "pipeline_cache = {}\n",
                    toml::Value::String(cache.display().to_string())
                ));
            }
            let backend = runtime::engine::backend::open::vulkan(boot_doc.as_bytes())?;
            let defaults = engine::Budgets::default();
            (
                backend,
                engine::Budgets {
                    max_lanes: opts.max_forward_requests.max(1),
                    max_tokens: opts.max_forward_tokens.max(1),
                    buckets: Vec::new(),
                    max_adapters: adapters.seats(),
                    page_size: defaults.page_size,
                    max_context: defaults.max_context,
                    slots: opts.max_state_slots.unwrap_or(256).max(1),
                    pages: opts.max_total_pages.unwrap_or(defaults.pages).max(1),
                    max_patches: patch_ceilings.0,
                    max_images: patch_ceilings.1,
                    max_voxels: None,
                    max_clips: None,
                },
                model_ir::Platform::Vulkan,
            )
        }
        #[cfg(feature = "wgpu")]
        EngineOptions::Wgpu(opts) => {
            let mut boot_doc = format!(
                "[wgpu]\nadapter_index = {}\ngpu_mem_utilization = {:?}\npower_preference = {}\n",
                opts.adapter_index,
                opts.gpu_mem_utilization,
                toml::Value::String(opts.power_preference.clone()),
            );
            if let Some(backends) = &opts.backends {
                boot_doc.push_str(&format!(
                    "backends = {}\n",
                    toml::Value::String(backends.clone())
                ));
            }
            if let Some(cache) = &opts.pipeline_cache {
                boot_doc.push_str(&format!(
                    "pipeline_cache = {}\n",
                    toml::Value::String(cache.display().to_string())
                ));
            }
            if let Some(memory) = opts.device_memory {
                boot_doc.push_str(&format!("device_memory = {}\n", memory.as_bytes()));
            }
            let backend = runtime::engine::backend::open::wgpu(boot_doc.as_bytes())?;
            let defaults = engine::Budgets::default();
            (
                backend,
                engine::Budgets {
                    max_lanes: opts.max_forward_requests.max(1),
                    max_tokens: opts.max_forward_tokens.max(1),
                    buckets: Vec::new(),
                    max_adapters: adapters.seats(),
                    page_size: defaults.page_size,
                    max_context: defaults.max_context,
                    slots: opts.max_state_slots.unwrap_or(256).max(1),
                    pages: opts.max_total_pages.unwrap_or(defaults.pages).max(1),
                    max_patches: patch_ceilings.0,
                    max_images: patch_ceilings.1,
                    max_voxels: None,
                    max_clips: None,
                },
                model_ir::Platform::Wgpu,
            )
        }
    };
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

    #[test]
    fn backend_every_case() {
        the_voxel_ceilings_a_deployment_states_reach_the_budget();
        the_pool_budget_seats_by_state_slots_not_pages();
        an_engine_takes_an_artifact_or_a_snapshot_and_nothing_else();
    }

    fn the_voxel_ceilings_a_deployment_states_reach_the_budget() {
        let opts = CudaNativeEngineOptions::default();
        let derived = cuda_budgets(&opts, 0, (None, None), (None, None));
        assert_eq!(derived.max_voxels, None, "unstated derives a ladder");
        assert_eq!(derived.max_clips, None);

        let stated = cuda_budgets(&opts, 0, (None, None), (Some(65_536), Some(4)));
        assert_eq!(
            stated.max_voxels,
            Some(65_536),
            "`[model] max_voxels` is the port ceiling the engine cuts against"
        );
        assert_eq!(stated.max_clips, Some(4));
    }

    fn the_pool_budget_seats_by_state_slots_not_pages() {
        let mut opts = CudaNativeEngineOptions {
            kv_page_size: Some(16),
            max_total_pages: Some(1024),
            ..Default::default()
        };
        let budgets = cuda_budgets(&opts, 0, (None, None), (None, None));
        assert_eq!(budgets.page_size, 16);
        assert_eq!(budgets.max_context, 4096);
        assert_eq!(budgets.slots, 256);
        assert_eq!(budgets.pages, 1024);
        opts.max_total_pages = None;
        assert_eq!(
            cuda_budgets(&opts, 0, (None, None), (None, None)).pages,
            65536
        );

        opts.max_state_slots = Some(4);
        assert_eq!(cuda_budgets(&opts, 0, (None, None), (None, None)).slots, 4);

        opts.max_state_slots = Some(0);
        assert_eq!(cuda_budgets(&opts, 0, (None, None), (None, None)).slots, 1);
    }

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
}

pub struct GroupEngine {
    pub caps: EngineCapabilities,
    pub facts: engine::LoadFacts,
    pub snapshot_dir: PathBuf,
    pub backend: runtime::engine::EngineBox,
}

pub struct ModelEngines {
    pub groups: Vec<GroupEngine>,
}

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
        #[cfg(feature = "vulkan")]
        Flavor::Vulkan => {
            let v: VulkanEngineOptions = m
                .engine
                .options
                .clone()
                .try_into()
                .map_err(|e| anyhow!("[engine] options for {:?}: {e}", m.name))?;
            Ok(EngineOptions::Vulkan(v))
        }
        #[cfg(feature = "wgpu")]
        Flavor::Wgpu => {
            let w: WgpuEngineOptions = m
                .engine
                .options
                .clone()
                .try_into()
                .map_err(|e| anyhow!("[engine] options for {:?}: {e}", m.name))?;
            Ok(EngineOptions::Wgpu(w))
        }
    }
}

#[cfg(test)]
mod topology_tests {
    use super::*;

    #[test]
    fn backend_1_every_case() {
        topology_rejects_dp_two();
        topology_rejects_indivisible();
        topology_rejects_zero_tp();
    }

    fn topology_rejects_dp_two() {
        let err = calculate_topology(2, 1).unwrap_err().to_string();
        assert!(err.contains("run 2 workers"), "got: {err}");
    }

    fn topology_rejects_indivisible() {
        let err = calculate_topology(3, 2).unwrap_err().to_string();
        assert!(err.contains("must be divisible"), "got: {err}");
    }

    fn topology_rejects_zero_tp() {
        let err = calculate_topology(4, 0).unwrap_err().to_string();
        assert!(err.contains("must be > 0"), "got: {err}");
    }
}
