pub mod flavor;

use std::path::Path;
use std::path::PathBuf;

#[cfg(feature = "engine-cuda")]
use anyhow::Context;
use anyhow::{Result, anyhow};

#[cfg(feature = "engine-cuda")]
use runtime::engine::backend::{DeviceBoot, Diagnostics, Knobs, Transport, World, ordinal_of};

use crate::backend::flavor::Flavor;
use crate::config;
#[cfg(feature = "cerebras")]
use crate::config::CerebrasEngineOptions;
#[cfg(feature = "engine-cuda")]
use crate::config::CudaEngineOptions;
#[cfg(all(feature = "metal", target_vendor = "apple"))]
use crate::config::MetalEngineOptions;
#[cfg(feature = "vulkan")]
use crate::config::VulkanEngineOptions;
#[cfg(feature = "wgpu")]
use crate::config::WgpuEngineOptions;
#[cfg(feature = "xla")]
use crate::config::XlaEngineOptions;

#[derive(Clone)]
pub enum EngineOptions {
    #[cfg(feature = "engine-cuda")]
    Cuda(CudaEngineOptions),
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    Metal(MetalEngineOptions),
    #[cfg(feature = "vulkan")]
    Vulkan(VulkanEngineOptions),
    #[cfg(feature = "wgpu")]
    Wgpu(WgpuEngineOptions),
    #[cfg(feature = "xla")]
    Xla(XlaEngineOptions),
    #[cfg(feature = "cerebras")]
    Cerebras(CerebrasEngineOptions),
}

pub use engine::Capabilities as EngineCapabilities;

/// The fire ceilings of a shell with no planner to derive its rungs from
/// the device: the most lanes and tokens one fire carries.
#[cfg_attr(
    not(any(
        feature = "vulkan",
        feature = "wgpu",
        feature = "xla",
        feature = "cerebras",
        all(feature = "metal", target_vendor = "apple")
    )),
    allow(dead_code, reason = "only the shells without a planner take them")
)]
#[derive(Clone, Copy)]
struct Fire {
    lanes: u32,
    tokens: u32,
}

#[cfg_attr(
    not(any(
        feature = "vulkan",
        feature = "wgpu",
        feature = "xla",
        feature = "cerebras",
        all(feature = "metal", target_vendor = "apple")
    )),
    allow(dead_code, reason = "only the shells without a planner take them")
)]
impl Fire {
    /// Its lanes and tokens.
    const fn of(self) -> (u32, u32) {
        (self.lanes, self.tokens)
    }
}

#[cfg(all(feature = "metal", target_vendor = "apple"))]
const METAL_FIRE: Fire = Fire {
    lanes: 512,
    tokens: 10240,
};
/// The Metal shell has no planner to derive a page from.
#[cfg(all(feature = "metal", target_vendor = "apple"))]
const METAL_PAGE_SIZE: u32 = 32;
#[cfg(feature = "vulkan")]
const VULKAN_FIRE: Fire = Fire {
    lanes: 512,
    tokens: 10240,
};
#[cfg(feature = "wgpu")]
const WGPU_FIRE: Fire = Fire {
    lanes: 512,
    tokens: 10240,
};
#[cfg(feature = "xla")]
const XLA_FIRE: Fire = Fire {
    lanes: 64,
    tokens: 2608,
};
#[cfg(feature = "cerebras")]
const CEREBRAS_FIRE: Fire = Fire {
    lanes: 4,
    tokens: 64,
};

/// The index in a device name (`vulkan:1`, `tpu:0`), for a shell that drives
/// one device; its family must be one of `families`.
#[cfg_attr(
    not(any(feature = "vulkan", feature = "wgpu", feature = "xla")),
    allow(dead_code, reason = "only the one-device shells read an index")
)]
fn device_index(devices: &[String], families: &[&str]) -> Result<u32> {
    let [device] = devices else {
        anyhow::bail!(
            "[engine] device: this engine drives one device, named as `{}:<n>`; got {devices:?}",
            families[0]
        );
    };
    let (family, index) = device.split_once(':').unwrap_or((device.as_str(), "0"));
    anyhow::ensure!(
        families.contains(&family),
        "[engine] device {device:?}: this engine's devices are named `{}:<n>`",
        families.join(":<n>`, `")
    );
    index
        .parse()
        .map_err(|_| anyhow!("[engine] device {device:?}: {index:?} is not a device number"))
}

/// The tokens a kv page holds on the cuda shell: the planner's page.
#[cfg(any(feature = "engine-cuda", test))]
const CUDA_PAGE_SIZE: u32 = 16;

/// The lanes one fire may carry on the cuda shell; the tokens are the
/// largest rung its card holds, derived at load.
#[cfg(any(feature = "engine-cuda", test))]
const CUDA_FIRE_LANES: u32 = 512;

#[cfg(any(feature = "engine-cuda", test))]
fn cuda_budgets(max_context: Option<u32>, declared: Option<u32>) -> engine::Budgets {
    let page_size = CUDA_PAGE_SIZE;
    let stated = max_context.filter(|&len| len > 0);
    // Unstated: the context the model declares (the engine shortens it to
    // what the card holds), 4096 where it declares none.
    let context_declared = stated.is_none() && declared.is_some();
    let max_context = stated
        .or(declared)
        .unwrap_or_else(|| engine::Budgets::default().max_context)
        .max(page_size);
    engine::Budgets {
        max_lanes: CUDA_FIRE_LANES,
        // 0: the engine takes its chip's `engine.forward_tokens`, 8192 where
        // the profile says nothing, and fits the rungs to the card.
        max_tokens: 0,
        buckets: Vec::new(),
        page_size,
        max_context,
        context_declared,
        // Unstated, 0: the engine seats a slot a lane (`max_lanes`).
        slots: 0,
        max_patches: None,
        max_images: None,
        max_voxels: None,
        max_clips: None,
    }
}

#[cfg(feature = "engine-cuda")]
fn device_boot(opts: &CudaEngineOptions, home: &Path, cache_dir: &Path) -> Result<DeviceBoot> {
    let mut knobs = Knobs {
        profiles: Some(home.join("profiles")),
        ..Knobs::default()
    };
    if let Some(words) = opts.diagnostics.as_deref() {
        knobs.diagnostics = words
            .parse::<Diagnostics>()
            .map_err(|error| anyhow!("[engine] diagnostics: {error}"))?;
    }
    knobs.expert_rebalance_steps = opts.expert_rebalance_steps;
    knobs.expert_load_window = opts.expert_load_window;
    knobs.expert_rebalance_moves = opts.expert_rebalance_moves;
    knobs.expert_rebalance_gain = opts.expert_rebalance_gain;
    // The count CUDA read once, at its first call, from the environment the
    // binary started it with: the shell reads no environment of its own.
    knobs.work_queues = std::env::var("CUDA_DEVICE_MAX_CONNECTIONS")
        .ok()
        .and_then(|word| word.trim().parse::<u32>().ok());
    if let Some(word) = opts.nccl_transport.as_deref() {
        knobs.nccl_transport = word
            .parse::<Transport>()
            .map_err(|error| anyhow!("[engine] nccl_transport: {error}"))?;
    }
    Ok(DeviceBoot {
        ordinal: ordinal_of(&opts.device),
        world: World::default(),
        comm: None,
        knobs,
        cache_dir: Some(cache_dir.to_path_buf()),
    })
}

#[cfg(feature = "engine-cuda")]
fn dump_device_boot(home: &Path, boot: &DeviceBoot, group_id: usize, rank: Option<usize>) {
    let dir = home.join("logs");
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
    cache_dir: &Path,
    budgets: engine::Budgets,
    residency: engine::Residency,
    platform: poem_ir::Platform,
    frames_in_flight: u8,
    overrides: &runtime::engine::load::Overrides,
) -> Result<(engine::Loaded, String, Option<PathBuf>)> {
    let mut request = runtime::engine::load::request(
        overrides,
        snapshot_dir,
        platform,
        budgets,
        residency,
        -1,
        frames_in_flight,
    )?;
    if request.residency.disk_kv_budget != Some(0) && request.residency.disk_kv_dir.is_none() {
        let dir = crate::disk::kv_dir(cache_dir, snapshot_dir)?;
        // Each rank of an expert-parallel group keeps its own requests' kv.
        request.residency.disk_kv_dir = Some(match overrides.ep {
            Some(_) => dir.join(format!("rank{}", overrides.rank.unwrap_or(0))),
            None => dir,
        });
    }
    let deployment = request.trace.name.clone();
    let disk_kv_dir = request.residency.disk_kv_dir.clone();
    let wide = request.trace.nodes.iter().any(|node| {
        matches!(
            node.op,
            poem_ir::Operation::Attn(poem_ir::Attn::Prefill { causal: false, .. })
        )
    });
    let loaded = backend.load(request).map_err(anyhow::Error::from)?;
    if wide && !loaded.caps.bidirectional_attention {
        return Err(anyhow!(
            "{deployment} attends both ways (a prefill that is not causal), and this \
             engine's attention applies its own causal bound and cannot lift it"
        ));
    }
    Ok((loaded, deployment, disk_kv_dir))
}

#[cfg(feature = "engine-cuda")]
/// The context the model at `snapshot_dir` declares, stretched by the yarn
/// factor a deployment turns at past the config.
fn declared_context(
    snapshot_dir: &Path,
    overrides: &runtime::engine::load::Overrides,
) -> Option<u32> {
    let config = crate::weights::config_of(snapshot_dir).ok()?;
    let stretch = overrides
        .yarn
        .map(|factor| (f64::from(factor), overrides.yarn_original.map(f64::from)));
    crate::weights::declared_context(&config, stretch)
}

fn validate_snapshot_dir(snapshot_dir: &Path) -> Result<()> {
    if snapshot_dir.is_dir()
        || (crate::weights::is_file(snapshot_dir) && crate::weights::is_artifact_path(snapshot_dir))
    {
        return Ok(());
    }
    Err(anyhow!(
        "model {snapshot_dir:?} is neither a .zt artifact nor a snapshot directory; \
         `pie model import` writes the former"
    ))
}

#[cfg(feature = "engine-cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn create_engine_backend_group(
    rank_options: &[EngineOptions],
    snapshot_dir: &Path,
    home: &Path,
    group_id: usize,
    frames_in_flight: u8,
    residency: engine::Residency,
    max_context: Option<u32>,
    overrides: &runtime::engine::load::Overrides,
) -> Result<GroupEngine> {
    let cache_dir = &crate::disk::engine_cache_dir(home);
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
        let EngineOptions::Cuda(opts) = rank_options else {
            return Err(anyhow!("cuda group creation requires cuda rank options"));
        };
        let boot = device_boot(opts, home, cache_dir)?;
        if opts.verbose {
            dump_device_boot(home, &boot, group_id, Some(rank));
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
    let overrides = &runtime::engine::load::Overrides {
        tp: Some(u32::try_from(ranks).context("a rank count fits u32")?),
        ..overrides.clone()
    };
    let (loaded, deployment, disk_kv_dir) = land(
        &mut backend,
        snapshot_dir,
        cache_dir,
        cuda_budgets(max_context, declared_context(snapshot_dir, overrides)),
        residency,
        poem_ir::Platform::Cuda,
        frames_in_flight,
        overrides,
    )?;
    Ok(GroupEngine {
        deployment,
        caps: loaded.caps,
        facts: loaded.facts,
        snapshot_dir: snapshot_dir.to_path_buf(),
        disk_kv_dir,
        backend,
    })
}

/// The ranks of an expert-parallel group, one device each, loaded at once:
/// each rank's load meets the others' before it arms, so each runs on a
/// thread of its own.
#[cfg(feature = "engine-cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn create_engine_lockstep(
    rank_options: &[EngineOptions],
    snapshot_dir: &Path,
    home: &Path,
    frames_in_flight: u8,
    residency: engine::Residency,
    max_context: Option<u32>,
    overrides: &runtime::engine::load::Overrides,
) -> Result<Vec<GroupEngine>> {
    let cache_dir = &crate::disk::engine_cache_dir(home);
    validate_snapshot_dir(snapshot_dir)?;
    let mut boots = Vec::with_capacity(rank_options.len());
    for (rank, options) in rank_options.iter().enumerate() {
        #[allow(
            irrefutable_let_patterns,
            reason = "`EngineOptions` has one variant in a CUDA-only build"
        )]
        let EngineOptions::Cuda(opts) = options else {
            return Err(anyhow!(
                "an expert-parallel group requires cuda rank options"
            ));
        };
        let boot = device_boot(opts, home, cache_dir)?;
        if opts.verbose {
            dump_device_boot(home, &boot, rank, Some(rank));
        }
        boots.push(boot);
    }
    let ep = u32::try_from(boots.len()).context("a rank count fits u32")?;
    let backends = runtime::engine::backend::open::cuda_lockstep(boots)?;
    let budgets = cuda_budgets(max_context, declared_context(snapshot_dir, overrides));
    let landed: Vec<Result<GroupEngine>> = std::thread::scope(|scope| {
        let loads: Vec<_> = backends
            .into_iter()
            .enumerate()
            .map(|(rank, mut backend)| {
                let residency = residency.clone();
                let budgets = budgets.clone();
                let overrides = runtime::engine::load::Overrides {
                    ep: Some(ep),
                    rank: Some(rank as u32),
                    ..overrides.clone()
                };
                scope.spawn(move || -> Result<GroupEngine> {
                    let (loaded, deployment, disk_kv_dir) = land(
                        &mut backend,
                        snapshot_dir,
                        cache_dir,
                        budgets,
                        residency,
                        poem_ir::Platform::Cuda,
                        frames_in_flight,
                        &overrides,
                    )
                    .with_context(|| format!("loading expert-parallel rank {rank}"))?;
                    Ok(GroupEngine {
                        deployment,
                        caps: loaded.caps,
                        facts: loaded.facts,
                        snapshot_dir: snapshot_dir.to_path_buf(),
                        disk_kv_dir,
                        backend,
                    })
                })
            })
            .collect();
        loads
            .into_iter()
            .map(|load| {
                load.join()
                    .unwrap_or_else(|_| Err(anyhow!("an expert-parallel rank's load panicked")))
            })
            .collect()
    });
    landed.into_iter().collect()
}

#[cfg_attr(
    not(feature = "engine-cuda"),
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
    home: &Path,
    group_id: usize,
    frames_in_flight: u8,
    residency: engine::Residency,
    max_context: Option<u32>,
    overrides: &runtime::engine::load::Overrides,
    opened: Option<runtime::engine::EngineBox>,
) -> Result<GroupEngine> {
    let cache_dir = &crate::disk::engine_cache_dir(home);
    let _ = (group_id, &opened);
    validate_snapshot_dir(snapshot_dir)?;

    let (mut backend, budgets, platform): (
        runtime::engine::EngineBox,
        engine::Budgets,
        poem_ir::Platform,
    ) = match options {
        #[cfg(not(any(
            feature = "engine-cuda",
            feature = "vulkan",
            feature = "wgpu",
            feature = "xla",
            feature = "cerebras",
            all(feature = "metal", target_vendor = "apple")
        )))]
        _ => unreachable!("`EngineOptions` has no variants in this build"),
        #[cfg(feature = "engine-cuda")]
        EngineOptions::Cuda(opts) => {
            let boot = device_boot(opts, home, cache_dir)?;
            if opts.verbose {
                dump_device_boot(home, &boot, group_id, None);
            }
            let backend = runtime::engine::backend::open::cuda(boot)?;
            (
                backend,
                cuda_budgets(max_context, declared_context(snapshot_dir, overrides)),
                poem_ir::Platform::Cuda,
            )
        }
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        EngineOptions::Metal(opts) => {
            let mut boot_doc = "[metal]\n".to_string();
            if let Some(words) = opts.diagnostics.as_deref() {
                boot_doc.push_str(&format!(
                    "diagnostics = {}\n",
                    toml::Value::String(words.to_string())
                ));
            }
            boot_doc.push_str(&format!(
                "profiles = {}\n",
                toml::Value::String(home.join("profiles").display().to_string())
            ));
            let backend = opened.map_or_else(
                || runtime::engine::backend::open::metal(boot_doc.as_bytes()),
                Ok,
            )?;
            let page_size = METAL_PAGE_SIZE;
            let max_context = max_context.unwrap_or_else(|| engine::Budgets::default().max_context);
            let (max_lanes, max_tokens) = METAL_FIRE.of();
            (
                backend,
                engine::Budgets {
                    max_lanes,
                    max_tokens,
                    buckets: Vec::new(),
                    page_size,
                    max_context,
                    context_declared: false,
                    slots: 0,
                    max_patches: None,
                    max_images: None,
                    max_voxels: None,
                    max_clips: None,
                },
                poem_ir::Platform::Metal,
            )
        }
        #[cfg(feature = "vulkan")]
        EngineOptions::Vulkan(opts) => {
            let mut boot_doc = format!(
                "[vulkan]\ndevice_index = {}\nvalidation = {}\n",
                opts.device_index, opts.validation
            );
            // Pipelines build in seconds on a phone; they persist under
            // the engine cache unless the config names another file.
            let cache = opts
                .pipeline_cache
                .clone()
                .unwrap_or_else(|| cache_dir.join("vulkan-pipelines.bin"));
            boot_doc.push_str(&format!(
                "pipeline_cache = {}\nprofiles = {}\n",
                toml::Value::String(cache.display().to_string()),
                toml::Value::String(home.join("profiles").display().to_string())
            ));
            let backend = opened.map_or_else(
                || runtime::engine::backend::open::vulkan(boot_doc.as_bytes()),
                Ok,
            )?;
            let defaults = engine::Budgets::default();
            let (max_lanes, max_tokens) = VULKAN_FIRE.of();
            (
                backend,
                engine::Budgets {
                    max_lanes,
                    max_tokens,
                    buckets: Vec::new(),
                    page_size: defaults.page_size,
                    max_context: max_context.unwrap_or(defaults.max_context),
                    context_declared: false,
                    slots: 0,
                    max_patches: None,
                    max_images: None,
                    max_voxels: None,
                    max_clips: None,
                },
                poem_ir::Platform::Vulkan,
            )
        }
        #[cfg(feature = "wgpu")]
        EngineOptions::Wgpu(opts) => {
            let mut boot_doc = format!(
                "[wgpu]\nadapter_index = {}\npower_preference = {}\n",
                opts.adapter_index,
                toml::Value::String(opts.power_preference.clone()),
            );
            if let Some(backends) = &opts.backends {
                boot_doc.push_str(&format!(
                    "backends = {}\n",
                    toml::Value::String(backends.clone())
                ));
            }
            boot_doc.push_str(&format!(
                "profiles = {}\n",
                toml::Value::String(home.join("profiles").display().to_string())
            ));
            let backend = opened.map_or_else(
                || runtime::engine::backend::open::wgpu(boot_doc.as_bytes()),
                Ok,
            )?;
            let defaults = engine::Budgets::default();
            let (max_lanes, max_tokens) = WGPU_FIRE.of();
            (
                backend,
                engine::Budgets {
                    max_lanes,
                    max_tokens,
                    buckets: Vec::new(),
                    page_size: defaults.page_size,
                    max_context: max_context.unwrap_or(defaults.max_context),
                    context_declared: false,
                    slots: 0,
                    max_patches: None,
                    max_images: None,
                    max_voxels: None,
                    max_clips: None,
                },
                poem_ir::Platform::Wgpu,
            )
        }
        #[cfg(feature = "xla")]
        EngineOptions::Xla(opts) => {
            let mut boot_doc = format!(
                "[xla]\ndevice = {}\nprewarm = {}\nprofiles = {}\n",
                opts.device_index,
                opts.prewarm,
                toml::Value::String(home.join("profiles").display().to_string()),
            );
            if let Some(plugin) = &opts.plugin {
                boot_doc.push_str(&format!(
                    "plugin = {}\n",
                    toml::Value::String(plugin.display().to_string())
                ));
            }
            if let Some(headroom) = opts.device_headroom {
                boot_doc.push_str(&format!("headroom = {}\n", headroom.as_bytes()));
            }
            if let Some(tokens) = opts.seat_tokens {
                boot_doc.push_str(&format!("seat_tokens = {tokens}\n"));
            }
            let backend = opened.map_or_else(
                || runtime::engine::backend::open::xla(boot_doc.as_bytes()),
                Ok,
            )?;
            let defaults = engine::Budgets::default();
            let (max_lanes, max_tokens) = XLA_FIRE.of();
            (
                backend,
                engine::Budgets {
                    max_lanes,
                    max_tokens,
                    buckets: Vec::new(),
                    page_size: defaults.page_size,
                    max_context: max_context
                        .filter(|&len| len > 0)
                        .unwrap_or(defaults.max_context)
                        .max(defaults.page_size),
                    context_declared: false,
                    slots: 0,
                    max_patches: None,
                    max_images: None,
                    max_voxels: None,
                    max_clips: None,
                },
                poem_ir::Platform::Xla,
            )
        }
        #[cfg(feature = "cerebras")]
        EngineOptions::Cerebras(opts) => {
            // The engine's `[cerebras]` table: compiled programs and cut tiles cache under the
            // PIE home, as wasmtime's do.
            let mut cerebras = toml::Table::new();
            cerebras.insert("target".into(), opts.target.clone().into());
            cerebras.insert("num_threads".into(), i64::from(opts.num_threads).into());
            cerebras.insert(
                "cache_dir".into(),
                crate::disk::engine_cache_dir(home)
                    .join("cerebras")
                    .display()
                    .to_string()
                    .into(),
            );
            if let Some(cmaddr) = &opts.cmaddr {
                cerebras.insert("cmaddr".into(), cmaddr.clone().into());
            }
            for (key, value) in [
                ("lookahead", opts.lookahead),
                ("posted_loop", opts.posted_loop),
            ] {
                if let Some(v) = value {
                    cerebras.insert(key.into(), i64::from(v).into());
                }
            }
            if let Some(asynchronous) = opts.asynchronous {
                cerebras.insert("async".into(), asynchronous.into());
            }
            for (key, table) in [
                ("wafer", &opts.wafer),
                ("toolchain", &opts.toolchain),
                ("debug", &opts.debug),
            ] {
                if let Some(t) = table {
                    cerebras.insert(key.into(), toml::Value::Table(t.clone()));
                }
            }
            let mut doc = toml::Table::new();
            doc.insert("cerebras".into(), toml::Value::Table(cerebras));
            let boot_doc = toml::to_string(&doc)?;
            let backend = runtime::engine::backend::open::cerebras(boot_doc.as_bytes())?;
            let defaults = engine::Budgets::default();
            let (max_lanes, max_tokens) = CEREBRAS_FIRE.of();
            (
                backend,
                engine::Budgets {
                    max_lanes,
                    max_tokens,
                    buckets: Vec::new(),
                    page_size: defaults.page_size,
                    max_context: max_context
                        .filter(|&len| len > 0)
                        .unwrap_or(defaults.max_context)
                        .max(defaults.page_size),
                    // a stated length is the positions the wafer program is built for
                    context_declared: max_context.is_some_and(|len| len > 0),
                    slots: 0,
                    max_patches: None,
                    max_images: None,
                    max_voxels: None,
                    max_clips: None,
                },
                poem_ir::Platform::Cerebras,
            )
        }
    };
    #[cfg_attr(
        not(feature = "engine-cuda"),
        allow(
            unreachable_code,
            reason = "`EngineOptions` has no variants in this build"
        )
    )]
    // A wafer program serves the language model alone: a checkpoint carrying a vision tower
    // is served as its text deployment.
    #[cfg(feature = "cerebras")]
    let text_only;
    #[cfg(feature = "cerebras")]
    let overrides = if platform == poem_ir::Platform::Cerebras {
        text_only = without_vision(snapshot_dir, overrides);
        &text_only
    } else {
        overrides
    };
    let (loaded, deployment, disk_kv_dir) = land(
        &mut backend,
        snapshot_dir,
        cache_dir,
        budgets,
        residency,
        platform,
        frames_in_flight,
        overrides,
    )?;

    Ok(GroupEngine {
        deployment,
        caps: loaded.caps,
        facts: loaded.facts,
        snapshot_dir: snapshot_dir.to_path_buf(),
        disk_kv_dir,
        backend,
    })
}

/// `overrides` with the checkpoint's vision part left off, where its deployment carries one and
/// the config does not already leave it off; else `overrides` as they are.
#[cfg(feature = "cerebras")]
fn without_vision(
    snapshot_dir: &Path,
    overrides: &runtime::engine::load::Overrides,
) -> runtime::engine::load::Overrides {
    use runtime::engine::load::{Platform, compose, identify};
    if overrides.off.iter().any(|part| part == "vision") {
        return overrides.clone();
    }
    let mut text = overrides.clone();
    text.off.push("vision".into());
    match identify(snapshot_dir, Platform::Cerebras)
        .and_then(|base| compose(&base, &text, Platform::Cerebras))
    {
        Ok(name) => {
            tracing::info!(deployment = name, "cerebras: the vision part left off");
            text
        }
        Err(_) => overrides.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_every_case() {
        the_rungs_and_ladders_are_the_engines_to_derive();
        the_pool_budget_seats_by_state_slots_not_pages();
        an_engine_takes_an_artifact_or_a_snapshot_and_nothing_else();
    }

    fn the_rungs_and_ladders_are_the_engines_to_derive() {
        let derived = cuda_budgets(None, None);
        assert_eq!(derived.max_tokens, 0, "the card's largest rung");
        assert_eq!(
            (derived.max_patches, derived.max_images),
            (None, None),
            "the patch ladder is derived"
        );
        assert_eq!((derived.max_voxels, derived.max_clips), (None, None));
    }

    fn the_pool_budget_seats_by_state_slots_not_pages() {
        let budgets = cuda_budgets(None, None);
        assert_eq!(budgets.page_size, 16);
        assert_eq!(budgets.max_context, 4096);
        assert_eq!(budgets.slots, 0, "the engine's chip sizes the slab");
        let capped = cuda_budgets(Some(2048), Some(32_768));
        assert_eq!(capped.max_context, 2048, "`[engine] max_context` caps it");
        assert!(!capped.context_declared);
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
    pub deployment: String,
    pub caps: EngineCapabilities,
    pub facts: engine::LoadFacts,
    pub snapshot_dir: PathBuf,
    /// Where the disk tier's slot files are, when it has one.
    pub disk_kv_dir: Option<PathBuf>,
    pub backend: runtime::engine::EngineBox,
}

pub struct ModelEngines {
    pub groups: Vec<GroupEngine>,
}

/// The devices each engine of the worker runs on: one tensor-parallel group,
/// or under expert parallelism one device for each of `ep_degree` ranks.
pub fn calculate_topology(
    world_size: usize,
    tp_degree: usize,
    ep_degree: usize,
) -> Result<Vec<Vec<usize>>> {
    if tp_degree == 0 {
        anyhow::bail!("tensor_parallel_size must be > 0");
    }
    if ep_degree == 0 {
        anyhow::bail!("expert_parallel_size must be > 0");
    }
    if ep_degree > 1 {
        if tp_degree > 1 {
            anyhow::bail!(
                "expert_parallel_size = {ep_degree} beside tensor_parallel_size = \
                 {tp_degree}: a worker splits its devices by expert or by tensor, \
                 not both yet"
            );
        }
        if world_size != ep_degree {
            anyhow::bail!(
                "[engine] device lists {world_size} devices and expert_parallel_size \
                 = {ep_degree}: an expert-parallel worker runs one rank on each device"
            );
        }
        return Ok((0..ep_degree).map(|rank| vec![rank]).collect());
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
            "[engine] device lists {world_size} devices with \
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
    not(feature = "engine-cuda"),
    allow(
        unused_variables,
        unreachable_code,
        reason = "with no `engine-*` feature `EngineOptions` is uninhabited, so \
                  every path that produces one diverges"
    )
)]
pub(crate) fn build_options(e: &config::EngineConfig, flavor: Flavor) -> Result<EngineOptions> {
    match flavor {
        #[cfg(feature = "engine-cuda")]
        Flavor::Cuda => {
            let mut c: CudaEngineOptions = e
                .options
                .clone()
                .try_into()
                .map_err(|why| anyhow!("[engine] options: {why}"))?;
            let device = e
                .device
                .first()
                .ok_or_else(|| anyhow!("[engine] cuda requires at least one device"))?;
            c.device = device.clone();
            Ok(EngineOptions::Cuda(c))
        }
        #[cfg(all(feature = "metal", target_vendor = "apple"))]
        Flavor::Metal => {
            let p: MetalEngineOptions = e
                .options
                .clone()
                .try_into()
                .map_err(|why| anyhow!("[engine] options: {why}"))?;
            Ok(EngineOptions::Metal(p))
        }
        #[cfg(feature = "vulkan")]
        Flavor::Vulkan => {
            let mut v: VulkanEngineOptions = e
                .options
                .clone()
                .try_into()
                .map_err(|why| anyhow!("[engine] options: {why}"))?;
            v.device_index = device_index(&e.device, &["vulkan"])?;
            Ok(EngineOptions::Vulkan(v))
        }
        #[cfg(feature = "wgpu")]
        Flavor::Wgpu => {
            let mut w: WgpuEngineOptions = e
                .options
                .clone()
                .try_into()
                .map_err(|why| anyhow!("[engine] options: {why}"))?;
            w.adapter_index = device_index(&e.device, &["wgpu"])?;
            Ok(EngineOptions::Wgpu(w))
        }
        #[cfg(feature = "xla")]
        Flavor::Xla => {
            let mut x: XlaEngineOptions = e
                .options
                .clone()
                .try_into()
                .map_err(|why| anyhow!("[engine] options: {why}"))?;
            x.device_index = device_index(&e.device, &["tpu", "xla"])?;
            Ok(EngineOptions::Xla(x))
        }
        #[cfg(feature = "cerebras")]
        Flavor::Cerebras => {
            let c: CerebrasEngineOptions = e
                .options
                .clone()
                .try_into()
                .map_err(|why| anyhow!("[engine] options: {why}"))?;
            Ok(EngineOptions::Cerebras(c))
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
        topology_spreads_expert_ranks();
        topology_rejects_expert_and_tensor_ranks();
    }

    fn topology_spreads_expert_ranks() {
        assert_eq!(
            calculate_topology(4, 1, 4).unwrap(),
            [vec![0], vec![1], vec![2], vec![3]]
        );
        let err = calculate_topology(8, 1, 4).unwrap_err().to_string();
        assert!(err.contains("one rank on each device"), "got: {err}");
        let err = calculate_topology(4, 1, 0).unwrap_err().to_string();
        assert!(err.contains("must be > 0"), "got: {err}");
    }

    fn topology_rejects_expert_and_tensor_ranks() {
        let err = calculate_topology(4, 2, 2).unwrap_err().to_string();
        assert!(err.contains("not both yet"), "got: {err}");
    }

    fn topology_rejects_dp_two() {
        let err = calculate_topology(2, 1, 1).unwrap_err().to_string();
        assert!(err.contains("run 2 workers"), "got: {err}");
    }

    fn topology_rejects_indivisible() {
        let err = calculate_topology(3, 2, 1).unwrap_err().to_string();
        assert!(err.contains("must be divisible"), "got: {err}");
    }

    fn topology_rejects_zero_tp() {
        let err = calculate_topology(4, 0, 1).unwrap_err().to_string();
        assert!(err.contains("must be > 0"), "got: {err}");
    }
}
