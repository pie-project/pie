use std::path::Path;

use anyhow::{Context, Result, anyhow};
use serde::Serialize;

use crate::backend::flavor::Flavor;
use crate::backend::{EngineCapabilities, EngineOptions, GroupEngine, ModelEngines};
use crate::config;
use crate::translate;
use crate::weights;

pub(crate) struct LoadedModelEngines {
    pub(crate) model: String,
    pub(crate) caps: EngineCapabilities,
    pub(crate) full_identity: crate::executor::ModelIdentity,
    pub(crate) encode_identity: crate::executor::ModelIdentity,
    pub(crate) kv_handle: Option<engine::KvHandle>,
    pub(crate) engines: ModelEngines,
    pub(crate) metadata: runtime::model::ModelMetadata,
}

#[cfg_attr(not(feature = "net"), allow(dead_code))]
pub(crate) struct LoadedPartnerMetadata {
    pub(crate) full_identity: crate::executor::ModelIdentity,
    pub(crate) encode_identity: crate::executor::ModelIdentity,
    pub(crate) kv_handle: Option<engine::KvHandle>,
    pub(crate) page_size: u32,
    pub(crate) supports_media_encode: bool,
    pub(crate) hidden_size: u32,
}

pub(crate) fn load_model_engines(
    user_cfg: &config::Config,
    home: &Path,
    component: crate::executor::ModelComponent,
    mut opened: Option<runtime::engine::EngineBox>,
) -> Result<LoadedModelEngines> {
    let (engine_groups, snapshot_dir, metadata) = {
        let m = &user_cfg.model;
        let flavor = crate::backend::flavor::resolve(m.engine.kind, &m.name)?;

        let world_size = m.engine.device.len();
        let tp_degree = if m.engine.tensor_parallel_size == 0 {
            world_size
        } else {
            m.engine.tensor_parallel_size as usize
        };
        let topology = crate::backend::calculate_topology(world_size, tp_degree)
            .with_context(|| format!("model {:?} topology", m.name))?;

        #[allow(unreachable_patterns)]
        if tp_degree > 1 {
            match flavor {
                #[cfg(feature = "cuda")]
                Flavor::Cuda => {}
                _ => anyhow::bail!(
                    "model {:?}: tensor_parallel_size={tp_degree} is only \
                     supported for cuda_native",
                    m.name,
                ),
            }
        }

        let mut embedded_base_opts = crate::backend::build_options(m, flavor)?;
        apply_embedded_verbose(&mut embedded_base_opts, user_cfg.server.verbose);
        let resolved_model = weights::resolve(
            &m.model,
            weights::Want {
                backend: Some(flavor.as_str()),
                overrides: Some(&m.overrides()?),
            },
            home,
        )
        .with_context(|| format!("resolving the model for {:?}", m.name))?;
        let lifted = resolved_model
            .metadata()
            .with_context(|| format!("reading the model metadata for {:?}", m.name))?;
        let snapshot_dir = resolved_model.path().to_path_buf();
        let mut group_engines: Vec<GroupEngine> = Vec::with_capacity(topology.len());
        for (group_idx, group) in topology.iter().enumerate() {
            group_engines.push(create_engine_group(
                m,
                group_idx,
                group,
                flavor,
                &embedded_base_opts,
                &snapshot_dir,
                home,
                tp_degree,
                component,
                u8::try_from(user_cfg.runtime.frame_dispatch_depth).unwrap_or(u8::MAX),
                opened.take(),
            )?);
        }
        (
            ModelEngines {
                groups: group_engines,
            },
            snapshot_dir,
            lifted,
        )
    };

    let caps = engine_groups
        .groups
        .first()
        .map(|group| group.caps.clone())
        .context("no engine capabilities available for control-plane registration")?;
    let kv_handle = engine_groups
        .groups
        .first()
        .and_then(|group| group.backend.export_kv_handle());
    let artifact_digest = if user_cfg.cluster.controller.is_some() || user_cfg.offload.enabled {
        weights::model_artifact_digest(&snapshot_dir)?
    } else {
        *blake3::hash(user_cfg.model.model.as_bytes()).as_bytes()
    };
    Ok(LoadedModelEngines {
        metadata,
        model: user_cfg.model.name.clone(),
        full_identity: weights::model_identity(
            user_cfg,
            &caps,
            &artifact_digest,
            crate::executor::ModelComponent::Full,
        )?,
        encode_identity: weights::model_identity(
            user_cfg,
            &caps,
            &artifact_digest,
            crate::executor::ModelComponent::Encode,
        )?,
        caps,
        kv_handle,
        engines: engine_groups,
    })
}

/// What a boot runs the model on.
pub enum Engine {
    /// The one `[model.engine]` names, opened here.
    Configured,
    /// One the host opened itself, such as the browser's WebGPU device.
    Opened(runtime::engine::EngineBox),
    /// None: the runtime serves programs that never run a forward pass.
    None,
}

/// What booted.
#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub model: String,
    pub deployment: String,
    pub trace: String,
    pub weight_bytes: u64,
    pub kv_pages: u32,
    pub kv_page_size: u32,
    pub max_lanes: u32,
    pub max_tokens: u32,
}

/// A booted runtime; dropping it leaves the runtime running. The runtime
/// boots once per process.
pub struct Embedded {
    pub(crate) runtime: runtime::bootstrap::BootstrapHandle,
    pub summary: Summary,
}

impl Embedded {
    /// The model onto `engine`, blocking: the weights land here and the
    /// runtime is not up yet; [`Loaded::start`] boots it with `builtins`.
    /// `home` holds the runtime's files: installed inferlets, languages and
    /// caches, and the model store.
    pub fn load(
        config: &config::Config,
        home: &Path,
        engine: Engine,
        builtins: Vec<runtime::bootstrap::BuiltinProgram>,
    ) -> Result<Loaded> {
        load(config, home, engine, builtins)
    }

    pub async fn shutdown(self) -> Result<()> {
        self.runtime.shutdown().await
    }
}

/// A model on its engine, the runtime not yet booted over it.
pub struct Loaded {
    #[cfg_attr(not(feature = "net"), allow(dead_code))]
    pub(crate) model: String,
    #[cfg_attr(not(feature = "net"), allow(dead_code))]
    pub(crate) partner: Option<LoadedPartnerMetadata>,
    summary: Summary,
    config: runtime::bootstrap::Config,
}

impl Loaded {
    pub fn summary(&self) -> &Summary {
        &self.summary
    }

    pub async fn start(self) -> Result<Embedded> {
        let runtime = runtime::bootstrap::bootstrap(self.config)
            .await
            .map_err(|e| anyhow!("runtime::bootstrap::bootstrap: {e}"))?;
        Ok(Embedded {
            runtime,
            summary: self.summary,
        })
    }
}

fn load(
    user_cfg: &config::Config,
    home: &Path,
    engine: Engine,
    builtins: Vec<runtime::bootstrap::BuiltinProgram>,
) -> Result<Loaded> {
    runtime::catalog::install(&home.join("models"));
    let opened = match engine {
        Engine::None => return load_without_engine(user_cfg, home, builtins),
        Engine::Configured => None,
        Engine::Opened(engine) => Some(engine),
    };
    let LoadedModelEngines {
        model,
        caps,
        full_identity,
        encode_identity,
        kv_handle,
        engines,
        metadata,
    } = load_model_engines(
        user_cfg,
        home,
        crate::executor::ModelComponent::Full,
        opened,
    )?;
    let hidden_size: u32 = serde_json::from_slice::<serde_json::Value>(&metadata.config)
        .ok()
        .and_then(|config| config.get("hidden_size")?.as_u64())
        .and_then(|size| u32::try_from(size).ok())
        .unwrap_or(0);
    let group = &engines.groups[0];
    let summary = Summary {
        model: artifact_name(&group.snapshot_dir),
        deployment: group.deployment.clone(),
        trace: group.facts.trace_name.clone(),
        weight_bytes: group.facts.weight_bytes,
        kv_pages: caps.pools.kv_pages,
        kv_page_size: caps.pools.kv_page_size,
        max_lanes: caps.limits.max_lanes,
        max_tokens: caps.limits.max_tokens,
    };
    let config = translate::build(user_cfg, home, builtins, engines, metadata)
        .context("translating to bootstrap::Config")?;
    Ok(Loaded {
        model,
        partner: Some(LoadedPartnerMetadata {
            full_identity,
            encode_identity,
            kv_handle,
            page_size: caps.pools.kv_page_size,
            supports_media_encode: caps.media_encode,
            hidden_size,
        }),
        summary,
        config,
    })
}

/// The runtime with no engine: the deployment is the artifact's serving stamp's.
fn load_without_engine(
    user_cfg: &config::Config,
    home: &Path,
    builtins: Vec<runtime::bootstrap::BuiltinProgram>,
) -> Result<Loaded> {
    let m = &user_cfg.model;
    let want = weights::Want {
        backend: None,
        overrides: Some(&m.overrides()?),
    };
    let resolved = weights::resolve(&m.model, want, home)
        .with_context(|| format!("resolving the model for {:?}", m.name))?;
    let artifact = resolved.path().to_path_buf();
    let metadata = resolved
        .metadata()
        .with_context(|| format!("reading the model metadata for {:?}", m.name))?;
    let deployment = checkpoint::file::serve::stamp_of(&artifact)?
        .map(|stamp| stamp.deployment)
        .ok_or_else(|| anyhow!("{} carries no serving stamp", artifact.display()))?;
    let config =
        translate::build_without_engine(user_cfg, home, builtins, &artifact, &deployment, metadata);
    Ok(Loaded {
        model: m.name.clone(),
        partner: None,
        summary: Summary {
            model: artifact_name(&artifact),
            deployment: deployment.clone(),
            trace: deployment,
            weight_bytes: 0,
            kv_pages: 0,
            kv_page_size: translate::ENGINELESS_PAGE_SIZE,
            max_lanes: 0,
            max_tokens: 0,
        },
        config,
    })
}

fn artifact_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

#[allow(
    clippy::too_many_arguments,
    reason = "independent inputs to one engine launch; a struct here \
              would be a parameter list with a name"
)]
#[cfg_attr(
    not(feature = "cuda"),
    allow(
        unused_variables,
        unreachable_code,
        reason = "with no `engine-*` feature `EngineOptions` is uninhabited, so \
                  every path that takes one diverges"
    )
)]
fn create_engine_group(
    m: &config::ModelConfig,
    group_idx: usize,
    group: &[usize],
    flavor: Flavor,
    base_opts: &EngineOptions,
    snapshot_dir: &Path,
    home: &Path,
    tp_degree: usize,
    component: crate::executor::ModelComponent,
    frames_in_flight: u8,
    opened: Option<runtime::engine::EngineBox>,
) -> Result<GroupEngine> {
    #[cfg(feature = "cuda")]
    {
        if flavor == Flavor::Cuda && tp_degree > 1 {
            let rank_opts = cuda_rank_options(m, group_idx, group, base_opts)?;
            return crate::backend::create_engine_backend_group(
                &rank_opts,
                snapshot_dir,
                home,
                m.adapter_mount().as_deref(),
                group_idx,
                component,
                frames_in_flight,
                &m.adapters,
                m.residency(),
                m.patch_ceilings(),
                m.voxel_ceilings(),
                &m.overrides()?,
            )
            .with_context(|| {
                format!(
                    "creating cuda TP engine group for model {:?} group {group_idx}",
                    m.name,
                )
            });
        }
    }

    #[cfg(not(feature = "cuda"))]
    let _ = (flavor, tp_degree);

    let first_engine_idx = group.first().copied().ok_or_else(|| {
        anyhow!(
            "model {:?}: group {group_idx} is empty; topology calculation produced no ranks",
            m.name,
        )
    })?;
    let device = group_engine(m, group_idx, first_engine_idx)?;
    let opts = embedded_opts_for_device(base_opts, device);

    crate::backend::create_engine_backend(
        &opts,
        snapshot_dir,
        home,
        m.adapter_mount().as_deref(),
        group_idx,
        component,
        frames_in_flight,
        &m.adapters,
        m.residency(),
        m.patch_ceilings(),
        m.voxel_ceilings(),
        &m.overrides()?,
        opened,
    )
    .with_context(|| format!("creating engine for model {:?} group {group_idx}", m.name,))
}

fn embedded_opts_for_device(base_opts: &EngineOptions, device: String) -> EngineOptions {
    #[cfg(not(feature = "cuda"))]
    let _ = &device;

    #[allow(unreachable_patterns)]
    match base_opts {
        #[cfg(feature = "cuda")]
        EngineOptions::CudaNative(opts) => {
            let mut opts = opts.clone();
            opts.device = device;
            EngineOptions::CudaNative(opts)
        }
        other => other.clone(),
    }
}

fn apply_embedded_verbose(options: &mut EngineOptions, verbose: bool) {
    #[cfg(feature = "cuda")]
    #[allow(
        irrefutable_let_patterns,
        reason = "`EngineOptions` has one variant in a CUDA-only build"
    )]
    if let EngineOptions::CudaNative(opts) = options {
        opts.verbose = verbose;
    }

    #[cfg(not(feature = "cuda"))]
    let _ = (options, verbose);
}

#[cfg(feature = "cuda")]
fn cuda_rank_options(
    m: &config::ModelConfig,
    group_idx: usize,
    group: &[usize],
    base_opts: &EngineOptions,
) -> Result<Vec<EngineOptions>> {
    let mut rank_opts = Vec::with_capacity(group.len());
    for &rank_engine_idx in group {
        let rank_engine = group_engine(m, group_idx, rank_engine_idx)?;
        #[allow(
            unreachable_patterns,
            reason = "`EngineOptions` has one variant in a CUDA-only build"
        )]
        match base_opts {
            EngineOptions::CudaNative(opts) => {
                let mut o = opts.clone();
                o.device = rank_engine;
                rank_opts.push(EngineOptions::CudaNative(o));
            }
            _ => unreachable!("flavor checked before building cuda rank options"),
        }
    }
    Ok(rank_opts)
}

fn group_engine(m: &config::ModelConfig, group_idx: usize, engine_idx: usize) -> Result<String> {
    m.engine
        .device
        .get(engine_idx)
        .cloned()
        .ok_or_else(|| {
            anyhow!(
                "model {:?}: group {group_idx} references device index {} but only {} devices configured",
                m.name,
                engine_idx,
                m.engine.device.len(),
            )
        })
}
