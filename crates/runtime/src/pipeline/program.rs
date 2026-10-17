use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use eta_compiler::codegen::cuda::region_analysis::RegionAnalysis;
use eta_compiler::codegen::launch::LaunchPackage;
use eta_compiler::codegen::program::{Backend, EmittedKernel, emit_program};
use eta_compiler::plan::CompiledStage;
use eta_ir::container::{self, ContainerDecodeError, PortSource, TraceContainer};
use eta_ir::container_hash;
use eta_ir::op::Op;
use eta_ir::registry::{ModelProfile, Port};
use eta_ir::validate::{BoundTrace, ValidateError, bind};
use lru::LruCache;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pricing {
    pub channel_bytes: u64,
    pub num_channels: usize,
    pub rows: u32,
}

#[derive(Debug)]
pub struct RegisteredProgram {
    pub bytes: Vec<u8>,
    pub hash: u64,
    pub bound: BoundTrace,
    pub compiled_stages: Vec<CompiledStage>,
    emitted: Mutex<HashMap<Backend, Arc<EmittedProgram>>>,
    pub channel_accesses: Vec<(bool, bool)>,
    pub reads_attn_score: bool,
    pub reads_mtp_logits: bool,
    launch: std::sync::OnceLock<LaunchPackage>,
    geometry_taint: std::sync::OnceLock<eta_compiler::eval::pareval::GeometryTaint>,
    shadow_plan: std::sync::OnceLock<Arc<crate::pipeline::fire::shadow::ShadowPlan>>,
    pub pricing: Pricing,
}

#[derive(Debug)]
pub struct EmittedProgram {
    pub emitter_version: u32,
    pub kernels: Vec<EmittedKernel>,
}

impl RegisteredProgram {
    pub fn launch(&self) -> &LaunchPackage {
        self.launch.get_or_init(|| {
            eta_compiler::codegen::launch::build(&self.bound, &self.compiled_stages)
        })
    }

    pub fn geometry_taint(&self) -> &eta_compiler::eval::pareval::GeometryTaint {
        self.geometry_taint
            .get_or_init(|| eta_compiler::eval::pareval::geometry_taint(&self.bound))
    }

    pub fn shadow_plan(&self) -> Arc<crate::pipeline::fire::shadow::ShadowPlan> {
        Arc::clone(self.shadow_plan.get_or_init(|| {
            Arc::new(crate::pipeline::fire::shadow::ShadowPlan::derive(
                &self.bound,
            ))
        }))
    }

    pub fn region_analysis(&self) -> Vec<RegionAnalysis> {
        eta_compiler::codegen::cuda::region_analysis::analyze_program(&self.compiled_stages)
    }

    pub fn emitted(&self, backend: &str) -> Option<Arc<EmittedProgram>> {
        let backend = Backend::parse(backend)?;
        let mut cache = self.emitted.lock().unwrap();
        Some(Arc::clone(cache.entry(backend).or_insert_with(|| {
            Arc::new(EmittedProgram {
                emitter_version: backend.emitter_version(),
                kernels: emit_program(backend, &self.compiled_stages, &self.bound),
            })
        })))
    }
}

#[derive(Debug)]
pub enum RegisterError {
    Decode(ContainerDecodeError),
    Bind(ValidateError),
    HashCollision(u64),
}

impl fmt::Display for RegisterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegisterError::Decode(e) => write!(f, "container decode failed: {e}"),
            RegisterError::Bind(e) => write!(f, "bind failed: {e}"),
            RegisterError::HashCollision(hash) => {
                write!(f, "program hash collision for 0x{hash:016x}")
            }
        }
    }
}
impl std::error::Error for RegisterError {}

pub const DEFAULT_CAPACITY: usize = 256;

pub struct Registry {
    inner: LruCache<u64, Arc<RegisteredProgram>>,
}

impl Registry {
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self {
            inner: LruCache::new(capacity),
        }
    }

    pub fn register(
        &mut self,
        bytes: Vec<u8>,
        profile: &ModelProfile,
    ) -> Result<Arc<RegisteredProgram>, RegisterError> {
        let hash = container_hash(&bytes);
        if let Some(hit) = self.inner.get(&hash) {
            if hit.bytes != bytes {
                return Err(RegisterError::HashCollision(hash));
            }
            return Ok(hit.clone());
        }
        let decoded = container::decode(&bytes).map_err(RegisterError::Decode)?;
        let pricing = price(&decoded);
        let channel_accesses = Self::channel_accesses(&decoded);
        let reads_attn_score = Self::reads_attn_score(&decoded);
        let reads_mtp_logits = Self::reads_mtp_logits(&decoded);
        let bound = bind(decoded, profile.clone()).map_err(RegisterError::Bind)?;
        let compiled_stages = eta_compiler::plan::compile_bound(&bound);
        let launch = std::sync::OnceLock::new();
        let entry = Arc::new(RegisteredProgram {
            bytes,
            hash,
            bound,
            compiled_stages,
            channel_accesses,
            reads_attn_score,
            reads_mtp_logits,
            launch,
            geometry_taint: std::sync::OnceLock::new(),
            shadow_plan: std::sync::OnceLock::new(),
            pricing,
            emitted: Mutex::new(HashMap::new()),
        });
        self.inner.put(hash, entry.clone());
        Ok(entry)
    }

    fn reads_attn_score(container: &TraceContainer) -> bool {
        container.stages.iter().any(|stage| {
            stage.ops.iter().any(|op| {
                matches!(
                    op,
                    Op::IntrinsicVal {
                        intr: eta_ir::op::IntrinsicId::AttnScore,
                        ..
                    }
                )
            })
        })
    }

    fn reads_mtp_logits(container: &TraceContainer) -> bool {
        container.stages.iter().any(|stage| {
            stage.ops.iter().any(|op| {
                matches!(
                    op,
                    Op::IntrinsicVal {
                        intr: eta_ir::op::IntrinsicId::MtpLogits
                            | eta_ir::op::IntrinsicId::MtpDrafts,
                        ..
                    }
                )
            })
        })
    }

    fn channel_accesses(container: &TraceContainer) -> Vec<(bool, bool)> {
        let mut accesses = vec![(false, false); container.channels.len()];
        for stage in &container.stages {
            for op in &stage.ops {
                match *op {
                    Op::ChanTake(channel) => accesses[channel as usize].0 = true,
                    Op::ChanPut { chan, .. } => accesses[chan as usize].1 = true,
                    _ => {}
                }
            }
        }
        for binding in &container.ports {
            let PortSource::Channel(channel) = &binding.source else {
                continue;
            };
            if matches!(
                binding.port,
                Port::EmbedTokens | Port::Positions | Port::Slots
            ) {
                accesses[*channel as usize].0 = true;
            }
        }
        accesses
    }

    pub fn lookup(&mut self, hash: u64) -> Option<Arc<RegisteredProgram>> {
        self.inner.get(&hash).cloned()
    }
}

fn price(c: &TraceContainer) -> Pricing {
    let channel_bytes = c
        .channels
        .iter()
        .map(|ch| {
            let elem = container::const_elem_size(ch.dtype.program_dtype()) as u64;
            let cells = (ch.capacity as u64) + 1;
            ch.shape.numel() * elem * cells
        })
        .sum();
    let port_len = |port| {
        c.ports
            .iter()
            .find(|p| p.port == port)
            .and_then(|p| match &p.source {
                container::PortSource::Const { shape, .. } => Some(shape.numel() as u32),
                container::PortSource::Channel(channel) => c
                    .channels
                    .get(*channel as usize)
                    .map(|decl| decl.shape.numel() as u32),
            })
    };
    let intrinsic_rows = || {
        c.stages
            .iter()
            .flat_map(|stage| stage.ops.iter())
            .filter_map(|op| match op {
                Op::IntrinsicVal {
                    intr:
                        eta_ir::op::IntrinsicId::Velocity
                        | eta_ir::op::IntrinsicId::Hidden
                        | eta_ir::op::IntrinsicId::Pixels
                        | eta_ir::op::IntrinsicId::Logits,
                    shape,
                    ..
                } => shape.dims().first().copied(),
                _ => None,
            })
            .max()
    };
    let rows = port_len(eta_ir::registry::Port::Readout)
        .map(|readout| readout.max(1))
        .or_else(|| {
            port_len(eta_ir::registry::Port::EmbedIndptr)
                .map(|indptr| indptr.saturating_sub(1).max(1))
        })
        .or_else(intrinsic_rows)
        .unwrap_or(1);
    Pricing {
        channel_bytes,
        num_channels: c.channels.len(),
        rows,
    }
}

use std::sync::{LazyLock, MutexGuard};

static GLOBAL: LazyLock<Mutex<Registry>> = LazyLock::new(|| {
    Mutex::new(Registry::new(
        NonZeroUsize::new(DEFAULT_CAPACITY).expect("nonzero capacity"),
    ))
});

fn global() -> MutexGuard<'static, Registry> {
    GLOBAL.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn register(
    bytes: Vec<u8>,
    profile: &ModelProfile,
) -> Result<Arc<RegisteredProgram>, RegisterError> {
    global().register(bytes, profile)
}

pub fn lookup(hash: u64) -> Option<Arc<RegisteredProgram>> {
    global().lookup(hash)
}

#[must_use]
pub fn with_host_codegen<'a>(
    plan: &'a ::engine::ProgramRegistration,
    engine_backend: Option<&str>,
) -> std::borrow::Cow<'a, ::engine::ProgramRegistration> {
    let Some(backend) = engine_backend.filter(|name| Backend::parse(name).is_some()) else {
        return std::borrow::Cow::Borrowed(plan);
    };
    let registered = lookup(plan.program_hash);

    let emitted = plan
        .emitted_kernels
        .is_empty()
        .then(|| registered.as_ref()?.emitted(backend))
        .flatten();

    let region_analysis = if plan.region_analysis.is_empty() && matches!(backend, "cuda" | "hip") {
        registered
            .as_ref()
            .map(|program| program.region_analysis())
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    if emitted.is_none() && region_analysis.is_empty() {
        return std::borrow::Cow::Borrowed(plan);
    }

    let mut next = plan.clone();
    if let Some(emitted) = emitted {
        next.emitter_version = emitted.emitter_version;
        next.emitted_kernels = emitted.kernels.clone();
    }
    if !region_analysis.is_empty() {
        next.region_analysis = region_analysis;
    }
    std::borrow::Cow::Owned(next)
}

pub fn model_profile() -> ModelProfile {
    let m = crate::model::model();
    profile_from(
        m.vocab_size(),
        crate::store::registry::get(0, 0).kv_page_size,
        m.num_layers(),
        m.eta_caps(),
        crate::model::velocity_facts(m.readings()),
        crate::model::pixels_facts(m.readings()),
    )
}

fn profile_from(
    vocab: u32,
    page_size: u32,
    num_layers: u32,
    eta: crate::model::EtaCaps,
    (has_velocity, velocity_width): (bool, u32),
    (has_pixels, pixels_width): (bool, u32),
) -> ModelProfile {
    ModelProfile {
        vocab,
        page_size,
        num_layers,
        activation: eta_ir::types::Dtype::F32,
        has_lora: eta.has_lora,
        has_mtp_logits: eta.has_mtp_logits,
        mtp_depth: eta.mtp_depth,
        draft_block: eta.draft_block,
        draft_mask_token: eta.draft_mask_token,
        draft_bidirectional: eta.draft_bidirectional,
        draft_proposals_from: eta.draft_proposals_from,
        has_value_head: eta.has_value_head,
        has_attn_score: eta.has_attn_score,
        has_attn_page_mask: eta.has_attn_page_mask,
        has_velocity,
        velocity_width,
        has_pixels,
        pixels_width,
        kernels: if eta.has_kv_envelopes {
            vec![eta_ir::registry::KernelInfo {
                name: "envelope_dot".into(),
                sink_scope: None,
                replayable: true,
            }]
        } else {
            Vec::new()
        },
    }
}

#[cfg(test)]
mod pricing_tests {
    use super::price;
    use eta_ir::container::{StageProgram, TraceContainer};
    use eta_ir::op::{IntrinsicId, Op};
    use eta_ir::registry::Stage;
    use eta_ir::types::{Dtype, Shape};

    #[test]
    fn a_float_lane_is_priced_by_its_velocity_rows() {
        let container = TraceContainer {
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: vec![Op::IntrinsicVal {
                    intr: IntrinsicId::Velocity,
                    shape: Shape::matrix(256, 64),
                    dtype: Dtype::F32,
                }],
            }],
            ..TraceContainer::default()
        };
        assert_eq!(price(&container).rows, 256);
        assert_eq!(price(&TraceContainer::default()).rows, 1);
    }
}
