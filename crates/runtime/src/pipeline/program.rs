//! Program registry: decodes, binds, and prices ETA programs once, cached by
//! `container_hash`; the engine ships the typed [`RegisteredProgram::launch`]
//! package rather than raw container bytes.

use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use lru::LruCache;
use eta_compiler::codegen::cuda::region_analysis::RegionAnalysis;
use eta_compiler::codegen::launch::LaunchPackage;
use eta_compiler::codegen::program::{Backend, EmittedKernel, emit_program};
use eta_compiler::plan::CompiledStage;
use eta_ir::container::{self, ContainerDecodeError, PortSource, TraceContainer};
use eta_ir::container_hash;
use eta_ir::op::Op;
use eta_ir::registry::{ModelProfile, Port};
use eta_ir::validate::{BoundTrace, ValidateError, bind};

/// Registration-time pricing: per-instance costs computed once per program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pricing {
    /// Per-instance channel arena bytes: `Σ channel numel × elem_size ×
    /// (capacity + 1)` (a capacity-N channel lowers to a ring of N+1 cells).
    pub channel_bytes: u64,
    /// Number of declared channels.
    pub num_channels: usize,
    /// Row count used to size stage buffers: from the readout port if
    /// present, else the embed indptr's lane count, else 1.
    pub rows: u32,
}

/// The interned, immutable artifact for one distinct registered ETA program.
#[derive(Debug)]
pub struct RegisteredProgram {
    /// Canonical container bytes (the host→engine wire artifact).
    pub bytes: Vec<u8>,
    /// `container_hash(bytes)` — the program-set identity and cache key.
    pub hash: u64,
    /// The validated, typed artifact.
    pub bound: BoundTrace,
    /// Compiler-owned normalized stages, signatures, and region partitions —
    /// input to [`Self::launch`], [`Self::region_analysis`], and emission.
    pub compiled_stages: Vec<CompiledStage>,
    /// Backend source generated for this program, keyed by the backend an
    /// engine advertised. Generated lazily and cached: generation is tens of
    /// kilobytes per region, and a program registers once but binds many times.
    emitted: Mutex<HashMap<Backend, Arc<EmittedProgram>>>,
    /// Dense-channel `(consume, publish)` mask, derived once from immutable IR.
    pub channel_accesses: Vec<(bool, bool)>,
    /// True when any stage materializes `IntrinsicId::AttnScore` — the sole
    /// signal that a lane captures attention scores (no port or flag for it).
    pub reads_attn_score: bool,
    /// True when any stage materializes `IntrinsicId::MtpLogits` — the sole
    /// signal that a lane drafts (`Lane::drafts`), which is what selects the
    /// model text's draft arm for the lane's rows.
    pub reads_mtp_logits: bool,
    /// This program in the shape an engine executes it, built on first use.
    /// See [`Self::launch`].
    launch: std::sync::OnceLock<LaunchPackage>,
    /// Static geometry-derivability taint, and the per-pass shadow fold
    /// schedule derived from it. Both are functions of `bound` alone; derived
    /// once since many instances share one registered program.
    geometry_taint: std::sync::OnceLock<eta_compiler::eval::pareval::GeometryTaint>,
    shadow_plan: std::sync::OnceLock<Arc<crate::pipeline::fire::shadow::ShadowPlan>>,
    /// Registration-time pricing. `Pricing::rows` is read at bind
    /// (`engine::BindExtents::sampled_rows`) rather than re-derived there.
    pub pricing: Pricing,
}

/// The generated kernels for one backend, plus the emitter version an engine's
/// compile cache must key on.
#[derive(Debug)]
pub struct EmittedProgram {
    pub emitter_version: u32,
    pub kernels: Vec<EmittedKernel>,
}

impl RegisteredProgram {
    /// This program in the shape an engine executes it: typed records rather
    /// than a wire format the engine would need to parse.
    pub fn launch(&self) -> &LaunchPackage {
        self.launch.get_or_init(|| {
            eta_compiler::codegen::launch::build(&self.bound, &self.compiled_stages)
        })
    }

    /// Whether the host can derive this program's submission geometry, and
    /// which channels the device decides. Derived once — see the field.
    pub fn geometry_taint(&self) -> &eta_compiler::eval::pareval::GeometryTaint {
        self.geometry_taint
            .get_or_init(|| eta_compiler::eval::pareval::geometry_taint(&self.bound))
    }

    /// The per-pass fold schedule every instance's [`HostShadow`] runs.
    ///
    /// [`HostShadow`]: crate::pipeline::fire::shadow::HostShadow
    pub fn shadow_plan(&self) -> Arc<crate::pipeline::fire::shadow::ShadowPlan> {
        Arc::clone(self.shadow_plan.get_or_init(|| {
            Arc::new(crate::pipeline::fire::shadow::ShadowPlan::derive(
                &self.bound,
            ))
        }))
    }

    /// Per-region decisions the CUDA engine needs at bind time (gates and the
    /// intrinsic side-table layout) — computed here so codegen doesn't answer
    /// the same question twice.
    pub fn region_analysis(&self) -> Vec<RegionAnalysis> {
        eta_compiler::codegen::cuda::region_analysis::analyze_program(&self.compiled_stages)
    }

    /// Backend source for this program, generated on first ask and cached.
    ///
    /// `backend` is what the engine advertised in
    /// `EngineCapabilities::codegen_backend`; an unrecognised name means the
    /// engine generates its own kernels, and nothing is emitted. That is what
    /// lets the CUDA and Metal engines move off their in-engine emitters
    /// independently.
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

/// A registration failure — surfaces the validator/decoder's own message.
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

/// Default bound: distinct-program churn must not grow the registry without
/// limit. Traces are small; this is generous.
pub const DEFAULT_CAPACITY: usize = 256;

/// A bounded LRU of registered ETA programs, keyed by `container_hash`.
pub struct Registry {
    inner: LruCache<u64, Arc<RegisteredProgram>>,
}

impl Registry {
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self {
            inner: LruCache::new(capacity),
        }
    }

    /// Register `bytes` against `profile`: hash-deduped; on a miss, decode + bind
    /// (validator is authoritative) + price once. Identical container bytes share
    /// one `Arc`.
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

    /// Whether any stage materializes the attention-score rectangle.
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

    /// Whether any stage materializes the draft head — its step-0 logits
    /// (`MtpLogits`) or its argmax chain (`MtpDrafts`); either one makes the
    /// lane a drafting lane.
    fn reads_mtp_logits(container: &TraceContainer) -> bool {
        container.stages.iter().any(|stage| {
            stage.ops.iter().any(|op| {
                matches!(
                    op,
                    Op::IntrinsicVal {
                        intr: eta_ir::op::IntrinsicId::MtpLogits | eta_ir::op::IntrinsicId::MtpDrafts,
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
                Port::EmbedTokens | Port::Positions | Port::WSlot | Port::WOff
            ) {
                accesses[*channel as usize].0 = true;
            }
        }
        accesses
    }

    /// Probe by identity hash (a hit bumps LRU recency).
    pub fn lookup(&mut self, hash: u64) -> Option<Arc<RegisteredProgram>> {
        self.inner.get(&hash).cloned()
    }

}

/// Compute registration-time pricing from the decoded container.
fn price(c: &TraceContainer) -> Pricing {
    let channel_bytes = c
        .channels
        .iter()
        .map(|ch| {
            let elem = container::const_elem_size(ch.dtype.program_dtype()) as u64;
            let cells = (ch.capacity as u64) + 1; // ring of N+1 cells
            ch.shape.numel() * elem * cells
        })
        .sum();
    // Prefer the readout port's row count (the `SampledRows` extent shared by
    // `Port::Readout` and `IntrinsicId::Logits`); fall back to the embed
    // indptr's lane count for a container that states no readout.
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
    let rows = port_len(eta_ir::registry::Port::Readout)
        .map(|readout| readout.max(1))
        .or_else(|| {
            // A CSR of `lanes + 1` bounds, so the lane count is one less.
            port_len(eta_ir::registry::Port::EmbedIndptr)
                .map(|indptr| indptr.saturating_sub(1).max(1))
        })
        .unwrap_or(1);
    Pricing {
        channel_bytes,
        num_channels: c.channels.len(),
        rows,
    }
}

// Process-wide registry

use std::sync::{LazyLock, MutexGuard};

static GLOBAL: LazyLock<Mutex<Registry>> = LazyLock::new(|| {
    Mutex::new(Registry::new(
        NonZeroUsize::new(DEFAULT_CAPACITY).expect("nonzero capacity"),
    ))
});

fn global() -> MutexGuard<'static, Registry> {
    GLOBAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Register into the process-wide registry. See [`Registry::register`].
pub fn register(
    bytes: Vec<u8>,
    profile: &ModelProfile,
) -> Result<Arc<RegisteredProgram>, RegisterError> {
    global().register(bytes, profile)
}

/// Probe the process-wide registry by identity hash. Only the `#[cfg(test)]`
/// `instance::instantiate` path probes by hash; production carries the
/// `Arc<RegisteredProgram>` from `register`.
pub fn lookup(hash: u64) -> Option<Arc<RegisteredProgram>> {
    global().lookup(hash)
}

/// Attach the host-generated kernels and region analysis an engine reads, if
/// this engine reads them and the caller did not already supply them.
///
/// Generation is memoised per program per backend, so a re-registration costs
/// a lookup. Borrowed back unchanged when there is nothing to attach.
#[must_use]
pub fn with_host_codegen<'a>(
    plan: &'a ::engine::ProgramRegistration,
    engine_backend: Option<&str>,
) -> std::borrow::Cow<'a, ::engine::ProgramRegistration> {
    // Must be the engine's own answer, not derived from `Engine::kind()`:
    // "metal" parses as a codegen backend, but a Metal engine may still run
    // its own emitter (see `Engine::codegen_backend`).
    let Some(backend) = engine_backend.filter(|name| Backend::parse(name).is_some()) else {
        return std::borrow::Cow::Borrowed(plan);
    };
    let registered = lookup(plan.program_hash);

    // The engine carries no emitter, so a fused region with no host source is
    // a registration failure rather than a slower path.
    let emitted = plan
        .emitted_kernels
        .is_empty()
        .then(|| registered.as_ref()?.emitted(backend))
        .flatten();

    // Region analysis only means something to the CUDA emitter's own kernels.
    let region_analysis = if plan.region_analysis.is_empty() && backend == "cuda" {
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
        // EmittedKernel is the compiler's own record; carried directly.
        next.emitted_kernels = emitted.kernels.clone();
    }
    if !region_analysis.is_empty() {
        next.region_analysis = region_analysis;
    }
    std::borrow::Cow::Owned(next)
}

/// Build the bind-time [`ModelProfile`] from the loaded model. Model-gated
/// intrinsics and second-party kernels default conservative until the model
/// surfaces them.
pub fn model_profile() -> ModelProfile {
    let m = crate::model::model();
    profile_from(
        m.vocab_size(),
        crate::store::registry::get(0, 0).kv_page_size,
        m.num_layers(),
        m.eta_caps(),
    )
}

/// The pure caps -> profile mapping, split out so it is testable without a
/// registered model.
fn profile_from(
    vocab: u32,
    page_size: u32,
    num_layers: u32,
    eta: crate::model::EtaCaps,
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
        // Second-party kernels the backend advertises. `envelope_dot` is
        // replayable (a pure function of the query and the page envelopes) and
        // has no sink scope: it produces a value, it does not consume one.
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

