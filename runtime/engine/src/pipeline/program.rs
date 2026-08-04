//! Program registry (thrust-3 P2.2/P2.3) — the host-side "register a traced
//! pass once, cache by identity" counterpart to the inferlet program cache.
//!
//! The wire artifact is the **canonical container bytes** (the `pie_ir`
//! IR's `container`); the guest cannot bind (bind needs the backend
//! [`ModelProfile`]). Registration:
//!
//! 1. `container_hash(bytes)` — the C3 identity + guest cache key (one number).
//! 2. hit ⇒ return the shared [`RegisteredProgram`]; miss ⇒
//! 3. `decode(bytes)` → `bind(container, profile)` (the authoritative
//!    validator; malformed traces fail here with the validator's message — the
//!    P2 exit), then derive registration-time **pricing** (channel bytes, rows,
//!    channel/stage counts — P2.3) once, and cache by hash.
//!
//! Pure host-side (no GPU). The driver keeps its own compile cache under the same
//! hash; the host→driver ship is the typed [`RegisteredProgram::launch`] package.
//!
//! [`model_profile`] builds the bind-time [`ModelProfile`] from the loaded
//! model: program-registration input, not fire-time glue.
//!
//! The registry probes ([`Registry::lookup`]/[`Registry::len`], the free
//! [`lookup`]) are `#[cfg(test)]`: production carries the
//! `Arc<RegisteredProgram>` that [`register`] returns rather than probing by
//! hash. [`Pricing`] is the one thing computed on the
//! production path with no production consumer yet — thrust-2 capacity
//! accounting is unwired — so it carries an annotated `allow` instead of a
//! blanket module-level one.

use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use lru::LruCache;
use pie_codegen::program::{Backend, EmittedKernel, emit_program};
use pie_driver_abi::plan::{DirectArgmax, RegionAnalysis};
use pie_ir::container::{self, ContainerDecodeError, PortSource, TraceContainer};
use pie_ir::container_hash;
use pie_ir::op::Op;
use pie_ir::registry::{ModelProfile, Port};
use pie_ir::validate::{BoundTrace, ValidateError, bind};
use pie_plan::CompiledStage;

/// Registration-time pricing (thrust-3 P2.3): per-instance costs computed once
/// per program and attached to its identity (feeds thrust-2's capacity
/// accounting through C3's opaque identity).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pricing {
    /// Exact per-instance channel arena bytes: `Σ channel numel × elem_size ×
    /// (capacity + 1)` (a capacity-N channel lowers to a ring of N+1 cells, §7.1).
    pub channel_bytes: u64,
    /// Number of declared channels.
    pub num_channels: usize,
    /// Read-out row count (from the `embed` indptr lanes, else 1) — the sampling
    /// row count thrust-2 prices the fire on.
    pub rows: u32,
}

/// The interned, immutable artifact for one distinct registered PTIR program.
#[derive(Debug)]
pub struct RegisteredProgram {
    /// Canonical container bytes (the host→driver wire artifact).
    pub bytes: Vec<u8>,
    /// `container_hash(bytes)` — the C3 program-set identity + program-id.
    pub hash: u64,
    /// The validated, typed artifact (types, readiness, §7.1 classes).
    pub bound: BoundTrace,
    /// Compiler-owned normalized stages, signatures, and region partitions —
    /// the input every host-side derivation reads: [`Self::launch`],
    /// [`Self::region_analysis`], and backend emission.
    pub compiled_stages: Vec<CompiledStage>,
    /// Backend source generated for this program, keyed by the backend a
    /// driver advertised. Emitted lazily and cached here because generation is
    /// tens of kilobytes per region and a program is registered once but bound
    /// many times.
    emitted: Mutex<HashMap<Backend, Arc<EmittedProgram>>>,
    /// Dense-channel `(consume, publish)` mask, derived once from immutable IR.
    pub channel_accesses: Vec<(bool, bool)>,
    /// This program in the shape a driver executes it, built on first use.
    /// See [`Self::launch`].
    launch: std::sync::OnceLock<pie_driver_abi::plan::LaunchPackage>,
    /// Static geometry-derivability taint, and the per-pass shadow fold
    /// schedule derived from it. Both are functions of `bound` alone, and a
    /// program is registered once but instantiated many times — at a cohort
    /// boundary a whole herd instantiates at once, so re-deriving a
    /// whole-trace taint fixpoint per instance lands squarely on the
    /// critical path.
    geometry_taint: std::sync::OnceLock<pie_eval::pareval::GeometryTaint>,
    shadow_plan: std::sync::OnceLock<Arc<crate::pipeline::fire::shadow::ShadowPlan>>,
    /// Registration-time pricing. Computed by [`price`] on every register,
    /// but nothing consumes it yet (thrust-2 capacity accounting is
    /// unwired) — the `allow` marks that gap rather than hiding it.
    #[allow(dead_code)]
    pub pricing: Pricing,
}

/// The generated kernels for one backend, plus the emitter version a driver's
/// compile cache must key on.
#[derive(Debug)]
pub struct EmittedProgram {
    pub emitter_version: u32,
    pub kernels: Vec<EmittedKernel>,
}

impl RegisteredProgram {
    /// **The launch package** — this program in the shape a driver executes it.
    ///
    /// This is what replaced the container bytes and the PTIB sidecar. A driver
    /// receives typed records instead of PTIR, so it has no wire format to parse and no
    /// plan to re-derive (`ptir-refactor.md` §2.3).
    pub fn launch(&self) -> &pie_driver_abi::plan::LaunchPackage {
        self.launch
            .get_or_init(|| pie_codegen::launch::build(&self.bound, &self.compiled_stages))
    }

    /// Whether the host can derive this program's submission geometry, and
    /// which channels the device decides. Derived once — see the field.
    pub fn geometry_taint(&self) -> &pie_eval::pareval::GeometryTaint {
        self.geometry_taint
            .get_or_init(|| pie_eval::pareval::geometry_taint(&self.bound))
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

    /// Every per-region decision the CUDA driver derives for itself in
    /// `region_support.hpp` — the bind-time gates and the intrinsic side-table
    /// analysis (`ptir-refactor.md` §4.2).
    ///
    /// The analysis is CUDA's, but it is not code generation: it is the same
    /// question the emitter had to answer to build the kernel, which is exactly
    /// why it must not be answered twice. Shipped on the same terms as
    /// `stage_identities` — the driver compares while both exist.
    pub fn region_analysis(&self) -> Vec<RegionAnalysis> {
        pie_codegen::cuda::region_analysis::analyze_program(&self.compiled_stages)
            .into_iter()
            .map(|region| RegionAnalysis {
                stage_index: region.stage_index,
                region_index: region.region_index,
                flags: region.flags,
                direct_argmax: region
                    .direct_argmax
                    .into_iter()
                    .map(|record| DirectArgmax {
                        node: record.node,
                        source_value: record.source_value,
                        intrinsic: record.intrinsic,
                        requires_single_row: record.requires_single_row,
                    })
                    .collect(),
                skipped: region.skipped,
            })
            .collect()
    }

    /// Backend source for this program, generated on first ask and cached.
    ///
    /// `backend` is what the driver advertised in
    /// `DriverCapabilities::codegen_backend`; an unrecognised name means the
    /// driver generates its own kernels, and nothing is emitted. That is what
    /// lets the CUDA and Metal drivers move off their in-driver emitters
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

/// A registration failure — surfaces the authoritative validator/decoder message
/// (the P2 exit: "malformed traces fail at bind with the P0 validator's message").
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

/// A bounded LRU of registered PTIR programs, keyed by `container_hash`.
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
        let bound = bind(decoded, profile.clone()).map_err(RegisterError::Bind)?;
        let compiled_stages = pie_plan::compile_bound(&bound);
        if std::env::var_os("PIE_PTIR_DUMP_PLAN").is_some() {
            for stage in &compiled_stages {
                eprintln!("{}", pie_plan::debug_stage_plan(stage));
                eprintln!("  metrics={:?}", stage.metrics());
            }
        }
        let launch = std::sync::OnceLock::new();
        let entry = Arc::new(RegisteredProgram {
            bytes,
            hash,
            bound,
            compiled_stages,
            channel_accesses,
            launch,
            geometry_taint: std::sync::OnceLock::new(),
            shadow_plan: std::sync::OnceLock::new(),
            pricing,
            emitted: Mutex::new(HashMap::new()),
        });
        self.inner.put(hash, entry.clone());
        Ok(entry)
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

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }
}

/// Compute registration-time pricing from the decoded container (P2.3).
fn price(c: &TraceContainer) -> Pricing {
    let channel_bytes = c
        .channels
        .iter()
        .map(|ch| {
            let elem = container::const_elem_size(ch.dtype.program_dtype()) as u64;
            let cells = (ch.capacity as u64) + 1; // ring of N+1 (§7.1)
            ch.shape.numel() * elem * cells
        })
        .sum();
    // Rows = embed indptr lanes (numel - 1) when the indptr folds to a constant.
    let rows = c
        .ports
        .iter()
        .find(|p| p.port == pie_ir::registry::Port::EmbedIndptr)
        .and_then(|p| match &p.source {
            container::PortSource::Const { shape, .. } => {
                Some((shape.numel() as u32).saturating_sub(1).max(1))
            }
            container::PortSource::Channel(channel) => c
                .channels
                .get(*channel as usize)
                .map(|decl| (decl.shape.numel() as u32).saturating_sub(1).max(1)),
        })
        .unwrap_or(1);
    Pricing {
        channel_bytes,
        num_channels: c.channels.len(),
        rows,
    }
}

// ---------------------------------------------------------------------------
// Process-wide registry
// ---------------------------------------------------------------------------

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

/// Build the bind-time [`ModelProfile`] from the loaded model (P2b: vocab +
/// page-size + layer caps; model-gated intrinsics + second-party kernels default
/// conservative until the model surfaces them).
pub fn model_profile() -> ModelProfile {
    let m = pie_model::model();
    profile_from(
        m.vocab_size(),
        crate::store::registry::get(0, 0).kv_page_size,
        m.num_layers(),
        m.ptir_caps(),
    )
}

/// The pure caps -> profile mapping, split out so it is testable without a
/// registered model.
fn profile_from(
    vocab: u32,
    page_size: u32,
    num_layers: u32,
    ptir: pie_model::PtirCaps,
) -> ModelProfile {
    ModelProfile {
        vocab,
        page_size,
        num_layers,
        activation: pie_ir::types::DType::F32,
        has_lora: ptir.has_lora,
        has_mtp_logits: ptir.has_mtp_logits,
        has_mtp_drafts: ptir.has_mtp_drafts,
        has_value_head: ptir.has_value_head,
        has_attn_score: ptir.has_attn_score,
        has_attn_page_mask: ptir.has_attn_page_mask,
        // Second-party kernels the backend advertises. `envelope_dot` is
        // replayable (a pure function of the query and the page envelopes) and
        // has no sink scope: it produces a value, it does not consume one.
        kernels: if ptir.has_kv_envelopes {
            vec![pie_ir::registry::KernelInfo {
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
mod tests {
    use super::*;
    use pie_ir::container::{
        ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram,
    };
    use pie_ir::op::{IntrinsicId, Op};
    use pie_ir::registry::{Port, Stage};
    use pie_ir::types::{DType, Shape};

    const VOCAB: u32 = 32;

    fn chan(shape: Shape, dtype: DType, role: HostRole, seeded: bool) -> ChannelDecl {
        ChannelDecl {
            shape,
            dtype: ChanDType::Concrete(dtype),
            capacity: 1,
            host_role: role,
            seeded,
        }
    }

    fn u32_port(port: Port, shape: Shape, values: &[u32]) -> PortBinding {
        PortBinding {
            port,
            source: PortSource::Const {
                dtype: DType::U32,
                shape,
                data: values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            },
        }
    }

    /// A minimal greedy-argmax pass: embed a seeded `tok`, epilogue argmaxes the
    /// logits and publishes the token to a host-read `out`.
    fn greedy(vocab: u32) -> TraceContainer {
        let ops = vec![
            Op::IntrinsicVal {
                intr: IntrinsicId::Logits,
                shape: Shape::matrix(1, vocab),
                dtype: DType::F32,
            },
            Op::Reshape {
                value: 0,
                shape: Shape::vector(vocab),
            },
            Op::ReduceArgmax(1),
            Op::Reshape {
                value: 2,
                shape: Shape::vector(1),
            },
            Op::ChanPut { chan: 1, value: 3 },
        ];
        TraceContainer {
            names: vec![],
            externs: vec![],
            channels: vec![
                chan(Shape::vector(1), DType::I32, HostRole::None, true), // 0 tok
                chan(Shape::vector(1), DType::I32, HostRole::Reader, false), // 1 out
            ],
            ports: vec![
                PortBinding {
                    port: Port::EmbedTokens,
                    source: PortSource::Channel(0),
                },
                PortBinding {
                    port: Port::EmbedIndptr,
                    source: PortSource::Const {
                        dtype: DType::U32,
                        shape: Shape::vector(2),
                        data: [0u32, 1].iter().flat_map(|w| w.to_le_bytes()).collect(),
                    },
                },
                u32_port(Port::Positions, Shape::vector(1), &[0]),
                u32_port(Port::Pages, Shape::vector(1), &[0]),
                u32_port(Port::PageIndptr, Shape::vector(2), &[0, 1]),
                PortBinding {
                    port: Port::KvLen,
                    source: PortSource::Const {
                        dtype: DType::U32,
                        shape: Shape::vector(1),
                        data: 1u32.to_le_bytes().to_vec(),
                    },
                },
                u32_port(Port::WSlot, Shape::vector(1), &[0]),
                u32_port(Port::WOff, Shape::vector(1), &[0]),
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops,
            }],
        }
    }

    /// The greedy pass plus a Quest attention tap: per layer, score this
    /// request's pages with `envelope_dot(query)` and publish the scores to a
    /// host-read channel. Deliberately stops short of the `attn_page_mask`
    /// sink — that consumer is not executable yet, and the point here is the
    /// bind-time availability of the KERNEL.
    fn quest_tap(vocab: u32, pages: u32) -> TraceContainer {
        let mut container = greedy(vocab);
        container.names = vec!["envelope_dot".to_string()];
        container.channels.push(chan(
            Shape::vector(pages),
            DType::F32,
            HostRole::Reader,
            false,
        ));
        container.stages.insert(
            0,
            StageProgram {
                stage: Stage::OnAttnProj,
                ops: vec![
                    Op::IntrinsicVal {
                        intr: IntrinsicId::Query,
                        shape: Shape::vector(pages),
                        dtype: DType::F32,
                    },
                    Op::KernelCall {
                        name: 0,
                        args: vec![0],
                        shape: Shape::vector(pages),
                        dtype: DType::F32,
                    },
                    Op::ChanPut { chan: 2, value: 1 },
                ],
            },
        );
        container
    }

    #[test]
    fn quest_tap_binds_only_against_a_backend_with_kv_envelopes() {
        let caps = |has: bool| pie_model::PtirCaps {
            has_mtp_logits: false,
            has_mtp_drafts: false,
            has_value_head: false,
            has_kv_envelopes: has,
            has_attn_score: false,
            has_attn_page_mask: false,
        };
        let bytes = quest_tap(VOCAB, 4).encode();

        // Without the capability the profile advertises no kernel, and bind is
        // where that has to be caught: a program that reaches the driver
        // naming a kernel it cannot launch would fail mid-fire instead.
        let mut registry = reg(2);
        let refused = registry.register(
            bytes.clone(),
            &ModelProfile {
                vocab: VOCAB,
                ..profile_from(VOCAB, 16, 4, caps(false))
            },
        );
        assert!(
            refused.is_err(),
            "envelope_dot must be unavailable without has_kv_envelopes"
        );

        let mut registry = reg(2);
        registry
            .register(
                bytes,
                &ModelProfile {
                    vocab: VOCAB,
                    ..profile_from(VOCAB, 16, 4, caps(true))
                },
            )
            .expect("quest tap must bind against a backend with kv envelopes");
    }

    fn reg(cap: usize) -> Registry {
        Registry::new(NonZeroUsize::new(cap).unwrap())
    }

    /// A dummy profile whose `vocab` matches the container's logits shape (bind
    /// checks `logits` against the profile's vocab).
    fn prof(vocab: u32) -> ModelProfile {
        ModelProfile {
            vocab,
            ..ModelProfile::dummy()
        }
    }

    #[test]
    fn register_dedups_identical_containers() {
        let mut r = reg(8);
        let p = prof(VOCAB);
        let bytes = greedy(VOCAB).encode();
        let a = r.register(bytes.clone(), &p).unwrap();
        let b = r.register(bytes.clone(), &p).unwrap();
        assert!(
            Arc::ptr_eq(&a, &b),
            "identical containers must share one Arc (cache hit)"
        );
        assert_eq!(r.len(), 1);
        assert_eq!(a.hash, container_hash(&bytes));
        // pricing: 2 channels × [1] × 4 bytes × (1+1 ring) = 16 bytes.
        assert_eq!(a.pricing.num_channels, 2);
        assert_eq!(a.pricing.channel_bytes, 16);
        assert_eq!(a.pricing.rows, 1);
    }

    #[test]
    fn engine_profile_is_authoritative_at_program_bind() {
        let mut registry = reg(2);
        let engine_profile = ModelProfile {
            vocab: VOCAB,
            page_size: 64,
            num_layers: 7,
            has_mtp_logits: false,
            has_mtp_drafts: true,
            has_value_head: false,
            ..ModelProfile::dummy()
        };
        let program = registry
            .register(greedy(VOCAB).encode(), &engine_profile)
            .unwrap();

        assert_eq!(program.bound.profile.vocab, VOCAB);
        assert_eq!(program.bound.profile.page_size, 64);
        assert_eq!(program.bound.profile.num_layers, 7);
        assert!(!program.bound.profile.has_mtp_logits);
        assert!(program.bound.profile.has_mtp_drafts);
        assert!(!program.bound.profile.has_value_head);
    }

    #[test]
    fn kv_envelope_capability_gates_the_envelope_dot_kernel() {
        let caps = |has: bool| pie_model::PtirCaps {
            has_mtp_logits: false,
            has_mtp_drafts: false,
            has_value_head: false,
            has_kv_envelopes: has,
            has_attn_score: false,
            has_attn_page_mask: false,
        };
        // A backend that cannot honour the envelope contract must advertise NO
        // second-party kernel, so a Quest program fails to bind rather than
        // reaching a driver that would refuse it mid-fire.
        assert!(profile_from(VOCAB, 16, 4, caps(false)).kernels.is_empty());

        let advertised = profile_from(VOCAB, 16, 4, caps(true));
        let kernel = advertised
            .kernel("envelope_dot")
            .expect("envelope_dot must be advertised when the backend has envelopes");
        assert!(
            kernel.replayable,
            "envelope_dot is a pure function of the query and the page \
             envelopes, so re-running a fire must reproduce it"
        );
        assert!(
            kernel.sink_scope.is_none(),
            "envelope_dot produces a value; it consumes no attention effect"
        );
    }

    #[test]
    fn pricing_reads_rows_from_channel_embed_indptr() {
        let mut container = greedy(VOCAB);
        container
            .channels
            .push(chan(Shape::vector(3), DType::U32, HostRole::None, true));
        container
            .ports
            .iter_mut()
            .find(|binding| binding.port == Port::EmbedIndptr)
            .unwrap()
            .source = PortSource::Channel(2);
        assert_eq!(price(&container).rows, 2);
    }

    #[test]
    fn distinct_containers_are_separate() {
        let mut r = reg(8);
        let a = r.register(greedy(8).encode(), &prof(8)).unwrap();
        let b = r.register(greedy(16).encode(), &prof(16)).unwrap();
        assert_ne!(a.hash, b.hash);
        assert!(!Arc::ptr_eq(&a, &b));
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn hash_hit_requires_identical_canonical_bytes() {
        let mut registry = reg(8);
        let bytes = greedy(VOCAB).encode();
        let program = registry.register(bytes.clone(), &prof(VOCAB)).unwrap();
        registry.inner.put(
            program.hash,
            Arc::new(RegisteredProgram {
                bytes: b"collision".to_vec(),
                hash: program.hash,
                bound: program.bound.clone(),
                compiled_stages: program.compiled_stages.clone(),
                channel_accesses: program.channel_accesses.clone(),
                launch: program.launch.clone(),
                geometry_taint: program.geometry_taint.clone(),
                shadow_plan: program.shadow_plan.clone(),
                pricing: program.pricing,
                emitted: Mutex::new(HashMap::new()),
            }),
        );
        assert!(matches!(
            registry.register(bytes, &prof(VOCAB)),
            Err(RegisterError::HashCollision(_))
        ));
    }

    #[test]
    fn malformed_bytes_fail_with_decoder_message() {
        let mut r = reg(4);
        let err = r
            .register(b"not a container".to_vec(), &prof(VOCAB))
            .unwrap_err();
        assert!(
            matches!(err, RegisterError::Decode(_)),
            "bad magic → decode error"
        );
        assert!(err.to_string().contains("decode"), "{err}");
    }

    #[test]
    fn malformed_trace_fails_at_bind_with_validator_message() {
        // A structurally-decodable container whose body is ill-typed: put a
        // Bool value into an I32 channel (a ChanPut dtype mismatch).
        let ops = vec![
            Op::IntrinsicVal {
                intr: IntrinsicId::Logits,
                shape: Shape::matrix(1, VOCAB),
                dtype: DType::F32,
            },
            Op::Reshape {
                value: 0,
                shape: Shape::vector(VOCAB),
            },
            Op::Gt(1, 1),                      // Bool [VOCAB]
            Op::ChanPut { chan: 1, value: 2 }, // out is I32 [1] → dtype+shape mismatch
        ];
        let mut c = greedy(VOCAB);
        c.stages[0].ops = ops;
        let bytes = c.encode();

        let mut r = reg(4);
        let err = r.register(bytes, &prof(VOCAB)).unwrap_err();
        assert!(
            matches!(err, RegisterError::Bind(_)),
            "ill-typed body → bind error, got {err}"
        );
        assert!(err.to_string().contains("bind failed"), "{err}");
    }

    #[test]
    fn lru_evicts_beyond_capacity() {
        let mut r = reg(2);
        let hashes: Vec<u64> = [8u32, 16, 32]
            .iter()
            .map(|&v| r.register(greedy(v).encode(), &prof(v)).unwrap().hash)
            .collect();
        assert_eq!(r.len(), 2);
        assert!(r.lookup(hashes[0]).is_none(), "LRU evicts the oldest");
        assert!(r.lookup(hashes[1]).is_some());
        assert!(r.lookup(hashes[2]).is_some());
    }

    #[test]
    fn process_wide_register_is_consistent() {
        use std::thread;
        let bytes = greedy(100).encode();
        let hash = container_hash(&bytes);
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let b = bytes.clone();
                thread::spawn(move || super::register(b, &prof(100)).unwrap().hash)
            })
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap(), hash);
        }
        assert!(super::lookup(hash).is_some());
    }
}
