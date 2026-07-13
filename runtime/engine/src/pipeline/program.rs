//! Program registry (thrust-3 P2.2/P2.3) — the host-side "register a traced
//! pass once, cache by identity" counterpart to the inferlet program cache.
//!
//! The wire artifact is the **canonical container bytes** (the `pie_ptir`
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
//! hash; the host→driver ship (container bytes + the PTIB `BoundTrace` sidecar)
//! rides the request path separately.
//!
//! [`model_profile`] builds the bind-time [`ModelProfile`] from the loaded
//! model: program-registration input, not fire-time glue.
//!
//! Complete pipeline domain API: some methods here (relaxed geometry
//! variants, per-channel introspection, the pure `instantiate`/registry
//! probe entry points, device-geometry lease internals) are not yet
//! called by the current single-model/mock-driver fire path, but are
//! exercised by this module's own unit tests and reserved for upcoming
//! wiring (multi-pass channels, device-geometry beams) — kept rather
//! than deleted, allowed rather than silently masked.
#![allow(dead_code)]

use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Arc;

use lru::LruCache;
use pie_ptir::compiler::{CompiledStage, StageSignature};
use pie_ptir::container::{self, ContainerDecodeError, PortSource, TraceContainer};
use pie_ptir::container_hash;
use pie_ptir::op::Op;
use pie_ptir::registry::{ModelProfile, Port};
use pie_ptir::validate::{BoundTrace, ValidateError, bind};

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
    /// Number of stage programs.
    pub num_stages: usize,
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
    /// Compiler-owned normalized stages, signatures, and region partitions.
    pub compiled_stages: Vec<CompiledStage>,
    /// Dense-channel `(consume, publish)` mask, derived once from immutable IR.
    pub channel_accesses: Vec<(bool, bool)>,
    /// The PTIB typed sidecar (`encode_bound(&bound)`) — the wire form of
    /// `BoundTrace` shipped beside the container bytes to the driver
    /// (seed-independent, hash-keyed; its inner `container_hash` == [`Self::hash`],
    /// which the driver asserts). Charlie's `bound.hpp` reads exactly this.
    pub sidecar: Vec<u8>,
    /// Registration-time pricing.
    pub pricing: Pricing,
}

impl RegisteredProgram {
    pub fn stage_signature(&self, stage: pie_ptir::registry::Stage) -> Option<&StageSignature> {
        self.compiled_stages
            .iter()
            .find(|compiled| compiled.normalized.stage == stage)
            .map(|compiled| &compiled.signature)
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
        let compiled_stages = pie_ptir::compiler::compile_bound(&bound);
        if std::env::var_os("PIE_PTIR_DUMP_PLAN").is_some() {
            for stage in &compiled_stages {
                eprintln!("{}", pie_ptir::compiler::debug_stage_plan(stage));
                eprintln!("  metrics={:?}", stage.metrics());
            }
        }
        // The PTIB sidecar is the host→driver wire form of `BoundTrace`
        // (seed-independent, hash-keyed) — computed once, cached beside pricing.
        let sidecar = pie_ptir::sidecar::encode_bound_with_plans(&bound, &compiled_stages);
        let entry = Arc::new(RegisteredProgram {
            bytes,
            hash,
            bound,
            compiled_stages,
            channel_accesses,
            sidecar,
            pricing,
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

    pub fn len(&self) -> usize {
        self.inner.len()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
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
        .find(|p| p.port == pie_ptir::registry::Port::EmbedIndptr)
        .and_then(|p| match &p.source {
            container::PortSource::Const { shape, .. } => {
                Some((shape.numel() as u32).saturating_sub(1).max(1))
            }
            container::PortSource::Channel(_) => None,
        })
        .unwrap_or(1);
    Pricing {
        channel_bytes,
        num_channels: c.channels.len(),
        num_stages: c.stages.len(),
        rows,
    }
}

// ---------------------------------------------------------------------------
// Process-wide registry
// ---------------------------------------------------------------------------

use std::sync::{LazyLock, Mutex, MutexGuard};

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

/// Probe the process-wide registry by identity hash.
pub fn lookup(hash: u64) -> Option<Arc<RegisteredProgram>> {
    global().lookup(hash)
}

/// Build the bind-time [`ModelProfile`] from the loaded model (P2b: vocab +
/// page-size + layer caps; model-gated intrinsics + second-party kernels default
/// conservative until the model surfaces them).
pub fn model_profile() -> ModelProfile {
    let m = pie_model::model();
    let ptir = m.ptir_caps();
    ModelProfile {
        vocab: m.vocab_size(),
        page_size: crate::store::registry::get(0, 0).kv_page_size,
        num_layers: 1,
        activation: pie_ptir::types::DType::F32,
        has_mtp_logits: ptir.has_mtp_logits,
        has_mtp_drafts: ptir.has_mtp_drafts,
        has_value_head: ptir.has_value_head,
        kernels: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pie_ptir::container::{
        ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram,
    };
    use pie_ptir::op::{IntrinsicId, Op};
    use pie_ptir::registry::{Port, Stage};
    use pie_ptir::types::{DType, Shape};

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
            ],
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops,
            }],
        }
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
    fn sidecar_roundtrips_and_matches_identity() {
        // The PTIB sidecar is the host→driver ship artifact; its inner
        // container_hash must equal the program identity (the driver asserts it).
        let mut r = reg(4);
        let bytes = greedy(VOCAB).encode();
        let prog = r.register(bytes.clone(), &prof(VOCAB)).unwrap();
        let decoded = pie_ptir::sidecar::decode_bound(&prog.sidecar).unwrap();
        assert_eq!(
            decoded.container_hash, prog.hash,
            "PTIB inner hash == container identity"
        );
        assert_eq!(decoded.container_hash, container_hash(&bytes));
        assert!(!prog.sidecar.is_empty());
        assert_eq!(decoded.stage_plans.len(), prog.compiled_stages.len());
        let signature = prog
            .stage_signature(Stage::Epilogue)
            .expect("epilogue signature");
        let plan = pie_ptir::compiler::decode_plan_header(&decoded.stage_plans[0].1).unwrap();
        assert_eq!(plan.signature_hash, signature.hash);
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
                sidecar: program.sidecar.clone(),
                pricing: program.pricing,
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
