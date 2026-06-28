//! `#9` — system-wide compiled-program cache (host side).
//!
//! Keyed by the canonical sampler-program **bytecode hash** (FNV-1a 64, identical
//! to the CUDA driver's `jit::fnv1a64`, so the host hash equals the driver
//! `ProgramHandle`), this bounded LRU cache interns each *distinct* sampler
//! program ONCE across requests and inferlets: one encode + one metadata
//! derivation, shared by `Arc`. Because [`pie_sampling_ir::encode`] is canonical
//! (same program ⟺ same bytes), the hash is a sound identity — simultaneously the
//! cache key, the `#10` cross-request group key, and the `#8` hash-match
//! fast-path recognizer key. One mechanism.
//!
//! Pure host-side (no GPU): the driver keeps its own compile cache under the same
//! hash; this is the runtime-side counterpart (program identity + marshaling
//! metadata + bounded eviction) that `#10`/`#11` build on. The kernel-JIT half
//! (GPU, NVRTC) is delta's, keyed by the same hash.
//!
//! The pattern mirrors `inference::structured::compiled_grammar`'s `get_or_compile`
//! cache: a `LazyLock<Mutex<LruCache<_, Arc<_>>>>` global, hit returns a clone,
//! miss builds and inserts.

use std::num::NonZeroUsize;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use lru::LruCache;
use pie_sampling_ir::{OutputKind, Readiness, SamplingProgram, ValidationError};

/// FNV-1a 64-bit program-identity hash — re-exported from `pie-sampling-ir` (the
/// single canonical impl beside `encode`, byte-identical to the driver's
/// `jit::fnv1a64` ⇒ host hash == driver `ProgramHandle`).
pub use pie_sampling_ir::program_hash;

/// The interned, immutable artifact for one distinct sampler program.
#[derive(Debug)]
pub struct CachedProgram {
    /// Canonical L0 bytecode (the C++ driver contract).
    pub bytecode: Vec<u8>,
    /// `program_hash(bytecode)` == driver `ProgramHandle` == `#10` group key.
    pub hash: u64,
    /// Per-output marshaling kinds, in declared order (host response routing).
    pub output_kinds: Vec<OutputKind>,
    /// Per-output element count (`ValueType.shape.numel()`), in declared order.
    /// `1` for a scalar/single-`Token`, `k` for a `[k]`-Token / `[k]` vector
    /// output. Drives the shape-aware marshal (`[k]`-Token routing) and the
    /// fast-path eligibility gate (single-`[1]`-Token only).
    pub output_elem_counts: Vec<u32>,
    /// Declared input-slot count (binding-arity check at attach).
    pub num_inputs: usize,
    /// Per-input-slot readiness (`Submit`/`Late`), in `Op::Input(i)` order, from
    /// the program's `InputDecl.ready`. The host gather routes a bound `tensor`
    /// per slot: `Submit` → `sampling_input_*` (gathered now), `Late` → the
    /// device-alias `sampling_late_*` channel (#27 cut #2). Defaults all `Submit`.
    pub input_readiness: Vec<Readiness>,
}

/// Default bound: high-concurrency unique-sampler churn (`#11`) must not grow the
/// cache without limit. Sampler programs are tiny, so this is generous.
pub const DEFAULT_CAPACITY: usize = 256;

/// A bounded LRU of interned programs, keyed by program hash. Held behind a
/// `Mutex` in the process-wide cache; constructed directly in tests.
pub struct ProgramCache {
    inner: LruCache<u64, Arc<CachedProgram>>,
}

impl ProgramCache {
    pub fn new(capacity: NonZeroUsize) -> Self {
        Self { inner: LruCache::new(capacity) }
    }

    /// Intern `program` (already validated by decode) given its canonical
    /// `bytecode`: returns the shared [`CachedProgram`], deriving the marshaling
    /// metadata only on a cache miss. Identical programs share one `Arc`.
    pub fn intern(
        &mut self,
        program: &SamplingProgram,
        bytecode: Vec<u8>,
    ) -> Result<Arc<CachedProgram>, ValidationError> {
        let hash = program_hash(&bytecode);
        if let Some(hit) = self.inner.get(&hash) {
            return Ok(hit.clone());
        }
        // Cold path only: derive the per-output marshaling kinds (re-validates,
        // but `program` is already decode-validated so this cannot fail here).
        let output_kinds = pie_sampling_ir::output_kinds(program)?;
        let output_elem_counts = pie_sampling_ir::output_types(program)?
            .iter()
            .map(|t| t.shape.numel() as u32)
            .collect();
        let entry = Arc::new(CachedProgram {
            bytecode,
            hash,
            output_kinds,
            output_elem_counts,
            num_inputs: program.inputs.len(),
            input_readiness: program.inputs.iter().map(|i| i.ready).collect(),
        });
        self.inner.put(hash, entry.clone());
        Ok(entry)
    }

    /// Fast-path probe by hash (the `#8` recognizer / `#10` group lookup). `None`
    /// if never interned or since evicted. Counts as a use (bumps LRU recency).
    pub fn lookup(&mut self, hash: u64) -> Option<Arc<CachedProgram>> {
        self.inner.get(&hash).cloned()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// The process-wide program cache.
static GLOBAL: LazyLock<Mutex<ProgramCache>> = LazyLock::new(|| {
    Mutex::new(ProgramCache::new(
        NonZeroUsize::new(DEFAULT_CAPACITY).expect("nonzero capacity"),
    ))
});

fn global() -> MutexGuard<'static, ProgramCache> {
    GLOBAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Intern a validated program into the process-wide cache, returning the shared
/// [`CachedProgram`]. See [`ProgramCache::intern`].
pub fn intern(
    program: &SamplingProgram,
    bytecode: Vec<u8>,
) -> Result<Arc<CachedProgram>, ValidationError> {
    global().intern(program, bytecode)
}

/// Probe the process-wide cache by program hash. See [`ProgramCache::lookup`].
pub fn lookup(hash: u64) -> Option<Arc<CachedProgram>> {
    global().lookup(hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pie_sampling_ir::{DType, InputDecl, Op, OutputDecl, Shape};

    fn argmax(vocab: u32) -> SamplingProgram {
        SamplingProgram {
            inputs: vec![InputDecl::new(Shape::vector(vocab), DType::F32)],
            ops: vec![Op::Input(0), Op::ReduceArgmax(0)],
            outputs: vec![OutputDecl::new(1, OutputKind::Token)],
        }
    }

    fn cache(cap: usize) -> ProgramCache {
        ProgramCache::new(NonZeroUsize::new(cap).unwrap())
    }

    fn enc(p: &SamplingProgram) -> Vec<u8> {
        pie_sampling_ir::encode(p)
    }

    #[test]
    fn intern_dedups_identical_programs() {
        let mut c = cache(8);
        let p = argmax(32);
        let a = c.intern(&p, enc(&p)).unwrap();
        let b = c.intern(&p, enc(&p)).unwrap();
        assert!(Arc::ptr_eq(&a, &b), "identical programs must share one Arc");
        assert_eq!(c.len(), 1);
        assert_eq!(a.hash, program_hash(&a.bytecode));
        assert_eq!(a.output_kinds, vec![OutputKind::Token]);
        assert_eq!(a.num_inputs, 1);
    }

    #[test]
    fn intern_distinct_programs_are_separate() {
        let mut c = cache(8);
        let (p, q) = (argmax(32), argmax(64)); // different vocab ⇒ different bytecode
        let a = c.intern(&p, enc(&p)).unwrap();
        let b = c.intern(&q, enc(&q)).unwrap();
        assert_ne!(a.hash, b.hash);
        assert!(!Arc::ptr_eq(&a, &b));
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn lru_evicts_beyond_capacity() {
        let mut c = cache(2);
        let progs = [argmax(8), argmax(16), argmax(32)];
        let hashes: Vec<u64> =
            progs.iter().map(|p| c.intern(p, enc(p)).unwrap().hash).collect();
        assert_eq!(c.len(), 2, "capacity 2 bounds the cache");
        // The least-recently-used (first) is evicted; the two newest are retained.
        assert!(c.lookup(hashes[0]).is_none());
        assert!(c.lookup(hashes[1]).is_some());
        assert!(c.lookup(hashes[2]).is_some());
    }

    #[test]
    fn lookup_miss_is_none() {
        let mut c = cache(4);
        assert!(c.lookup(0xdead_beef).is_none());
    }

    #[test]
    fn concurrent_intern_is_consistent() {
        use std::thread;
        // The PROCESS-WIDE cache: many threads interning the same program all
        // succeed and agree on the hash, and the cache ends with one live entry.
        let p = Arc::new(argmax(100));
        let bc = enc(&p);
        let hash = program_hash(&bc);
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let p = p.clone();
                let bc = bc.clone();
                thread::spawn(move || super::intern(&p, bc).unwrap().hash)
            })
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap(), hash);
        }
        assert!(super::lookup(hash).is_some());
    }
}
