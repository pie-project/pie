//! Behavioural parity with the C++ live `KvCache` — gate-kvcache-live.
//!
//! The oracle in `tests/oracle/kv_cache_live/` compiles the real
//! `store/kv_cache.cpp` (the layout oracle's exact tree) and drives the
//! LIVE object: layer views, accessors, page buffers, envelope seeding,
//! elastic forwarding. This test replays the same nine cases against
//! [`KvCache`] materialised from the already-proven [`KvCacheLayout`], and
//! requires the transcripts to be byte-identical.
//!
//! `tests/oracle/kv_cache_live/run.sh` can no longer be run — its inputs were deleted, see `oracle_census.rs`. It is kept as the description of how this golden was taken, which is read but not re-derived. It once regenerated
//! [`GOLDEN_FNV1A64`]. The pinned value is the **C++'s** hash.
//!
//! The oracle sweeps `PIE_CUDA_KV_ENVELOPES` with `setenv` because the C++
//! reads it inside `allocate`; the port takes the flag as a parameter (the
//! layout's `envelopes` argument), so this side passes the bool the env
//! var would have produced — the env parsing itself is pinned by the
//! layout parity suite.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::rc::Rc;

use driver_cuda::dtype::DType;
use driver_cuda::layout::KvCacheFormat;
use driver_cuda::pools::kv_cache::{KvCacheLayout, PerLayer};
use driver_cuda::pools::kv_cache_live::{ElasticPool, KvCache, KvCacheDeviceOps};
use driver_cuda::tensor::TensorSpec;

/// FNV-1a 64 of the C++ oracle's transcript.
const GOLDEN_FNV1A64: u64 = 0xe101e40015506783;

/// Rows the transcript must contain, so a truncated sweep cannot pass.
const GOLDEN_ROWS: usize = 174;

const SEP: char = '\u{1f}';

/// The shared recorder, reproduced: allocations rendered exactly as the
/// C++ recorder renders them, named `t#K` in allocation order, three queues
/// drained alloc-first at every flush.
struct FakeOps {
    alloc_log: Vec<String>,
    seed_log: Vec<String>,
    arena_log: Rc<RefCell<Vec<String>>>,
    names: HashMap<usize, String>,
    next_addr: usize,
    named: usize,
}

impl FakeOps {
    fn new() -> Self {
        Self {
            alloc_log: Vec::new(),
            seed_log: Vec::new(),
            arena_log: Rc::new(RefCell::new(Vec::new())),
            names: HashMap::new(),
            next_addr: 0x1000,
            named: 0,
        }
    }

    fn sym(&self, p: *mut c_void) -> String {
        if p.is_null() {
            return "null".into();
        }
        self.names
            .get(&(p as usize))
            .cloned()
            .unwrap_or_else(|| "unknown".into())
    }
}

impl KvCacheDeviceOps for FakeOps {
    fn alloc_tensor(&mut self, dtype: DType, shape: &[i64]) -> *mut c_void {
        let spec = TensorSpec::new(dtype, shape.to_vec()).expect("oracle shapes are valid");
        let dims = shape
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        self.alloc_log
            .push(format!("{}[{dims}]={}", dtype.name(), spec.nbytes()));
        if spec.nbytes() == 0 {
            return std::ptr::null_mut();
        }
        let p = self.next_addr;
        self.next_addr += 0x1000;
        self.names.insert(p, format!("t#{}", self.named));
        self.named += 1;
        p as *mut c_void
    }

    fn escape_arena(&mut self) {
        self.alloc_log.push("bind(default)".into());
    }

    fn restore_arena(&mut self) {
        self.alloc_log.push("bind(default)".into());
    }

    fn envelope_seed(
        &mut self,
        env_min: *mut u16,
        env_max: *mut u16,
        num_pages: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) {
        let p = |x: *mut u16| if x.is_null() { "null" } else { "p" };
        self.seed_log.push(format!(
            "seed(min={},max={},pages={num_pages},kvh={num_kv_heads},hd={head_dim})",
            p(env_min),
            p(env_max),
        ));
    }

    fn stream_synchronize(&mut self) {}
}

/// The elastic stub, reproduced — including its 4096-bytes-per-page model
/// of `committed_bytes`, which is what makes the clamp visible as a number.
struct FakeElastic {
    log: Rc<RefCell<Vec<String>>>,
    committed: usize,
}

const STUB_BYTES_PER_PAGE: usize = 4096;

impl ElasticPool for FakeElastic {
    fn ensure_fraction(&mut self, used: usize, capacity: usize) {
        self.log
            .borrow_mut()
            .push(format!("ensure({used}/{capacity})"));
        self.committed = if capacity == 0 {
            0
        } else {
            used * STUB_BYTES_PER_PAGE
        };
    }
    fn trim_fraction(&mut self, used: usize, capacity: usize) {
        self.log
            .borrow_mut()
            .push(format!("trim({used}/{capacity})"));
        self.committed = if capacity == 0 {
            0
        } else {
            used * STUB_BYTES_PER_PAGE
        };
    }
    fn committed_bytes(&self) -> usize {
        self.committed
    }
}

struct Harness {
    out: String,
    case: String,
}

type Cache = KvCache<FakeElastic>;

impl Harness {
    fn begin_case(&mut self, ops: &mut FakeOps, name: &str) {
        *ops = FakeOps::new();
        self.case = name.to_string();
        self.row("case-begin".into());
    }

    fn row(&mut self, body: String) {
        self.out.push_str(&self.case);
        self.out.push(SEP);
        self.out.push_str(&body);
        self.out.push('\n');
    }

    fn flush(&mut self, ops: &mut FakeOps) {
        for r in ops.alloc_log.drain(..).collect::<Vec<_>>() {
            self.row(format!("alloc{SEP}{r}"));
        }
        for r in ops.seed_log.drain(..).collect::<Vec<_>>() {
            self.row(format!("seed{SEP}{r}"));
        }
        for r in ops.arena_log.borrow_mut().drain(..).collect::<Vec<_>>() {
            self.row(format!("arena{SEP}{r}"));
        }
    }

    fn walk(&mut self, ops: &mut FakeOps, c: &Cache) {
        let l = c.layout();
        self.row(format!(
            "scalars{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}",
            l.num_layers(),
            l.num_pages(),
            l.page_size(),
            scalar_num_kv_heads(l),
            scalar_head_dim(l),
            c.format().name(),
            u8::from(l.page_order() == driver_cuda::pools::kv_cache::PageOrder::Hnd),
            u8::from(c.envelopes_enabled()),
        ));
        for layer in 0..l.num_layers() {
            let v = c.layer_view(layer);
            self.row(format!(
                "view{SEP}L{layer}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}\
                 {SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}\
                 {SEP}{}{SEP}{}",
                v.source_layer,
                v.num_pages,
                v.page_size,
                v.num_kv_heads,
                v.head_dim,
                v.scheme as u8,
                v.storage_dtype as u8,
                v.block_size,
                ops.sym(v.k_pages),
                ops.sym(v.v_pages),
                ops.sym(v.k_scales),
                ops.sym(v.v_scales),
                ops.sym(v.k_bf16_pages),
                ops.sym(v.v_bf16_pages),
                ops.sym(v.k_env_min.cast()),
                ops.sym(v.k_env_max.cast()),
                u8::from(v.hnd_layout),
                u8::from(v.native_bf16),
            ));
            self.row(format!(
                "acc{SEP}L{layer}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}",
                ops.sym(c.k(layer)),
                ops.sym(c.v(layer)),
                ops.sym(c.k_scale(layer)),
                ops.sym(c.v_scale(layer)),
                ops.sym(c.k_for_attention(layer)),
                ops.sym(c.v_for_attention(layer)),
                l.head_dim_at(layer),
                l.num_kv_heads_at(layer),
            ));
            let pb = c
                .page_buffers(layer)
                .iter()
                .map(|b| format!("{SEP}{}:{}", ops.sym(b.data), b.page_bytes))
                .collect::<String>();
            self.row(format!("pb{SEP}L{layer}{pb}"));
        }
        self.flush(ops);
    }
}

/// The C++ scalar accessors `num_kv_heads()` / `head_dim()` — the
/// constructor arguments, NOT the per-layer view. The Rust layout keeps
/// them private, but they are recoverable: `head_dim_at`/`num_kv_heads_at`
/// on an out-of-range index fall back to exactly these scalars.
fn scalar_head_dim(l: &KvCacheLayout) -> i32 {
    l.head_dim_at(i32::MAX)
}
fn scalar_num_kv_heads(l: &KvCacheLayout) -> i32 {
    l.num_kv_heads_at(i32::MAX)
}

fn bf16() -> KvCacheFormat {
    KvCacheFormat::from_name("bf16").unwrap()
}

fn transcript() -> String {
    let mut h = Harness {
        out: String::new(),
        case: String::new(),
    };
    let mut ops = FakeOps::new();

    // a. Homogeneous native bf16, envelopes off.
    h.begin_case(&mut ops, "a-hom-bf16");
    {
        let l = KvCacheLayout::plan(3, 4, 8, 2, 16, bf16(), false).unwrap();
        let c = Cache::materialize(l, &mut ops).unwrap();
        h.flush(&mut ops);
        h.walk(&mut ops, &c);
    }

    // b. Same stack with envelopes.
    h.begin_case(&mut ops, "b-hom-bf16-env");
    {
        let l = KvCacheLayout::plan(3, 4, 8, 2, 16, bf16(), true).unwrap();
        let c = Cache::materialize(l, &mut ops).unwrap();
        h.flush(&mut ops);
        h.walk(&mut ops, &c);
    }

    // c. Per-layer stack with KV sharing (the gemma-4 shape).
    h.begin_case(&mut ops, "c-per-layer-env");
    {
        let per = PerLayer {
            head_dim: vec![32, 32, 64, 64],
            kv_source_layer: vec![0, 0, 2, 2],
            num_kv_heads: vec![4, 4, 2, 2],
        };
        let l = KvCacheLayout::plan_per_layer(4, 3, 4, 4, per, bf16(), true).unwrap();
        let c = Cache::materialize(l, &mut ops).unwrap();
        h.flush(&mut ops);
        h.walk(&mut ops, &c);
    }

    // c2. Alias entries that DISAGREE with their source's — what separates
    //     a view reading the source's dims from one reading the layer's.
    h.begin_case(&mut ops, "c2-alias-dims-differ");
    {
        let per = PerLayer {
            head_dim: vec![32, 80, 64, 96],
            kv_source_layer: vec![0, 0, 2, 2],
            num_kv_heads: vec![4, 8, 2, 6],
        };
        let l = KvCacheLayout::plan_per_layer(4, 2, 4, 4, per, bf16(), true).unwrap();
        let c = Cache::materialize(l, &mut ops).unwrap();
        h.flush(&mut ops);
        h.walk(&mut ops, &c);
    }

    // d. A scaled quantized format; the envelope request is ignored.
    h.begin_case(&mut ops, "d-int8-scales-env-skipped");
    {
        let f = KvCacheFormat::from_name("int8_per_token_head").unwrap();
        let l = KvCacheLayout::plan(2, 3, 4, 2, 16, f, true).unwrap();
        let c = Cache::materialize(l, &mut ops).unwrap();
        h.flush(&mut ops);
        h.walk(&mut ops, &c);
    }

    // e. The FP4 block format.
    h.begin_case(&mut ops, "e-nvfp4");
    {
        let f = KvCacheFormat::from_name("nvfp4").unwrap();
        let l = KvCacheLayout::plan(1, 2, 4, 2, 32, f, false).unwrap();
        let c = Cache::materialize(l, &mut ops).unwrap();
        h.flush(&mut ops);
        h.walk(&mut ops, &c);
    }

    // f. Zero pages with envelopes requested.
    h.begin_case(&mut ops, "f-zero-pages-env");
    {
        let l = KvCacheLayout::plan(2, 0, 8, 2, 16, bf16(), true).unwrap();
        let c = Cache::materialize(l, &mut ops).unwrap();
        h.flush(&mut ops);
        h.walk(&mut ops, &c);
    }

    // g. The elastic forwarding.
    h.begin_case(&mut ops, "g-elastic");
    {
        let l = KvCacheLayout::plan(1, 10, 8, 2, 16, bf16(), false).unwrap();
        let mut c = Cache::materialize(l, &mut ops).unwrap();
        h.flush(&mut ops);
        h.row(format!("call{SEP}ensure-before-set(3)"));
        c.ensure_pages(3);
        h.row(format!("committed{SEP}{}", c.committed_bytes()));
        c.set_elastic_allocator(Some(FakeElastic {
            log: Rc::clone(&ops.arena_log),
            committed: 0,
        }));
        for pages in [-5, 0, 2, 99] {
            h.row(format!("call{SEP}ensure({pages})"));
            c.ensure_pages(pages);
            h.flush(&mut ops);
            h.row(format!("committed{SEP}{}", c.committed_bytes()));
        }
        h.row(format!("call{SEP}trim(4)"));
        c.trim_pages(4);
        h.flush(&mut ops);
        h.row(format!("committed{SEP}{}", c.committed_bytes()));
    }

    // h. `enable_envelopes` after the fact.
    h.begin_case(&mut ops, "h-enable-envelopes");
    {
        let l = KvCacheLayout::plan(1, 2, 4, 2, 16, bf16(), true).unwrap();
        let on = Cache::materialize(l, &mut ops).unwrap();
        h.flush(&mut ops);
        h.row(format!(
            "enable-when-on{SEP}{}",
            if on.enable_envelopes().is_ok() {
                "ok"
            } else {
                "threw"
            }
        ));
        let l = KvCacheLayout::plan(1, 2, 4, 2, 16, bf16(), false).unwrap();
        let off = Cache::materialize(l, &mut ops).unwrap();
        h.flush(&mut ops);
        h.row(format!(
            "enable-when-off{SEP}{}",
            if off.enable_envelopes().is_ok() {
                "ok"
            } else {
                "threw"
            }
        ));
    }

    h.out
}

fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[test]
fn the_port_reproduces_the_cpp_transcript() {
    let text = transcript();
    let rows = text.lines().count();
    assert_eq!(rows, GOLDEN_ROWS, "row count diverged — case shape changed");
    let hash = fnv1a64(text.as_bytes());
    if hash != GOLDEN_FNV1A64 {
        let path = std::env::temp_dir().join("kv_cache_live_rust_transcript.txt");
        std::fs::write(&path, &text).ok();
        panic!(
            "transcript hash 0x{hash:016x} != golden 0x{GOLDEN_FNV1A64:016x}; \
             rust transcript dumped to {}",
            path.display()
        );
    }
}
