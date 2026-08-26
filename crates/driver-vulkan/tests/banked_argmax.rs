//! **THE WHOLE CHAIN, ON A REAL CARD: a cached checkpoint through this driver
//! to an argmax, against the answer cuda banked.**
//!
//! `crates/driver-cuda/tests/banked_argmaxes.rs` is what this is measured
//! against and its header is the history: three checkpoints, one token in, an
//! argmax and a logit banked from the first end-to-end fire each SKU ever
//! managed. `driver-wgpu/tests/banked_argmax.rs` is the sibling on the other
//! shader plane and answers two of the three.
//!
//! # Why gpt-oss and not the small one
//!
//! `qwen35-d0.8b` is a third the size and eight times faster to load, and it
//! is not the row this plane can answer first. It is a HYBRID: eighteen of
//! its twenty-four layers are gated DeltaNet and want a recurrent slab each,
//! and **`driver-vulkan` allocates no recurrent slab anywhere** —
//! `resources` has a `Pool` and no `RecurrentPool`, and `Pools::slab` has
//! answered `None` for every layer since it was written. Standing a fixture
//! up for one would mean this test inventing a layout that no allocator on
//! this plane owns, and a layout invented by its own test is a layout that
//! agrees with itself.
//!
//! `gptoss-20b-bf16-mxfp4-kv-bf16` needs none: twenty-four layers of
//! attention, all one width, no recurrence at all. What it needs instead is
//! everything else this plane had never been asked for — a sink on every
//! layer, alternating sliding and full attention, mxfp4 experts, and twelve
//! gigabytes of weights against an adapter that binds four.
//!
//! # What this asks that nothing else in this crate does
//!
//! `tests/the_walk_is_the_program.rs` asks whether the walk visits the right
//! steps with a recorder standing where a device would. `tests/device.rs`
//! takes ONE `norm.rmsnorm` through the same `serve::run` this uses.
//! `tests/device_{fire,gemm,ssm,sink}.rs` each ask whether a family computes
//! the right numbers on the card. Every one of those is one link. This is the
//! chain:
//!
//! > weights produced → uploaded → pools allocated → a `Program` walked →
//! > every statement fired → logits read → argmax compared.
//!
//! # What walking a whole tower found here
//!
//! **This driver dropped every `InOut` copy on the floor.** The walk has
//! recorded them since it was written (`walk::fire::Fire::inout`): a point
//! that reads an operand and writes a result gets two rectangles whenever
//! `model_compiler::program::carve` gives the result its own slot, and the
//! operand's bytes have to be moved into it first. `serve::run` took
//! `dispatches` and not `blits`, so the kernel read whatever that slot held.
//!
//! Nothing in the tree could see it, and the reason is exact: `serve::run` is
//! this plane's device half and **had never been called with a real plan.**
//! Its one caller fires a `norm.rmsnorm`, which is not an `InOut` point. This
//! fire states 240 of them.
//!
//! # Why the answer is falsifiable
//!
//! Because the comparison is against a number this tree did not compute. A
//! self-consistency check — this plane against itself, or against an f64
//! model of what its shaders do — would pass for a tower that agreed with
//! itself and disagreed with the model. 11 at 14.4375 came off a CUDA card
//! through `kernels-cuda`, and the only thing the two planes share is the
//! model TEXT.
//!
//! `PHYSICAL_PAGE` is the other half: the fire's one token is written to and
//! read from PHYSICAL PAGE 3, not page 0, so a driver that bound the pool and
//! ignored the translation planes would attend over zeros.
//!
//! # The skip
//!
//! This plane has no `driver_wgpu::skip`, so the convention is
//! `tests/device_gemm.rs`'s: print the word and return. It is stated here
//! rather than borrowed, because the two halves are different questions and a
//! reader should see which one was absent — a CARD, a SHADER TREE, or a
//! CHECKPOINT.

#![cfg(all(feature = "device", feature = "native"))]
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard, OnceLock};

use driver_vulkan::baker::marks::{BufferId, Slice};
use driver_vulkan::baker::stage::{FireTable, KvGeometry, Pools, Slab};
use driver_vulkan::baker::walk::{Extent, Fire};
use driver_vulkan::baker::{Baked, Bank, Vulkan, arenas_of, encode::Encoder, join, readable_base};
use driver_vulkan::device::{Buffer, Device, Pipelines};
use driver_vulkan::resources::{Pool, Recurrent, RecurrentPool, Shape};
use driver_vulkan::serve::Embedded;
use kernels_vulkan::Capability;
use model_compiler::program::{Dt, Program};
use model_ir::plan::{CacheRow, FireClass, Plan};

// ── the banked answer ──────────────────────────────────────────────────

/// The catalog row.
/// One row of `driver-cuda`'s `BANKED`, which is the table this file answers
/// against and does not extend: an entry here is a claim CUDA published.
struct Banked {
    /// The catalog row.
    sku: &'static str,
    /// Where its snapshot sits under `~/.cache/huggingface/hub`.
    cache: &'static str,
    /// The argmax token id.
    id: usize,
    /// Its logit **as rendered to four decimals**.
    logit: &'static str,
}

/// **THE PURE-ATTENTION TOWER, and it was this plane's first for a reason.**
///
/// Twenty-four layers of attention with a sink, alternating sliding and full,
/// over mxfp4 experts, and no recurrence at all — which is what let it fire
/// here before `resources::RecurrentPool` existed.
const GPTOSS: Banked = Banked {
    sku: "gptoss-20b-bf16-mxfp4-kv-bf16",
    cache: "models--openai--gpt-oss-20b",
    id: 11,
    logit: "14.4375",
};

/// **THE HYBRID.** Six of its twenty-four layers are attention and eighteen
/// are gated DeltaNet, so it wants a recurrent slab per linear layer —
/// three planes each — and this plane allocated NONE until
/// `resources::RecurrentPool` landed beside the KV one. `Pools::slab` answered
/// `None` for every layer, so a walk over a hybrid refused at its first scan.
const QWEN35: Banked = Banked {
    sku: "qwen35-d0.8b-bf16-kv-bf16",
    cache: "models--Qwen--Qwen3.5-0.8B",
    id: 198,
    logit: "12.3125",
};

/// **THE TOWER THAT ATTENDS AT TWO WIDTHS AND SHARES ITS CACHES.**
///
/// Forty-two layers over twenty-four caches: one each for layers 0..21, then
/// `kv.22` between every later sliding layer and `kv.23` between every later
/// full one. And the two kinds do not attend at the same width — sliding at
/// `head_dim` 256, full at 512 — so a fixture that opens one pool per layer at
/// one width is wrong about this tower twice. It also carries a per-layer
/// embedding relay, which is the only reason `model_ir` has a sliced source at
/// all.
const GEMMA4: Banked = Banked {
    sku: "gemma4-e4b-bf16-kv-bf16",
    cache: "models--google--gemma-4-E4B-it",
    id: 785,
    logit: "7.5938",
};

// ── the fire's shape ───────────────────────────────────────────────────

/// **The token every banked answer was fired from.** `baker-smoke`'s default
/// prompt was the single id 785 and `banked-argmaxes.sh` never overrode it, so
/// the answers above are "one fire, one row, position zero".
const PROMPT: u32 = 785;

/// Recurrent seats. One, because this fire is one request.
const SLOTS: u32 = 1;

/// Token rows per KV page. Not a knob — cuda's `boot::KV_PAGE_SIZE`.
const PAGE: u32 = 16;

/// Pages the pool holds.
const PAGES: u32 = 4;

/// **The physical page this fire's one token lives in, and it is not zero.**
///
/// A paged cache is indirection or it is nothing: `kv_page_indices` says which
/// physical page a request's logical page zero is, and `kv_write_page` says
/// where the append lands. Both are set to this. A driver that bound the pool
/// and ignored the translation would write to slot 0 and attend over slot 0,
/// which for a one-token fire is *also* self-consistent — the append and the
/// read would agree with each other and disagree with the fixture. Putting the
/// token on page 3 of 4 makes them disagree with the ZEROS instead, which is a
/// wrong answer rather than a right one.
const PHYSICAL_PAGE: u32 = 3;

// ── the device, shared ─────────────────────────────────────────────────

static GPU: OnceLock<Option<Mutex<Device>>> = OnceLock::new();

/// A card, a compiled shader tree, or a printed skip.
///
/// This plane has two ways to have nothing to run — no device answered, and
/// the build carries no modules — and `tests/device_gemm.rs` states both. They
/// are stated here too rather than shared, because a test binary that borrowed
/// them would report the absence of the OTHER file's premise.
macro_rules! gpu {
    () => {{
        if !kernels_vulkan::embedded() {
            eprintln!(
                "skipped: built without kernels-vulkan/native, so there are no \
                 modules to build a pipeline from"
            );
            return;
        }
        let Some(device) = gpu() else {
            return;
        };
        device
    }};
}

fn gpu() -> Option<MutexGuard<'static, Device>> {
    let held = GPU.get_or_init(|| match Device::open() {
        Ok(d) => Some(Mutex::new(d)),
        Err(e) => {
            eprintln!("skipped: no device answered `Device::open`: {e}");
            None
        }
    });
    held.as_ref()
        .map(|m| m.lock().unwrap_or_else(std::sync::PoisonError::into_inner))
}

// ── what this fire stages beside its arena ─────────────────────────────

/// The driver's answer to [`Pools`], holding regions and nothing else.
///
/// Every method turns an allocation into a plain [`Slice`]; the executor reads
/// regions and mints handles, and which [`Buffer`] is behind one is this
/// struct's business alone.
struct Live {
    /// Per layer, the keys and values planes.
    kv: BTreeMap<u32, (Slice, Slice)>,
    /// The staged planes.
    tables: BTreeMap<FireTable, Slice>,
    /// Per gated-DeltaNet layer, `(state, conv, new_conv)`. Empty for a tower
    /// with no recurrence, which is most of them.
    slabs: BTreeMap<u32, (Slice, Slice, Slice)>,
    /// **Per layer, because a tower may attend at more than one width.**
    /// gpt-oss does not — every layer is `kv_heads: 8, head_dim: 64` — and
    /// this is a map anyway, because the trait takes a layer and a fixture
    /// that answered one geometry to every question would be right here by
    /// accident rather than by construction.
    geometry: BTreeMap<u32, KvGeometry>,
}

impl Pools for Live {
    fn kv(&self, layer: u32, values: bool) -> Option<Slice> {
        self.kv
            .get(&layer)
            .map(|(k, v)| if values { *v } else { *k })
    }

    fn slab(&self, layer: u32, which: Slab) -> Option<Slice> {
        // `None` for a layer with no slab, which is what a driver holding none
        // must answer: a scan handed a null carry answers fluently and wrongly.
        // Every layer of a pure-attention tower answers `None` here and every
        // gated-DeltaNet layer of a hybrid answers three planes.
        let (state, conv, new_conv) = self.slabs.get(&layer)?;
        Some(match which {
            Slab::State => *state,
            Slab::Conv => *conv,
            Slab::NewConv => *new_conv,
        })
    }

    fn kv_geometry(&self, layer: u32) -> KvGeometry {
        // Only asked of a layer `kv` answered for, and every one of those was
        // given a row when its pool was allocated.
        self.geometry[&layer]
    }

    fn table(&self, which: FireTable) -> Option<Slice> {
        self.tables.get(&which).copied()
    }
}

/// Everything the device holds for one load and one fire.
///
/// The buffers and the [`Live`] are built together and kept together because a
/// [`BufferId`] is an index into the first and every [`Slice`] in the second
/// carries one: separating them would be two lists that have to agree about a
/// numbering.
///
/// A `driver_vulkan::device::Buffer` is NOT reference counted — it is a raw
/// handle freed by [`Device::free`] — so the KV pool is held here rather than
/// having its buffers cloned out, and [`Loaded::buffers`] builds the borrow
/// list at fire time.
struct Loaded {
    /// The weight arenas, then the activation arena, then the staged planes.
    /// The KV pool's buffers are numbered after them by [`Loaded::buffers`].
    held: Vec<Buffer>,
    /// ONE POOL PER ATTENTION WIDTH. A `resources::Shape` states one width for
    /// every layer it holds, and `gemma4-e4b` attends at two.
    kv_pools: Vec<Pool>,
    /// The recurrent planes, when the tower has any. Held for the same reason
    /// the KV pools are: a `Buffer` here is a raw handle, not a refcount.
    recurrent: Option<RecurrentPool>,
    /// `(pool, row)` for every KV plane pair, in the order their buffers are
    /// numbered. A ROW is a CACHE and not a layer — see `kv_layers`.
    kv_rows: Vec<(usize, u16)>,
    /// Which layers own a slab, in the order their three planes are numbered
    /// — after every KV plane.
    rs_layers: Vec<u32>,
    pools: Live,
    banks: BTreeMap<String, Bank>,
    /// Which buffer the activation arena is. Not a constant: the weights take
    /// as many buffers before it as the adapter's ceiling needed.
    arena: BufferId,
    /// Bytes of the activation arena.
    arena_bytes: u64,
}

impl Loaded {
    /// One borrow per [`BufferId`], in the order the ids were handed out.
    fn buffers(&self) -> Vec<&Buffer> {
        let mut all: Vec<&Buffer> = self.held.iter().collect();
        for (pool, row) in &self.kv_rows {
            let pool = &self.kv_pools[*pool];
            all.push(pool.cache(*row, false).expect("the pool holds this row"));
            all.push(pool.cache(*row, true).expect("the pool holds this row"));
        }
        // AFTER EVERY KV PLANE, three per layer in `(state, conv, new_conv)`
        // order — which is the order `open` mints their ids in and the only
        // thing that keeps this list and those `BufferId`s the same numbering.
        if let Some(rs) = self.recurrent.as_ref() {
            for layer in &self.rs_layers {
                let at = *layer as u16;
                all.push(rs.state(at).expect("the pool holds this layer"));
                all.push(rs.conv(at).expect("the pool holds this layer"));
                all.push(rs.new_conv(at).expect("the pool holds this layer"));
            }
        }
        all
    }

    fn free(self, device: &Device) {
        for b in self.held {
            device.free(b);
        }
        for pool in self.kv_pools {
            pool.close(device);
        }
        if let Some(rs) = self.recurrent {
            rs.close(device);
        }
    }
}

/// Which layers own a KV pool, off the PLAN.
///
/// Read from `plan.caches` rather than from `Deployment::attention`, and the
/// difference is not cosmetic: `Deployment` fills the layers that state no
/// attention from the widest one that does, so its vector says something about
/// every layer of the tower and cannot say which of them actually hold pages.
/// **A LAYER TO A CACHE IS NOT ONE TO ONE**, and reading the layer out of the
/// cache row's NAME assumed it was. It works for `qwen35-d0.8b` and
/// `gptoss-20b`, whose rows are `kv.<stack layer>` — and `gemma4-e4b` SHARES:
/// its forty-two layers hold twenty-four caches, one each for layers 0..21 and
/// then two between all the rest, `kv.22` for the sliding ones and `kv.23` for
/// the full ones.
///
/// So the map comes from the STATEMENTS, which is where the association
/// actually lives: an op carries the layer it is at and the cache it names.
/// Nothing has to be parsed and sharing costs nothing to express.
fn kv_layers(plan: &Plan) -> BTreeMap<u32, String> {
    let mut kv = BTreeMap::new();
    for op in &plan.ops {
        let (Some(cache), Some(layer)) = (op.cache.as_deref(), op.layer) else {
            continue;
        };
        if plan
            .caches
            .iter()
            .any(|row| matches!(row, CacheRow::Kv { name, .. } if name == cache))
        {
            kv.insert(layer, cache.to_string());
        }
    }
    kv
}

/// Which layers own a RECURRENT slab, off the same column.
fn rs_layers(plan: &Plan) -> Vec<u32> {
    let mut rs = Vec::new();
    for row in &plan.caches {
        let CacheRow::State { name, .. } = row else {
            continue;
        };
        let Some(at) = name.rsplit('.').next().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        if !rs.contains(&at) {
            rs.push(at);
        }
    }
    rs.sort_unstable();
    rs
}

/// A staged plane, appended to the buffer table and answered as a whole region.
fn staged(device: &Device, held: &mut Vec<Buffer>, words: &[u32]) -> Slice {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let buffer = device.buffer(&bytes).expect("a fire table allocates");
    let slice = Slice::whole(BufferId(held.len() as u32), buffer.size());
    held.push(buffer);
    slice
}

/// Produce the weights, upload them, allocate the pool, stage the fire.
///
/// `None` when the checkpoint is not cached.
fn open(device: &Device, row: &Banked, baked: &Baked, program: &Program) -> Option<Loaded> {
    let snap = model::snapshot::Snapshot::open(row.cache)?;
    println!(
        "checkpoint {} — {} shard(s), {} tensors",
        snap.dir.display(),
        snap.shards(),
        snap.len()
    );

    // ── the weights ────────────────────────────────────────────────────
    let base = readable_base(row.sku).expect("the SKU offers a safetensors import");
    let import = model::import_of(row.sku, base).expect("the import table holds that flavor");
    let produced = model::produce::produce(&import, &baked.plan.params, 0, &|n| snap.read(n))
        .unwrap_or_else(|why| panic!("the checkpoint does not produce this plan: {why}"));

    // AS MANY ARENAS AS THIS ADAPTER'S CEILING NEEDS, which for this SKU is
    // several: 12.82 GiB of banks against a `budget` of four. Both halves of
    // that budget matter — an arena past `maxStorageBufferRange` allocates and
    // then cannot be bound.
    let arenas = arenas_of(&produced, device.budget())
        .unwrap_or_else(|why| panic!("the weights do not pack: {why}"));
    let total: u64 = arenas.iter().map(|a| a.bytes).sum();
    println!(
        "produced {} tensors into {} arena(s) of {:.2} GiB total (ceiling {:.2} GiB)",
        produced.len(),
        arenas.len(),
        total as f64 / (1024.0 * 1024.0 * 1024.0),
        device.budget() as f64 / (1024.0 * 1024.0 * 1024.0),
    );

    let mut held = Vec::new();
    for arena in &arenas {
        let buffer = device
            .empty(arena.bytes)
            .expect("the weight arena allocates");
        for &(i, at) in &arena.banks {
            device
                .write_at(&buffer, at, &produced[i].1.bytes)
                .expect("a bank uploads");
        }
        held.push(buffer);
    }
    assert_eq!(held.len(), arenas.len());

    // ONE CONTIGUOUS UPLOAD PER BANK, and the region each landed in is what a
    // `Const` slot binds. The repr is the PLAN's column and not the storage
    // dtype: a quantised bank's form lives only there, and `BoundOp::form`
    // reads it and nothing else.
    let repr_of = |name: &str| {
        baked
            .plan
            .params
            .iter()
            .find(|p| p.name == name)
            .map_or_else(String::new, |p| p.repr.clone())
    };
    let banks: BTreeMap<String, Bank> = arenas
        .iter()
        .enumerate()
        .flat_map(|(arena, a)| {
            let produced = &produced;
            a.banks.iter().map(move |&(i, at)| {
                let (name, t) = &produced[i];
                (
                    name.clone(),
                    Bank {
                        // THE ARENA'S INDEX *IS* THE BUFFER ID.
                        slice: Slice {
                            buffer: BufferId(arena as u32),
                            at,
                            bytes: t.bytes.len() as u64,
                        },
                        shape: t.shape.clone(),
                        dtype: t.dtype,
                        repr: repr_of(name),
                    },
                )
            })
        })
        .collect();
    join(&baked.plan, &banks).unwrap_or_else(|why| panic!("the load does not join: {why}"));
    println!("join: all {} params satisfied", baked.plan.params.len());

    // ── the activation arena ───────────────────────────────────────────
    //
    // `rows * row_pitch`, which is the whole of what a fire's arena is:
    // `model_compiler::program::carve` already reused every byte whose value's
    // life had ended, and the pitch is what came out of it.
    //
    // ZEROED AND NOT MERELY ALLOCATED. `Device::empty` says in its own doc
    // that Vulkan does not clear a fresh allocation and that pretending
    // otherwise reads correctly for a year.
    let arena_bytes = program.row_pitch;
    let arena_buf = device
        .empty(arena_bytes)
        .expect("the activation arena allocates");
    device
        .zero(&arena_buf, 0, arena_bytes)
        .expect("the activation arena zeroes");
    let arena = BufferId(held.len() as u32);
    held.push(arena_buf);
    println!("activation arena {arena_bytes} bytes for one row");

    // ── the fire's own planes ──────────────────────────────────────────
    //
    // Every one of them a whole staged region, because `Fire::runtime` binds
    // what `Pools::table` answers WITHOUT narrowing it.
    let mut tables = BTreeMap::new();
    for (which, words) in [
        (FireTable::TokenIds, vec![PROMPT]),
        (FireTable::Positions, vec![0]),
        (FireTable::RequestOfToken, vec![0]),
        (FireTable::QoIndptr, vec![0, 1]),
        // One BYTE per row, and the row is valid.
        (FireTable::RowValid, vec![1]),
        (FireTable::SamplingIndices, vec![0]),
        (FireTable::KvPageIndices, vec![PHYSICAL_PAGE]),
        (FireTable::KvPageIndptr, vec![0, 1]),
        (FireTable::KvWritePage, vec![PHYSICAL_PAGE]),
        (FireTable::KvWriteOffset, vec![0]),
        (FireTable::RecurrentSlots, vec![0]),
        // No custom mask. The enable plane's zero is what says so and the
        // stride's zero says it again; both are read by the sdpa arms, so the
        // plane is real and holds nothing rather than being absent.
        (FireTable::AttentionMask, vec![0]),
        (FireTable::AttentionMaskEnabled, vec![0]),
    ] {
        tables.insert(which, staged(device, &mut held, &words));
    }

    // ── the pool ───────────────────────────────────────────────────────
    let layers = kv_layers(&baked.plan);
    // ONE POOL PER WIDTH AND ONE ROW PER CACHE. `gemma4-e4b`'s sliding layers
    // take `head_dim` 256 and its full-attention ones 512, and a
    // `resources::Shape` states one width for every layer it holds — so two
    // widths are two pools. Within a pool the rows are CACHES, not layers,
    // because a cache may be shared.
    let width_of = |layer: u32| {
        let a = baked.deployment.attention[layer as usize];
        (a.kv_heads, a.head_dim)
    };
    let mut caches: BTreeMap<(u32, u32), Vec<&str>> = BTreeMap::new();
    for (layer, cache) in &layers {
        let of = caches.entry(width_of(*layer)).or_default();
        if !of.contains(&cache.as_str()) {
            of.push(cache.as_str());
        }
    }
    for of in caches.values_mut() {
        of.sort_unstable();
    }

    // The pools' buffers are numbered AFTER everything in `held`, two per ROW,
    // keys then values, pool by pool — which is the order `Loaded::buffers`
    // rebuilds and nothing else states, so the two have to be read together.
    let mut pools = Vec::new();
    let mut rows: Vec<(usize, u16)> = Vec::new();
    let mut kv = BTreeMap::new();
    let mut geometry = BTreeMap::new();
    let mut next = held.len() as u32;
    for (&(kv_heads, head_dim), names) in &caches {
        let pool = Pool::open(
            device,
            Shape {
                layers: names.len() as u16,
                kv_heads,
                head_dim,
                page_size: PAGE,
                pages: PAGES,
                bytes: 2,
            },
        )
        .expect("the kv pool opens");
        let bytes = pool.shape().layer_bytes();
        let at_pool = pools.len();
        let mut planes = Vec::with_capacity(names.len());
        for row in 0..names.len() {
            planes.push((
                Slice::whole(BufferId(next), bytes),
                Slice::whole(BufferId(next + 1), bytes),
            ));
            rows.push((at_pool, row as u16));
            next += 2;
        }
        pools.push(pool);
        for (layer, cache) in layers
            .iter()
            .filter(|(l, _)| width_of(**l) == (kv_heads, head_dim))
        {
            let at = names
                .iter()
                .position(|n| *n == cache.as_str())
                .expect("a cache of this width");
            kv.insert(*layer, planes[at]);
            let this = baked.deployment.attention[*layer as usize];
            geometry.insert(
                *layer,
                KvGeometry {
                    page_size: PAGE.cast_signed(),
                    // Elements between one token and the next WITHIN A HEAD,
                    // which on this plane is the whole row.
                    // `resources::Shape::row` is `kv_heads * head_dim` and the
                    // pool is laid out `[page][token][head][dim]`, so a token
                    // step crosses every head and a head is `head_dim` wide.
                    // Metal lays the same pool out `[page][head][token]` and
                    // states a different pair; the planes agree about the
                    // MEANING of the fields and not about the layout, which is
                    // why each states its own.
                    seq_stride: u64::from(this.kv_heads) * u64::from(this.head_dim),
                    head_stride: u64::from(this.head_dim),
                    kv_heads: this.kv_heads.cast_signed(),
                    head_dim: this.head_dim.cast_signed(),
                },
            );
        }
    }

    // ── the recurrent planes ───────────────────────────────────────────
    //
    // Numbered after every KV plane, three per layer in `(state, conv,
    // new_conv)` order — which `Loaded::buffers` rebuilds and nothing else
    // states, so the two have to be read together.
    let linear = rs_layers(&baked.plan);
    let mut slabs = BTreeMap::new();
    let recurrent = (!linear.is_empty()).then(|| {
        let rs = baked
            .deployment
            .recurrent
            .as_ref()
            .expect("a tower with recurrent cache rows states its geometry");
        let shape = Recurrent {
            linear_layers: linear.len() as u32,
            conv_dim: rs.conv_dim.unsigned_abs(),
            conv_k: rs.conv_k.unsigned_abs(),
            v_heads: rs.v_h.unsigned_abs(),
            v_dim: rs.v_d.unsigned_abs(),
            k_dim: rs.k_d.unsigned_abs(),
            slots: SLOTS,
        };
        let pool = RecurrentPool::open(
            device,
            shape,
            linear.iter().map(|l| *l as u16).collect::<Vec<_>>(),
        )
        .expect("the recurrent pool opens");
        for layer in &linear {
            let state = Slice::whole(BufferId(next), shape.state_bytes_per_layer());
            let conv = Slice::whole(BufferId(next + 1), shape.conv_bytes_per_layer());
            let new_conv = Slice::whole(BufferId(next + 2), shape.conv_bytes_per_layer());
            next += 3;
            slabs.insert(*layer, (state, conv, new_conv));
        }
        pool
    });
    println!(
        "pool: kv {} cache(s) over {} layer(s) x {PAGES} pages of {PAGE}, \
         recurrent {} layer(s) x {SLOTS} slot",
        rows.len(),
        layers.len(),
        linear.len(),
    );

    Some(Loaded {
        held,
        kv_pools: pools,
        recurrent,
        kv_rows: rows,
        rs_layers: linear,
        pools: Live {
            kv,
            tables,
            slabs,
            geometry,
        },
        banks,
        arena,
        arena_bytes,
    })
}

/// bf16 or f32 bytes to `f32`.
///
/// THE READ-OUT'S DTYPE IS THE SLOT'S, NOT A GUESS. A reader that assumed
/// `f32` for a bf16 read-out gets a vocabulary exactly half zeros — which
/// looks like a dead half of a tensor and is really two elements read as one.
fn widen(bytes: &[u8], dt: Dt) -> Vec<f32> {
    match dt {
        Dt::Bf16 => bytes
            .chunks_exact(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
            .collect(),
        Dt::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        other => panic!("the `out` seam lands at {other:?}, which this reader does not widen"),
    }
}

/// Fire one lane at `tier`, once, and hand back its argmax.
///
/// `None` when the checkpoint is not cached, which the callers turn into a
/// printed skip.
fn fire(
    device: &Device,
    row: &Banked,
    tier: Capability,
    class: FireClass,
) -> Option<(usize, f32, usize, usize)> {
    let sku = row.sku;
    let baked =
        Baked::of::<Vulkan>(sku).unwrap_or_else(|why| panic!("`{sku}` does not bake: {why}"));
    let unresolved = baked.unresolved::<Vulkan>();
    assert!(
        unresolved.is_empty(),
        "this plane does not claim every point `{sku}` states: {unresolved:?}",
    );
    let (word, program) = baked
        .lane(class, false)
        .unwrap_or_else(|why| panic!("no lane serves a {} of `{sku}`: {why}", class.suffix()));
    println!(
        "{} lane {word:#b}: {} steps over {} slots, row pitch {}",
        class.suffix(),
        program.steps.len(),
        program.slots.len(),
        program.row_pitch,
    );

    let loaded = open(device, row, &baked, program)?;
    let layers = baked.deployment.layers as usize;

    // ── the walk ───────────────────────────────────────────────────────
    let fired = Fire::over(
        &baked.plan,
        program,
        Extent {
            arena: Slice::whole(loaded.arena, loaded.arena_bytes),
            rows: 1,
            requests: 1,
            layers,
        },
        &loaded.banks,
        &loaded.pools,
    );
    let encoder = Encoder::over(&fired.bindings, &fired.cursor, tier);
    fired
        .walk(&encoder)
        .unwrap_or_else(|why| panic!("the walk refused: {why}"));
    let planned = encoder.finish();
    let mut planned = planned;
    let mut blits = fired.blits.borrow().clone();
    if std::env::var_os("PIE_VULKAN_DUMP_SLOTS").is_some() {
        // WHICH STATEMENT EACH DISPATCH BELONGS TO, and what each statement
        // WRITES. Before any truncation: a bisect needs the whole mapping to
        // choose the n it is about to cut at.
        for (i, d) in planned.iter().enumerate() {
            println!("PIE_DISPATCH {i} op{}", d.op);
        }
        for (at, step) in program.steps.iter().enumerate() {
            let op = &baked.plan.ops[step.op as usize];
            println!(
                "PIE_STEP {at} op{} {} outs {:?}",
                step.op, op.kernel, op.outputs
            );
        }
    }
    // STOP AFTER `n` DISPATCHES, so the arena at the end holds that statement's
    // output rather than whatever `carve` reused its slot for. `driver-wgpu`'s
    // gate takes the same variable and `driver-metal` has
    // `PIE_METAL_MAX_DISPATCH`, which is where that plane's own bisect lives —
    // the three print the same lines so two of them can be diffed.
    let stop: Option<usize> = std::env::var("PIE_STOP_AFTER")
        .ok()
        .and_then(|v| v.parse().ok());
    if let Some(n) = stop {
        planned.truncate(n);
        // The copies of the statements that remain, and no others: `serve::run`
        // refuses an `InOut` copy filed against a dispatch the run does not
        // hold, which is right for a real fire and an obstacle for a truncated
        // one.
        let kept: std::collections::BTreeSet<u32> = planned.iter().map(|d| d.op).collect();
        blits.retain(|b| kept.contains(&b.op));
        println!("STOPPED AFTER {n} of {} dispatches", planned.len());
    }
    println!(
        "walk: {} statements planned {} dispatches and {} in-place copies",
        program.steps.len(),
        planned.len(),
        blits.len(),
    );

    // ── the read-out ───────────────────────────────────────────────────
    //
    // The `out` seam names a VALUE and the program gives it a slot like any
    // other, so the logits are read out of the arena at the rectangle the walk
    // itself would bind — there is no second answer to keep in step with the
    // first.
    let out = fired
        .rect(baked.out)
        .unwrap_or_else(|why| panic!("the `out` seam has no rectangle: {why}"));
    let vocab = out.width.unsigned_abs() as usize;
    assert_eq!(
        vocab, baked.deployment.shape.vocab as usize,
        "the read-out is not the vocabulary wide",
    );

    // ── the fire ───────────────────────────────────────────────────────
    let mut pipelines = Pipelines::new();
    let all = loaded.buffers();
    let began = std::time::Instant::now();
    let report = driver_vulkan::serve::run(
        device,
        &mut pipelines,
        &Embedded,
        &all,
        &planned,
        &blits,
        tier,
    );
    let report = match report {
        Ok(r) => r,
        Err(why) => {
            drop(all);
            loaded.free(device);
            panic!("the fire did not run: {why}");
        }
    };
    println!(
        "fired {} dispatches in {} submission(s) — {} staged, {} modules read, \
         {} above baseline, {:.2}s",
        report.dispatches,
        report.submissions,
        report.staged,
        report.parsed,
        report.tiered,
        began.elapsed().as_secs_f64(),
    );
    assert_eq!(
        report.dispatches,
        planned.len(),
        "the device half ran a different number of dispatches than the walk planned",
    );
    assert_eq!(
        report.staged,
        blits.len(),
        "the device half moved a different number of `InOut` operands than the \
         walk asked for",
    );

    if std::env::var_os("PIE_VULKAN_DUMP_SLOTS").is_some() {
        // EVERY SLOT'S RECTANGLE, in the form `driver-wgpu` and `driver-metal`
        // print. Raw bytes reduced to mean, rms and max rather than a
        // checksum: two planes reducing in two orders differ by a bf16 ulp on
        // every statement, and a byte comparison calls that a difference and
        // leaves a bisect with nothing to bisect.
        let all = device
            .read_at(&loaded.held[loaded.arena.0 as usize], 0, loaded.arena_bytes)
            .expect("the arena reads back");
        for v in 0..program.slots.len() as u32 {
            let Ok(r) = fired.rect(v) else { continue };
            let span = r.rows.unsigned_abs() as u64 * r.width.unsigned_abs() as u64 * r.dt.size();
            if span == 0 || r.slice.at + span > loaded.arena_bytes {
                continue;
            }
            let raw = &all[r.slice.at as usize..(r.slice.at + span) as usize];
            println!(
                "PIE_SLOT {v} at{} {}",
                r.slice.at,
                summary(raw, r.dt.size())
            );
        }
        if stop.is_some() {
            // A truncated fire answers nothing and is not asked to: the point
            // is the arena it leaves behind.
            return None;
        }
    }

    let bytes = device
        .read_at(
            &loaded.held[loaded.arena.0 as usize],
            out.slice.at,
            out.slice.bytes,
        )
        .expect("the logits read back");
    let row = widen(&bytes, out.dt);
    assert_eq!(row.len(), vocab, "the read-out is not one whole row");
    let (id, logit) = row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("a non-empty vocabulary");
    let answer = (id, *logit, planned.len(), blits.len());
    drop(all);
    loaded.free(device);
    Some(answer)
}

/// **THE MILESTONE.** A cached checkpoint, through this driver, onto the card,
/// to the token cuda banked.
///
/// BASELINE AND NOT THE ADAPTER'S OWN TIER, deliberately. A tier is an
/// optimisation with a fallback and every symbol keeps a baseline module, so
/// baseline is the one configuration every device shares — which is what makes
/// a banked answer a claim about this tree rather than about this card.
/// **THE PLANE SERVES.** 579 dispatches in one submission, 240 staged `InOut`
/// copies, no refusal, and **token 11 at 14.5000** — the banked token, one
/// bf16 step off the banked 14.4375.
///
/// The logit was 10.1250 when the tower first fired end to end. Three defects
/// closed the gap, and none was reachable by anything narrower than a
/// value-by-value comparison against a plane that answers correctly.
///
/// Getting here closed four refusals, one per walk, each a real point that
/// could fire for nothing: op 0 `layout.embed` (no dense arm at all), op 8
/// `rope.yarn` (a precomputed ladder nothing stages), op 22
/// `mlp.swiglu_clamp_alpha` (no strided arm for a packed row), op 24
/// `moe.weighted_sum` (only the sorted fold, which needs a permutation no
/// point of this plane writes). `tests/doors.rs` held each against its
/// sentence and now holds all four fired.
///
/// **THE BISECT IS HERE NOW**, in the form the other two planes have:
/// `PIE_STOP_AFTER` cuts the dispatch list and `PIE_VULKAN_DUMP_SLOTS` prints
/// every rectangle as mean, rms and max — the same lines `driver-wgpu`
/// prints, so the two can be diffed statement by statement on the identical
/// program. It has found three things here already:
///
/// ```text
///   19  norm.add_bias           AGREE  rel 0.0000
///   20  moe.topk_softmax        was the first divergence, TWICE
///   22  mlp.swiglu_clamp_alpha  AGREE  rel 0.0000
///   23  moe.matmul_select_bias  DIFFER rel 0.28  — where it stands
/// ```
///
/// `moe.topk_softmax` was wrong in two independent ways and each hid behind
/// the other: it normalised the softmax over ALL thirty-two experts instead of
/// the four it kept, and it fired the `PIE_ACT` weight arm at a slot the point
/// declares `f32`. The EXPERTS CHOSEN were identical throughout — only the
/// weight plane beside them moved.
///
/// `moe.matmul_select_bias` then passed `x_slot_stride = 0` unconditionally,
/// which is right for an activation with one row per TOKEN and wrong for one
/// with a row per ROUTE. gpt-oss hands it the second, so all four experts
/// contracted against route zero's activations. `activation_strides` derives
/// the pair now, as `kernels-wgpu`'s `selected` always has.
/// Open a device, fire one banked row's lane, and compare — or say plainly
/// that it measured nothing.
///
/// **THE ID IS THE CLAIM AND THE LOGIT IS THE WITNESS**, asserted differently
/// for the reason `driver-metal`'s gate gives: cuda's and `driver-wgpu`'s
/// compare the rendered logit exactly and both can, because an L40S runs both
/// of them through one vendor's compiler. This is the same card through Slang
/// and SPIR-V, reducing in its own order, and gpt-oss answers 14.5000 where
/// cuda banked 14.4375 — one bf16 step, since the ulp at fourteen is 0.0625.
///
/// Loosening the ID would be giving up the claim; loosening the logit to one
/// ulp is saying what a bf16 logit is worth. A second ulp fails.
fn gate(row: &Banked, class: FireClass) {
    let device = gpu!();
    println!("device: {} — {} {}", device.name(), row.sku, class.suffix());
    let Some((id, logit, dispatches, staged)) = fire(&device, row, Capability::Baseline, class)
    else {
        eprintln!("skipped: `{}` is not cached", row.cache);
        return;
    };
    let rendered = format!("{logit:.4}");
    println!("ARGMAX {id} at {rendered} over {dispatches} dispatches, {staged} staged");
    // **ZERO, AND THAT IS THE CLAIM NOW.** This asserted `staged > 0` — "this
    // tower states `InOut` points and the walk found none, which would make
    // the staged-copy path untested rather than passing" — and it was right
    // until `model_compiler::program::alias_in_place` landed. An `InOut`
    // result now IS the rectangle its operand already is, which is what the
    // declaration always said, so there is nothing to copy: 240 dispatches
    // went out of gpt-oss's decode and 363 out of gemma-4's.
    //
    // The copy path did not go away with them. It stands for the cases
    // aliasing declines — an operand still live at its own statement, or a
    // result wider than the operand it rides — and no catalogue row reaches
    // one. So this asserts the ZERO: the day a tower needs a copy again, the
    // number moves and this says so, which is the same service the old
    // direction gave.
    assert_eq!(
        staged, 0,
        "an `InOut` result is the rectangle its operand already is, so a fire \
         that staged {staged} copies found an operand `alias_in_place` \
         declined — which is a real case and worth reading, not a failure of \
         this tower",
    );
    assert_eq!(
        id, row.id,
        "the banked answer for `{}` is token {} and this fire answered {id} at \
         {rendered}",
        row.sku, row.id,
    );
    let banked: f32 = row.logit.parse().expect("the banked logit parses");
    // DERIVED, NOT WRITTEN DOWN. A bf16 carries eight mantissa bits, so its
    // ulp is `2^(exponent - 7)`: 0.0625 at fourteen, where gpt-oss banks, and
    // 0.03125 at seven, where gemma-4 does. A constant is right for one row
    // and a trap for the next.
    let ulp = (banked.abs().log2().floor() - 7.0).exp2();
    assert!(
        (logit - banked).abs() <= ulp,
        "`{}` is banked at {banked} and this fire answered {rendered}, which is \
         {} away — past the {ulp} one bf16 step is at this magnitude",
        row.sku,
        (logit - banked).abs(),
    );
    if rendered != row.logit {
        println!("(one bf16 step off the banked {}, and no more)", row.logit);
    }
}

/// **THE PURE-ATTENTION TOWER.** See [`GPTOSS`].
#[test]
#[ignore = "loads a 20B checkpoint; `--ignored` runs it"]
fn gptoss_20b_answers_the_argmax_cuda_banked() {
    gate(&GPTOSS, FireClass::Decode);
}

/// **THE HYBRID, and the first fire on this plane to carry a recurrent slab.**
///
/// Eighteen of twenty-four layers are gated DeltaNet, and `Pools::slab`
/// answered `None` for every one of them until `resources::RecurrentPool`
/// landed — so a walk over this tower refused at its first scan and this plane
/// could serve only the pure-attention rows.
#[test]
#[ignore = "needs a Vulkan device and a cached checkpoint"]
fn qwen35_d0_8b_answers_the_argmax_cuda_banked() {
    gate(&QWEN35, FireClass::Decode);
}

/// **THE TOWER AT TWO WIDTHS.** See [`GEMMA4`].
///
/// Every earlier row here attends at one width over one cache per layer, and
/// this one does neither: twenty-four caches under forty-two layers, at two
/// head widths. Both were fixture assumptions rather than driver ones — the
/// `Pools` trait has always taken a layer and answered a geometry — but a
/// fixture that opens one pool per layer cannot express either.
#[test]
#[ignore = "loads a 15G checkpoint; `--ignored` runs it"]
fn gemma4_e4b_answers_the_argmax_cuda_banked() {
    gate(&GEMMA4, FireClass::Decode);
}

/// **THE OTHER LANE, AND THIS PLANE HAD NEVER FIRED IT.**
///
/// A one-row fire is a decode by the `qo_one` fact and cuda banked its answers
/// from one, so every row above spells the tower one way. The prefill lane is a
/// DIFFERENT PROGRAM over the same plan: `attention.prefill` where the decode
/// states `attention.decode`, and — for the hybrid — `ssm.causal_conv1d_chunked`
/// and `ssm.gated_delta_chunked` where it states the unchunked arms. Those arms
/// are claimed here and nothing had ever walked a real tower through them.
///
/// **The token must be the same token.** A prefill of one row at position zero
/// and a decode of one row at position zero are the same forward pass over the
/// same weights into the same empty pools; they differ in how the tower is
/// SPELLED and not in what it computes. So this asserts against the banked
/// answer rather than against the decode's, which is the stronger of the two
/// claims and the one that does not go stale if the decode regresses.
#[test]
#[ignore = "needs a Vulkan device and a cached checkpoint"]
fn the_qwen35_prefill_lane_answers_the_same_token() {
    gate(&QWEN35, FireClass::Prefill);
}

/// gpt-oss's prefill lane, which reaches `attention.prefill_lse` and the
/// `attention.sink` that merges its partials.
#[test]
#[ignore = "the second load of a 20B checkpoint; `--ignored` runs it"]
fn the_gptoss_prefill_lane_answers_the_same_token() {
    gate(&GPTOSS, FireClass::Prefill);
}

/// gemma-4's prefill lane, over two attention widths and twenty-four shared
/// caches.
#[test]
#[ignore = "the second load of a 15G checkpoint; `--ignored` runs it"]
fn the_gemma4_prefill_lane_answers_the_same_token() {
    gate(&GEMMA4, FireClass::Prefill);
}

/// A rectangle's bytes as three comparable numbers — `driver-wgpu`'s twin, and
/// `driver-metal`'s. Mean, rms and max rather than a checksum: two planes
/// reducing in two orders differ by a bf16 ulp on every statement, and a byte
/// comparison calls that a difference and leaves a bisect with nothing to
/// bisect.
fn summary(raw: &[u8], width: u64) -> String {
    let vals: Vec<f64> = if width == 2 {
        raw.chunks_exact(2)
            .map(|b| {
                f64::from(f32::from_bits(
                    u32::from(u16::from_le_bytes([b[0], b[1]])) << 16,
                ))
            })
            .collect()
    } else {
        raw.chunks_exact(4)
            .map(|b| f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
            .collect()
    };
    let n = vals.len().max(1) as f64;
    let mean = vals.iter().sum::<f64>() / n;
    let rms = (vals.iter().map(|x| x * x).sum::<f64>() / n).sqrt();
    let max = vals.iter().fold(0.0f64, |a, x| a.max(x.abs()));
    format!(
        "n{} mean {mean:.6e} rms {rms:.6e} max {max:.6e}",
        vals.len()
    )
}
