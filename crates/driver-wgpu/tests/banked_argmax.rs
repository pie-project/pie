//! **THE WHOLE CHAIN, ON A REAL ADAPTER: a cached checkpoint through this
//! driver to an argmax, against the answer cuda banked.**
//!
//! `crates/driver-cuda/tests/banked_argmaxes.rs` is what this is measured
//! against and its header is the history: three checkpoints, one token in, an
//! argmax and a logit banked from the first end-to-end fire each SKU ever
//! managed. `qwen35-d0.8b-bf16-kv-bf16` answered **198 at 12.3125** from the
//! single-token prompt `[785]`, and until this file no plane but cuda had ever
//! been asked the question.
//!
//! # What this asks that nothing else in this crate does
//!
//! `tests/the_walk_is_the_program.rs` asks whether the walk visits the right
//! steps with a recorder standing where a device would. `tests/device_fire.rs`
//! asks whether ONE claimed point, taken through the generated dispatch and a
//! `#[claims]` body, computes the right numbers on the adapter.
//! `tests/device_sink.rs` asks the same of three more. Each of those is one
//! link. This is the chain:
//!
//! > weights produced → uploaded → pools allocated → a `Program` walked →
//! > every statement fired → logits read → argmax compared.
//!
//! Nothing here is a fixture. The plan is `model::trace_of(SKU)(Wgpu)`, the
//! lane is `model_compiler::program::bound`'s, the weights are 260 real banks
//! out of a real safetensors snapshot, the pools are `resources::{Pool,
//! RecurrentPool}` at the geometry `model::deployment` read off the same plan,
//! and the dispatches are what `kernels-wgpu`'s own claim bodies asked for.
//! **459 statements, 24 layers, six of them full-attention and eighteen
//! gated-DeltaNet.**
//!
//! # Why the answer is falsifiable
//!
//! Because the comparison is against a number this tree did not compute. A
//! self-consistency check — this plane against itself, or against an f64 model
//! of what this plane's shaders do — would pass for a tower that agreed with
//! itself and disagreed with the model. 198 at 12.3125 came off a CUDA card
//! through `kernels-cuda`, and the only thing the two planes share is the model
//! TEXT.
//!
//! `PHYSICAL_PAGE` is the other half: the fire's one token is written to and
//! read from PHYSICAL PAGE 3, not page 0, so a driver that bound the pool and
//! ignored the translation planes would attend over zeros. It is a constant
//! rather than a test of its own because it is a property of the fixture; what
//! makes it a check is that page 3 of 4 is not reachable by accident.
//!
//! # What walking a whole tower found, which nothing narrower could
//!
//! **`attention.decode` and `attention.prefill` could not be fired on this
//! plane at all, on any adapter, for any SKU.** `attn/sdpa_paged.wgsl` declares
//! `sinks` at `@group(0) @binding(10)` for every variant and only the
//! `PIE_WITH_SINK` ones read it, and this plane's bind group covers what a
//! module READS (`reflect::of_module`). `kernels_wgpu::attn`'s unsplit `decode`
//! arm, its split and merge arms, `tiled` — which is both `attention.prefill`
//! and `attention.masked` — and `mma` each passed `points::absent` at that
//! slot, so each bound one buffer more than its layout has and
//! `Device::check_bindable` refused it: *"the module's layout has 10 entries
//! and 11 buffers were bound"*. The split arm passed two such stand-ins and the
//! merge arm ten.
//!
//! Nothing in the tree could see it. `tests/device_sink.rs` fires the
//! sink-bearing and `_lse` arms, which bind their reads exactly and are the
//! worked example of the right shape; `tests/device_fire.rs` fires one norm;
//! `tests/the_walk_is_the_program.rs` never meets a module. It took a walk over
//! a real tower — six attention layers of twenty-four — to reach the first one.
//!
//! The fix is in `kernels-wgpu`: every one of those stand-ins is gone, and
//! `points::absent` had no caller left and went with them. `Encode::absent`
//! stays, because the case it names is real — a binding a module READS that the
//! point states no operand for — and this is simply not one.
//!
//! # Why it skips rather than fails without the checkpoint
//!
//! Same reason `tests/device.rs` skips with no adapter, and stated in
//! `src/skip.rs`: a suite that needs 1.4 GiB of cached weights is not one a
//! build box has, and a test that FAILED there would be turned off. What must
//! not happen is passing quietly, so the skip prints and the fire itself
//! asserts.
//!
//! IT IS TWO SKIPS AND NOT ONE, for the reason `skip.rs` draws the line: the
//! workflow installs `mesa-vulkan-drivers` and then sets
//! `PIE_WGPU_REQUIRE_DEVICE`, so a missing ADAPTER is that runner failing to
//! keep its own word and is `skipped`. Nothing has ever asked that runner for
//! a checkpoint, so a missing SNAPSHOT is `unmeasured`, under
//! `PIE_WGPU_REQUIRE_WEIGHTS` — a switch no workflow sets and this tree's own
//! gate script does. Routing the snapshot through `skipped` would have turned
//! the device switch red for something never requested, which is exactly how
//! `skip.rs` says the switch that DOES work gets unset.

#![cfg(feature = "native")]
#![allow(clippy::print_stdout)]

use std::collections::{BTreeMap, BTreeSet};

use driver_wgpu::baker::marks::{BufferId, Slice};
use driver_wgpu::baker::stage::{FireTable, KvGeometry, Pools, Slab};
use driver_wgpu::baker::walk::{Extent, Fire};
use driver_wgpu::baker::{Baked, Bank, Wgpu, arenas_of, encode::Encoder, join, readable_base};
use driver_wgpu::device::{Buffer, Device, Pipelines};
use driver_wgpu::resources::{Pool, Recurrent, RecurrentPool, Shape};
use driver_wgpu::serve::Embedded;
use kernels_wgpu::Capability;
use model_compiler::program::{Dt, Program};
use model_ir::plan::{CacheRow, FireClass, Plan};

// ── the banked answer ──────────────────────────────────────────────────

/// One row of `driver-cuda`'s `BANKED`, which is the table this file answers
/// against and does not extend: an entry here is a claim that CUDA already
/// published, never a number this plane banked for itself.
struct Banked {
    /// The catalog row. The INSTRUCT checkpoints, which are the ones the
    /// answers were banked from — `models--Qwen--Qwen3.5-0.8B-Base` is a
    /// different row and would produce a different token with no complaint
    /// from anything.
    sku: &'static str,
    /// Where its snapshot sits under `~/.cache/huggingface/hub`.
    cache: &'static str,
    /// The argmax token id.
    id: usize,
    /// Its logit **as rendered to four decimals**, which is how cuda's gate
    /// compares it and for the reason that file gives: a bf16 logit carries no
    /// more digits than that, and comparing parsed floats would fail on a
    /// number that is right.
    logit: &'static str,
}

const QWEN35: Banked = Banked {
    sku: "qwen35-d0.8b-bf16-kv-bf16",
    cache: "models--Qwen--Qwen3.5-0.8B",
    id: 198,
    logit: "12.3125",
};

const GPTOSS: Banked = Banked {
    sku: "gptoss-20b-bf16-mxfp4-kv-bf16",
    cache: "models--openai--gpt-oss-20b",
    id: 11,
    logit: "14.4375",
};

/// **THE THIRD ROW, AND IT COST A CHANGE TO THE FLOOR TO GET HERE.**
///
/// `gemma4-e4b` attends at TWO widths: its sliding layers take `kv_heads: 2,
/// head_dim: 256` and its full-attention layers `global_kv_heads: 2,
/// global_head_dim: 512`. `baker::stage::Pools` used to answer
/// `kv_geometry()` once per FIRE with no layer argument, so the `seq_stride`
/// and `head_stride` a claim body read would have been right for one half of
/// the tower and wrong for the other — and wrong here means attending over
/// the wrong bytes rather than refusing. `driver-cuda` had always derived it
/// per layer (`bind::views::kv_view`); the shader planes now do too, and this
/// fixture opens one `resources::Pool` per width.
const GEMMA4: Banked = Banked {
    sku: "gemma4-e4b-bf16-kv-bf16",
    cache: "models--google--gemma-4-E4B-it",
    id: 785,
    logit: "7.5938",
};

/// **The token every banked answer was fired from.** `baker-smoke`'s default
/// prompt was the single id 785 and `banked-argmaxes.sh` never overrode it, so
/// the answers below are "one fire, one row, position zero".
const PROMPT: u32 = 785;

// ── the fire's shape ───────────────────────────────────────────────────

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

/// Recurrent seats. One, because this fire is one request.
const SLOTS: u32 = 1;

// THE BUFFER TABLE HAS NO FIXED ROWS ANY MORE, and one 20B checkpoint is why.
//
// It was `WEIGHTS = 0, ARENA = 1`. The weights are now as many buffers as this
// adapter's ceiling needs -- one for qwen3.5-0.8b, four for gpt-oss-20b -- so
// the activation arena is at `arenas.len()` and everything after it is
// numbered in the order `Loaded::open` allocates it. `Loaded::arena` carries
// the one id a caller outside this file needs. A plain comment and not a doc
// one: there is no item under it any more.

// ── what this fire stages beside its arena ─────────────────────────────

/// The driver's answer to [`Pools`], holding regions and nothing else.
///
/// Every method turns an allocation into a plain [`Slice`]; the executor reads
/// regions and mints handles, and which `wgpu::Buffer` is behind one is this
/// struct's business alone. `driver-metal`'s `serve::launch::FireStaging` is
/// the same seam on its plane.
struct Live {
    /// Per layer, the keys and values planes — absent for a layer that carries
    /// no KV, which on this hybrid is eighteen of twenty-four.
    kv: BTreeMap<u32, (Slice, Slice)>,
    /// Per layer, `(state, conv, new_conv)`. Absent for the six full-attention
    /// layers, which carry no recurrence.
    slabs: BTreeMap<u32, (Slice, Slice, Slice)>,
    /// The staged planes, by the row of `kernels::runtime::TIER1` that names
    /// them.
    tables: BTreeMap<FireTable, Slice>,
    /// **Per layer, because a tower may attend at more than one width.** A
    /// gemma-4 sliding layer is `head_dim` 256 and a full-attention one 512,
    /// so this is a map and not a field; every layer [`Self::kv`] answers for
    /// has a row here, and the two are filled together.
    geometry: BTreeMap<u32, KvGeometry>,
}

impl Pools for Live {
    fn kv(&self, layer: u32, values: bool) -> Option<Slice> {
        self.kv
            .get(&layer)
            .map(|(k, v)| if values { *v } else { *k })
    }

    fn slab(&self, layer: u32, which: Slab) -> Option<Slice> {
        let (state, conv, new_conv) = self.slabs.get(&layer)?;
        Some(match which {
            Slab::State => *state,
            Slab::Conv => *conv,
            Slab::NewConv => *new_conv,
        })
    }

    fn kv_geometry(&self, layer: u32) -> KvGeometry {
        // Only asked of a layer `kv` answered for, and every one of those was
        // given a row when its pool was allocated — so a miss here is this
        // fixture's own bookkeeping and not a tower with a gap in it.
        self.geometry[&layer]
    }

    fn table(&self, which: FireTable) -> Option<Slice> {
        self.tables.get(&which).copied()
    }
}

/// Everything the device holds for one load and one fire.
///
/// The `Vec<Buffer>` and the `Live` are built together and kept together
/// because a [`BufferId`] is an index into the FIRST and every [`Slice`] in the
/// second carries one: separating them would be two lists that have to agree
/// about a numbering, which is the class of mistake `BufferId::NONE` exists to
/// make loud and this shape makes impossible.
struct Loaded {
    held: Vec<Buffer>,
    pools: Live,
    banks: BTreeMap<String, Bank>,
    /// Which buffer the activation arena is. Not a constant: the weights take
    /// as many buffers before it as the adapter's ceiling needed.
    arena: BufferId,
    /// Bytes of the activation arena, which is `row_pitch * rows`.
    arena_bytes: u64,
}

/// Which layers own a KV pool, and which own a recurrent one, off the PLAN.
///
/// Read from `plan.caches` rather than from `Deployment::attention`, and the
/// difference is not cosmetic: `Deployment` fills the layers that state no
/// attention from the widest one that does, so its vector says something about
/// every layer of the tower and cannot say which six of them actually hold
/// pages. The cache rows can, because a row exists only where a layer declared
/// one.
fn cache_layers(plan: &Plan) -> (Vec<u32>, Vec<u32>) {
    let mut kv = Vec::new();
    let mut recurrent = Vec::new();
    for row in &plan.caches {
        let (name, into) = match row {
            CacheRow::Kv { name, .. } => (name, &mut kv),
            CacheRow::State { name, .. } => (name, &mut recurrent),
        };
        let Some(at) = name.rsplit('.').next().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        if !into.contains(&at) {
            into.push(at);
        }
    }
    kv.sort_unstable();
    recurrent.sort_unstable();
    (kv, recurrent)
}

/// A staged plane, appended to the buffer table and answered as a whole region.
fn staged(device: &Device, held: &mut Vec<Buffer>, words: &[u32]) -> Slice {
    let buffer = device.words(words).expect("a fire table allocates");
    let slice = Slice::whole(BufferId(held.len() as u32), buffer.size());
    held.push(buffer);
    slice
}

impl Loaded {
    /// Produce the weights, upload them, allocate the pools, stage the fire.
    ///
    /// Everything that happens before a dispatch exists. It is one function
    /// because the numbering is one numbering — see [`Loaded`] — and because
    /// every step of it is a step the milestone's report names.
    fn open(device: &Device, row: &Banked, baked: &Baked, program: &Program) -> Option<Self> {
        let snap = model::snapshot::Snapshot::open(row.cache)?;
        println!(
            "checkpoint {} — {} shard(s), {} tensors",
            snap.dir.display(),
            snap.shards(),
            snap.len()
        );

        // ── the weights ────────────────────────────────────────────────
        let base = readable_base(row.sku).expect("the SKU offers a safetensors import");
        let import = model::import_of(row.sku, base).expect("the import table holds that flavor");
        let produced = model::produce::produce(&import, &baked.plan.params, 0, &|n| snap.read(n))
            .unwrap_or_else(|why| panic!("the checkpoint does not produce this plan: {why}"));
        // AS MANY ARENAS AS THIS ADAPTER'S CEILING NEEDS, which for
        // qwen3.5-0.8b is one and for gpt-oss-20b is four. `Device::budget` is
        // the smaller of `max_buffer_size` and `max_storage_buffer_binding_size`
        // — both matter, because an arena past the second allocates and then
        // cannot be bound.
        let arenas = arenas_of(&produced, device.budget())
            .unwrap_or_else(|why| panic!("the weights do not pack: {why}"));
        let weight_bytes: u64 = arenas.iter().map(|a| a.bytes).sum();
        println!(
            "produced {} tensors into {} arena(s) of {:.2} GiB total (ceiling {:.2} GiB)",
            produced.len(),
            arenas.len(),
            weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            device.budget() as f64 / (1024.0 * 1024.0 * 1024.0),
        );

        let mut held = Vec::new();
        for arena in &arenas {
            let buffer = device
                .zeroed(arena.bytes)
                .expect("the weight arena allocates");
            for &(i, at) in &arena.banks {
                device
                    .write(&buffer, at, &produced[i].1.bytes)
                    .expect("a bank uploads");
            }
            held.push(buffer);
        }
        assert_eq!(held.len(), arenas.len());

        // ONE CONTIGUOUS UPLOAD PER BANK, and the region each landed in is what
        // a `Const` slot binds. The repr is the PLAN's column and not the
        // storage dtype: a quantised bank's form lives only there, and
        // `BoundOp::form` reads it and nothing else.
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
                            // THE ARENA'S INDEX *IS* THE BUFFER ID, which is
                            // what makes the split cost nothing downstream: a
                            // `Slice` has always named a buffer, so a bank in
                            // the fourth arena binds exactly like one in the
                            // first and no claim body can tell.
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

        // ── the activation arena ───────────────────────────────────────
        //
        // `rows * row_pitch`, which is the whole of what a fire's arena is:
        // `model_compiler::program::carve` already reused every byte whose
        // value's life had ended, and the pitch is what came out of it.
        let arena_bytes = program.row_pitch;
        held.push(
            device
                .zeroed(arena_bytes)
                .expect("the activation arena allocates"),
        );
        let arena = BufferId(held.len() as u32 - 1);
        println!("activation arena {arena_bytes} bytes for one row");

        // ── the pools ──────────────────────────────────────────────────
        let (kv_layers, rs_layers) = cache_layers(&baked.plan);

        // ONE POOL PER WIDTH, BECAUSE A TOWER MAY ATTEND AT MORE THAN ONE.
        // `gemma4-e4b`'s sliding layers take `head_dim` 256 and its
        // full-attention layers 512. A `resources::Shape` states one width for
        // every layer it holds, so two widths are two pools — and
        // `Pools::kv_geometry(layer)` is how a claim body is told which it got.
        //
        // Each pool is opened at the FULL layer count so `Pool::cache(layer)`
        // keeps indexing by the tower's own numbering, and only the layers of
        // that width are taken out of it; the rest are dropped with the pool
        // and never enter `held`. For a one-token fire at four pages that
        // over-allocates a few hundred KiB for a moment, which is the cheaper
        // half of the trade against renumbering layers per pool.
        let width_of = |layer: u32| {
            let a = baked.deployment.attention[layer as usize];
            (a.kv_heads, a.head_dim)
        };
        let mut widths: Vec<(u32, u32)> = kv_layers.iter().map(|l| width_of(*l)).collect();
        widths.sort_unstable();
        widths.dedup();

        // NO RECURRENT POOL FOR A TOWER THAT DECLARES NO RECURRENCE, and the
        // question is asked of the PLAN rather than of the deployment. A
        // `Deployment::recurrent` is `None` for gpt-oss, but a tower could in
        // principle carry the geometry and no cache rows; `rs_layers` is the
        // list of layers that actually declared a slab, so it is the one that
        // decides whether a pool is opened at all. `expect` here would refuse
        // every pure-attention SKU for wanting a slab none of them has.
        let rpool = (!rs_layers.is_empty()).then(|| {
            let rs = baked
                .deployment
                .recurrent
                .as_ref()
                .expect("a tower with recurrent cache rows states its geometry");
            RecurrentPool::open(
                device,
                Recurrent {
                    linear_layers: rs.linear_layers.len() as u32,
                    conv_dim: rs.conv_dim.unsigned_abs(),
                    conv_k: rs.conv_k.unsigned_abs(),
                    v_heads: rs.v_h.unsigned_abs(),
                    v_dim: rs.v_d.unsigned_abs(),
                    k_dim: rs.k_d.unsigned_abs(),
                    slots: SLOTS,
                },
                rs_layers.iter().map(|l| *l as u16).collect::<Vec<_>>(),
            )
            .expect("the recurrent pool opens")
        });
        println!(
            "pools: kv {} layer(s) x {PAGES} pages of {PAGE}, recurrent {} layer(s) x {SLOTS} slot",
            kv_layers.len(),
            rs_layers.len(),
        );

        // The pool owns its buffers, so the table takes CLONES of the handles
        // — a `wgpu::Buffer` is `Arc`-backed, so this is a refcount and not a
        // copy of a cache, and it is what lets one numbering cover pools and
        // staged tables alike.
        let mut kv = BTreeMap::new();
        let mut geometry = BTreeMap::new();
        for &(kv_heads, head_dim) in &widths {
            let pool = Pool::open(
                device,
                Shape {
                    layers: baked.deployment.layers as u16,
                    kv_heads,
                    head_dim,
                    page_size: PAGE,
                    pages: PAGES,
                    bytes: 2,
                },
            )
            .expect("the kv pool opens");
            for layer in kv_layers
                .iter()
                .filter(|l| width_of(**l) == (kv_heads, head_dim))
            {
                let mut region = |values: bool| {
                    let buffer = pool
                        .cache(*layer as u16, values)
                        .expect("the pool holds this layer")
                        .clone();
                    let slice = Slice::whole(BufferId(held.len() as u32), buffer.size());
                    held.push(buffer);
                    slice
                };
                let keys = region(false);
                let values = region(true);
                kv.insert(*layer, (keys, values));
                geometry.insert(
                    *layer,
                    KvGeometry {
                        page_size: PAGE.cast_signed(),
                        // Elements between one token and the next WITHIN A
                        // HEAD, which on this plane is the whole row:
                        // `attn/kv_write.wgsl` writes `slot * (n_kv_heads *
                        // head_dim) + h * head_dim + d` and
                        // `attn/sdpa_paged.wgsl` reads `(slot * n_kv_heads +
                        // kv_head) * head_dim`. Metal lays the same pool out
                        // `[page][head][token]` and states a different pair;
                        // the two planes agree about the MEANING of the fields
                        // and not about the layout, which is why each states
                        // its own.
                        seq_stride: u64::from(kv_heads) * u64::from(head_dim),
                        head_stride: u64::from(head_dim),
                        kv_heads: kv_heads.cast_signed(),
                    },
                );
            }
            // The pool owns its buffers and `held` took a handle to each one
            // this width uses; a `wgpu::Buffer` releases its allocation when
            // the LAST handle goes, so dropping the pool here frees exactly
            // the layers no statement will bind.
            drop(pool);
        }
        let mut slabs = BTreeMap::new();
        for layer in &rs_layers {
            let rpool = rpool.as_ref().expect("a recurrent layer has a pool");
            let mut region = |which: &str| {
                let buffer = rs_slab(rpool, *layer as u16, which);
                let slice = Slice::whole(BufferId(held.len() as u32), buffer.size());
                held.push(buffer);
                slice
            };
            let state = region("recurrent_state");
            let conv = region("conv_state");
            let new_conv = region("new_conv_state");
            slabs.insert(*layer, (state, conv, new_conv));
        }

        // ── the fire's own planes ──────────────────────────────────────
        //
        // Every one of them a whole staged region, because `Fire::runtime`
        // binds what `Pools::table` answers WITHOUT narrowing it — the shape a
        // rectangle wears is stated there and the extent is the staging's. So
        // `row_valid` is one byte of meaning in a four-byte allocation, which
        // is what its own doc means by "the declared element is a fiction the
        // DECLARATION carries and the buffer must not".
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
            // stride's zero says it again; both are read by the sdpa arms, so
            // the plane is real and holds nothing rather than being absent.
            (FireTable::AttentionMask, vec![0]),
            (FireTable::AttentionMaskEnabled, vec![0]),
        ] {
            tables.insert(which, staged(device, &mut held, &words));
        }

        // The recurrent pool is dropped here and its buffers are not: `held`
        // took a handle to each, and a `wgpu::Buffer` releases its allocation
        // when the LAST handle goes. The KV pools were dropped as each width
        // finished.
        drop(rpool);

        Some(Self {
            held,
            pools: Live {
                kv,
                slabs,
                tables,
                geometry,
            },
            banks,
            arena,
            arena_bytes,
        })
    }
}

/// One recurrent plane, by the name `resources::RecurrentPool` files it under.
///
/// A function so the `expect` message names the plane rather than the layer,
/// which is the half a reader cannot guess from a panic in a closure.
fn rs_slab(pool: &RecurrentPool, layer: u16, which: &str) -> Buffer {
    pool.slab(layer, which)
        .unwrap_or_else(|| panic!("the recurrent pool holds no `{which}` for layer {layer}"))
        .clone()
}

// ── the fire ───────────────────────────────────────────────────────────

/// Widen a read-out to `f32`, at the element the SLOT says it holds.
///
/// **Four is not a given**, which is the finding `serve`'s deleted `Unread`
/// recorded and the one every plane has had to relearn: a text's declared dtype
/// does not change what a kernel writes, and a reader that assumed `f32` for a
/// bf16 read-out got a vocabulary exactly half zeros — which looks like a dead
/// half of a tensor and is really two elements read as one.
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

/// Fire one lane of one banked row at `tier`, once, and hand back its argmax.
///
/// `None` when the checkpoint is not cached, which the callers turn into a
/// printed skip.
fn fire(
    device: &Device,
    banked: &Banked,
    tier: Capability,
    class: FireClass,
) -> Option<(usize, f32, usize)> {
    let sku = banked.sku;
    let baked = Baked::of::<Wgpu>(sku).unwrap_or_else(|why| panic!("`{sku}` does not bake: {why}"));
    let unresolved = baked.unresolved::<Wgpu>();
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

    let loaded = Loaded::open(device, banked, &baked, program)?;
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
    let planned = {
        let encoder = Encoder::over(&fired.bindings, &fired.cursor);
        fired
            .walk(&encoder)
            .unwrap_or_else(|why| panic!("the walk refused: {why}"));
        encoder.finish()
    };
    // STOP AFTER `n` DISPATCHES, so the arena at the end holds that statement's
    // output rather than whatever `carve` reused its slot for. Nothing else can
    // localise a disagreement to a statement: a whole tower fires either way,
    // and by the end almost every slot has been written over.
    let mut planned = planned;
    let mut blits = fired.blits.borrow().clone();
    if std::env::var_os("PIE_WGPU_DUMP_SLOTS").is_some() {
        // WHICH STATEMENT EACH DISPATCH BELONGS TO — see `driver-metal`'s twin.
        for (i, d) in planned.iter().enumerate() {
            println!("PIE_DISPATCH {i} op{}", d.op);
        }
    }
    let stop: Option<usize> = std::env::var("PIE_STOP_AFTER")
        .ok()
        .and_then(|v| v.parse().ok());
    if let Some(n) = stop {
        planned.truncate(n);
        // The copies of the statements that remain, and no others: `serve::run`
        // refuses an `InOut` copy filed against a dispatch the run does not
        // hold, which is the right answer for a real fire and an obstacle for
        // a truncated one.
        let kept: std::collections::BTreeSet<u32> = planned.iter().map(|d| d.op).collect();
        blits.retain(|b| kept.contains(&b.op));
        println!("STOPPED AFTER {n} of {} dispatches", program.steps.len());
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
    let buffers: Vec<&Buffer> = loaded.held.iter().collect();
    let began = std::time::Instant::now();
    let (report, bytes) = driver_wgpu::serve::run(
        device,
        &mut pipelines,
        &Embedded,
        &buffers,
        &planned,
        &blits,
        tier,
        if stop.is_some() {
            Some((loaded.arena, 0, loaded.arena_bytes))
        } else {
            Some((out.slice.buffer, out.slice.at, out.slice.bytes))
        },
    )
    .unwrap_or_else(|why| panic!("the fire did not run: {why}"));
    println!(
        "fired {} dispatches in {} submission(s) — {} shadowed, {} staged, \
         {} pipelines built, {:.2}s",
        report.dispatches,
        report.submissions,
        report.shadowed,
        report.staged,
        pipelines.built(),
        began.elapsed().as_secs_f64(),
    );
    assert_eq!(
        report.dispatches,
        planned.len(),
        "the device half ran a different number of dispatches than the walk planned",
    );
    if stop.is_some() {
        // A truncated fire answers nothing and is not asked to: the point is
        // the arena it leaves behind.
        for v in 0..program.slots.len() as u32 {
            let Ok(r) = fired.rect(v) else { continue };
            let span = r.rows.unsigned_abs() as u64 * r.width.unsigned_abs() as u64 * r.dt.size();
            if span == 0 || r.slice.at + span > loaded.arena_bytes {
                continue;
            }
            let raw = &bytes[r.slice.at as usize..(r.slice.at + span) as usize];
            println!(
                "PIE_SLOT {v} at{} {}",
                r.slice.at,
                summary(raw, r.dt.size())
            );
        }
        return None;
    }
    assert_eq!(
        report.shadowed, 0,
        "a `var<storage, read>` declaration came back somewhere in the shader \
         tree; `serve::Fired::shadowed` is zero for every real plan and its \
         only other symptom is that decoding got twice as slow",
    );
    assert_eq!(
        report.staged,
        blits.len(),
        "the device half moved a different number of `InOut` operands than the \
         walk asked for",
    );

    // EVERY SLOT'S RECTANGLE, when asked. `driver-metal`'s
    // `PIE_METAL_DUMP_SLOTS` prints the same thing in the same form on the
    // same program, so the first slot whose line differs is the first
    // statement the two planes disagree about. A slot is REUSED by `carve`, so
    // what survives to the end is its last writer's output — the same
    // statement on both planes, which is what keeps the comparison honest.
    //
    // RAW BYTES AND NOT WIDENED VALUES: the slots are bf16, f32 and i32 alike,
    // and a reader that had to know which would be a second place the two
    // planes could disagree about something other than the arithmetic.
    if std::env::var_os("PIE_WGPU_DUMP_SLOTS").is_some() {
        // ONE READ OF THE WHOLE ARENA and then sliced on the host, rather than
        // one `serve::run` per rectangle: a run of zero dispatches performs no
        // read-back, which the per-rectangle version discovered by printing
        // four hundred and fifty empty rows.
        let (_, all) = driver_wgpu::serve::run(
            device,
            &mut pipelines,
            &Embedded,
            &buffers,
            &planned,
            &blits,
            tier,
            Some((loaded.arena, 0, loaded.arena_bytes)),
        )
        .expect("a read of the whole arena");
        // WHICH STATEMENT LAST WROTE EACH SLOT, in program order. A slot is
        // reused by `carve`, so a slot that differs between two planes only
        // localises a statement once you know which one left it that way.
        for (at, step) in program.steps.iter().enumerate() {
            let op = &baked.plan.ops[step.op as usize];
            println!(
                "PIE_STEP {at} op{} {} outs {:?}",
                step.op, op.kernel, op.outputs
            );
        }
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
    }

    let row = widen(&bytes, out.dt);
    assert_eq!(row.len(), vocab, "the read-out is not one whole row");
    let (id, logit) = row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("a non-empty vocabulary");
    Some((id, *logit, planned.len()))
}

/// Open an adapter, fire one banked row's lane, and compare — or say plainly
/// that it measured nothing.
///
/// THE TWO SKIPS ARE DIFFERENT SKIPS and `src/skip.rs` is where the line is
/// drawn: an absent ADAPTER is a runner failing the guarantee it was set up to
/// make, and an absent SNAPSHOT is a thing no runner was ever asked for.
fn gate(banked: &Banked, tier: Capability, class: FireClass, why: &str) {
    let Ok(device) = Device::open() else {
        driver_wgpu::skip::skipped("no adapter answered `Device::open`");
        return;
    };
    println!(
        "adapter: {} ({:?}) — {} at {}",
        device.name(),
        device.backend(),
        banked.sku,
        tier.tag(),
    );
    let Some((id, logit, dispatches)) = fire(&device, banked, tier, class) else {
        driver_wgpu::skip::unmeasured(&format!("`{}` is not cached", banked.cache));
        return;
    };
    let rendered = format!("{logit:.4}");
    println!("ARGMAX {id} at {rendered} over {dispatches} dispatches");
    assert_eq!(
        (id, rendered.as_str()),
        (banked.id, banked.logit),
        "{why}: `{}` is banked at {} at {} and this fire answered {id} at \
         {rendered}",
        banked.sku,
        banked.id,
        banked.logit,
    );
}

/// **THE MILESTONE.** A cached checkpoint, through this driver, onto the card,
/// to the token cuda banked.
///
/// BASELINE AND NOT THE ADAPTER'S OWN TIER, deliberately. A tier is an
/// optimisation with a fallback and every symbol keeps a baseline module
/// (`serve::pick`), so baseline is the one configuration every adapter shares
/// — which is what makes a banked answer a claim about this tree rather than
/// about this card. `the_argmax_does_not_move_at_this_adapters_own_tier` is
/// the other half.
#[test]
fn qwen35_d0_8b_answers_the_argmax_cuda_banked() {
    gate(
        &QWEN35,
        Capability::Baseline,
        FireClass::Decode,
        "the decode lane",
    );
}

/// **THE SECOND SKU, and it is a different tower in every way that matters.**
///
/// qwen3.5-0.8b is a hybrid whose attention is six layers of twenty-four; this
/// is twenty-four layers of attention with a SINK on every one, alternating
/// sliding-window and full, over a mixture of experts whose weights are
/// **mxfp4** rather than bf16. So it fires `attention.{decode,decode_lse,sink}`
/// and `moe.{topk_softmax,matmul_select_bias,weighted_sum}` — the second of
/// which is the point whose mxfp4 arm carried the same stand-in binding defect
/// the attention family did, and which nothing but a whole tower would reach.
///
/// It shares no kernel with the first row except the norms, the rope and the
/// gemm, so a plane that answers both is a plane whose ATTENTION and whose
/// QUANTISED matmul are both right — not one that got a hybrid's narrow path
/// working.
#[test]
fn gptoss_20b_answers_the_argmax_cuda_banked() {
    gate(
        &GPTOSS,
        Capability::Baseline,
        FireClass::Decode,
        "the decode lane",
    );
}

/// **THE THIRD SKU, and the one that made `kv_geometry` take a layer.**
///
/// Two attention widths in one tower — 256 at the sliding layers, 512 at the
/// full ones — so this is the fire that would answer a token either way and
/// only answers the RIGHT one if each layer's strides are its own. It also
/// carries the `masked` fact, so its attention is `attention.masked` where the
/// other two rows state `attention.decode`.
#[test]
#[ignore = "one bank is past what this plane can bind; see the test below"]
fn gemma4_e4b_answers_the_argmax_cuda_banked() {
    gate(
        &GEMMA4,
        Capability::Baseline,
        FireClass::Decode,
        "the decode lane",
    );
}

/// **THE OTHER LANE.** One token through the PREFILL text, which must answer
/// the same thing.
///
/// A one-row fire is a decode by the `qo_one` fact, and cuda banked its answer
/// from one — so nothing in this tree had ever run the prefill lane of a real
/// tower end to end. It is a different program over the same plan: 24 layers of
/// `ssm.causal_conv1d_chunked` and `ssm.gated_delta_chunked` where the decode
/// lane states the unchunked arms, and `attention.prefill` where it states
/// `attention.decode`. Those are the two families whose stand-in bindings this
/// milestone had to fix, and until this test fired them the fix for
/// `attention.prefill` was reasoning rather than a number.
///
/// **The token must be the same token.** A prefill of one row at position zero
/// and a decode of one row at position zero are the same forward pass over the
/// same weights into the same empty pools; they differ in how the tower is
/// SPELLED and not in what it computes. So this asserts against the banked
/// answer rather than against the decode's, which is the stronger of the two
/// claims and the one that does not go stale if the decode regresses.
#[test]
fn the_prefill_lane_answers_the_same_token() {
    gate(
        &QWEN35,
        Capability::Baseline,
        FireClass::Prefill,
        "the prefill lane is the same tower spelled differently",
    );
}

/// gpt-oss's prefill lane, which reaches `attention.prefill_lse` and the
/// `attention.sink` that merges its partials — the arm this plane claimed
/// most recently and the one no fire had exercised over a real tower.
#[test]
#[ignore = "the second load of a 20B checkpoint; `--ignored` runs it"]
fn the_gptoss_prefill_lane_answers_the_same_token() {
    gate(
        &GPTOSS,
        Capability::Baseline,
        FireClass::Prefill,
        "the prefill lane is the same tower spelled differently",
    );
}

/// The same fire at whatever tier this adapter offers, which must not move the
/// answer.
///
/// A TIER IS AN OPTIMISATION AND THIS IS WHAT SAYS SO. `serve::pick` walks
/// `Capability::PREFERENCE` and lands on the best variant the tree has; a
/// `@subgroup` or `@matrix` module that computed something else would be
/// invisible to every test in this crate that fires at baseline, and visible
/// here as a different token.
///
/// `#[ignore]`d and not skipped: the load is 1.4 GiB and the fire compiles the
/// whole shader tree, so running it beside the milestone doubles the suite for
/// a check whose subject is a second configuration of the same chain.
#[test]
#[ignore = "the second load; `--ignored` runs it"]
fn the_argmax_does_not_move_at_this_adapters_own_tier() {
    let Ok(device) = Device::open() else {
        driver_wgpu::skip::skipped("no adapter answered `Device::open`");
        return;
    };
    let tier = device
        .tiers()
        .first()
        .copied()
        .unwrap_or(Capability::Baseline);
    gate(&QWEN35, tier, FireClass::Decode, "this adapter's own tier")
}

/// **WHAT GEMMA-4 IS ACTUALLY BLOCKED ON, and it is not what it looked like.**
///
/// Two things were in the way and only one of them was true.
///
/// THE FIRST WAS REAL AND IS FIXED. `gemma4-e4b` attends at two widths, and
/// `baker::stage::Pools` used to answer `kv_geometry()` once per FIRE with no
/// layer argument — so the strides a claim body read would have been right for
/// one half of the tower and wrong for the other, which here means attending
/// over the wrong bytes rather than refusing. `kv_geometry(layer)` is the fix
/// and `Loaded::open` opens one `resources::Pool` per width.
///
/// THE SECOND IS A BANK. `ple.table` is `[262144, 10752]` in bf16 — **5.25
/// GiB in one tensor**, and `layout.embed` binds it WHOLE. Splitting the arena
/// does not help and cannot: `arenas_of` packs banks into allocations, and
/// this is one bank past what one allocation may be BOUND at. This adapter
/// states 2 GiB; NVIDIA's Vulkan driver states `UINT32_MAX`, which is 4; the
/// WebGPU floor is 128 MiB. No shader plane can bind it as it stands.
///
/// Closing it is a change to the POINT and not to any driver: `layout.embed`
/// would have to take a table in shards and select among them, which is a new
/// shape of operand on the floor and a new indexing in four kernels. Until
/// then `gemma4_e4b_answers_the_argmax_cuda_banked` is `#[ignore]`d — and this
/// test is what keeps that `#[ignore]` honest, because it fails the day either
/// half of the sentence stops being true.
///
/// Needs no device and no checkpoint: both facts are in the plan.
#[test]
fn gemma4_attends_at_two_widths_and_is_blocked_on_one_bank() {
    let baked = Baked::of::<Wgpu>(GEMMA4.sku).expect("gemma-4 bakes");

    // Every point it states is claimed here. The blocker is not arithmetic.
    assert!(
        baked.unresolved::<Wgpu>().is_empty(),
        "this plane claims every point `{}` states; if that stopped being \
         true the blocker below is no longer the interesting one",
        GEMMA4.sku,
    );

    let widths: BTreeSet<(u32, u32)> = baked
        .plan
        .caches
        .iter()
        .filter_map(|row| match row {
            CacheRow::Kv { name, .. } => name.rsplit('.').next()?.parse::<usize>().ok(),
            CacheRow::State { .. } => None,
        })
        .map(|l| {
            let a = baked.deployment.attention[l];
            (a.kv_heads, a.head_dim)
        })
        .collect();
    assert_eq!(
        widths.len(),
        2,
        "gemma-4 is the row that made `kv_geometry` take a layer, and it is \
         because its tower holds two KV geometries: {widths:?}",
    );

    // The bank, from the plan's own column. `Repr` is bf16, so two bytes an
    // element, which is what `model::produce` will hand the arena.
    let table = baked
        .plan
        .params
        .iter()
        .find(|p| p.name == "ple.table")
        .expect("gemma-4 carries a per-layer embedding table");
    let bytes: u64 = table.shape.iter().product::<u64>() * 2;
    assert!(
        bytes > 4 * 1024 * 1024 * 1024,
        "`ple.table` is {bytes} bytes. This test claims it is past every \
         shader plane's binding ceiling, and the largest any of them states \
         is 4 GiB — if it has shrunk below that, the `#[ignore]` beside it \
         should come off and this assertion with it",
    );
}

/// A rectangle's bytes as three comparable numbers.
///
/// NOT A CHECKSUM. The first form of this printed one, and it said that
/// `norm.rmsnorm_plus_one` "differs" between two planes whose outputs were
/// `bf62` and `bf61` — adjacent bf16 values, one ulp apart, which is what two
/// reduction orders do and not what a wrong kernel does. A byte comparison
/// cannot tell those apart and every slot downstream of one inherits it, so
/// every slot differed and the bisect had nothing to bisect.
///
/// Mean, root-mean-square and the largest magnitude, decoded by the element's
/// WIDTH: two bytes is bf16 and four is f32. An `i32` plane is read as f32 and
/// the number is nonsense — identically nonsense on both planes, which is all
/// the comparison asks of it.
fn summary(raw: &[u8], width: u64) -> String {
    let vals: Vec<f64> = match width {
        2 => raw
            .chunks_exact(2)
            .map(|b| {
                f64::from(f32::from_bits(
                    u32::from(u16::from_le_bytes([b[0], b[1]])) << 16,
                ))
            })
            .collect(),
        _ => raw
            .chunks_exact(4)
            .map(|b| f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
            .collect(),
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
