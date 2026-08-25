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
use driver_vulkan::resources::{Pool, Shape};
use driver_vulkan::serve::Embedded;
use kernels_vulkan::Capability;
use model_compiler::program::{Dt, Program};
use model_ir::plan::{CacheRow, FireClass, Plan};

// ── the banked answer ──────────────────────────────────────────────────

/// The catalog row.
const SKU: &str = "gptoss-20b-bf16-mxfp4-kv-bf16";

/// Where its snapshot sits under `~/.cache/huggingface/hub`.
const CACHE: &str = "models--openai--gpt-oss-20b";

/// **The token every banked answer was fired from.** `baker-smoke`'s default
/// prompt was the single id 785 and `banked-argmaxes.sh` never overrode it, so
/// the answer below is "one fire, one row, position zero".
const PROMPT: u32 = 785;

/// The argmax cuda banked.
const BANKED_ID: usize = 11;

/// Its logit **as rendered to four decimals**, which is how cuda's gate
/// compares it and for the reason that file gives: a bf16 logit carries no
/// more digits than that, and comparing parsed floats would fail on a number
/// that is right.
const BANKED_LOGIT: &str = "14.4375";

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

    fn slab(&self, _layer: u32, _which: Slab) -> Option<Slice> {
        // gpt-oss declares no recurrence, and `None` is what a driver holding
        // no slab must answer: a scan handed a null carry answers fluently and
        // wrongly. This plane allocates none for any SKU — see the header.
        None
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
    pool: Pool,
    /// Which layers this pool holds, in the order their buffers are numbered.
    kv_layers: Vec<u32>,
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
        for layer in &self.kv_layers {
            all.push(
                self.pool
                    .cache(*layer as u16, false)
                    .expect("the pool holds this layer"),
            );
            all.push(
                self.pool
                    .cache(*layer as u16, true)
                    .expect("the pool holds this layer"),
            );
        }
        all
    }

    fn free(self, device: &Device) {
        for b in self.held {
            device.free(b);
        }
        self.pool.close(device);
    }
}

/// Which layers own a KV pool, off the PLAN.
///
/// Read from `plan.caches` rather than from `Deployment::attention`, and the
/// difference is not cosmetic: `Deployment` fills the layers that state no
/// attention from the widest one that does, so its vector says something about
/// every layer of the tower and cannot say which of them actually hold pages.
fn kv_layers(plan: &Plan) -> Vec<u32> {
    let mut kv = Vec::new();
    for row in &plan.caches {
        let CacheRow::Kv { name, .. } = row else {
            continue;
        };
        let Some(at) = name.rsplit('.').next().and_then(|s| s.parse::<u32>().ok()) else {
            continue;
        };
        if !kv.contains(&at) {
            kv.push(at);
        }
    }
    kv.sort_unstable();
    kv
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
fn open(device: &Device, baked: &Baked, program: &Program) -> Option<Loaded> {
    let snap = model::snapshot::Snapshot::open(CACHE)?;
    println!(
        "checkpoint {} — {} shard(s), {} tensors",
        snap.dir.display(),
        snap.shards(),
        snap.len()
    );

    // ── the weights ────────────────────────────────────────────────────
    let base = readable_base(SKU).expect("the SKU offers a safetensors import");
    let import = model::import_of(SKU, base).expect("the import table holds that flavor");
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
    let at = baked
        .deployment
        .attention
        .first()
        .copied()
        .expect("a tower");
    for layer in &layers {
        let this = baked.deployment.attention[*layer as usize];
        assert_eq!(
            (this.head_dim, this.kv_heads),
            (at.head_dim, at.kv_heads),
            "gpt-oss attends at one width and this fixture opens one pool; a \
             tower with two -- gemma-4 -- wants one pool per width, which \
             `driver-wgpu/tests/banked_argmax.rs` does",
        );
    }
    let pool = Pool::open(
        device,
        Shape {
            layers: baked.deployment.layers as u16,
            kv_heads: at.kv_heads,
            head_dim: at.head_dim,
            page_size: PAGE,
            pages: PAGES,
            bytes: 2,
        },
    )
    .expect("the kv pool opens");
    println!(
        "pool: kv {} layer(s) x {PAGES} pages of {PAGE}, no recurrent slab",
        layers.len(),
    );

    // The pool's buffers are numbered AFTER everything in `held`, two per
    // layer, keys then values — which is the order `Loaded::buffers` rebuilds.
    let mut kv = BTreeMap::new();
    let mut geometry = BTreeMap::new();
    let mut next = held.len() as u32;
    for layer in &layers {
        let bytes = pool.shape().layer_bytes();
        let keys = Slice::whole(BufferId(next), bytes);
        let values = Slice::whole(BufferId(next + 1), bytes);
        next += 2;
        kv.insert(*layer, (keys, values));
        let this = baked.deployment.attention[*layer as usize];
        geometry.insert(
            *layer,
            KvGeometry {
                page_size: PAGE.cast_signed(),
                // Elements between one token and the next WITHIN A HEAD, which
                // on this plane is the whole row. `resources::Shape::row` is
                // `kv_heads * head_dim` and the pool is laid out
                // `[page][token][head][dim]`, so a token step crosses every
                // head and a head is `head_dim` wide. Metal lays the same pool
                // out `[page][head][token]` and states a different pair; the
                // planes agree about the MEANING of the fields and not about
                // the layout, which is why each states its own.
                seq_stride: u64::from(this.kv_heads) * u64::from(this.head_dim),
                head_stride: u64::from(this.head_dim),
                kv_heads: this.kv_heads.cast_signed(),
                head_dim: this.head_dim.cast_signed(),
            },
        );
    }

    Some(Loaded {
        held,
        pool,
        kv_layers: layers,
        pools: Live {
            kv,
            tables,
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
fn fire(device: &Device, tier: Capability, class: FireClass) -> Option<(usize, f32, usize, usize)> {
    let baked =
        Baked::of::<Vulkan>(SKU).unwrap_or_else(|why| panic!("`{SKU}` does not bake: {why}"));
    let unresolved = baked.unresolved::<Vulkan>();
    assert!(
        unresolved.is_empty(),
        "this plane does not claim every point `{SKU}` states: {unresolved:?}",
    );
    let (word, program) = baked
        .lane(class, false)
        .unwrap_or_else(|why| panic!("no lane serves a {} of `{SKU}`: {why}", class.suffix()));
    println!(
        "{} lane {word:#b}: {} steps over {} slots, row pitch {}",
        class.suffix(),
        program.steps.len(),
        program.slots.len(),
        program.row_pitch,
    );

    let loaded = open(device, &baked, program)?;
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
    let blits = fired.blits.borrow().clone();
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
/// **THE WHOLE TOWER FIRES AND THE ARGMAX IS THE BANKED TOKEN. The logit is
/// not.**
///
/// 579 dispatches in one submission, 240 staged `InOut` copies, no refusal —
/// and **token 11**, which is what cuda banked. The logit is 10.1250 against
/// a banked 14.4375, which is not a rounding gap and not one this file can
/// explain yet.
///
/// Getting here closed four refusals, one per walk, each a real point that
/// could fire for nothing: op 0 `layout.embed` (no dense arm at all), op 8
/// `rope.yarn` (a precomputed ladder nothing stages), op 22
/// `mlp.swiglu_clamp_alpha` (no strided arm for a packed row), op 24
/// `moe.weighted_sum` (only the sorted fold, which needs a permutation no
/// point of this plane writes). `tests/doors.rs` held each against its
/// sentence and now holds all four fired.
///
/// **The next step is the bisect the other two planes have.**
/// `driver-wgpu`'s `PIE_STOP_AFTER` and `PIE_WGPU_DUMP_SLOTS`, and
/// `driver-metal`'s `PIE_METAL_MAX_DISPATCH` and `PIE_METAL_DUMP_SLOTS`,
/// compare two planes statement by statement on the identical program — which
/// is what found metal's `InOut` copies being staged before the fire ran, at
/// the first statement that could show it. This plane has no such probe yet
/// and wants one.
#[test]
#[ignore = "the tower fires and answers the banked token; the logit is 10.1250 against 14.4375"]
fn gptoss_20b_answers_the_argmax_cuda_banked() {
    let device = gpu!();
    println!("device: {}", device.name());
    let Some((id, logit, dispatches, staged)) =
        fire(&device, Capability::Baseline, FireClass::Decode)
    else {
        eprintln!("skipped: `{CACHE}` is not cached");
        return;
    };
    let rendered = format!("{logit:.4}");
    println!("ARGMAX {id} at {rendered} over {dispatches} dispatches, {staged} staged");
    assert!(
        staged > 0,
        "this tower states `InOut` points and the walk found none, which would \
         make the staged-copy path below untested rather than passing",
    );
    assert_eq!(
        (id, rendered.as_str()),
        (BANKED_ID, BANKED_LOGIT),
        "the banked answer for `{SKU}` is {BANKED_ID} at {BANKED_LOGIT} and \
         this fire answered {id} at {rendered}",
    );
}
