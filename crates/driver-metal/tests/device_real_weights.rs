//! **A real checkpoint's weights, through the generic executor, and what came
//! out.**
//!
//! `device_text_fire.rs` proves the fire executes against sentinels;
//! `device_checkpoint_names.rs` proves every name binds against a checkpoint.
//! Neither looks at a NUMBER, and the gap between them is where a driver hides
//! its worst defects: a fire that runs to completion over correctly-addressed
//! weights and computes nonsense is indistinguishable from a working one
//! unless somebody reads the output.
//!
//! So this reads the output. Not against a reference — that is the accuracy
//! gate's job and it wants one — but against the three failure modes that
//! account for most of the distance:
//!
//!   * **all zeros.** A projection told its extents are zero no-ops; a weight
//!     bound to an unwritten arena slot contributes nothing. Both leave the
//!     residual stream exactly as the embedding left it, or empty.
//!   * **non-finite.** A norm handed a zero epsilon divides by the root of the
//!     mean square alone; a NaN anywhere spreads to everything downstream
//!     within one layer.
//!   * **degenerate.** Every row identical means the per-token axis is not
//!     reaching the kernels — a launch whose grid collapsed, or a gather
//!     reading token 0 for every lane.
//!
//! None of those three is subtle and all three are invisible without a read.
//! Passing here is not correctness; it is the floor beneath which correctness
//! cannot be discussed.
//!
//! # Running it
//!
//! ```text
//! PIE_METAL_SMOKE_CHECKPOINT=<an MLX snapshot dir with a config.json> \
//!   cargo test -p driver-metal --features metal-4 --test device_real_weights \
//!   -- --include-ignored --test-threads=1
//! ```
//!
//! Both flags are load-bearing. `--test-threads=1` because twelve tests each
//! mapping an 18 GB checkpoint at once is a SIGKILL, not a slowdown.
//!
//! `--include-ignored` because every test here is `#[ignore]`d, and that is a
//! correction rather than a convenience. Each one used to open by reading the
//! environment variable and `return`ing with an `eprintln!` when it was
//! unset -- and libtest swallows a PASSING test's stderr. So the suite
//! reported `ok. 12 passed` in 0.00s, CI never sets the variable, and the
//! strongest gate in the crate reported twelve passes of nothing for as long
//! as it has existed. A test that reports the same result whether or not it
//! ran is not a test. `ignored` is the one word libtest has for "this did not
//! run", so the suite says it.
//!
//! Running it for the first time found what it was built to find: a real
//! gemma-4-31b decode leaves nineteen of twenty-two arena regions unwritten
//! across 1255 non-empty dispatches, and fills three with ~1e27. The same
//! twelve are green on a real llama, MLX token-for-token agreement included.
//! See `.wiki/driver/real-metal-north-star.md` §15.
//!
//! It found six defects in its first afternoon, and the last two are the ones
//! that argue for the file:
//!
//!   1. **No barrier between dispatches.** Metal does not order two dispatches
//!      in one compute encoder and the executor's loop emitted none. Three
//!      runs of one fire gave widest activations of 11.7, 23.1 and 4.5e12 --
//!      TWO OF THE THREE looked entirely plausible.
//!   2. **The readout's dtype.** The text said `F32`, `affine_qmv_fast` writes
//!      bfloat, and the logits came back exactly half zero.
//!   3. **Unzeroed arena and KV pool.** A fresh Metal buffer is usually zero
//!      and nothing promises it, so an attention read past what a fire wrote
//!      attended to whatever the allocator last held.
//!   4. **The single-row gather.** `embed_gather_4bit` reads `id[0]` and
//!      writes one row whatever grid it is handed, and the text picked it by
//!      CLASS -- but a decode of four requests is four rows. One readout lane
//!      of four held anything, and NOTHING ELSE WAS WRONG: every launch stated
//!      four rows, every grid covered them, and every other kernel read the
//!      row where the grid put it.
//!   5. **Contiguous attention over a paged pool.** The text chose by CLASS
//!      where the POOL's layout decides, so a decode walked
//!      `[page, token, head, dim]` with `sdpa_vector_decode`'s arithmetic.
//!   6. **`v_new` bound to nothing.** `dispatch::reorder` defaulted a row's
//!      output count to ONE, and `kv_append` names no `Out` -- it writes the
//!      POOL. So the last INPUT was taken for an output and `In(1)` had
//!      nothing to resolve to. The K pages filled, the V pages were zero in
//!      every layer, and the attention that read them answered zero without
//!      failing. The widest activation went 1.1 -> 14.75 when it was fixed,
//!      which is the difference between a residual stream and a rumour of
//!      one.
//!
//! Every statement in the first layer writes both rows now, attention
//! included. What is NOT established is that any of the numbers is the right
//! number -- that still wants a reference, and this gate is the floor beneath
//! it rather than a substitute for it.
//!
//! Three measurements track what is left, each pinned so it can only improve:
//! declared outputs nothing fills (**0**, was 5), readout lanes that hold
//! anything (**4** of 4, was 1), and the arena's non-zero share (**99%**, was
//! 26%).
//!
//! Gated on `PIE_METAL_SMOKE_CHECKPOINT`, the same variable the other
//! checkpoint tests take. Run against
//! `mlx-community/Llama-3.2-1B-Instruct-4bit`.
//!
//! # gpt-oss-20b: it loads now, and it NaNs
//!
//! Measured 2026-08-10 against `mlx-community/gpt-oss-20b-MXFP4-Q4`, which
//! became runnable here the day `stage_plan_weights` stopped holding the
//! model twice (12.1 GB peak; the old path wanted about twice that and this
//! machine has 32 GB).
//!
//! `a_real_checkpoints_weights_produce_finite_varied_activations` **fails**
//! on it: 909,207 NaNs. That is the first numeric result gpt-oss has ever
//! produced here -- every prior gate was structural (names resolve, the fire
//! encodes, the launches are legal grids), and all of those still pass.
//!
//! What the bisection says, which is where anyone picking this up should
//! start: **layer 0 is entirely finite and plausible.** Twelve statements,
//! every one writing both rows, magnitudes from 2.5 to 36, the KV pool
//! holding real keys and values --
//!
//! ```text
//! [ 8] sdpa_paged_decode_sink_bfloat16_d_64   max|v| 10.25
//! [ 9] affine_qmv_fast_residual_...           max|v| 31.25
//! [10] rms_single_row_bfloat16                max|v|  2.51
//! [11] affine_qmv_fast_...  (the router)      max|v|  3.90
//! ```
//!
//! So the attention half is right, the sink kernel runs, and the router is
//! handed a sane activation. The NaN is downstream of statement 11 -- in the
//! six-statement routed FFN (`route_sort`, `route_gather`, two
//! `routed_qmv`, the clamped `swiglu`, `combine_sorted`) or in what a later
//! layer does with its output.
//!
//! Two candidates worth checking first, in order: the SwiGLU's `limit` and
//! `alpha` (gpt-oss clamps the gate above, clamps the linear branch both ways
//! and adds one to it -- a wrong bound there is an overflow, not a rounding
//! error), and the expert bank's row count after the sort pads each group up
//! to a tile.


use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use driver_metal::device::{Allocation, Context};
use driver_metal::program::Compiler;
use driver_metal::lowering::dispatch::Geometry;
use driver_metal::bind::encode::Pipelines;
use driver_metal::lowering::executor::{FireTable, Resolver, Slice};
use driver_metal::lowering::frame::{Step, lower_step};
use driver_metal::pools::kv::Pool;
    use driver_metal::layout::kv::Shape;
use driver_metal::weights::load::load;
use driver_metal::lowering::resolve::{Names, Store};
use driver_metal::layout::region::Region as _;
use model::families::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
use model::families::llama_like::forward::llama_like_metal;
use model_compiler::trace::FireClass;

fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .join("kernels-metal/kernels")
}

fn snapshot() -> Option<PathBuf> {
    std::env::var_os("PIE_METAL_SMOKE_CHECKPOINT").map(PathBuf::from)
}

/// WHICH MODEL a snapshot is, at what affine point, and in what shape.
///
/// Eleven tests below open the same checkpoint and every one of them used to
/// spell the same four lines: normalize `config.json` into a `pie.model/1`
/// descriptor, parse the descriptor back into a private `ModelFacts`, project
/// THAT into a `DecodeGeometry`, and hand the descriptor to the loader as
/// well. Four steps, three intermediate documents, and eleven copies of the
/// sequence — so a change to any of it was eleven edits and the first one
/// missed was a test comparing a real GPU's output against the wrong shape.
///
/// It is one step now and it is worth naming what that step IS: the
/// checkpoint's TENSORS pick a `model::catalog` row, and everything else is a
/// projection of the row. No document is believed, because none is read for
/// anything except the quantization — which a row genuinely cannot state,
/// since `mlx-community` publishes the same weights at 4 bits group 64 and at
/// 8 bits group 32 and the two pack to shapes no extent distinguishes.
///
/// A refusal here PANICS rather than skipping. These are the A/B tests: they
/// are the only place a real Metal device's numbers are compared against
/// anything, and a skip that prints to stderr is how that comparison quietly
/// stops happening.
fn served(
    snapshot: &Path,
) -> (
    &'static dyn model::catalog::Variant,
    model::encoding::Encoding,
    driver_metal::batch::DecodeGeometry,
) {
    let meta = model_loader::checkpoint::read::parse_checkpoint_metadata(snapshot)
        .unwrap_or_else(|e| panic!("{} did not read as a checkpoint: {e:?}", snapshot.display()));
    let row = model::catalog::identify(&meta, &model::catalog::Override::None)
        .unwrap_or_else(|e| panic!("{}: {e}", snapshot.display()));
    let config = match model_loader::checkpoint::read::read_meta(
        &meta,
        model::encoding::CONFIG_OBJECT,
    ) {
        Ok(Some(bytes)) => String::from_utf8(bytes).expect("the embedded config is utf8"),
        _ => std::fs::read_to_string(snapshot.join("config.json"))
            .unwrap_or_else(|e| panic!("{}/config.json: {e}", snapshot.display())),
    };
    let encoding = model::encoding::Encoding::from_config_json(&config)
        .unwrap_or_else(|e| panic!("{}: no encoding in the config: {e}", snapshot.display()));
    let deployment = row
        .deployment(model::catalog::Deployed::single())
        .unwrap_or_else(|e| panic!("`{}` does not deploy: {e}", row.id()));
    let dg = driver_metal::batch::geometry_from_deployment(
        &deployment,
        row.load_shape(),
        driver_metal::batch::AffineFormat {
            bits: encoding.bits,
            group: encoding.group_size,
        },
    )
    .unwrap_or_else(|e| panic!("`{}` projects no decodable geometry: {}", row.id(), e.0));
    (row, encoding, dg)
}

/// What a run of the whole arena found.
#[derive(Debug, Default)]
struct Census {
    finite_nonzero: usize,
    zero: usize,
    nan: usize,
    inf: usize,
    /// The widest magnitude seen, which says whether anything saturated.
    max_abs: f32,
}

/// Count what is in `bytes`, read at `element` bytes per value.
///
/// The element width is NOT a constant over an arena, and assuming it was is
/// the first thing this gate got wrong about itself. 89% of a llama-1B
/// decode's arena is the readout, which is `DType::F32`; the rest is the
/// residual stream, which is bf16. Reading the f32 half as bf16 reports the
/// LOW sixteen bits of every logit as a number, which came out as 5.8e11 and
/// looked exactly like saturation.
///
/// `Arg::Arena` states `bytes` per element for precisely this reason -- its
/// own doc says a driver that windows a rectangle needs the stride and that
/// every hand windowing in the CUDA executor multiplied by two -- so the
/// census asks the lowering rather than guessing.
fn census(bytes: &[u8], element: usize) -> Census {
    let mut c = Census::default();
    for chunk in bytes.chunks_exact(element) {
        let v = if element == 4 {
            f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
        } else {
            // A bf16 is the TOP half of an f32, so widening is a shift.
            f32::from_bits(u32::from(u16::from_le_bytes([chunk[0], chunk[1]])) << 16)
        };
        if v.is_nan() {
            c.nan += 1;
        } else if v.is_infinite() {
            c.inf += 1;
        } else if v == 0.0 {
            c.zero += 1;
        } else {
            c.finite_nonzero += 1;
            c.max_abs = c.max_abs.max(v.abs());
        }
    }
    c
}

/// The checkpoint's weights, the fire's tables, and the pool's geometry.
struct Live<'a> {
    store: Store<'a>,
    tables: &'a driver_metal::bind::tables::Staged,
    shape: Shape,
    pages: &'a dyn Fn(u16, bool) -> Option<Slice>,
}

impl Resolver for Live<'_> {
    fn weight(&mut self, name: &str) -> Option<Slice> {
        self.store.weight(name)
    }
    fn named(&mut self, value: model_compiler::trace::ValueId) -> Option<Slice> {
        self.store.named(value)
    }
    fn kv(&mut self, layer: u16, values: bool) -> Option<Slice> {
        (self.pages)(layer, values)
    }
    fn fire(&mut self, which: FireTable) -> Option<Slice> {
        self.tables.at(which)
    }
    fn pool(&mut self, which: FireTable) -> Option<u32> {
        Some(match which {
            FireTable::KvHeadStride => self.shape.head_dim,
            FireTable::KvSeqStride => self.shape.kv_heads * self.shape.head_dim,
            FireTable::KvPageSize => self.shape.page_size,
            _ => return None,
        })
    }
}

/// How many dispatches the fire plans, and how many have an empty grid.
fn plan_count(
    lowered: &model_compiler::lower::Lowered,
    facts: &model::families::llama_like::forward::facts::LlamaLikeFacts,
    live: &mut Live<'_>,
) -> String {
    let dispatches = driver_metal::lowering::dispatch::plan(
        lowered,
        driver_metal::lowering::dispatch::table(),
        driver_metal::lowering::executor::Frame {
            arena: Slice {
                address: 0x1_0000_0000,
                bytes: 1 << 30,
            },
        },
        Geometry {
            q_heads: facts.q_heads,
            kv_heads: facts.kv_heads,
            head_dim: facts.head_dim,
            rotary_dims: facts.head_dim,
            n_experts: facts.n_experts,
            experts_per_token: facts.experts_per_token,
        },
        live,
    )
    .expect("the fire plans");
    let empty = dispatches
        .iter()
        .filter(|d| d.grid.contains(&0) || d.threadgroup.contains(&0))
        .count();
    format!("{} ({empty} with an empty grid)", dispatches.len())
}

/// The fire's own tables, staged into one region exactly as the engine seam
/// stages them.
///
/// FOUR DIFFERENT tokens at four different positions, which is what makes the
/// per-token checks able to fail at all. A zeroed region for every table was
/// the first draft and it decodes token 0 at position 0 on every lane -- a
/// legitimate input, and a degenerate one that says nothing about whether the
/// per-token axis works.
fn stage_tables(
    context: &Context,
    step: &Step<'_>,
    page_size: u32,
    freqs: &[f32],
) -> driver_metal::bind::tables::Staged {
    let n = step.token_ids.len() as u32;
    let positions: Vec<u32> = (0..n).collect();
    let each: Vec<u32> = (0..n).collect();
    let indptr: Vec<u32> = (0..=n).collect();
    let w_off: Vec<u32> = positions.iter().map(|p| p % page_size.max(1)).collect();
    let inv_freq: Vec<u32> = freqs.iter().map(|f| f.to_bits()).collect();
    driver_metal::bind::tables::stage(
        context,
        driver_metal::bind::tables::Frame {
            token_ids: step.token_ids,
            position_ids: &positions,
            req_of_token: &each,
            kv_page_indices: &each,
            kv_page_indptr: &indptr,
            kv_write_page: &each,
            kv_write_offset: &w_off,
            rope_frequencies: &inv_freq,
            sampling_indices: step.sampling_indices,
        },
    )
    .expect("the tables stage")
}

#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn a_real_checkpoints_weights_produce_finite_varied_activations() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let (row, encoding, dg) = served(&snapshot);
    let loaded = load(&context, &snapshot, row, &encoding).expect("the checkpoint loads");
    let (facts, metal) =
        driver_metal::model::text::facts_from(&dg, |t| loaded.tensors.contains_key(t));

    // Four lanes, one token each: the decode a scheduler posts.
    let step = Step {
        token_ids: &[128_000, 9906, 1917, 128_001],
        qo_indptr: &[0, 1, 2, 3, 4],
        sampling_indices: &[0, 1, 2, 3],
        ..Step::default()
    };
    let plan = llama_like_metal(&facts, &metal, FireClass::Decode);
    let lowered = lower_step(&plan, &step).expect("the step lowers");

    let shape = Shape {
        layers: facts.layers,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        page_size: 16,
        pages: 64,
        element_bytes: 2,
        global_head_dim: 0,
        global_kv_heads: 0,
        full_attn_every: 0,
    };
    let pool = Pool::allocate(&context, shape).expect("a pool");
    let pages = |layer: u16, values: bool| {
        pool.layer(u32::from(layer)).map(|l| Slice {
            address: if values {
                l.v.gpu_address()
            } else {
                l.k.gpu_address()
            },
            bytes: shape.layer_bytes_at(0),
        })
    };

    let freqs = driver_metal::model::rope::frequencies(
        facts.head_dim,
        metal.rope_theta,
        (dg.rope_freq_factor > 0.0).then_some(driver_metal::model::rope::Rescale {
            factor: dg.rope_freq_factor,
            low: dg.rope_low_freq_factor,
            high: dg.rope_high_freq_factor,
            original_max: dg.rope_original_max_position as f32,
        }),
    );
    let staged = stage_tables(&context, &step, shape.page_size, &freqs);

    let named = HashMap::new();
    let mut live = Live {
        store: Store::new(Names::mlx(), &loaded.tensors, &named),
        tables: &staged,
        shape,
        pages: &pages,
    };

    let geometry = Geometry {
        q_heads: facts.q_heads,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        rotary_dims: facts.head_dim,
        n_experts: facts.n_experts,
        experts_per_token: facts.experts_per_token,
    };
    let (timing, arena) = driver_metal::fire::run::run_keeping_arena(
        &context,
        &compiler,
        &mut pipelines,
        &lowered,
        geometry,
        &mut live,
    )
    .expect("the fire runs against real weights");

    assert!(
        timing.encode > std::time::Duration::ZERO,
        "nothing was encoded"
    );
    assert!(
        live.store.missed().is_empty(),
        "the fire asked for {} name(s) the checkpoint does not answer, so the \
         census below would be about sentinels: {:?}",
        live.store.missed().len(),
        live.store.missed()
    );

    // Did the KV pool get anything? An attention that reads a pool nothing
    // wrote answers zero and looks exactly like an attention that is broken.
    for l in 0..2.min(shape.layers) {
        let layer = pool.layer(l).expect("a layer");
        let n = shape.layer_bytes_at(0) as usize;
        // SAFETY: the command buffer retired.
        let (k, v) = unsafe {
            (
                core::slice::from_raw_parts(
                    layer.k
                        .host_span(0, n as u64)
                        .expect("the pages are addressable")
                        .as_ptr()
                        .cast_const(),
                    n,
                ),
                core::slice::from_raw_parts(
                    layer.v
                        .host_span(0, n as u64)
                        .expect("the pages are addressable")
                        .as_ptr()
                        .cast_const(),
                    n,
                ),
            )
        };
        // The first row of each, as numbers. A byte count says the pool was
        // written; it cannot say WHICH tensor landed there, and "the attention
        // answers with K" is exactly the question of which.
        let head = |r: &[u8]| {
            r.chunks_exact(2)
                .take(6)
                .map(|c| {
                    let x = f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16);
                    format!("{x:.6}")
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        eprintln!(
            "  kv layer {l}: {} of {n} K bytes non-zero, {} V\n    K[0..6] [{}]\n    V[0..6] [{}]",
            k.iter().filter(|&&b| b != 0).count(),
            v.iter().filter(|&&b| b != 0).count(),
            head(k),
            head(v),
        );
    }

    let mut read = vec![0u8; arena.len() as usize];
    // SAFETY: the command buffer retired before `run_keeping_arena` returned,
    // so nothing is writing the arena.
    unsafe {
        let raw = core::slice::from_raw_parts(
            arena.contents().as_ptr().cast_const().cast::<u8>(),
            arena.len() as usize,
        );
        read.copy_from_slice(raw);
    }

    // Every arena region the lowering states, censused at ITS element width.
    // Regions rather than the whole buffer: an arena is mixed-dtype and one
    // census over all of it is meaningful only for the dtype that happens to
    // dominate.
    let mut regions: Vec<(usize, usize, usize)> = lowered
        .args
        .iter()
        .filter_map(|a| match a {
            model_compiler::lower::Arg::Arena { at, width, bytes } => {
                Some((*at, *width as usize * *bytes as usize, *bytes as usize))
            }
            _ => None,
        })
        .collect();
    regions.sort_unstable();
    regions.dedup();
    // Widen each region to where the NEXT one starts. `width * bytes` is one
    // ROW and a decode's value is `rows` of them, so censusing the stated
    // width looks at the first token's slice and calls it the region -- which
    // cannot tell "nothing wrote this" from "the write landed at the wrong
    // offset inside it".
    //
    // An arena REUSES offsets: the same `at` carries different tensors at
    // different points in the schedule, so the same start appears several
    // times with different widths, and a stated `width * bytes` can run past
    // where the next region begins. Counting those descriptors one after
    // another censuses the same bytes once per descriptor. That is not a
    // presentation detail -- gemma-4-31b's arena holds 2,496,512 bf16 values
    // and the run counted 2,673,664 of them, which is how this was found.
    //
    // So the census walks DISTINCT starts and gives each the bytes up to the
    // next one. Each byte is then read exactly once. Where descriptors at one
    // start disagree about the element width, the widest wins: reading f32 as
    // bf16 reports the low sixteen bits of every value as a number, and those
    // bit patterns are NaN about as often as not.
    let mut by_start: BTreeMap<usize, usize> = BTreeMap::new();
    for (at, _, bytes) in &regions {
        let e = by_start.entry(*at).or_insert(*bytes);
        *e = (*e).max(*bytes);
    }
    let starts: Vec<usize> = by_start.keys().copied().collect();
    let mut regions: Vec<(usize, usize, usize)> = Vec::with_capacity(starts.len());
    for (i, at) in starts.iter().enumerate() {
        let end = starts.get(i + 1).copied().unwrap_or(read.len()).min(read.len());
        if *at >= end {
            continue;
        }
        regions.push((*at, end - at, by_start[at]));
    }

    // Which statement each arena offset belongs to, and whether it is that
    // statement's OUTPUT. A region nothing wrote is diagnosable only if the
    // report says which launch was supposed to write it.
    let mut writers: HashMap<usize, String> = HashMap::new();
    for launch in &lowered.launches {
        let symbol = &lowered.kernels[launch.kernel as usize];
        let args = &lowered.args[launch.args.start as usize..launch.args.end as usize];
        // The trace states inputs, then OUTPUTS, then weights, and the row
        // says how many of the widthed operands are results — the same split
        // `dispatch::reorder` makes. A region that is only ever an INPUT is
        // one nothing was ever supposed to write.
        let results = kernels::sig_in(kernels_metal::KERNELS, symbol)
            .map(|sig| {
                sig.operands
                    .iter()
                    .filter_map(|o| match o.source {
                        kernels::Source::Out(i) => Some(usize::from(i) + 1),
                        _ => None,
                    })
                    .max()
                    .unwrap_or(1)
            })
            .unwrap_or(1);
        let widthed: Vec<&model_compiler::lower::Arg> = args
            .iter()
            .filter(|a| !matches!(a, model_compiler::lower::Arg::Weight(_)))
            .collect();
        let split = widthed.len().saturating_sub(results);
        for arg in widthed.iter().skip(split) {
            if let model_compiler::lower::Arg::Arena { at, .. } = arg {
                writers
                    .entry(*at)
                    .or_insert_with(|| format!("written by {symbol}"));
            }
        }
    }

    {
        let mut hist: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
        for l in &lowered.launches {
            *hist.entry(l.rows.end - l.rows.start).or_default() += 1;
        }
        eprintln!("launch rows histogram: {hist:?}");
    }
    eprintln!(
        "{} launch(es) -> {} dispatch(es)",
        lowered.launches.len(),
        plan_count(&lowered, &facts, &mut live)
    );

    let mut c = Census::default();
    let mut unwritten: Vec<String> = Vec::new();
    let mut widest_by_element: Vec<(usize, f32)> = Vec::new();
    for (at, len, element) in &regions {
        let end = (at + len).min(read.len());
        if *at >= end {
            continue;
        }
        let r = census(&read[*at..end], *element);
        c.finite_nonzero += r.finite_nonzero;
        c.zero += r.zero;
        c.nan += r.nan;
        c.inf += r.inf;
        if r.finite_nonzero == 0 && writers.contains_key(at) {
            unwritten.push(format!(
                "  @{at} ({} elements x{element}): {}",
                len / element,
                writers[at]
            ));
        }
        widest_by_element.push((*element, r.max_abs));
        c.max_abs = c.max_abs.max(r.max_abs);
        eprintln!(
            "  @{at:>8} {len:>8} B x{element}: {:>7} nz, {:>7} zero, max |v| = {}{}",
            r.finite_nonzero,
            r.zero,
            r.max_abs,
            if r.finite_nonzero == 0 {
                format!(
                    "   <- NOTHING WROTE THIS ({})",
                    writers
                        .get(at)
                        .map_or("NO LAUNCH WRITES IT — read-only", String::as_str)
                )
            } else {
                String::new()
            }
        );
    }
    let widest = |e: usize| {
        widest_by_element
            .iter()
            .filter(|(el, _)| *el == e)
            .map(|(_, v)| *v)
            .fold(0.0f32, f32::max)
    };
    eprintln!(
        "arena {} B in {} region(s): {} finite non-zero, {} zero, {} NaN, {} inf; \
         widest |v| = {} (bf16 {}, f32 {})",
        read.len(),
        regions.len(),
        c.finite_nonzero,
        c.zero,
        c.nan,
        c.inf,
        c.max_abs,
        widest(2),
        widest(4),
    );

    // ── the three failure modes ──
    assert_eq!(
        c.nan, 0,
        "the fire produced {} NaN(s). A NaN anywhere spreads to everything \
         downstream within one layer, so this is not a rounding question.",
        c.nan
    );
    assert_eq!(
        c.inf, 0,
        "the fire produced {} infinity(ies), which is what a norm handed a \
         zero epsilon does to a near-zero row.",
        c.inf
    );
    // Measured 648205 non-zero to 8179 zero: 99% of the arena holds a value.
    // It was 171268 to 485116 while the gather was the single-row one, which
    // is the same defect the lane count below names.
    assert!(
        c.finite_nonzero > c.zero * 10,
        "the arena is {} zero to {} non-zero. A projection told its extents \
         are zero no-ops and leaves exactly this, so a near-empty arena is a \
         fire that ran and did not compute.",
        c.zero,
        c.finite_nonzero
    );

    // MAGNITUDES, and the bounds are loose on purpose: what is being caught is
    // saturation, not inaccuracy. A llama-1B decode measures its widest
    // activation under one and its widest logit around 0.08 -- both small,
    // both finite -- and the bounds sit orders of magnitude out so a real
    // drift trips them and a different checkpoint does not.
    assert!(
        c.max_abs > 1e-4 && c.max_abs < 1e3,
        "the widest value anywhere is {}, which is saturation or silence \
         rather than a forward pass.",
        c.max_abs
    );

    // The READOUT, by name rather than by dtype: it is the widest region the
    // text states, because a vocabulary is wider than anything else in a
    // decode.
    let readout = regions
        .iter()
        .max_by_key(|(_, len, _)| *len)
        .copied()
        .expect("the text states a readout");
    let (at, len, element) = readout;
    let end = (at + len).min(read.len());
    // Row ZERO of it, not all four: three of the four are empty and the lane
    // count below is what tracks that. What this asks is whether the readout
    // that DID run produced a distribution.
    //
    // Exactly half zero would mean something else entirely -- a kernel writing
    // bf16 into a slot sized for f32 -- and that is a defect this gate found
    // and closed on its first run.
    let lane_bytes = (end - at) / 4;
    let r = census(&read[at..at + lane_bytes], element);
    assert!(
        r.finite_nonzero > r.zero,
        "the readout's first lane is {} zero to {} non-zero. Half zero is a \
         dtype disagreement; mostly zero is a readout that did not run.",
        r.zero,
        r.finite_nonzero
    );
    assert!(
        r.max_abs > 1e-4,
        "every logit is under 1e-4, so the readout accumulated nothing."
    );

    // ── the regions a launch declares and does not fill ──
    //
    // ZERO, down from FIVE. All five were the same defect the lane count below
    // names: the text picked the single-row `embed_gather_4bit`, so every lane
    // but the first was zero from statement zero onward, and the branches only
    // those lanes fed never held anything.
    //
    // The NUMBER is what made it findable. It said "five regions", the writer
    // attribution said which statements, and a prefix bisection
    // (`the_second_lane_stops_somewhere_and_this_says_where`) put the stop at
    // statement 0 -- three steps, each narrowing, none of them a guess.
    eprintln!("{} declared output(s) nothing filled", unwritten.len());
    assert!(
        unwritten.is_empty(),
        "{} statement(s) declare an output nothing filled. A statement whose \
         output stays zero is a branch of the forward pass that computes \
         nothing.\n{}",
        unwritten.len(),
        unwritten.join("\n")
    );

    // ── THE PER-TOKEN AXIS ──
    //
    // Four lanes decoded four different tokens, so the readout should hold
    // four different rows. It holds ONE: 128256 of 513024 values non-zero,
    // which is exactly one row of a 128256-wide vocabulary, and rows one
    // through three are zero all the way through.
    //
    // Measured 2026-08-10, and it is the largest remaining gap between this
    // executor and a model that answers. Nothing about it is a grid: every
    // launch states `rows 0..4`, `qmv_mb` puts the row on `grid.x` and
    // `qmv_fast_impl` reads it there (`y += tid.x * out_vec_size`), and the
    // dispatches come out `[128, 512, 1]` over `[32, 2, 1]` -- four
    // threadgroups on x, one per row. All 227 launches plan and none has an
    // empty grid.
    //
    // So the arithmetic is right and the rows still do not appear, which
    // means the next thing to look at is what the FIRST statement writes:
    // every later row being zero is what a gather that emitted one row looks
    // like four launches downstream. Reading between dispatches is the
    // instrument that settles it and this file does not have one yet.
    //
    // Pinned at one, and the number to want is four.
    let lanes = {
        let row = &read[at..end];
        let stride = row.len() / 4;
        (0..4)
            .filter(|i| {
                row[i * stride..(i + 1) * stride]
                    .chunks_exact(element)
                    .any(|c| c.iter().any(|&b| b != 0))
            })
            .count()
    };
    eprintln!("{lanes} of 4 readout lane(s) hold anything");
    assert_eq!(
        lanes, 4,
        "the per-token axis lost a lane: {lanes} of four readout rows hold \
         anything. A fire that answers one token for four is the failure this \
         gate exists to catch, because every magnitude check passes through it."
    );
}

/// **Where the second lane stops.**
///
/// The instrument the test above says it lacks: run the first `n` dispatches
/// of the fire and read the arena, for every `n`, and report the first prefix
/// after which no arena region holds anything in its second row.
///
/// A bisection rather than a guess. "Every later row is zero" is true of a
/// gather that emitted one row and of a projection that did, and four
/// launches downstream they look identical -- so the only thing that
/// distinguishes them is running fewer launches.
///
/// It found the single-row gather at statement 0 and, once that was fixed,
/// `sdpa_paged_decode` writing NEITHER row while every statement around it
/// writes both. That second finding is still open.
///
/// A report, not an assertion: what it prints is a map, and a map that fails
/// the build is a map nobody reads.
#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn the_second_lane_stops_somewhere_and_this_says_where() {
    bisect(FireClass::Decode);
}

/// The same walk over the PREFILL lane, which states a different half of the
/// kernel table: `affine_qmm_t` where a decode states `affine_qmv_fast`, and
/// a causal attention over a prefix where a decode has one key.
///
/// SIXTEEN tokens, because `Rule::Qmm` refuses a row count its tile does not
/// divide and `QMM_BMS` starts there. MLX's stages for the same prefix, for
/// comparison against what this prints:
///
///   embed 0.361, L0 attn_norm 2.207, L0 q_proj 1.320
#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn the_prefill_lane_too() {
    bisect(FireClass::Prefill);
}

fn bisect(class: FireClass) {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let (row, encoding, dg) = served(&snapshot);
    let loaded = load(&context, &snapshot, row, &encoding).expect("the checkpoint loads");
    let (facts, _metal) =
        driver_metal::model::text::facts_from(&dg, |t| loaded.tensors.contains_key(t));
    let (_, metal) =
        driver_metal::model::text::facts_from(&dg, |t| loaded.tensors.contains_key(t));

    // A decode posts four independent lanes; a prefill posts one sequence.
    let decode = class == FireClass::Decode;
    let step = if decode {
        Step {
            token_ids: &[128_000, 9906, 1917, 128_001],
            qo_indptr: &[0, 1, 2, 3, 4],
            sampling_indices: &[0, 1, 2, 3],
            ..Step::default()
        }
    } else {
        // TWO rows, which the GEMM's tile does not divide -- the guard's
        // GEMV arm is what serves them.
        Step {
            token_ids: &[128_000, 9906],
            qo_indptr: &[0, 2],
            sampling_indices: &[0],
            ..Step::default()
        }
    };
    let plan = llama_like_metal(&facts, &metal, class);
    let lowered = lower_step(&plan, &step).expect("the step lowers");

    let shape = Shape {
        layers: facts.layers,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        page_size: 16,
        pages: 64,
        element_bytes: 2,
        global_head_dim: 0,
        global_kv_heads: 0,
        full_attn_every: 0,
    };
    let pool = Pool::allocate(&context, shape).expect("a pool");
    let pages = |layer: u16, values: bool| {
        pool.layer(u32::from(layer)).map(|l| Slice {
            address: if values {
                l.v.gpu_address()
            } else {
                l.k.gpu_address()
            },
            bytes: shape.layer_bytes_at(0),
        })
    };
    let freqs = driver_metal::model::rope::frequencies(
        facts.head_dim,
        metal.rope_theta,
        (dg.rope_freq_factor > 0.0).then_some(driver_metal::model::rope::Rescale {
            factor: dg.rope_freq_factor,
            low: dg.rope_low_freq_factor,
            high: dg.rope_high_freq_factor,
            original_max: dg.rope_original_max_position as f32,
        }),
    );
    let staged = if decode {
        stage_tables(&context, &step, shape.page_size, &freqs)
    } else {
        stage_prefill(&context, &step, shape.page_size, &freqs)
    };

    let named = HashMap::new();
    let mut live = Live {
        store: Store::new(Names::mlx(), &loaded.tensors, &named),
        tables: &staged,
        shape,
        pages: &pages,
    };
    let geometry = Geometry {
        q_heads: facts.q_heads,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        rotary_dims: facts.head_dim,
        n_experts: facts.n_experts,
        experts_per_token: facts.experts_per_token,
    };

    // Every launch's OUTPUT rectangle, so a prefix can be judged by what its
    // last statement was supposed to write rather than by the whole arena.
    let outs: Vec<(usize, usize, usize, String)> = lowered
        .launches
        .iter()
        .map(|l| {
            let symbol = lowered.kernels[l.kernel as usize].clone();
            let args = &lowered.args[l.args.start as usize..l.args.end as usize];
            let last = args
                .iter()
                .rev()
                .find_map(|a| match a {
                    model_compiler::lower::Arg::Arena { at, width, bytes } => {
                        Some((*at, *width as usize, *bytes as usize))
                    }
                    _ => None,
                })
                .unwrap_or((0, 0, 0));
            (last.0, last.1, last.2, symbol)
        })
        .collect();

    // The prefixes worth running: the first twelve statements are one layer's
    // worth, which is where a per-row defect either appears or does not.
    let mut first_bad: Option<(usize, String)> = None;
    for n in 1..=12.min(lowered.launches.len()) {
        let arena = Allocation::new(
            &context,
            (lowered.arena_bytes as u64).max(1),
            "bisect arena",
        )
        .expect("an arena");
        // SAFETY: freshly allocated.
        unsafe { arena.zero(0, arena.len()).expect("it zeroes") };
        let dispatches = driver_metal::lowering::dispatch::plan(
            &lowered,
            driver_metal::lowering::dispatch::table(),
            driver_metal::lowering::executor::Frame {
                arena: Slice {
                    address: arena.gpu_address(),
                    bytes: arena.len(),
                },
            },
            geometry,
            &mut live,
        )
        .expect("the fire plans");
        let prefix = &dispatches[..n];
        let prepared = driver_metal::fire::run::prepare(&context, &lowered, prefix)
            .expect("the prefix prepares");
        pipelines
            .ensure(&context, &compiler, prefix)
            .expect("the pipelines compile");
        let mut stepper = driver_metal::device::Stepper::new(&context).expect("a stepper");
        stepper
            .run(|encoder| {
                driver_metal::bind::encode::encode(
                    encoder,
                    &prepared.table,
                    &pipelines,
                    &prepared.params,
                    prefix,
                )
            })
            .expect("the prefix runs");

        let mut read = vec![0u8; arena.len() as usize];
        // SAFETY: the command buffer retired.
        unsafe {
            let raw = core::slice::from_raw_parts(
                arena.contents().as_ptr().cast_const().cast::<u8>(),
                arena.len() as usize,
            );
            read.copy_from_slice(raw);
        }

        // The nth statement's own output, row 0 against row 1.
        let (at, width, element, symbol) = &outs[n - 1];
        let row = width * element;
        let live_row = |i: usize| {
            let (a, b) = (at + i * row, (at + (i + 1) * row).min(read.len()));
            a < b && read[a..b].iter().any(|&x| x != 0)
        };
        let (r0, r1) = (live_row(0), live_row(1));
        // The magnitude too, because "it wrote something" and "it wrote the
        // right something" are different questions and the second is the one a
        // reference can answer. MLX's numbers for the same snapshot at
        // position zero, for comparison:
        //
        //   embed 0.361, attn_norm 2.207, q_proj 1.320, v 0.413,
        //   o_proj 0.114, after attn 0.312, L0 out 20.03, L1 out 408.75
        let widest = {
            let (a, b) = (*at, (at + row).min(read.len()));
            read[a..b]
                .chunks_exact(*element)
                .map(|c| {
                    if *element == 4 {
                        f32::from_le_bytes([c[0], c[1], c[2], c[3]]).abs()
                    } else {
                        f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16).abs()
                    }
                })
                .fold(0.0f32, f32::max)
        };
        let head: Vec<String> = read[*at..(at + row).min(read.len())]
            .chunks_exact(*element)
            .take(6)
            .map(|c| {
                let v = if *element == 4 {
                    f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                } else {
                    f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)
                };
                format!("{v:.6}")
            })
            .collect();
        eprintln!(
            "  [{:>2}] {symbol:<44} @{at} row0 {} row1 {} max|v| {widest:.5} [{}]",
            n - 1,
            if r0 { "yes" } else { "NO " },
            if r1 { "yes" } else { "NO " },
            head.join(", "),
        );
        if r0 && !r1 && first_bad.is_none() {
            first_bad = Some((n - 1, symbol.clone()));
        }
    }

    // The pool, after the whole prefix: which tensor actually landed where.
    {
        let layer = pool.layer(0).expect("a layer");
        let n = shape.layer_bytes_at(0) as usize;
        // SAFETY: the command buffers retired.
        let (k, v) = unsafe {
            (
                core::slice::from_raw_parts(
                    layer.k
                        .host_span(0, n as u64)
                        .expect("the pages are addressable")
                        .as_ptr()
                        .cast_const(),
                    n,
                ),
                core::slice::from_raw_parts(
                    layer.v
                        .host_span(0, n as u64)
                        .expect("the pages are addressable")
                        .as_ptr()
                        .cast_const(),
                    n,
                ),
            )
        };
        let head = |r: &[u8]| {
            r.chunks_exact(2)
                .take(6)
                .map(|c| {
                    let x = f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16);
                    format!("{x:.6}")
                })
                .collect::<Vec<_>>()
                .join(", ")
        };
        eprintln!("  pool K[0..6] [{}]", head(k));
        eprintln!("  pool V[0..6] [{}]", head(v));
    }

    match &first_bad {
        Some((i, symbol)) => eprintln!(
            "\nThe second lane stops at statement {i}, `{symbol}`: it wrote row 0 \
             and not row 1."
        ),
        None => eprintln!("\nEvery statement in the first layer wrote both rows."),
    }
}

/// Which statement first writes a NaN, over the WHOLE fire.
///
/// # Why this is a search and not a walk
///
/// [`bisect`] re-runs a prefix per statement, which is fine for the twelve
/// that make one layer and quadratic for the four hundred that make a fire.
/// This binary-searches instead: the shortest prefix whose arena holds a NaN
/// is the statement that made it, and that is ~9 runs for a 24-layer model
/// rather than ~480.
///
/// The claim it can make is narrow and worth being exact about. A NaN in the
/// arena after `n` statements and none after `n-1` means statement `n` WROTE
/// one -- it does not say the arithmetic in statement `n` is wrong, because a
/// kernel handed a bad operand produces a bad answer honestly. What it does
/// is turn "somewhere in a 20B model" into one symbol and one layer, and
/// everything after that is reading.
///
/// Prints and passes. A checkpoint with no NaN says so and this is a no-op;
/// making it an assertion would fail every green run for the one model that
/// is not.
#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn the_first_statement_that_writes_a_nan_says_which_one_it_is() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let (row, encoding, dg) = served(&snapshot);
    let loaded = load(&context, &snapshot, row, &encoding).expect("the checkpoint loads");
    let (facts, metal) =
        driver_metal::model::text::facts_from(&dg, |t| loaded.tensors.contains_key(t));

    let step = Step {
        token_ids: &[128_000],
        qo_indptr: &[0, 1],
        sampling_indices: &[0],
        ..Step::default()
    };
    let plan = llama_like_metal(&facts, &metal, FireClass::Decode);
    let lowered = lower_step(&plan, &step).expect("the step lowers");
    let geometry = Geometry {
        q_heads: facts.q_heads,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        rotary_dims: facts.head_dim,
        n_experts: facts.n_experts,
        experts_per_token: facts.experts_per_token,
    };

    let shape = Shape {
        layers: facts.layers,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        page_size: 16,
        pages: 64,
        element_bytes: 2,
        global_head_dim: 0,
        global_kv_heads: 0,
        full_attn_every: 0,
    };
    let pool = Pool::allocate(&context, shape).expect("a pool");
    let freqs = driver_metal::model::rope::frequencies(
        facts.head_dim,
        metal.rope_theta,
        (dg.rope_freq_factor > 0.0).then_some(driver_metal::model::rope::Rescale {
            factor: dg.rope_freq_factor,
            low: dg.rope_low_freq_factor,
            high: dg.rope_high_freq_factor,
            original_max: dg.rope_original_max_position as f32,
        }),
    );
    let staged = stage_tables(&context, &step, shape.page_size, &freqs);
    let named = HashMap::new();

    // Which arena spans hold FLOATS, off the text's own declared dtypes.
    // `Arg::Arena` carries a byte WIDTH, which cannot tell an i32 from an f32,
    // and the plan can.
    let int_offsets: std::collections::HashSet<usize> = plan
        .values
        .iter()
        .enumerate()
        .filter(|(_, v)| matches!(v.dtype, model_compiler::trace::DType::I32))
        .filter_map(|(id, _)| lowered.value_offset.get(id).copied())
        .collect();
    let mut float_spans: Vec<(usize, usize, usize)> = lowered
        .args
        .iter()
        .filter_map(|a| match a {
            model_compiler::lower::Arg::Arena { at, width, bytes } => {
                Some((*at, *width as usize * *bytes as usize, *bytes as usize))
            }
            _ => None,
        })
        .filter(|(at, _, _)| !int_offsets.contains(at))
        .collect();
    float_spans.sort_unstable();
    float_spans.dedup();
    eprintln!(
        "{} float span(s), {} integer value(s) excluded",
        float_spans.len(),
        int_offsets.len()
    );

    // Run the first `n` statements and say whether the arena holds a NaN.
    let mut nan_after = |n: usize| -> bool {
        let arena = Allocation::new(
            &context,
            (lowered.arena_bytes as u64).max(1),
            "nan search arena",
        )
        .expect("an arena");
        // SAFETY: freshly allocated. Zeroed so an unwritten slot reads as a
        // zero and not as whatever the allocator had -- a stale NaN would
        // otherwise be attributed to whichever statement ran last.
        unsafe { arena.zero(0, arena.len()).expect("it zeroes") };
        let pages = |layer: u16, values: bool| {
            pool.layer(u32::from(layer)).map(|l| Slice {
                address: if values {
                    l.v.gpu_address()
                } else {
                    l.k.gpu_address()
                },
                bytes: shape.layer_bytes_at(0),
            })
        };
        let mut live = Live {
            store: Store::new(Names::mlx(), &loaded.tensors, &named),
            tables: &staged,
            shape,
            pages: &pages,
        };
        let dispatches = driver_metal::lowering::dispatch::plan(
            &lowered,
            driver_metal::lowering::dispatch::table(),
            driver_metal::lowering::executor::Frame {
                arena: Slice {
                    address: arena.gpu_address(),
                    bytes: arena.len(),
                },
            },
            geometry,
            &mut live,
        )
        .expect("the fire plans");
        let prefix = &dispatches[..n.min(dispatches.len())];
        let prepared = driver_metal::fire::run::prepare(&context, &lowered, prefix)
            .expect("the prefix prepares");
        pipelines
            .ensure(&context, &compiler, prefix)
            .expect("the pipelines compile");
        let mut stepper = driver_metal::device::Stepper::new(&context).expect("a stepper");
        stepper
            .run(|encoder| {
                driver_metal::bind::encode::encode(
                    encoder,
                    &prepared.table,
                    &pipelines,
                    &prepared.params,
                    prefix,
                )
            })
            .expect("the prefix runs");
        // SAFETY: the command buffer retired.
        let raw = unsafe {
            core::slice::from_raw_parts(
                arena.contents().as_ptr().cast_const().cast::<u8>(),
                arena.len() as usize,
            )
        };
        // FLOAT regions only. A routed FFN's arena is half INDEX buffers --
        // `route_sort` writes a permutation, a per-row expert, a per-tile
        // expert and an inverse, all integers -- and an index read as a float
        // is a NaN whenever its top bits happen to be an all-ones exponent.
        // `-1`, the sentinel a padded tile carries, is 0xFFFFFFFF, which is
        // exactly that. So a detector that reads the whole arena as floats
        // reports the sort as the first NaN of every mixture, every time, and
        // says nothing.
        float_spans.iter().any(|&(at, len, element)| {
            raw[at..(at + len).min(raw.len())]
                .chunks_exact(element)
                .any(|c| {
                    let v = if element == 4 {
                        f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                    } else {
                        f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)
                    };
                    v.is_nan()
                })
        })
    };

    let total = lowered.launches.len();
    if !nan_after(total) {
        eprintln!("the whole fire ({total} statements) is NaN-free");
        return;
    }
    // The smallest prefix that has one.
    let (mut lo, mut hi) = (0usize, total);
    while lo + 1 < hi {
        let mid = lo + (hi - lo) / 2;
        if nan_after(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let launch = &lowered.launches[hi - 1];
    let symbol = &lowered.kernels[launch.kernel as usize];
    eprintln!(
        "\nthe first NaN appears at statement {} of {total}: `{symbol}`, layer {:?}, rows {:?}",
        hi - 1,
        launch.layers,
        launch.rows
    );
    // Its neighbours, because a symbol alone does not say what it was handed.
    for i in hi.saturating_sub(4)..(hi + 2).min(total) {
        let l = &lowered.launches[i];
        eprintln!(
            "  [{i:3}]{} {} layer {:?}",
            if i == hi - 1 { " <-" } else { "   " },
            lowered.kernels[l.kernel as usize],
            l.layers
        );
    }
}

/// **The first number held to a reference.**
///
/// One token at position ZERO, and the position is chosen rather than
/// convenient: rope is the identity there (cos 0 = 1, sin 0 = 0), so
/// llama-3.2's rope SCALING -- which this text does not state -- cannot make
/// the two implementations disagree, and attention attends to exactly one key,
/// its own. What is left is every piece of arithmetic that is not
/// position-dependent: the gather, five norms a layer, q/k/v/o, the gated MLP,
/// the final norm and the readout.
///
/// The reference is MLX itself, run over the same snapshot with
/// `mx.quantized_matmul` -- the same affine codec the checkpoint was written
/// with, so a disagreement is about the DRIVER and not about who read the
/// 4-bit format correctly. Its answer for `<|begin_of_text|>` (128000) is
/// argmax **16309** with logits spanning [-4.61, 6.41].
///
/// **It agrees.** Same argmax, the same top five in the same order, every
/// logit within bf16 of MLX's, and the same span. The driver's readout is bf16
/// where MLX accumulates wider, so the tolerance is a statement about the
/// FORMAT rather than slack for a wrong answer.
///
/// Getting here cost one more defect, and it was two statements into the fire:
/// `RmsParams::w_stride` is the distance between consecutive CHANNELS of the
/// gain vector -- `ws[w_stride * i]`, one for a contiguous row, and
/// `rms.metal`'s own header says so. The statement passed the AXIS. Every norm
/// read `w[2048 * i]`, strode out of the gain vector on its second channel,
/// and multiplied by whatever followed it in the checkpoint. Channel 1 came
/// out -0.016 where MLX says +0.052: the wrong sign, from the wrong tensor.
///
/// It survived everything. The fire ran, every statement wrote every row, no
/// NaN, no infinity, 99% of the arena non-zero, and the logits were a
/// plausible-looking near-uniform distribution over 128256 tokens. Only a
/// reference could see it, which is the argument for having one.
///
/// # Why position zero and not position one
///
/// Not caution -- a KNOWN gap, and stating where it is beats pretending it is
/// not there. llama-3.2's config carries
/// `rope_scaling: {rope_type: llama3, factor: 32, low_freq_factor: 1,
/// high_freq_factor: 4, original_max_position_embeddings: 8192}`, and the text
/// passes `dsl::metal::rope` a bare theta. So the driver's rotation is the
/// unscaled one and a comparison at any position but zero would be measuring
/// that rather than the executor.
///
/// The shader for it already exists: `rope_neox_freqs_decode` takes
/// `inv_freq` as a device buffer rather than deriving frequencies from a base,
/// which is exactly the shape llama-3's rescaling wants. What is missing is
/// the table -- a load-time derivation from the config, so a `Source` beside
/// the fire tables rather than anything a text can state. That is the next
/// thing this file should be pointed at.
#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn one_token_at_position_zero_agrees_with_mlx() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let (row, encoding, dg) = served(&snapshot);
    let loaded = load(&context, &snapshot, row, &encoding).expect("the checkpoint loads");
    let (facts, metal) =
        driver_metal::model::text::facts_from(&dg, |t| loaded.tensors.contains_key(t));

    // ONE request, ONE token, position zero.
    let step = Step {
        token_ids: &[128_000],
        qo_indptr: &[0, 1],
        sampling_indices: &[0],
        ..Step::default()
    };
    let plan = llama_like_metal(&facts, &metal, FireClass::Decode);
    let lowered = lower_step(&plan, &step).expect("the step lowers");

    let shape = Shape {
        layers: facts.layers,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        page_size: 16,
        pages: 16,
        element_bytes: 2,
        global_head_dim: 0,
        global_kv_heads: 0,
        full_attn_every: 0,
    };
    let pool = Pool::allocate(&context, shape).expect("a pool");
    let pages = |layer: u16, values: bool| {
        pool.layer(u32::from(layer)).map(|l| Slice {
            address: if values {
                l.v.gpu_address()
            } else {
                l.k.gpu_address()
            },
            bytes: shape.layer_bytes_at(0),
        })
    };
    let freqs = driver_metal::model::rope::frequencies(
        facts.head_dim,
        metal.rope_theta,
        (dg.rope_freq_factor > 0.0).then_some(driver_metal::model::rope::Rescale {
            factor: dg.rope_freq_factor,
            low: dg.rope_low_freq_factor,
            high: dg.rope_high_freq_factor,
            original_max: dg.rope_original_max_position as f32,
        }),
    );
    let staged = stage_tables(&context, &step, shape.page_size, &freqs);

    let named = HashMap::new();
    let mut live = Live {
        store: Store::new(Names::mlx(), &loaded.tensors, &named),
        tables: &staged,
        shape,
        pages: &pages,
    };

    let (_, arena) = driver_metal::fire::run::run_keeping_arena(
        &context,
        &compiler,
        &mut pipelines,
        &lowered,
        Geometry {
            q_heads: facts.q_heads,
            kv_heads: facts.kv_heads,
            head_dim: facts.head_dim,
            rotary_dims: facts.head_dim,
            n_experts: facts.n_experts,
            experts_per_token: facts.experts_per_token,
        },
        &mut live,
    )
    .expect("the fire runs");

    let mut read = vec![0u8; arena.len() as usize];
    // SAFETY: the command buffer retired before the call returned.
    unsafe {
        let raw = core::slice::from_raw_parts(
            arena.contents().as_ptr().cast_const().cast::<u8>(),
            arena.len() as usize,
        );
        read.copy_from_slice(raw);
    }

    // The readout: the widest region the text states, because a vocabulary is
    // wider than anything else in a decode.
    // The text's OWN statement of where its answer is, not a guess at it.
    // This used to take the widest arena region, which was right by
    // luck: the gemma text holds TWO vocabulary-wide buffers, because
    // the logit softcap is out of place, and the tie-break picked the
    // capped one.
    let (at, width, element) = {
        let r = lowered.readout.expect("the text states an exit seam");
        (r.at, r.vocab as usize, r.bytes as usize)
    };
    let vocab = width;
    let logits: Vec<f32> = read[at..at + vocab * element]
        .chunks_exact(element)
        .map(|c| {
            if element == 4 {
                f32::from_le_bytes([c[0], c[1], c[2], c[3]])
            } else {
                f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)
            }
        })
        .collect();

    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let (lo, hi) = logits
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    eprintln!("argmax {} over {vocab} logits, span [{lo}, {hi}]", order[0]);
    for (i, &t) in order.iter().take(5).enumerate() {
        eprintln!("  top{i}: {t} logit {:.6}", logits[t]);
    }

    // MLX's answer for the same token over the same snapshot, top five, in
    // order, with the logits it gave them.
    //
    // The tokens must match exactly; the logits are compared with a bf16
    // tolerance because the driver's readout IS bf16 -- 8 mantissa bits, so
    // about 0.4% near six -- where MLX accumulates wider. A tolerance is
    // therefore a statement about the FORMAT and not slack for a wrong answer.
    const MLX: [(usize, f32); 5] = [
        (16309, 6.406_25),
        (2, 5.949_219),
        (1757, 5.859_375),
        (791, 5.781_25),
        (475, 5.601_562),
    ];
    for (i, (want, logit)) in MLX.iter().enumerate() {
        let got = order[i];
        assert_eq!(
            got, *want,
            "rank {i}: MLX says token {want} and this says {got}. At position \
             zero rope is the identity and attention has one key, so nothing \
             position-dependent can explain a difference."
        );
        let mine = logits[got];
        assert!(
            (mine - logit).abs() < 0.05,
            "token {want}: MLX logit {logit}, this {mine} — further apart than \
             bf16 explains."
        );
    }

    // The SPAN, because five agreeing logits at the top is consistent with a
    // distribution that is wrong everywhere else. MLX: [-4.613, 6.406].
    assert!(
        (hi - 6.406).abs() < 0.05 && (lo + 4.613).abs() < 0.05,
        "the distribution spans [{lo}, {hi}] where MLX spans [-4.613, 6.406]."
    );
}

/// **A two-token PREFILL, held to the same reference.**
///
/// Everything the position-zero gate could not reach: rope at a position that
/// rotates, attention over a prefix rather than one key, and the M>1 lane's
/// own symbols — a prefill states `affine_qmm_t` where a decode states
/// `affine_qmv_fast`, so this is a different half of the kernel table.
///
/// The readout is the LAST token's, which is what a prefill produces and what
/// a sampler wants. MLX's answer for `[128000, 9906]` at position 1 is argmax
/// **0** with the distribution spanning [-5.42, 18.56].
///
/// **It agrees.** Same argmax, same top five in order, span [-5.41, 18.63]
/// against MLX's [-5.42, 18.56]. So the M>1 lane is held to a reference too,
/// and between them the two gates cover both halves of the kernel table.
///
/// Three things had to land for it, and each was invisible to the other gate:
///
///   * the projection GUARD, because `qmm_t.metal` needs `M % BM == 0` and two
///     rows tile nothing — which took `region_out` on the arms and
///     `Lowering::region_outs` under them;
///   * the ROW GATHER, because a prefill's stream is one row per TOKEN and its
///     readout one per REQUEST. Without it the readout read row 0 and answered
///     the FIRST token's distribution — exactly right, for a question nobody
///     asked;
///   * `Source::RequestCount` as `Ty::InPacked`, because how many rows to
///     gather is the fire's number and it is a FIELD of a packed struct rather
///     than an operand.
#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn a_two_token_prefill_agrees_with_mlx() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let (row, encoding, dg) = served(&snapshot);
    let loaded = load(&context, &snapshot, row, &encoding).expect("the checkpoint loads");
    let (facts, metal) =
        driver_metal::model::text::facts_from(&dg, |t| loaded.tensors.contains_key(t));

    // ONE request, TWO tokens: a prefill.
    //
    // `sampling_indices: &[1]` — the LAST token's, which is what a prefill
    // produces and what a sampler wants. Asking for index 0 returns position
    // zero's distribution, and it returns it EXACTLY: 16309 at 6.40625,
    // matching the decode gate and MLX. Worth knowing, because it means the
    // readout gather is right and the difference below is the sequence.
    let step = Step {
        token_ids: &[128_000, 9906],
        qo_indptr: &[0, 2],
        sampling_indices: &[1],
        ..Step::default()
    };
    let plan = llama_like_metal(&facts, &metal, FireClass::Prefill);
    let lowered = lower_step(&plan, &step).expect("the step lowers");

    let shape = Shape {
        layers: facts.layers,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        page_size: 16,
        pages: 16,
        element_bytes: 2,
        global_head_dim: 0,
        global_kv_heads: 0,
        full_attn_every: 0,
    };
    let pool = Pool::allocate(&context, shape).expect("a pool");
    let pages = |layer: u16, values: bool| {
        pool.layer(u32::from(layer)).map(|l| Slice {
            address: if values {
                l.v.gpu_address()
            } else {
                l.k.gpu_address()
            },
            bytes: shape.layer_bytes_at(0),
        })
    };
    let freqs = driver_metal::model::rope::frequencies(
        facts.head_dim,
        metal.rope_theta,
        (dg.rope_freq_factor > 0.0).then_some(driver_metal::model::rope::Rescale {
            factor: dg.rope_freq_factor,
            low: dg.rope_low_freq_factor,
            high: dg.rope_high_freq_factor,
            original_max: dg.rope_original_max_position as f32,
        }),
    );
    // Both tokens are ONE request's, so `req_of_token` is all zeros and both
    // land in that request's first page at their own offsets. `stage_tables`
    // states one request per token, which is a decode's shape — so the tables
    // here are the prefill's own.
    let staged = stage_prefill(&context, &step, shape.page_size, &freqs);

    let named = HashMap::new();
    let mut live = Live {
        store: Store::new(Names::mlx(), &loaded.tensors, &named),
        tables: &staged,
        shape,
        pages: &pages,
    };

    let (_, arena) = driver_metal::fire::run::run_keeping_arena(
        &context,
        &compiler,
        &mut pipelines,
        &lowered,
        Geometry {
            q_heads: facts.q_heads,
            kv_heads: facts.kv_heads,
            head_dim: facts.head_dim,
            rotary_dims: facts.head_dim,
            n_experts: facts.n_experts,
            experts_per_token: facts.experts_per_token,
        },
        &mut live,
    )
    .expect("the prefill runs");

    let mut read = vec![0u8; arena.len() as usize];
    // SAFETY: the command buffer retired before the call returned.
    unsafe {
        let raw = core::slice::from_raw_parts(
            arena.contents().as_ptr().cast_const().cast::<u8>(),
            arena.len() as usize,
        );
        read.copy_from_slice(raw);
    }

    // The text's OWN statement of where its answer is, not a guess at it.
    // This used to take the widest arena region, which was right by
    // luck: the gemma text holds TWO vocabulary-wide buffers, because
    // the logit softcap is out of place, and the tie-break picked the
    // capped one.
    let (at, width, element) = {
        let r = lowered.readout.expect("the text states an exit seam");
        (r.at, r.vocab as usize, r.bytes as usize)
    };
    let logits: Vec<f32> = read[at..at + width * element]
        .chunks_exact(element)
        .map(|c| {
            if element == 4 {
                f32::from_le_bytes([c[0], c[1], c[2], c[3]])
            } else {
                f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)
            }
        })
        .collect();
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let (lo, hi) = logits
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    eprintln!("prefill argmax {} span [{lo}, {hi}]", order[0]);
    for (i, &t) in order.iter().take(5).enumerate() {
        eprintln!("  top{i}: {t} logit {:.6}", logits[t]);
    }

    const MLX: [(usize, f32); 5] = [
        (0, 18.562_5),
        (11, 18.234_375),
        (5127, 17.937_5),
        (1070, 17.468_75),
        (323, 17.296_875),
    ];
    for (i, (want, logit)) in MLX.iter().enumerate() {
        assert_eq!(
            order[i], *want,
            "rank {i}: MLX says token {want} and this says {}",
            order[i]
        );
        assert!(
            (logits[order[i]] - logit).abs() < 0.2,
            "token {want}: MLX logit {logit}, this {} — further apart than bf16 \
             explains at this magnitude.",
            logits[order[i]]
        );
    }
    assert!(
        (hi - 18.5625).abs() < 0.2 && (lo + 5.422).abs() < 0.2,
        "the distribution spans [{lo}, {hi}] where MLX spans [-5.422, 18.563]."
    );
}

/// **The rotation reaches every row of a prefill, and not only the first.**
///
/// This needs no reference, which is the point — it is a gate a laptop with
/// the checkpoint can run when nothing has captured MLX for the case.
///
/// Rope at position ZERO is the identity: `theta = scale * 0 * inv_freq` is
/// zero, so `cos` is one and `sin` is zero and the pair comes back unchanged.
/// So a prefill of the SAME token twice writes two K rows that are the same
/// projection of the same embedding, and the ONLY thing that can separate them
/// is the rotation. Row 0 must be the raw projection and row 1 must be it
/// turned by one position. Two IDENTICAL rows therefore say exactly one thing:
/// the rotation never reached row 1.
///
/// Which is what a single-row kernel over a multi-row grid does. `Rule::Rope`
/// dispatches `[rotary_dims/2, q_heads, rows]`, and a kernel that declares
/// `uint2 pos [[thread_position_in_grid]]` is never handed `pos.z` — every row
/// of the grid computes row 0's index, races on row 0's memory, and leaves
/// rows 1.. untouched.
///
/// A logit comparison cannot be trusted to catch this on its own: it reads one
/// distribution at the end of twenty-eight layers, and the constants it checks
/// have to come from somewhere. This reads the rotation's own output.
#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn a_prefill_rotates_its_second_row() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let (row, encoding, dg) = served(&snapshot);
    let loaded = load(&context, &snapshot, row, &encoding).expect("the checkpoint loads");
    let (facts, metal) =
        driver_metal::model::text::facts_from(&dg, |t| loaded.tensors.contains_key(t));

    // The SAME token twice. Same embedding, same projection, so the two K rows
    // leave the matmul bit-identical and only the rotation can part them.
    let step = Step {
        token_ids: &[9906, 9906],
        qo_indptr: &[0, 2],
        sampling_indices: &[1],
        ..Step::default()
    };
    let plan = llama_like_metal(&facts, &metal, FireClass::Prefill);
    let lowered = lower_step(&plan, &step).expect("the step lowers");

    let shape = Shape {
        layers: facts.layers,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        page_size: 16,
        pages: 16,
        element_bytes: 2,
        global_head_dim: 0,
        global_kv_heads: 0,
        full_attn_every: 0,
    };
    let pool = Pool::allocate(&context, shape).expect("a pool");
    let pages = |layer: u16, values: bool| {
        pool.layer(u32::from(layer)).map(|l| Slice {
            address: if values {
                l.v.gpu_address()
            } else {
                l.k.gpu_address()
            },
            bytes: shape.layer_bytes_at(0),
        })
    };
    let freqs = driver_metal::model::rope::frequencies(
        facts.head_dim,
        metal.rope_theta,
        (dg.rope_freq_factor > 0.0).then_some(driver_metal::model::rope::Rescale {
            factor: dg.rope_freq_factor,
            low: dg.rope_low_freq_factor,
            high: dg.rope_high_freq_factor,
            original_max: dg.rope_original_max_position as f32,
        }),
    );
    assert!(
        metal.rope_freq_table,
        "this checkpoint rescales its ladder, so the freqs lane is the one under test"
    );
    let staged = stage_prefill(&context, &step, shape.page_size, &freqs);

    let named = HashMap::new();
    let mut live = Live {
        store: Store::new(Names::mlx(), &loaded.tensors, &named),
        tables: &staged,
        shape,
        pages: &pages,
    };

    driver_metal::fire::run::run_keeping_arena(
        &context,
        &compiler,
        &mut pipelines,
        &lowered,
        Geometry {
            q_heads: facts.q_heads,
            kv_heads: facts.kv_heads,
            head_dim: facts.head_dim,
            rotary_dims: facts.head_dim,
            n_experts: facts.n_experts,
            experts_per_token: facts.experts_per_token,
        },
        &mut live,
    )
    .expect("the prefill runs");

    // Layer zero's keys. `stage_prefill` writes row r into page zero at slot r,
    // and the layer is `[pages, page_size, kv_heads * head_dim]`.
    let layer = pool.layer(0).expect("layer zero is pooled");
    let row_bytes = shape.row_bytes_at(0) as usize;
    let mut keys = vec![0u8; row_bytes * 2];
    // SAFETY: the command buffer retired before `run_keeping_arena` returned,
    // and the pool's K region is at least two rows wide.
    unsafe {
        let raw = core::slice::from_raw_parts(
            layer
                .k
                .host_span(0, keys.len() as u64)
                .expect("the pages are addressable")
                .as_ptr()
                .cast_const(),
            keys.len(),
        );
        keys.copy_from_slice(raw);
    }
    let (row0, row1) = keys.split_at(row_bytes);

    assert!(
        row0.iter().any(|b| *b != 0),
        "row zero's key is all zeros, so nothing was written and the rest of \
         this gate would pass for the wrong reason"
    );
    assert!(
        row0 != row1,
        "the two K rows are byte-identical after a prefill of the same token \
         twice. Rope at position zero is the identity, so row one should be \
         the same projection turned by one position — identical rows mean the \
         rotation never reached row one."
    );
}

/// **The same, on the ladder a deployment that does NOT rescale takes.**
///
/// The gate above and this one look identical and are not: they differ only in
/// `rope_freq_table`, which is the whole of what parts `neox_freqs_mb` from
/// `neox_mb`. Running both is what separated the two defects that produced the
/// same symptom.
///
/// The lane choice was one of them — the rescaled branch named its DECODE
/// symbol whatever the fire was. The other was underneath both lanes:
/// `Rule::Rope` took its head axis from the fire's `q_heads` while the rotation
/// is stated once per tensor, so k's launch covered thirty-two heads of an
/// eight-head buffer and `neox_mb` strided its rows by q's width. Fixing the
/// lane alone leaves this failing; fixing the axis alone leaves the one above
/// failing. Neither is visible from a single lane, so both lanes stay.
#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn a_prefill_rotates_its_second_row_on_the_base_ladder() {    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let (row, encoding, dg) = served(&snapshot);
    let loaded = load(&context, &snapshot, row, &encoding).expect("the checkpoint loads");
    let (facts, mut metal) =
        driver_metal::model::text::facts_from(&dg, |t| loaded.tensors.contains_key(t));

    // The SAME token twice. Same embedding, same projection, so the two K rows
    // leave the matmul bit-identical and only the rotation can part them.
    let step = Step {
        token_ids: &[9906, 9906],
        qo_indptr: &[0, 2],
        sampling_indices: &[1],
        ..Step::default()
    };
    metal.rope_freq_table = false;
    let plan = llama_like_metal(&facts, &metal, FireClass::Prefill);
    let lowered = lower_step(&plan, &step).expect("the step lowers");

    let shape = Shape {
        layers: facts.layers,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        page_size: 16,
        pages: 16,
        element_bytes: 2,
        global_head_dim: 0,
        global_kv_heads: 0,
        full_attn_every: 0,
    };
    let pool = Pool::allocate(&context, shape).expect("a pool");
    let pages = |layer: u16, values: bool| {
        pool.layer(u32::from(layer)).map(|l| Slice {
            address: if values {
                l.v.gpu_address()
            } else {
                l.k.gpu_address()
            },
            bytes: shape.layer_bytes_at(0),
        })
    };
    let freqs = driver_metal::model::rope::frequencies(
        facts.head_dim,
        metal.rope_theta,
        (dg.rope_freq_factor > 0.0).then_some(driver_metal::model::rope::Rescale {
            factor: dg.rope_freq_factor,
            low: dg.rope_low_freq_factor,
            high: dg.rope_high_freq_factor,
            original_max: dg.rope_original_max_position as f32,
        }),
    );
    let staged = stage_prefill(&context, &step, shape.page_size, &freqs);

    let named = HashMap::new();
    let mut live = Live {
        store: Store::new(Names::mlx(), &loaded.tensors, &named),
        tables: &staged,
        shape,
        pages: &pages,
    };

    driver_metal::fire::run::run_keeping_arena(
        &context,
        &compiler,
        &mut pipelines,
        &lowered,
        Geometry {
            q_heads: facts.q_heads,
            kv_heads: facts.kv_heads,
            head_dim: facts.head_dim,
            rotary_dims: facts.head_dim,
            n_experts: facts.n_experts,
            experts_per_token: facts.experts_per_token,
        },
        &mut live,
    )
    .expect("the prefill runs");

    // Layer zero's keys. `stage_prefill` writes row r into page zero at slot r,
    // and the layer is `[pages, page_size, kv_heads * head_dim]`.
    let layer = pool.layer(0).expect("layer zero is pooled");
    let row_bytes = shape.row_bytes_at(0) as usize;
    let mut keys = vec![0u8; row_bytes * 2];
    // SAFETY: the command buffer retired before `run_keeping_arena` returned,
    // and the pool's K region is at least two rows wide.
    unsafe {
        let raw = core::slice::from_raw_parts(
            layer
                .k
                .host_span(0, keys.len() as u64)
                .expect("the pages are addressable")
                .as_ptr()
                .cast_const(),
            keys.len(),
        );
        keys.copy_from_slice(raw);
    }
    let (row0, row1) = keys.split_at(row_bytes);

    assert!(
        row0.iter().any(|b| *b != 0),
        "row zero's key is all zeros, so nothing was written and the rest of \
         this gate would pass for the wrong reason"
    );
    assert!(
        row0 != row1,
        "the two K rows are byte-identical after a prefill of the same token \
         twice. Rope at position zero is the identity, so row one should be \
         the same projection turned by one position — identical rows mean the \
         rotation never reached row one."
    );
}

/// One request's tables: every token belongs to request zero and lands in that
/// request's page at its own offset.
///
/// `stage_tables` states one request PER TOKEN, which is a decode's shape. A
/// prefill is the other one, and getting it wrong makes every token its own
/// sequence — which attends to nothing and is exactly the answer a broken
/// attention gives.
fn stage_prefill(
    context: &Context,
    step: &Step<'_>,
    page_size: u32,
    freqs: &[f32],
) -> driver_metal::bind::tables::Staged {
    let n = step.token_ids.len() as u32;
    let positions: Vec<u32> = (0..n).collect();
    let zeros: Vec<u32> = vec![0; n as usize];
    let w_off: Vec<u32> = positions.iter().map(|p| p % page_size.max(1)).collect();
    let inv_freq: Vec<u32> = freqs.iter().map(|f| f.to_bits()).collect();
    driver_metal::bind::tables::stage(
        context,
        driver_metal::bind::tables::Frame {
            token_ids: step.token_ids,
            position_ids: &positions,
            req_of_token: &zeros,
            kv_page_indices: &[0],
            kv_page_indptr: &[0, 1],
            kv_write_page: &zeros,
            kv_write_offset: &w_off,
            rope_frequencies: &inv_freq,
            sampling_indices: step.sampling_indices,
        },
    )
    .expect("the tables stage")
}

/// **A generation, token for token, against MLX.**
///
/// The standard `device_smoke.rs` holds the retiring path to — decode a
/// sequence and compare every token — through the generic executor instead.
/// It is the last thing between `batch/dispatch_llama.rs` and the bin.
///
/// One prefill of `[BOS, "Hello"]` then three decodes, each reading the KV the
/// last one wrote. That carryover is what a single-fire gate cannot reach: an
/// append that lands one row off, a page index that does not advance, a stride
/// that is right for the first token and wrong for the second — none of them
/// show until a second fire reads what a first one wrote.
///
/// MLX's greedy continuation from the same prompt is `0, 358, 2846, 12304`,
/// computed by recomputing the WHOLE prefix at every step so that nothing is
/// carried on the reference side. A KV bug here cannot hide in a shared
/// assumption.
#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn a_generation_agrees_with_mlx_token_for_token() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let (row, encoding, dg) = served(&snapshot);
    let loaded = load(&context, &snapshot, row, &encoding).expect("the checkpoint loads");
    let (facts, metal) =
        driver_metal::model::text::facts_from(&dg, |t| loaded.tensors.contains_key(t));

    // ONE pool for the whole generation. That is the point: every fire after
    // the first reads what its predecessors wrote.
    let shape = Shape {
        layers: facts.layers,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        page_size: 16,
        pages: 16,
        element_bytes: 2,
        global_head_dim: 0,
        global_kv_heads: 0,
        full_attn_every: 0,
    };
    let pool = Pool::allocate(&context, shape).expect("a pool");
    let pages = |layer: u16, values: bool| {
        pool.layer(u32::from(layer)).map(|l| Slice {
            address: if values {
                l.v.gpu_address()
            } else {
                l.k.gpu_address()
            },
            bytes: shape.layer_bytes_at(0),
        })
    };
    let freqs = driver_metal::model::rope::frequencies(
        facts.head_dim,
        metal.rope_theta,
        (dg.rope_freq_factor > 0.0).then_some(driver_metal::model::rope::Rescale {
            factor: dg.rope_freq_factor,
            low: dg.rope_low_freq_factor,
            high: dg.rope_high_freq_factor,
            original_max: dg.rope_original_max_position as f32,
        }),
    );
    let inv_freq: Vec<u32> = freqs.iter().map(|f| f.to_bits()).collect();

    const MLX: [u32; 4] = [0, 358, 2846, 12304];
    let mut seq: Vec<u32> = vec![128_000, 9906];
    let mut got: Vec<u32> = Vec::new();

    for (turn, &want) in MLX.iter().enumerate() {
        // The first fire is the PREFILL of the prompt; every one after is a
        // decode of the last token at its own position.
        let (tokens, first): (Vec<u32>, u32) = if turn == 0 {
            (seq.clone(), 0)
        } else {
            (vec![*seq.last().expect("a sequence")], seq.len() as u32 - 1)
        };
        let n = tokens.len() as u32;
        let positions: Vec<u32> = (first..first + n).collect();
        let class = if n > 1 {
            FireClass::Prefill
        } else {
            FireClass::Decode
        };

        let step = Step {
            token_ids: &tokens,
            qo_indptr: &[0, n],
            sampling_indices: &[n - 1],
            ..Step::default()
        };
        let plan = llama_like_metal(&facts, &metal, class);
        let lowered = lower_step(&plan, &step).expect("the step lowers");

        // One request, one page list. The write destinations advance with the
        // POSITION, which is what makes each fire land after the last.
        let zeros: Vec<u32> = vec![0; n as usize];
        let w_off: Vec<u32> = positions.iter().map(|p| p % shape.page_size).collect();
        let staged = driver_metal::bind::tables::stage(
            &context,
            driver_metal::bind::tables::Frame {
                token_ids: &tokens,
                position_ids: &positions,
                req_of_token: &zeros,
                kv_page_indices: &[0],
                kv_page_indptr: &[0, 1],
                kv_write_page: &zeros,
                kv_write_offset: &w_off,
                rope_frequencies: &inv_freq,
                sampling_indices: &[n - 1],
            },
        )
        .expect("the tables stage");

        let named = HashMap::new();
        let mut live = Live {
            store: Store::new(Names::mlx(), &loaded.tensors, &named),
            tables: &staged,
            shape,
            pages: &pages,
        };
        let (_, arena) = driver_metal::fire::run::run_keeping_arena(
            &context,
            &compiler,
            &mut pipelines,
            &lowered,
            Geometry {
                q_heads: facts.q_heads,
                kv_heads: facts.kv_heads,
                head_dim: facts.head_dim,
                rotary_dims: facts.head_dim,
                n_experts: facts.n_experts,
                experts_per_token: facts.experts_per_token,
            },
            &mut live,
        )
        .expect("the fire runs");

        let mut read = vec![0u8; arena.len() as usize];
        // SAFETY: the command buffer retired before the call returned.
        unsafe {
            let raw = core::slice::from_raw_parts(
                arena.contents().as_ptr().cast_const().cast::<u8>(),
                arena.len() as usize,
            );
            read.copy_from_slice(raw);
        }

        let (at, width, element) = {
            // The text's OWN statement of where its answer is, not a guess at it.
            // This used to take the widest arena region, which was right by luck:
            // the gemma text holds TWO vocabulary-wide buffers, because the logit
            // softcap is out of place, and the tie-break picked the capped one.
            let r = lowered.readout.expect("the text states an exit seam");
            (r.at, r.vocab as usize, r.bytes as usize)
        };
        let logits: Vec<f32> = read[at..at + width * element]
            .chunks_exact(element)
            .map(|c| {
                if element == 4 {
                    f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                } else {
                    f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)
                }
            })
            .collect();
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .expect("a readout has an argmax");

        eprintln!("turn {turn}: {next} (MLX {want})");
        got.push(next);
        seq.push(next);
    }

    assert_eq!(
        got, MLX,
        "the generation diverged from MLX. A first token that agrees and a \
         second that does not is the KV carryover, which no single-fire gate \
         reaches."
    );
}

/// The REPLAYED path, over a real checkpoint, against the encoded one.
///
/// # The gap this closes
///
/// `submit` serves by replaying a recorded indirect command buffer — 424
/// dispatches issued as one `executeCommandsInBuffer`, 311× cheaper than
/// encoding them. `device_icb.rs` proves the two paths agree byte-for-byte,
/// but on **sentinel weights**: one region answering every name. The tests
/// above prove the ENCODED path agrees with MLX, but they call
/// `run_keeping_arena`, which encodes.
///
/// So each half was covered and nothing spanned both. What a real checkpoint
/// adds is **address diversity**: the weight arena, the fire tables, the
/// activation arena, the scalars, and two spans per layer for the KV pool.
/// Every one of those addresses becomes a `setKernelBuffer` on a recorded
/// command through `Regions::resolve`, and a resolution to the wrong span is
/// a kernel reading another layer's cache — silently.
#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn a_replayed_fire_over_real_weights_agrees_with_the_encoded_one() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let (row, encoding, dg) = served(&snapshot);
    let loaded = load(&context, &snapshot, row, &encoding).expect("the checkpoint loads");
    let (facts, metal) =
        driver_metal::model::text::facts_from(&dg, |t| loaded.tensors.contains_key(t));

    let step = Step {
        token_ids: &[128_000],
        qo_indptr: &[0, 1],
        sampling_indices: &[0],
        ..Step::default()
    };
    let plan = llama_like_metal(&facts, &metal, FireClass::Decode);
    let lowered = lower_step(&plan, &step).expect("the step lowers");

    let shape = Shape {
        layers: facts.layers,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        page_size: 16,
        pages: 16,
        element_bytes: 2,
        global_head_dim: 0,
        global_kv_heads: 0,
        full_attn_every: 0,
    };
    let pool = Pool::allocate(&context, shape).expect("a pool");
    let pages = |layer: u16, values: bool| {
        pool.layer(u32::from(layer)).map(|l| Slice {
            address: if values {
                l.v.gpu_address()
            } else {
                l.k.gpu_address()
            },
            bytes: shape.layer_bytes_at(0),
        })
    };
    let freqs = driver_metal::model::rope::frequencies(
        facts.head_dim,
        metal.rope_theta,
        (dg.rope_freq_factor > 0.0).then_some(driver_metal::model::rope::Rescale {
            factor: dg.rope_freq_factor,
            low: dg.rope_low_freq_factor,
            high: dg.rope_high_freq_factor,
            original_max: dg.rope_original_max_position as f32,
        }),
    );
    let staged = stage_tables(&context, &step, shape.page_size, &freqs);
    let geometry = Geometry {
        q_heads: facts.q_heads,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        rotary_dims: facts.head_dim,
        n_experts: facts.n_experts,
        experts_per_token: facts.experts_per_token,
    };
    let (at, vocab, element) = {
        let r = lowered.readout.expect("the text states an exit seam");
        (r.at, r.vocab as usize, r.bytes as usize)
    };
    let logits_of = |bytes: &[u8]| -> Vec<f32> {
        bytes[at..at + vocab * element]
            .chunks_exact(element)
            .map(|c| {
                if element == 4 {
                    f32::from_le_bytes([c[0], c[1], c[2], c[3]])
                } else {
                    f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)
                }
            })
            .collect()
    };

    // ── The ENCODED path, which the tests above hold to MLX. ──
    let named = HashMap::new();
    let mut live = Live {
        store: Store::new(Names::mlx(), &loaded.tensors, &named),
        tables: &staged,
        shape,
        pages: &pages,
    };
    let (_, arena) = driver_metal::fire::run::run_keeping_arena(
        &context,
        &compiler,
        &mut pipelines,
        &lowered,
        geometry,
        &mut live,
    )
    .expect("the encoded fire runs");
    // SAFETY: the command buffer retired before the call returned.
    let encoded = logits_of(unsafe {
        core::slice::from_raw_parts(
            arena.contents().as_ptr().cast_const().cast::<u8>(),
            arena.len() as usize,
        )
    });

    // ── The REPLAYED path, which is what serves. ──
    //
    // Every region a fire's operands may land in, registered so the recording
    // can turn an address back into a buffer. This is the list a seam builds:
    // the weights, the pool's layers, the tables — and `submit` adds the
    // arena and the scalars it leases.
    let mut regions = driver_metal::device::Regions::new();
    regions.add(&loaded.region);
    for l in 0..shape.layers {
        if let Some(layer) = pool.layer(l) {
            layer.k.register(&mut regions);
            layer.v.register(&mut regions);
        }
    }
    regions.add(&staged.region);
    regions.set_null(&staged.region);

    let mut recordings = driver_metal::fire::Recordings::new();
    let scratch = driver_metal::fire::Scratch::new();
    let mut stepper = driver_metal::device::Stepper::new(&context).expect("a stepper");
    let mut live = Live {
        store: Store::new(Names::mlx(), &loaded.tensors, &named),
        tables: &staged,
        shape,
        pages: &pages,
    };
    let replayed = {
        let mut machine = driver_metal::fire::run::Machine {
            context: &context,
            compiler: &compiler,
            pipelines: &mut pipelines,
            stepper: &mut stepper,
            scratch: &scratch,
            regions: &mut regions,
            recordings: Some(&mut recordings),
        };
        let fire = driver_metal::fire::run::submit(&mut machine, &lowered, geometry, &mut live)
            .expect("the replayed fire commits");
        machine
            .stepper
            .wait_for(fire.value)
            .expect("the replayed fire retires");
        // SAFETY: waited for above.
        logits_of(unsafe {
            core::slice::from_raw_parts(
                fire.arena.contents().as_ptr().cast_const().cast::<u8>(),
                fire.arena.len() as usize,
            )
        })
    };

    // The fire was RECORDED, not silently encoded. `submit` falls back when a
    // recording cannot be made -- right for serving, useless here, and
    // otherwise this compares the encode path with itself. Falsified: emptying
    // the region registry makes this fail with `left: 0`.
    assert_eq!(
        recordings.recorded(),
        1,
        "the fire was not recorded, so this proved nothing"
    );

    let mut order: Vec<usize> = (0..encoded.len()).collect();
    order.sort_by(|&a, &b| encoded[b].total_cmp(&encoded[a]));
    let mut replayed_order: Vec<usize> = (0..replayed.len()).collect();
    replayed_order.sort_by(|&a, &b| replayed[b].total_cmp(&replayed[a]));
    eprintln!(
        "encoded argmax {} ({:.6}), replayed argmax {} ({:.6}) over {vocab} logits",
        order[0], encoded[order[0]], replayed_order[0], replayed[replayed_order[0]]
    );

    // BIT-IDENTICAL, not within a tolerance. The two paths issue the same
    // kernels over the same buffers in the same order; the only difference is
    // who tells the GPU about them. Anything but equality is a recording that
    // bound something else.
    assert_eq!(
        encoded, replayed,
        "the replayed fire computes different logits from the encoded one -- \
         and the encoded one is what the MLX gates above hold to a reference"
    );
    assert!(
        encoded.iter().any(|v| *v != 0.0 && v.is_finite()),
        "both paths produced nothing usable, so the comparison proved nothing"
    );
}

/// Two requests' tables for a batched PREFILL: request r owns page r and its
/// tokens are positioned from zero inside it.
///
/// This is the third staging shape in this file and the one no gate had.
/// `stage_tables` states one request PER TOKEN (a decode fleet) and
/// `stage_prefill` states ONE request holding every token. A served frame is
/// routinely neither: several requests, each with several tokens.
fn stage_prefill_fleet(
    context: &Context,
    step: &Step<'_>,
    page_size: u32,
    freqs: &[f32],
) -> driver_metal::bind::tables::Staged {
    let bounds = step.qo_indptr;
    let requests = bounds.len() as u32 - 1;
    let mut positions = Vec::new();
    let mut req_of_token = Vec::new();
    let mut write_page = Vec::new();
    let mut write_offset = Vec::new();
    for r in 0..requests {
        let (lo, hi) = (bounds[r as usize], bounds[r as usize + 1]);
        for p in 0..(hi - lo) {
            positions.push(p);
            req_of_token.push(r);
            write_page.push(r);
            write_offset.push(p % page_size.max(1));
        }
    }
    // One page each, so request r's page list is exactly `[r]`.
    let page_indices: Vec<u32> = (0..requests).collect();
    let page_indptr: Vec<u32> = (0..=requests).collect();
    let inv_freq: Vec<u32> = freqs.iter().map(|f| f.to_bits()).collect();
    driver_metal::bind::tables::stage(
        context,
        driver_metal::bind::tables::Frame {
            token_ids: step.token_ids,
            position_ids: &positions,
            req_of_token: &req_of_token,
            kv_page_indices: &page_indices,
            kv_page_indptr: &page_indptr,
            kv_write_page: &write_page,
            kv_write_offset: &write_offset,
            rope_frequencies: &inv_freq,
            sampling_indices: step.sampling_indices,
        },
    )
    .expect("the tables stage")
}

/// Read one sampled row's logits out of a retired arena.
fn logits_at(read: &[u8], lowered: &model_compiler::lower::Lowered, row: usize) -> Vec<f32> {
    let r = lowered.readout.expect("the text states an exit seam");
    let (width, element) = (r.vocab as usize, r.bytes as usize);
    let at = r.at + row * width * element;
    read[at..at + width * element]
        .chunks_exact(element)
        .map(|c| {
            if element == 4 {
                f32::from_le_bytes([c[0], c[1], c[2], c[3]])
            } else {
                f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)
            }
        })
        .collect()
}

/// Everything a prefill needs that does not change between fires.
struct Rig<'a> {
    context: &'a Context,
    compiler: &'a Compiler,
    loaded: &'a driver_metal::weights::load::Loaded,
    facts: &'a LlamaLikeFacts,
    metal: &'a LlamaLikeMetalFacts,
    dg: &'a driver_metal::batch::DecodeGeometry,
}

/// Run one prefill fire and hand back every sampled row's logits.
///
/// Staged as a FLEET whatever the request count, which costs nothing: at one
/// request `stage_prefill_fleet` produces exactly `stage_prefill`'s tables —
/// positions from zero, request zero, page zero. So the comparison below runs
/// both fires through one staging path and a difference between them cannot be
/// the harness.
/// Which storage a pool under test is built on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Storage {
    /// One allocation per layer side, sized once.
    Fixed,
    /// A sparse address space per layer side, committed to the same size.
    Elastic,
}

fn prefill_logits(rig: &Rig<'_>, pipelines: &mut Pipelines, step: &Step<'_>) -> Vec<Vec<f32>> {
    prefill_logits_on(rig, pipelines, step, Storage::Fixed)
}

fn prefill_logits_on(
    rig: &Rig<'_>,
    pipelines: &mut Pipelines,
    step: &Step<'_>,
    storage: Storage,
) -> Vec<Vec<f32>> {
    let Rig {
        context,
        compiler,
        loaded,
        facts,
        metal,
        dg,
    } = *rig;
    let plan = llama_like_metal(facts, metal, FireClass::Prefill);
    let lowered = lower_step(&plan, step).expect("the step lowers");

    let shape = Shape {
        layers: facts.layers,
        kv_heads: facts.kv_heads,
        head_dim: facts.head_dim,
        page_size: 16,
        pages: 16,
        element_bytes: 2,
        global_head_dim: 0,
        global_kv_heads: 0,
        full_attn_every: 0,
    };
    // The arena outlives the pool: an elastic buffer charges its tiles back
    // on drop, and dropping the arena first would leave nothing to charge.
    let arena_for_pool = driver_metal::device::Arena::new(1024 * 1024 * 1024, 0);
    let pool = match storage {
        Storage::Fixed => Pool::allocate(context, shape).expect("a pool"),
        Storage::Elastic => {
            let mut stepper = driver_metal::device::Stepper::new(context).expect("stepper");
            Pool::allocate_elastic(context, &mut stepper, &arena_for_pool, shape)
                .expect("an elastic pool")
        }
    };
    let pages = |layer: u16, values: bool| {
        pool.layer(u32::from(layer)).map(|l| Slice {
            address: if values {
                l.v.gpu_address()
            } else {
                l.k.gpu_address()
            },
            bytes: shape.layer_bytes_at(0),
        })
    };
    let freqs = driver_metal::model::rope::frequencies(
        facts.head_dim,
        metal.rope_theta,
        (dg.rope_freq_factor > 0.0).then_some(driver_metal::model::rope::Rescale {
            factor: dg.rope_freq_factor,
            low: dg.rope_low_freq_factor,
            high: dg.rope_high_freq_factor,
            original_max: dg.rope_original_max_position as f32,
        }),
    );
    let staged = stage_prefill_fleet(context, step, shape.page_size, &freqs);

    let named = HashMap::new();
    let mut live = Live {
        store: Store::new(Names::mlx(), &loaded.tensors, &named),
        tables: &staged,
        shape,
        pages: &pages,
    };
    let (_, arena) = driver_metal::fire::run::run_keeping_arena(
        context,
        compiler,
        pipelines,
        &lowered,
        Geometry {
            q_heads: facts.q_heads,
            kv_heads: facts.kv_heads,
            head_dim: facts.head_dim,
            rotary_dims: facts.head_dim,
            n_experts: facts.n_experts,
            experts_per_token: facts.experts_per_token,
        },
        &mut live,
    )
    .expect("the prefill runs");

    let mut read = vec![0u8; arena.len() as usize];
    // SAFETY: the command buffer retired before the call returned.
    unsafe {
        let raw = core::slice::from_raw_parts(
            arena.contents().as_ptr().cast_const().cast::<u8>(),
            arena.len() as usize,
        );
        read.copy_from_slice(raw);
    }
    (0..step.sampling_indices.len())
        .map(|row| logits_at(&read, &lowered, row))
        .collect()
}

/// **A request's answer does not depend on what shares its fire.**
///
/// `device_smoke.rs`'s tombstone names this as one of two claims no current
/// gate makes, and it is the one a served frame exercises constantly: the
/// engine batches whatever is ready. Every device gate here runs either ONE
/// request holding every token or one request PER token — a decode fleet.
/// Several requests each holding several tokens, which is the shape of a
/// batched prefill, was never run.
///
/// The check needs no reference. The same prompt is prefilled alone and then
/// again beside a second, longer, unrelated request, and the two answers must
/// agree bit for bit. Anything that leaks between requests — a position that
/// counts from the fire rather than the request, a mask that lets row two see
/// row one's sequence, a page index that does not advance — moves the first
/// request's distribution, and the fire's own arithmetic is the only thing
/// that could have moved it.
#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn a_request_prefills_the_same_way_beside_another_one() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let (row, encoding, dg) = served(&snapshot);
    let loaded = load(&context, &snapshot, row, &encoding).expect("the checkpoint loads");
    let (facts, metal) =
        driver_metal::model::text::facts_from(&dg, |t| loaded.tensors.contains_key(t));

    let rig = Rig {
        context: &context,
        compiler: &compiler,
        loaded: &loaded,
        facts: &facts,
        metal: &metal,
        dg: &dg,
    };

    let alone = Step {
        token_ids: &[128_000, 9906],
        qo_indptr: &[0, 2],
        sampling_indices: &[1],
        ..Step::default()
    };
    let solo = prefill_logits(&rig, &mut pipelines, &alone);

    // The same two tokens, then a THIRD-and-fourth of someone else's. The
    // second request is deliberately a different length and different tokens,
    // so nothing about it can be mistaken for the first's.
    let together = Step {
        token_ids: &[128_000, 9906, 128_000, 3923, 374],
        qo_indptr: &[0, 2, 5],
        sampling_indices: &[1, 4],
        ..Step::default()
    };
    let batched = prefill_logits(&rig, &mut pipelines, &together);

    // The SECOND request run alone. This is the sensitive direction and the
    // reason the comparison is not one-sided: attention is causal, so nothing
    // the second request does can reach the first one's rows even if the fire
    // were staged as a single sequence. The second request is where a leak
    // shows — it is the one that could read the first's keys, count its
    // positions from the fire instead of from itself, or land in its page.
    let second_alone = Step {
        token_ids: &[128_000, 3923, 374],
        qo_indptr: &[0, 3],
        sampling_indices: &[2],
        ..Step::default()
    };
    let solo_b = prefill_logits(&rig, &mut pipelines, &second_alone);

    assert_eq!(batched.len(), 2, "the fire samples both requests");
    assert!(
        solo_b[0].iter().any(|v| v.is_finite() && *v != 0.0),
        "the solo prefills produced distributions at all"
    );
    for (which, alone, batched) in [
        ("first", &solo[0], &batched[0]),
        ("second", &solo_b[0], &batched[1]),
    ] {
        let worst = alone
            .iter()
            .zip(batched)
            .enumerate()
            .max_by(|x, y| (x.1.0 - x.1.1).abs().total_cmp(&(y.1.0 - y.1.1).abs()))
            .expect("a vocabulary");
        assert_eq!(
            alone, batched,
            "the {which} request's distribution moved when the other joined \
             its fire. Widest disagreement at token {}: alone {}, batched {}.",
            worst.0, worst.1.0, worst.1.1
        );
    }
}

/// **An elastic pool answers exactly as a fixed one does.**
///
/// The point of elastic KV is that a pool can be resized without every
/// address bound into an argument table moving. The point of THIS gate is
/// that the change is free before anyone resizes anything: the pages are in
/// placement heaps behind a sparse buffer instead of in one allocation, and
/// nothing above them may be able to tell.
///
/// A weaker check — that the fire runs, or that the activations are finite —
/// would pass over a pool whose rows landed a page apart, because attention
/// over the wrong keys is still finite. So the comparison is bit-for-bit
/// against the same fire on a fixed pool: same weights, same tokens, same
/// staging, one storage difference. Real weights matter here for the same
/// reason they matter to the rope gates — a synthetic pool of zeros gives the
/// same answer whatever it is read through.
///
/// **What it does not reach.** Five tokens over a page size of sixteen touch
/// the first page of each layer and no other, so a commit that covered only
/// the front of every buffer would pass this — measured, by halving it. What
/// fails it is a pool with nothing mapped, which is the shape the mistake
/// takes when a commit is skipped rather than shortened. The per-page
/// arithmetic is gated in `device_elastic.rs` instead, where a span past what
/// is mapped is refused rather than served.
#[test]
#[ignore = "needs PIE_METAL_SMOKE_CHECKPOINT; run with --include-ignored --test-threads=1"]
fn an_elastic_pool_answers_exactly_as_a_fixed_one_does() {
    let Some(snapshot) = snapshot() else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let Ok(context) = Context::new() else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let compiler = Compiler::new(&context).expect("a compiler");
    let mut pipelines = Pipelines::new(kernels_dir());

    let (row, encoding, dg) = served(&snapshot);
    let loaded = load(&context, &snapshot, row, &encoding).expect("the checkpoint loads");
    let (facts, metal) =
        driver_metal::model::text::facts_from(&dg, |t| loaded.tensors.contains_key(t));

    let rig = Rig {
        context: &context,
        compiler: &compiler,
        loaded: &loaded,
        facts: &facts,
        metal: &metal,
        dg: &dg,
    };

    // Two requests of different lengths, so the fire reads pages belonging to
    // more than one sequence: a storage seam that only showed up past the
    // first request's rows would survive a single-sequence check.
    let step = Step {
        token_ids: &[128_000, 9906, 128_000, 3923, 374],
        qo_indptr: &[0, 2, 5],
        sampling_indices: &[1, 4],
        ..Step::default()
    };

    let fixed = prefill_logits_on(&rig, &mut pipelines, &step, Storage::Fixed);
    let elastic = prefill_logits_on(&rig, &mut pipelines, &step, Storage::Elastic);

    assert_eq!(
        fixed.len(),
        elastic.len(),
        "the same fire sampled a different number of rows"
    );
    for (row, (want, got)) in fixed.iter().zip(&elastic).enumerate() {
        assert_eq!(
            want.len(),
            got.len(),
            "row {row}: the two pools produced different vocabulary widths"
        );
        let differ = want
            .iter()
            .zip(got)
            .enumerate()
            .find(|(_, (a, b))| a.to_bits() != b.to_bits());
        assert!(
            differ.is_none(),
            "row {row}: an elastic pool changed the answer. \
             {differ:?} — the pages are the only thing that differs between \
             these two runs, so a logit that moved means the fire read \
             different bytes: heaps that do not sit where the sparse buffer \
             says, a commit that did not reach every layer, or a zero that \
             did not clear what a fixed allocation happened to have clear"
        );
    }
}
