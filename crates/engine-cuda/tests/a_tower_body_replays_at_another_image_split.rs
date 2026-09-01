//! **A TWO-UNIT ARTIFACT ARMS BODIES FOR BOTH OF ITS CAPTURE UNITS, AND A
//! VISION FIRE REPLAYS AT AN IMAGE GEOMETRY THE CAPTURE NEVER SAW** — the
//! multi-unit bodies wave's gate.
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 --release \
//!     --test a_tower_body_replays_at_another_image_split -- --ignored --nocapture
//! ```
//!
//! # What changed, in one paragraph
//!
//! A `record::BodyKey` named ONE lattice point, and a fire that launches two
//! execs has one per capture unit — the token rectangle's and the patch
//! rectangle's. So every vision SKU was refused the bodies path outright, by
//! five separate sites all reading the compiler's `fold_refused`, and every
//! fire of every tower load walked eagerly for the life of the load. The key
//! carries a `record::AxisKey` per unit now — that unit's own bucket and its
//! own ladder over its own seriation — so a two-unit composition is NAMED
//! rather than refused, which is multimodal §1's "6 + 6, not 6 x 6" instead
//! of the product it declines. Everything else was already per-unit:
//! `record::Cut` has carried its unit since tier 2, a body has been
//! `Vec<GraphExec>` since it was written, and the replay loop submits one
//! stretch at a time.
//!
//! # The claims
//!
//! ```text
//! (a) THE BOOT ARMS TOWER KEYS. `armed_at_load` moves on a vision load, and
//!     the boot line's `tower n/m` field is where an operator reads how many
//!     — printed under `--nocapture`, beside the four token kinds
//! (b) A VISION FIRE REPLAYS AT AN IMAGE GEOMETRY THE ARMING NEVER
//!     SYNTHESIZED. The arming pass fires each PATCH RUNG whole (128 patch
//!     rows for the 128 rung); this fires a 10x10 image, which is ONE HUNDRED
//!     patch rows landing on that same rung. `hits` moves and `captures` does
//!     not, so the exec that ran was captured at 128 patch rows and served a
//!     fire with 100 — the patch axis's live-geometry seat retiring 28 rows a
//!     token-axis argument would have had nothing to say about
//! (c) AND SO DOES A SECOND ONE, ON A DIFFERENT RUNG. An 8x8 image is 64
//!     patch rows and keys to the 64 rung, which is a different
//!     `record::AxisKey` and therefore a different body. Two rungs replaying
//!     is what says the patch bucket is a real coordinate of the key rather
//!     than a number that happens to be constant
//! (d) AND AT A DIFFERENT TOKEN SPLIT OF ONE KEY. The same image with one
//!     more leading text row is the same token bucket and the same patch
//!     rung — one key, two splits — which is the tier-1 claim, asked on the
//!     composition that could not reach it before
//! (e) AND EVERY ONE OF THEM ANSWERS, BIT FOR BIT, WHAT THE EAGER WALK OF THE
//!     SAME FIRE ANSWERS. A replay that answered different logits would be
//!     worse than no replay
//! (f) AND THE SEAL HELD: `captures` does not move once past the boot, so the
//!     serving path recorded nothing
//! ```
//!
//! (b) is the load-bearing one and it is why the geometry moves. A body
//! promises to serve every fire of its key whatever the ROWS do, and the key
//! carries no row counts on either axis — so the honest test of a tower body
//! is not "does the fire that captured it replay" but "does a fire with FEWER
//! PATCH ROWS than the capture had replay, and answer what the eager walk
//! answers". Both halves of the second unit are on trial there: the arena's
//! patch column, which is carved at the patch BUCKET for a bodied fire (a
//! column cut at the capturing fire's own patch rows would be shorter than
//! the grid that addresses it), and the tower launches' grids, which are
//! issued at the patch bucket and retired by the same `[rows, row_offset,
//! lanes, lane_offset]` seat the trunk's are.
//!
//! # What this file is NOT
//!
//! It is not `a_vision_sku_loads_and_fires_an_image.rs`, which asks whether
//! the LOAD happens and the launches run, eagerly, and owns the media path's
//! own three refusals. This one takes that load as given and asks one
//! question about the router. If the vision gate is red, everything here is
//! about the wrong thing.
//!
//! # G4, which this file also stands under
//!
//! The wave's oath is that A TEXT-ONLY LOAD IS BYTE-FOR-BYTE UNMOVED — the
//! same keys, the same `Eq`, the same `Hash`, the same `Display`, the same
//! admissibility table, the same arena carve. `BodyKey::patch` is `None` on
//! every one-unit artifact and every surface reads through it unchanged;
//! `record`'s own key tests assert that half, where it is checkable without a
//! device. What this file adds is the other direction: the two-unit load that
//! the invariant was kept FOR actually arms and actually replays.

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, PoisonError};

use engine_cuda::{Boot, Graphs, Media, Seated, Shell};
use model_compiler::Budget;
use model_dsl::Platform;

/// The catalog's vision row — the same one
/// `a_vision_sku_loads_and_fires_an_image.rs` names, and reached the same way:
/// by NAME, because a stock qwen35 import answers the text-only SKU first.
const SKU: &str = "qwen35-d0.8b-vision-bf16-kv-bf16";

/// `RuntimeInput::Patches` is `[Patches, 1536]`, read off the trace's
/// declarations rather than guessed.
const PATCH_WIDTH: usize = 1536;

/// `PatchEmbedRows` is `[Patches, 4]`.
const TAPS: usize = 4;

/// `visual.pos_embed` is `[2304, 768]`, so the stored grid is 48 x 48.
const GRID_SIDE: usize = 48;

/// `spatial_merge_size`.
const MERGE: usize = 2;

/// **THE TWO IMAGE GEOMETRIES, AND THE ARITHMETIC THAT PICKED THEM.**
///
/// The derived patch ladder for this SKU at `max_tokens = 256` is
/// `buckets = [64, 128, 256]`, `max_patches = 256`, `max_images = 4` — which
/// `a_vision_sku_loads_and_fires_an_image.rs` asserts and this file therefore
/// does not re-derive. The token lattice beside it is
/// `api::default_lattice(256)` = `[8, 16, 32, 64, 128, 256]`.
///
/// ```text
///          patches   patch rung   soft tokens   token rows   token bucket
///  8 x 8        64           64            16           20             32
/// 10 x 10      100          128            25           29             32
/// ```
///
/// So the two fires land on ONE token bucket and TWO patch rungs, which is
/// exactly the separation claim (c) wants: the only coordinate that differs is
/// the one this wave added. And the 10 x 10 fire lands on the 128 rung with a
/// hundred rows rather than a hundred and twenty-eight, which is claim (b).
///
/// **BOTH RUNGS ARE ARMABLE AND THAT IS ARITHMETIC, NOT LUCK.**
/// `Shell::arm_bodies`' tower arm skips a `(token bucket, patch rung)` pair
/// whose placeholder rows do not fit: the tower folds `MERGE * MERGE` patch
/// rows into one soft token and the merge scatters those onto token rows of
/// the same lane, so the rung owes `rung / 4` rows and the bucket has to hold
/// them. 64 owes 16 and 128 owes 32, and the token bucket both fires round to
/// is 32.
const SMALL: usize = 8;

const LARGE: usize = 10;

/// Where the first soft token lands, per fire — the leading text rows. Claim
/// (d) is the two of them on ONE image geometry: 2 and 3 leading rows are 29
/// and 30 token rows, which is one bucket and therefore one key.
const FIRST_SPLIT: i32 = 2;

const SECOND_SPLIT: i32 = 3;

/// The trailing text rows every fire carries, so that the placeholders are
/// never the tail of the fire — a soft-token run that ends at the last row is
/// a fire whose scatter cannot be distinguished from a fire whose scatter ran
/// one row short.
const TRAILING: usize = 2;

/// One CUDA device, one test at a time. Every gate in this suite that loads a
/// shell takes it, for the reason `bodies_gate.rs` states: two shells on one
/// card is a memory census about neither of them.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

fn snapshot() -> Option<PathBuf> {
    let home = std::env::var_os("PIE_HOME")
        .map_or_else(|| dirs_home().join(".pie"), PathBuf::from);
    let dir = home.join("models").join("Qwen--Qwen3.5-0.8B");
    dir.join("archive.zt").is_file().then_some(dir)
}

fn dirs_home() -> PathBuf {
    std::env::var_os("HOME").map_or_else(|| PathBuf::from("/root"), Into::into)
}

fn to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if (b & 0x7fff_ffff) > 0x7f80_0000 {
        return ((b >> 16) | 0x0040) as u16;
    }
    let rounding = 0x7fff + ((b >> 16) & 1);
    (b.wrapping_add(rounding) >> 16) as u16
}

/// **THE MERGE-BLOCK-MAJOR PATCH ORDER**, transcribed from
/// `multimodal::QwenImageConfig::qwen_patchify_hwc`: block row, block column,
/// then row and column inside the block. Every patch-axis stream a fire
/// carries is laid in this order, which is the statute the folds ask for.
fn patch_order(h: usize, w: usize) -> Vec<(usize, usize)> {
    let (bh, bw) = (h / MERGE, w / MERGE);
    let mut out = Vec::with_capacity(h * w);
    for ih_blk in 0..bh {
        for iw_blk in 0..bw {
            for ih in 0..MERGE {
                for iw in 0..MERGE {
                    out.push((ih_blk * MERGE + ih, iw_blk * MERGE + iw));
                }
            }
        }
    }
    out
}

/// `_interpolation_axis_taps_weights`, bilinear with `align_corners = True` —
/// what `Qwen3_5VisionModel.__init__` states, transcribed.
fn axis_taps(index: usize, size: usize, side: usize) -> ([usize; 2], [f32; 2]) {
    #[allow(clippy::cast_precision_loss)]
    let src = index as f32 * (side as f32 - 1.0) / (size.saturating_sub(1).max(1)) as f32;
    let floor = src.floor();
    let mut taps = [0usize; 2];
    let mut weights = [0f32; 2];
    for (t, offset) in [0f32, 1f32].into_iter().enumerate() {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let tap = (floor as i64 + offset as i64).clamp(0, side as i64 - 1) as usize;
        taps[t] = tap;
        weights[t] = (1.0 - (src - floor - offset).abs()).max(0.0);
    }
    (taps, weights)
}

/// One lane's images, as the submission carries them.
struct Shot {
    rows: Vec<u32>,
    patches: Vec<u8>,
    routes: Vec<i32>,
    positions: Vec<i32>,
    embed_rows: Vec<i32>,
    embed_weights: Vec<f32>,
}

/// A synthetic image of `side x side` patches: deterministic values, real
/// geometry — the vision gate's `one_image`, parameterized on the grid because
/// this file's whole subject is the grid moving.
///
/// `first_placeholder` is the lane-relative token row the first soft token
/// lands on; the rows past the fold say `-1`, which is the drop sentinel
/// `layout.scatter_live_rows` reads.
fn one_image(h: usize, w: usize, first_placeholder: i32) -> Shot {
    let order = patch_order(h, w);
    let patches_n = h * w;
    let soft = patches_n / (MERGE * MERGE);
    let mut patches = Vec::with_capacity(patches_n * PATCH_WIDTH * 2);
    let mut positions = Vec::with_capacity(patches_n * 3);
    let mut embed_rows = Vec::with_capacity(patches_n * TAPS);
    let mut embed_weights = Vec::with_capacity(patches_n * TAPS);

    for (at, &(row, col)) in order.iter().enumerate() {
        for lane in 0..PATCH_WIDTH {
            #[allow(clippy::cast_precision_loss)]
            let v = (((at * PATCH_WIDTH + lane) % 251) as f32 / 251.0) - 0.5;
            patches.extend_from_slice(&to_bf16(v).to_le_bytes());
        }
        // `(t, h, w)`: a still image has no time axis, and the tower states
        // `sections[0] == 0`, so nothing reads the first column.
        positions.extend_from_slice(&[0, row as i32, col as i32]);

        let (h_taps, h_w) = axis_taps(row, h, GRID_SIDE);
        let (w_taps, w_w) = axis_taps(col, w, GRID_SIDE);
        for a in 0..2 {
            for b in 0..2 {
                embed_rows.push((h_taps[a] * GRID_SIDE + w_taps[b]) as i32);
                embed_weights.push(h_w[a] * w_w[b]);
            }
        }
    }

    let mut routes = vec![-1i32; patches_n];
    for (j, route) in routes.iter_mut().take(soft).enumerate() {
        *route = first_placeholder + j as i32;
    }

    Shot {
        rows: vec![patches_n as u32],
        patches,
        routes,
        positions,
        embed_rows,
        embed_weights,
    }
}

/// One fire of this gate: an image at `side x side`, `leading` text rows in
/// front of its soft tokens and [`TRAILING`] behind them.
struct Fire {
    tokens: Vec<u32>,
    shot: Shot,
    /// **THIS FIRE'S OWN WORD, FROM THE MODEL'S OWN `Classify`** (decision
    /// #18). A shell may not invent a lane's word and neither may a gate — and
    /// it is per fire rather than per file because the word is a function of
    /// the ROW COUNT as well as of the media bit: `qo_one` is what separates a
    /// decode lane from a prefill one, so one word reused across two row counts
    /// would be a lane whose class and whose rows disagree.
    word: u64,
}

fn fire_of(classify: model_dsl::ClassifyFn, side: usize, leading: i32, seed: u32) -> Fire {
    let soft = side * side / (MERGE * MERGE);
    let rows = leading as usize + soft + TRAILING;
    Fire {
        tokens: (0..rows as u32).map(|i| seed + i).collect(),
        shot: one_image(side, side, leading),
        word: classify(&model_dsl::Request::new(rows as u32, false).with_media(true)),
    }
}

/// Fire it, and answer the logits.
///
/// **THE SLOT IS OPENED FIRST, WHICH IS NOT HOUSEKEEPING.** A fire appends to
/// its lane's kv cache, so the second fire of one slot attends a history the
/// first one wrote — and this file's whole claim is a bit-for-bit diff between
/// an eager fire and a replayed one of the SAME composition. Two fires that
/// differ in their cache differ in their logits for a reason that has nothing
/// to do with the router. `Shell::open` is what makes each one the first.
fn fire_it(shell: &mut Shell, fire: &Fire, slot: u32) -> Vec<f32> {
    shell.open(slot).expect("the slot opens");
    let media = [Media {
        lane: 0,
        rows: &fire.shot.rows,
        patches: &fire.shot.patches,
        routes: &fire.shot.routes,
        positions: &fire.shot.positions,
        token_positions: &[],
        embed_rows: &fire.shot.embed_rows,
        embed_weights: &fire.shot.embed_weights,
    }];
    let lane = engine_cuda::Lane {
        slot,
        word: fire.word,
        tokens: &fire.tokens,
    };
    let mut scores = Vec::new();
    let out = shell
        .fire_media(&[Seated::of(lane)], &[], &media, &mut scores)
        .expect("a vision fire answers");
    out.into_iter().next().expect("one lane, one readout")
}

/// The loaded vision shell in the TIERED mode, and the media word its lanes
/// carry — or `None` and a sentence saying what was missing.
///
/// **`Graphs::On` AT LOAD, WHICH IS WHAT MAKES CLAIM (a) SAYABLE.**
/// `Shell::arm_bodies` refuses to run at all unless the mode records, so a
/// load that stated `Off` and turned the mode on afterwards would mint its
/// bodies from TRAFFIC and this file would be a different test.
fn ready(what: &str) -> Option<(Shell, model_dsl::ClassifyFn)> {
    let Some(checkpoint) = snapshot() else {
        eprintln!("skipping {what}: no Qwen3.5-0.8B artifact in $PIE_HOME/models");
        return None;
    };
    let trace = models::trace_of(SKU).expect("the catalog ships the vision row")(Platform::Cuda);

    // The ladder the ENGINE derives, from the same function the engine door
    // calls — a gate that stated its own would prove that SOME ladder serves.
    let budgets = engine::load::Budgets {
        max_tokens: 256,
        max_lanes: 4,
        ..engine::load::Budgets::default()
    };
    let ladder = engine_cuda::api::patch_ladder(&trace, &budgets)
        .expect("a plan that states patch rows derives a ladder");

    let container: PathBuf = checkpoint.join("archive.zt");
    let source = ztensor_compat::index(&container).expect("the artifact opens");
    let Ok(contract) = models::import_of(SKU).expect("the catalog ships an import")(&source) else {
        eprintln!(
            "skipping {what}: {container:?} holds no `model.visual.*` planes — the stored \
             artifact was imported as the text-only SKU"
        );
        return None;
    };
    drop(source);

    // **THE MODEL'S OWN `Classify`, CARRIED RATHER THAN CALLED** (decision
    // #18). A shell may not invent a lane's word and neither may a gate: the
    // bit that says "this lane submitted an image" is the model's business,
    // and `Fault::MaskWord`'s sibling refusals are what would catch a guess.
    // Handed back so that each fire computes its word from its own row count
    // — see [`Fire::word`].
    let classify = models::classify_of(SKU).expect("the catalog ships a classify");

    let shell = Shell::load(Boot {
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: &container,
        budget: Budget::new(budgets.max_lanes, budgets.max_tokens),
        patches: Some(ladder),
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        ordinal: 0,
        graphs: Graphs::On,
        knobs: engine_cuda::Knobs::default(),
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        weight_cache_dir: None,
    })
    .expect("the vision shell loads");
    Some((shell, classify))
}

// ── the gate ─────────────────────────────────────────────────────────────

/// **THE SIX CLAIMS, ON ONE LOAD.**
///
/// One load and two modes, for `bodies_gate.rs`'s reason: two loads would
/// differ in their weight residency, their arena carve and their autotuner
/// state, and the diff would be about those instead of about the router.
///
/// The eager arm runs FIRST and is warmed, because the dense tuner tunes a
/// GEMM shape on its second sighting and a cold arm against a warm one is two
/// tactic ladders rather than two routers.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local Qwen3.5-0.8B artifact; run it with `-- --ignored`"]
fn a_tower_body_replays_at_another_image_split_and_says_what_the_walk_said() {
    let _serial = serialized();
    let Some((mut shell, classify)) = ready("the tower body gate") else {
        return;
    };

    // **CLAIM (a): THE BOOT ARMED SOMETHING.** Read before any fire, so
    // nothing here can be traffic's. The boot line printed above this — under
    // `--nocapture` — is where the TOWER count is spelled out beside the four
    // token kinds; what a test can assert without parsing stderr is that the
    // enumeration seated keys at all, and claims (b) and (c) below are what
    // prove two of them were tower keys, which is strictly stronger than
    // reading a number off a line.
    let armed = shell.body_stats();
    eprintln!("at boot: {armed}");
    assert!(
        armed.tally.armed_at_load >= 1,
        "the boot armed nothing at all on a two-unit artifact, so nothing below \
         is about arming. Until the multi-unit bodies wave this was the EXPECTED \
         reading — `Shell::arm_bodies` returned on `CompiledModel::fold_refused` \
         before it enumerated anything — and a zero here now means that clause, \
         or the one it became (`Shell::keyable_units`), is turning the load away: \
         {armed}"
    );

    // The three fires: two image geometries and two token splits of one of
    // them. The word seeds differ so that no two fires are the same token
    // stream, which is what keeps claim (e) from comparing a fire to itself.
    let large_first = fire_of(classify, LARGE, FIRST_SPLIT, 100);
    let large_second = fire_of(classify, LARGE, SECOND_SPLIT, 300);
    let small = fire_of(classify, SMALL, FIRST_SPLIT, 500);

    // ── the golden, and it is the eager walk of each fire ────────────────
    shell.set_mode(Graphs::Off);
    for _ in 0..2 {
        let _ = fire_it(&mut shell, &large_first, 0);
        let _ = fire_it(&mut shell, &large_second, 1);
        let _ = fire_it(&mut shell, &small, 2);
    }
    let eager_large_first = fire_it(&mut shell, &large_first, 0);
    let eager_large_second = fire_it(&mut shell, &large_second, 1);
    let eager_small = fire_it(&mut shell, &small, 2);

    assert!(
        eager_large_first.iter().all(|logit| logit.is_finite()),
        "the eager arm answered a non-finite logit, so the golden is not one"
    );
    assert_ne!(
        eager_large_first, eager_small,
        "two different image geometries answered identical logits, so the tower \
         is not conditioning the trunk and every claim below is about nothing"
    );

    // ── the tiered router, on the bodies the BOOT captured ───────────────
    let before = shell.body_stats();
    shell.set_mode(Graphs::On);
    let bodied_large_first = fire_it(&mut shell, &large_first, 0);
    let bodied_large_second = fire_it(&mut shell, &large_second, 1);
    let bodied_small = fire_it(&mut shell, &small, 2);
    let after = shell.body_stats();
    eprintln!("after three vision fires: {after}");

    // **CLAIMS (b), (c) AND (d): ALL THREE REPLAYED.**
    //
    // Three hits over two patch rungs and two token splits. The 10x10 fires
    // bring 100 patch rows to a body captured at 128 — the rung's own whole
    // count, which is what `BodySynth::Tower` synthesizes — so a hit there is
    // the patch axis's seat retiring 28 rows the graph was gridded for. The
    // 8x8 fire keys to a different `record::AxisKey` entirely, which is what
    // says the patch bucket is a coordinate of the key rather than a constant.
    assert!(
        after.tally.hits >= before.tally.hits + 3,
        "three vision fires produced fewer than three hits. Each of them keys to \
         a body the boot armed — two patch rungs, one token bucket, two splits \
         of one of them — and the numbers in that key are ceilings rather than \
         measurements, so no in-key fire can outgrow the grids the capture \
         froze: {after}"
    );

    // **CLAIM (f): THE SEAL HELD.** Past `Shell::arm_bodies` the serving path
    // records nothing at all, so a moving capture counter is a fire that
    // reached a key the boot did not arm and minted one on somebody's critical
    // path — which a sealed map is supposed to make impossible.
    assert_eq!(
        after.tally.captures, before.tally.captures,
        "the serving path captured something. The map is sealed at the end of \
         `Shell::arm_bodies`, so a fire whose key holds no body keeps its eager \
         numbers and is counted (`sealed_declines`) rather than warming toward a \
         capture: {after}"
    );
    assert_eq!(
        after.tally.refusals, before.tally.refusals,
        "a vision composition was refused admission by name. Since the multi-unit \
         bodies wave the only thing left to refuse a key outright is a widening \
         that left nothing captured, and a tower artifact is not one: {after}"
    );

    // **CLAIM (e): AND EACH ONE SAID WHAT THE WALK SAID.** Bit for bit — the
    // same kernels run over the same rows through the same page table, so a
    // difference of one ULP is a bug and not noise.
    for (what, eager, bodied) in [
        ("10x10 at split 2", &eager_large_first, &bodied_large_first),
        ("10x10 at split 3", &eager_large_second, &bodied_large_second),
        ("8x8 at split 2", &eager_small, &bodied_small),
    ] {
        assert_eq!(
            eager.len(),
            bodied.len(),
            "{what}: the replay answered a different readout width"
        );
        let moved = eager
            .iter()
            .zip(bodied.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            moved, 0,
            "{what}: {moved} of {} logits differ between the eager walk and the \
             replay. A tower body's execs are captured at the PATCH bucket and \
             retired by the patch axis's own live-geometry seat; a difference \
             here is a launch reading rows the seat did not retire, or an arena \
             column carved at a fire's patch rows instead of at the key's",
            eager.len(),
        );
    }
}
