//! **THE DEPLOYMENT DOOR: A VISION SKU BOOTS AND FIRES.**
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 \
//!     --test a_vision_sku_loads_and_fires_an_image -- --ignored --nocapture
//! ```
//!
//! `Boot::patches` was a literal `None` at the engine door, which made every
//! vision SKU a load that could not happen — a plan stating `Dim::Patches`
//! against no ladder is `model_compiler::Error::Unsized`, named at the door,
//! and `qwen_3`'s catalog comment says so in as many words. This file is the
//! evidence that the door is open, and it opens it the way a deployment does:
//!
//! ```text
//! (a) THE DERIVED LADDER IS THE ONE THAT LOADS. The gate does not invent a
//!     `PatchLadder`; it calls `engine_cuda::api::patch_ladder` — the same
//!     function `Engine::load` calls — against the text's own trace and the
//!     contract's default budgets, and boots on what comes back. A gate that
//!     stated its own ladder would prove that SOME ladder serves
//! (b) AND THE FIRE CARRIES AN IMAGE. One synthetic 8x8-patch image through
//!     `fire_media`, whose logits are finite and the vocabulary's width
//! (c) AND THE TOWER CONDITIONED THE TRUNK: the same lane fired with no media
//!     answers different logits, because the image-placeholder rows keep
//!     their token embedding when nothing scatters over them
//! (d) AND A TEXT-ONLY FIRE OF THE SAME SHELL STILL WORKS — the axis-empty
//!     fire the whole design is arranged around
//! ```
//!
//! **WHAT THIS IS NOT.** M-1's five gates — the text-only regression, the
//! polymorphic mixed fire, width invariance, a sensible caption, the three
//! refusals — are the text lane's, and they want a real image and a
//! tokenizer. This one wants to know the LOAD happens and the launches run.
//!
//! # The three stops this gate found, and what each one was
//!
//! It was written red on purpose and it named where it stopped. All three are
//! closed, and none of them was the ladder or the media path:
//!
//! ```text
//! 1. attention.plan_prefill: "the host qo_indptr holds 0 entries for a batch
//!    of 1". `hoist` puts every prepare region in front of every capture
//!    region, so `partition` stamped them "the unit open where it stands" --
//!    unit 0 by POSITION. On a two-unit plan unit 0 is the TOWER, so both
//!    `Windows::of` and `walk` cut a hoisted `plan_prefill` at the PATCH
//!    table and left its `indptr_host` empty. A prepare region names no exec;
//!    it takes the primary unit now
//! 2. `value N reads where this fire's tower rows land` -- `PatchRoutes`,
//!    unbound on a text-only fire. THE EMBED MERGE IS A TRUNK-UNIT NODE, so
//!    the walk's zero-row skip reads its TOKEN window, which is full. §1's
//!    `media` fact is what keeps it out, and the text lane landed the bit
//!    against §15's contract
//! 3. "the activation's rows are the rows the result lands" -- a tower GEMM
//!    whose DESTINATION had zero rows. `model_exec::store::arena::rect` sized
//!    every rectangle with `FireRows::text_only`, on the reading that a tower
//!    rectangle would be asked for through `Composition::value_window`. It is
//!    not: `Run::whole` resolves every arena value through the slot table,
//!    tower rectangles included, so every one of them was sized at nothing --
//!    which does not fault, it computes
//! ```
//!
//! Two of the three are the same shape of mistake: a number that is true on a
//! one-axis plan (unit 0 is the token unit; a fire has no patch rows) written
//! down as if it were true always.
//!
use engine_cuda::{Boot, Media, Seated, Shell};
use model_compiler::Budget;
use model_dsl::Platform;

/// The catalog's vision row. Last in `IMPORTS` on purpose, so a stock qwen35
/// import still answers the text-only SKU; a gate reaches it by NAME.
const SKU: &str = "qwen35-d0.8b-vision-bf16-kv-bf16";

/// The tower's own numbers, read off the trace's declarations rather than
/// guessed: `RuntimeInput::Patches` is `[Patches, 1536]` and
/// `PatchEmbedRows` is `[Patches, 4]`.
const PATCH_WIDTH: usize = 1536;

const TAPS: usize = 4;

/// `visual.pos_embed` is `[2304, 768]`, so the stored grid is 48 x 48.
const GRID_SIDE: usize = 48;

/// `spatial_merge_size`.
const MERGE: usize = 2;

/// The synthetic image: 8 x 8 patches, which is a whole number of 2 x 2 merge
/// blocks and lands exactly on the patch ladder's first rung.
const GRID_H: usize = 8;

const GRID_W: usize = 8;

const PATCHES: usize = GRID_H * GRID_W;

/// Its soft tokens, after the 2 x 2 merge.
const SOFT_TOKENS: usize = PATCHES / (MERGE * MERGE);

fn snapshot() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("PIE_HOME").map_or_else(
        || dirs_home().join(".pie"),
        std::path::PathBuf::from,
    );
    let dir = home.join("models").join("Qwen--Qwen3.5-0.8B");
    dir.join("archive.zt").is_file().then_some(dir)
}

fn dirs_home() -> std::path::PathBuf {
    std::env::var_os("HOME").map_or_else(|| std::path::PathBuf::from("/root"), Into::into)
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
/// then row and column inside the block. Every patch-axis stream this fire
/// carries is laid in this order, which is the statute the folds ask for.
fn patch_order() -> Vec<(usize, usize)> {
    let (bh, bw) = (GRID_H / MERGE, GRID_W / MERGE);
    let mut out = Vec::with_capacity(PATCHES);
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

/// A synthetic image: deterministic patch values, real geometry.
///
/// `first_placeholder` is the lane-relative token row the first soft token
/// lands on; the `PATCHES - SOFT_TOKENS` rows past the fold say `-1`, which is
/// the drop sentinel `layout.scatter_live_rows` reads.
fn one_image(first_placeholder: i32) -> Shot {
    let order = patch_order();
    let mut patches = Vec::with_capacity(PATCHES * PATCH_WIDTH * 2);
    let mut positions = Vec::with_capacity(PATCHES * 3);
    let mut embed_rows = Vec::with_capacity(PATCHES * TAPS);
    let mut embed_weights = Vec::with_capacity(PATCHES * TAPS);

    for (at, &(row, col)) in order.iter().enumerate() {
        // A deterministic ramp in [-1, 1): a real image would be pixels, and a
        // gate about the LOAD wants numbers it can reproduce.
        for lane in 0..PATCH_WIDTH {
            #[allow(clippy::cast_precision_loss)]
            let v = (((at * PATCH_WIDTH + lane) % 251) as f32 / 251.0) - 0.5;
            patches.extend_from_slice(&to_bf16(v).to_le_bytes());
        }
        // `(t, h, w)`: a still image has no time axis, and the tower states
        // `sections[0] == 0`, so nothing reads the first column.
        positions.extend_from_slice(&[0, row as i32, col as i32]);

        let (h_taps, h_w) = axis_taps(row, GRID_H, GRID_SIDE);
        let (w_taps, w_w) = axis_taps(col, GRID_W, GRID_SIDE);
        for a in 0..2 {
            for b in 0..2 {
                embed_rows.push((h_taps[a] * GRID_SIDE + w_taps[b]) as i32);
                embed_weights.push(h_w[a] * w_w[b]);
            }
        }
    }

    let mut routes = vec![-1i32; PATCHES];
    for (j, route) in routes.iter_mut().take(SOFT_TOKENS).enumerate() {
        *route = first_placeholder + j as i32;
    }

    Shot {
        rows: vec![PATCHES as u32],
        patches,
        routes,
        positions,
        embed_rows,
        embed_weights,
    }
}

/// The loaded vision shell, and the ladder it was booted on — or `None` and a
/// sentence saying what was missing.
fn ready(what: &str) -> Option<(Shell, model_compiler::PatchLadder)> {
    let Some(checkpoint) = snapshot() else {
        eprintln!("skipping {what}: no Qwen3.5-0.8B artifact in $PIE_HOME/models");
        return None;
    };
    let trace = model::trace_of(SKU).expect("the catalog ships the vision row")(Platform::Cuda);

    // **(a) THE LADDER THE ENGINE WOULD DERIVE**, from the same function the
    // engine door calls and the contract's own default budgets.
    let budgets = engine::load::Budgets {
        max_tokens: 256,
        max_lanes: 4,
        ..engine::load::Budgets::default()
    };
    let ladder = engine_cuda::api::patch_ladder(&trace, &budgets)
        .expect("a plan that states patch rows derives a ladder");

    let container = checkpoint.join("archive.zt");
    let source = ztensor_compat::index(&container).expect("the artifact opens");
    let Ok(contract_probe) = model::import_of(SKU).expect("the catalog ships an import")(&source)
    else {
        eprintln!(
            "skipping {what}: {container:?} holds no `model.visual.*` planes — the stored \
             artifact was imported as the text-only SKU"
        );
        return None;
    };
    let _ = &contract_probe;
    let contract = contract_probe;
    drop(source);

    let shell = Shell::load(Boot {
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        // The CONTAINER and not the directory: the stored artifact is one
        // `.zt` file, and `Boot::checkpoint` takes either.
        checkpoint: &container,
        budget: Budget::new(budgets.max_lanes, budgets.max_tokens),
        patches: Some(ladder.clone()),
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        ordinal: 0,
        // Eager: this gate is about the load and the launches, and a recorded
        // fire is a different subject with its own file.
        graphs: engine_cuda::Graphs::Off,
        knobs: engine_cuda::Knobs::default(),
        program_cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        weight_cache_dir: None,
    })
    .expect("the vision shell loads");
    Some((shell, ladder))
}

/// (a): the derived ladder is a ladder this shell boots on, and its numbers
/// are the ones the derivation argues for.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local Qwen3.5-0.8B artifact; run it with `-- --ignored`"]
fn the_ladder_the_engine_derives_is_the_one_a_vision_sku_loads_on() {
    let Some((_shell, ladder)) = ready("the derived ladder loads") else {
        return;
    };
    assert_eq!(
        ladder.max_patches, 256,
        "at max_tokens = 256 the patch ceiling is the token rectangle's"
    );
    assert_eq!(ladder.buckets, vec![64, 128, 256]);
    assert_eq!(ladder.max_images, 4);
    assert!(
        PATCHES as u32 <= ladder.buckets[0],
        "the synthetic image lands on the ladder's first rung"
    );
}

/// (b), (c) and (d): the fire carries an image, the tower conditions the
/// trunk, and the axis-empty fire still works.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local Qwen3.5-0.8B artifact; run it with `-- --ignored`"]
fn a_fire_with_one_image_answers_and_a_fire_without_one_answers_differently() {
    let Some((mut shell, _)) = ready("a fire with an image") else {
        return;
    };

    // Two text rows, then the image's soft tokens, then two more.
    const FIRST_PLACEHOLDER: i32 = 2;
    let rows = FIRST_PLACEHOLDER as usize + SOFT_TOKENS + 2;
    let tokens: Vec<u32> = (0..rows as u32).map(|i| 100 + i).collect();
    // **TWO WORDS, BECAUSE THE MERGE IS GUARDED** (multimodal §15). A lane
    // that submitted images carries the `media` bit and its rows run the embed
    // merge; a lane that did not takes the class where the merge is absent, so
    // nothing resolves `PatchRoutes` on a text-only fire. The tower itself
    // needs no bit — its rectangles are `Dim::Patches` and an axis-empty fire
    // has zero patch rows.
    let classify = model::classify_of(SKU).expect("the catalog ships a classify");
    let word_with_media =
        classify(&model_dsl::Request::new(rows as u32, false).with_media(true));
    let word_text_only = classify(&model_dsl::Request::new(rows as u32, false));
    assert_ne!(
        word_with_media, word_text_only,
        "the media bit has to move the word, or the guard on the merge is a guard on nothing"
    );

    let shot = one_image(FIRST_PLACEHOLDER);
    let media = [Media {
        lane: 0,
        rows: &shot.rows,
        patches: &shot.patches,
        routes: &shot.routes,
        positions: &shot.positions,
        token_positions: &[],
        embed_rows: &shot.embed_rows,
        embed_weights: &shot.embed_weights,
    }];
    let lane = engine_cuda::Lane {
        slot: 0,
        word: word_with_media,
        tokens: &tokens,
    };

    // **THE AXIS-EMPTY FIRE FIRST**, so a failure localizes: if the plan
    // cannot fire without an image, nothing below is about the media path.
    let mut scores = Vec::new();
    let without = shell
        .fire_media(
            &[Seated::of(engine_cuda::Lane {
                slot: 1,
                word: word_text_only,
                tokens: &tokens,
            })],
            &[],
            &[],
            &mut scores,
        )
        .expect("a fire carrying no image answers");

    let mut scores = Vec::new();
    let with_image = shell
        .fire_media(&[Seated::of(lane.clone())], &[], &media, &mut scores)
        .expect("a fire carrying one image answers");
    let vocab = with_image[0].len();
    assert!(vocab > 1000, "the readout is the vocabulary's width, got {vocab}");
    assert!(
        with_image[0].iter().all(|logit| logit.is_finite()),
        "a logit came back non-finite"
    );

    // (d) THE AXIS-EMPTY FIRE — fired above, checked here.
    assert_eq!(without[0].len(), vocab);
    assert!(without[0].iter().all(|logit| logit.is_finite()));

    // (c) THE TOWER CONDITIONED THE TRUNK. Without media nothing scatters over
    // the placeholder rows, so they keep their token embedding and the readout
    // moves. Equal logits would mean the tower ran and changed nothing.
    let moved = with_image[0]
        .iter()
        .zip(&without[0])
        .filter(|(a, b)| (*a - *b).abs() > 1e-3)
        .count();
    assert!(
        moved > vocab / 100,
        "the image moved {moved} of {vocab} logits, which is not a tower that conditioned \
         anything"
    );
}

// ── M-1's FIVE GATES ────────────────────────────────────────────────────
//
// The file above asks whether the LOAD happens and the launches run. These
// ask the campaign's own five (`.wiki/alto/campaign.md` §2, M-1), which are
// claims about what a fire ANSWERS, and four of them are asked here. What (b)
// needs and does not have is written at the bottom.

/// The text-only row this artifact also satisfies — gate (a)'s other side.
const TEXT_SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// The same load, on a stated SKU. `ready` is this at [`SKU`] with the ladder
/// returned; a gate that compares two readings of one artifact needs both.
fn ready_as(sku: &str, what: &str) -> Option<Shell> {
    let checkpoint = snapshot()?;
    let trace = model::trace_of(sku).expect("the catalog ships the row")(Platform::Cuda);
    let budgets = engine::load::Budgets {
        max_tokens: 256,
        max_lanes: 4,
        ..engine::load::Budgets::default()
    };
    let patches = engine_cuda::api::patch_ladder(&trace, &budgets);
    let container = checkpoint.join("archive.zt");
    let source = ztensor_compat::index(&container).expect("the artifact opens");
    let Ok(contract) = model::import_of(sku).expect("the catalog ships an import")(&source) else {
        eprintln!("skipping {what}: {container:?} does not satisfy `{sku}`");
        return None;
    };
    drop(source);
    Some(
        Shell::load(Boot {
            residency: engine_cuda::experts::Plan::default(),
            trace,
            contract: &contract,
            checkpoint: &container,
            budget: Budget::new(budgets.max_lanes, budgets.max_tokens),
            patches,
            profile: None,
            page_size: 16,
            context: 512,
            // **EIGHT, BECAUSE EVERY IDENTITY BELOW FIRES TWICE.** A slot is a
            // sequence and a fire APPENDS to it, so re-firing one lane on the
            // slot it already wrote compares nine keys against eighteen. Each
            // comparison here gets disjoint, untouched slots on both sides.
            slots: 8,
            ordinal: 0,
            graphs: engine_cuda::Graphs::Off,
            knobs: engine_cuda::Knobs::default(),
            program_cache_dir: None,
            runahead: engine::runahead::Runahead::F1,
            weight_cache_dir: None,
        })
        .expect("the shell loads"),
    )
}

/// Every logit as its bits — the only comparison an identity claim may make.
fn bits(logits: &[f32]) -> Vec<u32> {
    logits.iter().map(|x| x.to_bits()).collect()
}

fn text_word(sku: &str, rows: u32) -> u64 {
    model::classify_of(sku).expect("a classify")(&model_dsl::Request::new(rows, false))
}

fn media_word(sku: &str, rows: u32) -> u64 {
    model::classify_of(sku).expect("a classify")(&model_dsl::Request::new(rows, false).with_media(true))
}

/// **GATE (a): A FIRE WITH NO IMAGE LANE IS THE FIRE THIS ENGINE ALWAYS
/// FIRED** — asked as an identity between two READINGS of one artifact.
///
/// The campaign words it "byte-identical in tokens AND ms". The ms half is a
/// throughput question and it has an answer of its own, measured at c256 and
/// written into `qwen_3::IMPORTS` (a vision load stands its fold down and pays
/// ~15% there). This is the half that is about ANSWERS, and it is the half a
/// text lane's correctness rests on: the same tokens, fired against the
/// text-only SKU and against the vision SKU of the same checkpoint, must come
/// back bit for bit. If the tower or the merge touched a text lane at all,
/// they would not.
///
/// Sequentially, not side by side: two shells of one model is two copies of
/// the weights, and the claim needs neither of them alive at once.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local Qwen3.5-0.8B artifact; run it with `-- --ignored`"]
fn a_text_only_fire_of_a_vision_load_is_the_text_only_load_bit_for_bit() {
    const ROWS: u32 = 12;
    let tokens: Vec<u32> = (0..ROWS).map(|i| 100 + i).collect();

    let Some(mut vision) = ready_as(SKU, "gate (a), the vision reading") else {
        return;
    };
    let mut scores = Vec::new();
    let from_vision = vision
        .fire_media(
            &[Seated::of(engine_cuda::Lane {
                slot: 0,
                word: text_word(SKU, ROWS),
                tokens: &tokens,
            })],
            &[],
            &[],
            &mut scores,
        )
        .expect("a text-only fire of a vision load answers");
    let from_vision = bits(&from_vision[0]);
    drop(vision);

    let Some(mut text) = ready_as(TEXT_SKU, "gate (a), the text-only reading") else {
        return;
    };
    let mut scores = Vec::new();
    let from_text = text
        .fire_media(
            &[Seated::of(engine_cuda::Lane {
                slot: 0,
                word: text_word(TEXT_SKU, ROWS),
                tokens: &tokens,
            })],
            &[],
            &[],
            &mut scores,
        )
        .expect("the text-only load answers");
    let from_text = bits(&from_text[0]);

    assert_eq!(from_vision.len(), from_text.len(), "two readouts, two widths");
    let moved = from_vision
        .iter()
        .zip(&from_text)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        moved, 0,
        "{moved} of {} logits differ between the vision reading of this checkpoint and \
         the text-only one, on a fire that carried no image. A tower that costs a text \
         lane one bit is a tower that is not windowed",
        from_text.len(),
    );
}

/// **GATE (c), THE POLYMORPHIC ONE: image lanes and text lanes in ONE fire,
/// and the text lanes do not know.**
///
/// This is the thesis the second row axis exists for. A fire carries one lane
/// that submitted an image and two that did not; every text lane's logits must
/// be what the same lane answers in a fire with no image in it at all — bit
/// for bit, because the two computations are the same computation and any
/// difference is the axis leaking across lanes.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local Qwen3.5-0.8B artifact; run it with `-- --ignored`"]
fn a_mixed_fire_leaves_its_text_lanes_bit_identical() {
    const FIRST_PLACEHOLDER: i32 = 2;
    let image_rows = FIRST_PLACEHOLDER as usize + SOFT_TOKENS + 2;
    let image_tokens: Vec<u32> = (0..image_rows as u32).map(|i| 100 + i).collect();
    let text_a: Vec<u32> = (0..9u32).map(|i| 300 + i).collect();
    let text_b: Vec<u32> = (0..5u32).map(|i| 700 + i).collect();

    let Some(mut shell) = ready_as(SKU, "gate (c), the mixed fire") else {
        return;
    };

    // The text lanes alone, in a fire with no media at all.
    // Fresh slots, and the mixed fire below uses different ones: a fire
    // appends, so the same lane fired twice on one slot is a longer sequence
    // the second time and the comparison would be measuring that.
    let alone: Vec<Seated<'_>> = vec![
        Seated::of(engine_cuda::Lane { slot: 4, word: text_word(SKU, text_a.len() as u32), tokens: &text_a }),
        Seated::of(engine_cuda::Lane { slot: 5, word: text_word(SKU, text_b.len() as u32), tokens: &text_b }),
    ];
    let mut scores = Vec::new();
    let solo = shell
        .fire_media(&alone, &[], &[], &mut scores)
        .expect("the text lanes answer alone");
    let solo: Vec<Vec<u32>> = solo.iter().map(|l| bits(l)).collect();

    // The same two lanes, beside an image lane, in one fire.
    let shot = one_image(FIRST_PLACEHOLDER);
    let media = [Media {
        lane: 0,
        rows: &shot.rows,
        patches: &shot.patches,
        routes: &shot.routes,
        positions: &shot.positions,
        token_positions: &[],
        embed_rows: &shot.embed_rows,
        embed_weights: &shot.embed_weights,
    }];
    let mixed_lanes: Vec<Seated<'_>> = vec![
        Seated::of(engine_cuda::Lane { slot: 0, word: media_word(SKU, image_rows as u32), tokens: &image_tokens }),
        Seated::of(engine_cuda::Lane { slot: 1, word: text_word(SKU, text_a.len() as u32), tokens: &text_a }),
        Seated::of(engine_cuda::Lane { slot: 2, word: text_word(SKU, text_b.len() as u32), tokens: &text_b }),
    ];
    let mut scores = Vec::new();
    let mixed = shell
        .fire_media(&mixed_lanes, &[], &media, &mut scores)
        .expect("a mixed fire answers");

    assert_eq!(mixed.len(), 3, "three lanes in, three readouts out");
    assert!(
        mixed[0].iter().all(|l| l.is_finite()),
        "the image lane's own readout came back non-finite"
    );

    // **AND THE CLAIM RUNS BOTH WAYS.** A text lane must not feel the image
    // lane, and the image lane must not feel the text lanes: its own rows, its
    // own patch rectangle and its own routes are unchanged, and the only thing
    // that moved is where `compose` seriated it. The media class orders SECOND
    // here, so the image lane's rows sit at a non-zero offset in the mixed fire
    // and at zero in the solo one — which is exactly the rebase
    // `Media::routes` promises, asked as an answer rather than as a doc.
    let mut scores = Vec::new();
    let image_solo = shell
        .fire_media(
            &[Seated::of(engine_cuda::Lane {
                slot: 3,
                word: media_word(SKU, image_rows as u32),
                tokens: &image_tokens,
            })],
            &[],
            &media,
            &mut scores,
        )
        .expect("the image lane answers alone");
    let moved = bits(&mixed[0])
        .iter()
        .zip(&bits(&image_solo[0]))
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        moved, 0,
        "the IMAGE lane moved {moved} of {} logits when two text lanes joined its fire;          its rows and its patch rectangle are unchanged, so what moved is the route          rebase or the seriated offset",
        image_solo[0].len(),
    );
    for (at, want) in solo.iter().enumerate() {
        let got = bits(&mixed[at + 1]);
        let moved = got.iter().zip(want).filter(|(a, b)| a != b).count();
        assert_eq!(
            moved, 0,
            "text lane {at} moved {moved} of {} logits when an image lane joined its fire; \
             the patch axis is leaking across the token rectangle",
            want.len(),
        );
    }
}

/// **GATE (d): WIDTH INVARIANCE, ON THE AXIS THAT IS NEW.**
///
/// The token axis's width invariance is an existing gate (`cuda_width_invariance`)
/// and this text does not move it. What is new is the PATCH ladder: a fire's
/// patch count is rounded up to a rung, so the same image submitted in a fire
/// that lands on a WIDER rung must answer the same logits — the padding rows
/// are padding and not neighbours. Two images on the same rung would not ask
/// it; a second lane whose image pushes the fire onto the next rung does.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local Qwen3.5-0.8B artifact; run it with `-- --ignored`"]
fn a_wider_patch_rung_does_not_move_the_image_it_padded() {
    const FIRST_PLACEHOLDER: i32 = 2;
    let rows = FIRST_PLACEHOLDER as usize + SOFT_TOKENS + 2;
    let tokens: Vec<u32> = (0..rows as u32).map(|i| 100 + i).collect();

    let Some(mut shell) = ready_as(SKU, "gate (d), the patch rung") else {
        return;
    };
    let shot = one_image(FIRST_PLACEHOLDER);
    let one = [Media {
        lane: 0,
        rows: &shot.rows,
        patches: &shot.patches,
        routes: &shot.routes,
        positions: &shot.positions,
        token_positions: &[],
        embed_rows: &shot.embed_rows,
        embed_weights: &shot.embed_weights,
    }];
    let mut scores = Vec::new();
    let narrow = shell
        .fire_media(
            &[Seated::of(engine_cuda::Lane {
                slot: 0,
                word: media_word(SKU, rows as u32),
                tokens: &tokens,
            })],
            &[],
            &one,
            &mut scores,
        )
        .expect("one image answers");
    let narrow = bits(&narrow[0]);

    // A second lane with its own image: 128 patch rows, which is the ladder's
    // SECOND rung, so lane 0's tower rows are now padded to a wider rectangle.
    let second = one_image(FIRST_PLACEHOLDER);
    let two = [
        one[0],
        Media {
            lane: 1,
            rows: &second.rows,
            patches: &second.patches,
            routes: &second.routes,
            positions: &second.positions,
            token_positions: &[],
            embed_rows: &second.embed_rows,
            embed_weights: &second.embed_weights,
        },
    ];
    // Two FRESH slots, so the first lane of this fire is the same untouched
    // sequence the fire above answered — the only difference between them is
    // the width of the patch rectangle.
    let lanes = vec![
        Seated::of(engine_cuda::Lane {
            slot: 2,
            word: media_word(SKU, rows as u32),
            tokens: &tokens,
        }),
        Seated::of(engine_cuda::Lane {
            slot: 3,
            word: media_word(SKU, rows as u32),
            tokens: &tokens,
        }),
    ];
    let mut scores = Vec::new();
    let wide_all = shell
        .fire_media(&lanes, &[], &two, &mut scores)
        .expect("two images answer");
    let wide = bits(&wide_all[0]);

    // **LOCALIZE FIRST.** The two lanes of the wide fire carry the SAME tokens
    // and the SAME image, so they must answer each other exactly. If they do
    // not, the fault is per-lane — the patch segments, the route rebase — and
    // has nothing to do with the rung; if they do, whatever moved moved for
    // both, which is the rectangle.
    let twin = bits(&wide_all[1]);
    let across = wide.iter().zip(&twin).filter(|(a, b)| a != b).count();
    assert_eq!(
        across, 0,
        "two identical images in one fire answered differently ({across} of {} logits); \
         the fault is per-lane and not the rung",
        wide.len(),
    );

    let moved = narrow.iter().zip(&wide).filter(|(a, b)| a != b).count();
    assert_eq!(
        moved, 0,
        "the first image's readout moved {moved} of {} logits when a second image widened \
         the patch rectangle; a rung's padding is being attended over",
        narrow.len(),
    );
}

/// **GATE (e): THE THREE REFUSALS, BY NAME.**
///
/// Every one of them is host-side and fires BEFORE a launch, which is the
/// whole point: `layout.scatter_live_rows` is a copy with an index and the
/// arena does not fault on an address that stays inside one `cudaMalloc`, so a
/// submission that disagrees with itself has to be caught by arithmetic.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local Qwen3.5-0.8B artifact; run it with `-- --ignored`"]
fn the_three_media_refusals_fire_by_name() {
    const FIRST_PLACEHOLDER: i32 = 2;
    let rows = FIRST_PLACEHOLDER as usize + SOFT_TOKENS + 2;
    let tokens: Vec<u32> = (0..rows as u32).map(|i| 100 + i).collect();
    let shot = one_image(FIRST_PLACEHOLDER);
    let good = Media {
        lane: 0,
        rows: &shot.rows,
        patches: &shot.patches,
        routes: &shot.routes,
        positions: &shot.positions,
        token_positions: &[],
        embed_rows: &shot.embed_rows,
        embed_weights: &shot.embed_weights,
    };

    // (i) AN IMAGE AGAINST A TEXT WITH NO PATCH AXIS.
    if let Some(mut text) = ready_as(TEXT_SKU, "refusal (i), towerless") {
        let mut scores = Vec::new();
        let refused = text.fire_media(
            &[Seated::of(engine_cuda::Lane {
                slot: 0,
                word: text_word(TEXT_SKU, rows as u32),
                tokens: &tokens,
            })],
            &[],
            &[good],
            &mut scores,
        );
        let why = refused.err().expect("a text-only load refuses an image").to_string();
        assert!(
            why.contains("tower") || why.contains("patch"),
            "the towerless refusal does not name the axis it lacks: {why}"
        );
    }

    let Some(mut shell) = ready_as(SKU, "refusals (ii) and (iii)") else {
        return;
    };
    let lane = Seated::of(engine_cuda::Lane {
        slot: 0,
        word: media_word(SKU, rows as u32),
        tokens: &tokens,
    });

    // (ii) GEOMETRY THAT DISAGREES WITH ITS PAYLOAD: the row count says one
    // patch fewer than the bytes carry.
    let short_rows = vec![shot.rows[0] - 1];
    let mut scores = Vec::new();
    let refused = shell.fire_media(
        &[lane.clone()],
        &[],
        &[Media { rows: &short_rows, ..good }],
        &mut scores,
    );
    let why = refused
        .err()
        .expect("a payload that disagrees with its geometry is refused")
        .to_string();
    assert!(
        why.contains("patch") || why.contains("byte"),
        "the payload refusal does not name what disagreed: {why}"
    );

    // (iii) A ROUTE PAST THE LANE'S OWN ROWS. Lane-relative, so `rows` is the
    // bound, and one past it is the row this fire does not have.
    let mut far = shot.routes.clone();
    far[0] = rows as i32;
    let mut scores = Vec::new();
    let refused = shell.fire_media(
        &[lane],
        &[],
        &[Media { routes: &far, ..good }],
        &mut scores,
    );
    let why = refused
        .err()
        .expect("a route past the lane's rows is refused")
        .to_string();
    assert!(
        why.contains("route") || why.contains("row"),
        "the route refusal does not name the bound it broke: {why}"
    );
}

// **GATE (b) IS NOT HERE, AND HERE IS WHAT IT WANTS.** "A deterministic,
// sensible caption on a real image" needs three things this crate does not
// have: a decoded image (an `image` dependency, which lives in `runtime`), the
// resize-and-patchify policy that turns it into `[patches, C·T·P²]`
// (`runtime::inferlet::host::media::multimodal::QwenImageConfig::
// qwen_patchify_hwc`, one crate up and the wrong direction for a shell test),
// and a tokenizer to read the answer back. A synthetic ramp answers
// deterministically and says nothing about sense, so asserting on it would be
// a gate that cannot fail for the reason it names. It belongs where the
// preprocessing already is — a runtime-level gate, or an inferlet once §6.6's
// guest door opens — and it is the one of the five that a Shell-level file
// cannot honestly make.
