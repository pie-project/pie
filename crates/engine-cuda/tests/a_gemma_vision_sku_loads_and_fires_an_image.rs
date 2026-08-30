//! **M-2's HALF OF THE FIVE GATES, ON GEMMA'S TOWER.**
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 \
//!     --test a_gemma_vision_sku_loads_and_fires_an_image -- --ignored --nocapture
//! ```
//!
//! A sibling of `a_vision_sku_loads_and_fires_an_image` and not a
//! generalization of it, because the two towers differ in exactly the places a
//! parameter would have hidden: gemma pools where qwen merges (`k = 3` against
//! `k = 2`, so a legal grid is a different shape), its patch row is
//! `C · P²` with no temporal extent (768 against 1536), and its position table
//! is TWO SEPARABLE LOOKUPS rather than a bilinear resample — two taps whose
//! weights are ones, the `y` index reaching into the second half of one
//! flattened table. A shared `one_image` would have carried three `if`s and
//! said none of that.
//!
//! The claims are the campaign's own, and (b) is door-blocked here for the
//! reason it is there: a caption wants a decoded image, the resize policy that
//! lives in `runtime`, and a tokenizer.

use engine_cuda::{Boot, Media, Seated, Shell};
use model_compiler::Budget;
use model_dsl::Platform;

const SKU: &str = "gemma4-e4b-vision-bf16-kv-bf16";
const TEXT_SKU: &str = "gemma4-e4b-bf16-kv-bf16";

/// `in_channels · patch_size²`. Gemma's patch has no temporal extent.
const PATCH_WIDTH: usize = 3 * 16 * 16;

/// `pooling_kernel_size`. `layout.pool_rows` folds `POOL²` consecutive rows.
const POOL: usize = 3;

/// Two: the x table and the y table, read as one bank (`Tower`'s own note).
const TAPS: usize = 2;

/// `position_embedding_size` — where the y half of the flattened table begins.
const AXIS_POSITIONS: i32 = 10_240;

/// A 6 x 6 patch grid: a whole number of 3 x 3 pooling blocks, and 36 rows,
/// which lands inside the patch ladder's first rung.
const GRID_H: usize = 6;
const GRID_W: usize = 6;
const PATCHES: usize = GRID_H * GRID_W;
const SOFT_TOKENS: usize = PATCHES / (POOL * POOL);

fn store() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("PIE_HOME").map_or_else(
        || {
            std::path::PathBuf::from(
                std::env::var_os("HOME").unwrap_or_else(|| "/root".into()),
            )
            .join(".pie")
        },
        std::path::PathBuf::from,
    );
    let dir = home.join("models").join("google--gemma-4-E4B-it");
    dir.join("archive.zt").is_file().then_some(dir)
}

fn to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if (b & 0x7fff_ffff) > 0x7f80_0000 {
        return ((b >> 16) | 0x0040) as u16;
    }
    let rounding = 0x7fff + ((b >> 16) & 1);
    (b.wrapping_add(rounding) >> 16) as u16
}

/// **THE POOL-BLOCK-MAJOR PATCH ORDER**, which is §2's merge-block statute at
/// `k = 3`: block row, block column, then row and column inside the block.
/// `layout.pool_rows` reads no geometry and folds consecutive rows, so this
/// ordering IS the pooling's correctness.
fn patch_order() -> Vec<(usize, usize)> {
    let (bh, bw) = (GRID_H / POOL, GRID_W / POOL);
    let mut out = Vec::with_capacity(PATCHES);
    for ih_blk in 0..bh {
        for iw_blk in 0..bw {
            for ih in 0..POOL {
                for iw in 0..POOL {
                    out.push((ih_blk * POOL + ih, iw_blk * POOL + iw));
                }
            }
        }
    }
    out
}

struct Shot {
    rows: Vec<u32>,
    patches: Vec<u8>,
    routes: Vec<i32>,
    positions: Vec<i32>,
    embed_rows: Vec<i32>,
    embed_weights: Vec<f32>,
}

/// One synthetic image with real geometry.
fn one_image(first_placeholder: i32) -> Shot {
    let order = patch_order();
    let mut patches = Vec::with_capacity(PATCHES * PATCH_WIDTH * 2);
    let mut positions = Vec::with_capacity(PATCHES * 3);
    let mut embed_rows = Vec::with_capacity(PATCHES * TAPS);
    let mut embed_weights = Vec::with_capacity(PATCHES * TAPS);

    for (at, &(row, col)) in order.iter().enumerate() {
        for lane in 0..PATCH_WIDTH {
            #[allow(clippy::cast_precision_loss)]
            let v = (((at * PATCH_WIDTH + lane) % 251) as f32 / 251.0) - 0.5;
            patches.extend_from_slice(&to_bf16(v).to_le_bytes());
        }
        // `(t, h, w)`; the tower states `sections[0] == 0`, so `t` is read by
        // nothing.
        positions.extend_from_slice(&[0, row as i32, col as i32]);
        // `table[0][x] + table[1][y]`, as two taps into one flattened bank
        // with weights of one — the x index is the COLUMN and the y index the
        // row, which is what `_position_embeddings` indexes with.
        embed_rows.push(col as i32);
        embed_rows.push(AXIS_POSITIONS + row as i32);
        embed_weights.push(1.0);
        embed_weights.push(1.0);
    }

    let mut routes = vec![-1i32; PATCHES];
    for (j, route) in routes.iter_mut().take(SOFT_TOKENS).enumerate() {
        *route = first_placeholder + j as i32;
    }

    Shot { rows: vec![PATCHES as u32], patches, routes, positions, embed_rows, embed_weights }
}

fn ready_as(sku: &str, what: &str) -> Option<Shell> {
    let dir = store()?;
    let trace = model::trace_of(sku).expect("the catalog ships the row")(Platform::Cuda);
    let budgets = engine::load::Budgets {
        max_tokens: 256,
        max_lanes: 4,
        ..engine::load::Budgets::default()
    };
    let patches = engine_cuda::api::patch_ladder(&trace, &budgets);
    let container = dir.join("archive.zt");
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
            // Disjoint slots on both sides of every identity: a fire APPENDS.
            slots: 8,
            ordinal: 0,
            graphs: engine_cuda::Graphs::Off,
            knobs: engine_cuda::Knobs::default(),
            program_cache_dir: None,
            runahead: engine::runahead::Runahead::F1,
            weight_cache_dir: None,
        })
        .expect("the gemma vision shell loads"),
    )
}

fn bits(logits: &[f32]) -> Vec<u32> {
    logits.iter().map(|x| x.to_bits()).collect()
}

fn text_word(sku: &str, rows: u32) -> u64 {
    model::classify_of(sku).expect("a classify")(&model_dsl::Request::new(rows, false))
}

fn media_word(sku: &str, rows: u32) -> u64 {
    model::classify_of(sku).expect("a classify")(
        &model_dsl::Request::new(rows, false).with_media(true),
    )
}

fn shot_media<'a>(lane: u32, shot: &'a Shot) -> Media<'a> {
    Media {
        lane,
        rows: &shot.rows,
        patches: &shot.patches,
        routes: &shot.routes,
        positions: &shot.positions,
        token_positions: &[],
        embed_rows: &shot.embed_rows,
        embed_weights: &shot.embed_weights,
    }
}

/// The ladder a gemma vision load derives, and that this image lands on it.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local gemma-4-E4B artifact; run it with `-- --ignored`"]
fn the_derived_ladder_seats_a_pooled_image() {
    let trace = model::trace_of(SKU).expect("the catalog ships the row")(Platform::Cuda);
    let ladder = engine_cuda::api::patch_ladder(
        &trace,
        &engine::load::Budgets { max_tokens: 256, max_lanes: 4, ..engine::load::Budgets::default() },
    )
    .expect("a plan that states patch rows derives a ladder");
    assert!(
        PATCHES as u32 <= ladder.buckets[0],
        "a {PATCHES}-row image must land on the ladder's first rung {}",
        ladder.buckets[0],
    );
    assert_eq!(PATCHES % (POOL * POOL), 0, "a legal grid is whole pooling blocks");
}

/// **(a)**: a text-only fire of the vision load is the text-only load, bit for
/// bit — the same claim the qwen file makes, on the family whose tower pools.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local gemma-4-E4B artifact; run it with `-- --ignored`"]
fn a_text_only_fire_of_the_gemma_vision_load_is_the_text_only_load_bit_for_bit() {
    const ROWS: u32 = 12;
    let tokens: Vec<u32> = (0..ROWS).map(|i| 100 + i).collect();

    let Some(mut vision) = ready_as(SKU, "gate (a), the vision reading") else {
        return;
    };
    let mut scores = Vec::new();
    let from_vision = vision
        .fire_media(
            &[Seated::of(engine_cuda::Lane { slot: 0, word: text_word(SKU, ROWS), tokens: &tokens })],
            &[],
            &[],
            &mut scores,
        )
        .expect("a text-only fire of a gemma vision load answers");
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

    let moved = from_vision.iter().zip(&from_text).filter(|(a, b)| a != b).count();
    assert_eq!(
        moved, 0,
        "{moved} of {} logits differ between gemma's vision reading and its text-only one \
         on a fire that carried no image",
        from_text.len(),
    );
}

/// **(b-partial), (c) and (d)**: the image fires and conditions the trunk, the
/// text lanes beside it do not move, and a second image does not move the
/// first.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local gemma-4-E4B artifact; run it with `-- --ignored`"]
fn a_gemma_image_fires_and_leaves_its_neighbours_alone() {
    const FIRST_PLACEHOLDER: i32 = 2;
    let rows = FIRST_PLACEHOLDER as usize + SOFT_TOKENS + 2;
    let tokens: Vec<u32> = (0..rows as u32).map(|i| 100 + i).collect();
    let text: Vec<u32> = (0..9u32).map(|i| 300 + i).collect();

    let Some(mut shell) = ready_as(SKU, "gemma's image fire") else {
        return;
    };
    let shot = one_image(FIRST_PLACEHOLDER);

    // The text lane alone, and the image lane alone, on fresh slots.
    let mut scores = Vec::new();
    let text_solo = shell
        .fire_media(
            &[Seated::of(engine_cuda::Lane {
                slot: 4,
                word: text_word(SKU, text.len() as u32),
                tokens: &text,
            })],
            &[],
            &[],
            &mut scores,
        )
        .expect("the text lane answers alone");
    let text_solo = bits(&text_solo[0]);

    let mut scores = Vec::new();
    let with_image = shell
        .fire_media(
            &[Seated::of(engine_cuda::Lane {
                slot: 0,
                word: media_word(SKU, rows as u32),
                tokens: &tokens,
            })],
            &[],
            &[shot_media(0, &shot)],
            &mut scores,
        )
        .expect("a gemma fire carrying one image answers");
    let vocab = with_image[0].len();
    assert!(with_image[0].iter().all(|l| l.is_finite()), "a logit came back non-finite");
    let image_solo = bits(&with_image[0]);

    // The same lane with no media: the placeholder rows keep their token
    // embedding, so the readout must MOVE. Equal logits would mean a tower
    // that ran and changed nothing.
    let mut scores = Vec::new();
    let without = shell
        .fire_media(
            &[Seated::of(engine_cuda::Lane {
                slot: 5,
                word: text_word(SKU, rows as u32),
                tokens: &tokens,
            })],
            &[],
            &[],
            &mut scores,
        )
        .expect("the same rows answer without an image");
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

    // (c) MIXED: the image lane beside the text lane, both untouched slots.
    let second = one_image(FIRST_PLACEHOLDER);
    let mixed = vec![
        Seated::of(engine_cuda::Lane {
            slot: 1,
            word: media_word(SKU, rows as u32),
            tokens: &tokens,
        }),
        Seated::of(engine_cuda::Lane {
            slot: 2,
            word: text_word(SKU, text.len() as u32),
            tokens: &text,
        }),
    ];
    let mut scores = Vec::new();
    let out = shell
        .fire_media(&mixed, &[], &[shot_media(0, &second)], &mut scores)
        .expect("a mixed gemma fire answers");
    let moved = bits(&out[1]).iter().zip(&text_solo).filter(|(a, b)| a != b).count();
    assert_eq!(
        moved, 0,
        "the text lane moved {moved} of {} logits when a gemma image lane joined its fire",
        text_solo.len(),
    );
    let moved = bits(&out[0]).iter().zip(&image_solo).filter(|(a, b)| a != b).count();
    assert_eq!(
        moved, 0,
        "the image lane moved {moved} of {} logits when a text lane joined its fire",
        image_solo.len(),
    );

    // (d) A WIDER PATCH RECTANGLE: two images, and the first is unmoved.
    let third = one_image(FIRST_PLACEHOLDER);
    let two = vec![
        Seated::of(engine_cuda::Lane {
            slot: 3,
            word: media_word(SKU, rows as u32),
            tokens: &tokens,
        }),
        Seated::of(engine_cuda::Lane {
            slot: 6,
            word: media_word(SKU, rows as u32),
            tokens: &tokens,
        }),
    ];
    let mut scores = Vec::new();
    let out = shell
        .fire_media(
            &two,
            &[],
            &[shot_media(0, &second), shot_media(1, &third)],
            &mut scores,
        )
        .expect("two gemma images answer");
    let across = bits(&out[0]).iter().zip(&bits(&out[1])).filter(|(a, b)| a != b).count();
    assert_eq!(across, 0, "two identical gemma images answered differently ({across} logits)");
    let moved = bits(&out[0]).iter().zip(&image_solo).filter(|(a, b)| a != b).count();
    assert_eq!(
        moved, 0,
        "the first image moved {moved} of {} logits when a second widened the patch rectangle",
        image_solo.len(),
    );
}

/// **(e)**: the three refusals, by name, on gemma's row.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local gemma-4-E4B artifact; run it with `-- --ignored`"]
fn the_three_media_refusals_fire_by_name_on_gemma() {
    const FIRST_PLACEHOLDER: i32 = 2;
    let rows = FIRST_PLACEHOLDER as usize + SOFT_TOKENS + 2;
    let tokens: Vec<u32> = (0..rows as u32).map(|i| 100 + i).collect();
    let shot = one_image(FIRST_PLACEHOLDER);
    let good = shot_media(0, &shot);

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
        let why = refused.err().expect("a text-only gemma refuses an image").to_string();
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

    let short_rows = vec![shot.rows[0] - 1];
    let mut scores = Vec::new();
    let why = shell
        .fire_media(&[lane.clone()], &[], &[Media { rows: &short_rows, ..good }], &mut scores)
        .err()
        .expect("a payload that disagrees with its geometry is refused")
        .to_string();
    assert!(
        why.contains("patch") || why.contains("byte"),
        "the payload refusal does not name what disagreed: {why}"
    );

    let mut far = shot.routes.clone();
    far[0] = rows as i32;
    let mut scores = Vec::new();
    let why = shell
        .fire_media(&[lane], &[], &[Media { routes: &far, ..good }], &mut scores)
        .err()
        .expect("a route past the lane's rows is refused")
        .to_string();
    assert!(
        why.contains("route") || why.contains("row"),
        "the route refusal does not name the bound it broke: {why}"
    );
}
