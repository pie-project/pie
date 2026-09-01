//! **THE METAL VISION ROW, FIRED AT A REAL PICTURE** — metal-verify-queue
//! Session I's entry (m), which first light left standing red and which this
//! file both reproduces and localizes without a server in the room.
//!
//! The CUDA sibling (`engine-cuda/tests/a_vision_sku_loads_and_fires_an_image
//! .rs`) fires a synthetic ramp through a bf16 tower and asks whether the
//! launches happen. This file asks the question that gate cannot: does the
//! tower carry the PICTURE. It builds three solid squares, runs them through
//! the PRODUCTION front-end (`models::media::vision_front_end`), derives the
//! six patch vectors exactly as `runtime::pipeline::media::lane_media` does,
//! fires the real `qwen35-d0.8b-vision-mlxu4-kv-bf16` load, and reads the
//! caption off a greedy continuation.
//!
//! # What it found — fixed since, and kept as the record
//!
//! **THE TOWER FIRED, CONDITIONED THE TRUNK BY CONTENT** — red and blue
//! moved ~246 000 of 248 320 logits apart — **and every colour captioned
//! `" black"`.** The soft tokens were not constant and they were not the
//! picture: they were the picture read down the wrong axis. The fault is
//! fixed (the import permutes the MLX bank through a gather lowering) and
//! all three gates below are green; the analysis stays because the bank
//! gate's meaning depends on it.
//!
//! `mlx_lm` stores the tower's patch projection as an **MLX** `Conv3d` kernel,
//! which is CHANNELS-LAST: `vision_tower.patch_embed.proj.weight` is
//! `[hidden, T, P, P, C]` = `[768, 2, 16, 16, 3]`. `transformers` stores the
//! same projection as a **torch** `Conv3d`, which is channel-major:
//! `model.visual.patch_embed.proj.weight` is `[hidden, C, T, P, P]` =
//! `[768, 3, 2, 16, 16]`. Both hold 1 179 648 elements, so
//! `qwen_3::import::flattened`'s count check passes either way, and the ONE
//! rewrite that file makes — a `transmute` to `[hidden, C·T·P²]`, argued as
//! "the same bytes in the same order" — is the same bytes in a DIFFERENT
//! order for every `-vision-mlxu4-` row in the catalog. (Since fixed:
//! `qwen_3::import` reads the MLX bank through `Expr::Gather`, lowered as a
//! `GatherWrite` in the shared staging path.)
//!
//! The payload the front-end ships is channel-major by statute
//! (`model/tests/qwen3_5_media_is_the_pinned_arithmetic.rs`'s
//! `a_patch_row_is_channel_then_temporal_then_row_then_column`), so the tower
//! contracts a channel-major vector against a channels-last bank: finite,
//! deterministic, image-dependent, and nonsense — which is what a 0.8-billion
//! trunk answers `" black"` to whatever colour it is handed.
//!
//! [`the_bank_s_lane_order_is_the_one_that_names_the_colours`] is the proof:
//! it fires the same picture twice, once with the front-end's own lane order
//! and once re-laid channels-last so the wrong transmute cancels, and asserts
//! that EXACTLY ONE of the two names all three colours. Today it is the
//! channels-last one (`" red"` / `"Green"` / `"Blue"`); the day
//! `qwen_3::import` permutes the MLX bank it will be the other, and that gate
//! stays true and stays informative across the fix — today the channel-major
//! arm is the one that names them.
//!
//! # The three claims of the gate proper
//!
//! `tests/inferlets/test_curated.py::test_image_captioning`'s own, asked here
//! where a failure localizes instead of at the end of a server:
//!
//! 1. **the tower conditions the trunk** — an image moves the logits;
//! 2. **it conditions them by CONTENT** — red and blue move them differently;
//! 3. **and the captions differ and name their colour**, which is the claim
//!    the whole media door exists for.
//!
//! Everything is greedy and every number is the front-end's own, so a caption
//! that moves between runs is a bug and not a sample.

use engine::frame::Shell as FrameShell;
use engine_metal::serve::Media;
use engine_metal::{Boot, Lane, Seated, Shell, StepView};
use model_compiler::Budget as LoadBudget;
use model_ir::Platform;
use models::media::{Budget, Rgb8};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};

const SKU: &str = "qwen35-d0.8b-vision-mlxu4-kv-bf16";
const SIDE: u32 = 256;
const MAX_TOKENS: u32 = 512;
const MAX_LANES: u32 = 4;
const CONTEXT: u32 = 1024;
const NEW_TOKENS: usize = 12;

const SYSTEM: &str = "You are a helpful assistant that describes images.";
const QUESTION: &str = "What is the dominant colour of the image above? Answer with one word.";

/// One device, one test at a time.
fn serialized() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The `mlx-community` Qwen3.5 4-bit snapshots in the hugging face cache that
/// actually publish a tower.
///
/// **THE FILTER IS THE TOWER AND NOT THE NAME**, because the repository holds
/// two snapshots and only one of them is the published artifact: a local
/// `mlx_lm.convert` output lands under `snapshots/local` and converts the TEXT
/// row, so it has every trunk plane and no `vision_tower.*` at all. A search
/// that took the newest directory would refuse at the patch embed and read as
/// a broken import.
fn snapshots() -> Vec<PathBuf> {
    let usable = |p: &Path| p.join("tokenizer.json").exists() && container(p).is_some();
    let homes = [
        std::env::var("HOME").unwrap_or_default(),
        "/Users/ingim".to_string(),
    ];
    let mut out: Vec<PathBuf> = Vec::new();
    for home in homes {
        let root = Path::new(&home)
            .join(".cache/huggingface/hub")
            .join("models--mlx-community--Qwen3.5-0.8B-4bit/snapshots");
        let Ok(entries) = std::fs::read_dir(root) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if usable(&path) && !out.contains(&path) {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// The container the contract is checked against — one file of the snapshot,
/// whichever one holds the tensors.
fn container(snapshot: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(snapshot)
        .ok()?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            (name.ends_with(".safetensors") || name.ends_with(".zt")).then_some(path)
        })
        .collect();
    found.sort();
    found.into_iter().next()
}

/// The lent resample. Every square this file makes is already the size
/// `smart_resize` asks for, so the front-end's call is the identity — and a
/// call with a DIFFERENT target is a fixture bug, not a resize.
fn no_resample(src: &Rgb8, th: u32, tw: u32) -> Rgb8 {
    assert_eq!(
        (src.h, src.w),
        (th, tw),
        "this fixture sizes its squares at the front-end's own target; a real resize here \
         would mean the policy moved"
    );
    src.clone()
}

fn solid(color: [u8; 3]) -> Rgb8 {
    let mut data = Vec::with_capacity((SIDE * SIDE * 3) as usize);
    for _ in 0..SIDE * SIDE {
        data.extend_from_slice(&color);
    }
    Rgb8::new(SIDE, SIDE, data).expect("a solid square is a frame")
}

fn to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if (b & 0x7fff_ffff) > 0x7f80_0000 {
        return ((b >> 16) | 0x0040) as u16;
    }
    let rounding = 0x7fff + ((b >> 16) & 1);
    (b.wrapping_add(rounding) >> 16) as u16
}

/// One lane's media row, derived exactly as `runtime::pipeline::media` derives
/// it: the front-end's span plus the anchor the run scan found.
struct Shot {
    rows: Vec<u32>,
    patches: Vec<u8>,
    routes: Vec<i32>,
    positions: Vec<i32>,
    embed_rows: Vec<i32>,
    embed_weights: Vec<f32>,
    token_positions: Vec<i32>,
    tokens: Vec<u32>,
    /// Each span's soft-token count, in submission order.
    softs: Vec<u32>,
    /// Where each span's run starts, as an offset into this lane's token rows.
    anchors: Vec<u32>,
    /// **WHERE THE M-ROPE CURSOR ACTUALLY ENDED**, which is not `tokens.len()`
    /// once a run is in the lane: an image spends `t·h·w` token rows and only
    /// `max(h, w)` positions. Gate (s)'s second half is the distance between
    /// these two numbers.
    cursor_end: u32,
}

/// Build the whole submission for one colour: the chatml prompt with the
/// span's own spelling spliced in, and the six patch vectors beside it.
fn shot(color: [u8; 3], tok: &tokenizer::Tokenizer, channels_last: bool) -> Shot {
    shot_of(&[color], tok, QUESTION, 0, channels_last)
}

/// One span's payload re-laid `(c, t, r, q)` -> `(t, r, q, c)`, which is the
/// order an MLX `Conv3d` bank is stored in.
///
/// **THE DIAGNOSTIC LANE ORDER.** It exists to answer one question
/// ([`the_bank_s_lane_order_is_the_one_that_names_the_colours`]) and will be
/// deleted with it.
fn relaid(payload: &[f32], rows: u32) -> Vec<f32> {
    const P: usize = 16;
    const TP: usize = 2;
    let mut out = vec![0.0f32; payload.len()];
    let width = 3 * TP * P * P;
    for row in 0..rows as usize {
        let src = &payload[row * width..(row + 1) * width];
        let dst = &mut out[row * width..(row + 1) * width];
        for c in 0..3 {
            for t in 0..TP {
                for r in 0..P {
                    for q in 0..P {
                        dst[((t * P + r) * P + q) * 3 + c] = src[((c * TP + t) * P + r) * P + q];
                    }
                }
            }
        }
    }
    out
}

/// Neutral prose that names no colour, repeated to whatever row count gate (s)
/// asks for. A caption that names a colour after this named it from the
/// picture and not from its prompt.
const FILLER: &str =
    "The archive keeps its ledgers in the east wing, and the clerks arrive before the bells. ";

/// **ONE LANE, N IMAGES, AND AS MUCH TEXT IN FRONT OF THEM AS THE CALLER
/// ASKS** — `runtime::pipeline::media::lane_media`'s derivation, transcribed.
///
/// Session M's two gates are exactly the two cases one image and a short
/// prompt cannot tell apart:
///
/// * **(r)** a lane's `routes` are ONE fold-space prefix, `patches / fold`
///   long over the LANE's total payload rows — so the spans' addresses go
///   down BACK TO BACK and the single `-1` tail pads the end of the lane.
///   Per-span padding is the same vector for one span and drops every span
///   after the first.
/// * **(s)** an image's triples are `cursor + (t, h, w)` at the cursor its run
///   STARTS at, and the cursor then advances by `position_span` — `max(h, w)`
///   — and not by the run's token count. With filler rows in front of the
///   image, a derivation missing the run-start offset rotates the picture as
///   though it sat at the start of the sequence.
fn shot_of(
    colors: &[[u8; 3]],
    tok: &tokenizer::Tokenizer,
    question: &str,
    filler: usize,
    channels_last: bool,
) -> Shot {
    let front = models::media::vision_front_end("qwen3_5").expect("the catalog ships qwen's tower");
    let d = front.delimiters();
    let one = |s: &str| {
        let ids = tok.encode(s);
        assert_eq!(ids.len(), 1, "{s:?} is one reserved id");
        ids[0]
    };
    let spans: Vec<_> = colors
        .iter()
        .map(|rgb| {
            let mut span = front
                .encode(&solid(*rgb), Budget::Still, no_resample)
                .expect("a solid square preprocesses");
            span.spell_with(vec![one(d.prefix)], one(d.placeholder), vec![one(d.suffix)]);
            span
        })
        .collect();

    // -- THE LEDGER. `chat::system` / `chat::user` / `chat::cue`, spelled here
    //    because this file holds no chat crate.
    let mut tokens = tok.encode(&format!("<|im_start|>system\n{SYSTEM}<|im_end|>\n"));
    tokens.extend(tok.encode("<|im_start|>user\n"));
    // **THE DISTANCE FROM THE SEQUENCE START, AS A KNOB** -- gate (s).
    if filler > 0 {
        let unit = tok.encode(FILLER);
        assert!(!unit.is_empty(), "the filler sentence is at least one row");
        let mut rows = Vec::with_capacity(filler + unit.len());
        while rows.len() < filler {
            rows.extend_from_slice(&unit);
        }
        rows.truncate(filler);
        tokens.extend(rows);
    }
    let mut anchors: Vec<u32> = Vec::with_capacity(spans.len());
    for (i, span) in spans.iter().enumerate() {
        if i > 0 {
            tokens.extend(tok.encode("\nand then\n"));
        }
        anchors.push((tokens.len() + span.prefix.len()) as u32);
        tokens.extend(span.tokens());
    }
    tokens.extend(tok.encode(&format!("{question}<|im_end|>\n")));
    tokens.extend(tok.encode("<|im_start|>assistant\n"));

    // -- THE SIX, as `lane_media` derives them.
    let mut rows_of: Vec<u32> = Vec::with_capacity(spans.len());
    let mut softs: Vec<u32> = Vec::with_capacity(spans.len());
    let mut patches: Vec<u8> = Vec::new();
    let mut routes: Vec<i32> = Vec::new();
    let mut positions: Vec<i32> = Vec::new();
    let mut embed_rows: Vec<i32> = Vec::new();
    let mut embed_weights: Vec<f32> = Vec::new();
    for (span, &anchor) in spans.iter().zip(&anchors) {
        rows_of.push(span.rows);
        softs.push(span.token_count);
        let payload = if channels_last {
            relaid(&span.payload, span.rows)
        } else {
            span.payload.clone()
        };
        patches.extend(payload.iter().flat_map(|&v| to_bf16(v).to_le_bytes()));
        // **THE LANE'S ADDRESSES, BACK TO BACK, WITH NOTHING BETWEEN THEM.**
        routes.extend((0..span.token_count).map(|k| (anchor + k) as i32));
        for yx in span.positions.chunks_exact(2) {
            positions.extend_from_slice(&[0, yx[0] as i32, yx[1] as i32]);
        }
        embed_rows.extend_from_slice(&span.embed_rows);
        embed_weights.extend_from_slice(&span.embed_weights);
    }
    // **AND ONE `-1` TAIL, AT THE END OF THE LANE** -- the rows the fold
    // spends, padding the vector out to the `[Dim::Patches]` rectangle.
    let owed: usize = rows_of.iter().map(|&r| r as usize).sum();
    while routes.len() < owed {
        routes.push(-1);
    }

    // **THE TRUNK'S TRIPLES CARRY THE RUN'S START POSITION** -- `get_rope_index`,
    // and `lane_media`'s own cursor walk. A text row takes the CURSOR, not its
    // token row; the run's rows take the cursor plus their merged-grid
    // coordinate on all three axes; and the cursor then advances by
    // `position_span` (`max(h, w)`), which is fewer than the `h . w` token rows
    // the run spent -- so the two part company for every row after the image.
    let mut token_positions = Vec::with_capacity(3 * tokens.len());
    let mut cursor: u32 = 0;
    let mut p: u32 = 0;
    let mut next = 0usize;
    while p < tokens.len() as u32 {
        match anchors.get(next).filter(|&&a| a == p) {
            Some(_) => {
                let span = &spans[next];
                let g = span.grid;
                let (gh, gw) = (g.h.max(1), g.w.max(1));
                let hw = gh * gw;
                for k in 0..span.token_count {
                    let rem = k % hw;
                    token_positions.extend_from_slice(&[
                        (cursor + k / hw) as i32,
                        (cursor + rem / gw) as i32,
                        (cursor + rem % gw) as i32,
                    ]);
                }
                cursor += span.position_span;
                p += span.token_count;
                next += 1;
            }
            None => {
                token_positions.extend_from_slice(&[cursor as i32; 3]);
                cursor += 1;
                p += 1;
            }
        }
    }

    Shot {
        rows: rows_of,
        patches,
        routes,
        positions,
        embed_rows,
        embed_weights,
        token_positions,
        tokens,
        softs,
        anchors,
        cursor_end: cursor,
    }
}

impl Shot {
    fn media(&self) -> Media<'_> {
        Media {
            lane: 0,
            rows: &self.rows,
            patches: &self.patches,
            routes: &self.routes,
            positions: &self.positions,
            embed_rows: &self.embed_rows,
            embed_weights: &self.embed_weights,
            token_positions: &self.token_positions,
        }
    }
}

fn ready(what: &str) -> Option<(Shell, tokenizer::Tokenizer)> {
    ready_as(SKU, what)
}

/// The same load on a STATED row — the vision row and its text-only twin are
/// the same artifact read two ways, and gate (o) needs both.
fn ready_as(sku: &str, what: &str) -> Option<(Shell, tokenizer::Tokenizer)> {
    ready_with(sku, what, MAX_TOKENS, CONTEXT)
}

/// The same load with the two budgets a long-context fire has to raise.
///
/// Gate (s) submits ~2 000 filler rows in front of the image in ONE prefill,
/// so both the fire's row budget and the slot's context have to hold it; every
/// other gate in this file fits the defaults and states them.
fn ready_with(
    sku: &str,
    what: &str,
    max_tokens: u32,
    context: u32,
) -> Option<(Shell, tokenizer::Tokenizer)> {
    if !engine_metal::device::present() {
        eprintln!("skipping {what}: this machine publishes no Metal device");
        return None;
    }
    let trace = models::trace_of(sku).expect("the catalog ships the row")(Platform::Metal);
    let budgets = engine::load::Budgets {
        max_tokens,
        max_lanes: MAX_LANES,
        ..engine::load::Budgets::default()
    };
    // `None` for the text-only twin, whose plan states no patch row — the
    // ladder is derived off `Dim::Patches` and never off a flag.
    let ladder = engine_metal::api::patch_ladder(&trace, &budgets);
    let import = models::import_of(sku).expect("the catalog ships an import");
    let found = snapshots().into_iter().find_map(|snapshot| {
        let container = container(&snapshot)?;
        let source = ztensor_compat::index(&container).ok()?;
        // Read for this shell (§J4c): a family's text may state a `Dtype`
        // PLACEMENT, and a contract read under a different setup than the
        // trace describes different planes. See `four_bit_first_light::ready`.
        let contract = models::placing_for(Platform::Metal, || import(&source)).ok()?;
        Some((snapshot, contract))
    });
    let Some((checkpoint, contract)) = found else {
        eprintln!(
            "skipping {what}: no mlx-community Qwen3.5-0.8B-4bit snapshot in the HF cache \
             satisfies {sku}"
        );
        return None;
    };
    let tok = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint ships a tokenizer");
    let shell = Shell::load(Boot {
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        // §M-4c, as `serve_smoke` states it: an unstamped snapshot proceeds,
        // and the deployment's facts are stated honestly all the same.
        tp_size: 1,
        precision: models::precision_of(sku)
            .expect("the catalog states this row's precision")
            .to_string(),
        budget: LoadBudget::new(MAX_LANES, max_tokens),
        patches: ladder,
        profile: None,
        page_size: 16,
        context,
        slots: 4,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
    })
    .expect("the vision shell loads");
    Some((shell, tok))
}

fn word(rows: u32, media: bool) -> u64 {
    word_of(SKU, rows, media)
}

fn word_of(sku: &str, rows: u32, media: bool) -> u64 {
    let classify = models::classify_of(sku).expect("the catalog ships a classify");
    classify(&model_dsl::Request::new(rows, false).with_media(media))
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0u32;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best as usize] {
            best = i as u32;
        }
    }
    best
}

/// One media prefill, then `NEW_TOKENS` greedy text decodes in the same slot.
fn caption(
    shell: &mut Shell,
    tok: &tokenizer::Tokenizer,
    slot: u32,
    shot: &Shot,
) -> (Vec<f32>, Vec<u32>, String) {
    caption_n(shell, tok, slot, shot, NEW_TOKENS)
}

/// The same, with the decode budget stated.
///
/// **A BUDGET IS A FIXTURE PARAMETER AND NOT A CONSTANT**, because this
/// checkpoint opens with an empty `<think>` block and sometimes opens with two
/// — four rows and sometimes eight before the answer starts. A gate whose
/// prompt asks a longer question (gate (r) asks about the SECOND of two
/// images, and the model answers it in a sentence rather than a word) runs out
/// of `NEW_TOKENS` mid-clause and reads as a caption that named no colour.
fn caption_n(
    shell: &mut Shell,
    tok: &tokenizer::Tokenizer,
    slot: u32,
    shot: &Shot,
    new_tokens: usize,
) -> (Vec<f32>, Vec<u32>, String) {
    shell.open(slot).expect("the slot opens");
    let lanes = [Seated::of(Lane {
        slot,
        word: word(shot.tokens.len() as u32, true),
        tokens: &shot.tokens,
    })];
    let media = [shot.media()];
    let prefill = fire(shell, &lanes, &media);
    let mut produced = vec![argmax(&prefill)];
    for _ in 0..new_tokens {
        let fed = [*produced.last().expect("a step feeds the last back")];
        let lanes = [Seated::of(Lane {
            slot,
            word: word(1, false),
            tokens: &fed,
        })];
        let row = fire(shell, &lanes, &[]);
        produced.push(argmax(&row));
    }
    let text = tok.decode(&produced, false);
    (prefill, produced, text)
}

/// The three phases, spelled here because the native `Shell::fire` hardcodes
/// an empty media slice — this shell's media door is `crate::api`'s, and a
/// test that wants one fires the frame trait itself.
fn fire(shell: &mut Shell, lanes: &[Seated<'_>], media: &[Media<'_>]) -> Vec<f32> {
    let prepared = FrameShell::prepare(
        shell,
        StepView {
            lanes,
            attachments: &[],
            media,
            done: None,
        },
        None,
    )
    .expect("the step stages");
    let enqueued = FrameShell::enqueue(shell, prepared).expect("the step enqueues");
    let landed = FrameShell::settle(shell, enqueued).expect("the step settles");
    let mut rows = shell.rows_of(&landed).expect("the step answers");
    rows.remove(0)
}

const COLORS: [(&str, [u8; 3]); 3] = [
    ("red", [255, 0, 0]),
    ("green", [0, 255, 0]),
    ("blue", [0, 0, 255]),
];

#[test]
fn three_colours_caption_differently_and_name_what_they_are() {
    let _lock = serialized();
    let Some((mut shell, tok)) = ready("three colours caption differently") else {
        return;
    };
    let mut said: Vec<(String, Vec<f32>, String)> = Vec::new();
    for (slot, (name, rgb)) in COLORS.iter().enumerate() {
        let s = shot(*rgb, &tok, false);
        assert_eq!(s.softs[0], 64, "a 256x256 square is 64 soft tokens");
        assert_eq!(s.rows[0], 256, "and 256 patch rows behind them");
        let (logits, ids, text) = caption(&mut shell, &tok, slot as u32, &s);
        eprintln!(
            "{name:>6}: anchor {} soft {} -> {text:?}  ids {:?}",
            s.anchors[0], s.softs[0], ids
        );
        said.push((name.to_string(), logits, text));
    }
    for i in 0..said.len() {
        for j in i + 1..said.len() {
            let (a, b) = (&said[i], &said[j]);
            let moved =
                a.1.iter()
                    .zip(&b.1)
                    .filter(|(x, y)| (*x - *y).abs() > 1e-3)
                    .count();
            eprintln!(
                "  {} vs {}: {moved} logits apart, captions {:?} / {:?}",
                a.0, b.0, a.2, b.2
            );
        }
    }
    let distinct: std::collections::BTreeSet<&str> =
        said.iter().map(|(_, _, t)| t.as_str()).collect();
    assert_eq!(
        distinct.len(),
        said.len(),
        "three different pictures answered {distinct:?}. The tower fired and the logits \
         moved, so this is not a scatter that dropped or a payload that never arrived — \
         it is the patch embedding read down the wrong axis. See this file's header and \
         `the_bank_s_lane_order_is_the_one_that_names_the_colours`: `mlx_lm` stores \
         `vision_tower.patch_embed.proj.weight` channels-last, [hidden, T, P, P, C], and \
         `models::qwen_3::import`'s `flattened` transmute reads it as [hidden, C, T, P, P]"
    );
    for (name, _, text) in &said {
        assert!(
            text.to_lowercase().contains(name.as_str()),
            "the {name} square captioned {text:?}, which names no colour it is"
        );
    }
}

/// **WHICH LANE ORDER THE LOADED BANK IS IN, ASKED AS AN ANSWER.**
///
/// One picture, two payloads: the front-end's own `(c, t, r, q)` row, and the
/// same row re-laid `(t, r, q, c)`. A patch embedding is a contraction, so
/// whichever of the two matches the BANK's stored axis order is the one that
/// computes the checkpoint's own numbers — and the other is a permutation of
/// them. Exactly one of the two therefore names the colours, and which one it
/// is IS the bank's layout, read off the model rather than off a shape.
///
/// This is the gate that localizes [`three_colours_caption_differently_and_
/// name_what_they_are`]'s failure to `qwen_3::import` rather than to anything
/// this engine does, and it keeps saying something true after the import is
/// fixed — the answer simply moves to the other arm.
#[test]
fn the_bank_s_lane_order_is_the_one_that_names_the_colours() {
    let _lock = serialized();
    let Some((mut shell, tok)) = ready("which lane order the bank is in") else {
        return;
    };
    let mut named: Vec<(&str, usize, Vec<String>)> = Vec::new();
    for (order, channels_last) in [("channel-major", false), ("channels-last", true)] {
        let mut said = Vec::new();
        let mut hits = 0usize;
        for (slot, (name, rgb)) in COLORS.iter().enumerate() {
            let s = shot(*rgb, &tok, channels_last);
            let (_, _, text) = caption(&mut shell, &tok, slot as u32, &s);
            if text.to_lowercase().contains(name) {
                hits += 1;
            }
            said.push(text);
        }
        eprintln!("  {order:>13} payload: {hits}/3 named their colour — {said:?}");
        named.push((order, hits, said));
    }
    let winners: Vec<&str> = named
        .iter()
        .filter(|(_, hits, _)| *hits == COLORS.len())
        .map(|(order, _, _)| *order)
        .collect();
    assert_eq!(
        winners.len(),
        1,
        "exactly one lane order can be the bank's own, and {} answered for all three \
         colours — {named:?}",
        winners.len()
    );
    eprintln!(
        "the loaded patch-embed bank is laid out {}: a payload in that order names every \
         colour and the other order names none",
        winners[0]
    );
}

/// **SESSION I's GATE (o): TEXT-LANE INVARIANCE.**
///
/// A text-only fire of `qwen35-d0.8b-mlxu4-kv-bf16` is byte-identical to the
/// same fire of the `-vision-` row with no image attached. It is a PROPERTY
/// and not a hope: the tower's rectangles are all `Dim::Patches`, an
/// axis-empty fire has zero of them, `Inputs::write` stages nothing and no
/// seat is bound — so the two loads walk the same regions over the same
/// numbers. The one thing that could move is the trunk's rotation, and the
/// vision row's `MropePositions` stream fills `(p, p, p)` for a lane that
/// submitted nothing, which is the angle the scalar entry would have used.
///
/// Bit-identical and not "close": a difference here is a different plan, not a
/// different rounding.
#[test]
fn a_text_only_fire_of_the_vision_row_is_the_text_rows_bit_for_bit() {
    let _lock = serialized();
    const TEXT_SKU: &str = "qwen35-d0.8b-mlxu4-kv-bf16";
    let Some((mut vision, tok)) = ready_as(SKU, "text-lane invariance (vision row)") else {
        return;
    };
    let Some((mut text, _)) = ready_as(TEXT_SKU, "text-lane invariance (text row)") else {
        return;
    };
    let prompt =
        tok.encode("<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n");
    let rows = prompt.len() as u32;

    let fire_on = |shell: &mut Shell, sku: &str| {
        shell.open(0).expect("the slot opens");
        let lanes = [Seated::of(Lane {
            slot: 0,
            word: word_of(sku, rows, false),
            tokens: &prompt,
        })];
        fire(shell, &lanes, &[])
    };
    let with_tower = fire_on(&mut vision, SKU);
    let without = fire_on(&mut text, TEXT_SKU);

    assert_eq!(with_tower.len(), without.len(), "one vocabulary, two loads");
    let apart = with_tower
        .iter()
        .zip(&without)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    eprintln!(
        "text-lane invariance: {rows} rows, {} logits, {apart} bits apart; argmax {} vs {}",
        with_tower.len(),
        argmax(&with_tower),
        argmax(&without),
    );
    assert_eq!(
        apart, 0,
        "a text-only fire of the vision row moved {apart} logits against the text row — the \
         patch axis is supposed to cost an axis-empty fire nothing at all"
    );
}


// ── SESSION M's TWO DEVICE GATES (metal-verify-queue, "Session M banked") ───
//
// Both bugs were found and fixed host-side in `runtime::pipeline::media`, and
// both are pinned there by assertions that fail on the old derivation. What a
// host gate cannot ask is whether the SCATTER and the ROTATION then answer the
// picture. These two ask it.

/// **GATE (r): TWO IMAGES IN ONE LANE, ON A REAL FOLD.**
///
/// Gate (p) is two image LANES in one fire, which exercises `patch_offset /
/// fold`. This is two image SPANS in ONE lane, which exercises the shape of
/// the submission's own `routes` vector — and for one span the two shapes are
/// the same vector, which is why the bug was silent. Both shells read a lane's
/// routes as ONE fold-space prefix `patches / fold` long over the lane's TOTAL
/// payload rows, so the spans' addresses go down back to back with a single
/// `-1` tail at the end. The producer used to pad each span to its OWN payload
/// row count, which put image 0's tail INSIDE the live prefix and image 1's
/// addresses past the end of it: image 1's soft tokens were all dropped, every
/// length still agreed, and the pass captioned image 0 twice.
///
/// **THREE CLAIMS, WEAKEST LAST.**
///
/// 1. **THE VECTOR** — the cheap pre-check the queue names. The live prefix
///    holds both runs' anchors with no `-1` between them, and the tail is one
///    tail at the end.
/// 2. **THE SECOND PICTURE REACHES THE TRUNK** — `(red, blue)` and
///    `(red, red)` are the same token ledger, the same route vector and the
///    same row counts, differing only in the second square's pixels. A dropped
///    span answers them IDENTICALLY. This claim is free of the model's taste
///    and is the one that catches the bug.
/// 3. **AND THE CAPTION NAMES THE SECOND COLOUR**, asked in both orders so a
///    model that simply prefers one word cannot pass it.
#[test]
fn two_images_in_one_lane_both_reach_the_trunk() {
    let _lock = serialized();
    let Some((mut shell, tok)) = ready("two images in one lane") else {
        return;
    };
    const SECOND: &str =
        "What is the dominant colour of the SECOND image? Answer with one colour word.";
    // Long enough for the empty `<think>` block (sometimes two) and a sentence.
    const ANSWER_TOKENS: usize = 24;
    let red = [255u8, 0, 0];
    let blue = [0u8, 0, 255];

    // ── 1. THE VECTOR.
    let pair = shot_of(&[red, blue], &tok, SECOND, 0, false);
    assert_eq!(pair.rows.len(), 2, "one lane, two spans");
    assert_eq!(pair.softs, vec![64, 64], "two 256x256 squares are 64 soft tokens each");
    let live: usize = pair.softs.iter().map(|&n| n as usize).sum();
    let owed: usize = pair.rows.iter().map(|&r| r as usize).sum();
    assert_eq!(pair.routes.len(), owed, "one route per payload row");
    let want: Vec<i32> = pair
        .anchors
        .iter()
        .zip(&pair.softs)
        .flat_map(|(&a, &n)| (0..n).map(move |k| (a + k) as i32))
        .collect();
    assert_eq!(
        &pair.routes[..live],
        &want[..],
        "the live prefix is the lane's spans' addresses CONCATENATED; a `-1` inside it is \
         the per-span padding this gate exists to catch"
    );
    assert!(
        pair.routes[live..].iter().all(|&r| r == -1),
        "the `-1` tail is one tail at the end of the lane"
    );
    eprintln!(
        "  routes: {} payload rows, live prefix {live} = anchors {:?} x softs {:?}, tail {} x -1",
        owed,
        pair.anchors,
        pair.softs,
        owed - live
    );

    // ── 2. THE SECOND PICTURE REACHES THE TRUNK.
    let same = shot_of(&[red, red], &tok, SECOND, 0, false);
    assert_eq!(
        same.tokens, pair.tokens,
        "the two lanes are the same ledger; only the second square's pixels differ"
    );
    assert_eq!(same.routes, pair.routes, "and the same route vector");
    let (pair_logits, _, pair_text) = caption_n(&mut shell, &tok, 0, &pair, ANSWER_TOKENS);
    let (same_logits, _, same_text) = caption_n(&mut shell, &tok, 1, &same, ANSWER_TOKENS);
    let moved = pair_logits
        .iter()
        .zip(&same_logits)
        .filter(|(a, b)| (*a - *b).abs() > 1e-3)
        .count();
    eprintln!(
        "  (red, blue) -> {pair_text:?}\n  (red, red)  -> {same_text:?}\n  {moved} of {} logits apart",
        pair_logits.len()
    );
    assert!(
        moved > 0,
        "changing ONLY the second image's pixels moved no logit at all — the second span's \
         soft tokens never reached the trunk, which is the two-images-one-lane bug exactly"
    );

    // ── 3. AND THE CAPTION NAMES THE SECOND COLOUR, in both orders.
    let flipped = shot_of(&[blue, red], &tok, SECOND, 0, false);
    let (_, _, flipped_text) = caption_n(&mut shell, &tok, 2, &flipped, ANSWER_TOKENS);
    eprintln!("  (blue, red) -> {flipped_text:?}");
    for (order, second, text) in [
        ("(red, blue)", "blue", &pair_text),
        ("(blue, red)", "red", &flipped_text),
    ] {
        assert!(
            text.to_lowercase().contains(second),
            "{order} asked for the SECOND colour and captioned {text:?}, which does not name \
             {second}. A run that names only the FIRST colour is the per-span padding \
             surviving somewhere — check claim 1's vector at the producer."
        );
    }
}

/// **GATE (s): THE M-ROPE OFFSET IN A LONG CONTEXT.**
///
/// The image triples used to be the RAW merged-grid coordinate `(0, 0..h,
/// 0..w)` with no run-start offset, so an image's positions ran BACKWARD past
/// everything before it — and a caption prompt is ~40 tokens, so the rotation
/// barely noticed. The error grows with the distance from the sequence start,
/// which is why the bug stayed latent and why this gate puts ~2 000 rows of
/// filler in front of the picture.
///
/// **THE CLAIM IS THAT BOTH ARMS NAME THE COLOUR.** Before the fix the long
/// arm degrades and the short one does not; after it, both stand.
///
/// **AND THE SECOND HALF IS THE DECODE STEP**, which this gate MEASURES and
/// does not fix. A lane's `token_positions` is submitted only on the fire that
/// carries the run — `StepMedia::validate` refuses a media row naming no spans
/// — so a decode fire falls back to scalar `(p, p, p)` at the ABSOLUTE token
/// row, which is `soft − position_span` past where the M-RoPE cursor actually
/// ended. Upstream carries `mrope_position_deltas` for exactly this. The
/// arithmetic is asserted here so the size of the divergence is on the record;
/// the delta belongs on the SEQUENCE and not in `lane_media`, so nothing here
/// improvises it.
#[test]
fn an_image_deep_in_the_context_still_names_its_colour() {
    let _lock = serialized();
    const FILLER_ROWS: usize = 2000;
    const LONG_CONTEXT: u32 = 4096;
    let Some((mut shell, tok)) = ready_with(SKU, "an image deep in the context", 4096, LONG_CONTEXT)
    else {
        return;
    };
    let mut said: Vec<(&str, String)> = Vec::new();
    let mut divergence: Vec<u32> = Vec::new();
    for (slot, (arm, filler)) in [("short", 0usize), ("long", FILLER_ROWS)]
        .into_iter()
        .enumerate()
    {
        let s = shot_of(&[[0, 0, 255]], &tok, QUESTION, filler, false);
        assert!(
            (s.tokens.len() as u32) < LONG_CONTEXT,
            "{arm}: {} rows do not fit a {LONG_CONTEXT}-token context",
            s.tokens.len()
        );
        // **THE DECODE-SIDE DIVERGENCE, MEASURED.** The cursor advanced by
        // `position_span` over the run and the row index by `token_count`, so
        // the first decode row is fed `tokens.len()` where M-RoPE ended at
        // `cursor_end`. It is a property of the IMAGE and not of the filler.
        divergence.push(s.tokens.len() as u32 - s.cursor_end);
        let (_, _, text) = caption(&mut shell, &tok, slot as u32, &s);
        eprintln!(
            "  {arm:>5}: {} rows ({filler} filler), anchor {}, m-rope cursor ended at {} -> \
             the decode is fed {} and is {} positions PAST it; caption {text:?}",
            s.tokens.len(),
            s.anchors[0],
            s.cursor_end,
            s.tokens.len(),
            divergence[slot],
        );
        said.push((arm, text));
    }
    assert_eq!(
        divergence[0], divergence[1],
        "the decode-side divergence is the image's own (`token_count - position_span`) and \
         must not depend on how much text precedes it; it moved {divergence:?}"
    );
    assert_eq!(
        divergence[0], 56,
        "a 256x256 square is 64 soft tokens over an 8x8 merged grid, so its run spends 64 \
         token rows and 8 positions: every decode row after it is fed 56 positions past the \
         M-RoPE cursor. That is `mrope_position_deltas`' job and it belongs on the SEQUENCE, \
         so this gate measures it and does not fix it."
    );
    for (arm, text) in &said {
        assert!(
            text.to_lowercase().contains("blue"),
            "the {arm} arm captioned {text:?}, which names no colour it is. A LONG arm that \
             fails beside a green SHORT one is the run-start offset missing from the image's \
             triples ({}); both failing is a different fault.",
            "runtime::pipeline::media::lane_media"
        );
    }
}
