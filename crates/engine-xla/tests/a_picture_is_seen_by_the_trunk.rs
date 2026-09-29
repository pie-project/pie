//! Qwen3.5's image tower on the device, against numbers Hugging Face
//! computes for the same picture and prompt. Asked for with
//! `PIE_XLA_ARTIFACT` naming a Qwen3.5 vision artifact (e.g.
//! `~/.pie/models/Qwen--Qwen3.5-0.8B/*vision*.xla.zt`).
//!
//! The picture is the `image-captioning` inferlet's: a solid 224 x 224
//! square, which the preprocessing lifts to 256 x 256 (a 16 x 16 patch grid,
//! 64 merged rows). The prompt is that inferlet's too: the system turn, the
//! span, the user turn, the bare assistant header. Upstream (transformers,
//! f32 and bf16) answers `<think>\n\n</think>\n\nred<|im_end|>` for it.
//!
//! The square is prefilled twice: in one fire, and in the inferlet's three
//! (the text before, the run alone, the text after), every fire rotating at
//! upstream's `get_rope_index` positions — the text after a span sits
//! `h·w - max(h, w)` behind its token row. The inferlet once rotated that
//! tail at its raw row and the served caption of the red square became
//! "a solid, solid-filled rectangle"; the tower's rows were never at fault
//! (cosine 0.998 against transformers' f32 rows, as close as its own bf16).
//!
//! `PIE_XLA_VISION_DUMP=<dir>` writes the tower's rows and each first-step
//! logit row as raw f32 for comparing against a reference offline.

mod common;

use engine_xla::{Boot, DeviceBoot, Lane, Seated, Shell};
use model_compiler::{Budget, PatchLadder};
use model_dsl::{Operands, Platform, Request};
use models::media::{Rgb8, VisionFrontEnd};

/// `<|im_start|>system\nYou are a helpful assistant that describes images.<|im_end|>\n`
const BEFORE: [u32; 14] = [
    248045, 8678, 198, 2523, 513, 264, 10631, 17313, 421, 16067, 5167, 13, 248046, 198,
];
/// `<|im_start|>user\nWhat is the dominant colour of the image above? Answer
/// with one word.<|im_end|>\n<|im_start|>assistant\n`
const AFTER: [u32; 23] = [
    248045, 846, 198, 3710, 369, 279, 23681, 12106, 314, 279, 2099, 3294, 30, 21134, 440, 799,
    3299, 13, 248046, 198, 248045, 74455, 198,
];
const VISION_START: u32 = 248053;
const IMAGE_PAD: u32 = 248056;
const VISION_END: u32 = 248054;
const IM_END: u32 = 248046;
const THINK_CLOSE: u32 = 248069;

fn argmax(xs: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, v) in xs.iter().enumerate() {
        if *v > xs[best] {
            best = at;
        }
    }
    best as u32
}

fn bf16(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    ((rounded >> 16) as u16).to_le_bytes()
}

fn nearest(src: &Rgb8, h: u32, w: u32) -> Rgb8 {
    let mut data = Vec::with_capacity((h * w * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let (sy, sx) = (y * src.h / h, x * src.w / w);
            let at = ((sy * src.w + sx) * 3) as usize;
            data.extend_from_slice(&src.data[at..at + 3]);
        }
    }
    Rgb8 { h, w, data }
}

fn dump(name: &str, values: &[f32]) {
    let Some(dir) = std::env::var_os("PIE_XLA_VISION_DUMP") else {
        return;
    };
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(std::path::Path::new(&dir).join(name), bytes).expect("the dump writes");
}

/// One prefill of the square and a greedy decode after it; the first
/// logits, the probes of a one-fire prefill, and the tokens.
fn caption(
    shell: &mut Shell,
    word: &dyn Fn(u32, bool) -> u64,
    span: &models::media::EncodedSpan,
    split: bool,
) -> (Vec<f32>, Vec<(u32, Vec<f32>)>, Vec<u32>) {
    let pads = span.token_count as usize;
    let before = BEFORE.len();
    let run_end = before + pads + 2;

    let mut tokens: Vec<u32> = BEFORE.to_vec();
    tokens.push(VISION_START);
    tokens.extend(std::iter::repeat_n(IMAGE_PAD, pads));
    tokens.push(VISION_END);
    tokens.extend(AFTER);

    // Upstream's `get_rope_index`: text advances by one, the span's rows sit
    // at `(start, start + y, start + x)` and advance the cursor by `max(h, w)`.
    let lag = span.token_count - span.position_span;
    let row_position = |row: usize| -> u32 {
        if row < run_end { row as u32 } else { row as u32 - lag }
    };
    let mut triples = Vec::with_capacity(tokens.len() * 3);
    let gw = span.grid.w as i32;
    for row in 0..tokens.len() {
        let p = row_position(row) as i32;
        if (before + 1..before + 1 + pads).contains(&row) {
            let k = (row - before - 1) as i32;
            let start = (before + 1) as i32;
            triples.extend([start, start + k / gw, start + k % gw]);
        } else if row == run_end - 1 {
            triples.extend([(before + 1) as i32 + span.position_span as i32; 3]);
        } else {
            triples.extend([p; 3]);
        }
    }

    let patches: Vec<u8> = span.payload.iter().flat_map(|&v| bf16(v)).collect();
    let rows = [span.rows];
    let grid: Vec<i32> = span
        .positions
        .chunks_exact(2)
        .flat_map(|yx| [0, yx[0] as i32, yx[1] as i32])
        .collect();
    // A route per merged row: the pad row it lands on, counted in the fire
    // that carries the run.
    let routes = |anchor: usize| -> Vec<i32> {
        let mut r: Vec<i32> = (0..pads).map(|k| (anchor + k) as i32).collect();
        r.resize(span.rows as usize, -1);
        r
    };

    shell.open(0).expect("slot 0 opens");
    let (logits, probes) = if split {
        let mut fire = |from: usize, to: usize, media: bool| -> Vec<f32> {
            let positions: Vec<u32> = (from..to).map(row_position).collect();
            let run_routes = routes(1);
            let mut seated = Seated::of(Lane {
                slot: 0,
                word: word((to - from) as u32, media),
                tokens: &tokens[from..to],
            });
            seated.positions = &positions;
            if media {
                seated.media = Some(engine_xla::Media {
                    rows: &rows,
                    patches: &patches,
                    routes: &run_routes,
                    positions: &grid,
                    embed_rows: &span.embed_rows,
                    embed_weights: &span.embed_weights,
                    token_positions: &triples[3 * from..3 * to],
                });
            }
            shell.fire_seated(&[seated]).expect("a prefill fires").remove(0)
        };
        fire(0, before, false);
        fire(before, run_end, true);
        let last = fire(run_end, tokens.len(), false);
        (last, Vec::new())
    } else {
        let all_routes = routes(before + 1);
        let mut seated = Seated::of(Lane {
            slot: 0,
            word: word(tokens.len() as u32, true),
            tokens: &tokens,
        });
        seated.media = Some(engine_xla::Media {
            rows: &rows,
            patches: &patches,
            routes: &all_routes,
            positions: &grid,
            embed_rows: &span.embed_rows,
            embed_weights: &span.embed_weights,
            token_positions: &triples,
        });
        let fired = shell.fire_full(&[seated]).expect("the prefill fires");
        let probes = fired.probes.into_iter().map(|(_, w, v)| (w, v)).collect();
        (fired.rows[0].clone(), probes)
    };

    let next = row_position(tokens.len());
    let mut produced = vec![argmax(&logits)];
    for step in 0..15u32 {
        let last = *produced.last().expect("a token");
        if last == IM_END {
            break;
        }
        let fed = [last];
        let at = [next + step];
        let mut seated = Seated::of(Lane {
            slot: 0,
            word: word(1, false),
            tokens: &fed,
        });
        seated.positions = &at;
        let row = shell.fire_seated(&[seated]).expect("a decode fires").remove(0);
        produced.push(argmax(&row));
    }
    (logits, probes, produced)
}

#[test]
fn a_solid_square_is_named_by_its_colour() {
    let Some(m) = common::model() else {
        eprintln!("not asked: set PIE_XLA_ARTIFACT to a Qwen3.5 vision artifact");
        return;
    };
    if !m.sku.name.contains("vision") {
        eprintln!("not asked: {} carries no image tower", m.sku.name);
        return;
    }
    let trace = (m.sku.trace)(Platform::Xla);
    let classify = m.sku.classify;
    let word = |len: u32, media: bool| classify(&Request::new(len, false).with_media(media));

    // The first patch rows (embedding plus position table) and the tower's
    // output, which the trunk's scatter lands onto the pad rows.
    let mut probes = Vec::new();
    for node in &trace.nodes {
        let mut ins = Vec::new();
        let mut outs = Vec::new();
        node.op.inputs(&mut ins);
        node.op.outputs(&mut outs);
        if node.op.name() == "elementwise.residual_add" && probes.is_empty() {
            probes.extend(outs.first().copied());
        }
        if node.op.name() == "layout.scatter_live_rows" {
            probes.push(ins[0]);
            break;
        }
    }
    assert_eq!(probes.len(), 2, "the plan embeds patches and scatters the tower's rows");

    let vision = models::qwen_3::media::Qwen35Vision::new();
    let tokenizer = common::tokenizer(&m);
    let context = 256;
    let _device = engine_xla::bench::lock_device();
    let mut shell = Shell::load(Boot {
        trace,
        contract: &m.contract,
        checkpoint: &m.checkpoint,
        budget: Budget::new(2, context),
        page_size: 16,
        context,
        slots: 2,
        pages: 2 * context / 16,
        device: &DeviceBoot::default(),
        patches: Some(PatchLadder::new(256, 1)),
    })
    .expect("the shell loads");
    shell.probe(probes);


    for (name, rgb) in [("red", [255u8, 0, 0]), ("green", [0, 255, 0]), ("blue", [0, 0, 255])] {
        let side = 224u32;
        let picture = Rgb8::new(side, side, rgb.repeat((side * side) as usize)).expect("rgb");
        let span = vision
            .encode(&picture, models::media::Budget::Still, nearest)
            .expect("the square encodes");
        assert_eq!(span.token_count, 64, "224 x 224 lifts to a 16 x 16 patch grid");

        let (whole, probes, whole_tokens) = caption(&mut shell, &word, &span, false);
        for (at, (width, values)) in probes.iter().enumerate() {
            dump(&format!("xla_{name}_probe{at}.f32"), values);
            assert!(
                values.iter().all(|v| v.is_finite()),
                "probe {at} ({width} wide) holds a non-finite value"
            );
        }
        dump(&format!("xla_{name}_logits.f32"), &whole);
        let (split, _, split_tokens) = caption(&mut shell, &word, &span, true);
        dump(&format!("xla_{name}_split_logits.f32"), &split);
        let worst = whole
            .iter()
            .zip(&split)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("{name}: one fire {whole_tokens:?}, three fires {split_tokens:?}, max |Δlogit| {worst}");
        // The same rows at the same rotations, cut three ways: the tail must
        // still see the span's keys. Text alone cut in two moves the logits
        // by ~0.2; a tail bounded at its rotation instead of its row moved
        // them by ~20 (correlation 0.39).
        assert!(
            worst < 1.5,
            "a {name} square prefilled in three fires reads different logits than in one: \
             max |Δ| {worst}"
        );

        for (how, produced) in [("one fire", &whole_tokens), ("three fires", &split_tokens)] {
            let after = produced
                .iter()
                .position(|&t| t == THINK_CLOSE)
                .map_or(&produced[..], |at| &produced[at + 1..]);
            let text = tokenizer.decode(after, true).to_lowercase();
            eprintln!("{name} ({how}): {text:?}");
            assert!(
                text.contains(name),
                "the caption of a solid {name} square prefilled in {how} does not name it: {text:?}"
            );
        }
    }
}
