//! Qwen3.5-0.8B's text trunk on the device against Hugging Face's f32
//! forward of the same prompt. Asked for with `PIE_XLA_ARTIFACT` naming a
//! Qwen3.5-0.8B bf16 artifact, text or vision (e.g.
//! `~/.pie/models/Qwen--Qwen3.5-0.8B/*d0-8b-{bf16,vision-bf16}-kv-bf16.xla.zt`).
//!
//! transformers in bf16 lands within 0.22 of its own f32 logits on the ids
//! below (0.38 over the whole vocab, correlation 0.9996); the device lands
//! within 0.10 (0.20, correlation 0.99989). The vision plan's trunk rotated
//! M-RoPE pairs `(i, i + d/2)` at `theta^(-2i/d)` over the whole 256-wide
//! head instead of upstream's 64-wide rotated prefix: from the first full
//! attention layer on the residual stream sat 12% off (bf16: 1%), the logits
//! 3.8 off here (6.1 over the vocab, correlation 0.907).
//!
//! `PIE_XLA_TRUNK_DUMP=<dir>` writes every probed value (`PIE_XLA_TRUNK_OPS`,
//! a comma list of op names, default the residual adds) and the logits as
//! raw f32, with a `manifest.txt` naming each file's layer, op and width, for
//! a layer-by-layer comparison against a reference offline.

mod common;

use std::collections::HashMap;
use std::fmt::Write as _;

use engine_xla::{Boot, DeviceBoot, Lane, Shell};
use poem_compiler::Budget;
use poem_dsl::{Operands, Platform, Request};

/// The `image-captioning` inferlet's text turns without the picture.
const PROMPT: [u32; 37] = [
    248045, 8678, 198, 2523, 513, 264, 10631, 17313, 421, 16067, 5167, 13, 248046, 198, 248045,
    846, 198, 3710, 369, 279, 23681, 12106, 314, 279, 2099, 3294, 30, 21134, 440, 799, 3299, 13,
    248046, 198, 248045, 74455, 198,
];

/// transformers' f32 logits of the last prompt row (Qwen3.5-0.8B snapshot,
/// `Qwen3_5ForConditionalGeneration`, CPU) at the 24 highest ids and every
/// 4099th id.
const IDS: [u32; 85] = [
    0, 9, 16, 17, 18, 21, 27, 30, 196, 760, 4099, 4754, 7676, 8198, 9008, 12297, 13314, 16396,
    20495, 23892, 24594, 27775, 28693, 32792, 36891, 40990, 45089, 49188, 53287, 57386, 61485,
    65584, 69267, 69683, 71093, 73782, 74455, 77881, 81980, 86079, 90178, 94277, 98376, 102475,
    106574, 110673, 114772, 118871, 122970, 127069, 131168, 135267, 139366, 143465, 147564, 148013,
    151663, 155762, 159861, 163960, 168059, 172158, 176257, 180356, 184455, 188554, 192653, 196752,
    200851, 204950, 209049, 213148, 217247, 221346, 225445, 229544, 233643, 237742, 241841, 245940,
    248045, 248046, 248058, 248068, 248069,
];
const UPSTREAM: [f32; 85] = [
    7.5333, 13.3616, 15.0337, 13.2323, 13.5798, 12.7595, 17.0193, 16.4004, 12.8645, 13.8990,
    -0.4498, 14.7407, 13.7775, -2.2394, 14.8447, 2.2835, 13.9148, -1.8909, 1.7621, 12.8815, 1.0956,
    12.7375, -1.1291, -1.5254, -0.4783, 2.5491, -1.8969, -4.5659, 0.0067, -4.9702, -2.1957, 2.0186,
    17.1565, 0.2867, 13.0812, -1.7100, 17.3996, 1.8675, -2.1807, -0.8257, -1.6987, -0.6348, 2.2747,
    -0.5511, -1.3648, 2.5340, -0.1421, 1.4243, -1.5899, -1.0337, -0.8114, -2.8459, -4.0638,
    -2.3293, -2.0434, 13.1271, 2.8182, 0.5275, -1.2241, -3.0995, -3.4788, 2.4493, -1.5550, 0.3126,
    0.9981, 0.5832, -0.7969, 0.0063, 2.1442, 0.9744, -2.5079, -1.0482, -2.7244, 0.6404, -1.7722,
    -4.0096, -2.8459, -2.6396, -0.5003, -1.6977, 15.2330, 18.1293, 16.4301, 35.9386, 21.7368,
];

/// transformers' own bf16 forward lands 0.22 off; the M-RoPE mispairing
/// landed 3.8 off.
const MAX_DELTA: f32 = 0.25;

/// A token-by-token walk of the prompt (see the assertion).
const WALK_DELTA: f32 = 0.35;

fn correlation(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().map(|&v| f64::from(v)).sum::<f64>() / n,
        b.iter().map(|&v| f64::from(v)).sum::<f64>() / n,
    );
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x) - ma, f64::from(y) - mb);
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa * bb).sqrt()
}

fn dump_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("PIE_XLA_TRUNK_DUMP").map(Into::into)
}

fn dump(name: &str, values: &[f32]) {
    let Some(dir) = dump_dir() else {
        return;
    };
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(dir.join(name), bytes).expect("the dump writes");
}

#[test]
fn the_first_logits_track_upstream() {
    let Some(m) = common::model() else {
        eprintln!("not asked: set PIE_XLA_ARTIFACT to a Qwen3.5-0.8B bf16 artifact");
        return;
    };
    if !m.sku.name.starts_with("qwen35-d0.8b") {
        eprintln!("not asked: {} is not Qwen3.5-0.8B", m.sku.name);
        return;
    }
    let trace = m.sku.trace(Platform::Xla);
    let facts = m.sku.trace(models::Platform::Xla).facts;
    let word = |len: u32| facts.word(&Request::new(len, false));

    let wanted: Vec<String> = std::env::var("PIE_XLA_TRUNK_OPS")
        .unwrap_or_else(|_| "elementwise.residual_add".into())
        .split(',')
        .map(str::to_string)
        .collect();
    let mut named = HashMap::new();
    let mut probes = Vec::new();
    if dump_dir().is_some() {
        for node in &trace.nodes {
            if !wanted.iter().any(|w| w == node.op.name()) {
                continue;
            }
            let mut outs = Vec::new();
            node.op.outputs(&mut outs);
            for (at, value) in outs.into_iter().enumerate() {
                let layer = node.layer.map_or(-1, |l| l as i64);
                named.insert(value, format!("L{layer:02}_{}_{at}", node.op.name()));
                probes.push(value);
            }
        }
    }

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
        // A vision artifact's plan sizes its tower by the patch ladder.
        patches: m
            .sku
            .name
            .contains("vision")
            .then(|| poem_compiler::PatchLadder::new(256, 1)),
    })
    .expect("the shell loads");
    shell.probe(probes);

    shell.open(0).expect("slot 0 opens");
    let fired = shell
        .fire_full(&[engine_xla::Seated::of(Lane {
            slot: 0,
            word: word(PROMPT.len() as u32),
            tokens: &PROMPT,
        })])
        .expect("the prefill fires");
    // The fire may keep its readout on the device; a buffer must not
    // outlive the client the shell owns.
    let (mut rows, probed) = (fired.rows, fired.probes);
    drop(fired.kept);
    let logits = rows.remove(0);
    dump("xla_logits.f32", &logits);
    let mut manifest = String::new();
    for (seq, (value, width, values)) in probed.iter().enumerate() {
        let name = format!("{seq:03}_{}.f32", named[value]);
        writeln!(manifest, "{name} {width}").expect("a string takes a line");
        dump(&name, values);
    }
    if let Some(dir) = dump_dir() {
        std::fs::write(dir.join("manifest.txt"), manifest).expect("the manifest writes");
    }

    // The same prompt walked one token at a time.
    shell.open(1).expect("slot 1 opens");
    let mut walked = Vec::new();
    for (at, token) in PROMPT.into_iter().enumerate() {
        let lane = Lane {
            slot: 1,
            word: word(1),
            tokens: &[token],
        };
        if at + 1 < PROMPT.len() {
            walked = shell.fire(&[lane]).expect("a decode fires").remove(0);
            continue;
        }
        // The last step's probes, for the same layer-by-layer reading.
        let fired = shell
            .fire_full(&[engine_xla::Seated::of(lane)])
            .expect("a decode fires");
        let (mut rows, probed) = (fired.rows, fired.probes);
        drop(fired.kept);
        walked = rows.remove(0);
        let mut manifest = String::new();
        for (seq, (value, width, values)) in probed.iter().enumerate() {
            let name = format!("walk_{seq:03}_{}.f32", named[value]);
            writeln!(manifest, "{} {width}", &name[5..]).expect("a string takes a line");
            dump(&name, values);
        }
        if let Some(dir) = dump_dir() {
            std::fs::write(dir.join("walk_manifest.txt"), manifest).expect("the manifest writes");
        }
    }
    drop(shell);
    dump("xla_walked_logits.f32", &walked);

    let against = |row: &[f32]| -> (f32, f64) {
        let at: Vec<f32> = IDS.iter().map(|&id| row[id as usize]).collect();
        let worst = at
            .iter()
            .zip(&UPSTREAM)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        (worst, correlation(&at, &UPSTREAM))
    };
    let (worst, r) = against(&logits);
    let (walked_worst, walked_r) = against(&walked);
    eprintln!(
        "{}: vs transformers f32, prefill max |Δlogit| {worst:.4} (correlation {r:.6}), \
         token by token {walked_worst:.4} ({walked_r:.6})",
        m.sku.name
    );
    assert!(
        worst <= MAX_DELTA,
        "the first logits sit {worst} from transformers' f32 (its own bf16: 0.22)"
    );
    assert!(
        r >= 0.9999,
        "the first logits correlate {r} with transformers' f32"
    );
    // The walk rounds the recurrent state to the pool's bf16 once per token
    // (transformers keeps it in f32): it lands about where transformers'
    // bf16 prefill does, 0.22 here, 0.38 over the vocab.
    assert!(
        walked_worst <= WALK_DELTA,
        "the prompt walked token by token sits {walked_worst} from transformers' f32"
    );
    assert!(
        walked_r >= 0.9999,
        "the walked logits correlate {walked_r} with transformers' f32"
    );
}
