//! A capturing lane's score rows served end to end, against the CPU oracle
//! (`engine_cpu::Forward::at_context`): Qwen3's miniature loaded by the
//! shell, four lanes prefilled together and then decoded together, the
//! second and the fourth capturing scores (the first and the third not, so
//! the fire compacts its capturing lanes), the second reading pages too.
//! Each capturing lane's key row (`attn.scores`, a mass a key of the load's
//! context summed over the layers) and page row (`attn.scores.pages`) is the
//! oracle's for the same tokens at two contexts. The rows lie in the fire's
//! guest region, not the arena: the span lays nothing for them, a fire that
//! captures takes no device byte past what the load reserved, and a lane
//! reading no scores reads none. A fire records as any other: the capture
//! table is staged at its arm's fixed seat, and the capturing lanes' pattern
//! is part of the fire's slot table, so the recording a pattern keys binds
//! its own rows.

use engine_cpu::forward::{Output, Values};
use engine_cpu::Forward;
use poem::{Dtype, Request};
use poem_ir::{ParamSource, Trace};

use crate::backend::{Lane, Seated};
use crate::model_logits::{PAGE, fixture, load};

const MODEL: &str = "qwen3-micro-text";
const KEYS: &str = "attn.scores";
const PAGES: &str = "attn.scores.pages";

fn fnv(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// The fixture checkpoint's plane `name` as the suite writes it
/// (`model_logits::write_checkpoint`): seeded by name, a bf16 plane cut to
/// its bf16 bits.
fn plane(trace: &Trace, name: &str, count: usize) -> Vec<f32> {
    let param = trace
        .params
        .iter()
        .find(|param| param.name == name && param.source == ParamSource::Checkpoint)
        .unwrap_or_else(|| panic!("`{name}` is a checkpoint plane"));
    let norm = name.ends_with("norm");
    let mut seed = fnv(name);
    (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let unit = ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
            let value = if norm { 1.0 + 0.05 * unit } else { 0.08 * unit };
            match param.dtype {
                Dtype::Bf16 => f32::from_bits(value.to_bits() & 0xffff_0000),
                _ => value,
            }
        })
        .collect()
}

fn word(trace: &Trace, rows: u32, captures: bool) -> u64 {
    trace
        .facts
        .word(&Request::new(rows, false).capturing_scores(captures))
}

/// A lane's capture: none, its key row, or its key and page rows.
#[derive(Clone, Copy, PartialEq)]
enum Reads {
    Nothing,
    Keys,
    Pages,
}

/// The lanes of `pattern`: their prompts, whether they capture and whether
/// they read pages.
fn lanes(pattern: &[Reads]) -> Vec<(Vec<u32>, bool, bool)> {
    let prompt = |n: u32, k: u32| (0..n).map(|at| (at * k + 3) % 256).collect::<Vec<_>>();
    let prompts = [prompt(23, 37), prompt(45, 53), prompt(9, 29), prompt(20, 61)];
    pattern
        .iter()
        .zip(prompts)
        .map(|(&reads, prompt)| (prompt, reads != Reads::Nothing, reads == Reads::Pages))
        .collect()
}

/// The tokens each lane decodes after its prompt, the same for both.
const DECODE: [u32; 2] = [17, 101];

/// The oracle's key and page rows for `prompt` and then each of `DECODE`.
fn oracle(trace: &Trace, context: u32, prompt: &[u32]) -> Vec<[Vec<f32>; 2]> {
    let mut me = Forward::new(trace.clone(), None, &|name, count| plane(trace, name, count))
        .unwrap_or_else(|why| panic!("the oracle loads {MODEL}: {why}"))
        .at_context(context, PAGE);
    let read = |outs: Vec<Output>| -> [Vec<f32>; 2] {
        [KEYS, PAGES].map(|name| match outs.iter().find(|out| out.name == name) {
            Some(Output {
                values: Values::F32(v),
                ..
            }) => v.clone(),
            _ => panic!("the oracle reads out no `{name}`"),
        })
    };
    let mut steps = vec![read(
        me.step(prompt, 0, word(trace, prompt.len() as u32, true))
            .expect("the oracle prefills"),
    )];
    let mut at = prompt.len() as u32;
    for token in DECODE {
        steps.push(read(
            me.step(&[token], at, word(trace, 1, true))
                .expect("the oracle decodes"),
        ));
        at += 1;
    }
    steps
}

/// How far `got` is from `want`: the root of the summed squared difference
/// over the root of `want`'s summed square.
fn apart(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "a row as wide as the oracle's");
    let diff: f32 = got.iter().zip(want).map(|(a, b)| (a - b) * (a - b)).sum();
    let norm: f32 = want.iter().map(|b| b * b).sum();
    (diff / norm.max(f32::MIN_POSITIVE)).sqrt()
}

fn served(context: u32, pattern: &[Reads]) {
    let fixture = fixture(MODEL, None);
    let trace = &fixture.trace;
    assert!(
        trace.nodes.iter().any(|node| matches!(
            node.op,
            poem_ir::Operation::Attn(poem_ir::Attn::ScoreCapture { .. })
        )),
        "{MODEL} captures scores"
    );
    let Some(mut shell) = load(&fixture, context) else {
        return;
    };
    let lanes = lanes(pattern);
    let n = lanes.len();
    let reads: Vec<Vec<String>> = lanes
        .iter()
        .map(|(_, captures, pages)| match (captures, pages) {
            (false, _) => Vec::new(),
            (true, false) => vec![KEYS.to_string()],
            (true, true) => vec![KEYS.to_string(), PAGES.to_string()],
        })
        .collect();
    let want: Vec<Option<Vec<[Vec<f32>; 2]>>> = lanes
        .iter()
        .map(|(prompt, captures, _)| captures.then(|| oracle(trace, context, prompt)))
        .collect();

    let books = |shell: &crate::backend::Shell| {
        engine_gpu::backend::Context::books(shell.device()).device()
    };
    let mut held = vec![0u32; n];
    for step in 0..=DECODE.len() {
        let tokens: Vec<Vec<u32>> = lanes
            .iter()
            .map(|(prompt, ..)| if step == 0 { prompt.clone() } else { vec![DECODE[step - 1]] })
            .collect();
        let seated: Vec<Seated<'_>> = (0..n)
            .map(|lane| Seated {
                held: Some(held[lane]),
                reads: &reads[lane],
                ..Seated::of(Lane {
                    slot: lane as u32,
                    word: word(trace, tokens[lane].len() as u32, lanes[lane].1),
                    tokens: &tokens[lane],
                })
            })
            .collect();
        let before = books(&shell);
        let outs = shell.fire_outputs(&seated).expect("the fire lands");
        assert_eq!(
            books(&shell),
            before,
            "context {context}, step {step}: a capturing fire takes no device byte past the load's"
        );
        for (lane, (_, _, pages)) in lanes.iter().enumerate() {
            let of = |name: &str| {
                outs.iter()
                    .find(|(at, read, _)| *at == lane as u32 && read == name)
                    .map(|(.., values)| values.clone())
            };
            let Some(want) = &want[lane] else {
                assert!(
                    of(KEYS).is_none() && of(PAGES).is_none(),
                    "lane {lane} reads no scores"
                );
                continue;
            };
            let [keys, page_rows] = &want[step];
            let got = of(KEYS).unwrap_or_else(|| panic!("lane {lane} reads its key row"));
            assert_eq!(got.len(), context as usize, "a key row is the load's context");
            let off = apart(&got, keys);
            eprintln!("context {context}, step {step}, lane {lane}: keys {off:.2e} apart");
            assert!(off < 2e-2, "lane {lane}'s key row at step {step}: {off} apart");
            // Every layer adds a mass of one.
            let layers = trace
                .nodes
                .iter()
                .filter(|node| {
                    matches!(
                        node.op,
                        poem_ir::Operation::Attn(poem_ir::Attn::ScoreCapture { .. })
                    )
                })
                .count() as f32;
            let mass: f32 = got.iter().sum();
            assert!(
                (mass - layers).abs() < 1e-2 * layers,
                "lane {lane}'s row holds {mass}, and {layers} layers add one each"
            );
            if *pages {
                let got = of(PAGES).unwrap_or_else(|| panic!("lane {lane} reads its page row"));
                assert_eq!(got.len(), context.div_ceil(PAGE) as usize);
                let off = apart(&got, page_rows);
                assert!(off < 2e-2, "lane {lane}'s page row at step {step}: {off} apart");
            } else {
                assert!(of(PAGES).is_none(), "lane {lane} reads no pages");
            }
        }
        for lane in 0..n {
            held[lane] += tokens[lane].len() as u32;
        }
    }
}

/// Two capturing lanes among four, one reading pages, at two contexts.
#[test]
fn a_capturing_lane_serves_the_oracles_score_rows() {
    use Reads::{Keys, Nothing, Pages};
    for context in [256, 512] {
        served(context, &[Nothing, Pages, Nothing, Keys]);
    }
}

/// One capturing lane among three; and a fire none of whose lanes captures,
/// which reads no score row and takes no byte for one.
#[test]
fn a_fire_lays_rows_for_its_capturing_lanes_alone() {
    use Reads::{Nothing, Pages};
    served(256, &[Nothing, Pages, Nothing]);
    served(256, &[Nothing, Nothing, Nothing]);
}

/// The span lays no byte for a capturing lane's rows: a load that captures
/// lays every rung's carve as the compiler's arena, which places them
/// nowhere.
#[test]
fn the_arena_holds_no_score_row() {
    let fixture = fixture(MODEL, None);
    let compiled = poem_compiler::compile_axes(
        &fixture.trace,
        &poem_compiler::Budgets::of(poem_compiler::Budget::new(4, 256)),
        &poem_compiler::DeviceProfile::default(),
    )
    .expect("the axes compile");
    let mut captures = 0;
    for (at, decl) in fixture.trace.values.iter().enumerate() {
        let poem_ir::Ty::Tensor { shape, .. } = &decl.ty else {
            continue;
        };
        if shape.first() != Some(&poem_ir::Dim::Capturing) {
            continue;
        }
        captures += 1;
        let placed = &compiled.arena.placements[compiled.arena.root(poem_ir::ValueId(at as u32)).0 as usize];
        assert!(
            matches!(placed, poem_compiler::Placement::Capture { .. }),
            "value {at} is a capturing lane's row, placed {placed:?}"
        );
        assert_eq!(placed.bytes(), 0, "value {at} takes arena bytes");
    }
    assert!(captures > 0, "{MODEL} captures");
}
