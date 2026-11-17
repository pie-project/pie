//! The score capture's contract (Mac's `the_score_rows_are_the_oracles`,
//! engine-metal), and a capturing lane's score rows served end to end, against the CPU oracle
//! (`engine_cpu::Forward::at_context`): Qwen3's miniature loaded by the
//! shell, four lanes prefilled together and then decoded together, the
//! second and the fourth capturing scores (the first and the third not, so
//! the fire compacts its capturing lanes), the second reading pages too.
//! Each capturing lane's key row (`attn.scores`, a mass a key of the load's
//! context summed over the layers) and page row (`attn.scores.pages`) is the
//! oracle's for the same tokens at two contexts. The rows lie at the reading
//! instance's place in the pool ([`rows_for`]), not the arena: the span lays
//! nothing for them, a fire that captures takes no device byte past what the
//! load reserved, and a lane reading no scores reads none. A fire records as any other: the capture
//! table is staged at its arm's fixed seat, and the capturing lanes' pattern
//! is part of the fire's slot table, so the recording a pattern keys binds
//! its own rows.

use engine_cpu::forward::{Output, Values};
use engine_cpu::Forward;
use poem::Request;
use poem_ir::Trace;

use engine_gpu::guest::session::{ScoreRows, lay_instance, piece_places, score_rows};

use crate::backend::{Lane, Seated};
use crate::model_logits::{PAGE, fixture, load, load_lanes, seeded_bf16};

const MODEL: &str = "qwen3-micro-text";
const KEYS: &str = "attn.scores";
const PAGES: &str = "attn.scores.pages";

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

/// Where a reader instance of `reads` holds a lane's score rows, as its
/// state lays them (`guest::session::lay_instance`): in chunk `k` below the
/// activation span, a chunk a reader. None where `reads` captures nothing.
fn rows_for(shell: &crate::backend::Shell, reads: &[String], k: u32) -> Option<ScoreRows> {
    let spec = shell.score_spec(reads.iter().map(|name| (name.as_str(), 0)));
    if spec.lanes.is_empty() {
        return None;
    }
    let pools = shell.pools();
    let lay = lay_instance(&[], &spec, pools.guest_pieces()).expect("a score row fits a chunk");
    let at = pools.layout().total_chunks - pools.span_chunks() - 1 - k;
    let places = piece_places(
        pools,
        &engine::ScratchPlace::Chunks {
            run: lay.run,
            pieces: vec![at],
        },
        &lay,
    )
    .expect("the reader's chunk is committed");
    score_rows(&lay, &places).first().map(|&(_, rows)| rows)
}

/// The tokens each lane decodes after its prompt, the same for both.
const DECODE: [u32; 2] = [17, 101];

/// The oracle's key and page rows for `prompt` and then each of `DECODE`.
fn oracle(trace: &Trace, context: u32, prompt: &[u32]) -> Vec<[Vec<f32>; 2]> {
    let mut me = Forward::new(trace.clone(), None, &seeded_bf16)
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

    // The device's reservations by owner, but for the per-dispatch scalars
    // wgpu stages in a uniform arena every fire draws on (grown a chunk at a
    // time by the dispatches it binds, whatever they are): what a fire must
    // leave as it found it.
    let books = |shell: &crate::backend::Shell| {
        let ledger = engine_gpu::backend::Context::books(shell.device()).ledger();
        let uniforms: u64 = ledger
            .entries()
            .filter(|&(_, owner, _)| owner == "uniforms")
            .map(|(.., bytes)| bytes)
            .sum();
        let held: Vec<_> = ledger
            .entries()
            .filter(|&(_, owner, _)| owner != "uniforms")
            .map(|(class, owner, bytes)| (format!("{class:?}"), owner, bytes))
            .collect();
        (held, uniforms)
    };
    let places: Vec<Option<ScoreRows>> = (0..n)
        .map(|lane| rows_for(&shell, &reads[lane], lane as u32))
        .collect();
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
                scores: places[lane],
                ..Seated::of(Lane {
                    slot: lane as u32,
                    word: word(trace, tokens[lane].len() as u32, lanes[lane].1),
                    tokens: &tokens[lane],
                })
            })
            .collect();
        let (before, uniforms) = books(&shell);
        let outs = shell.fire_outputs(&seated).expect("the fire lands");
        let (after, grown) = books(&shell);
        eprintln!("context {context}, step {step}: the uniform arena {uniforms} -> {grown} bytes");
        assert_eq!(
            after, before,
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

/// The score capture's contract. A fire lays one score row a capturing lane
/// and none for a lane that captures nothing. Its rows are the CPU oracle's
/// (`engine_cpu::Forward` on this backend's own trace and the fixture's
/// planes): the attention mass a key of the load's context, summed over the
/// layers and averaged over the query heads, and that summed a kv page of
/// the load's own (its chunk recipe's, not the boot's). A fire of 256 lanes,
/// one capturing, lays that lane's rows alone.
#[test]
fn the_score_rows_are_the_oracles() {
    const LANES: u32 = 256;
    const CONTEXT: u32 = 64;
    let fixture = fixture(MODEL, None);
    let trace = &fixture.trace;
    let Some(mut shell) = load_lanes(&fixture, PAGE, CONTEXT, LANES, 512) else {
        return;
    };
    let prompt: Vec<u32> = (0..6u32).map(|at| (11 + at * 37) % 256).collect();
    let one: Vec<Vec<u32>> = (1..LANES).map(|lane| vec![(lane * 7) % 256]).collect();
    let reads = vec![KEYS.to_string(), PAGES.to_string()];
    let place = rows_for(&shell, &reads, 0);
    assert!(place.is_some(), "{MODEL} captures for a reader of its scores");
    let fire = |shell: &mut crate::backend::Shell, captures: bool, place: Option<ScoreRows>| {
        let seated: Vec<Seated<'_>> = (0..LANES)
            .map(|slot| {
                let tokens: &[u32] = if slot == 0 { &prompt } else { &one[slot as usize - 1] };
                let capturing = captures && slot == 0;
                Seated {
                    held: Some(0),
                    reads: if capturing { &reads } else { &[] },
                    scores: if capturing { place } else { None },
                    ..Seated::of(Lane {
                        slot,
                        word: word(trace, tokens.len() as u32, capturing),
                        tokens,
                    })
                }
            })
            .collect();
        shell.fire_seated(&seated).expect("the fire lands");
    };

    // A fire of no capturing lane lays no row.
    fire(&mut shell, false, place);
    assert!(
        shell.score_rows().expect("the rows read back").is_empty(),
        "a fire of no capturing lane laid a score row"
    );

    // One capturing lane among 256 lays one lane's rows.
    fire(&mut shell, true, place);
    let rows = shell.score_rows().expect("the rows read back");
    assert_eq!(rows.len(), 1, "one capturing lane of {LANES} laid {} rows", rows.len());
    let (source, keys, pages) = &rows[0];
    assert_eq!(*source, 0, "the row is the capturing lane's");
    assert_eq!(keys.len(), CONTEXT as usize, "a mass a key of the context");
    // The load's kv page, as its chunk recipe lays it.
    let page = shell.paging().page_size;
    assert_eq!(pages.len(), CONTEXT.div_ceil(page) as usize, "a mass a page of it");

    let mut oracle = Forward::new(trace.clone(), None, &seeded_bf16)
        .expect("the oracle loads")
        .at_context(CONTEXT, page);
    let outs = oracle
        .step(&prompt, 0, word(trace, prompt.len() as u32, true))
        .expect("the oracle steps");
    let floats = |name: &str| -> Vec<f32> {
        match outs.iter().find(|out| out.name == name) {
            Some(Output {
                values: Values::F32(v),
                ..
            }) => v.clone(),
            _ => panic!("the oracle reads out no `{name}`"),
        }
    };
    let (want_keys, want_pages) = (floats(KEYS), floats(PAGES));
    let most = |got: &[f32], want: &[f32]| -> f32 {
        assert_eq!(got.len(), want.len());
        got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0.0, f32::max)
    };
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
    let mass: f32 = keys.iter().sum();
    let (keys_apart, pages_apart) = (most(keys, &want_keys), most(pages, &want_pages));
    eprintln!(
        "the row against the oracle: keys {keys_apart:e}, pages {pages_apart:e} apart at most, a \
         row of mass {mass} ({layers} layers)"
    );
    // bf16 activations against the oracle's: a few hundredths of one
    // layer's mass a key at most; a row of another head, layer or lane is
    // off by tenths.
    assert!(keys_apart < 2e-2, "the key row is not the oracle's: {keys_apart} apart");
    assert!(pages_apart < 4e-2, "the page row is not the oracle's: {pages_apart} apart");
    assert!((mass - layers).abs() < 1e-3 * layers, "a row of mass {mass}, not {layers}");

    // The reader parked and unparked at another place (its rows dead
    // between fires): the same fire lays the same rows there, and leaves
    // the old place as it was.
    let moved = rows_for(&shell, &reads, 1);
    assert_ne!(moved, place, "another place");
    let read = |shell: &crate::backend::Shell, at: ScoreRows| -> Vec<f32> {
        let mut raw = vec![0u8; CONTEXT as usize * 4];
        engine_gpu::backend::Buffer::read(
            shell.pools().segment_buffer(at.keys.segment).expect("committed"),
            at.keys.at,
            &mut raw,
        )
        .expect("read");
        raw.as_chunks::<4>().0.iter().map(|w| f32::from_le_bytes(*w)).collect()
    };
    let before = read(&shell, place.expect("placed"));
    fire(&mut shell, true, moved);
    let again = shell.score_rows().expect("the rows read back");
    assert_eq!(again.len(), 1);
    assert_eq!(&again[0].1, keys, "the key row at the new place");
    assert_eq!(&again[0].2, pages, "the page row at the new place");
    assert_eq!(read(&shell, place.expect("placed")), before, "the old place untouched");
}

/// A served fire whose lane captures scores with no reader attached to hold
/// its rows is refused by name (the runtime stamps the switch only on a fire
/// carrying its reader); before the load is through, the same fire (a warm
/// or arming fire, which has no reader) lays no row and serves.
#[test]
fn a_served_capturing_lane_without_a_reader_is_refused_by_name() {
    let fixture = fixture(MODEL, None);
    let trace = &fixture.trace;
    let Some(mut shell) = load(&fixture, 256) else {
        return;
    };
    let reads = vec![KEYS.to_string()];
    let prompt: Vec<u32> = (0..6u32).map(|at| (11 + at * 37) % 256).collect();
    let fire = |shell: &mut crate::backend::Shell| {
        let seated = [Seated {
            held: Some(0),
            reads: &reads,
            ..Seated::of(Lane {
                slot: 0,
                word: word(trace, prompt.len() as u32, true),
                tokens: &prompt,
            })
        }];
        shell.fire_seated(&seated)
    };
    fire(&mut shell).expect("a load's own fire with no reader lays no row");
    assert!(shell.score_rows().expect("read back").is_empty());
    shell.serve_from_now();
    let refused = fire(&mut shell).expect_err("a served capturing lane with no reader");
    let why = refused.to_string();
    assert!(
        why.contains("lane 0 captures scores, and no instance reading them is attached"),
        "{why}"
    );
}
