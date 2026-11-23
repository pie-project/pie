//! The score capture's contract (Mac's `the_score_rows_are_the_oracles`,
//! engine-metal), and a capturing lane's score rows served end to end, against the CPU oracle
//! (`engine_cpu::Forward::at_context`): Qwen3's miniature loaded by the
//! shell, four lanes prefilled together and then decoded together, the
//! second and the fourth capturing scores (the first and the third not, so
//! the fire compacts its capturing lanes), the second reading pages too.
//! Each capturing lane's key row (`attn.scores`, a mass a key of the load's
//! context summed over the layers) and page row (`attn.scores.pages`) is the
//! oracle's for the same tokens at two contexts. The rows lie in a reading
//! pair's score room in the pool (`guest::state`), not the arena: the span
//! lays nothing for them, a fire that captures takes no device byte past
//! what the load reserved, and a lane reading no scores reads none. A fire
//! records as any other: the capture table is staged at its arm's fixed
//! seat, and the capturing lanes' pattern is part of the fire's slot table,
//! so the recording a pattern keys binds its own rows.

use engine_cpu::forward::{Output, Values};
use engine_cpu::Forward;
use poem::Request;
use poem_ir::Trace;

use crate::backend::{Lane, Seated, Shell};
use engine_gpu::serve::DeviceCell;
use crate::model_logits::{PAGE, fixture, load, load_lanes, seeded_bf16};

const MODEL: &str = "qwen3-micro-text";
const KEYS: &str = "attn.scores";
const PAGES: &str = "attn.scores.pages";

fn word(trace: &Trace, rows: u32, captures: bool) -> u64 {
    trace
        .facts
        .word(&Request::new(rows, false).capturing_scores(captures))
}

/// The score room a pair reading `attn.scores` keeps (`Shell::score_room`),
/// laid as the runtime would: a run of whole chunks right under the span.
fn room(shell: &Shell) -> DeviceCell<crate::backend::Api> {
    let bytes = shell.score_room([(KEYS, 0u32)]);
    assert!(bytes > 0, "a read of `{KEYS}` keeps a score room");
    let pools = shell.pools();
    let layout = pools.layout();
    let run = u32::try_from(bytes.div_ceil(layout.chunk_bytes)).expect("a run of chunks");
    let first = layout.total_chunks - pools.span_chunks() - run;
    let (buffer, at) = pools.chunk_place(first).expect("the room's chunks are committed");
    assert!(
        at / layout.chunk_bytes + u64::from(run) <= u64::from(layout.segment_chunks),
        "the room lies in one segment"
    );
    DeviceCell {
        buffer: buffer.clone(),
        at,
        bytes,
    }
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
    served_with(context, pattern, None);
}

/// [`served`]; where `moved` is given, the load arms its bodies and serves
/// from them, and before that step its room moves one room's run down the
/// pool (its bytes with it, as an unpark or an evacuation moves them).
fn served_with(context: u32, pattern: &[Reads], moved: Option<usize>) {
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
    let first = room(&shell);
    let mut room = first.clone();
    if moved.is_some() {
        shell.record_bodies(None).expect("the load arms its bodies");
        shell.recorded_serving();
        shell.unkept_serving();
        shell.eager_serving();
        shell.serving = true;
    }
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
    let mut held = vec![0u32; n];
    for step in 0..=DECODE.len() {
        if moved == Some(step) {
            // One room's run down: whole chunks in the same segment.
            let chunk = shell.pools().layout().chunk_bytes;
            let run = first.bytes.div_ceil(chunk) * chunk;
            assert!(first.at >= run, "a room's run lies below the first in its segment");
            let to = DeviceCell {
                at: first.at - run,
                ..first.clone()
            };
            let mut frame = engine_gpu::backend::Context::frame(shell.device()).expect("a frame");
            engine_gpu::backend::Frame::copy(
                &mut frame,
                &first.buffer,
                first.at,
                &to.buffer,
                to.at,
                first.bytes,
            )
            .expect("the room's bytes move");
            engine_gpu::backend::Frame::commit(frame).expect("the move lands");
            room = to;
        }
        let tokens: Vec<Vec<u32>> = lanes
            .iter()
            .map(|(prompt, ..)| if step == 0 { prompt.clone() } else { vec![DECODE[step - 1]] })
            .collect();
        let seated: Vec<Seated<'_>> = (0..n)
            .map(|lane| Seated {
                held: Some(held[lane]),
                reads: &reads[lane],
                scores: lanes[lane].1.then_some(&room),
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
    if moved.is_some() {
        assert_eq!(shell.recorded_serving(), 0, "a fire after ready recorded a body");
        if <crate::backend::Api as engine_gpu::serve::Serve>::RECORDS {
            assert_eq!(
                shell.unkept_serving(),
                0,
                "a fire after ready walked with no armed body: the moved room is the fire's"
            );
        }
    }
}

/// A pair's room moved (an unpark, an evacuation) between two replays of
/// one armed body: the replay after reads the rows where the room now lies,
/// the oracle's, and nothing records or walks unarmed.
#[test]
fn an_armed_body_reads_its_room_where_it_moved() {
    use Reads::{Keys, Nothing, Pages};
    served_with(256, &[Nothing, Pages, Nothing, Keys], Some(DECODE.len()));
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

/// A load whose trace captures names every kernel a capture fires, all at
/// the baseline tier (no subgroup, fp16 or cooperative-matrix form: a device
/// at its baseline runs them), and holds no lane class unservable for want
/// of one: the normaliser, the mass at the form the device's workgroup
/// memory takes, and the page sums.
#[test]
fn a_capturing_load_names_its_capture_kernels_at_the_baseline_tier() {
    let fixture = fixture(MODEL, None);
    let Some(shell) = load(&fixture, 256) else {
        return;
    };
    let named = shell.named();
    let unservable: Vec<_> = named.unservable.iter().map(|held| held.class).collect();
    assert!(unservable.is_empty(), "classes held unservable: {unservable:?}");
    for prefix in ["attn_score_norm_", "attn_score_mass_", "attn_score_pages"] {
        let kernel = named
            .kernels
            .iter()
            .find(|kernel| kernel.key.starts_with(prefix))
            .unwrap_or_else(|| panic!("the load names no `{prefix}` kernel"));
        assert_eq!(kernel.unit, "attn/score.slang");
        assert_eq!(
            kernels_slang::key(kernel.key, kernels_slang::Tier::Baseline),
            Some(kernel.key),
            "`{}` is the baseline tier's",
            kernel.key
        );
    }
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
/// the load's own (its chunk recipe's, not the boot's). A fire as wide as the
/// load carries, one lane capturing, lays that lane's rows alone. The boot
/// asks for 256 lanes and 512 tokens, but the load's fire top is the rung its
/// ruling picks on this device (`engine::load::fire_top`), which may be
/// narrower: the fire takes the lanes and tokens the load states
/// (`Shell::budget`), the capturing lane's prompt and a token each for the
/// rest.
#[test]
fn the_score_rows_are_the_oracles() {
    const CONTEXT: u32 = 512;
    let fixture = fixture(MODEL, None);
    let trace = &fixture.trace;
    let Some(mut shell) = load_lanes(&fixture, PAGE, CONTEXT, 256, 512) else {
        return;
    };
    let room = room(&shell);
    let prompt: Vec<u32> = (0..6u32).map(|at| (11 + at * 37) % 256).collect();
    let budget = shell.budget();
    let tokens = budget.max_tokens.checked_sub(prompt.len() as u32).unwrap_or_else(|| {
        panic!(
            "the load's fire top of {} tokens holds no {}-token prompt",
            budget.max_tokens,
            prompt.len()
        )
    });
    let lanes = budget.max_lanes.min(1 + tokens);
    assert!(
        lanes > 1,
        "the load carries {} lanes and {} tokens: no lane beside the capturing one",
        budget.max_lanes,
        budget.max_tokens
    );
    eprintln!(
        "a fire of {lanes} lanes against the load's {} lanes and {} tokens",
        budget.max_lanes, budget.max_tokens
    );
    let one: Vec<Vec<u32>> = (1..lanes).map(|lane| vec![(lane * 7) % 256]).collect();
    let reads = vec![KEYS.to_string(), PAGES.to_string()];
    let fire = |shell: &mut Shell, captures: bool| {
        let seated: Vec<Seated<'_>> = (0..lanes)
            .map(|slot| {
                let tokens: &[u32] = if slot == 0 { &prompt } else { &one[slot as usize - 1] };
                let capturing = captures && slot == 0;
                Seated {
                    held: Some(0),
                    reads: if capturing { &reads } else { &[] },
                    scores: capturing.then_some(&room),
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
    fire(&mut shell, false);
    assert!(
        shell.score_rows().expect("the rows read back").is_empty(),
        "a fire of no capturing lane laid a score row"
    );

    // One capturing lane among them lays one lane's rows.
    fire(&mut shell, true);
    let rows = shell.score_rows().expect("the rows read back");
    assert_eq!(rows.len(), 1, "one capturing lane of {lanes} laid {} rows", rows.len());
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
}
