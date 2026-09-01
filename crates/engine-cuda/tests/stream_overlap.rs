//! **P6's TWO GATES, ON THE DEVICE: the tokens do not move, and here is what
//! the overlap is worth.**
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 --release --test stream_overlap -- --nocapture
//! ```
//!
//! # Why the off arm is a load and not a flag
//!
//! `side_streams` is a `DeviceProfile` field, so streams-off is a different
//! ARTIFACT — stream 0 on every region, no event point anywhere, the arena
//! carved against a sequential relation — and not a shell that declines to
//! use one it baked. That is the only arrangement in which "streams cost
//! nothing when they are off" is a statement about anything. Each arm here is
//! its own `Shell::load`, and the arms alternate, because a boot warms the
//! process-global scratch, the JIT and the dense autotuner (build log 11) and
//! whichever arm ran first would otherwise be the cold one.
//!
//! # What holds the graph these blocks measure
//!
//! A BODY, since the tier-2 campaign deleted the exact-shape keyed cache: the
//! router is bodies-or-eager, `Graphs::On` means tiered, and the fork's one
//! observable — `nodes` and `edges` over a captured topology — moved onto
//! `BodyStats` with the path that used to carry it. Each arm below therefore
//! stands `bodies` down at LOAD and turns it on afterwards, which is the one
//! deployment-unlike thing this file does and is argued where it is written:
//! an armed load captures its whole lattice at boot and seals the map, and the
//! three-class masked fire under test is not a shape that enumeration
//! synthesizes, so it would meet the seal and walk. An eager fire has no
//! topology, and a fork that cannot be seen cannot be measured.
//!
//! # What is measured, and what it is not
//!
//! Whole-fire wall clock around `fire_seated`, which synchronizes: the
//! ATTENTION SECTION alone would need timing events inside the graph, and the
//! graph is the thing under test. So the figure below is the fire, and the
//! overlap it contains is three attention arms out of forty-two layers of a
//! fire that is otherwise 13.9 GiB of weight reads. **A small percentage here
//! is the expected result**, and the honest way to report it is beside what it
//! is a fraction of — which is why gemma is measured at two window widths and
//! why the milliseconds are printed beside the percent. The saving grows with
//! the windows; the percentage does not, because the denominator grows faster.
//!
//! The dev lineage's 17% (tart status 2026-08-04) is 17% of an attention
//! section. It is not this number and this number is not a refutation of it.

/// How many mixed fires each timed block runs.
const FIRES: usize = 12;

/// How many times each arm is loaded and measured. The arms alternate.
const ROUNDS: usize = 3;

/// The wide composition's per-window row count. 320 rows in each of the two
/// non-decode windows plus one decode row is 641, inside the 768-row budget
/// gemma is loaded at here.
const WIDE: usize = 320;

/// One timed block: the median and the mean of `FIRES` mixed fires, in
/// milliseconds, plus the tokens each lane said.
#[derive(Debug)]
struct Block {
    median_ms: f64,
    mean_ms: f64,
    tokens: (u32, u32, u32),
    captures: u64,
    /// Fires served FROM a recorded body — `BodyStats::hits`. It was
    /// `Stats::replays` while there were two recording paths; the keyed one is
    /// gone and a replay is a hit.
    hits: u64,
    nodes: usize,
    /// **THE ONLY OBSERVABLE A FORK HAS.** Capture lowers an event record and
    /// the wait behind it into a graph EDGE, not into nodes, so a forked
    /// capture and a sequential one hold the same launches and a different
    /// topology. See `device::graph::Graph::edges`.
    ///
    /// **AND IT RIDES `BodyStats` NOW.** The counter used to live on the keyed
    /// cache's own stats; when the tier-2 campaign deleted that path the pair
    /// (`nodes`, `edges`) moved onto the bodies stats rather than dying with
    /// it, precisely because this file is their only reader and the claim
    /// below is the only place a fork is observable from outside at all.
    edges: usize,
    /// What P6 baked for this arm: `(streams, events, forked regions, side
    /// streams opened)`.
    streams: (u32, u32, usize, usize),
    /// How many token rows each of the two non-decode windows carried.
    rows: u32,
}

#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn the_three_class_fire_says_the_same_thing_with_the_streams_on_and_here_is_the_cost() {
    let _serial = gemma::serialized();
    if gemma::ready_or_skip("the stream overlap gate").is_none() {
        return;
    }

    // Two compositions, one pair of loads: a short three-class fire (the C1b
    // golden's own shape) and one whose prefill and masked windows are an
    // order of magnitude wider.
    let mut off: Vec<Block> = Vec::new();
    let mut on: Vec<Block> = Vec::new();
    let mut wide_off: Vec<Block> = Vec::new();
    let mut wide_on: Vec<Block> = Vec::new();
    for round in 0..ROUNDS {
        // Alternate, so that neither arm is systematically the cold one.
        let (first, second) = if round % 2 == 0 { (0, 2) } else { (2, 0) };
        for side in [first, second] {
            let short = gemma::measure(side, 0);
            let wide = gemma::measure(side, WIDE);
            if side == 0 {
                off.push(short);
                wide_off.push(wide);
            } else {
                on.push(short);
                wide_on.push(wide);
            }
        }
    }

    eprintln!("\n── P6, gemma4-e4b-bf16, L40S, release, three-class mixed fire ──");
    for (name, arm) in [("streams off", &off), ("streams on ", &on)] {
        for (at, block) in arm.iter().enumerate() {
            eprintln!(
                "{name} run {at}: median {:.3} ms  mean {:.3} ms  \
                 (captures {} hits {} nodes {} edges {} forks {:?})",
                block.median_ms,
                block.mean_ms,
                block.captures,
                block.hits,
                block.nodes,
                block.edges,
                block.streams,
            );
        }
    }
    let median = |arm: &[Block]| {
        let mut all: Vec<f64> = arm.iter().map(|block| block.median_ms).collect();
        all.sort_by(f64::total_cmp);
        all[all.len() / 2]
    };
    let (quiet, forked) = (median(&off), median(&on));
    eprintln!(
        "── short windows ({} rows each): off {quiet:.3} ms, on {forked:.3} ms — \
         {:+.2}% ──",
        off[0].rows,
        (forked - quiet) / quiet * 100.0,
    );
    for (name, arm) in [("streams off", &wide_off), ("streams on ", &wide_on)] {
        for (at, block) in arm.iter().enumerate() {
            eprintln!(
                "wide {name} run {at}: median {:.3} ms  mean {:.3} ms  \
                 ({} rows, nodes {} edges {})",
                block.median_ms, block.mean_ms, block.rows, block.nodes, block.edges,
            );
        }
    }
    let (wide_quiet, wide_forked) = (median(&wide_off), median(&wide_on));
    eprintln!(
        "── wide windows ({} rows each): off {wide_quiet:.3} ms, on \
         {wide_forked:.3} ms — {:+.2}% ──\n",
        wide_off[0].rows,
        (wide_forked - wide_quiet) / wide_quiet * 100.0,
    );

    // **THE GATE IS THE TOKENS.** Every block of both arms answered the same
    // three tokens, which is the claim P6's safety argument makes and the one
    // a race would break.
    let expected = off[0].tokens;
    for (name, arm) in [("streams off", &off), ("streams on", &on)] {
        for (at, block) in arm.iter().enumerate() {
            assert_eq!(
                block.tokens, expected,
                "{name} run {at} answered {:?} where the first off run answered \
                 {expected:?} — a fork changed a number",
                block.tokens,
            );
        }
    }

    // **AND THE TWO ARMS ARE TWO ARTIFACTS**, asserted rather than printed:
    // a measurement whose arms baked the same graph is a measurement of noise.
    assert_eq!(
        off[0].streams,
        (1, 0, 0, 0),
        "the off arm forked, so it is not an off arm",
    );
    let (streams, events, forked, open) = on[0].streams;
    assert!(
        streams == 3 && events > 0 && forked > 0 && open == 2,
        "the on arm baked {:?} — gemma's three attention arms should reach three \
         streams and this shell should have opened two beside the main one",
        on[0].streams,
    );
    // The launches are the same launches — the fork moved none of them — and
    // the TOPOLOGY is what changed. A chain of `n` nodes has `n - 1` edges;
    // every fork/join pair adds one and removes none.
    assert_eq!(
        on[0].nodes, off[0].nodes,
        "a fork moved a launch, which it may not: it says where the next one \
         lands, never whether it happens",
    );
    assert!(
        on[0].edges > off[0].edges,
        "the forked capture holds {} edges against the sequential one's {} — \
         the side streams never joined the graph, so the on arm IS the off arm",
        on[0].edges,
        off[0].edges,
    );
}

/// The same question on qwen, whose fork groups are its FULL-attention layers
/// only: the linear-attention arms share a process-global staging plane
/// (`engine_cuda::EXCLUSIVE`) and P6 orders them.
///
/// Six forked regions against gemma's hundred and four, so the expected effect
/// is smaller still — which is the point of running it: the number that says
/// "this SKU has almost nothing to overlap" is as much a result as the one
/// that says otherwise.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_qwen_decode_beside_a_prefill_says_the_same_thing_with_the_streams_on() {
    let _serial = gemma::serialized();
    if qwen::ready_or_skip("the qwen stream gate").is_none() {
        return;
    }

    let mut off: Vec<Block> = Vec::new();
    let mut on: Vec<Block> = Vec::new();
    for round in 0..ROUNDS {
        let (first, second) = if round % 2 == 0 { (0, 2) } else { (2, 0) };
        let a = qwen::measure(first);
        let b = qwen::measure(second);
        for (side, block) in [(first, a), (second, b)] {
            if side == 0 { off.push(block) } else { on.push(block) }
        }
    }

    eprintln!("\n── P6, qwen35-d0.8b-bf16, L40S, release, decode + prefill ──");
    for (name, arm) in [("streams off", &off), ("streams on ", &on)] {
        for (at, block) in arm.iter().enumerate() {
            eprintln!(
                "{name} run {at}: median {:.3} ms  mean {:.3} ms  \
                 (captures {} hits {} nodes {} edges {} forks {:?})",
                block.median_ms,
                block.mean_ms,
                block.captures,
                block.hits,
                block.nodes,
                block.edges,
                block.streams,
            );
        }
    }
    let median = |arm: &[Block]| {
        let mut all: Vec<f64> = arm.iter().map(|block| block.median_ms).collect();
        all.sort_by(f64::total_cmp);
        all[all.len() / 2]
    };
    let (quiet, forked) = (median(&off), median(&on));
    eprintln!(
        "── median of medians: off {quiet:.3} ms, on {forked:.3} ms — {:+.2}% ──\n",
        (forked - quiet) / quiet * 100.0,
    );

    let expected = off[0].tokens;
    for (name, arm) in [("streams off", &off), ("streams on", &on)] {
        for (at, block) in arm.iter().enumerate() {
            assert_eq!(
                block.tokens, expected,
                "{name} run {at} answered {:?} where the first off run answered \
                 {expected:?}",
                block.tokens,
            );
        }
    }
    assert_eq!(off[0].streams, (1, 0, 0, 0));
    let (streams, events, forked, open) = on[0].streams;
    assert!(
        streams > 1 && events > 0 && forked > 0 && open + 1 == streams as usize,
        "the on arm baked {:?} — qwen's full-attention arms are what the slab \
         rule leaves it, and the shell opens one stream per compiled one",
        on[0].streams,
    );
    assert!(
        on[0].edges > off[0].edges,
        "the forked capture holds {} edges against {} — the side stream never \
         joined the graph",
        on[0].edges,
        off[0].edges,
    );
}

/// The load, the fire, and the clock — for gemma's three-class composition.
mod gemma {
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, MutexGuard, PoisonError};
    use std::time::Instant;

    use engine::fire::{Mask, Masking};
    use engine_cuda::{Boot, Seated, Shell};
    use model_compiler::{Budget, DeviceProfile};
    use model_dsl::{Classify, Platform, Request};

    use super::{Block, FIRES};

    const SKU: &str = "gemma4-e4b-bf16-kv-bf16";

    /// One shell at a time per process — `kernels-cuda`'s scratch slabs are
    /// process-global and keyed by name.
    static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

    pub fn serialized() -> MutexGuard<'static, ()> {
        ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub fn word(query_len: u32, masked: bool) -> u64 {
        model::gemma_4::forward::Facts::of(&Request::new(query_len, masked)).word()
    }

    pub fn argmax(logits: &[f32]) -> u32 {
        let mut best = (0usize, f32::NEG_INFINITY);
        for (at, &value) in logits.iter().enumerate() {
            assert!(value.is_finite(), "logit {at} is {value}");
            if value > best.1 {
                best = (at, value);
            }
        }
        best.0 as u32
    }

    pub fn turn(ask: &str) -> String {
        format!("<start_of_turn>user\n{ask}<end_of_turn>\n<start_of_turn>model\n")
    }

    pub fn snapshot() -> Option<PathBuf> {
        if let Ok(stated) = std::env::var("PIE_GEMMA_SNAPSHOT") {
            let path = PathBuf::from(stated);
            return path.is_dir().then_some(path);
        }
        let home = std::env::var("HOME").ok()?;
        let direct = Path::new(&home).join(".pie/models/google--gemma-4-E2B-it");
        if direct.join("tokenizer.json").exists() {
            return Some(direct);
        }
        let snapshots = Path::new(&home)
            .join(".cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots");
        std::fs::read_dir(snapshots)
            .ok()?
            .filter_map(|entry| Some(entry.ok()?.path()))
            .find(|path| path.join("tokenizer.json").exists())
    }

    pub fn container(snapshot: &Path) -> Option<PathBuf> {
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

    /// Is there a device and a checkpoint? Says what is missing and skips.
    pub fn ready_or_skip(what: &str) -> Option<()> {
        if !engine_cuda::device::present() {
            eprintln!("skipping {what}: no CUDA device on this machine");
            return None;
        }
        let Some(checkpoint) = snapshot() else {
            eprintln!("skipping {what}: no gemma snapshot (set PIE_GEMMA_SNAPSHOT)");
            return None;
        };
        container(&checkpoint).map(|_| ())
    }

    /// A load at exactly `side_streams`, everything else equal.
    ///
    /// **THE PROFILE IS STATED RATHER THAN PROBED**, and both arms state the
    /// same one but for the field under test: `sms` feeds P4's fallback
    /// threshold and P6's `sm_hint`, so letting the shell probe it in one arm
    /// and not the other would be two different bakes for two reasons.
    fn load(side_streams: u32) -> (Shell, tokenizer::Tokenizer) {
        let checkpoint = snapshot().expect("a gemma snapshot");
        let container = container(&checkpoint).expect("a tensor container");
        let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
            .expect("the checkpoint's tokenizer loads");
        let trace = model::trace_of(SKU).expect("the catalog ships gemma")(Platform::Cuda);
        let source = ztensor_compat::index(&container).expect("the checkpoint opens");
        let contract = model::import_of(SKU).expect("the catalog ships an import")(&source)
            .expect("the import contract fits its own checkpoint");
        drop(source);
        let shell = Shell::load(Boot {
        // Full residency: the whole weight table on the device, which is what
        // an uncapped `Residency` plans (alto design §7).
        residency: engine_cuda::experts::Plan::default(),
            trace,
            contract: &contract,
            checkpoint: &checkpoint,
            budget: Budget::new(4, 768),
            patches: None,
            profile: Some(DeviceProfile {
                sms: 142,
                side_streams,
                ..DeviceProfile::default()
            }),
            page_size: 16,
            context: 1024,
            slots: 4,
            ordinal: 0,
            // **RECORDED, BECAUSE A FORK ONLY EXISTS IN A GRAPH.** The eager
            // walk is the serialization of the same DAG by construction
            // (`model_exec::fire::EagerSink`), so an eager arm would measure
            // nothing and prove nothing.
            graphs: engine_cuda::Graphs::On,
            // **`bodies` STOOD DOWN AT LOAD AND TURNED ON BELOW.** A load
            // that states the word arms its lattice and seals the map inside
            // `Shell::load`, so every capture happens at boot and a fire of a
            // composition the enumeration did not synthesize — this file's
            // three-class masked fire is one — meets a sealed map and walks
            // eagerly. An eager fire has no topology to count edges on, which
            // is this whole file's observable. Stood down, the map stays open
            // and the warm block below captures the composition actually under
            // test.
            knobs: engine_cuda::Knobs {
                bodies: false,
                ..engine_cuda::Knobs::default()
            },
            program_cache_dir: None,
            // F1's depth, kept: these gates fire one step at a time and
            // read its numbers, so a deeper ring would carve slots nothing
            // claims. `Runahead::of` is the door a deployment comes through.
            runahead: engine::runahead::Runahead::F1,
            // The warm-boot weight artifact cache is off for a gate: a test
            // that shared one would be asserting about the last run.
            weight_cache_dir: None,
        })
        .expect("the shell loads");
        (shell, tokenizer)
    }

    /// One arm, measured: load, warm, then `FIRES` timed three-class fires.
    /// A prompt of at least `want` tokens, built out of one sentence repeated.
    /// What is asked of it is a WINDOW WIDTH, not a continuation anyone reads.
    fn padded(tok: &tokenizer::Tokenizer, ask: &str, want: usize) -> Vec<u32> {
        let mut text = format!("<start_of_turn>user\n{ask} ");
        while tok.encode(&text).len() < want {
            text.push_str("The quick brown fox jumps over the lazy dog. ");
        }
        text.push_str("<end_of_turn>\n<start_of_turn>model\n");
        tok.encode(&text)
    }

    /// One arm at one composition: `rows` is roughly how wide each of the two
    /// non-decode windows is.
    ///
    /// **THE SIZE IS THE POINT OF THE SECOND CALL.** An overlap can only be
    /// worth what the smaller of the two arms costs, and at a twenty-token
    /// prefill window beside a twenty-token masked one that is almost nothing.
    /// Running the same fire at a window an order of magnitude wider is how
    /// the measurement says whether the effect is a size effect or a ceiling.
    pub fn measure(side_streams: u32, rows: usize) -> Block {
        let (mut shell, tok) = load(side_streams);
        // The word, said here rather than at load, for the reason `load`
        // states: an armed load has nothing left for these fires to capture.
        shell.set_bodies(true);
        let carried =
            tok.encode(&turn("What is the capital of France? Answer in one word."));
        let fresh = padded(&tok, "Name the largest planet. One word.", rows);
        let masked = padded(&tok, "What colour is the sky on a clear day? One word.", rows);
        let keep =
            Masking::Extent(Mask::new(vec![0, masked.len() as u32], masked.len() as u64));

        shell.open(0).expect("slot 0 opens");
        let seated = shell
            .fire_seated(&[Seated::of(engine_cuda::Lane {
                slot: 0,
                word: word(carried.len() as u32, false),
                tokens: &carried,
            })])
            .expect("the carried lane prefills");
        let fed = [argmax(&seated[0])];

        let one = |shell: &mut Shell, timed: &mut Vec<f64>| {
            shell.open(1).expect("slot 1 re-opens");
            shell.open(2).expect("slot 2 re-opens");
            let began = Instant::now();
            let fire = shell
                .fire_seated(&[
                    Seated::of(engine_cuda::Lane {
                        slot: 1,
                        word: word(fresh.len() as u32, false),
                        tokens: &fresh,
                    }),
                    Seated::masked(
                        engine_cuda::Lane {
                            slot: 2,
                            word: word(masked.len() as u32, true),
                            tokens: &masked,
                        },
                        &keep,
                    ),
                    Seated::of(engine_cuda::Lane {
                        slot: 0,
                        word: word(1, false),
                        tokens: &fed,
                    }),
                ])
                .expect("the three-class fire");
            timed.push(began.elapsed().as_secs_f64() * 1000.0);
            (
                argmax(&fire[0]),
                argmax(&fire[1]),
                argmax(&fire[2]),
            )
        };

        // Warm: the tuner tunes on the second sighting and the body captures
        // on the fire after that (`record::WARM_FIRES`), so three fires are
        // thrown away before the clock means anything.
        let mut discard = Vec::new();
        let mut tokens = (0, 0, 0);
        for _ in 0..3 {
            tokens = one(&mut shell, &mut discard);
        }
        let before = shell.body_stats().hits;

        let mut timed = Vec::with_capacity(FIRES);
        for _ in 0..FIRES {
            let said = one(&mut shell, &mut timed);
            assert_eq!(said, tokens, "a fire changed its answer mid-block");
        }
        let stats = shell.body_stats();
        assert!(
            stats.hits > before,
            "the block replayed nothing, so it measured the eager walk and \
             there is no captured topology under the numbers below. A moved \
             `refusals` says the admissibility rule turned this composition \
             away, which would be a finding about the composition rather than \
             about the fork: {stats}",
        );

        let mut sorted = timed.clone();
        sorted.sort_by(f64::total_cmp);
        Block {
            median_ms: sorted[sorted.len() / 2],
            mean_ms: timed.iter().sum::<f64>() / timed.len() as f64,
            tokens,
            captures: stats.captures,
            hits: stats.hits,
            nodes: stats.nodes,
            edges: stats.edges,
            streams: shell.streams(),
            rows: fresh.len() as u32,
        }
    }
}

/// The same, for qwen's decode-beside-prefill fire.
mod qwen {
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use engine_cuda::{Boot, Seated, Shell};
    use model_compiler::{Budget, DeviceProfile};
    use model_dsl::{Classify, Platform, Request};

    use super::{Block, FIRES};

    const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

    fn word(query_len: u32) -> u64 {
        model::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
    }

    fn snapshot() -> Option<PathBuf> {
        if let Ok(stated) = std::env::var("PIE_SMOKE_SNAPSHOT") {
            let path = PathBuf::from(stated);
            return path.is_dir().then_some(path);
        }
        let home = std::env::var("HOME").ok()?;
        let direct = Path::new(&home).join(".pie/models/Qwen--Qwen3.5-0.8B");
        if direct.join("tokenizer.json").exists() {
            return Some(direct);
        }
        let snapshots =
            Path::new(&home).join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots");
        std::fs::read_dir(snapshots)
            .ok()?
            .filter_map(|entry| Some(entry.ok()?.path()))
            .find(|path| path.join("tokenizer.json").exists())
    }

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

    pub fn ready_or_skip(what: &str) -> Option<()> {
        if !engine_cuda::device::present() {
            eprintln!("skipping {what}: no CUDA device on this machine");
            return None;
        }
        let Some(checkpoint) = snapshot() else {
            eprintln!("skipping {what}: no Qwen3.5-0.8B snapshot (set PIE_SMOKE_SNAPSHOT)");
            return None;
        };
        container(&checkpoint).map(|_| ())
    }

    fn load(side_streams: u32) -> (Shell, tokenizer::Tokenizer) {
        let checkpoint = snapshot().expect("a qwen snapshot");
        let container = container(&checkpoint).expect("a tensor container");
        let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
            .expect("the checkpoint's tokenizer loads");
        let trace = model::trace_of(SKU).expect("the catalog ships the SKU")(Platform::Cuda);
        let source = ztensor_compat::index(&container).expect("the checkpoint opens");
        let contract = model::import_of(SKU).expect("the catalog ships an import")(&source)
            .expect("the import contract fits its own checkpoint");
        drop(source);
        let shell = Shell::load(Boot {
        // Full residency: the whole weight table on the device, which is what
        // an uncapped `Residency` plans (alto design §7).
        residency: engine_cuda::experts::Plan::default(),
            trace,
            contract: &contract,
            checkpoint: &checkpoint,
            budget: Budget::new(4, 256),
            patches: None,
            profile: Some(DeviceProfile {
                sms: 142,
                side_streams,
                ..DeviceProfile::default()
            }),
            page_size: 16,
            context: 512,
            slots: 4,
            ordinal: 0,
            graphs: engine_cuda::Graphs::On,
            // **`bodies` STOOD DOWN AT LOAD AND TURNED ON BELOW.** A load
            // that states the word arms its lattice and seals the map inside
            // `Shell::load`, so every capture happens at boot and a fire of a
            // composition the enumeration did not synthesize — this file's
            // three-class masked fire is one — meets a sealed map and walks
            // eagerly. An eager fire has no topology to count edges on, which
            // is this whole file's observable. Stood down, the map stays open
            // and the warm block below captures the composition actually under
            // test.
            knobs: engine_cuda::Knobs {
                bodies: false,
                ..engine_cuda::Knobs::default()
            },
            program_cache_dir: None,
            // F1's depth, kept: these gates fire one step at a time and
            // read its numbers, so a deeper ring would carve slots nothing
            // claims. `Runahead::of` is the door a deployment comes through.
            runahead: engine::runahead::Runahead::F1,
            // The warm-boot weight artifact cache is off for a gate: a test
            // that shared one would be asserting about the last run.
            weight_cache_dir: None,
        })
        .expect("the shell loads");
        (shell, tokenizer)
    }

    fn argmax(logits: &[f32]) -> u32 {
        let mut best = (0usize, f32::NEG_INFINITY);
        for (at, &value) in logits.iter().enumerate() {
            if value > best.1 {
                best = (at, value);
            }
        }
        best.0 as u32
    }

    pub fn measure(side_streams: u32) -> Block {
        let (mut shell, tok) = load(side_streams);
        // The word, said here rather than at load, for the reason `load`
        // states: an armed load has nothing left for these fires to capture.
        shell.set_bodies(true);
        let carried = tok.encode("The capital of France is");
        let fresh = tok.encode("The largest planet in the solar system is");

        shell.open(0).expect("slot 0 opens");
        let seated = shell
            .fire_seated(&[Seated::of(engine_cuda::Lane {
                slot: 0,
                word: word(carried.len() as u32),
                tokens: &carried,
            })])
            .expect("the carried lane prefills");
        let fed = [argmax(&seated[0])];

        let one = |shell: &mut Shell, timed: &mut Vec<f64>| {
            shell.open(1).expect("slot 1 re-opens");
            let began = Instant::now();
            let fire = shell
                .fire_seated(&[
                    Seated::of(engine_cuda::Lane {
                        slot: 1,
                        word: word(fresh.len() as u32),
                        tokens: &fresh,
                    }),
                    Seated::of(engine_cuda::Lane {
                        slot: 0,
                        word: word(1),
                        tokens: &fed,
                    }),
                ])
                .expect("the mixed fire");
            timed.push(began.elapsed().as_secs_f64() * 1000.0);
            (argmax(&fire[0]), argmax(&fire[1]), 0)
        };

        let mut discard = Vec::new();
        let mut tokens = (0, 0, 0);
        for _ in 0..3 {
            tokens = one(&mut shell, &mut discard);
        }
        let before = shell.body_stats().hits;

        let mut timed = Vec::with_capacity(FIRES);
        for _ in 0..FIRES {
            let said = one(&mut shell, &mut timed);
            assert_eq!(said, tokens, "a fire changed its answer mid-block");
        }
        let stats = shell.body_stats();
        assert!(
            stats.hits > before,
            "the block replayed nothing, so it measured the eager walk and \
             there is no captured topology under the numbers below. A moved \
             `refusals` says the admissibility rule turned this composition \
             away, which would be a finding about the composition rather than \
             about the fork: {stats}",
        );

        let mut sorted = timed.clone();
        sorted.sort_by(f64::total_cmp);
        Block {
            median_ms: sorted[sorted.len() / 2],
            mean_ms: timed.iter().sum::<f64>() / timed.len() as f64,
            tokens,
            captures: stats.captures,
            hits: stats.hits,
            nodes: stats.nodes,
            edges: stats.edges,
            streams: shell.streams(),
            rows: fresh.len() as u32,
        }
    }
}
