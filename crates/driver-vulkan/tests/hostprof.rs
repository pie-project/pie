//! Where a decode step's HOST milliseconds go.
//!
//! `tests/device.rs` measures the device: two timestamps around every dispatch
//! account for a submission end to end, and
//! `the_projections_dominate_both_steps_now_that_the_decode_splits_its_keys`
//! holds the result. What that test also recorded, in a doc-comment table with
//! no code behind it, is that **1.9 ms of a decode step happens outside
//! `run_all` altogether** and does not move with context -- so it is fixed
//! per-step work, and at 24 tokens of history it is a third of the wall step.
//!
//! **Both halves of that sentence are now false, and the arithmetic behind it
//! was wrong before it was stale.** It subtracted `fire/run_all` from the wall
//! and called the remainder host. Since `crate::replay` most steps never enter
//! `run_all` at all -- 94% leave by `fire/replay/submit`, a span that wraps
//! `queue_submit` AND `wait_for_fences`, so it is the card working and not the
//! host preparing. Subtracting only the first route counted the GPU as host:
//! it reported 1.61 ms of a 1.74 ms step, where the honest split is 0.08.
//!
//! | | device (submit + wait) | outside the submit | wall |
//! |---|---|---|---|
//! | 24 tokens | 1.513 | 0.096 | 1.609 |
//! | 384 tokens | 1.657 | 0.082 | 1.739 |
//!
//! So the host is about a twentieth of a step and does not grow with context.
//! The remaining work is on the CARD, which is where the earlier profiles had
//! been sending it all along -- and the mistake is worth keeping because it
//! flattered in the direction work gets aimed. A profile that says the host
//! owns the step sends the next optimisation to the host.
//!
//! **Read the two lines, not one.** `device (submit + wait)` and `outside the
//! submit (host)` sum to the wall by construction, so a claim that either end
//! dominates can be checked against the other.
//!
//! This file is the code behind that table. It fires real decode steps against
//! a real checkpoint with `PIE_VULKAN_HOST_PHASES` on and prints
//! [`driver_vulkan::phase`]'s totals per step, so the number can be reproduced
//! and a change to it can be attributed rather than argued about.
//!
//! # How to run it
//!
//! ```text
//! PIE_VULKAN_HOST_PHASES=1 cargo test -p driver-vulkan --features native \
//!     --release --test hostprof -- --nocapture
//! ```
//!
//! **Release, always.** This is host code and a debug build misattributes it
//! by an order of magnitude -- the SPIR-V walk that dominated the first
//! profile is thirty times its release cost unoptimised, which would have
//! aimed the work at it whether or not it mattered.
//!
//! Nothing is asserted about milliseconds. A shared box varies by 1.7x between
//! runs, which is the same reason `device.rs` asserts shares and not times; the
//! MEDIAN over many steps is printed instead, and a reader compares two runs of
//! this file rather than one run against a constant.
//!
//! # Why it duplicates `device.rs`'s fixture
//!
//! An integration test cannot import another one. The checkpoint load, the
//! text and the shell are the same four steps `device.rs` does; they are
//! copied rather than shared for the reason its own `names_a_decode_binds`
//! gives -- a `mod` between them would drag the checkpoint dependency into
//! every test in the other file.

#![cfg(feature = "native")]

use std::collections::BTreeMap;

/// Every allocation this process makes, counted.
///
/// A phase table says WHERE the host milliseconds go and cannot say WHAT they
/// are. Planning 452 rectangles is a bounded amount of index arithmetic and a
/// handful of buffer ranges, so the first question about 1.4 microseconds of
/// it is how many times it went to the allocator -- and that is a number, not
/// an argument. A relaxed increment per allocation is a nanosecond or two on
/// a path that already costs tens, and it is on in every build of this file
/// so the count is quoted against the same binary the milliseconds are.
struct Counted;

/// How many allocations since the last [`allocations`] reading.
static ALLOCS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

unsafe impl std::alloc::GlobalAlloc for Counted {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe { std::alloc::System.realloc(ptr, layout, new) }
    }
}

#[global_allocator]
static COUNTING: Counted = Counted;

/// The allocation counter's reading.
fn allocations() -> u64 {
    ALLOCS.load(std::sync::atomic::Ordering::Relaxed)
}

/// What one allocate-and-free of a small `Vec` costs in this build.
///
/// The plan makes ~27 of them per rectangle and 452 rectangles a step, so the
/// product of this number and the allocation count is how much of the host
/// step is the allocator rather than the arithmetic the plan exists to do.
fn alloc_cost_ms() -> f64 {
    const N: usize = 200_000;
    let mut sink = 0usize;
    let at = std::time::Instant::now();
    for i in 0..N {
        let v: Vec<u64> = Vec::with_capacity(8 + (i & 7));
        sink += v.capacity();
        drop(v);
    }
    let ms = at.elapsed().as_secs_f64() * 1000.0 / N as f64;
    assert!(sink > 0);
    ms
}

/// What one [`driver_vulkan::phase::span`] pair costs in this build.
///
/// The tool is 3,200 spans a decode step, so a profile that does not state
/// this cannot tell its own overhead from the phase it is attributing. A
/// LOWER bound: the calibration loop hits a one-row table where a real span
/// scans as many rows as the fire has phases.
fn span_cost_ms() -> f64 {
    for _ in 0..10_000 {
        let _s = driver_vulkan::phase::span("hostprof/calibration");
    }
    const N: usize = 200_000;
    let at = std::time::Instant::now();
    for _ in 0..N {
        let _s = driver_vulkan::phase::span("hostprof/calibration");
    }
    let ms = at.elapsed().as_secs_f64() * 1000.0 / N as f64;
    driver_vulkan::phase::reset();
    ms
}

/// The prompt `device.rs` uses, and the same one for the same reason: six
/// tokens that repeat, so a decode after `n` copies has `6n` of history.
const PERIOD: [u32; 6] = [15_339, 1_723, 88_204, 6_100, 41_777, 2_930];

/// The one model this file profiles.
///
/// qwen3-0.6b at four bits, because that is the model every number in
/// `.wiki/kernel-x/vulkan-refactor.md` and in `device.rs`'s step-shape table
/// is quoted against. A second model would double the run time and answer a
/// question nobody has asked: the host cost is the PLAN's, and both models
/// lower to within 7% of the same rectangle count.
const MODEL_ID: &str = "qwen3-0.6b";

/// `model.embed_tokens.weight`'s packed shape, which is how a snapshot says
/// which model it is.
const EMBED: &[i64] = &[151_936, 128];

fn facts() -> model::shared::llama_like::forward::facts::LlamaLikeFacts {
    model::shared::llama_like::forward::facts::LlamaLikeFacts::qwen3_0_6b()
}

fn snapshots() -> Vec<String> {
    match std::env::var("PIE_CHECKPOINT") {
        Ok(v) => v
            .split(':')
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Every weight name this model's decode plan binds.
fn names_a_decode_binds() -> Vec<String> {
    use model::shared::llama_like::forward::facts::LlamaLikeMetalFacts;
    use model::shared::llama_like::forward::llama_like_metal;
    use model_compiler::lower::{Arg, Fire, Row, lower};
    use model_ir::trace::FireClass;

    let text = llama_like_metal(
        &facts(),
        &LlamaLikeMetalFacts {
            add_bias: true,
            ..LlamaLikeMetalFacts::synthetic()
        },
        FireClass::Decode,
    );
    let low = lower(
        &text,
        &[Row::default()],
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the plan lowers");
    let names: std::collections::BTreeSet<String> = low
        .args
        .iter()
        .filter_map(|a| match a {
            Arg::Weight(n) if !n.starts_with("scale.") => Some(n.clone()),
            _ => None,
        })
        .collect();
    names.into_iter().collect()
}

/// The checkpoint's bytes under the names a plan binds, or `None` to skip.
fn load_weights() -> Option<BTreeMap<String, Vec<u8>>> {
    for dir in snapshots() {
        let path = std::path::Path::new(&dir);
        let Ok(meta) = model_loader::checkpoint::read::parse_checkpoint_metadata(path) else {
            continue;
        };
        let hidden = meta
            .tensors
            .iter()
            .find(|t| t.name == "model.embed_tokens.weight")
            .map(|t| t.shape.clone())
            .unwrap_or_default();
        if hidden != EMBED {
            continue;
        }
        let row = model::catalog::find(MODEL_ID)?;
        let config = std::fs::read_to_string(path.join("config.json")).ok()?;
        let encoding = model::encoding::Encoding::from_config_json(&config).ok()?;
        let target = model_loader::plan::StorageTarget::for_backend(
            model_loader::types::BackendKind::Vulkan,
            0,
            1,
        );
        let (plan, _) = model::boot::compile_load_plan_for(
            path,
            &meta,
            &target,
            row,
            &encoding,
            model::boot::Binding::MLX_IN_PLACE,
        )
        .ok()?;
        let storage = match model_loader::executor::Execution::new(&plan, path).run() {
            Ok(storage) => storage,
            Err(e) => {
                eprintln!("the loader would not execute the plan: {e}");
                return None;
            }
        };
        let naming = driver_vulkan::names::Naming::mlx();
        let mut out = BTreeMap::new();
        for traced in names_a_decode_binds() {
            let bytes = naming
                .spellings(&traced)
                .iter()
                .find_map(|s| storage.tensors.get(s.as_str()))
                .unwrap_or_else(|| panic!("`{traced}` resolves to nothing the loader produced"));
            out.insert(traced, bytes.clone());
        }
        return Some(out);
    }
    eprintln!("no snapshot PIE_CHECKPOINT names is the 4-bit `{MODEL_ID}` this file states");
    None
}

/// The checkpoint, read once for the process.
fn weights() -> Option<&'static BTreeMap<String, Vec<u8>>> {
    static HELD: std::sync::OnceLock<Option<&'static BTreeMap<String, Vec<u8>>>> =
        std::sync::OnceLock::new();
    *HELD.get_or_init(|| load_weights().map(|m| &*Box::leak(Box::new(m))))
}

/// A shell serving the model, with its weights held.
fn shelled(real: &'static BTreeMap<String, Vec<u8>>, pages: u32) -> driver_vulkan::shell::Shell {
    use driver_vulkan::shell::{Deployment, Shell, Text};
    use model::shared::llama_like::forward::facts::LlamaLikeMetalFacts;
    use model::shared::llama_like::forward::llama_like_metal;
    use model_ir::trace::FireClass;

    let facts = facts();
    let metal = LlamaLikeMetalFacts {
        add_bias: true,
        ..LlamaLikeMetalFacts::synthetic()
    };
    let text = Text {
        decode: llama_like_metal(&facts, &metal, FireClass::Decode),
        prefill: llama_like_metal(&facts, &metal, FireClass::Prefill),
        geometry: driver_vulkan::dispatch::Geometry {
            q_heads: facts.q_heads,
            kv_heads: facts.kv_heads,
            head_dim: facts.head_dim,
            rotary_dims: facts.head_dim,
            n_experts: 0,
            experts_per_token: 0,
            ..Default::default()
        },
        layers: facts.layers as u16,
    };

    let mut shell = Shell::open(
        text,
        Deployment {
            pages,
            ..Deployment::default()
        },
    )
    .unwrap_or_else(|e| panic!("the shell: {e}"));
    for (name, bytes) in real {
        shell.hold(name, bytes).expect("a weight");
    }
    shell
}

/// The host phases of a decode step, at two context lengths.
///
/// Prints; asserts only that the tool was on and that the numbers are
/// self-consistent. See the module doc for why there is no millisecond
/// ceiling here.
/// # RUN THIS UNDER `--release`, or every number it prints is about rustc
///
/// The host phases are ordinary Rust and the device phases are not, so an
/// unoptimised build inflates one side of the very ratio this file exists to
/// report -- and inflates it enough to invert the conclusion.
///
/// Measured both ways on the same machine and checkpoint, at batch 8:
/// `step/logits` is 31.2 ms/step in a `cargo test` build and 0.25 in a
/// `cargo test --release` one, a factor of 125. Nearly all of it is the
/// bf16-to-f32 widening in `serve::logits_of`, which is a shift over 151,936
/// values a row -- vectorised in release, a bounds-checked scalar loop
/// without it. The debug build therefore says 60% of a decode step is host
/// and the release build says 2%, and only the second is about this
/// repository.
///
/// This is not hypothetical: the sibling table in `tests/device.rs` prints a
/// "% of the wall step off-device" line that is subject to exactly the same
/// distortion, and it was read as a finding before it was read as a build
/// profile.
#[test]
fn where_a_decode_steps_host_milliseconds_go() {
    // Whether this build carries the modules. A `native` build of
    // `kernels-vulkan` embeds them in its rlib; a portable one embeds an
    // empty table. This file is `#![cfg(feature = "native")]`, so the table
    // is populated in every build that compiles it -- the check is kept as
    // the same skip the directory lookup used to be, because the feature
    // edge is a Cargo fact and this is a runtime one.
    if !kernels_vulkan::embedded() {
        eprintln!("skipped: built without kernels-vulkan/native, so there are no modules");
        return;
    }
    let Some(real) = weights() else {
        return;
    };
    // Not required. With the tool off this still prints the wall, which is
    // the control every phase number is quoted against: the spans are
    // 3600-odd `Instant` pairs a step and a profile that cannot say what its
    // own tool costs is a profile that has not been checked.
    let timed = std::env::var_os("PIE_VULKAN_HOST_PHASES").is_some();
    assert!(
        std::env::var_os("PIE_VULKAN_TIMING").is_none(),
        "PIE_VULKAN_TIMING perturbs the submit by 2 ms a fire; the host phases \
         must be measured with the device tool OFF"
    );

    // Fifty steps at each length. The host cost per step is 2 ms and a
    // `Instant` pair is tens of nanoseconds, so the tool is a rounding error
    // on what it measures; fifty is enough that one descheduled step does not
    // move the median.
    const STEPS: usize = 50;

    let span_cost = if timed { span_cost_ms() } else { 0.0 };
    let alloc_cost = alloc_cost_ms();
    eprintln!("one small allocation costs {:.1} ns", alloc_cost * 1e6);
    eprintln!(
        "one span pair costs {:.1} ns (lower bound)",
        span_cost * 1e6
    );

    // `batch` is how many CONVERSATIONS decode together in one step, which is
    // what a server's scheduler gathers and what every number this file used
    // to print left out. At one the step is a single row and the host has one
    // lowering to build; at eight it has eight rows to place, and whether the
    // host cost per step is flat or proportional to the batch is the question
    // the extra arm exists to answer. It matters because the DEVICE cost is
    // known to be proportional -- `tests/qmv_bench.rs` shows the matvec
    // re-reading its weights per row -- so a host cost that is also
    // proportional would mean nothing about a batch is amortised anywhere.
    let run = |repeats: usize, pages: u32, batch: u64| {
        let mut shell = shelled(real, pages);
        let mut prompt: Vec<u32> = Vec::new();
        for _ in 0..repeats {
            prompt.extend_from_slice(&PERIOD);
        }
        let step = |shell: &mut driver_vulkan::shell::Shell,
                    turns: Vec<driver_vulkan::turns::Turn>| {
            shell.step(&turns).unwrap_or_else(|e| panic!("{e}"));
        };
        // One prefill per conversation, each in its own step: a step that
        // prefilled all of them at once would be a different shape from the
        // decode being measured and would need its own token budget.
        for who in 1..=batch {
            step(
                &mut shell,
                vec![driver_vulkan::turns::Turn {
                    who,
                    tokens: prompt.clone(),
                }],
            );
        }
        let decode = || -> Vec<driver_vulkan::turns::Turn> {
            (1..=batch)
                .map(|who| driver_vulkan::turns::Turn {
                    who,
                    tokens: vec![PERIOD[0]],
                })
                .collect()
        };
        // Warm: the first decode after a prefill builds every pipeline, and
        // the lowering cache is cold until a decode of this row count has
        // been asked for once.
        for _ in 0..3 {
            step(&mut shell, decode());
        }
        driver_vulkan::phase::reset();
        let before = allocations();
        let mut walls: Vec<f64> = Vec::with_capacity(STEPS);
        for _ in 0..STEPS {
            let at = std::time::Instant::now();
            step(&mut shell, decode());
            walls.push(at.elapsed().as_secs_f64() * 1000.0);
        }
        let allocs = (allocations() - before) as f64 / STEPS as f64;
        let rows = driver_vulkan::phase::rows();
        walls.sort_by(f64::total_cmp);
        let median = walls[walls.len() / 2];
        let mean = walls.iter().sum::<f64>() / walls.len() as f64;

        eprintln!(
            "\n=== {} tokens of history, batch {batch}, {STEPS} decode steps ===",
            repeats * PERIOD.len()
        );
        eprintln!("  {:<34} {:>9}  {:>7}", "phase", "ms/step", "entries");
        let ms = |name: &str| -> f64 {
            rows.iter()
                .find(|(n, _, _)| *n == name)
                .map_or(0.0, |(_, ms, _)| *ms)
                / STEPS as f64
        };
        for (name, total, n) in &rows {
            eprintln!(
                "  {name:<34} {:>9.3}  {:>7}",
                total / STEPS as f64,
                *n as f64 / STEPS as f64
            );
        }
        let fire = ms("step/fire");
        let run_all = ms("fire/run_all");
        // The two routes a step's work can leave by, and BOTH have to come
        // off the wall before what is left can be called host.
        //
        // This subtracted `run_all` alone, which was right when every step
        // recorded one. Since `crate::replay`, 94% of steps take
        // `fire/replay/submit` instead -- and that span wraps `queue_submit`
        // AND `wait_for_fences`, so it is the GPU executing, not the host
        // preparing. `run_all` then averages 0.133 ms at 0.06 entries while
        // the replay carries 1.501, and subtracting only the first reported
        // 1.61 of a 1.74 ms step as "host" when nine tenths of it was the
        // card.
        //
        // The failure is worth naming because it flatters in the direction
        // work gets aimed: a profile that says the host owns the step sends
        // the next optimisation to the host.
        let submit = ms("fire/replay/submit");
        let entries: f64 = rows.iter().map(|(_, _, n)| *n as f64).sum::<f64>() / STEPS as f64;
        // Only with the tool ON. With it off every phase total is zero, so
        // both lines would read `median` and the second would name the whole
        // step "host" -- which is exactly the misreading this pair replaced,
        // reintroduced by a different route. The wall below is printed either
        // way, because the wall is the control.
        if timed {
            eprintln!(
                "  {:<34} {:>9.3}",
                "device (submit + wait)",
                run_all + submit
            );
            eprintln!(
                "  {:<34} {:>9.3}",
                "outside the submit (host)",
                median - run_all - submit
            );
        }
        eprintln!("  {:<34} {:>9.0}", "allocations/step", allocs);
        eprintln!(
            "  {:<34} {:>9.3}",
            "allocator (ms/step)",
            allocs * alloc_cost
        );
        eprintln!("  {:<34} {:>9.0}", "spans/step", entries);
        eprintln!(
            "  {:<34} {:>9.3}",
            "tool overhead (>=, ms/step)",
            entries * span_cost
        );
        eprintln!("  {:<34} {:>9.3}", "wall (median)", median);
        eprintln!("  {:<34} {:>9.3}", "wall (mean)", mean);
        assert_eq!(
            timed,
            fire > 0.0 && run_all > 0.0,
            "the host tool and its totals disagree about whether it is on"
        );
        (median, run_all + submit)
    };

    let (short_wall, short_dev) = run(4, 64, 1);
    let (long_wall, long_dev) = run(64, 512, 1);
    // The same short history, decoded as a batch. Pages scale with the
    // conversations: each holds its own history.
    for batch in [2u64, 4, 8] {
        run(4, 64 * batch as u32, batch);
    }
    // Guarded for the same reason the table is: with the tool off both
    // totals are zero and this line would report the whole wall as host,
    // which is the misreading the split exists to prevent.
    if timed {
        eprintln!(
            "\nhost outside the submit: short {:.3} ms of {:.3}, long {:.3} of {:.3}",
            short_wall - short_dev,
            short_wall,
            long_wall - long_dev,
            long_wall
        );
    } else {
        eprintln!(
            "\nwall only, tool off: short {short_wall:.3} ms, long {long_wall:.3} ms. \
             Set PIE_VULKAN_HOST_PHASES=1 for the host/device split."
        );
    }
}

/// Four conversations batched into one decode answer what each answers alone.
///
/// This exists because of what the batched-decode lane DID. A batch of eight
/// one-token turns used to be planned as a prefill and reach
/// `sdpa_paged_tiled`, where a 32-row query tile held rows belonging to eight
/// different sequences with eight different key runs. `GuardPred::WindowOne`
/// now routes that fire to the decode pair instead, which is worth ~2ms a
/// step at every batch of two or more -- and which is a change to WHICH
/// KERNEL READS WHOSE KEYS. Nothing else in the suite would notice if the
/// new lane read them wrongly: `cargo test -p model` checks the plan TEXT,
/// and `two_requests_in_one_fire_do_not_read_each_others_history` is a
/// synthetic fixture that never sees these weights.
///
/// The witness has to be built so it can fail. Four IDENTICAL prompts would
/// agree with each other even if the lane pooled every row's keys into one
/// run, so the prompts differ, and the test asserts BOTH halves: each
/// conversation matches itself run alone, and the four disagree among
/// themselves. The second assertion is the one that keeps the first
/// meaningful, and it is the assertion this test's first draft was missing.
///
/// Greedy rather than sampled, and twelve tokens rather than one, because a
/// single argmax can agree by luck on a plateau while a twelve-token chain
/// re-feeds its own answer and diverges permanently after the first
/// disagreement.
#[test]
fn batched_decode_answers_what_a_single_decode_answers() {
    let Some(real) = weights() else {
        return;
    };
    // Each conversation gets a DIFFERENT prompt. A batch of identical
    // prompts would agree even if the lane leaked one row's keys into
    // another's, so the only witness worth having is one where the rows
    // disagree with each other and each still matches itself run alone.
    let prompt = |who: u64| -> Vec<u32> {
        vec![PERIOD[0], 2000 * who as u32, 3000 + who as u32, 700 * who as u32 + 11]
    };
    let greedy = |whos: &[u64]| -> Vec<Vec<u32>> {
        let mut shell = shelled(real, 64 * whos.len() as u32 + 64);
        for &who in whos {
            shell
                .step(&[driver_vulkan::turns::Turn {
                    who,
                    tokens: prompt(who),
                }])
                .unwrap();
        }
        let mut next: Vec<u32> = whos.iter().map(|&w| prompt(w)[3]).collect();
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); whos.len()];
        for _ in 0..12 {
            let turns: Vec<_> = whos
                .iter()
                .enumerate()
                .map(|(i, &who)| driver_vulkan::turns::Turn {
                    who,
                    tokens: vec![next[i]],
                })
                .collect();
            let step = shell.step(&turns).unwrap();
            for (i, &at) in step.readout_of.iter().enumerate() {
                let row = step.logits.row(at).expect("a row per turn");
                let best = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .expect("a non-empty vocabulary")
                    .0 as u32;
                out[i].push(best);
                next[i] = best;
            }
        }
        out
    };
    let four = greedy(&[1, 2, 3, 4]);
    for (i, &who) in [1u64, 2, 3, 4].iter().enumerate() {
        let alone = greedy(&[who]);
        eprintln!("who {who} alone: {:?}", alone[0]);
        eprintln!("who {who} in four: {:?}", four[i]);
        assert_eq!(
            four[i], alone[0],
            "conversation {who} answered differently in a batch of four than alone"
        );
    }
    assert!(
        four.iter().any(|s| s != &four[0]),
        "the four conversations agreed with each other, so this proves nothing"
    );
}
