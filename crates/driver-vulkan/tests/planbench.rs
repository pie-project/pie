//! What a single rectangle costs to PLAN, with no card in the room.
//!
//! `tests/hostprof.rs` says a qwen3-0.6b decode step spends about 1.4 ms on
//! the host outside `run_all`, and that most of it is `fire/plan` over 452
//! rectangles. It cannot say what a rectangle's microsecond and a half is
//! MADE of, for two reasons a profile has to answer for:
//!
//! * its own tool is 47 ns a span pair and the plan is seven spans a
//!   rectangle, so a third of a fine-grained reading would be the reading;
//! * it fires a real GPU, so every number is quoted against a shared card
//!   whose wall step moves 1.7x between runs of the same binary.
//!
//! So this file plans the same 452 rectangles of the same lowering with a
//! `Buffer::placeholder` where the arena goes -- `plan_routine` is arithmetic
//! over offsets and never dereferences the handle, which is what
//! `tests/arena.rs` has always relied on -- and times STAGES by running each
//! one over every rectangle, many times, under one `Instant`. Attribution by
//! repetition rather than by instrumentation: no span is paid inside the loop
//! and the stages are the same calls `plan_routine` makes, in its order.
//!
//! # How to run it
//!
//! ```text
//! cargo test -p driver-vulkan --features native --release --test planbench \
//!     -- --nocapture
//! ```
//!
//! **Release, always**, for the reason `hostprof.rs` gives. Nothing is
//! asserted about nanoseconds; the test asserts only that the rectangles it
//! timed all planned, so that a stage cannot get cheap by refusing.

#![cfg(feature = "native")]

use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
use model::shared::llama_like::forward::llama_like_metal;
use model_compiler::lower::{Fire, Lowered, Row, lower};
use model_ir::trace::FireClass;

/// Where a `native` build of `kernels-vulkan` left the modules.
const SPV_DIR: Option<&str> = option_env!("PIE_KERNELS_VULKAN_SPV_DIR");

/// The strictest `minStorageBufferOffsetAlignment` a conformant device
/// reports, which is what `tests/arena.rs` plans against and is a superset of
/// what this machine's card asks.
const ALIGNMENT: u64 = 256;

/// An arena big enough that no weight or seam value is what refuses.
const GENEROUS: u64 = 1 << 30;

/// Says yes to everything, so that what is timed is the planning and not a
/// resolver's tables. Transcribed from `tests/arena.rs`, which explains each
/// answer.
struct Everything(driver_vulkan::device::Buffer);

impl driver_vulkan::binding::Resolve for Everything {
    fn weight(&self, _: &str) -> Option<&driver_vulkan::device::Buffer> {
        Some(&self.0)
    }
    fn named(&self, _: model_ir::trace::ValueId) -> Option<&driver_vulkan::device::Buffer> {
        Some(&self.0)
    }
    fn kv(&self, _: u16, _: bool) -> Option<&driver_vulkan::device::Buffer> {
        Some(&self.0)
    }
    fn table(
        &self,
        _: driver_vulkan::binding::FireTable,
    ) -> Option<&driver_vulkan::device::Buffer> {
        Some(&self.0)
    }
    fn number(&self, which: driver_vulkan::binding::FireNumber) -> Option<u32> {
        Some(match which {
            driver_vulkan::binding::FireNumber::KvPageSize => 0x0011_1111,
            driver_vulkan::binding::FireNumber::KvHeadStride => 0x0022_2222,
            driver_vulkan::binding::FireNumber::KvSeqStride => 0x0033_3333,
            driver_vulkan::binding::FireNumber::AttentionMaskStride
            | driver_vulkan::binding::FireNumber::KvHistoryBucket => 0,
        })
    }
}

/// The same, but answering weights the way the driver really does.
///
/// [`crate::resources::Weights`] holds `BTreeMap<String, Buffer>` over every
/// tensor name a checkpoint carries -- 704 of them for qwen3-0.6b -- and the
/// names share long prefixes (`model.layers.13.self_attn.q_proj.weight`), so
/// a lookup is ten string comparisons that agree for their first thirty
/// characters, over nodes scattered across the heap. [`Everything`] answers in
/// one instruction, so the difference between the two `bind` numbers below is
/// exactly what the real store costs.
struct Realistic {
    held: std::collections::BTreeMap<String, driver_vulkan::device::Buffer>,
    other: driver_vulkan::device::Buffer,
}

impl driver_vulkan::binding::Resolve for Realistic {
    fn weight(&self, name: &str) -> Option<&driver_vulkan::device::Buffer> {
        self.held.get(name).or(Some(&self.other))
    }
    fn named(&self, _: model_ir::trace::ValueId) -> Option<&driver_vulkan::device::Buffer> {
        Some(&self.other)
    }
    fn kv(&self, _: u16, _: bool) -> Option<&driver_vulkan::device::Buffer> {
        Some(&self.other)
    }
    fn table(
        &self,
        _: driver_vulkan::binding::FireTable,
    ) -> Option<&driver_vulkan::device::Buffer> {
        Some(&self.other)
    }
    fn number(&self, which: driver_vulkan::binding::FireNumber) -> Option<u32> {
        Some(match which {
            driver_vulkan::binding::FireNumber::KvPageSize => 0x0011_1111,
            driver_vulkan::binding::FireNumber::KvHeadStride => 0x0022_2222,
            driver_vulkan::binding::FireNumber::KvSeqStride => 0x0033_3333,
            driver_vulkan::binding::FireNumber::AttentionMaskStride
            | driver_vulkan::binding::FireNumber::KvHistoryBucket => 0,
        })
    }
}

/// The same store, hashed instead of ordered.
///
/// The one change under test: whether the driver's weight store should be a
/// `HashMap`. A `BTreeMap` lookup of `model.layers.13.self_attn.q_proj.weight`
/// among 704 such names is ten comparisons that agree for thirty characters
/// each, over nodes the allocator scattered; a hash is one pass over the name
/// and one probe.
struct Hashed {
    held: std::collections::HashMap<String, driver_vulkan::device::Buffer>,
    other: driver_vulkan::device::Buffer,
}

impl driver_vulkan::binding::Resolve for Hashed {
    fn weight(&self, name: &str) -> Option<&driver_vulkan::device::Buffer> {
        self.held.get(name).or(Some(&self.other))
    }
    fn named(&self, _: model_ir::trace::ValueId) -> Option<&driver_vulkan::device::Buffer> {
        Some(&self.other)
    }
    fn kv(&self, _: u16, _: bool) -> Option<&driver_vulkan::device::Buffer> {
        Some(&self.other)
    }
    fn table(
        &self,
        _: driver_vulkan::binding::FireTable,
    ) -> Option<&driver_vulkan::device::Buffer> {
        Some(&self.other)
    }
    fn number(&self, which: driver_vulkan::binding::FireNumber) -> Option<u32> {
        Some(match which {
            driver_vulkan::binding::FireNumber::KvPageSize => 0x0011_1111,
            driver_vulkan::binding::FireNumber::KvHeadStride => 0x0022_2222,
            driver_vulkan::binding::FireNumber::KvSeqStride => 0x0033_3333,
            driver_vulkan::binding::FireNumber::AttentionMaskStride
            | driver_vulkan::binding::FireNumber::KvHistoryBucket => 0,
        })
    }
}

/// The one decode this file times: qwen3-0.6b at one row, the same lowering
/// `hostprof.rs` fires and the same 452 rectangles.
fn decode() -> (Lowered, driver_vulkan::dispatch::Geometry) {
    let facts = LlamaLikeFacts::qwen3_0_6b();
    let metal = LlamaLikeMetalFacts {
        add_bias: true,
        ..LlamaLikeMetalFacts::synthetic()
    };
    let plan = llama_like_metal(&facts, &metal, FireClass::Decode);
    let low = lower(
        &plan,
        &[Row {
            samples: true,
            ..Row::default()
        }],
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the decode lowers");
    (
        low,
        driver_vulkan::dispatch::Geometry {
            q_heads: facts.q_heads,
            kv_heads: facts.kv_heads,
            head_dim: facts.head_dim,
            rotary_dims: facts.head_dim,
            n_experts: facts.n_experts,
            experts_per_token: facts.experts_per_token,
            ..Default::default()
        },
    )
}

/// Where a decode rectangle's planning time goes.
#[test]
fn what_one_rectangle_costs_to_plan() {
    let Some(dir) = SPV_DIR else {
        eprintln!("no modules: build with `--features native` and `slangc` on PATH");
        return;
    };
    let mut modules: std::collections::BTreeMap<String, Vec<u8>> = Default::default();
    for entry in std::fs::read_dir(dir)
        .expect("the module directory reads")
        .flatten()
    {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("spv")
            && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
        {
            modules.insert(stem.to_owned(), std::fs::read(&path).expect("a module"));
        }
    }
    let reflection =
        driver_vulkan::serve::Reflection::new(&modules, kernels_vulkan::Capability::Baseline);

    let (low, geometry) = decode();
    let buf = driver_vulkan::device::Buffer::placeholder(low.arena_bytes as u64);
    let store = Everything(driver_vulkan::device::Buffer::placeholder(GENEROUS));
    let arena = driver_vulkan::binding::Arena {
        buffer: &buf,
        bytes: low.arena_bytes as u64,
    };

    let rectangles = low.launches.len();
    assert!(rectangles > 0, "the decode states no rectangles");

    /// How many times each stage walks all 452 rectangles.
    const ROUNDS: usize = 200;

    // Warm: the routine memo and the reflection are both per-thread caches
    // that are cold on the first walk, and a cold walk is `hostprof`'s first
    // step and not its median.
    let mut planned = 0usize;
    let mut dispatches = 0usize;
    for launch in &low.launches {
        let symbol = &low.kernels[launch.kernel as usize];
        let routine = driver_vulkan::hold::routine_for(symbol).expect("a routine");
        if let Ok(d) = driver_vulkan::serve::plan_routine(
            &low,
            launch,
            symbol,
            routine,
            arena,
            &store,
            geometry,
            &reflection,
            ALIGNMENT,
        ) {
            planned += 1;
            dispatches += d.len();
        }
    }
    assert_eq!(
        planned, rectangles,
        "a rectangle refused, so the timing below is of refusals"
    );

    // MIN OF FIVE, not the mean. This box is shared with other work and the
    // same binary has been seen 1.9x apart between runs; the minimum of a
    // repeated pure walk is the one estimator that does not move with what
    // else the machine is doing, because nothing can make a stage FASTER than
    // it is.
    let bench = |label: &str, f: &mut dyn FnMut()| {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let at = std::time::Instant::now();
            for _ in 0..ROUNDS {
                f();
            }
            let ns = at.elapsed().as_secs_f64() * 1e9 / (ROUNDS * rectangles) as f64;
            best = best.min(ns);
        }
        eprintln!(
            "  {label:<28} {best:>8.1} ns/rect   {:>7.3} ms/step",
            best * rectangles as f64 / 1e6
        );
    };

    eprintln!(
        "\n=== qwen3-0.6b decode, {rectangles} rectangles, {dispatches} dispatches, \
         {ROUNDS} rounds, min of 5 ==="
    );

    // The store the driver really has: 704 tensor names in a `BTreeMap`,
    // which every stage below that resolves a weight goes through.
    let mut held = std::collections::BTreeMap::new();
    for arg in &low.args {
        if let model_compiler::lower::Arg::Weight(name) = arg {
            held.insert(
                name.clone(),
                driver_vulkan::device::Buffer::placeholder(GENEROUS),
            );
        }
    }
    let names = held.len();
    let real = Realistic {
        held,
        other: driver_vulkan::device::Buffer::placeholder(GENEROUS),
    };

    bench("routine_for", &mut || {
        for launch in &low.launches {
            let symbol = &low.kernels[launch.kernel as usize];
            std::hint::black_box(driver_vulkan::hold::routine_for(symbol));
        }
    });
    bench("bind (stand-in store)", &mut || {
        for launch in &low.launches {
            std::hint::black_box(
                driver_vulkan::binding::bind(&low, launch, arena, &store, ALIGNMENT).is_ok(),
            );
        }
    });
    bench("bind (704-name store)", &mut || {
        for launch in &low.launches {
            std::hint::black_box(
                driver_vulkan::binding::bind(&low, launch, arena, &real, ALIGNMENT).is_ok(),
            );
        }
    });
    let hashed = Hashed {
        held: real
            .held
            .keys()
            .map(|k| {
                (
                    k.clone(),
                    driver_vulkan::device::Buffer::placeholder(GENEROUS),
                )
            })
            .collect(),
        other: driver_vulkan::device::Buffer::placeholder(GENEROUS),
    };
    bench("bind (704-name hashed)", &mut || {
        for launch in &low.launches {
            std::hint::black_box(
                driver_vulkan::binding::bind(&low, launch, arena, &hashed, ALIGNMENT).is_ok(),
            );
        }
    });
    bench("spelled (memo)", &mut || {
        for launch in &low.launches {
            let symbol = &low.kernels[launch.kernel as usize];
            std::hint::black_box(driver_vulkan::hold::spelled(symbol));
        }
    });
    bench("affine_of + tile_of", &mut || {
        for launch in &low.launches {
            let symbol = &low.kernels[launch.kernel as usize];
            std::hint::black_box(driver_vulkan::hold::affine_of(symbol));
            std::hint::black_box(driver_vulkan::hold::tile_of(symbol));
        }
    });
    bench("traced_results + split", &mut || {
        for launch in &low.launches {
            let symbol = &low.kernels[launch.kernel as usize];
            let routine = driver_vulkan::hold::routine_for(symbol).expect("a routine");
            let args = &low.args[launch.args.start as usize..launch.args.end as usize];
            let results = driver_vulkan::hold::traced_results(routine);
            std::hint::black_box(driver_vulkan::hold::split(args, results));
        }
    });
    // Everything `plan_routine` does BEFORE the body: the operands, the
    // facts, the handles and the binder. What is left when this is subtracted
    // from the whole is the encoder and the routine body.
    bench("through the binder", &mut || {
        for launch in &low.launches {
            let symbol = &low.kernels[launch.kernel as usize];
            let routine = driver_vulkan::hold::routine_for(symbol).expect("a routine");
            let bound = driver_vulkan::binding::bind(&low, launch, arena, &real, ALIGNMENT)
                .expect("the operands bind");
            let args = &low.args[launch.args.start as usize..launch.args.end as usize];
            let results = driver_vulkan::hold::traced_results(routine);
            let (ins, outs, weights) = driver_vulkan::hold::split(args, results);
            let params: Vec<Option<u32>> = (launch.params.start as usize
                ..launch.params.end as usize)
                .map(|at| low.params.get(at).copied())
                .collect();
            let widths: Vec<u32> = args
                .iter()
                .filter_map(|a| match a {
                    model_compiler::lower::Arg::Arena { width, .. }
                    | model_compiler::lower::Arg::Named { width, .. } => Some(*width),
                    model_compiler::lower::Arg::Weight(_)
                    | model_compiler::lower::Arg::Raised { .. } => None,
                })
                .collect();
            // The same numbers PER ARGUMENT, which is how `Handles` indexes
            // them -- a weight holds a place there and carries no width, so
            // the compacted list above would shift every operand after it.
            let arg_widths: Vec<i32> = args
                .iter()
                .map(|a| match a {
                    model_compiler::lower::Arg::Arena { width, .. }
                    | model_compiler::lower::Arg::Named { width, .. } => (*width).cast_signed(),
                    model_compiler::lower::Arg::Weight(_)
                    | model_compiler::lower::Arg::Raised { .. } => 0,
                })
                .collect();
            let (group, bits) = driver_vulkan::hold::affine_of(symbol).unwrap_or((0, 0));
            let facts = driver_vulkan::hold::Facts {
                rows: launch.rows.end - launch.rows.start,
                width: widths.last().copied().unwrap_or(0),
                in_width: widths.first().copied().unwrap_or(0),
                q_heads: geometry.q_heads,
                kv_heads: geometry.kv_heads,
                head_dim: geometry.head_dim,
                rotary_dims: geometry.rotary_dims,
                n_experts: geometry.n_experts,
                experts_per_token: geometry.experts_per_token,
                group,
                bits,
                tile: driver_vulkan::hold::tile_of(symbol),
                layer: launch.layers.start,
                requests: low.n_requests,
                ..Default::default()
            };
            let mut handles =
                driver_vulkan::hold::Handles::new(
                    &bound,
                    &arg_widths,
                    &ins,
                    &outs,
                    &weights,
                    &params,
                    &real,
                );
            let r = driver_vulkan::hold::routine_for(symbol).expect("a routine");
            std::hint::black_box(
                driver_vulkan::bind::bind(r.args, r.sources, &mut handles, facts).is_ok(),
            );
            std::hint::black_box(handles.staged().len());
        }
    });
    bench("plan_routine (stand-in)", &mut || {
        for launch in &low.launches {
            let symbol = &low.kernels[launch.kernel as usize];
            let routine = driver_vulkan::hold::routine_for(symbol).expect("a routine");
            std::hint::black_box(
                driver_vulkan::serve::plan_routine(
                    &low,
                    launch,
                    symbol,
                    routine,
                    arena,
                    &store,
                    geometry,
                    &reflection,
                    ALIGNMENT,
                )
                .is_ok(),
            );
        }
    });
    bench("plan_routine (704-name)", &mut || {
        for launch in &low.launches {
            let symbol = &low.kernels[launch.kernel as usize];
            let routine = driver_vulkan::hold::routine_for(symbol).expect("a routine");
            std::hint::black_box(
                driver_vulkan::serve::plan_routine(
                    &low,
                    launch,
                    symbol,
                    routine,
                    arena,
                    &real,
                    geometry,
                    &reflection,
                    ALIGNMENT,
                )
                .is_ok(),
            );
        }
    });
    // What a fire really accumulates: the dispatches are KEPT until the
    // command buffer is recorded, so the allocator cannot hand the same block
    // back for the next rectangle and the working set grows through the walk.
    // Dropping each plan, as every stage above does, measures a loop no fire
    // runs.
    bench("plan_routine (kept, hashed)", &mut || {
        let mut kept: Vec<driver_vulkan::dispatch::Dispatch<'_>> = Vec::with_capacity(rectangles);
        for launch in &low.launches {
            let symbol = &low.kernels[launch.kernel as usize];
            let routine = driver_vulkan::hold::routine_for(symbol).expect("a routine");
            if let Ok(made) = driver_vulkan::serve::plan_routine(
                &low,
                launch,
                symbol,
                routine,
                arena,
                &hashed,
                geometry,
                &reflection,
                ALIGNMENT,
            ) {
                kept.extend(made);
            }
        }
        std::hint::black_box(kept.len());
    });
    eprintln!("  ({names} weight names in the store)");
}
