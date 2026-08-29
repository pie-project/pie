//! The `masked` axis, end to end — and the gate that says the C1 axis runs.
//!
//! **WHAT THIS FILE IS FOR.** `masked` is design §0's second supergraph axis
//! and the first one beyond decode/prefill: a per-lane fact the model
//! declares, a run-length mask on the submission, an `attention.masked` arm
//! over its own window. C1 wrote the bits path and then found the catalog
//! could not run it, for three reasons that were each somebody's to fix and
//! none of them the mask path's (build log 20). This file used to PIN those
//! three refusals so that the day one was fixed the test asserting it would
//! fail and say so. C1b fixed all three, so every one of them has flipped:
//! what is asserted now is the fix, in the same place and against the same
//! catalog.
//!
//! ```text
//! blocker            what it was                        what it is now
//! kv::probe          one geometry per kv SPACE, and     facts are keyed by
//!                    gemma states two                   ROW and by PLAN
//! the schedule       one plan_prefill read by two       gemma mints six, one
//!                    classes -> Fault::Straddled        per (reading x class)
//! the windowed arm   "fa2 has no custom+sliding"        it always had one:
//!                                                       VariantCustom IS it
//! ```
//!
//! ```text
//! cargo test -p engine-cuda --test masked_axis
//! cargo test -p engine-cuda --features cuda-13 --test masked_axis -- --nocapture
//! ```

use engine::engine_api::fire::{Mask, Masking};
use engine_cuda::{Fault, LaneMask, Seated};
use model_compiler::{Budget, DeviceProfile, compile};
use model_dsl::Platform;
use model_ir::{Attention, Operation, Trace};

/// A deployment's ceilings, small: nothing in the host half loads a
/// checkpoint.
fn budget() -> Budget {
    Budget::new(8, 512)
}

/// How many `attention.masked` arms a SKU's trace carries.
fn masked_arms(trace: &Trace) -> usize {
    trace.nodes
        .iter()
        .filter(|node| matches!(node.op, Operation::Attention(Attention::Masked { .. })))
        .count()
}

/// **GEMMA IS THE ONLY FAMILY THAT DECLARES THE AXIS.**
///
/// Stated rather than assumed, because every other claim in this file is
/// about gemma and a reader is entitled to know why: `masked` is a
/// model-declared fact (design §8), the bits are a runtime input, and a plan
/// with no `attention.masked` node has nowhere for them to go. A qwen lane
/// carrying a mask is not "the axis without the gemma facts" — it is a mask
/// nothing reads.
#[test]
fn gemma_is_the_only_family_that_declares_the_masked_axis() {
    let mut declaring: Vec<(String, usize)> = Vec::new();
    for (sku, _, trace, _) in model::catalog() {
        let arms = masked_arms(&trace(Platform::Cuda));
        if arms > 0 {
            declaring.push((sku.to_string(), arms));
        }
    }
    assert!(
        declaring.iter().all(|(sku, _)| sku.starts_with("gemma4-")),
        "a family beyond gemma declares `attention.masked`, and the gates in \
         this file were written against gemma alone: {declaring:?}"
    );
    assert!(
        !declaring.is_empty(),
        "no SKU declares `attention.masked` at all, and then the axis has no \
         model text to be exercised by"
    );
}

/// **BLOCKER 1, FLIPPED: one kv space, two readings, and the facts are keyed
/// by what actually holds them.**
///
/// Gemma alternates a 256-wide sliding attention with a 512-wide global one
/// and puts every layer's cache row in ONE space. That is the truth, not a
/// mis-declaration: a space is the PAGE-ID space — one page size, one block
/// per slot, one page list per lane — and gemma's two layer kinds are one
/// sequence, at the same lengths, written at the same offsets. What is NOT a
/// space fact is how wide the row a page id addresses is, or how far back a
/// reader may look. `kv::probe` used to fold both up to the space and refuse
/// a family that stated two; now it keys them by the cache ROW (the bytes one
/// page holds) and by the PLAN VALUE (the reading one schedule is carved
/// for), which is where the IR puts them — `CacheRow::Kv { planes }` is per row,
/// and `Attention::PlanDecode { q_heads, kv_heads, head_dim, window }` states
/// the reading its schedule is carved for, which `Attention::Decode
/// { head_dim, window }` then restates per launch.
///
/// The dev lineage says the same thing in its own vocabulary: one `KvCache`,
/// one `num_pages`/`page_size`, and `per_layer_head_dim_` /
/// `per_layer_num_kv_heads_` vectors beside a `per_layer_window_left` on the
/// weights.
#[test]
fn gemma_s_one_kv_space_carries_two_readings_and_probes_them_apart() {
    let trace = model::trace_of("gemma4-e4b-bf16-kv-bf16").expect("the catalog ships gemma")(
        Platform::Cuda,
    );
    let facts = engine_cuda::store::kv::probe(&trace).expect(
        "gemma's caches probe: the readings are keyed by row and by plan, not folded \
         up to the space",
    );

    // Every layer that OWNS a cache row declares one, at its own kind's
    // width: 20 sliding rows of 2x256 and 4 global rows of 2x512 (the other
    // 18 layers borrow, so they declare none).
    let mut widths: Vec<(u32, u32)> = facts
        .rows
        .iter()
        .flatten()
        .map(|row| (row.head_dim, row.kv_heads))
        .collect();
    widths.sort_unstable();
    widths.dedup();
    assert_eq!(
        widths,
        vec![(256, 2), (512, 2)],
        "gemma's cache rows are two head widths at two heads each"
    );

    // And six schedules, one per (reading x class): sliding and global, for
    // the decode arm, the prefill arm and the masked arm.
    let mut readings: Vec<(u32, Option<u32>)> = facts
        .plans
        .iter()
        .flatten()
        .map(|schedule| (schedule.reading.head_dim, schedule.reading.window))
        .collect();
    assert_eq!(readings.len(), 6, "gemma mints six schedules: {readings:?}");
    readings.sort_unstable();
    readings.dedup();
    assert_eq!(
        readings,
        vec![(256, Some(512)), (512, None)],
        "the six are two readings, three schedules each"
    );

    // The one space is still one space — that is the point of the ruling.
    let spaces = trace
        .caches
        .iter()
        .filter_map(|row| match row {
            model_ir::CacheRow::Kv { space, .. } => Some(*space),
            model_ir::CacheRow::State { .. } => None,
        })
        .max()
        .map_or(0, |top| top + 1);
    assert_eq!(spaces, 1, "gemma is served out of ONE page-id space");
}

/// And every other family still probes, including the one that had the same
/// refusal.
///
/// **THE OTHER HALF OF A FIX.** gpt-oss alternates a 128-wide window with
/// full attention over one space at ONE row width, so its refusal was purely
/// about the window — and it cleared the same way, by minting one schedule
/// per reading in its own model text.
#[test]
fn every_sku_probes_its_caches() {
    let mut refused: Vec<String> = Vec::new();
    for (sku, _, trace, _) in model::catalog() {
        if let Err(fault) = engine_cuda::store::kv::probe(&trace(Platform::Cuda)) {
            refused.push(format!("`{sku}`: {fault}"));
        }
    }
    assert!(refused.is_empty(), "\n{}\n", refused.join("\n"));
}

/// **BLOCKER 2, FLIPPED: gemma's masked arm plans its own prefill.**
///
/// `plan_p` used to be minted once and read by both `attention.prefill` and
/// `attention.masked`, which stand in different classes. The compiler narrows
/// a prepare node by demand to the UNION of the classes reading its struct
/// (design build log 7) — right for a shared value, wrong for two windowed
/// readers — so the schedule was carved over both classes and each arm handed
/// it its own, narrower, rebased boundaries. Every work item past the first
/// request then indexed a `qo_indptr` that had already ended.
///
/// The fix was model text, and the net that catches the next one is upstream
/// of this file: `model_compiler::compile` refuses a straddle by name
/// (`model_compiler::Error::Straddled`) off `ClassTable::node_mask`, and
/// `crates/model/tests/no_schedule_straddles_its_readers.rs` asks the same
/// predicate one pass earlier, with no compiler in the room. This is the
/// shell's own restatement over a `CompiledModel`, kept because the shell asks it at
/// load and a `CompiledModel` can arrive from anywhere.
#[test]
fn no_sku_straddles_a_schedule() {
    let mut straddled: Vec<String> = Vec::new();
    for (sku, _, trace, _) in model::catalog() {
        let trace = trace(Platform::Cuda);
        let Ok(compiled) = compile(&trace, &budget(), &DeviceProfile::default()) else {
            straddled.push(format!("`{sku}`: does not bake"));
            continue;
        };
        if let Err(fault) = engine_cuda::window::no_schedule_straddles_its_readers(&trace, &compiled) {
            straddled.push(format!("`{sku}`: {fault}"));
        }
    }
    assert!(straddled.is_empty(), "\n{}\n", straddled.join("\n"));
}

/// **BLOCKER 3, FLIPPED: the windowed custom-mask arm was there all along.**
///
/// C1 recorded "fa2 instantiates no custom-mask + sliding-window arm", and it
/// was a misreading of the variant's own template arguments. `VariantCustom`
/// is `flashinfer::DefaultAttention<use_custom_mask = true, use_sliding_window
/// = true, ...>`: its `REGISTER_LOGITS_MASK` ANDs the custom bit with
/// `kv_idx + qo_len + window_left >= kv_len + qo_idx`, and `window_left` is
/// `params.window_left` when that is non-negative and `kv_len` — which makes
/// the term vacuous — when it is not. The unwindowed masked path has been
/// firing the windowed arm at `window_left = -1` since C1. Nothing needed
/// instantiating; a refusal in `attn::masked` stood in front of it, and what
/// it feared ("a windowed schedule would discard positions the mask may
/// keep") is the wrong way round for a model that STATES a window: gemma's
/// masked reading is causal ∧ mask ∧ window, and the window is the second
/// conjunct rather than an approximation of the first.
///
/// Host-side this can only assert the shape of the claim — that gemma still
/// states a window on most of its masked arms and not on all of them, so the
/// arm is partly windowed and the gate below exercises both halves. The
/// arithmetic is [`the_masked_arm_says_what_the_causal_arm_says`].
#[test]
fn gemma_states_a_sliding_window_on_most_of_its_masked_arms() {
    let trace = model::trace_of("gemma4-e4b-bf16-kv-bf16").expect("the catalog ships gemma")(
        Platform::Cuda,
    );
    let windowed = trace
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            Operation::Attention(Attention::Masked { window, .. }) => Some(*window),
            _ => None,
        })
        .filter(Option::is_some)
        .count();
    let arms = masked_arms(&trace);
    assert_eq!((windowed, arms), (35, 42), "five layers of every six slide");
}
/// The bits a lane's runs expand to are the bits the device text addresses.
///
/// **ONE TEST THAT IS ABOUT THE ARITHMETIC AND NOT ABOUT A REFUSAL.** The
/// custom-mask variant reads `qo_idx * kv_len + kv_idx`, LSB-first inside
/// each byte, at `maybe_custom_mask + maybe_mask_indptr[batch_idx]` — a BYTE
/// offset on this device text, not the bit offset upstream flashinfer
/// carries. This walks the staged bytes exactly that way, over a fire whose
/// lanes are DIFFERENT lengths and whose masked lane is not the first, so the
/// span table is doing real work.
#[test]
fn the_staged_bits_read_back_the_way_the_device_text_addresses_them() {
    // Lane 0: unmasked, 3 held, 2 new. Lane 1: masked, 5 held, 3 new, with
    // positions 0 and 6 dropped. Lane 2: unmasked decode.
    let mask = Masking::Extent(Mask::new(vec![1, 5, 1, 1], 8));
    let staged = engine_cuda::mask::stage(&[
        LaneMask {
            mask: None,
            have: 3,
            rows: 2,
        },
        LaneMask {
            mask: Some(&mask),
            have: 5,
            rows: 3,
        },
        LaneMask {
            mask: None,
            have: 9,
            rows: 1,
        },
    ])
    .expect("the mask covers its lane")
    .expect("a masked fire stages bits");

    // Lane 1 is 3 x 8 = 24 cells = 3 bytes, and it is the only lane holding
    // any, so the table is 0,0,3,3.
    assert_eq!(staged.indptr, vec![0, 0, 3, 3]);
    assert_eq!(staged.bits.len(), 3);

    let base = staged.indptr[1] as usize;
    let kv = 8usize;
    for q in 0..3usize {
        for k in 0..kv {
            let cell = q * kv + k;
            let read = (staged.bits[base + cell / 8] >> (cell % 8)) & 1 == 1;
            // The runs keep 1..=5 and 7; the causal bound keeps k <= 5 + q.
            let want = ((1..=5).contains(&k) || k == 7) && k <= 5 + q;
            assert_eq!(read, want, "cell ({q}, {k}) of lane 1");
        }
    }
    // The one position both terms drop, from opposite directions: key 7 is
    // KEPT by the runs and reachable only by the last query row.
    let seven = |q: usize| {
        let cell = q * kv + 7;
        (staged.bits[base + cell / 8] >> (cell % 8)) & 1 == 1
    };
    assert!(!seven(0) && !seven(1) && seven(2));
}

/// A mask against a plan with no masked arm is refused BY NAME, at the fire.
///
/// **THE REFUSAL THIS WAVE REPLACED, AND WHY IT MOVED.** The shell used to
/// answer `Unsupported { verb: "explicit attention masks" }` for every model,
/// which said the CUDA plane could not carry a mask. It can: the bits stage,
/// the seats bind, the span table slices. What decides is the artifact, so
/// the refusal is now `Fault::Maskless` and it is asked against the loaded
/// plan.
///
/// Skips without a device and a checkpoint, like every other test in this
/// tree that needs one.
#[test]
fn a_mask_against_a_maskless_artifact_is_refused_by_name() {
    if !engine_cuda::device::present() {
        eprintln!("skipping the maskless refusal: no CUDA device on this machine");
        return;
    }
    let Some((mut shell, _)) = common::ready("the maskless refusal") else {
        return;
    };
    shell.open(0).expect("slot 0 opens");
    let tokens = [9707u32, 11, 1879];
    let mask = Masking::Extent(Mask::new(vec![0, 3], 3));
    let refused = shell.fire_seated(&[engine_cuda::Seated::masked(
        engine_cuda::Lane {
            slot: 0,
            word: common::word(tokens.len() as u32),
            tokens: &tokens,
        },
        &mask,
    )]);
    assert!(
        matches!(refused, Err(Fault::Maskless { lane: 0 })),
        "a masked lane against qwen — which bakes no `attention.masked` arm — \
         must be refused by name, not run unmasked: {refused:?}"
    );

    // And the same lane WITHOUT a mask still fires, so the refusal is about
    // the mask and not about the submission around it.
    let fired = shell.fire(&[engine_cuda::Lane {
        slot: 0,
        word: common::word(tokens.len() as u32),
        tokens: &tokens,
    }]);
    assert!(
        fired.is_ok(),
        "the same lane without a mask must still fire: {fired:?}"
    );
}

// ── THE GATE: gemma, on a device, with all three classes co-firing ─────────

/// **THE §0 THREE-CLASS GOLDEN.** A decode lane, a prefill lane and a MASKED
/// lane in one fire say what each says alone.
///
/// This is what C1b was for. Design §0's claim is that a fire is a batch of
/// lanes standing in different classes and that composing them changes no
/// lane's arithmetic; C1 could state that for two classes (decode beside
/// prefill, `serve_smoke`) and not for three, because the third class had no
/// model that could load. The lanes here are deliberately different lengths
/// and the masked one is not first, so the window table, the mask span table
/// and the per-class page arithmetic are all doing real work.
///
/// Greedy, because an identity between two runs is only available if the
/// sampling is a function of the logits alone.
#[test]
fn a_mixed_fire_of_all_three_classes_says_what_each_lane_says_alone() {
    let _serial = gemma::serialized();
    let Some((mut shell, tok)) = gemma::ready("the three-class golden") else {
        return;
    };
    const STEPS: usize = 8;

    let carried = tok.encode(&gemma::turn("What is the capital of France? Answer in one word."));
    let fresh = tok.encode(&gemma::turn("Name the largest planet. One word."));
    let masked = tok.encode(&gemma::turn("What colour is the sky on a clear day? One word."));
    // The mask keeps every position of the masked lane's prompt, so the arm
    // has to reproduce the causal answer exactly — the same claim
    // `the_masked_arm_says_what_the_causal_arm_says` makes at length, made
    // here inside a mixed fire.
    let keep = Masking::Extent(Mask::new(vec![0, masked.len() as u32], masked.len() as u64));

    // Solo: each lane on its own slot, alone in its fire.
    let solo_carried = gemma::solo(&mut shell, 0, &carried, None, STEPS);
    let solo_fresh = gemma::solo(&mut shell, 1, &fresh, None, STEPS);
    let solo_masked = gemma::solo(&mut shell, 2, &masked, Some(&keep), STEPS);

    // Mixed. Fire one seats the carried lane's prefill so that from fire two
    // on it is a DECODE lane beside a prefill and a masked one — three
    // classes, one fire, which is the composition this wave exists for. The
    // prefill and masked lanes are re-seated every step (a fresh slot each
    // time) so the shape repeats.
    shell.open(0).expect("slot 0 opens");
    let seated = shell
        .fire_seated(&[Seated::of(engine_cuda::Lane {
            slot: 0,
            word: gemma::word(carried.len() as u32, false),
            tokens: &carried,
        })])
        .expect("the carried lane prefills");
    // (The solo runs above already fired every one of these shapes, so the
    // tuner is warm on all of them by here.)
    let mut mixed_carried = vec![gemma::argmax(&seated[0])];
    let mut mixed_fresh: Vec<u32> = Vec::new();
    let mut mixed_masked: Vec<u32> = Vec::new();

    // STEPS + 1 fires, and the first one's prefill/masked answers are
    // dropped: it is the cold sighting of this composition's GEMM shapes and
    // `gemma::solo` warms its own for the same reason (build log 11).
    for step in 0..=STEPS {
        shell.open(1).expect("slot 1 re-opens");
        shell.open(2).expect("slot 2 re-opens");
        let fed = [*mixed_carried.last().expect("a token to feed")];
        let fire = shell
            .fire_seated(&[
                Seated::of(engine_cuda::Lane {
                    slot: 1,
                    word: gemma::word(fresh.len() as u32, false),
                    tokens: &fresh,
                }),
                Seated::masked(
                    engine_cuda::Lane {
                        slot: 2,
                        word: gemma::word(masked.len() as u32, true),
                        tokens: &masked,
                    },
                    &keep,
                ),
                Seated::of(engine_cuda::Lane {
                    slot: 0,
                    word: gemma::word(1, false),
                    tokens: &fed,
                }),
            ])
            .unwrap_or_else(|why| panic!("the three-class fire at step {step}: {why}"));
        if step > 0 {
            mixed_fresh.push(gemma::argmax(&fire[0]));
            mixed_masked.push(gemma::argmax(&fire[1]));
        }
        // The carried lane's first token came off its own prefill above, so
        // it takes one fewer of these than the two re-seated lanes do.
        if mixed_carried.len() < STEPS {
            mixed_carried.push(gemma::argmax(&fire[2]));
        }
    }

    eprintln!(
        "solo    decode {:?} | prefill {:?} | masked {:?}",
        tok.decode(&solo_carried, false),
        tok.decode(&solo_fresh[..1], false),
        tok.decode(&solo_masked[..1], false),
    );
    eprintln!(
        "mixed   decode {:?} | prefill {:?} | masked {:?}",
        tok.decode(&mixed_carried, false),
        tok.decode(&mixed_fresh[..1], false),
        tok.decode(&mixed_masked[..1], false),
    );

    assert_eq!(
        solo_carried, mixed_carried,
        "the DECODE lane of a three-class fire said {:?} where it said {:?} alone",
        tok.decode(&mixed_carried, false),
        tok.decode(&solo_carried, false),
    );
    // Every step re-seats the prefill and masked lanes at the same prompt, so
    // each of their answers is the same first token the solo run produced.
    assert!(
        mixed_fresh.iter().all(|&t| t == solo_fresh[0]),
        "the PREFILL lane of a three-class fire answered {:?} across {STEPS} \
         identical seatings, where it answered {:?} alone",
        tok.decode(&mixed_fresh, false),
        tok.decode(&solo_fresh[..1], false),
    );
    assert!(
        mixed_masked.iter().all(|&t| t == solo_masked[0]),
        "the MASKED lane of a three-class fire answered {:?} across {STEPS} \
         identical seatings, where it answered {:?} alone",
        tok.decode(&mixed_masked, false),
        tok.decode(&solo_masked[..1], false),
    );
}

/// **THE WINDOWED ARM'S ARITHMETIC.** A mask that keeps everything must
/// produce what the causal arm produces — on a sequence long enough that the
/// 512-wide sliding window genuinely truncates the prefix.
///
/// Two different kernels over two differently carved schedules: the causal
/// arm is `MaskMode::kCausal` with `VariantWindow`, the masked arm is
/// `kCustom` with `VariantCustom`, and the second reads a bit per (row, key)
/// with the causal bound already folded into the bits. If the window did not
/// compose with the mask — the thing C1 believed had no instantiation — the
/// two would disagree here and only here, because a short prompt never
/// reaches back past 512 and hides the whole question.
#[test]
fn the_masked_arm_says_what_the_causal_arm_says() {
    let _serial = gemma::serialized();
    let Some((mut shell, tok)) = gemma::ready("the windowed masked arm") else {
        return;
    };

    // Past the window on purpose: 512 is what gemma's sliding layers state,
    // so a prompt of ~700 tokens makes 35 of the 42 layers drop the front of
    // their own prefix.
    let long = gemma::long_prompt(&tok, 700);
    assert!(
        long.len() > 512,
        "the point of this test is a prefix the 512-wide window truncates, and this \
         one is {} tokens",
        long.len()
    );
    let keep = Masking::Extent(Mask::new(vec![0, long.len() as u32], long.len() as u64));

    let causal = gemma::solo(&mut shell, 0, &long, None, 4);
    let masked = gemma::solo(&mut shell, 1, &long, Some(&keep), 4);
    eprintln!(
        "{} tokens: causal {:?} / masked {:?}",
        long.len(),
        tok.decode(&causal, false),
        tok.decode(&masked, false),
    );
    assert_eq!(
        causal, masked,
        "the windowed masked arm disagreed with the windowed causal arm over a \
         {}-token prefix: {:?} against {:?}",
        long.len(),
        tok.decode(&masked, false),
        tok.decode(&causal, false),
    );

    // And a mask that DROPS something must change the answer, or the bits
    // were never read and the identity above proved nothing.
    //
    // Keep position ZERO and nothing else. Dropping a middle slab would be a
    // weaker control here — the long prompt is one sentence repeated, so its
    // halves say the same thing — and dropping the tail would leave rows with
    // no key at all under the 512-wide window, which is a different question
    // (an empty softmax) than the one this file asks. Every query row keeps
    // key 0 by the causal bound, so every row still attends exactly one key.
    let only_first =
        Masking::Extent(Mask::new(vec![0, 1, long.len() as u32 - 1], long.len() as u64));
    let cut = gemma::solo(&mut shell, 2, &long, Some(&only_first), 1);
    eprintln!("only-first-key {:?}", tok.decode(&cut, false));
    assert_ne!(
        causal[0], cut[0],
        "a mask that keeps one key of {} changed nothing, so the bits are not \
         reaching the kernel",
        long.len()
    );
}

/// **CAPTURE COVERS A MASKED COMPOSITION.** A new key captures once and
/// replays identically.
///
/// The masked class is a third window in the fire, a third attention schedule
/// in the prepare phase and a mask slab whose span table is sliced per window
/// — every one of which is a pointer or an extent a capture could freeze at
/// this fire's value instead of this KEY's. The counter is watched because
/// "it did not capture again" is not a property any output has.
#[test]
fn a_masked_composition_captures_once_and_replays_identically() {
    let _serial = gemma::serialized();
    let Some((mut shell, tok)) = gemma::ready("the masked replay") else {
        return;
    };
    const STEPS: usize = 6;

    let masked = tok.encode(&gemma::turn("Name a colour. One word."));
    let keep = Masking::Extent(Mask::new(vec![0, masked.len() as u32], masked.len() as u64));

    let mut run = |mode: engine_cuda::Graphs| {
        shell.set_mode(mode);
        let mut said = Vec::new();
        for _ in 0..STEPS {
            shell.open(0).expect("slot 0 re-opens");
            let fire = shell
                .fire_seated(&[Seated::masked(
                    engine_cuda::Lane {
                        slot: 0,
                        word: gemma::word(masked.len() as u32, true),
                        tokens: &masked,
                    },
                    &keep,
                )])
                .expect("the masked fire");
            said.push(gemma::argmax(&fire[0]));
        }
        said
    };

    let eager = run(engine_cuda::Graphs::Off);
    let shaped = run(engine_cuda::Graphs::Shaped);
    let replayed = run(engine_cuda::Graphs::On);

    let stats = shell.graph_stats();
    eprintln!(
        "masked capture: {} captured ({} nodes), {} replayed, {} declined",
        stats.captures, stats.nodes, stats.replays, stats.declined,
    );
    assert!(
        stats.captures >= 1 && stats.replays >= 1,
        "the masked composition neither captured nor replayed, so this compared \
         eager against eager: {stats:?}"
    );
    assert_eq!(
        eager, shaped,
        "graph-shaped schedules changed the masked answer before any graph existed"
    );
    assert_eq!(
        shaped, replayed,
        "the replayed masked fire disagreed with the eager one it was captured from: \
         {:?} against {:?}",
        tok.decode(&replayed, false),
        tok.decode(&shaped, false),
    );
}

/// **LAUNCH ISOLATION, AT THE SHELL.** Two sequences through one boot, on one
/// slot, say the same thing.
///
/// The serving stack's own gate is `tests/gpu/cuda_launch_isolation`; this is
/// the same discipline asked where the fix lives (build log 19: `have == 0 ->
/// pools.clear(slot)`), and asked of gemma, which is the family this wave
/// added. Gemma declares no recurrent bank, so what is at stake here is the
/// kv pool and the mask slab: a second sequence must not be able to tell that
/// the first one used the same pages.
#[test]
fn two_sequences_through_one_boot_say_the_same_thing() {
    let _serial = gemma::serialized();
    let Some((mut shell, tok)) = gemma::ready("gemma launch isolation") else {
        return;
    };
    let prompt = tok.encode(&gemma::turn("What is the capital of France? Answer in one word."));
    let keep = Masking::Extent(Mask::new(vec![0, prompt.len() as u32], prompt.len() as u64));

    // A masked sequence between the two, on the same slot, so the second
    // launch follows a DIFFERENT class as well as a different sequence.
    let first = gemma::solo(&mut shell, 0, &prompt, None, 6);
    let _between = gemma::solo(&mut shell, 0, &prompt, Some(&keep), 3);
    let second = gemma::solo(&mut shell, 0, &prompt, None, 6);

    eprintln!(
        "launch 1 {:?} / launch 2 {:?}",
        tok.decode(&first, false),
        tok.decode(&second, false),
    );
    assert_eq!(
        first, second,
        "the second sequence through one boot said {:?} where the first said {:?}",
        tok.decode(&second, false),
        tok.decode(&first, false),
    );
}

/// **THE WINDOWED PREFILL, ON THE DEVICE: EVERY ROW UNDER ITS OWN MASK.**
///
/// The shape `Masking::Rows` exists for, asked of the kernel rather than of
/// the expansion. A sliding-window prefill gives row `i` the keys `[i - w, i]`
/// and row `i + 1` the keys `[i + 1 - w, i + 1]` — two restrictions that are
/// not nested, so no `Masking::Extent` is either of them, and the lowering
/// that had only that form refused the fire by name (`palo B-mask`) after an
/// older one had silently used row ZERO's mask on every row.
///
/// **THE CONTROL IS THE SAME LAST ROW REACHED BY THE PROVEN PATH.** Only the
/// LAST row of this fire is windowed; every earlier row keeps everything,
/// which under the causal bound is the plain causal prefix. So the same last
/// row can be built a second way out of parts this file already trusts: fire
/// the first `n - 1` tokens as an ordinary causal prefill (no mask at all),
/// then feed the last token as a ONE-ROW lane carrying that row's window as a
/// `Masking::Extent` — the decode-shaped custom mask, the one form this shell
/// served before per-row masks existed and the one
/// `the_masked_arm_says_what_the_causal_arm_says` pins. Both readings see the
/// identical KV, built identically, and the identical key set on the row that
/// is read back. They must agree.
///
/// **AND THE WINDOW MUST BITE**, or the agreement above is two ways of
/// spelling the unmasked answer: the same prompt through the plain causal arm
/// must say something ELSE. Between them the two claims are decisive about
/// the thing that actually changed — a row's mask is that row's. Under the
/// row-zero substitution the per-row fire would keep everything on every row
/// (row 0's mask is the all-keeping one here) and land on the CAUSAL answer,
/// which is exactly what the second assertion refuses.
#[test]
fn a_windowed_prefill_masks_each_row_with_its_own_window() {
    let _serial = gemma::serialized();
    let Some((mut shell, tok)) = gemma::ready("the per-row window") else {
        return;
    };

    let prompt = tok.encode(&gemma::turn("What is the capital of France? Answer in one word."));
    let n = u32::try_from(prompt.len()).expect("a prompt of sane length");
    assert!(n > 4, "the window has to have something to cut: {n} tokens");
    // ONE KEY BACK. The last row keeps `[n - 2, n - 1]` and nothing else —
    // narrow on purpose, because the second assertion needs the window to
    // change the answer and a wide one over a short prompt might not.
    let front = n - 2;
    let window = Mask::new(vec![front, n - front], u64::from(n));
    let keep_all = Mask::new(vec![0, n], u64::from(n));

    // Rows 0..n-2 keep everything (causality does the rest); row n-1 is the
    // one that differs, which is what makes this mask two-dimensional.
    let mut rows = vec![keep_all; prompt.len()];
    rows[prompt.len() - 1] = window.clone();
    let per_row = Masking::Rows(rows);
    let extent = Masking::Extent(window);

    // Each reading is fired TWICE and the second kept — the dense autotuner
    // tunes a GEMM shape on its second sighting, so a cold fire and a warm
    // one are two tactic ladders (`gemma::solo` argues it at length).
    let mut windowed = 0u32;
    for _ in 0..2 {
        shell.open(0).expect("slot 0 opens");
        let said = shell
            .fire_seated(&[Seated::masked(
                engine_cuda::Lane {
                    slot: 0,
                    word: gemma::word(n, true),
                    tokens: &prompt,
                },
                &per_row,
            )])
            .expect("a per-row mask is ACCEPTED — the first half of the claim");
        windowed = gemma::argmax(&said[0]);
    }

    let mut control = 0u32;
    for _ in 0..2 {
        shell.open(1).expect("slot 1 opens");
        shell
            .fire_seated(&[Seated::of(engine_cuda::Lane {
                slot: 1,
                word: gemma::word(n - 1, false),
                tokens: &prompt[..prompt.len() - 1],
            })])
            .expect("the causal prefix fires");
        let said = shell
            .fire_seated(&[Seated::masked(
                engine_cuda::Lane {
                    slot: 1,
                    word: gemma::word(1, true),
                    tokens: &prompt[prompt.len() - 1..],
                },
                &extent,
            )])
            .expect("the one-row control fires");
        control = gemma::argmax(&said[0]);
    }

    let mut causal = 0u32;
    for _ in 0..2 {
        shell.open(2).expect("slot 2 opens");
        let said = shell
            .fire_seated(&[Seated::of(engine_cuda::Lane {
                slot: 2,
                word: gemma::word(n, false),
                tokens: &prompt,
            })])
            .expect("the unmasked prefill fires");
        causal = gemma::argmax(&said[0]);
    }

    eprintln!(
        "{n} rows: per-row {:?} / one-row control {:?} / causal {:?}",
        tok.decode(&[windowed], false),
        tok.decode(&[control], false),
        tok.decode(&[causal], false),
    );
    assert_eq!(
        windowed, control,
        "the per-row mask's last row said {:?} where the same row over the same \
         KV under the same key set through the one-row extent path said {:?}",
        tok.decode(&[windowed], false),
        tok.decode(&[control], false),
    );
    assert_ne!(
        windowed, causal,
        "a two-key window over a {n}-token prompt changed nothing, so either the \
         last row's own mask never reached the kernel or row zero's was used \
         for every row — which is the substitution `Masking::Rows` exists to end"
    );
}

/// The load, shared with `serve_smoke` in shape and stated here rather than
/// imported because a test binary is its own crate.
mod common {
    use std::path::{Path, PathBuf};

    use engine_cuda::{Boot, Shell};
    use model_compiler::Budget;
    use model_dsl::{Classify, Platform, Request};

    const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

    pub fn word(query_len: u32) -> u64 {
        model::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
    }

    fn snapshot() -> Option<PathBuf> {
        if let Ok(stated) = std::env::var("PIE_SMOKE_SNAPSHOT") {
            let path = PathBuf::from(stated);
            return path.is_dir().then_some(path);
        }
        let home = std::env::var("HOME").ok()?;
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

    pub fn ready(what: &str) -> Option<(Shell, tokenizer::Tokenizer)> {
        let Some(checkpoint) = snapshot() else {
            eprintln!("skipping {what}: no Qwen3.5-0.8B snapshot (set PIE_SMOKE_SNAPSHOT)");
            return None;
        };
        let Some(container) = container(&checkpoint) else {
            eprintln!("skipping {what}: {checkpoint:?} holds no tensor container");
            return None;
        };
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
            profile: None,
            page_size: 16,
            context: 512,
            slots: 4,
            ordinal: 0,
            graphs: engine_cuda::Graphs::Off,
            knobs: engine_cuda::Knobs::default(),
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
        Some((shell, tokenizer))
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// THE DEVICE-GEOMETRY CLASS, AGAINST THE HOST-GEOMETRY FIRE IT MUST EQUAL
// ─────────────────────────────────────────────────────────────────────────────

/// **THE WHOLE FIRE GEOMETRY OFF THE RINGS, AND THE SAME LOGITS.**
///
/// `GeometryClass::DeviceGeometry` is the claim that a fire's ids, its
/// positions, its readable extent, its page table, its write descriptor and
/// its attention mask can all come out of an attached instance's channel cells
/// instead of out of the submission. There is exactly one way to make that
/// claim decisive: fire the SAME two lanes twice — once with every one of
/// those values in the submission where the shell has always read them, once
/// with every one of them in a channel and nothing in the submission at all —
/// and require the two rectangles of logits to agree BIT FOR BIT.
///
/// Bit for bit and not nearly, because the two fires run the same kernels over
/// the same bytes: any disagreement is a number this shell resolved
/// differently, and there is no rounding to hide behind. A page CSR off by a
/// page, a `last_page_len` off by a token, a write descriptor pointing one
/// cell early, a mask packed MSB-first, the token ids read from `tail` instead
/// of `head` — each of them moves a logit.
///
/// **GEMMA BECAUSE THE MASK NEEDS AN ARM.** The device-resolved mask is half
/// the class (a beam search's ancestry is `gather(mask, parent)` and lives
/// nowhere but the device), and `attention.masked` is gemma's alone — see
/// `gemma_is_the_only_family_that_declares_the_masked_axis` at the top of this
/// file.
#[test]
fn a_device_geometry_fire_is_the_host_geometry_fire_it_describes() {
    let _one = gemma::serialized();
    let Some((mut shell, tokenizer)) = gemma::ready("the device-geometry gate") else {
        return;
    };
    let prompt = tokenizer.encode(&gemma::turn("Name one colour."));
    let held = prompt.len() as u32;

    // The two lanes' tokens, and the ancestry each attends. The mask DROPS
    // something (key 3 of every row) so that the fire it describes is not the
    // fire an all-keeping mask would describe: a gate whose mask kept
    // everything would pass with the bits never read.
    let feed: [u32; devgeo::LANES] = [prompt[prompt.len() - 1], prompt[0]];
    let extent = held + 1;
    let dense = devgeo::dense_mask(extent);

    // ── The control: everything in the submission, where it has always been.
    //    Fired TWICE and the second kept — the dense autotuner tunes a GEMM
    //    shape on its second sighting, so a cold fire and a warm one are two
    //    tactic ladders (see `gemma::solo`'s note). The device fire below is
    //    the same shape, so warming here warms it too.
    let control_mask = devgeo::control_masking(extent);
    let mut control = Vec::new();
    for _ in 0..2 {
        devgeo::prime(&mut shell, &prompt);
        control = devgeo::fire_host(&mut shell, &feed, held, &control_mask);
    }

    // ── The subject: the same two lanes, with nothing in the submission.
    devgeo::prime(&mut shell, &prompt);
    let (instance, table) = devgeo::bind_geometry_instance(&mut shell, &feed, held, &dense);
    let subject = devgeo::fire_device(&mut shell, instance, &table);

    assert_eq!(
        subject.len(),
        control.len(),
        "both fires read out {} lane(s)",
        devgeo::LANES
    );
    for (lane, (device, host)) in subject.iter().zip(&control).enumerate() {
        assert_eq!(
            device.len(),
            host.len(),
            "lane {lane}'s two readings are the same width"
        );
        let differing = device
            .iter()
            .zip(host.iter())
            .enumerate()
            .find(|(_, (d, h))| d.to_bits() != h.to_bits());
        assert!(
            differing.is_none(),
            "lane {lane}: the device-resolved geometry and the submitted one \
             disagree at {differing:?} — the ports resolved a different fire"
        );
    }
}

/// **AND THE WRITE DESCRIPTOR IS LOAD-BEARING.**
///
/// The equality above would still hold if `w_slot`/`w_off` were read and
/// thrown away, because it states exactly the descriptor
/// `store::kv::geometry_with` derives. This states a DIFFERENT one — the same
/// page, one cell earlier, over the token the prompt already wrote — and
/// requires the logits to move. That cell is inside the lane's own readable
/// extent and the mask keeps it, so a fire that honoured the descriptor
/// attends a rewritten key and a fire that ignored it does not.
///
/// It is the second half of what "the device resolves the geometry" means: a
/// beam search's whole fork mechanism is `B` lanes appending into cells the
/// seat's `have + row` arithmetic cannot name, and a shell that read the
/// descriptor without using it would serve every beam the first beam's cell.
#[test]
fn an_explicit_write_descriptor_lands_somewhere_the_seat_would_not_have() {
    let _one = gemma::serialized();
    let Some((mut shell, tokenizer)) = gemma::ready("the write-descriptor gate") else {
        return;
    };
    let prompt = tokenizer.encode(&gemma::turn("Name one colour."));
    let held = prompt.len() as u32;
    let feed: [u32; devgeo::LANES] = [prompt[prompt.len() - 1], prompt[0]];
    let extent = held + 1;
    let dense = devgeo::dense_mask(extent);

    devgeo::prime(&mut shell, &prompt);
    let (derived, table) = devgeo::bind_geometry_instance(&mut shell, &feed, held, &dense);
    let at_the_tail = devgeo::fire_device(&mut shell, derived, &table);

    devgeo::prime(&mut shell, &prompt);
    let (moved, table) = devgeo::bind_geometry_instance_at(&mut shell, &feed, held, &dense, held - 1);
    let one_cell_early = devgeo::fire_device(&mut shell, moved, &table);

    let same = at_the_tail
        .iter()
        .zip(&one_cell_early)
        .all(|(tail, early)| {
            tail.iter()
                .zip(early.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        });
    assert!(
        !same,
        "moving `w_off` back one cell changed nothing, so the write descriptor \
         the ports resolved is not the one the append used"
    );
}

/// The device-geometry gate's fixture: one guest program that is nothing but
/// descriptor ports, and the two fires it is compared through.
///
/// **THE PROGRAM HAS NO BODY ON PURPOSE.** What is under test is the
/// descriptor-port plane — `program::ports` reading committed cells and
/// `serve::prepare` using them — and a stage that computed anything would put
/// its own arithmetic between the seeds this module writes and the geometry
/// the fire resolves. The channels are seeded and the epilogue does nothing,
/// so the cell the port reads is the cell this file wrote, and a wrong logit
/// is the shell's reading of it.
mod devgeo {
    use crate::gemma;
    use engine::engine_api::fire::{Mask, Masking};
    use engine::engine_api::program::ProgramRegistration;
    use engine::tensor_ir::registry::GeometryClass;
    use engine_cuda::{Lane, Seated, Shell};
    use tensor_ir::container::{
        ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
    };
    use tensor_ir::registry::{ModelProfile, Port, Stage};
    use tensor_ir::types::{DType, Shape};

    /// Two lanes: the smallest fire in which "which lane does this value
    /// belong to" is a question at all.
    pub const LANES: usize = 2;
    /// The key width the mask rectangle is built at — the POOL's width and not
    /// the extent's, which is the ordinary shape (see `engine_cuda::mask`'s "a
    /// mask may be LONGER") and the one a beam search always has.
    pub const POOL: usize = 64;
    /// `gemma::ready`'s boot: 1024 tokens of context at 16 tokens a page.
    const PAGES_PER_SLOT: u32 = 64;
    const PAGE_SIZE: u32 = 16;

    /// Which pool slot lane `l` sits in, and the pages that slot owns.
    fn slot(lane: usize) -> u32 {
        lane as u32
    }

    fn pages_of(lane: usize, extent: u32) -> Vec<u32> {
        let base = slot(lane) * PAGES_PER_SLOT;
        (0..extent.div_ceil(PAGE_SIZE).max(1))
            .map(|page| base + page)
            .collect()
    }

    /// Both lanes back to "the prompt and nothing else", so the two fires
    /// under comparison start from one KV state.
    pub fn prime(shell: &mut Shell, prompt: &[u32]) {
        for lane in 0..LANES {
            let slot = slot(lane);
            shell.open(slot).expect("the slot opens");
            shell
                .fire(&[Lane {
                    slot,
                    word: gemma::word(prompt.len() as u32, false),
                    tokens: prompt,
                }])
                .expect("the prompt prefills");
        }
    }

    /// The ancestry every lane attends: everything the extent holds except key
    /// 3, over a rectangle `POOL` wide.
    pub fn dense_mask(extent: u32) -> Vec<bool> {
        (0..LANES)
            .flat_map(|_| (0..POOL).map(|key| key != 3 && (key as u32) < extent))
            .collect()
    }

    /// The same restriction, written independently as run lengths over the
    /// same `POOL`-wide axis — masked-out first. Independent on purpose: it is
    /// what makes the equality a claim about `mask::from_dense` and not a
    /// tautology.
    pub fn control_masking(extent: u32) -> Masking {
        let kept = extent.min(POOL as u32);
        Masking::Extent(Mask::new(
            vec![0, 3, 1, kept - 4, POOL as u32 - kept],
            POOL as u64,
        ))
    }

    /// The control: every value in the submission.
    pub fn fire_host(
        shell: &mut Shell,
        feed: &[u32; LANES],
        held: u32,
        mask: &Masking,
    ) -> Vec<Vec<f32>> {
        let tables: Vec<Vec<u32>> = (0..LANES).map(|l| pages_of(l, held + 1)).collect();
        let tokens: Vec<[u32; 1]> = feed.iter().map(|&token| [token]).collect();
        let seated: Vec<Seated<'_>> = (0..LANES)
            .map(|lane| Seated {
                pages: &tables[lane],
                held: Some(held),
                ..Seated::masked(
                    Lane {
                        slot: slot(lane),
                        word: gemma::word(1, true),
                        tokens: &tokens[lane],
                    },
                    mask,
                )
            })
            .collect();
        shell
            .fire_seated(&seated)
            .expect("the host-geometry fire runs")
    }

    /// The subject: two lanes carrying a slot, a word and the working set's
    /// flat table — which is everything a device-geometry lane carries, and
    /// the table is not geometry: it is the map the geometry is resolved
    /// THROUGH (`Seated::translation`).
    pub fn fire_device(shell: &mut Shell, instance: u64, table: &[u32]) -> Vec<Vec<f32>> {
        let seated: Vec<Seated<'_>> = (0..LANES)
            .map(|lane| {
                // NO ROWS AND NO IDS: a device-geometry submission states its
                // row split nowhere, exactly as the runtime ships it.
                Seated {
                    translation: table,
                    ..Seated::of(Lane {
                        slot: slot(lane),
                        word: gemma::word(1, true),
                        tokens: &[],
                    })
                }
            })
            .collect();
        shell
            .fire_attached(
                &seated,
                &[engine_cuda::serve::Attached {
                    lane: 0,
                    instance,
                    at: engine::engine_api::fire::Boundary::Epilogue,
                }],
            )
            .expect("the device-geometry fire runs")
    }

    /// Bind an instance whose seeds ARE this fire's geometry, with the write
    /// descriptor the seat would have derived.
    pub fn bind_geometry_instance(
        shell: &mut Shell,
        feed: &[u32; LANES],
        held: u32,
        dense: &[bool],
    ) -> (u64, Vec<u32>) {
        bind_geometry_instance_at(shell, feed, held, dense, held)
    }

    /// As [`bind_geometry_instance`], landing every lane's row at flat
    /// position `at` of its own page run instead of at its tail.
    pub fn bind_geometry_instance_at(
        shell: &mut Shell,
        feed: &[u32; LANES],
        held: u32,
        dense: &[bool],
        at: u32,
    ) -> (u64, Vec<u32>) {
        let extent = held + 1;
        let per_lane = extent.div_ceil(PAGE_SIZE).max(1) as usize;
        // **THE GUEST STATES RELATIVE INDEXES AND THE TABLE IS NOT THE
        // IDENTITY.** `ws.reserve` hands a guest `0 .. n` and nothing else
        // (`kv-working-set`: "never a physical page id"), so the seeds below
        // are `0, 1, 2, 3` — and lane 1's two of them map to pool pages 64 and
        // 65, which is a slot away from where an untranslated read would land.
        // A shell that pushed the seed straight into the page CSR would attend
        // pool pages 2 and 3, and the logits would not be the control's.
        let table: Vec<u32> = (0..LANES).flat_map(|lane| pages_of(lane, extent)).collect();
        let pages: Vec<u32> = (0..table.len() as u32).collect();
        let page_indptr: Vec<u32> = (0..=LANES as u32).map(|lane| lane * per_lane as u32).collect();
        let w_slot: Vec<u32> = (0..LANES)
            .map(|lane| (lane * per_lane) as u32 + at / PAGE_SIZE)
            .collect();
        let w_off: Vec<u32> = (0..LANES).map(|_| at % PAGE_SIZE).collect();

        let seeds: Vec<Seed> = vec![
            Seed::i32(Port::EmbedTokens, Shape::vector(LANES as u32), feed.iter().map(|&t| t as i32).collect()),
            Seed::u32(Port::EmbedIndptr, Shape::vector(LANES as u32 + 1), (0..=LANES as u32).collect()),
            Seed::u32(Port::Positions, Shape::vector(LANES as u32), vec![held; LANES]),
            // IN WIRE-TAG ORDER, which is what `tensor_ir::validate` requires
            // of a container's port table (`PortsUnsorted`): pages (3) and its
            // CSR (4) stand before the extent (5).
            Seed::u32(Port::Pages, Shape::vector(pages.len() as u32), pages),
            Seed::u32(Port::PageIndptr, Shape::vector(page_indptr.len() as u32), page_indptr),
            Seed::u32(Port::KvLen, Shape::vector(LANES as u32), vec![extent; LANES]),
            Seed::u32(Port::WSlot, Shape::vector(LANES as u32), w_slot),
            Seed::u32(Port::WOff, Shape::vector(LANES as u32), w_off),
            Seed::bools(Port::AttnMask, Shape::matrix(LANES as u32, POOL as u32), dense.to_vec()),
        ];
        let registration = registration(&seeds);
        let program = shell
            .register_program(&registration)
            .expect("a port-only program registers");
        let wire: Vec<(u32, Vec<u8>)> = seeds
            .iter()
            .enumerate()
            .map(|(index, seed)| (index as u32, seed.wire()))
            .collect();
        let ids: Vec<u64> = (0..seeds.len() as u64).collect();
        let instance = shell
            .bind_program(
                program,
                &wire,
                engine::Extents::default(),
                GeometryClass::DeviceGeometry,
                &[],
                &ids,
            )
            .expect("the instance binds in the device-geometry class");
        (instance, table)
    }

    /// One channel, its port and its seed.
    pub struct Seed {
        port: Port,
        shape: Shape,
        dtype: DType,
        value: engine::Value,
    }

    impl Seed {
        fn i32(port: Port, shape: Shape, lanes: Vec<i32>) -> Seed {
            Seed { port, shape, dtype: DType::I32, value: engine::Value::I32(lanes) }
        }
        fn u32(port: Port, shape: Shape, lanes: Vec<u32>) -> Seed {
            Seed { port, shape, dtype: DType::U32, value: engine::Value::U32(lanes) }
        }
        fn bools(port: Port, shape: Shape, lanes: Vec<bool>) -> Seed {
            Seed {
                port,
                shape,
                dtype: DType::Bool,
                value: engine::Value::Bool(lanes.into_iter().map(u8::from).collect()),
            }
        }
        /// The seed as the wire cell `bind_program` takes.
        fn wire(&self) -> Vec<u8> {
            let lanes = self.shape.numel() as usize;
            let mut bytes = vec![0u8; engine::wire_cell_bytes(self.dtype, lanes)];
            engine::encode_wire(&self.value, &mut bytes);
            bytes
        }
    }

    /// The port-only program, all the way to what `register_program` takes.
    fn registration(seeds: &[Seed]) -> ProgramRegistration {
        let mut container = TraceContainer {
            names: Vec::new(),
            externs: Vec::new(),
            channels: Vec::new(),
            ports: Vec::new(),
            stages: vec![StageProgram {
                stage: Stage::Epilogue,
                ops: Vec::new(),
            }],
        };
        for (index, seed) in seeds.iter().enumerate() {
            container.channels.push(ChannelDecl {
                shape: seed.shape,
                dtype: ChanDType::Concrete(seed.dtype),
                capacity: 1,
                host_role: HostRole::None,
                seeded: true,
            });
            container.ports.push(PortBinding {
                port: seed.port,
                source: PortSource::Channel(index as u32),
            });
        }
        let bound = tensor_ir::validate::bind(container, ModelProfile::dummy())
            .expect("a port-only container binds");
        let stages = tensor_compiler::plan::compile_bound(&bound);
        let launch = tensor_compiler::codegen::launch::build(&bound, &stages);
        let backend = tensor_compiler::codegen::program::Backend::Cuda;
        let emitted = tensor_compiler::codegen::program::emit_program(backend, &stages, &bound);
        ProgramRegistration {
            program_hash: bound.hash,
            emitted_kernels: emitted
                .into_iter()
                .map(|kernel| engine::engine_api::program::EmittedKernel {
                    kind: kernel.kind,
                    stage_index: kernel.stage_index,
                    region_index: kernel.region_index,
                    entry_name: kernel.entry_name,
                    source: kernel.source,
                    error: kernel.error,
                })
                .collect(),
            emitter_version: backend.emitter_version(),
            region_analysis: Vec::new(),
            launch,
            reference_ptir: Vec::new(),
        }
    }
}

/// The gemma load, and the greedy loop the gates above share.
mod gemma {
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, MutexGuard, PoisonError};

    use engine::engine_api::fire::Masking;
    use engine_cuda::{Boot, Lane, Seated, Shell};
    use model_compiler::Budget;
    use model_dsl::{Classify, Platform, Request};

    const SKU: &str = "gemma4-e4b-bf16-kv-bf16";

    /// One shell at a time per process — `kernels-cuda`'s scratch slabs are
    /// process-global and keyed by name, so two shells firing at once stage
    /// into the same bytes. The same mutex `serve_smoke` holds, for the same
    /// reason.
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

    /// One turn of gemma's own chat format. E4B is instruction-tuned and a
    /// bare completion prompt makes it echo; the format is what the gates'
    /// identities are asked over, not what they are about.
    pub fn turn(ask: &str) -> String {
        format!("<start_of_turn>user\n{ask}<end_of_turn>\n<start_of_turn>model\n")
    }

    /// A prompt of at least `want` tokens, so the 512-wide sliding window has
    /// something to truncate. Built out of one sentence repeated, which is
    /// fine: what is asked of it is an identity between two arms, not a
    /// continuation anyone reads.
    pub fn long_prompt(tok: &tokenizer::Tokenizer, want: usize) -> Vec<u32> {
        let mut text = String::from("<start_of_turn>user\n");
        while tok.encode(&text).len() < want {
            text.push_str("The quick brown fox jumps over the lazy dog. ");
        }
        text.push_str("<end_of_turn>\n<start_of_turn>model\n");
        tok.encode(&text)
    }

    /// One sequence, alone in its fires: prefill (masked or not) then greedy
    /// decode. The decode steps are never masked — the axis is per lane and
    /// per fire, and a one-token query has no prefix of its own to mask.
    ///
    /// **IT FIRES THE PROMPT TWICE AND KEEPS THE SECOND.** The dense
    /// autotuner tunes a GEMM shape on its SECOND sighting and runs the
    /// untuned cuBLAS ladder on the first (build log 11), so a cold fire and a
    /// warm one are two tactic ladders and their logits differ in the last
    /// bits. Every identity in this file is between STEADY STATES; comparing
    /// a cold solo run against a warm mixed one would be comparing the tuner
    /// against itself. Measured on gemma's masked arm: the two readings of one
    /// prompt put `"Blue"` at 20.375 and `"blue"` at 20.000 — two ulp of a
    /// bf16 logit apart — and the cold/warm pair disagreed about which was
    /// first.
    pub fn solo(
        shell: &mut Shell,
        slot: u32,
        prompt: &[u32],
        mask: Option<&Masking>,
        steps: usize,
    ) -> Vec<u32> {
        warm(shell, slot, prompt, mask);
        shell.open(slot).expect("the slot opens");
        let lane = Lane {
            slot,
            word: word(prompt.len() as u32, mask.is_some()),
            tokens: prompt,
        };
        let seated = match mask {
            Some(mask) => Seated::masked(lane, mask),
            None => Seated::of(lane),
        };
        let prefill = shell.fire_seated(&[seated]).expect("the prefill fires");
        let mut said = vec![argmax(&prefill[0])];
        for step in 1..steps {
            let fed = [*said.last().expect("a token to feed")];
            let decode = shell
                .fire(&[Lane {
                    slot,
                    word: word(1, false),
                    tokens: &fed,
                }])
                .unwrap_or_else(|why| panic!("decode step {step}: {why}"));
            said.push(argmax(&decode[0]));
        }
        said
    }

    /// One throwaway fire at this prompt's shape, so the tuner has seen it.
    fn warm(shell: &mut Shell, slot: u32, prompt: &[u32], mask: Option<&Masking>) {
        shell.open(slot).expect("the slot opens");
        let lane = Lane {
            slot,
            word: word(prompt.len() as u32, mask.is_some()),
            tokens: prompt,
        };
        let seated = match mask {
            Some(mask) => Seated::masked(lane, mask),
            None => Seated::of(lane),
        };
        shell.fire_seated(&[seated]).expect("the warming fire");
    }

    fn snapshot() -> Option<PathBuf> {
        if let Ok(stated) = std::env::var("PIE_GEMMA_SNAPSHOT") {
            let path = PathBuf::from(stated);
            return path.is_dir().then_some(path);
        }
        let home = std::env::var("HOME").ok()?;
        let snapshots = Path::new(&home)
            .join(".cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots");
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

    /// A loaded gemma, or `None` and a sentence saying what was missing.
    ///
    /// **THE BUDGETS ARE THE L40S's.** E4B is 13.9 GiB of bf16 weights, and
    /// its kv row is the SUM over the 24 layers that own one — 20 sliding at
    /// 2x2x256 and 4 global at 2x2x512, which is 56 KiB a token. At a
    /// 1024-token context over 4 slots that is 224 MiB of pool; the six
    /// schedules' grants are the other big number (a graph-shaped prefill
    /// schedule at head width 512 wants ~150 MiB of partials on its own), and
    /// the arena reserves `max_tokens` rows of a 262144-wide logit column.
    /// 1024/768 fits a 46 GiB card with room; larger tokens would not.
    pub fn ready(what: &str) -> Option<(Shell, tokenizer::Tokenizer)> {
        if !engine_cuda::device::present() {
            eprintln!("skipping {what}: no CUDA device on this machine");
            return None;
        }
        let Some(checkpoint) = snapshot() else {
            eprintln!(
                "skipping {what}: no gemma-4-E4B-it snapshot in the hugging face cache \
                 (set PIE_GEMMA_SNAPSHOT)"
            );
            return None;
        };
        let Some(container) = container(&checkpoint) else {
            eprintln!("skipping {what}: {checkpoint:?} holds no tensor container");
            return None;
        };
        let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
            .expect("the checkpoint's tokenizer loads");
        let trace = model::trace_of(SKU).expect("the catalog ships gemma")(Platform::Cuda);
        let source = ztensor_compat::index(&container).expect("the checkpoint opens");
        let contract = model::import_of(SKU).expect("the catalog ships an import")(&source)
            .expect("the import contract fits its own checkpoint");
        drop(source);

        let booted = std::time::Instant::now();
        let shell = Shell::load(Boot {
        // Full residency: the whole weight table on the device, which is what
        // an uncapped `Residency` plans (alto design §7).
        residency: engine_cuda::experts::Plan::default(),
            trace,
            contract: &contract,
            checkpoint: &checkpoint,
            budget: Budget::new(4, 768),
            profile: None,
            page_size: 16,
            context: 1024,
            slots: 4,
            ordinal: 0,
            graphs: engine_cuda::Graphs::Off,
            knobs: engine_cuda::Knobs::default(),
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
        let (weights, arena, pools, inputs) = shell.footprint();
        eprintln!(
            "gemma4-e4b loaded in {:.1}s — weights {:.2} GiB, arena {:.1} MiB, \
             pools {:.1} MiB, inputs {:.1} MiB",
            booted.elapsed().as_secs_f64(),
            weights as f64 / (1u64 << 30) as f64,
            arena as f64 / (1 << 20) as f64,
            pools as f64 / (1 << 20) as f64,
            inputs as f64 / (1 << 20) as f64,
        );
        Some((shell, tokenizer))
    }
}
