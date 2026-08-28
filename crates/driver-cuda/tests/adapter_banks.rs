//! The adapter axis, end to end — and the gate that says the C2 axis runs.
//!
//! **WHAT THIS FILE IS FOR.** LoRA is palo design §8's CORRECTION class: an
//! additive `ΔW·x` on an already-materialised output, over the rows of the
//! lanes that routed to an adapter, with the per-row adapter id arriving as
//! runtime data and the weights living in a device bank a serving verb writes.
//! Nothing about it is a hook and nothing about it recaptures. The claims:
//!
//! ```text
//! (a) the zero-adapter fire is the fire this shell always fired  — tokens and ms
//! (b) a registered ZERO adapter is the identity                  — bit for bit
//! (b') a real adapter moves the tokens, deterministically
//!      and an adapterless lane beside it does not               — no leak
//! (c) two adapters in one fire, each saying what it says alone
//! (d) registering another adapter captures nothing
//! (e) the refusals: an unknown bank, an id past capacity, a short plane,
//!     an adapter against a word that does not route
//! ```
//!
//! **THE SHAPE OF THE ADAPTERS USED HERE.** Every gate builds its planes by
//! hand rather than reading a PEFT checkpoint, because what is under test is
//! the SEAT and not anybody's fine-tune. Two constructions carry the whole
//! file:
//!
//! * the **zero** adapter — `A = 0`, `B` arbitrary. Its correction is exactly
//!   zero (`B·(0·x) = 0`, and adding a bf16 zero is exact), so a lane routed
//!   to it must produce the base model's logits BIT FOR BIT. That is the
//!   strongest single statement there is about an additive correction, and it
//!   fails on any of the ways the plumbing can be wrong: a mis-strided bank, a
//!   waist that is not cleared, a window that covers the wrong rows, a routes
//!   vector read at a lane offset instead of a row offset.
//! * the **scaled** family — one `(A, B)` pair at scales `0, ¼, ½, 1`. A
//!   correction is LINEAR in `B`, so the logit displacement it produces must
//!   vanish at zero and grow with the scale. Which is a prediction about the
//!   arithmetic that a kernel reading the wrong bank row would not satisfy,
//!   and one a test can check without knowing what any particular fine-tune
//!   ought to say.
//!
//! ```text
//! cargo test -p driver-cuda --features cuda-13 --test adapter_banks -- --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use driver_cuda::{AdapterPlane, Boot, Graphs, Lane, Seated, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

const PROMPT: &str = "The capital of France is";

/// How many greedy decode fires follow a prefill.
const STEPS: usize = 12;

/// One shell at a time per process — `kernels-cuda`'s scratch slabs are
/// process-global and keyed by name (`serve_smoke.rs` argues it whole).
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The lane word the model's own `Classify` computes — the two facts qwen
/// declares, and no third opinion about either.
fn word(query_len: u32, adapter: bool) -> u64 {
    model::qwen_3::forward::Facts::of(&Request::new(query_len, false).adapted(adapter)).word()
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        assert!(value.is_finite(), "logit {at} is {value}");
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

/// The largest absolute difference between two readouts.
fn displacement(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), right.len(), "two readouts of one vocabulary");
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

// ── the adapters ─────────────────────────────────────────────────────────

/// One adapter's planes for every bank of a load, built to `fill`.
///
/// `fill(bank_name, element_index) -> f32` and the bytes come out bf16,
/// because that is what the bank declared and `register_adapter` refuses a
/// plane that is not exactly one slot.
fn planes(shell: &Shell, fill: &dyn Fn(&str, usize) -> f32) -> Vec<(String, Vec<u8>)> {
    shell
        .banks()
        .iter()
        .map(|&(name, _, slot)| {
            let count = usize::try_from(slot).expect("a slot fits this host") / 2;
            let mut bytes = Vec::with_capacity(count * 2);
            for at in 0..count {
                bytes.extend_from_slice(&bf16_bits(fill(name, at)).to_le_bytes());
            }
            (name.to_string(), bytes)
        })
        .collect()
}

/// f32 to bf16, round-to-nearest-even — the same conversion the loader does,
/// stated here because a test that truncated would be registering a slightly
/// different adapter than the one it describes.
fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

fn register(shell: &mut Shell, id: u32, built: &[(String, Vec<u8>)]) {
    let planes: Vec<AdapterPlane<'_>> = built
        .iter()
        .map(|(bank, bytes)| AdapterPlane {
            bank: bank.as_str(),
            bytes,
        })
        .collect();
    shell
        .register_adapter(id, &planes)
        .unwrap_or_else(|why| panic!("registering adapter {id}: {why}"));
}

/// The ZERO adapter: `A = 0` everywhere, `B` a visible non-zero.
///
/// `B` is deliberately NOT zero. An adapter whose every plane were zero would
/// pass the identity gate even if the shell had silently skipped the arm; with
/// a loud `B` behind a zero `A`, the identity holds only if the projection
/// half really ran and really produced a zero waist.
fn zero_adapter(shell: &Shell) -> Vec<(String, Vec<u8>)> {
    planes(shell, &|bank, at| {
        if bank.ends_with(".lora_a") {
            0.0
        } else {
            // A non-trivial pattern rather than a constant, so a stride error
            // in the combine reads a different number than the right one.
            0.5 - ((at % 7) as f32) * 0.1
        }
    })
}

/// A LOUD adapter: entries big enough that the twenty-four stacked
/// corrections take the continuation somewhere else entirely.
///
/// What the token gates want. `seed` distinguishes two of them, and it enters
/// the pattern rather than a scale so the two are genuinely different maps
/// and not one map at two magnitudes.
fn loud_adapter(shell: &Shell, seed: usize) -> Vec<(String, Vec<u8>)> {
    planes(shell, &|bank, at| {
        let sign = if (at + seed) % 2 == 0 { 1.0 } else { -1.0 };
        if bank.ends_with(".lora_a") {
            sign * 0.02 * (((at % (11 + seed)) as f32) + 1.0)
        } else {
            sign * 0.02 * (((at % (7 + seed)) as f32) + 1.0)
        }
    })
}

/// A FAINT adapter at `scale`: the same `A` at every scale, `B` proportional.
///
/// **THE LINEAR REGIME IS WHERE A LINEARITY CLAIM IS CHECKABLE**, and this
/// entry exists because the loud one is nowhere near it — and because the
/// first two guesses at "faint" were not either.
///
/// The magnitude is CALIBRATED, not chosen. Twenty-four stacked corrections
/// reach the logits through an `lm_head` that is 248320 rows wide, and the
/// amplification is large: measured on this SKU, `A ~ 1e-3` with
/// `B ~ 1e-3` already displaces the readout by 5.80, and quadrupling from
/// there gives 13.44 and then 15.97 — a model changing its mind, not a
/// correction growing. `B ~ 3e-5` at the same `A` puts the base of the sweep
/// near 0.2, which is two decades under the bend and two bf16 ulp over the
/// readout's own quantization at a logit magnitude of ~20.
fn faint_adapter(shell: &Shell, scale: f32) -> Vec<(String, Vec<u8>)> {
    planes(shell, &|bank, at| {
        let sign = if at % 2 == 0 { 1.0 } else { -1.0 };
        if bank.ends_with(".lora_a") {
            sign * 1e-3 * (((at % 13) as f32) + 1.0)
        } else {
            scale * sign * 3e-5 * (((at % 11) as f32) + 1.0)
        }
    })
}

// ── the runs ─────────────────────────────────────────────────────────────

/// One sequence alone in its fires: prefill then `STEPS` greedy decodes,
/// routed to `adapter` throughout. Returns the tokens and the prefill's own
/// readout.
///
/// **IT FIRES THE PROMPT TWICE AND KEEPS THE SECOND.** The dense autotuner
/// tunes a GEMM shape on its second sighting (palo build log 11), so a cold
/// solo run and a warm mixed one are two tactic ladders. Every identity in
/// this file is between STEADY STATES, exactly as `masked_axis`'s are.
fn solo(
    shell: &mut Shell,
    slot: u32,
    prompt: &[u32],
    adapter: Option<u32>,
    steps: usize,
) -> (Vec<u32>, Vec<f32>) {
    let seat = |slot: u32| {
        let lane = Lane {
            slot,
            word: word(prompt.len() as u32, adapter.is_some()),
            tokens: prompt,
        };
        match adapter {
            Some(id) => Seated::adapted(lane, id),
            None => Seated::of(lane),
        }
    };
    shell.open(slot).expect("the slot opens");
    shell.fire_seated(&[seat(slot)]).expect("the warming fire");
    shell.open(slot).expect("the slot re-opens");
    let prefill = shell.fire_seated(&[seat(slot)]).expect("the prefill fires");
    let readout = prefill[0].clone();
    let mut said = vec![argmax(&readout)];
    for step in 1..steps {
        let fed = [*said.last().expect("a token to feed")];
        let lane = Lane {
            slot,
            word: word(1, adapter.is_some()),
            tokens: &fed,
        };
        let seated = match adapter {
            Some(id) => Seated::adapted(lane, id),
            None => Seated::of(lane),
        };
        let decode = shell
            .fire_seated(&[seated])
            .unwrap_or_else(|why| panic!("decode step {step}: {why}"));
        said.push(argmax(&decode[0]));
    }
    (said, readout)
}

// ── (a) the zero-adapter fire is the fire this shell always fired ────────

/// **THE 1.00x CLAIM, AND IT IS A CLAIM ABOUT AN ABSENCE.**
///
/// tart's correction class is 1.01x the no-divergence floor WITH adapters in
/// the batch; what a deployment that has registered none is entitled to is
/// 1.00x — the axis must be free when nobody uses it. The mechanism is design
/// §0's, not a special case: the correction is guarded on `has_adapter`, a
/// fire no lane routed has zero rows in that class, and
/// `driver::fire::walk` skips a zero-row region before it dispatches a node.
/// So the launch sequence of a plain fire is the launch sequence of a plain
/// fire on a plan with no bank at all.
///
/// Asserted two ways, because either alone is weak. The TOKENS pin the
/// arithmetic — a correction that added anything would move them — and the
/// MILLISECONDS pin the launches, against the shell's own measurement of the
/// same loop rather than against a number written down last week.
#[test]
fn a_fire_no_lane_routed_costs_the_axis_nothing() {
    let _serial = serialized();
    let Some((mut shell, tok)) = ready("the zero-adapter floor") else {
        return;
    };
    let prompt = tok.encode(PROMPT);

    // The reference: the same loop, warmed, measured.
    let (first, _) = solo(&mut shell, 0, &prompt, None, STEPS);
    let mut runs: Vec<(Vec<u32>, f64)> = Vec::new();
    for _ in 0..3 {
        shell.open(0).expect("slot 0 opens");
        let at = Instant::now();
        let (said, _) = solo(&mut shell, 0, &prompt, None, STEPS);
        runs.push((said, at.elapsed().as_secs_f64() * 1000.0 / (STEPS + 2) as f64));
    }
    for (said, _) in &runs {
        assert_eq!(
            *said, first,
            "two adapterless runs of one prompt through one boot disagree: {:?} vs {:?}",
            tok.decode(said, false),
            tok.decode(&first, false),
        );
    }
    let millis: Vec<f64> = runs.iter().map(|(_, ms)| *ms).collect();
    eprintln!(
        "zero-adapter fire: {:?} | {:.3} / {:.3} / {:.3} ms per fire",
        tok.decode(&first, false),
        millis[0],
        millis[1],
        millis[2],
    );

    // And the axis is genuinely absent, which is a fact about the WALK rather
    // than about the numbers: no region carrying the correction has rows in
    // this composition, so no `linear.lora_correct` was dispatched at all.
    // Read off the bake, because "a launch that did not happen" has no output.
    let corrections = shell
        .trace()
        .nodes
        .iter()
        .filter(|node| {
            matches!(
                node.op,
                model_ir::Operation::Linear(model_ir::Linear::LoraCorrect { .. })
            )
        })
        .count();
    assert_eq!(
        corrections,
        24,
        "the SKU should state one correction per layer, and the gates below \
         are about arms this plan does not carry otherwise"
    );
}

// ── (b) the zero adapter is the identity, bit for bit ────────────────────

/// **AN ADDITIVE CORRECTION OF ZERO IS EXACTLY ZERO.**
///
/// `A = 0` makes the waist zero, `B · 0` is zero, and a bf16 `y + 0` is `y`.
/// So a lane routed to this adapter — a lane whose WORD says `has_adapter`,
/// whose rows are inside the correction's window, and every one of whose
/// twenty-four correction arms therefore launched — must produce the base
/// model's logits bit for bit.
///
/// It is the strongest claim in the file and the cheapest to break. A bank
/// read at the wrong stride, a waist slab left over from the previous fire, a
/// routes vector indexed by lane instead of by row, a window resolved at the
/// fire's rectangle instead of the region's: every one of them shows up here
/// as a logit that moved.
#[test]
fn a_zero_adapter_is_the_identity_bit_for_bit() {
    let _serial = serialized();
    let Some((mut shell, tok)) = ready("the zero adapter") else {
        return;
    };
    let prompt = tok.encode(PROMPT);
    let built = zero_adapter(&shell);
    register(&mut shell, 0, &built);

    let (base, base_logits) = solo(&mut shell, 0, &prompt, None, STEPS);
    let (routed, routed_logits) = solo(&mut shell, 1, &prompt, Some(0), STEPS);

    eprintln!(
        "zero adapter: base {:?} | routed {:?}",
        tok.decode(&base, false),
        tok.decode(&routed, false),
    );
    assert_eq!(
        displacement(&base_logits, &routed_logits),
        0.0,
        "a lane routed to an all-zero-A adapter moved the logits, so the \
         correction is not additive-with-zero"
    );
    assert_eq!(
        base, routed,
        "a lane routed to an all-zero-A adapter said {:?} where the base model \
         said {:?}",
        tok.decode(&routed, false),
        tok.decode(&base, false),
    );
}

/// **AND THE DISPLACEMENT VANISHES WITH THE SCALE.**
///
/// A correction is linear in `B`: the same `A`, the same `x`, and `B` at
/// `0, ¼, ½, 1` must displace the first fire's logits by an amount that is
/// zero at zero and strictly increasing after it. That is a prediction about
/// the arithmetic — not about what any fine-tune ought to say — and a combine
/// that read the wrong bank row, or a projection that indexed the bank by the
/// route's value rather than by the row's route, would not satisfy it.
///
/// Strictly increasing rather than exactly proportional, deliberately: the
/// readout is the LAST layer's logits and twenty-four corrections stack
/// through a residual stream with three nonlinearities in each layer, so
/// proportionality is not what the model computes. Monotone in the scale is,
/// and it is what a wrong bank read breaks.
#[test]
fn the_displacement_vanishes_with_the_adapter() {
    let _serial = serialized();
    let Some((mut shell, tok)) = ready("the scaled family") else {
        return;
    };
    let prompt = tok.encode(PROMPT);
    let (_, base_logits) = solo(&mut shell, 0, &prompt, None, 1);

    let mut seen: Vec<(f32, f32)> = Vec::new();
    for scale in [0.0f32, 1.0, 4.0, 16.0] {
        let built = faint_adapter(&shell, scale);
        register(&mut shell, 1, &built);
        let (_, routed) = solo(&mut shell, 1, &prompt, Some(1), 1);
        seen.push((scale, displacement(&base_logits, &routed)));
    }
    eprintln!("scale -> max |dlogit|: {seen:?}");

    assert_eq!(
        seen[0].1, 0.0,
        "an adapter whose B is zero displaced the logits by {}, and `B·(A·x)` \
         at B = 0 is zero",
        seen[0].1
    );
    // Quadrupling the scale must more than double the displacement: growth is
    // the claim, and a factor-of-two floor keeps a bf16 readout's own
    // quantization from being what the assertion is reading.
    for pair in seen[1..].windows(2) {
        assert!(
            pair[1].1 > pair[0].1 * 2.0,
            "the displacement at scale {} is {} and at four times that scale it \
             is {}, which is not growth — a correction is linear in B",
            pair[0].0,
            pair[0].1,
            pair[1].1,
        );
    }
}

// ── (b') and (c): the mixed fire ─────────────────────────────────────────

/// **TWO ADAPTERS AND A BASE LANE IN ONE FIRE, EACH SAYING WHAT IT SAYS
/// ALONE.**
///
/// This is design §0's golden for the C2 axis, and it is the claim the whole
/// routed-bank shape exists to make: adapter diversity is absorbed INSIDE the
/// op, by per-row ids, with no branch and no second graph. Three lanes, three
/// different answers, one fire.
///
/// **THE MIXED/SOLO IDENTITY IS ON TOKENS AND HAS TO BE** (palo build log 21,
/// which found the same thing on the masked axis). A batched fire's shared
/// GEMMs genuinely run at a different `M` than a solo fire's — three lanes of
/// five rows is one 15-row matmul, not three 5-row ones — so cuBLAS picks a
/// different tactic and the last bits of a bf16 logit move. What §0 claims is
/// that the WINDOWS are the same computation, not that a batched matmul
/// rounds like an unbatched one. Measured here: the unadapted lane's logits
/// move by 0.17 at a magnitude of ~20, which is one bf16 ulp.
///
/// The leak question is asked separately and exactly, by
/// [`a_correction_reaches_no_row_outside_its_window`], where the fire's shape
/// is held fixed and only the bank's contents change.
#[test]
fn three_lanes_two_adapters_and_a_base_lane_agree_with_their_solo_runs() {
    let _serial = serialized();
    let Some((mut shell, tok)) = ready("the three-lane adapter golden") else {
        return;
    };
    let prompt = tok.encode(PROMPT);
    {
        let built = loud_adapter(&shell, 0);
        register(&mut shell, 0, &built);
    }
    {
        let built = loud_adapter(&shell, 3);
        register(&mut shell, 1, &built);
    }

    // Solo, warmed, one prefill readout each.
    let (_, solo_base) = solo(&mut shell, 0, &prompt, None, 1);
    let (_, solo_first) = solo(&mut shell, 1, &prompt, Some(0), 1);
    let (_, solo_second) = solo(&mut shell, 2, &prompt, Some(1), 1);

    let base_token = argmax(&solo_base);
    let first_token = argmax(&solo_first);
    let second_token = argmax(&solo_second);
    eprintln!(
        "solo: base {:?} | adapter 0 {:?} | adapter 1 {:?}",
        tok.decode(&[base_token], false),
        tok.decode(&[first_token], false),
        tok.decode(&[second_token], false),
    );
    assert_ne!(
        first_token, base_token,
        "adapter 0 produced the base model's token, so nothing was corrected"
    );
    assert_ne!(
        second_token, first_token,
        "two different adapters produced one token, so the route is not reaching \
         the bank"
    );

    // Mixed: the adapted lanes on either side of the base one, so the base
    // lane's rows are surrounded rather than at an end of the fire.
    for round in 0..3 {
        shell.open(0).expect("slot 0 opens");
        shell.open(1).expect("slot 1 opens");
        shell.open(2).expect("slot 2 opens");
        let mixed = shell
            .fire_seated(&[
                Seated::adapted(
                    Lane {
                        slot: 1,
                        word: word(prompt.len() as u32, true),
                        tokens: &prompt,
                    },
                    0,
                ),
                Seated::of(Lane {
                    slot: 0,
                    word: word(prompt.len() as u32, false),
                    tokens: &prompt,
                }),
                Seated::adapted(
                    Lane {
                        slot: 2,
                        word: word(prompt.len() as u32, true),
                        tokens: &prompt,
                    },
                    1,
                ),
            ])
            .unwrap_or_else(|why| panic!("the three-lane adapter fire, round {round}: {why}"));

        assert_eq!(
            argmax(&mixed[1]),
            base_token,
            "round {round}: the UNADAPTED lane of a mixed fire said {:?} where it \
             said {:?} alone",
            tok.decode(&[argmax(&mixed[1])], false),
            tok.decode(&[base_token], false),
        );
        assert_eq!(
            argmax(&mixed[0]),
            first_token,
            "round {round}: adapter 0's lane said {:?} in a mixed fire and {:?} \
             alone",
            tok.decode(&[argmax(&mixed[0])], false),
            tok.decode(&[first_token], false),
        );
        assert_eq!(
            argmax(&mixed[2]),
            second_token,
            "round {round}: adapter 1's lane said {:?} in a mixed fire and {:?} \
             alone",
            tok.decode(&[argmax(&mixed[2])], false),
            tok.decode(&[second_token], false),
        );
    }
}

/// **NO CORRECTION REACHES A ROW OUTSIDE ITS WINDOW, AND THIS IS THE EXACT
/// VERSION OF THAT CLAIM.**
///
/// The gate above compares a batched fire against solo ones, so its identity
/// is on tokens: the shared GEMMs run at a different `M` and the last bits of
/// a bf16 logit move for a reason that has nothing to do with adapters. Here
/// the confound is removed by holding EVERYTHING fixed but the bank's
/// contents.
///
/// Two fires, same three lanes, same three slots, same three words, same
/// classes, same windows, same `M` on every matmul, same attention schedules.
/// The only difference is what adapter 0 and adapter 1 hold: in one fire they
/// are loud, in the other they are the ZERO adapter, whose correction the gate
/// above proved is exactly nothing. So the middle lane — the one that routes
/// nowhere — must read bit for bit the same in both, and any difference is a
/// correction that wrote a row its window does not cover.
#[test]
fn a_correction_reaches_no_row_outside_its_window() {
    let _serial = serialized();
    let Some((mut shell, tok)) = ready("the exact leak gate") else {
        return;
    };
    let prompt = tok.encode(PROMPT);

    let fire = |shell: &mut Shell| -> Vec<Vec<f32>> {
        shell.open(0).expect("slot 0 opens");
        shell.open(1).expect("slot 1 opens");
        shell.open(2).expect("slot 2 opens");
        shell
            .fire_seated(&[
                Seated::adapted(
                    Lane {
                        slot: 1,
                        word: word(prompt.len() as u32, true),
                        tokens: &prompt,
                    },
                    0,
                ),
                Seated::of(Lane {
                    slot: 0,
                    word: word(prompt.len() as u32, false),
                    tokens: &prompt,
                }),
                Seated::adapted(
                    Lane {
                        slot: 2,
                        word: word(prompt.len() as u32, true),
                        tokens: &prompt,
                    },
                    1,
                ),
            ])
            .expect("the three-lane fire")
    };

    // The quiet reading: two zero adapters, whose correction is exactly zero.
    {
        let built = zero_adapter(&shell);
        register(&mut shell, 0, &built);
        register(&mut shell, 1, &built);
    }
    fire(&mut shell); // warm this composition's GEMM shapes
    let quiet = fire(&mut shell);

    // The loud reading: same fire, different bytes in the banks.
    {
        let built = loud_adapter(&shell, 0);
        register(&mut shell, 0, &built);
    }
    {
        let built = loud_adapter(&shell, 3);
        register(&mut shell, 1, &built);
    }
    let loud = fire(&mut shell);

    eprintln!(
        "same fire, two bank contents: adapted lanes moved by {:.4} and {:.4}, \
         the base lane by {:.4}",
        displacement(&quiet[0], &loud[0]),
        displacement(&quiet[2], &loud[2]),
        displacement(&quiet[1], &loud[1]),
    );
    assert!(
        displacement(&quiet[0], &loud[0]) > 0.0 && displacement(&quiet[2], &loud[2]) > 0.0,
        "neither adapted lane moved when the banks changed, so this fire is not \
         reading the banks at all and the assertion below is vacuous"
    );
    assert_eq!(
        displacement(&quiet[1], &loud[1]),
        0.0,
        "the lane that routes nowhere moved when two OTHER lanes' adapters \
         changed: a correction wrote rows its window does not cover"
    );
    let _ = &tok;
}

// ── (d) registering captures nothing ─────────────────────────────────────

/// **A REGISTRATION IS A POOL WRITE AND A TABLE ROW** (decision 17).
///
/// The graph key is a fire's COMPOSITION — which classes have how many rows
/// and lanes — and a bank's contents are not in it. The bank's addresses were
/// reserved at load and do not move. So a deployment adds its eighth adapter
/// between two fires of one shape and the second one replays the first's
/// graph.
///
/// Watched through the capture COUNTER, because "it did not recapture" is not
/// a property any output has. The counter is also what says the graph path was
/// exercised at all: a run in which nothing ever captured would pass a
/// no-recapture assertion vacuously.
#[test]
fn registering_another_adapter_captures_nothing() {
    let _serial = serialized();
    let Some((mut shell, tok)) = ready("the no-recapture counter") else {
        return;
    };
    let prompt = tok.encode(PROMPT);
    {
        let built = loud_adapter(&shell, 0);
        register(&mut shell, 0, &built);
    }
    shell.set_mode(Graphs::On);

    // Warm and capture this composition. A key captures on its SECOND fire —
    // the first is the eager pass that warms the scratch slabs, the JIT and
    // the autotuner (build log 11) — so the loop below is deliberately longer
    // than two.
    let (first, _) = solo(&mut shell, 0, &prompt, Some(0), 4);
    let after_first = shell.graph_stats();
    assert!(
        after_first.captures > 0,
        "nothing captured, so a no-recapture assertion would be vacuous \
         (stats: {after_first:?})"
    );

    // Seven more adapters, one after another, and a fire of the SAME shape
    // between each pair.
    for id in 1..8u32 {
        {
        let built = loud_adapter(&shell, id as usize);
        register(&mut shell, id, &built);
    }
        let before = shell.graph_stats().captures;
        let (said, _) = solo(&mut shell, 0, &prompt, Some(0), 4);
        let after = shell.graph_stats();
        assert_eq!(
            after.captures, before,
            "registering adapter {id} cost {} capture(s); a bank's contents are \
             not in the graph key",
            after.captures - before
        );
        assert_eq!(
            said, first,
            "registering adapter {id} changed what adapter 0's lane says: {:?} \
             vs {:?}",
            tok.decode(&said, false),
            tok.decode(&first, false),
        );
    }

    let end = shell.graph_stats();
    eprintln!(
        "eight adapters registered: captures {} -> {}, replays {}, nodes {}",
        after_first.captures, end.captures, end.replays, end.nodes,
    );
    assert_eq!(
        end.captures, after_first.captures,
        "seven registrations cost {} capture(s) between them",
        end.captures - after_first.captures
    );
    assert!(
        end.replays > after_first.replays,
        "no fire replayed after the registrations, so the counter is watching \
         nothing"
    );
}

// ── (e) the refusals ─────────────────────────────────────────────────────

/// **EVERY WAY A REGISTRATION CAN BE WRONG IS A SENTENCE WITH A NUMBER IN
/// IT** — the budget is a shape, so a caller that overran one is told which
/// one and by how much, at the door, before a byte is written.
#[test]
fn a_registration_the_banks_cannot_seat_is_refused_by_name() {
    let _serial = serialized();
    let Some((mut shell, _)) = ready("the registration refusals") else {
        return;
    };
    let good = zero_adapter(&shell);
    let (bank, adapters, slot) = {
        let banks = shell.banks();
        let (name, adapters, slot) = banks[0];
        (name.to_string(), adapters, slot)
    };

    let unknown = shell.register_adapter(
        0,
        &[AdapterPlane {
            bank: "layer.0.not_a_bank",
            bytes: &good[0].1,
        }],
    );
    assert!(
        format!("{}", unknown.expect_err("an unknown bank is refused"))
            .contains("not a bank this plan declares"),
        "an unknown bank should say so"
    );

    let past = shell.register_adapter(
        adapters,
        &[AdapterPlane {
            bank: &bank,
            bytes: &good[0].1,
        }],
    );
    let said = format!("{}", past.expect_err("an id past capacity is refused"));
    assert!(
        said.contains(&format!("seats {adapters} adapters")),
        "an id past capacity should name the capacity: {said}"
    );

    let short = vec![0u8; usize::try_from(slot).expect("a slot fits") - 2];
    let clipped = shell.register_adapter(
        0,
        &[AdapterPlane {
            bank: &bank,
            bytes: &short,
        }],
    );
    let said = format!("{}", clipped.expect_err("a short plane is refused"));
    assert!(
        said.contains("bytes per adapter"),
        "a short plane should name both byte counts: {said}"
    );
}

/// **AN ADAPTER AND A WORD THAT DISAGREE ARE REFUSED BEFORE ANYTHING
/// LAUNCHES** — `Fault::AdapterWord`, the mask's twin.
///
/// Both directions are a wrong answer that looks like a right one. A lane that
/// routed and whose word puts it outside the correction's window would answer
/// with the BASE MODEL under an adapter's name, which is exactly the silent
/// failure decision 17 makes the capacity a budget rather than an admission
/// cap to avoid. A lane inside the window that named no adapter would send the
/// arm at a routes vector this fire never staged.
#[test]
fn an_adapter_and_a_word_that_disagree_are_refused() {
    let _serial = serialized();
    let Some((mut shell, tok)) = ready("the word/adapter agreement") else {
        return;
    };
    let prompt = tok.encode(PROMPT);
    {
        let built = zero_adapter(&shell);
        register(&mut shell, 0, &built);
    }

    shell.open(0).expect("slot 0 opens");
    let routed_but_unstamped = shell.fire_seated(&[Seated::adapted(
        Lane {
            slot: 0,
            word: word(prompt.len() as u32, false),
            tokens: &prompt,
        },
        0,
    )]);
    let said = format!(
        "{}",
        routed_but_unstamped.expect_err("an unstamped routed lane is refused")
    );
    assert!(
        said.contains("outside the correction's window"),
        "the refusal should say the id would never be read: {said}"
    );

    shell.open(0).expect("slot 0 re-opens");
    let stamped_but_unrouted = shell.fire_seated(&[Seated::of(Lane {
        slot: 0,
        word: word(prompt.len() as u32, true),
        tokens: &prompt,
    })]);
    let said = format!(
        "{}",
        stamped_but_unrouted.expect_err("a stamped unrouted lane is refused")
    );
    assert!(
        said.contains("names no adapter"),
        "the refusal should say the arm has nothing to route with: {said}"
    );
}

// ── the load ─────────────────────────────────────────────────────────────

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

/// A loaded shell, or `None` and a sentence saying what was missing.
///
/// **`max_adapters` IS THE MODEL TEXT's OWN CAPACITY, ASKED FOR IN FULL.** The
/// budget is what the deployment intends to register and the bank's leading
/// axis is what the plan seats; `model_compiler::compile` refuses the load
/// when the first is bigger than the second, so asking for exactly what the
/// text declares is both the honest ask and the one that exercises the check.
fn ready(what: &str) -> Option<(Shell, tokenizer::Tokenizer)> {
    if !driver_cuda::device::present() {
        eprintln!("skipping {what}: no CUDA device on this machine");
        return None;
    }
    let Some(checkpoint) = snapshot() else {
        eprintln!(
            "skipping {what}: no Qwen3.5-0.8B snapshot in the hugging face cache \
             (set PIE_SMOKE_SNAPSHOT)"
        );
        return None;
    };
    let Some(container) = container(&checkpoint) else {
        eprintln!("skipping {what}: {checkpoint:?} holds no tensor container");
        return None;
    };
    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");
    let trace = model::trace_of(SKU).expect("the catalog ships the SKU")(Platform::Cuda);
    let seats = trace
        .params
        .iter()
        .filter(|param| param.source == model_ir::ParamSource::Registered)
        .map(|param| param.shape.first().copied().unwrap_or(0))
        .min()
        .expect("the SKU declares adapter banks");
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = model::import_of(SKU).expect("the catalog ships an import")(&source)
        .expect("the import contract fits its own checkpoint");
    drop(source);

    let shell = Shell::load(Boot {
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        budget: Budget {
            max_adapters: u32::try_from(seats).expect("a capacity fits a u32"),
            ..Budget::new(4, 256)
        },
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        ordinal: 0,
        graphs: Graphs::Off,
    })
    .expect("the shell loads");
    let (weights, arena, pools, inputs) = shell.footprint();
    eprintln!(
        "{SKU} loaded — weights {:.2} GiB, arena {:.1} MiB, pools {:.1} MiB, \
         inputs {:.1} MiB, {} banks x {} adapters",
        weights as f64 / (1u64 << 30) as f64,
        arena as f64 / (1 << 20) as f64,
        pools as f64 / (1 << 20) as f64,
        inputs as f64 / (1 << 20) as f64,
        shell.banks().len(),
        seats,
    );
    Some((shell, tokenizer))
}
