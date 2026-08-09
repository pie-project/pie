//! EVERY ROW ANSWERS FOR EVERY BACKEND, AND NO ROW ANSWERS BY ACCIDENT.
//!
//! This file is the guard that replaces a deleted table. Until this
//! refactor, `driver-metal/src/model/text.rs` held `LLAMA_LIKE` — eleven
//! architecture STRINGS (`llama`, `llama3`, `llama4`, `mistral`, `phi3`,
//! `olmo2`, `qwen2`, `qwen3`, `qwen3_moe`, `gpt_oss`, `gemma4`) reduced
//! by a punctuation-stripping `canonical()` — and asked it, in the
//! driver, before any text was traced: "do I serve this?"
//!
//! It was the THIRD dispatch key. The catalog exists because the first
//! two disagreed — an `architectures[0]` string chose a derivation, a
//! `model_type` string chose a chat template, and a checkpoint that
//! satisfied neither got a `_ =>` arm — and this table disagreed with
//! both in the same way. It listed `gpt_oss`, whose every publication
//! either fails this crate's manifest or names tensors `driver-metal`
//! has no handle for, so the answer to "served?" depended on which of
//! the two you asked. It omitted `gemma3`, whose
//! model the Metal text actually states. And it named `llama4`, for
//! which no row exists at all.
//!
//! A string table cannot be held to a text, because a string is not a
//! text. What CAN be held is this: every row in the catalog, asked for
//! a Metal load, either produces a Metal text or refuses in words. Not
//! a panic — a panic in a driver's load path is a process, not a
//! message. Not a CUDA plan — that is the silent corruption the whole
//! backend axis exists to prevent, and it is what a row DID return
//! before `Deployed::backend` existed: the Metal driver received
//! `llama_like.cuda.decode` and lowered CUDA symbol names against Metal
//! pipelines.
//!
//! # Why this is an integration test and not a unit test
//!
//! The property is about the CATALOG, not about any generation. A unit
//! test in a generation module can only see its own rows, and the
//! failure this catches is a generation ADDED without either arm being
//! wired — which is invisible from inside every module that exists. So
//! it walks `catalog()`, and its lower bound on the row count is there
//! because a walk of an empty iterator passes every assertion in it.

use model::catalog::{self, Backend, Deployed, MetalBinding};
use model::deployment::Refusal;
use model_compiler::kernels::Backend as TracedBackend;
use model_compiler::trace::FireClass;

/// A load's six observations, as `driver-metal`'s `observed()` builds
/// them for the encoding every published MLX checkpoint uses.
///
/// The values are `mlx-community`'s g64/b4 default with the three
/// kernel capabilities this build compiles. Nothing here should be
/// read as a MODEL fact — that is the whole claim `MetalBinding` makes
/// — and `a_row_answers_the_same_way_at_every_encoding` below is what
/// holds it to that.
const BINDING: MetalBinding = MetalBinding {
    quant_group: 64,
    quant_bits: 4,
    router_quant_group: 0,
    router_quant_bits: 0,
    moe_mxfp4: false,
    fuse_residual_gemv: true,
    paged_multi_batch: true,
    qmm_multi_batch: true,
    // TRUE, and the only one of the four whose value is a copy of something
    // this crate cannot see. `driver-metal::model::binding::build_kernels` is
    // where the claim lives; the layering forbids a dependency on it, so this
    // restates it and `every_role_the_mlx_map_answers_is_one_some_trace_asks
    // _for` is what catches the restatement going stale — with `false` here,
    // the Metal text states no bias, no trace asks for `q_bias`, and the map
    // entry that answers it looks like dead weight.
    add_bias: true,
};

/// The catalog is big enough that a walk of it means something.
///
/// Every test here iterates `catalog()`, and an iterator that yields
/// nothing satisfies a `for` loop full of assertions. Nineteen
/// generations are wired; the bound is deliberately well under that, so
/// it catches a catalog that has collapsed rather than one that has
/// merely grown or shrunk by a row.
fn assert_the_catalog_was_walked(rows: usize) {
    assert!(
        rows > 10,
        "only {rows} rows were enumerated, so this file is not reading the \
         catalog and its assertions have run over nothing"
    );
}

/// EVERY row either produces a Metal text or refuses Metal in words.
///
/// The guard the deleted `LLAMA_LIKE` table could not be. Three
/// outcomes are possible for a row asked to trace a Metal load, and
/// only two of them are allowed:
///
/// * a plan whose family is a METAL family — the row has a Metal text;
/// * a `Refusal` — the row says, in a sentence, what this build has
///   not got;
/// * a plan whose family is a CUDA family — the failure. It is what a
///   row returned before the backend was a parameter of the question,
///   and the driver that received it lowered CUDA symbol names against
///   Metal pipelines. Nothing faulted at trace time; the fire did.
///
/// The third is not reachable by construction any more, which is why
/// this asserts it rather than trusting it: `Variant::trace` is
/// nineteen independent implementations, and "I forgot to match on the
/// backend" is a one-line omission that compiles.
#[test]
fn every_row_either_traces_metal_or_refuses_it_in_words() {
    let mut rows = 0usize;
    let mut served = Vec::new();
    let mut refused = Vec::new();
    for row in catalog::catalog() {
        rows += 1;
        for class in [FireClass::Prefill, FireClass::Decode] {
            match row.trace(class, Deployed::metal(&BINDING)) {
                Ok(plan) => {
                    assert_eq!(
                        TracedBackend::of_family(&plan.family),
                        Some(TracedBackend::Metal),
                        "`{}` answered a METAL load with `{}`. A row that traces \
                         one backend's text for both is the silent corruption \
                         this axis exists to prevent — the driver receiving this \
                         would lower CUDA symbol names against Metal pipelines",
                        row.id(),
                        plan.family
                    );
                    assert!(
                        !plan.ops.is_empty(),
                        "`{}` traced a Metal plan of no operations",
                        row.id()
                    );
                    if class == FireClass::Decode {
                        served.push(row.id());
                    }
                }
                Err(Refusal::Unsupported(why)) => {
                    assert!(
                        why.len() > 40,
                        "`{}` refused Metal with `{why}`, which does not name \
                         what is missing. `Refusal::Unsupported` replaced a \
                         single sentence used for nine unrelated causes, and a \
                         refusal that says nothing is that sentence again",
                        row.id()
                    );
                    if class == FireClass::Decode {
                        refused.push(row.id());
                    }
                }
                Err(other) => panic!(
                    "`{}` refused a Metal load with `{other:?}`. A build with no \
                     text for a row is `Unsupported` — the checkpoint is fine, \
                     and a pie whose Metal half had that text would serve the \
                     same row unchanged",
                    row.id()
                ),
            }
        }
    }
    assert_the_catalog_was_walked(rows);
    assert!(
        !served.is_empty(),
        "no row in the catalog traces a Metal text, so `llama_like_metal` is \
         reachable by nothing and this whole axis is dead code"
    );
    assert!(
        !refused.is_empty(),
        "every row traces Metal, which cannot be right: this build has ONE \
         Metal text and twelve `*_cuda` texts, so the generations whose \
         forward is not llama-like must be refusing and are not"
    );
}

/// Only the family's generations serve Metal, and none of them is dark.
///
/// A list, in a test, which is the thing this refactor deleted from a
/// driver — and the difference is where it lives and what it is
/// checked against. `LLAMA_LIKE` was consulted to DECIDE the answer,
/// in the driver, from a string that reached it through a config file.
/// This is derived from the answers the rows give and compared against
/// the GENERATIONS a reader was told serve Metal.
///
/// The expected set is built from each generation's own `rows()` rather
/// than from a list of row ids or a prefix match on them. Both of those
/// were tried and both were wrong: `ministral-8b` is a `mistral_3` row
/// whose id does not begin with `mistral`, and `embeddinggemma-300m` is
/// a `gemma_3` row whose id names no gemma-3 at all. A row id is a
/// NAME, and deciding anything from the shape of a name is the habit
/// this whole refactor is unwinding.
///
/// # Why this is two weaker claims and not one equality
///
/// It asserted that the serving rows are EXACTLY the listed
/// generations' rows, and that is false in both of the ways a row can
/// differ from its generation.
///
/// `phi-3-mini-4k` is a listed row that refuses: its heads are 96 wide
/// and `sdpa_paged.metal` instantiates 64, 128, 256 and 512, so the
/// text would name a symbol no shader defines. CUDA pads to 128 and
/// strips; Metal has no pad in the text. `gemma-4-26b-a4b` is a listed
/// row that refuses on ANY backend — the build cannot LOAD a routed
/// gemma-4 block, and it says the same to CUDA. (The refusal used to
/// say the text was missing. It is written; see `Gemma4::untraced`.)
/// Neither is a generation losing Metal, and an equality cannot say so.
///
/// So the two directions are stated separately, and the second is the
/// one that was missing. A whole generation going dark is what actually
/// happened: gemma-4 spent a merge refusing Metal by name while
/// `llama_like_metal` stated every width it needed, and this test
/// AGREED with the refusal because the roster had been edited to match
/// it. A roster compared against itself proves nothing. Requiring each
/// listed generation to keep at least one serving row is a claim about
/// the TEXT, and it is the claim that would have failed.
#[test]
fn the_rows_that_serve_metal_are_the_llama_like_ones() {
    // The TEN generations whose forward reaches `llama_like_metal`: the
    // seven that call the family projection directly, plus gemma-3,
    // gemma-4 and gpt-oss, whose own projections write their fields over
    // the family's and call the same text.
    let generations: [(&str, &[&'static dyn catalog::Variant]); 10] = [
        ("qwen_2", model::qwen_2::rows()),
        ("qwen_3", model::qwen_3::rows()),
        ("llama_3", model::llama_3::rows()),
        ("mistral_3", model::mistral_3::rows()),
        ("phi_3", model::phi_3::rows()),
        ("olmo_2", model::olmo_2::rows()),
        ("olmo_3", model::olmo_3::rows()),
        ("gemma_3", model::gemma_3::rows()),
        ("gemma_4", model::gemma_4::rows()),
        // The newest, and the one that took the most to get here: attention
        // sinks, a clamped SwiGLU, mxfp4 expert banks, three kinds of bias
        // the shared text did not state, and a YaRN ladder the driver's
        // geometry declined to derive.
        ("gpt_oss", model::gpt_oss::rows()),
    ];
    let expected: Vec<&'static str> = generations
        .iter()
        .flat_map(|(_, g)| g.iter().map(|r| r.id()))
        .collect();

    // ONE: nothing outside those generations reaches a Metal text.
    let mut rows = 0usize;
    for row in catalog::catalog() {
        rows += 1;
        let id = row.id();
        if row
            .trace(FireClass::Decode, Deployed::metal(&BINDING))
            .is_ok()
        {
            assert!(
                expected.contains(&id),
                "`{id}` serves Metal and belongs to no generation this test \
                 lists. A generation gaining a Metal text is a thing a reader \
                 should have to notice"
            );
        }
    }
    assert_the_catalog_was_walked(rows);
    assert!(
        expected.len() < rows,
        "every row in the catalog belongs to a llama-like generation, so this \
         test is comparing a set against itself"
    );

    // TWO: and none of them has gone dark. Per GENERATION, because a
    // row may refuse on its own measured grounds — a head width with no
    // kernel, an expert block with no text — while its generation is
    // served. A generation with no serving row at all is the gemma-4
    // regression, and nothing else here would have caught it.
    for (name, g) in &generations {
        assert!(
            g.iter().any(|r| r
                .trace(FireClass::Decode, Deployed::metal(&BINDING))
                .is_ok()),
            "no `{name}` row answers a Metal load, so this generation has \
             gone dark. Either its text was lost or every row of it refuses \
             for a reason worth reading; both are edits to this test, and \
             neither is silent"
        );
    }
}

/// No door serves a routed bank at a point the shader never stamped.
///
/// `llama_like_metal` is one text with three doors. Twelve generations
/// reach it through `shared::llama_like::project::trace`; `gemma_3` and
/// `gemma_4` reach it directly, because each writes its own fields over
/// the family's projection and so cannot use the shared `trace`.
///
/// The refusals guarding that text are about the KERNEL SET and not
/// about a row — `trace`'s own doc says so — and they lived at ONE
/// door. The other two carried the shard check alone. So the question
/// this asks is not "does each door hold a copy of the ladder" but the
/// property the ladder is for: can a routed bank reach
/// `affine_qmv_routed` at a group it was never stamped at, through any
/// door at all.
///
/// # What the measurement found, including what it falsified
///
/// It cannot, and the two doors stop it for DIFFERENT reasons — which
/// is worth writing down, because the obvious reading was wrong:
///
///   * The shared door refuses by [`NO_METAL_ROUTED_ENCODING`].
///     `qwen3-30b-a3b` and `qwen3-235b-a22b` are the rows that say so.
///   * gemma-4's door never gets the question. `gemma-4-26b-a4b` is
///     refused EARLIER and by its own text — this build cannot load
///     a routed gemma-4 block — so its only routed row stops before
///     any binding is consulted.
///
/// The first draft of this test asserted that gemma-4's door produced
/// the routed refusal, on the assumption that a routed gemma-4 reached
/// it. It does not, and the test said so. The refusal is still wired at
/// that door, where it will matter the day the routed text lands; what
/// is asserted here is only what is true today.
///
/// # Why `a_row_answers_the_same_way_at_every_encoding` does not cover it
///
/// That test is one-directional on purpose: it forbids a row REFUSED at
/// the probe and SERVED at a real encoding. This is the other
/// direction — served where it cannot run — and a permissive door is
/// invisible to a test that only looks for refusals that are too
/// strict.
#[test]
fn no_door_serves_a_routed_bank_at_an_unstamped_point() {
    // Group 128 is a point the routed matvec is not stamped at:
    // `AffineQ::group_size` is a template constant and the shader
    // compiles `affine_qmv_routed` at group 64 / 4 bits alone.
    let unstamped = MetalBinding {
        quant_group: 128,
        quant_bits: 4,
        router_quant_group: 0,
        router_quant_bits: 0,
        moe_mxfp4: false,
        ..BINDING
    };

    let mut rows = 0usize;
    let mut named_the_encoding: Vec<&'static str> = Vec::new();
    let mut routed_rows = 0usize;
    for row in catalog::catalog() {
        rows += 1;
        // A row is routed if its LOAD says so; this is the catalog's
        // own answer and not a guess from the row's name.
        let routed = row.load_shape().n_experts > 0;
        if !routed {
            continue;
        }
        routed_rows += 1;
        match row.trace(FireClass::Decode, Deployed::metal(&unstamped)) {
            Ok(_) => panic!(
                "`{}` is routed and SERVED at group {}, which the shader \
                 never stamped `affine_qmv_routed` at. Its expert bank would \
                 be dequantised with every scale read from the wrong offset",
                row.id(),
                unstamped.quant_group
            ),
            Err(Refusal::Unsupported(m)) if m.contains("routed matvec") => {
                named_the_encoding.push(row.id());
            }
            // Refused for some earlier reason of its own — no routed
            // text, no Metal text at all. Also safe, and the reason is
            // in the row's own words.
            Err(_) => {}
        }
    }
    assert_the_catalog_was_walked(rows);

    assert!(
        routed_rows > 0,
        "the catalog has no routed row, so this test asked nothing"
    );
    assert!(
        !named_the_encoding.is_empty(),
        "{routed_rows} routed row(s) were asked and NONE reached \
         `NO_METAL_ROUTED_ENCODING`. Every one of them is being stopped by \
         some earlier refusal, so the encoding check is unexercised and this \
         test would pass with it deleted"
    );
}

/// A REFUSAL before the bytes arrive must hold for every encoding.
///
/// `driver-metal`'s `serve/load.rs` asks "do you serve this row" BEFORE
/// any weights have arrived, through a probe binding — so a row that
/// refuses at the probe and would have been served at the encoding the
/// load turns out to have is turning away a checkpoint this build can
/// run. The catalog's counterpart to `driver-metal`'s own
/// `a_row_is_served_the_same_way_at_every_encoding`, asked from this
/// side of the boundary so that neither crate is the only place it
/// holds — including its DIRECTION, which is the part that matters.
///
/// # Why this is one-directional and was not
///
/// It asserted EQUALITY, on the premise that "whether a build has a
/// text is not a question about bytes". For the expert bank it is:
/// `quant/qmv.metal` instantiates `affine_qmv_routed` at group 64
/// alone, because `AffineQ::group_size` is a template constant, so a
/// routed row whose bank arrives at g128/b8 names a kernel no shader
/// defines. `qwen3-30b-a3b` is the row that says so, and
/// `shared::llama_like::project::NO_METAL_ROUTED_ENCODING` is the
/// sentence it says it with.
///
/// The permissive direction was never the premise. The probe exists so
/// 17 GB of gemma is not staged to reach an answer identification
/// already had, and a later refusal at the real encoding costs a load
/// rather than a wrong answer. What it may not do is refuse something
/// a real encoding would have served, and that is what this holds.
#[test]
fn a_row_answers_the_same_way_at_every_encoding() {
    let encodings = [
        BINDING,
        MetalBinding {
            quant_group: 32,
            quant_bits: 4,
            router_quant_group: 0,
            router_quant_bits: 0,
            moe_mxfp4: true,
            ..BINDING
        },
        MetalBinding {
            quant_group: 128,
            quant_bits: 8,
            router_quant_group: 0,
            router_quant_bits: 0,
            moe_mxfp4: false,
            ..BINDING
        },
        MetalBinding {
            fuse_residual_gemv: false,
            paged_multi_batch: false,
            qmm_multi_batch: false,
            ..BINDING
        },
    ];
    let mut rows = 0usize;
    for row in catalog::catalog() {
        rows += 1;
        let want = row
            .trace(FireClass::Decode, Deployed::metal(&BINDING))
            .is_ok();
        for b in &encodings {
            assert!(
                want || row.trace(FireClass::Decode, Deployed::metal(b)).is_err(),
                "`{}` is refused at the probe binding and served at g{}/b{} \
                 (bank mxfp4: {}), so the pre-staging refusal turns away a \
                 checkpoint this build can run",
                row.id(),
                b.quant_group,
                b.quant_bits,
                b.moe_mxfp4
            );
        }
    }
    assert_the_catalog_was_walked(rows);
}

/// An ENCODING is not a model fact, measured on the plan.
///
/// Two Metal loads of one row that differ ONLY in the bytes their
/// weights arrived as must state the same PROGRAM: the same operations
/// in the same order over the same values. What may differ is which
/// instantiation of a kernel each names.
///
/// This is the property the deleted `facts_from_with` could not have
/// had. It built the model's facts and the encoding's facts in ONE pass
/// over one `DecodeGeometry`, so nothing separated them and nothing
/// could check that they were separate — which is how a `norm_variant`
/// came to be decided by which norm tensors a checkpoint shipped.
#[test]
fn two_encodings_of_one_row_state_the_same_program() {
    let four = BINDING;
    let eight = MetalBinding {
        quant_group: 128,
        quant_bits: 8,
        router_quant_group: 0,
        router_quant_bits: 0,
        ..BINDING
    };
    let mut compared = 0usize;
    for row in catalog::catalog() {
        let (Ok(a), Ok(b)) = (
            row.trace(FireClass::Decode, Deployed::metal(&four)),
            row.trace(FireClass::Decode, Deployed::metal(&eight)),
        ) else {
            continue;
        };
        compared += 1;
        assert_eq!(
            a.family,
            b.family,
            "`{}`: an encoding changed the family",
            row.id()
        );
        assert_eq!(
            a.values,
            b.values,
            "`{}`: an encoding changed a value's shape or dtype, which makes it \
             a model fact — and a model fact belongs to the row",
            row.id()
        );
        assert_eq!(
            a.ops.len(),
            b.ops.len(),
            "`{}`: an encoding moved an op",
            row.id()
        );
        for (i, (x, y)) in a.ops.iter().zip(&b.ops).enumerate() {
            assert_eq!(x.layer, y.layer, "`{}`: op {i} moved layers", row.id());
            assert_eq!(
                x.inputs,
                y.inputs,
                "`{}`: op {i} reads different values",
                row.id()
            );
            assert_eq!(
                x.outputs,
                y.outputs,
                "`{}`: op {i} writes different values",
                row.id()
            );
        }
    }
    assert!(
        compared > 10,
        "only {compared} rows traced a Metal text at both encodings, so this \
         compared almost nothing"
    );
}

/// The CUDA half is untouched, which is the other half of the claim.
///
/// `Deployed::single()` states `Backend::Cuda`, so every caller written
/// before the backend existed reaches the arm it always reached. A
/// refactor that moved the Metal answer into the catalog and quietly
/// changed the CUDA one would be a worse outcome than the string table:
/// the string table at least only ever spoke for Metal.
#[test]
fn a_cuda_load_still_reaches_a_cuda_text() {
    assert!(matches!(Deployed::single().backend, Backend::Cuda));
    let mut rows = 0usize;
    for row in catalog::catalog() {
        rows += 1;
        for class in [FireClass::Prefill, FireClass::Decode] {
            let Ok(plan) = row.trace(class, Deployed::single()) else {
                // A row may refuse CUDA on its own grounds — `csm` has no
                // forward text at all, `kimi-k3`'s MLA output gate is
                // unstated, gemma-4's A4B rows want two texts this build
                // has not got. Those refusals predate the backend axis
                // and are not this file's subject.
                continue;
            };
            assert_eq!(
                TracedBackend::of_family(&plan.family),
                Some(TracedBackend::Cuda),
                "`{}` answered a CUDA load with `{}`",
                row.id(),
                plan.family
            );
        }
    }
    assert_the_catalog_was_walked(rows);
}
