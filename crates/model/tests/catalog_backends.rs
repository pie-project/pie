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
//! both in the same way. It listed `gemma4`, which the Metal load path
//! refused on entirely separate grounds, so the answer to "served?"
//! depended on which of the two you asked. It omitted `gemma3`, whose
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
#![cfg(feature = "forward")]

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
    moe_mxfp4: false,
    fuse_residual_gemv: true,
    paged_multi_batch: true,
    qmm_multi_batch: true,
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

/// The rows that serve Metal are exactly the rows whose text is the
/// family's, and they are named.
///
/// A list, in a test, which is the thing this refactor deleted from a
/// driver — and the difference is where it lives and what it is
/// checked against. `LLAMA_LIKE` was consulted to DECIDE the answer,
/// in the driver, from a string that reached it through a config file.
/// This is derived from the answers the rows give and compared against
/// the GENERATIONS a reader was told serve Metal; if a generation gains
/// a Metal text, this fails and is edited, which is the point of
/// writing it down.
///
/// The expected set is built from each generation's own `rows()` rather
/// than from a list of row ids or a prefix match on them. Both of those
/// were tried and both were wrong: `ministral-8b` is a `mistral_3` row
/// whose id does not begin with `mistral`, and `embeddinggemma-300m` is
/// a `gemma_3` row whose id names no gemma-3 at all. A row id is a
/// NAME, and deciding anything from the shape of a name is the habit
/// this whole refactor is unwinding.
#[test]
fn the_rows_that_serve_metal_are_the_llama_like_ones() {
    // The eight generations whose forward IS `llama_like`: the seven
    // that call the family projection directly, plus gemma-3, whose own
    // projection writes gemma's seven fields over the family's and calls
    // the same text.
    let expected: Vec<&'static str> = [
        model::qwen_2::rows(),
        model::qwen_3::rows(),
        model::llama_3::rows(),
        model::mistral_3::rows(),
        model::phi_3::rows(),
        model::olmo_2::rows(),
        model::olmo_3::rows(),
        model::gemma_3::rows(),
    ]
    .iter()
    .flat_map(|g| g.iter().map(|r| r.id()))
    .collect();

    let mut rows = 0usize;
    for row in catalog::catalog() {
        rows += 1;
        let id = row.id();
        let serves = row.trace(FireClass::Decode, Deployed::metal(&BINDING)).is_ok();
        assert_eq!(
            serves,
            expected.contains(&id),
            "`{id}` {} Metal, and the generations listed in this test say \
             otherwise. A generation gaining or losing a Metal text is a thing \
             a reader should have to notice",
            if serves { "serves" } else { "refuses" }
        );
    }
    assert_the_catalog_was_walked(rows);
    assert!(
        expected.len() < rows,
        "every row in the catalog belongs to a llama-like generation, so this \
         test is comparing a set against itself"
    );
}

/// WHETHER a build has a text is not a question about bytes.
///
/// `driver-metal`'s `serve/load.rs` asks "do you serve this row" BEFORE
/// any weights have arrived, through a probe binding — so a row whose
/// answer moved with a group size, a bit width or the expert bank's
/// format would be answering a question it cannot yet ask, and the
/// pre-staging refusal would be a guess. The catalog's counterpart to
/// `driver-metal`'s own `a_row_is_served_the_same_way_at_every_encoding`,
/// asked from this side of the boundary so that neither crate is the
/// only place it holds.
#[test]
fn a_row_answers_the_same_way_at_every_encoding() {
    let encodings = [
        BINDING,
        MetalBinding { quant_group: 32, quant_bits: 4, moe_mxfp4: true, ..BINDING },
        MetalBinding { quant_group: 128, quant_bits: 8, moe_mxfp4: false, ..BINDING },
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
        let want = row.trace(FireClass::Decode, Deployed::metal(&BINDING)).is_ok();
        for b in &encodings {
            assert_eq!(
                row.trace(FireClass::Decode, Deployed::metal(b)).is_ok(),
                want,
                "`{}` answers differently at g{}/b{} (bank mxfp4: {}), so \
                 whether this build HAS a text for it depends on how its bytes \
                 arrived — and the load path asks before any have",
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
    let eight = MetalBinding { quant_group: 128, quant_bits: 8, ..BINDING };
    let mut compared = 0usize;
    for row in catalog::catalog() {
        let (Ok(a), Ok(b)) = (
            row.trace(FireClass::Decode, Deployed::metal(&four)),
            row.trace(FireClass::Decode, Deployed::metal(&eight)),
        ) else {
            continue;
        };
        compared += 1;
        assert_eq!(a.family, b.family, "`{}`: an encoding changed the family", row.id());
        assert_eq!(
            a.values,
            b.values,
            "`{}`: an encoding changed a value's shape or dtype, which makes it \
             a model fact — and a model fact belongs to the row",
            row.id()
        );
        assert_eq!(a.ops.len(), b.ops.len(), "`{}`: an encoding moved an op", row.id());
        for (i, (x, y)) in a.ops.iter().zip(&b.ops).enumerate() {
            assert_eq!(x.layer, y.layer, "`{}`: op {i} moved layers", row.id());
            assert_eq!(x.inputs, y.inputs, "`{}`: op {i} reads different values", row.id());
            assert_eq!(x.outputs, y.outputs, "`{}`: op {i} writes different values", row.id());
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
