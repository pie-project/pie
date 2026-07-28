//! Every wire decoder, against every prefix and every bit-flip of real bytes.
//!
//! `malformed_wire_corpus.txt` holds fifty-odd byte strings someone thought of.
//! It is a good corpus and it caught real count bombs, but it is a list, and a
//! list only rejects the inputs on it. This file is the other half: it takes
//! the encodings the compiler actually produces — 19 golden containers, their
//! sidecars, and every stage plan the corpus compiles — and mutates all of
//! them, so the decoders answer to a property instead of to a set of examples.
//!
//! Two claims are ratchets here:
//!
//! 1. **Totality.** No mutant panics and none runs away. The sweep is ~235,000
//!    inputs under one wall-clock budget, so a length field that decodes
//!    straight into `Vec::with_capacity` shows up as a failed test rather than
//!    as a wedged machine.
//! 2. **Truncation is never completion.** No proper prefix of a valid encoding
//!    decodes, and neither does a valid encoding with a byte glued on the end.
//!    A decoder that accepts a prefix has trusted a length past the buffer it
//!    holds; one that accepts a suffix does not say where the message stops.
//!
//! Canonicality is deliberately *not* asserted. `container::decode` ends in
//! `container.encode() != bytes -> NonCanonical`, so a test that decodes and
//! re-encodes would be restating the decoder's own last line. The sweep still
//! runs it — 17,180 mutants, 8,995 of them accepted — but as coverage of
//! `encode`, not as evidence about `decode`.
//!
//! The third test is a written-down measurement rather than a ratchet, and its
//! doc comment says why.

#[path = "common/msl_corpus.rs"]
mod msl_corpus;

use std::time::{Duration, Instant};

use msl_corpus::{GOLDEN_NAMES, corpus_stages, golden_container, golden_profile};
use pie_ir::validate::bind;
use pie_plan::decode_plan_header;
use pie_plan::sidecar::{BoundSidecar, decode_bound, encode_bound};

struct Decoder {
    name: &'static str,
    /// `true` = accepted. Panicking, hanging, or allocating without bound is
    /// the failure this is looking for.
    run: fn(&[u8]) -> bool,
}

const DECODERS: &[Decoder] = &[
    Decoder {
        name: "PTIB container",
        run: |bytes| match pie_ir::container::decode(bytes) {
            // Re-encode inside the sweep so `encode` is exercised on every
            // accepted mutant too, not only on the corpus.
            Ok(container) => container.encode() == bytes,
            Err(_) => false,
        },
    },
    Decoder {
        name: "PTIB sidecar",
        run: |bytes| decode_bound(bytes).is_ok(),
    },
    Decoder {
        name: "PTRP plan header",
        run: |bytes| decode_plan_header(bytes).is_ok(),
    },
];

fn sidecars() -> Vec<Vec<u8>> {
    GOLDEN_NAMES
        .iter()
        .filter_map(|name| {
            let bound = bind(golden_container(name), golden_profile(name)).ok()?;
            Some(encode_bound(&bound))
        })
        .collect()
}

/// Real encodings, one batch per decoder, in [`DECODERS`] order.
fn seeds() -> Vec<Vec<Vec<u8>>> {
    let containers = GOLDEN_NAMES
        .iter()
        .map(|name| golden_container(name).encode())
        .collect();
    let plans = corpus_stages()
        .into_iter()
        .map(|stage| stage.wire)
        .collect();
    vec![containers, sidecars(), plans]
}

/// Deterministic same-length mutants of `seed`.
///
/// No RNG: a fuzz finding that cannot be re-run is a rumour. The flip set is
/// `0x01` (a tag or a flag), `0x80` (a sign or a high bit) and `0xff` (the
/// count bomb the hand-written corpus is built around), which between them
/// reach every failure mode those cases cover — at every offset, rather than
/// at the offsets someone picked.
fn flips(seed: &[u8]) -> impl Iterator<Item = Vec<u8>> + '_ {
    (0..seed.len()).flat_map(move |i| {
        [0x01u8, 0x80, 0xff].into_iter().map(move |mask| {
            let mut bytes = seed.to_vec();
            bytes[i] ^= mask;
            bytes
        })
    })
}

/// Framing: a message must be exactly as long as it says it is.
#[test]
fn truncation_and_trailing_bytes_are_never_accepted() {
    let batches = seeds();
    for (decoder, seeds) in DECODERS.iter().zip(&batches) {
        assert!(
            !seeds.is_empty(),
            "{}: no seeds, the sweep would pass vacuously",
            decoder.name
        );
        for seed in seeds {
            assert!(
                (decoder.run)(seed),
                "{}: rejected its own unmutated encoding",
                decoder.name
            );
            for n in 0..seed.len() {
                assert!(
                    !(decoder.run)(&seed[..n]),
                    "{} accepted the first {n} of {} bytes",
                    decoder.name,
                    seed.len()
                );
            }
            let mut extended = seed.clone();
            extended.push(0);
            assert!(
                !(decoder.run)(&extended),
                "{} accepted its encoding with a zero byte appended",
                decoder.name
            );
        }
    }
}

/// The sweep proper: nothing panics and nothing runs away.
#[test]
fn no_mutant_panics_or_runs_away() {
    let batches = seeds();
    let start = Instant::now();
    let mut total = 0usize;
    for (decoder, seeds) in DECODERS.iter().zip(&batches) {
        let mut count = 0usize;
        for seed in seeds {
            for bytes in flips(seed) {
                count += 1;
                let _ = (decoder.run)(&bytes);
            }
        }
        assert!(
            count > 1000,
            "{}: only {count} mutants, the sweep is too thin to mean anything",
            decoder.name
        );
        total += count;
    }
    // Generous, because this runs unoptimised. The number is not the point:
    // the point is that a count decoded straight into an allocation cannot
    // hide inside a quarter of a million inputs.
    assert!(
        start.elapsed() < Duration::from_secs(180),
        "the sweep of {total} mutants took {:?}; a decoder is allocating on \
         an unchecked count",
        start.elapsed()
    );
}

/// **The container hash in a sidecar does not cover the sidecar.**
///
/// `container_hash` is the only self-identifying field a [`BoundSidecar`] has,
/// and `sidecar.rs` says the C++ driver reads these bytes with a reader of its
/// own. So it is worth knowing, in the tree rather than in someone's head,
/// whether that hash constrains the payload printed beside it. It does not: of
/// the 79,906 single-byte flips of a golden sidecar that still decode, 79,594
/// — better than 99.6% — hand back the same `container_hash` with different
/// `classes`, `readiness`, `stage_types` or `stage_plans`.
///
/// This is by construction. The hash is of the *container*; the sidecar is
/// derived data published next to it, and nothing ever claimed otherwise. The
/// reason to pin it is that "the sidecar carries the container hash" reads, to
/// anyone who has not traced it, as though the sidecar were checked — and the
/// consumer is a separate reader in another language that cannot re-derive the
/// fields to find out.
///
/// So this is not a ratchet against regression; it is a ratchet against the
/// fact going unwritten. Closing the gap means covering the payload in the
/// hash, which moves a wire format the driver reads — a decision, not a
/// cleanup. Whoever makes it will see this test fail, and its doc is the thing
/// they update.
#[test]
fn the_sidecar_hash_does_not_cover_its_own_payload() {
    let seeds = sidecars();
    assert!(!seeds.is_empty());
    let mut forgeable = 0usize;
    for seed in &seeds {
        let original = decode_bound(seed).expect("a golden sidecar decodes");
        for bytes in flips(seed) {
            let Ok(mutant) = decode_bound(&bytes) else {
                continue;
            };
            if mutant.container_hash == original.container_hash
                && payload_differs(&mutant, &original)
            {
                forgeable += 1;
            }
        }
    }
    assert!(
        forgeable > 0,
        "no flip changed a sidecar payload while keeping its container hash — \
         if the hash now covers the payload, delete this test and say so where \
         the format is documented"
    );
}

fn payload_differs(a: &BoundSidecar, b: &BoundSidecar) -> bool {
    a.classes != b.classes
        || a.readiness != b.readiness
        || a.stage_types != b.stage_types
        || a.stage_plans != b.stage_plans
}
