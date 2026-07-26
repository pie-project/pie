//! The container decoder, against every prefix and every bit-flip of real bytes.
//!
//! A hand-written corpus of malformed inputs catches real count bombs, but a
//! list only rejects the inputs on it — and it ages into a description of
//! formats nobody parses any more. This file is the other half: it takes the
//! encodings the compiler actually produces and mutates all of them, so the
//! decoder answers to a property instead of to a set of examples.
//!
//! The seeds are every container the compiler can build — the golden corpus and
//! the extended corpus both. That is the whole live surface: `PTIR` is the only
//! byte format anything outside this directory parses
//! (`runtime/engine/src/pipeline/program.rs` on program registration,
//! `pipeline/fire/kv.rs`, `driver/dummy`). Nothing else belongs in this sweep:
//! fuzzing a format that only this crate writes and only this crate reads is
//! fuzzing a round-trip, and it will pass no matter what either side does.
//!
//! Two claims are ratchets here:
//!
//! 1. **Totality.** No mutant panics and none runs away. The sweep runs under
//!    one wall-clock budget, so a length field that decodes straight into
//!    `Vec::with_capacity` shows up as a failed test rather than as a wedged
//!    machine.
//! 2. **Truncation is never completion.** No proper prefix of a valid encoding
//!    decodes, and neither does a valid encoding with a byte glued on the end.
//!    A decoder that accepts a prefix has trusted a length past the buffer it
//!    holds; one that accepts a suffix does not say where the message stops.
//!
//! Canonicality is deliberately *not* asserted. `container::decode` ends in
//! `container.encode() != bytes -> NonCanonical`, so a test that decodes and
//! re-encodes would be restating the decoder's own last line. The sweep still
//! runs it, but as coverage of `encode`, not as evidence about `decode`.

#[path = "common/msl_corpus.rs"]
mod msl_corpus;

use std::time::{Duration, Instant};

use msl_corpus::{GOLDEN_NAMES, extended_traces, golden_container};

/// A decoder and the real encodings it is swept over.
///
/// Seeds live beside the decoder rather than in a parallel array: the two used
/// to be zipped together by position, which silently drops a decoder if the
/// arrays fall out of step.
struct Sweep {
    name: &'static str,
    /// `true` = accepted. Panicking, hanging, or allocating without bound is
    /// the failure this is looking for.
    run: fn(&[u8]) -> bool,
    seeds: fn() -> Vec<Vec<u8>>,
}

const SWEEPS: &[Sweep] = &[Sweep {
    name: "PTIR container",
    run: |bytes| match pie_ir::container::decode(bytes) {
        // Re-encode inside the sweep so `encode` is exercised on every
        // accepted mutant too, not only on the corpus.
        Ok(container) => container.encode() == bytes,
        Err(_) => false,
    },
    seeds: containers,
}];

fn containers() -> Vec<Vec<u8>> {
    GOLDEN_NAMES
        .iter()
        .map(|name| golden_container(name).encode())
        .chain(
            extended_traces()
                .into_iter()
                .map(|(_, container, _)| container.encode()),
        )
        .collect()
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
    for sweep in SWEEPS {
        let seeds = (sweep.seeds)();
        assert!(
            !seeds.is_empty(),
            "{}: no seeds, the sweep would pass vacuously",
            sweep.name
        );
        for seed in &seeds {
            assert!(
                (sweep.run)(seed),
                "{}: rejected its own unmutated encoding",
                sweep.name
            );
            for n in 0..seed.len() {
                assert!(
                    !(sweep.run)(&seed[..n]),
                    "{} accepted the first {n} of {} bytes",
                    sweep.name,
                    seed.len()
                );
            }
            let mut extended = seed.clone();
            extended.push(0);
            assert!(
                !(sweep.run)(&extended),
                "{} accepted its encoding with a zero byte appended",
                sweep.name
            );
        }
    }
}

/// The sweep proper: nothing panics and nothing runs away.
#[test]
fn no_mutant_panics_or_runs_away() {
    let start = Instant::now();
    let mut total = 0usize;
    for sweep in SWEEPS {
        let mut count = 0usize;
        for seed in &(sweep.seeds)() {
            for bytes in flips(seed) {
                count += 1;
                let _ = (sweep.run)(&bytes);
            }
        }
        assert!(
            count > 1000,
            "{}: only {count} mutants, the sweep is too thin to mean anything",
            sweep.name
        );
        total += count;
    }
    // Generous, because this runs unoptimised. The number is not the point:
    // the point is that a count decoded straight into an allocation cannot
    // hide inside a sweep this size.
    assert!(
        start.elapsed() < Duration::from_secs(180),
        "the sweep of {total} mutants took {:?}; a decoder is allocating on \
         an unchecked count",
        start.elapsed()
    );
}
