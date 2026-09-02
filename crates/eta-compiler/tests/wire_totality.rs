//! Fuzzes the container decoder over real encodings and their byte-flip
//! mutants: checks no mutant panics or runs away, and that no proper prefix
//! or extended suffix of a valid encoding ever decodes.

#[path = "common/msl_corpus.rs"]
mod msl_corpus;


use msl_corpus::{GOLDEN_NAMES, extended_traces, golden_container};

/// A decoder and the real encodings it is swept over.
struct Sweep {
    name: &'static str,
    /// `true` = accepted.
    run: fn(&[u8]) -> bool,
    seeds: fn() -> Vec<Vec<u8>>,
}

const SWEEPS: &[Sweep] = &[Sweep {
    name: "ETA container",
    run: |bytes| match eta_ir::container::decode(bytes) {
        // Re-encode so `encode` is exercised on every accepted mutant too.
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

