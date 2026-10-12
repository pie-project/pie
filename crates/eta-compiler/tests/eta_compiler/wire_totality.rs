use crate::msl_corpus::{GOLDEN_NAMES, extended_traces, golden_container};

struct Sweep {
    name: &'static str,
    run: fn(&[u8]) -> bool,
    seeds: fn() -> Vec<Vec<u8>>,
}

const SWEEPS: &[Sweep] = &[Sweep {
    name: "ETA container",
    run: |bytes| match eta_ir::container::decode(bytes) {
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
