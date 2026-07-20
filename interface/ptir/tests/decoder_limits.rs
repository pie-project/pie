use pie_ptir::compiler::decode_plan_header;
use pie_ptir::sidecar::decode_bound;
use std::time::{Duration, Instant};

#[derive(Clone)]
struct Case {
    kind: &'static str,
    accept: bool,
    name: &'static str,
    bytes: Vec<u8>,
}

fn corpus() -> Vec<Case> {
    include_str!("malformed_wire_corpus.txt")
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let mut fields = line.split_whitespace();
            let kind = fields.next().unwrap();
            let expected = fields.next().unwrap();
            let name = fields.next().unwrap();
            let hex = fields.next().unwrap();
            assert!(fields.next().is_none(), "extra corpus field in {name}");
            assert_eq!(hex.len() % 2, 0, "odd hex length in {name}");
            let bytes = hex
                .as_bytes()
                .chunks_exact(2)
                .map(|pair| u8::from_str_radix(core::str::from_utf8(pair).unwrap(), 16).unwrap())
                .collect();
            Case {
                kind,
                accept: expected == "accept",
                name,
                bytes,
            }
        })
        .collect()
}

fn accepted(case: &Case) -> bool {
    match case.kind {
        "PTIB" => decode_bound(&case.bytes).is_ok(),
        "PTRP" => decode_plan_header(&case.bytes).is_ok(),
        other => panic!("unknown corpus kind {other}"),
    }
}

#[test]
fn rust_decoder_matches_shared_resource_limit_corpus() {
    let cases = corpus();
    let short_bombs = cases
        .iter()
        .filter(|case| {
            !case.accept
                && (20..=40).contains(&case.bytes.len())
                && case.bytes.windows(4).any(|word| word == [0xff; 4])
        })
        .count();
    assert!(
        short_bombs >= 12,
        "corpus must retain adversarial 20-40 byte count bombs"
    );
    for case in &cases {
        assert_eq!(
            accepted(case),
            case.accept,
            "{} {} acceptance mismatch",
            case.kind,
            case.name
        );
    }
}

#[test]
fn malformed_corpus_rejection_is_bounded() {
    let rejected: Vec<_> = corpus().into_iter().filter(|case| !case.accept).collect();
    let start = Instant::now();
    for _ in 0..4096 {
        for case in &rejected {
            assert!(!accepted(case), "{} unexpectedly accepted", case.name);
        }
    }
    assert!(
        start.elapsed() < Duration::from_secs(2),
        "malformed corpus rejection exceeded the CPU budget"
    );
}
