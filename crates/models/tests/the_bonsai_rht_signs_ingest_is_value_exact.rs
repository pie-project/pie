//! Ternary-Bonsai's RHT sign diagonals land in its contract value for value:
//! the package's GGUF format decodes the GGUF's `prism.hadamard.sign_{widths,
//! values}` metadata into three constant tensors, keyed by width, identical to
//! the PrismML llama.cpp fork's loaded sign arrays.
//!
//! The ground truth is the fork's own decode of the real
//! `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, frozen into a committed fixture so this
//! test is self-contained without the 6 GB file:
//!
//! * `bonsai/rht_signs.packed` — every one of the 28672 signs, bit-packed
//!   (bit=1 => +1, bit=0 => -1, LSB-first, the three widths concatenated in
//!   stored order 5120, 6144, 17408; ceil(28672/8)=3584 bytes). This is the
//!   value-for-value oracle.
//! * The `EXPECTED_*` summary below (per-width +1/-1 counts and first/last 16),
//!   a human-legible restatement checked against the packed bytes.
//!
//! A GGUF stating the oracle's table (and naming, without holding, every
//! tensor the import reads) is read through the real ztensor GGUF reader and
//! the package's format, and each sign constant the contract carries must equal
//! the oracle. A table the Bonsai rotation cannot use is refused.

mod sparse_gguf;

use std::collections::BTreeMap;

use poem::Dtype;
use sparse_gguf::{Kv, Scratch, reads};

const WIDTH_HIDDEN: u32 = 5120;
const WIDTH_SSM_OUT: u32 = 6144;
const WIDTH_FFN_DOWN: u32 = 17408;
const BONSAI_SIGN_WIDTHS: [u32; 3] = [WIDTH_HIDDEN, WIDTH_SSM_OUT, WIDTH_FFN_DOWN];

/// The value-for-value oracle: the fork's 28672 loaded signs, bit-packed.
const PACKED: &[u8] = include_bytes!("bonsai/rht_signs.packed");
const TOTAL: usize = 28672;

/// Per-width summary frozen from the fork decode (see `bonsai/rht_signs.json`).
/// `(width, count_pos, count_neg, first16, last16)`.
struct Expected {
    width: u32,
    pos: usize,
    neg: usize,
    first16: [i8; 16],
    last16: [i8; 16],
}

const EXPECTED: [Expected; 3] = [
    Expected {
        width: WIDTH_HIDDEN,
        pos: 2481,
        neg: 2639,
        first16: [-1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
        last16: [-1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1],
    },
    Expected {
        width: WIDTH_SSM_OUT,
        pos: 3032,
        neg: 3112,
        first16: [-1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1],
        last16: [1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1],
    },
    Expected {
        width: WIDTH_FFN_DOWN,
        pos: 8655,
        neg: 8753,
        first16: [1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1],
        last16: [-1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1],
    },
];

/// Unpack the committed oracle into the three `±1` vectors, keyed by width, in
/// the stored width order.
fn oracle() -> BTreeMap<u32, Vec<i8>> {
    assert_eq!(
        PACKED.len(),
        TOTAL.div_ceil(8),
        "packed fixture is 3584 bytes"
    );
    let mut bits = Vec::with_capacity(TOTAL);
    for i in 0..TOTAL {
        let bit = (PACKED[i / 8] >> (i % 8)) & 1;
        bits.push(if bit == 1 { 1i8 } else { -1i8 });
    }
    let mut out = BTreeMap::new();
    let mut off = 0usize;
    for w in BONSAI_SIGN_WIDTHS {
        let span = w as usize;
        out.insert(w, bits[off..off + span].to_vec());
        off += span;
    }
    assert_eq!(off, TOTAL, "widths tile the oracle exactly");
    out
}

fn check_summary(width: u32, vec: &[i8]) {
    let e = EXPECTED
        .iter()
        .find(|e| e.width == width)
        .unwrap_or_else(|| panic!("no expected summary for width {width}"));
    assert_eq!(vec.len(), width as usize, "width {width} length");
    assert!(
        vec.iter().all(|&s| s == 1 || s == -1),
        "width {width} is ±1"
    );
    let pos = vec.iter().filter(|&&s| s == 1).count();
    let neg = vec.iter().filter(|&&s| s == -1).count();
    assert_eq!((pos, neg), (e.pos, e.neg), "width {width} +1/-1 counts");
    assert_eq!(&vec[..16], &e.first16, "width {width} first 16");
    assert_eq!(
        &vec[width as usize - 16..],
        &e.last16,
        "width {width} last 16"
    );
}

#[test]
fn the_committed_oracle_is_self_consistent() {
    let signs = oracle();
    assert_eq!(
        signs.keys().copied().collect::<Vec<_>>(),
        vec![WIDTH_HIDDEN, WIDTH_SSM_OUT, WIDTH_FFN_DOWN],
    );
    for (w, vec) in &signs {
        check_summary(*w, vec);
    }
}

/// The Bonsai GGUF metadata stating the sign table `widths` / `values` over
/// Hadamard blocks of `block`.
fn table(block: u32, widths: &[u32], values: &[i8]) -> Vec<(&'static str, Kv)> {
    vec![
        ("general.architecture", Kv::Str("qwen35".into())),
        ("prism.hadamard.gdn_v_grouped", Kv::Bool(true)),
        ("prism.hadamard.sign_mode", Kv::Str("explicit".into())),
        ("prism.hadamard.block_size", Kv::U32(block)),
        (
            "prism.hadamard.sign_widths",
            Kv::I32s(widths.iter().map(|&w| w as i32).collect()),
        ),
        (
            "prism.hadamard.sign_values",
            Kv::I32s(values.iter().map(|&v| i32::from(v)).collect()),
        ),
    ]
}

/// The `±1` values of the bf16 constant the recorded read of `weight` lands.
fn landed(log: &[String], weight: &str) -> Vec<i8> {
    let read = log
        .iter()
        .find(|r| r.contains(&format!("name: \"{weight}\"")))
        .unwrap_or_else(|| panic!("no read of `{weight}`"));
    let bytes = &read[read.find("bytes: [").expect("a constant's bytes") + 8..];
    let bytes: Vec<u8> = bytes[..bytes.find(']').unwrap()]
        .split(", ")
        .map(|b| b.parse().unwrap())
        .collect();
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| match u16::from_le_bytes(*b) {
            0x3F80 => 1,
            0xBF80 => -1,
            other => panic!("`{weight}` holds {other:#06x}, not a bf16 ±1"),
        })
        .collect()
}

#[test]
fn the_contract_carries_the_forks_signs_value_for_value() {
    let oracle = oracle();
    let values: Vec<i8> = BONSAI_SIGN_WIDTHS
        .iter()
        .flat_map(|w| oracle[w].iter().copied())
        .collect();
    let scratch = Scratch::new("bonsai-signs");
    let log = reads(
        &scratch.0,
        "qwen36-27b-bonsai",
        Dtype::Ptq1_0,
        &table(1024, &BONSAI_SIGN_WIDTHS, &values),
    );
    let mut checked = 0;
    for width in BONSAI_SIGN_WIDTHS {
        let signs = landed(&log, &format!("prism.hadamard.signs.{width}"));
        assert_eq!(
            signs, oracle[&width],
            "width {width}: signs differ from the fork oracle"
        );
        check_summary(width, &signs);
        checked += signs.len();
    }
    assert_eq!(checked, TOTAL, "all 28672 signs checked");
}

#[test]
fn a_table_off_the_1024_block_is_refused() {
    let oracle = oracle();
    let values: Vec<i8> = BONSAI_SIGN_WIDTHS
        .iter()
        .flat_map(|w| oracle[w].iter().copied())
        .collect();
    let scratch = Scratch::new("bonsai-signs-block");
    let refused = std::panic::catch_unwind(|| {
        reads(
            &scratch.0,
            "qwen36-27b-bonsai",
            Dtype::Ptq1_0,
            &table(512, &BONSAI_SIGN_WIDTHS, &values),
        )
    });
    assert!(
        refused.is_err(),
        "a 512-wide Hadamard block is not the Bonsai rotation's"
    );
}
