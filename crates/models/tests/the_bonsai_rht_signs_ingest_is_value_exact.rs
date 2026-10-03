//! M3b: pie ingests the Bonsai RHT sign diagonals from the GGUF's
//! `prism.hadamard.sign_{widths,values}` metadata and exposes them as registered
//! params keyed by width — value-for-value identical to the PrismML llama.cpp
//! fork's loaded sign arrays.
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
//! Two layers of check:
//!   1. Always (CI, no GGUF, no Metal): a synthetic in-memory GGUF carrying a
//!      known sign table is parsed through the REAL ztensor GGUF reader and
//!      decoded, proving the file -> attributes -> `decode_signs` plumbing; and
//!      the committed packed fixture is asserted self-consistent.
//!   2. Guarded (`BONSAI_GGUF` points at the real file): pie decodes the real
//!      metadata and every one of the 28672 signs must equal the packed oracle,
//!      with matching per-width +1/-1 counts and endpoints.

use std::collections::BTreeMap;

use models::qwen_3::rotation::{
    self, BONSAI_SIGN_WIDTHS, SignVector, WIDTH_FFN_DOWN, WIDTH_HIDDEN, WIDTH_SSM_OUT,
};

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

/// A GGUF v3 metadata value: only the shapes this fixture needs.
enum Kv {
    U32(u32),
    Str(&'static str),
    /// `int32` array (ggml metadata elem type 5).
    I32Array(Vec<i32>),
}

/// Emit a minimal GGUF v3 file (zero tensors) carrying the given metadata KV, so
/// the real `ztensor` GGUF reader parses it and `decode_signs` runs over genuine
/// `Source::attributes()`.
fn gguf_with_kv(kvs: &[(&str, Kv)]) -> Vec<u8> {
    const ALIGN: usize = 32;
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
    out.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (key, value) in kvs {
        out.extend_from_slice(&(key.len() as u64).to_le_bytes());
        out.extend_from_slice(key.as_bytes());
        match value {
            Kv::U32(n) => {
                out.extend_from_slice(&4u32.to_le_bytes()); // type: uint32
                out.extend_from_slice(&n.to_le_bytes());
            }
            Kv::Str(s) => {
                out.extend_from_slice(&8u32.to_le_bytes()); // type: string
                out.extend_from_slice(&(s.len() as u64).to_le_bytes());
                out.extend_from_slice(s.as_bytes());
            }
            Kv::I32Array(xs) => {
                out.extend_from_slice(&9u32.to_le_bytes()); // type: array
                out.extend_from_slice(&5u32.to_le_bytes()); // elem type: int32
                out.extend_from_slice(&(xs.len() as u64).to_le_bytes());
                for x in xs {
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
    }
    while !out.len().is_multiple_of(ALIGN) {
        out.push(0);
    }
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

#[test]
fn a_real_gguf_reader_parses_and_decodes_a_synthetic_sign_table() {
    // block 4; width 4 = [+1,-1,-1,+1], width 8 = [-1,-1,+1,+1,-1,+1,+1,-1].
    let bytes = gguf_with_kv(&[
        ("prism.hadamard.sign_mode", Kv::Str("explicit")),
        ("prism.hadamard.block_size", Kv::U32(4)),
        ("prism.hadamard.sign_widths", Kv::I32Array(vec![4, 8])),
        (
            "prism.hadamard.sign_values",
            Kv::I32Array(vec![1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1]),
        ),
    ]);
    let dir = std::env::temp_dir().join(format!("m3b_signs_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("synthetic.gguf");
    std::fs::write(&path, &bytes).unwrap();

    let src = ztensor_compat::index(&path).expect("the real gguf reader opens it");
    // The generic decoder handles any positive block (small here for readability);
    // this exercises the file -> reader -> attributes -> decode_signs plumbing.
    let signs = rotation::decode_signs(src.attributes()).expect("decode over Source::attributes");

    assert_eq!(signs.len(), 2);
    assert_eq!(signs[&4].signs, vec![1, -1, -1, 1]);
    assert_eq!(signs[&8].signs, vec![-1, -1, 1, 1, -1, 1, 1, -1]);
    // The registered params mirror the fork's on-device tensor names.
    let params = rotation::sign_params(&signs);
    let names: Vec<_> = params.iter().map(|w| w.name.clone()).collect();
    assert_eq!(
        names,
        vec!["prism.hadamard.signs.4", "prism.hadamard.signs.8"]
    );

    // The Bonsai ingest wrapper additionally pins explicit-mode signs to the
    // canonical 1024-wide Hadamard block, so this non-1024 synthetic table is
    // rejected there even though the generic decoder accepts it above.
    assert!(
        rotation::signs_from_gguf(&src).is_err(),
        "signs_from_gguf rejects a non-1024 block size"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_real_bonsai_gguf_signs_match_the_fork_value_for_value() {
    let Some(path) = std::env::var_os("BONSAI_GGUF") else {
        eprintln!(
            "m3b: BONSAI_GGUF unset; skipping the real-file value-for-value check \
             (the committed oracle is still asserted by the other tests)"
        );
        return;
    };
    if !std::path::Path::new(&path).exists() {
        eprintln!("m3b: BONSAI_GGUF={path:?} does not exist; skipping");
        return;
    }

    let src = ztensor_compat::index(&path).expect("open the real Bonsai GGUF");
    let signs = rotation::signs_from_gguf(&src).expect("decode the real sign metadata");
    let oracle = oracle();

    // Same widths, same order.
    assert_eq!(
        signs.keys().copied().collect::<Vec<_>>(),
        vec![WIDTH_HIDDEN, WIDTH_SSM_OUT, WIDTH_FFN_DOWN],
        "the three Bonsai sign widths",
    );

    let mut checked = 0usize;
    for (w, sv) in &signs {
        let want = &oracle[w];
        // Value-for-value: every sign equals the fork's loaded array.
        assert_eq!(
            &sv.signs, want,
            "width {w}: signs differ from the fork oracle"
        );
        check_summary(*w, &sv.signs);
        // The f32 view the Hadamard op consumes is exact ±1.
        assert_eq!(sv.to_f32().len(), *w as usize);
        checked += sv.signs.len();
    }
    assert_eq!(checked, TOTAL, "all 28672 signs checked");

    // Sanity: the decoded map is exactly what the width-keyed vectors say.
    let rebuilt: BTreeMap<u32, Vec<i8>> =
        signs.iter().map(|(w, sv)| (*w, sv.signs.clone())).collect();
    assert_eq!(rebuilt, oracle, "the whole decoded table equals the oracle");
    let _: &SignVector = &signs[&WIDTH_HIDDEN];
    eprintln!("m3b: {checked} signs across 3 widths match the fork value-for-value");
}
