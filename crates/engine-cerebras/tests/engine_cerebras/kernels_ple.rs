//! Per-layer-embedding n-gram hashing over lanes with a kept window.

use engine_cerebras::bench::Bench;
use kernels_cerebras::attn::ple;
use kernels_cerebras::tensor::{RaggedTensor, RecurrentPool};

const EOS: i32 = 1;
const MULTS: [u64; 3] = [
    0x9E37_79B9_7F4A_7C15,
    0xC2B2_AE3D_27D4_EB4F,
    0x1656_67B1_9E37_79F9,
];
const PRIMES: [u64; 4] = [1_000_003, 999_983, 7_919, 104_729];
const OFFSETS: [u64; 4] = [0, 1_000_003, 2_000_000, 2_010_000];

/// Hashes one row's window `[id, id-1, id-2]`: two orders of two heads.
fn hash_window(window: &[i32]) -> Vec<i32> {
    let mut out = Vec::new();
    let mut mixed = 0u64;
    for (p, w) in window.iter().enumerate() {
        mixed ^= (i64::from(*w) as u64).wrapping_mul(MULTS[p]);
        if p == 0 {
            continue;
        }
        for i in 0..2 {
            let at = (p - 1) * 2 + i;
            let r = (mixed % PRIMES[at]).wrapping_add(OFFSETS[at]);
            out.push((r & 0xFFFF_FFFF) as u32 as i32);
        }
    }
    out
}

/// Two lanes of a chunked fire read their kept windows first, an eos cuts
/// the context, and each lane's last two ids land as its next window.
#[test]
fn chunked_ngram_ids_hash_each_rows_window_and_keep_the_tail() {
    let ids = [5i32, 6, EOS, 8, 9, 3];
    let indptr = [0i32, 4, 6];
    // Slot 2 holds [nothing, 4]; slot 0 holds [7, 2]; cells are id + 1.
    let slab = [8i32, 3, 0, 0, 0, 5];
    let slot_of_row = [2i32, 2, 2, 2, 0, 0];
    let mut b = Bench::new();
    let idt = b.i32(6, 1, &ids);
    let ip = b.i32(3, 1, &indptr);
    let state = b.i32(3, 2, &slab);
    let slots = b.i32(6, 1, &slot_of_row);
    let out = b.zeros(dtype::Dtype::I32, 6, 4);
    let pool = RecurrentPool {
        state,
        slots,
        conv_state: state,
        new_conv_state: state,
    };
    let ran = b
        .run(|ctx| {
            ple::ngram_ids_chunked(
                ctx,
                RaggedTensor {
                    data: idt,
                    indptr: ip,
                },
                &pool,
                EOS as u32,
                &MULTS,
                &PRIMES,
                &OFFSETS,
                2,
                None,
                out,
            )
        })
        .unwrap();
    if !ran {
        return;
    }
    let windows = [
        [5, 4, EOS], // kept cell 4, then nothing (eos)
        [6, 5, 4],
        [EOS, 6, 5],
        [8, EOS, EOS], // past the eos every older id reads as eos
        [9, 2, 7],     // lane 2's kept window [7, 2]
        [3, 9, 2],
    ];
    let want: Vec<i32> = windows.iter().flat_map(|w| hash_window(w)).collect();
    assert_eq!(b.read_i32(out), want);
    // Slot 2 keeps [EOS, 8] + 1, slot 0 keeps [9, 3] + 1, slot 1 untouched.
    assert_eq!(b.read_i32(state), vec![10, 4, 0, 0, EOS + 1, 9]);
}

/// One row per lane: the row hashes against its slot's window and pushes
/// its id in.
#[test]
fn decode_ngram_ids_hash_against_the_kept_window() {
    let ids = [5i32, 9];
    let slab = [8i32, 3, 0, 0, 0, 5];
    let slot_of_row = [2i32, 0];
    let mut b = Bench::new();
    let idt = b.i32(2, 1, &ids);
    let state = b.i32(3, 2, &slab);
    let slots = b.i32(2, 1, &slot_of_row);
    let out = b.zeros(dtype::Dtype::I32, 2, 4);
    let pool = RecurrentPool {
        state,
        slots,
        conv_state: state,
        new_conv_state: state,
    };
    let ran = b
        .run(|ctx| {
            ple::ngram_ids(
                ctx, idt, &pool, EOS as u32, &MULTS, &PRIMES, &OFFSETS, 2, None, out,
            )
        })
        .unwrap();
    if !ran {
        return;
    }
    let want: Vec<i32> = [[5, 4, EOS], [9, 2, 7]]
        .iter()
        .flat_map(|w| hash_window(w))
        .collect();
    assert_eq!(b.read_i32(out), want);
    assert_eq!(b.read_i32(state), vec![3, 10, 0, 0, 5, 6]);
}
