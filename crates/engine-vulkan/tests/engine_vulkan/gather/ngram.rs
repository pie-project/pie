//! The n-gram hashers on the first Vulkan device, beside the hand-written kernels,
//! against a host reference in wrapping u64: the decode form (a token a
//! lane) and the chunked form (a request's run a lane), with an
//! end-of-sequence id cutting a window and an empty request; the ids and the
//! rolled recurrent state are compared exactly.

use kernels_vulkan::attn::ple;
use kernels_vulkan::tensor::RecurrentPool;
use kernels_vulkan::{RaggedTensor, Tensor};
use poem_ir::Dtype;

use crate::gpu::Gpu;

const EOS: i32 = 2;
const MULTS: [u64; 3] = [
    0x9e37_79b9_7f4a_7c15,
    0xbf58_476d_1ce4_e5b9,
    0x94d0_49bb_1331_11eb,
];
const PRIMES: [u64; 4] = [1_000_003, 999_983, 4_000_037, 3_999_971];
const OFFSETS: [u64; 4] = [0, 1_000_003, 2_000_000, 6_000_037];
const HEADS_PER: usize = 2;
const SPAN: usize = MULTS.len() - 1;
const HEADS: usize = PRIMES.len();

fn cell(held: i32) -> i32 {
    if held == 0 { EOS } else { held - 1 }
}

fn hash(mut window: [i32; 3]) -> [i32; HEADS] {
    let mut crossed = false;
    for id in &mut window[1..] {
        if crossed {
            *id = EOS;
        }
        crossed |= *id == EOS;
    }
    let mut out = [0; HEADS];
    for order in 2..=MULTS.len() {
        let mixed = (0..order).fold(0u64, |m, p| {
            m ^ (i64::from(window[p]) as u64).wrapping_mul(MULTS[p])
        });
        for k in 0..HEADS_PER {
            let head = (order - 2) * HEADS_PER + k;
            out[head] = (mixed % PRIMES[head] + OFFSETS[head]) as i32;
        }
    }
    out
}

fn ints(gpu: &Gpu, buf: u32, n: usize) -> Vec<i32> {
    gpu.down_f32(buf, n)
        .iter()
        .map(|v| v.to_bits() as i32)
        .collect()
}

#[test]
fn the_ngram_hashers_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let st = |t: Tensor| kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype);
    let t = |buf: u32, r: usize, w: usize, d: Dtype| Tensor::new(buf, r as u32, w as u32, d);
    let plane: Vec<u32> = ple::hash_constants(&MULTS, &PRIMES, &OFFSETS)
        .iter()
        .flat_map(|v| [*v as u32, (*v >> 32) as u32])
        .collect();
    let hb = gpu.up_u32(&plane);
    let ht = t(hb, 1, MULTS.len() + 2 * HEADS, Dtype::U64);
    // Three slots of window state, `id + 1` with 0 for none yet.
    let state0 = [0i32, 0, 41, 120_001, 3, 3];
    let pool = |state: u32, slots: u32, rows: usize| RecurrentPool {
        state: t(state, 3, SPAN, Dtype::I32),
        slots: t(slots, rows, 1, Dtype::U32),
        conv_state: t(state, 3, SPAN, Dtype::I32),
        new_conv_state: t(state, 3, SPAN, Dtype::I32),
    };

    // Decode: lane r is slot slots[r].
    // Vocabulary-sized ids, so the 64-bit products carry between halves.
    let ids = [151_000i32, 5, 99_999];
    let slots = [2u32, 0, 1];
    let mut want_state = state0;
    let mut want = Vec::new();
    for (r, &id) in ids.iter().enumerate() {
        let slab = &mut want_state[slots[r] as usize * SPAN..][..SPAN];
        let window = [id, cell(slab[SPAN - 1]), cell(slab[SPAN - 2])];
        want.extend(hash(window));
        slab.rotate_left(1);
        slab[SPAN - 1] = id + 1;
    }
    let (ib, sb) = (gpu.up_i32(&ids), gpu.up_u32(&slots));
    for slang in [false, true] {
        let who = if slang { "slang " } else { "" };
        let (state, out) = (gpu.up_i32(&state0), gpu.up_i32(&[0; 3 * HEADS]));
        let (it, ot) = (t(ib, 3, 1, Dtype::I32), t(out, 3, HEADS, Dtype::I32));
        if slang {
            gpu.run_slang(|c| {
                kernels_slang::gather::ngram::ngram_ids(
                    c,
                    st(it),
                    st(t(state, 3, SPAN, Dtype::I32)),
                    st(t(sb, 3, 1, Dtype::U32)),
                    st(ht),
                    EOS as u32,
                    &MULTS,
                    &PRIMES,
                    &OFFSETS,
                    HEADS_PER as u32,
                    st(ot),
                )
            });
        } else {
            gpu.run(|c| {
                ple::ngram_ids(
                    c,
                    it,
                    &pool(state, sb, 3),
                    ht,
                    EOS as u32,
                    &MULTS,
                    &PRIMES,
                    &OFFSETS,
                    HEADS_PER as u32,
                    ot,
                )
            });
        }
        assert_eq!(ints(&gpu, out, 3 * HEADS), want, "{who}ngram ids, decode");
        assert_eq!(
            ints(&gpu, state, 3 * SPAN),
            want_state,
            "{who}ngram state, decode"
        );
        eprintln!("{who}ngram decode: exact");
    }

    // Chunked: requests of 4, 0 and 3 tokens; a request's slot is its first
    // token's.
    let ids = [131_071i32, EOS, 70_000, 70_000, 30, 65_537, 148_000];
    let indptr = [0i32, 4, 4, 7];
    let slots = [1u32, 1, 1, 1, 0, 0, 0];
    let mut want_state = state0;
    let mut want = Vec::new();
    for lane in 0..3 {
        let (begin, end) = (indptr[lane] as usize, indptr[lane + 1] as usize);
        if end == begin {
            continue;
        }
        let slab = &mut want_state[slots[begin] as usize * SPAN..][..SPAN];
        let run = &ids[begin..end];
        for tk in 0..run.len() {
            let back = |p: usize| {
                if tk >= p {
                    run[tk - p]
                } else {
                    cell(slab[SPAN - (p - tk)])
                }
            };
            want.extend(hash([run[tk], back(1), back(2)]));
        }
        let next: Vec<i32> = (0..SPAN)
            .map(|p| {
                let src = run.len() as isize - SPAN as isize + p as isize;
                if src >= 0 {
                    run[src as usize] + 1
                } else {
                    slab[p + run.len()]
                }
            })
            .collect();
        slab.copy_from_slice(&next);
    }
    let (ib, pb, sb) = (gpu.up_i32(&ids), gpu.up_i32(&indptr), gpu.up_u32(&slots));
    for slang in [false, true] {
        let who = if slang { "slang " } else { "" };
        let (state, out) = (gpu.up_i32(&state0), gpu.up_i32(&[0; 7 * HEADS]));
        let (it, pt, ot) = (
            t(ib, 7, 1, Dtype::I32),
            t(pb, 4, 1, Dtype::I32),
            t(out, 7, HEADS, Dtype::I32),
        );
        if slang {
            gpu.run_slang(|c| {
                kernels_slang::gather::ngram::ngram_ids_chunked(
                    c,
                    st(it),
                    st(pt),
                    st(t(state, 3, SPAN, Dtype::I32)),
                    st(t(sb, 7, 1, Dtype::U32)),
                    st(ht),
                    EOS as u32,
                    &MULTS,
                    &PRIMES,
                    &OFFSETS,
                    HEADS_PER as u32,
                    st(ot),
                )
            });
        } else {
            let ragged = RaggedTensor {
                data: it,
                indptr: pt,
            };
            gpu.run(|c| {
                ple::ngram_ids_chunked(
                    c,
                    ragged,
                    &pool(state, sb, 7),
                    ht,
                    EOS as u32,
                    &MULTS,
                    &PRIMES,
                    &OFFSETS,
                    HEADS_PER as u32,
                    ot,
                )
            });
        }
        assert_eq!(ints(&gpu, out, 7 * HEADS), want, "{who}ngram ids, chunked");
        assert_eq!(
            ints(&gpu, state, 3 * SPAN),
            want_state,
            "{who}ngram state, chunked"
        );
        eprintln!("{who}ngram chunked: exact");
    }
}
