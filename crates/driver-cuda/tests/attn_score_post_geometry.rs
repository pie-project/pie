//! The four capture post-kernels' launches, checked against their rows.
//!
//! # Why this file exists at all
//!
//! `tests/attn_score_parity.rs` is the behavioural gate for the score
//! captures, and it cannot cover these. Its golden was produced by compiling
//! `driver-cuda/csrc/src/model/attn_score.cu`, which commit `4569b9e4b`
//! deleted along with the rest of the pre-rewrite crate, so
//! `tests/oracle/attn_score/run.sh` cannot run and `GOLDEN_FNV1A64` is
//! frozen. Three of the four launches below never appeared in that program
//! anyway — they were the tail of `attention_flashinfer.cu`'s capture
//! dispatches, a translation unit the oracle never compiled — so the parity
//! recorder records nothing for them, on purpose and with the omission
//! stated at each method.
//!
//! What is left uncovered by that decision is the part most likely to be
//! wrong: **the operand list and the geometry the driver transcribed from a
//! C++ `<<<>>>` that has now been deleted.** A shifted argument there is not
//! a crash. `Args::bind` checks types, and every one of these kernels takes
//! its pointers in a row followed by its `int`s, so swapping
//! `kv_page_indptr` for `kv_last_page_lens` — or `num_q_heads` for
//! `page_size` — binds cleanly and produces a plausible score row. That is
//! precisely the class of error the module header calls worse than a fault.
//!
//! # What is checked, and what is NOT
//!
//! This is a **transcription check**, not a behavioural one. It needs no
//! GPU, no NVRTC and no CUDA toolkit: it resolves each symbol through
//! `kernels_cuda_new::unit`, which is a static table, and compares the row's
//! operand list against the C++ argument list quoted below it.
//!
//! It does NOT check that the kernels compute the right floats. Nothing in
//! this tree does — the C++ that would have been the oracle is deleted, and
//! `new-horizon.md` §53 records that as an outstanding gap rather than
//! pretending otherwise.

use kernels::Ty;

/// One post-kernel: the symbol the driver fires, and the C++ launcher's
/// argument list in order, as `(name, type)`.
///
/// The names are the ROW's operand names. The types are what the row must
/// declare for the driver's `ArgValue`s to bind. Both are transcribed from
/// the launcher quoted in the doc comment above each entry, and the launcher
/// text itself is pinned by `kernels-cuda-new/tests/launch_rules.rs` against
/// `driver-cuda/src/fire/attn_score.rs`, which quotes it verbatim.
struct Post {
    symbol: &'static str,
    operands: &'static [(&'static str, Ty)],
    /// The launcher this was transcribed from, for the failure message.
    launcher: &'static str,
}

/// The four launches `fire::attn_score` issues, in capture order.
const POSTS: &[Post] = &[
    // `dispatch_attention_flashinfer_decode_capture_bf16`'s tail:
    //
    //     const dim3 grid(static_cast<unsigned>(cache.num_requests),
    //                     static_cast<unsigned>(cache.num_q_heads));
    //     device::attn_score_normalize<<<grid, 256, 0, stream>>>(
    //         score_out, score_indptr_d, kv_page_indptr_d,
    //         kv_last_page_lens_d, cache.page_size);
    //
    // `scores` is `BufMut` because the kernel divides in place; there is no
    // second buffer and no `kv_len`, which the body derives from the page
    // CSR. A row that "helpfully" added a length operand would bind and be
    // ignored.
    Post {
        symbol: "attn::attn_score_normalize",
        operands: &[
            ("scores", Ty::BufMut),
            ("score_indptr", Ty::I32s),
            ("kv_page_indptr", Ty::U32s),
            ("kv_last_page_lens", Ty::U32s),
            ("page_size", Ty::I32),
        ],
        launcher: "device::attn_score_normalize<<<grid, 256, 0, stream>>>",
    },
    // `dispatch_attention_flashinfer_prefill_capture_bf16`'s tail:
    //
    //     const dim3 norm_grid(static_cast<unsigned>(cache.num_requests),
    //                          static_cast<unsigned>(cache.num_q_heads),
    //                          static_cast<unsigned>(window));
    //     device::attn_prefill_score_normalize<<<norm_grid, 256, 0, stream>>>(
    //         score_out, score_indptr_d, qo_indptr_d, kv_page_indptr_d,
    //         kv_last_page_lens_d, cache.page_size, window);
    //
    // `qo_indptr` is the operand the decode form does not have, and `window`
    // is both the third grid extent and the last operand.
    Post {
        symbol: "attn::attn_prefill_score_normalize",
        operands: &[
            ("scores", Ty::BufMut),
            ("score_indptr", Ty::I32s),
            ("qo_indptr", Ty::U32s),
            ("kv_page_indptr", Ty::U32s),
            ("kv_last_page_lens", Ty::U32s),
            ("page_size", Ty::I32),
            ("window", Ty::I32),
        ],
        launcher: "device::attn_prefill_score_normalize<<<norm_grid, 256, 0, stream>>>",
    },
    // Immediately after it:
    //
    //     const dim3 fold_grid(static_cast<unsigned>(cache.num_requests), 32u);
    //     device::attn_prefill_score_fold<<<fold_grid, 256, 0, stream>>>(
    //         score_out, folded_out, score_indptr_d, qo_indptr_d,
    //         kv_page_indptr_d, kv_last_page_lens_d, cache.page_size,
    //         cache.num_q_heads, window);
    //
    // The only one of the four with two buffers, and the only one taking
    // `num_q_heads` as an OPERAND rather than a grid extent — it collapses
    // the head axis instead of indexing it.
    Post {
        symbol: "attn::attn_prefill_score_fold",
        operands: &[
            ("scores", Ty::Buf),
            ("folded", Ty::BufMut),
            ("score_indptr", Ty::I32s),
            ("qo_indptr", Ty::U32s),
            ("kv_page_indptr", Ty::U32s),
            ("kv_last_page_lens", Ty::U32s),
            ("page_size", Ty::I32),
            ("num_q_heads", Ty::I32),
            ("window", Ty::I32),
        ],
        launcher: "device::attn_prefill_score_fold<<<fold_grid, 256, 0, stream>>>",
    },
    // The decode fold, which made this journey one migration earlier:
    //
    //     const dim3 grid(static_cast<unsigned>(num_requests), 64u);
    //     device::attn_score_fold_heads<<<grid, 256, 0, stream>>>(
    //         scores, score_indptr_d, kv_page_indptr_d, kv_last_page_lens_d,
    //         page_size, num_q_heads, folded);
    //
    // Included here because it is the same shape and the same hazard, and
    // because a check that covers three of four launches invites the reader
    // to wonder about the fourth.
    Post {
        symbol: "attn::attn_score_fold_heads",
        operands: &[
            ("scores", Ty::Buf),
            ("score_indptr", Ty::I32s),
            ("kv_page_indptr", Ty::U32s),
            ("kv_last_page_lens", Ty::U32s),
            ("page_size", Ty::I32),
            ("num_q_heads", Ty::I32),
            ("folded", Ty::BufMut),
        ],
        launcher: "device::attn_score_fold_heads<<<grid, 256, 0, stream>>>",
    },
];

/// Every symbol `fire::attn_score` fires resolves to exactly one row, whose
/// operands are the launcher's arguments in the launcher's order.
///
/// A failure here means one of three things, and the message says which:
/// the symbol was renamed in `kernels-cuda-new` and the driver still says
/// the old name; the row's operand list drifted from the C++ it was
/// transcribed from; or the driver's `ArgValue` list was written against a
/// different row than the one it names.
#[test]
fn every_post_kernel_row_matches_the_launcher_it_came_from() {
    for post in POSTS {
        let Some((_, unit)) = kernels_cuda_new::unit::unit_of(post.symbol) else {
            panic!(
                "{}: in no JIT unit. `fire::attn_score` fires this symbol by name; if it \
                 was renamed in `kernels-cuda-new`, the driver's constant must follow or \
                 the launch panics at run time instead of failing here.\n  launcher: {}",
                post.symbol, post.launcher
            );
        };
        let Some(row) = unit.row(post.symbol) else {
            panic!(
                "{} named unit `{}` and is not one of its rows",
                post.symbol, unit.name
            );
        };
        let got = row.sig.operands;
        assert_eq!(
            got.len(),
            post.operands.len(),
            "{}: the row takes {} operands and the launcher passed {}.\n  launcher: {}\n  \
             row: {:?}\n  A count that drifts is caught by `Args::bind`; a count that \
             matches with the wrong ORDER is not, which is why the names are checked too.",
            post.symbol,
            got.len(),
            post.operands.len(),
            post.launcher,
            got.iter().map(|o| o.name).collect::<Vec<_>>()
        );
        for (i, (name, ty)) in post.operands.iter().enumerate() {
            assert_eq!(
                got[i].name, *name,
                "{}: operand {i} is `{}` and the launcher passed `{name}` there.\n  \
                 launcher: {}\n  These kernels take their pointers in a run and their \
                 `int`s in a run, so a swap inside either run BINDS and produces a \
                 plausible score row. That is the error this assertion exists for.",
                post.symbol, got[i].name, post.launcher
            );
            assert_eq!(
                got[i].ty, *ty,
                "{}: operand `{name}` is {:?} in the row and {ty:?} in the driver's \
                 transcription of {}",
                post.symbol, got[i].ty, post.launcher
            );
        }
    }
}

/// The three `ATTN_SCORE_POST` rows are hosted by one unit, and the decode
/// fold by another.
///
/// Not decoration: `cache::module` compiles a UNIT, so every symbol in the
/// first group costs one NVRTC compile between them and the fourth costs a
/// second. If a later edit scattered them across units, each capture would
/// pay a separate compile on its first fire — a latency regression no
/// behavioural test would see, which is the same class of invisible cost
/// `FOLD_GRID_Y` is documented against.
#[test]
fn the_three_post_kernels_share_one_unit() {
    let unit_of = |sym: &str| {
        kernels_cuda_new::unit::unit_of(sym)
            .unwrap_or_else(|| panic!("{sym}: in no JIT unit"))
            .1
            .name
    };
    let normalize = unit_of("attn::attn_score_normalize");
    for sym in [
        "attn::attn_prefill_score_normalize",
        "attn::attn_prefill_score_fold",
    ] {
        assert_eq!(
            unit_of(sym),
            normalize,
            "{sym} is hosted by a different unit than `attn::attn_score_normalize`; \
             the capture would pay two NVRTC compiles where the C++ paid none"
        );
    }
    assert_ne!(
        unit_of("attn::attn_score_fold_heads"),
        normalize,
        "the decode fold has joined the post unit — if that was deliberate, this \
         assertion is the place to say so, because it changes what a first fire costs"
    );
}
