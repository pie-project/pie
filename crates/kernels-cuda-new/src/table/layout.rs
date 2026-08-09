//! Pure addressing: gather, scatter, split, concat, transpose, embed.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.

use kernels::kernel;
use kernels::operands;
use kernels::Source;
use kernels::KernelSig;

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    kernel!(split_rows "layout::split_bf16_rows",
        operands = operands![
            src: Buf <- Source::In(0),
            left: BufMut <- Source::Out(0),
            right: BufMut <- Source::Out(1),
            n: I32 <- Source::Rows,
            left_dim: I32 <- Source::OutWidth(0),
            right_dim: I32 <- Source::OutWidth(1),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(split_qwen_gdn_ba "layout::split_qwen_gdn_ba_bf16",
        operands = operands![
            ba: Buf <- Source::In(0),
            b_out: BufMut <- Source::Out(0),
            a_out: BufMut <- Source::Out(1),
            n: I32 <- Source::Rows,
            v_h: I32 <- Source::OutWidth(0),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // ── FOUR ROWS WERE HERE AND ARE DELETED. §54's whole finding. ────────
    //
    // `layout::copy_if_valid_slot`, `layout::concat_bf16_rows`,
    // `layout::deinterleave_rows_bf16` and `layout::deinterleave_vec_bf16`.
    // Each had a JIT unit, migrated device text in
    // `kernels-cuda-new/csrc/src/layout/*.cuh`, a `dsl::cuda` wrapper — and
    // NO CALLER IN ANY LANGUAGE.
    //
    // The consumer sweep, all five doors, because a deletion is a claim
    // about a whole consumer set and one door is not the set:
    //
    //   * `crates/model/src` — searched for the SYMBOL STRING and,
    //     separately, for the wrapper name `dsl::cuda::{copy_if_valid_slot,
    //     concat_rows, deinterleave_rows, deinterleave_vec}`. Two different
    //     tokens; a sweep for one of them once reported a live symbol as
    //     uncalled. Zero hits for either, for all four.
    //   * `model-compiler/src/lower.rs::semantic()` — no mapping. (21 rows
    //     reach production with no DSL wrapper at all, so this door has to
    //     be checked even when the wrapper is dead.)
    //   * hand-written `ffi::pie_k_*` arms in `driver-cuda/src` — the door
    //     that is invisible to every check reading generated text. There are
    //     eight such names in the whole crate and none of them is these.
    //   * C++ — `.cu`, `.cuh`, `.cpp`, `.hpp` across the tree.
    //   * the generated dispatch itself — all four carried `Source::Unbound`
    //     operands, so `emit_rust_dispatch` skipped them WHOLE and no arm of
    //     either kind was ever written for them. They could not have been
    //     fired even if something had asked.
    //
    // §28's root cause, in its clearest surviving instance: the DSL surface
    // was generated from launcher headers, so a wrapper existed for every
    // launcher and READ AS DEMAND to any tool that stopped at it. Four rows
    // looked live because four wrappers existed. The wrappers went with the
    // rows — `model/tests/kernels_table.rs::the_table_covers_the_dsl_surface`
    // asserts these two sets are equal, so deleting one side alone would
    // have failed it and deleting both keeps it balanced.
    //
    // WHAT SURVIVED, AND IT IS NOT NOTHING. `families::layout`'s DEVICE rows
    // stay. `copy_if_valid_slot`'s in particular is the only witness in the
    // tree for `kernels::LaunchRule::Single` and is fired three times by
    // `kernels-cuda-new/tests/launch_rules.rs`, which resolves through
    // `unit::unit_of` and never through this table. Deleting a table row is
    // a claim about who CALLS a symbol; it says nothing about whether the
    // kernel exists, and here the kernel does.
    // THE EPILOGUE'S GATHER. A prefill streams one row per token and reads
    // one distribution per request, so the rows that are actually sampled
    // have to be collected before the final norm and the head — and they
    // are not a contiguous run, which is why this is a gather rather than
    // a slice.
    //
    // It had no row and no arm, and the reason it was never missed is
    // worth keeping: `driver-cuda`'s shell built every fire row as
    // `samples: true`, so `sampled < window.len()` was false on every
    // fire and `lower::epilogue` never stated the gather. The moment the
    // shell read the step's real readout list, every prefill asked for
    // this and got `NoArm`.
    //
    // The last operand is the row WIDTH, not a vocabulary: the header
    // names it `vocab` but the caller passes `H`, because this gathers
    // hidden rows on their way INTO the head.
    // ── `layout::embed_bf16`, MOVED HERE FROM `table::driver_internal` ────
    //
    // The first launch of every fire. It moved because
    // `driver_internal`'s own module doc names the exit — *"a row leaves
    // when a statement learns to say it"* — and a statement already said
    // it: `model-compiler/src/lower.rs:1462` is
    // `Embed { .. } => Semantic::Kernels(&["layout::embed_bf16"])`. The row
    // sat in a table `table::TABLES` deliberately excludes, so
    // `table::sig` did not resolve it and `check_plan` could not see the
    // symbol the lowering names.
    //
    // The move is also what makes the C++ deletable.
    // `execution::RUST_SERVED` is gated on `table::sig(symbol)` resolving
    // with non-empty operands, so a `driver_internal` row can never be
    // taken over — deletion is its only close, and deleting a row a
    // lowering names is not available. Here, `RUST_SERVED` is, and
    // `driver-cuda/src/fire/embed.rs` is the launcher.
    //
    // WHAT FORCED `WeightNamed`: a vocab table is not something a trace
    // produces, so the embedding's weight is only ever the statement's own
    // NAME and never a slot in the argument run. The sourcing is carried
    // across unchanged — all seven operands, exactly as `driver_internal`
    // stated them — because a move that edits the bindings is two changes
    // wearing one diff.
    kernel!(embed "layout::embed_bf16",
        operands = operands![
            token_ids: I32s <- Source::Ctx("token_ids"),
            weight: Buf <- Source::WeightNamed,
            y: BufMut <- Source::Out(0),
            num_tokens: I32 <- Source::Rows,
            hidden: I32 <- Source::OutWidth(0),
            vocab: I32 <- Source::Ctx("vocab"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(gather_rows "layout::gather_bf16_rows",
        operands = operands![
            src: U16s <- Source::In(0),
            row_indices: I32s <- Source::SamplingIndices,
            dst: U16sMut <- Source::Out(0),
            num_dst_rows: I32 <- Source::Rows,
            width: I32 <- Source::OutWidth(0),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    // The PLE relay: [N, L, D] -> [L, N, D], so a layer reads a
    // contiguous slice. Addressing, not arithmetic.
    // The relay's three extents, off the RESULT. It is `[L, Tokens,
    // ple_dim]` -- the layer axis leads, which is the whole reason this
    // statement exists -- so the layer count and the per-layer width are
    // its own dims and the token count is the fire's. The arm read all
    // three from config, on the reading that `Tokens` being off the
    // leading axis left it with nothing to derive from; the leading axis
    // is exactly what carries two of them.
    kernel!(transpose_nld_to_lnd "layout::transpose_bf16_nld_to_lnd",
        operands = operands![
            src: U16s <- Source::In(0),
            dst: U16sMut <- Source::Out(0),
            n: I32 <- Source::Rows,
            // Neither extent is the plan's, which is what put this row on
            // the generator's wall. The PLE dim is a fire fact the driver
            // holds, and the layer count is what is left of the operand's
            // row once that is divided out — which is exactly the
            // arithmetic the hand arm did, refusal on an unset `ple_dim`
            // included.
            layers: I32 <- Source::Div(&Source::Width(&Source::In(0)), &Source::CtxNonZero("ple_dim")),
            dim: I32 <- Source::Ctx("ple_dim"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(verify_stash_store "qwen35_verify_stash_store"),
    kernel!(verify_stash_load "qwen35_verify_stash_load"),
];
