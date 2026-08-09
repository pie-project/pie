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
    // A copy that skips requests whose slot id is invalid: the launch happens
    // for every request every time and the slot decides whether it does
    // anything, so the dispatch is fixed and a CUDA graph replays.
    kernel!(copy_if_valid_slot "layout::copy_if_valid_slot", whole = true,
        operands = operands![
            src: U8s,
            dst: U8sMut,
            bytes: Usize,
            slot_ids: I32s,
            request: Usize,
            stream: Stream,
        ]),
    kernel!(concat_rows "layout::concat_bf16_rows",
        operands = operands![
            left: Buf,
            right: Buf,
            out: BufMut,
            n: I32,
            left_dim: I32,
            right_dim: I32,
            stream: Stream,
        ]),
    // Splits a packed gate/up bank by HALVES, where `deinterleave_rows`
    // splits by parity. Same shape, different layout, checkpoint decides.
    kernel!(split_gate_up "layout::split_gate_up_bf16",
        operands = operands![
            packed: Buf,
            gate_out: BufMut,
            up_out: BufMut,
            n_tokens: I32,
            inter: I32,
            stream: Stream,
        ]),
    // gpt-oss interleaves gate and up ROW BY ROW, so splitting them is a
    // parity deinterleave and not a slice. Weight-shaped, no token extent.
    kernel!(deinterleave_rows "layout::deinterleave_rows_bf16",
        operands = operands![
            fused: Buf,
            gate_out: BufMut,
            up_out: BufMut,
            i: I32,
            h: I32,
            stream: Stream,
        ]),
    kernel!(deinterleave_vec "layout::deinterleave_vec_bf16",
        operands = operands![
            fused: Buf,
            gate_out: BufMut,
            up_out: BufMut,
            i: I32,
            stream: Stream,
        ]),
    // A vocab-sharded embedding: the rank holds `[local_vocab, hidden]` from
    // `vocab_offset` and writes zeros elsewhere, and the all-reduce after it
    // makes the row whole. The shard is a property of the WEIGHT, not of the
    // row range, so this splits like any gather.
    kernel!(embed_vocab_shard "layout::embed_bf16_vocab_shard",
        operands = operands![
            token_ids: I32s,
            weight: Buf,
            y: BufMut,
            num_tokens: I32,
            hidden: I32,
            local_vocab: I32,
            vocab_offset: I32,
            stream: Stream,
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
            layers: I32 <- Source::InWidthOver(0, "ple_dim"),
            dim: I32 <- Source::Ctx("ple_dim"),
            stream: Stream <- Source::Ctx("stream"),
        ]),
    kernel!(verify_stash_store "qwen35_verify_stash_store"),
    kernel!(verify_stash_load "qwen35_verify_stash_load"),
];
