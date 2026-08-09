//! `attn/attention_naive.cu`'s two surviving launchers, in Rust — and with
//! them the file.
//!
//! Three of that file's five launchers were deleted in an earlier pass
//! (`attention_naive_bf16`, `attention_mtp_history_bf16`,
//! `attention_mtp_paged_history_bf16`), on the measurement that *"the
//! cluster's whole consumer set is the cluster"*. These two are the MTP
//! state pair, and they are PORTED rather than deleted: `dsl::cuda`'s
//! `mtp_shift_hidden` and `mtp_update_pending_hidden` are the wrappers a
//! multi-token-prediction model states, and a launcher behind a live DSL
//! surface is not the same thing as a launcher behind nothing.
//!
//! # The fallback that went with the deleted three, restated so it survives
//!
//! `attention_naive.cu`'s header recorded it and there is nowhere else it is
//! written down:
//!
//! > The fallback in `attention_mtp_paged_history_bf16` — a host `if` on
//! > `max_global_tokens + history_steps > 8192` choosing a different kernel —
//! > went with it, and it is worth recording what it was, because it is the
//! > same shape as the FlashInfer host dispatch this migration could not
//! > state: a predicate over two operands selecting between two kernels with
//! > different shared-memory budgets. No `LaunchRule` says that. It did not
//! > need one, because nothing called it.
//!
//! # The symbol split, and why these two needed one
//!
//! `families/attn.rs`' device rows named `attn::mtp_shift_hidden_bf16` and
//! `attn::mtp_update_pending_hidden_bf16` — the SAME strings as the
//! `table::attn` rows. §52.11 is *a walk may drive a JIT'd kernel; it may not
//! be one*, and `execution`'s `a_walk_is_only_a_walk` enforces it through
//! `unit_of`, so a symbol a unit hosts cannot be walked. The device rows are
//! renamed `_dev`, exactly as `attn::write_kv_explicit_bf16_dev` was (§60.6);
//! the TABLE symbol does not move, because a table symbol is what a trace
//! records.
//!
//! The `_bf16` suffix was wrong on the device rows anyway, for the reason
//! `MLA_PAGED_SIGS` gives about `attn::write_mla`: both are
//! `template <typename T>` and the row picks `T`, so a format suffix on the
//! row's own name advertises a choice at a level that does not make it.
//!
//! # Both rows are unsourced, which is why `RUST_SERVED` costs nothing here
//!
//! §60.7. Neither row ever generated a dispatch arm — `crate::abi` skips a
//! row with any `Source::Unbound` operand whole — so taking them over drops
//! the shim entry and changes nothing else. What it buys is the `.cu`.

use std::ffi::c_void;

use kernels_cuda_new::runtime::{ArgValue, Launch};

/// `attn::mtp_shift_hidden_bf16` — the table symbol.
pub const SHIFT_SYMBOL: &str = "attn::mtp_shift_hidden_bf16";

/// `attn::mtp_update_pending_hidden_bf16` — the table symbol.
pub const UPDATE_SYMBOL: &str = "attn::mtp_update_pending_hidden_bf16";

/// `attn::mtp_shift_hidden_dev` — the device row.
const SHIFT_DEVICE: &str = "attn::mtp_shift_hidden_dev";

/// `attn::mtp_update_pending_hidden_dev` — the device row.
const UPDATE_DEVICE: &str = "attn::mtp_update_pending_hidden_dev";

/// `attention_naive.cu:25` — `constexpr int BLOCK = device::BLOCK;`.
///
/// One constant, shared with the header, and 256 in both.
const BLOCK: u32 = 256;

/// Whether an MTP state launch ran.
///
/// `#[must_use]` for `fire/gemv.rs`' reason.
#[must_use]
pub enum Mtp {
    /// The kernel was launched on the caller's stream.
    Launched,
    /// Nothing was launched, and why.
    Declined(MtpDecline),
}

/// Every way the MTP state pair declines. Each is a clause of one `if`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtpDecline {
    /// `total_tokens <= 0` — an empty batch, and `mtp_shift_hidden`'s grid.
    NoTokens,
    /// `num_requests <= 0` — no requests, and `mtp_update_pending`'s grid.
    NoRequests,
    /// `hidden_size <= 0` — a zero-width model.
    NoWidth,
    /// `pending_hidden == nullptr` — MTP is not armed for this fire.
    ///
    /// This is the clause that is NOT a geometry check and the reason the
    /// guard could never have been a rule's `Ungeometric::Empty`: the pending
    /// buffer is the MTP state itself, and a fire without one is a fire this
    /// pair has nothing to do for. The C++ returned; so does this.
    NoPendingState,
}

/// `attn/attention_naive.cu:54` — `mtp_shift_hidden_bf16`.
///
/// The previous step's pending hidden state becomes this step's first token,
/// per request; every other token keeps the target's own hidden state.
///
/// ```text
/// :64   device::mtp_shift_hidden<bf16><<<total_tokens, BLOCK, 0, stream>>>(
/// :65       target_hidden, pending_hidden, qo_indptr, slot_ids, out,
/// :66       num_requests, hidden_size);
/// ```
///
/// One block per TOKEN, which is `LaunchRule::PerRow` to the digit.
/// `total_tokens` is the grid and does not reach the kernel; `num_requests`
/// does, because it bounds `find_request_u32`'s scan and a request count is
/// not a row count.
///
/// # Panics
///
/// If the kernel table and this driver disagree; see [`super::hand::fire`].
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `stream` is the caller's stream.
#[allow(clippy::too_many_arguments)]
pub unsafe fn mtp_shift_hidden_bf16(
    target_hidden: *const c_void,
    pending_hidden: *const c_void,
    qo_indptr: *const u32,
    slot_ids: *const i32,
    out: *mut c_void,
    total_tokens: i32,
    num_requests: i32,
    hidden_size: i32,
    stream: *mut c_void,
) -> Mtp {
    // `attention_naive.cu:60-63`, one `if` with four clauses, split so the
    // caller learns which one refused.
    if total_tokens <= 0 {
        return Mtp::Declined(MtpDecline::NoTokens);
    }
    if num_requests <= 0 {
        return Mtp::Declined(MtpDecline::NoRequests);
    }
    if hidden_size <= 0 {
        return Mtp::Declined(MtpDecline::NoWidth);
    }
    if pending_hidden.is_null() {
        return Mtp::Declined(MtpDecline::NoPendingState);
    }
    #[allow(clippy::cast_sign_loss)]
    let launch = Launch {
        grid: [total_tokens as u32, 1, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    };
    let values = [
        ArgValue::Ptr(target_hidden.cast_mut()),
        ArgValue::Ptr(pending_hidden.cast_mut()),
        ArgValue::Ptr(qo_indptr.cast_mut().cast()),
        ArgValue::Ptr(slot_ids.cast_mut().cast()),
        ArgValue::Ptr(out),
        ArgValue::I32(num_requests),
        ArgValue::I32(hidden_size),
    ];
    super::hand::fire(SHIFT_DEVICE, launch, &values, stream);
    Mtp::Launched
}

/// `attn/attention_naive.cu:77` — `mtp_update_pending_hidden_bf16`.
///
/// Stashes each request's LAST hidden state into the pending buffer, so the
/// next step's [`mtp_shift_hidden_bf16`] has something to shift in.
///
/// ```text
/// :85   device::mtp_update_pending_hidden<bf16><<<num_requests, BLOCK, 0, stream>>>(
/// :86       target_hidden, pending_hidden, qo_indptr, slot_ids,
/// :87       num_requests, hidden_size);
/// ```
///
/// One block per REQUEST — `LaunchRule::PerRequest`, and the reason the twin
/// above is `PerRow` and this one is not: the statement records a `StateRef`
/// and no result, so it names no rectangle of its own.
///
/// # Panics
///
/// If the kernel table and this driver disagree; see [`super::hand::fire`].
///
/// # Safety
///
/// [`mtp_shift_hidden_bf16`]'s.
#[allow(clippy::too_many_arguments)]
pub unsafe fn mtp_update_pending_hidden_bf16(
    target_hidden: *const c_void,
    pending_hidden: *mut c_void,
    qo_indptr: *const u32,
    slot_ids: *const i32,
    num_requests: i32,
    hidden_size: i32,
    stream: *mut c_void,
) -> Mtp {
    // `attention_naive.cu:84-86`.
    if num_requests <= 0 {
        return Mtp::Declined(MtpDecline::NoRequests);
    }
    if hidden_size <= 0 {
        return Mtp::Declined(MtpDecline::NoWidth);
    }
    if pending_hidden.is_null() {
        return Mtp::Declined(MtpDecline::NoPendingState);
    }
    #[allow(clippy::cast_sign_loss)]
    let launch = Launch {
        grid: [num_requests as u32, 1, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    };
    let values = [
        ArgValue::Ptr(target_hidden.cast_mut()),
        ArgValue::Ptr(pending_hidden),
        ArgValue::Ptr(qo_indptr.cast_mut().cast()),
        ArgValue::Ptr(slot_ids.cast_mut().cast()),
        ArgValue::I32(num_requests),
        ArgValue::I32(hidden_size),
    ];
    super::hand::fire(UPDATE_DEVICE, launch, &values, stream);
    Mtp::Launched
}
