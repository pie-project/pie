//! What happens when a trace states one of the tensor-parallel collectives.
//!
//! Two families' symbols, one file, because they are two halves of one
//! decision. `comm::` is the custom P2P reduction over IPC-mapped peer memory
//! and `dist::` is NCCL, and a sharded model text picks between them **by
//! message size**: `mistral_7b_v03.cuda.tp2.decode` states
//!
//! ```text
//!   Guard { pred: TokensLE(512) }
//!     then  comm::all_reduce_bf16          (in 9 -> out 11)
//!     else  dist::all_reduce_bf16_out      (in 9 -> out 12)
//! ```
//!
//! — the same operands, the same shape, twice, with `all_reduce_p2p_max_rows`
//! deciding. So the model text has already made the crossover call before a
//! fire arrives here. What it cannot know is anything about THIS rank's
//! plane, and that is what the arms below add.
//!
//! # The fallback, decided
//!
//! `CustomAllReduce::can_handle` is a QUERY, not a refusal — its own doc says
//! a decline is *"the caller's cue to fall back to `ncclAllReduce`"* — so an
//! arm that turned one into an error would be converting a routine
//! size/registration check into a fire failure. It does not.
//!
//! **`comm::all_reduce_bf16` falls back to `dist::all_reduce_bf16_out`.**
//! That is the NCCL arm with exactly this operand shape — one input, one
//! separate output, the sum over the group — so a fallback that succeeded
//! would produce the right answer, and a fallback that refuses produces a
//! refusal naming NCCL. It refuses today: `cudarc`'s `nccl` feature is off,
//! so no binding exists to call. Either way the caller never gets `Ok` with
//! this rank's unreduced partial sitting in the destination, which is the
//! failure this whole path exists to prevent.
//!
//! **`comm::all_reduce_residual_rmsnorm_bf16` has no fallback, and refuses.**
//! There is no fused NCCL symbol to fall back TO. The unfused composition
//! exists — `dist::all_reduce_bf16` over the partial, then
//! `norm::residual_add_rmsnorm_bf16` — and `model-dsl`'s `cuda::tp` can spell
//! both, but it is not the same computation with the same buffers: the fused
//! landing lands its sum in the RESIDUAL operand (output 0, aliased over
//! input 1), and the unfused pair lands it in the partial and would need a
//! third buffer this statement does not carry to move it. Composing it here
//! would mean an arm inventing storage, so the arm refuses instead and names
//! the decline. A refusal is a wrong answer nobody gets; a composition that
//! wrote the sum to the wrong buffer is a wrong answer everybody gets.
//!
//! # What these arms deliberately do NOT do: register buffers
//!
//! `can_handle` answers [`Decline::Unregistered`] for any input whose base
//! allocation has never been through `CustomAllReduce::register_buffer`, and
//! nothing in this driver registers one — so once device text lands, every
//! fire on this path will fall back until something does. That is a known and
//! stated gap, not an oversight, and it is left open on purpose.
//!
//! Registering here would be tempting: `register_buffer` is idempotent per
//! base address, so an arm could call it the first time it saw a buffer.
//! **It would also be a deadlock.** `register_buffer` returns EARLY when the
//! base is already known and runs an all-gather when it is not, so it is a
//! collective that some ranks may skip — and a rank that skips the gather
//! while its peers enter it leaves them blocked forever. Whether every rank
//! first-sights the same address at the same statement depends on the ranks'
//! arenas agreeing, which they need not: a rank's shard sizes differ, so its
//! allocations can too.
//!
//! Registration belongs in a SETUP step that every rank runs unconditionally,
//! beside `build_tp_plane`, over the buffers a fire will use. That needs the
//! fire arena to exist at load, which it does not. A fallback is a wrong
//! answer nobody gets; a conditional collective is a hang everybody gets.
//!
//! # What is unexercised
//!
//! Everything below the `Refusal` return, and the reason changed under it.
//! It used to be that `kernels_cuda::comm::CAN_LAUNCH` was `false` — neither
//! collective had device text in this tree — so
//! `serve::load::tp_serving_refusal` refused `tp_size > 1` at `create` and no
//! fire could reach these arms in any configuration the driver accepted.
//! **Both headers are internalised now and `CAN_LAUNCH` is `true`**, so a
//! `tp_size = 2` deployment with a group key reaches them.
//!
//! What is left unexercised is narrower and is a fact about the BOX rather
//! than about the tree: a collective needs a peer, and the machine this was
//! written on has one GPU. So these arms have been compiled, their refusal
//! mapping has been type-checked against every `Decline` variant, and **no
//! fire has ever gone through them**. `kernels_cuda::comm`'s own header says
//! the same thing about the launches under them, at length.
//!
//! They were written and registered before any of that for the reason
//! `bind/arms/mod.rs`'s header gives: a symbol with no row at all answers
//! `Route::Rows` and is indistinguishable from one nothing has heard of, and
//! a model naming it is refused for the wrong reason.

use core::ffi::c_void;

use kernels::Refusal;
use kernels_cuda::comm::{AllReduce, Decline};
use kernels_cuda::jit::Ctx;

use super::super::cx::Cx;
use super::Bound;
use crate::fire::all_reduce::with_current;

/// bf16 is two bytes, and `can_handle` measures a message in bytes while a
/// statement measures it in elements.
const BF16_BYTES: usize = 2;

/// One [`Decline`] as the [`Refusal`] that carries the most of it.
///
/// `Refusal` is `Copy` with `&'static str` payloads, so a decline's numbers
/// survive only where a variant has integer fields for them —
/// [`Refusal::Wide`] does, and the three size declines are exactly the ones
/// worth keeping numbers for. The rest map to the variant that names the
/// right KIND of absence, which is what a caller reading a log needs first.
fn refusal_for(decline: &Decline) -> Refusal {
    match decline {
        Decline::NoInstance | Decline::NotInitialised => {
            Refusal::Absent { what: "a constructed custom all-reduce for this rank" }
        }
        Decline::NullInput => Refusal::Null { what: "the all-reduce's input" },
        Decline::Bytes { bytes, max_bytes } => Refusal::Wide {
            what: "the P2P all-reduce's message (or it is not a multiple of 16 bytes)",
            at: i64::try_from(*bytes).unwrap_or(i64::MAX),
            max: i64::try_from(*max_bytes).unwrap_or(i64::MAX),
        },
        Decline::NotFullyConnected { .. } => Refusal::Absent {
            what: "peer access between every ordered pair of a group wider than two",
        },
        Decline::CaptureUnknown => {
            Refusal::Device { why: "`cudaStreamIsCapturing` failed on the fire's stream" }
        }
        Decline::Unregistered => {
            Refusal::Absent { what: "a `register_buffer` for the all-reduce's input" }
        }
        Decline::AboveCrossover { bytes, crossover, .. } => Refusal::Wide {
            what: "the P2P all-reduce's message, above the crossover where NCCL wins",
            at: i64::try_from(*bytes).unwrap_or(i64::MAX),
            max: i64::try_from(*crossover).unwrap_or(i64::MAX),
        },
        Decline::NoFusionWorkspace => Refusal::Absent {
            what: "a fusion workspace (world size 2 with both fusion extents positive builds one)",
        },
        Decline::FusionTokens { tokens, max_tokens } => Refusal::Wide {
            what: "the fused landing's token count",
            at: i64::from(*tokens),
            max: i64::from(*max_tokens),
        },
        Decline::FusionHidden { .. } => Refusal::Unstated {
            what: "a hidden size equal to the one the fusion workspace was sized for",
        },
        Decline::FusionWorldSize { .. } => {
            Refusal::Unstated { what: "a world size of two, which is all the fused landing takes" }
        }
        Decline::FusionHiddenNotOctet { .. } => Refusal::Unstated {
            what: "a hidden size that is a multiple of 8, the kernel's vector width in bf16",
        },
        Decline::PatternNotInstantiated { .. } => Refusal::Unstated {
            what: "an `AllReduceFusionPattern` in `kernels_cuda::comm::INSTANTIATED`",
        },
        Decline::WorldSizeUnsupported { .. } => Refusal::Unstated {
            what: "a TP world size the kernel is instantiated at (the fused landing takes 2, 4, \
                   8, 16; the plain reduction takes 2, 4, 6, 8)",
        },
        // `Refusal::Narrow` carries the value and not the divisor, so the
        // divisor is in the sentence: 8 bf16 elements is one 16-byte vector,
        // and a count that is not a multiple of it is what the kernel cannot
        // address rather than a count that is merely small.
        Decline::Vector { count, .. } => Refusal::Narrow {
            what: "the all-reduce's element count, which must be a non-zero multiple of 8 -- \
                   the kernel's 16-byte vector width in bf16",
            at: i64::try_from(*count).unwrap_or(i64::MAX),
        },
        Decline::FusionBlockWidth { threads, max, .. } => Refusal::Wide {
            what: "the fused landing's threads per block, which is `hidden / 8` because \
                   `comm::CLUSTER_SIZE` is pinned to 1",
            at: i64::from(*threads),
            max: i64::from(*max),
        },
        Decline::FusionBlockNarrow { threads, .. } => Refusal::Narrow {
            what: "the two-shot fused kernel's threads per block, which must cover one per rank",
            at: i64::from(*threads),
        },
        Decline::NoTemplateId { .. } => Refusal::Absent {
            what: "a template-id in `kernels_cuda::comm::inst` for the resolved point",
        },
        Decline::DeviceQuery { what } => Refusal::Absent { what },
        // The one decline that is already a `Refusal` -- `jit::Ctx::launch`'s,
        // which has logged the detail once per instantiation. Forwarded whole
        // rather than flattened, so a caller sees the layer that refused.
        Decline::Launch(why) => *why,
    }
}

/// `comm::all_reduce_bf16` — one input, one output, the sum over the group.
///
/// The count is an ELEMENT count (`custom_all_reduce.hpp:100`), so it is the
/// fire's rows times the output's width; `can_handle` wants BYTES, which is
/// twice that.
fn all_reduce_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let input = cx.arg_in(0)?.cast_const();
    let output = cx.arg_out(0)?;
    let count = usize::try_from(cx.rows().count.saturating_mul(cx.out_width(0)?)).unwrap_or(0);
    if count == 0 {
        return Err(Refusal::Empty { what: "the all-reduce's element count" });
    }
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };

    // `with_current` answers `None` when this thread has no plane at all,
    // which is [`Decline::NoInstance`] by another spelling -- flattened here
    // so the fallback has ONE decline to report rather than an `Option` of
    // one.
    let attempt = with_current(|car| {
        // The admissibility test, and the only place this rank's plane gets a
        // say. A decline is a ROUTING answer, not a failure -- see the module
        // header. `can_handle` measures bytes; the row counts elements.
        match car.can_handle(input, count * BF16_BYTES, stream) {
            Err(decline) => AllReduce::Declined(decline),
            Ok(()) => {
                kernels_cuda::comm::all_reduce_bf16(&ctx, car.plane(), input, output, count)
            }
        }
    })
    .unwrap_or(AllReduce::Declined(Decline::NoInstance));

    match attempt {
        AllReduce::Launched => Ok(()),
        AllReduce::Declined(decline) => {
            fall_back_out_of_place(&ctx, input, output, count, &decline)
        }
    }
}

/// `dist::all_reduce_bf16_out`, and the refusal that stands in for it.
///
/// # Which of two absences a reader is told about
///
/// Both arms are missing, so a decline here always ends in a refusal, and
/// there are two true things to say. The rule is: **report the P2P decline
/// when the P2P path is structurally unavailable, and NCCL's when the P2P
/// path merely declined THIS message.**
///
/// `AboveCrossover`, `Bytes` and `NotFullyConnected` are the second kind —
/// they are `can_handle` saying "this one is NCCL's", which is the
/// crossover query working exactly as designed, and the honest report is that
/// NCCL is not here. Everything else is the first kind: no device text, no
/// plane, an unregistered buffer. Telling an operator about NCCL when their
/// build has no all-reduce kernel at all sends them to the wrong repository.
fn fall_back_out_of_place(
    ctx: &Ctx,
    input: *const c_void,
    output: *mut c_void,
    count: usize,
    why: &Decline,
) -> Result<(), Refusal> {
    let elems = i64::try_from(count).unwrap_or(i64::MAX);
    match kernels_cuda::dist::all_reduce_bf16_out(ctx, input, output, elems) {
        // If NCCL ever lands, this is the whole of the fallback: same
        // operands, same semantics, and the caller is none the wiser.
        Ok(()) => Ok(()),
        Err(nccl) => Err(match why {
            Decline::AboveCrossover { .. }
            | Decline::Bytes { .. }
            | Decline::NotFullyConnected { .. } => nccl,
            structural => refusal_for(structural),
        }),
    }
}

/// `comm::all_reduce_residual_rmsnorm_bf16` — the fused reduction, residual
/// add and RMSNorm.
///
/// Two inputs and two outputs, and the aliasing matters: the row states
/// `in_place = &[(0, 1)]`, which `kernels/src/lib.rs:1985` defines as
/// `(output index, input index)` — **output 0 is placed at input 1's
/// address**. `lower::Buffers` has already done that, so `arg_out(0)` and
/// `arg_in(1)` are the same pointer at bind time and the kernel's single
/// `residual_inout` slot takes it. Reading it as `arg_out(0)` rather than
/// `arg_in(1)` is deliberate and follows `norm::rmsnorm_residual_add_bf16`'s
/// arm, which has the same alias: the operand being written is the output,
/// and spelling it as an input would make the aliasing look accidental.
fn all_reduce_residual_rmsnorm_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    let input = cx.arg_in(0)?.cast_const();
    let residual_inout = cx.arg_out(0)?;
    let rms_gamma = cx.weight(0)?.cast_const();
    let norm_out = cx.arg_out(1)?;
    let tokens = cx.rows().count;
    let hidden = cx.out_width(0)?;
    let eps = cx.rms_eps()?;
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };

    // `hidden` is the FULL width here, not this rank's shard: every rank
    // holds a partial sum of the whole vector, which is what makes the
    // reduction a sum rather than a concatenation.
    let bytes = usize::try_from(tokens.saturating_mul(hidden)).unwrap_or(0) * BF16_BYTES;
    if bytes == 0 {
        return Err(Refusal::Empty { what: "the fused all-reduce's element count" });
    }

    let outcome = with_current(|car| {
        if let Err(decline) = car.can_handle(input, bytes, stream) {
            return AllReduce::Declined(decline);
        }
        kernels_cuda::comm::all_reduce_residual_rmsnorm_bf16(
            &ctx,
            car.plane(),
            input,
            residual_inout,
            rms_gamma,
            norm_out,
            tokens,
            hidden,
            eps,
        )
    });

    match outcome {
        Some(AllReduce::Launched) => Ok(()),
        // No fallback, and the module header argues why: there is no fused
        // NCCL symbol, and the unfused pair lands its sum in a different
        // buffer than this statement's aliasing says it must.
        Some(AllReduce::Declined(decline)) => Err(refusal_for(&decline)),
        None => Err(refusal_for(&Decline::NoInstance)),
    }
}

/// The tensor-parallel collectives.
///
/// The `dist::` three are [`Bound`] rows with an ARM rather than
/// `unbound` reasons, because `kernels_cuda::dist` is a real host program
/// that refuses — the refusal is the body's, at the point of fire, and
/// duplicating it here as a `&'static str` would be the second account of one
/// absence that `dist`'s own header was just corrected for having.
pub static ARMS: &[Bound] = &[
    Bound { symbol: "comm::all_reduce_bf16", arm: Some(all_reduce_arm), unbound: None },
    Bound {
        symbol: "comm::all_reduce_residual_rmsnorm_bf16",
        arm: Some(all_reduce_residual_rmsnorm_arm),
        unbound: None,
    },
    Bound { symbol: "dist::all_reduce_bf16", arm: Some(dist_all_reduce_arm), unbound: None },
    Bound { symbol: "dist::all_reduce_bf16_out", arm: Some(dist_all_reduce_out_arm), unbound: None },
    Bound { symbol: "dist::all_gather_bf16", arm: Some(dist_all_gather_arm), unbound: None },
];

/// `dist::all_reduce_bf16` — in place, `in_place = &[(0, 0)]`.
fn dist_all_reduce_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    let elems = i64::from(cx.rows().count).saturating_mul(i64::from(cx.out_width(0)?));
    kernels_cuda::dist::all_reduce_bf16(&ctx, cx.arg_out(0)?, elems)
}

/// `dist::all_reduce_bf16_out` — a separate destination and no alias pair.
fn dist_all_reduce_out_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    let elems = i64::from(cx.rows().count).saturating_mul(i64::from(cx.out_width(0)?));
    kernels_cuda::dist::all_reduce_bf16_out(&ctx, cx.arg_in(0)?.cast_const(), cx.arg_out(0)?, elems)
}

/// `dist::all_gather_bf16` — each rank's shard concatenated on every rank.
///
/// The count is PER RANK, so it is the input's width and not the output's:
/// the output is `world_size` times as wide, and handing the kernel the wide
/// figure would have every rank write past its own band.
fn dist_all_gather_arm(cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
    // SAFETY: `stream` is the fire's own, live across the launch.
    let ctx = unsafe { Ctx::on(stream) };
    let per_rank = i64::from(cx.rows().count).saturating_mul(i64::from(cx.in_width(0)?));
    kernels_cuda::dist::all_gather_bf16(
        &ctx,
        cx.arg_in(0)?.cast_const(),
        cx.arg_out(0)?,
        per_rank,
    )
}
