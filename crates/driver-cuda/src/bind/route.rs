//! Trace symbol -> what runs it. Derived, with nothing hand-written left.
//!
//! ```text
//!   no routine, or one outside the vocabulary  -> Unknown
//!   the row says the driver fires it           -> Driver
//!   no operand column                          -> Driver
//!   a parameter nothing states                 -> Unbound, naming it
//!   otherwise                                  -> Bound, at the routine's symbol
//! ```
//!
//! # This was `bind/arms/`: thirteen files, 6,621 lines, 173 rows
//!
//! Every row was one of five things, and each turned out to be derivable or to
//! be work an arm was doing that belonged somewhere else.
//!
//! **129 said only that a symbol exists and its column runs it.** That is what
//! having a routine with a column already means. They needed one fact to go:
//! `Routine::internal`, because `gemm::act_x_wt_bf16` is a real routine with a
//! complete column and admitting it would bind `beta` off the statement where
//! the symbol a text may state pins it at zero.
//!
//! **27 of the 29 refusals wrote in prose what a `None` in the source column
//! says exactly.** `unstated_parameter` answers for them, and more precisely:
//! `rope::rope_yarn_bf16`'s reason was *"llama-3's `low_freq_factor`/
//! `high_freq_factor`, which no statement or context carries"* and the derived
//! answer is `factor` -- the parameter, which cannot fall out of date with the
//! signature it indexes. A test refuses any row that writes both.
//!
//! **8 said the driver fires this by path**, which is `#[routine(driver)]` now.
//! It could not be derived from the column's shape: those eight ranged from no
//! column at all to one that resolves at every position, and that last is the
//! case the fact exists for -- `moe::moe_grouped_gemm_bf16` is a routine AND a
//! driver op of the same name, so a resolver reading only the column would
//! bind every operand correctly and run the other implementation.
//!
//! **2 were `quant`'s MXFP4 decode arms**, and what they carried was a real
//! correction: `Const<Tensor<u8>>` derives the weight chain, both halves of
//! which answer the bank's base, while the kernel's first act is
//! `packed_ptrs[expert]`. The arm bound the per-expert array by hand and a
//! `PerExpert` wrapper made `keys::WeightScales` mean the scale array rather
//! than the scale plane -- a driver deciding a key means something other than
//! what it says. `keys::WeightExpertPtrs` and `WeightExpertScalePtrs` say it.
//!
//! **9 were FA2's**, and they are the reason this file has a section below.
//!
//! # What is NOT derived, and stays stated
//!
//! Three facts, all on the routine: `internal`, `driver`, and the source
//! column itself. Each is a claim a signature can make and a column cannot,
//! and each was found by deleting a row and asking what broke.
//!
//! # The FA2 arms, and what a hand-written arm actually carries
//!
//! Eight launchers with a hand-written arm each, plus
//! `attn::dequant_kv_cache_layer_to_bf16_active` sharing one of their bodies.
//! The header listed four things holding them here and said *"none is a
//! fact"*, which was true and was not the reason. Every one of the four was
//! **work an arm did between resolving the operands and launching**, and a
//! column binds an argument list — so the arms were not carrying knowledge the
//! signatures lacked, they were carrying operations with nowhere else to
//! happen. Each went to the place that was already doing that kind of work.
//!
//! 1. **The upload** — `attn/fa2/mod.rs`'s `upload_plan`, called by the
//!    launcher. `keys::Fa2{Decode,Prefill}Upload{Src,Len,Dst}` are the copy's
//!    operands, resolved off the same plan cache and the same carve the
//!    sixteen decode leaves come from. Split by family for a harder reason
//!    than symmetry: `AttnCtx` holds TWO workspace carves, and a prefill
//!    descriptor landing in the decode one clobbers it with no fault.
//!
//! 2. **`no_join_extras`** — `#[routine(no_join)]`, checked in
//!    `table::dispatch` before it binds an operand. A precondition is a
//!    different AXIS from a `Source` rather than a `Source` nobody wrote:
//!    every source fills a slot and this one requires two to be empty.
//!
//! 3. **`dequant_prelude`** — `attn/fa2/mod.rs`'s, same name, called by the
//!    five launchers whose arms called it. The condition is a BOOT fact, so it
//!    stays a decline inside the callee rather than becoming a branch.
//!
//! 4. **`lse_slab` and `o_or`** — `keys::AttnLseOut` already answered the
//!    first, and `table::operand` grew the slot chain for the second:
//!    `Or(Slot(Out, 0), Named(..))` is what `o_or` was, said once at the
//!    parameter instead of six times in six arms.
//!
//! And a fifth the header did not count, because it was the planless pair's
//! alone: that arm **planned**. It walked the host CSRs, called
//! `plan::plan_prefill` and filled the cache. The note beside its
//! `#[unbound] plan` parameter said planning is *"`driver-cuda`'s vocabulary"*
//! — but `plan_prefill` is in `kernels-cuda` and always was. What was actually
//! the driver's is the cache's ALLOCATION and the two host CSRs, which are
//! `keys::Fa2PrefillPlanCache`, `keys::QoIndptrHost` and `keys::KvPageIndptrHost`
//! now. `plan_own_prefill` is the rest, and the pair lost its one unsourced
//! column entry with it.
//!
//! # What did not move, and is not missing
//!
//! `AttnCtx` still holds two workspace carves and a plan still writes its
//! schedule into the one it was raised against, so a prefill reading the
//! decode carve still clobbers it invisibly. That is why the upload keys are
//! per-family and why the planless pair's upload goes to `AttnWorkspaceInt`
//! and not to the prefill carve: it plans against the decode one, as the
//! entry point it replaces did.
//!
//! `window_left` is `Env<keys::WindowLeft>` and not `Param<0, i32>`, though
//! every attention row states it in `params[0]`: `LaunchSpec::params` is
//! `Vec<u32>`, so an unbounded window arrives as `0xFFFF_FFFF` and
//! `as_declared`'s `U32 -> I32` refuses above `i32::MAX`.
//!
use core::ffi::c_void;

use kernels::Refusal;
use kernels::routine::{In, Out};

use super::cx::Cx;
use super::table;

/// The `In<N, T>` a hand arm would otherwise write out longhand.
///
/// A generic `fn` and not a closure: a closure cannot be generic over a const
/// parameter, so an arm could bind slot 1's pointer with slot 0's width.
pub fn in_region<const N: usize, E: kernels::Elem<Read = *const E>>(
    cx: &Cx<'_>,
    ptr: *const E,
    rows: i32,
) -> In<E> {
    In { ptr, rows, width: cx.in_width(N).unwrap_or(0) }
}

/// [`in_region`]'s output half. Same argument, same reason for the const.
pub fn out_region<const N: usize, E: kernels::Elem<Write = *mut E>>(
    cx: &Cx<'_>,
    ptr: *mut E,
    rows: i32,
) -> Out<E> {
    Out { ptr, rows, width: cx.out_width(N).unwrap_or(0) }
}


/// What will fire one symbol, decided once at model load.
///
/// An `Option` cannot tell "nothing declares this" from "something declares it
/// and nothing can run it" -- a broken model against an unsupported one.
#[derive(Clone, Copy, Debug, Default)]
pub enum Route {
    /// The derived column runs it, at the symbol named here.
    ///
    /// THE SYMBOL AND NOT A ROW, because most rows do not exist any more. A
    /// trace may state a name that stands for a CHOICE rather than a routine
    /// -- `attn::write_kv_to_pages` is declared so `check_plan` can measure a
    /// text against it, and [`Boot::route`](super::Boot::route) resolves it to
    /// `_bf16` or `_quantised` from the KV dtype the boot settled -- so what
    /// this carries is the name whose COLUMN is about to be bound, which is
    /// not always the name the text spelled.
    Bound(&'static str),
    /// It is declared and no arm can run it, ever, for this reason.
    Unbound(&'static str),
    /// The driver's own operation, not a kernel.
    Driver,
    /// Nothing declares this symbol.
    ///
    /// THE DEFAULT, where `Rows` used to be. That variant meant *"a
    /// `KernelSig` declares it and the generated match fires it"* and was the
    /// answer for every symbol the table had not heard of -- which made it two
    /// claims at once: *"the sweep still owes this a port"* and *"the driver
    /// fires it by hand"*. [`route`] derives now and cannot produce it: a
    /// routine with a column answers `Bound`, one without answers `Driver`,
    /// and a name no routine declares answers this. The four symbols that were
    /// only ever reachable through the fallthrough say `Bound::driver` instead.
    ///
    /// Defaulting to `Unknown` and not to something permissive is the same
    /// choice `Rows` got wrong: a `LaunchSpec` nobody routed must refuse.
    #[default]
    Unknown,
}

impl Route {
    /// Why this symbol cannot be fired at all, if it cannot.
    #[must_use]
    pub const fn refusal(self) -> Option<Refusal> {
        match self {
            Self::Unbound(why) => Some(Refusal::Unstated { what: why }),
            Self::Unknown => Some(Refusal::Undeclared),
            Self::Bound(_) | Self::Driver => None,
        }
    }

}

/// The parameter this routine cannot bind, if one of them cannot.
///
/// A non-nullable parameter whose source is absent: nothing states it, nothing
/// answers it by name, and a null is not allowed to land there. The refusal
/// names the PARAMETER, which is what a reader needs -- twenty-seven of the
/// twenty-nine hand-written reasons in this file's tables were a sentence
/// about exactly this, and the parameter says it more precisely than the
/// sentence did.
///
/// It reads `sources` positionally against `derived` and treats a short
/// `sources` as all-`None`, because that is what `table::operands` does: it
/// zips the column against `sources.iter().chain(repeat(None))`.
fn unstated_parameter(row: &'static kernels::KernelSig) -> Option<&'static str> {
    (0..row.derived.len()).find_map(|i| {
        let stated = row.sources.get(i).copied().flatten().is_some();
        (!stated && !row.derived[i].nullable).then_some(row.derived[i].name)
    })
}

/// What will fire `symbol` — the ONE resolution, in the crate that owns the
/// trace vocabulary.
///
/// # Derived, whole
///
/// This used to be a lookup and nothing else, against 173 rows. Every question
/// it answered is asked of the routine now:
///
/// * no routine, or one outside the trace vocabulary -> [`Route::Unknown`]
/// * the row says the driver fires it -> [`Route::Driver`]
/// * no operand column -> [`Route::Driver`]
/// * a parameter nothing states -> [`Route::Unbound`], naming it
/// * otherwise -> [`Route::Bound`], at the routine's own symbol
///
/// There is no table in front of it any more. The last two rows were
/// `quant`'s MXFP4 arms, and what they knew is `keys::WeightExpertPtrs`.
#[must_use]
pub fn route(symbol: &str) -> Route {
    let Some(row) = kernels::sig_in(kernels_cuda::sigs(), symbol) else {
        return Route::Unknown;
    };
    if row.internal {
        // DECLARED AND STILL UNKNOWN, which is the pair this fact exists for:
        // `gemm::act_x_wt_bf16` is a real routine with a complete column, and
        // admitting it would bind `beta` off the statement where the symbol a
        // text may state (`gemm::act_x_w`) pins it at zero.
        return Route::Unknown;
    }
    if row.driver {
        // STATED, because the column's shape cannot be read for it. These
        // range from no column at all (`comm::all_reduce_bf16`) through a
        // deliberately empty source run (`gemm::lora_qkv_correction`) to one
        // that resolves at every position -- and that last is the case the
        // fact exists for: `moe::moe_grouped_gemm_bf16` is a routine AND a
        // driver op of the same name, so a resolver reading only the column
        // would bind every operand correctly and run the other implementation.
        return Route::Driver;
    }
    if row.derived.is_empty() {
        // A row with no column has nothing for the binder to bind, whatever it
        // may have meant to say. `attn::write_kv_to_pages` is the case:
        // `Boot::route` intercepts it before this is asked, because it names a
        // CHOICE the boot's KV storage settles rather than a routine.
        return Route::Driver;
    }
    match unstated_parameter(row) {
        Some(what) => Route::Unbound(what),
        None => Route::Bound(row.symbol),
    }
}

#[cfg(test)]
mod agreement {
    use super::{Route, route};

    /// A row the registry refuses and a row whose column cannot resolve are
    /// the SAME row.
    ///
    /// The two used to be one fact written twice: a parameter that nothing
    /// supplies said nothing at all by being a bare `i32`, while the reason it
    /// could not be bound was prose on a `Bound::unbound` in a table. Ninety-
    /// five parameters wore the first and thirty-one symbols the second, and
    /// twenty-three of them were the same routines.
    ///
    /// There is one spelling now and it is the parameter's, so this is no
    /// longer a check that two writings agree — it is the claim that
    /// [`Route::Bound`] MEANS the column resolves.
    #[test]
    fn a_refused_row_is_one_whose_column_cannot_resolve() {
        let mut wrong: Vec<String> = Vec::new();
        // OVER THE WHOLE REGISTRY, not over the tables. It used to walk
        // `FAMILIES` because that was where the answer lived; `route` derives
        // now, so the claim can be made about every symbol a text may state
        // rather than about the hundred and seventy that happened to have a
        // row. That is a strictly larger statement made with strictly less.
        for row in kernels_cuda::sigs() {
            // `Route::Bound` IS THE CLAIM THAT THE COLUMN RESOLVES. Anything
            // else is a different path -- `Driver` says `bind::dispatch`'s own
            // match fires it, and that match reads no column at all, so an
            // unstated parameter on one of those is not a disagreement.
            // `gemm::mla_absorb_q_to_latent_bf16` is the case: its column has
            // a `None` and its body takes a cuBLAS handle.
            if !matches!(route(row.symbol), Route::Bound(_)) {
                continue;
            }
            let Some(what) = super::unstated_parameter(row) else { continue };
            wrong.push(format!(
                "  {}: `{what}` has no source and no null may land there, and \
                 `route` answers `Bound` -- a fire would reach the column and \
                 die on `Unstated`",
                row.symbol
            ));
        }
        assert!(
            wrong.is_empty(),
            "the registry and the signatures disagree about which rows can bind:\n{}",
            wrong.join("\n")
        );
    }

    /// Nothing is refused in prose, because there is no table to write prose in.
    ///
    /// This test read the tables and failed on any row whose own column
    /// already said what its prose said. It found all twenty-seven, they were
    /// deleted, and what it asserts now is that none came back: every refusal
    /// this driver makes names a PARAMETER, and a parameter cannot fall out of
    /// date with the signature it indexes.
    #[test]
    fn every_refusal_names_a_parameter() {
        for row in kernels_cuda::sigs() {
            let Route::Unbound(what) = route(row.symbol) else { continue };
            assert!(
                row.derived.iter().any(|d| d.name == what),
                "`{}` refuses with `{what}`, which is not one of its parameters -- \
                 so it is prose, and prose is what this driver stopped keeping",
                row.symbol
            );
        }
    }
}
