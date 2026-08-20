//! Trace symbol → what runs it.
//!
//! One module per kernel family. A symbol no arm can serve is still a row,
//! carrying the reason, so it refuses by naming the missing fact rather than
//! answering nothing. [`row_census`] counts the rows at compile time.

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

mod attn;
/// The tensor-parallel collectives, `comm`'s and `dist`'s in one file: a
/// sharded model text picks between them by message size.
mod comm;
/// The FlashInfer FA2 dispatches. `attn`'s namespace, and a file of its own
/// because they read the plan caches and nothing else does.
mod fa2;
/// The matmuls, none armed here; the rows exist so live symbols reach a fire
/// with a registry entry.
mod gemm;
mod layout;
mod mlp;
mod moe;
mod norm;
mod quant;
mod rope;
mod sample;
mod ssm;
mod xqa;

/// What runs one statement.
pub type Arm = fn(&Cx<'_>, *mut c_void) -> Result<(), Refusal>;

/// One symbol's binding, or the reason it has none.
///
/// Three states, spelled by `arm` and `unbound` between them:
///
/// * `Some(arm)` -- this runs it. [`Route::Bound`].
/// * `None` + `Some(why)` -- nothing can, ever. [`Route::Unbound`], a LOAD-time
///   refusal.
/// * `None` + `None` -- the driver's hand dispatch runs it. [`Route::Driver`];
///   write it as [`Bound::driver`]. Without this state a hand-dispatched symbol
///   is indistinguishable from one no registry has heard of.
#[derive(Debug)]
pub struct Bound {
    /// The symbol a trace states.
    pub symbol: &'static str,
    /// What runs it.
    pub arm: Option<Arm>,
    /// Why nothing does.
    ///
    /// `None` beside `arm: None` is not "no reason given" -- see the type doc.
    pub unbound: Option<&'static str>,
}

/// The families, in registration order.
static FAMILIES: &[&[Bound]] = &[
    rope::ARMS,
    layout::ARMS,
    mlp::ARMS,
    norm::ARMS,
    quant::ARMS,
    moe::ARMS,
    ssm::ARMS,
    sample::ARMS,
    attn::ARMS,
    fa2::ARMS,
    xqa::ARMS,
    gemm::ARMS,
    comm::ARMS,
];

/// How the registry's rows are declared: `(armed, refused, driver)`.
///
/// A `const fn`, so it is recomputed every build. It cannot separate crossed
/// from hand-armed: both are `arm: Some(..)` and const-eval may not compare
/// function pointers.
#[must_use]
pub const fn row_census() -> (usize, usize, usize) {
    let (mut armed, mut refused, mut driver) = (0usize, 0usize, 0usize);
    let mut f = 0;
    while f < FAMILIES.len() {
        let rows = FAMILIES[f];
        let mut i = 0;
        while i < rows.len() {
            match (rows[i].arm.is_some(), rows[i].unbound.is_some()) {
                (true, _) => armed += 1,
                (false, true) => refused += 1,
                (false, false) => driver += 1,
            }
            i += 1;
        }
        f += 1;
    }
    (armed, refused, driver)
}

// `refused` rising means a symbol left the plane; `driver` rising means a row
// wants something `Cx` may not offer (northstar §3.3).
const _: () = {
    let (armed, refused, driver) = row_census();
    // 139 since `mlp::chunked_swiglu_into_bf16` joined (one kernel, two
    // statement shapes, two contracts), `gemm::act_x_wt_bias_bf16` stopped
    // being blocked on a `beta` its symbol states, and
    // `norm::rmsnorm_residual_add_scale_rmsnorm_bf16` on a scale its
    // statement carries, and `norm::rmsnorm_gated_fp32_in_bf16` on a head
    // width `keys::GdnVDim` always answered. The last three moved OUT of
    // `refused`, which is why that count drops by three -- and gpt-oss's two
    // MXFP4 decode rows moved the other way, from armed to refused, because
    // `compute-sanitizer` showed the kernel indexing a per-expert POINTER
    // ARRAY nothing in this tree builds. A load-time refusal that names the
    // reason beats an illegal address that poisons the context and surfaces
    // on an unrelated kernel's module load.
    assert!(armed == 138);
    assert!(refused == 31);
    assert!(driver == 3);
};

/// The binding for one symbol, if any family declares one.
#[must_use]
pub fn bound(symbol: &str) -> Option<&'static Bound> {
    FAMILIES.iter().flat_map(|f| f.iter()).find(|b| b.symbol == symbol)
}

impl Bound {
    /// A symbol the driver's hand dispatch runs, declared so the registry
    /// accounts for it.
    ///
    /// The body needs something a [`Cx`] must not offer -- a cuBLAS handle, a
    /// fire-scoped state pointer, a resolver-owned aux slot.
    #[must_use]
    pub const fn driver(symbol: &'static str) -> Self {
        Self { symbol, arm: None, unbound: None }
    }

    /// A symbol the DERIVED column runs: the row states every operand.
    #[must_use]
    pub const fn derived(symbol: &'static str) -> Self {
        Self { symbol, arm: Some(table::derived_arm), unbound: None }
    }

    /// Run this binding, or refuse with the reason it has none.
    ///
    /// # Errors
    ///
    /// Whatever the arm refuses, or [`Refusal::Unstated`] naming the fact no
    /// statement carries.
    pub fn call(&self, cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
        match self.arm {
            Some(arm) => arm(cx, stream),
            // A [`Bound::driver`] row never reaches here, so the `unwrap_or`
            // is the message for a row somebody wrote wrong.
            None => Err(Refusal::Unstated { what: self.unbound.unwrap_or(self.symbol) }),
        }
    }
}

/// What will fire one symbol, decided once at model load.
///
/// An `Option` cannot tell "nothing declares this" from "something declares it
/// and nothing can run it" -- a broken model against an unsupported one.
#[derive(Clone, Copy, Debug, Default)]
pub enum Route {
    /// This binding runs it.
    Bound(&'static Bound),
    /// It is declared and no arm can run it, ever, for this reason.
    Unbound(&'static str),
    /// The driver's own operation, not a kernel.
    Driver,
    /// A `KernelSig` declares it and the generated match fires it.
    #[default]
    Rows,
    /// Nothing declares this symbol.
    Unknown,
}

impl Route {
    /// Why this symbol cannot be fired at all, if it cannot.
    #[must_use]
    pub const fn refusal(self) -> Option<Refusal> {
        match self {
            Self::Unbound(why) => Some(Refusal::Unstated { what: why }),
            Self::Unknown => Some(Refusal::Undeclared),
            Self::Bound(_) | Self::Driver | Self::Rows => None,
        }
    }

    /// Whether the sweep still owes this symbol a port.
    #[must_use]
    pub const fn is_row_world(self) -> bool {
        matches!(self, Self::Rows)
    }
}

/// What will fire `symbol` — the ONE resolution, in the crate that owns the
/// trace vocabulary.
#[must_use]
pub fn route(symbol: &str) -> Route {
    if let Some(b) = bound(symbol) {
        return match (b.arm, b.unbound) {
            (Some(_), _) => Route::Bound(b),
            (None, Some(why)) => Route::Unbound(why),
            // Must NOT answer `Route::Unbound`: that is a load-time refusal
            // and would reject every model naming a hand-dispatched symbol.
            (None, None) => Route::Driver,
        };
    }
    // Everything else falls through to the hand-dispatch match.
    Route::Rows
}

#[cfg(test)]
mod agreement {
    use super::{FAMILIES, Route, route};

    /// A row the registry refuses and a row whose column cannot resolve are
    /// the SAME row.
    ///
    /// The two used to be one fact written twice: a parameter that nothing
    /// supplies was spelled by being a bare `i32`, which said nothing at all,
    /// while the reason it could not be bound was prose on `Bound::unbound`
    /// over here. Ninety-five parameters wore the first and thirty-one
    /// symbols the second, and twenty-three of them were the same routines.
    ///
    /// The parameter says it now — `Env<T, keys::Unstated>`, whose `Source` is
    /// genuinely `None` — so the STATE is derivable and only the REASON is
    /// prose. This is what keeps the two from drifting: a routine that gains
    /// an unbindable parameter must gain a reason here, and one that loses its
    /// last such parameter must lose it.
    #[test]
    fn a_refused_row_is_one_whose_column_cannot_resolve() {
        let mut wrong: Vec<String> = Vec::new();
        for rows in FAMILIES {
            for b in *rows {
                let Some(r) = kernels_cuda::routine(b.symbol) else { continue };
                // `untraced` rows carry no column at all; they are not
                // "unresolvable", they are "not resolved this way".
                if r.derived.is_empty() {
                    continue;
                }
                let unresolvable = r.sources.iter().any(Option::is_none);
                let refused = matches!(route(b.symbol), Route::Unbound(_));
                if unresolvable && !refused {
                    wrong.push(format!(
                        "  {}: a parameter says `keys::Unstated` and the registry \
                         does not refuse the row -- a fire would reach it and \
                         die on `Unstated`",
                        b.symbol
                    ));
                }
                // THE CONVERSE IS NOT DERIVABLE ANY MORE, and the reason is
                // the `Env` -> `ask` move: a fact a body ASKS for is not a
                // parameter, so it has no entry in `sources` to be `None`.
                // Every row below refuses for a fact this driver cannot
                // supply -- an MLA layer view, the score-capture CSR, a join's
                // aux operand -- and every one of those facts is asked in the
                // body now, so the column is fully sourced while the row is
                // still, truthfully, unbindable.
                //
                // Asserting it anyway would demand that a standing refusal be
                // deleted because its evidence moved, which is the test
                // pushing a wrong change rather than catching one. The half
                // above still holds and is the half that matters: a parameter
                // that says `Unstated` MUST be refused.
                let _ = refused;
            }
        }
        assert!(
            wrong.is_empty(),
            "the registry and the signatures disagree about which rows can bind:\n{}",
            wrong.join("\n")
        );
    }
}
