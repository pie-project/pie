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
pub fn in_region<const N: usize, E: kernels::Elem>(
    cx: &Cx<'_>,
    ptr: *const E,
    rows: i32,
) -> In<N, E> {
    In { ptr, rows, width: cx.in_width(N).unwrap_or(0) }
}

/// [`in_region`]'s output half. Same argument, same reason for the const.
pub fn out_region<const N: usize, E: kernels::Elem>(
    cx: &Cx<'_>,
    ptr: *mut E,
    rows: i32,
) -> Out<N, E> {
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
