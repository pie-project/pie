//! Trace symbol → what runs it.
//!
//! One module per kernel family, each holding the arms that used to be a
//! `bind!` block inside `kernels-cuda`. They moved because they read the
//! DRIVER's vocabulary: a backend kernels crate exposes routines and cannot
//! know what a trace states, so joining the two is this side's job.
//!
//! An arm that cannot be written at all is still a row here, carrying the
//! sentence that says why. That is what makes an unservable symbol a refusal
//! naming the missing fact rather than a lookup that answers nothing.

use core::ffi::c_void;

use kernels::Refusal;

use super::cx::Cx;

mod attn;
/// The tensor-parallel collectives — `comm`'s two and `dist`'s three, in one
/// file because a sharded model text picks between the two families by
/// message size and an arm for either has to know what the other does.
mod comm;
/// The FlashInfer FA2 lattice's six dispatches. `attn`'s namespace, and a
/// file of its own because they read the plan caches and nothing else does.
mod fa2;
/// The matmuls. **Not one arm among them**, which is why this file did not
/// exist and is exactly why it has to: three symbols live deployments lower
/// to were reaching a fire with no registry entry at all.
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
/// # THREE states, not two, and the third is the one that was missing
///
/// `arm` and `unbound` spell them between them, and every combination means
/// something:
///
/// * `Some(arm)` — this binding runs it. [`Route::Bound`].
/// * `None` + `Some(why)` — nothing can run it, ever, for that reason.
///   [`Route::Unbound`], which [`Route::refusal`] turns into a LOAD-time
///   refusal: a model naming this symbol is rejected before it fires.
/// * `None` + `None` — **the DRIVER's hand dispatch runs it**, and this row
///   exists so the registry accounts for the symbol. [`Route::Driver`], which
///   falls through to `bind::dispatch`'s match. Write it as
///   [`Bound::driver`], never as a bare literal.
///
/// The third state existed as a `Route` variant from the beginning and
/// nothing produced it, so a hand-dispatched symbol was indistinguishable
/// from one no registry had heard of — both answered [`Route::Rows`], and a
/// fire refused `NoArm` naming neither what was missing nor who would supply
/// it. That is what `executor_bind.rs`'s
/// `every_lowered_symbol_runs_or_says_why_not` measures, and it is why the
/// distinction had to become writable rather than merely nameable.
#[derive(Debug)]
pub struct Bound {
    /// The symbol a trace states.
    pub symbol: &'static str,
    /// What runs it.
    pub arm: Option<Arm>,
    /// Why nothing does, in the words the arm that would have used them.
    ///
    /// `None` alongside `arm: None` is not "no reason given" — see the type's
    /// own doc for what the pair means.
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

/// The binding for one symbol, if any family declares one.
#[must_use]
pub fn bound(symbol: &str) -> Option<&'static Bound> {
    FAMILIES.iter().flat_map(|f| f.iter()).find(|b| b.symbol == symbol)
}

impl Bound {
    /// A symbol the DRIVER's hand dispatch runs, declared so the registry
    /// accounts for it.
    ///
    /// Not an arm and not a refusal: `bind::dispatch`'s match holds the body,
    /// because it needs something a [`Cx`] must not offer — a cuBLAS handle,
    /// a fire-scoped state pointer, an aux slot the resolver owns. [`Cx`] is
    /// query-only by `northstar.md` §3.3 and a `Cx` that could hand any of
    /// those over is the surface that section says must not exist.
    ///
    /// **The entry buys the account, not the dispatch.** Without it the
    /// symbol answers [`Route::Rows`] and is indistinguishable from one no
    /// registry has heard of; with it, a reader asking "what runs this?" is
    /// told, in the file where every other answer for the family lives.
    #[must_use]
    pub const fn driver(symbol: &'static str) -> Self {
        Self { symbol, arm: None, unbound: None }
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
            // A [`Bound::driver`] row never reaches here -- `route` answers
            // `Route::Driver` for it and the dispatch falls through without
            // building a `Cx` -- so the `unwrap_or` is the message for a row
            // that stated no reason and is not driver-dispatched either,
            // which is a row somebody wrote wrong.
            None => Err(Refusal::Unstated { what: self.unbound.unwrap_or(self.symbol) }),
        }
    }
}

/// What will fire one symbol, decided once at model load.
///
/// Four answers, not two: an `Option` cannot tell "nothing declares this" from
/// "something declares it and nothing can run it", and the difference is the
/// difference between a broken model and an unsupported one.
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
            // The third state — see [`Bound`]. It falls through to the hand
            // dispatch, and it must NOT answer `Route::Unbound`: that answer
            // is a load-time refusal, so encoding a hand-dispatched symbol
            // that way would reject every model naming it.
            (None, None) => Route::Driver,
        };
    }
    // Everything else falls through to the hand-dispatch match below: the
    // driver's own ops and the lattices that have not crossed. They were two
    // variants when `execution::service` could tell them apart; it is gone
    // with the classification table, and both always fell through together.
    Route::Rows
}
