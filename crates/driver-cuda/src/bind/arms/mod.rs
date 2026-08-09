//! Trace symbol → what runs it.
//!
//! One module per kernel family, each holding the arms that used to be a
//! `bind!` block inside `kernels-cuda-new`. They moved because they read the
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
#[derive(Debug)]
pub struct Bound {
    /// The symbol a trace states.
    pub symbol: &'static str,
    /// What runs it.
    pub arm: Option<Arm>,
    /// Why nothing does, in the words the arm that would have used them.
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
    xqa::ARMS,
];

/// The binding for one symbol, if any family declares one.
#[must_use]
pub fn bound(symbol: &str) -> Option<&'static Bound> {
    FAMILIES.iter().flat_map(|f| f.iter()).find(|b| b.symbol == symbol)
}

impl Bound {
    /// Run this binding, or refuse with the reason it has none.
    ///
    /// # Errors
    ///
    /// Whatever the arm refuses, or [`Refusal::Unstated`] naming the fact no
    /// statement carries.
    pub fn call(&self, cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
        match self.arm {
            Some(arm) => arm(cx, stream),
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
        return match b.arm {
            Some(_) => Route::Bound(b),
            None => Route::Unbound(b.unbound.unwrap_or("this symbol is not trace-fired")),
        };
    }
    // Everything else falls through to the hand-dispatch match below: the
    // driver's own ops and the lattices that have not crossed. They were two
    // variants when `execution::service` could tell them apart; it is gone
    // with the classification table, and both always fell through together.
    Route::Rows
}
