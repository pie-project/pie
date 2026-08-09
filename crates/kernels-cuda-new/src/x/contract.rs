//! §3.4 — [`Contract`], [`Entry`], [`Refusal`]: the declaration the readers
//! that cannot call read, and the two ways a fire ends.
//!
//! # Who reads a contract
//!
//! `model-compiler` does, and it is GPU-free: it links no CUDA, it holds no
//! device pointer, and **it must not be able to tell whether a symbol is
//! cuBLAS or a JIT'd kernel.** A contract is the whole of what it is allowed
//! to know — the trace-facing shape of a symbol, which is a claim about
//! values and never about geometry, occupancy or instantiation.
//!
//! Everything a contract does NOT carry is the measure of the design. It
//! does not carry an operand list, because the binding is a `fn`'s parameter
//! list. It does not carry a launch rule, because the geometry is an
//! expression. It does not carry a template argument, because that is
//! between the declaration and the device text.
//!
//! # Why `KernelSig` is still derived from it
//!
//! `kernels::KernelSig` is the row every existing consumer reads —
//! `model-compiler::kernels::check_plan` refuses any `OpKind::Launch` symbol
//! no row declares — and it survives §5 as the portable backends' vocabulary
//! (§4's fleet-scope correction: metal, vulkan and wgpu keep row-world).
//! [`Contract::sig`] is the one function that produces it, so the fn-world
//! declaration is written once and the row is derived. That is what
//! "nothing is written twice" means for a fact with two readers.

use core::ffi::c_void;

use kernels::{Cap, KernelSig, LaunchRule, Prepare};

use crate::x::cx::Cx;

/// What a trace may say about one symbol.
///
/// The ten fields are exactly `KernelSig`'s trace-facing ones. The five it
/// drops — `file`, `launch`, `operands`, `returns`, `axes` — are the ones
/// that described a launcher rather than a statement, and each of them is
/// now either a `fn`'s body or a `fn`'s signature.
#[derive(Clone, Copy, Debug)]
pub struct Contract {
    /// What the DSL calls it. `kernels::KernelSig::name`.
    pub name: &'static str,
    /// What the trace calls it, and the key everything else joins on.
    pub symbol: &'static str,
    /// This statement consumes its whole operand, not a row range.
    pub whole: bool,
    /// What the fire must have prepared before this statement can run.
    pub needs: Prepare,
    /// Capabilities this symbol does NOT have.
    pub lacks: &'static [Cap],
    /// The state store this statement writes, if any.
    pub sink: Option<&'static str>,
    /// `(input, output)` pairs that must be given the same address.
    pub in_place: &'static [(u32, u32)],
    /// This statement participates in the depth-prefix plan.
    pub depth_prefix_plan: bool,
    /// `(slot, arity)` auxiliary values this statement publishes.
    pub publishes_aux: &'static [(u8, u8)],
    /// The semantic op this symbol lowers from, if it is a lowering target.
    pub lowered_as: Option<&'static str>,
}

impl Contract {
    /// A contract that claims nothing.
    ///
    /// The base every `contract!` invocation updates from, so a declaration
    /// states only what is true of it and the reader's eye is drawn to
    /// exactly the fields that are unusual.
    pub const DEFAULT: Self = Self {
        name: "",
        symbol: "",
        whole: false,
        needs: Prepare::None,
        lacks: &[],
        sink: None,
        in_place: &[],
        depth_prefix_plan: false,
        publishes_aux: &[],
        lowered_as: None,
    };

    /// This contract as the row `model-compiler` and the portable backends
    /// read.
    ///
    /// **`operands` is empty, and that is a statement.** `abi.rs`'s
    /// `stated()` drops a row with no operands before `emit_c_shim` ever
    /// sees it, so an empty operand list is the third of the three
    /// mechanisms by which a symbol loses its ahead-of-time C entry — the
    /// other two being `device::JIT_DISPATCHED` and
    /// `execution::RUST_SERVED`. It is the right one for fn-world: those
    /// two say "something else launches this", and an empty operand list
    /// says "there is no ahead-of-time launcher to describe", which is the
    /// true thing.
    ///
    /// `launch: LaunchRule::Unstated` for the same reason, and it means
    /// here what it always meant: nothing can be dispatched from this row.
    /// Dispatch comes from the [`Entry`]'s bind.
    #[must_use]
    pub const fn sig(&self) -> KernelSig {
        KernelSig {
            name: self.name,
            symbol: self.symbol,
            whole: self.whole,
            needs: self.needs,
            lacks: self.lacks,
            sink: self.sink,
            in_place: self.in_place,
            depth_prefix_plan: self.depth_prefix_plan,
            publishes_aux: self.publishes_aux,
            lowered_as: self.lowered_as,
            ..SIG_BASE
        }
    }
}

/// A `KernelSig` that claims nothing, to update from.
///
/// `kernels::kernel!` carries the same base inline. Naming it lets the
/// generated device rows and [`Contract::sig`] share one, so a
/// nineteenth field arrives in one place rather than three.
pub const SIG_BASE: KernelSig = KernelSig {
    name: "",
    symbol: "",
    file: None,
    launch: LaunchRule::Unstated,
    whole: false,
    needs: Prepare::None,
    lacks: &[],
    sink: None,
    in_place: &[],
    depth_prefix_plan: false,
    publishes_aux: &[],
    operands: &[],
    returns: "",
    axes: &[],
    grid_param: None,
    head_param: None,
    heads_param: None,
    lowered_as: None,
};

/// Why a host program declined to launch.
///
/// Not an error: a refusal is a correct outcome the caller must handle. The
/// distinction matters enough that [`Fired`] exists so that "it declined"
/// cannot be spelled like "it ran".
///
/// **What is NOT a refusal**: a symbol in no unit, a unit that will not
/// compile, an operand list the signature rejects. Those are drift between
/// this driver and its kernel table — a broken JIT — and the fire path
/// panics with the symbol named, because a `false` there would report a
/// broken table as an unknown kernel and send the reader to the wrong file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// An extent is zero or negative: there is nothing to launch.
    ///
    /// `rope.cu:85` — `const int half = head_dim / 2; if (half <= 0)
    /// return;` — is the shape. Under the row world this was a bare `return`
    /// inside a launcher and the caller could not tell it apart from a
    /// launch; here it has a name.
    Empty {
        /// Which extent, in the kernel's own word for it.
        what: &'static str,
    },
    /// An extent is real but below the kernel's smallest unit of work.
    Narrow {
        /// Which extent.
        what: &'static str,
        /// What it was.
        at: i32,
    },
    /// An operand the fire did not carry.
    ///
    /// A null where the launcher does not accept one, or an index past the
    /// bound argument list.
    Absent {
        /// Which operand, by the name the `fn` gives its parameter.
        what: &'static str,
    },
    /// A fact no statement and no context carries.
    ///
    /// The refusals the row world wrote as prose beside an unsourced
    /// operand. They survive the port as this — but see [`Entry::unbound`],
    /// which is where a refusal that is true of EVERY fire belongs, because
    /// §0 asks for every refusal the system can make to be made at model
    /// load.
    Unstated {
        /// The fact, named.
        what: &'static str,
    },
}

/// A refusal is a sentence, so it prints as one.
///
/// The driver logs these and a human reads them. `{:?}` would print
/// `Empty { what: "half" }`, which is a struct; what a reader needs is
/// *"nothing to launch: half is zero"*. One impl here rather than a format
/// string at each of the driver's seams, so every refusal reads the same
/// way wherever it surfaces.
impl core::fmt::Display for Refusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Empty { what } => write!(f, "nothing to launch: {what} is zero"),
            Self::Narrow { what, at } => {
                write!(f, "{what} is {at}, below the kernel's smallest unit of work")
            }
            Self::Absent { what } => write!(f, "the fire does not carry {what}"),
            Self::Unstated { what } => write!(f, "nothing states {what}"),
        }
    }
}

/// How a host program ended.
///
/// `driver-cuda`'s `fire::gemv`'s `#[must_use] enum Gemv { Launched,
/// Declined(Decline) }` is the established shape and this is it,
/// generalised: **"it declined" cannot be spelled like "it ran"**. A
/// `Result<(), Refusal>` would let a caller `let _ =` it; `#[must_use]` on a
/// two-arm enum makes the caller name the arm.
#[must_use = "a launch that declined did not run, and the caller has to say what happens instead"]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fired {
    /// The launch went to the device.
    Launched,
    /// It did not, for this reason.
    Declined(Refusal),
}

impl Fired {
    /// This outcome as a bind body's result.
    ///
    /// The one place the two spellings meet: a `fn` returns [`Fired`]
    /// because its callers are hosts that must handle a decline, and a bind
    /// body returns `Result` because it is a chain of `?`s over [`Cx`]
    /// queries that can each refuse.
    ///
    /// # Errors
    ///
    /// The refusal, when the launch declined.
    pub const fn ok(self) -> Result<(), Refusal> {
        match self {
            Self::Launched => Ok(()),
            Self::Declined(why) => Err(why),
        }
    }
}

/// What a trace's statement is bound to.
///
/// The signature §3.4 gives: a `fn` from the query-only context and a
/// stream to a launch or a refusal. It is a plain function pointer, which
/// is what lets §5 step 4 intern `&'static Entry` into the lowered model
/// once at load and leave nothing to look up per fire.
pub type Bind = fn(&Cx<'_>, *mut c_void) -> Result<(), Refusal>;

/// One symbol's whole declaration: what a trace may say, and what happens
/// when it says it.
#[derive(Clone, Copy, Debug)]
pub struct Entry {
    /// What a trace may say. The part `model-compiler` reads.
    pub contract: &'static Contract,
    /// What happens when a trace says it.
    ///
    /// `None` means this symbol is not trace-fired: its `fn` exists and is
    /// public, but no statement can carry the facts it needs, so a trace
    /// that names it is refused **at model load** rather than at the fire.
    /// §1's ladder allows exactly this — "a kernel that is never trace-fired
    /// simply has none".
    pub bind: Option<Bind>,
    /// WHY there is no bind, in the words the row that preceded it used.
    ///
    /// Always `Some` when `bind` is `None`, and this is the sentence a
    /// load-time refusal prints. The row world wrote these beside a
    /// `Source::Unbound` where only a reader could find them; here the
    /// refusal is the diagnostic.
    pub unbound: Option<&'static str>,
}

impl Entry {
    /// Fire this entry's bind, or say why there is none.
    ///
    /// # Errors
    ///
    /// [`Refusal::Unstated`] when the symbol has no bind, otherwise
    /// whatever the bind refused with.
    pub fn call(&self, cx: &Cx<'_>, stream: *mut c_void) -> Result<(), Refusal> {
        match self.bind {
            Some(bind) => bind(cx, stream),
            None => Err(Refusal::Unstated {
                what: self.unbound.unwrap_or(self.contract.symbol),
            }),
        }
    }
}
