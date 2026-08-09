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
    rows_param: None,
    lowered_as: None,
};

/// Why a host program declined to launch.
///
/// Not an error: a refusal is a correct outcome the caller must handle. The
/// distinction matters enough that [`Fired`] exists so that "it declined"
/// cannot be spelled like "it ran".
///
/// **What is NOT a refusal**: a symbol that HAS a declaration but whose unit
/// will not compile, or an operand list the signature rejects. Those are
/// drift between this driver and its kernel table — a broken JIT — and the
/// fire path panics with the symbol named, because a `false` there would
/// report a broken table as an unknown kernel and send the reader to the
/// wrong file.
///
/// [`Refusal::Undeclared`] is the opposite case and the distinction is worth
/// holding: a trace naming a symbol NOTHING declares is not internal drift,
/// it is a model asking for a kernel that does not exist, and the right
/// answer is to refuse the load rather than to panic a fire.
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
    /// An extent is above a ceiling the compiled kernel cannot exceed.
    ///
    /// The mirror of [`Refusal::Narrow`], and it exists because writing one
    /// backwards reads as a lie. `norm`'s `hc_post` refuses `hc_mult > 8`,
    /// which was an `assert!` in the C++ launcher and cannot stay one — a
    /// `fn` reached from a `bind!` must not panic — and the first port of it
    /// had to say *"`hc_mult` is 12, below the kernel's smallest unit of
    /// work"*, which is the opposite of what happened.
    ///
    /// `max` is carried rather than folded into `what` because the ceiling is
    /// a property of the compiled unit — an instantiation width, a shared
    /// memory budget — and a reader who meets this needs to know whether the
    /// input was unreasonable or the unit was built narrow.
    Wide {
        /// Which extent.
        what: &'static str,
        /// What it was.
        at: i32,
        /// The largest this unit was compiled for.
        max: i32,
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
    /// Nothing declares this symbol — no contract, no row.
    ///
    /// The one refusal that is not about a launch: it is about a LOWERING,
    /// and it is made at model load by [`crate::x::route`] before any fire
    /// exists. `model-compiler`'s `check_plan` makes the same refusal from
    /// the other end, and the two are not redundant — `check_plan` runs
    /// where the trace is BUILT, and a driver may be handed a plan built
    /// somewhere else, by an older compiler, or over a wire.
    ///
    /// It carries no field because the thing it names is the symbol, and a
    /// symbol is what every caller already has in hand.
    Undeclared,
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
            Self::Wide { what, at, max } => {
                write!(f, "{what} is {at}, above the {max} this unit was compiled for")
            }
            Self::Absent { what } => write!(f, "the fire does not carry {what}"),
            Self::Unstated { what } => write!(f, "nothing states {what}"),
            Self::Undeclared => write!(f, "no contract and no row declares it"),
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
///
/// # A BODY THAT FIRES MORE THAN ONCE
///
/// §2.3's `Composed`/`Walk` — one statement, two *different* kernels in an
/// ordered pair — is a `fn` with two calls in it, and **no floor
/// convenience is coming for it.** That is a decision, not an omission:
/// `Composed` is on §4's ledger of vocabulary that dies, and a combinator
/// here would re-mint it as vocabulary one level down. Two statements in a
/// Rust `fn`, in order, with the ordering constraint in the body's own doc,
/// is the whole of what the row world needed a `Control` variant for. The
/// first worked example is `layout`'s `fire::envelope::merge_written`
/// (`reset_started_pages`, then `merge_written`, above 128 tokens);
/// `norm` and `ssm` follow.
///
/// **But there is one rule, and it is not obvious.** A multi-launch body
/// must resolve every refusal condition BEFORE its first launch. A
/// `Declined` returned after something has already gone to the device is a
/// lie of exactly the kind this enum exists to prevent — it says nothing
/// ran, and something ran, and the device state the caller now reasons about
/// is neither the before nor the after. So: take every [`Cx`] query with
/// `?`, check every emptiness and geometry precondition for *both* kernels,
/// and only then launch. The `?`s are free to be anywhere in a
/// single-launch body and are not in a composed one.
///
/// ## HOIST, DO NOT FLATTEN — the failure mode of following that too far
///
/// The rule is *evaluate every refusal above the first launch*. It is **not**
/// *turn every refusal into an unconditional precondition*, and `ssm` found
/// the difference the hard way. `nemotron_h::mamba_split_bf16` has a refusal
/// nested inside a branch — `conv_dt_total <= 0`, reachable only when
/// `gate.is_null()`. Flattened, it reads:
///
/// ```ignore
/// if conv_dt_total <= 0 { return Fired::Declined(..) }   // WRONG
/// ```
///
/// and it declines fires that run correctly today: `conv_dim + num_heads`
/// can be non-positive while the *gated* arm, which never divides by it,
/// launches fine. **Hoisting a refusal out of an arm it does not belong to
/// invents one.** The correct hoist keeps the guard and moves only the
/// evaluation:
///
/// ```ignore
/// let ungated = gate.is_null();
/// if ungated && conv_dt_total <= 0 { return Fired::Declined(..) }
/// ```
///
/// A refusal carries its arm's condition with it. What must be above the
/// first launch is the *decision*, not a weakened version of it.
///
/// A refusal that genuinely cannot be hoisted — because the second kernel's
/// geometry depends on the FIRST one's device-side output — is not a
/// refusal. It is a device-side branch, and the answer is a kernel that
/// handles the empty case, not a host that returns `Declined` after a
/// launch. No host can read that value without a synchronise, and a fire is
/// a straight line.
///
/// ## A refusal upstream does not make is not yours to add
///
/// `ssm`'s `chunk_prefill` guards `r`/`v_h`/`k_d`/`v_d` and NOT `k_h`, which
/// reaches a kernel that divides by it. That is `gated_delta_net.cu:305`'s
/// own reading, reproduced rather than repaired, and the port said so. A
/// port that silently adds a guard changes behaviour under cover of a
/// migration, and the next reader cannot tell which conditions came from the
/// device text. Reproduce, and write the sentence.
///
/// [`Cx`]: crate::x::Cx
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

/// What will fire one symbol, decided once at model load.
///
/// §5 step 4 — the dispatch flip. §5 writes the flip as
/// `lowered.kernels: Vec<&'static Entry>`, which is right about the key and
/// wrong about two other things, both found by building it:
///
/// * **The owner.** `Lowered` is `model-compiler`'s, and a lowering carrying
///   an [`Entry`] would tell a GPU-free crate exactly which symbols are
///   JIT'd — the one thing §3.4 says it must not be able to see. The intern
///   lives on `driver-cuda`'s op join. (§5.1 ②, one level up.)
/// * **The arity.** `Option<&Entry>` has two answers and the question has
///   four. `None` was carrying "not ported yet", "the driver's own op" and
///   "nothing declares this at all" in one value, which is why step 4's
///   second half looked un-landable: **"unknown symbols refuse at load"
///   cannot be written against a value that cannot say "unknown".** Naming
///   the four is the whole of what makes the refusal expressible.
///
/// # Why this is `_cuda`-gated
///
/// For the same reason [`crate::x::FAMILIES`] is and [`crate::x::SIGS`] is
/// not. A `Route` can tell a JIT'd kernel from a cuBLAS one — that is its
/// entire job — so a crate that can name one is a crate that can tell them
/// apart, and `model-compiler` must not be. The gate is where that rule is
/// enforced rather than remembered.
#[cfg(feature = "_cuda")]
#[derive(Clone, Copy, Debug, Default)]
pub enum Route {
    /// fn-world: this entry's bind fires it.
    Bound(&'static Entry),
    /// fn-world declares it and no bind can fire it, ever, for this reason.
    ///
    /// Not "this fire cannot" — [`Entry::unbound`] is a fact about the
    /// symbol, true of every fire, which is exactly why it belongs at load.
    Unbound(&'static Entry, &'static str),
    /// The driver's own operation, not a kernel.
    ///
    /// `execution::Service::DriverOp` — a symbol a trace may state whose
    /// implementation is a driver call rather than a launch. Today:
    /// `pie_lora_qkv_correction`, `qwen35_verify_stash_*`.
    ///
    /// **This is not a leftover and does not retire with the sweep.** A
    /// driver op has no device text, so there is no `.cuh` for it to be the
    /// second truth of; and it cannot become a [`Bound`](Self::Bound),
    /// because a [`Bind`] receives only a [`Cx`], which is query-only by
    /// §3.3 — no device API, no allocator, no stream mutation — and
    /// `pie_lora_qkv_correction` needs a cuBLAS handle. Handing a bind body
    /// a cuBLAS handle would hand it a device API with a settable stream, a
    /// math mode and a workspace, which is precisely the surface §3.3 says
    /// must not exist. See `x/mod.rs`'s note on the hand match.
    Driver,
    /// Row-world: a `KernelSig` declares it and the generated match or the
    /// hand match answers.
    ///
    /// **Temporary.** This variant is constructed only when
    /// `table::sig(symbol)` finds a row that [`crate::x::SIGS`] did not
    /// derive — that is, a family the §5 step-5 sweep has not reached.
    ///
    /// # What removes it
    ///
    /// `table::ROW_TABLES` becoming empty. At that moment every symbol in
    /// `table::KERNELS` comes from `x::SIGS`, so every row lookup that finds
    /// something also finds an [`Entry`], this arm is unreachable, and the
    /// two lines that build it collapse into [`Unknown`](Self::Unknown).
    /// That is step 6's first deletion and its precondition is mechanical:
    /// one `const` list reaching length zero, not a judgement call.
    ///
    /// # And it is the default
    ///
    /// A `Route` that no [`route`](crate::x::route) call produced belongs to
    /// the row world, because that is what every unresolved launch meant
    /// before this enum existed. The default is the SAFE direction: it
    /// dispatches, where [`Unknown`](Self::Unknown) would refuse. A
    /// `LaunchSpec` built by hand in a test therefore keeps behaving as it
    /// did, and the default disappears with the variant.
    #[default]
    Rows,
    /// Nothing declares this symbol.
    ///
    /// The load-time refusal §5 step 4 asks for. See [`Refusal::Undeclared`]
    /// for why it is not redundant with `check_plan`.
    Unknown,
}

#[cfg(feature = "_cuda")]
impl Route {
    /// The entry that will fire this, if one will.
    #[must_use]
    pub const fn entry(self) -> Option<&'static Entry> {
        match self {
            Self::Bound(entry) => Some(entry),
            _ => None,
        }
    }

    /// Why this symbol cannot be fired at all, if it cannot.
    ///
    /// `None` for the two routes that fire ([`Bound`](Self::Bound),
    /// [`Driver`](Self::Driver)) and for [`Rows`](Self::Rows), which fires
    /// or refuses at the fire — the row world cannot answer "is there an
    /// arm" without a bound launch, and re-deriving `emit_rust_dispatch`'s
    /// rule here would be writing the emitter's decision twice.
    ///
    /// That gap is the honest limit of this refusal, and it closes when
    /// [`Rows`](Self::Rows) does.
    #[must_use]
    pub const fn refusal(self) -> Option<Refusal> {
        match self {
            Self::Unbound(_, why) => Some(Refusal::Unstated { what: why }),
            Self::Unknown => Some(Refusal::Undeclared),
            Self::Bound(_) | Self::Driver | Self::Rows => None,
        }
    }

    /// Whether the §5 step-5 sweep still owes this symbol a port.
    ///
    /// The sweep's own progress bar, readable from any lowering: a driver
    /// can print how much of the model it loaded still runs on rows.
    #[must_use]
    pub const fn is_row_world(self) -> bool {
        matches!(self, Self::Rows)
    }
}
