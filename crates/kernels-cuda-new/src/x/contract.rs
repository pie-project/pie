use core::ffi::c_void;

use kernels::{Cap, KernelSig, LaunchRule, Prepare};

use crate::x::cx::Cx;

/// What a trace may say about one symbol.
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
    /// Which of this kernel's OUTPUTS fill which of the layer's AUX slots,
    pub publishes_aux: &'static [(u8, u8)],
    /// The name a LOWERING gives this kernel, where it differs.
    pub lowered_as: Option<&'static str>,
}

impl Contract {
    /// A contract that claims nothing.
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
            ..SIG_BASE
        }
    }
}

/// A `KernelSig` that claims nothing, to update from.
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
    operands: &[],
    axes: &[],
    grid_param: None,
    head_param: None,
    heads_param: None,
    rows_param: None,
};

/// Why a host program declined to launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// An extent is zero or negative: there is nothing to launch.
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
    Wide {
        /// Which extent.
        what: &'static str,
        /// What it was.
        at: i32,
        /// The largest this unit was compiled for.
        max: i32,
    },
    /// An operand the fire did not carry.
    Absent {
        /// Which operand, by the name the `fn` gives its parameter.
        what: &'static str,
    },
    /// A fact no statement and no context carries.
    Unstated {
        /// The fact, named.
        what: &'static str,
    },
    /// Nothing declares this symbol — no contract, no row.
    Undeclared,
}

/// A refusal is a sentence, so it prints as one.
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
    pub const fn ok(self) -> Result<(), Refusal> {
        match self {
            Self::Launched => Ok(()),
            Self::Declined(why) => Err(why),
        }
    }
}

/// What a trace's statement is bound to.
pub type Bind = fn(&Cx<'_>, *mut c_void) -> Result<(), Refusal>;

/// One symbol's whole declaration: what a trace may say, and what happens
#[derive(Clone, Copy, Debug)]
pub struct Entry {
    /// What a trace may say. The part `model-compiler` reads.
    pub contract: &'static Contract,
    /// What happens when a trace says it.
    pub bind: Option<Bind>,
    /// WHY there is no bind, in the words the row that preceded it used.
    pub unbound: Option<&'static str>,
}

impl Entry {
    /// Fire this entry's bind, or say why there is none.
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
#[cfg(feature = "_cuda")]
#[derive(Clone, Copy, Debug, Default)]
pub enum Route {
    /// fn-world: this entry's bind fires it.
    Bound(&'static Entry),
    /// fn-world declares it and no bind can fire it, ever, for this reason.
    Unbound(&'static Entry, &'static str),
    /// The driver's own operation, not a kernel.
    Driver,
    /// Row-world: a `KernelSig` declares it and the generated match or the
    #[default]
    Rows,
    /// Nothing declares this symbol.
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
    #[must_use]
    pub const fn refusal(self) -> Option<Refusal> {
        match self {
            Self::Unbound(_, why) => Some(Refusal::Unstated { what: why }),
            Self::Unknown => Some(Refusal::Undeclared),
            Self::Bound(_) | Self::Driver | Self::Rows => None,
        }
    }

    /// Whether the §5 step-5 sweep still owes this symbol a port.
    #[must_use]
    pub const fn is_row_world(self) -> bool {
        matches!(self, Self::Rows)
    }
}
