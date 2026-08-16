//! What a dispatch IS, after the thing that built one stopped living here.
//!
//! This file was the join: [`binding`](crate::binding) answered where an
//! operand lived, [`geometry`](crate::geometry) answered how many workgroups
//! a `Rule` wanted, and `plan_one` turned one [`Launch`](
//! model_compiler::lower::Launch) plus one `KernelSig` row into a
//! [`Dispatch`]. Six hundred lines, and the last reader of
//! `kernels_vulkan::KERNELS` in this crate.
//!
//! # What left, and where it went
//!
//! * `plan_one` -- TAKEN OVER BY [`crate::arm`]. A routine's body states its
//!   own operands and its own grid in ordinary Rust, so there is no row to
//!   join against and nothing to join.
//! * `dims_of` -- the `grid_param` / `head_param` / `heads_param` override,
//!   the one that made gemma-4 work: a statement's own scalar beating the
//!   fire-wide number, because gemma-4's full-attention layers rotate a
//!   quarter of each head where its sliding layers rotate all of one. THAT
//!   READING IS NOT RETIRED. It is `arm::Facts`, which an arm builds from the
//!   statement first and the fire second, and `arm.rs`'s `affine_of` states
//!   the same "zero is absent" rule this function's `stated` did.
//! * `Built` and `Sources` -- the two argument bundles `plan_one` took.
//!   [`crate::serve::plan_routine`] takes what it needs directly.
//! * `rule_of` -- resolved a symbol to its row's `Rule`. There are no rows.
//!
//! What stays is what a dispatch is rather than how one is made: [`Geometry`],
//! the fire-wide shape; [`Dispatch`], the four things a recorded rectangle
//! carries; and [`Undispatchable`], the refusals -- which [`crate::serve`]
//! still raises, including `Unknown` for a symbol no arm answers to.
//!
//! Nothing here needs a device, so the whole vocabulary can be measured
//! against real plans on a machine with no GPU in it.

use crate::binding::{Params, Unbindable};
use crate::device::Bound;
use crate::geometry::Ungeometric;

/// The fire-wide shape a plan is executed at.
///
/// Everything a [`Rule`](crate::geometry::Rule) needs that a single statement
/// does not state. A statement may override three of these -- see
/// [`crate::arm::Facts`], which is where that override lives now.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Geometry {
    /// Query heads.
    pub q_heads: u32,
    /// Key/value heads.
    pub kv_heads: u32,
    /// Elements per head.
    pub head_dim: u32,
    /// Channels a partial rope rotates.
    pub rotary_dims: u32,
    /// Experts the router scores.
    pub n_experts: u32,
    /// Experts each token routes to.
    pub experts_per_token: u32,
}

/// One recordable dispatch: everything a command buffer needs, and nothing
/// that needs a command buffer to compute.
#[derive(Clone, Debug, PartialEq)]
pub struct Dispatch<'a> {
    /// The entrypoint to run.
    ///
    /// Borrowed from [`Lowered::kernels`] on the plan-ordered path, which
    /// costs nothing and is what it has always been. Owned when a ROUTINE
    /// composed it: a body that varies over an instantiation axis builds the
    /// whole spelling itself — `affine_qmm_t_bf16_gs_128_b_4` — and the
    /// `String` it built lives in the body's frame, which ends before the
    /// dispatch is recorded. `Cow` is the smaller of the two prices; the
    /// other was making every body's entrypoint `&'static str`, which forbids
    /// exactly the axis instantiation half this table needs.
    pub symbol: std::borrow::Cow<'a, str>,
    /// The operands, in binding order over the module's non-hole slots.
    pub buffers: Vec<Bound<'a>>,
    /// Which of [`Self::buffers`] the shader may WRITE through, in the same
    /// order and of the same length.
    ///
    /// Read off the kernel row's operand types -- [`kernels::Ty::BufMut`] is
    /// "the launcher may write through this" and everything else is a read --
    /// which is the only place the distinction exists. SPIR-V does not carry
    /// it usefully: `slangc` decorates a buffer `NonWritable` only when the
    /// shader said `readonly`, and this tree's shaders mostly do not.
    ///
    /// What it is FOR is the barrier between two dispatches. A fire is a few
    /// hundred rectangles over one arena and the recording used to put a
    /// full pipeline barrier between every pair of them, which on this card
    /// is 8 microseconds each -- measured, 3.8 milliseconds of a 7.2
    /// millisecond decode. Most neighbouring pairs do not touch the same
    /// bytes, and this is what lets the recorder tell which ones do.
    ///
    /// A row that states no operands has no answer here, so every slot is
    /// marked written: the coarse reading is the safe one, and it is what
    /// this driver did for every slot of every launch before this existed.
    pub writes: Vec<bool>,
    /// Where this launch's scalars go, and the bytes to put there.
    ///
    /// Not a `Vec<u8>`, because "the bytes to push" is only half the answer:
    /// the module decides whether they ride push constants or a struct in a
    /// storage buffer, and the reachable symbols split almost evenly on it.
    /// A dispatch that flattened both into one byte run would be a dispatch
    /// whose caller has to ask the shader the same question again.
    pub params: Params,
    /// Where in [`Self::buffers`] the caller's scalar block goes.
    ///
    /// [`Params::Block`] states a BINDING index; this states an index into
    /// the dense list [`Device::run`](crate::device::Device::run) takes,
    /// which skips the module's descriptor holes. The two agree on every
    /// module in this tree -- no module that reads its scalars from a buffer
    /// has a hole -- and [`plan_one`] refuses rather than assume it.
    pub block_at: Option<usize>,
    /// Workgroups in each dimension. Never contains a zero; see
    /// [`Undispatchable::Empty`].
    pub groups: [u32; 3],
    /// Which traced op produced this, for a refusal to point at.
    pub op: u32,
}

/// Why a launch could not be turned into a dispatch.
#[derive(Clone, Debug, PartialEq)]
pub enum Undispatchable {
    /// The plan names a symbol no row in the kernel table states.
    Unknown {
        /// The symbol the plan named.
        symbol: String,
    },
    /// The row's slots and the module's bindings do not line up.
    Layout(
        /// Which reading disagreed.
        crate::binding::Unlayoutable,
    ),
    /// An operand could not be bound.
    Operand {
        /// Which operand of the launch, counting from zero.
        at: usize,
        /// Why.
        why: Unbindable,
    },
    /// The scalars the plan states do not fit where the module reads them.
    Scalars {
        /// How many the plan states.
        stated: usize,
        /// How many the module's block or push range has room for.
        room: usize,
    },
    /// The row names a scalar this driver cannot work out. See
    /// [`crate::binding::Misplaced::Unresolved`].
    Unresolved {
        /// The symbol whose row named it.
        symbol: String,
        /// Which operand, counting from zero.
        at: usize,
        /// The operand's name, as the row spells it.
        name: &'static str,
        /// The `kernels::Source` variant, rendered.
        source: String,
    },
    /// The row walks the KV cache with strides and no page table, and this
    /// driver's pool is paged. See [`crate::binding::Misplaced::Contiguous`].
    Contiguous {
        /// The symbol whose row named it.
        symbol: String,
        /// Which operand, counting from zero.
        at: usize,
        /// The operand's name, as the row spells it.
        name: &'static str,
    },
    /// The rule and dims do not describe a grid.
    Ungeometric {
        /// Why.
        why: Ungeometric,
    },
    /// The rectangle sits under a guard nothing has evaluated.
    ///
    /// Not a defect: a conditional arm is a launch a caller records only
    /// after deciding the branch. Refused rather than recorded because
    /// recording every arm would RUN every arm.
    Conditional {
        /// The symbol the guarded rectangle names.
        symbol: String,
        /// Which conditional region, as an index into `Lowered::conds`.
        cond: u32,
    },
    /// The plan states a different number of operands than the module binds.
    ///
    /// Almost always the plan being short of a resource the driver is
    /// expected to own -- a paged KV cache, a page table -- rather than a
    /// mistake. Refused because binding fewer descriptors than a shader
    /// reads is what `VUID-vkCmdDispatch-None-08114` exists to catch, and
    /// binding more would put a tensor at a slot on the theory that nothing
    /// looks there.
    Arity {
        /// The symbol whose module disagrees.
        symbol: String,
        /// How many operands the plan states.
        stated: usize,
        /// How many bindings the module really has: its layout less its
        /// holes.
        module: usize,
    },
    /// The grid computed to zero in some dimension.
    ///
    /// Legal Vulkan and always a defect: it runs nothing, reports success,
    /// and leaves the output holding whatever it held before. Refused here
    /// rather than at the device so that the traced op is still in hand to
    /// name.
    Empty {
        /// The grid that would have been dispatched.
        groups: [u32; 3],
    },
    /// A crossed ROUTINE would not launch this statement.
    ///
    /// The routine path's single refusal, carrying whatever the arm, the body
    /// or the encoder said. Distinct from every variant above because those
    /// are the TABLE path's vocabulary -- a row that named an operand it does
    /// not have, a rule with no grid -- and a routine has none of those parts
    /// to get wrong. What it can refuse instead is an extent that came out
    /// empty, an entrypoint this build did not produce, and an argument list
    /// the module's bindings do not fit.
    Refused {
        /// The symbol the plan named, which is not necessarily the entrypoint
        /// the body asked for.
        symbol: String,
        /// What the routine said.
        why: kernels::routine::Refusal,
    },
}

impl std::fmt::Display for Undispatchable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown { symbol } => write!(f, "no kernel row states `{symbol}`"),
            Self::Refused { symbol, why } => {
                write!(f, "the routine for `{symbol}` refused: {why:?}")
            }
            Self::Layout(why) => write!(f, "{why}"),
            Self::Operand { at, why } => write!(f, "operand {at}: {why}"),
            Self::Scalars { stated, room } => {
                write!(f, "{stated} scalars stated, room for {room}")
            }
            Self::Unresolved {
                symbol,
                at,
                name,
                source,
            } => write!(
                f,
                "`{symbol}` operand {at} (`{name}`) is sourced from {source}, \
                 which this driver does not know how to work out"
            ),
            Self::Contiguous { symbol, at, name } => write!(
                f,
                "`{symbol}` operand {at} (`{name}`) is a contiguous KV stride, \
                 and this driver's pool is paged: the row would read real \
                 memory at the wrong tokens"
            ),
            Self::Conditional { symbol, cond } => {
                write!(f, "`{symbol}` sits under unevaluated guard {cond}")
            }
            Self::Arity {
                symbol,
                stated,
                module,
            } => write!(
                f,
                "`{symbol}` states {stated} operands and its module binds {module}"
            ),
            Self::Ungeometric { why } => write!(f, "no grid: {why}"),
            Self::Empty { groups } => {
                write!(
                    f,
                    "a grid of {groups:?} would run nothing and report success"
                )
            }
        }
    }
}

impl std::error::Error for Undispatchable {}
