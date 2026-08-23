//! Turning a lowered fire into dispatches. The executor's other half.
//!
//! [`executor`] resolves a launch's operands to addresses; [`geometry`] turns
//! a rectangle into a thread grid. This is the walk that uses both, and its
//! whole shape is the claim the crate rests on:
//!
//! ```text
//! for launch in &lowered.launches {
//!     let routine = crossed(symbol)?;         // the stem names the body
//!     let args = bind(lowered, launch, ..)?;  // the driver resolves names
//!     let facts = facts_of(lowered, launch, geometry);
//!     routine.body(arm(handles, facts)?, planner)?; // the body states the rest
//! }
//! ```
//!
//! **No row is consulted.** There was a table here: a row per entry point,
//! stating its operands as a `Source` enum and its rectangle as a
//! `LaunchRule`, and two interpreters — `reorder` and `eval` — that walked
//! them. Both are deleted. What replaced them is a routine per kernel,
//! written as ordinary Rust in `kernels-metal` and shared with every other
//! backend, plus an arm here that says which of THIS driver's handles fill
//! its parameters.
//!
//! The trade is deliberate and it is the north star's: an arm per routine is
//! more code than a row per kernel, but a row could only ever state what its
//! enum could spell, and every kernel that needed something the enum could
//! not say went unreachable in silence rather than failing to compile.
//!
//! # Portable, and that is deliberate
//!
//! Nothing here touches a Metal object. A dispatch is a symbol, a file, a
//! grid, a threadgroup and a list of resolved operands — all of which are
//! decided before any device is involved, and all of which are therefore
//! provable in a build with no GPU. `encode` is the half that needs one.
//!
//! [`executor`]: crate::lowering::executor
//! [`geometry`]: crate::lowering::executor

use core::ops::Range;

use model_compiler::lower::{Arg, Launch, Lowered};

use crate::lowering::executor::{BindRefusal, BoundArg, Frame, Resolver, Slice, bind};

/// The fire-invariant half of [`Dims`]: what every launch of one fire shares.
///
/// The rectangle states the rows and the operands state the widths, so these
/// are the only quantities left — and they are the fire's geometry, handed in
/// by the caller that already knows it. The driver does not derive them:
/// deriving a head count from a buffer size is exactly the "model definition
/// inside the driver" that `batch/geometry.rs`'s `DecodeGeometry` is retiring
/// for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Geometry {
    /// Query heads.
    pub q_heads: u32,
    /// Key/value heads.
    pub kv_heads: u32,
    /// Elements per head.
    pub head_dim: u32,
    // `rotary_dims` WAS HERE TOO. `DecodeGeometry` in `batch/geometry.rs`
    // carried one and this carried the other, both fire-wide, both filled
    // from layer zero's attention row and offered to every layer as though a
    // stack could only rotate one width. gemma-4 is the counter-example, and
    // it is not an exotic one.
    //
    // The reason both could go is that the consumer went first: `RotaryWidth`
    // was the fallback of seven `ParamOr<3, ..>` sites, answered through
    // `arm.rs` off `Facts::rotary_dims`, and the key does not exist in the
    // tree any more -- those sites read the width the statement carries. Two
    // fields survived their reader by long enough that only a public-door
    // census noticed, and only after somebody built the test target.
    /// Experts the router scores.
    pub n_experts: u32,
    /// Experts each token routes to.
    pub experts_per_token: u32,
    /// The deployment's affine quantisation group size, and the bits per
    /// weight.
    ///
    /// The one pair of facts a `kernel!` row never had to state, because the
    /// TRACE stated them: `embed_gather_mb_4bit_bfloat16_gs_64_b_4` carries
    /// its own instantiation in its name and `covers_point` peeled it off. A
    /// routine composes that spelling instead of receiving it, so the numbers
    /// have to arrive as numbers.
    ///
    /// `plan_routine` checks the composition against the trace's own
    /// spelling, which is what makes carrying them here safe: a deployment
    /// whose embedding is quantised differently from the rest of it refuses
    /// by name rather than gathering from the wrong plane.
    pub group: u32,
    /// See [`Geometry::group`].
    pub bits: u32,
    /// The affine point the ROUTER GATES arrived at, when it differs from
    /// [`Self::group`]/[`Self::bits`], or zero when one point serves.
    ///
    /// gpt-oss states two: `mlx-community/gpt-oss-20b-MXFP4-Q4` lists 98
    /// tensors at group 64 / 4 bits and its 24 `mlp.router` gates at group
    /// 64 / EIGHT, and its text names `affine_qmv_fast_bfloat16_gs_64_b_8`
    /// for the gate beside `_b_4` for everything else. `model::binding`'s
    /// `observed` already reads the gate's own point off the checkpoint --
    /// `build_kernels_at` exists for no other row -- so the number reached
    /// the binding and stopped there, and every gate projection composed the
    /// dense spelling and refused `Misspelled`.
    ///
    /// Zero rather than "equal to the dense point" because `observed` states
    /// it ONLY when it differs, and two spellings of one encoding is how a
    /// text stops being comparable to itself across two checkpoints of one
    /// row.
    ///
    /// The cost of reading the gate at the stack's width is not a crash:
    /// `the_router_gate_is_read_at_its_own_width` measures it as "a fluent
    /// model routing every token to almost the right experts, cosine 0.84
    /// against the reference logits and not one NaN to notice it by".
    pub router_group: u32,
    /// See [`Geometry::router_group`].
    pub router_bits: u32,
    /// The per-head width the FULL-attention layers use, or zero for a stack
    /// whose layers all share [`Self::head_dim`].
    ///
    /// gemma-4 states two, and both of them reach a symbol: its text names
    /// `sdpa_paged_decode_bfloat16_d_<width>` for each, and `project`'s
    /// `metal_kernel_refusal` checks BOTH against this backend's SDPA axis
    /// before a load is accepted. So a fire-wide head width is not a
    /// simplification here -- it is a number that is wrong for one of the two
    /// layer kinds, and the routine composing a symbol from it spells one the
    /// trace did not state.
    ///
    /// Rope has never had this problem, because a partial rotation is a
    /// scalar the STATEMENT carries and [`arm::stated`] prefers it over the
    /// fire's. SDPA's head width is an `Env` the routine composes, so it has
    /// nowhere to ride but here.
    ///
    /// [`arm::stated`]: crate::lowering::hold
    pub global_head_dim: u32,
    /// The key/value head count the FULL-attention layers use, or zero for
    /// one shape everywhere. See [`Self::global_head_dim`].
    pub global_kv_heads: u32,
    /// One full-attention layer every `full_attn_every`, or zero for a stack
    /// that does not alternate.
    ///
    /// The same field, the same name and the same rule as
    /// [`layout::kv::Shape::full_attn_every`], which sizes the POOL's pages
    /// per layer off it. That the pool already alternated while the lowering
    /// did not is how the two halves of one deployment disagreed: pages laid
    /// out at 256 wide, read by a kernel instantiated at 128.
    ///
    /// [`layout::kv::Shape::full_attn_every`]: crate::layout::kv::Shape::full_attn_every
    pub full_attn_every: u32,
    /// The value heads and per-head width the LINEAR-attention layers carry,
    /// or `(0, 0)` for a stack with none.
    ///
    /// A THIRD pair beside [`Self::head_dim`] and [`Self::global_head_dim`],
    /// and not a third spelling of the same thing: a gated-deltanet layer has
    /// no keys and values in the attention sense at all. qwen3.5's linear
    /// layers run 32 value heads of 128 next to full layers at 2 heads of
    /// 256, and the recurrent slab, the gated norm and the scan grid are all
    /// sized by the former while the KV pool is sized by the latter.
    ///
    /// `heads_at` cannot answer this, because it answers about PAGES.
    pub v_heads: u32,
    /// See [`Self::v_heads`].
    pub v_dim: u32,
}

impl Geometry {
    /// Whether layer `l` attends the whole context.
    ///
    /// Character-for-character [`layout::kv::Shape::is_full_attention`], and
    /// deliberately not a call to it: this module is the fire-invariant
    /// lowering half, which holds no pool and must lower a text with no pool
    /// allocated at all. Duplicating four tokens is cheaper than a dependency
    /// that would make the host-only half need the pool's.
    ///
    /// [`layout::kv::Shape::is_full_attention`]: crate::layout::kv::Shape::is_full_attention
    #[must_use]
    pub const fn is_full_attention(&self, l: u32) -> bool {
        self.full_attn_every > 1 && (l + 1).is_multiple_of(self.full_attn_every)
    }

    /// This layer's key/value head count and per-head width.
    ///
    /// The fire's own pair for every deployment but gemma-4, and for that one
    /// the full layers' pair on the layers that are full.
    /// This layer's RECURRENT head count and width.
    ///
    /// Falls back to the attention pair when the stack states no recurrent
    /// one, which is what keeps `device_gdn.rs`'s rig -- a GDN fire whose
    /// only shape IS the recurrent shape -- reading the same numbers it
    /// always did.
    #[must_use]
    pub const fn recurrent_at(&self) -> (u32, u32) {
        if self.v_heads > 0 && self.v_dim > 0 {
            (self.v_heads, self.v_dim)
        } else {
            (self.kv_heads, self.head_dim)
        }
    }

    /// This layer's key/value head count and per-head width.
    ///
    /// The fire's own pair for every deployment but gemma-4, and for that one
    /// the full layers' pair on the layers that are full.
    #[must_use]
    pub const fn heads_at(&self, l: u32) -> (u32, u32) {
        if self.is_full_attention(l) {
            (
                if self.global_kv_heads > 0 {
                    self.global_kv_heads
                } else {
                    self.kv_heads
                },
                if self.global_head_dim > 0 {
                    self.global_head_dim
                } else {
                    self.head_dim
                },
            )
        } else {
            (self.kv_heads, self.head_dim)
        }
    }
}

/// One encodable dispatch: everything a command encoder needs, and nothing
/// that needs a command encoder to compute.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dispatch<'a> {
    /// The entry point to run. Borrowed from [`Lowered::kernels`] — the
    /// lowering's own spelling, unmodified, because the shader exports it
    /// under that name.
    pub symbol: &'a str,
    /// The shader that defines `symbol`, from the family's `ENTRYPOINTS`.
    pub file: &'static str,
    /// The line that makes `symbol` exist in `file`, or empty if the file
    /// already declares it. See [`kernels::routine::Fire::stamp`].
    pub stamp: &'static str,
    /// Total threads per axis.
    pub grid: [u32; 3],
    /// Threads per threadgroup per axis.
    pub threadgroup: [u32; 3],
    /// Operands **in the order the kernel reads them**, when its row states
    /// them; in the trace's stated order when it does not.
    ///
    /// The two are not the same order, and assuming they were is what made
    /// every launch of every text misbind. The trace states inputs, outputs
    /// then weights — the compiler's convention — while `affine_qmv_fast`
    /// declares `w, scales, biases, x, y`. A row that states its operands
    /// says which slot takes what, and `reorder` applies it.
    ///
    /// A row that states none is bound positionally, which is what every row
    /// got before and is wrong for most of them. That is why
    /// `tests/text_conformance.rs` counts them.
    pub args: Vec<BoundArg>,
    /// The byte ranges this dispatch reads and the ones it may write.
    ///
    /// The reason an encoder can stop putting a barrier after every single
    /// dispatch. Two launches that write different ranges and read only ranges
    /// nobody has written since the last barrier are independent, and a
    /// decode's `q`/`k`/`v` and `gate`/`up` projections are exactly that.
    pub touches: Touches,
    /// Where each scalar binds, and how wide it is there.
    ///
    /// Three facts, because three are needed and the row states all of them.
    ///
    /// * **Which buffer.** Two spellings exist in the tree: `moe/route.metal`
    ///   takes `constant RouterParams&`, one buffer holding every field, and
    ///   `quant/qmv.metal` takes its two extents as separate buffers.
    /// * **Which scalar**, as a byte offset into this dispatch's staged run.
    /// * **How wide.** `attn/sdpa_vector.metal` declares its strides
    ///   `const constant size_t&` — **eight bytes** — while the trace's params
    ///   are `u32`. A driver that handed a four-byte slot to an eight-byte
    ///   read would give the kernel four bytes of the next scalar as the high
    ///   half of this one. The row's `Ty` says which, so the stage widens.
    pub param_slots: Vec<ParamSlot>,
    /// The scalar arguments the statement states, in its stated order.
    ///
    /// A kernel takes numbers no operand shape gives — a QKV split's two
    /// widths, a strided kernel's row pitch. The **text** states them; this
    /// forwards them without knowing what they mean, which is the difference
    /// between a driver that passes a constant and one that re-derives it
    /// from a config it had to understand.
    pub params: Vec<u32>,
    /// The layers this rectangle covers.
    pub layers: Range<u16>,
    /// Which traced op produced it — where a refusal points.
    pub op: u32,
}

/// One scalar's placement: which buffer, where in the staged run, how wide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamSlot {
    /// The argument-table index this binds at.
    pub slot: usize,
    /// Byte offset into this dispatch's staged scalars.
    pub at: u32,
    /// Bytes the kernel reads there — four or eight for a scalar.
    pub bytes: u32,
    /// This slot is a POINTER to a struct holding every remaining scalar,
    /// rather than one scalar.
    ///
    /// Both spellings are in the tree and the row's `Ty` tells them apart: a
    /// `Buf` param is `constant RouterParams&` — one buffer, every field —
    /// while an `I32` param is `const constant int&`, one buffer per number.
    /// A packed slot's run is as long as the statement's scalars; a scalar
    /// slot's is its own width.
    pub packed: bool,
    /// Which of the statement's scalars this is, or `None` for a slot the row
    /// names past what the statement states.
    pub value: Option<u8>,
}

/// Why a launch could not become a dispatch.
///
/// Every variant is drift: a fire that cannot be dispatched was lowered
/// against a table or a binding other than the one loaded, and no retry
/// changes that.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Undispatchable {
    /// An operand did not resolve. See [`BindRefusal`].
    Unbound {
        /// The symbol whose launch refused.
        symbol: String,
        /// The traced op it came from.
        op: u32,
        /// Which rule could not be applied.
        why: BindRefusal,
    },
    /// The lowering states a symbol no ROUTINE claims, so nothing in this
    /// tree can fire it. `attn::split_qkv_bf16` is today's instance.
    ///
    /// It was `NoRow`, and the rename is the crossing: the contract lived in
    /// a `kernel!` row and the refusal was "no row declares this". There are
    /// no rows. `lowering::routine::crossed` matches the symbol against the
    /// stems the registry states, and a symbol no stem claims is a name
    /// nothing can dispatch -- which is the same refusal about a different
    /// statement, and it is worth saying which one is missing.
    Unclaimed {
        /// The symbol no routine claims.
        symbol: String,
        /// The traced op that named it.
        op: u32,
    },
    /// The routine composed an entrypoint the trace did not name.
    ///
    /// Both strings say which kernel to run and they were derived by
    /// unrelated readings -- the trace's by `model-compiler` from the
    /// checkpoint, the routine's by its body from this fire's
    /// [`Geometry`]. A disagreement means one of the two readings is wrong
    /// about the deployment, and running either is worse than running
    /// neither.
    Misspelled {
        /// What the trace named.
        symbol: String,
        /// The traced op that named it.
        op: u32,
        /// What the routine composed instead.
        composed: &'static str,
    },
    /// The routine declined the rectangle. See [`kernels::routine::Refusal`].
    ///
    /// Distinct from [`Undispatchable::Ungeometric`] in who spoke, not in what
    /// went wrong: a rule refuses from a table row, a routine refuses from its
    /// own body, and only the second can say which of ITS parameters was at
    /// fault by the name its signature gives it.
    Refused {
        /// The symbol whose launch refused.
        symbol: String,
        /// The traced op that named it.
        op: u32,
        /// What the routine would not accept.
        why: kernels::routine::Refusal,
    },
    /// The rectangle sits under a conditional region, so whether it runs is a
    /// question this walk cannot answer.
    ///
    /// `GuardMode::Union` keeps every arm and tags it, for a backend that can
    /// turn the tree back into conditional graph nodes. **Metal has no such
    /// API**: `Stepper` re-encodes every step, so the merged rectangle list IS
    /// the encode loop and `GuardMode::Resolve` is the mode that fits — the
    /// guards are answered before a rectangle exists.
    ///
    /// Reaching here means a fire was lowered in `Union` mode and handed to
    /// this walk, which would encode **every arm of every guard
    /// unconditionally**. That is not a slower answer, it is a different one.
    Conditional {
        /// The symbol whose rectangle is conditional.
        symbol: String,
        /// The traced op that named it.
        op: u32,
        /// Which region of [`Lowered::conds`] it sits under.
        cond: u32,
    },
}

/// Elements per row of the operand that sizes this launch.
///
/// The rectangle's operands are stated **inputs, outputs, then weights**
/// ([`Launch::args`]), and a weight carries no row width because its extent is
/// the tensor's. So the last operand with a width is the launch's last
/// *output*, and an output's row width is what every rule in the vocabulary
/// means by "width": a projection's output width, a norm's row width, an MLP's
/// intermediate.
///
/// Zero when the launch states no widthed operand at all, which leaves the
/// rule to refuse rather than this to guess.
fn sizing_width(lowered: &Lowered, launch: &Launch) -> u32 {
    widths(lowered, launch).next_back().unwrap_or(0)
}

/// Elements per row of the launch's FIRST widthed operand — its first input.
///
/// What sizes a statement that reads one packed buffer and writes several: no
/// one output spells the grid, because each is a fraction of the work.
fn input_width(lowered: &Lowered, launch: &Launch) -> u32 {
    widths(lowered, launch).next().unwrap_or(0)
}

/// The row widths this launch's operands state, in the trace's order.
fn widths<'a>(lowered: &'a Lowered, launch: &Launch) -> impl DoubleEndedIterator<Item = u32> + 'a {
    lowered.args[launch.args.start as usize..launch.args.end as usize]
        .iter()
        .filter_map(|arg| match arg {
            Arg::Arena { width, .. } | Arg::Named { width, .. } => Some(*width),
            // A raise has no row width to contribute; see `Arg::Raised`.
            Arg::Weight(_) | Arg::Raised { .. } => None,
        })
}

/// What a dispatch reads and what it may write, as byte ranges.
///
/// Ranges, not operands, because the question an encoder asks is whether two
/// launches can run at once and the answer is whether their bytes meet.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Touches {
    /// Every range the dispatch may read. A superset of nothing — a weight,
    /// a fire table and an input all sit here.
    pub reads: Vec<Slice>,
    /// Every range the dispatch may write. A subset of the operands, and
    /// **conservative in every direction the row leaves unstated.**
    pub writes: Vec<Slice>,
}

impl Touches {
    /// Every operand as both a read and a write.
    ///
    /// What a dispatch built by hand means: nothing told it which of its
    /// operands are results, so an encoder must order it against everything.
    /// Also what [`touches`] answers for a row that states no operands at all
    /// — those are bound positionally and have said nothing about direction,
    /// and guessing "no writes" would let them race silently.
    #[must_use]
    pub fn everything(args: &[BoundArg]) -> Self {
        let all: Vec<Slice> = args.iter().map(|a| a.slice).collect();
        Self {
            reads: all.clone(),
            writes: all,
        }
    }
}

/// Record a range, merging into an identical one rather than growing the set.
///
/// A fire binds the same weight and the same table over and over, and the sets
/// this feeds are scanned linearly.
/// Its only consumer is `bind::encode::Hazards::note`, and `bind` is behind
/// `metal-4` while this module is portable — so in the portable half nothing
/// calls this and `dead_code` is right about it. Stated as a conditional
/// allowance rather than silenced outright, because "unused in the half that
/// has no device" and "unused" are different facts and only the first one is
/// true here.
#[cfg_attr(not(feature = "metal-4"), allow(dead_code))]
pub(crate) fn merge(set: &mut Vec<Slice>, slice: Slice) {
    if slice.address == 0 || slice.bytes == 0 {
        return;
    }
    if let Some(seen) = set.iter_mut().find(|s| s.address == slice.address) {
        seen.bytes = seen.bytes.max(slice.bytes);
        return;
    }
    set.push(slice);
}

/// The GEMM tile a symbol names, as `(bm, bn)`.
///
/// `model-compiler` chooses the tile when it builds the entrypoint and writes
/// it into the name; nothing else in the trace carries it. So a routine that
/// needs the two numbers -- `quant::qmm_t` composes the same name back from
/// them -- gets them by reading the string, and the spelling check below is
/// what holds the two readings to each other.
///
/// `rsplit_once` rather than a suffix test, because the shipped names do not
/// all end at the tile: `affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32` and
/// `..._bm_32_bn_32_wm_1_wn_2` are both in `entrypoints.generated.txt`, and
/// the warp split after the tile is not part of it.
#[must_use]
pub fn named_tile(symbol: &str) -> Option<(u32, u32)> {
    let (_, tail) = symbol.rsplit_once("_bm_")?;
    let (bm, rest) = leading_number(tail)?;
    let (bn, _) = leading_number(rest.strip_prefix("_bn_")?)?;
    Some((bm, bn))
}

/// The decimal run at the front of `s`, and what follows it.
fn leading_number(s: &str) -> Option<(u32, &str)> {
    let end = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    Some((s[..end].parse().ok()?, &s[end..]))
}

/// What an arm is told about a launch, before its routine is asked anything.
///
/// The statement carries the widths and the layer, the fire carries the head
/// counts, and the SYMBOL carries the tile — three sources, one struct, and
/// no arm reaches past it. Read it to see what a routine can possibly know.
#[must_use]
pub fn facts_of(
    lowered: &Lowered,
    launch: &Launch,
    geometry: Geometry,
) -> crate::lowering::hold::Facts {
    // THIS STATEMENT'S affine point, which is the fire's unless the statement
    // is projecting a router gate and the gates arrived at their own.
    //
    // Read off the weight NAMES the statement carries, because that is the
    // only thing that distinguishes the gate projection: it is an ordinary
    // quantised matvec on an ordinary routine, and `affine_qmv_fast` has no
    // way to tell whose matrix it was handed. `model::binding` already puts
    // this name to a checkpoint to LEARN the point; this puts it to a
    // statement to spend it.
    let point = if geometry.router_bits > 0
        && lowered.args[launch.args.start as usize..launch.args.end as usize]
            .iter()
            .any(|a| match a {
                Arg::Weight(name) => crate::model::binding::ROUTER_POINT_AT_ANY_LAYER
                    .iter()
                    .any(|g| name.ends_with(g)),
                _ => false,
            }) {
        (geometry.router_group, geometry.router_bits)
    } else {
        (geometry.group, geometry.bits)
    };
    crate::lowering::hold::facts(
        launch,
        geometry,
        lowered.n_requests,
        sizing_width(lowered, launch),
        input_width(lowered, launch),
        named_tile(&lowered.kernels[launch.kernel as usize]),
        point,
    )
}

/// Every launch of a lowered fire, in order, as dispatches.
///
/// The whole executor. It does not branch on a family, a fire class or a
/// kernel — it walks what the compiler stated.
///
/// # Errors
///
/// The first [`Undispatchable`]. Nothing partial is returned: a fire that
/// cannot be dispatched whole would otherwise run its prefix and leave the
/// arena half-written, which is indistinguishable from a model that answers
/// nonsense.
pub fn plan<'a, S: Resolver>(
    lowered: &'a Lowered,
    frame: Frame,
    geometry: Geometry,
    resolver: &mut S,
) -> Result<Vec<Dispatch<'a>>, Undispatchable> {
    let mut out = Vec::with_capacity(lowered.launches.len());
    for launch in &lowered.launches {
        // A body may state more than one dispatch, which the table path could
        // not: a two-pass reduction is two entrypoints over one statement,
        // and a plane carrying only one would push the second back into the
        // lowering. So this extends rather than pushes.
        out.extend(plan_launch(lowered, launch, frame, geometry, resolver)?);
    }
    Ok(out)
}

/// ONE launch, as the dispatches its routine asked for.
///
/// [`plan`] is this in a loop. It is public because a launch is the unit a
/// test can put a single statement to, and because a caller that wants to
/// know which launches of a plane are dispatchable — rather than whether the
/// whole plane is — has to ask one at a time.
///
/// # Errors
///
/// [`Undispatchable::Conditional`] for a guarded launch, [`Undispatchable::Unclaimed`]
/// for a symbol no row names or no routine has an arm for, and whatever
/// [`plan_routine`] refuses.
pub fn plan_launch<'a, S: Resolver>(
    lowered: &'a Lowered,
    launch: &Launch,
    frame: Frame,
    geometry: Geometry,
    resolver: &mut S,
) -> Result<Vec<Dispatch<'a>>, Undispatchable> {
    // Every launch takes the ROUTINE path. There is no other one left to
    // take: all ninety-nine routines this backend builds have an arm, so a
    // symbol that reaches here without one is not a family waiting its turn
    // -- it is a name nothing in this tree can dispatch, and saying so is the
    // whole of what the fallback used to hide.
    let symbol = &lowered.kernels[launch.kernel as usize];
    // Before the lookup, because the guard is a fact about the LAUNCH and the
    // lookup is a question about the symbol. This backend re-encodes every
    // step and has no conditional graph node, so a guarded launch is
    // undispatchable whatever kernel it names -- and asked in the other order
    // an unclaimed symbol answered `Unclaimed`, which is true and is not the
    // reason.
    if launch.cond != Launch::NO_COND {
        return Err(Undispatchable::Conditional {
            symbol: symbol.clone(),
            op: launch.op,
            cond: launch.cond,
        });
    }
    // The trace names the fully INSTANTIATED entrypoint --
    // `silu_mul_bfloat16`, `affine_qmv_fast_bfloat16_gs_64_b_4` -- and a
    // routine is named after the row without the axis points. The registry
    // states the mapping itself, as a stem, so that no row is asked: routing
    // through `sig_in` was circular, and a family whose rows were deleted
    // would have made its own routines unreachable.
    let Some(routine) = crate::lowering::routine::crossed(symbol) else {
        return Err(Undispatchable::Unclaimed {
            symbol: symbol.clone(),
            op: launch.op,
        });
    };
    plan_routine(lowered, launch, routine, frame, geometry, resolver)
}

/// One launch of a crossed routine, as the dispatches its body asked for.
///
/// The arm resolves the statement's operands into handles and states the
/// routine's argument list; the body states the rectangle and the entrypoint.
/// Neither half can state the other's, which is the point.
///
/// # Errors
///
/// [`Undispatchable::Unbound`] for an operand the statement does not carry,
/// and [`Undispatchable::Refused`] for a rectangle the routine will not
/// launch — an extent of zero, a width no shader is compiled at.
fn plan_routine<'a, S: Resolver>(
    lowered: &'a Lowered,
    launch: &Launch,
    routine: &'static kernels::routine::Routine<kernels_metal::routine::Metal>,
    frame: Frame,
    geometry: Geometry,
    resolver: &mut S,
) -> Result<Vec<Dispatch<'a>>, Undispatchable> {
    let symbol = &lowered.kernels[launch.kernel as usize];
    let bound = bind(lowered, launch, frame, resolver).map_err(|why| Undispatchable::Unbound {
        symbol: symbol.clone(),
        op: launch.op,
        why,
    })?;
    let args = &lowered.args[launch.args.start as usize..launch.args.end as usize];
    let params: Vec<Option<u32>> = lowered.params
        [launch.params.start as usize..launch.params.end as usize]
        .iter()
        .map(|&p| Some(p))
        .collect();
    let facts = facts_of(lowered, launch, geometry);
    let refused = |why| Undispatchable::Refused {
        symbol: symbol.clone(),
        op: launch.op,
        why,
    };
    // HOW MANY OF THE WIDTHED OPERANDS ARE RESULTS -- read off the row.
    //
    // It used to be counted as the `BufMut` in the signature, which is wrong
    // for every routine whose writable arguments include STATE:
    // `attn::kv_append_paged` declares `k_pages` and `v_pages` as `BufMut`
    // and both are the KV pool, which the driver holds and no traced value
    // stands for. Counting them made a statement carrying two inputs and no
    // result read as one carrying no input and two results, so the arm asked
    // for `input(0)` and was told the statement does not carry one -- every
    // fire, every layer, on every text in the suite.
    //
    // The repair was to ASK the arm, by running it twice: once over an
    // undivided statement purely to count the `output(i)` asks, and once over
    // the split that count implies. That worked because the arm knows the
    // difference a type could not express -- a result is what it asks
    // `output(i)` for, the pool is what it asks `kv(..)` for.
    //
    // The type expresses it now. `OutSlot<0, BufMut>` is a result and
    // `Env<BufMut>` is not, which is `Side::Declared` against `Side::OfType`,
    // and the count is a filter over a `&'static [Side]` the row already
    // carries. `kv_append_paged` keeps both its `BufMut` and declares
    // neither.
    //
    // What makes the count trustworthy rather than merely plausible is
    // `routine::tests::every_arm_binds_the_slot_its_signature_states`: it
    // runs all ninety-one arms and asserts, for each, that the number of
    // declared results equals the highest result index the arm asks for plus
    // one. Those two agree only while each result appears in a signature
    // exactly once, which is a property nothing else enforces -- so the probe
    // was not deleted until the thing that replaces it was checked against it.
    //
    // THE COLUMN IT IS READ OFF CHANGED AND THE FACT DID NOT. `Side` was a
    // second column restating what the mark beside it already said, and it is
    // deleted; the `Source` a mark resolves to carries the same fact and
    // cannot drift from it. `Out` resolves to `Slot(Kind::Out, _)` and `InOut`
    // to `Alias(_, _)` -- one address wearing both slots, which is still a
    // result the statement declares.
    let results = routine
        .sources
        .iter()
        .filter(|s| {
            matches!(
                s,
                Some(kernels::Source::Slot(kernels::Kind::Out, _) | kernels::Source::Alias(..))
            )
        })
        .count();
    let (ins, outs, weights) = crate::lowering::hold::split(args, results);
    let mut handles =
        crate::lowering::hold::Handles::new(&bound.args, &ins, &outs, &weights, &params, resolver);
    // THE VIEWS OUTLIVE THE BODY. `bind` boxes a host view per `Ty::Raised`
    // operand (`In<Struct<KvCache>>` and its siblings) and hands its ADDRESS
    // into `values`; the body reads it below, so the holder sits on this
    // frame until `planner.finish()`. A recorded fire replays encoded
    // dispatches and runs no body, so no address outlives this function.
    let mut views = crate::lowering::views::Views::over(args, &ins);
    let values = crate::lowering::bind::bind(
        routine.args,
        routine.sources,
        &mut handles,
        facts,
        &mut views,
    )
    .map_err(refused)?;
    let bound = handles.bound().to_vec();
    // THE STAGED BLOCK IS COPIED OUT BEFORE THE MOVE. `Staged<'_>` borrows the
    // word run, and the cell below takes the `Handles` by value; the planner
    // holds the block for the whole body, so it cannot be a borrow of
    // something the cell owns.
    let (block, words) = {
        let s = handles.staged();
        (s.block, s.words.to_vec())
    };
    // `RefCell`, BECAUSE A BODY MAY STILL ASK. With `Env` out of the parameter
    // list a fact only the fire can answer is no longer bound into `values`
    // before the body runs -- the body asks for it, and answering MINTS.
    let handles = core::cell::RefCell::new(handles);
    let staged = crate::lowering::hold::Staged {
        block,
        words: &words,
    };
    let planner = crate::lowering::routine::Planner::new(
        routine,
        &bound,
        staged,
        launch.layers.clone(),
        launch.op,
    )
    .answering(&handles, facts);
    (routine.body)(&planner, &values).map_err(refused)?;
    let plan = planner.finish();
    // The two spellings, held to each other.
    //
    // The trace names the fully instantiated entrypoint and the body composes
    // one from the facts its arm supplied, so for a statement that becomes
    // ONE dispatch the two strings are the same string derived twice -- once
    // by `model-compiler` from the checkpoint, once here from `Geometry`.
    // Nothing else compares them, and the failure they would otherwise hide
    // is the worst kind this backend has: a `_gs_64_b_4` gather over a
    // `_gs_128_b_8` table does not fault, it returns fluent nonsense, because
    // the two pack to identical extents.
    //
    // Only for a single dispatch. A body may state two -- a two-pass
    // reduction is two entrypoints over one statement -- and then neither is
    // "the" symbol the trace named.
    if let [only] = plan.as_slice()
        && only.symbol != symbol
    {
        {
            return Err(Undispatchable::Misspelled {
                symbol: symbol.clone(),
                op: launch.op,
                composed: only.symbol,
            });
        }
    }
    Ok(plan)
}

/// The distinct `(file, entry point, stamp)` triples a dispatch list needs
/// compiled.
///
/// In first-use order, deduplicated: a fire naming one symbol 28 times
/// compiles it once. This is what the device half hands to
/// `Compiler::compile_batch`, and it is here rather than there because it is a
/// property of the list, not of the GPU.
///
/// The stamp rides with the pair rather than being looked up later, because
/// there is nowhere to look it up: it is composed at the fire, by the routine
/// body, and this list is the only thing that survives the body.
#[must_use]
pub fn pipelines_needed<'a>(
    dispatches: &[Dispatch<'a>],
) -> Vec<(&'static str, &'a str, &'static str)> {
    let mut out: Vec<(&'static str, &'a str, &'static str)> = Vec::new();
    for d in dispatches {
        let point = (d.file, d.symbol, d.stamp);
        if !out.contains(&point) {
            out.push(point);
        }
    }
    out
}

// -- The vocabulary and the width it implies ---------------------------------
//
// Both were in `model/run.rs` until the gate became a feature and a portable
// test naming them stopped compiling. Neither touches a device: one is a
// `&'static` table and the other counts operands. That is
// `.wiki/driver/real-metal-north-star.md` §7's own instruction -- *move the
// arithmetic, not the crate* -- and the build is what noticed.

/// The widest operand count any statement of a fire binds, plus its scalars.
///
/// An argument table is created with a fixed bind count and a binding past it
/// is an error rather than a silent no-op — so the table has to be built for
/// the widest statement in the fire, not for a guess.
#[must_use]
pub fn table_width(dispatches: &[Dispatch<'_>]) -> usize {
    dispatches
        .iter()
        // One slot per operand, and ONE more for the packed params — the
        // scalars ride as a single struct, which is what every shader in the
        // tree takes (`constant RouterParams&` and its siblings).
        .map(|d| {
            let params = if d.params.is_empty() {
                0
            } else {
                d.param_slots.iter().map(|p| p.slot + 1).max().unwrap_or(0)
            };
            d.args.len().max(params)
        })
        .max()
        .unwrap_or(1)
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lowering::executor::FireTable;
    use model_ir::trace::ValueId;

    /// Answers every name, so a test is about the walk rather than the store.
    #[derive(Default)]
    struct Anything;

    impl Resolver for Anything {
        fn weight(&mut self, _: &str) -> Option<Slice> {
            Some(Slice {
                address: 0x1000,
                bytes: 1 << 30,
            })
        }
        fn named(&mut self, _: ValueId) -> Option<Slice> {
            Some(Slice {
                address: 0x2000,
                bytes: 1 << 30,
            })
        }
    }

    /// One launch of `symbol` over `rows`, with the args given.
    fn one(symbol: &str, rows: u32, args: Vec<Arg>) -> Lowered {
        Lowered {
            // A hand-built lowering states no per-argument rows; zero is "no opinion".
            arg_rows: Vec::new(),
            // One request: these fixtures state one row.
            n_requests: 1,
            // A dispatch fixture is one launch, not a whole fire, so it has no
            // exit to state.
            readout: None,

            launches: vec![Launch {
                kernel: 0,
                rows: 0..rows,
                layers: 0..1,
                op: 7,
                cond: Launch::NO_COND,
                params: 0..0,
                args: 0..args.len() as u32,
                peel: None,
            }],
            kernels: vec![symbol.to_string()],
            rectangles: 1,
            arena_bytes: 4096,
            value_offset: Vec::new(),
            value_owner: Vec::new(),
            epilogue_gather: usize::MAX,
            epilogue_norm: usize::MAX,
            args,
            params: Vec::new(),
            structural: Vec::new(),
            residue: Vec::new(),
            conds: Vec::new(),
            // A dispatch fixture states no attention schedule to raise.
            preps: Vec::new(),
        }
    }

    fn frame() -> Frame {
        Frame {
            arena: Slice {
                address: 0x8000,
                bytes: 4096,
            },
        }
    }

    #[test]
    fn the_sizing_width_is_the_last_output_not_the_first_input() {
        // Args are stated inputs, outputs, then weights, and a weight has no
        // row width. So the last widthed operand is the output, and that is
        // what every rule means by "width".
        let low = one(
            "sized",
            3,
            vec![
                Arg::Arena {
                    at: 0,
                    width: 11,
                    bytes: 2,
                },
                Arg::Arena {
                    at: 64,
                    width: 22,
                    bytes: 2,
                },
                Arg::Weight("w".into()),
            ],
        );
        assert_eq!(sizing_width(&low, &low.launches[0]), 22);
    }

    #[test]
    fn a_launch_states_its_rows_and_the_fire_states_the_rest() {
        let low = one(
            "sized",
            5,
            vec![Arg::Arena {
                at: 0,
                width: 64,
                bytes: 2,
            }],
        );
        let geometry = Geometry {
            q_heads: 16,
            kv_heads: 4,
            head_dim: 128,
            ..Geometry::default()
        };
        let launch = &low.launches[0];
        let facts = facts_of(&low, launch, geometry);
        assert_eq!(facts.rows, 5, "the rectangle states the rows");
        assert_eq!(facts.width, 64, "the operand states the width");
        assert_eq!(facts.q_heads(), 16, "the fire states the rest");
    }

    #[test]
    fn a_symbol_with_no_row_names_itself_and_the_op_it_came_from() {
        let low = one(
            "attn::split_qkv_bf16",
            1,
            vec![Arg::Arena {
                at: 0,
                width: 8,
                bytes: 16,
            }],
        );
        assert_eq!(
            plan(&low, frame(), Geometry::default(), &mut Anything),
            Err(Undispatchable::Unclaimed {
                symbol: "attn::split_qkv_bf16".into(),
                op: 7
            })
        );
    }

    #[test]
    fn a_fire_that_cannot_be_dispatched_whole_returns_nothing_partial() {
        // A prefix of dispatches would leave the arena half-written, which is
        // indistinguishable from a model that answers nonsense.
        let mut low = one(
            "sized",
            1,
            vec![Arg::Arena {
                at: 0,
                width: 8,
                bytes: 2,
            }],
        );
        low.kernels.push("no_rule".into());
        let second = Launch {
            kernel: 1,
            ..low.launches[0].clone()
        };
        low.launches.push(second);
        assert!(plan(&low, frame(), Geometry::default(), &mut Anything).is_err());
    }

    #[test]
    fn a_conditional_rectangle_refuses_because_metal_cannot_answer_a_guard() {
        // `GuardMode::Union` keeps every arm for a backend that can build
        // conditional graph nodes. Metal has no such API and re-encodes every
        // step, so a union-lowered fire reaching this walk would encode every
        // arm of every guard unconditionally — a different answer, not a
        // slower one.
        //
        // `sized` is a symbol no stem claims, and that is the second half of
        // what this pins: the guard is refused BEFORE the routine lookup, so
        // the error names the guard rather than the name.
        let mut low = one(
            "sized",
            1,
            vec![Arg::Arena {
                at: 0,
                width: 8,
                bytes: 2,
            }],
        );
        low.launches[0].cond = 3;
        assert_eq!(
            plan(&low, frame(), Geometry::default(), &mut Anything),
            Err(Undispatchable::Conditional {
                symbol: "sized".into(),
                op: 7,
                cond: 3
            })
        );
    }

    #[test]
    fn one_symbol_named_many_times_is_compiled_once() {
        let d = Dispatch {
            symbol: "sized",
            file: "f.metal",
            stamp: "",
            grid: [1, 1, 1],
            threadgroup: [1, 1, 1],
            args: Vec::new(),
            touches: Touches::default(),
            param_slots: vec![ParamSlot {
                slot: 0,
                at: 0,
                bytes: 4,
                packed: true,
                value: Some(0),
            }],
            params: Vec::new(),
            layers: 0..1,
            op: 0,
        };
        let list = vec![
            d.clone(),
            d.clone(),
            Dispatch {
                symbol: "other",
                ..d.clone()
            },
        ];
        assert_eq!(
            pipelines_needed(&list),
            vec![("f.metal", "sized", ""), ("f.metal", "other", "")]
        );

        // AND THE STAMP IS PART OF THE POINT, not a decoration hanging off
        // it. Two dispatches can name one symbol in one file and still be two
        // things to compile: the stamp is the line that makes the symbol
        // EXIST, so a list carrying two of them is a list asking for two
        // different bodies under one name. Deduplicating on the pair alone
        // would hand the compiler the first and let the second run against
        // it.
        let stamped = vec![
            Dispatch {
                stamp: "PIE_STAMP_f(\"sized\", 32)",
                ..d.clone()
            },
            Dispatch {
                stamp: "PIE_STAMP_f(\"sized\", 64)",
                ..d.clone()
            },
            Dispatch {
                stamp: "PIE_STAMP_f(\"sized\", 32)",
                ..d.clone()
            },
        ];
        assert_eq!(
            pipelines_needed(&stamped),
            vec![
                ("f.metal", "sized", "PIE_STAMP_f(\"sized\", 32)"),
                ("f.metal", "sized", "PIE_STAMP_f(\"sized\", 64)"),
            ],
            "one symbol under two stamps is two pipelines, and the repeat of \
             the first is still one"
        );
    }

    /// A crossed symbol takes the routine path, and the plan is the body's.
    ///
    /// `sample` is the first family wired, and its row is why: it states no
    /// `launch` rule, so `eval` refuses it `Unstated` and the table path has
    /// never been able to dispatch `argmax_logits` at all. The rectangle below
    /// is therefore the FIRST statement of this kernel's grid anywhere in the
    /// driver -- there is no prior behaviour to preserve, which is the
    /// cheapest possible place to prove a seam.
    #[test]
    fn a_crossed_symbol_plans_through_its_routine() {
        let args = vec![
            Arg::Arena {
                at: 0,
                width: 128,
                bytes: 2,
            },
            Arg::Arena {
                at: 512,
                width: 4,
                bytes: 4,
            },
            Arg::Arena {
                at: 1024,
                width: 1,
                bytes: 4,
            },
            Arg::Arena {
                at: 1536,
                width: 1,
                bytes: 4,
            },
        ];
        let mut lowered = one("argmax_logits_bfloat16", 3, args);
        // `rows` is the statement's now, not a fact the tables answer.
        lowered.launches[0].params = 0..1;
        lowered.params = vec![3];

        assert!(
            crate::lowering::routine::crossed(&lowered.kernels[0]).is_some(),
            "the stem resolves the spelling the TRACE states"
        );

        let plan = plan(&lowered, frame(), Geometry::default(), &mut Anything)
            .expect("the routine plans it");

        assert_eq!(plan.len(), 1, "one statement, one dispatch");
        let got = &plan[0];
        assert_eq!(got.symbol, "argmax_logits_bfloat16", "the whole spelling");
        assert_eq!(got.file, "sample/argmax.metal");
        assert_eq!(got.grid[1], 3, "one row of the rectangle per token");
        assert_eq!(got.grid[0], got.threadgroup[0], "one group reduces a row");
        // The trace's order is inputs then results -- logits, params, token,
        // flag. The kernel's order interleaves them, and the arm is what
        // turns one into the other.
        assert_eq!(
            got.args.iter().map(|a| a.slice.address).collect::<Vec<_>>(),
            vec![0x8000, 0x8000 + 1024, 0x8000 + 512, 0x8000 + 1536],
            "the operands, in the KERNEL's order"
        );
    }

    /// An empty rectangle is a refusal from the routine, not a dispatch of
    /// nothing.
    #[test]
    fn a_routine_refusing_the_rectangle_refuses_the_walk() {
        let mut lowered = one(
            "argmax_logits_bfloat16",
            0,
            vec![
                Arg::Arena {
                    at: 0,
                    width: 128,
                    bytes: 2,
                },
                Arg::Arena {
                    at: 512,
                    width: 4,
                    bytes: 4,
                },
                Arg::Arena {
                    at: 1024,
                    width: 1,
                    bytes: 4,
                },
                Arg::Arena {
                    at: 1536,
                    width: 1,
                    bytes: 4,
                },
            ],
        );
        // An empty rectangle is now something the STATEMENT says, and the
        // routine refuses on reading it rather than on the launch's range.
        lowered.launches[0].params = 0..1;
        lowered.params = vec![0];

        let why = plan(&lowered, frame(), Geometry::default(), &mut Anything)
            .expect_err("no row to sample");
        assert!(
            matches!(
                why,
                Undispatchable::Refused {
                    why: kernels::routine::Refusal::Empty { what: "rows" },
                    ..
                }
            ),
            "{why:?}"
        );
    }

    /// Answers the fire's tables too, so an operand that comes from one is
    /// distinguishable from an operand that does not.
    #[derive(Default)]
    struct Tables;

    impl Resolver for Tables {
        fn weight(&mut self, name: &str) -> Option<Slice> {
            // Distinct per name, so a swap of two weight operands is visible
            // rather than an equality between two copies of `0x1000`.
            Some(Slice {
                address: 0x1000 + 0x100 * u64::from(name.len() as u32),
                bytes: 1 << 20,
            })
        }
        fn named(&mut self, _: ValueId) -> Option<Slice> {
            Some(Slice {
                address: 0x2000,
                bytes: 1 << 20,
            })
        }
        fn fire(&mut self, which: FireTable) -> Option<Slice> {
            Some(Slice {
                address: 0x4000 + 0x100 * which as u64,
                bytes: 1 << 20,
            })
        }
    }

    /// The fire's own axes, for the families that read them.
    fn geom() -> Geometry {
        Geometry {
            q_heads: 4,
            kv_heads: 4,
            head_dim: 64,
            group: 64,
            bits: 4,
            ..Geometry::default()
        }
    }

    /// The two rotations that had no row now dispatch, and `neox_strided`
    /// still refuses the statement that cannot say its stride.
    ///
    /// `neox_prop_mb` is gemma's prefill rotation and `neox_strided` is the
    /// packed-QKV one. Both rows were bare -- `kernel!(neox_prop_mb
    /// "neox_prop_mb", file = ..., axes = &[BF16])` and nothing else -- so
    /// `eval` had no rule and refused, which on gemma meant the prefill's
    /// rotation silently did not happen. A routine states its own bindings,
    /// so there is nothing left to leave out.
    ///
    /// The stride is the exception. Every other scalar here has a fire-wide
    /// fallback that is right for a single-shape deployment; a row pitch does
    /// not, because a pitch equal to the row width is exactly the case this
    /// kernel is NOT for. So it is read from the statement or refused.
    #[test]
    fn the_two_rotations_no_row_could_reach_now_dispatch() {
        let scale = 1.0f32.to_bits();
        let base = 500_000.0f32.to_bits();
        // TWICE, BECAUSE THE ROTATION IS IN PLACE. `x` is an `InOut` and
        // claims an operand slot AND a result slot at one address, so the
        // statement lists the buffer on both sides -- inputs then outputs,
        // which is the order `Handles` splits `args` by. One entry gave the
        // binder an output and no input, and `Source::Alias(0, 0)` asks for
        // both.
        let cell = Arg::Arena {
            at: 0,
            width: 256,
            bytes: 2,
        };
        // POSITIONS IS AN OPERAND NOW. It was `ctx.ask::<_, keys::Positions>()`
        // -- a fact the driver's tables answered -- until the no-ask series
        // put it in the signature, so the statement lists it between the
        // input and output halves that `Handles` splits `args` into.
        let positions = Arg::Arena {
            at: 2048,
            width: 5,
            bytes: 4,
        };
        let arena = vec![cell.clone(), positions, cell];

        let mut prop = one("neox_prop_mb_bfloat16", 5, arena.clone());
        prop.launches[0].params = 0..5;
        prop.params = vec![scale, base, 64, 32, 5];
        let plan =
            plan(&prop, frame(), geom(), &mut Tables).expect("the routine states its own bindings");
        assert_eq!(
            plan[0].grid,
            [16, 4, 5],
            "half the rotation, per head, per row"
        );

        // The stride, missing.
        let mut strided = one("neox_strided_bfloat16", 5, arena.clone());
        strided.launches[0].params = 0..4;
        strided.params = vec![scale, base, 64, 32];
        // Four words where the signature wants six: the pitch is the fifth and
        // `rows` the sixth, and a run that stops short of the pitch is exactly
        // the statement this test says cannot dispatch.
        let why =
            super::plan(&strided, frame(), geom(), &mut Tables).expect_err("no pitch, no dispatch");
        assert!(
            matches!(why, Undispatchable::Refused { .. }),
            "a stride the statement does not carry is a refusal, not the row \
                 width: {why:?}"
        );

        // The stride, stated, and narrower than the row it strides over.
        strided.launches[0].params = 0..6;
        strided.params = vec![scale, base, 64, 32, 128, 5];
        let why = super::plan(&strided, frame(), geom(), &mut Tables)
            .expect_err("a pitch narrower than the row makes consecutive rows overlap");
        assert!(
            matches!(
                why,
                Undispatchable::Refused {
                    why: kernels::routine::Refusal::Narrow { at: 128, .. },
                    ..
                }
            ),
            "{why:?}"
        );

        strided.params = vec![scale, base, 64, 32, 512, 5];
        let plan = super::plan(&strided, frame(), geom(), &mut Tables)
            .expect("a pitch wider than the row tiles");
        assert_eq!(plan[0].grid, [16, 4, 5]);
    }

    /// Every strided norm this backend has now dispatches, and none of them
    /// could before.
    ///
    /// `rms_strided_row`, `rms_strided_head_row`, `gated_rms`,
    /// `gated_rms_strided` and `residual_add_strided` were all bare rows --
    /// `kernel!(name "name", file = ..., axes = &[BF16])` and nothing more --
    /// so `eval` refused every one. That is the whole prefill path over a
    /// packed layout: a QK-norm across a prompt, a gated head norm, a
    /// residual over non-contiguous rows.
    #[test]
    fn the_strided_norms_no_row_could_reach_now_dispatch() {
        let eps = 1e-5f32.to_bits();
        let arena = |at: usize| Arg::Arena {
            at,
            width: 256,
            bytes: 2,
        };
        // Four 64-wide reductions inside each 256-wide row, five rows.
        //
        // ONE BLOCK NO LONGER SERVES ALL FIVE. It did while `heads` and `rows`
        // were facts the routines asked the driver's tables for; the no-ask
        // series made both positional, so the run a statement carries is now
        // the signature's own and the two gated norms take a different one
        // (`eps, vd, heads, rows`) from the two rms rows
        // (`eps, axis, w_stride, plus_one, gain, rows`).
        let rms = vec![eps, 64, 1, 0, 1.0f32.to_bits(), 5];
        let gated = vec![eps, 64, 4, 5];

        for (symbol, args, params, grid, tg) in [
            (
                "rms_strided_row",
                vec![arena(0), Arg::Weight("norm".into()), arena(2048)],
                rms.clone(),
                // One threadgroup per ROW: the base is `gid * row_pitch`, so
                // a row holds one norm and the axis only sizes the group.
                [16 * 5, 1, 1],
                [16, 1, 1],
            ),
            (
                "rms_strided_head_row",
                vec![arena(0), Arg::Weight("norm".into()), arena(2048)],
                rms.clone(),
                // Four heads on their own axis, five rows on a third.
                [16, 4, 5],
                [16, 1, 1],
            ),
            (
                "gated_rms",
                vec![
                    arena(0),
                    arena(1024),
                    Arg::Weight("norm".into()),
                    arena(2048),
                ],
                gated.clone(),
                // The pool's shape, not the statement's: four 64-wide value
                // heads. The old rule had no row axis at all.
                [64, 4, 5],
                [64, 1, 1],
            ),
            (
                "gated_rms_strided",
                vec![
                    arena(0),
                    arena(1024),
                    Arg::Weight("norm".into()),
                    arena(2048),
                ],
                gated.clone(),
                [64, 4, 5],
                [64, 1, 1],
            ),
            (
                "residual_add_strided",
                vec![arena(0), arena(1024), arena(2048)],
                // The pitch and the row count, neither of which anything else
                // can supply now that both are the signature's.
                vec![512, 5],
                [256, 5, 1],
                [256, 1, 1],
            ),
        ] {
            let mut lowered = one(&format!("{symbol}_bfloat16"), 5, args);
            lowered.launches[0].params = 0..params.len() as u32;
            lowered.params = params;

            let by_routine = plan(&lowered, frame(), geom(), &mut Tables)
                .unwrap_or_else(|why| panic!("{symbol}: the routine states its bindings: {why:?}"));
            assert_eq!(by_routine[0].grid, grid, "{symbol}: the rectangle");
            assert_eq!(by_routine[0].threadgroup, tg, "{symbol}: the group");
        }
    }

    /// A routine handed the wrong axis facts composes a name the trace never
    /// stated, and the walk refuses rather than gathering over the wrong
    /// table.
    ///
    /// This is the failure the spelling check exists for and it is otherwise
    /// silent: a `_gs_64_b_4` kernel over a `_gs_128_b_8` weight reads the
    /// same extent, faults nothing, and returns fluent garbage. Here the
    /// `Geometry` says 64/4 while the trace says 128/8 -- which is what a
    /// misread of `MetalBinding`'s quantization bytes would produce.
    #[test]
    fn a_routine_composing_a_name_the_trace_did_not_state_is_refused() {
        let mut lowered = one(
            "embed_gather_mb_4bit_bfloat16_gs_128_b_8",
            3,
            vec![
                Arg::Weight("embed".into()),
                Arg::Weight("embed_scales".into()),
                Arg::Weight("embed_biases".into()),
                // The tokens to gather, which the routine used to ask for and
                // now takes as an operand.
                Arg::Arena {
                    at: 2048,
                    width: 3,
                    bytes: 4,
                },
                Arg::Arena {
                    at: 0,
                    width: 64,
                    bytes: 2,
                },
            ],
        );
        // `[group, bits]`, which is the run this routine's two `Const` marks
        // claim. It was `[64]` alone -- one number where two are read -- and a
        // statement short of a scalar is refused before the spelling this case
        // is about can be composed at all. 64 and 4 are what the geometry
        // below says, so the composed name is `_gs_64_b_4` against the traced
        // `_gs_128_b_8`, which is the disagreement.
        //
        // TWICE NOW, and the second time the same sentence explains it: the
        // no-ask series added `token_ids` and `rows` to this signature, so the
        // run is three words and the operand list carries the tokens. A
        // statement short of either is refused before the spelling can be
        // composed, which would make this case pass for the wrong reason.
        lowered.launches[0].params = 0..3;
        lowered.params = vec![64, 4, 3];

        let why = plan(
            &lowered,
            frame(),
            Geometry {
                group: 64,
                bits: 4,
                ..Geometry::default()
            },
            &mut Tables,
        )
        .expect_err("the composed spelling is not the traced one");
        assert!(
            matches!(
                why,
                Undispatchable::Misspelled {
                    composed: "embed_gather_mb_4bit_bfloat16_gs_64_b_4",
                    ..
                }
            ),
            "{why:?}"
        );
    }
}
