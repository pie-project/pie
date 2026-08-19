//! Turning a lowering's operands into descriptor ranges.
//!
//! [`model_compiler::lower`] hands a driver a flat list of [`Arg`]s and says
//! *"bind these"*. On Metal that is nearly free: `setBuffer:offset:` takes an
//! address and a byte offset and no length at all, so `driver-metal`'s binder
//! resolves each operand to its base plus its offset, reports the rest of the
//! arena as the extent, and is done. The extent is not load-bearing there
//! because nothing reads it.
//!
//! Vulkan has no such call. A storage descriptor is a buffer, an offset AND a
//! range, and all three are checked. So this module has to answer a question
//! Metal never asked: **how many bytes is this operand?**
//!
//! # The extent is in the plan, and it is exact
//!
//! It is `rows × width × bytes`: the launch states its row count, and the
//! operand states its row width and its element size. Measured over every
//! arena operand six real texts produce in both fire classes -- 14324 of them
//! -- every extent lands inside the arena, and the tightest fit has **zero**
//! bytes to spare. An operand ends exactly where the arena does.
//!
//! That zero is the useful part. It says the formula is not a lower bound that
//! happens to be safe, it is the real extent: if it were an under-estimate the
//! slack would never reach zero, and if it were an over-estimate it would have
//! run past. `tests/arena.rs` keeps that number.
//!
//! # Why the range matters more here than a length usually does
//!
//! The arena is ONE buffer holding every activation. `VK_WHOLE_SIZE` on an
//! operand at offset `at` therefore covers every tensor allocated after it,
//! and a kernel indexing past its own rows would read or write a neighbour
//! silently -- `tests/device.rs` demonstrates exactly that, and demonstrates
//! that a real range confines it instead, because `robustBufferAccess` is on.
//! So the range is not bookkeeping. It is the only thing standing between a
//! stray index and another tensor's bytes.
//!
//! # What this module does not do
//!
//! It resolves operands. It does not build the parameter side of the call:
//! seven of the reachable symbols want their scalars as push constants and six
//! want them as a buffer of their own, which `tests/arena.rs` measures and a
//! later layer will act on.

use model_compiler::lower::{Arg, Launch, Lowered};
use model_ir::trace::ValueId;

use crate::device::{Bound, Buffer};

/// The frame's arena: one buffer, every activation.
#[derive(Clone, Copy, Debug)]
pub struct Arena<'a> {
    /// The buffer the offsets are into.
    pub buffer: &'a Buffer,
    /// How many bytes of it the plan was allowed to place into.
    ///
    /// Stated separately from the buffer's own size because a driver may hold
    /// a larger arena than a given fire needs, and the question an operand
    /// asks is whether it fits the PLAN's arena. A buffer big enough to
    /// contain a mistake still contains one.
    pub bytes: u64,
}

/// Where the operands this crate cannot resolve come from.
///
/// Two of the three [`Arg`] kinds name something the plan does not hold: a
/// weight by its trace name and a seam value by its id. Both are the driver's
/// own tables, so both are asked for rather than looked up.
///
/// Takes `&self` rather than `&mut self` -- unlike the Metal one, which can
/// return a copied address -- because a Vulkan binding borrows the buffer it
/// names, and the borrow has to outlive the call that produced it.
pub trait Resolve {
    /// The buffer holding a weight, by the name the trace states.
    fn weight(&self, name: &str) -> Option<&Buffer>;
    /// The buffer holding a seam value the backend binds by name.
    fn named(&self, value: ValueId) -> Option<&Buffer>;

    /// The KV cache for one layer, keys or values.
    ///
    /// STATE, not an operand: no traced value stands for it, so no plan
    /// mentions it and no arena holds it. `kv_append_paged` names both and
    /// the paged attentions read both.
    ///
    /// Defaulted to `None` so that a resolver serving a text without paged
    /// attention does not have to state a method it will never be asked for.
    /// The refusal that produces is [`Unbindable::NoDriverResource`], which
    /// names what was missing.
    fn kv(&self, _layer: u16, _values: bool) -> Option<&Buffer> {
        None
    }

    /// One of the fire's own numbers.
    ///
    /// A pool's shape, not a statement's scalar. A text that stated its page
    /// size would be right for one deployment and silently wrong for the
    /// next, so the row names the number and the driver answers it.
    fn number(&self, _which: FireNumber) -> Option<u32> {
        None
    }

    /// One of the fire's own tables.
    ///
    /// Also state. The kernel row names WHICH; this forwards the name and
    /// never reads what it means, which is what keeps the driver from having
    /// an opinion about a table's contents.
    fn table(&self, _which: FireTable) -> Option<&Buffer> {
        None
    }
}

/// The fire-wide numbers a kernel row may name.
///
/// Scalars rather than buffers, so these are appended to the parameter run
/// where the row places them -- not given a descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FireNumber {
    /// How many rows one KV page holds.
    KvPageSize,
    /// The stride between heads in the cache.
    KvHeadStride,
    /// The stride between positions in the cache.
    KvSeqStride,
    /// The pitch of this fire's mask rectangle, in keys.
    ///
    /// Zero when the fire states no mask, and the shader reads that as "apply
    /// the causal rule alone". Not a fact about the cache, which is why
    /// [`crate::resources::Pool`] answers it and `Shape` does not.
    AttentionMaskStride,
    /// The fire's longest history, ROUNDED UP TO A POWER OF TWO.
    ///
    /// One past the largest position any row of the fire attends from, which
    /// is how many keys the busiest decode row walks. It decides how many ways
    /// [`kernels_vulkan::attn::decode_splits`] cuts the key range, and it is
    /// bucketed because that grid is RECORDED: `crate::replay` re-submits a
    /// decode's command buffer across tokens, so a number that moved every
    /// token would re-plan every token. A power-of-two bucket moves a handful
    /// of times in a sequence's life.
    ///
    /// Zero from a resolver that does not know -- and zero means one split,
    /// which is the single-pass path this backend has always taken.
    KvHistoryBucket,
}

/// The fire-wide tables a kernel row may name.
///
/// Transcribed from `driver-metal`, which reached the same list by the same
/// route: these are the [`kernels::Source`] variants that are neither an
/// operand of the statement nor a scalar, so nothing but the driver can
/// supply them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FireTable {
    /// The token id at each row.
    TokenIds,
    /// The position of each row in its sequence.
    Positions,
    /// Which request each row belongs to.
    RequestOfToken,
    /// The pages each request's KV occupies.
    KvPageIndices,
    /// Where each request's run starts in [`Self::KvPageIndices`].
    KvPageIndptr,
    /// The attention mask.
    AttentionMask,
    /// Whether the mask is in force.
    AttentionMaskEnabled,
    /// The page each row's KV is written to.
    KvWritePage,
    /// The offset within that page.
    KvWriteOffset,
    /// The rope frequency table.
    RopeFrequencies,
    /// Which rows the readout samples.
    SamplingIndices,
    /// Scratch for the flash decode's partial softmaxes.
    ///
    /// `splits * rows * q_heads * (head_dim + 2)` floats: an unnormalised
    /// weighted-V accumulator per `(split, row, head)`, then a `(max,
    /// sum_exp)` pair each. Written by every workgroup of the split pass and
    /// read by the fold, and by nothing else -- so it is never zeroed and
    /// never read back.
    ///
    /// A driver resource and not an operand, for the same reason the KV cache
    /// is one: no traced value stands for it, so no plan mentions it and no
    /// arena holds it.
    AttnPartials,
}

impl std::fmt::Display for Unbindable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PastArena { at, extent, arena } => {
                write!(f, "{extent} bytes at {at} runs off an arena of {arena}")
            }
            Self::UnknownWeight(name) => write!(f, "no buffer holds the weight `{name}`"),
            Self::UnknownNamed(value) => write!(f, "no buffer holds the named value {value}"),
            Self::Constant(name) => {
                write!(
                    f,
                    "`{name}` is a dispatch constant riding a weight slot, not a buffer"
                )
            }
            Self::NoOperand => write!(f, "the row names an operand this statement does not state"),
            Self::NoKvCache { layer, values } => write!(
                f,
                "no {} cache for layer {layer}",
                if *values { "value" } else { "key" }
            ),
            Self::NoDriverResource(what) => write!(f, "the driver holds no {what:?} table"),
            Self::Unaddressable(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for Unbindable {}

/// The marker a constant rides the weight-name slot under.
///
/// Transcribed from `driver-metal`: `dsl::cuda::scalar_mul` puts a scalar in
/// an operand slot so the launch's arity holds, and states that no binder
/// looks for it. Metal binds a zero-length region and calls that honest.
///
/// This backend cannot. A descriptor of range zero is invalid in Vulkan --
/// `Bound::within` refuses it as an overrun -- so a slot under this prefix is one
/// the caller must fill some other way, and saying so is the only correct
/// answer. It is [`Unbindable::Constant`] rather than a silent empty range.
const SCALE_PREFIX: &str = "scale.";

/// Why an operand could not become a descriptor range.
#[derive(Clone, Debug, PartialEq)]
pub enum Unbindable {
    /// An arena operand's rectangle runs past the arena the plan sized.
    ///
    /// Never seen in a real lowering -- and the tightest real operand ends
    /// exactly AT the end, so a plan one row wider would land here.
    PastArena {
        /// Where in the arena the operand starts.
        at: usize,
        /// The rectangle it wanted: `rows × width × bytes`.
        extent: u64,
        /// What the plan said the arena holds.
        arena: u64,
    },
    /// The plan names a weight the resolver does not hold.
    UnknownWeight(
        /// The name the plan stated.
        String,
    ),
    /// The plan names a seam value the resolver does not bind.
    UnknownNamed(
        /// The value id the plan stated.
        ValueId,
    ),
    /// The slot holds a dispatch constant, not a buffer.
    ///
    /// See `SCALE_PREFIX`. The refusal is the finding: there is no range
    /// that means "nothing" in Vulkan.
    Constant(
        /// The constant's name, with the `scale.` prefix removed.
        String,
    ),
    /// The row names an operand of the statement that the statement does not
    /// have.
    ///
    /// A row states the shape of every deployment its kernel serves, so a
    /// statement reaching only part of that shape is ordinary -- but a slot
    /// left holding nothing is a descriptor the shader reads, so it is a
    /// refusal rather than a gap. A row says "gap" by stating `None`.
    NoOperand,
    /// The row names the KV cache and the resolver does not hold one.
    NoKvCache {
        /// Which layer was asked for.
        layer: u16,
        /// Values rather than keys.
        values: bool,
    },
    /// The row names a fire table the resolver does not hold.
    NoDriverResource(
        /// Which table.
        FireTable,
    ),
    /// The device cannot address the offset or extent this operand needs.
    ///
    /// Carries what [`Bound::within`] said, so an alignment failure and a
    /// length failure stay distinguishable at the point they are reported.
    Unaddressable(
        /// What the range check said.
        crate::device::Failed,
    ),
}

/// How many bytes one operand of this launch covers.
///
/// `None` for the operand kinds whose extent is not the plan's to state: a
/// weight is as big as its tensor and a seam value is as big as the backend
/// made it, and in both cases the resolver's buffer already says so.
#[must_use]
pub fn extent(arg: &Arg, launch: &Launch) -> Option<u64> {
    match arg {
        Arg::Arena { width, bytes, .. } => {
            let rows = u64::from(launch.rows.end - launch.rows.start);
            // Saturating rather than wrapping: a plan with an absurd width
            // should produce a refusal below, not a small number that binds.
            Some(
                rows.saturating_mul(u64::from(*width))
                    .saturating_mul(u64::from(*bytes)),
            )
        }
        Arg::Named { .. } | Arg::Weight(_) => None,
    }
}

/// Resolve one operand into a descriptor range.
///
/// `min_offset` is the device's `minStorageBufferOffsetAlignment`, which the
/// range is checked against here rather than at dispatch: an operand that
/// cannot be addressed is a fact about the plan and the device together, and
/// the earliest place both are known is here.
///
/// # Errors
///
/// [`Unbindable`], naming which of the rules could not be applied.
pub fn resolve<'a, R: Resolve>(
    arg: &Arg,
    launch: &Launch,
    arena: Arena<'a>,
    resolver: &'a R,
    min_offset: u64,
) -> Result<Bound<'a>, Unbindable> {
    match arg {
        Arg::Arena { at, .. } => {
            let extent = extent(arg, launch).expect("an arena operand states its rectangle");
            let at64 = *at as u64;
            // Checked before `Bound::within` so the refusal names the ARENA. The
            // buffer may well be larger than the plan's arena, in which case
            // `Bound::within` would accept a range the plan had no right to.
            if at64.saturating_add(extent) > arena.bytes {
                return Err(Unbindable::PastArena {
                    at: *at,
                    extent,
                    arena: arena.bytes,
                });
            }
            Bound::within(arena.buffer, at64, extent, min_offset).map_err(Unbindable::Unaddressable)
        }
        Arg::Named { value, .. } => resolver
            .named(*value)
            .map(Bound::whole)
            .ok_or(Unbindable::UnknownNamed(*value)),
        Arg::Weight(name) => {
            if let Some(rest) = name.strip_prefix(SCALE_PREFIX) {
                return Err(Unbindable::Constant(rest.to_owned()));
            }
            resolver
                .weight(name)
                .map(Bound::whole)
                .ok_or_else(|| Unbindable::UnknownWeight(name.clone()))
        }
    }
}

/// Resolve every operand of one launch, in the order the plan states them.
///
/// # Errors
///
/// The first [`Unbindable`] any operand produces, and the index of the operand
/// that produced it. Nothing partial comes back: a dispatch with some ranges
/// resolved is a dispatch that would read whatever the descriptor set happened
/// to hold in the others, which on a reused set is the previous launch's
/// operand and not garbage -- so it would look plausible.
pub fn bind<'a, R: Resolve>(
    lowered: &Lowered,
    launch: &Launch,
    arena: Arena<'a>,
    resolver: &'a R,
    min_offset: u64,
) -> Result<Vec<Bound<'a>>, (usize, Unbindable)> {
    let span = launch.args.start as usize..launch.args.end as usize;
    let mut bound = Vec::with_capacity(span.len());
    for (i, arg) in lowered.args[span].iter().enumerate() {
        bound.push(resolve(arg, launch, arena, resolver, min_offset).map_err(|e| (i, e))?);
    }
    Ok(bound)
}

/// What one of a module's binding slots holds.
///
/// The middle term the positional binder did not have. A plan states its
/// operands in TRACE order -- inputs, then outputs, then weights -- and a
/// shader binds them in the order its kernel row states, and those are not
/// the same order. `rms_single_row`'s row is `In(0), Weight(0), Out(0),
/// params`, and `norm/rms.slang` decorates exactly that; the trace hands over
/// `In(0), Out(0), Weight(0)`.
#[derive(Clone, Debug, PartialEq)]
pub enum Slot<'a> {
    /// A range of a buffer: an operand, a weight, or a driver resource.
    Buffer(Bound<'a>),
    /// The slot the row reserves for this launch's scalars.
    ///
    /// The caller allocates and fills it -- see [`params`] -- because a
    /// buffer needs a device and this module is arithmetic.
    Params,
    /// A slot the row states and nothing fills.
    ///
    /// Source `None`. `kv_append_paged` has six, kept so that the rest of
    /// its row stays at the positions a shared ring ABI put them; the module
    /// leaves them as descriptor holes and nothing reads them.
    Nothing,
}

/// Where a launch's scalars go.
///
/// A plan states its operands and its scalars separately, which is already the
/// shape Vulkan wants -- descriptors on one side, a push block on the other.
/// But only half the reachable kernels take them that way. `tests/arena.rs`
/// measures the split over every symbol three real texts launch: six take
/// their scalars as a push block, and six take them as a plain struct in a
/// storage buffer of their own. Seven, once descriptor holes are subtracted:
/// `affine_qmv_routed` reads as short of a buffer until the slot no shader
/// reads is discounted, and then it is an ordinary push-block kernel.
///
/// Which one a kernel wants is not a naming convention and not a list to keep
/// up to date. It is read off the compiled module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Params {
    /// The scalars are the module's push-constant block.
    ///
    /// The bytes are laid out at the offsets the SHADER declares, not packed
    /// end to end, because those are not always the same thing and the
    /// difference is silent. `crate::lowering::pack` stood for that reason and
    /// this named it; the row packer is deleted on all three shader backends
    /// and `lowering::routine::bind` applies the same rule from a body's own
    /// arguments.
    Push(Vec<u8>),
    /// The scalars are a struct in a storage buffer, at this binding.
    ///
    /// The caller allocates it, writes these bytes, and binds it at `at`. It
    /// is the caller's because a buffer needs a device and this module is
    /// arithmetic; keeping it that way is what lets the split be decided on a
    /// machine with no GPU.
    Block {
        /// The struct's bytes, exactly as long as the shader's block.
        bytes: Vec<u8>,
        /// Which binding to put it at.
        at: usize,
    },
    /// The module declares neither, and the launch states no scalars.
    None,
}

/// Why a launch's scalars could not be placed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Misplaced {
    /// The module wants its scalars somewhere, and the two somewheres
    /// disagree about how many there are.
    ///
    /// This is the defect the whole parameter side exists to catch, and it has
    /// no symptom on this backend: `robustBufferAccess` is on, so a block
    /// short of what the shader reads returns ZEROS rather than faulting.
    /// `tests/device.rs` shows a dispatch accepting one, producing 256 zeros,
    /// and the validation layer saying nothing at all.
    Count {
        /// How many scalars the plan states.
        stated: usize,
        /// How many the module's push block holds.
        push: usize,
        /// The sizes of every knowable block the module declares, in bytes.
        blocks: Vec<u32>,
    },
    /// The row addresses the KV cache CONTIGUOUSLY, and this driver's pool is
    /// paged.
    ///
    /// [`kernels::Source::Named(<kernels::keys::KvHeadStride as kernels::keys::Fact>::KEY)`] and [`kernels::Source::Named(<kernels::keys::KvSeqStride as kernels::keys::Fact>::KEY)`]
    /// appear on exactly the rows that walk the cache with two strides and no
    /// page table -- `kv_append`, `sdpa_vector_decode`,
    /// `sdpa_vector_decode_swa`. The paged writer beside them takes
    /// `page_size` and `n_kv_heads` and consults the page table instead.
    ///
    /// `resources::Shape` allocates `[page, token, head, dim]` for every fire
    /// this driver runs, so `Shape::number` can answer both strides for it --
    /// and that is the defect, not the fix. The arithmetic is right only while
    /// a fire's pages happen to be physically consecutive from zero: true of
    /// one freshly-allocated sequence, false of the second. It reads real
    /// memory at every step and attends to the WRONG TOKENS, and
    /// `robustBufferAccess` has nothing to say because nothing is out of
    /// bounds.
    ///
    /// `crates/model` reached the same conclusion from the other side and
    /// stopped emitting these rows (*"no contiguous attention over a paged
    /// pool"*), which guards the texts that exist. The pool's layout is the
    /// DRIVER's fact, and this is the last place that knows it.
    Contiguous {
        /// Which operand of the row, counting from zero.
        at: usize,
        /// The operand's name, as the row spells it.
        name: &'static str,
    },
    /// The row names a scalar this driver cannot work out.
    ///
    /// A NAMED refusal where there used to be a zero, twice: `_ => 0` in the
    /// packed-struct arm and `_ => {}` in the scalar run. Zero is the worst
    /// available default because it is PLAUSIBLE -- a width is a row pitch, so
    /// at zero the shader writes element 0 of every row or the rule builds a
    /// grid of nothing, and `vkQueueSubmit` returns success over a buffer that
    /// kept whatever it held.
    ///
    /// This file had already met the shape and repaired one instance of it:
    /// there is an arm for `Source::OutWidth` and a note saying it was split
    /// out "when `kernels::Source::OutWidth` arrived". The coverage stopped
    /// there, so the next row naming `InWidth`, `OutElements` or `InElements`
    /// would have got the zero.
    ///
    /// The scalar run splits on the operand's KIND rather than on a list of
    /// source names, which is what closes it: a buffer contributes no scalar
    /// because binding it is `reorder`'s job, and everything else must resolve
    /// or be named here. So a new BUFFER source cannot enter the run by
    /// omission and a new DERIVED one cannot enter it as a zero.
    ///
    /// It deliberately does NOT answer the row family. `OutRows` is a value's
    /// leading extent -- `Rows` for a token-shaped value, a load-time constant
    /// for a fixed one, a padded block-major count for the MoE aligned path --
    /// so answering the fire's rows would be right for most values and
    /// silently wrong for exactly the ones the source exists to distinguish.
    Unresolved {
        /// Which operand of the row, counting from zero.
        at: usize,
        /// The operand's name, as the row spells it.
        name: &'static str,
        /// The [`kernels::Source`] variant, rendered -- `Source` is not `Eq`,
        /// so it cannot be carried whole in a type that is.
        source: String,
    },
}

/// Place one launch's scalars the way its module wants them.
///
/// # Errors
///
/// [`Misplaced::Count`] when neither of the module's two shapes can hold what
/// the plan states.
pub fn params(
    lowered: &Lowered,
    launch: &Launch,
    declared: &crate::spirv::Declared,
) -> Result<Params, Misplaced> {
    let stated = &lowered.params[launch.params.start as usize..launch.params.end as usize];
    params_from(stated, declared)
}

pub(crate) fn params_from(
    stated: &[u32],
    declared: &crate::spirv::Declared,
) -> Result<Params, Misplaced> {
    // Asked in this order because push is the stronger claim: it accounts for
    // every descriptor as well as every scalar, and a module that declares a
    // push block of the right size is not also hiding a parameter buffer.
    if declared.push_offsets.len() == stated.len() {
        if stated.is_empty() {
            return Ok(Params::None);
        }
        // Sized from the block's own extent rather than from four bytes per
        // scalar: `vkCmdPushConstants` takes a size, and a block with a gap
        // in it needs the gap written or the range does not cover the members
        // after it.
        let end = declared
            .push_offsets
            .iter()
            .map(|o| *o as usize + 4)
            .max()
            .unwrap_or(0);
        let mut bytes = vec![0u8; end];
        for (word, offset) in stated.iter().zip(&declared.push_offsets) {
            let at = *offset as usize;
            bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
        }
        return Ok(Params::Push(bytes));
    }

    // Found by SIZE and not by position. Looking for it at the binding one
    // past the operand count is the obvious guess and is wrong for two of the
    // six: `combine_sorted` binds its 12-byte block at 3 of 5 and `route_sort`
    // its 28-byte block at 4 of 6, each with an operand after it. Where a
    // parameter block sits is the kernel's own ABI.
    //
    // Searching by size is also what CONSTRAINS how a shader may spell the
    // block, which is not obvious from here and bites anyone editing the
    // kernel tree. A block has to declare a fixed extent, so it cannot be a
    // Slang `StructuredBuffer<T>` -- that is a runtime array, reflection
    // reports no size for it (correctly), and every launch would be refused
    // with "n scalars stated, room for 0". The tree therefore keeps one
    // GLSL-syntax construct, `PIE_PARAMS` in `kernels/common/bf16.slang`,
    // which is why `build.rs` passes `-allow-glsl`.
    let want = stated.len() as u32 * 4;
    if want > 0
        && let Some(at) = declared.block_bytes.iter().position(|b| *b == Some(want))
    {
        let mut bytes = Vec::with_capacity(want as usize);
        for word in stated {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        return Ok(Params::Block { bytes, at });
    }

    if stated.is_empty() && declared.push_offsets.is_empty() {
        return Ok(Params::None);
    }

    Err(Misplaced::Count {
        stated: stated.len(),
        push: declared.push_offsets.len(),
        blocks: declared.block_bytes.iter().flatten().copied().collect(),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    /// A `Buffer` this test can name without a device.
    ///
    /// Binding never dereferences the handle -- it produces offsets and
    /// lengths against it -- so a zeroed one is enough to ask every question
    /// in this module, and asking them without a GPU is the point.
    fn buffer(bytes: u64) -> Buffer {
        Buffer::placeholder(bytes)
    }

    #[derive(Default)]
    struct Store {
        weights: BTreeMap<String, Buffer>,
        named: BTreeMap<ValueId, Buffer>,
    }

    impl Resolve for Store {
        fn weight(&self, name: &str) -> Option<&Buffer> {
            self.weights.get(name)
        }
        fn named(&self, value: ValueId) -> Option<&Buffer> {
            self.named.get(&value)
        }
    }

    /// A launch over `rows` rows with `n` operands starting at zero.
    fn launch(rows: u32, n: u32) -> Launch {
        Launch {
            kernel: 0,
            rows: 0..rows,
            layers: 0..1,
            op: 0,
            args: 0..n,
            params: 0..0,
            peel: None,
            cond: 0,
        }
    }

    #[test]
    fn an_arena_operands_extent_is_its_rectangle_and_not_one_row() {
        let arg = Arg::Arena {
            at: 0,
            width: 128,
            bytes: 2,
        };
        // The distinction Metal never had to draw: one row is 256 bytes, and
        // a 64-row prefill launch of the same operand covers 64 times that.
        assert_eq!(extent(&arg, &launch(1, 1)), Some(256));
        assert_eq!(extent(&arg, &launch(64, 1)), Some(64 * 256));
    }

    #[test]
    fn a_weight_and_a_seam_value_do_not_get_their_extent_from_the_plan() {
        assert_eq!(extent(&Arg::Weight("w".into()), &launch(1, 1)), None);
        assert_eq!(
            extent(
                &Arg::Named {
                    value: 0,
                    width: 8,
                    bytes: 2,
                },
                &launch(1, 1)
            ),
            None
        );
    }

    #[test]
    fn an_arena_operand_is_bound_at_its_offset_for_its_rectangle() {
        let buf = buffer(1 << 20);
        let arena = Arena {
            buffer: &buf,
            bytes: 1 << 20,
        };
        let arg = Arg::Arena {
            at: 512,
            width: 64,
            bytes: 2,
        };
        let store = Store::default();
        let b = resolve(&arg, &launch(4, 1), arena, &store, 16).expect("bindable");
        assert_eq!(b.offset(), 512);
        // Four rows of 64 elements at 2 bytes -- NOT the 512 one row would
        // give, and not the rest of the arena `VK_WHOLE_SIZE` would give.
        assert_eq!(b.len(), 512);
    }

    #[test]
    fn an_operand_whose_rectangle_runs_past_the_arena_is_refused() {
        let buf = buffer(1 << 20);
        // The buffer is a megabyte; the PLAN said the arena is 1024 bytes.
        // The refusal has to come from the plan's number, or a driver holding
        // a generously sized arena would accept an operand that addresses
        // another fire's bytes.
        let arena = Arena {
            buffer: &buf,
            bytes: 1024,
        };
        let arg = Arg::Arena {
            at: 768,
            width: 64,
            bytes: 2,
        };
        let err =
            resolve(&arg, &launch(4, 1), arena, &Store::default(), 16).expect_err("runs past");
        assert_eq!(
            err,
            Unbindable::PastArena {
                at: 768,
                extent: 512,
                arena: 1024
            }
        );
    }

    #[test]
    fn an_operand_the_device_cannot_address_from_is_refused_as_such() {
        let buf = buffer(1 << 20);
        let arena = Arena {
            buffer: &buf,
            bytes: 1 << 20,
        };
        // 260 is inside the arena and a multiple of 4, so the plan and Metal
        // are both content. It is not a multiple of 256.
        let arg = Arg::Arena {
            at: 260,
            width: 64,
            bytes: 2,
        };
        let err =
            resolve(&arg, &launch(1, 1), arena, &Store::default(), 256).expect_err("misaligned");
        assert!(
            matches!(
                err,
                Unbindable::Unaddressable(crate::device::Failed::Unaligned { .. })
            ),
            "an offset the device cannot use is not the same refusal as one the \
             plan oversized: {err:?}"
        );
    }

    #[test]
    fn a_weight_and_a_seam_value_come_from_the_resolver_whole() {
        let buf = buffer(1 << 20);
        let arena = Arena {
            buffer: &buf,
            bytes: 1 << 20,
        };
        let mut store = Store::default();
        store.weights.insert("layer.3.q_proj".into(), buffer(4096));
        store.named.insert(7, buffer(64));

        let w = resolve(
            &Arg::Weight("layer.3.q_proj".into()),
            &launch(1, 1),
            arena,
            &store,
            16,
        )
        .expect("held");
        // Whole, because the plan does not state a weight's extent and the
        // tensor's own size is the right answer.
        assert_eq!((w.offset(), w.len()), (0, 4096));

        let n = resolve(
            &Arg::Named {
                value: 7,
                width: 8,
                bytes: 2,
            },
            &launch(1, 1),
            arena,
            &store,
            16,
        )
        .expect("bound");
        assert_eq!((n.offset(), n.len()), (0, 64));
    }

    #[test]
    fn a_name_the_resolver_does_not_hold_is_named_in_the_refusal() {
        let buf = buffer(1 << 20);
        let arena = Arena {
            buffer: &buf,
            bytes: 1 << 20,
        };
        let store = Store::default();
        assert_eq!(
            resolve(
                &Arg::Weight("layer.3.q_proj".into()),
                &launch(1, 1),
                arena,
                &store,
                16
            )
            .expect_err("not held"),
            Unbindable::UnknownWeight("layer.3.q_proj".into())
        );
        assert_eq!(
            resolve(
                &Arg::Named {
                    value: 7,
                    width: 8,
                    bytes: 2,
                },
                &launch(1, 1),
                arena,
                &store,
                16
            )
            .expect_err("not bound"),
            Unbindable::UnknownNamed(7)
        );
    }

    /// The one place this backend must refuse where Metal proceeds.
    #[test]
    fn a_slot_holding_a_dispatch_constant_cannot_be_a_range_on_this_backend() {
        let buf = buffer(1 << 20);
        let arena = Arena {
            buffer: &buf,
            bytes: 1 << 20,
        };
        // Metal binds `{address: 0, bytes: 0}` here and calls it honest,
        // because a zero-length Metal binding is legal and unread. Vulkan has
        // no such descriptor, so the honest answer is a refusal that says
        // which constant the caller still owes.
        assert_eq!(
            resolve(
                &Arg::Weight("scale.rope_theta".into()),
                &launch(1, 1),
                arena,
                &Store::default(),
                16
            )
            .expect_err("a constant is not a range"),
            Unbindable::Constant("rope_theta".into())
        );
    }

    #[test]
    fn a_launch_binds_whole_or_not_at_all() {
        let buf = buffer(1 << 20);
        let arena = Arena {
            buffer: &buf,
            bytes: 1 << 20,
        };
        let mut store = Store::default();
        store.weights.insert("held".into(), buffer(64));
        let lowered = lowered(vec![
            Arg::Arena {
                at: 0,
                width: 8,
                bytes: 2,
            },
            Arg::Weight("held".into()),
            Arg::Weight("absent".into()),
        ]);

        let (i, err) = bind(&lowered, &launch(1, 3), arena, &store, 16).expect_err("one is absent");
        // The index is reported because a refusal that only names the weight
        // cannot say WHICH operand slot the dispatch was going to leave stale.
        assert_eq!(i, 2);
        assert_eq!(err, Unbindable::UnknownWeight("absent".into()));

        // The same launch minus the absent operand binds all of what remains.
        assert_eq!(
            bind(&lowered, &launch(1, 2), arena, &store, 16)
                .expect("both held")
                .len(),
            2
        );
    }

    #[test]
    fn operands_come_back_in_the_order_the_plan_states_them() {
        let buf = buffer(1 << 20);
        let arena = Arena {
            buffer: &buf,
            bytes: 1 << 20,
        };
        // Three arena operands at distinguishable offsets. Descriptor slots
        // are positional, so an order this crate rearranged would hand every
        // kernel its inputs shuffled and no test of one operand would notice.
        let lowered = lowered(
            [1024usize, 256, 2048]
                .into_iter()
                .map(|at| Arg::Arena {
                    at,
                    width: 8,
                    bytes: 2,
                })
                .collect(),
        );
        let store = Store::default();
        let bound = bind(&lowered, &launch(1, 3), arena, &store, 16).expect("bindable");
        assert_eq!(
            bound.iter().map(Bound::offset).collect::<Vec<_>>(),
            [1024, 256, 2048]
        );
    }

    /// A `Lowered` holding nothing but the operands under test.
    fn lowered(args: Vec<Arg>) -> Lowered {
        Lowered {
            launches: Vec::new(),
            kernels: Vec::new(),
            rectangles: 0,
            arena_bytes: 0,
            value_offset: Vec::new(),
            value_owner: Vec::new(),
            epilogue_gather: usize::MAX,
            epilogue_norm: usize::MAX,
            args,
            structural: Vec::new(),
            residue: Vec::new(),
            params: Vec::new(),
            n_requests: 0,
            conds: Vec::new(),
            readout: None,
        }
    }
}

/// The refusals [`descriptors`] produces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unlayoutable {
    /// The row states more slots than the module declares, and the excess is
    /// not all unbound.
    ///
    /// A row states every deployment its kernel serves, so a row longer than
    /// one module is ordinary -- but only where the tail is `None`. A
    /// buffer past the end is a buffer nothing can hold.
    Overlong {
        /// How many slots the row states.
        stated: usize,
        /// How many the module declares.
        module: u32,
    },
    /// The scalar block's slot is not where the row puts its parameters.
    ///
    /// The module says which binding is the block; the row says which operand
    /// is `Const { v: 0 }`. Both are read off separately, so their disagreement is
    /// a finding rather than a fact -- and it means the shader would read its
    /// scalars out of an operand.
    BlockElsewhere {
        /// Where the SPIR-V puts the block.
        module: usize,
        /// Where the row's parameters land.
        row: usize,
    },
    /// The row leaves a slot unbound that the module decorates and reads.
    ///
    /// Measured to be exactly the descriptor holes, on both the modules that
    /// have any: `affine_qmv_routed` 1 and 1, `kv_append_paged` 6 and 6. So a
    /// mismatch means one of the two readings is wrong.
    Unfilled {
        /// The slot the row leaves empty.
        at: usize,
    },
}

impl core::fmt::Display for Unlayoutable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Overlong { stated, module } => {
                write!(
                    f,
                    "the row states {stated} slots and the module declares {module}"
                )
            }
            Self::BlockElsewhere { module, row } => {
                write!(
                    f,
                    "the module's block is binding {module} and the row's is {row}"
                )
            }
            Self::Unfilled { at } => write!(f, "binding {at} is read and nothing fills it"),
        }
    }
}

impl core::error::Error for Unlayoutable {}
