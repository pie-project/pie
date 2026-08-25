//! Turning a statement's operands into descriptor ranges.
//!
//! `model_compiler::lower` handed a driver a flat list of `Arg`s and said
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
//! run past. `tests/arena.rs` kept that number and is deleted with the
//! lowering it measured; the number is written down here instead.
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
//! # `extent`, `resolve` and `bind` STOOD HERE, and what carries the claim now
//!
//! All three took a `model_compiler::lower::Launch` -- the flat argument row a
//! lowering handed a driver -- and turned it into a run of
//! [`crate::device::Bound`]s. There
//! is no `Launch`. [`crate::walk::fire::Fire::rect`] answers where a value
//! lives from the STATEMENT, and [`crate::baker::marks`] is the region type it
//! answers in.
//!
//! The extent arithmetic is not retired with them, it MOVED, and the
//! measurement that pinned it is worth keeping: an arena operand is `rows ×
//! width × bytes`, and over every arena operand six real texts produced in
//! both fire classes -- 14,324 of them -- every extent landed inside the arena
//! and the tightest fit had **zero** bytes to spare. That zero is the useful
//! part. It says the formula is not a lower bound that happens to be safe: if
//! it were an under-estimate the slack would never reach zero, and if it were
//! an over-estimate it would have run past.
//!
//! # What this module does not do
//!
//! It resolves operands. It does not build the parameter side of the call --
//! [`params_from`] decides only WHERE a run of scalar words goes, push block
//! or storage struct, and the reachable symbols split almost evenly on it, so
//! neither answer could be assumed.

use model_ir::plan::ValueId;

use crate::device::Buffer;

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
/// Two of the three operand kinds a statement can name are not the plan's to
/// hold: a weight by its trace name and a seam value by its id. Both are the
/// driver's own tables, so both are asked for rather than looked up.
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
    /// The refusal that produces is `kernels::plane::Refusal::Absent`, raised
    /// by [`crate::walk::fire::Fire`] where the cache is asked for, which
    /// names the layer as well as the fact.
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
    /// bucketed because that grid was RECORDED: `crate::replay` re-submitted a
    /// decode's command buffer across tokens, so a number that moved every
    /// token would re-plan every token. A power-of-two bucket moves a handful
    /// of times in a sequence's life. `replay` is deleted and the bucket is
    /// kept, because the reason survives the mechanism: a grid that changes
    /// every token is a pipeline barrier's worth of re-planning every token.
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

impl FireTable {
    /// The table a runtime NAME lands in, or `None` for a name this driver
    /// stages nothing under.
    ///
    /// ONE TRANSLATION AND NOT TWO. The no-ask channel reaches this driver
    /// twice: a plan may mint a stream as a runtime VALUE, which
    /// `crate::runtime::Streams` mapped by value id -- deleted with the
    /// `ForwardPlan` it read -- and a claim body may ask for one by name
    /// through
    /// [`Encode::staged`](kernels_vulkan::plane::Encode::staged). Both are the
    /// same question -- *which buffer holds `positions` this fire?* -- and a
    /// second list of the same five names is a second thing to keep right.
    /// `Streams::of` walked the plan's runtime table through THIS, so the two
    /// halves could not disagree; with it gone, [`crate::walk::fire::Fire`]'s
    /// `runtime` is the one reader left and this is still the one list.
    ///
    /// # What is deliberately absent
    ///
    /// `qo_indptr`, `row_valid` and `first_token` are tier-1 names this driver
    /// stages nothing for, and they are left out rather than pointed at
    /// something plausible: a launch that names one is refused by name instead
    /// of reading a stand-in of zeros fluently.
    ///
    /// `rope.yarn_inv_freq` is the interesting absence. This driver DOES stage
    /// a rope frequency table and it is not that one --
    /// [`crate::rope::frequencies`] raises llama-3's piecewise-in-wavelength
    /// rescale, where YaRN's ramp between `beta_fast` and `beta_slow` is a
    /// different ladder entirely. Answering `rope.yarn_inv_freq` with
    /// `rope.frequencies` would rotate a YaRN deployment against the wrong
    /// frequencies and report success, which is the failure this whole table
    /// exists to make impossible.
    #[must_use]
    pub fn named(name: &str) -> Option<Self> {
        Some(match name {
            "positions" => Self::Positions,
            "token_ids" => Self::TokenIds,
            "request_of_token" => Self::RequestOfToken,
            "sampling_indices" => Self::SamplingIndices,
            // Tier-2 on the vocabulary, tier-1 on this driver: the rope table
            // is staged every fire and the rope routines take it as an
            // operand. The name is dotted because the load-time signature
            // check refuses an undotted runtime name outside the tier-1 floor
            // -- the plane owns the spelling, not the vocabulary.
            "rope.frequencies" => Self::RopeFrequencies,
            _ => return None,
        })
    }
}

// `SCALE_PREFIX`, `pub enum Unbindable`, `extent`, `resolve` and `bind` STOOD
// HERE -- 224 lines, and all five were about a `model_compiler::lower::Arg`.
//
// `bind` walked one `Launch`'s argument span and `resolve` turned each `Arg`
// into a `Bound`: an `Arena` operand against `extent`'s `rows × width ×
// bytes`, a `Named` or a `Weight` against the `Resolve` below, a `Raised`
// against the arena as a PLACEHOLDER so the positional list stayed aligned
// with the trace. Nothing partial ever came back, and that was the point --
// a dispatch with some ranges resolved would read whatever the descriptor set
// happened to hold in the others, which on a reused set is the PREVIOUS
// launch's operand and not garbage, so it would look plausible.
//
// `Unbindable` named the five ways it could fail: `PastArena` (a rectangle
// running past the arena the plan sized), `UnknownWeight`, `UnknownNamed`,
// `NoDriverResource`, `Constant` (a `scale.`-prefixed weight name --
// `dsl::cuda::scalar_mul` put a scalar in an operand slot so the launch's
// arity held, and Vulkan cannot bind the zero-length region metal calls
// honest) and `Unaddressable` (what `Bound::within` said).
//
// WHAT REPLACES THE WHOLE OF IT is one statement's worth of the same
// arithmetic done by the walk: `walk::fire::Fire::rect` resolves a `ValueId`
// against the arena, the banks and the fire's tables, and `baker::marks`
// mints the region. The refusals are `kernels::plane::Refusal`, raised where
// the operand is asked for rather than collected into a per-launch enum, and
// `crate::device::Failed` for the addressability half -- which is the one
// check that is still Vulkan's alone and still lives on `Bound::within`.

// `pub enum Slot<'a>` and `pub enum Unlayoutable` STOOD HERE, one at each end
// of this file, and they were two halves of a function that had ALREADY LEFT:
// `descriptors`, which cut a row's slot list down to the module's declared
// bindings. It went at `5bd280339`, its two vocabularies did not, and both
// have had no reader since -- so this is a deletion the culling found rather
// than one the baker forced.
//
// `Slot` was the middle term the positional binder did not have: `Buffer` (a
// range), `Params` (the slot a row reserves for its scalars) and `Nothing` (a
// slot the row states and nothing fills). A plan stated its operands in TRACE
// order -- inputs, then outputs, then weights -- and a shader binds them in
// the order its kernel row states, and those are not the same order.
// `Unlayoutable` named the three ways the two readings disagreed.
//
// THE DESCRIPTOR HOLES THEY EXISTED FOR ARE STILL REAL AND ARE ANSWERED
// ELSEWHERE. `kv_append_paged` declares six bindings nothing reads, kept so
// that the rest of its row stays at the positions a shared ring ABI put them.
// A `#[claims]` body writes its argument list in the SHADER's order -- see
// `baker::dispatch::Dispatch::args`, which is why the `reorder` pass is gone
// -- and what a module actually declares is `spirv::Declared::bindings`, which
// `device::Pipelines::get` takes the maximum of against the caller's count.

/// Where a launch's scalars go.
///
/// A plan states its operands and its scalars separately, which is already the
/// shape Vulkan wants -- descriptors on one side, a push block on the other.
/// But only half the reachable kernels take them that way. `tests/arena.rs`
/// measured it before it was deleted with the lowering:
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
    /// this named it; the row packer is deleted on all three shader backends,
    /// and what applies the same rule from a body's own arguments is
    /// [`crate::baker::encode`]'s `lay_out` feeding [`params_from`].
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

/// Place a run of scalar words the way its module wants them.
///
/// PUSH BLOCK OR STORAGE STRUCT, and the decision is read off the compiled
/// module rather than off a naming convention: of the reachable symbols,
/// seven take their scalars as a push block and six take them as a plain
/// struct in a storage buffer of their own. Neither answer could be assumed
/// and the split is almost even.
///
/// `stated` is one word per scalar in signature order, two for a `Usize` with
/// the low half first and the pair aligned to an even word --
/// `baker::encode`'s `lay_out` is what produces it and
/// `baker::dispatch::Dispatch::params` states the convention.
///
/// # `params` and `push_from` STOOD BESIDE THIS
///
/// `params` was the same call with a `model_compiler::lower::Lowered` and a
/// `Launch` in front of it, slicing the launch's own scalar span out of the
/// plan; there is no plan-wide scalar run to slice.
///
/// `push_from` placed a routine's arguments MEMBER by member rather than word
/// by word, and it went with `kernels_vulkan::routine::ArgValue`. Its reason
/// is worth keeping because this function still cannot do what it did: this
/// one zips `stated` against [`crate::spirv::Declared::push_offsets`] and
/// writes FOUR BYTES at each, which reads every member as one 32-bit word.
/// That is true of every scalar a lowering stated and false of a 64-bit
/// stride. `kv_append`'s block is `{ int head_dim; PIE_STRIDE k_head_stride;
/// PIE_STRIDE k_seq_stride; }` -- three members at offsets 0, 8 and 16 -- and
/// the words that fill it are six. Six against three is [`Misplaced::Count`],
/// and had the counts happened to agree the writer would still have put four
/// bytes where eight go. The 64-bit strides belong to the contiguous KV path,
/// which no text in this tree launches, which is why a block that could not be
/// filled at all went unnoticed for as long as it did.
///
/// # Errors
///
/// [`Misplaced::Count`] when neither of the module's two shapes can hold what
/// the caller states.
pub fn params_from(stated: &[u32], declared: &crate::spirv::Declared) -> Result<Params, Misplaced> {
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

// `impl core::fmt::Display for Unlayoutable` and its `Error` impl stood
// below the enum. Both went with it.
