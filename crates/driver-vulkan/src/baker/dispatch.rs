//! One encodable dispatch: everything a command buffer needs, and nothing that
//! needs a command buffer to compute.
//!
//! `driver-metal/src/baker/dispatch.rs` and `driver-wgpu`'s are the siblings,
//! and this is the plane that takes ONE ANSWER FROM EACH. That is worth stating
//! at the top, because it is the clearest evidence in the tree that these three
//! files could not have been one:
//!
//! | | grid | barriers |
//! |---|---|---|
//! | metal | TOTAL THREADS, group stated by the body | the driver's, via `Touches` |
//! | wgpu | LANES, group declared by the WGSL | wgpu-core's, and it will not be told not to |
//! | vulkan | LANES, group declared by the SPIR-V | the driver's, via [`Touches`] |
//!
//! # LANES, NOT WORKGROUPS
//!
//! Metal's `Dispatch::grid` is TOTAL THREADS and its `threadgroup` is stated by
//! the body, because MSL declares no workgroup size and nothing else in the
//! path knows one. **Slang does declare it** — `[numthreads(256, 1, 1)]` —
//! and it survives into the SPIR-V as `OpExecutionMode LocalSize`, which
//! `crate::spirv`'s own comment calls *"the divisor a grid is built with"*.
//! Every claim body in `kernels-vulkan` therefore states LANES only, passing a
//! bare `[u32; 3]` to `Fire::apply` and leaving `group` at zero.
//!
//! So this struct carries [`Dispatch::lanes`] and no group, and the divisor is
//! read off the reflected module at encode time, exactly as `crate::encode`
//! already does it:
//!
//! ```ignore
//! let local = module.local;
//! let groups = [fire.lanes[0].div_ceil(local.at(0)), ..];
//! ```
//!
//! Naming the field `lanes` rather than `groups` is not cosmetic.
//! `crate::dispatch::Dispatch` — the legacy one — carries `groups`, ALREADY
//! DIVIDED, and a value moved from one struct to the other without the division
//! would run 1/256th of the work and report success.
//!
//! # There IS a `Touches`, and that is where wgpu parts company
//!
//! wgpu's sibling carries no hazard set, because `wgpu-core` emits a real
//! barrier before every dispatch and `skip_barrier` is always false for a
//! writable buffer. Vulkan offers no such service: `vkCmdDispatch` runs
//! concurrently with its neighbours until a `vkCmdPipelineBarrier` says
//! otherwise, and where those barriers go is this driver's decision. The crate
//! header states the cost of getting it wrong the other way —
//! `crate::dispatch::Dispatch::writes` exists so the recorder can *"skip a
//! pipeline barrier between two dispatches that do not touch the same bytes —
//! most neighbouring pairs — rather than insert one unconditionally between
//! every pair, which is measurably expensive over a fire of a few hundred
//! rectangles."*
//!
//! [`Touches`] is that column, said as RANGES rather than as a parallel
//! `Vec<bool>`. The question an encoder asks is whether two launches may run at
//! once, and the answer is whether their bytes meet — which a bool per operand
//! can only answer by re-joining it against the operand list.
//!
//! # There is no `ParamSlot`, and that is the third divergence
//!
//! Metal's `ParamSlot` carries an argument-table index; wgpu's carries a byte
//! offset into one uniform block. **On this plane the placement is the
//! MODULE's**, and it is not even the same KIND of placement twice:
//! `crate::binding::Params` has three arms because `tests/arena.rs` measured
//! the split over every symbol three real texts launch — six take their scalars
//! as a push block and six as a plain struct in a storage buffer of their own.
//!
//! So what survives the body here is a run of WORDS, in
//! [`Dispatch::params`]'s stated convention, and `crate::binding::params_from`
//! reads the reflected declaration to decide where they go. A `ParamSlot` on
//! this plane would be an offset computed against a layout the driver does not
//! choose.
//!
//! # The plan stays data
//!
//! [`super::encode::Encoder`] BUILDS these rather than recording, for the
//! reason the crate header already gives: *"A fire is a few hundred rectangles
//! and this driver plans all of them, then records them into ONE command buffer
//! with barriers only between the pairs that touch the same bytes."* It is also
//! what makes the walk testable: the encoder is behind a `dyn Encode`, so a
//! recorder can stand where it stands.

use super::marks::{Bound, Slice};

/// One encodable dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dispatch {
    /// The entry point to run, as the claim body named it (`Fire::entrypoint`).
    pub symbol: &'static str,
    /// The ARTIFACT that defines `symbol` (`Fire::file`).
    ///
    /// THE BODY'S OWN ANSWER AND NOT A NAME DERIVED TWICE. A claim body asks
    /// [`kernels_vulkan::plane::Encode::best`] for the tier this adapter
    /// advertises and hands it to `kernels_vulkan::module::path`, which steps
    /// down to the tier the build actually compiled — so `file` is the module
    /// the body CHOSE. The crate header records what deriving it a second time
    /// cost: 146 cooperative-matrix modules and 20 fp16 ones were dead on every
    /// device from the first commit, and nothing failed.
    pub file: &'static str,
    /// The line that makes `symbol` exist in `file`, or empty when the file
    /// already declares it. See `kernels::plane::Fire::stamp`.
    pub stamp: &'static str,
    /// TOTAL INVOCATIONS per axis, as the claim body computed them.
    ///
    /// Divided by the module's own `LocalSize` at encode time to get the
    /// `vkCmdDispatch` argument. See this module's header for why the name is
    /// `lanes` and not `groups`.
    pub lanes: [u32; 3],
    /// Operands IN THE ORDER THE CLAIM BODY STATED THEM, which is the order the
    /// entry point declares its bindings in.
    ///
    /// THE `reorder` PASS IS GONE AND THIS IS WHY. A trace stated inputs, then
    /// outputs, then weights — the compiler's convention — while a Slang
    /// entrypoint declares whatever order it declares, so every launch went
    /// through a permutation read off a row's `operands` column, and a row that
    /// stated none was bound positionally and wrong. A claim body writes the
    /// argument list itself, in the shader's order, because it is the thing
    /// that knows the shader. There is nothing left to permute.
    ///
    /// DESCRIPTOR HOLES ARE THE DEVICE HALF'S. `crate::binding` already knows
    /// which of a module's bindings nothing reads (`kv_append_paged` has six),
    /// and this list is the body's dense run.
    pub args: Vec<Bound>,
    /// The byte ranges this dispatch reads and the ones it may write.
    pub touches: Touches,
    /// The scalar arguments, as the words a parameter block is built from.
    ///
    /// ONE WORD PER SCALAR IN SIGNATURE ORDER, AND TWO FOR A `Usize`, LOW
    /// FIRST, the second aligned to an even word. That convention is
    /// `crate::encode`'s `words` and it is not this module's invention: nothing
    /// in this shader tree declares a 64-bit integer, so an extent arrives as
    /// two `uint`s, and `PIE_STRIDE` is `uint2`, whose push-constant alignment
    /// is eight bytes. `attn/kv_write.slang` is the witness —
    /// `struct Push { int head_dim; PIE_STRIDE k_head_stride; ... }` puts the
    /// first stride at offset 8 and leaves a four-byte hole after `head_dim`.
    ///
    /// A run that gets it wrong is LOUD rather than silent, which is the only
    /// reason a packed run is safe to hand over at all:
    /// `crate::device::Device::dispatch` refuses a push run whose length is not
    /// exactly the pipeline's declared range.
    pub params: Vec<u32>,
    /// The layer this rectangle covers — where a refusal points.
    pub layer: u16,
    /// Which statement of the plan produced it.
    pub op: u32,
}

/// What a dispatch reads and what it may write, as byte ranges.
///
/// Ranges, not operands, because the question an encoder asks is whether two
/// launches can run at once and the answer is whether their bytes meet.
///
/// IT IS `driver-metal`'s TYPE AT THIS PLANE'S REGIONS, and the two are kept
/// apart for the reason a `Slice` is: metal's names an address, this one names
/// an allocation and an offset, and a set that mixed the two would be comparing
/// numbers from different address spaces.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Touches {
    /// Every range the dispatch may read.
    pub reads: Vec<Slice>,
    /// Every range the dispatch may write.
    pub writes: Vec<Slice>,
}

impl Touches {
    /// The CONSERVATIVE answer: every operand as both a read and a write.
    ///
    /// What a fire's own dispatches never use, and the reason it is here
    /// anyway: [`super::encode`]'s `lay_out` reads the direction off the value
    /// a claim body stated — `ArgValue::Buffer { writes: true, .. }` is what
    /// `.arg_mut()` produces — and a hand-built `Dispatch` has no body to have
    /// stated one. So the readers are the device tests that write a launch out
    /// by hand, and what they get is a set that orders every pair rather than a
    /// set that lies about which of them may overlap.
    #[must_use]
    pub fn everything(args: &[Bound]) -> Self {
        let all: Vec<Slice> = args.iter().map(|a| a.slice).collect();
        Self {
            reads: all.clone(),
            writes: all,
        }
    }

    /// Whether these two dispatches touch the same bytes, and therefore whether
    /// a barrier has to stand between them.
    ///
    /// READ-AFTER-WRITE, WRITE-AFTER-READ AND WRITE-AFTER-WRITE, all three.
    /// Two dispatches that only READ the same region need nothing between them,
    /// which is the case this exists to answer `false` for: a fire binds one
    /// weight arena in every statement of a layer, so a rule that ordered every
    /// pair sharing a byte would order the whole fire.
    #[must_use]
    pub fn hazards_after(&self, earlier: &Self) -> bool {
        overlaps(&earlier.writes, &self.reads)
            || overlaps(&earlier.reads, &self.writes)
            || overlaps(&earlier.writes, &self.writes)
    }
}

/// Whether any range of `a` meets any range of `b`.
///
/// Both sets are a handful of entries — a dispatch binds a few operands — so
/// this is a nested scan rather than an interval tree, which is the same choice
/// `merge` below makes for the same reason.
fn overlaps(a: &[Slice], b: &[Slice]) -> bool {
    a.iter().any(|x| {
        b.iter().any(|y| {
            x.buffer == y.buffer
                && !x.is_nothing()
                && !y.is_nothing()
                && x.at < y.at + y.bytes
                && y.at < x.at + x.bytes
        })
    })
}

/// Record a range, merging into an identical one rather than growing the set.
///
/// A fire binds the same weight and the same table over and over, and the sets
/// this feeds are scanned linearly. Merging on `(buffer, at)` rather than on
/// the whole region is what makes two bindings of one weight at two extents one
/// entry: the wider of the two covers both.
pub(crate) fn merge(set: &mut Vec<Slice>, slice: Slice) {
    if slice.is_nothing() {
        return;
    }
    if let Some(seen) = set
        .iter_mut()
        .find(|s| s.buffer == slice.buffer && s.at == slice.at)
    {
        seen.bytes = seen.bytes.max(slice.bytes);
        return;
    }
    set.push(slice);
}

/// The distinct `(file, entry point, stamp)` triples a dispatch list needs
/// compiled.
///
/// In first-use order, deduplicated: a fire naming one symbol 28 times compiles
/// it once. This is what the device half hands to the pipeline cache, and it is
/// here rather than there because it is a property of the list, not of the GPU.
///
/// The stamp rides with the pair rather than being looked up later, because
/// there is nowhere to look it up: it is composed at the fire, by the claim
/// body, and this list is the only thing that survives the body.
#[must_use]
pub fn pipelines_needed(
    dispatches: &[Dispatch],
) -> Vec<(&'static str, &'static str, &'static str)> {
    let mut out: Vec<(&'static str, &'static str, &'static str)> = Vec::new();
    for d in dispatches {
        let point = (d.file, d.symbol, d.stamp);
        if !out.contains(&point) {
            out.push(point);
        }
    }
    out
}

/// The widest operand count any statement of a fire binds.
///
/// A descriptor set is allocated per dispatch against that pipeline's own
/// layout, so nothing here is forced to a common width — what this answers is a
/// capacity question the device half asks once (how big a pool to size, how
/// many `VkWriteDescriptorSet`s to reuse) and a diagnostic one (which statement
/// is the widest).
///
/// Scalars contribute nothing, and on this plane that is only USUALLY true:
/// half the reachable symbols take their scalars as a struct in a storage
/// buffer of their own, which IS a binding. The device half adds that one when
/// `crate::binding::params_from` answers `Params::Block`, because it is the
/// thing that read the module.
#[must_use]
pub fn widest_binding_count(dispatches: &[Dispatch]) -> usize {
    dispatches
        .iter()
        .map(|d| d.args.len())
        .max()
        .unwrap_or(1)
        .max(1)
}

/// How many WORDS of scalar the widest statement of a fire states.
///
/// The device half writes each dispatch's block into one staging allocation
/// when the module wants a `Params::Block`, so it wants the largest single run
/// up front. Zero when no statement states a scalar.
#[must_use]
pub fn widest_param_words(dispatches: &[Dispatch]) -> usize {
    dispatches.iter().map(|d| d.params.len()).max().unwrap_or(0)
}
