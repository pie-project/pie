//! One encodable dispatch: everything a command encoder needs, and nothing
//! that needs a command encoder to compute.
//!
//! LIFTED OUT OF `lowering::dispatch`, WHICH DIED WITH THE BY-NAME CROSSING.
//! That module was two things wearing one name: a GRID PLANNER, which read a
//! `kernel!` row's launch rule and a lowered launch's operand widths and
//! decided how many threads to run, and this — the encoder's own vocabulary.
//! The first is gone: a grid is computed inside the claim body that fires it
//! (`kernels-metal/src/norm.rs::rms_grid` is the shape) and arrives on
//! `Fire::lanes`, so nothing outside the plane crate plans one any more. The
//! second had no legacy in it at all, and it moves here whole.
//!
//! # The plan stays data
//!
//! `fire::run` used to do four things to a plan before an encoder existed:
//! size the argument table from the whole of it, batch-compile every pipeline
//! it names, fingerprint it for indirect-command-buffer replay, and only then
//! submit. That is why [`super::encode::Encoder`] BUILDS these rather than
//! encoding — the walk runs on any host, and all four passes still see the
//! whole fire before a device does.

use core::ops::Range;

use super::marks::{Bound, Slice};

/// One encodable dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dispatch {
    /// The entry point to run, as the claim body named it
    /// (`Fire::entrypoint`).
    pub symbol: &'static str,
    /// The shader that defines `symbol` (`Fire::file`).
    pub file: &'static str,
    /// The line that makes `symbol` exist in `file`, or empty if the file
    /// already declares it. See [`kernels::routine::Fire::stamp`].
    pub stamp: &'static str,
    /// Total threads per axis.
    pub grid: [u32; 3],
    /// Threads per threadgroup per axis.
    pub threadgroup: [u32; 3],
    /// Operands IN THE ORDER THE CLAIM BODY STATED THEM, which is the order
    /// the entry point declares its buffers in.
    ///
    /// THE `reorder` PASS IS GONE AND THIS IS WHY. A trace stated inputs,
    /// then outputs, then weights — the compiler's convention — while
    /// `affine_qmv_fast` declares `w, scales, biases, x, y`, so every launch
    /// went through a permutation read off a `kernel!` row's `operands`
    /// column, and a row that stated none was bound positionally and wrong.
    /// A claim body writes the argument list itself, in the shader's order,
    /// because it is the thing that knows the shader. There is nothing left
    /// to permute.
    pub args: Vec<Bound>,
    /// The byte ranges this dispatch reads and the ones it may write.
    pub touches: Touches,
    /// Where each scalar binds, and how wide it is there.
    pub param_slots: Vec<ParamSlot>,
    /// The scalar arguments, in the order the body stated them.
    pub params: Vec<u32>,
    /// The layers this rectangle covers — where a refusal points.
    pub layers: Range<u16>,
    /// Which statement of the plan produced it.
    pub op: u32,
}

/// One scalar's placement: which buffer, where in the staged run, how wide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamSlot {
    /// The argument-table index this binds at.
    pub slot: usize,
    /// Byte offset into this dispatch's staged scalars.
    pub at: u32,
    /// Bytes the kernel reads there.
    ///
    /// Four or eight, and the difference is not cosmetic:
    /// `attn/sdpa_vector.metal` declares its strides
    /// `const constant size_t&` — eight bytes — while most scalars are
    /// `u32`. A driver that handed a four-byte slot to an eight-byte read
    /// would give the kernel four bytes of the NEXT scalar as this one's
    /// high half. The `ArgValue` variant says which, so the stage widens.
    pub bytes: u32,
    /// Which of the body's scalars this is: the index of its first word in
    /// [`Dispatch::params`].
    pub value: u8,
}

/// What a dispatch reads and what it may write, as byte ranges.
///
/// Ranges, not operands, because the question an encoder asks is whether two
/// launches can run at once and the answer is whether their bytes meet.
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
    /// anyway: `super::encode::directed` reads the direction off the value a
    /// claim body stated — `.arg_mut()` is a write and `.arg()` is a read —
    /// and a hand-built `Dispatch` has no body to have stated one. So the
    /// readers are the device tests that write a launch out by hand, and what
    /// they get is a set that orders every pair rather than a set that lies
    /// about which of them may overlap.
    #[must_use]
    pub fn everything(args: &[super::marks::Bound]) -> Self {
        let all: Vec<Slice> = args.iter().map(|a| a.slice).collect();
        Self {
            reads: all.clone(),
            writes: all,
        }
    }
}

/// Record a range, merging into an identical one rather than growing the set.
///
/// A fire binds the same weight and the same table over and over, and the
/// sets this feeds are scanned linearly.
#[cfg_attr(not(feature = "metal-4"), allow(dead_code))]
pub(crate) fn merge(set: &mut Vec<Slice>, slice: Slice) {
    if slice.is_nothing() {
        return;
    }
    if let Some(seen) = set.iter_mut().find(|s| s.address == slice.address) {
        seen.bytes = seen.bytes.max(slice.bytes);
        return;
    }
    set.push(slice);
}

/// The distinct `(file, entry point, stamp)` triples a dispatch list needs
/// compiled.
///
/// In first-use order, deduplicated: a fire naming one symbol 28 times
/// compiles it once. This is what the device half hands to
/// `Compiler::compile_batch`, and it is here rather than there because it is
/// a property of the list, not of the GPU.
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

/// The widest operand count any statement of a fire binds, plus its scalars.
///
/// An argument table is created with a fixed bind count and a binding past it
/// is an error rather than a silent no-op — so the table has to be built for
/// the widest statement in the fire, not for a guess.
#[must_use]
pub fn table_width(dispatches: &[Dispatch]) -> usize {
    dispatches
        .iter()
        .map(|d| {
            let params = d
                .param_slots
                .iter()
                .map(|p| p.slot + 1)
                .max()
                .unwrap_or_default();
            d.args.len().max(params)
        })
        .max()
        .unwrap_or(1)
        .max(1)
}
