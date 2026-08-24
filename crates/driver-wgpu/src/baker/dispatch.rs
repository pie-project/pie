//! One encodable dispatch: everything a command encoder needs, and nothing
//! that needs a command encoder to compute.
//!
//! `driver-metal/src/baker/dispatch.rs` is the sibling and most of this is that
//! file. What it says about its own history applies here word for word: the old
//! `lowering::dispatch` was two things wearing one name, a GRID PLANNER that
//! read a `LaunchRule` and a lowered launch's operand widths, and the encoder's
//! own vocabulary. The first is gone — a grid is computed inside the claim body
//! that fires it and arrives on `Fire::lanes`, and `tests/rules.rs` already
//! recorded that no routine names a `LaunchRule` any more — and the second had
//! no legacy in it and moves here whole.
//!
//! # LANES, NOT WORKGROUPS — the one number that differs from metal
//!
//! Metal's `Dispatch::grid` is TOTAL THREADS and its `threadgroup` is stated by
//! the body, because MSL declares no workgroup size and nothing else in the
//! path knows one. WGSL DOES declare it — `@compute @workgroup_size(256)` — and
//! every claim body in `kernels-wgpu` states LANES only, passing a bare
//! `[u32; 3]` and leaving `group` at zero.
//!
//! So this struct carries [`Dispatch::lanes`] and no group, and the divisor is
//! read off the reflected module at encode time, which `src/encode.rs` already
//! does:
//!
//! ```ignore
//! let local = pipeline.module().local;
//! let groups = [lanes[0].div_ceil(local.at(0)), ..];
//! ```
//!
//! Naming the field `lanes` rather than `grid` is not cosmetic. `grid` on this
//! plane would read as "workgroup counts" to anyone who has written WebGPU, and
//! `dispatch_workgroups` takes exactly that — so a driver that passed lanes
//! where workgroups were wanted would run 256× the work and still finish. The
//! reverse, which is the one that actually happened, is worse and is on record:
//! `kernels_wgpu::attn`'s `tiled_lanes` disagreed with its shader's real
//! `@workgroup_size` and dispatched a QUARTER of the query heads, silently,
//! until a workgroup census caught it.
//!
//! # There is no `Touches`, and that is measured
//!
//! Metal's `Dispatch` carries the byte ranges a dispatch reads and may write,
//! because a Metal compute encoder runs concurrent dispatches by default and
//! the driver decides where the barriers go. WebGPU does not offer that choice:
//! `src/device.rs`'s module docs record the measurement — wgpu-core emits a real
//! barrier before every dispatch and `skip_barrier` is always false for a
//! writable buffer — so a `Touches` set here would be computed, carried and
//! never read.
//!
//! It is left out rather than carried empty. What metal spends it on, this
//! plane gets for free and cannot decline.
//!
//! # The plan stays data
//!
//! [`super::encode::Encoder`] BUILDS these rather than encoding, for metal's
//! three reasons minus the one that does not apply: the device half wants the
//! WHOLE fire before it starts, to size the bind groups for the widest
//! statement in it and to batch-compile every pipeline it names. (Metal's third
//! — fingerprinting for indirect-command-buffer replay — has no WebGPU
//! equivalent; there are no ICBs.) And it is what makes the walk testable: the
//! encoder is behind a `dyn Encode`, so a recorder can stand where it stands.

use super::marks::Bound;

/// One encodable dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dispatch {
    /// The entry point to run, as the claim body named it (`Fire::entrypoint`).
    pub symbol: &'static str,
    /// The shader that defines `symbol` (`Fire::file`).
    pub file: &'static str,
    /// The line that makes `symbol` exist in `file`, or empty if the file
    /// already declares it. See [`kernels::routine::Fire::stamp`].
    pub stamp: &'static str,
    /// TOTAL INVOCATIONS per axis, as the claim body computed them.
    ///
    /// Divided by the module's own `@workgroup_size` at encode time to get the
    /// `dispatch_workgroups` argument. See this module's header for why the
    /// name is `lanes` and not `grid`.
    pub lanes: [u32; 3],
    /// Operands IN THE ORDER THE CLAIM BODY STATED THEM, which is the order the
    /// entry point declares its `@group(0)` bindings in.
    ///
    /// THE `reorder` PASS IS GONE AND THIS IS WHY. A trace stated inputs, then
    /// outputs, then weights — the compiler's convention — while a WGSL
    /// entrypoint declares whatever order it declares, so every launch went
    /// through a permutation read off a row's `operands` column, and a row that
    /// stated none was bound positionally and wrong. A claim body writes the
    /// argument list itself, in the shader's order, because it is the thing
    /// that knows the shader. There is nothing left to permute.
    pub args: Vec<Bound>,
    /// Where each scalar sits in the uniform block, and how wide it is there.
    pub param_slots: Vec<ParamSlot>,
    /// The scalar arguments, in the order the body stated them.
    pub params: Vec<u32>,
    /// The layer this rectangle covers — where a refusal points.
    pub layer: u16,
    /// Which statement of the plan produced it.
    pub op: u32,
}

impl Dispatch {
    /// This dispatch's `@group(1) @binding(0)` block, as the bytes to write.
    ///
    /// THE JOIN BETWEEN THE TWO HALVES OF THE SCALAR RUN, and it is here rather
    /// than in the device half because it is arithmetic: `params` holds the
    /// words a body stated and `param_slots` holds where each one goes, and
    /// putting them together needs no adapter. A build with no GPU can
    /// therefore check the block a statement would write, which is what
    /// `src/reflect.rs`'s `Declared::uniform_offsets` exists to be checked
    /// against.
    ///
    /// The gaps are ZEROS rather than whatever was there, because a uniform
    /// block's padding is read by nothing and a driver that left it uninitialised
    /// would hand a shader bytes that differ run to run — which is the kind of
    /// thing that makes a numeric test flaky rather than failing.
    #[must_use]
    pub fn uniform(&self) -> Vec<u8> {
        let len = self
            .param_slots
            .iter()
            .map(|p| p.at + p.bytes)
            .max()
            .unwrap_or_default() as usize;
        let mut out = vec![0u8; len];
        for slot in &self.param_slots {
            let words = (slot.bytes / 4) as usize;
            let from = slot.value as usize;
            for (w, word) in self
                .params
                .get(from..from + words)
                .unwrap_or_default()
                .iter()
                .enumerate()
            {
                let at = slot.at as usize + w * 4;
                if let Some(dst) = out.get_mut(at..at + 4) {
                    dst.copy_from_slice(&word.to_le_bytes());
                }
            }
        }
        out
    }
}

/// One scalar's placement in this dispatch's uniform block.
///
/// # ONE UNIFORM BLOCK, WHERE METAL HAS A TABLE SLOT
///
/// This is the field-for-field divergence from `driver-metal`'s `ParamSlot`,
/// and it is the crate's oldest stated fact: **WebGPU has no push constants.**
/// `src/lib.rs` opens with it. A launch's scalars are the FIELDS OF ONE uniform
/// buffer the shell writes and binds at `@group(1) @binding(0)`.
///
/// Metal's `ParamSlot` therefore carries a `slot` — the argument-table index a
/// scalar binds at — and this one does not, because on this plane a scalar
/// binds at no index of its own. What it carries instead is the byte offset
/// inside the block, which is the thing `src/reflect.rs` reads back off the
/// module (`Declared::uniform_offsets`) so that the two can be compared rather
/// than assumed to agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamSlot {
    /// Byte offset into this dispatch's uniform block.
    pub at: u32,
    /// Bytes the shader reads there.
    ///
    /// Four or eight, and the difference is not cosmetic: `kernels_wgpu`'s
    /// `Lang::USIZE` is `vec2<u32>`, eight bytes, where most scalars are `u32`.
    /// A driver that handed a four-byte field to an eight-byte read would give
    /// the shader four bytes of the NEXT scalar as this one's high half. The
    /// `ArgValue` variant says which, so the packing widens.
    pub bytes: u32,
    /// Which of the body's scalars this is: the index of its first word in
    /// [`Dispatch::params`].
    pub value: u8,
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
/// Metal sizes ONE argument table for the whole fire from this, because it has
/// one table. WebGPU builds a bind group per dispatch against that pipeline's
/// own layout, so nothing here is forced to a common width — what this answers
/// is a capacity question the device half asks once (how big a scratch of
/// entries to reuse) and a diagnostic one (which statement is the widest).
///
/// Scalars contribute nothing, which is the other half of the no-push-constants
/// fact: they are not entries of `@group(0)`, they are fields of `@group(1)`.
#[must_use]
pub fn widest_binding_count(dispatches: &[Dispatch]) -> usize {
    dispatches
        .iter()
        .map(|d| d.args.len())
        .max()
        .unwrap_or(1)
        .max(1)
}

/// How many bytes of uniform block the widest statement of a fire needs.
///
/// The device half allocates one uniform staging buffer per fire and writes
/// each dispatch's block into it at an aligned offset, so it wants the largest
/// single block up front. Zero when no statement states a scalar.
#[must_use]
pub fn widest_uniform_bytes(dispatches: &[Dispatch]) -> u32 {
    dispatches
        .iter()
        .map(|d| {
            d.param_slots
                .iter()
                .map(|p| p.at + p.bytes)
                .max()
                .unwrap_or_default()
        })
        .max()
        .unwrap_or_default()
}
