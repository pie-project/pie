//! What this driver knows about a fire, and which routine a symbol reaches.
//!
//! This file held seventy-nine ARMS: a `pub fn` per routine that read the
//! statement's slots and this driver's fire tables and handed back the
//! argument list a shader wanted. They are gone. What supplies a routine now
//! is the routine's own signature -- each parameter states where it comes
//! from -- read by `kernels::bind` and answered by [`crate::bind`]. What is
//! left here is the two things a signature cannot say: which fire this is
//! ([`Facts`]), and which routine a plan's instantiated symbol reaches
//! ([`routine_for`], over [`LIVE`]).
//!
//! # What the deletion found
//!
//! An arm was a second opinion about what a kernel takes, written beside the
//! kernel's own signature and compared to it by nothing. Running the arms and
//! the column against the same statement and diffing the BUFFERS -- which
//! memory each argument points at, the one thing comparable without a device
//! -- crossed eighty-two routines and split on two:
//!
//! - `copy_logits_bf16`: the arm handed the shader the statement's second
//!   INPUT where the column places the parameter block.
//! - `router_topk_scaled`: the arm handed the second input where the column
//!   places the first WEIGHT -- the per-expert scale, which is what the
//!   scaled form exists to read.
//!
//! Both are the same two `driver-wgpu`'s arms got wrong, in the same two
//! routines, written independently. The same mistaken idea was written twice
//! and neither crate's tests could see it, because the arm WAS each crate's
//! notion of what the kernel took. One binder, asked once, is the whole
//! argument for this refactor in two lines.
//!
//! # What was not an arm
//!
//! Three rules were living among the arms and are now in [`crate::bind`],
//! where the binder's backend half is:
//!
//! - The split decode's `attn.partials` and `attn.splits`. How many ways to
//!   fold a long decode's key range is a judgement about THIS fire -- history
//!   depth against head count -- and a signature is fire-invariant, so it can
//!   name the number but cannot compute it. [`Handles::decode_splits`] still
//!   makes that judgement; `named` is what calls it.
//! - `cast_qmm_input_strided`'s `count`, which the shader does not read and
//!   the arm filled anyway with `rows x pitch` so that a later reader would
//!   find the number the field's name promises. That is arithmetic the column
//!   can spell, so the column spells it now, and the routine no longer has an
//!   argument nothing states.
//!
//! # How this differs from `driver-metal` and `driver-wgpu`
//!
//! **A missing driver resource is a refusal here, not a zero handle.** Metal's
//! `Handles::table` answers a null slice for a fire table the resolver does
//! not hold, because a Metal argument slot left unbound holds whatever address
//! the previous dispatch put there. This driver's [`crate::binding::Resolve`]
//! returns `Option<&Buffer>` and [`crate::binding::reorder`] already refuses
//! with [`Unbindable::NoDriverResource`](crate::binding::Unbindable).
//!
//! **`params_block` is the module's choice, not the crate's.** Metal mints a
//! handle standing for the statement's scalar run because MSL has no push
//! constants. Here a routine hands its scalars over as [`ArgValue`] words and
//! [`crate::encode::Encoder`] asks [`crate::binding::params_from`] which of
//! push constants and a storage struct this module declared. The reachable
//! symbols divide almost evenly on it, so neither answer could have been
//! assumed on either side.
//!
//! **This driver serves no recurrent state.** [`crate::frames`] refuses a plan
//! that names recurrent slots, so the five GDN routines' `recurrent_slots`
//! goes unanswered here and that is the right answer rather than a gap.

use kernels::routine::Refusal;
use model_compiler::lower::Arg;

use crate::binding::{FireNumber, FireTable, Resolve};
use crate::device::Bound;

/// The extents a routine takes from its environment rather than its
/// statement.
///
/// These are the numbers `Dims` carried for the launch rules, minus the four
/// `*_param` overrides: a routine that needs a per-statement head width takes
/// it as an ARGUMENT, off the scalar run, because its signature can name it.
/// The columns existed to let a row point at a scalar it could not name.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Facts {
    /// Rows the rectangle covers.
    pub rows: u32,
    /// Elements per row of the operand that sizes the launch — the last
    /// widthed operand, which is the last result.
    pub width: u32,
    /// Elements per row of the first widthed operand, the first input.
    pub in_width: u32,
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
    /// The deployment's affine group size and bits per weight.
    ///
    /// Not from [`crate::dispatch::Geometry`], which does not carry them: on
    /// this backend the two numbers are spelled into the ENTRYPOINT — the
    /// `_gs_128_b_4` suffix `kernels::sig_in` matches a row through — and
    /// [`affine_of`] reads them back off the plan's symbol. Metal's `Geometry`
    /// carries them because its entrypoint names do not.
    pub group: u32,
    /// See [`Facts::group`].
    pub bits: u32,
    /// The layer this rectangle covers.
    ///
    /// The layer span of a rectangle is always one wide, so its start is the
    /// layer, and that is what [`crate::binding::reorder`] reads for
    /// [`kernels::Source::Named(<kernels::keys::KvKeys as kernels::keys::Fact>::KEY)`]. An arm needs it for the same reason: the
    /// KV cache is per-layer state.
    pub layer: u16,
    /// Requests the fire serves.
    ///
    /// Not an extent of any rectangle: it is the count a `row_gather` writes,
    /// and the one statement that needs it takes it as an argument rather than
    /// as a lane count. This is the plan's `n_requests`, which is what
    /// `kernels::Source::Named(<kernels::keys::RequestCount as kernels::keys::Fact>::KEY)` resolved to.
    ///
    /// Sizing that gather by [`Facts::rows`] is the defect
    /// `.wiki/kernel-x/vulkan-refactor.md` §10 records, and it lives in
    /// `binding::extent` rather than here.
    pub requests: u32,
    /// RECURRENT heads and width, which a GDN block reads instead of the
    /// attention pair. Filled by `Geometry::recurrent`, so a stack that
    /// states no recurrent shape reads the attention one and nothing moves.
    pub v_heads: u32,
    /// See [`Facts::v_heads`].
    pub v_dim: u32,
    /// The GEMM tile the symbol spells, if it spells one.
    ///
    /// `affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32` is a 32x32 tile. Like
    /// [`Facts::group`] and [`Facts::bits`] this is read off the PLAN's
    /// symbol and not off the fire, because it is a choice about THIS
    /// multiply's shape rather than a property of the deployment -- there is
    /// no fire-wide number that could stand in for it, and the two
    /// substitutions tried on the metal backend were both measured wrong
    /// against a real checkpoint: handing the GEMM's symbol the matvec's grid
    /// made a prefill entirely NaN, and rounding an axis up made it finite
    /// and wrong.
    pub tile: Option<(u32, u32)>,
}

/// The affine group size and bits an entrypoint spells.
///
/// `affine_qmv_fast_bfloat16_gs_64_b_4` is group 64 at 4 bits. Read off the
/// PLAN's symbol rather than taken from the fire, because that is where this
/// backend writes them down: `kernels::sig_in` matches a row by peeling these
/// same suffixes, so a symbol that reached the launch path has them.
///
/// `None` when the symbol spells no affine axis, which is every kernel that is
/// not a quantized matmul.
#[must_use]
pub fn affine_of(symbol: &str) -> Option<(u32, u32)> {
    // The tile suffix comes AFTER the codec one, so it has to come off first:
    // `..._gs_64_b_4_bm_32_bn_32` splits at the last `_b_` into a `bits` of
    // `4_bm_32_bn_32`, which parses as nothing and reads as "this symbol
    // spells no codec". Peeling is not optional -- every routed and tiled
    // GEMM in the fleet wears both.
    let symbol = tile_of(symbol).map_or(symbol, |_| {
        symbol.rsplit_once("_bm_").map_or(symbol, |(head, _)| head)
    });
    let (head, bits) = symbol.rsplit_once("_b_")?;
    let (_, group) = head.rsplit_once("_gs_")?;
    // `rsplit_once` leaves whatever followed, and a real axis point is the
    // whole of it: `_b_4_bm_16` must not read as four bits with a tail.
    Some((group.parse().ok()?, bits.parse().ok()?))
}

/// The GEMM tile an entrypoint spells, as `(bm, bn)`.
///
/// `affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32` is `(32, 32)`. `None` when
/// the symbol spells no tile, which is every kernel that is not a tiled
/// multiply -- and, deliberately, every tiled multiply whose suffix is
/// malformed: a tile is refused rather than guessed, because a GEMM handed a
/// tile its module was not compiled for writes a finite wrong answer.
#[must_use]
pub fn tile_of(symbol: &str) -> Option<(u32, u32)> {
    let (head, bn) = symbol.rsplit_once("_bn_")?;
    let (_, bm) = head.rsplit_once("_bm_")?;
    Some((bm.parse().ok()?, bn.parse().ok()?))
}

/// The statement's operands and scalars, and the handles an arm builds from
/// them.
///
/// An arm names a value — `o.input(0)`, `o.weight(2)`, `o.kv(layer, false)` —
/// and gets back a handle it can pass to a routine. The handle is an index
/// into [`Handles::bound`], which is what [`crate::encode::Encoder`] resolves
/// a body's [`ArgValue::Buffer`] through, and the indices are assigned HERE
/// rather than fixed by the trace's order: a fire table and a KV page are not
/// operands and have no place in it.
///
/// Two lifetimes, for the reason [`crate::encode::Encoder`] has two: `'a` is
/// the DEVICE memory a [`Bound`] points into -- the arena and the weight
/// store, which outlive the whole fire -- while `'h` is the caller's own
/// per-launch scratch. Fused, the handles an arm produced would borrow
/// vectors that are dropped before the command buffer is recorded.
pub struct Handles<'a, 'h> {
    /// Every handle an arm has asked for, in the order it asked.
    bound: Vec<Bound<'a>>,
    /// The statement's widthed operands that are INPUTS, as indices into the
    /// launch's bound arguments.
    ins: &'h [usize],
    /// The same, for its RESULTS.
    outs: &'h [usize],
    /// The same, for the weights it names.
    weights: &'h [usize],
    /// What the launch bound, in the trace's order.
    args: &'h [Bound<'a>],
    /// Elements per row of each of those, in the SAME order, and zero for the
    /// ones that have none.
    ///
    /// THE HALF A `Bound` DOES NOT CARRY. A [`Bound`] is a buffer, an offset
    /// and a length; the rectangle over it is stated by the lowering's `Arg`,
    /// which this driver drops on the way in. That was invisible while a
    /// signature's marks were plain buffers, and stopped being invisible when
    /// `Tensor<E>` began carrying its own width: `kernels::bind`'s `shaped`
    /// asks a backend for the width beside the handle, this one answered
    /// `Unstated`, the binder read the default and bound EVERY operand at
    /// width zero -- so the first body to read `out.width` refused `Empty`,
    /// which is every fire this driver has ever planned.
    ///
    /// A weight's entry is zero and that is not a hole: its extents are the
    /// checkpoint's and a statement does not restate them. `driver-wgpu`
    /// carries the lowering's `Arg` slice itself and reads the same number off
    /// it; the shape here is a projection of that, because this driver's
    /// `Handles` is built from `Bound`s rather than from `Arg`s.
    widths: &'h [i32],
    /// The statement's own scalar run.
    params: &'h [Option<u32>],
    /// What answers for the things the STATEMENT does not carry: a fire's
    /// position table, a layer's KV pages, a pool's page size.
    ///
    /// A `dyn` because [`Arm`] is a plain function pointer and cannot be
    /// generic. The cost is one virtual call per table, once per LOWERING, so
    /// it is not on any encode path.
    resolver: &'a dyn Resolve,
}

impl<'a, 'h> Handles<'a, 'h> {
    /// The handles for one launch.
    #[must_use]
    pub fn new(
        args: &'h [Bound<'a>],
        widths: &'h [i32],
        ins: &'h [usize],
        outs: &'h [usize],
        weights: &'h [usize],
        params: &'h [Option<u32>],
        resolver: &'a dyn Resolve,
    ) -> Self {
        Self {
            // Sized for the widest signature in the tree rather than grown
            // from nothing: a handle vector that starts empty reallocates at
            // one, two, four and eight handles, per RECTANGLE.
            bound: Vec::with_capacity(8),
            ins,
            outs,
            weights,
            args,
            widths,
            params,
            resolver,
        }
    }

    /// What the encoder resolves a body's handles through.
    #[must_use]
    pub fn bound(&self) -> &[Bound<'a>] {
        &self.bound
    }

    /// Take a handle for `bound`, whatever it is.
    fn take(&mut self, bound: Bound<'a>) -> u32 {
        let at = u32::try_from(self.bound.len()).unwrap_or(u32::MAX);
        self.bound.push(bound);
        at
    }

    fn pick(&mut self, at: Option<usize>, what: &'static str) -> Result<u32, Refusal> {
        let at = at.ok_or(Refusal::Absent { what })?;
        let bound = *self.args.get(at).ok_or(Refusal::Absent { what })?;
        Ok(self.take(bound))
    }

    /// The statement's `i`-th input.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] when the statement has fewer, which is an arm
    /// asking for an operand its trace does not carry.
    pub fn input(&mut self, i: usize) -> Result<u32, Refusal> {
        let at = self.ins.get(i).copied();
        self.pick(at, "an input the statement does not carry")
    }

    /// The statement's `i`-th result.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] when the statement has fewer.
    pub fn output(&mut self, i: usize) -> Result<u32, Refusal> {
        let at = self.outs.get(i).copied();
        self.pick(at, "a result the statement does not carry")
    }

    /// The statement's `i`-th result, read rather than written.
    ///
    /// For the routines that take their own output as an input — a residual
    /// added in place, a gate applied to the tensor it gates. The aliasing is
    /// stated on the routine as `in_place`, and this is the arm honouring it.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] when the statement has fewer.
    pub fn output_read(&mut self, i: usize) -> Result<u32, Refusal> {
        let at = self.outs.get(i).copied();
        self.pick(at, "a result the statement does not carry")
    }

    /// The `i`-th weight the statement names.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] when the statement names fewer.
    pub fn weight(&mut self, i: usize) -> Result<u32, Refusal> {
        let at = self.weights.get(i).copied();
        self.pick(at, "a weight the statement does not name")
    }

    /// Elements per row of the statement's `i`-th INPUT.
    ///
    /// Asked beside [`Self::input`] rather than instead of it: a `Tensor<E>`
    /// mark is a handle AND a rectangle, so the shared binder puts both
    /// questions to a backend. See [`Handles::widths`] for what answering only
    /// the first one cost.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] when the statement carries no such input, the same
    /// refusal [`Self::input`] gives for the same index.
    pub fn in_width(&self, i: usize) -> Result<i32, Refusal> {
        self.width_at(
            self.ins.get(i).copied(),
            "an input the statement does not carry",
        )
    }

    /// The same, for the statement's `i`-th RESULT. See [`Self::in_width`].
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] when the statement declares no such result.
    pub fn out_width(&self, i: usize) -> Result<i32, Refusal> {
        self.width_at(
            self.outs.get(i).copied(),
            "a result the statement does not carry",
        )
    }

    /// One operand's row width, by its place in the launch's arguments.
    fn width_at(&self, at: Option<usize>, what: &'static str) -> Result<i32, Refusal> {
        let at = at.ok_or(Refusal::Absent { what })?;
        self.widths.get(at).copied().ok_or(Refusal::Absent { what })
    }

    /// One of the FIRE's tables: the token ids, the positions, the sampled
    /// rows, the KV page directory.
    ///
    /// Not the statement's, which is why a row had to name these with a
    /// [`kernels::Source`] of their own and why an arm asks for them by name
    /// rather than by index. They are this fire's data — what is being run —
    /// where an operand is the model's structure.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] when this fire holds no such table, which is what
    /// [`crate::binding::reorder`] reports as
    /// [`Unbindable::NoDriverResource`](crate::binding::Unbindable). A decode
    /// has no sampling indices, and a statement that asks for them in one is a
    /// trace mismatch.
    pub fn table(&mut self, which: FireTable) -> Result<u32, Refusal> {
        let buffer = self.resolver.table(which).ok_or(Refusal::Absent {
            what: "a fire table this run does not hold",
        })?;
        Ok(self.take(Bound::whole(buffer)))
    }

    /// The same, written through.
    ///
    /// # Errors
    ///
    /// See [`Handles::table`].
    pub fn table_mut(&mut self, which: FireTable) -> Result<u32, Refusal> {
        let buffer = self.resolver.table(which).ok_or(Refusal::Absent {
            what: "a fire table this run does not hold",
        })?;
        Ok(self.take(Bound::whole(buffer)))
    }

    /// A layer's KV cache, keys or values.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] when this fire has no paged cache.
    pub fn kv(&mut self, layer: u16, values: bool) -> Result<u32, Refusal> {
        let buffer = self.resolver.kv(layer, values).ok_or(Refusal::Absent {
            what: "a KV cache this run does not hold",
        })?;
        Ok(self.take(Bound::whole(buffer)))
    }

    /// The same, read rather than written — what a paged attention takes.
    ///
    /// # Errors
    ///
    /// See [`Handles::kv`].
    pub fn kv_read(&mut self, layer: u16, values: bool) -> Result<u32, Refusal> {
        let buffer = self.resolver.kv(layer, values).ok_or(Refusal::Absent {
            what: "a KV cache this run does not hold",
        })?;
        Ok(self.take(Bound::whole(buffer)))
    }

    /// A number the driver keeps for the KV pool — a stride, a page size.
    ///
    /// Not a handle: these reach a kernel as scalars, and a routine that needs
    /// one takes it as an argument.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] when this fire's resolver does not answer for it.
    pub fn number(&self, which: FireNumber) -> Result<u32, Refusal> {
        self.resolver.number(which).ok_or(Refusal::Absent {
            what: "a pool number this run does not hold",
        })
    }

    /// The statement's `i`-th scalar, as the signed number a kernel reads.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] when the statement's run is shorter, or the slot is
    /// empty — a trace that stated a scalar's position and not its value.
    pub fn param(&self, i: usize) -> Result<i32, Refusal> {
        let held = self
            .params
            .get(i)
            .copied()
            .flatten()
            .ok_or(Refusal::Absent {
                what: "a scalar the statement does not carry",
            })?;
        Ok(held.cast_signed())
    }

    /// The statement's `i`-th scalar, as the float a kernel reads.
    ///
    /// The trace carries every scalar as a `u32` and the BITS are the value,
    /// so this reinterprets rather than converting — `1.0f32` rides as
    /// `0x3f80_0000`, and a conversion would hand the kernel 1065353216.0.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] when the statement's run is shorter.
    pub fn param_f32(&self, i: usize) -> Result<f32, Refusal> {
        let held = self
            .params
            .get(i)
            .copied()
            .flatten()
            .ok_or(Refusal::Absent {
                what: "a scalar the statement does not carry",
            })?;
        Ok(f32::from_bits(held))
    }

    /// The statement's scalars, as the one packed buffer a kernel reads them
    /// through.
    ///
    /// Six of this backend's modules read their parameters out of a storage
    /// block rather than a push range -- `route_sort`'s 28 bytes at binding 4
    /// of 6, `combine_sorted`'s 12 at 3 of 5 -- and a routine that forwards
    /// `ctx.params()` is one of them. The block is a DESCRIPTOR there,
    /// so the arm has to put something in its slot.
    ///
    /// What it cannot put there is an address. The scalar runs of a whole
    /// fire are gathered into ONE staging buffer after every rectangle is
    /// planned (`serve::fire`), because a buffer allocated per dispatch is a
    /// buffer freed while the queue still holds a pointer to it -- so at the
    /// moment an arm runs, the memory this block will live in does not exist.
    ///
    /// So this mints a SENTINEL. [`crate::encode`] recognises it, keeps the
    /// slot it would have occupied as the dispatch's `block_at`, and hands
    /// the bytes to the same staging path the table path uses. A body that
    /// forwards the handle gets its descriptor; a body that drops it -- which
    /// is `mlp::geglu_tanh`, whose block slangc deleted for being unread --
    /// leaves no slot behind, which is what its module declares.
    ///
    /// Deliberately not a `Result`: a statement carrying no scalars still has
    /// a block, of one zero word. The shader dereferences the pointer whether
    /// or not it reads a field, and a descriptor pointing at nothing is a
    /// device fault rather than a refusal.
    pub fn params_block(&mut self) -> u32 {
        BLOCK
    }

    /// A handle for a slot the routine drops. See [`UNBOUND`].
    pub fn unbound(&mut self) -> u32 {
        UNBOUND
    }

    /// One of the fire's tables, or [`UNBOUND`] if this fire holds none.
    ///
    /// For a table a routine may or may not dispatch through. The flash
    /// decode's partials are the one: a caller that never sizes them leaves
    /// the split count at one, the body takes the single-pass path and the
    /// handle is dropped -- so an absent table is a fact the arm reads, not a
    /// refusal it raises.
    pub fn table_or_unbound(&mut self, which: FireTable) -> u32 {
        match self.resolver.table(which) {
            Some(buffer) => self.take(Bound::whole(buffer)),
            None => UNBOUND,
        }
    }

    /// How many ways this statement's decode splits its key range.
    ///
    /// Zero splits is not a thing: this answers 1 -- the single-pass path --
    /// whenever the pool states no history, holds no partial buffer, or the
    /// rule says the fold would not pay. See
    /// [`kernels_vulkan::attn::decode_splits`].
    pub(crate) fn decode_splits(&self, f: Facts) -> i32 {
        if self.resolver.table(FireTable::AttnPartials).is_none() {
            return 1;
        }
        let Some(bucket) = self.resolver.number(FireNumber::KvHistoryBucket) else {
            return 1;
        };
        kernels_vulkan::attn::decode_splits(
            i32::try_from(bucket).unwrap_or(i32::MAX),
            f.q_heads.cast_signed(),
            f.rows.cast_signed(),
        )
    }

    /// A GDN state slab, by layer and by name.
    ///
    /// # Errors
    ///
    /// [`Refusal::Unstated`], always, on this backend: nothing here allocates
    /// the recurrent pool. `kernels::Source::GdnSlab` reaches
    /// [`crate::binding`]'s catch-all and the eight `ssm` rows were all bare,
    /// so the family has never dispatched.
    ///
    /// Writing the arms against a refusal rather than leaving them unwritten
    /// is the point: it makes the gap a MISSING ALLOCATION with one name,
    /// instead of eight routines nothing can reach and no statement of what
    /// they would need. When a driver allocates the slabs, this is the one
    /// function that changes.
    pub fn slab(&mut self, _layer: u16, _which: &'static str) -> Result<u32, Refusal> {
        Err(Refusal::Unstated {
            what: "a GDN slab: this driver allocates none",
        })
    }

    /// The words [`Handles::params_block`] stands for.
    ///
    /// The statement's run in order, one word each, with an empty slot read
    /// as zero -- the same words `binding::params_from` would have packed
    /// from a row, and in the same order, because both are the trace's.
    ///
    /// Not the WHOLE block. A routine may pack a further word into it that
    /// the statement does not carry: `layout::row_gather`'s `count` is the
    /// second field of `RowGatherParams` and reaches the shader as an
    /// [`kernels_vulkan::routine::InPacked`], which is a member of the struct
    /// and not an operand. Those arrive as the body's own scalars, so
    /// [`crate::encode`] appends them here.
    #[must_use]
    pub fn staged(&self) -> Vec<u32> {
        self.params.iter().map(|p| p.unwrap_or(0)).collect()
    }
}

/// The handle [`Handles::params_block`] mints for the staged scalar run.
///
/// `u32::MAX` because it must not collide with a real index into
/// [`Handles::bound`], and because a path that failed to recognise it would
/// fail LOUDLY: `encode`'s buffer lookup is a `get`, so an unrecognised
/// sentinel is `Refusal::Absent` rather than a descriptor pointing at
/// whichever range happened to be last.
pub const BLOCK: u32 = u32::MAX;

/// A handle for a slot the ROUTINE drops.
///
/// Three of this backend's signatures take a buffer their body never
/// dispatches -- `moe::router_topk`'s `_per_expert_scale`, `moe::qmv_routed`'s
/// `_bias`, `moe::mxfp4_qmv_routed_bias`'s `_biases` -- because they share a
/// signature with a form that DOES read it. Metal binds nothing there, and has
/// to: an unset Metal slot holds the previous dispatch's address. Here the
/// argument never reaches a descriptor at all, because the body drops it
/// before `ctx.dispatch`.
///
/// So this mints a value that cannot be resolved. If a body ever does forward
/// it, `encode`'s buffer lookup is a `get` and the answer is
/// [`Refusal::Absent`] -- the loud failure, not a descriptor pointing at
/// whichever range happened to be last.
pub const UNBOUND: u32 = u32::MAX - 1;

/// The widthed operands of a launch split into inputs and results, and the
/// weights it names.
///
/// The trace concatenates inputs, then results, then weights, and the binder
/// keeps that order. `results` is how many of the widthed ones are results,
/// which the ROUTINE knows — it is the count of writable types in its
/// signature, [`kernels::Binds::Writes`] — where the table path read it off
/// the row's `Out` sources.
#[must_use]
pub fn split(args: &[Arg], results: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let widthed: Vec<usize> = args
        .iter()
        .enumerate()
        .filter(|(_, a)| !matches!(a, Arg::Weight(_)))
        .map(|(i, _)| i)
        .collect();
    let weights: Vec<usize> = args
        .iter()
        .enumerate()
        .filter(|(_, a)| matches!(a, Arg::Weight(_)))
        .map(|(i, _)| i)
        .collect();
    let results = results.min(widthed.len());
    // `split_off` and not `split_at`: the tail moves into its own vector and
    // the head keeps the one already allocated, where two `to_vec`s allocated
    // twice more and copied both halves.
    let mut ins = widthed;
    let outs = ins.split_off(ins.len() - results);
    (ins, outs, weights)
}

/// How many of a routine's WRITABLE operands the STATEMENT supplies.
///
/// [`split`]'s `results` is meant to be this number, and it is exactly the
/// count of writable types in the signature -- for all hundred of them.
///
/// # The two that used to be subtracted, and why they no longer are
///
/// `attn::kv_append` and `attn::kv_append_paged` write the KV cache, and the
/// cache is not an operand the trace carries: the body draws it from the POOL.
/// While that draw was a MARK, the pair's signatures counted two writable
/// types the statement does not supply, and this function subtracted two to
/// get back to the statement's own count.
///
/// Both bodies ask for the cache now -- `ctx.ask::<_, keys::KvKeys>()`, which
/// is a fact and not an argument -- so neither signature has a writable type
/// in it at all. The subtraction was left in place across that move and was
/// then taking two from ZERO: `attempt to subtract with overflow`, on the
/// driver lane, at the first `kv_append` of every fire. A debug build panicked
/// and a release build would have wrapped to `usize::MAX`, which `split` clamps
/// with `results.min(widthed.len())` -- so it would have handed BOTH inputs to
/// `outs`, left `ins` empty, and refused `Absent` at `o.input(0)`. That is
/// exactly the failure the subtraction was added to fix, arrived at from the
/// other side.
///
/// `driver-wgpu`'s `results` is this function without the special case, and
/// has been since its bodies started asking. There is no case left to state.
///
/// # Why it asks [`kernels::Ty::binds`] rather than testing `Ty::BufMut`
///
/// Because `BufMut` is one writable type of ten and this counted only that one
/// -- the narrow count `driver-wgpu`'s `results` already refused to copy, its
/// doc naming `ssm`'s `F32sMut` recurrent state as the case it misses. A
/// signature naming its activation element binds `Ty::Bf16sMut` and would have
/// been the second such case. The classification is `kernels`' to make, once.
#[must_use]
pub fn traced_results(routine: &kernels_vulkan::routine::Routine) -> usize {
    routine
        .args
        .iter()
        .filter(|ty| ty.binds() == kernels::Binds::Writes)
        .count()
}

// THE ARMS STOOD HERE: seventy-nine functions, one per crossed kernel,
// each reading the statement positionally and handing a body the list it
// wanted. They are gone, and `kernels::bind` reads the `sources` column
// each routine already states -- see `crate::bind`.
//
// Two of them had silently parted from the signature they duplicated, and
// they are the SAME TWO that had parted on `driver-wgpu`, written wrong
// twice independently:
//
//   * `copy_logits_bf16` bound the statement's input 1 where the column
//     says the packed run, so the run went nowhere and an arena operand
//     took its place.
//   * `router_topk_scaled` bound the statement's input 1 where the column
//     says weight 0.
//
// Neither was caught by a test in either backend, because the arm WAS each
// test's idea of what the kernel took. What found them is
// `bind::equivalence`, which ran both readings against one statement and
// compared the MEMORY each pointed at -- and which is retired now, having
// nothing left to compare.

/// One crossed routine: the entrypoint stem a plan spells it with, the
/// routine itself, and the arm that feeds it.
///
/// The stem is the piece that lets a `kernel!` row be DELETED. Until this
/// existed the fork resolved a plan's symbol through
/// `kernels::sig_in(KERNELS, ..)` -- which is the very table the refactor is
/// emptying, so retiring a family's rows would have made its own routines
/// unreachable. `.wiki/kernel-x/refactor-bigplan.md` §7 puts this at Stage 5
/// and Stage 3 at the same time; Stage 3 asks each family to delete its rows
/// as it lands, which cannot happen while the lookup is the rows. It belongs
/// here, first.
///
/// The stem is transcribed out of the row it replaces -- `kernel!`'s second
/// column, the entrypoint base before any axis suffix -- so this is a move
/// rather than a second opinion, and the row is deleted in the same commit
/// that adds the stem.
pub struct Crossed {
    /// The entrypoint base, before any instantiation suffix.
    ///
    /// `affine_qmv_fast` for a routine named `qmv_fast`: the two differ, which
    /// is why a prefix test against [`Crossed::routine`]'s own name would not
    /// do and the stem has to be stated.
    pub stem: &'static str,
    /// The routine, with its body -- or `None` for a stem this driver
    /// RESERVES without serving.
    ///
    /// A family does not always cross whole. `mlp` has four routines against
    /// five rows: `silu_mul_strided` was never written, on this backend or on
    /// Metal. And `silu_mul` is a prefix of it followed by a `_`, so the match
    /// rule below would hand every `silu_mul_strided_bfloat16` to the
    /// contiguous body -- which reads three operands at the wrong pitches and
    /// returns success.
    ///
    /// A reserved stem is the narrowest fix: it wins the longest-match and
    /// answers `None`, so the symbol falls through to the `kernel!` row that
    /// is still there for it. It goes away when the routine is written.
    pub routine: Option<&'static kernels_vulkan::routine::Routine>,
}

/// The routine this driver calls for the symbol a plan named, if its family
/// is crossed.
///
/// A plan names the fully instantiated entrypoint --
/// `argmax_logits_bfloat16`, `affine_qmm_t_bf16_gs_128_b_4_bm_32_bn_32` --
/// and a routine is named after the kernel. This is the join, and it is the
/// arm registry's own: it matches the longest [`Crossed::stem`] the symbol
/// begins with, requiring what follows to be empty or to start with `_`.
///
/// # Why longest-match, and why the underscore
///
/// Both halves are load-bearing and neither is obvious.
///
/// **Longest.** `qmm_t` and `qmm_t_splitk` are different routines and the
/// first stem is a prefix of the second's symbols. First-match would send
/// every split-K rectangle to the single-pass body, which binds real buffers,
/// dispatches, and writes a partial sum where a total belongs.
///
/// **The underscore.** Without it `qmm_t` also claims `qmm_t_strided`, which
/// is the same defect one letter further along. An axis suffix always begins
/// with a separator -- `_bfloat16`, `_gs_64_b_4`, `_bm_32_bn_32` -- so a stem
/// that runs into the middle of a longer name is not a match.
///
/// A crossed family with no arm is declared and dark -- comparable against the
/// other backends by `kernels/tests/shader_backends_agree.rs`, callable by
/// nothing -- and that is the state every family not listed here is in.
/// Retiring a family's `kernel!` rows is what adding it here BUYS, and the two
/// go in one commit.
#[must_use]
pub fn routine_for(symbol: &str) -> Option<&'static kernels_vulkan::routine::Routine> {
    spelled(symbol).map(|s| s.routine)
}

/// Everything a SYMBOL says about itself, worked out once per symbol.
///
/// The routine and the arm [`arm_for`] answers, and beside them the three
/// facts a rectangle used to re-derive from the same string every time it was
/// planned: the affine axis [`affine_of`] peels off it, the tile [`tile_of`]
/// peels off it, and the result count [`traced_results`] reads off the
/// routine. None of the three varies between two rectangles naming the same
/// symbol, and a qwen3-0.6b decode states 452 rectangles over NINE symbols.
///
/// Measured, release, `tests/planbench.rs` -- which plans the same 452
/// rectangles with no card in the room, so the number is not a shared GPU's:
/// `affine_of + tile_of` alone was **66 to 133 ns of a 950 ns rectangle**,
/// which is 7-14% of the planning cost, spent on `rsplit_once` and `parse`
/// over an answer that had been computed 443 times already.
#[derive(Clone, Copy)]
pub struct Spelled {
    /// The routine this symbol's family crosses to.
    pub routine: &'static kernels_vulkan::routine::Routine,
    /// The affine group size, or zero where the symbol spells no codec.
    pub group: u32,
    /// The affine bit width, or zero with [`Spelled::group`].
    pub bits: u32,
    /// The GEMM tile the symbol spells, if it spells one.
    pub tile: Option<(u32, u32)>,
    /// How many of the widthed operands are RESULTS: [`traced_results`].
    pub results: usize,
}

/// What `symbol` spells, from a per-thread memo.
///
/// # Why it is memoised
///
/// The stem search is a sweep. [`LIVE`] is 101 entries and the answer is the
/// LONGEST matching stem, so every call walks all of them; a qwen3-0.6b
/// decode asks 452 times about nine distinct symbols. Measured, release, per
/// decode step: 0.113 ms of a 1.88 ms host step for 45,652 prefix
/// comparisons, of which 45,643 asked a question already answered.
///
/// The memo is sound because every part of the answer is immutable: [`LIVE`]
/// is a `LazyLock` built once, and this function is pure in its argument. It
/// is per THREAD rather than behind a lock so that two drivers firing at once
/// do not queue on it -- the table is nine entries a model, so the
/// duplication is nothing.
///
/// Keyed by `String`, probed by `&str`: `HashMap`'s `Borrow` lets a hit cost
/// no allocation, and a miss allocates once per distinct symbol per thread
/// rather than once per rectangle.
#[must_use]
pub fn spelled(symbol: &str) -> Option<Spelled> {
    thread_local! {
        static MEMO: std::cell::RefCell<
            std::collections::HashMap<String, Option<Spelled>>,
        > = std::cell::RefCell::new(std::collections::HashMap::new());
    }
    MEMO.with(|memo| {
        if let Some(held) = memo.borrow().get(symbol) {
            return *held;
        }
        let found = LIVE
            .iter()
            .filter(|c| {
                symbol
                    .strip_prefix(c.stem)
                    .is_some_and(|rest| rest.is_empty() || rest.starts_with('_'))
            })
            .max_by_key(|c| c.stem.len());
        let answer = found.and_then(|f| {
            let routine = f.routine?;
            let (group, bits) = affine_of(symbol).unwrap_or((0, 0));
            Some(Spelled {
                routine,
                group,
                bits,
                tile: tile_of(symbol),
                results: traced_results(routine),
            })
        });
        memo.borrow_mut().insert(symbol.to_owned(), answer);
        answer
    })
}

/// One routine by name, for a registry line to point at.
///
/// ONE SLICE AND NOT TEN. This took a family's own `ROUTINES` beside the name,
/// and the families no longer have one: `#[routine]` registers into a
/// `linkme` distributed slice, so `kernels_vulkan::ROUTINES` is every routine
/// the crate declares and the linker assembles it. There is no membership list
/// to add a routine to -- which is the last hand-written thing about one, and
/// the last that could be forgotten.
///
/// Searching the whole slice rather than a family's is safe because the names
/// are unique across it, which `kernels-vulkan`'s own
/// `no_symbol_is_declared_twice` keeps true.
///
/// A `panic` rather than an `Option` because it is reached at most once per
/// entry and only ever fails when this file names a routine the crate does not
/// have -- which is a mistake in the line below it, not a condition.
fn of(name: &'static str) -> &'static kernels_vulkan::routine::Routine {
    match kernels_vulkan::ROUTINES.iter().find(|r| r.name == name) {
        Some(r) => r,
        None => panic!("the arm registry names a routine this crate does not hold"),
    }
}

static LIVE: std::sync::LazyLock<Vec<Crossed>> = std::sync::LazyLock::new(|| {
    vec![
        // sample -- sample/argmax.slang
        Crossed {
            stem: "argmax_logits",
            routine: Some(of("argmax_logits")),
        },
        // ptir -- ptir/logits_copy.slang
        Crossed {
            stem: "copy_logits_bf16",
            routine: Some(of("copy_logits_bf16")),
        },
        // mlp -- mlp/gated.slang
        Crossed {
            stem: "geglu_tanh",
            routine: Some(of("geglu_tanh")),
        },
        Crossed {
            stem: "geglu_tanh_strided",
            routine: Some(of("geglu_tanh_strided")),
        },
        Crossed {
            stem: "gptoss_swiglu",
            routine: Some(of("gptoss_swiglu")),
        },
        Crossed {
            stem: "silu_mul",
            routine: Some(of("silu_mul")),
        },
        // layout -- layout/embed.slang, layout/ple.slang, layout/gather.slang
        Crossed {
            stem: "embed_gather_4bit",
            routine: Some(of("embed_gather_4bit")),
        },
        Crossed {
            stem: "embed_gather_mb_4bit",
            routine: Some(of("embed_gather_mb_4bit")),
        },
        Crossed {
            stem: "embed_gather_scaled_4bit",
            routine: Some(of("embed_gather_scaled_4bit")),
        },
        Crossed {
            stem: "embed_gather_scaled_mb_4bit",
            routine: Some(of("embed_gather_scaled_mb_4bit")),
        },
        Crossed {
            stem: "ple_combine",
            routine: Some(of("ple_combine")),
        },
        Crossed {
            stem: "row_gather",
            routine: Some(of("row_gather")),
        },
        Crossed {
            stem: "silu_mul_strided",
            routine: Some(of("silu_mul_strided")),
        },
        // rope
        Crossed {
            stem: "neox_decode",
            routine: Some(of("neox_decode")),
        },
        Crossed {
            stem: "neox_mb",
            routine: Some(of("neox_mb")),
        },
        Crossed {
            stem: "neox_prop_decode",
            routine: Some(of("neox_prop_decode")),
        },
        Crossed {
            stem: "neox_prop_mb",
            routine: Some(of("neox_prop_mb")),
        },
        Crossed {
            stem: "neox_freqs_decode",
            routine: Some(of("neox_freqs_decode")),
        },
        Crossed {
            stem: "neox_freqs_mb",
            routine: Some(of("neox_freqs_mb")),
        },
        Crossed {
            stem: "neox_strided",
            routine: Some(of("neox_strided")),
        },
        // norm
        Crossed {
            stem: "rms_single_row",
            routine: Some(of("rms_single_row")),
        },
        Crossed {
            stem: "vnorm_single_row",
            routine: Some(of("vnorm_single_row")),
        },
        Crossed {
            stem: "rms_residual",
            routine: Some(of("rms_residual")),
        },
        Crossed {
            stem: "rms_residual_scaled",
            routine: Some(of("rms_residual_scaled")),
        },
        Crossed {
            stem: "rms_strided_row",
            routine: Some(of("rms_strided_row")),
        },
        Crossed {
            stem: "rms_strided_head_row",
            routine: Some(of("rms_strided_head_row")),
        },
        Crossed {
            stem: "rms_rope",
            routine: Some(of("rms_rope")),
        },
        Crossed {
            stem: "gated_rms",
            routine: Some(of("gated_rms")),
        },
        Crossed {
            stem: "gated_rms_strided",
            routine: Some(of("gated_rms_strided")),
        },
        Crossed {
            stem: "layer_scalar_mul",
            routine: Some(of("layer_scalar_mul")),
        },
        Crossed {
            stem: "residual_add",
            routine: Some(of("residual_add")),
        },
        Crossed {
            stem: "residual_add_strided",
            routine: Some(of("residual_add_strided")),
        },
        Crossed {
            stem: "add_bias",
            routine: Some(of("add_bias")),
        },
        // ssm
        Crossed {
            stem: "gdn_prep",
            routine: Some(of("gdn_prep")),
        },
        Crossed {
            stem: "gdn_prep_slotted",
            routine: Some(of("gdn_prep_slotted")),
        },
        Crossed {
            stem: "gdn_prep_prefill",
            routine: Some(of("gdn_prep_prefill")),
        },
        Crossed {
            stem: "gdn_core",
            routine: Some(of("gdn_core")),
        },
        Crossed {
            stem: "gdn_core_slotted",
            routine: Some(of("gdn_core_slotted")),
        },
        Crossed {
            stem: "gdn_core_recurrent",
            routine: Some(of("gdn_core_recurrent")),
        },
        Crossed {
            stem: "gdn_core_recurrent_slotted",
            routine: Some(of("gdn_core_recurrent_slotted")),
        },
        Crossed {
            stem: "gdn_core_recurrent_prefill",
            routine: Some(of("gdn_core_recurrent_prefill")),
        },
        // moe
        Crossed {
            stem: "router_topk",
            routine: Some(of("router_topk")),
        },
        Crossed {
            stem: "router_topk_scaled",
            routine: Some(of("router_topk_scaled")),
        },
        Crossed {
            stem: "route_sort",
            routine: Some(of("route_sort")),
        },
        Crossed {
            stem: "route_gather",
            routine: Some(of("route_gather")),
        },
        Crossed {
            stem: "combine_sorted",
            routine: Some(of("combine_sorted")),
        },
        Crossed {
            stem: "shared_expert_combine",
            routine: Some(of("shared_expert_combine")),
        },
        Crossed {
            stem: "shared_expert_combine_strided",
            routine: Some(of("shared_expert_combine_strided")),
        },
        Crossed {
            stem: "affine_qmv_routed",
            routine: Some(of("qmv_routed")),
        },
        Crossed {
            stem: "affine_qmv_routed_bias",
            routine: Some(of("qmv_routed_bias")),
        },
        Crossed {
            stem: "mxfp4_qmv_routed_bias",
            routine: Some(of("mxfp4_qmv_routed_bias")),
        },
        Crossed {
            stem: "affine_qmm_t_routed",
            routine: Some(of("qmm_t_routed")),
        },
        Crossed {
            stem: "affine_qmm_t_routed_fp16",
            routine: Some(of("qmm_t_routed_fp16")),
        },
        Crossed {
            stem: "mxfp4_qmm_t_routed_bias",
            routine: Some(of("mxfp4_qmm_t_routed_bias")),
        },
        // attn
        Crossed {
            stem: "sdpa_paged_decode",
            routine: Some(of("sdpa_paged_decode")),
        },
        Crossed {
            stem: "sdpa_paged_decode_sink",
            routine: Some(of("sdpa_paged_decode_sink")),
        },
        Crossed {
            stem: "sdpa_paged_tiled",
            routine: Some(of("sdpa_paged_tiled")),
        },
        Crossed {
            stem: "sdpa_paged_tiled_sink",
            routine: Some(of("sdpa_paged_tiled_sink")),
        },
        Crossed {
            stem: "sdpa_paged_tiled_strided",
            routine: Some(of("sdpa_paged_tiled_strided")),
        },
        Crossed {
            stem: "sdpa_paged_mma",
            routine: Some(of("sdpa_paged_mma")),
        },
        Crossed {
            stem: "sdpa_paged_mma_sink",
            routine: Some(of("sdpa_paged_mma_sink")),
        },
        Crossed {
            stem: "sdpa_vector_decode",
            routine: Some(of("sdpa_vector_decode")),
        },
        Crossed {
            stem: "sdpa_vector_decode_swa",
            routine: Some(of("sdpa_vector_decode_swa")),
        },
        Crossed {
            stem: "sdpa_vector_decode_sink",
            routine: Some(of("sdpa_vector_decode_sink")),
        },
        Crossed {
            stem: "kv_append",
            routine: Some(of("kv_append")),
        },
        Crossed {
            stem: "kv_append_paged",
            routine: Some(of("kv_append_paged")),
        },
        Crossed {
            stem: "split_qkv_bf16",
            routine: Some(of("split_qkv_bf16")),
        },
        Crossed {
            stem: "gate",
            routine: Some(of("gate")),
        },
        Crossed {
            stem: "q_gate_split",
            routine: Some(of("q_gate_split")),
        },
        Crossed {
            stem: "logit_softcap",
            routine: Some(of("logit_softcap")),
        },
        // quant
        Crossed {
            stem: "cast_qmm_input_bfloat16_to_float16",
            routine: Some(of("cast_qmm_input_bfloat16_to_float16")),
        },
        Crossed {
            stem: "cast_qmm_input_strided_bfloat16_to_float16",
            routine: Some(of("cast_qmm_input_strided_bfloat16_to_float16")),
        },
        Crossed {
            stem: "affine_encode_u4_bf16",
            routine: Some(of("encode_u4_bf16")),
        },
        Crossed {
            stem: "affine_encode_u4_f32",
            routine: Some(of("encode_u4_f32")),
        },
        Crossed {
            stem: "mxfp4_dequant_bf16",
            routine: Some(of("mxfp4_dequant_bf16")),
        },
        Crossed {
            stem: "qmm_splitk_reduce",
            routine: Some(of("qmm_splitk_reduce")),
        },
        Crossed {
            stem: "qmm_splitk_reduce_f32",
            routine: Some(of("qmm_splitk_reduce_f32")),
        },
        Crossed {
            stem: "affine_qmm_t",
            routine: Some(of("qmm_t")),
        },
        Crossed {
            stem: "affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4",
            routine: Some(of("qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4")),
        },
        Crossed {
            stem: "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2",
            routine: Some(of("qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2")),
        },
        Crossed {
            stem: "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2",
            routine: Some(of("qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2")),
        },
        Crossed {
            stem: "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1",
            routine: Some(of("qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1")),
        },
        Crossed {
            stem: "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4",
            routine: Some(of("qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4")),
        },
        Crossed {
            stem: "affine_qmm_t_bias",
            routine: Some(of("qmm_t_bias")),
        },
        Crossed {
            stem: "affine_qmm_t_bias_fp16_precast",
            routine: Some(of("qmm_t_bias_fp16_precast")),
        },
        Crossed {
            stem: "affine_qmm_t_fp16_precast",
            routine: Some(of("qmm_t_fp16_precast")),
        },
        Crossed {
            stem: "affine_qmm_t_residual",
            routine: Some(of("qmm_t_residual")),
        },
        Crossed {
            stem: "affine_qmm_t_residual_fp16_precast",
            routine: Some(of("qmm_t_residual_fp16_precast")),
        },
        Crossed {
            stem: "affine_qmm_t_splitk",
            routine: Some(of("qmm_t_splitk")),
        },
        Crossed {
            stem: "affine_qmm_t_splitk_f32",
            routine: Some(of("qmm_t_splitk_f32")),
        },
        Crossed {
            stem: "affine_qmm_t_splitk_fp16_precast",
            routine: Some(of("qmm_t_splitk_fp16_precast")),
        },
        Crossed {
            stem: "affine_qmm_t_splitk_fp16_precast_f32",
            routine: Some(of("qmm_t_splitk_fp16_precast_f32")),
        },
        Crossed {
            stem: "affine_qmm_t_strided",
            routine: Some(of("qmm_t_strided")),
        },
        Crossed {
            stem: "affine_qmm_t_strided_fp16_precast",
            routine: Some(of("qmm_t_strided_fp16_precast")),
        },
        Crossed {
            stem: "affine_qmm_t_strided_fp16_precast_residual",
            routine: Some(of("qmm_t_strided_fp16_precast_residual")),
        },
        Crossed {
            stem: "affine_qmm_t_strided_residual",
            routine: Some(of("qmm_t_strided_residual")),
        },
        Crossed {
            stem: "affine_qmv_fast",
            routine: Some(of("qmv_fast")),
        },
        Crossed {
            stem: "affine_qmv_fast_residual",
            routine: Some(of("qmv_fast_residual")),
        },
        Crossed {
            stem: "affine_qmv_tail",
            routine: Some(of("qmv_tail")),
        },
        Crossed {
            stem: "affine_qmv_tail_bias",
            routine: Some(of("qmv_tail_bias")),
        },
        Crossed {
            stem: "affine_qmv_wide_strided",
            routine: Some(of("qmv_wide_strided")),
        },
    ]
});

#[cfg(test)]
mod tests {
    use super::*;

    /// A plan's fully instantiated symbol reaches the routine that serves it,
    /// through the registry's OWN lookup.
    ///
    /// This is the join the whole fork rests on. A plan names
    /// `argmax_logits_bfloat16` and the routine is `argmax_logits`, after the
    /// kernel rather than after the entrypoint, so something has to bridge
    /// them -- and until this test's subject existed that something was the
    /// `kernel!` table, which is what the refactor is deleting.
    #[test]
    fn a_plans_instantiated_symbol_reaches_the_routine_that_serves_it() {
        for (symbol, want) in [
            ("argmax_logits_bfloat16", "argmax_logits"),
            ("copy_logits_bf16", "copy_logits_bf16"),
        ] {
            let routine =
                routine_for(symbol).unwrap_or_else(|| panic!("nothing serves `{symbol}`"));
            assert_eq!(routine.name, want);
        }
    }

    /// The lookup needs no `kernel!` row, which is the whole point of it.
    ///
    /// This asserted `kernels::sig_in(KERNELS, "argmax_logits_bfloat16")
    /// .is_none()` beside the line below -- the rows' ABSENCE rather than
    /// their presence, so that a family could delete its rows in the same
    /// commit that added its arms. Every family has, `KERNELS` is empty, and
    /// this crate no longer names it: the `sig_in` half would now be a claim
    /// about a table that has no rows for anything, which is not the claim.
    ///
    /// What is left is the half that still says something: the registry finds
    /// an entrypoint by its own stems.
    #[test]
    fn the_lookup_reads_no_kernel_row() {
        assert!(routine_for("argmax_logits_bfloat16").is_some());
    }

    /// The longest stem wins, and a stem that runs into the middle of a longer
    /// name is not a match.
    ///
    /// Both halves are defects that dispatch. `qmm_t` is a prefix of every
    /// `qmm_t_splitk` symbol, so first-match would send a split-K rectangle to
    /// the single-pass body -- which binds real buffers, succeeds, and leaves
    /// a partial sum where a total belongs. Dropping the separator rule is the
    /// same defect one letter along, with `qmm_t` claiming `qmm_t_strided`.
    ///
    /// Written against stems this file states for the test rather than against
    /// the live registry, so it keeps saying the same thing as families land.
    #[test]
    fn the_longest_stem_wins_and_a_stem_may_not_end_mid_word() {
        fn pick<'a>(stems: &[&'a str], symbol: &str) -> Option<&'a str> {
            stems
                .iter()
                .filter(|s| {
                    symbol
                        .strip_prefix(**s)
                        .is_some_and(|rest| rest.is_empty() || rest.starts_with('_'))
                })
                .max_by_key(|s| s.len())
                .copied()
        }
        let stems = ["qmm_t", "qmm_t_splitk", "qmm_t_strided"];
        assert_eq!(
            pick(&stems, "qmm_t_splitk_bfloat16_gs_64_b_4"),
            Some("qmm_t_splitk"),
            "first-match would run the single-pass body over a split-K rectangle"
        );
        assert_eq!(pick(&stems, "qmm_t_bfloat16_gs_64_b_4"), Some("qmm_t"));
        assert_eq!(
            pick(&["qmm_t"], "qmm_t_strided_bfloat16"),
            Some("qmm_t"),
            "and this is why `qmm_t_strided` must be in the list rather than \
             left to fall through"
        );
        assert_eq!(
            pick(&["argmax_logit"], "argmax_logits_bfloat16"),
            None,
            "a stem ending mid-word is not a match"
        );
    }

    /// No family is dark any more.
    ///
    /// While the crossing ran, a family whose rows were gone but whose arms
    /// were unwritten had to be UNREACHABLE -- reaching it would have run a
    /// body against operands nothing had worked out -- and this test named
    /// one such symbol and asserted `arm_for` refused it.
    ///
    /// There is no such symbol left to name, so the test says the thing the
    /// naming was for instead: every routine this crate declares is filed in
    /// `LIVE` with an arm. Written over the routine list rather than over a
    /// symbol, because the failure it guards is a routine that crossed and
    /// was never wired, and that is a hole in the list, not a bad stem.
    #[test]
    fn no_routine_this_crate_declares_is_left_without_an_arm() {
        let unwired: Vec<&str> = kernels_vulkan::routines()
            .iter()
            .map(|r| r.name)
            .filter(|name| {
                !LIVE
                    .iter()
                    .any(|c| c.routine.is_some_and(|r| r.name == *name))
            })
            .collect();
        assert!(
            unwired.is_empty(),
            "these routines crossed and reach no arm: {}",
            unwired.join(", ")
        );
    }

    /// Every family that has landed is landed WHOLE, and every stem resolves
    /// to the routine it is filed against.
    ///
    /// A family that crossed halfway is the worst of the intermediate states:
    /// its rows are gone, so the launch path cannot fall back, and the
    /// routines that were missed are unreachable. The rows and the arms have
    /// to move together, and this is what says so.
    #[test]
    fn a_landed_family_is_landed_whole_and_every_stem_finds_its_own_routine() {
        for c in LIVE.iter() {
            let Some(want) = c.routine else {
                // A RESERVED stem resolves to nothing on purpose, and that
                // is the whole of what it has to do.
                assert!(
                    routine_for(c.stem).is_none(),
                    "`{}` is reserved and must serve nothing",
                    c.stem
                );
                continue;
            };
            let found = routine_for(c.stem)
                .unwrap_or_else(|| panic!("`{}` does not resolve itself", c.stem));
            assert_eq!(
                found.name, want.name,
                "`{}` resolves to the wrong routine",
                c.stem
            );
        }
        // ONE LIST, WHERE THERE WERE TEN. Each module used to publish its own
        // `ROUTINES`, and this walk named all ten so that a module added
        // without a line here would be missed rather than silently uncovered.
        // The crate publishes a single `routines()` now -- the census the
        // proc macro builds -- so naming it is naming all of them, and the
        // gap this loop was shaped to catch cannot open.
        for r in kernels_vulkan::routines() {
            assert!(
                LIVE.iter()
                    .any(|c| c.routine.is_some_and(|held| held.name == r.name)),
                "`{}`'s family has landed, so it needs an arm too",
                r.name
            );
        }
    }

    /// The longest stem wins, so a stem that is another's prefix does not
    /// steal it.
    ///
    /// `silu_mul` is a prefix of `silu_mul_strided` followed by a separator.
    /// While the strided routine was unwritten this was answered by RESERVING
    /// the longer stem; now both are served, and the rule that has to hold is
    /// the same one: `silu_mul_strided_bfloat16` reaches the strided body and
    /// not the contiguous one, which would read three operands at the wrong
    /// pitches and return success.
    #[test]
    fn the_longest_stem_wins_over_the_one_that_is_its_prefix() {
        let strided = routine_for("silu_mul_strided_bfloat16").expect("a strided body");
        assert_eq!(strided.name, "silu_mul_strided");
        let flat = routine_for("silu_mul_bfloat16").expect("a contiguous body");
        assert_eq!(flat.name, "silu_mul");
    }

    /// The affine axis is read off the entrypoint, which is where this backend
    /// writes it down.
    ///
    /// Metal's `Geometry` carries the group and the bits; this one does not,
    /// because here they are spelled into the name `kernels::sig_in` matches a
    /// row through. A kernel with no affine axis has none, and that is `None`
    /// rather than a zero that would read as a group size.
    #[test]
    fn the_affine_axis_comes_off_the_entrypoint() {
        assert_eq!(
            affine_of("affine_qmv_fast_bfloat16_gs_64_b_4"),
            Some((64, 4))
        );
        assert_eq!(
            affine_of("affine_qmv_fast_bfloat16_gs_128_b_8"),
            Some((128, 8))
        );
        assert_eq!(affine_of("argmax_logits_bfloat16"), None);
        assert_eq!(affine_of("copy_logits_bf16"), None);
    }

    /// The trace concatenates inputs, then results, then weights, and a
    /// routine's own `BufMut` count is what says where the first two divide.
    #[test]
    fn the_split_puts_the_results_last_and_the_weights_aside() {
        let args = [
            Arg::Arena {
                at: 0,
                width: 4,
                bytes: 2,
            },
            Arg::Arena {
                at: 8,
                width: 4,
                bytes: 2,
            },
            Arg::Weight("layer.0.q_proj".to_owned()),
            Arg::Arena {
                at: 16,
                width: 4,
                bytes: 2,
            },
        ];
        let (ins, outs, weights) = split(&args, 1);
        assert_eq!(ins, vec![0, 1], "two inputs");
        assert_eq!(outs, vec![3], "the one result is the LAST widthed operand");
        assert_eq!(
            weights,
            vec![2],
            "a weight is not widthed and takes no place"
        );
    }

    /// Every entrypoint this crate can be asked for finds an arm.
    ///
    /// The stems here are ENTRYPOINT stems, not routine names, and the two
    /// are not always the same word: `quant::qmm_t`'s entrypoints all begin
    /// `affine_qmm_t`, because the row that used to state them named the
    /// routine one way and the shader another. Thirty rows in `quant` and
    /// `moe` are like that, and while the table still held them the bridge
    /// was the row. It is this list now, so the list has to say the shader's
    /// word.
    ///
    /// Stated as a sweep over `entrypoints()` rather than as a list of the
    /// thirty, because the failure it catches is not "this stem is wrong" but
    /// "a symbol a plan can name reaches nothing" -- which is silent: the
    /// dispatch is dropped and the frame runs short.
    #[test]
    fn every_entrypoint_a_plan_can_name_finds_an_arm() {
        let dark: Vec<String> = kernels_vulkan::entrypoints()
            .into_iter()
            .filter(|e| routine_for(e).is_none())
            .collect();
        assert!(
            dark.is_empty(),
            "{} entrypoints reach no arm, first few: {}",
            dark.len(),
            dark.iter().take(8).cloned().collect::<Vec<_>>().join(", ")
        );
    }

    /// The two routines whose results the POOL supplies STATE none.
    ///
    /// Both write the KV cache and neither takes it as an operand: the bodies
    /// ask for it (`ctx.ask::<_, keys::KvKeys>()`), which is a fact and not an
    /// argument, so there is no writable type in either signature to discount.
    ///
    /// The regression this pins is the discount that outlived the marks.
    /// `traced_results` subtracted two for this pair on the grounds that their
    /// signatures counted two writable buffers the statement does not supply;
    /// after the draw became an ask, the count it subtracted from was ZERO and
    /// the subtraction panicked the driver lane at the first `kv_append` of
    /// every fire. Asserting the signature's own count is what makes the
    /// discount's absence checkable rather than assumed.
    #[test]
    fn a_result_the_pool_supplies_is_not_a_result_the_statement_carries() {
        for name in ["kv_append", "kv_append_paged"] {
            let r = kernels_vulkan::routines()
                .iter()
                .find(|r| r.name == name)
                .copied()
                .expect("the routine is declared");
            assert_eq!(
                r.args
                    .iter()
                    .filter(|ty| ty.binds() == kernels::Binds::Writes)
                    .count(),
                0,
                "`{name}` asks for both cache planes, so it states no mutable buffer"
            );
            assert_eq!(
                traced_results(r),
                0,
                "`{name}` draws both from the pool, so the statement carries none"
            );
        }
    }

    /// EVERY routine takes its results from the statement.
    ///
    /// There is no exception list any more, and that is the claim: a discount
    /// naming routines by name is a list that can grow silently wrong, and the
    /// one this crate carried did -- it kept subtracting after the routines it
    /// named stopped stating what it subtracted.
    #[test]
    fn every_routines_results_are_the_ones_it_declares() {
        for r in kernels_vulkan::routines() {
            let declared = r
                .args
                .iter()
                .filter(|ty| ty.binds() == kernels::Binds::Writes)
                .count();
            assert_eq!(
                traced_results(r),
                declared,
                "`{}` must be counted whole",
                r.name
            );
        }
    }
}
