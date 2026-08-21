//! Calling a routine, and the plan a routine's dispatch lands in.
//!
//! `.wiki/kernel-x/refactor-bigplan.md` §7 Stage 2. The driver half of the
//! seam `kernels-metal/src/routine.rs` opened: the thing that turns a routine
//! body's statement — *this entrypoint, over these threads, with these
//! arguments* — into the [`Dispatch`] the rest of this driver already knows
//! how to encode.
//!
//! # Why this replaces `plan_one` and not `encode_one`
//!
//! `fire::run` does four things to a plan before an encoder exists: it sizes
//! the argument table from the whole of it, it batch-compiles every pipeline
//! it names, it fingerprints it for indirect-command-buffer replay, and only
//! then does it submit. `Dispatch`'s own doc is *"everything a command encoder
//! needs, and nothing that needs a command encoder to compute."*
//!
//! So a routine body does not encode. [`Planner`] implements
//! [`kernels_metal::routine::Encode`] by BUILDING dispatches, and all four
//! passes are untouched. The plan stays data.
//!
//! # What is empty here, and why it ships that way
//!
//! [`arm_for`] answers for `sample` and for nothing else. Every other family
//! has crossed — its routines are written and compared against the other
//! backends — and none of them has an ARM, so every other symbol takes the
//! table path and behaves exactly as before.
//!
//! The split is deliberate and it is the refactor's whole shape: a routine is
//! a statement about a kernel, checkable without a device, and an arm is a
//! statement about a TRACE, checkable only against one. They fail
//! differently, so they land in different commits.

use core::cell::RefCell;

use kernels::Ty;
use kernels::routine::{Refusal, Routine};
use kernels_metal::routine::{ArgValue, Encode, Fire, Metal};

use crate::lowering::hold::Staged;

use crate::lowering::dispatch::{Dispatch, ParamSlot, Touches};
use crate::lowering::executor::{BoundArg, Slice};

/// An operand slot that addresses nothing.
///
/// What a scalar's argument-table entry holds: the value rides
/// [`Dispatch::params`] and is bound through a [`ParamSlot`], so the slot
/// itself is a zero address. The same answer `reorder` gives on the table
/// path, for the same reason — and the reason the two planes produce
/// `Dispatch`es nothing downstream can tell apart.
const NOTHING: BoundArg = BoundArg {
    slice: Slice {
        address: 0,
        bytes: 0,
    },
    width: 0,
};

/// What a routine body's dispatches are accumulated into.
///
/// `Encode::dispatch` takes `&self` — [`kernels::routine::Backend::Ctx`]
/// reaches a body as a shared reference — so the accumulation is behind a
/// [`RefCell`]. That is the whole of the interior mutability: one push per
/// dispatch, and `driver-vulkan`'s `Encoder` is behind the same one.
///
/// A body may state more than one dispatch, and this keeps them in the order
/// it stated them. That is not hypothetical on this backend: a two-pass
/// reduction is two entrypoints over one statement, and a plane that could
/// only carry one would push the second back into the lowering, which is
/// where it was before the refactor.
pub struct Planner<'r> {
    /// The routine's declared argument types, positionally.
    ///
    /// Read for exactly one thing the VALUE cannot say: an [`Ty::InPacked`]
    /// argument arrives as an [`ArgValue::U32`] and is not one. The type is
    /// the only place that difference is written down, which is why a planner
    /// is given the routine and not just its values.
    types: &'r [Ty],
    /// The buffers this launch bound, in the trace's order. A body's handle is
    /// an index into this.
    handles: &'r [BoundArg],
    /// WHAT THIS FIRE ANSWERS, for a body that asks.
    ///
    /// `Env` left the parameter list, so a fact only the fire can answer is no
    /// longer bound into the values before the body runs. The body asks, and
    /// this is the same [`crate::lowering::hold::Handles`] the binder used,
    /// through the same [`kernels::bind::one`].
    ///
    /// `RefCell` because answering MINTS: a staged fact takes a handle, and
    /// the body holds only a `&self`.
    answers: Option<(
        &'r RefCell<crate::lowering::hold::Handles<'r>>,
        crate::lowering::hold::Facts,
    )>,
    /// The layers the rectangle covers, and which traced op it came from.
    ///
    /// From the LAUNCH, not from the body: where in the trace a rectangle sits
    /// is a fact a routine has no way to know and no business stating.
    layers: core::ops::Range<u16>,
    /// See [`Planner::layers`].
    op: u32,
    /// The launch's staged scalars, and the handle that stands for them.
    staged: Staged<'r>,
    /// What the body asked for, in the order it asked.
    out: RefCell<Vec<Dispatch<'static>>>,
}

impl<'r> Planner<'r> {
    /// A planner for one launch of `routine`, over the operands `handles`.
    #[must_use]
    pub fn new(
        routine: &'r Routine<Metal>,
        handles: &'r [BoundArg],
        staged: Staged<'r>,
        layers: core::ops::Range<u16>,
        op: u32,
    ) -> Self {
        Self {
            types: routine.args,
            handles,
            layers,
            op,
            staged,
            out: RefCell::new(Vec::new()),
            answers: None,
        }
    }

    /// The same planner, able to ANSWER a body that asks.
    ///
    /// Separate from [`Self::new`] because a probe planner has no fire behind
    /// it and must not pretend to: a body that asks on one gets
    /// [`Refusal::Unstated`], which is the honest answer.
    #[must_use]
    pub fn answering(
        mut self,
        handles: &'r RefCell<crate::lowering::hold::Handles<'r>>,
        facts: crate::lowering::hold::Facts,
    ) -> Self {
        self.answers = Some((handles, facts));
        self
    }

    /// The dispatches the body asked for, in the order it asked.
    #[must_use]
    pub fn finish(self) -> Vec<Dispatch<'static>> {
        self.out.into_inner()
    }
}

/// Lay one dispatch's arguments out as the argument table and scalar run.
///
/// Metal's argument table has **one slot per operand**, buffers and scalars
/// alike: a buffer's slot holds its address, and a scalar's holds nothing
/// while its bits ride the staged run and a [`ParamSlot`] joins the two. So
/// this walks the body's own list once and the position in that list IS the
/// slot — which is the whole of what `reorder` and `param_layout` computed
/// from a row's `operands` column, now derived from the signature instead.
///
/// # Errors
///
/// [`Refusal::Absent`] for a handle the launch did not bind — a body reaching
/// past its statement, which is not answered with a zero address. On the table
/// path that answer was a live defect: `mxfp4_qmv_routed_bias` read an
/// additive bias off a null pointer for every expert logit and nothing in the
/// path said a word.
///
/// [`Refusal::Unstated`] for a packed field, which this seam does not lay out
/// yet — see the `InPacked` arm.
/// What a laid-out argument run IS: the handles a dispatch binds, the slots
/// its parameter block is described by, and the words of the block itself.
/// Named because the tuple is three parallel statements about ONE run and a
/// reader who meets it bare has to reconstruct which is which.
type LaidOut = (Vec<BoundArg>, Vec<ParamSlot>, Vec<u32>);

fn lay_out(
    values: &[ArgValue],
    types: &[Ty],
    handles: &[BoundArg],
    staged: Staged<'_>,
) -> Result<LaidOut, Refusal> {
    let mut args = Vec::with_capacity(values.len());
    let mut params: Vec<u32> = Vec::new();
    let mut slots: Vec<ParamSlot> = Vec::new();
    let mut at = 0u32;
    let mut block: Option<usize> = None;
    let mut fields: Vec<u32> = Vec::new();

    for (slot, value) in values.iter().enumerate() {
        // The one thing the VALUE cannot say. `InPacked` arrives as an
        // `ArgValue::U32` and is not one: it is a FIELD of the struct an
        // earlier argument binds, so it takes no slot of its own and its word
        // is appended to that struct's run.
        //
        // `layout::row_gather` is the shape. `RowGatherParams` is width then
        // count; the statement carries the width, the driver knows the
        // request count, and the shader declares no buffer for it -- the
        // count is the struct's second word and nothing else. The type is now
        // the only statement of that. Under `kernel!` it was said twice, in
        // the row and in the shader's struct, and only one of the two was
        // ever checked.
        if types.get(slot).copied() == Some(Ty::InPacked) {
            let ArgValue::U32(word) = *value else {
                return Err(Refusal::Kind {
                    at: slot,
                    want: Ty::InPacked,
                });
            };
            if block.is_none() {
                return Err(Refusal::Unstated {
                    what: "a packed field with no block before it to be a field OF",
                });
            }
            fields.push(word);
            args.push(NOTHING);
            continue;
        }
        // The packed block, which is a pointer into the staged run and not an
        // address of its own. Its struct covers the run from its first word to
        // the END -- `ParamSlot::packed` is what the encoder reads -- so a
        // dispatch can hold at most one, and it is laid FIRST so that
        // anything else in the run falls after its last declared field and is
        // simply not read. `norm::rms_strided_row` is the shape: a
        // `constant RmsParams&` at buffer 3 and a `constant int& row_pitch`
        // at buffer 4, which the two cannot be if the struct starts anywhere
        // but zero.
        if staged.block
            == Some(match *value {
                // A `Shaped` HANDLE IS STILL A HANDLE, here as much as in the
                // arm below. Leaving it out made the staged block unmatchable
                // the moment a params handle arrived with its rectangle beside
                // it -- `packed` was never set, the encoder bound a scalar run
                // where a struct pointer belongs, and no test could see it
                // because both are a handle at a slot.
                ArgValue::Buffer(handle)
                | ArgValue::BufferMut(handle)
                | ArgValue::Shaped { handle, .. } => handle,
                _ => u32::MAX,
            })
        {
            if block.is_some() {
                return Err(Refusal::Unstated {
                    what: "a second packed block: one staged run cannot hold two structs",
                });
            }
            block = Some(slot);
            args.push(NOTHING);
            continue;
        }
        match *value {
            // A `Shaped` HANDLE IS STILL A HANDLE: it carries the rectangle
            // the statement gave the operand, which the marks read and an
            // encoder does not.
            ArgValue::Buffer(handle)
            | ArgValue::BufferMut(handle)
            | ArgValue::Shaped { handle, .. } => {
                let bound = handles
                    .get(handle as usize)
                    .ok_or(Refusal::Absent { what: "a buffer" })?;
                args.push(*bound);
                continue;
            }
            // Every scalar lands the same way: a zero argument slot, its bits
            // appended to the staged run, and a `ParamSlot` joining the two.
            // The kinds differ only in WIDTH, which is the difference that
            // matters — an eight-byte read of a four-byte slot takes the next
            // scalar as its high half, and the kernels that read
            // `const constant size_t&` are the ones that would.
            ArgValue::I32(v) => scalar(
                &mut params,
                &mut slots,
                &mut at,
                slot,
                4,
                &[v.cast_unsigned()],
            ),
            ArgValue::U32(v) => scalar(&mut params, &mut slots, &mut at, slot, 4, &[v]),
            ArgValue::F32(v) => scalar(&mut params, &mut slots, &mut at, slot, 4, &[v.to_bits()]),
            ArgValue::Usize(v) => scalar(
                &mut params,
                &mut slots,
                &mut at,
                slot,
                8,
                &[
                    // Low word first. `params` is a run of `u32` the stage copies
                    // verbatim into a little-endian buffer, and every Apple GPU
                    // is little-endian; `driver-vulkan`'s `words` splits the same
                    // value the same way.
                    (v & 0xffff_ffff) as u32,
                    (v >> 32) as u32,
                ],
            ),
            // A raised view is HOST data the body already read; it names no
            // slot and packs no scalar, and metal's argument list is
            // positional -- so a body that passed one back has stated a slot
            // nothing can fill, which is a refusal and not a skip.
            ArgValue::Raised(_) => {
                return Err(Refusal::Unstated {
                    what: "a raised view in a dispatch argument list: a view is \
                           host data a body reads, not a slot it binds",
                });
            }
        }
        args.push(NOTHING);
    }
    if let Some(slot) = block {
        // The struct's own fields, after the statement's -- `Ty::InPacked`
        // above. The order is the signature's, which is the struct's.
        let mut prefix = staged.words.to_vec();
        prefix.extend_from_slice(&fields);
        // Every loose scalar moves down by the struct's length. They were
        // laid from zero because the walk did not yet know whether a block
        // would claim that address, and the block must have it: its run has
        // no end but the run's, so a scalar BEFORE it would be read as a
        // field while a scalar after it is past the last one declared.
        let shift = u32::try_from(prefix.len() * 4).map_err(|_| Refusal::Grid {
            what: "a staged struct too long to address",
            at: prefix.len() as i64 * 4,
        })?;
        let moved = u8::try_from(prefix.len()).unwrap_or(u8::MAX);
        for s in &mut slots {
            s.at += shift;
            s.value = s.value.map(|v| v.saturating_add(moved));
        }
        prefix.append(&mut params);
        params = prefix;
        slots.push(ParamSlot {
            slot,
            at: 0,
            bytes: 4,
            packed: true,
            value: Some(0),
        });
    }
    Ok((args, slots, params))
}

/// Append one scalar's words and the slot that binds them.
///
/// `at` advances to the value's natural alignment FIRST, so an eight-byte
/// stride starts on an eight-byte boundary rather than wherever the previous
/// four-byte extent happened to end. That is the rule `param_layout` follows,
/// and it has to be the same rule because the same shaders read both runs.
fn scalar(
    params: &mut Vec<u32>,
    slots: &mut Vec<ParamSlot>,
    at: &mut u32,
    slot: usize,
    bytes: u32,
    words: &[u32],
) {
    let first = u8::try_from(params.len()).unwrap_or(u8::MAX);
    params.extend_from_slice(words);
    *at = at.next_multiple_of(bytes);
    slots.push(ParamSlot {
        slot,
        at: *at,
        bytes,
        packed: false,
        value: Some(first),
    });
    *at += bytes;
}

/// What a dispatch reads and what it may write, off the VALUES.
///
/// The direction was always there and this used to throw it away: a routine
/// spells a written buffer `BufMut` or `F32sMut` and a read one `Buf`,
/// `I32s`, `U32s`, `U8s`, `F32s`. So a launch's writes are exactly the
/// handles bound at the mutable arguments, and every other bound handle is a
/// read.
///
/// The conservative answer this replaces -- every operand as both -- was
/// honest and expensive. `Touches` decides whether an encoder may run two
/// launches at once, so calling a read a write inserts a barrier between two
/// dispatches that never meet, and on a decode where the routine path now
/// carries every launch that is a barrier per launch rather than per hazard.
///
/// # It read the SIGNATURE at the value's position, and those are two lists
///
/// `types[i]` is the i-th PARAMETER; `values[i]` is the i-th BUFFER SLOT.
/// They are the same list only for a routine whose entrypoint numbers its
/// buffers with no holes. Twenty-three do not: an entrypoint that declares
/// buffers at 2, 3, 6, 7, 8 and 10 fills the holes with a `pad` taken once in
/// the signature and bound five times in the dispatch, and every argument
/// after the first hole then read its neighbour's type.
///
/// `gdn_core_recurrent_prefill` is the one that showed. Its `core_out` sits
/// at slot 3 and the signature's fourth entry is `pre_q`, so the scan's only
/// real output was declared a READ; `mixed`, bound at the `pad` in slot 1,
/// was declared a WRITE in its place. The encoder saw no hazard between the
/// scan and the `gated_rms` that consumes it, ran them at once, and qwen3.6
/// answered a two-token prompt differently every time it was asked --
/// thirteen logits apart, from the same program fired twice.
/// `qmm_splitk_reduce` declared its `y` the same way and
/// `cast_qmm_input_bfloat16_to_float16` declared `half_out` not at all.
///
/// `ShaderValue::buffer_mut` exists precisely so the direction rides on the
/// value, and Metal had taken its default. It does not any more:
/// [`ArgValue::BufferMut`] is what a `BufMut` or `F32sMut` argument produces,
/// at whatever slot the routine binds it to.
///
/// It stays conservative in the direction that matters. A handle bound to
/// nothing is a zero-length slice, which meets nothing; a scalar contributes
/// neither; a `pad` is a read, which costs a barrier and never loses one.
fn directed(args: &[BoundArg], values: &[ArgValue]) -> Touches {
    let mut touches = Touches::default();
    for (i, arg) in args.iter().enumerate() {
        if arg.slice.bytes == 0 {
            continue;
        }
        match values.get(i) {
            Some(ArgValue::BufferMut(_)) => touches.writes.push(arg.slice),
            // Everything else that reached a slot is a read. A scalar binds
            // no slice and a `pad` binds a real one, which is why this is
            // conservative in the direction that costs a barrier and never in
            // the one that loses a hazard.
            _ => touches.reads.push(arg.slice),
        }
    }
    touches
}

impl Encode for Planner<'_> {
    fn resolve(
        &self,
        ty: kernels::Ty,
        source: kernels::Source,
    ) -> Result<ArgValue, Refusal> {
        let (handles, facts) = self.answers.ok_or(Refusal::Unstated {
            what: "a fact, on a planner with no fire behind it",
        })?;
        crate::lowering::bind::one(ty, source, &mut handles.borrow_mut(), facts)
    }

    fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
        // A body with nothing to do should have refused already. A zero here
        // would become `dispatchThreads:` over an empty grid, which runs
        // nothing and reports success — so the buffer keeps whatever it held
        // and the model answers from stale bytes. The two are told apart on
        // purpose: `Refusal::Empty` is a body that noticed, and this is a body
        // that computed an extent and got zero.
        if fire.lanes.contains(&0) {
            return Err(Refusal::Grid {
                what: "the threads a routine asked for",
                at: 0,
            });
        }
        // Metal STATES its threadgroup rather than reflecting it — MSL
        // declares no workgroup size — so nothing else in the path would
        // catch a zero, and `threadsPerThreadgroup` of zero is undefined
        // rather than empty.
        if fire.group.contains(&0) {
            return Err(Refusal::Grid {
                what: "the threadgroup a routine asked for",
                at: 0,
            });
        }

        // `bound` is what the slots hold; `args` is what the routine SAID.
        // The direction comes off the latter -- see `ArgValue::BufferMut`.
        //
        // THE LIVE HANDLE LIST, NOT THE SNAPSHOT `Planner::new` WAS GIVEN.
        // Binding the column mints a handle per operand and `plan_routine`
        // copies the list out before it runs the body -- but a body that ASKS
        // mints more, into the `Handles` the cell holds, and their indices
        // point past the copy. The lookup below then answered
        // `Refusal::Absent { what: "a buffer" }` for every fact a body reached
        // through `ctx.ask`: the positions table, the KV pages, the mask.
        //
        // A probe planner has no cell and keeps the snapshot, which is right:
        // nothing can have minted anything.
        // AND THE STAGED BLOCK IS LIVE FOR THE SAME REASON. `ctx.params()`
        // goes through `resolve` like every other ask, so a body that forwards
        // its params run MINTS the block while it runs -- after this planner's
        // snapshot was taken. Reading `self.staged` here left `block` at
        // `None`, `lay_out` never recognised the pointer, and the encoder
        // bound a scalar run where a `constant RmsParams&` belongs. Both are
        // a handle at a slot, so nothing downstream could tell them apart.
        let live = self.answers.map(|(cell, _)| cell.borrow());
        let handles = live.as_deref().map_or(self.handles, |h| h.bound());
        let staged = live.as_deref().map_or(self.staged, super::hold::Handles::staged);
        let (bound, param_slots, params) = lay_out(args, self.types, handles, staged)?;
        drop(live);
        self.out.borrow_mut().push(Dispatch {
            symbol: fire.entrypoint,
            file: fire.file,
            stamp: fire.stamp,
            grid: fire.lanes,
            threadgroup: fire.group,
            touches: directed(&bound, args),
            args: bound,
            param_slots,
            params,
            layers: self.layers.clone(),
            op: self.op,
        });
        Ok(())
    }
}

/// Every routine this backend builds: its name, its entrypoint STEM, and the
/// arm that plumbs its operands.
///
/// The stem is the piece that lets a `kernel!` row be DELETED. A plan names
/// the fully instantiated entrypoint — `silu_mul_bfloat16`,
/// `affine_qmv_fast_bfloat16_gs_64_b_4` — and a routine is named after the
/// row without the axis points. Until this column existed the mapping between
/// the two lived only in the row's second column, so asking the table which
/// body to run was circular: retiring a family made its own routines
/// unreachable.
///
/// It is transcribed out of `kernel!`'s second column, and mostly equals the
/// name. Where it does not, the shader ships under a prefix the routine does
/// not carry: every `affine_*` quant entrypoint is spelled from a routine
/// called `qmm_t` or `qmv_fast`.
const LIVE: &[(&str, &str)] = &[
    ("gdn_core", "gdn_core"),
    ("gdn_prep", "gdn_prep"),
    ("qmv_routed", "affine_qmv_routed"),
    ("route_gather", "route_gather"),
    ("route_sort", "route_sort"),
    ("router_topk", "router_topk"),
    ("gate", "gate"),
    ("kv_append", "kv_append"),
    ("logit_softcap", "logit_softcap"),
    ("q_gate_split", "q_gate_split"),
    ("qmm_t", "affine_qmm_t"),
    ("qmm_t_bias", "affine_qmm_t_bias"),
    ("qmv_fast", "affine_qmv_fast"),
    ("qmv_tail", "affine_qmv_tail"),
    ("add_bias", "add_bias"),
    ("gated_rms", "gated_rms"),
    ("residual_add", "residual_add"),
    ("rms_residual", "rms_residual"),
    ("neox_decode", "neox_decode"),
    ("neox_mb", "neox_mb"),
    ("neox_prop_mb", "neox_prop_mb"),
    ("neox_freqs_mb", "neox_freqs_mb"),
    ("neox_strided", "neox_strided"),
    ("ple_combine", "ple_combine"),
    ("row_gather", "row_gather"),
    ("geglu_tanh", "geglu_tanh"),
    ("gptoss_swiglu", "gptoss_swiglu"),
    ("silu_mul", "silu_mul"),
    ("argmax_logits", "argmax_logits"),
    // THE SEVENTY THE FLATTENING DROPPED. `LIVE` was `&[(&[Routine],
    // &[(name, stem)])]` -- twenty-eight FAMILY rows holding ninety-nine
    // arms between them -- and collapsing it to one pair per row kept a
    // row's first arm and lost the rest. Nothing said so: `arm_for` still
    // answered for what remained, and a routine whose arm was gone came
    // back `Unclaimed` at dispatch, which reads like a family that never
    // crossed rather than one whose crossing was deleted.
    ("gdn_core_recurrent", "gdn_core_recurrent"),
    ("gdn_core_recurrent_prefill", "gdn_core_recurrent_prefill"),
    ("gdn_core_recurrent_slotted", "gdn_core_recurrent_slotted"),
    ("gdn_core_slotted", "gdn_core_slotted"),
    ("gdn_prep_prefill", "gdn_prep_prefill"),
    ("gdn_prep_slotted", "gdn_prep_slotted"),
    ("copy_logits_bf16", "copy_logits_bf16"),
    ("combine_sorted", "combine_sorted"),
    ("mxfp4_qmm_t_routed_bias", "mxfp4_qmm_t_routed_bias"),
    ("mxfp4_qmv_routed_bias", "mxfp4_qmv_routed_bias"),
    ("qmm_t_routed", "affine_qmm_t_routed"),
    ("qmm_t_routed_fp16", "affine_qmm_t_routed_fp16"),
    ("qmv_routed_bias", "affine_qmv_routed_bias"),
    ("router_topk_scaled", "router_topk_scaled"),
    ("shared_expert_combine", "shared_expert_combine"),
    ("shared_expert_combine_strided", "shared_expert_combine_strided"),
    ("kv_append_paged", "kv_append_paged"),
    ("sdpa_paged_decode", "sdpa_paged_decode"),
    ("sdpa_paged_decode_sink", "sdpa_paged_decode_sink"),
    ("sdpa_paged_mma", "sdpa_paged_mma"),
    ("sdpa_paged_mma_sink", "sdpa_paged_mma_sink"),
    ("sdpa_paged_tiled", "sdpa_paged_tiled"),
    ("sdpa_paged_tiled_sink", "sdpa_paged_tiled_sink"),
    ("sdpa_paged_tiled_strided", "sdpa_paged_tiled_strided"),
    ("sdpa_vector_decode", "sdpa_vector_decode"),
    ("sdpa_vector_decode_sink", "sdpa_vector_decode_sink"),
    ("sdpa_vector_decode_swa", "sdpa_vector_decode_swa"),
    ("split_qkv_bf16", "split_qkv_bf16"),
    ("cast_qmm_input_bfloat16_to_float16", "cast_qmm_input_bfloat16_to_float16"),
    ("cast_qmm_input_strided_bfloat16_to_float16", "cast_qmm_input_strided_bfloat16_to_float16"),
    ("encode_u4_bf16", "affine_encode_u4_bf16"),
    ("encode_u4_f32", "affine_encode_u4_f32"),
    ("mxfp4_dequant_bf16", "mxfp4_dequant_bf16"),
    ("qmm_splitk_reduce", "qmm_splitk_reduce"),
    ("qmm_splitk_reduce_f32", "qmm_splitk_reduce_f32"),
    ("qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4", "affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4"),
    ("qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2", "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2"),
    ("qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2", "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2"),
    ("qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1", "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1"),
    ("qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4", "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4"),
    ("qmm_t_bias_fp16_precast", "affine_qmm_t_bias_fp16_precast"),
    ("qmm_t_fp16_precast", "affine_qmm_t_fp16_precast"),
    ("qmm_t_residual", "affine_qmm_t_residual"),
    ("qmm_t_residual_fp16_precast", "affine_qmm_t_residual_fp16_precast"),
    ("qmm_t_splitk", "affine_qmm_t_splitk"),
    ("qmm_t_splitk_f32", "affine_qmm_t_splitk_f32"),
    ("qmm_t_splitk_fp16_precast", "affine_qmm_t_splitk_fp16_precast"),
    ("qmm_t_splitk_fp16_precast_f32", "affine_qmm_t_splitk_fp16_precast_f32"),
    ("qmm_t_strided", "affine_qmm_t_strided"),
    ("qmm_t_strided_fp16_precast", "affine_qmm_t_strided_fp16_precast"),
    ("qmm_t_strided_fp16_precast_residual", "affine_qmm_t_strided_fp16_precast_residual"),
    ("qmm_t_strided_residual", "affine_qmm_t_strided_residual"),
    ("qmv_fast_residual", "affine_qmv_fast_residual"),
    ("qmv_tail_bias", "affine_qmv_tail_bias"),
    ("qmv_wide_strided", "affine_qmv_wide_strided"),
    ("gated_rms_strided", "gated_rms_strided"),
    ("layer_scalar_mul", "layer_scalar_mul"),
    ("residual_add_strided", "residual_add_strided"),
    ("rms_residual_scaled", "rms_residual_scaled"),
    ("rms_single_row", "rms_single_row"),
    ("rms_strided_head_row", "rms_strided_head_row"),
    ("rms_strided_row", "rms_strided_row"),
    ("vnorm_single_row", "vnorm_single_row"),
    ("neox_prop_decode", "neox_prop_decode"),
    ("neox_freqs_decode", "neox_freqs_decode"),
    ("embed_gather_4bit", "embed_gather_4bit"),
    ("embed_gather_mb_4bit", "embed_gather_mb_4bit"),
    ("embed_gather_scaled_4bit", "embed_gather_scaled_4bit"),
    ("embed_gather_scaled_mb_4bit", "embed_gather_scaled_mb_4bit"),
    ("geglu_tanh_strided", "geglu_tanh_strided"),
];

/// The routine `name` names, out of the crate-wide slice.
///
/// [`LIVE`] USED TO CARRY THE SCOPE as well as the pairs: each row was
/// `(&[Routine], &[(name, stem)])`, and the first half existed only to bound
/// the `r.name == name` search to one family. Names are unique across the
/// crate -- `no_symbol_is_declared_twice` is what says so -- so the bound was
/// never doing anything, and with the families gone there is nothing to write
/// it with.
fn row(name: &str) -> Option<&'static Routine<Metal>> {
    kernels_metal::ROUTINES.iter().find(|r| r.name == name)
}

/// The routine this driver calls for `name`, if its family has crossed.
///
/// `name` is the ROW's name — the kernel, not the entrypoint — because that is
/// what a routine is named after and what `kernels::sig_in` already resolves a
/// lowered symbol to. The entrypoint is the body's to spell, which is the
/// direction the whole refactor runs in.
///
/// The pair is the routine and the ARM that feeds it: a signature says what a
/// routine takes and in what order, and never where any of it comes from.
/// See [`crate::lowering::hold`].
///
/// Answers for every routine this backend builds. A family reaches [`LIVE`]
/// only once its arms are written, and all ten have; a routine ADDED without
/// one answers `None` here, which
/// `every_routine_this_backend_builds_has_an_arm` refuses.
#[must_use]
pub fn arm_for(name: &str) -> Option<&'static Routine<Metal>> {
    LIVE.iter().find(|(n, _)| *n == name)?;
    row(name)
}

/// The routine a plan's SYMBOL names, by the longest stem it starts with.
///
/// This is the fork. A trace states `affine_qmv_fast_bfloat16_gs_64_b_4` and
/// this answers with `quant::qmv_fast` and its arm, without asking a row
/// anything — which is what lets the rows go.
///
/// # Why longest-match, and why the underscore
///
/// Stems nest. `qmm_t` is a prefix of every `qmm_t_splitk` symbol, and
/// first-match would send a split-K rectangle to the single-pass body: it
/// binds real buffers, dispatches, and leaves a partial sum where a total
/// belongs. Nothing downstream would notice.
///
/// And a stem may not end mid-word. Every instantiation appends its axis
/// points with a separator — `_bfloat16`, `_gs_64_b_4`, `_bm_32_bn_32` — so
/// what follows a stem is either nothing or an underscore. Without that rule
/// `rms_norm` would claim `rms_norm_gated`'s symbols by prefix alone.
#[must_use]
pub fn crossed(symbol: &str) -> Option<&'static Routine<Metal>> {
    let claims = |stem: &str| {
        symbol
            .strip_prefix(stem)
            .is_some_and(|rest| rest.is_empty() || rest.starts_with('_'))
    };
    // The dark stems compete on the same terms and answer nothing. A symbol
    // this backend does not cross must not fall through to a SHORTER stem
    // that happens to prefix it: `silu_mul_strided_bfloat16` clears the
    // underscore rule against `silu_mul`, and without this line it would
    // resolve to the routine for a kernel with a different argument table.
    //
    // `plan_routine`'s spelling check would catch it -- the body composes
    // `silu_mul_bfloat16` and the trace said otherwise -- but that is a
    // refusal at dispatch naming the wrong cause. This is the refusal at the
    // fork, naming the right one.
    let dark = DARK
        .iter()
        .filter(|(stem, _)| claims(stem))
        .map(|(stem, _)| stem.len())
        .max();
    LIVE.iter()
        .filter(|(_, stem)| claims(stem))
        .max_by_key(|(_, stem)| stem.len())
        .filter(|(_, stem)| dark.is_none_or(|d| stem.len() > d))
        .and_then(|(name, _)| row(name))
}

/// Every stem this backend crosses, with the routine it names.
///
/// [`crossed`] answers "which routine claims this SYMBOL", which is the
/// dispatch question. This is the census question: what does the registry
/// claim at all. It exists because the rows used to answer it -- a sweep over
/// `kernels_metal::KERNELS` reached every kernel this backend has, and every
/// family has retired its rows, so a sweep keyed on the table now reaches
/// nothing and says so by passing.
pub fn stems() -> impl Iterator<Item = (&'static str, &'static Routine<Metal>)> {
    LIVE.iter().filter_map(|(name, stem)| Some((*stem, row(name)?)))
}

/// Entrypoint stems this backend deliberately does NOT cross, and why.
///
/// A dark stem is not a gap in the port. It is a shipped shader that no text
/// names and no routine can express, listed here so that the stem lookup
/// refuses it outright instead of handing its symbols to a shorter stem that
/// prefixes them.
///
/// The list may only shrink by a kernel crossing or by its shader leaving the
/// tree. It is checked against the rows in `text_conformance.rs`, which is
/// what stops an entry outliving its argument.
pub const DARK: &[(&str, &str)] = &[(
    "silu_mul_strided",
    "no text names it and no statement produces the `row_pitch` it wants, so \
     a routine for it would be a call with no caller. This used to say the \
     hole was the obstacle -- the entrypoint declares `row_pitch` at \
     buffer(4) with buffer(3) left empty, and an argument list is positional \
     -- and that stopped being true when `pad` became the idiom for exactly \
     this: twenty-one routines now bind a valid address at an index their \
     shader does not declare, because a slot nothing declares needs an \
     address and not a meaning. What remains is the producer, not the shape.",
)];

#[cfg(test)]
mod tests {
    // The argument-count sweep REPORTS its number: a floor that only says
    // "above 500" cannot tell a reader the column moved from 900 to 572, and
    // that migration is exactly what the assertion above it is about.
    #![allow(clippy::print_stdout)]

    use super::*;

    /// A launch that stages no scalars and mints no packed block.
    const NO_BLOCK: Staged<'static> = Staged {
        block: None,
        words: &[],
    };
    use kernels::Grid;
    use kernels_metal::routine::{Bind, Ctx, In, Out, Tensor, Usize, bf16};

    fn handle(address: u64, width: u32) -> BoundArg {
        BoundArg {
            slice: Slice {
                address,
                bytes: u64::from(width) * 2,
            },
            width,
        }
    }

    /// A body written for this test, in the shape a crossed family's will
    /// have: buffers and scalars the trace supplies, extents the environment
    /// does.
    ///
    /// It is not a real kernel and does not need to be. What is under test is
    /// the SEAM — that a body's one statement becomes the dispatch the encoder
    /// takes — and a body that named a real entrypoint would test the same
    /// thing while inviting the reader to check it against a shader, which is
    /// what `tests/` is for once a family has crossed.
    fn scale_rows(
        ctx: &Ctx<'_>,
        x: In<Tensor<bf16>>,
        out: Out<Tensor<bf16>>,
        width: u32,
        stride: Usize,
        // A PLAIN NUMBER: this probe's `fn` is not a `#[routine]`, so `Env`
        // was never marking a column here -- it only kept the scalar from
        // reading as an operand, which nothing counts any more.
        rows: u32,
    ) -> Result<(), Refusal> {
        if rows == 0 {
            return Err(Refusal::Empty { what: "rows" });
        }
        ctx.fire(
            Fire::at("norm/scale.metal", "scale_rows_bfloat16")
                .apply(Grid::of([width, rows, 1], [32, 1, 1])),
            &[x.arg(), out.arg(), width.arg(), stride.arg()],
        )
    }

    /// The row `scale_rows` derives, for a planner to read its types off.
    static ROUTINE: std::sync::LazyLock<Routine<Metal>> =
        std::sync::LazyLock::new(|| kernels::routine!(Metal, "scale_rows", scale_rows, namespace = ""));

    /// The planner turns a body's statement into the dispatch the encoder
    /// already knows how to run.
    ///
    /// Everything the table path computes from a row — the operand order, the
    /// argument-table slots, the scalar run's offsets and widths — this
    /// computes from the body's own argument list, and the point is that it
    /// lands in the SAME `Dispatch`. Nothing downstream can tell which plane
    /// produced one, which is what makes a family's crossing a change to one
    /// file rather than to the driver.
    #[test]
    fn a_body_s_statement_becomes_the_dispatch_the_encoder_takes() {
        let handles = [handle(0x1000, 64), handle(0x2000, 64)];
        let planner = Planner::new(&ROUTINE, &handles, NO_BLOCK, 0..1, 7);

        scale_rows(&planner, In::new(Tensor::new(0)), Out::new(Tensor::new(1)), 64, Usize(128), 8)
            .expect("eight rows is a launch");

        let plan = planner.finish();
        assert_eq!(plan.len(), 1, "one statement is one dispatch");
        let d = &plan[0];
        assert_eq!(d.symbol, "scale_rows_bfloat16");
        assert_eq!(d.file, "norm/scale.metal");
        assert_eq!(
            d.grid,
            [64, 8, 1],
            "THREADS, not threadgroups -- `dispatchThreads:` takes the first, \
             and a body that wrote the other number would launch 64 threads \
             where it meant 64 groups of 32"
        );
        assert_eq!(d.threadgroup, [32, 1, 1]);
        assert_eq!(
            d.args,
            vec![handles[0], handles[1], NOTHING, NOTHING],
            "the two buffers at the slots the BODY put them in, and a zero \
             slot for each scalar -- a scalar's value rides `params`"
        );
        assert_eq!(
            d.params,
            vec![64, 128, 0],
            "the width, then the stride's two words low first"
        );
        assert_eq!(
            d.param_slots,
            vec![
                ParamSlot {
                    slot: 2,
                    at: 0,
                    bytes: 4,
                    packed: false,
                    value: Some(0)
                },
                ParamSlot {
                    slot: 3,
                    at: 8,
                    bytes: 8,
                    packed: false,
                    value: Some(1)
                },
            ],
            "the eight-byte stride starts on an eight-byte boundary and not at \
             four, where the four-byte width left off. A kernel reading \
             `const constant size_t&` at 4 would take two halves of two \
             different scalars."
        );
        assert_eq!(d.layers, 0..1, "the launch's, not the body's");
        assert_eq!(d.op, 7, "likewise");
    }

    /// A body whose entrypoint numbers its buffers with HOLES, which is the
    /// shape twenty-three shipped routines have.
    ///
    /// `pad` is taken once and bound at every hole, so the dispatch list and
    /// the parameter list are different lists: `out` is the signature's
    /// second entry and the dispatch's fourth. See [`directed`].
    fn holed(
        ctx: &Ctx<'_>,
        pad: In<Tensor<bf16>>,
        out: Out<Tensor<bf16>>,
        x: In<Tensor<bf16>>,
        width: u32,
    ) -> Result<(), Refusal> {
        ctx.fire(
            Fire::at("norm/scale.metal", "holed_bfloat16")
                .apply(Grid::of([width, 1, 1], [32, 1, 1])),
            &[pad.arg(), pad.arg(), x.arg(), out.arg(), width.arg()],
        )
    }

    static HOLED: std::sync::LazyLock<Routine<Metal>> =
        std::sync::LazyLock::new(|| kernels::routine!(Metal, "holed", holed, namespace = ""));

    /// A padded routine still says which buffer it WRITES.
    ///
    /// The direction used to come from the signature indexed at the value's
    /// position, which for a padded body is a different argument entirely:
    /// this one's `out` landed on `x`'s type and was declared a read, while
    /// the `pad` in slot 1 was declared a write. An encoder reading that runs
    /// this dispatch alongside the one that consumes `out`, and the answer
    /// stops being the same twice -- which is exactly what qwen3.6 did until
    /// `ArgValue::BufferMut` existed.
    #[test]
    fn a_body_with_pad_slots_still_declares_the_buffer_it_writes() {
        let handles = [handle(0x1000, 64), handle(0x2000, 64), handle(0x3000, 64)];
        let planner = Planner::new(&HOLED, &handles, NO_BLOCK, 0..1, 0);
        holed(&planner, In::new(Tensor::new(0)), Out::new(Tensor::new(1)), In::new(Tensor::new(2)), 64).expect("a launch");
        let plan = planner.finish();
        let d = &plan[0];
        assert_eq!(
            d.touches.writes,
            vec![handles[1].slice],
            "the one mutable argument, at the slot the BODY bound it to"
        );
        assert!(
            !d.touches.writes.contains(&handles[0].slice),
            "`pad` is read-only however many holes it fills"
        );
        assert!(
            d.touches.reads.contains(&handles[2].slice)
                && d.touches.reads.contains(&handles[0].slice),
            "everything else that reached a slot is a read"
        );
    }

    /// A refusal leaves no dispatch behind.
    #[test]
    fn a_body_that_refuses_does_not_reach_the_plan() {
        let handles = [handle(0x1000, 64), handle(0x2000, 64)];
        let planner = Planner::new(&ROUTINE, &handles, NO_BLOCK, 0..1, 7);
        assert_eq!(
            scale_rows(&planner, In::new(Tensor::new(0)), Out::new(Tensor::new(1)), 64, Usize(128), 0),
            Err(Refusal::Empty { what: "rows" })
        );
        assert!(planner.finish().is_empty());
    }

    /// A body reaching past its statement's operands is refused, not zeroed.
    #[test]
    fn a_handle_the_launch_did_not_bind_is_refused_and_not_zeroed() {
        let handles = [handle(0x1000, 64)];
        let planner = Planner::new(&ROUTINE, &handles, NO_BLOCK, 0..1, 7);
        assert_eq!(
            scale_rows(&planner, In::new(Tensor::new(0)), Out::new(Tensor::new(1)), 64, Usize(128), 8),
            Err(Refusal::Absent { what: "a buffer" })
        );
        assert!(
            planner.finish().is_empty(),
            "and nothing partial is left behind: a dispatch with one operand \
             bound and one missing would run"
        );
    }

    /// A grid of no threads is refused, because the hardware would not report
    /// it.
    ///
    /// Distinct from [`Refusal::Empty`] on purpose. `Empty` is a body that
    /// noticed it had nothing to do; this is a body that computed an extent
    /// and got zero, which is an arithmetic bug in the body and not a fact
    /// about the batch.
    #[test]
    fn a_dispatch_of_no_threads_is_refused_rather_than_run() {
        let handles = [handle(0x1000, 64), handle(0x2000, 64)];
        let planner = Planner::new(&ROUTINE, &handles, NO_BLOCK, 0..1, 7);
        assert_eq!(
            scale_rows(&planner, In::new(Tensor::new(0)), Out::new(Tensor::new(1)), 0, Usize(128), 8),
            Err(Refusal::Grid {
                what: "the threads a routine asked for",
                at: 0
            })
        );
        assert!(planner.finish().is_empty());
    }

    /// The derived row is the signature, and it is what the planner reads
    /// types off.
    #[test]
    fn the_derived_row_states_which_arguments_the_environment_supplies() {
        assert_eq!(ROUTINE.name, "scale_rows");
        assert_eq!(
            ROUTINE.args,
            &[
                // `Bf16s`, not `Buf`: the ELEMENT states the type now, and
                // `In<Tensor<bf16>>` records which element it carries where
                // the untyped handle recorded only that it was a buffer.
                Ty::Bf16s,
                Ty::Bf16sMut,
                Ty::U32,
                Ty::Usize,
                Ty::U32,
            ]
        );
    }

    /// Every arm in the registry names a routine that exists.
    ///
    /// `arm_for` finds the arm first and the routine second, and answers
    /// `None` if either is missing -- which for a MISSPELLED arm name is
    /// indistinguishable from a family that has not crossed. The symbol keeps
    /// taking the table path, every test stays green, and the wiring the
    /// commit claims to have done is not there.
    ///
    /// That exact failure has already happened once on this refactor, in
    /// `plan`: the fork asked for a row's bare name while the trace states
    /// the instantiated one, so two families were "wired" to nothing. This is
    /// the same shape one level down.
    #[test]
    fn every_arm_names_a_routine_that_exists() {
        for (name, _) in LIVE {
            assert!(
                row(name).is_some(),
                "{name}: an arm for a routine no module declares -- `arm_for` \
                 answers `None`, so nothing could dispatch the symbol at all"
            );
            assert!(arm_for(name).is_some(), "{name}");
        }
    }

    /// Every routine this backend builds has an arm.
    ///
    /// Stated as a number rather than a spot check so that a routine ADDED to
    /// `kernels-metal` without an arm fails here instead of silently taking
    /// a table path that Stage 5 is about to delete.
    #[test]
    fn every_routine_this_backend_builds_has_an_arm() {
        let live: usize = LIVE.len();
        let built: usize = kernels_metal::ROUTINES.len();
        assert_eq!(live, built, "an arm per routine, all ten families");
        assert_eq!(live, 99);
        for name in ["rms_single_row", "sdpa_paged_decode", "qmm_t", "gdn_core"] {
            assert!(arm_for(name).is_some(), "{name}");
        }
    }

    /// **Every arm hands its routine the list its routine declares** — the
    /// right number of values, each of the right kind.
    ///
    /// The two halves of a crossing are written in different crates. A
    /// routine states its signature in `kernels-metal`; the arm that fills it
    /// is in `arm::`; [`LIVE`] is the only place the two names meet. Nothing
    /// compared them, because an arm returns `Vec<ArgValue>` and a `Vec` has
    /// no arity: a list one short is [`Refusal::Arity`] and a value of the
    /// wrong shape is [`Refusal::Kind`], both raised inside `KernelFn::invoke`
    /// at DISPATCH — a device away and a whole fire late.
    ///
    /// The module doc above says an arm is a call and "an argument list one
    /// short does not compile". That is true of the routine's own `pub fn`
    /// and false of the arm, which builds a `Vec` and hands it over
    /// positionally. This test is the missing half of that sentence.
    ///
    /// # It is not hypothetical
    ///
    /// Twelve quant routines gained a `pad` argument in one commit — their
    /// shared argument table declares a slot at 7 and nothing was binding it —
    /// and each of their arms had to grow a value in the same breath, while
    /// `split_k` lost one. Fifteen argument lists were reconciled by COUNTING,
    /// by hand, because a `vec![]` of the wrong length is a perfectly good
    /// `vec![]` and the compiler had nothing to say about any of them.
    ///
    /// # Why a fixture, and why a generous one
    ///
    /// An arm's length is not a static fact. It is what the arm's body
    /// builds, out of handles it asks for one at a time, so the only way to
    /// ask how long it is, is to RUN it. The statement below is therefore
    /// wider than any real one and its resolver answers every question: every
    /// operand index is in range, every scalar is stated, every table and pool
    /// and slab resolves. A refusal here is the arm's own and never the
    /// fixture running out.
    ///
    /// That generosity is also why this cannot replace the conformance tests.
    /// It compares the arm against the ROUTINE; `tests/text_conformance.rs`
    /// compares the routine's dispatch list against the SHADER. Neither sees
    /// the other's seam, and an argument can be the right kind in the right
    /// slot of the wrong list.
    #[test]
    fn every_routine_states_the_argument_list_it_declares() {
        use crate::lowering::dispatch::Geometry;
        use crate::lowering::hold::{Facts, Handles};
        use crate::lowering::executor::{FireTable, Resolver};
        use kernels_metal::routine::ArgValue;

        /// One region, big enough that nothing an arm asks for falls off it.
        const SOMEWHERE: Slice = Slice {
            address: 0x1_0000,
            bytes: 1 << 20,
        };

        /// A resolver that answers everything, including the three questions
        /// whose real answers are optional — the KV pool, the GDN slabs, the
        /// fire tables. A `None` from any of them is a refusal about the
        /// DRIVER's state, and this test is not about the driver's state.
        struct Everything;

        impl Resolver for Everything {
            fn weight(&mut self, _: &str) -> Option<Slice> {
                Some(SOMEWHERE)
            }
            fn named(&mut self, _: model_ir::trace::ValueId) -> Option<Slice> {
                Some(SOMEWHERE)
            }
            fn kv(&mut self, _: u16, _: bool) -> Option<Slice> {
                Some(SOMEWHERE)
            }
            fn slab(&mut self, _: u16, _: &'static str) -> Option<Slice> {
                Some(SOMEWHERE)
            }
            fn fire(&mut self, _: FireTable) -> Option<Slice> {
                Some(SOMEWHERE)
            }
            fn pool(&mut self, _: FireTable) -> Option<u32> {
                Some(16)
            }
        }

        // What `Arg::unpack` will do to each value at dispatch, asked ahead of
        // time. An unlisted kind panics rather than defaulting: a `Ty` this
        // backend starts using should be classified deliberately, not swept
        // into whichever arm a wildcard sat in.
        let fits = |ty: Ty, v: ArgValue| match ty {
            Ty::BufMut
            | Ty::Buf
            | Ty::Bf16s
            | Ty::Bf16sMut
            | Ty::I32s
            | Ty::I32sMut
            | Ty::U32s
            | Ty::U32sMut
            | Ty::U8s
            | Ty::U8sMut
            | Ty::F32s
            | Ty::F32sMut
            // THE HALF PAIR, WHICH THE MARKS MADE REACHABLE. `Tensor<f16>` is
            // an element like any other now, so a routine can declare one --
            // the precast matmuls do -- and it binds a handle exactly as the
            // other twelve do.
            | Ty::F16s
            | Ty::F16sMut => matches!(
                v,
                // `Shaped` IS A BUFFER, and `kind()` already says so -- it is
                // a handle that carries the rectangle the statement gave the
                // operand, which the binder produces wherever a mark's region
                // is known. Leaving it out here read as "the binder bound a
                // buffer where a buffer was wanted", which is not a sentence
                // anyone could act on.
                ArgValue::Buffer(_) | ArgValue::BufferMut(_) | ArgValue::Shaped { .. }
            ),
            Ty::I32 => matches!(v, ArgValue::I32(_)),
            // `InPacked` carries a `u32`'s value and is not a `u32`: it is a
            // FIELD of the preceding params struct, which is why it has a kind
            // of its own and the same binding.
            Ty::U32 | Ty::InPacked => matches!(v, ArgValue::U32(_)),
            Ty::F32 => matches!(v, ArgValue::F32(_)),
            Ty::Usize => matches!(v, ArgValue::Usize(_)),
            other => panic!("no metal routine took `{other:?}` when this was written"),
        };

        let args: Vec<BoundArg> = (0..24)
            .map(|i: u64| handle(0x1_0000 + i * 0x1000, 64))
            .collect();
        let ins: Vec<usize> = (0..8).collect();
        let outs: Vec<usize> = (8..16).collect();
        let weights: Vec<usize> = (16..24).collect();
        // Non-zero, because arms divide by these: a group size of zero is
        // `Refusal::Empty` and a zero axis makes a head count a division by it.
        let params: Vec<Option<u32>> = (1..=16).map(Some).collect();
        let facts = Facts {
            geometry: Geometry {
                q_heads: 8,
                kv_heads: 2,
                head_dim: 64,
                rotary_dims: 64,
                n_experts: 8,
                experts_per_token: 2,
                ..Geometry::default()
            },
            rows: 4,
            width: 64,
            in_width: 64,
            tile: Some((32, 32)),
            point: (64, 4),
            layer: 0,
            requests: 2,
        };

        let mut wrong: Vec<String> = Vec::new();
        for (name, stem) in LIVE {
            {
                let Some(routine) = row(name) else {
                    wrong.push(format!("  {stem} -> `{name}`: no routine of that name"));
                    continue;
                };
                for (at, source) in routine.sources.iter().enumerate() {
                    if source.is_none() {
                        wrong.push(format!(
                            "  {stem} -> `{name}`: argument {at} has no source, \
                             so no signature says where it comes from"
                        ));
                    }
                }
                let mut resolver = Everything;
                let mut handles =
                    Handles::new(&args, &ins, &outs, &weights, &params, &mut resolver);
                let built = match crate::lowering::bind::bind(
                    routine.args,
                    routine.sources,
                    &mut handles,
                    facts,
                ) {
                    Ok(built) => built,
                    Err(why) => {
                        wrong.push(format!("  {stem} -> `{name}`: refused: {why}"));
                        continue;
                    }
                };
                if built.len() != routine.args.len() {
                    wrong.push(format!(
                        "  {stem} -> `{name}`: the binder hands {} value(s), the \
                         routine takes {}",
                        built.len(),
                        routine.args.len()
                    ));
                    continue;
                }
                for (at, (value, ty)) in built.iter().zip(routine.args).enumerate() {
                    if !fits(*ty, *value) {
                        wrong.push(format!(
                            "  {stem} -> `{name}`: argument {at} is {ty:?} and \
                             the binder bound {}",
                            value.kind()
                        ));
                    }
                }
            }
        }
        assert!(
            wrong.is_empty(),
            "a routine and its own signature disagree about the call between \
             them. This ran against ninety-one hand-written arms while STAGES \
             2 through 6 moved what they knew into the signatures; it runs \
             against `bind` now, so a disagreement here is a routine that \
             cannot dispatch at all:\n{}",
            wrong.join("\n")
        );
    }

    /// The longest stem wins, and a stem may not end mid-word.
    ///
    /// Both halves are load-bearing over the shipped names. `affine_qmm_t`
    /// prefixes every `affine_qmm_t_splitk` symbol, and first-match would run
    /// the single-pass body over a split-K rectangle -- which binds, launches
    /// and leaves a partial sum where a total belongs. `rms_norm` prefixes
    /// `rms_norm_gated` without an underscore boundary to stop it.
    #[test]
    fn the_longest_stem_wins_and_a_stem_may_not_end_mid_word() {
        let by = |symbol: &str| crossed(symbol).map(|r| r.name);
        assert_eq!(
            by("affine_qmm_t_splitk_bfloat16_gs_64_b_4"),
            Some("qmm_t_splitk"),
            "the longer stem takes its own symbol"
        );
        assert_eq!(
            by("affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32"),
            Some("qmm_t")
        );
        assert_eq!(by("silu_mul_bfloat16"), Some("silu_mul"));
        assert_eq!(
            by("silu_mul_strided_bfloat16"),
            None,
            "a DARK stem answers nothing rather than falling through to the \
             shorter stem that prefixes it"
        );
        assert_eq!(by("qmm_t"), None, "the quant symbols carry the prefix");
        assert_eq!(
            by("affine_qmm_t_nonsense_axis"),
            Some("qmm_t"),
            "an unknown axis point is still this routine's symbol"
        );
    }

    /// Every stem resolves to its own routine, and no two claim each other's
    /// symbol.
    ///
    /// The registry is hand-kept and the stems are transcribed, so the
    /// falsifier is a typo: a stem that matches nothing dispatches nothing,
    /// and a stem that is another's whole symbol makes the pair ambiguous in
    /// a way `max_by_key` resolves silently.
    #[test]
    fn every_stem_finds_its_own_routine() {
        {
            for (name, stem) in LIVE {
                let got = crossed(stem)
                    .unwrap_or_else(|| panic!("`{stem}` resolves nothing"))
                    .name;
                assert_eq!(got, *name, "`{stem}` resolves the wrong routine");
            }
        }
    }



    /// Every source in the column is one the binder has a case for --
    /// including the halves a full statement never reaches.
    ///
    /// # Why the other gate cannot ask this
    ///
    /// `every_routine_states_the_argument_list_it_declares` binds each
    /// routine against a statement carrying every scalar, so every chain
    /// takes its FIRST half and the second is never evaluated. That is the
    /// shape of statement every deployment in the suite produces, which is
    /// why a missing case in the second half is invisible until a model that
    /// states nothing arrives.
    ///
    /// Binding the whole routine with no scalars does not work either: a
    /// required `Param<N, T>` refuses first, `bind` stops at it, and every
    /// argument after it -- including the chain -- goes unevaluated. Twenty-
    /// one routines refuse that way, which is enough to hide any number of
    /// unanswerable sources behind them.
    ///
    /// So each argument is bound ALONE, on a one-element slice of the
    /// routine's own columns. Nothing masks anything, every chain takes its
    /// fallback, and the real `bind` is what runs -- so this asks about the
    /// binder rather than about a second list that describes it.
    ///
    /// # What is a failure and what is not
    ///
    /// `Refusal::Absent` is the honest answer to "the statement does not
    /// carry this scalar", and with no scalars most routines produce
    /// several. `Refusal::Unstated` is the binder saying it has NO CASE, and
    /// that is the failure -- a source in the column no code path answers.
    ///
    /// It found one on landing. `keys::RotaryWidth` is the fallback of seven
    /// `ParamOr<3, ..>` sites, `arm.rs` answered it from `Facts::rotary_dims`,
    /// and `bind::named` had no case for it.
    #[test]
    fn every_source_in_the_column_is_one_the_binder_answers() {
        use crate::lowering::executor::{FireTable, Resolver};
        use crate::lowering::dispatch::Geometry;
        use crate::lowering::hold::{Facts, Handles};

        const SOMEWHERE: Slice = Slice {
            address: 0x1_0000,
            bytes: 1 << 20,
        };

        struct Everything;

        impl Resolver for Everything {
            fn weight(&mut self, _: &str) -> Option<Slice> {
                Some(SOMEWHERE)
            }
            fn named(&mut self, _: model_ir::trace::ValueId) -> Option<Slice> {
                Some(SOMEWHERE)
            }
            fn kv(&mut self, _: u16, _: bool) -> Option<Slice> {
                Some(SOMEWHERE)
            }
            fn slab(&mut self, _: u16, _: &'static str) -> Option<Slice> {
                Some(SOMEWHERE)
            }
            fn fire(&mut self, _: FireTable) -> Option<Slice> {
                Some(SOMEWHERE)
            }
            fn pool(&mut self, _: FireTable) -> Option<u32> {
                Some(16)
            }
        }

        let args: Vec<BoundArg> = (0..24)
            .map(|i: u64| handle(0x1_0000 + i * 0x1000, 64))
            .collect();
        let ins: Vec<usize> = (0..8).collect();
        let outs: Vec<usize> = (8..16).collect();
        let weights: Vec<usize> = (16..24).collect();
        // THE POINT OF THE TEST: no scalars, so every chain falls through to
        // the half a shipped statement never reaches.
        let params: Vec<Option<u32>> = Vec::new();
        let facts = Facts {
            geometry: Geometry {
                q_heads: 8,
                kv_heads: 2,
                head_dim: 64,
                v_heads: 2,
                v_dim: 64,
                rotary_dims: 32,
                n_experts: 8,
                experts_per_token: 2,
                ..Geometry::default()
            },
            rows: 4,
            width: 64,
            in_width: 64,
            // Present, so that `TileM`/`TileN` reach their number rather than
            // the `Unstated` that a device stating no tile deserves.
            tile: Some((32, 32)),
            point: (64, 4),
            layer: 0,
            requests: 2,
        };

        let mut unanswered: Vec<String> = Vec::new();
        let mut asked = 0usize;
        {
            for (name, _stem) in LIVE {
                let Some(routine) = row(name) else {
                    continue;
                };
                for at in 0..routine.args.len() {
                    let mut resolver = Everything;
                    let mut handles =
                        Handles::new(&args, &ins, &outs, &weights, &params, &mut resolver);
                    asked += 1;
                    let one = crate::lowering::bind::bind(
                        &routine.args[at..=at],
                        &routine.sources[at..=at],
                        &mut handles,
                        facts,
                    );
                    if let Err(Refusal::Unstated { what }) = one {
                        unanswered.push(format!("  {name} argument {at}: {what}"));
                    }
                }
            }
        }

        assert!(
            unanswered.is_empty(),
            "a source in the column has no case in `bind`, so the argument \
             refuses whenever that source is reached -- which for the second \
             half of a chain is only on the deployments that state \
             nothing:\n{}",
            unanswered.join("\n")
        );

        // A FLOOR, because a gate that asks nothing looks like a gate that
        // passes. This one would ask nothing if `LIVE` emptied or if the
        // `find` stopped matching, and it has happened twice on this ladder.
        let total: usize = LIVE
            .iter()
            .filter_map(|(name, _)| row(name))
            .map(|routine| routine.args.len())
            .sum();
        assert_eq!(asked, total, "every argument of every routine is asked");
        // THE FLOOR MOVED BECAUSE THE COLUMN DID. It read `> 900` against a
        // column of a thousand-odd, and the marks migration took it to 572:
        // every fact a body now reaches for with `ctx.ask` has left the
        // parameter run, and `args` enumerates the run. The floor is still a
        // floor -- it catches `LIVE` emptying or the `find` going quiet, which
        // is what it is for -- and 500 is the same distance below 572 that 900
        // was below the old count.
        assert!(
            asked > 500,
            "only {asked} arguments asked; the column is 570-odd"
        );
        println!("arguments asked: {asked}");
    }

}
