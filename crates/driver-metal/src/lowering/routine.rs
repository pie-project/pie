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
use kernels::routine::{Provenance, Refusal, Routine};
use kernels_metal::routine::{ArgValue, Encode, Fire, Metal};

use crate::lowering::arm::{self, Arm, Staged};

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
    types: &'r [(Ty, Provenance)],
    /// The buffers this launch bound, in the trace's order. A body's handle is
    /// an index into this.
    handles: &'r [BoundArg],
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
        }
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
fn lay_out(
    values: &[ArgValue],
    types: &[(Ty, Provenance)],
    handles: &[BoundArg],
    staged: Staged<'_>,
) -> Result<(Vec<BoundArg>, Vec<ParamSlot>, Vec<u32>), Refusal> {
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
        if types.get(slot).map(|(ty, _)| *ty) == Some(Ty::InPacked) {
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
                ArgValue::Buffer(handle) => handle,
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
            ArgValue::Buffer(handle) => {
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

/// What a dispatch reads and what it may write, off the SIGNATURE.
///
/// The direction was always there and this used to throw it away: a routine
/// spells a written buffer `BufMut` or `F32sMut` and a read one `Buf`,
/// `I32s`, `U32s`, `U8s`, `F32s`. So a launch's writes are exactly the
/// handles at the mutable positions, and every other bound handle is a read.
///
/// The conservative answer this replaces -- every operand as both -- was
/// honest and expensive. `Touches` decides whether an encoder may run two
/// launches at once, so calling a read a write inserts a barrier between two
/// dispatches that never meet, and on a decode where the routine path now
/// carries every launch that is a barrier per launch rather than per hazard.
///
/// It stays conservative in the direction that matters. A handle bound to
/// nothing is a zero-length slice, which meets nothing; a scalar contributes
/// neither. What it will not do is silently under-report a write, because a
/// `BufMut` in the signature is the same fact the shader's `device T*`
/// states and the cross-backend gate compares.
fn directed(args: &[BoundArg], types: &[(Ty, Provenance)]) -> Touches {
    let mut touches = Touches::default();
    for (i, arg) in args.iter().enumerate() {
        if arg.slice.bytes == 0 {
            continue;
        }
        match types.get(i).map(|(ty, _)| *ty) {
            Some(Ty::BufMut | Ty::F32sMut) => touches.writes.push(arg.slice),
            // A slot the signature does not reach is the staged parameter
            // block, which the encoder writes and no other launch reads.
            Some(_) => touches.reads.push(arg.slice),
            None => {}
        }
    }
    touches
}

impl Encode for Planner<'_> {
    fn dispatch(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
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

        let (args, param_slots, params) = lay_out(args, self.types, self.handles, self.staged)?;
        self.out.borrow_mut().push(Dispatch {
            symbol: fire.entrypoint,
            file: fire.file,
            grid: fire.lanes,
            threadgroup: fire.group,
            touches: directed(&args, self.types),
            args,
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
const LIVE: &[(&[Routine<Metal>], &[(&str, &str, Arm)])] = &[
    (
        kernels_metal::ssm::ROUTINES,
        &[
            ("gdn_core", "gdn_core", arm::gdn_core as Arm),
            (
                "gdn_core_recurrent",
                "gdn_core_recurrent",
                arm::gdn_core_recurrent as Arm,
            ),
            (
                "gdn_core_recurrent_prefill",
                "gdn_core_recurrent_prefill",
                arm::gdn_core_recurrent_prefill as Arm,
            ),
            (
                "gdn_core_recurrent_slotted",
                "gdn_core_recurrent_slotted",
                arm::gdn_core_recurrent_slotted as Arm,
            ),
            (
                "gdn_core_slotted",
                "gdn_core_slotted",
                arm::gdn_core_slotted as Arm,
            ),
            ("gdn_prep", "gdn_prep", arm::gdn_prep as Arm),
            (
                "gdn_prep_prefill",
                "gdn_prep_prefill",
                arm::gdn_prep_prefill as Arm,
            ),
            (
                "gdn_prep_slotted",
                "gdn_prep_slotted",
                arm::gdn_prep_slotted as Arm,
            ),
        ],
    ),
    (
        kernels_metal::ptir::ROUTINES,
        &[(
            "copy_logits_bf16",
            "copy_logits_bf16",
            arm::copy_logits_bf16 as Arm,
        )],
    ),
    (
        kernels_metal::moe::ROUTINES,
        &[
            (
                "combine_sorted",
                "combine_sorted",
                arm::combine_sorted as Arm,
            ),
            (
                "mxfp4_qmm_t_routed_bias",
                "mxfp4_qmm_t_routed_bias",
                arm::mxfp4_qmm_t_routed_bias as Arm,
            ),
            (
                "mxfp4_qmv_routed_bias",
                "mxfp4_qmv_routed_bias",
                arm::mxfp4_qmv_routed_bias as Arm,
            ),
            (
                "qmm_t_routed",
                "affine_qmm_t_routed",
                arm::qmm_t_routed as Arm,
            ),
            (
                "qmm_t_routed_fp16",
                "affine_qmm_t_routed_fp16",
                arm::qmm_t_routed_fp16 as Arm,
            ),
            ("qmv_routed", "affine_qmv_routed", arm::qmv_routed as Arm),
            (
                "qmv_routed_bias",
                "affine_qmv_routed_bias",
                arm::qmv_routed_bias as Arm,
            ),
            ("route_gather", "route_gather", arm::route_gather as Arm),
            ("route_sort", "route_sort", arm::route_sort as Arm),
            ("router_topk", "router_topk", arm::router_topk as Arm),
            (
                "router_topk_scaled",
                "router_topk_scaled",
                arm::router_topk_scaled as Arm,
            ),
            (
                "shared_expert_combine",
                "shared_expert_combine",
                arm::shared_expert_combine as Arm,
            ),
            (
                "shared_expert_combine_strided",
                "shared_expert_combine_strided",
                arm::shared_expert_combine_strided as Arm,
            ),
        ],
    ),
    (
        kernels_metal::attn::ROUTINES,
        &[
            ("gate", "gate", arm::gate as Arm),
            ("kv_append", "kv_append", arm::kv_append as Arm),
            (
                "kv_append_paged",
                "kv_append_paged",
                arm::kv_append_paged as Arm,
            ),
            ("logit_softcap", "logit_softcap", arm::logit_softcap as Arm),
            ("q_gate_split", "q_gate_split", arm::q_gate_split as Arm),
            (
                "sdpa_paged_decode",
                "sdpa_paged_decode",
                arm::sdpa_paged_decode as Arm,
            ),
            (
                "sdpa_paged_decode_sink",
                "sdpa_paged_decode_sink",
                arm::sdpa_paged_decode_sink as Arm,
            ),
            (
                "sdpa_paged_mma",
                "sdpa_paged_mma",
                arm::sdpa_paged_mma as Arm,
            ),
            (
                "sdpa_paged_mma_sink",
                "sdpa_paged_mma_sink",
                arm::sdpa_paged_mma_sink as Arm,
            ),
            (
                "sdpa_paged_tiled",
                "sdpa_paged_tiled",
                arm::sdpa_paged_tiled as Arm,
            ),
            (
                "sdpa_paged_tiled_sink",
                "sdpa_paged_tiled_sink",
                arm::sdpa_paged_tiled_sink as Arm,
            ),
            (
                "sdpa_paged_tiled_strided",
                "sdpa_paged_tiled_strided",
                arm::sdpa_paged_tiled_strided as Arm,
            ),
            (
                "sdpa_vector_decode",
                "sdpa_vector_decode",
                arm::sdpa_vector_decode as Arm,
            ),
            (
                "sdpa_vector_decode_sink",
                "sdpa_vector_decode_sink",
                arm::sdpa_vector_decode_sink as Arm,
            ),
            (
                "sdpa_vector_decode_swa",
                "sdpa_vector_decode_swa",
                arm::sdpa_vector_decode_swa as Arm,
            ),
            (
                "split_qkv_bf16",
                "split_qkv_bf16",
                arm::split_qkv_bf16 as Arm,
            ),
        ],
    ),
    (
        kernels_metal::quant::ROUTINES,
        &[
            (
                "cast_qmm_input_bfloat16_to_float16",
                "cast_qmm_input_bfloat16_to_float16",
                arm::cast_qmm_input_bfloat16_to_float16 as Arm,
            ),
            (
                "cast_qmm_input_strided_bfloat16_to_float16",
                "cast_qmm_input_strided_bfloat16_to_float16",
                arm::cast_qmm_input_strided_bfloat16_to_float16 as Arm,
            ),
            (
                "encode_u4_bf16",
                "affine_encode_u4_bf16",
                arm::encode_u4_bf16 as Arm,
            ),
            (
                "encode_u4_f32",
                "affine_encode_u4_f32",
                arm::encode_u4_f32 as Arm,
            ),
            (
                "mxfp4_dequant_bf16",
                "mxfp4_dequant_bf16",
                arm::mxfp4_dequant_bf16 as Arm,
            ),
            (
                "qmm_splitk_reduce",
                "qmm_splitk_reduce",
                arm::qmm_splitk_reduce as Arm,
            ),
            (
                "qmm_splitk_reduce_f32",
                "qmm_splitk_reduce_f32",
                arm::qmm_splitk_reduce_f32 as Arm,
            ),
            ("qmm_t", "affine_qmm_t", arm::qmm_t as Arm),
            (
                "qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4",
                "affine_qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4",
                arm::qmm_t_bfloat16_gs_64_b_4_bm_128_bn_32_wm_4 as Arm,
            ),
            (
                "qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2",
                "affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2",
                arm::qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32_wm_1_wn_2 as Arm,
            ),
            (
                "qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2",
                "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2",
                arm::qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_1_wn_2 as Arm,
            ),
            (
                "qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1",
                "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1",
                arm::qmm_t_bfloat16_gs_64_b_4_bm_64_bn_32_wm_2_wn_1 as Arm,
            ),
            (
                "qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4",
                "affine_qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4",
                arm::qmm_t_bfloat16_gs_64_b_4_bm_64_bn_64_wn_4 as Arm,
            ),
            ("qmm_t_bias", "affine_qmm_t_bias", arm::qmm_t_bias as Arm),
            (
                "qmm_t_bias_fp16_precast",
                "affine_qmm_t_bias_fp16_precast",
                arm::qmm_t_bias_fp16_precast as Arm,
            ),
            (
                "qmm_t_fp16_precast",
                "affine_qmm_t_fp16_precast",
                arm::qmm_t_fp16_precast as Arm,
            ),
            (
                "qmm_t_residual",
                "affine_qmm_t_residual",
                arm::qmm_t_residual as Arm,
            ),
            (
                "qmm_t_residual_fp16_precast",
                "affine_qmm_t_residual_fp16_precast",
                arm::qmm_t_residual_fp16_precast as Arm,
            ),
            (
                "qmm_t_splitk",
                "affine_qmm_t_splitk",
                arm::qmm_t_splitk as Arm,
            ),
            (
                "qmm_t_splitk_f32",
                "affine_qmm_t_splitk_f32",
                arm::qmm_t_splitk_f32 as Arm,
            ),
            (
                "qmm_t_splitk_fp16_precast",
                "affine_qmm_t_splitk_fp16_precast",
                arm::qmm_t_splitk_fp16_precast as Arm,
            ),
            (
                "qmm_t_splitk_fp16_precast_f32",
                "affine_qmm_t_splitk_fp16_precast_f32",
                arm::qmm_t_splitk_fp16_precast_f32 as Arm,
            ),
            (
                "qmm_t_strided",
                "affine_qmm_t_strided",
                arm::qmm_t_strided as Arm,
            ),
            (
                "qmm_t_strided_fp16_precast",
                "affine_qmm_t_strided_fp16_precast",
                arm::qmm_t_strided_fp16_precast as Arm,
            ),
            (
                "qmm_t_strided_fp16_precast_residual",
                "affine_qmm_t_strided_fp16_precast_residual",
                arm::qmm_t_strided_fp16_precast_residual as Arm,
            ),
            (
                "qmm_t_strided_residual",
                "affine_qmm_t_strided_residual",
                arm::qmm_t_strided_residual as Arm,
            ),
            ("qmv_fast", "affine_qmv_fast", arm::qmv_fast as Arm),
            (
                "qmv_fast_residual",
                "affine_qmv_fast_residual",
                arm::qmv_fast_residual as Arm,
            ),
            ("qmv_tail", "affine_qmv_tail", arm::qmv_tail as Arm),
            (
                "qmv_tail_bias",
                "affine_qmv_tail_bias",
                arm::qmv_tail_bias as Arm,
            ),
            (
                "qmv_wide_strided",
                "affine_qmv_wide_strided",
                arm::qmv_wide_strided as Arm,
            ),
        ],
    ),
    (
        kernels_metal::norm::ROUTINES,
        &[
            ("add_bias", "add_bias", arm::add_bias as Arm),
            ("gated_rms", "gated_rms", arm::gated_rms as Arm),
            (
                "gated_rms_strided",
                "gated_rms_strided",
                arm::gated_rms_strided as Arm,
            ),
            (
                "layer_scalar_mul",
                "layer_scalar_mul",
                arm::layer_scalar_mul as Arm,
            ),
            ("residual_add", "residual_add", arm::residual_add as Arm),
            (
                "residual_add_strided",
                "residual_add_strided",
                arm::residual_add_strided as Arm,
            ),
            ("rms_residual", "rms_residual", arm::rms_residual as Arm),
            (
                "rms_residual_scaled",
                "rms_residual_scaled",
                arm::rms_residual_scaled as Arm,
            ),
            (
                "rms_single_row",
                "rms_single_row",
                arm::rms_single_row as Arm,
            ),
            (
                "rms_strided_head_row",
                "rms_strided_head_row",
                arm::rms_strided_head_row as Arm,
            ),
            (
                "rms_strided_row",
                "rms_strided_row",
                arm::rms_strided_row as Arm,
            ),
            (
                "vnorm_single_row",
                "vnorm_single_row",
                arm::vnorm_single_row as Arm,
            ),
        ],
    ),
    (
        kernels_metal::rope::ROUTINES,
        &[
            ("neox_decode", "neox_decode", arm::neox_decode as Arm),
            ("neox_mb", "neox_mb", arm::neox_mb as Arm),
            (
                "neox_prop_decode",
                "neox_prop_decode",
                arm::neox_prop_decode as Arm,
            ),
            ("neox_prop_mb", "neox_prop_mb", arm::neox_prop_mb as Arm),
            (
                "neox_freqs_decode",
                "neox_freqs_decode",
                arm::neox_freqs_decode as Arm,
            ),
            ("neox_freqs_mb", "neox_freqs_mb", arm::neox_freqs_mb as Arm),
            ("neox_strided", "neox_strided", arm::neox_strided as Arm),
        ],
    ),
    (
        kernels_metal::layout::ROUTINES,
        &[
            (
                "embed_gather_4bit",
                "embed_gather_4bit",
                arm::embed_gather_4bit as Arm,
            ),
            (
                "embed_gather_mb_4bit",
                "embed_gather_mb_4bit",
                arm::embed_gather_mb_4bit as Arm,
            ),
            (
                "embed_gather_scaled_4bit",
                "embed_gather_scaled_4bit",
                arm::embed_gather_scaled_4bit as Arm,
            ),
            (
                "embed_gather_scaled_mb_4bit",
                "embed_gather_scaled_mb_4bit",
                arm::embed_gather_scaled_mb_4bit as Arm,
            ),
            ("ple_combine", "ple_combine", arm::ple_combine as Arm),
            ("row_gather", "row_gather", arm::row_gather as Arm),
        ],
    ),
    (
        kernels_metal::mlp::ROUTINES,
        &[
            ("geglu_tanh", "geglu_tanh", arm::geglu_tanh as Arm),
            (
                "geglu_tanh_strided",
                "geglu_tanh_strided",
                arm::geglu_tanh_strided as Arm,
            ),
            ("gptoss_swiglu", "gptoss_swiglu", arm::gptoss_swiglu as Arm),
            ("silu_mul", "silu_mul", arm::silu_mul as Arm),
        ],
    ),
    (
        kernels_metal::sample::ROUTINES,
        &[("argmax_logits", "argmax_logits", arm::argmax_logits as Arm)],
    ),
];

/// The routine this driver calls for `name`, if its family has crossed.
///
/// `name` is the ROW's name — the kernel, not the entrypoint — because that is
/// what a routine is named after and what `kernels::sig_in` already resolves a
/// lowered symbol to. The entrypoint is the body's to spell, which is the
/// direction the whole refactor runs in.
///
/// The pair is the routine and the ARM that feeds it: a signature says what a
/// routine takes and in what order, and never where any of it comes from.
/// See [`crate::lowering::arm`].
///
/// Answers for every routine this backend builds. A family reaches [`LIVE`]
/// only once its arms are written, and all ten have; a routine ADDED without
/// one answers `None` here, which
/// `every_routine_this_backend_builds_has_an_arm` refuses.
#[must_use]
pub fn arm_for(name: &str) -> Option<(&'static Routine<Metal>, Arm)> {
    LIVE.iter().find_map(|(routines, arms)| {
        let arm = arms.iter().find(|(n, _, _)| *n == name)?.2;
        let routine = routines.iter().find(|r| r.name == name)?;
        Some((routine, arm))
    })
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
pub fn crossed(symbol: &str) -> Option<(&'static Routine<Metal>, Arm)> {
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
        .flat_map(|(routines, arms)| arms.iter().map(move |arm| (routines, arm)))
        .filter(|(_, (_, stem, _))| claims(stem))
        .max_by_key(|(_, (_, stem, _))| stem.len())
        .filter(|(_, (_, stem, _))| dark.is_none_or(|d| stem.len() > d))
        .and_then(|(routines, (name, _, arm))| {
            Some((routines.iter().find(|r| r.name == *name)?, *arm))
        })
}

/// Every stem this backend crosses, with the routine it names.
///
/// [`crossed`] answers "which routine claims this SYMBOL", which is the
/// dispatch question. This is the census question: what does the registry
/// claim at all. It exists because the rows used to answer it -- a sweep over
/// `kernels_metal::KERNELS` reached every kernel this backend has, and every
/// family has retired its rows, so a sweep keyed on the table now reaches
/// nothing and says so by passing.
#[must_use]
pub fn stems() -> impl Iterator<Item = (&'static str, &'static Routine<Metal>)> {
    LIVE.iter().flat_map(|(routines, arms)| {
        arms.iter().filter_map(move |(name, stem, _)| {
            Some((*stem, routines.iter().find(|r| r.name == *name)?))
        })
    })
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
    use super::*;

    /// A launch that stages no scalars and mints no packed block.
    const NO_BLOCK: Staged<'static> = Staged {
        block: None,
        words: &[],
    };
    use kernels_metal::routine::{Bind, Buf, BufMut, Ctx, Env, Usize};

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
        x: Buf,
        out: BufMut,
        width: u32,
        stride: Usize,
        rows: Env<u32>,
    ) -> Result<(), Refusal> {
        if *rows == 0 {
            return Err(Refusal::Empty { what: "rows" });
        }
        ctx.dispatch(
            Fire {
                entrypoint: "scale_rows_bfloat16",
                file: "norm/scale.metal",
                lanes: [width, *rows, 1],
                group: [32, 1, 1],
            },
            &[x.v(), out.v(), width.v(), stride.v()],
        )
    }

    /// The row `scale_rows` derives, for a planner to read its types off.
    static ROUTINE: std::sync::LazyLock<Routine<Metal>> =
        std::sync::LazyLock::new(|| kernels_metal::routine!(scale_rows));

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

        scale_rows(&planner, Buf(0), BufMut(1), 64, Usize(128), Env(8))
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

    /// A refusal leaves no dispatch behind.
    #[test]
    fn a_body_that_refuses_does_not_reach_the_plan() {
        let handles = [handle(0x1000, 64), handle(0x2000, 64)];
        let planner = Planner::new(&ROUTINE, &handles, NO_BLOCK, 0..1, 7);
        assert_eq!(
            scale_rows(&planner, Buf(0), BufMut(1), 64, Usize(128), Env(0)),
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
            scale_rows(&planner, Buf(0), BufMut(1), 64, Usize(128), Env(8)),
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
            scale_rows(&planner, Buf(0), BufMut(1), 0, Usize(128), Env(8)),
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
                (Ty::Buf, Provenance::Trace),
                (Ty::BufMut, Provenance::Trace),
                (Ty::U32, Provenance::Trace),
                (Ty::Usize, Provenance::Trace),
                (Ty::U32, Provenance::Env),
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
        for (routines, arms) in LIVE {
            for (name, _, _) in *arms {
                assert!(
                    routines.iter().any(|r| r.name == *name),
                    "{name}: an arm for a routine this family does not have \
                     -- `arm_for` answers `None`, so nothing could dispatch \
                     the symbol at all"
                );
                assert!(arm_for(name).is_some(), "{name}");
            }
        }
    }

    /// Every routine this backend builds has an arm.
    ///
    /// Stated as a number rather than a spot check so that a routine ADDED to
    /// `kernels-metal` without an arm fails here instead of silently taking
    /// a table path that Stage 5 is about to delete.
    #[test]
    fn every_routine_this_backend_builds_has_an_arm() {
        let live: usize = LIVE.iter().map(|(_, arms)| arms.len()).sum();
        let built: usize = LIVE.iter().map(|(routines, _)| routines.len()).sum();
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
    fn every_arm_fills_the_argument_list_its_routine_declares() {
        use crate::lowering::arm::{Facts, Handles};
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
            | Ty::I32s
            | Ty::I32sMut
            | Ty::U32s
            | Ty::U32sMut
            | Ty::U8s
            | Ty::U8sMut
            | Ty::F32s
            | Ty::F32sMut => matches!(v, ArgValue::Buffer(_)),
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
            rows: 4,
            width: 64,
            in_width: 64,
            q_heads: 8,
            kv_heads: 2,
            head_dim: 64,
            rotary_dims: 64,
            n_experts: 8,
            experts_per_token: 2,
            group: 64,
            bits: 4,
            tile: Some((32, 32)),
            layer: 0,
            requests: 2,
        };

        let mut wrong: Vec<String> = Vec::new();
        for (routines, arms) in LIVE {
            for (name, stem, arm) in *arms {
                let Some(routine) = routines.iter().find(|r| r.name == *name) else {
                    wrong.push(format!("  {stem} -> `{name}`: no routine of that name"));
                    continue;
                };
                let mut resolver = Everything;
                let mut handles =
                    Handles::new(&args, &ins, &outs, &weights, &params, &mut resolver);
                let built = match arm(&mut handles, facts) {
                    Ok(built) => built,
                    Err(why) => {
                        wrong.push(format!("  {stem} -> `{name}`: refused: {why}"));
                        continue;
                    }
                };
                if built.len() != routine.args.len() {
                    wrong.push(format!(
                        "  {stem} -> `{name}`: the arm hands {} value(s), the \
                         routine takes {}",
                        built.len(),
                        routine.args.len()
                    ));
                    continue;
                }
                for (at, (value, (ty, _))) in built.iter().zip(routine.args).enumerate() {
                    if !fits(*ty, *value) {
                        wrong.push(format!(
                            "  {stem} -> `{name}`: argument {at} is {ty:?} and \
                             the arm bound {}",
                            value.kind()
                        ));
                    }
                }
            }
        }
        assert!(
            wrong.is_empty(),
            "an arm and its routine disagree about the call between them. \
             Until this test existed the disagreement was a `Refusal` at \
             dispatch, which names the arity but not the arm:\n{}",
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
        let by = |symbol: &str| crossed(symbol).map(|(r, _)| r.name);
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
        for (_, arms) in LIVE {
            for (name, stem, _) in *arms {
                let got = crossed(stem)
                    .unwrap_or_else(|| panic!("`{stem}` resolves nothing"))
                    .0
                    .name;
                assert_eq!(got, *name, "`{stem}` resolves the wrong routine");
            }
        }
    }
}
