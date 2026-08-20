//! The adapter correction: every staged LoRA lane's `x·Aᵀ·Bᵀ` delta landed
//! on the materialised q and v projections.
//!
//! It is a `gemm` and not a family of its own because every launch in its three
//! passes is a matmul: the solo lanes are pairs of [`super::act_x_wt_bf16`],
//! the grouped lanes are three [`super::grouped_act_x_wt_bf16`] over a staged
//! pointer slab, and the scale pass is `quant::scale_rows_bf16`. **There is no
//! device text here** — the LoRA seam is batched cuBLAS, so no `__global__` was
//! ever written for it. `gemm::mla_absorb_*` is the precedent, in this same
//! directory.
//!
//! The symbol is `gemm::lora_qkv_correction` and its row derives from the `fn`
//! below, because **the module path IS the trace namespace**, taken from
//! `module_path!()` by [`jit::Family::new`] and therefore unable to drift. A
//! bare symbol has no family to derive from — [`jit::Family::symbol`] is a
//! namespace, `::`, and a name — so its row could only ever be hand-written.
//!
//! [`jit::Family::new`]: crate::jit::Family::new
//! [`jit::Family::symbol`]: crate::jit::Family::symbol
//!
//! The launch half of `driver-cuda`'s `fire/lora.rs` lives here: [`Lane`],
//! [`Group`] and the pointer-slab arithmetic over them, because every field of
//! both exists to be READ by [`lora_qkv_correction`]. [`LoraLaneView`] and its
//! site vocabulary come with them, being the words the three passes are written
//! in. The staging half stays in the driver: `LoraFireState::stage` validates
//! the lane table, casts the fp32 adapters to bf16 and lays out the slab
//! through `fire::sideband_arena::DeviceMemory`, and neither of this crate's
//! device-memory shapes serves that — [`Ctx`] is per call and `jit::device`'s
//! scratch is per process, where staging wants a per-fire bump arena.
//!
//! [`Staged`] is the seam, and it is a borrow rather than a copy because the
//! two `Vec`s are the fire's and outlive every launch made from them.

use kernels_macros::routine;
use core::ffi::c_void;

use crate::jit::Ctx;
use crate::jit::abi::bf16;
use kernels::Refusal;
// `act_x_wt_bf16`'s weight position wears `Const<Tensor<c_void>>` so that
// `#[routine]` derives `Source::Named(<keys::NamedWeight as keys::Fact>::KEY)`
// for it rather than `In(1)`. LoRA is the one caller for whom that derived claim is false —
// these are adapter matrices off a `LoraLane` — and it costs nothing, because
// this file never fires through the binder: it is a host program calling a Rust
// function, and the column it would have used is one it never reads. The three
// calls below likewise build an `In` and an `Out` by hand for extents that live
// inside the regions; the SHAPE they carry is true, and every number passed is
// the lane view's own.
use kernels::routine::{Const, In, InOut, Out};
// `#[routine]` on the correction below is fully qualified,
// because the family spells it that way in all four files that hold a launcher.

// ── the sink's vocabulary ────────────────────────────────────────────────

/// `q_proj` — consumed.
pub const LORA_SITE_Q: u64 = 1 << 0;
/// `k_proj` — reserved.
pub const LORA_SITE_K: u64 = 1 << 1;
/// `v_proj` — consumed.
pub const LORA_SITE_V: u64 = 1 << 2;
/// `o_proj` — reserved.
pub const LORA_SITE_O: u64 = 1 << 3;
/// gate/up — reserved.
pub const LORA_SITE_GATE_UP: u64 = 1 << 4;
/// `down_proj` — reserved.
pub const LORA_SITE_DOWN: u64 = 1 << 5;
/// Every bit the vocabulary defines.
pub const LORA_SITES_KNOWN: u64 =
    LORA_SITE_Q | LORA_SITE_K | LORA_SITE_V | LORA_SITE_O | LORA_SITE_GATE_UP | LORA_SITE_DOWN;
/// The bits v0 actually applies.
///
/// A lane naming any other known bit binds fine and is refused loudly at
/// first use — a silently ignored site would be a request whose adapter never
/// applied while every sample still returned. The refusal is the staging's,
/// which is why that constant and this one are here together: they are one
/// vocabulary, and splitting them across the crate boundary would let the
/// list of sites and the list of sites-with-an-arm drift apart.
pub const LORA_SITES_CONSUMED: u64 = LORA_SITE_Q | LORA_SITE_V;

/// The adapter FORM — the sink's arity selects it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum LoraForm {
    /// 3 args: low-rank `y += B(Ax)`.
    #[default]
    LowRank = 0,
    /// 2 args: SCALE `y = l ⊙ y` (IA3) — `a` holds the l vector, `b`
    /// null, rank/d_in zero.
    Scale = 1,
}

/// One lane's resolved `lora` sink. Ports `LoraLaneView`.
#[derive(Debug, Clone, Copy)]
pub struct LoraLaneView {
    /// The A channel's committed cell, f32, `[num_layers, R, d_in]`.
    pub a: *const c_void,
    /// The B channel's committed cell, f32, `[num_layers, d_out, R]`;
    /// the LoRA scale is folded into the contents.
    pub b: *const c_void,
    /// The SITES placement bitmask — structure, not contents.
    pub sites_bits: u64,
    /// The lane's span in fire token rows.
    pub token_start: u32,
    /// Rows in the span.
    pub token_count: u32,
    /// Adapter geometry, element counts.
    pub num_layers: u32,
    /// The rank; zero on a scale lane.
    pub rank: u32,
    /// Input width; zero on a scale lane.
    pub d_in: u32,
    /// Output width.
    pub d_out: u32,
    /// Low-rank or scale.
    pub form: LoraForm,
}

// ── what the staging leaves for the launcher ─────────────────────────────

/// One staged lane: its view, and the bf16 casts the staging made for it.
///
/// Fields are `pub` because the writer is one crate up and the reader is the
/// function below. That asymmetry is the honest shape while `stage` needs a
/// `DeviceMemory` this crate does not have — see the module header — and it
/// is not a licence for anything else to read them: nothing but
/// [`lora_qkv_correction`] does.
#[derive(Debug, Clone, Copy)]
pub struct Lane {
    /// The sink as the plan and the session resolved it.
    pub view: LoraLaneView,
    /// `a` cast to bf16 in the fire's staging arena, `[layers, R, d_in]`.
    pub a_bf16: *mut c_void,
    /// `b` cast to bf16, `[layers, d_out, R]`. Null on a scale lane.
    pub b_bf16: *mut c_void,
    /// This lane's offset into the shared `xAᵀ` scratch, in elements.
    ///
    /// Meaningful only when [`Self::grouped`]: a solo lane writes `xAᵀ` at
    /// the scratch's base, because its two GEMMs are issued back to back and
    /// nothing else is in flight against it.
    pub xa_offset: usize,
    /// Whether a group claimed this lane. A grouped lane is skipped by the
    /// solo pass and reached through its group's slab slots instead.
    pub grouped: bool,
}

/// One same-shape group: the lanes that share a GEMM shape, and the row
/// counts each of the three grouped calls needs.
///
/// The key is `(rank, d_in, d_out)` and the precondition is pairwise-disjoint
/// token spans — one `cublasGemmGroupedBatchedEx` runs its `beta = 1`
/// accumulations concurrently, so two lanes writing the same rows would race.
/// Both are the staging's to establish; this record is what it establishes.
#[derive(Debug, Clone, Default)]
pub struct Group {
    /// The shared rank.
    pub rank: i32,
    /// The shared input width.
    pub d_in: i32,
    /// The shared output width.
    pub d_out: i32,
    /// Indices into [`Staged::lanes`]. Only the COUNT is read here — the
    /// pointers themselves were resolved into the slab at staging time,
    /// which is what makes this pass capture-safe.
    pub members: Vec<usize>,
    /// How many members carry [`LORA_SITE_Q`].
    pub nq: i32,
    /// How many carry [`LORA_SITE_V`].
    pub nv: i32,
    /// Each member's token count, for the `xAᵀ` call.
    pub m: Vec<i32>,
    /// Each q-site member's token count.
    pub mq: Vec<i32>,
    /// Each v-site member's token count.
    pub mv: Vec<i32>,
    /// This group's first slot in one layer's slab stride.
    pub slab_off: usize,
}

/// What one fire's staging leaves for [`lora_qkv_correction`] to launch.
///
/// A borrow, not a copy: the two slices are the fire's own vectors, and they
/// outlive every launch issued from them. The pointer slab is device memory
/// the staging uploaded once, at fire setup, which is what lets this pass be
/// recorded into a captured graph at all.
#[derive(Debug, Clone, Copy)]
pub struct Staged<'a> {
    /// Every usable lane, in staging order.
    pub lanes: &'a [Lane],
    /// The same-shape groups, empty when grouping is off or found nothing.
    pub groups: &'a [Group],
    /// The device pointer slab: `[layers][slab_stride]` device addresses.
    pub ptr_slab: *mut c_void,
    /// One layer's worth of slots.
    pub slab_stride: usize,
}

/// `base + row * width` bf16 elements — the C++ `bf16_row`.
///
/// `pub` because the staging one crate up addresses the same rows to fill the
/// slab that this file reads back, and two copies of a row-stride computation
/// either side of a boundary is the defect the descent that moved this file
/// was looking for. It is `* 2` and not `* size_of::<bf16>()` for the same
/// reason the C++ was: the width is in ELEMENTS and the type is fixed by the
/// name.
///
/// # Safety
///
/// `base` must address at least `(row + 1) * width` bf16 elements. The result
/// is an address, never dereferenced here.
pub fn bf16_row(base: *const c_void, row: u32, width: i32) -> *const c_void {
    let off = row as usize * usize::try_from(width.max(0)).unwrap_or(0) * 2;
    // SAFETY: offset arithmetic only; the result is handed to a GEMM and is
    // never dereferenced here.
    unsafe { base.cast::<u8>().add(off).cast() }
}

/// `gemm::lora_qkv_correction` — the staged adapter delta, applied.
///
/// Three passes, in the C++'s order, and the order is the arithmetic:
///
/// 1. **Solo lanes.** `xAᵀ` into the shared scratch base, then `(xAᵀ)Bᵀ`
///    accumulated `β = 1` into the q and v row windows the lane's SITES bits
///    name.
/// 2. **Grouped lanes.** Slot arithmetic over the staged pointer slab and
///    three grouped GEMMs — measured at up to 24.75x over separate launches
///    when shapes share (stage0-l40s §3.1).
/// 3. **The SCALE pass, last** — after every delta, so a same-site low-rank
///    plus scale composes as `s ⊙ (y + B(Ax))`, which is DoRA's order. A lone
///    scale lane is IA3, unchanged.
///
/// The LAYER is the op tag's, never `param1` — the bug the C++'s first live
/// A/B caught.
///
/// # Errors
///
/// [`Refusal::Absent`] if this context carries no cuBLAS handle, and whatever
/// the three matmuls or the row scale refuse. Every extent they check is one
/// the staging already guarded, so a refusal here means the staging and this
/// disagree about the state they share.
///
/// # Safety
///
/// `qkv_in`, `q_out`, `v_out` and `xa_scratch` must address live device
/// memory for the fire's row count at widths `h`, `hq`, `hk` and the group
/// scratch's, and every pointer [`Staged`] carries must be one the staging
/// laid down for THIS fire — a slab from a previous fire is a set of
/// addresses that have since been reused.
#[routine(untraced, driver)]
pub fn lora_qkv_correction(
    ctx: &Ctx<'_>,
    staged: Staged<'_>,
    layer: i32,
    qkv_in: *const c_void,
    h: i32,
    hq: i32,
    hk: i32,
    q_out: *mut c_void,
    v_out: *mut c_void,
    xa_scratch: *mut c_void) -> Result<(), Refusal> {
    let layer_u = usize::try_from(layer).unwrap_or(0);

    for lane in staged.lanes {
        if lane.grouped {
            continue;
        }
        let v = &lane.view;
        if v.form == LoraForm::Scale {
            continue; // the scale pass below, after every delta
        }
        let t = i32::try_from(v.token_count).unwrap_or(0);
        let r = i32::try_from(v.rank).unwrap_or(0);
        let a_l = bf16_row(
            lane.a_bf16.cast_const(),
            u32::try_from(layer_u * v.rank as usize).unwrap_or(0),
            i32::try_from(v.d_in).unwrap_or(0),
        );
        let b_l = bf16_row(
            lane.b_bf16.cast_const(),
            u32::try_from(layer_u * v.d_out as usize).unwrap_or(0),
            r,
        );
        let x = bf16_row(qkv_in, v.token_start, h);
        // THE THREE EXTENTS ARE STILL HERE AND THEY ARE ARGUMENTS TO A
        // STRUCT NOW. `act_x_wt_bf16` lost `m`, `n` and `k` in STAGE 3
        // (`gemm/mod.rs`'s ledger), so this call site supplies the same
        // three numbers as the regions' own fields: `m = t`, `k = h` on the
        // activation, `n = r` on the result.
        //
        // CONSTRUCTING A REGION BY HAND IS LEGITIMATE HERE AND WOULD NOT BE
        // ON THE GEMV LEG, which is the distinction `gemv.rs`'s header
        // draws. These three numbers are the lane view's own -- `v.rank`,
        // `v.d_in`, `v.token_count`, read off the staging that allocated
        // `xa_scratch` -- so the shape a `In`/`Out` carries here is the
        // shape the buffer actually has. Nothing is being invented to fill
        // a field.
        super::act_x_wt_bf16_beta(
            ctx,
            In { ptr: x, rows: t, width: h },
            Const { v: a_l },
            Out { ptr: xa_scratch, rows: t, width: r },
            0.0,
        )?;
        let d_out = i32::try_from(v.d_out).unwrap_or(0);
        if v.sites_bits & LORA_SITE_Q != 0 {
            super::act_x_wt_bf16_beta(
                ctx,
                In { ptr: xa_scratch.cast_const(), rows: t, width: r },
                Const { v: b_l },
                Out {
                    ptr: bf16_row(q_out.cast_const(), v.token_start, hq).cast_mut(),
                    rows: t,
                    width: d_out,
                },
                1.0,
            )?;
        }
        if v.sites_bits & LORA_SITE_V != 0 {
            super::act_x_wt_bf16_beta(
                ctx,
                In { ptr: xa_scratch.cast_const(), rows: t, width: r },
                Const { v: b_l },
                Out {
                    ptr: bf16_row(v_out.cast_const(), v.token_start, hk).cast_mut(),
                    rows: t,
                    width: d_out,
                },
                1.0,
            )?;
        }
    }

    for g in staged.groups {
        let n = g.members.len();
        // The slab was fully staged at fire setup — slot arithmetic and
        // launches, nothing else, which is what a captured body requires.
        // SAFETY: `layer_u * slab_stride + slab_off` is inside the
        // `layers * slab_stride` slab the staging allocated and filled; the
        // arithmetic is the staging's own layout read back.
        let slot = unsafe {
            staged
                .ptr_slab
                .cast::<*const c_void>()
                .add(layer_u * staged.slab_stride + g.slab_off)
        };
        let x_ptrs: *const *const c_void = slot;
        // SAFETY: the three runs are consecutive by the same layout —
        // `[x a xa]` n each, then `[q_act q_w q_y]` nq each, then the v
        // triple — which is the order the staging wrote them in.
        unsafe {
            let a_ptrs = x_ptrs.add(n);
            let xa_ptrs = x_ptrs.add(2 * n);
            // THE FOUR TABLES ARE `#[source(Unbound)]` AT THE SIGNATURE, and
            // these are the call sites that note cites as its evidence: every
            // argument below is visibly slab arithmetic -- `x_ptrs.add(n)`,
            // `base.add(2 * g.nv as usize)`, `g.m.as_ptr()` on a host
            // `Vec<i32>` -- so there is no operand behind any of them. The
            // three pointer runs are DEVICE addresses out of `ptr_slab`;
            // only `g.m` is host, which is the split cuBLAS asks for.
            super::grouped_act_x_wt_bf16(
                ctx,
                x_ptrs,
                a_ptrs,
                xa_ptrs.cast::<*mut c_void>().cast_mut(),
                g.m.as_ptr(),
                i32::try_from(n).unwrap_or(0),
                0.0,
            )?;
            if g.nq > 0 {
                let base = x_ptrs.add(3 * n);
                super::grouped_act_x_wt_bf16(
                    ctx,
                    base,
                    base.add(g.nq as usize),
                    base.add(2 * g.nq as usize).cast::<*mut c_void>().cast_mut(),
                    g.mq.as_ptr(),
                    g.nq,
                    1.0,
                )?;
            }
            if g.nv > 0 {
                let base = x_ptrs.add(3 * n + 3 * g.nq as usize);
                super::grouped_act_x_wt_bf16(
                    ctx,
                    base,
                    base.add(g.nv as usize),
                    base.add(2 * g.nv as usize).cast::<*mut c_void>().cast_mut(),
                    g.mv.as_ptr(),
                    g.nv,
                    1.0,
                )?;
            }
        }
    }

    for lane in staged.lanes {
        let v = &lane.view;
        if v.form != LoraForm::Scale {
            continue;
        }
        let t = i32::try_from(v.token_count).unwrap_or(0);
        let d_out = i32::try_from(v.d_out).unwrap_or(0);
        let l_l = bf16_row(lane.a_bf16.cast_const(), u32::try_from(layer_u).unwrap_or(0), d_out);
        if v.sites_bits & LORA_SITE_Q != 0 {
            crate::quant::scale_rows::<bf16>(
                ctx,
                // THE SAME RECTANGLE THE `act_x_wt_bf16` CALL ABOVE BUILDS
                // FROM THE SAME POINTER, which is why this is a conversion
                // and not an invention. `scale_rows` deleted its `rows` and
                // `width` parameters when its `OutSlot<0, _>` became
                // `InOut<Tensor<_>>`; the two scalars that used to follow this
                // argument were `t` and `d_out`, and they are the two the
                // region carries now.
                InOut {
                    ptr: bf16_row(q_out.cast_const(), v.token_start, hq)
                        .cast_mut()
                        .cast::<crate::jit::abi::bf16>(),
                    rows: t,
                    width: d_out,
                },
                // THE SCALE VECTOR NOW CARRIES ITS EXTENT, because `InSlot`
                // is gone and F1 says why: a layout is 1:1 with the address
                // and never absent -- what was absent was a TRANSPORT that
                // dropped it. This caller has always known both numbers; the
                // wrapper simply had nowhere to put them.
                In {
                    ptr: l_l.cast::<crate::jit::abi::bf16>(),
                    rows: t,
                    width: d_out,
                },
            )?;
        }
        if v.sites_bits & LORA_SITE_V != 0 {
            crate::quant::scale_rows::<bf16>(
                ctx,
                // As the Q site above, and the same pair of numbers.
                InOut {
                    ptr: bf16_row(v_out.cast_const(), v.token_start, hk)
                        .cast_mut()
                        .cast::<crate::jit::abi::bf16>(),
                    rows: t,
                    width: d_out,
                },
                // THE SCALE VECTOR NOW CARRIES ITS EXTENT, because `InSlot`
                // is gone and F1 says why: a layout is 1:1 with the address
                // and never absent -- what was absent was a TRANSPORT that
                // dropped it. This caller has always known both numbers; the
                // wrapper simply had nowhere to put them.
                In {
                    ptr: l_l.cast::<crate::jit::abi::bf16>(),
                    rows: t,
                    width: d_out,
                },
            )?;
        }
    }
    Ok(())
}
