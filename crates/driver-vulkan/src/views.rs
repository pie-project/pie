//! The raised views this driver builds — the answer half of `In<Struct<..>>`.
//!
//! A swept routine no longer asks `ctx.ask::<_, keys::KvKeys>()` per fact;
//! it takes ONE operand, `In<Struct<KvCache>>`, and reads the view the
//! driver built. Identity lives in `kernels::runtime` (`"kv_cache"`,
//! `"recurrent_state"`, `"attention_mask"`), the carrier in
//! `kernels_vulkan::views`, and the answer HERE: the statement's operand
//! arrives as `Arg::Raised { key, .. }`, [`Views::raise`] matches the key,
//! and the fields are minted through the same [`Handles`] doors the per-key
//! answering used — [`Handles::kv`], [`Handles::table_or_unbound`],
//! [`Handles::slab`], [`Handles::number`].
//!
//! # Lifetime
//!
//! The carrier crosses as `ArgValue::Raised(address)`. On this plane a view
//! holds HANDLES — indices into the launch's own bound list — so a view is
//! per LAUNCH; what must hold is that its address outlives the body that
//! reads it. Each view is boxed (a stable address) and the holder lives on
//! `serve::plan_routine`'s stack past the body call. A captured REPLAY
//! (`crate::replay`) re-submits the recorded command buffer and runs no
//! body, so no address survives into a replay.
//!
//! # Absence is deferred, not refused
//!
//! The per-key answering refused an absent table AT THE ASK, which was
//! per-field lazy. A view is built whole, so an absent optional field
//! (a mask table, a write descriptor) mints [`crate::hold::UNBOUND`]: a body
//! that drops the field loses nothing, and one that binds it is refused at
//! encode by name — the same late, loud failure the split decode's partials
//! already use. The two REQUIRED absences keep their old posture: a missing
//! KV pool refuses the launch, and a recurrent slab refuses always, because
//! this driver allocates none.

use kernels::routine::Refusal;
use kernels::shader::{Tensor, Usize};
use kernels::{Kind, Source};
use kernels_vulkan::routine::ArgValue;
use kernels_vulkan::views::{MaskView, PagedKvView, RecurrentView, SplitView};

use crate::binding::{FireNumber, FireTable};
use crate::hold::{Facts, Handles};
use model_compiler::lower::Arg;

/// The raised views of ONE launch, kept alive until its body has run.
pub struct Views<'a> {
    /// The launch's operands, as the lowering states them — the half
    /// [`Handles`] does not carry on this plane (it is built from `Bound`s).
    args: &'a [Arg],
    /// Which of them are the INPUTS, in slot order.
    ins: &'a [usize],
    /// Boxed so every address is stable however the vectors grow.
    kv: Vec<Box<PagedKvView>>,
    rs: Vec<Box<RecurrentView>>,
    mask: Vec<Box<MaskView>>,
    split: Vec<Box<SplitView>>,
}

impl<'a> Views<'a> {
    /// An empty holder over one launch's operand list.
    #[must_use]
    pub fn over(args: &'a [Arg], ins: &'a [usize]) -> Self {
        Self {
            args,
            ins,
            kv: Vec::new(),
            rs: Vec::new(),
            mask: Vec::new(),
            split: Vec::new(),
        }
    }

    /// Answer one `Ty::Raised` argument: build the view the statement's
    /// operand names and hand back its address.
    ///
    /// # Errors
    ///
    /// [`Refusal::Unstated`] when the source is not an input slot, when the
    /// statement placed an ordinary operand where the signature marks a
    /// view, or when the key is not one this driver builds; whatever a
    /// required field's own absence refuses with otherwise.
    pub fn raise(
        &mut self,
        source: Source,
        o: &mut Handles<'_, '_>,
        f: Facts,
    ) -> Result<ArgValue, Refusal> {
        let Source::Slot(Kind::In, n) = source else {
            return Err(Refusal::Unstated {
                what: "a raised operand whose source is not one of the statement's inputs",
            });
        };
        let at = *self.ins.get(usize::from(n)).ok_or(Refusal::Absent {
            what: "an input operand the signature marks as a raised view",
        })?;
        let Some(Arg::Raised { key, .. }) = self.args.get(at) else {
            return Err(Refusal::Unstated {
                what: "a raised view where the statement placed an ordinary operand",
            });
        };
        match key.as_str() {
            "kv_cache" => {
                let view = Box::new(kv(o, f)?);
                let at = std::ptr::from_ref::<PagedKvView>(view.as_ref()) as usize;
                self.kv.push(view);
                Ok(ArgValue::Raised(at))
            }
            "recurrent_state" => {
                let view = Box::new(recurrent(o, f)?);
                let at = std::ptr::from_ref::<RecurrentView>(view.as_ref()) as usize;
                self.rs.push(view);
                Ok(ArgValue::Raised(at))
            }
            "attention_mask" => {
                let view = Box::new(mask(o));
                let at = std::ptr::from_ref::<MaskView>(view.as_ref()) as usize;
                self.mask.push(view);
                Ok(ArgValue::Raised(at))
            }
            "attn.split_policy" => {
                let view = Box::new(split(o, f));
                let at = std::ptr::from_ref::<SplitView>(view.as_ref()) as usize;
                self.split.push(view);
                Ok(ArgValue::Raised(at))
            }
            _ => Err(Refusal::Unstated {
                what: "a runtime object this driver does not build a view for",
            }),
        }
    }
}

/// The paged KV cache, one view per launch, at the RECTANGLE's layer
/// (`Facts::layer`, which `plan_routine` reads off `launch.layers.start`).
///
/// The strides are what `Pool::number` answers — `Shape::number` answers
/// `None` for a stride over a paged pool, which lands here as ZERO, the
/// value this crate already reads as "no extent". Only the page-table half
/// is real on a paged fire; a contiguous kernel handed a zero stride refuses
/// at its grid rather than attending to the wrong context.
fn kv(o: &mut Handles<'_, '_>, f: Facts) -> Result<PagedKvView, Refusal> {
    Ok(PagedKvView {
        keys: Tensor::new(o.kv(f.layer, false)?),
        values: Tensor::new(o.kv(f.layer, true)?),
        page_indices: Tensor::new(o.table_or_unbound(FireTable::KvPageIndices)),
        page_indptr: Tensor::new(o.table_or_unbound(FireTable::KvPageIndptr)),
        write_page: Tensor::new(o.table_or_unbound(FireTable::KvWritePage)),
        write_offset: Tensor::new(o.table_or_unbound(FireTable::KvWriteOffset)),
        page_size: o.number(FireNumber::KvPageSize).unwrap_or(0).cast_signed(),
        seq_stride: Usize(u64::from(o.number(FireNumber::KvSeqStride).unwrap_or(0))),
        head_stride: Usize(u64::from(o.number(FireNumber::KvHeadStride).unwrap_or(0))),
    })
}

/// The recurrent-state view — which [`Handles::slab`] refuses on this
/// driver, always, because it allocates no slabs. The refusal is the same
/// sentence the per-key answering gave, from the same door; when a
/// recurrent pool lands, `slab` is still the one function that changes.
fn recurrent(o: &mut Handles<'_, '_>, f: Facts) -> Result<RecurrentView, Refusal> {
    Ok(RecurrentView {
        state: Tensor::new(o.slab(f.layer, "recurrent_state")?),
        // No slot table exists on this driver yet; a body that binds this
        // handle is refused at encode by name. Unreachable today behind the
        // slab refusal above.
        slots: Tensor::new(o.unbound()),
        conv_state: Tensor::new(o.slab(f.layer, "conv_state")?),
        new_conv_state: Tensor::new(o.slab(f.layer, "new_conv_state")?),
    })
}

/// The custom-mask triple. This driver stages a real (or zeroed) mask every
/// fire, so the tables resolve; the pitch is the fire's own number, zero
/// while no mask is in force — which is what the zeroed table means.
fn mask(o: &mut Handles<'_, '_>) -> MaskView {
    MaskView {
        mask: Tensor::new(o.table_or_unbound(FireTable::AttentionMask)),
        enabled: Tensor::new(o.table_or_unbound(FireTable::AttentionMaskEnabled)),
        stride: o.number(FireNumber::AttentionMaskStride).unwrap_or(0),
    }
}

/// The decode split policy — the judgement `Whence::Splits` used to make.
///
/// How many ways to fold a decode's key range is a fact about THIS fire
/// (history depth against head and row count), so it is the driver's to
/// compute and the statement's only to CARRY. A fire with no partials
/// buffer answers one split — the single-pass path — and the handle it
/// carries is then never read, which the two agree on by construction.
fn split(o: &mut Handles<'_, '_>, f: Facts) -> SplitView {
    SplitView {
        partials: Tensor::new(o.table_or_unbound(FireTable::AttnPartials)),
        splits: o.decode_splits(f),
    }
}
