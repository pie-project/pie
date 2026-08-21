//! The raised views this driver builds — the answer half of `In<Struct<..>>`.
//!
//! A swept routine no longer asks `ctx.ask::<_, keys::KvKeys>()` per fact;
//! it takes ONE operand, `In<Struct<KvCache>>`, and reads the view the
//! driver built. Identity lives in `kernels::runtime` (`"kv_cache"`,
//! `"recurrent_state"`, `"attention_mask"`), the carrier in
//! `kernels_metal::views`, and the answer HERE: the statement's operand
//! arrives as `Arg::Raised { key, .. }`, [`Views::raise`] matches the key,
//! and the fields are minted through the same [`Handles`] doors the per-key
//! answering used — [`Handles::kv`], [`Handles::table`], [`Handles::slab`],
//! [`Handles::pooled`].
//!
//! # Lifetime
//!
//! The carrier crosses as `ArgValue::Raised(address)`. On this plane a view
//! holds HANDLES — indices into the launch's own minted list — so a view is
//! per LAUNCH; what must hold is that its address outlives the body that
//! reads it. Each view is boxed (a stable address) and the holder lives on
//! `dispatch::plan_routine`'s stack past the body call. A RECORDED fire
//! (`crate::fire::recordings`) replays encoded dispatches and runs no body,
//! so no address survives into a replay.
//!
//! # Absence binds [`super::hold`]'s NOTHING, mostly
//!
//! [`Handles::kv`] and [`Handles::table`] answer an absent pool or table
//! with a zero-address region, which is this backend's honest null — the
//! encoder binds it and a shader that reads it faults loudly rather than
//! reading a neighbour. The one exception keeps its old posture:
//! [`Handles::slab`] REFUSES when the driver holds no recurrent pool,
//! because a scan handed a null carry answers fluently and wrongly.

use kernels::routine::Refusal;
use kernels::shader::{Tensor, Usize};
use kernels::{Kind, Source};
use kernels_metal::routine::ArgValue;
use kernels_metal::views::{MaskView, PagedKvView, RecurrentView, SplitView};

use super::executor::FireTable;
use super::hold::{Facts, Handles};
use model_compiler::lower::Arg;

/// The raised views of ONE launch, kept alive until its body has run.
pub struct Views<'a> {
    /// The launch's operands, as the lowering states them — the half
    /// [`Handles`] does not carry on this plane (it is built from
    /// `BoundArg`s).
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
    /// view, or when the key is not one this driver builds; a slab's own
    /// refusal for a recurrent view on a driver with no pool.
    pub fn raise(
        &mut self,
        source: Source,
        o: &mut Handles<'_>,
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
                let view = Box::new(kv(o, f));
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
                // This plane fires unsplit; one split is the whole answer
                // and the partials handle is then never read.
                let view = Box::new(SplitView {
                    partials: Tensor::new(0),
                    splits: 1,
                });
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
/// (`Facts::layer`, which `facts_of` reads off the launch's span).
///
/// The pool's numbers come through [`Handles::pooled`], zero when the store
/// has no pool behind it — the same zero [`super::hold`]'s NOTHING stands
/// for, and a paged read at page size zero refuses at its grid.
fn kv(o: &mut Handles<'_>, f: Facts) -> PagedKvView {
    let pooled = |o: &mut Handles<'_>, which| o.pooled(which).unwrap_or(0);
    let page_size = pooled(o, FireTable::KvPageSize);
    let seq_stride = pooled(o, FireTable::KvSeqStride);
    let head_stride = pooled(o, FireTable::KvHeadStride);
    PagedKvView {
        keys: Tensor::new(o.kv(f.layer, false)),
        values: Tensor::new(o.kv(f.layer, true)),
        page_indices: Tensor::new(o.table(FireTable::KvPageIndices)),
        page_indptr: Tensor::new(o.table(FireTable::KvPageIndptr)),
        write_page: Tensor::new(o.table(FireTable::KvWritePage)),
        write_offset: Tensor::new(o.table(FireTable::KvWriteOffset)),
        page_size: page_size.cast_signed(),
        seq_stride: Usize(u64::from(seq_stride)),
        head_stride: Usize(u64::from(head_stride)),
    }
}

/// The recurrent-state view: the three slabs and the fire's slot table.
///
/// [`Handles::slab`] refuses on a driver with no recurrent pool, and the
/// refusal is the point — see its doc for why a null carry is worse.
fn recurrent(o: &mut Handles<'_>, f: Facts) -> Result<RecurrentView, Refusal> {
    Ok(RecurrentView {
        state: Tensor::new(o.slab(f.layer, "recurrent_state")?),
        slots: Tensor::new(o.table(FireTable::RecurrentSlots)),
        conv_state: Tensor::new(o.slab(f.layer, "conv_state")?),
        new_conv_state: Tensor::new(o.slab(f.layer, "new_conv_state")?),
    })
}

/// The custom-mask triple. This driver stages the enable plane as zeros and
/// no mask beside it (`Frame` has no field for a user mask), so the pitch is
/// ZERO and means it — the same answer the retired
/// `keys::AttentionMaskStride` arm gave, moved into the view.
fn mask(o: &mut Handles<'_>) -> MaskView {
    MaskView {
        mask: Tensor::new(o.table(FireTable::AttentionMask)),
        enabled: Tensor::new(o.table(FireTable::AttentionMaskEnabled)),
        stride: 0,
    }
}
