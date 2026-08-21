//! The raised views this driver builds — the answer half of `In<Struct<..>>`.
//!
//! A swept routine no longer asks `ctx.ask::<_, keys::KvKeys>()` fourteen
//! times; it takes ONE operand, `In<Struct<KvCache>>`, and reads the fields
//! of the view the driver built. Identity lives in `kernels::runtime`
//! (`"kv_cache"`, `"recurrent_state"`, `"attention_mask"`), the carrier in
//! `kernels_wgpu::views`, and the answer HERE: the statement's operand
//! arrives as `Arg::Raised { key, .. }`, [`Views::raise`] matches the key,
//! and the view's fields are minted through the same [`Handles`] doors the
//! per-key answering used — [`Handles::kv`], [`Handles::table`],
//! [`Handles::slab`], [`Handles::fire_number`] — so `plan`'s resolution loop
//! resolves them exactly as it resolved a body's own asks.
//!
//! # Lifetime
//!
//! The carrier crosses as `ArgValue::Raised(address)`. On this plane a view
//! holds HANDLES — indices into the launch's own minted list — so a view is
//! per LAUNCH, not per fire: a handle from one launch means nothing in the
//! next. What must hold is that the address outlives the body that reads it,
//! and [`Views`] is how `plan` states that: each view is boxed (a stable
//! address), pushed here, and the whole holder lives on `plan`'s stack past
//! `stating`. Nothing device-side ever sees the address — the body reads the
//! view on the host and hands back its fields' handles.
//!
//! # The paged-stride guard, relocated
//!
//! `bind::contiguous_pool` used to refuse `KvSeqStride`/`KvHeadStride` over
//! a paged pool, because `resources::Shape::number` derives a number there
//! that addresses the wrong tokens. One view now carries both the strides
//! and the page tables, so the refusal became a ZEROING: over a paged pool
//! the stride fields are 0 — the value this crate already reads as "the
//! statement gave no extent" — and only the page-table half is real. A
//! contiguous kernel handed a zero stride refuses at its grid rather than
//! attending to the wrong context.

use kernels::routine::Refusal;
use kernels::shader::{Tensor, Usize};
use kernels::{Kind, Source};
use kernels_wgpu::routine::ArgValue;
use kernels_wgpu::views::{MaskView, PagedKvView, RecurrentView, SplitView};

use super::hold::Handles;
use crate::binding::{FireNumber, FireTable};

/// The raised views of ONE launch, kept alive until its body has run.
///
/// `Default` is the empty holder `plan` starts each launch with; the probe
/// pass gets a scratch one whose views are dropped with it.
#[derive(Debug, Default)]
pub struct Views {
    /// Boxed so every address is stable however the vectors grow.
    kv: Vec<Box<PagedKvView>>,
    rs: Vec<Box<RecurrentView>>,
    mask: Vec<Box<MaskView>>,    split: Vec<Box<SplitView>>,
}

impl Views {
    /// Answer one `Ty::Raised` argument: build the view the statement's
    /// operand names and hand back its address.
    ///
    /// # Errors
    ///
    /// [`Refusal::Unstated`] when the source is not an input slot, when the
    /// statement placed an ordinary operand where the signature marks a
    /// view, or when the key is not one this driver builds.
    pub fn raise(
        &mut self,
        source: Source,
        o: &mut Handles<'_>,
    ) -> Result<ArgValue, Refusal> {
        let Source::Slot(Kind::In, n) = source else {
            return Err(Refusal::Unstated {
                what: "a raised operand whose source is not one of the statement's inputs",
            });
        };
        let key = o.raised_key(n.into())?;
        match key.as_str() {
            "kv_cache" => {
                let view = Box::new(kv(o)?);
                let at = std::ptr::from_ref::<PagedKvView>(view.as_ref()) as usize;
                self.kv.push(view);
                Ok(ArgValue::Raised(at))
            }
            "recurrent_state" => {
                let view = Box::new(recurrent(o)?);
                let at = std::ptr::from_ref::<RecurrentView>(view.as_ref()) as usize;
                self.rs.push(view);
                Ok(ArgValue::Raised(at))
            }
            "attention_mask" => {
                let view = Box::new(mask(o)?);
                let at = std::ptr::from_ref::<MaskView>(view.as_ref()) as usize;
                self.mask.push(view);
                Ok(ArgValue::Raised(at))
            }
            "attn.split_policy" => {
                let view = Box::new(split(o));
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

/// The handle inside a value the mints above produced.
///
/// Every one of them mints `ArgValue::Buffer`, so this cannot fail for a
/// value they answered; a `Refusal` rather than a panic keeps that a claim
/// this file makes rather than one it assumes.
fn at(v: ArgValue) -> Result<u32, Refusal> {
    use kernels::shader::ShaderValue;
    v.as_buffer().ok_or(Refusal::Unstated {
        what: "a view-field mint that answered with something other than a buffer",
    })
}

/// The paged KV cache, one view per launch. The layer is the RECTANGLE's:
/// [`Handles::kv`] and the layer-carrying `Asked` variants read it off
/// `launch.layers.start` at resolution, so a view cannot disagree with the
/// launch it is built for.
fn kv(o: &mut Handles<'_>) -> Result<PagedKvView, Refusal> {
    let page_size = o.fire_number(FireNumber::KvPageSize);
    // The relocated `contiguous_pool` guard — see the module doc.
    let stride = |o: &Handles<'_>, which: FireNumber| {
        if page_size == 0 { u64::from(o.fire_number(which)) } else { 0 }
    };
    let seq_stride = stride(o, FireNumber::KvSeqStride);
    let head_stride = stride(o, FireNumber::KvHeadStride);
    Ok(PagedKvView {
        keys: Tensor::new(at(o.kv(false))?),
        values: Tensor::new(at(o.kv(true))?),
        page_indices: Tensor::new(at(o.table(FireTable::KvPageIndices))?),
        page_indptr: Tensor::new(at(o.table(FireTable::KvPageIndptr))?),
        write_page: Tensor::new(at(o.table(FireTable::KvWritePage))?),
        write_offset: Tensor::new(at(o.table(FireTable::KvWriteOffset))?),
        page_size: page_size.cast_signed(),
        seq_stride: Usize(seq_stride),
        head_stride: Usize(head_stride),
    })
}

/// The recurrent-state view. [`Handles::slab`] mints a handle whose
/// RESOLUTION refuses by name (`Unplanned::NoSlab`) when this driver holds
/// no pool — the same posture the per-key answering had, moved to the launch
/// that actually binds the field.
fn recurrent(o: &mut Handles<'_>) -> Result<RecurrentView, Refusal> {
    Ok(RecurrentView {
        state: Tensor::new(at(o.slab("recurrent_state"))?),
        slots: Tensor::new(at(o.table(FireTable::RecurrentSlots))?),
        conv_state: Tensor::new(at(o.slab("conv_state"))?),
        new_conv_state: Tensor::new(at(o.slab("new_conv_state"))?),
    })
}

/// The custom-mask triple. This driver stages a real (or zeroed) mask every
/// fire, so the tables always resolve; the pitch is the fire's own number.
fn mask(o: &mut Handles<'_>) -> Result<MaskView, Refusal> {
    Ok(MaskView {
        mask: Tensor::new(at(o.table(FireTable::AttentionMask))?),
        enabled: Tensor::new(at(o.table(FireTable::AttentionMaskEnabled))?),
        stride: o.fire_number(FireNumber::AttentionMaskStride),
    })
}

/// The decode split policy. This driver splits when its pool staged a
/// scratch plane for the fire (the presence test the optional
/// `keys::AttnScratch` ask used to make); the kernel body still holds the
/// occupancy rule, so `splits: 2` here means only "a scratch plane exists".
fn split(o: &mut Handles<'_>) -> SplitView {
    use kernels::shader::ShaderValue;
    let scratch = o.table(FireTable::AttnScratch);
    match scratch.as_buffer() {
        Some(h) => SplitView { partials: Tensor::new(h), splits: 2 },
        None => SplitView { partials: Tensor::new(0), splits: 1 },
    }
}
