//! The per-fire VIEW ARENA: every runtime object this driver answers, built
//! once per fire and addressed by the resolver at bind.
//!
//! The contract is `.wiki/designs/design-no-ask.md` §B7: a routine takes a
//! driver-owned object as an operand (`In<Struct<KvCache>>`), the trace mints
//! a value for it out of the closed vocabulary (`ForwardPlan::runtime`), and
//! the driver answers the NAME with the address of a view struct whose field
//! types `kernels_cuda::views` pins. What `bind/facts.rs` answered as ~40
//! separate keys, one ask at a time, is built HERE as one struct per
//! (fire, layer) and handed over as one address.
//!
//! # Lifetime
//!
//! A [`FireViews`] lives beside `AttnCtx`/`GdnCtx` in the launch path: built
//! after both, dropped after the fire's walk. The vectors are filled once and
//! never grown, so every address handed to the binder stays valid for every
//! bind+launch of the fire.
//!
//! # CUDA graph capture
//!
//! A captured fire is SAFE against this arena for the same reason it is safe
//! against `AttnCtx`: a routine body reads the view on the HOST, at capture
//! time, and bakes field VALUES into its kernel nodes — the view's address
//! never reaches the device. A replay therefore reuses the baked fields, and
//! every field here is copied from `AttnCtx`/`GdnCtx` (pooled, epoch-stable
//! buffers) or from load-resident weight banks, exactly the sources
//! `capture_digest` already guards. A view struct that outlived one fire
//! would still be re-BUILT next fire; the replay never re-reads it.

use core::ffi::c_void;
use std::collections::{BTreeMap, BTreeSet};

use kernels_cuda::views::{ExpertWeightsView, MaskView, PagedKvView, RecurrentView};
use model_compiler::lower::{Arg, Lowered};
use model_ir::trace::{RuntimeBinding, ValueId};

use super::{AttnCtx, DispatchPlan, GdnCtx};

/// The staged per-fire STREAMS, by runtime name — the tensors the driver
/// uploads for the fire being bound. These answer `Resolver::named` for the
/// values the plan's runtime table marks as streams; the device pointers are
/// the same pooled `fire_arrays` slots the fire's arms always read, so a
/// capture that baked one keeps addressing the slot every later fire
/// re-uploads into.
#[derive(Debug, Clone, Copy, Default)]
pub struct FireStreams {
    /// `"positions"` — per-token absolute positions, i32.
    pub positions: *mut c_void,
    /// `"token_ids"` — the fire's token ids, i32.
    pub token_ids: *mut c_void,
    /// `"request_of_token"` — which request each token row belongs to, i32.
    /// Staged only when the plan names it; null otherwise.
    pub request_of_token: *mut c_void,
    /// `"qo_indptr"` — the query-window CSR, device copy.
    pub qo_indptr: *mut c_void,
    /// `"row_valid"` — per-row validity, one byte per row.
    pub row_valid: *mut c_void,
    /// `"sampling_indices"` — the rows a sampling gather collects; null when
    /// the fire samples every row, which is a REFUSAL for a statement that
    /// names it (a gather addressing no indices must not bind null).
    pub sampling_indices: *mut c_void,
    /// `"first_token"` — the fire's write origin. A SCALAR smuggled through
    /// the pointer channel: the swept routines read `first_token.ptr as i32`,
    /// so the answer is the value itself as an address, zero included.
    pub first_token: i32,
    /// `"qo_indptr.host"` — the HOST qo CSR the planless prefill walks; null
    /// outside the planless leg.
    pub qo_indptr_host: *const u32,
    /// `"kv_page_indptr.host"` — its page-CSR sibling.
    pub kv_page_indptr_host: *const u32,
    /// `"fa2.prefill"` AS A RUNTIME OBJECT — the plan CACHE the planless
    /// prefill fills at fire time. Distinct from a prep-published plan value,
    /// which the launch path's own `raised` map answers first.
    pub prefill_plan_cache: *mut c_void,
}

impl FireStreams {
    /// The device pointer for one stream name, or `None` for a name this
    /// struct does not carry or a stream the fire did not stage. `None` is a
    /// refusal at the bind (`UnknownNamed`), never a null argument.
    #[must_use]
    pub fn named(&self, name: &str) -> Option<*mut c_void> {
        let nn = |p: *mut c_void| (!p.is_null()).then_some(p);
        match name {
            "positions" => nn(self.positions),
            "token_ids" => nn(self.token_ids),
            "request_of_token" => nn(self.request_of_token),
            "qo_indptr" => nn(self.qo_indptr),
            "row_valid" => nn(self.row_valid),
            "sampling_indices" => nn(self.sampling_indices),
            // The smuggled scalar: zero is a real origin, so it is answered,
            // not refused — see the field.
            "first_token" => Some(self.first_token as usize as *mut c_void),
            _ => None,
        }
    }
}

/// Every runtime OBJECT this fire can answer, built once (`FireViews::build`)
/// and addressed by the resolver's `raised`.
#[derive(Debug)]
pub struct FireViews {
    /// `"kv_cache"`, one view per model layer. `None` for a layer whose
    /// storage dtype this driver cannot spell — refused at bind rather than
    /// launched over a wrong scheme byte.
    pub kv: Vec<Option<PagedKvView>>,
    /// `"recurrent_state"`, one view per model layer; the slab fields are
    /// null on a layer that is not linear.
    pub recurrent: Vec<RecurrentView>,
    /// `"attention_mask"` — published on every fire (`HasCustomMask` is a
    /// folded predicate): the resident form is the plan's own causal mask.
    pub mask: MaskView,
    /// `"attn.score"` — the observation the driver keeps: the fire's CSR
    /// and the boot-configured window. Nulls/zero when nothing observes.
    pub score: kernels_cuda::views::ScoreView,
    /// `"moe.expert_weights"` — PER STATEMENT, keyed by the trace value: the bank
    /// differs per layer AND per projection (gate_up vs down), so the view is
    /// built from the weight each statement names, not from the fire.
    pub expert_weights: BTreeMap<ValueId, ExpertWeightsView>,
    /// The streams, kept beside the objects so one struct answers both halves.
    pub streams: FireStreams,
}

/// The dtype code `PagedKvView::storage_dtype` carries — the same numbering
/// `kernels_cuda::attn::KvDType` pinned when this was `keys::KvStorageDtype`.
fn kv_dtype_code(d: crate::dtype::DType) -> Option<i32> {
    use crate::dtype::DType as D;
    use kernels_cuda::attn::KvDType;
    Some(match d {
        D::Bf16 => KvDType::Bf16 as i32,
        D::Fp16 => KvDType::Fp16 as i32,
        D::Int8 => KvDType::Int8 as i32,
        D::Fp8E4M3 => KvDType::Fp8E4M3 as i32,
        D::Fp8E5M2 => KvDType::Fp8E5M2 as i32,
        _ => return None,
    })
}

/// The scheme byte, TRANSLATED rather than transmuted — `bind/facts.rs`'s
/// `kv_layer` rule, kept.
fn kv_scheme_code(s: crate::bind::abi::KvCacheScheme) -> i32 {
    use crate::bind::abi::KvCacheScheme as S;
    use kernels_cuda::attn::KvScheme;
    (match s {
        S::Native => KvScheme::Native,
        S::Fp8PerTensor => KvScheme::Fp8PerTensor,
        S::Int8PerTokenHead => KvScheme::Int8PerTokenHead,
        S::Fp8PerTokenHead => KvScheme::Fp8PerTokenHead,
        S::Fp4Block => KvScheme::Fp4Block,
    }) as i32
}

impl FireViews {
    /// Build the arena for one fire. Pure host reads: nothing here touches
    /// the device or allocates device memory.
    #[must_use]
    pub fn build(attn: Option<&AttnCtx>, gdn: Option<&GdnCtx>, streams: FireStreams) -> Self {
        let kv = attn.map_or_else(Vec::new, |a| {
            a.layers.iter().map(|v| kv_view(a, v)).collect()
        });
        let recurrent = gdn.map_or_else(Vec::new, |g| {
            (0..g.conv_state.len().max(g.recurrent_state.len()))
                .map(|l| recurrent_view(g, l))
                .collect()
        });
        let mask = attn.map_or(
            MaskView {
                mask: core::ptr::null(),
                indptr: core::ptr::null(),
                enabled: false,
                stride: 0,
            },
            |a| MaskView {
                mask: a.mask_d,
                indptr: a.mask_indptr_d,
                // Published on every fire; when nothing custom is staged the
                // resident form is the plan's own causal mask, so "enabled"
                // is the pointer's presence — the reading the fa2 arms
                // always used for `keys::AttnMask`'s null-is-an-answer.
                enabled: !a.mask_d.is_null(),
                // CUDA's mask is CSR-shaped (`indptr` locates each request's
                // rows); a row stride is a shader-plane spelling and no CUDA
                // routine reads this field.
                stride: 0,
            },
        );
        let score = attn.map_or(
            kernels_cuda::views::ScoreView {
                indptr: core::ptr::null(),
                window: 0,
            },
            |a| kernels_cuda::views::ScoreView {
                indptr: a.score_indptr_d,
                window: a.score_window,
            },
        );
        Self {
            kv,
            recurrent,
            mask,
            score,
            expert_weights: BTreeMap::new(),
            streams,
        }
    }

    /// Fill [`Self::expert_weights`] from the statements of one lowering.
    ///
    /// Per-STATEMENT, because the bank is: every launch that takes the
    /// `"moe.expert_weights"` object names its own packed bank as its weight, and
    /// the `_ptrs`/`_scales_ptrs`/`_bias_ptrs` arrays were carved BESIDE that
    /// bank at load (`serve::load::build_moe_expert_ptrs`). The map is keyed
    /// by the trace VALUE the statement places.
    ///
    /// Two statements answering ONE value with two different banks is a text
    /// the driver cannot serve — the mint deduplicated what must stay apart —
    /// so the colliding value is REMOVED and both statements refuse at bind
    /// (`RaisedUnbound`), loud rather than smoothly wrong.
    pub fn fill_expert_weights(
        &mut self,
        lowered: &Lowered,
        dplan: &DispatchPlan,
        mut weight: impl FnMut(&str) -> Option<*const c_void>,
    ) {
        use kernels::raises::Raise;
        let mut poisoned: BTreeSet<ValueId> = BTreeSet::new();
        for (i, launch) in lowered.launches.iter().enumerate() {
            let spec = dplan.spec(i);
            let run = launch.args.start as usize..launch.args.end as usize;
            for arg in lowered.args.get(run).unwrap_or(&[]) {
                let Arg::Raised { value, key } = arg else {
                    continue;
                };
                if key != kernels_cuda::views::ExpertWeights::KEY || poisoned.contains(value) {
                    continue;
                }
                let Some(bank) = spec.weight.as_deref() else {
                    continue;
                };
                let mut suffixed = |s: &str| weight(&format!("{bank}{s}"));
                // The two pointer arrays the kernels index per expert are
                // required; the bias array is a checkpoint's to omit.
                let (Some(ptrs), Some(scale_ptrs)) = (suffixed("_ptrs"), suffixed("_scales_ptrs"))
                else {
                    continue;
                };
                let view = ExpertWeightsView {
                    ptrs: ptrs.cast::<u8>(),
                    scale_ptrs: scale_ptrs.cast::<u8>(),
                    bias_ptrs: suffixed("_bias_ptrs").map_or(core::ptr::null(), |p| p.cast::<u8>()),
                };
                match self.expert_weights.get(value) {
                    None => {
                        self.expert_weights.insert(*value, view);
                    }
                    Some(held) if held.ptrs == view.ptrs && held.scale_ptrs == view.scale_ptrs => {}
                    Some(_) => {
                        // The collision: one value, two banks. Refuse both.
                        self.expert_weights.remove(value);
                        poisoned.insert(*value);
                        eprintln!(
                            "[driver-cuda] bind: two statements name one \
                             `expert_weights` value with two different banks; \
                             refusing both (the trace must mint one value per \
                             bank)"
                        );
                    }
                }
            }
        }
    }

    /// One runtime object's address, by the vocabulary name the trace minted
    /// it under. `None` is a refusal at the bind ([`RaisedUnbound`]), never a
    /// null: the names this returns `None` for — `"moe.banks"`,
    /// `"gemm.groups"`, the `dsv4.*` state, `"mtp.pending_hidden"` — are
    /// objects this driver does not STAGE yet (nothing allocates their banks
    /// or slabs), and a refusal that names the key is the honest answer until
    /// something does.
    ///
    /// [`RaisedUnbound`]: super::BindRefusal::RaisedUnbound
    #[must_use]
    pub fn raised(&self, name: &str, layer: Option<u32>, value: ValueId) -> Option<*const c_void> {
        let of = |p: *const c_void| (!p.is_null()).then_some(p);
        match name {
            // Per-layer objects: a mint without a layer is a text error and
            // refuses here rather than defaulting to layer zero.
            "kv_cache" => self
                .kv
                .get(layer? as usize)?
                .as_ref()
                .map(|v| core::ptr::from_ref(v).cast::<c_void>()),
            "recurrent_state" => self
                .recurrent
                .get(layer? as usize)
                .map(|v| core::ptr::from_ref(v).cast::<c_void>()),
            "attention_mask" => Some(core::ptr::from_ref(&self.mask).cast::<c_void>()),
            "attn.score" => Some(core::ptr::from_ref(&self.score).cast::<c_void>()),
            "moe.expert_weights" => self
                .expert_weights
                .get(&value)
                .map(|v| core::ptr::from_ref(v).cast::<c_void>()),
            // The planless prefill's three: the plan cache it fills and the
            // two host CSR mirrors it walks. Null means this fire did not
            // stage the planless leg, which is a refusal for a statement
            // that names it.
            "fa2.prefill" => of(self.streams.prefill_plan_cache.cast_const()),
            "qo_indptr.host" => of(self.streams.qo_indptr_host.cast::<c_void>()),
            "kv_page_indptr.host" => of(self.streams.kv_page_indptr_host.cast::<c_void>()),
            _ => None,
        }
    }
}

/// The set of trace values the plan's runtime table binds, for excluding them
/// from the seam-pin walk: a runtime stream is staged by the driver, so
/// allocating a seam pin for its `Arg::Named` would be resident memory nothing
/// reads.
#[must_use]
pub fn runtime_values(runtime: &[RuntimeBinding]) -> BTreeSet<ValueId> {
    runtime.iter().map(|b| b.value).collect()
}

/// One layer's [`PagedKvView`], from the layer's pool descriptor and the
/// fire-wide CSRs/write descriptors on [`AttnCtx`].
fn kv_view(a: &AttnCtx, v: &crate::bind::abi::KvCacheLayerView) -> Option<PagedKvView> {
    // The two strides, ported from the shader planes' stride tables (the keys
    // `KvSeqStride`/`KvHeadStride`; `driver-metal/src/lowering/resolve.rs`
    // answered `head_dim` and `kv_heads * head_dim` for the NHD layout it
    // owns). In elements, over the page's own axes:
    //   NHD: a page is `[page_size, num_kv_heads, head_dim]` — one token step
    //        crosses every head (`kv_heads * head_dim`), one head is
    //        `head_dim` wide.
    //   HND: a page is `[num_kv_heads, page_size, head_dim]` — one token step
    //        is `head_dim`, one head spans the page (`page_size * head_dim`).
    let (seq_stride, head_stride) = if v.hnd_layout {
        (
            i64::from(v.head_dim),
            i64::from(v.page_size) * i64::from(v.head_dim),
        )
    } else {
        (
            i64::from(v.num_kv_heads) * i64::from(v.head_dim),
            i64::from(v.head_dim),
        )
    };
    Some(PagedKvView {
        keys: v.k_pages.cast::<u8>(),
        values: v.v_pages.cast::<u8>(),
        bf16_keys: v.k_bf16_pages.cast::<u8>(),
        bf16_values: v.v_bf16_pages.cast::<u8>(),
        page_indices: a.kv_page_indices_d.cast::<i32>(),
        page_indptr: a.kv_page_indptr_d.cast::<i32>(),
        last_page_lens: a.kv_last_page_lens_d.cast::<i32>(),
        key_scales: v.k_scales.cast_const(),
        value_scales: v.v_scales.cast_const(),
        // The write half is NULLABLE BY CONTRACT (`KvWritePageOrNull`'s
        // spelling folded into the view): a fire that appends nothing
        // carries nulls and the appending kernels test for them.
        write_page: a.w_page_d.cast::<i32>(),
        write_offset: a.w_off_d.cast::<i32>(),
        page_size: v.page_size,
        seq_stride,
        head_stride,
        layout: i32::from(v.hnd_layout),
        storage_dtype: kv_dtype_code(v.storage_dtype)?,
        scheme_byte: kv_scheme_code(v.scheme),
        native_bf16: v.is_native_bf16(),
        has_envelopes: v.has_envelopes(),
        env_min: v.k_env_min.cast_const(),
        env_max: v.k_env_max.cast_const(),
        block_size: v.block_size,
        max_pages_per_request: a.max_pages_per_request,
        pages_in_batch: a.num_pages_in_batch,
    })
}

/// One layer's [`RecurrentView`], from the fire's [`GdnCtx`]. A layer that is
/// not linear gets null slabs — its view exists so the vector stays
/// layer-indexed, and no statement of a well-formed text names it.
fn recurrent_view(g: &GdnCtx, layer: usize) -> RecurrentView {
    let base = |v: &[u64]| -> *mut c_void {
        match v.get(layer) {
            Some(&b) if b != 0 => b as *mut c_void,
            _ => core::ptr::null_mut(),
        }
    };
    let slab = base(&g.recurrent_state);
    let conv = base(&g.conv_state);
    RecurrentView {
        slab,
        slot_ids: g.slot_ids_d,
        slot_stride_elems: g.state_stride_elems,
        // `RecurrentSlots` was the same request→slot table under the mamba
        // spelling; one indirection serves both.
        slots: g.slot_ids_d,
        // The `state`/`conv_state`/`new_conv_state` triple is the shader
        // planes' double-buffered spelling. CUDA's routines read
        // `slab`/`conv_slab` and update in place, so the state half aliases
        // the slab and the swap plane stays null until a CUDA routine wants
        // one — a null a routine must refuse, never a silently-wrong alias.
        state: slab,
        conv_state: conv,
        new_conv_state: core::ptr::null_mut(),
        conv_slab: conv,
        conv_stride: g.conv_stride_elems,
    }
}

/// Every runtime name this driver ANSWERS — streams through
/// [`FireStreams::named`], objects through [`FireViews::raised`], the fa2
/// preps through the resolver's raise map. The rebirth of
/// `every_plane_is_answered`: a test walks every catalogued SKU's
/// `plan.runtime` against this list, so a text minting a name nothing
/// answers fails the build, not the fire.
pub const ANSWERED: &[&str] = &[
    // streams
    "positions",
    "token_ids",
    "request_of_token",
    "qo_indptr",
    "row_valid",
    "sampling_indices",
    "first_token",
    // objects
    "kv_cache",
    "recurrent_state",
    "attention_mask",
    "attn.score",
    "moe.expert_weights",
    "fa2.prefill",
    "fa2.decode",
    "qo_indptr.host",
    "kv_page_indptr.host",
];

/// Runtime names this driver KNOWS but deliberately refuses until their
/// staging owners land (`RaisedUnbound`, by name): the moe/gemm pointer
/// banks and the dsv4/mtp slabs nothing in driver-cuda allocates today.
pub const UNSTAGED: &[&str] = &[
    "moe.banks",
    "gemm.groups",
    "dsv4.state_kv",
    "dsv4.state_score",
    "dsv4.ape",
    "dsv4.comp_kv_pages",
    "mtp.pending_hidden",
];
