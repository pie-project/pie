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

use kernels_cuda::views::{MaskView, PagedKvView, RecurrentView};

use super::{AttnCtx, GdnCtx};

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
    /// `"fa2.decode"` — the decode SCHEDULE `raise_attn_plans` raised for
    /// this fire: workspaces stamped inside the plan-update fence, the
    /// variant the lane stated, `window_left = -1`.
    ///
    /// It reached the walk as a loose argument until the fa2 points became
    /// claim bodies, because the walk's one reader was a hand-written arm
    /// that could be handed anything. A body has only the key.
    pub decode_plan_cache: *mut c_void,
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
    // `expert_weights` STOOD HERE — one `ExpertWeightsView` per trace VALUE,
    // filled by walking the legacy lowering's launches for the ones that
    // raised `"moe.expert_weights"` and reading each one's packed bank off
    // its `LaunchSpec`. There is no launch list to walk, and the values it
    // was keyed by were the legacy trace's.
    //
    // `"moe.expert_weights"` IS THEREFORE A NAMED REFUSAL, not a deletion:
    // the kernels still take the view, and the one thing missing is the join
    // from a STATEMENT to its bank — which a `Program`'s statement carries in
    // `Op::weights` and nothing has read yet. (`fire::moe_ptrs`, which carved
    // the `_ptrs`/`_scales_ptrs`/`_bias_ptrs` arrays at load, is gone with the
    // rest of the legacy MoE leg.) The MoE SKUs are not
    // servable through this driver today for other reasons as well (a3b and
    // dsv4 both refuse lanes), so a refusal by name is the honest answer
    // until the first of them resolves.
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
            streams,
        }
    }

    /// One runtime object's address, by the vocabulary name the trace minted
    /// it under.
    ///
    /// # This is the door a `#[claims]` body opens
    ///
    /// `bind::resolve_arg` used to call it, on every `Arg::Raised` of every
    /// lowered launch; R2 deleted that walk and left this function with no
    /// caller, naming the generated dispatch as its target. THAT IS NOW THE
    /// CALLER, through one indirection: a raise is declared by KEY
    /// (`kernels::raises::Struct<KvCache>`), a claim body asks its `Ctx` for
    /// the key (`Ctx::raised::<Fa2Decode>()`), and `impl Answered for
    /// FireViews` below hands the ask straight here with no layer.
    ///
    /// NO LAYER, AND THAT IS THE SPLIT. The two per-layer pools are the ones
    /// a STATEMENT names (`Op::cache`), so they arrive through the `Cache`
    /// mark and `BoundOp::pages`/`recurrent` — `baker::fire::Fire`'s own
    /// indexing of [`Self::kv`] and [`Self::recurrent`]. What comes through
    /// the key door is exactly what no statement names: the fire's
    /// schedules, its host CSR mirrors, its mask, its streams. A key asked
    /// with no layer against a per-layer object therefore refuses here
    /// rather than defaulting to layer zero.
    ///
    /// [`ANSWERED`] is the vocabulary this must cover. `None` is a refusal at
    /// the bind ([`RaisedUnbound`]), never a null: the names this returns
    /// `None` for — `"moe.banks"`, `"gemm.groups"`, the `dsv4.*` state,
    /// `"mtp.pending_hidden"` — are objects this driver does not STAGE yet
    /// (nothing allocates their banks or slabs), and a refusal that names the
    /// key is the honest answer until something does. The one key whose null
    /// is an ANSWER is `"row_valid"`, and the CALLER says so: a body asks for
    /// it through `Ctx::staged`, which reads `None` as the null every
    /// appending kernel tests for.
    ///
    /// [`RaisedUnbound`]: super::BindRefusal::RaisedUnbound
    #[must_use]
    pub fn raised(&self, name: &str, layer: Option<u32>) -> Option<*const c_void> {
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
            // See the note where `expert_weights` stood: the object exists,
            // its banks are carved at load, and the statement->bank join is
            // not written. `None` is a refusal at the bind, by name.
            "moe.expert_weights" => None,
            // The planless prefill's three: the plan cache it fills and the
            // two host CSR mirrors it walks. Null means this fire did not
            // stage the planless leg, which is a refusal for a statement
            // that names it.
            "fa2.prefill" => of(self.streams.prefill_plan_cache.cast_const()),
            "qo_indptr.host" => of(self.streams.qo_indptr_host.cast::<c_void>()),
            "kv_page_indptr.host" => of(self.streams.kv_page_indptr_host.cast::<c_void>()),
            // THE DECODE SCHEDULE, ANSWERED BY KEY. It was withheld here for
            // one measured reason, and that reason retired with the legacy
            // walk: there used to be TWO decode schedules whenever a stack
            // kept one per layer kind, and picking between them needed the
            // window on the statement's `LaunchSpec`, which a key does not
            // carry. A `Program` is ONE LANE and `Baked::attn_ask` raises the
            // schedule that lane's statements ask for — refusing the lane
            // outright when two of them ask for different ones ("states two
            // decode attention schedules and this driver raises one"). So
            // there is exactly one per fire, this key names it, and the
            // ambiguity that made a one-valued answer dangerous is a
            // load-time refusal instead.
            "fa2.decode" => of(self.streams.decode_plan_cache.cast_const()),
            // THE STREAMS, THROUGH THE SAME DOOR. A stream is a raise whose
            // payload is a plane rather than a struct (`kernels_cuda::views`
            // declares `RowValid` and `RequestOfToken` beside the view
            // structs), and a body that needs one needs it for the reason a
            // body needs a schedule: no statement carries it. The SHAPE is
            // not answered and is not asked for — every kernel that reads one
            // of these indexes it by the row it is already on.
            "row_valid" | "request_of_token" | "positions" | "token_ids" | "qo_indptr" => {
                self.streams.named(name).map(|p| p.cast_const())
            }
            _ => None,
        }
    }
}

/// The by-key door, as the floor spells it: `Ctx::raised::<R>()` asks for
/// `R::KEY` and this is what answers.
///
/// NO LAYER CROSSES IT, for [`FireViews::raised`]'s reason — the per-layer
/// pools ride the `Cache` mark, which a statement names and `BoundOp`
/// resolves. Everything a claim body pulls off `self` is fire-wide.
impl kernels::raises::Answered for FireViews {
    fn raised(&self, key: &'static str) -> Option<*const c_void> {
        Self::raised(self, key, None)
    }
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
        // The fire's query CSR and row validity, carried on the pool row so
        // a `#[claims]` body that names ONE cache row can resolve an
        // append's destination without staging on the operand column's
        // behalf. Both are the same `AttnCtx` fields the appending routines
        // take as loose operands; `row_valid_d` is null on a fire with no
        // rejected rows, which is what the kernels test for.
        qo_indptr: a.qo_indptr_d.cast::<i32>(),
        row_valid: a.row_valid_d,
        requests: a.num_requests,
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
/// [`FireStreams::named`], objects through [`FireViews::raised`], both
/// reachable from a claim body by key through `impl Answered`.
///
/// THE CENSUS AND NOT A GATE, since R2: the test that walked every catalogued
/// SKU's `plan.runtime` against this list died with `plan.runtime`, and a
/// baker `Plan` mints no runtime names at all — a body asks for what it needs
/// by the key its `Raise` declares, and an unanswered key is a refusal with
/// the key in it. What the list is for now is reading: one place that says
/// what a fire of this driver can be asked for.
///
/// `"fa2.decode"` IS ON IT AGAIN. It was struck when the legacy walk could
/// keep two decode schedules and a key could not say which; `Baked::attn_ask`
/// refuses a lane that asks for two, so one fire has one, and
/// [`FireViews::raised`] says so at the arm.
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
    "fa2.prefill",
    "fa2.decode",
    "qo_indptr.host",
    "kv_page_indptr.host",
];

/// Runtime names this driver KNOWS but deliberately refuses until their
/// staging owners land (`RaisedUnbound`, by name): the moe/gemm pointer
/// banks and the dsv4/mtp slabs nothing in driver-cuda allocates today.
pub const UNSTAGED: &[&str] = &[
    // See `FireViews::raised`: the banks are carved, the join is not.
    "moe.expert_weights",
    "moe.banks",
    "gemm.groups",
    "dsv4.state_kv",
    "dsv4.state_score",
    "dsv4.ape",
    "dsv4.comp_kv_pages",
    "mtp.pending_hidden",
];
