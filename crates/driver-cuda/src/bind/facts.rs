//! The driver's answer to every fact a bind arm can ask for.
//!
//! Absence is spelled `Option`, never a type's own idea of empty: a `None`
//! becomes `Refusal::Unstated` naming the fact. A stub is NOT caught at load
//! — `unfireable` reads the symbol table alone, so `arm: Some(f)` is reported
//! bound whatever `f` does. A symbol that cannot fire is `arm: None`.

use core::ffi::c_void;
use core::ptr::NonNull;

use kernels_cuda::attn::{
    AttnWorkspace, KvDType, KvLayer, KvScheme, MlaLayer, MlaPlan, Plan, Rows,
};
use kernels_cuda::rope::Yarn;
use kernels_cuda::ssm::{Gdn, Slab};

use super::{AttnCtx, BoundLaunch, DispatchCtx, GdnCtx, LaunchSpec};

/// One launch's whole world, as a bind body may read it.
pub struct Fire<'a> {
    /// The launch, with every operand resolved.
    pub bound: &'a BoundLaunch<'a>,
    /// The op join: the operand split, the statement's params, the weight.
    pub spec: &'a LaunchSpec,
    /// The fire's model-wide facts.
    pub ctx: &'a DispatchCtx,
    /// The attention half of the fire, when it has one.
    pub attn: Option<&'a AttnCtx>,
    /// The gated-delta-net half, when it has one.
    pub gdn: Option<&'a GdnCtx>,
    /// This region's row count, already narrowed.
    pub rows: i32,
    /// The named weight, resolved ONCE by the caller: `Facts` is not `&mut`.
    pub w_named: *const c_void,
    /// The second named weight, resolved the same way.
    pub w_named2: *const c_void,
    /// Weights the statement names by SUFFIX. A slice: `quant` reaches three.
    pub w_suffixed: &'a [(&'static str, *const c_void)],
}

impl Fire<'_> {
    /// This launch's layer, as an index.
    pub fn layer_index(&self) -> usize {
        usize::from(self.bound.layers.start)
    }

    /// The `i`th arg of the run, by absolute index.
    pub fn arg(&self, i: usize) -> Option<*mut c_void> {
        self.bound.args.get(i).map(|a| a.ptr)
    }

    /// The `i`th arg's row width. `None` for zero, which a WEIGHT carries.
    pub fn width(&self, i: usize) -> Option<i32> {
        let w = self.bound.args.get(i)?.width;
        i32::try_from(w).ok().filter(|w| *w > 0)
    }


    /// Which AltUp stream ran through the real layer. `None` below two.
    pub fn altup_active(&self) -> Option<i32> {
        (self.ctx.altup_streams > 1).then_some(self.ctx.altup_active)
    }

    /// A per-layer constant the model names, keyed by that name.
    pub fn named_scale(&self, name: &str) -> Option<f32> {
        self.ctx.scales.get(name).copied()
    }

    /// The `i`th auxiliary buffer. Always `None`, and not for want of a
    /// borrow: the map `aux_of` reads is never inserted into.
    pub const fn aux(&self, _i: usize) -> Option<*mut c_void> {
        None
    }

    /// This fire's linear-attention shape and state addressing.
    pub fn gdn(&self) -> Option<Gdn> {
        let g = self.gdn?;
        Some(Gdn {
            k_h: g.k_h,
            v_h: g.v_h,
            k_d: g.k_d,
            v_d: g.v_d,
            conv_dim: g.conv_dim,
            conv_k: g.conv_k,
            n_groups: g.n_groups,
            conv_stride_elems: g.conv_stride_elems,
            state_stride_elems: g.state_stride_elems,
            slot_ids_d: g.slot_ids_d,
            write_state: g.write_state,
        })
    }

    /// This layer's latent cache — `kv_layer`'s MLA sibling. VACUOUS.
    pub const fn mla_layer(&self) -> Option<MlaLayer> {
        None
    }

    /// The plan `Prepare::MlaPlan` built for this fire. VACUOUS, as above.
    pub const fn mla_plan(&self) -> Option<MlaPlan> {
        None
    }

    /// The `i`th result placement. Always `None`, deliberately: `walk.rs`
    /// puts the placement at `args[n_in]`, so [`Fire::arg_out`] answers it.
    pub const fn result(&self, _i: usize) -> Option<*mut c_void> {
        None
    }

    /// The statement's bias weight, when it carries one — `weight2`. NOT
    /// `keys::WeightBias`, the `_bias` SUFFIX: same type, no error if swapped.
    pub fn weight_bias(&self) -> Option<*mut c_void> {
        self.weight_named(1)
    }

    /// The WNA16 weight's group size, in elements along K. Always `None`:
    /// no `QuantMeta` is built anywhere, so the driver does not have it.
    pub const fn wna16_group_size(&self) -> Option<i32> {
        None
    }
}

impl Fire<'_> {
    pub(super) fn arg_in(&self, i: usize) -> Option<*mut c_void> {
        if i >= self.spec.n_in {
            return None;
        }
        self.arg(i)
    }

    pub(super) fn arg_out(&self, i: usize) -> Option<*mut c_void> {
        if i >= self.spec.n_out {
            return None;
        }
        self.arg(self.spec.n_in + i)
    }

    pub(super) fn weight(&self, i: usize) -> Option<*mut c_void> {
        self.arg(self.spec.n_in + self.spec.n_out + i)
    }

    /// Whether the router renormalises its top-k weights. Always `Some`.
    pub fn moe_norm_topk(&self) -> Option<bool> {
        Some(self.ctx.moe_norm_topk)
    }

    /// The router's routed scaling factor. [`Fire::moe_norm_topk`]'s pair.
    pub fn moe_routed_scaling(&self) -> Option<f32> {
        Some(self.ctx.moe_routed_scaling)
    }

    /// How many experts one token visits. `0` is filtered: it routes nothing.
    pub fn experts_per_token(&self) -> Option<i32> {
        (self.ctx.experts_per_token > 0).then_some(self.ctx.experts_per_token)
    }

    /// The `i`th input's row count — `self.rows`, with the index checked.
    pub fn in_rows(&self, i: usize) -> Option<i32> {
        (i < self.spec.n_in).then_some(self.rows)
    }

    /// The attention workspace this fire was given — the DECODE carve.
    pub fn attn_workspace(&self) -> Option<AttnWorkspace> {
        let w = self.attn?.workspace;
        Some(AttnWorkspace {
            float_buffer: w.float_buffer,
            float_bytes: w.float_bytes,
            int_buffer: w.int_buffer,
            int_bytes: w.int_bytes,
        })
    }

    /// The PREFILL carve. A second accessor and not a parameter: a prefill
    /// plan handed the DECODE carve overwrites what that plan staged.
    pub fn attn_prefill_workspace(&self) -> Option<AttnWorkspace> {
        let w = self.attn?.prefill_workspace;
        Some(AttnWorkspace {
            float_buffer: w.float_buffer,
            float_bytes: w.float_bytes,
            int_buffer: w.int_buffer,
            int_bytes: w.int_bytes,
        })
    }

    /// The softmax scale this fire was planned with.
    pub fn sm_scale(&self) -> Option<f32> {
        Some(self.attn?.sm_scale)
    }

    /// `write_kv_to_pages`' first-token scalar.
    pub fn first_token(&self) -> Option<i32> {
        Some(self.attn?.first_token)
    }

    /// The pages this fire's CSR names.
    pub fn num_pages_in_batch(&self) -> Option<i32> {
        Some(self.attn?.num_pages_in_batch)
    }

    /// The widest single request's page count — XQA's page-table row stride.
    pub fn max_pages_per_request(&self) -> Option<i32> {
        let n = self.attn?.max_pages_per_request;
        (n > 0).then_some(n)
    }

    /// Per-row target page for this fire's KV append. Null is absence and
    /// not a value: a fire that appends no KV carries one.
    pub fn w_page_d(&self) -> Option<*const u32> {
        let p = self.attn?.w_page_d;
        (!p.is_null()).then_some(p)
    }

    /// Per-row offset-in-page for the append. [`Fire::w_page_d`]'s pair.
    pub fn w_off_d(&self) -> Option<*const u32> {
        let p = self.attn?.w_off_d;
        (!p.is_null()).then_some(p)
    }

    /// The destination the fused QKV kernel writes Q into. Null-checked.
    pub fn q_out(&self) -> Option<*mut c_void> {
        let p = self.attn?.q_out;
        (!p.is_null()).then_some(p)
    }

    /// The sliding-window span this fire attends — `super::window_of`.
    pub fn window_left(&self) -> Option<i32> {
        Some(super::window_of(
            self.spec,
            self.attn?,
            u32::try_from(self.layer_index()).unwrap_or(0),
        ))
    }

    /// The attention logit soft cap.
    pub fn logits_soft_cap(&self) -> Option<f32> {
        Some(self.attn?.logits_soft_cap)
    }

    /// The LSE scratch the decode dispatch writes. NOT null-checked, unlike
    /// `w_page_d`: returning `None` would invent a state nothing produces.
    pub fn lse_out(&self) -> Option<*mut f32> {
        Some(self.attn?.lse_out_d)
    }

    /// gpt-oss's clamped-GLU ceiling. Always `Some`, a deployment default.
    pub fn glu_limit(&self) -> Option<f32> {
        Some(self.ctx.glu_limit)
    }

    /// The clamped GLU's alpha, carried the same way.
    pub fn glu_alpha(&self) -> Option<f32> {
        Some(self.ctx.glu_alpha)
    }

    /// A weight the statement names by suffix. A linear scan of at most three.
    pub fn weight_suffixed(&self, suffix: &str) -> Option<*mut c_void> {
        self.w_suffixed
            .iter()
            .find(|(s, _)| *s == suffix)
            .map(|(_, p)| p.cast_mut())
            .filter(|p| !p.is_null())
    }

    /// The weights a statement names by NAME: `0` is `spec.weight`, `1` is
    /// `spec.weight2`. Null is absence, and which kind is the caller's job.
    pub fn weight_named(&self, i: usize) -> Option<*mut c_void> {
        let p = match i {
            0 => self.w_named,
            1 => self.w_named2,
            _ => return None,
        };
        NonNull::new(p.cast_mut()).map(NonNull::as_ptr)
    }

    pub(super) fn in_width(&self, i: usize) -> Option<i32> {
        if i >= self.spec.n_in {
            return None;
        }
        self.width(i)
    }

    pub(super) fn out_width(&self, i: usize) -> Option<i32> {
        if i >= self.spec.n_out {
            return None;
        }
        self.width(self.spec.n_in + i)
    }

    /// The region, and the lane space it sits in: `total` is the fire's rows.
    pub fn rows(&self) -> Rows {
        Rows {
            start: i32::try_from(self.bound.rows.start).unwrap_or(0),
            count: self.rows,
            total: self.ctx.rows_total,
        }
    }

    pub(super) fn layer(&self) -> usize {
        self.layer_index()
    }

    /// `OpKind::Launch::params` — the wire scalars the statement carries.
    pub fn param(&self, i: usize) -> Option<u32> {
        self.spec.params.get(i).copied()
    }

    pub(super) fn positions(&self) -> Option<*const i32> {
        NonNull::new(self.ctx.positions).map(|p| p.as_ptr().cast_const().cast::<i32>())
    }

    /// The fire's token ids. Null is absence, `positions`' convention.
    pub fn token_ids(&self) -> Option<*const i32> {
        NonNull::new(self.ctx.token_ids).map(|p| p.as_ptr().cast_const().cast::<i32>())
    }

    /// How many tokens the vocabulary holds. Zero is absence, not a width.
    pub fn vocab(&self) -> Option<i32> {
        (self.ctx.vocab > 0).then_some(self.ctx.vocab)
    }

    /// The rows a sampling gather collects, or `None` for every row.
    pub fn sampling_indices(&self) -> Option<*const i32> {
        NonNull::new(self.ctx.sampling_indices.cast_mut()).map(|p| p.as_ptr().cast_const())
    }

    /// The per-layer-embedding width. Zero is absence — the layer count divides.
    pub fn ple_dim(&self) -> Option<i32> {
        (self.ctx.ple_dim > 0).then_some(self.ctx.ple_dim)
    }

    /// The fire's peel window, `[start, count]`, device-resident.
    pub fn peel_window(&self) -> Option<NonNull<u32>> {
        NonNull::new(self.ctx.peel_window.cast_mut())
    }

    /// Elements per attention head. Zero is absence and not a width.
    pub fn head_dim(&self) -> Option<i32> {
        (self.ctx.head_dim > 0).then_some(self.ctx.head_dim)
    }

    pub(super) fn num_q_heads(&self) -> Option<i32> {
        (self.ctx.num_q_heads > 0).then_some(self.ctx.num_q_heads)
    }

    pub(super) fn num_kv_heads(&self) -> Option<i32> {
        (self.ctx.num_kv_heads > 0).then_some(self.ctx.num_kv_heads)
    }

    /// The rotary base, the fire-wide one. Zero is absence.
    pub fn rope_theta(&self) -> Option<f32> {
        (self.ctx.rope_theta > 0.0).then_some(self.ctx.rope_theta)
    }

    /// The rotary base for THIS statement's layer. gemma-4 splits it by kind.
    pub fn theta(&self) -> Option<f32> {
        let t = self.ctx.theta(self.layer_index());
        (t > 0.0).then_some(t)
    }

    pub(super) fn rms_eps(&self) -> Option<f32> {
        (self.ctx.eps > 0.0).then_some(self.ctx.eps)
    }

    /// The logit soft cap. Zero is absence, not a cap of zero.
    pub fn final_logit_softcap(&self) -> Option<f32> {
        (self.ctx.final_logit_softcap > 0.0).then_some(self.ctx.final_logit_softcap)
    }

    /// The engine's cuBLAS handle, with THIS fire's stream already bound.
    pub fn cublas(&self) -> *mut c_void {
        self.ctx.cublas
    }
    /// The statement's per-head width — what tells `RmsnormPerHead` from plain.
    pub fn per_head_dim(&self) -> Option<i32> {
        self.spec.per_head_dim.map(|d| i32::try_from(d).unwrap_or(0))
    }

    /// GPT-J adjacent-pair rotation, vs NeoX half/half. `false` means NeoX.
    pub fn rope_interleaved(&self) -> bool {
        self.ctx.rope_interleaved
    }

    /// How many channels rotate, for a statement that does NOT state it.
    pub fn rotary_width(&self) -> Option<i32> {
        self.spec
            .rope_partial
            .or_else(|| {
                self.ctx.rotary_by_layer.get(self.layer_index()).copied().filter(|r| *r > 0)
            })
            .and_then(|r| i32::try_from(r).ok())
    }

    /// The checkpoint's YaRN quartet. `None` when the deployment states none.
    pub fn yarn(&self) -> Option<Yarn> {
        let [factor, beta_fast, beta_slow, attention_factor] = self.ctx.yarn;
        (self.ctx.yarn_original_max > 0).then_some(Yarn {
            factor,
            beta_fast,
            beta_slow,
            attention_factor,
            original_max_position: self.ctx.yarn_original_max,
        })
    }

    /// This layer's paged KV cache.
    pub fn kv_layer(&self) -> Option<KvLayer> {
        use crate::bind::abi::KvCacheScheme as S;
        use crate::dtype::DType as D;
        let v = self.attn?.layers.get(self.layer_index())?;
        Some(KvLayer {
            k_pages: v.k_pages,
            v_pages: v.v_pages,
            page_size: v.page_size,
            head_dim: v.head_dim,
            num_kv_heads: v.num_kv_heads,
            hnd: v.hnd_layout,
            // TRANSLATED rather than transmuted, which is what the `_` arms say.
            scheme: match v.scheme {
                S::Native => KvScheme::Native,
                S::Fp8PerTensor => KvScheme::Fp8PerTensor,
                S::Int8PerTokenHead => KvScheme::Int8PerTokenHead,
                S::Fp8PerTokenHead => KvScheme::Fp8PerTokenHead,
                S::Fp4Block => KvScheme::Fp4Block,
            },
            storage_dtype: match v.storage_dtype {
                D::Bf16 => KvDType::Bf16,
                D::Fp16 => KvDType::Fp16,
                D::Int8 => KvDType::Int8,
                D::Fp8E4M3 => KvDType::Fp8E4M3,
                D::Fp8E5M2 => KvDType::Fp8E5M2,
                _ => return None,
            },
            block_size: v.block_size,
            num_pages: v.num_pages,
            k_scales: v.k_scales,
            v_scales: v.v_scales,
            k_bf16_pages: v.k_bf16_pages,
            v_bf16_pages: v.v_bf16_pages,
            k_env_min: v.k_env_min,
            k_env_max: v.k_env_max,
            has_envelopes: v.has_envelopes(),
            is_native_bf16: v.is_native_bf16(),
        })
    }

    /// The fire's per-request plan arrays. `requests: 0` is not an absence.
    pub fn plan(&self) -> Option<Plan> {
        let a = self.attn?;
        Some(Plan {
            qo_indptr: a.qo_indptr_d,
            kv_page_indices: a.kv_page_indices_d,
            kv_page_indptr: a.kv_page_indptr_d,
            kv_last_page_lens: a.kv_last_page_lens_d,
            row_valid: a.row_valid_d,
            requests: a.num_requests,
        })
    }

    /// One of a gated-delta-net layer's two state slabs, by enum and not name.
    pub fn slab(&self, which: Slab) -> Option<*mut c_void> {
        let g = self.gdn?;
        let layer = self.spec.state.as_ref()?.layer as usize;
        let v: &[u64] = match which {
            Slab::Conv => &g.conv_state,
            Slab::Recurrent => &g.recurrent_state,
        };
        match v.get(layer) {
            Some(&base) if base != 0 => Some(base as *mut c_void),
            _ => None,
        }
    }
}
