//! The driver's answer to [`kernels_cuda_new::x::Facts`].
//!
//! # What this is
//!
//! `.wiki/kernel-x/northstar.md` §3.3 asks for a query-only context a bind
//! body reads its facts out of, and says the thing plainly: *"this is a
//! promotion, not an invention"*. The API already existed — it was
//! [`dispatch_generated`](super::dispatch_generated)'s scaffolding
//! (`width_of`, `rotary_width`, `attn_plan`, `kv_view`, `is_set`) reading
//! [`DispatchCtx`], [`AttnCtx`] and [`GdnCtx`], and it was reachable only
//! from inside a generated file. This module is the same reads, named, with
//! a public caller.
//!
//! # Why the trait lives in `kernels-cuda-new` and the impl lives here
//!
//! A bind body lives in `kernels-cuda-new/src/x/`, holding the `.cuh` it
//! fires by `include_str!` (northstar §5.1 ①). The facts are the fire's,
//! which is here.
//! `driver-cuda` depends on `kernels-cuda-new` and not the other way round,
//! so the vocabulary is declared there as a trait and answered here — the
//! only direction that is not a cycle. `Cx` holds `&dyn Facts`, so a bind
//! never names a driver type and the driver never names a bind.
//!
//! # Why every method is a read
//!
//! `Fire<'_>` borrows everything it answers from, shared. There is no
//! device API on it, no allocator, no stream — the stream is passed to
//! `Entry::call` beside the `Cx` and not through it — and no `&mut`
//! anywhere. That is §3.3's safety argument stated as a type: a bind body
//! has no surface to misbehave on, so the unsafe block it eventually writes
//! is one launch and nothing else.
//!
//! # The one thing that is NOT a promotion
//!
//! `is_set` is gone. The scaffolding's `IsSet` trait answered "did the fire
//! state this?" for five different types by comparing against each type's
//! own idea of empty (`0`, `0.0`, null, `false`), and a generated guard
//! called it. Every method here answers `Option` instead, so the same
//! question is the language's and a `None` becomes a
//! [`Refusal::Unstated`](kernels_cuda_new::x::Refusal) that names the fact.
//! The zero-is-absence conventions did not go anywhere — they are written
//! into the bodies below, each beside the field it decides.

#![cfg(feature = "bridge")]

use core::ffi::c_void;
use core::ptr::NonNull;

use kernels_cuda_new::x::{Facts, KvDType, KvLayer, KvScheme, Plan, Rows, Slab, Yarn};

use super::{AttnCtx, BoundLaunch, DispatchCtx, GdnCtx, LaunchSpec};

/// One launch's whole world, as a bind body may read it.
///
/// Assembled at the dispatch site out of the four things
/// [`dispatch`](super::dispatch) already holds, and borrowed from all of
/// them. Nothing is copied except the scalars.
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
    /// The named weight, resolved once by the caller.
    ///
    /// A [`Resolver`](super::Resolver) is `&mut` and `Facts` is not, which
    /// is deliberate: resolving a name can consult a store, and a bind body
    /// that could do that could also make the store answer differently the
    /// second time. So the resolve happens once, at the dispatch site,
    /// before the `Cx` exists.
    pub w_named: *const c_void,
    /// The second named weight, resolved the same way.
    pub w_named2: *const c_void,
    /// Weights the statement names by SUFFIX, resolved once by the caller.
    ///
    /// `w_named`'s reason, for a set that is not two: `quant`'s routed MXFP4
    /// pair reaches `_scales`, `_gate_bias` and `_up_bias` on one fire, and
    /// `weight_names.rs:505` records that the trace states none of them. So
    /// this is a slice rather than two more fields — the arity is the
    /// statement's, not the struct's.
    ///
    /// Empty for every fire that names no suffix, which is most of them.
    pub w_suffixed: &'a [(&'static str, *const c_void)],
}

impl Fire<'_> {
    /// This launch's layer, as an index.
    fn layer_index(&self) -> usize {
        usize::from(self.bound.layers.start)
    }

    /// The `i`th arg of the run, by absolute index.
    fn arg(&self, i: usize) -> Option<*mut c_void> {
        self.bound.args.get(i).map(|a| a.ptr)
    }

    /// The `i`th arg's row width.
    ///
    /// `None` for a width of zero, which is what a WEIGHT operand carries
    /// (`BoundArg::width` is "zero for a weight, whose extent is the
    /// tensor's") and also what a launch that states a three-dimensional
    /// value carries. Both are "nothing stated a row width here", which is
    /// the sentence `Option` spells and the sentence a `0` did not.
    fn width(&self, i: usize) -> Option<i32> {
        let w = self.bound.args.get(i)?.width;
        i32::try_from(w).ok().filter(|w| *w > 0)
    }
}

impl Facts for Fire<'_> {
    fn arg_in(&self, i: usize) -> Option<*mut c_void> {
        if i >= self.spec.n_in {
            return None;
        }
        self.arg(i)
    }

    fn arg_out(&self, i: usize) -> Option<*mut c_void> {
        if i >= self.spec.n_out {
            return None;
        }
        self.arg(self.spec.n_in + i)
    }

    fn weight(&self, i: usize) -> Option<*mut c_void> {
        self.arg(self.spec.n_in + self.spec.n_out + i)
    }

    /// Whether the router renormalises its top-k weights.
    ///
    /// `DispatchCtx` has carried this since `bind/mod.rs:1179`, filled from
    /// `model.deployment.norm_topk_prob` at `fire/launch.rs:3435`. The row
    /// world reached it as `Source::Ctx("moe_norm_topk")` and the generated
    /// dispatcher rendered it `ctx.moe_norm_topk`; this is the same field
    /// reached the fn-world way.
    ///
    /// Always `Some`: the field is a `bool` with a deployment default, not an
    /// optional fact, so a refusal here would be inventing an absence.
    fn moe_norm_topk(&self) -> Option<bool> {
        Some(self.ctx.moe_norm_topk)
    }

    /// The router's routed scaling factor.
    ///
    /// [`Facts::moe_norm_topk`]'s pair, carried the same way and `Some` for
    /// the same reason.
    fn moe_routed_scaling(&self) -> Option<f32> {
        Some(self.ctx.moe_routed_scaling)
    }

    /// The `i`th input's row count.
    ///
    /// **This is `self.rows`, and the index is checked rather than used.**
    /// `bind/mod.rs:1596`'s `rows_of(b, i, rows)` — which every generated arm
    /// called for `Source::InRows` — is `{ let _ = (b, i); rows }`: every
    /// operand of a fire spans the same already-narrowed region, and the
    /// parameter exists so a future per-operand narrowing has a place to go.
    ///
    /// The bound is kept anyway, because `Cx::in_rows` refuses with
    /// `Refusal::Absent` and an out-of-range index IS an absent operand. A
    /// method that answered `self.rows` for `i = 99` would report a row count
    /// for a buffer the fire does not have.
    fn in_rows(&self, i: usize) -> Option<i32> {
        (i < self.spec.n_in).then_some(self.rows)
    }

    /// The attention workspace this fire was given.
    ///
    /// `AttnCtx::workspace`, which is the DECODE workspace. `prefill_workspace`
    /// is a second field and this query does not choose between them — the
    /// two families that asked both name the decode one, and a query that
    /// picked by guessing which phase a fire is in would be inventing a fact.
    fn attn_workspace(&self) -> Option<AttnWorkspace> {
        let w = self.attn?.workspace;
        Some(AttnWorkspace {
            float_buffer: w.float_buffer,
            float_bytes: w.float_bytes,
            int_buffer: w.int_buffer,
            int_bytes: w.int_bytes,
        })
    }

    /// The softmax scale this fire was planned with.
    fn sm_scale(&self) -> Option<f32> {
        Some(self.attn?.sm_scale)
    }

    /// `write_kv_to_pages`' first-token scalar.
    ///
    /// **All four of these have a producer**, which is what separates them
    /// from [`Facts::mla_layer`]: `AttnCtx` has carried them since before
    /// fn-world existed, and the queries were the missing half rather than
    /// the fill.
    fn first_token(&self) -> Option<i32> {
        Some(self.attn?.first_token)
    }

    /// The pages this fire's CSR names, which the dequant staging walks.
    fn num_pages_in_batch(&self) -> Option<i32> {
        Some(self.attn?.num_pages_in_batch)
    }

    /// Per-row target page for this fire's KV append.
    ///
    /// Null is absence rather than a value: a fire that appends no KV carries
    /// a null here, and a body that took the pointer anyway would index it.
    fn w_page_d(&self) -> Option<*const u32> {
        let p = self.attn?.w_page_d;
        (!p.is_null()).then_some(p)
    }

    /// Per-row offset-in-page for the append. [`Facts::w_page_d`]'s pair, and
    /// null-checked for the same reason.
    fn w_off_d(&self) -> Option<*const u32> {
        let p = self.attn?.w_off_d;
        (!p.is_null()).then_some(p)
    }

    /// gpt-oss's clamped-GLU ceiling.
    ///
    /// `DispatchCtx` has carried it since `bind/mod.rs:1193`. Always `Some`:
    /// a config value with a deployment default is not an optional fact.
    fn glu_limit(&self) -> Option<f32> {
        Some(self.ctx.glu_limit)
    }

    /// The clamped GLU's alpha, carried the same way.
    fn glu_alpha(&self) -> Option<f32> {
        Some(self.ctx.glu_alpha)
    }

    /// A weight the statement names by suffix.
    ///
    /// A linear scan, and it should stay one: the longest `w_suffixed` any
    /// fire carries is three. A map would cost an allocation per fire to
    /// avoid two comparisons.
    ///
    /// **Absence is not an error here** — see [`Cx::weight_suffixed`]. Two of
    /// `quant`'s three suffixes are nullable and one is not, in the same bind
    /// body, so the caller names the refusal.
    fn weight_suffixed(&self, suffix: &str) -> Option<*mut c_void> {
        self.w_suffixed
            .iter()
            .find(|(s, _)| *s == suffix)
            .map(|(_, p)| p.cast_mut())
            .filter(|p| !p.is_null())
    }

    /// The weights a statement names by NAME rather than by position.
    ///
    /// Two of them, and the index picks: `0` is `spec.weight` and `1` is
    /// `spec.weight2` — the generated arms' `w_named` and `w_named2`. Null
    /// is absence here for the reason the scaffolding gives at its own
    /// binding site: a statement that names no weight and a store that
    /// lacks the name are different situations with the same answer at this
    /// seam, and telling them apart is the caller's job, not a bind's.
    fn weight_named(&self, i: usize) -> Option<*mut c_void> {
        let p = match i {
            0 => self.w_named,
            1 => self.w_named2,
            _ => return None,
        };
        NonNull::new(p.cast_mut()).map(NonNull::as_ptr)
    }

    fn in_width(&self, i: usize) -> Option<i32> {
        if i >= self.spec.n_in {
            return None;
        }
        self.width(i)
    }

    fn out_width(&self, i: usize) -> Option<i32> {
        if i >= self.spec.n_out {
            return None;
        }
        self.width(self.spec.n_in + i)
    }

    /// The region, and the lane space it sits in.
    ///
    /// `total` is [`DispatchCtx::rows_total`] — the whole fire's row count,
    /// which a `_devwin` launch spans regardless of how many rows its own
    /// region serves. The row world could not state it as a `Source` and
    /// said so; here it is one field of one struct.
    fn rows(&self) -> Rows {
        Rows {
            start: i32::try_from(self.bound.rows.start).unwrap_or(0),
            count: self.rows,
            total: self.ctx.rows_total,
        }
    }

    fn layer(&self) -> usize {
        self.layer_index()
    }

    /// `OpKind::Launch::params` — the wire scalars the statement carries.
    fn param(&self, i: usize) -> Option<u32> {
        self.spec.params.get(i).copied()
    }

    fn positions(&self) -> Option<*const i32> {
        NonNull::new(self.ctx.positions)
            .map(|p| p.as_ptr().cast_const().cast::<i32>())
    }

    /// The fire's token ids.
    ///
    /// `*mut c_void` on [`DispatchCtx`] and `*const i32` here, because the
    /// buffer is written by the sampler and only ever READ by a kernel; the
    /// row world spelled the same narrowing as `Source::Ctx("token_ids")`
    /// against an `Operand` of `Ty::I32s`.
    ///
    /// Null is absence, which is the same convention [`positions`] uses one
    /// method up: a fire that carries no tokens leaves the field null rather
    /// than pointing it at an empty allocation.
    ///
    /// [`positions`]: Facts::positions
    fn token_ids(&self) -> Option<*const i32> {
        NonNull::new(self.ctx.token_ids).map(|p| p.as_ptr().cast_const().cast::<i32>())
    }

    /// How many tokens the vocabulary holds.
    ///
    /// Zero is absence and not a width, [`head_dim`]'s convention: a fire
    /// that states no vocabulary leaves it at zero, and an embedding gather
    /// that bounds-checked against zero would refuse every token.
    ///
    /// [`head_dim`]: Facts::head_dim
    fn vocab(&self) -> Option<i32> {
        (self.ctx.vocab > 0).then_some(self.ctx.vocab)
    }

    /// The rows a sampling gather collects, or `None` when it gathers every
    /// row.
    ///
    /// Already `*const i32` on [`DispatchCtx`], so this is a null test and
    /// nothing else. The row grammar's `Source::SamplingIndices` — a source
    /// spelling with exactly one consumer — is what retires with the row.
    fn sampling_indices(&self) -> Option<*const i32> {
        NonNull::new(self.ctx.sampling_indices.cast_mut()).map(|p| p.as_ptr().cast_const())
    }

    /// The per-layer-embedding width.
    ///
    /// Zero is absence, and here the row grammar SAID SO: the layer count
    /// was `Div(Width(In(0)), CtxNonZero("ple_dim"))`, and `CtxNonZero`
    /// exists because dividing by this field when it is zero faults.
    /// `Option` is that statement in the type system.
    fn ple_dim(&self) -> Option<i32> {
        (self.ctx.ple_dim > 0).then_some(self.ctx.ple_dim)
    }

    /// The fire's peel window, `[start, count]`, device-resident.
    ///
    /// **This is the operand `table/rope.rs` refused.** Its note read *"a
    /// device word the driver writes between replays; no `Source` reads
    /// device memory"*, and it was right about `Source` — a `Source` binds
    /// values a trace states, and this one is written by the driver after
    /// the trace is fixed, which is the whole point of a captured replay
    /// serving different splits. Under a `fn` it is a pointer that is
    /// either there or not, and `qk_rmsnorm_rope_devwin`'s bind reads it in
    /// one line.
    fn peel_window(&self) -> Option<NonNull<u32>> {
        NonNull::new(self.ctx.peel_window.cast_mut())
    }

    /// Elements per attention head.
    ///
    /// Zero is absence and not a width: `DispatchCtx::head_dim` is `i32`
    /// and a fire that states no head geometry leaves it at zero, which
    /// every reader refused already. A negative value cannot arise and is
    /// treated the same.
    fn head_dim(&self) -> Option<i32> {
        (self.ctx.head_dim > 0).then_some(self.ctx.head_dim)
    }

    fn num_q_heads(&self) -> Option<i32> {
        (self.ctx.num_q_heads > 0).then_some(self.ctx.num_q_heads)
    }

    fn num_kv_heads(&self) -> Option<i32> {
        (self.ctx.num_kv_heads > 0).then_some(self.ctx.num_kv_heads)
    }

    /// The rotary base, the fire-wide one.
    ///
    /// Rows spelled this `CtxNonZero("rope_theta")` — the `NonZero` was the
    /// row world saying `Option` in the only vocabulary it had.
    fn rope_theta(&self) -> Option<f32> {
        (self.ctx.rope_theta > 0.0).then_some(self.ctx.rope_theta)
    }

    /// The rotary base for THIS statement's layer.
    ///
    /// `Source::CtxByLayer("theta")`, which is
    /// [`DispatchCtx::theta`]'s whole reason for existing: gemma-4 splits
    /// theta by layer kind (sliding 1e4, full 1e6) and the fallback to the
    /// uniform value is deliberately on this side, because whether a
    /// family's per-layer vector is short is the driver's question.
    fn theta(&self) -> Option<f32> {
        let t = self.ctx.theta(self.layer_index());
        (t > 0.0).then_some(t)
    }

    fn rms_eps(&self) -> Option<f32> {
        (self.ctx.eps > 0.0).then_some(self.ctx.eps)
    }

    /// The logit soft cap, absent when the deployment states none.
    ///
    /// Zero is absence, not a cap of zero — which is the reading
    /// `Source::CtxNonZero("final_logit_softcap")` already had, moved into
    /// the type. Gemma-2/3/3n are the only deployments that state it.
    fn final_logit_softcap(&self) -> Option<f32> {
        (self.ctx.final_logit_softcap > 0.0).then_some(self.ctx.final_logit_softcap)
    }

    /// GPT-J adjacent-pair rotation, vs NeoX half/half.
    ///
    /// A `bool` and not an `Option<bool>`: a fire that states nothing means
    /// NeoX, which is what `false` says, and there is no third answer.
    fn rope_interleaved(&self) -> bool {
        self.ctx.rope_interleaved
    }

    /// How many channels rotate.
    ///
    /// Promoted verbatim from `dispatch_generated`'s `rotary_width`, whose
    /// preference order is the statement's own param, then the semantic
    /// `Rope { partial }`, then the fire's per-layer table — the first two
    /// are one fact under two spellings and both are live (qwen3_5's
    /// prefill states the launch, its decode records the semantic op).
    fn rotary_width(&self) -> Option<i32> {
        self.spec
            .params
            .first()
            .copied()
            .filter(|r| *r > 0)
            .or(self.spec.rope_partial)
            .or_else(|| {
                self.ctx
                    .rotary_by_layer
                    .get(self.layer_index())
                    .copied()
                    .filter(|r| *r > 0)
            })
            .and_then(|r| i32::try_from(r).ok())
    }

    /// The checkpoint's YaRN quartet, and the length it was trained at.
    ///
    /// `None` when the deployment states no YaRN block, which is
    /// `yarn_original_max <= 0` — the same test `rope.cu:367`'s guard makes
    /// (`yarn_factor > 1.f && yarn_original_max_position > 0`), so a bind
    /// that wants the un-ramped branch asks for [`Yarn::NONE`] explicitly
    /// rather than getting it by an accident of zeros.
    fn yarn(&self) -> Option<Yarn> {
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
    ///
    /// `has_kv_layer` and `kv_view` were two functions in the scaffolding
    /// because *"the generator emits the test into the branch GUARD and the
    /// read into the argument list, and a guard cannot bind"*. A `fn` can
    /// bind, so they are one method and the pair's whole reason is gone.
    fn kv_layer(&self) -> Option<KvLayer> {
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
            // The two enums are TRANSLATED rather than transmuted, and the
            // fallthrough is deliberate on each. `KvScheme` mirrors all five
            // of `KvCacheScheme`, so its `_` is unreachable and says so;
            // `KvDType` mirrors five of `DType`'s twelve, because a KV page
            // is never `Int4Packed` or `Mxfp4Packed` — those are weight
            // representations — so its `_` is a producer that reached a state
            // the mirror says it cannot, and refusing is the honest answer.
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
            // Answered, not handed over as inputs — see the fields' docs.
            has_envelopes: v.has_envelopes(),
            is_native_bf16: v.is_native_bf16(),
        })
    }

    /// The fire's per-request plan arrays.
    ///
    /// Every field is a device pointer the launcher takes loose. `None`
    /// when the fire has no attention half at all; a fire that has one but
    /// planned no requests answers with `requests: 0`, because zero
    /// requests is a rectangle and not a missing fact.
    fn plan(&self) -> Option<Plan> {
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

    /// One of a gated-delta-net layer's two state slabs.
    ///
    /// The scaffolding's `gdn_slab(g, state, field)` took the field as a
    /// `&str` and matched `"conv_state"` / `"recurrent_state"`. Two
    /// variants of an enum here, so the misspelling that would have
    /// silently declined is not writable.
    fn slab(&self, which: Slab) -> Option<*mut c_void> {
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
