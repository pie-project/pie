use kernels_cerebras::attn;
use model_exec::{DispatchAttention, KernelError};
use model_ir::{Attention, Operands, StructKind};

use crate::run::{Run, StructSlot};

impl DispatchAttention for Run<'_> {
    fn dispatch(&mut self, op: &Attention) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        self.attention(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn attention(&mut self, op: &Attention) -> Result<(), kernels_cerebras::Error> {
        if self.rs_seat().is_some() {
            return Err(kernels_cerebras::Error::Backend {
                op: op.name(),
                detail: "rollback seats (recurrent-state verbs) are not placed on this backend"
                    .to_string(),
            });
        }
        match op {
            Attention::PlanDecode { kv_len, plan, .. } => {
                match self.declared(*plan) {
                    StructKind::AttnDecodePlan => {}
                    other => panic!(
                        "`attention.plan_decode` defines a {other:?}, which is no decode plan kind"
                    ),
                }
                let kv_len = self.tensor(*kv_len);
                let fire = self.bindings();
                let (positions, t) = (fire.cache_rows, fire.tables);
                let built = attn::plan_decode(
                    self.ctx(),
                    kv_len,
                    self.cut_rows(positions),
                    self.cut_rows(t.request_of_token),
                    self.cut_rows(t.mask),
                    self.cut_rows(t.mask_enabled),
                    t.mask_stride,
                )?;
                self.put(*plan, StructSlot::Decode(built));
                Ok(())
            }
            Attention::PlanPrefill { kv_len, plan, .. } => {
                match self.declared(*plan) {
                    StructKind::AttnPrefillPlan => {}
                    other => panic!(
                        "`attention.plan_prefill` defines a {other:?}, which is no prefill plan kind"
                    ),
                }
                let kv_len = self.tensor(*kv_len);
                let fire = self.bindings();
                let (positions, t) = (fire.cache_rows, fire.tables);
                let built = attn::plan_prefill(
                    self.ctx(),
                    kv_len,
                    self.cut_rows(positions),
                    self.cut_rows(t.request_of_token),
                    self.cut_rows(t.mask),
                    self.cut_rows(t.mask_enabled),
                    t.mask_stride,
                )?;
                self.put(*plan, StructSlot::Prefill(built));
                Ok(())
            }
            Attention::Decode {
                q,
                plan,
                cache,
                window,
                head_dim,
                sm_scale,
                o,
            } => attn::decode(
                self.ctx(),
                self.tensor(*q),
                self.decode_plan(*plan),
                self.pool(*cache),
                *window,
                *head_dim,
                *sm_scale,
                self.tensor(*o),
            ),
            Attention::DecodeLse {
                q,
                plan,
                cache,
                window,
                head_dim,
                sm_scale,
                o,
                lse,
            } => attn::decode_lse(
                self.ctx(),
                self.tensor(*q),
                self.decode_plan(*plan),
                self.pool(*cache),
                *window,
                *head_dim,
                *sm_scale,
                self.tensor(*o),
                self.tensor(*lse),
            ),
            Attention::Prefill {
                q,
                plan,
                cache,
                window,
                head_dim,
                kv_heads,
                sm_scale,
                o,
            } => attn::prefill(
                self.ctx(),
                self.ragged(*q),
                self.prefill_plan(*plan),
                self.pool(*cache),
                *window,
                *head_dim,
                *kv_heads,
                *sm_scale,
                self.tensor(*o),
            ),
            Attention::PrefillLse {
                q,
                plan,
                cache,
                window,
                head_dim,
                kv_heads,
                sm_scale,
                o,
                lse,
            } => attn::prefill_lse(
                self.ctx(),
                self.ragged(*q),
                self.prefill_plan(*plan),
                self.pool(*cache),
                *window,
                *head_dim,
                *kv_heads,
                *sm_scale,
                self.tensor(*o),
                self.tensor(*lse),
            ),
            Attention::Masked {
                q,
                plan,
                mask,
                cache,
                window,
                head_dim,
                kv_heads: _,
                causal,
                sm_scale,
                o,
            } => {
                let mask = self.cut_rows(self.tensor(*mask));
                attn::masked(
                    self.ctx(),
                    self.ragged(*q),
                    self.prefill_plan(*plan),
                    mask,
                    self.pool(*cache),
                    *window,
                    *causal,
                    *head_dim,
                    *sm_scale,
                    self.tensor(*o),
                )
            }
            Attention::MaskedLse {
                q,
                plan,
                mask,
                cache,
                window,
                head_dim,
                kv_heads: _,
                causal,
                sm_scale,
                o,
                lse,
            } => {
                let mask = self.cut_rows(self.tensor(*mask));
                attn::masked_lse(
                    self.ctx(),
                    self.ragged(*q),
                    self.prefill_plan(*plan),
                    mask,
                    self.pool(*cache),
                    *window,
                    *causal,
                    *head_dim,
                    *sm_scale,
                    self.tensor(*o),
                    self.tensor(*lse),
                )
            }
            Attention::KvAppend {
                k,
                v,
                cache,
                write_page,
                write_offset,
            } => attn::kv_append(
                self.ctx(),
                self.tensor(*k),
                self.tensor(*v),
                self.pool(*cache),
                self.tensor(*write_page),
                self.tensor(*write_offset),
            ),
            Attention::KvAppendShared {
                plane,
                cache,
                write_page,
                write_offset,
            } => attn::kv_append_shared(
                self.ctx(),
                self.tensor(*plane),
                self.pool(*cache),
                self.tensor(*write_page),
                self.tensor(*write_offset),
            ),
            Attention::SsmCausalConv1d {
                x,
                weight,
                state,
                conv_width,
                dilation,
                y,
            } => attn::ssm::causal_conv1d(
                self.ctx(),
                self.tensor(*x),
                self.tensor(*weight),
                &self.recurrent(*state),
                *conv_width,
                *dilation,
                self.tensor(*y),
            ),
            Attention::SsmCausalConv1dChunked {
                x,
                weight,
                state,
                conv_width,
                dilation,
                y,
            } => attn::ssm::causal_conv1d_chunked(
                self.ctx(),
                self.ragged(*x),
                self.tensor(*weight),
                &self.recurrent(*state),
                *conv_width,
                *dilation,
                self.tensor(*y),
            ),
            Attention::SsmGdnPrep {
                ba,
                dt_bias,
                a_log,
                gates,
            } => attn::ssm::gdn_prep(
                self.ctx(),
                self.tensor(*ba),
                self.tensor(*dt_bias),
                self.tensor(*a_log),
                self.tensor(*gates),
            ),
            Attention::SsmGatedDelta {
                qkv,
                z,
                gates,
                state,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                y,
            } => attn::ssm::gated_delta(
                self.ctx(),
                self.tensor(*qkv),
                self.tensor(*z),
                self.tensor(*gates),
                &self.recurrent(*state),
                *k_heads,
                *v_heads,
                *k_dim,
                *v_dim,
                self.tensor(*y),
            ),
            Attention::SsmGatedDeltaChunked {
                qkv,
                z,
                gates,
                state,
                k_heads,
                v_heads,
                k_dim,
                v_dim,
                y,
            } => attn::ssm::gated_delta_chunked(
                self.ctx(),
                self.ragged(*qkv),
                self.tensor(*z),
                self.tensor(*gates),
                &self.recurrent(*state),
                *k_heads,
                *v_heads,
                *k_dim,
                *v_dim,
                self.tensor(*y),
            ),
            Attention::Dense {
                q,
                k,
                v,
                segments,
                head_dim,
                sm_scale,
                o,
            } => attn::dense::bidirectional(
                self.ctx(),
                self.tensor(*q),
                self.tensor(*k),
                self.tensor(*v),
                self.tensor(*segments),
                *head_dim,
                *sm_scale,
                self.tensor(*o),
            ),
            Attention::DecodeSelected {
                q,
                plan,
                selection,
                cache,
                window,
                head_dim,
                sm_scale,
                ratio,
                o,
            } => attn::decode_selected(
                self.ctx(),
                self.tensor(*q),
                self.decode_plan(*plan),
                self.tensor(*selection),
                self.pool(*cache),
                *window,
                *head_dim,
                *sm_scale,
                *ratio,
                self.tensor(*o),
            ),
            Attention::PrefillSelected {
                q,
                plan,
                selection,
                cache,
                window,
                head_dim,
                kv_heads,
                sm_scale,
                ratio,
                o,
            } => attn::prefill_selected(
                self.ctx(),
                self.ragged(*q),
                self.prefill_plan(*plan),
                self.tensor(*selection),
                self.pool(*cache),
                *window,
                *head_dim,
                *kv_heads,
                *sm_scale,
                *ratio,
                self.tensor(*o),
            ),
            Attention::IndexKvAppend {
                k,
                keys,
                write_page,
                write_offset,
            } => attn::index::kv_append(
                self.ctx(),
                self.tensor(*k),
                self.pool(*keys),
                self.tensor(*write_page),
                self.tensor(*write_offset),
            ),
            Attention::IndexBlockMean {
                boundary_pos,
                boundary_req,
                keys,
                head_dim,
                ratio,
                entries,
            } => attn::index::block_mean(
                self.ctx(),
                self.tensor(*boundary_pos),
                self.tensor(*boundary_req),
                self.pool(*keys),
                *head_dim,
                *ratio,
                self.tensor(*entries),
            ),
            Attention::IndexTopk {
                q,
                weights,
                keys,
                heads,
                head_dim,
                top_k,
                ratio,
                selection,
            } => {
                let fire = self.bindings();
                let (positions, request_of_token) = (fire.cache_rows, fire.tables.request_of_token);
                attn::index::topk(
                    self.ctx(),
                    self.tensor(*q),
                    weights.map(|w| self.tensor(w)),
                    self.pool(*keys),
                    self.cut_rows(positions),
                    self.cut_rows(request_of_token),
                    *heads,
                    *head_dim,
                    *top_k,
                    *ratio,
                    self.tensor(*selection),
                )
            }
            Attention::PoolBoundaryDecode {
                positions,
                row_valid,
                ratio,
                boundary_pos,
                boundary_req,
                boundary_rope,
            } => attn::index::boundary_decode(
                self.ctx(),
                self.tensor(*positions),
                self.cut_rows(self.bindings().tables.request_of_token),
                self.tensor(*row_valid),
                *ratio,
                self.tensor(*boundary_pos),
                self.tensor(*boundary_req),
                self.tensor(*boundary_rope),
            ),
            Attention::PoolBoundaryPrefill {
                positions,
                row_valid,
                ratio,
                boundary_pos,
                boundary_req,
                boundary_rope,
            } => attn::index::boundary_prefill(
                self.ctx(),
                self.ragged(*positions),
                self.cut_rows(self.bindings().tables.request_of_token),
                self.tensor(*row_valid),
                *ratio,
                self.tensor(*boundary_pos),
                self.tensor(*boundary_req),
                self.tensor(*boundary_rope),
            ),
            Attention::PoolKvAppend {
                entries,
                boundary_pos,
                boundary_req,
                pool: into,
                write_page,
                write_offset,
            } => attn::index::pool_kv_append(
                self.ctx(),
                self.tensor(*entries),
                self.tensor(*boundary_pos),
                self.tensor(*boundary_req),
                self.pool(*into),
                self.tensor(*write_page),
                self.tensor(*write_offset),
            ),
            Attention::PleNgramIds {
                ids,
                state,
                eos,
                mults,
                primes,
                offsets,
                heads_per_ngram,
                map,
                ngram_ids,
            } => attn::ple::ngram_ids(
                self.ctx(),
                self.tensor(*ids),
                &self.recurrent(*state),
                *eos,
                mults,
                primes,
                offsets,
                *heads_per_ngram,
                map.map(|m| self.tensor(m)),
                self.tensor(*ngram_ids),
            ),
            Attention::PleNgramIdsChunked {
                ids,
                state,
                eos,
                mults,
                primes,
                offsets,
                heads_per_ngram,
                map,
                ngram_ids,
            } => attn::ple::ngram_ids_chunked(
                self.ctx(),
                self.ragged(*ids),
                &self.recurrent(*state),
                *eos,
                mults,
                primes,
                offsets,
                *heads_per_ngram,
                map.map(|m| self.tensor(m)),
                self.tensor(*ngram_ids),
            ),
            Attention::SelectorWalk {
                cand,
                unary,
                hp,
                tokens,
                pred,
                succ,
                first,
                picks,
            } => attn::dynconv::selector_walk(
                self.ctx(),
                self.ragged(*cand),
                self.tensor(*unary),
                hp.map(|h| self.tensor(h)),
                self.tensor(*tokens),
                self.tensor(*pred),
                self.tensor(*succ),
                *first,
                self.tensor(*picks),
            ),
            Attention::BlockDynConv {
                x,
                coeff,
                base,
                side,
                taps,
                group,
                y,
            } => attn::dynconv::block_dyn_conv(
                self.ctx(),
                self.ragged(*x),
                self.tensor(*coeff),
                self.tensor(*base),
                *side,
                *taps,
                *group,
                self.tensor(*y),
            ),
            other => Err(kernels_cerebras::Error::Unsupported { op: other.name() }),
        }
    }
}
