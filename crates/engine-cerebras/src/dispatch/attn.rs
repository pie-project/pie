use kernels_cerebras::attn;
use poem_exec::{DispatchAttn, KernelError};
use poem_ir::{Attn, Operands, StructKind};

use crate::run::{Run, StructSlot};

impl DispatchAttn for Run<'_> {
    fn dispatch(&mut self, op: &Attn) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        self.attention(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn attention(&mut self, op: &Attn) -> Result<(), kernels_cerebras::Error> {
        if self.rs_seat().is_some() {
            return Err(kernels_cerebras::Error::Backend {
                op: op.name(),
                detail: "rollback seats (recurrent-state verbs) are not placed on this backend"
                    .to_string(),
            });
        }
        match op {
            Attn::PlanDecode { kv_len, plan, .. } => {
                match self.declared(*plan) {
                    StructKind::AttnDecodePlan => {}
                    other => panic!(
                        "`attn.plan_decode` defines a {other:?}, which is no decode plan kind"
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
            Attn::PlanPrefill { kv_len, plan, .. } => {
                match self.declared(*plan) {
                    StructKind::AttnPrefillPlan => {}
                    other => panic!(
                        "`attn.plan_prefill` defines a {other:?}, which is no prefill plan kind"
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
            Attn::Decode {
                q,
                plan,
                cache,
                window,
                head_dim,
                sm_scale,
                o,
                selection: None,
                bias: None,
                selected: None,
                rel: None,
                lse: None,
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
            Attn::Decode {
                q,
                plan,
                cache,
                window,
                head_dim,
                sm_scale,
                o,
                lse: Some(lse),
                selection: None,
                bias: None,
                selected: None,
                rel: None,
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
            Attn::Prefill {
                q,
                plan,
                cache,
                window,
                head_dim,
                kv_heads,
                sm_scale,
                o,
                mask: None,
                selection: None,
                bias: None,
                floor: None,
                causal: true,
                selected: None,
                rel: None,
                lse: None,
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
            Attn::Prefill {
                q,
                plan,
                cache,
                window,
                head_dim,
                kv_heads,
                sm_scale,
                o,
                lse: Some(lse),
                floor: None,
                mask: None,
                selection: None,
                bias: None,
                causal: true,
                selected: None,
                rel: None,
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
            Attn::Prefill {
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
                selection: None,
                bias: None,
                floor: None,
                selected: None,
                rel: None,
                lse: None,
            } if mask.is_some() || !*causal => {
                let mask = self.cut_rows(self.mask_bits(*mask));
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
            Attn::Prefill {
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
                lse: Some(lse),
                selection: None,
                bias: None,
                floor: None,
                selected: None,
                rel: None,
            } if mask.is_some() || !*causal => {
                let mask = self.cut_rows(self.mask_bits(*mask));
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
            Attn::KvAppend {
                k,
                v: Some(v),
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
            Attn::KvAppend {
                k: plane,
                v: None,
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
            Attn::Dense {
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
            Attn::Decode {
                q,
                plan,
                selection: Some(selection),
                cache,
                window,
                head_dim,
                sm_scale,
                o,
                bias: None,
                selected:
                    Some(poem_ir::Blocks {
                        ratio,
                        per_head: false,
                    }),
                rel: None,
                lse: None,
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
            Attn::Prefill {
                q,
                plan,
                selection: Some(selection),
                cache,
                window,
                head_dim,
                kv_heads,
                sm_scale,
                o,
                mask: None,
                bias: None,
                floor: None,
                causal: true,
                selected:
                    Some(poem_ir::Blocks {
                        ratio,
                        per_head: false,
                    }),
                rel: None,
                lse: None,
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
            Attn::IndexKvAppend {
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
            Attn::IndexBlockMean {
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
            Attn::IndexTopk {
                q,
                weights,
                keys,
                heads,
                head_dim,
                top_k,
                ratio,
                score: poem_ir::IndexScore::Pooled,
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
            Attn::PoolBoundary {
                positions,
                row_valid,
                ratio,
                boundary_pos,
                boundary_req,
                boundary_rope,
                decode: true,
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
            Attn::PoolBoundary {
                positions,
                row_valid,
                ratio,
                boundary_pos,
                boundary_req,
                boundary_rope,
                decode: false,
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
            Attn::PoolKvAppend {
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
            other => Err(kernels_cerebras::Error::Unsupported { op: other.name() }),
        }
    }
}
