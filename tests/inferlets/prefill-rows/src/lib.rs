//! Teacher-forced prefill of stated ids on an attention-kind model, every
//! row read out: what a reference's encoder logits are compared against
//! when the question is the trunk alone (`scripts/diffusiongemma_parity_ref.py`).

use inferlet::eta::shared_prelude::*;
use serde::{Deserialize, Serialize};
use std::ops::RangeBounds;

/// The two kinds bind state through different `attention` signatures; what
/// they share for this program is the guest's to say (text-completion's idiom).
trait BindState {
    fn bind_state<R, W>(
        &self,
        ws: &WorkingSet,
        geom: KvGeometry<'_, R, W>,
        rs: &[RsWorkingSet],
    ) -> ::std::result::Result<(), String>
    where
        R: RangeBounds<u32>,
        W: RangeBounds<u32>;
}

impl BindState for inferlet::eta::attention::ForwardPass {
    fn bind_state<R, W>(
        &self,
        ws: &WorkingSet,
        geom: KvGeometry<'_, R, W>,
        rs: &[RsWorkingSet],
    ) -> ::std::result::Result<(), String>
    where
        R: RangeBounds<u32>,
        W: RangeBounds<u32>,
    {
        debug_assert!(rs.is_empty());
        self.attention(ws, geom)
    }
}

impl BindState for inferlet::eta::hybrid::ForwardPass {
    fn bind_state<R, W>(
        &self,
        ws: &WorkingSet,
        geom: KvGeometry<'_, R, W>,
        rs: &[RsWorkingSet],
    ) -> ::std::result::Result<(), String>
    where
        R: RangeBounds<u32>,
        W: RangeBounds<u32>,
    {
        self.attention(
            Some(KvBinding {
                working_set: ws,
                geometry: geom,
            }),
            rs,
            RsGeometry {
                fold_len: None,
                buffer: 0..0,
            },
        )
    }
}

#[derive(Deserialize)]
struct Input {
    ids: String,
}

#[derive(Serialize, Default)]
struct Rows {
    argmax: Vec<i32>,
    entropy: Vec<f32>,
    top8_ids: Vec<u32>,
    top8_probs: Vec<f32>,
}

#[derive(Serialize)]
struct Output {
    prefill_rows: Rows,
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    match model::pass_kind() {
        model::ForwardKind::Attention => run_attention(input).await,
        model::ForwardKind::Hybrid => run_hybrid(input).await,
        model::ForwardKind::Recurrent => Err("no recurrent-only path here".into()),
        model::ForwardKind::Diffusion => Err("use diffusion-parity --only_prefill for a diffusion model".into()),
    }
}

macro_rules! define_run {
    ($name:ident, $kind:ident) => {
        async fn $name(input: Input) -> Result<Output> {
            use inferlet::eta::$kind::ForwardPass;
            let ids: Vec<u32> = inferlet::serde_json::from_str(&input.ids)
                .map_err(|why| format!("ids json: {why}"))?;
            let n = u32::try_from(ids.len()).map_err(|_| "prompt is too long")?;
            let page_size = kv_page_size();
            let max_pages = n.div_ceil(page_size).max(1);
            let ws = WorkingSet::new();
            ws.reserve(max_pages).context("reserve KV")?;
            let rs_ws: Vec<RsWorkingSet> = if model::pass_kind() == model::ForwardKind::Hybrid {
                vec![RsWorkingSet::new()]
            } else {
                Vec::new()
            };
            let pipe = Pipeline::new();
            let prompt_i32: Vec<i32> = ids.iter().map(|&t| t as i32).collect();
            let mut rows = Rows::default();
            for (at, &(base, end)) in prefill_chunks(n, None).iter().enumerate() {
                let len = end - base;
                let toks = Channel::from(&prompt_i32[base as usize..end as usize]).named(&format!("toks_{at}"));
                let embed_indptr = Channel::from([0u32, len]).named(&format!("embed_indptr_{at}"));
                let positions = Channel::from_iter(base..end).named(&format!("positions_{at}"));
                let pages = Channel::from_iter(0..max_pages).named(&format!("pages_{at}"));
                let page_indptr = Channel::from([0u32, end.div_ceil(page_size)]).named(&format!("page_indptr_{at}"));
                let w_slot = Channel::from_iter((base..end).map(|p| p / page_size)).named(&format!("w_slot_{at}"));
                let w_off = Channel::from_iter((base..end).map(|p| p % page_size)).named(&format!("w_off_{at}"));
                let kv_len = Channel::from([end]).named(&format!("kv_len_{at}"));
                let readout = Channel::from_iter(0..len).named(&format!("readout_{at}"));
                let arg_out = Channel::new([len], dtype::i32).named(&format!("arg_{at}"));
                let ent_out = Channel::new([len], dtype::f32).named(&format!("ent_{at}"));
                let ids_out = Channel::new([len, 8], dtype::u32).named(&format!("ids_{at}"));
                let probs_out = Channel::new([len, 8], dtype::f32).named(&format!("probs_{at}"));

                let fwd = ForwardPass::new();
                fwd.embed(&toks, &embed_indptr)?;
                fwd.bind_state(
                    &ws,
                    KvGeometry {
                        readable_pages: ..,
                        writable_pages: ..,
                        kv_len: &kv_len,
                        pages: &pages,
                        page_indptr: &page_indptr,
                        w_slot: &w_slot,
                        w_off: &w_off,
                        positions: &positions,
                        mask: None,
                    },
                    &rs_ws,
                )?;
                fwd.readout(&readout)?;
                fwd.epilogue(move || {
                    let logits = intrinsics::logits();
                    let probs = softmax(&logits);
                    let (top_p, top_i) = top_k(&probs, 8);
                    arg_out.put(reduce_argmax(&logits));
                    ent_out.put(entropy(&probs));
                    ids_out.put(top_i);
                    probs_out.put(top_p);
                });
                fwd.submit(&pipe).with_context(|| format!("prefill submit @{base}"))?;
                rows.argmax.extend(arg_out.take_host::<Vec<i32>>().await?);
                rows.entropy.extend(ent_out.take_host::<Vec<f32>>().await?);
                rows.top8_ids.extend(ids_out.take_host::<Vec<u32>>().await?);
                rows.top8_probs.extend(probs_out.take_host::<Vec<f32>>().await?);
            }
            pipe.close();
            Ok(Output { prefill_rows: rows })
        }
    };
}

define_run!(run_attention, attention);
define_run!(run_hybrid, hybrid);
