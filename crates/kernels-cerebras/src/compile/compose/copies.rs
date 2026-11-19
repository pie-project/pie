//! The copies the role kernels read off each placed block's PEs ([`KernelCopies`]), by the plan's
//! kernel set and facts (hidden pad, padded vocabulary rows, expert encoding, token loop). Sources
//! are the cut's checkpoint tensors ([`crate::compile::weights::Binding::from`]).

use crate::compile::block::head::{HeadCopies, Hemb};
use crate::compile::block::moe::MoeCopies;
use crate::compile::block::row_runs::RunCopies;
use crate::compile::block::units::UnitCopies;
use crate::compile::kernels::split::Dims;
use crate::compile::plan::Plan;
use crate::compile::weights::{From, Slice, Transform, from};

use super::KernelCopies;
use crate::compile::kernels::KernelSet;

type R<T> = Result<T, String>;

/// The copies the kernels of `set` read off `plan`'s blocks.
pub fn of(plan: &Plan, set: KernelSet) -> R<KernelCopies> {
    match set {
        KernelSet::Own => snake(plan),
        KernelSet::Shared => Ok(stacked(plan)),
        KernelSet::Windows => Ok(runs(plan)),
        KernelSet::Experts => split(plan),
    }
}

/// The head's rows padded to its vocabulary rows with the table's last real row.
fn pad_rows(plan: &Plan) -> Transform {
    let h = &plan.partition.head;
    Transform::PadLastRow {
        to: h.rows * h.vocab_rows,
    }
}

/// The snake's units (zero-padded past the hidden where the rows do not divide it), every CY
/// table optional, the head east of them with the embedding after it when the token loop runs.
fn snake(plan: &Plan) -> R<KernelCopies> {
    let (s, p) = (&plan.shape, &plan.partition);
    let placed = p.hidden_placed();
    let pad = |t: &str| -> From {
        let tr = if placed != s.hidden {
            vec![Transform::PadZero { dim: 1, to: placed }]
        } else {
            vec![]
        };
        vec![(t.to_string(), tr)]
    };
    Ok(KernelCopies {
        units: Some(UnitCopies {
            cols: p.cols,
            rows: p.rows,
            w_from: pad("layers.*.{q,k,v,o,gate,up,down}_proj, linear_attn.*"),
            gam_b16: true,
            gam_from: pad("layers.*.{input,post_attention}_layernorm, norm"),
            cy: &[
                ("cw", true),
                ("gpar", true),
                ("gnw", true),
                ("qkw", true),
                ("rot", true),
            ],
            cy_optional: true,
        }),
        head: Some(HeadCopies {
            cols: p.head.cols,
            rows: p.head.rows,
            hw: |_, rows, cols| Slice::of("hw", None, vec![rows, cols]),
            hw_from: vec![("lm_head".into(), vec![pad_rows(plan)])],
            hg_from: from("norm"),
            hemb: if plan.contract.token_loop {
                Hemb::After(from(&format!("embed_tokens ({} rows)", s.vocab)))
            } else {
                Hemb::None
            },
        }),
        ..Default::default()
    })
}

/// The stacked units, the head east of them (fed the embedding inline).
fn stacked(plan: &Plan) -> KernelCopies {
    let p = &plan.partition;
    KernelCopies {
        units: Some(UnitCopies {
            cols: p.cols,
            rows: p.rows,
            w_from: Vec::new(),
            gam_b16: false,
            gam_from: Vec::new(),
            cy: &[("qkw", true), ("rot", false)],
            cy_optional: false,
        }),
        head: Some(HeadCopies {
            cols: p.head.cols,
            rows: p.head.rows,
            hw: |_, rows, cols| Slice::of("hw", None, vec![rows, cols]),
            hw_from: vec![("embed_tokens".into(), vec![pad_rows(plan)])],
            hg_from: from("norm"),
            hemb: Hemb::Inline(Vec::new()),
        }),
        ..Default::default()
    }
}

/// Per unit and run block its weights (bf16 -> f16), gammas, q / k norms and RoPE tables; the head
/// split across two files (one would pass the 2 GiB a staged file may take), rows padded.
fn runs(plan: &Plan) -> KernelCopies {
    let p = &plan.partition;
    let f16 = [
        "self_attn.{q,k,v,o}_proj.weight",
        "mlp.{gate,up,down}_proj.weight",
    ]
    .into_iter()
    .map(|n| (n.to_string(), vec![Transform::Bf16ToF16]))
    .collect();
    KernelCopies {
        runs: Some(RunCopies {
            w_from: f16,
            cy: &[("qkw", true), ("rot", false)],
        }),
        head: Some(HeadCopies {
            cols: p.head.cols,
            rows: p.head.rows,
            hw: |i, rows, (c0, c1)| Slice::of(&format!("hw{i}"), None, vec![rows, (0, c1 - c0)]),
            hw_from: vec![(
                "embed_tokens.weight".into(),
                vec![pad_rows(plan), Transform::Bf16ToF16],
            )],
            hg_from: from("norm.weight"),
            hemb: Hemb::None,
        }),
        ..Default::default()
    }
}

/// The attention | MoE halves' words a PE, the experts' banks re-encoded as the kernels read them
/// (MXFP4 as E5M2, 16-bit as f16).
fn split(plan: &Plan) -> R<KernelCopies> {
    let d = Dims::of(plan)?;
    let k = d.k;
    let lwa = d.hcc * d.nqa + d.qo * d.hcc;
    Ok(KernelCopies {
        moe: Some(MoeCopies {
            w: k * lwa,
            wx: k * d.nex * d.xe,
            wr: k * d.hcc,
            cx: [2 * k * d.hcc, k * d.hcc, k * d.nex * d.hcc],
            cya: [k * d.nqa, k * d.qh, d.nlv * 8 * 2 * d.nrot],
            cym: [k, k * d.nex * 2 * d.ic, k * d.nex * 2],
            wx_from: if plan.shape.moe.is_some_and(|m| m.quantized) {
                vec![(
                    "model.layers.*.mlp.experts.{gate_up,down}_proj_blocks".to_string(),
                    vec![Transform::Mxfp4ToE5m2],
                )]
            } else {
                vec![(
                    "model.layers.*.mlp.experts.*.{gate_proj,up_proj,down_proj}".to_string(),
                    vec![Transform::Bf16ToF16],
                )]
            },
        }),
        ..Default::default()
    })
}
