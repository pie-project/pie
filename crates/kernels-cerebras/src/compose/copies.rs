//! What the role kernels read off the PEs of each block the composer places ([`KernelCopies`]):
//! the copies onto each kind, by the layout the plan's layers take ([`Layout`]) and the plan's
//! facts (the placed hidden's zero pad, the head's padded vocabulary rows, the experts' encoding,
//! the token loop's embedding). The sources each copy names are the cut's checkpoint tensors
//! (documentation: [`crate::weights::Binding::from`]).

use crate::block::head::{HeadCopies, Hemb};
use crate::block::moe::MoeCopies;
use crate::block::row_runs::RunCopies;
use crate::block::units::UnitCopies;
use crate::kernels::split::Dims;
use crate::plan::Plan;
use crate::weights::{From, Slice, Transform, from};

use super::{KernelCopies, Layout};

type R<T> = Result<T, String>;

/// The copies the kernels of `layout` read off `plan`'s blocks.
pub fn of(plan: &Plan, layout: Layout) -> R<KernelCopies> {
    match layout {
        Layout::Snake => snake(plan),
        Layout::Stacked => Ok(stacked(plan)),
        Layout::Runs => Ok(runs(plan)),
        Layout::Split => split(plan),
    }
}

/// The head's rows padded to its vocabulary rows with the table's last real row.
fn pad_rows(plan: &Plan) -> Transform {
    let h = &plan.partition.head;
    Transform::PadLastRow {
        to: h.rows * h.vocab_rows,
    }
}

/// The snake's units (their weights and gammas zero-padded past the hidden where the rows do not
/// divide it: Qwen3-8B), every CY table optional (a model without GDN has no conv), the head east
/// of them with the embedding after it when the token loop runs.
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

/// Per unit and run block its weights (bf16 -> f16) and gammas, its CY row's q / k norms and RoPE
/// tables; the head by row groups in its two halves' files (one is over the 2 GiB a staged file
/// may take), its rows padded with the embedding's last.
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

/// The attention | MoE halves' words a PE (the band's are its own), the experts' banks
/// re-encoded as the kernels read them (MXFP4 as E5M2, 16-bit as f16).
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
