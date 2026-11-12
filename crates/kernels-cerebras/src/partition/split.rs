//! Attention | MoE halves: gpt-oss and Qwen3 MoE as they plan: units of AX attention + MX MoE
//! columns (CX after the attention columns), PY rows, a snake of two unit columns that starts
//! upward (so the first and the last unit sit on the band), the scores segments on the attention
//! columns, the router's on the CY row's MoE columns, the head as a band below the units with its
//! control row, the token loop over the band. Blocks (by [`crate::compose`]): `MoeUnits`,
//! `AttnScores`, `Router`, `Band`, `BandLoop`; kernels: [`crate::kernels::split`]; cut:
//! [`crate::tiles::split`].

use crate::partition::ceil_to;
use crate::plan::{
    Contract, HeadPartition, KvEntries, Partition, Plan, Preset, ReduceGroups, RowRun, Segments,
    Shape,
};

type R<T> = Result<T, String>;

/// The keys a plan's preset gives: attention and MoE columns a unit (AX, MX), rows (PY), the unit
/// slots (UNX x UNY, every slot a unit), the scores and router segments (SWA, SWR), the band's
/// vocabulary columns (HVC), KV entries a row (NE), the RoPE tables' levels (NLV), the reduction
/// groups (G2, G2W); the logits' prescale (HSCALE: the band head's 2^15) and the posted loop (LOOP).
///
/// Units of K = layers / units, a row's dims the hidden over PY; a scores segment a KV head's
/// attention columns; the band head HVC columns of VR vocabulary rows (a multiple of 8, the
/// head control's argmax). Overrides: `NE`, `G2`, `G2W`, `CAP`, `LOOP`, `WL` (the token loop;
/// on unless 0).
pub fn plan(s: &Shape, pr: &Preset) -> R<Plan> {
    let (ax, mx, py) = (pr.need("AX")?, pr.need("MX")?, pr.need("PY")?);
    let (unx, uny) = (pr.need("UNX")?, pr.need("UNY")?);
    let (swa, hvc) = (pr.need("SWA")?, pr.need("HVC")?);
    let units = unx * uny;
    let n = s.layers.len() as u32;
    if units == 0 || !n.is_multiple_of(units) {
        return Err(format!("split: {n} layers on {units} units"));
    }
    let k = n / units;
    if s.layers
        .chunks(k as usize)
        .any(|u| u != &s.layers[..k as usize])
    {
        return Err("split: every unit the same layer kinds".into());
    }
    if !ax.is_multiple_of(swa) || ax / swa != s.attn.kv_heads {
        return Err(format!(
            "split: {ax} attention columns in segments of {swa} for {} KV heads",
            s.attn.kv_heads
        ));
    }
    if !s.hidden.is_multiple_of(py) {
        return Err(format!("split: {} hidden over {py} rows", s.hidden));
    }
    // the cut's and the kernels' divisibility: a column's kv dims rotated pairs, its expert
    // dims an even share of each expert, every expert's router output on its own columns,
    // the router's pick tree over blocks of 8 experts, MXFP4's two codes a word
    let moe = s.moe.ok_or("split: no experts")?;
    let (hd, hc, inter) = (s.attn.head_dim, s.hidden / py, moe.expert_intermediate);
    let dc = hd / swa;
    if !hd.is_multiple_of(swa) || dc % 2 != 0 || !s.attn.q_heads.is_multiple_of(s.attn.kv_heads) {
        return Err(format!(
            "split: head dim {hd} over {swa} columns a KV head (rotated pairs a column)"
        ));
    }
    if mx == 0 || !inter.is_multiple_of(mx) || !mx.is_multiple_of(moe.experts) {
        return Err(format!(
            "split: {inter} expert dims over {mx} MoE columns ({} experts' router outputs)",
            moe.experts
        ));
    }
    // a router segment holds every expert's output once (its words index the experts)
    if pr.need("SWR")? != moe.experts {
        return Err(format!(
            "split: router segments of {} columns for {} experts",
            pr.need("SWR")?,
            moe.experts
        ));
    }
    if !moe.experts.is_multiple_of(8) || moe.top_k == 0 || moe.top_k > moe.experts {
        return Err(format!(
            "split: top {} of {} experts (the pick tree takes blocks of 8)",
            moe.top_k, moe.experts
        ));
    }
    if moe.quantized && (hc % 2 != 0 || (inter / mx) % 2 != 0) {
        return Err(format!(
            "split: MXFP4 experts pack two rows a word ({hc} dims a row, {} a column)",
            inter / mx
        ));
    }
    let ne = pr.need("NE")?;
    let partition = Partition {
        unit_grid: Some((unx, uny)),
        units,
        layers_per_unit: k,
        rows: py,
        cols: ax + mx,
        row_dims: vec![RowRun {
            rows: (0, py),
            dims: s.hidden / py,
            attention: true,
        }],
        kv_rows: py,
        kv_entries: KvEntries {
            per_row: ne,
            ..Default::default()
        },
        reduce: ReduceGroups {
            row: pr.need("G2")?,
            col: pr.need("G2")?,
            head: pr.need("G2")?,
            token_loop: pr.need("G2W")?,
        },
        segments: Segments {
            scores: swa,
            router: Some(pr.need("SWR")?),
            ..Default::default()
        },
        attn_cols: Some(ax),
        head: HeadPartition {
            cols: hvc,
            rows: py,
            vocab_rows: ceil_to(s.vocab, hvc, 8),
            ..Default::default()
        },
        arrangement: None,
    };
    let contract = Contract {
        positions: ne * py,
        rope_levels: pr.need("NLV")?,
        rope_bits: 3,
        cap: pr.int("CAP").unwrap_or(256),
        loop_len: pr.need("LOOP")?,
        token_loop: pr.int("WL").is_none_or(|v| v != 0),
        logit_scale: pr.float("HSCALE").unwrap_or(1.0) as f32,
    };
    Ok(Plan {
        shape: s.clone(),
        partition,
        contract,
        levers: pr.levers.clone(),
    })
}
