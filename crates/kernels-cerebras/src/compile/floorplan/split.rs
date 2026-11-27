//! The attention | MoE halves constructor (gpt-oss, Qwen3 MoE): units of AX attention + MX MoE
//! columns over PY rows, snaked in two unit columns starting upward so the first and last unit
//! sit on the band; the head is a band below the units, with the token loop over it.
//! Kernels: [`crate::compile::kernels::split`]; cut: [`crate::compile::tiles::split`].

use crate::compile::floorplan::ceil_to;
use crate::compile::plan::{
    Contract, HeadPartition, KvEntries, Partition, Plan, Preset, ReduceGroups, RowRun, Segments,
    Shape,
};

type R<T> = Result<T, String>;

/// Preset keys: attention and MoE columns a unit (AX, MX), rows (PY), unit slots (UNX x UNY),
/// scores and router segments (SWA, SWR), band vocabulary columns (HVC), KV entries a row (NE),
/// RoPE levels (NLV), reduction groups (G2, G2W), logits prescale (HSCALE), posted loop (LOOP).
/// Overrides: `NE`, `G2`, `G2W`, `G3W`, `CAP`, `LOOP`, `WL` (the token loop; on unless 0), `SEATS`.
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
    // the cut's and the kernels' divisibility: rotated kv pairs a column, an even share of each
    // expert, the router's pick tree over blocks of 8 experts, MXFP4's two codes a word
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
    // a ring of seats shares the full KV pool, NE PY: a sequence holds what the others' least
    // windows (a launch's laps, whole rows) leave
    let seats = pr.int("SEATS").unwrap_or(1).max(1);
    let cap = pr.int("CAP").unwrap_or(256);
    let least = (cap / seats).max(1).div_ceil(py) * py;
    if seats > ne || (seats - 1) * least >= ne * py {
        return Err(format!("split: {seats} seats over {ne} KV entries a row"));
    }
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
            token_loop3: pr.int("G3W").unwrap_or(0),
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
        positions: ne * py - (seats - 1) * least,
        rope_levels: pr.need("NLV")?,
        rope_bits: 3,
        cap,
        loop_len: pr.need("LOOP")?,
        token_loop: pr.int("WL").is_none_or(|v| v != 0),
        logit_scale: pr.float("HSCALE").unwrap_or(1.0) as f32,
    };
    Ok(Plan {
        shape: s.clone(),
        partition,
        contract,
        levers: pr.levers.clone(),
        seats,
    })
}
