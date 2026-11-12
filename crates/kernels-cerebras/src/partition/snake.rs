//! The snake: Qwen3.5 (GDN layers + gated full attention: 9B / 4B / 0.8B), Qwen3 and Llama 3
//! (dense attention, no output gate: Qwen3-8B, Llama 3.1 8B) as they plan: a snake of PX x PY
//! units (at most one gap slot, whose PEs relay), the scores segments on every weight row, GDN's
//! (k, q) multicast and the o-norm segments where there are GDN layers, the head east of the
//! units, the token loop. Kernels: [`crate::kernels::snake`]; cut: [`crate::tiles::snake`].

use crate::compose::snake_gap;
use crate::partition::units_of;
use crate::plan::{
    Contract, HeadPartition, KvEntries, LayerKind, Partition, Plan, Preset, ReduceGroups, RowRun,
    Segments, Shape,
};

type R<T> = Result<T, String>;

/// The keys a plan's preset gives: weight columns and rows a unit (PX, PY), the unit slots (UNX x
/// UNY; the units are the layers over K, one short of the slots leaves a gap slot), layers a unit
/// (K), the head's rows and vocabulary rows a head PE (HR, VR), KV entries a row (NE).
///
/// Units of K layers, every unit the same kinds (GDN and gated attention, or Qwen3's plain
/// attention); a row's dims the hidden over PY (zero-padded past it); a scores segment a KV
/// head's columns, GDN's a key head's, the o-norm's a value head's (Qwen3: the scores'); the
/// head PY / 2 columns; the RoPE tables enough levels for the context (`CTX`, the positions
/// asked for, where fewer than NE PY). Overrides: the
/// reduction groups (`G2`, `G2X` / `G2Y` / `G2H`, `G2W`), the RoPE tables' levels and digit
/// bits (`RLV`, `RBITS`), `CAP`, `LOOP`, the token loop (`WL`).
pub fn plan(s: &Shape, pr: &Preset) -> R<Plan> {
    let (px, py, k) = (pr.need("PX")?, pr.need("PY")?, pr.need("K")?);
    let unit_grid = (pr.need("UNX")?, pr.need("UNY")?);
    let units = units_of(&s.layers, k, "snake")?;
    snake_gap(units, unit_grid)?;
    let pattern = &s.layers[..k as usize];
    if s.layers.chunks(k as usize).any(|u| u != pattern) {
        return Err("snake: every unit the same layer kinds".into());
    }
    let per = |heads: u32, what: &str| {
        if heads == 0 || !px.is_multiple_of(heads) {
            return Err(format!("snake: {heads} {what} over {px} columns"));
        }
        Ok(px / heads)
    };
    // a head's dims split evenly over its columns
    let split = |dims: u32, cols: u32, what: &str| {
        if dims.is_multiple_of(cols) {
            Ok(())
        } else {
            Err(format!("snake: {what}'s {dims} dims over {cols} columns"))
        }
    };
    let scores = per(s.attn.kv_heads, "KV heads")?;
    split(s.attn.head_dim, scores, "an attention head")?;
    let (gdn, onorm) = match s.gdn {
        Some(g) => {
            let (gdn, onorm) = (
                per(g.key_heads, "key heads")?,
                per(g.value_heads, "value heads")?,
            );
            split(g.key_dim, gdn, "a key head")?;
            split(g.value_dim, onorm, "a value head")?;
            (gdn, onorm)
        }
        None => (scores, scores),
    };
    // the MLP's columns split evenly over the unit's (the cut has no padded column)
    if !s.mlp.intermediate.is_multiple_of(px) {
        return Err(format!(
            "snake: the MLP's {} over {px} columns",
            s.mlp.intermediate
        ));
    }
    let g2 = pr.int("G2").unwrap_or(0);
    let ne = pr.need("NE")?;
    // a position is RLV digits of RBITS bits: enough levels for the context NE PY (at least 4)
    let rope_bits = pr.int("RBITS").unwrap_or(3);
    let reach = pr.int("CTX").map_or(ne * py, |c| c.min(ne * py));
    let rope_levels = pr
        .int("RLV")
        .unwrap_or(4.max((32 - (reach - 1).leading_zeros()).div_ceil(rope_bits)));
    let partition = Partition {
        units,
        layers_per_unit: k,
        unit_grid: Some(unit_grid),
        rows: py,
        cols: px,
        row_dims: vec![RowRun {
            rows: (0, py),
            dims: s.hidden.div_ceil(py),
            attention: pattern.iter().any(|l| !matches!(l, LayerKind::Gdn)),
        }],
        kv_rows: py,
        kv_entries: KvEntries {
            per_row: ne,
            ..Default::default()
        },
        reduce: ReduceGroups {
            row: pr.int("G2X").unwrap_or(g2),
            col: pr.int("G2Y").unwrap_or(g2),
            head: pr.int("G2H").unwrap_or(g2),
            token_loop: pr.int("G2W").unwrap_or(24),
        },
        segments: Segments {
            scores,
            gdn,
            onorm,
            ..Default::default()
        },
        attn_cols: None,
        head: HeadPartition {
            cols: py / 2,
            rows: pr.need("HR")?,
            vocab_rows: pr.need("VR")?,
            ..Default::default()
        },
        arrangement: None,
    };
    let contract = Contract {
        // bounded by the CY's RoPE tables: RLV digits of RBITS bits
        positions: (ne * py).min(1 << (rope_bits * rope_levels).min(31)),
        rope_levels,
        rope_bits,
        cap: pr.int("CAP").unwrap_or(256),
        loop_len: pr.int("LOOP").unwrap_or(32),
        token_loop: pr.int("WL").unwrap_or(0) != 0,
        logit_scale: 1.0,
    };
    Ok(Plan {
        shape: s.clone(),
        partition,
        contract,
        levers: pr.levers.clone(),
    })
}
