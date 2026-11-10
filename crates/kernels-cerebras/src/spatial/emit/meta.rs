//! The build's record of its plan (`meta.json`): the plan whole and its cut as flat keys, for the
//! Python runners. Nothing reads a plan back out of it.

use std::collections::BTreeMap;

use crate::spatial::plan::Plan;

/// The record of `plan` (`meta.json`, for the Python runners): the family, the whole plan
/// (`plan`), and its cut as flat keys.
pub fn of_plan(plan: &Plan) -> BTreeMap<String, serde_json::Value> {
    let (s, p, c) = (&plan.shape, &plan.partition, &plan.contract);
    let mut m = BTreeMap::new();
    let mut put = |k: &str, v: serde_json::Value| {
        m.insert(k.to_string(), v);
    };
    put("family", plan.family.name().into());
    put(
        "plan",
        serde_json::to_value(plan).unwrap_or(serde_json::Value::Null),
    );
    put("H", s.hidden.into());
    put("V", s.vocab.into());
    put("EPS", f64::from(s.norm.eps).into());
    put("K", p.layers_per_unit.into());
    put("PX", p.cols.into());
    put("PY", p.rows.into());
    put("NE", p.kv_entries.per_row.into());
    put("PYW", p.kv_rows.into());
    if let Some(g) = p.kv_entries.global {
        put("NEG", g.into());
    }
    if let Some(n) = p.kv_entries.sliding {
        put("NES", n.into());
    }
    put("HC", p.head.cols.into());
    put("HR", p.head.rows.into());
    put("VR", p.head.vocab_rows.into());
    put("G2X", p.reduce.row.into());
    put("G2Y", p.reduce.col.into());
    put("G2H", p.reduce.head.into());
    put("G2W", p.reduce.token_loop.into());
    put("RLV", c.rope_levels.into());
    put("RBITS", c.rope_bits.into());
    put("CAP", c.cap.into());
    put("LOOP", c.loop_len.into());
    put("WL", c.token_loop.into());
    put("POS", c.positions.into());
    if let Some(cap) = s.head.final_softcap {
        put("SOFTCAP", f64::from(cap).into());
    }
    m
}
