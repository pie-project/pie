//! The prep's meta (`<prefix>meta.json`), typed: every key the families, the host and the tools
//! read, plus the overrides `PIE_CEREBRAS_SPLIT` applies. Replaces the untyped
//! `serde_json::Map` reads of `mod.rs::SplitSpec` (`int`, `need`, the `NEX` / `geo` / `RA`
//! sniffing) and the meta writers of the preps.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::spatial::split::plan::{FamilyId, Plan};

/// The key holding a whole [`Plan`] (a meta written by [`Meta::of_plan`]): a family reads it back
/// as is, which is how a plan made elsewhere (the spatial compiler) reaches the build.
pub const PLAN_KEY: &str = "plan";

/// The meta as stored: known keys typed by the family readers through [`Meta::int`] etc.;
/// the whole map is kept so unknown keys round-trip unchanged.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Meta {
    pub map: BTreeMap<String, serde_json::Value>,
}

impl Meta {
    /// Reads `<dir>/<prefix>meta.json` and applies `k=v` overrides (`SplitSpec::parse`'s rules:
    /// a bool key stays bool).
    pub fn read(dir: &Path, prefix: &str, overrides: &str) -> Result<Meta, String> {
        let path = dir.join(format!("{prefix}meta.json"));
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut meta = Meta {
            map: serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?,
        };
        meta.apply(overrides)?;
        Ok(meta)
    }

    /// Applies `k=v,...` overrides (integers; a key that holds a bool stays a bool).
    pub fn apply(&mut self, overrides: &str) -> Result<(), String> {
        for kv in overrides.split(',').filter(|s| !s.is_empty()) {
            let (k, x) = kv
                .split_once('=')
                .ok_or_else(|| format!("split meta override {kv}: k=v"))?;
            let n: i64 = x.parse().map_err(|e| format!("override {kv}: {e}"))?;
            let val = match self.map.get(k) {
                Some(serde_json::Value::Bool(_)) => serde_json::Value::Bool(n != 0),
                _ => serde_json::Value::from(n),
            };
            self.map.insert(k.to_string(), val);
        }
        Ok(())
    }

    pub fn int(&self, k: &str) -> Option<u32> {
        match self.map.get(k)? {
            serde_json::Value::Bool(b) => Some(u32::from(*b)),
            v => v.as_u64().map(|x| x as u32),
        }
    }

    pub fn need(&self, k: &str) -> Result<u32, String> {
        self.int(k).ok_or_else(|| format!("split meta: no {k}"))
    }

    pub fn float(&self, k: &str) -> Option<f64> {
        self.map.get(k)?.as_f64()
    }

    pub fn str(&self, k: &str) -> Option<&str> {
        self.map.get(k)?.as_str()
    }

    /// The plan a meta carries whole ([`PLAN_KEY`]), if it does.
    pub fn plan(&self) -> Option<Result<Plan, String>> {
        let v = self.map.get(PLAN_KEY)?;
        Some(serde_json::from_value(v.clone()).map_err(|e| format!("split meta plan: {e}")))
    }

    /// The family the meta was prepped for: an explicit `family` key, else today's sniffing
    /// (`NEX`: gpt-oss; `geo` + `RA`: Gemma 12B; `geo`: Gemma; otherwise Qwen).
    pub fn family(&self) -> FamilyId {
        if let Some(f) = self.str("family").and_then(FamilyId::of_name) {
            return f;
        }
        if self.map.contains_key("NEX") {
            FamilyId::GptOss
        } else if self.map.contains_key("geo") {
            if self.map.contains_key("RA") {
                FamilyId::Gemma12
            } else {
                FamilyId::Gemma
            }
        } else {
            FamilyId::Qwen
        }
    }

    /// The meta a plan states, for the tools: the `family` key, the whole plan ([`PLAN_KEY`]),
    /// and the flat keys every reader of a prep's meta knows (the host's and `link.sh`'s), so a
    /// tool that reads only those still works. Family-specific prep keys are the preps' to write.
    pub fn of_plan(plan: &Plan) -> Meta {
        let (s, p, c) = (&plan.shape, &plan.partition, &plan.contract);
        let mut m = BTreeMap::new();
        let mut put = |k: &str, v: serde_json::Value| {
            m.insert(k.to_string(), v);
        };
        put("family", s.family.name().into());
        put(
            PLAN_KEY,
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
        Meta { map: m }
    }
}
