//! Lower a supergraph plan: sweep the fact words, dedup surviving
//! behaviors into lanes, and join role points against the plane's claims.

use model_ir::kernels::Backend;
use model_ir::plan::Plan;

pub struct Lowered {
    pub lanes: Vec<Lane>,
    pub resolution: Resolution,
}

/// One behavior class: the fact words it serves and the ops that survive.
pub struct Lane {
    pub words: Vec<u64>,
    pub ops: Vec<u32>,
}

pub struct Resolution {
    /// role point -> the plane symbol its claim answers with.
    pub resolved: Vec<(String, String)>,
    /// Stated role points no routine of this plane claims — the backlog.
    pub unresolved: Vec<String>,
    /// A plane-gated symbol stated on the wrong plane; a refused plan.
    pub violations: Vec<String>,
}

#[must_use]
pub fn lower(plan: &Plan) -> Lowered {
    Lowered {
        lanes: lanes(plan),
        resolution: resolve(plan),
    }
}

#[must_use]
pub fn lanes(plan: &Plan) -> Vec<Lane> {
    let facts = plan.facts.len();
    assert!(facts <= 20, "a plan over {facts} facts");
    let mut lanes: Vec<Lane> = Vec::new();
    for word in 0..1u64 << facts {
        let ops: Vec<u32> = plan
            .ops
            .iter()
            .enumerate()
            .filter(|(_, op)| op.cond.holds(word))
            .map(|(i, _)| i as u32)
            .collect();
        match lanes.iter_mut().find(|lane| lane.ops == ops) {
            Some(lane) => lane.words.push(word),
            None => lanes.push(Lane { words: vec![word], ops }),
        }
    }
    lanes
}

#[must_use]
pub fn resolve(plan: &Plan) -> Resolution {
    let mut resolution = Resolution {
        resolved: Vec::new(),
        unresolved: Vec::new(),
        violations: Vec::new(),
    };
    let mut seen: Vec<&str> = Vec::new();
    for op in &plan.ops {
        let kernel = op.kernel.as_str();
        if seen.contains(&kernel) {
            continue;
        }
        seen.push(kernel);
        if let Some(symbol) = kernel.strip_prefix("cuda::") {
            if plan.plane == Backend::Cuda {
                resolution.resolved.push((kernel.to_string(), symbol.to_string()));
            } else {
                resolution.violations.push(kernel.to_string());
            }
            continue;
        }
        if model_ir::kernels::point_claims(plan.plane).contains(&kernel) {
            resolution.resolved.push((kernel.to_string(), format!("points::{kernel}")));
            continue;
        }
        match model_ir::kernels::canon_symbol(plan.plane, kernel) {
            Some(symbol) => resolution.resolved.push((kernel.to_string(), symbol.to_string())),
            None => resolution.unresolved.push(kernel.to_string()),
        }
    }
    resolution
}
