use model_ir::{Attention, Operands, Operation, StructKind, Trace, Ty, ValueId};

use crate::error::{Fault, Result};

pub use model_exec::store::kv::{
    Geometry, Paging, Reader, Seat, SpaceFacts, indptr, pad_indptr, reads, row_of, space_of,
};

pub fn geometry(paging: &Paging, seats: &[Seat]) -> Result<Geometry> {
    Ok(model_exec::store::kv::geometry(paging, seats)?)
}

pub fn geometry_with(paging: &Paging, seats: &[Seat], tables: &[&[u32]]) -> Result<Geometry> {
    Ok(model_exec::store::kv::geometry_with(paging, seats, tables)?)
}

pub fn width_of(trace: &Trace, value: ValueId) -> Result<u64> {
    Ok(model_exec::store::kv::width_of(trace, value)?)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScheduleFacts {
    pub kind: StructKind,
    pub reading: SpaceFacts,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Facts {
    pub rows: Vec<Option<SpaceFacts>>,
    pub plans: Vec<Option<ScheduleFacts>>,
}

impl Facts {
    #[must_use]
    pub fn row(&self, at: usize) -> Option<SpaceFacts> {
        self.rows.get(at).copied().flatten()
    }
}

pub fn probe(trace: &Trace) -> Result<Facts> {
    let mut out = Facts {
        rows: vec![None; trace.caches.len()],
        plans: vec![None; trace.values.len()],
    };

    for node in &trace.nodes {
        let Some(read) = reads(&node.op) else {
            continue;
        };
        let Some(row) = row_of(trace, read.cache) else {
            continue;
        };
        let seat = out.rows.get_mut(row).ok_or_else(|| Fault::Unbound {
            what: format!("cache row {row}, which this plan does not declare"),
        })?;
        match seat {
            None => {
                *seat = Some(SpaceFacts {
                    head_dim: read.head_dim,
                    kv_heads: read.kv_heads.unwrap_or(0),
                    q_heads: 0,
                    window: None,
                });
            }
            Some(known) => {
                if known.head_dim != read.head_dim {
                    return Err(Fault::Unbound {
                        what: format!(
                            "cache row {row}, whose readers disagree about its shape: \
                             head_dim {} against head_dim {}",
                            known.head_dim, read.head_dim
                        ),
                    });
                }
                if let Some(heads) = read.kv_heads {
                    if known.kv_heads != 0 && known.kv_heads != heads {
                        return Err(Fault::Unbound {
                            what: format!(
                                "cache row {row}, whose readers state {} and {heads} kv heads",
                                known.kv_heads
                            ),
                        });
                    }
                    known.kv_heads = heads;
                }
            }
        }
    }

    for node in &trace.nodes {
        let Some(carve) = carves(&node.op) else {
            continue;
        };
        let kind = kind_of(trace, carve.plan, node.op.name())?;
        let seat = out
            .plans
            .get_mut(carve.plan.0 as usize)
            .ok_or_else(|| Fault::Unbound {
                what: format!(
                    "plan value {}, which this plan does not declare",
                    carve.plan.0
                ),
            })?;
        *seat = Some(ScheduleFacts {
            kind,
            reading: carve.reading,
        });
    }

    for node in &trace.nodes {
        if let Some(read) = reads(&node.op) {
            let width = width_of(trace, read.q)?;
            if read.head_dim == 0 || width % u64::from(read.head_dim) != 0 {
                return Err(Fault::Unbound {
                    what: format!(
                        "plan value {}, whose query rectangle is {width} wide and whose head \
                         width is {} — not a whole number of heads",
                        read.plan.0, read.head_dim
                    ),
                });
            }
            agrees(
                &out,
                read.plan,
                node.op.name(),
                Restated {
                    head_dim: read.head_dim,
                    kv_heads: read.kv_heads,
                    q_heads: u32::try_from(width / u64::from(read.head_dim)).unwrap_or(u32::MAX),
                    window: read.window,
                },
            )?;
        }
        if let Some(read) = latents(&node.op) {
            agrees(
                &out,
                read.plan,
                node.op.name(),
                Restated {
                    head_dim: read.kv_lora_rank,
                    kv_heads: None,
                    q_heads: read.heads,
                    window: None,
                },
            )?;
        }
    }
    Ok(out)
}

struct Carving {
    plan: ValueId,
    reading: SpaceFacts,
}

fn carves(op: &Operation) -> Option<Carving> {
    let Operation::Attention(op) = op else {
        return None;
    };
    match op {
        Attention::PlanDecode {
            q_heads,
            kv_heads,
            head_dim,
            window,
            plan,
            ..
        }
        | Attention::PlanPrefill {
            q_heads,
            kv_heads,
            head_dim,
            window,
            plan,
            ..
        } => Some(Carving {
            plan: *plan,
            reading: SpaceFacts {
                head_dim: *head_dim,
                kv_heads: *kv_heads,
                q_heads: *q_heads,
                window: *window,
            },
        }),
        Attention::MlaPlan {
            heads,
            kv_lora_rank,
            plan,
            ..
        } => Some(Carving {
            plan: *plan,
            reading: SpaceFacts {
                head_dim: *kv_lora_rank,
                kv_heads: 0,
                q_heads: *heads,
                window: None,
            },
        }),
        _ => None,
    }
}

fn kind_of(trace: &Trace, value: ValueId, carver: &'static str) -> Result<StructKind> {
    let declared = trace.values.get(value.0 as usize).map(|decl| &decl.ty);
    let Some(Ty::Struct(kind)) = declared else {
        return Err(Fault::Unbound {
            what: format!(
                "plan value {}, which the plan op `{carver}` carves a schedule into though it \
                 declares no host struct — a schedule's kind is what says which builder runs \
                 and how much workspace that builder stages",
                value.0
            ),
        });
    };
    Ok(*kind)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Restated {
    head_dim: u32,
    kv_heads: Option<u32>,
    q_heads: u32,
    window: Option<u32>,
}

fn agrees(facts: &Facts, plan: ValueId, launch: &'static str, restated: Restated) -> Result<()> {
    let schedule = facts
        .plans
        .get(plan.0 as usize)
        .copied()
        .flatten()
        .ok_or_else(|| Fault::Unbound {
            what: format!(
                "plan value {}, which the launch `{launch}` reads at {restated:?} though no \
                 plan op in this plan carves it — a schedule's reading is stated by the op \
                 that builds it, so a value nothing carved is a schedule nothing planned",
                plan.0
            ),
        })?;
    let seat = schedule.reading;
    let disagreement = if seat.head_dim != restated.head_dim {
        Some("the head width")
    } else if seat.window != restated.window {
        Some("the sliding window")
    } else if seat.q_heads != restated.q_heads {
        Some("the query heads")
    } else if restated
        .kv_heads
        .is_some_and(|heads| seat.kv_heads != heads)
    {
        Some("the kv heads")
    } else {
        None
    };
    let Some(about) = disagreement else {
        return Ok(());
    };
    Err(Fault::Unbound {
        what: format!(
            "plan value {}, whose schedule is carved for {seat:?} while the launch `{launch}` \
             restates {restated:?} — they disagree about {about}, so the launch restates a \
             reading its schedule was not carved for. A schedule is carved for ONE reading \
             (the window sizes its kv chunking, the head width its tile), so the model text \
             states the second reading on a second plan op rather than pointing a second \
             reader at this one",
            plan.0
        ),
    })
}

struct LatentReader {
    plan: ValueId,
    heads: u32,
    kv_lora_rank: u32,
}

fn latents(op: &Operation) -> Option<LatentReader> {
    let Operation::Attention(op) = op else {
        return None;
    };
    match op {
        Attention::MlaDecode {
            plan,
            heads,
            kv_lora_rank,
            ..
        }
        | Attention::MlaPrefill {
            plan,
            heads,
            kv_lora_rank,
            ..
        }
        | Attention::MlaDecodeSelected {
            plan,
            heads,
            kv_lora_rank,
            ..
        }
        | Attention::MlaPrefillSelected {
            plan,
            heads,
            kv_lora_rank,
            ..
        } => Some(LatentReader {
            plan: *plan,
            heads: *heads,
            kv_lora_rank: *kv_lora_rank,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rows_heads_are_read_off_the_ops_that_restate_them() {
        let trace = models::sku("qwen35-d0.8b-bf16-kv-bf16")
            .expect("the catalog ships the smoke's SKU")
            .trace;
        let plan = trace(model_dsl::Platform::Cuda);
        let facts = probe(&plan).expect("a hybrid SKU's caches read");

        let stated: Vec<SpaceFacts> = facts.rows.iter().flatten().copied().collect();
        assert!(!stated.is_empty(), "qwen3.5 declares kv rows");
        for row in &stated {
            assert_eq!(row.head_dim, 256);
            assert_eq!(row.kv_heads, 2);
        }

        let mut readings: Vec<SpaceFacts> = facts
            .plans
            .iter()
            .flatten()
            .map(|schedule| schedule.reading)
            .collect();
        readings.dedup();
        assert_eq!(
            readings,
            vec![
                SpaceFacts {
                    head_dim: 256,
                    kv_heads: 2,
                    q_heads: 8,
                    window: None,
                };
                readings.len()
            ],
        );

        let kinds: Vec<StructKind> = facts
            .plans
            .iter()
            .flatten()
            .map(|schedule| schedule.kind)
            .collect();
        assert!(
            kinds.contains(&StructKind::AttnDecodePlan),
            "qwen3.5 carves a decode schedule: {kinds:?}"
        );
        assert!(
            kinds.contains(&StructKind::AttnPrefillPlan),
            "and prefill ones: {kinds:?}"
        );

        let states = plan
            .caches
            .iter()
            .filter(|row| matches!(row, model_ir::CacheRow::State { .. }))
            .count();
        assert_eq!(
            states, 36,
            "18 gdn layers, a conv bank and a delta bank each"
        );
    }
}
