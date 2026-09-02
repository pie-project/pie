//! What the plan says about its own caches: rows' readings and the schedules
//! plan ops carve. Page arithmetic lives in [`model_exec::store::kv`] and is
//! re-exported below; `probe` stays shell-local (three passes, including the
//! latent/mla arms) since this shell reads a schedule off the op that carves
//! it, unlike the Metal shell.

use model_ir::{Attention, Operands, Operation, StructKind, Trace, Ty, ValueId};

use crate::error::{Fault, Result};

pub use model_exec::store::kv::{
    Geometry, Paging, Reader, Seat, SpaceFacts, indptr, pad_indptr, reads, row_of, space_of,
};

/// Compute one fire's geometry — [`model_exec::store::kv::geometry`], in this
/// shell's `Result`.
///
/// # Errors
///
/// [`Fault::Ceiling`] for a lane past the pool's slots or past its slot's
/// page block.
pub fn geometry(paging: &Paging, seats: &[Seat]) -> Result<Geometry> {
    Ok(model_exec::store::kv::geometry(paging, seats)?)
}

/// Compute one fire's geometry against a caller-supplied page table —
/// [`model_exec::store::kv::geometry_with`], in this shell's `Result`.
///
/// # Errors
///
/// [`Fault::Ceiling`] for a lane past the pool's slots (shell-owned only),
/// past its slot's page block (shell-owned only), or — caller-owned — one
/// whose stated pages do not cover the tokens it is about to hold.
pub fn geometry_with(paging: &Paging, seats: &[Seat], tables: &[&[u32]]) -> Result<Geometry> {
    Ok(model_exec::store::kv::geometry_with(paging, seats, tables)?)
}

/// A value's row width — [`model_exec::store::kv::width_of`], in this shell's
/// `Result`.
///
/// # Errors
///
/// [`Fault::Unbound`] for a value its own plan does not declare, for one that
/// declares a host struct rather than a rectangle, and for one whose width
/// carries a symbolic dim.
pub fn width_of(trace: &Trace, value: ValueId) -> Result<u64> {
    Ok(model_exec::store::kv::width_of(trace, value)?)
}

/// One attention schedule as its plan op states it: which struct it defines
/// (the builder that runs) and the reading it is carved for. Both are needed:
/// two schedules can share a [`SpaceFacts`] reading yet want workspaces of
/// very different sizes, since the buffer each builder stages is shaped by
/// its own kernel's grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScheduleFacts {
    /// The host struct the plan op's output value declares.
    pub kind: StructKind,
    /// The reading that schedule is carved for.
    pub reading: SpaceFacts,
}

/// Everything the plan restates about its own caches, keyed two ways.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Facts {
    /// Per `Trace::caches` ROW: what the paged launches that walk it restate
    /// about it. `None` for a row no paged launch reads — which is a row
    /// allocated as declared and checked against nothing, not a refusal.
    pub rows: Vec<Option<SpaceFacts>>,
    /// Per `Trace::values` ID: the SCHEDULE one attention plan op carves —
    /// which struct it is and the reading it is carved for, as that op states
    /// them. `None` for every value no plan op defines.
    pub plans: Vec<Option<ScheduleFacts>>,
}

impl Facts {
    /// What the paged launches restate about one cache row, or `None` for a
    /// row none of them reads (a lookup, not a refusal: the row's bytes come
    /// off its declaration regardless).
    #[must_use]
    pub fn row(&self, at: usize) -> Option<SpaceFacts> {
        self.rows.get(at).copied().flatten()
    }
}

/// Read every cache row's and every attention schedule's facts off the plan:
/// a row's off the launches that walk it, a schedule's off the plan op that
/// carves it — and then check every launch's restatement against the schedule
/// it reads.
///
/// # Errors
///
/// [`Fault::Unbound`] for a cache row two ops give two widths, for a plan op
/// whose output value declares no host struct, for a launch that restates a
/// reading its schedule was not carved for, for a launch over a plan value no
/// plan op carved, or for a q rectangle that is not a whole number of heads
/// wide.
///
pub fn probe(trace: &Trace) -> Result<Facts> {
    let mut out = Facts {
        rows: vec![None; trace.caches.len()],
        plans: vec![None; trace.values.len()],
    };

    // Pass one: the rows. kv_heads is stated by the prefill arms alone, so a
    // row only a decode reads carries zero.
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

    // Pass two: the schedules, off the ops that carve them (not off whoever
    // consumes the value). Also records the declared struct kind, since the
    // reading alone does not say which builder runs.
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

    // Pass three: every launch restates its share of the reading it was
    // handed; a restatement the schedule was not carved for is a refusal.
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

/// One plan op's own statement: the value it defines and the reading it
/// carves that schedule for.
struct Carving {
    plan: ValueId,
    reading: SpaceFacts,
}

/// The three plan ops, as one shape.
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
        // A latent schedule is carved in the absorbed reading: every query
        // head reads the one shared latent plane and writes kv_lora_rank
        // floats, so rank is the head width. No kv head count (plane is
        // shared, not per kv head) and no window (no sliding spans).
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

/// The host struct a plan op's output value declares. Read off the value
/// rather than the op: one op family builds either an `AttnPrefillPlan` or
/// an `AttnPrefillPlanSm90`, and the value's `Ty` says which.
///
/// # Errors
///
/// [`Fault::Unbound`] for a plan op whose output declares a rectangle rather
/// than a host struct: nothing can build such a value, and nothing can say what
/// workspace building it would want.
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

/// What one launch restates about the schedule it reads. `kv_heads` is `None`
/// for every launch that states none — the decode and masked arms, and the
/// latent ones over their shared plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Restated {
    head_dim: u32,
    kv_heads: Option<u32>,
    q_heads: u32,
    window: Option<u32>,
}

/// Check one launch's restatement against the seat of the plan it reads.
///
/// # Errors
///
/// [`Fault::Unbound`] naming the plan value, the launch and both readings —
/// including the case where the value carries no seat at all, because a launch
/// over a schedule no op carved reads a carving nobody stated.
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

/// What one latent (mla) launch restates: `q` here is the absorbed query
/// (already mapped into latent space), so its width is `heads x
/// kv_lora_rank`.
struct LatentReader {
    plan: ValueId,
    heads: u32,
    kv_lora_rank: u32,
}

/// The four latent launches, as one shape. They do not feed the row pass:
/// `CacheRow::Kv` already states a latent cache's pages, and no latent op
/// restates a kv head count.
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
        // Pins probe() against a real hybrid SKU (qwen3.5): row/schedule
        // facts, and that state (recurrent) caches are excluded.
        let trace = models::sku("qwen35-d0.8b-bf16-kv-bf16")
            .expect("the catalog ships the smoke's SKU")
            .trace;
        let plan = trace(model_dsl::Platform::Cuda);
        let facts = probe(&plan).expect("a hybrid SKU's caches read");

        // qwen's 18 attention layers each declare a kv row.
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

        // recurrent (state) rows are excluded: space_of answers None for them
        let states = plan
            .caches
            .iter()
            .filter(|row| matches!(row, model_ir::CacheRow::State { .. }))
            .count();
        assert_eq!(states, 36, "18 gdn layers, a conv bank and a delta bank each");
    }
}
