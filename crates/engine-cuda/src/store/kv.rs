//! What the plan says about its own caches: the rows' readings and the
//! schedules the plan ops carve.
//!
//! **THE PAGE ARITHMETIC IS NO LONGER HERE.** `Paging`, `Seat`, `Geometry`,
//! `geometry`/`geometry_with` and `indptr` — every item this file used to
//! carry a `// model_exec::store candidate` marker over — live in
//! [`model_exec::store::kv`], byte-identical on both shells and host-tested
//! there. What is re-exported below is that module, spelled at the path this
//! shell's callers already use; what is written out below it is the half that
//! did NOT survive the merge, because the two shells' probes disagree.
//!
//! # Why this shell's `probe` stayed
//!
//! The Metal twin reads a schedule's reading off the LAUNCHES that consume
//! it (two passes, refusing when two launches disagree). This one reads it
//! off the plan op that CARVES it (three passes), keeps a [`StructKind`]
//! beside the reading because the reading alone does not say which builder
//! runs, and walks the latent (mla) arms the other shell has no kernels for.
//! That is a behaviour difference rather than a spelling one, so it is stated
//! twice on purpose until the Metal plane grows the same ops.
//!
//! # The three questions
//!
//! A paged kv space answers three, and confusing them is how a cache silently
//! reads somebody else's tokens:
//!
//! ```text
//! which pages does lane L own?     the PAGING — static, per slot, per load
//! how much of them is live?        kv_len / last_page_len — per fire, per lane
//! where does token T land?         write_page / write_offset — per fire, per row
//! ```
//!
//! The first is a deployment's budget. The second is what the attention
//! schedule is planned against. The third is what the append kernel writes
//! through, and it is stated per token rather than derived from a position
//! because a derivation cannot spell a fresh-page write that is not the page
//! run's tail (`kernels/attn/kv.cuh`, the explicit-descriptor writer).

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

/// One attention schedule as its plan op states it: which struct it defines —
/// the builder that runs, and the workspace the builder wants — and the
/// reading it is carved for.
///
/// The two are not the same question, and the shell needs both. A latent
/// schedule and a paged one can be carved for readings that look alike in
/// [`SpaceFacts`] and still want workspaces two orders of magnitude apart,
/// because the buffer each builder stages is shaped by its own kernels'
/// grid ([`Inputs::reserve`](crate::inputs::Inputs::reserve)).
///
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScheduleFacts {
    /// The host struct the plan op's output value declares.
    pub kind: StructKind,
    /// The reading that schedule is carved for.
    pub reading: SpaceFacts,
}

/// Everything the plan restates about its own caches, keyed two ways.
///
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
    /// row none of them reads.
    ///
    /// A LOOKUP AND NOT A REFUSAL: the row's bytes come off its declaration,
    /// so a row nothing restates is a row with nothing to check — the latent,
    /// index and pool launches walk their pages without ever naming a kv head
    /// count, and refusing them was the whole of the M22 finding.
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

    // Pass one: the rows. `kv_heads` is stated by the prefill arms alone, so
    // a row only a decode reads carries zero — the reading this shell has
    // always had, kept.
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

    // Pass two: the schedules, off the ops that CARVE them. A plan value is
    // carved for ONE reading — query heads, kv heads, head width, window — and
    // the op that builds it is where that reading has an author, so this pass
    // reads it there rather than inferring it from whoever happens to consume
    // the value. Beside the reading it takes the value's declared STRUCT,
    // because the reading alone does not say which builder runs: an absorbed
    // latent schedule and a paged one both state a head width, and the two
    // stage workspaces of entirely different shapes.
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

    // Pass three: every launch restates its share of the reading it was handed
    // — the paged ones a head width, a window, a kv head count (the prefill
    // arms) and a q rectangle whose width names its query heads, the latent
    // ones their `heads` and `kv_lora_rank` outright — and a restatement the
    // schedule was not carved for is a refusal rather than a schedule read at
    // the wrong tile.
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
        // A latent schedule is carved in the ABSORBED reading, which is the
        // only one its kernels have: every query head reads the one shared
        // latent plane and writes `kv_lora_rank` floats, so the rank IS the
        // head width the planner sizes at (`plan_mla`'s `head_dim_o`, which
        // sizes the split-kv partial buffer at one packed row per (token,
        // head) — `kernels-cuda/src/attn/sched_mla.rs`, and `mla_fa2::pack`'s
        // `o_stride_h = rank` beside it). There is no second kv head count to
        // state — the plane is shared, not per kv head — and no window: the
        // latent kernels carve no sliding spans.
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

/// The host struct a plan op's output value declares.
///
/// It is read off the VALUE rather than off the op because that is where the
/// trace states it — one op family builds two of them (`attention.plan_prefill`
/// mints an `AttnPrefillPlan` or, where the trace asks for the sm90 kernels, an
/// `AttnPrefillPlanSm90`) and the value's `Ty` is what says which, the same way
/// `Run::declared` reads it at build time.
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

/// What one latent (mla) launch restates. It states its numbers outright
/// rather than through a rectangle: `q` here is the ABSORBED query, already
/// mapped into latent space by `attention.mla_absorb_q`, so its width is the
/// same `heads x kv_lora_rank` the op names.
struct LatentReader {
    plan: ValueId,
    heads: u32,
    kv_lora_rank: u32,
}

/// The four latent launches, as one shape. They do not feed the row pass:
/// a latent cache's pages are the compressed plane and the rope plane, which
/// `CacheRow::Kv` already states, and no latent op restates a kv head count.
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
        // A `CacheRow::Kv` carries a per-token row and an element, and nothing
        // else: a row's heads and head width live on `Attention::Decode`/
        // `Prefill`, which walk it. A schedule's reading is stated by the plan
        // op that carves it, and the launches restate their share. Reading
        // both here is what keeps a pool's shape and a model's text from being
        // two opinions.
        let trace = models::trace_of("qwen35-d0.8b-bf16-kv-bf16")
            .expect("the catalog ships the smoke's SKU");
        let plan = trace(model_dsl::Platform::Cuda);
        let facts = probe(&plan).expect("a hybrid SKU's caches read");

        // qwen's 18 attention layers each declare a kv row, and they agree.
        let stated: Vec<SpaceFacts> = facts.rows.iter().flatten().copied().collect();
        assert!(!stated.is_empty(), "qwen3.5 declares kv rows");
        for row in &stated {
            assert_eq!(row.head_dim, 256);
            assert_eq!(row.kv_heads, 2);
        }

        // And one schedule of each kind, each carved for the one reading its
        // launches share.
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

        // The kind rides beside the reading, because the reading does not say
        // which builder runs: qwen's schedules share one reading and are still
        // a decode schedule beside prefill ones, and each names the struct its
        // own plan op defines.
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

        // And the recurrent rows are not in it: `space_of` answers `None` for
        // a state cache, which is what keeps the 36 gdn banks out of the page
        // arithmetic entirely.
        let states = plan
            .caches
            .iter()
            .filter(|row| matches!(row, model_ir::CacheRow::State { .. }))
            .count();
        assert_eq!(states, 36, "18 gdn layers, a conv bank and a delta bank each");
    }
}
