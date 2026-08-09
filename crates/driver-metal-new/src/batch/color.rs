//! Linear-scan colouring of activation live ranges onto a buffer pool.
//!
//! Every dispatch of the decode DAG reads its input activations from scratch
//! and writes its output there, and the pool is a handful of ping-pong
//! buffers. Which value lives in which buffer is a colouring problem over
//! live ranges: a value's interval runs from its first use to its last —
//! **extended to the end of the concurrency run its last use sits in**,
//! because the encoder drops barriers inside such a run, so a buffer last
//! read at ordinal `i` must not be rewritten by anything still running
//! alongside `i`. Overlap is inclusive, which is what makes a same-dispatch
//! write-after-read and two concurrent outputs interfere and land in
//! different buffers.
//!
//! Ported from `scratch_color.hpp`, which the C++ extracted "so a second
//! model family does not arrive with a second copy that drifts". Three
//! things do not survive:
//!
//! * **A value nobody uses still cost a buffer.** An unused value has no
//!   interval, but the C++ walked it through the allocator anyway: its
//!   `def = last = -1` never matched `free_at < def`, so every unused value
//!   took a fresh colour, inflating `colors_used` — the number the scratch
//!   region is *sized by* — by values that occupy nothing. Unused values
//!   colour to [`None`] here and cost nothing.
//! * **An ordinal past the run table silently lost its extension.** The
//!   C++ guarded the extension with `l < run_ends.size()` and fell through
//!   — precisely the case where the barrier-free-run rule goes unapplied,
//!   which is the hazard the extension exists to prevent. A use beyond the
//!   table is refused as malformed.
//! * **`hazard_free` was a flag the caller must remember to read.** The
//!   self-check is the file's honest admission that an allocator bug is a
//!   silent corruption; keeping it behind a `bool` re-creates the "did the
//!   validator run" problem. The check runs unconditionally and a detected
//!   hazard is an [`Err`] that cannot be ignored.

/// One buffer of one dispatch and the activation value it carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Use {
    /// The dispatch's DAG ordinal.
    pub ordinal: u32,
    /// The activation slot this dispatch binds the value at.
    pub bind_index: u8,
    /// The dataflow value id.
    pub value: u32,
    /// Whether this use writes the value (an output slot).
    pub is_write: bool,
}

/// Which pool buffer each value lives in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Coloring {
    /// `color[v]`: the pool buffer of value `v`, or [`None`] when nothing
    /// uses it.
    pub color: Vec<Option<u32>>,
    /// Distinct pool buffers needed — what the scratch region is sized by.
    pub colors_used: u32,
}

/// Why the live ranges could not be coloured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColoringError {
    /// A use names a value at or past `value_count`.
    ValueOutOfRange {
        /// The offending use's value id.
        value: u32,
    },
    /// A use's ordinal has no entry in the run table, so its concurrency
    /// extension cannot be applied. The C++ silently skipped the extension
    /// here — the one case the extension exists for.
    RunsTooShort {
        /// The ordinal past the table.
        ordinal: u32,
    },
    /// The self-check found two overlapping values sharing a buffer: an
    /// allocator bug, surfaced instead of corrupting an activation.
    HazardDetected {
        /// The lower value id of the pair.
        first: u32,
        /// The higher value id.
        second: u32,
    },
}

/// Colour the live ranges onto pool buffers.
///
/// `run_ends[i]` is the last ordinal of the concurrency run containing `i`,
/// or `i` itself when it runs alone; it must cover every used ordinal.
/// `no_recycle` gives every used value its own buffer — the dump/diagnostic
/// mode that preserves every intermediate and separates scratch-aliasing
/// races from in-kernel ones.
///
/// # Errors
///
/// [`ColoringError`]; see each variant.
pub fn color_live_ranges(
    uses: &[Use],
    run_ends: &[u32],
    value_count: usize,
    no_recycle: bool,
) -> Result<Coloring, ColoringError> {
    // First and last use per value, as inclusive ordinals.
    let mut interval: Vec<Option<(u32, u32)>> = vec![None; value_count];
    for use_ in uses {
        let slot = interval
            .get_mut(use_.value as usize)
            .ok_or(ColoringError::ValueOutOfRange { value: use_.value })?;
        *slot = Some(match *slot {
            None => (use_.ordinal, use_.ordinal),
            Some((def, last)) => (def.min(use_.ordinal), last.max(use_.ordinal)),
        });
    }
    // The concurrency extension: a value's last use reaches the end of the
    // run it sits in.
    for slot in interval.iter_mut().flatten() {
        let end = *run_ends
            .get(slot.1 as usize)
            .ok_or(ColoringError::RunsTooShort { ordinal: slot.1 })?;
        slot.1 = slot.1.max(end);
    }

    let mut order: Vec<usize> = (0..value_count)
        .filter(|&v| interval[v].is_some())
        .collect();
    order.sort_by_key(|&v| interval[v].map(|(def, _)| def));

    let mut color: Vec<Option<u32>> = vec![None; value_count];
    // `free_at[b]`: the ordinal after which buffer `b` is free.
    let mut free_at: Vec<u32> = Vec::new();
    for &v in &order {
        let Some((def, last)) = interval[v] else {
            continue;
        };
        let chosen = if no_recycle {
            free_at.push(last);
            free_at.len() - 1
        } else {
            // Strictly before: inclusive overlap means a buffer freed AT
            // `def` still interferes.
            match free_at.iter().position(|&free| free < def) {
                Some(buffer) => {
                    free_at[buffer] = last;
                    buffer
                }
                None => {
                    free_at.push(last);
                    free_at.len() - 1
                }
            }
        };
        color[v] = Some(chosen as u32);
    }

    // The self-check, unconditional: no two overlapping values share a
    // buffer. Quadratic in the value count, which is tens.
    for a in 0..value_count {
        let Some((def_a, last_a)) = interval[a] else {
            continue;
        };
        for b in (a + 1)..value_count {
            let Some((def_b, last_b)) = interval[b] else {
                continue;
            };
            let overlap = def_a.max(def_b) <= last_a.min(last_b);
            if overlap && color[a] == color[b] {
                return Err(ColoringError::HazardDetected {
                    first: a as u32,
                    second: b as u32,
                });
            }
        }
    }

    Ok(Coloring {
        colors_used: free_at.len() as u32,
        color,
    })
}

/// One buffer of one dispatch, and the pool colour it resolved to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchBind {
    /// The activation slot the dispatch binds.
    pub bind_index: u8,
    /// The pool buffer behind it.
    pub color: u32,
}

/// The full scratch schedule: what to bind, and how big to make it.
///
/// `per_dispatch` answers what to bind; `coloring` answers how big. Sizing
/// a pool slot needs the widest value sharing it, and a value's width is a
/// property of the VALUE -- a routed model's expert stack is
/// `experts_per_token` times taller than the dense tensor beside it, and
/// there is no way to see that from a bind index. That is why the colouring
/// travels with the bind table instead of being consumed by it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScratchSchedule {
    /// `per_dispatch[ordinal]`: the scratch binds of that dispatch.
    pub per_dispatch: Vec<Vec<ScratchBind>>,
    /// The colouring the binds came from (`color[value]`, `colors_used`).
    pub coloring: Coloring,
}

/// Why a schedule was not produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduleError {
    /// The colouring itself refused; hazards land here and remain
    /// unignorable -- the C++ carried a `hazard_free` flag the encoder had
    /// to remember to read.
    Coloring(ColoringError),
    /// A use's ordinal is at or past the DAG size, so it has no dispatch to
    /// fan out to. The C++ indexed the table with it anyway.
    OrdinalPastDag {
        /// The offending ordinal.
        ordinal: u32,
        /// The DAG's dispatch count.
        dag_size: usize,
    },
}

impl std::fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScheduleError::Coloring(err) => write!(f, "scratch schedule: {err:?}"),
            ScheduleError::OrdinalPastDag { ordinal, dag_size } => write!(
                f,
                "scratch schedule: a use at ordinal {ordinal} in a {dag_size}-dispatch DAG"
            ),
        }
    }
}

impl std::error::Error for ScheduleError {}

/// Colour a family's dataflow and fan the result out per dispatch.
///
/// Ported from `model/family_coloring.hpp`, minus its template: the C++
/// took any `Use`-shaped type because every family declared its own struct
/// with identical fields, and widening them was half the adapter. Rust
/// families produce [`Use`] itself, so what remains is the part that was
/// always real -- the fan-out from a per-value colouring to the per-dispatch
/// bind lists the encoder walks.
///
/// # Errors
///
/// [`ScheduleError`]; a use past the DAG is refused before the colouring's
/// answer is fanned out.
pub fn schedule_scratch(
    dag_size: usize,
    uses: &[Use],
    run_ends: &[u32],
    value_count: usize,
    no_recycle: bool,
) -> Result<ScratchSchedule, ScheduleError> {
    for use_ in uses {
        if use_.ordinal as usize >= dag_size {
            return Err(ScheduleError::OrdinalPastDag {
                ordinal: use_.ordinal,
                dag_size,
            });
        }
    }
    let coloring = color_live_ranges(uses, run_ends, value_count, no_recycle)
        .map_err(ScheduleError::Coloring)?;
    let mut per_dispatch: Vec<Vec<ScratchBind>> = vec![Vec::new(); dag_size];
    for use_ in uses {
        let color = coloring.color[use_.value as usize]
            .expect("a value with a use has an interval, so it was coloured");
        per_dispatch[use_.ordinal as usize].push(ScratchBind {
            bind_index: use_.bind_index,
            color,
        });
    }
    Ok(ScratchSchedule {
        per_dispatch,
        coloring,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(ordinal: u32, value: u32) -> Use {
        Use {
            ordinal,
            bind_index: 0,
            value,
            is_write: false,
        }
    }

    fn write(ordinal: u32, value: u32) -> Use {
        Use {
            ordinal,
            bind_index: 1,
            value,
            is_write: true,
        }
    }

    /// Each ordinal runs alone.
    fn solo_runs(n: u32) -> Vec<u32> {
        (0..n).collect()
    }

    #[test]
    fn the_schedule_answers_bind_and_size_from_one_colouring() {
        // A two-dispatch chain: d0 writes v0, d1 reads v0 and writes v1.
        let uses = [write(0, 0), read(1, 0), write(1, 1)];
        let schedule =
            schedule_scratch(2, &uses, &solo_runs(2), 2, false).expect("a sound DAG schedules");
        assert_eq!(schedule.per_dispatch.len(), 2);
        let c0 = schedule.coloring.color[0].unwrap();
        let c1 = schedule.coloring.color[1].unwrap();
        assert_eq!(
            schedule.per_dispatch[0],
            [ScratchBind {
                bind_index: 1,
                color: c0
            }]
        );
        assert_eq!(
            schedule.per_dispatch[1],
            [
                ScratchBind {
                    bind_index: 0,
                    color: c0
                },
                ScratchBind {
                    bind_index: 1,
                    color: c1
                }
            ]
        );
        // The colouring travels with the table: sizing needs color-of-VALUE.
        assert!(schedule.coloring.colors_used >= 2);
    }

    #[test]
    fn a_use_past_the_dag_is_refused_before_fanning_out() {
        let uses = [write(3, 0)];
        assert_eq!(
            schedule_scratch(2, &uses, &solo_runs(4), 1, false),
            Err(ScheduleError::OrdinalPastDag {
                ordinal: 3,
                dag_size: 2
            })
        );
    }

    #[test]
    fn a_colouring_refusal_reaches_the_scheduler_unignorable() {
        // Ordinal 1 is past a one-entry run table: RunsTooShort, carried.
        let uses = [write(0, 0), read(1, 0)];
        assert_eq!(
            schedule_scratch(2, &uses, &[0], 1, false),
            Err(ScheduleError::Coloring(ColoringError::RunsTooShort {
                ordinal: 1
            }))
        );
    }

    #[test]
    fn a_chain_of_ping_pong_values_needs_two_buffers() {
        // v0: written @0, read @1. v1: written @1, read @2. v2: written @2.
        // Inclusive overlap at the shared ordinal keeps neighbours apart;
        // v2 can reuse v0's buffer, freed strictly before ordinal 2.
        let uses = [
            write(0, 0),
            read(1, 0),
            write(1, 1),
            read(2, 1),
            write(2, 2),
        ];
        let coloring = color_live_ranges(&uses, &solo_runs(3), 3, false).expect("colours");
        assert_eq!(coloring.colors_used, 2);
        assert_ne!(coloring.color[0], coloring.color[1]);
        assert_ne!(coloring.color[1], coloring.color[2]);
        assert_eq!(
            coloring.color[0], coloring.color[2],
            "v2 reuses v0's buffer"
        );
    }

    #[test]
    fn a_same_dispatch_write_after_read_lands_in_different_buffers() {
        let uses = [write(0, 0), read(1, 0), write(1, 1)];
        let coloring = color_live_ranges(&uses, &solo_runs(2), 2, false).expect("colours");
        assert_ne!(
            coloring.color[0], coloring.color[1],
            "the dispatch reading v0 writes v1 in the same ordinal"
        );
    }

    #[test]
    fn a_concurrency_run_extends_the_interval_to_its_end() {
        // v0 last read @1, but ordinals 1..=3 run barrier-free; v1 is
        // written @2, inside the run. Without the extension they would
        // share a buffer and the concurrent writer would clobber the read.
        let runs = [0, 3, 3, 3];
        let uses = [write(0, 0), read(1, 0), write(2, 1)];
        let coloring = color_live_ranges(&uses, &runs, 2, false).expect("colours");
        assert_ne!(coloring.color[0], coloring.color[1]);

        // The same shape with solo ordinals reuses the buffer — the
        // extension, not the shape, is what separates them.
        let solo = color_live_ranges(&uses, &solo_runs(3), 2, false).expect("colours");
        assert_eq!(solo.color[0], solo.color[1]);
    }

    #[test]
    fn no_recycle_gives_every_used_value_its_own_buffer() {
        let uses = [write(0, 0), read(1, 0), write(1, 1), read(2, 1)];
        let coloring = color_live_ranges(&uses, &solo_runs(3), 2, true).expect("colours");
        assert_eq!(coloring.colors_used, 2);
        assert_ne!(coloring.color[0], coloring.color[1]);
    }

    #[test]
    fn a_value_nobody_uses_costs_no_buffer() {
        // Three declared values, one used. The C++ gave each unused value a
        // fresh colour, inflating the count the scratch region is sized by.
        let uses = [write(0, 1)];
        let coloring = color_live_ranges(&uses, &solo_runs(1), 3, false).expect("colours");
        assert_eq!(coloring.colors_used, 1);
        assert_eq!(coloring.color[0], None);
        assert_eq!(coloring.color[2], None);
        let no_recycle = color_live_ranges(&uses, &solo_runs(1), 3, true).expect("colours");
        assert_eq!(
            no_recycle.colors_used, 1,
            "no_recycle counts used values only"
        );
    }

    #[test]
    fn a_use_past_the_run_table_is_refused_not_unextended() {
        let uses = [write(0, 0), read(5, 0)];
        assert_eq!(
            color_live_ranges(&uses, &solo_runs(3), 1, false),
            Err(ColoringError::RunsTooShort { ordinal: 5 }),
            "the C++ silently skipped the extension here"
        );
        assert_eq!(
            color_live_ranges(&[write(0, 7)], &solo_runs(1), 1, false),
            Err(ColoringError::ValueOutOfRange { value: 7 })
        );
    }

    #[test]
    fn values_defined_out_of_order_still_colour_by_first_use() {
        // v1 is defined before v0 in ordinal terms; the allocator scans in
        // def order, not id order.
        let uses = [write(0, 1), read(1, 1), write(2, 0)];
        let coloring = color_live_ranges(&uses, &solo_runs(3), 2, false).expect("colours");
        assert_eq!(
            coloring.colors_used, 1,
            "v0 reuses v1's buffer after it dies"
        );
        assert_eq!(coloring.color[0], coloring.color[1]);
    }
}
