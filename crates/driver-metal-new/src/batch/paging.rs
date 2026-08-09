//! Firing a step whose routed experts do not all fit on the device: the
//! portable half.
//!
//! The mechanism exists once, for every family that has a routed bank. It
//! was written inside the llama engine first and gpt-oss needed the same
//! thing; the second copy would have been the same eighty lines with
//! different type names, and the two would have drifted at the first fix
//! applied to one.
//!
//! What it does is decided by one fact about the hardware: an Apple Silicon
//! GPU has no demand paging, and `requestResidency` wires every page of a
//! mapping whether a kernel reads it or not. So a bank bigger than the
//! working set cannot be mapped -- it has to be read through a region that
//! stays wired while its CONTENTS change, and the host has to change them
//! between dispatches. That makes a step several ordered command buffers
//! instead of one, with the host awake between them: after each mixture
//! layer's router runs, the host pages that layer's experts in and rewrites
//! the ids the router wrote IN PLACE from expert numbers to slot numbers.
//! The rewrite is why no kernel changed for any of this: a routed matvec
//! does `base += expert_ids[sel] * stride`, and a slot number in a shorter
//! stack is the same instruction against different bytes.
//!
//! The cost is one submit-and-wait per mixture layer. It is a bad trade for
//! a model that fits and the only trade there is for a model that does not.
//!
//! What is here is what needs no device: [`plan_paging`] -- every refusal
//! is a case that would otherwise fail mid-decode, where the diagnosis is
//! right and nothing can be done about it -- and [`renumber_routing`], the
//! in-place rewrite. The segment orchestration (submit, wait, page, repeat)
//! drives a command queue and belongs to `src/metal/`; one of its rules is
//! stated here because it is a budget fact, not a queue fact: the previous
//! segment's pins are given back BEFORE the next page-in, because its
//! command buffer has completed -- exactly the condition that makes its
//! slots reusable -- and doing it after would hold two layers at once and
//! need twice the budget.

/// One mixture layer's cut: one past the last dispatch of the segment that
/// ends with its router.
///
/// The ids buffer is per LAYER, not one for the stack, and stays with the
/// device half. A single handle assumed every layer's routing decision
/// colours onto the same pool slot, which is a property of a greedy
/// colouring rather than a guarantee -- and the day it stopped holding, the
/// host would have rewritten a buffer the next segment does not read, which
/// is wrong tokens and not an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cut {
    /// One past the last dispatch of this segment.
    pub end: usize,
}

/// The segment table a paged fire runs: each mixture layer's cut, then the
/// tail, which pages nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PagingPlan {
    /// Segment ends, ascending; the last is the whole step.
    ends: Vec<usize>,
    /// The most distinct experts one fire can want resident at once.
    worst_case_experts: u32,
}

impl PagingPlan {
    /// `(begin, end)` for each segment in order. The caller recognises the
    /// first segment by `begin == 0` and the last by `end == dag_size`,
    /// which is where any pre/post hook belongs. The segment after cut `k`
    /// is the one whose page-in rewrites cut `k`'s ids.
    pub fn segments(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let begins = std::iter::once(0).chain(self.ends.iter().copied());
        begins.zip(self.ends.iter().copied())
    }

    /// How many mixture layers the step pages for — one fewer than the
    /// segment count, because the tail routes nothing.
    ///
    /// A caller supplying one routing buffer per layer checks its slice
    /// against this. The C++ instead carried the handle inside its `Cut` and
    /// appended a tail cut holding a null one, so "the tail has no ids" was a
    /// value to remember at every use rather than a length that cannot be
    /// wrong.
    #[must_use]
    pub fn mixture_layers(&self) -> usize {
        self.ends.len().saturating_sub(1)
    }

    /// How many segments the step runs (mixture layers plus the tail).
    #[must_use]
    pub fn len(&self) -> usize {
        self.ends.len()
    }

    /// Never true -- a plan always carries at least the tail -- but the
    /// emptiness convention comes with [`len`](Self::len).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// The most distinct experts one fire can route to at once.
    #[must_use]
    pub fn worst_case_experts(&self) -> u32 {
        self.worst_case_experts
    }
}

/// Why a shape cannot be paged.
#[derive(Debug, PartialEq, Eq)]
pub enum PagingRefused {
    /// The DAG's routed-layer count disagrees with what the slab staged.
    LayerCount {
        /// Routed layers cut out of the DAG.
        in_dag: usize,
        /// Layers the slab holds banks for.
        staged: u32,
    },
    /// Two cuts share a concurrency run: the ends are not strictly
    /// ascending, so a segment would hold two routing decisions and the
    /// page-in between them would have nowhere to stand.
    TwoRoutersInOneRun,
    /// A cut past the end of the step.
    CutPastEnd {
        /// The offending cut.
        end: usize,
        /// The step's dispatch count.
        dag_size: usize,
    },
    /// The slab cannot hold the worst case. Every expert a single fire can
    /// route to must be resident AT ONCE: one dispatch reads them all, so
    /// there is no order in which a smaller cache could serve it. (Which is
    /// also why the slab never needs to be bigger than ONE layer's bank
    /// however many layers the model has -- the whole reduction.) Refused
    /// with the numbers that would work.
    SlabTooSmall {
        /// The fire width the bound was computed for.
        max_rows: u32,
        /// `min(n_experts, max_rows * experts_per_token)`.
        worst: u32,
        /// The bytes that worst case needs resident.
        needed_bytes: u64,
        /// The experts the budget holds.
        budget_slots: u32,
    },
}

impl std::fmt::Display for PagingRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PagingRefused::LayerCount { in_dag, staged } => write!(
                f,
                "expert paging: {in_dag} routed layers in the DAG against {staged} staged"
            ),
            PagingRefused::TwoRoutersInOneRun => {
                write!(f, "expert paging: two routers in one concurrency run")
            }
            PagingRefused::CutPastEnd { end, dag_size } => write!(
                f,
                "expert paging: a cut at {end} past the end of a {dag_size}-dispatch step"
            ),
            PagingRefused::SlabTooSmall {
                max_rows,
                worst,
                needed_bytes,
                budget_slots,
            } => write!(
                f,
                "expert_slab_bytes is too small: a fire of {max_rows} rows can route \
                 to {worst} distinct experts at once, which needs {needed_bytes} \
                 bytes, but the budget holds {budget_slots} experts"
            ),
        }
    }
}

impl std::error::Error for PagingRefused {}

/// What the plan checks against: the slab's shape, read off it by the
/// device half. Plain numbers rather than the slab itself, so the check is
/// testable -- and stated -- without one.
#[derive(Clone, Copy, Debug)]
pub struct SlabShape {
    /// Layers the slab stages.
    pub layers: u32,
    /// Slots the budget holds.
    pub slots: u32,
    /// Bytes one slot costs (every routed tensor of one expert).
    pub slot_bytes: u64,
}

/// Validate a paged step's shape and build its segment table.
///
/// # Errors
///
/// Every [`PagingRefused`] variant is a case that would otherwise fail
/// mid-decode, where the diagnosis is right and nothing can be done about
/// it.
pub fn plan_paging(
    cuts: &[Cut],
    dag_size: usize,
    slab: SlabShape,
    n_experts: u32,
    experts_per_token: u32,
    max_rows: u32,
) -> Result<PagingPlan, PagingRefused> {
    if cuts.len() != slab.layers as usize {
        return Err(PagingRefused::LayerCount {
            in_dag: cuts.len(),
            staged: slab.layers,
        });
    }
    let mut previous = 0;
    for (k, cut) in cuts.iter().enumerate() {
        if k > 0 && cut.end <= previous {
            return Err(PagingRefused::TwoRoutersInOneRun);
        }
        if cut.end > dag_size {
            return Err(PagingRefused::CutPastEnd {
                end: cut.end,
                dag_size,
            });
        }
        previous = cut.end;
    }
    let worst = u64::from(n_experts)
        .min(u64::from(max_rows) * u64::from(experts_per_token))
        .try_into()
        .unwrap_or(u32::MAX);
    if slab.slots < worst {
        return Err(PagingRefused::SlabTooSmall {
            max_rows,
            worst,
            needed_bytes: u64::from(worst) * slab.slot_bytes,
            budget_slots: slab.slots,
        });
    }
    let mut ends: Vec<usize> = cuts.iter().map(|cut| cut.end).collect();
    ends.push(dag_size); // the tail, which pages nothing
    Ok(PagingPlan {
        ends,
        worst_case_experts: worst,
    })
}

/// The ids buffer is shorter than the rows it is being read at, or a
/// resident lookup refused.
#[derive(Debug, PartialEq, Eq)]
pub enum RenumberRefused<E> {
    /// `rows` rows at this stride do not fit the buffer.
    ShortIds {
        /// Bytes the walk would touch.
        need: usize,
        /// Bytes the buffer holds.
        have: usize,
    },
    /// The residency callback refused; carries its error.
    Resident(E),
}

/// How one layer's routing readback is laid out in its buffer.
///
/// Three integers that must agree with each other and with the fire. They
/// travel together because passing them positionally is how `rows` and
/// `experts_per_token` come to be swapped at a call site: both are small
/// counts, both are `usize`, and the swap compiles. `PARITY-LOADER.md`
/// records the same defect in `plan_heap`'s two adjacent `int` widths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IdsLayout {
    /// Rows the fire routes.
    pub rows: usize,
    /// Routing decisions per row.
    pub experts_per_token: usize,
    /// Bytes between two rows' decisions; `0` means packed. See
    /// [`renumber_routing`] for why the distinction is not cosmetic.
    pub row_stride_bytes: usize,
}

/// Rewrite one layer's routing decisions in place, expert numbers to slot
/// numbers.
///
/// `row_stride_bytes` is how far apart two rows' decisions sit, and zero
/// means "packed", which is what the router kernel produces whenever one
/// dispatch covers every row -- it advances `expert_ids` by
/// `row * experts_per_token` and never by a row pitch. It is not always
/// packed: qwen3.5 encodes a prefill as one DAG PER TOKEN over a shared
/// scratch pool, and when the prompt is too wide to route as one group each
/// token routes off its own binding, a whole scratch row apart. Reading a
/// strided fire as packed renumbers row 0 `rows` times and leaves every
/// other row holding TRUE expert ids, which then index slots -- experts
/// read from whatever slot happened to hold them, which is fluent wrong
/// text rather than an error.
///
/// An id outside `[0, n_experts)` is a padded slot and is left alone.
/// `resident` is the slab's `ensure_resident` for the layer being paged;
/// its refusals pass through.
///
/// # Errors
///
/// [`RenumberRefused::ShortIds`] before any byte is touched -- a partial
/// rewrite is a buffer in a state no caller asked for -- and
/// [`RenumberRefused::Resident`] from the callback.
pub fn renumber_routing<E>(
    ids: &mut [u8],
    rows: usize,
    experts_per_token: usize,
    row_stride_bytes: usize,
    n_experts: u32,
    mut resident: impl FnMut(u32) -> Result<u32, E>,
) -> Result<(), RenumberRefused<E>> {
    let packed = experts_per_token * size_of::<i32>();
    let stride = if row_stride_bytes == 0 {
        packed
    } else {
        row_stride_bytes
    };
    if rows > 0 {
        let need = (rows - 1) * stride + packed;
        if need > ids.len() {
            return Err(RenumberRefused::ShortIds {
                need,
                have: ids.len(),
            });
        }
    }
    for row in 0..rows {
        for lane in 0..experts_per_token {
            let at = row * stride + lane * size_of::<i32>();
            let cell: &mut [u8] = &mut ids[at..at + size_of::<i32>()];
            let expert = i32::from_le_bytes(cell.try_into().expect("four bytes"));
            let Ok(expert) = u32::try_from(expert) else {
                continue; // a padded slot
            };
            if expert >= n_experts {
                continue; // a padded slot
            }
            let slot = resident(expert).map_err(RenumberRefused::Resident)?;
            cell.copy_from_slice(&i32::try_from(slot).unwrap_or(i32::MAX).to_le_bytes());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> SlabShape {
        SlabShape {
            layers: 3,
            slots: 32,
            slot_bytes: 1000,
        }
    }

    #[test]
    fn a_plan_is_the_cuts_plus_a_tail_that_pages_nothing() {
        let cuts = [Cut { end: 4 }, Cut { end: 9 }, Cut { end: 15 }];
        let plan = plan_paging(&cuts, 20, shape(), 128, 4, 8).expect("a sound shape plans");
        let segments: Vec<_> = plan.segments().collect();
        assert_eq!(segments, [(0, 4), (4, 9), (9, 15), (15, 20)]);
        assert_eq!(plan.len(), 4);
        assert_eq!(plan.worst_case_experts(), 32, "8 rows x 4 slots");
    }

    #[test]
    fn the_tail_is_not_a_mixture_layer_so_the_routing_buffers_are_one_fewer() {
        // The count a caller checks its per-layer routing buffers against. The
        // off-by-one it prevents is not an error at the boundary: an extra
        // buffer would be paired with the tail, which routes nothing, and the
        // host would rewrite ids no segment reads.
        let cuts = [Cut { end: 4 }, Cut { end: 9 }, Cut { end: 15 }];
        let plan = plan_paging(&cuts, 20, shape(), 128, 4, 8).expect("a sound shape plans");
        assert_eq!(plan.len(), 4, "three mixture layers plus the tail");
        assert_eq!(plan.mixture_layers(), 3);
    }

    #[test]
    fn every_way_the_cut_table_can_lie_is_named() {
        let dag = 20;
        assert_eq!(
            plan_paging(&[Cut { end: 4 }], dag, shape(), 128, 4, 8),
            Err(PagingRefused::LayerCount {
                in_dag: 1,
                staged: 3
            })
        );
        assert_eq!(
            plan_paging(
                &[Cut { end: 4 }, Cut { end: 4 }, Cut { end: 9 }],
                dag,
                shape(),
                128,
                4,
                8
            ),
            Err(PagingRefused::TwoRoutersInOneRun)
        );
        assert_eq!(
            plan_paging(
                &[Cut { end: 4 }, Cut { end: 9 }, Cut { end: 21 }],
                dag,
                shape(),
                128,
                4,
                8
            ),
            Err(PagingRefused::CutPastEnd {
                end: 21,
                dag_size: dag
            })
        );
    }

    #[test]
    fn the_worst_case_is_every_expert_one_dispatch_can_read() {
        let cuts = [Cut { end: 4 }, Cut { end: 9 }, Cut { end: 15 }];
        // 16 rows x 4 experts each = 64 distinct in the worst case, capped
        // by the bank's 48. One dispatch reads them all; there is no order
        // in which 32 slots could serve it.
        let err = plan_paging(&cuts, 20, shape(), 48, 4, 16).expect_err("32 < 48");
        assert_eq!(
            err,
            PagingRefused::SlabTooSmall {
                max_rows: 16,
                worst: 48,
                needed_bytes: 48_000,
                budget_slots: 32
            }
        );
        // A single row cannot want more than experts_per_token at once.
        assert!(plan_paging(&cuts, 20, shape(), 48, 4, 1).is_ok());
    }

    fn ids_of(lanes: &[i32]) -> Vec<u8> {
        lanes.iter().flat_map(|lane| lane.to_le_bytes()).collect()
    }

    fn lanes_of(ids: &[u8]) -> Vec<i32> {
        ids.chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn a_packed_rewrite_maps_every_lane_and_leaves_padding_alone() {
        // Two rows, three lanes each; -1 and 99 are padding (99 >= 8).
        let mut ids = ids_of(&[5, -1, 3, 99, 5, 0]);
        let mapped: std::cell::RefCell<Vec<u32>> = std::cell::RefCell::new(Vec::new());
        renumber_routing(&mut ids, 2, 3, 0, 8, |expert| {
            mapped.borrow_mut().push(expert);
            Ok::<u32, ()>(expert + 100)
        })
        .expect("in bounds");
        assert_eq!(lanes_of(&ids), [105, -1, 103, 99, 105, 100]);
        assert_eq!(
            *mapped.borrow(),
            [5, 3, 5, 0],
            "padding never reaches the slab"
        );
    }

    #[test]
    fn a_strided_fire_rewrites_each_row_at_its_own_pitch() {
        // Two rows a whole scratch row (16 bytes) apart, two lanes each.
        // Read as packed this would renumber row 0 twice and leave row 1
        // holding true expert ids -- fluent wrong text. The stride keeps
        // each row's rewrite at its own offset.
        let mut ids = vec![0u8; 32];
        ids[0..8].copy_from_slice(&ids_of(&[1, 2]));
        ids[16..24].copy_from_slice(&ids_of(&[3, 4]));
        renumber_routing(&mut ids, 2, 2, 16, 8, |expert| Ok::<u32, ()>(expert + 10))
            .expect("in bounds");
        assert_eq!(lanes_of(&ids[0..8]), [11, 12]);
        assert_eq!(lanes_of(&ids[16..24]), [13, 14]);
    }

    #[test]
    fn a_short_buffer_is_refused_before_any_byte_moves() {
        let mut ids = ids_of(&[1, 2, 3]);
        let untouched = ids.clone();
        let err = renumber_routing(&mut ids, 2, 2, 0, 8, Ok::<u32, ()>)
            .expect_err("two packed rows need 16 bytes");
        assert_eq!(err, RenumberRefused::ShortIds { need: 16, have: 12 });
        assert_eq!(
            ids, untouched,
            "a partial rewrite is a state nobody asked for"
        );
    }

    #[test]
    fn a_residency_refusal_passes_through() {
        let mut ids = ids_of(&[1]);
        let err = renumber_routing(&mut ids, 1, 1, 0, 8, |_| Err::<u32, &str>("slab says no"))
            .expect_err("the callback refused");
        assert_eq!(err, RenumberRefused::Resident("slab says no"));
    }
}
