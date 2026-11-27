//! What a guest session shares across APIs: the bindings a stage's
//! tables lie at, and how wide it dispatches. The tables themselves are
//! `eta_exec::tables`, one definition for every backend.

use eta_compiler::codegen::wgsl;
use eta_exec::ValueDesc;
pub use eta_exec::tables::{
    Cfg, DESC_WORDS, StageTables, Stepping, alias_reshapes, desc_words, program_table_sizes,
    stage_tables, table_sizes, words_le,
};

pub use eta_compiler::codegen::steps::Bases;

use crate::backend::{Backend, Buffer as _};
use crate::error::{Fault, Result};
use crate::guest::held::Held;

/// The bindings a step module reads (the WGSL runtime's): its status word,
/// as atomics, and its tables, in the segment holding the instance's slots;
/// its heap, in the segment holding its state; each a whole segment, so one
/// group serves every instance whose slots and heap share two segments.
pub const STATUS_BINDING: u32 = 0;
pub const TABLES_BINDING: u32 = 1;
pub const HEAP_BINDING: u32 = 4;

/// The binding of a step's record, read at a dynamic offset a dispatch:
/// its node, rung and selector and the instance's bases
/// (`eta_compiler::codegen::steps::StepUniform`).
pub const STEP_BINDING: u32 = wgsl::STEP_BINDING;

/// How far apart a stage's step records lie, for a device whose storage
/// offsets align to `align`.
#[must_use]
pub fn step_stride(align: u64) -> u64 {
    u64::from(wgsl::STEP_BYTES).max(align).next_power_of_two()
}

/// The bytes a stage's status word takes where it lies in the instance's
/// slots.
pub const STATUS_SPAN: u64 = eta_exec::STATUS_BYTES as u64;

/// What an instance's tables take in its `Instance` slots, in the order a
/// bind places them: each stage's four bind-time tables and its steps'
/// records (`stride` apart), then the status word its stages share. The
/// one list a registration's facts and a bind's placement both read.
#[must_use]
pub fn instance_table_sizes(
    plans: &[eta_compiler::codegen::launch::LaunchStagePlan],
    code: &[crate::guest::Lowered],
    stride: u64,
) -> Vec<u64> {
    let mut sizes: Vec<u64> = plans
        .iter()
        .zip(code)
        .flat_map(|(plan, lowered)| {
            let mut stage = table_sizes(plan).to_vec();
            stage.push(lowered.steps.len().max(1) as u64 * stride);
            stage
        })
        .collect();
    sizes.push(STATUS_SPAN);
    sizes
}

/// A stage's step records, `stride` apart, in dispatch order, each with its
/// node's area of `areas` (`StageTables::areas`) and the instance's
/// `bases`.
#[must_use]
pub fn step_records(steps: &[wgsl::Step], areas: &[u32], stride: u64, bases: Bases) -> Vec<u8> {
    let stride = usize::try_from(stride).expect("a step stride fits the host");
    let mut bytes = vec![0u8; steps.len().max(1) * stride];
    for (at, step) in steps.iter().enumerate() {
        let area = areas.get(step.node as usize).copied().unwrap_or(0);
        let record = step.uniform(area).bytes(bases);
        bytes[at * stride..at * stride + record.len()].copy_from_slice(&record);
    }
    bytes
}

/// Where a stage's instance lies, as its step records state it: the words
/// its status, its four tables (descriptors, params, offsets, config) and
/// its heap start at in the segments a backend binds.
///
/// This is the bind's check of the runtime's isolation facts (the WGSL
/// runtime's third): the status, the tables and the step records lie in one
/// run of the instance's slots (`slot_bytes` each) in one segment, the heap
/// (`heap`: its segment, first byte and bytes) in its segment, each base
/// and extent in whole words inside its segment and the `u32` words a
/// kernel indexes, and, where the heap lies in the slots' segment, it
/// overlaps none of the run the instance's tables take: a heap write, the
/// one write a guest's data steers, reaches no host-written word. Refused by
/// name.
pub fn bases<B: Backend>(
    status: &Held<B>,
    tables: &[Held<B>; 4],
    args: &Held<B>,
    (heap, heap_at, heap_bytes): (&B::Buffer, u64, u64),
    slot_bytes: u64,
) -> Result<Bases> {
    let segment = status.buffer();
    let span = |held: &Held<B>| Laid {
        segment: held.buffer().slab_id(),
        run: held.run_span(slot_bytes),
        at: held.offset(),
        bytes: held.bytes(),
    };
    let [descs, params, offs, cfg] = tables;
    located(
        (segment.slab_id(), segment.bytes()),
        [
            span(status),
            span(descs),
            span(params),
            span(offs),
            span(cfg),
            span(args),
        ],
        Laid {
            segment: heap.slab_id(),
            run: (0, heap.bytes()),
            at: heap_at,
            bytes: heap_bytes,
        },
    )
}

/// Where a stage's instance lies for a backend that binds each span on its
/// own (native Vulkan): its four tables in words of the one run of slots
/// they lie in, which the tables binding spans, its status and its heap at
/// word 0 of their own exact bindings. The bases then hold wherever the run
/// moves. Refused by name where a table lies off the run or a word. The run
/// (its segment's slab id, first and last byte) is the second of the pair.
pub fn run_bases<B: Backend>(
    tables: &[Held<B>; 4],
    slot_bytes: u64,
) -> Result<(Bases, (u64, u64, u64))> {
    let run = tables[0].run_span(slot_bytes);
    let segment = tables[0].buffer().slab_id();
    let mut words = [0u32; 4];
    for ((held, what), word) in tables.iter().zip(&LAID[1..5]).zip(&mut words) {
        let at = held.offset().wrapping_sub(run.0);
        if held.buffer().slab_id() != segment
            || held.run_span(slot_bytes) != run
            || !at.is_multiple_of(4)
            || held.offset() < run.0
            || held.offset() + held.bytes() > run.1
        {
            return Err(Fault::Program {
                at: "guest::session",
                why: format!("its {what} lies outside the one run of slots its tables lie in"),
            });
        }
        *word = u32::try_from(at / 4).map_err(|_| Fault::Program {
            at: "guest::session",
            why: format!("its {what} lies past the words a kernel indexes"),
        })?;
    }
    let [descs, params, offs, cfg] = words;
    Ok((
        Bases {
            status: 0,
            descs,
            params,
            offs,
            cfg,
            heap: 0,
        },
        (segment, run.0, run.1),
    ))
}

/// A span a bind checks: bytes `[at, at + bytes)` of the segment of slab
/// `segment`, in the run of slots `run` there (the heap: its segment's
/// bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Laid {
    pub segment: u64,
    pub run: (u64, u64),
    pub at: u64,
    pub bytes: u64,
}

const LAID: [&str; 6] = [
    "status",
    "descriptors",
    "op params",
    "offsets",
    "config",
    "step records",
];

/// [`bases`] over where each span lies: the status, the four tables and
/// the step records (`laid`, in that order) in the segment `segment` (its
/// slab id and bytes), and the heap. Refused by name where a span lies off
/// the status's run or segment, off a word, past the `u32` words a kernel
/// indexes, or, the heap, over the run in one segment.
pub fn located(segment: (u64, u64), laid: [Laid; 6], heap: Laid) -> Result<Bases> {
    let refuse = |why: String| Fault::Program {
        at: "guest::session",
        why,
    };
    let word = |what: &str, span: Laid, end_at: u64| -> Result<u32> {
        if !span.at.is_multiple_of(4)
            || span
                .at
                .checked_add(span.bytes)
                .is_none_or(|end| end > end_at || span.at < span.run.0)
        {
            return Err(refuse(format!(
                "its {what} at byte {} of {} bytes lies outside its segment of {end_at} bytes                  or its slots, or off a word",
                span.at, span.bytes
            )));
        }
        u32::try_from(span.at / 4).map_err(|_| {
            refuse(format!(
                "its {what} at byte {} lies past the words a kernel indexes",
                span.at
            ))
        })
    };
    let run = laid[0].run;
    let mut words = [0u32; 6];
    for ((what, span), word_at) in LAID.iter().zip(laid).zip(&mut words) {
        if span.segment != segment.0 || span.run != run {
            return Err(refuse(format!(
                "its {what} lies outside the one run of slots its status lies in"
            )));
        }
        *word_at = word(what, span, segment.1.min(run.1))?;
    }
    let [status, descs, params, offs, cfg, _records] = words;
    let bases = Bases {
        status,
        descs,
        params,
        offs,
        cfg,
        heap: word("heap", heap, heap.run.1)?,
    };
    if u64::from(bases.heap) + heap.bytes / 4 > u64::from(u32::MAX) {
        return Err(refuse(format!(
            "its heap of {} bytes from word {} runs past the words a kernel indexes",
            heap.bytes, bases.heap
        )));
    }
    if heap.segment == segment.0 && heap.at < run.1 && run.0 < heap.at + heap.bytes {
        return Err(refuse(format!(
            "its heap at bytes {}..{} overlaps bytes {}..{} of its segment, the slots its              tables lie in",
            heap.at,
            heap.at + heap.bytes,
            run.0,
            run.1
        )));
    }
    Ok(bases)
}

pub const WORKGROUP: u32 = eta_compiler::codegen::wgsl::WORKGROUP;

pub fn groups_for(descriptors: &[ValueDesc], workgroups: u32) -> u32 {
    let widest = descriptors.iter().map(|d| d.len).max().unwrap_or(1).max(1);
    let wanted = widest.div_ceil(WORKGROUP);
    wanted.clamp(1, workgroups.max(1))
}

/// Each of `steps`' workgroups, at most `workgroups` (the device holds at
/// once): the jobs its call runs over its node's first operand
/// ([`eta_compiler::codegen::wgsl::step_jobs`]) a workgroup's invocations
/// each, else what the stage's widest value wants ([`groups_for`]). A step
/// whose jobs are far fewer than the widest value (a reduction's levels,
/// its finish) launches no workgroup only to find no job.
#[must_use]
pub fn step_groups(
    plan: &eta_compiler::codegen::launch::LaunchStagePlan,
    descriptors: &[ValueDesc],
    steps: &[eta_compiler::codegen::wgsl::Step],
    workgroups: u32,
) -> Vec<u32> {
    let widest = groups_for(descriptors, u32::MAX);
    steps
        .iter()
        .map(|step| {
            let jobs = plan
                .ops
                .get(step.node as usize)
                .and_then(|op| op.args.first())
                .and_then(|&a0| descriptors.get(a0 as usize))
                .and_then(|a| {
                    eta_compiler::codegen::wgsl::step_jobs(step.call, step.rung, a.len, rows_of(a))
                });
            let wanted = jobs.map_or(widest, |jobs| {
                u32::try_from(jobs.div_ceil(u64::from(WORKGROUP))).unwrap_or(u32::MAX)
            });
            wanted.clamp(1, workgroups.max(1))
        })
        .collect()
}

/// A value's rows as the runtime counts them (`rows_of`): its dims but the
/// last, one where its rank is under two.
fn rows_of(value: &ValueDesc) -> u32 {
    let rank = (value.rank as usize).min(value.dims.len());
    if rank < 2 {
        return 1;
    }
    value.dims[..rank - 1].iter().product::<u32>().max(1)
}

#[cfg(test)]
mod tests {
    use super::{Laid, located};

    const SEGMENT: (u64, u64) = (7, 1 << 20);
    const RUN: (u64, u64) = (4096, 8192);

    fn laid() -> [Laid; 6] {
        [4096, 4112, 4224, 4352, 4400, 4608].map(|at| Laid {
            segment: SEGMENT.0,
            run: RUN,
            at,
            bytes: 16,
        })
    }

    fn heap(segment: u64, at: u64, bytes: u64) -> Laid {
        Laid {
            segment,
            run: (0, 1 << 20),
            at,
            bytes,
        }
    }

    /// The bind refuses by name a span off its run or segment or a word, a
    /// heap past the `u32` words a kernel indexes, and a heap over the run
    /// its tables take in one segment.
    #[test]
    fn a_bind_refuses_a_span_a_kernel_must_not_reach() {
        let refused = |laid: [Laid; 6], heap: Laid, why: &str| {
            let fault = located(SEGMENT, laid, heap).expect_err(why).to_string();
            assert!(fault.contains(why), "{fault} names `{why}`");
        };
        let fine = heap(9, 256, 64);
        let mut past_run = laid();
        past_run[2].at = 8190;
        refused(past_run, fine, "op params at byte 8190");
        let mut other_run = laid();
        other_run[5].run = (8192, 12288);
        refused(other_run, fine, "step records lies outside the one run");
        let mut other_segment = laid();
        other_segment[1].segment = 8;
        refused(other_segment, fine, "descriptors lies outside the one run");
        let mut off_word = laid();
        off_word[3].at = 4354;
        refused(off_word, fine, "offsets at byte 4354");
        refused(laid(), heap(9, (1 << 20) - 32, 64), "heap at byte");
        let far = Laid {
            segment: 9,
            run: (0, 1 << 36),
            at: (u64::from(u32::MAX) - 4) * 4,
            bytes: 64,
        };
        refused(laid(), far, "runs past the words a kernel indexes");
        refused(
            laid(),
            heap(SEGMENT.0, 8000, 64),
            "overlaps bytes 4096..8192",
        );
        refused(
            laid(),
            heap(SEGMENT.0, 0, 4100),
            "overlaps bytes 4096..8192",
        );
    }
}
