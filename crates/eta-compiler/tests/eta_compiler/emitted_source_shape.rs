use eta_compiler::codegen::lane_scratch::LANE_SCRATCH;
use eta_compiler::codegen::program::{Backend, EmittedKernel, emit_program};
use eta_compiler::plan::compile_bound;

use crate::msl_corpus::{
    GOLDEN_NAMES, extended_traces, golden_container, golden_served, synthetic_traces,
};

fn every_emitted_kernel() -> Vec<(String, Backend, EmittedKernel)> {
    let mut traces: Vec<_> = GOLDEN_NAMES
        .iter()
        .map(|name| {
            (
                (*name).to_string(),
                golden_container(name),
                golden_served(name),
            )
        })
        .collect();
    for (name, container, served) in synthetic_traces().into_iter().chain(extended_traces()) {
        traces.push((name.to_string(), container, served));
    }

    let mut out = Vec::new();
    for (name, container, served) in traces {
        let Ok(bound) = served.bind(container) else {
            continue;
        };
        let stages = compile_bound(&bound);
        for &backend in Backend::ALL {
            for kernel in emit_program(backend, &stages, &bound) {
                out.push((name.clone(), backend, kernel));
            }
        }
    }
    out
}

#[test]
fn every_emitted_kernel_is_a_source_or_a_reason() {
    for (trace, backend, kernel) in every_emitted_kernel() {
        let where_ = format!(
            "{trace} {backend:?} stage {} region {}",
            kernel.stage_index, kernel.region_index
        );
        assert_ne!(
            kernel.source.is_empty(),
            kernel.error.is_empty(),
            "{where_}: source and error must be exactly one of the two \
             (source {} bytes, error {:?})",
            kernel.source.len(),
            kernel.error
        );
        if kernel.source.is_empty() {
            assert!(
                kernel.entry_name.is_empty(),
                "{where_}: a refusal still named an entry point {:?}",
                kernel.entry_name
            );
            continue;
        }
        assert!(
            !kernel.entry_name.is_empty(),
            "{where_}: emitted a source with no entry name"
        );
        assert!(
            kernel.source.contains(&kernel.entry_name),
            "{where_}: source does not define its own entry point {:?}",
            kernel.entry_name
        );
    }
}

/// A grouped kernel's lane finds its value scratch at its own entry of the
/// lane scratch table ([`LANE_SCRATCH`]), a 64-bit address, bound where the
/// table's facts state: on every backend that emits grouped kernels, and
/// no lane's base is lane 0's plus a stride.
#[test]
fn every_grouped_lane_scratch_base_is_its_lane_scratch_entry() {
    let metal = format!(
        "const device ulong* {} [[buffer({})]]",
        LANE_SCRATCH.name, LANE_SCRATCH.metal_buffer
    );
    let metal_base = format!(
        "reinterpret_cast<device uchar*>({}[dispatch_lane]);",
        LANE_SCRATCH.name
    );
    let simt = format!("const m1_u64* {},", LANE_SCRATCH.name);
    let simt_base = format!(
        "m1_u8* scratch = reinterpret_cast<m1_u8*>({}[dispatch_lane]);",
        LANE_SCRATCH.name
    );
    let mut seated = [0usize; 3];
    for (trace, backend, kernel) in every_emitted_kernel() {
        let where_ = format!(
            "{trace} {backend:?} stage {} region {} ({})",
            kernel.stage_index, kernel.region_index, kernel.entry_name
        );
        assert!(
            !kernel.source.contains("all_scratch"),
            "{where_}: a lane's scratch is still addressed from one region by a stride"
        );
        if !kernel.source.contains("dispatch_lane") {
            continue;
        }
        let (table, base, at) = match backend {
            Backend::Metal => (&metal, &metal_base, 0),
            Backend::Cuda => (&simt, &simt_base, 1),
            Backend::Hip => (&simt, &simt_base, 2),
        };
        if !kernel.source.contains(table.as_str()) {
            continue;
        }
        seated[at] += 1;
        assert!(
            kernel.source.contains(base.as_str()),
            "{where_}: binds the lane scratch table and reads no lane's base from it"
        );
    }
    assert!(
        seated.iter().all(|&count| count > 0),
        "the corpus emits grouped kernels on every backend that reads lane bases: {seated:?}"
    );
}
