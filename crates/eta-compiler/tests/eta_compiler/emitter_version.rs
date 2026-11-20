use crate::msl_corpus::{corpus_bound, corpus_stages, extended_stages};
use eta_compiler::codegen::cuda::CUDA_GENERATED_EMITTER_VERSION;
use eta_compiler::codegen::hip::HIP_GENERATED_EMITTER_VERSION;
use eta_compiler::codegen::metal::METAL_M1_EMITTER_VERSION;
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::codegen::wgsl::{WGSL_STEP_EMITTER_VERSION, step_calls, step_module};

const PINNED: &[(&str, u16, u64)] = &[
    ("cuda", 85, 0x9389_6e9d_eece_7251),
    ("hip", 45, 0xf542_b370_db4d_4553),
    ("metal", 76, 0x3f1b_da25_e649_6779),
];

/// The stepwise WGSL emitter's version and the fingerprint of what it
/// writes: every call's module, in the order a stage's steps index them.
const WGSL_PINNED: (u16, u64) = (10, 0x8709_d20e_5db8_8971);

fn wgsl_fingerprint() -> u64 {
    let mut bytes = Vec::new();
    for call in step_calls() {
        let module = step_module("main", &call);
        bytes.extend_from_slice(&(module.len() as u64).to_le_bytes());
        bytes.extend_from_slice(module.as_bytes());
    }
    eta_ir::fnv1a64(&bytes)
}

fn fingerprint(backend: Backend) -> u64 {
    let stages: Vec<_> = corpus_stages()
        .into_iter()
        .chain(extended_stages())
        .map(|stage| stage.plan)
        .collect();
    let mut bytes = Vec::new();
    for kernel in emit_program(backend, &stages, &corpus_bound()) {
        bytes.extend_from_slice(&(kernel.kind as u32).to_le_bytes());
        bytes.extend_from_slice(&kernel.stage_index.to_le_bytes());
        bytes.extend_from_slice(&kernel.region_index.to_le_bytes());
        for text in [&kernel.entry_name, &kernel.source, &kernel.error] {
            bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
            bytes.extend_from_slice(text.as_bytes());
        }
        // A streamed kernel's step table (its dispatch words and the nodes
        // each step computes) is emitted bytes an engine lays scratch by.
        if !kernel.steps.is_empty() {
            bytes.extend_from_slice(&(kernel.steps.len() as u64).to_le_bytes());
            for word in &kernel.steps {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
    }
    eta_ir::fnv1a64(&bytes)
}

fn backend_of(name: &str) -> Backend {
    match name {
        "cuda" => Backend::Cuda,
        "hip" => Backend::Hip,
        "metal" => Backend::Metal,
        other => panic!("no backend named `{other}`"),
    }
}

#[test]
fn emitter_version_every_case() {
    the_pinned_versions_are_the_compiled_ones();
    each_emitter_version_still_describes_its_output();
    the_stepwise_wgsl_version_still_describes_its_output();
}

fn the_stepwise_wgsl_version_still_describes_its_output() {
    let (version, pinned) = WGSL_PINNED;
    assert_eq!(
        version, WGSL_STEP_EMITTER_VERSION,
        "the stepwise WGSL emitter version constant is {WGSL_STEP_EMITTER_VERSION}, but \
         WGSL_PINNED still says {version}"
    );
    let actual = wgsl_fingerprint();
    assert_eq!(
        actual, pinned,
        "stepwise WGSL emitter version {version} was pinned to {pinned:#018x} but now emits \
         {actual:#018x}: bump WGSL_STEP_EMITTER_VERSION and put the new fingerprint in \
         WGSL_PINNED, in the same commit"
    );
}

fn the_pinned_versions_are_the_compiled_ones() {
    for (name, version, _) in PINNED {
        let constant = match backend_of(name) {
            Backend::Cuda => CUDA_GENERATED_EMITTER_VERSION,
            Backend::Hip => HIP_GENERATED_EMITTER_VERSION,
            Backend::Metal => METAL_M1_EMITTER_VERSION,
        };
        assert_eq!(
            *version, constant,
            "the {name} emitter version constant is {constant}, but PINNED still says {version}. \
             If the emitted bytes changed, update both; if they did not, the constant moved for \
             nothing and every engine's cache was discarded."
        );
    }
}

fn each_emitter_version_still_describes_its_output() {
    let mut moved = Vec::new();
    for (name, version, pinned) in PINNED {
        let actual = fingerprint(backend_of(name));
        if actual != *pinned {
            moved.push(format!(
                "{name}: emitter version {version} was pinned to {pinned:#018x} but now emits \
                 {actual:#018x}"
            ));
        }
    }
    assert!(
        moved.is_empty(),
        "the emitted bytes changed, so the engines' compile caches must be \
         invalidated:\n  {}\n\nBump the backend's emitter version constant and \
         put the new fingerprint in PINNED, in the same commit. If the change \
         was meant to be output-neutral, this is the bug report.",
        moved.join("\n  ")
    );
}
