//! The CUDA kernel emitters, byte for byte against the C++ oracle.
//!
//! The sibling of `metal_msl_golden.rs`, for the CUDA backend.
//! `compiler/tests/golden-cuda/` is a dump of
//! `driver/cuda/src/pipeline/generated/` over the shared corpus, produced by
//! `compiler/tests/oracle/cuda_codegen_dump.cpp`; the Rust port in
//! `compiler/codegen/src/cuda/` must reproduce it.
//!
//! A fused CUDA kernel is 40-70 KB and the corpus has 320 regions, so unlike
//! the Metal dump the fused bodies are pinned by FNV-1a of the tail rather than
//! checked in whole — with one region per stage also kept verbatim
//! (`emit_fused_region_cuda_verbatim`) so a failure can be read, not just
//! detected.

#[path = "common/msl_corpus.rs"]
mod msl_corpus;

use std::fmt::Write as _;
use std::path::PathBuf;

use pie_codegen::cuda::{
    CUDA_GENERATED_EMITTER_VERSION, emit_fused_region, emit_singleton_region,
    second_party_region_supported, singleton_runtime_source, validate_generated_region,
};

use msl_corpus::{corpus_bound, corpus_stages, region_shape};

fn golden_cuda_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("golden-cuda")
}

fn fnv1a64(text: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in text.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

struct Dump {
    emitter: &'static str,
    body: String,
    runtime: String,
}

impl Dump {
    fn new(emitter: &'static str) -> Self {
        Self {
            emitter,
            body: String::new(),
            runtime: singleton_runtime_source(),
        }
    }

    fn open_case(&mut self, id: &str) {
        let _ = writeln!(self.body, "=== {id}");
    }

    fn field(&mut self, key: &str, value: &str) {
        let _ = writeln!(self.body, "{key}: {value}");
    }

    fn end(&mut self) {
        self.body.push_str("=== end\n");
    }

    /// Split an emitted kernel into the elided runtime prefix and its tail.
    fn split<'a>(&self, source: &'a str) -> (&'static str, &'a str) {
        if !self.runtime.is_empty() && source.starts_with(&self.runtime) {
            ("@runtime", &source[self.runtime.len()..])
        } else {
            ("none", source)
        }
    }

    /// `GeneratedKernelSource` carries its own ok/error, so there is no
    /// separate failure path — the C++ dumps the same four fields either way.
    fn kernel(&mut self, entry_name: &str, op_tag: u8, emitted: &Result<String, String>) {
        let _ = writeln!(self.body, "ok: {}", emitted.is_ok());
        let _ = writeln!(
            self.body,
            "error: {}",
            emitted.as_ref().err().map_or("", |error| error.as_str())
        );
        let _ = writeln!(self.body, "entry_name: {entry_name}");
        // `GeneratedKernelSource::op_tag` defaults to 0 and only the singleton
        // emitter sets it, but the C++ dump prints it either way.
        let _ = writeln!(self.body, "op_tag: 0x{op_tag:02x}");
        let Ok(source) = emitted else {
            self.end();
            return;
        };
        let (prefix, tail) = self.split(source);
        let _ = writeln!(
            self.body,
            "source: bytes={} prefix={prefix} tail_bytes={}",
            source.len(),
            tail.len()
        );
        self.body.push_str(tail);
        if !tail.is_empty() && !tail.ends_with('\n') {
            self.body.push_str("\n\\no-trailing-newline\n");
        }
        self.end();
    }

    /// The hashed form used for the 320 fused kernels.
    fn kernel_digest(&mut self, entry_name: &str, emitted: &Result<String, String>) {
        let _ = writeln!(self.body, "ok: {}", emitted.is_ok());
        let _ = writeln!(
            self.body,
            "error: {}",
            emitted.as_ref().err().map_or("", |error| error.as_str())
        );
        let _ = writeln!(self.body, "entry_name: {entry_name}");
        if let Ok(source) = emitted {
            let (prefix, tail) = self.split(source);
            let _ = writeln!(
                self.body,
                "source: bytes={} prefix={prefix} tail_bytes={} fnv1a64=0x{:016x}",
                source.len(),
                tail.len(),
                fnv1a64(tail)
            );
        }
        self.end();
    }
}

fn compare(dump: &Dump) {
    let path = golden_cuda_dir().join(format!("{}.txt", dump.emitter));
    let oracle = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{} missing ({error})", path.display()));
    let header: String = oracle
        .lines()
        .take_while(|line| line.starts_with('#'))
        .map(|line| format!("{line}\n"))
        .collect();
    let expected = &oracle[header.len()..];

    if std::env::var("PTIR_REGEN").is_ok() {
        std::fs::write(&path, header.clone() + &dump.body).unwrap();
        return;
    }
    if expected == dump.body {
        return;
    }
    let mut case = String::from("<before the first case>");
    for (index, (mine, theirs)) in dump.body.lines().zip(expected.lines()).enumerate() {
        if let Some(id) = theirs.strip_prefix("=== ") {
            case = id.to_string();
        }
        assert_eq!(
            mine,
            theirs,
            "{} diverged from the C++ oracle at line {} (case `{case}`)",
            dump.emitter,
            index + 1
        );
    }
    assert_eq!(
        dump.body.lines().count(),
        expected.lines().count(),
        "{} emitted a different number of lines than the C++ oracle",
        dump.emitter
    );
}

/// The runtime template is 45 KB elided from every other case, so it is pinned
/// on its own — otherwise the two sides could differ on the bytes nobody diffs.
#[test]
fn runtime_matches_oracle() {
    let mut dump = Dump::new("singleton_runtime_cuda_source");
    let runtime = singleton_runtime_source();
    dump.open_case("runtime");
    dump.field("bytes", &runtime.len().to_string());
    dump.field("fnv1a64", &format!("0x{:016x}", fnv1a64(&runtime)));
    dump.end();
    compare(&dump);
}

/// Entry names the emitters must accept or reject on their own terms.
const ENTRY_NAMES: &[&str] = &[
    "ptir_fused_0123456789abcdef_r0",
    "",
    "0starts_with_digit",
    "has-a-dash",
    "has space",
    "_underscore_start",
    "A",
];

#[test]
fn singleton_matches_oracle() {
    let mut dump = Dump::new("emit_singleton_region_cuda");
    for tag in 0..256u32 {
        let tag = tag as u8;
        let name = format!("ptir_singleton_0x{tag:02x}");
        dump.open_case(&format!("tag_0x{tag:02x}"));
        dump.field("entry_name", &name);
        let emitted = emit_singleton_region(&name, tag);
        dump.kernel(&name, tag, &emitted);
    }
    for name in ENTRY_NAMES {
        dump.open_case(&format!("entry_name[{name}]"));
        dump.field("entry_name", name);
        // 0x10 is a real op tag, so only the name is under test.
        let emitted = emit_singleton_region(name, 0x10);
        dump.kernel(name, 0x10, &emitted);
    }
    compare(&dump);
}

#[test]
fn region_emitters_match_oracle() {
    let stages = corpus_stages();
    let mut fused = Dump::new("emit_fused_region_cuda");
    let mut verbatim = Dump::new("emit_fused_region_cuda_verbatim");
    let mut validate = Dump::new("validate_generated_region");
    let mut second_party = Dump::new("second_party_region_supported");

    for stage in &stages {
        let signature = format!("{:016x}", stage.plan.signature.hash);
        for (partition_name, partition) in [
            ("singleton", &stage.plan.singleton),
            ("fused", &stage.plan.fused),
        ] {
            for (index, region) in partition.regions.iter().enumerate() {
                let id = format!("{} {partition_name}#{index}", stage.id());
                let shape = region_shape(region);
                let entry = format!("ptir_fused_{signature}_r{index}");

                let emitted = emit_fused_region(&entry, &stage.plan, region);
                fused.open_case(&id);
                fused.field("entry_name", &entry);
                fused.field("region", &shape);
                fused.kernel_digest(&entry, &emitted);
                if index == 0 {
                    verbatim.open_case(&id);
                    verbatim.field("entry_name", &entry);
                    verbatim.field("region", &shape);
                    verbatim.kernel(&entry, 0, &emitted);
                }

                validate.open_case(&id);
                validate.field("region", &shape);
                let verdict = validate_generated_region(&stage.plan, region);
                validate.field("ok", &verdict.is_ok().to_string());
                validate.field(
                    "error",
                    verdict.as_ref().err().map_or("", |error| error.as_str()),
                );
                validate.end();

                second_party.open_case(&id);
                second_party.field("region", &shape);
                second_party.field(
                    "supported",
                    &second_party_region_supported(&stage.plan, region).to_string(),
                );
                second_party.end();
            }
        }
    }
    compare(&fused);
    compare(&verbatim);
    compare(&validate);
    compare(&second_party);
}

/// The emitter version is part of the driver's compile-cache key, so a silent
/// bump would quietly invalidate or, worse, reuse the wrong cached cubin.
#[test]
fn emitter_version_matches_oracle() {
    let dump = std::fs::read_to_string(golden_cuda_dir().join("emit_fused_region_cuda.txt"))
        .expect("the fused oracle dump exists");
    let recorded: u16 = dump
        .lines()
        .find_map(|line| line.strip_prefix("# emitter_version: "))
        .expect("the oracle header records the emitter version")
        .trim()
        .parse()
        .unwrap();
    assert_eq!(recorded, CUDA_GENERATED_EMITTER_VERSION);
}

/// Whole-program emission: the table a driver receives.
///
/// The per-region emitters are pinned against the C++ oracle above; this pins
/// the walk around them — that every region gets an entry, that entry names are
/// the ones the drivers used to generate for themselves, and that a failure is
/// recorded rather than dropped.
#[test]
fn emit_program_covers_every_region() {
    use pie_codegen::program::{Backend, KERNEL_FUSED, emit_program};

    let stages: Vec<_> = corpus_stages()
        .into_iter()
        .map(|stage| stage.plan)
        .collect();
    let kernels = emit_program(Backend::Cuda, &stages, &corpus_bound());

    let expected: usize = stages.iter().map(|stage| stage.fused.regions.len()).sum();
    assert_eq!(
        kernels.len(),
        expected,
        "CUDA emission must produce one kernel per fused region"
    );
    for kernel in &kernels {
        assert_eq!(kernel.kind, KERNEL_FUSED);
        // Exactly one of source/error is set: a kernel is either emitted or
        // explained, never silently absent.
        assert_ne!(
            kernel.source.is_empty(),
            kernel.error.is_empty(),
            "kernel {}#{} has neither source nor error",
            kernel.stage_index,
            kernel.region_index
        );
        if !kernel.source.is_empty() {
            let stage = &stages[kernel.stage_index as usize];
            let entry = format!(
                "ptir_fused_{:016x}_r{}",
                stage.signature.hash, kernel.region_index
            );
            assert_eq!(kernel.entry_name, entry);
            assert!(
                kernel.source.contains(&entry),
                "the source defines its entry"
            );
        }
    }
}

/// The Metal walk emits four families; check each shows up and is named the way
/// `m1_runtime.cpp` names them.
#[test]
fn emit_program_metal_covers_every_family() {
    use pie_codegen::program::{
        Backend, KERNEL_COMMIT, KERNEL_FUSED, KERNEL_GROUPED, KERNEL_READINESS, KERNEL_SINGLETON,
        emit_program,
    };

    let stages: Vec<_> = corpus_stages()
        .into_iter()
        .map(|stage| stage.plan)
        .collect();
    let kernels = emit_program(Backend::Metal, &stages, &corpus_bound());
    for kind in [
        KERNEL_SINGLETON,
        KERNEL_FUSED,
        KERNEL_GROUPED,
        KERNEL_READINESS,
        KERNEL_COMMIT,
    ] {
        assert!(
            kernels.iter().any(|kernel| kernel.kind == kind),
            "no kernel of kind {kind} was emitted"
        );
    }
    for kernel in &kernels {
        assert_ne!(
            kernel.source.is_empty(),
            kernel.error.is_empty(),
            "kernel kind={} {}#{} has neither source nor error",
            kernel.kind,
            kernel.stage_index,
            kernel.region_index
        );
        if !kernel.source.is_empty() {
            assert!(
                kernel.source.contains(&kernel.entry_name),
                "the source defines its entry `{}`",
                kernel.entry_name
            );
        }
    }
}
