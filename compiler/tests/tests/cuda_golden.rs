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

#[path = "common/device_text.rs"]
mod device_text;
#[path = "common/msl_corpus.rs"]
mod msl_corpus;
#[path = "common/provenance.rs"]
mod provenance;

use std::fmt::Write as _;
use std::path::PathBuf;

use pie_codegen::cuda::{
    CUDA_GENERATED_EMITTER_VERSION, emit_fused_region, emit_singleton_region,
    second_party_region_supported, singleton_runtime_source, validate_generated_region,
};

use device_text::OracleInputs;
use msl_corpus::{corpus_bound, corpus_stages, region_shape};

fn golden_cuda_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("golden-cuda")
}

/// The live CUDA device source and the oracle-era copy of whichever part of it
/// has moved since the dump was taken. See `common/device_text.rs`.
fn oracle_inputs() -> OracleInputs {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    OracleInputs::load(
        manifest.join("../codegen/runtime/cuda"),
        golden_cuda_dir().join("oracle-inputs"),
    )
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
    inputs: OracleInputs,
}

impl Dump {
    fn new(emitter: &'static str) -> Self {
        let inputs = oracle_inputs();
        Self {
            emitter,
            body: String::new(),
            runtime: inputs.rewrite(&singleton_runtime_source()),
            inputs,
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
        let source = self.inputs.rewrite(source);
        let (prefix, tail) = self.split(&source);
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
            let source = self.inputs.rewrite(source);
            let (prefix, tail) = self.split(&source);
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
        provenance::regenerate_foreign(&path, &header, &dump.body);
        return;
    }
    if expected == dump.body {
        return;
    }
    let mut case = String::from("<before the first case>");
    let hint = dump.inputs.hint();
    for (index, (mine, theirs)) in dump.body.lines().zip(expected.lines()).enumerate() {
        if let Some(id) = theirs.strip_prefix("=== ") {
            case = id.to_string();
        }
        assert_eq!(
            mine,
            theirs,
            "{} diverged from the C++ oracle at line {} (case `{case}`)\n{hint}",
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
    let runtime = dump.runtime.clone();
    dump.open_case("runtime");
    dump.field("bytes", &runtime.len().to_string());
    dump.field("fnv1a64", &format!("0x{:016x}", fnv1a64(&runtime)));
    dump.end();
    compare(&dump);
}

/// What makes the oracle-era rewrite safe: the emitter splices each device file
/// in whole and verbatim, so putting the oracle-era text back can only ever
/// restore bytes the kernel side owns. An emitter that started editing, folding
/// or regenerating one of these blocks would fail here first, and the rewrite
/// would stop firing for it rather than quietly cover the change up.
#[test]
fn every_live_device_file_is_spliced_into_a_kernel_verbatim() {
    let inputs = oracle_inputs();
    let runtime = singleton_runtime_source();
    let singleton = emit_singleton_region("ptir_singleton_probe", 0x10)
        .expect("the singleton emitter accepts op tag 0x10");
    let stages = corpus_stages();
    let fused =
        stages
            .iter()
            .flat_map(|stage| {
                stage.plan.fused.regions.iter().map(move |region| {
                    emit_fused_region("ptir_fused_probe_r0", &stage.plan, region)
                })
            })
            .find_map(Result::ok)
            .expect("at least one corpus region emits a fused CUDA kernel");

    let files = inputs.live_files();
    assert_eq!(
        files.len(),
        6,
        "compiler/codegen/runtime/cuda/ holds six device files; a new one needs \
         a home in this assertion and, once it moves, in golden-cuda/oracle-inputs/"
    );
    for (name, text) in files {
        let hosts = [&runtime, &singleton, &fused];
        assert!(
            hosts.iter().any(|host| host.contains(&text)),
            "{name} is include_str!d into the emitters but no emitted kernel \
             contains it verbatim, so the oracle-era rewrite cannot stand in \
             for it"
        );
    }
}

/// A copy that no longer differs from its live file stands in for nothing.
#[test]
fn oracle_inputs_are_live_files_that_moved() {
    oracle_inputs().assert_entries_are_live_files_that_moved();
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
    use pie_codegen::program::{Backend, PIE_KERNEL_FUSED, emit_program};

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
        assert_eq!(kernel.kind, PIE_KERNEL_FUSED);
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
        Backend, PIE_KERNEL_COMMIT, PIE_KERNEL_FUSED, PIE_KERNEL_GROUPED, PIE_KERNEL_READINESS,
        PIE_KERNEL_SINGLETON, emit_program,
    };

    let stages: Vec<_> = corpus_stages()
        .into_iter()
        .map(|stage| stage.plan)
        .collect();
    let kernels = emit_program(Backend::Metal, &stages, &corpus_bound());
    for kind in [
        PIE_KERNEL_SINGLETON,
        PIE_KERNEL_FUSED,
        PIE_KERNEL_GROUPED,
        PIE_KERNEL_READINESS,
        PIE_KERNEL_COMMIT,
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

/// `pie_plan::stage_identity` must equal the CUDA driver's
/// `compiled_stage_identity` byte for byte.
///
/// The driver keys its CUDA-graph cache on this value, and a wrong key is
/// silent: it does not fail, it reuses the wrong graph. So the two
/// implementations are pinned here while both exist, the same way the emitters
/// were pinned by `golden-{msl,cuda}/` (`ptir-refactor.md` §3.2). The expected
/// column was produced by compiling `driver/cuda/src/pipeline/program_identity.hpp`
/// on the host and running it over `corpus_stages()`; regenerate with
/// `PTIR_REGEN=1` only after re-deriving it the same way.
#[test]
fn stage_identity_matches_the_driver() {
    let header = "# GENERATED from driver/cuda/src/pipeline/program_identity.hpp\n\
                  # <golden>#<stage_index> <compiled_stage_identity as hex>\n";
    let mut rendered = String::new();
    for stage in corpus_stages() {
        let _ = writeln!(
            rendered,
            "{} {:016x}",
            stage.id(),
            pie_plan::stage_identity(&stage.plan)
        );
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("golden-stage-identity.txt");
    if std::env::var("PTIR_REGEN").is_ok() {
        provenance::regenerate_foreign(&path, header, &rendered);
        return;
    }
    let file = std::fs::read_to_string(&path).expect("golden-stage-identity.txt");
    let expected: String = file
        .lines()
        .skip_while(|line| line.starts_with('#'))
        .map(|line| format!("{line}\n"))
        .collect();
    assert_eq!(
        rendered, expected,
        "pie_plan::stage_identity has diverged from the CUDA driver's \
         compiled_stage_identity"
    );
}

/// Escape hatch for re-deriving `golden-stage-identity.txt` against the C++
/// driver: `PTIR_DUMP_PLANS=<path> cargo test -p pie-compiler-tests --test
/// cuda_golden dump_corpus_plans` writes the PTRP wire bytes the host oracle
/// in `program_identity.hpp` decodes. Not a check; it fails nothing.
#[test]
fn dump_corpus_plans() {
    let Ok(path) = std::env::var("PTIR_DUMP_PLANS") else {
        return;
    };
    let mut out = String::new();
    for stage in corpus_stages() {
        let hex: String = stage.wire.iter().map(|b| format!("{b:02x}")).collect();
        let _ = writeln!(out, "id={} bytes={hex}", stage.id());
    }
    std::fs::write(path, out).unwrap();
}

/// Emit the CUDA kernel table for the traces the driver's own tests register.
///
/// The driver no longer generates kernels, so a C++ test that registers a
/// program has to be handed the same table the engine would hand it. It cannot
/// run the Rust emitter, so the table is written here as a fixture beside the
/// traces (`ptir-refactor.md` §4.3 — retargeting the driver test surface at the
/// launch package).
///
/// The corpus is `compiler/tests/driver-corpus/`: one hex container per file,
/// named for the program the driver's tests register under that name. It is not
/// a subset of `compiler/tests/golden/` (`staged_dispatch` exists only here), so
/// it stays a corpus of its own — but a corpus only, holding containers and
/// nothing else. It used to live in `driver/cuda/tests/golden-ptir/` and carry
/// decoded plans, sidecars, and readiness verdicts beside each container: a
/// second source of truth for a decoder that no longer exists
/// (`ptir-refactor.md` §4.3).
///
/// Format, one record per kernel:
///   `kernel <kind> <stage> <region> <entry-or-dash> <source-byte-count>`
///   followed by exactly that many bytes of source and a newline.
#[test]
fn emit_driver_test_kernel_fixtures() {
    use pie_codegen::program::{Backend, emit_program};
    use pie_ir::container::decode as decode_container;
    use pie_ir::validate::bind;

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let traces = manifest.join("driver-corpus");
    let out_dir = manifest.join("../../driver/fixtures");
    std::fs::create_dir_all(&out_dir).unwrap();

    let mut written = 0;
    let mut unbindable: Vec<String> = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(&traces)
        .expect("driver corpus directory")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "ptir"))
        .collect();
    entries.sort();
    for path in entries {
        let hex = std::fs::read_to_string(&path).unwrap();
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let Ok(container) = decode_container(&msl_corpus::unhex(hex.trim())) else {
            unbindable.push(format!("{name}: container does not decode"));
            continue;
        };
        let bound = match bind(container, msl_corpus::golden_profile(&name)) {
            Ok(bound) => bound,
            Err(error) => {
                unbindable.push(format!(
                    "{name}: does not bind against today's profile: {error:?}"
                ));
                continue;
            }
        };
        let stages = pie_plan::compile_bound(&bound);
        let kernels = emit_program(Backend::Cuda, &stages, &bound);
        let mut body = String::new();
        for kernel in &kernels {
            let _ = writeln!(
                body,
                "kernel {} {} {} {} {}",
                kernel.kind,
                kernel.stage_index,
                kernel.region_index,
                if kernel.entry_name.is_empty() {
                    "-"
                } else {
                    &kernel.entry_name
                },
                kernel.source.len()
            );
            body.push_str(&kernel.source);
            body.push('\n');
        }
        std::fs::write(out_dir.join(format!("{name}.kernels")), body).unwrap();

        // The launch package travels with the kernels for the same reason: the
        // driver's C++ tests cannot run the host's planner, and the driver no
        // longer decodes PTIR, so the one thing it *does* accept has to be
        // handed to them as a fixture. Written as a relocatable image of the
        // very same `#[repr(C)]` records the engine ships (see
        // `pie_driver_abi::image`), not as a second wire format.
        std::fs::write(
            out_dir.join(format!("{name}.launch")),
            pie_driver_abi::image::encode(&pie_codegen::launch::build(&bound, &stages)),
        )
        .unwrap();

        // Same reasoning for the region analysis. Its record is nested, so it
        // is written as one line per region followed by one line per direct
        // argmax:
        //   `region <stage> <region> <flags> <argmax-count> <skipped...>`
        //   `argmax <node> <source_value> <intrinsic> <requires_single_row>`
        let mut regions = String::new();
        for region in pie_codegen::region_analysis::analyze_program(&stages) {
            let mut header = format!(
                "region {} {} {} {}",
                region.stage_index,
                region.region_index,
                region.flags,
                region.direct_argmax.len()
            );
            for node in &region.skipped {
                let _ = write!(header, " {node}");
            }
            let _ = writeln!(regions, "{header}");
            for record in &region.direct_argmax {
                let _ = writeln!(
                    regions,
                    "argmax {} {} {} {}",
                    record.node, record.source_value, record.intrinsic, record.requires_single_row
                );
            }
        }
        std::fs::write(out_dir.join(format!("{name}.regions")), regions).unwrap();
        written += 1;
    }
    assert!(written > 0, "no driver-test kernel fixtures were written");

    // A corpus trace that no longer binds is drift, not a missing feature: the
    // C++ tests that register these fail for want of a fixture, which is the
    // correct outcome -- but say so here, where the cause is visible.
    // `neg_*` are the reject cases; not binding is what they are for.
    let drifted: Vec<&String> = unbindable
        .iter()
        .filter(|reason| !reason.starts_with("neg_"))
        .collect();
    for reason in &drifted {
        // The one stream write the compiler crates allow: the assert below
        // only reports a count, and the count is useless without the names.
        #[allow(clippy::print_stderr, reason = "names the fixtures the assert counts")]
        {
            eprintln!("[driver kernel fixtures] skipped {reason}");
        }
    }
    // Zero, not a tolerance. The last survivor was `staged_dispatch`, and it
    // had not drifted at all -- `golden_profile` was guessing vocab 8 for a
    // trace whose `Logits` is `[1, 4]`. A tolerance let that read as decay for
    // as long as the C++ test that needs the fixture was failing for other
    // reasons.
    assert!(
        drifted.is_empty(),
        "vendored driver traces no longer bind: {drifted:?}"
    );
}
