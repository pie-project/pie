//! The Metal MSL emitters, byte for byte against the C++ oracle.
//!
//! `compiler/tests/golden-msl/` is a dump of
//! `driver/metal/src/pipeline/m1_codegen.cpp` over a fixed corpus, produced by
//! `compiler/tests/oracle/m1_codegen_dump.cpp` (that file carries the build
//! command). This test drives the Rust port in `compiler/codegen/src/metal/`
//! over the same corpus, formats it identically, and requires the bytes to
//! match. Any difference is a bug in the port: the C++ is the oracle until it
//! is deleted, and this comparison is what survives it.
//!
//! Regenerating with `PTIR_REGEN=1` writes the *Rust* output over the goldens,
//! so only do it after re-deriving the C++ side with the oracle and seeing an
//! empty `git diff` — blessing from Rust alone erases the comparison this file
//! exists to make.
//!
//! Two deliberate scope limits:
//!
//! * `validate_singleton_plan` is compared on the **unmutated** corpus, and on
//!   the rejection path only its verdict and message are compared, not the
//!   `operations` vector. The oracle also dumps 23 wire-level mutations per
//!   stage, but those model a plan decoded from bytes; the Rust validator takes
//!   a native [`pie_plan::CompiledStage`] whose types make most of them
//!   unrepresentable. And the C++ fills its out-param as it walks, so a late
//!   rejection leaves a partially built vector behind — `m1_runtime.cpp` returns
//!   `reject_deterministic(error)` without reading it, so those entries are
//!   unobservable. Reproducing them would mean returning junk inside an `Err`.
//! * `emit_grouped_nucleus` skips the cases the oracle marked
//!   `cxx-oracle-undefined`. The C++ guard omits `!region.library` (its TopK
//!   sibling has it) and a Generated region's `library_op` byte is
//!   `0 == PTIR_LIBRARY_NUCLEUS_SAMPLE`, so it reaches an out-of-bounds read of
//!   `region.inputs[0..3]`. There is no defined oracle output to match.

#[path = "common/msl_corpus.rs"]
mod msl_corpus;
#[path = "common/provenance.rs"]
mod provenance;

use std::fmt::Write as _;
use std::path::PathBuf;

use pie_codegen::metal::preamble::grouped_preamble;
use pie_codegen::metal::{
    M1ChannelEffect, METAL_M1_EMITTER_VERSION, METAL_M1_MAX_CHANNELS, RUNTIME_TEMPLATE,
    emit_commit, emit_fused_region, emit_grouped_commit, emit_grouped_fused_region,
    emit_grouped_nucleus, emit_grouped_readiness, emit_grouped_topk, emit_readiness,
    emit_singleton_region, validate_singleton_plan,
};

use msl_corpus::{CorpusStage, corpus_stages, is_library, op_tag, region_shape};

/// FNV-1a 64, the hash the oracle header records for the shared prefixes.
fn fnv1a64(text: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in text.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// The embedded runtime template must be the same bytes the C++ emitter read
/// from `driver/metal/src/kernels/ptir_m1_runtime.metal`. Everything else here
/// compares kernel *tails* with the shared prefix elided, so without this the
/// two could diverge on the 34 KB nobody diffs.
#[test]
fn embedded_runtime_matches_the_oracle() {
    let dump = std::fs::read_to_string(golden_msl_dir().join("emit_singleton_region_msl.txt"))
        .expect("the singleton oracle dump exists");
    let field = |key: &str| -> (usize, u64) {
        let line = dump
            .lines()
            .find(|line| line.starts_with(key))
            .unwrap_or_else(|| panic!("the oracle header has no `{key}` line"));
        let bytes = line
            .split("bytes=")
            .nth(1)
            .unwrap()
            .split(' ')
            .next()
            .unwrap();
        let hash = line.split("fnv1a64=0x").nth(1).unwrap();
        (
            bytes.parse().unwrap(),
            u64::from_str_radix(hash.trim(), 16).unwrap(),
        )
    };

    let runtime = runtime_prefix();
    assert_eq!(
        (runtime.len(), fnv1a64(&runtime)),
        field("# @runtime:"),
        "compiler/codegen/runtime/metal/ptir_m1_runtime.metal has drifted from the copy \
         the C++ oracle read out of driver/metal/src/kernels/"
    );

    let grouped = grouped_preamble();
    assert_eq!(
        (grouped.len(), fnv1a64(grouped)),
        field("# @grouped:"),
        "the grouped preamble has drifted from the C++ emitter"
    );
}

fn golden_msl_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("golden-msl")
}

/// The runtime prefix every emitted kernel starts with, elided from the dump
/// the way the oracle elides it.
fn runtime_prefix() -> String {
    let mut prefix = String::from(RUNTIME_TEMPLATE);
    prefix.push('\n');
    prefix
}

/// Runtime + grouped preamble. The oracle recovers this by probing the emitter
/// (it cannot call the private helper from C++); here the helper is in scope, so
/// this restates the emitter's own composition order rather than guessing it.
fn grouped_prefix() -> String {
    runtime_prefix() + grouped_preamble()
}

struct Dump {
    emitter: &'static str,
    body: String,
    runtime: String,
    grouped: String,
}

impl Dump {
    fn new(emitter: &'static str) -> Self {
        Self {
            emitter,
            body: String::new(),
            runtime: runtime_prefix(),
            grouped: grouped_prefix(),
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

    fn source(&mut self, text: &str) {
        self.body.push_str("ok: true\n");
        let (prefix, tail) = if !self.grouped.is_empty() && text.starts_with(&self.grouped) {
            ("@runtime+@grouped", &text[self.grouped.len()..])
        } else if !self.runtime.is_empty() && text.starts_with(&self.runtime) {
            ("@runtime", &text[self.runtime.len()..])
        } else {
            ("none", text)
        };
        let _ = writeln!(
            self.body,
            "source: bytes={} prefix={prefix} tail_bytes={}",
            text.len(),
            tail.len()
        );
        self.body.push_str(tail);
        if !tail.is_empty() && !tail.ends_with('\n') {
            self.body.push_str("\n\\no-trailing-newline\n");
        }
        self.end();
    }

    fn result(&mut self, emitted: Result<String, String>) {
        match emitted {
            Ok(source) => self.source(&source),
            Err(error) => {
                self.body.push_str("ok: false\n");
                let _ = writeln!(self.body, "error: {error}");
                self.end();
            }
        }
    }
}

/// Cases where this compiler *deliberately* disagrees with the recorded C++
/// oracle, each with the reason.
///
/// The oracle binary is gone (deleted with the C++ emitters), so these dumps
/// are the only surviving record of its behaviour. That makes `PTIR_REGEN=1`
/// the wrong tool for a fix: regenerating would overwrite the evidence and
/// leave no trace that the divergence was intended. An entry here is the
/// trace. Adding one is a claim that the oracle was *wrong*, so it needs to
/// say why.
const INTENDED_ORACLE_DIVERGENCES: &[(&str, &str)] = &[
    // The oracle emitted these: it bound `intrinsic_val` ids 3 (`query`) and 4
    // (`value_head`) to the `logits` buffer, because that is the only buffer
    // `ptir_m1_runtime.metal` binds for op tag 0xA0 and neither the emitter nor
    // the Metal driver looks at `p.intr` except for `MtpDrafts`. The kernel ran,
    // faulted nothing, and returned logits rows in place of the requested
    // intrinsic. `intrinsics_bindable` now rejects the region instead.
    (
        "pentathlon_iter#0 singleton#0",
        "oracle emitted a kernel binding intrinsic id 3 (query) to `logits`",
    ),
    (
        "pentathlon_iter#0 fused#0",
        "oracle emitted a kernel binding intrinsic id 3 (query) to `logits`",
    ),
    (
        "pentathlon_iter#1 fused#2",
        "oracle emitted a kernel binding intrinsic id 4 (value_head) to `logits`",
    ),
    // Same defect on the interpreted path: `m1_runtime.cpp` binds
    // `inputs.logits_bf16` for every `PTIR_OP_INTRINSIC_VAL` and only consults
    // `op.intr` to size the row range, so the oracle accepted these plans too
    // (`pentathlon_iter#1` verbatim; `#0` it rejected, but for an unrelated
    // kernel boundary further down the stage).
    (
        "pentathlon_iter#0",
        "oracle accepted intrinsic id 3 (query) on the interpreted path",
    ),
    (
        "pentathlon_iter#1",
        "oracle accepted intrinsic id 4 (value_head) on the interpreted path",
    ),
];

/// One-line gist of a case body: the verdict plus its error or byte count.
fn summarize(case: &str) -> String {
    case.lines()
        .filter(|line| {
            line.starts_with("ok:") || line.starts_with("error:") || line.starts_with("source:")
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

fn intended_divergence(case: &str) -> Option<&'static str> {
    INTENDED_ORACLE_DIVERGENCES
        .iter()
        .find(|(id, _)| *id == case)
        .map(|(_, reason)| *reason)
}

/// Split a dump body into `(case-id, case-text)` pairs.
fn cases(body: &str) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    for line in body.lines() {
        if let Some(id) = line.strip_prefix("=== ")
            && id != "end"
        {
            out.push((id.to_string(), String::new()));
            continue;
        }
        if let Some((_, text)) = out.last_mut() {
            text.push_str(line);
            text.push('\n');
        }
    }
    out
}

/// Compare one emitter's cases against the oracle dump, ignoring the oracle's
/// `#` header (it records provenance, not behaviour).
fn compare(dump: &Dump) {
    let path = golden_msl_dir().join(format!("{}.txt", dump.emitter));
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
    let mine_cases = cases(&dump.body);
    let their_cases = cases(expected);
    assert_eq!(
        mine_cases.iter().map(|(id, _)| id).collect::<Vec<_>>(),
        their_cases.iter().map(|(id, _)| id).collect::<Vec<_>>(),
        "{} case list diverged from the oracle dump",
        dump.emitter
    );
    let mut unexpected: Vec<String> = Vec::new();
    let mut stale: Vec<String> = Vec::new();
    for ((id, mine), (_, theirs)) in mine_cases.iter().zip(&their_cases) {
        match (intended_divergence(id), mine == theirs) {
            (Some(_), false) => {}
            (Some(reason), true) => stale.push(format!("{id} ({reason})")),
            (None, true) => {}
            (None, false) => unexpected.push(format!(
                "  `{id}`\n    rust  : {}\n    oracle: {}",
                summarize(mine),
                summarize(theirs)
            )),
        }
    }
    assert!(
        unexpected.is_empty(),
        "{} diverged from the C++ oracle in {} case(s):\n{}",
        dump.emitter,
        unexpected.len(),
        unexpected.join("\n")
    );
    assert!(
        stale.is_empty(),
        "{} cases are listed in INTENDED_ORACLE_DIVERGENCES but now match the \
         oracle -- drop the entries:\n{}",
        dump.emitter,
        stale.join("\n")
    );
    // Line counts legitimately differ once a case is in the ledger (a rejection
    // is one line where an emitted kernel is hundreds). The per-case comparison
    // above, gated on identical case-id lists, is the stronger check.
}

fn effect_from_flags(flags: u32, capacity: u32) -> M1ChannelEffect {
    M1ChannelEffect {
        requires_full: flags & 1 != 0,
        requires_empty: flags & 2 != 0,
        take: flags & 4 != 0,
        put: flags & 8 != 0,
        capacity,
    }
}

fn effect_cases() -> Vec<(String, Vec<M1ChannelEffect>)> {
    let mut cases: Vec<(String, Vec<M1ChannelEffect>)> = vec![("empty".into(), Vec::new())];
    for flags in 0..16u32 {
        cases.push((format!("flags_{flags}"), vec![effect_from_flags(flags, 1)]));
    }
    for capacity in [0u32, 1, 2, 3, 7, 64, 65535, 4294967295] {
        cases.push((
            format!("capacity_{capacity}"),
            vec![effect_from_flags(13, capacity)],
        ));
    }
    for flags in 0..16u32 {
        cases.push((
            format!("pair_{flags}"),
            vec![
                effect_from_flags(flags, 1),
                effect_from_flags(15 - flags, 3),
            ],
        ));
    }
    cases.push((
        "mix5".into(),
        vec![
            effect_from_flags(1, 1),
            effect_from_flags(8, 2),
            effect_from_flags(12, 4),
            effect_from_flags(2, 8),
            effect_from_flags(15, 16),
        ],
    ));
    for count in [
        METAL_M1_MAX_CHANNELS - 1,
        METAL_M1_MAX_CHANNELS,
        METAL_M1_MAX_CHANNELS + 1,
    ] {
        let channels = (0..count)
            .map(|channel| effect_from_flags((channel as u32 * 7 + 1) % 16, channel as u32 + 1))
            .collect();
        cases.push((format!("wide_{count}"), channels));
    }
    cases
}

#[test]
fn singleton_region_matches_oracle() {
    let mut dump = Dump::new("emit_singleton_region_msl");
    for tag in 0..256u32 {
        dump.open_case(&format!("tag_0x{tag:02x}"));
        let name = format!("ptir_m1_feedfacecafebeef_r{tag}");
        dump.field("function_name", &name);
        dump.source(&emit_singleton_region(&name, tag as u8));
    }
    compare(&dump);
}

#[test]
fn readiness_and_commit_match_oracle() {
    let mut readiness = Dump::new("emit_readiness_msl");
    let mut commit = Dump::new("emit_commit_msl");
    for (id, channels) in effect_cases() {
        let ready_name = format!("ptir_m1_{id}_ready");
        let commit_name = format!("ptir_m1_{id}_commit");

        readiness.open_case(&id);
        readiness.field("function_name", &ready_name);
        readiness.field("channels", &channels.len().to_string());
        for (index, effect) in channels.iter().enumerate() {
            readiness.field(
                &format!("channel[{index}]"),
                &format!(
                    "requires_full={} requires_empty={} take={} put={} capacity={}",
                    effect.requires_full as u8,
                    effect.requires_empty as u8,
                    effect.take as u8,
                    effect.put as u8,
                    effect.capacity
                ),
            );
        }
        readiness.source(&emit_readiness(&ready_name, &channels));

        commit.open_case(&id);
        commit.field("function_name", &commit_name);
        commit.field("channels", &channels.len().to_string());
        commit.source(&emit_commit(&commit_name, &channels));
    }
    compare(&readiness);
    compare(&commit);
}

#[test]
fn grouped_effects_match_oracle() {
    let mut readiness = Dump::new("emit_grouped_readiness_msl");
    let mut commit = Dump::new("emit_grouped_commit_msl");
    let names = [
        format!("ptir_m3_generic_ready_v{METAL_M1_EMITTER_VERSION}"),
        format!("ptir_m3_generic_commit_v{METAL_M1_EMITTER_VERSION}"),
        "k".to_string(),
        "ptir_m3_0123456789abcdef_effects".to_string(),
    ];
    for name in &names {
        readiness.open_case(name);
        readiness.field("function_name", name);
        readiness.source(&emit_grouped_readiness(name));
        commit.open_case(name);
        commit.field("function_name", name);
        commit.source(&emit_grouped_commit(name));
    }
    compare(&readiness);
    compare(&commit);
}

/// Every region of every corpus stage through all four plan-taking emitters —
/// the nesting the oracle walks.
#[test]
fn plan_emitters_match_oracle() {
    let stages = corpus_stages();
    let mut fused = Dump::new("emit_fused_region_msl");
    let mut grouped = Dump::new("emit_grouped_fused_region_msl");
    let mut nucleus = Dump::new("emit_grouped_nucleus_msl");
    let mut topk = Dump::new("emit_grouped_topk_msl");

    for stage in &stages {
        let signature = format!("{:016x}", stage.plan.signature.hash);
        for (partition_name, partition) in [
            ("singleton", &stage.plan.singleton),
            ("fused", &stage.plan.fused),
        ] {
            let singleton = partition_name == "singleton";
            let mut seen = [false; 256];
            for (index, region) in partition.regions.iter().enumerate() {
                if singleton {
                    // Singleton regions are 1:1 with ops and their text is a
                    // function of the op tag, so one region per distinct tag
                    // covers the partition without dozens of near-identical
                    // 10 KB kernels.
                    if region.nodes.len() != 1
                        || region.nodes[0].index() >= stage.plan.normalized.ops.len()
                    {
                        continue;
                    }
                    let tag = op_tag(&stage.plan, region.nodes[0].get());
                    if seen[tag as usize] {
                        continue;
                    }
                    seen[tag as usize] = true;
                }
                let id = format!("{} {partition_name}#{index}", stage.id());
                let shape = region_shape(region);

                let name = format!("ptir_m2_{signature}_r{index}");
                fused.open_case(&id);
                fused.field("function_name", &name);
                fused.field("region", &shape);
                fused.result(emit_fused_region(&name, &stage.plan, region));

                let name = format!(
                    "{}{signature}_r{index}",
                    if singleton { "ptir_m3s_" } else { "ptir_m3_" }
                );
                grouped.open_case(&id);
                grouped.field("function_name", &name);
                grouped.field("region", &shape);
                grouped.result(emit_grouped_fused_region(&name, &stage.plan, region));

                let name = format!("ptir_m3_{signature}_r{index}");
                nucleus.open_case(&id);
                nucleus.field("function_name", &name);
                nucleus.field("region", &shape);
                if !is_library(region) && (region.inputs.len() < 3 || region.outputs.is_empty()) {
                    nucleus.field(
                        "skipped",
                        "cxx-oracle-undefined (emit_grouped_nucleus_msl indexes \
                         region.inputs[0]/outputs[0] without checking region.library)",
                    );
                    nucleus.end();
                } else {
                    nucleus.result(emit_grouped_nucleus(&name, &stage.plan, region));
                }

                topk.open_case(&id);
                topk.field("function_name", &name);
                topk.field("region", &shape);
                topk.result(emit_grouped_topk(&name, &stage.plan, region));
            }
        }
    }
    compare(&fused);
    compare(&grouped);
    compare(&nucleus);
    compare(&topk);
}

/// `validate_singleton_plan` on the unmutated corpus; see the module docs for
/// the two scope limits.
#[test]
fn validate_singleton_plan_matches_oracle_on_clean_plans() {
    let oracle = std::fs::read_to_string(golden_msl_dir().join("validate_singleton_plan.txt"))
        .expect("the validate oracle dump exists");
    let stages = corpus_stages();
    assert!(!stages.is_empty(), "the corpus is empty");

    let mut divergences = Vec::new();
    for stage in &stages {
        let theirs = Verdict::parse(&oracle_case(&oracle, &format!("{} none", stage.id())));
        let mine = Verdict::of(stage);
        // The verdict and the message are the contract. `operations` is only
        // read by the driver when the verdict is `ok`.
        let same = mine.ok == theirs.ok
            && mine.error == theirs.error
            && (!mine.ok || mine.operations == theirs.operations);
        if let Some(reason) = intended_divergence(&stage.id()) {
            assert!(
                !same,
                "{} is listed in INTENDED_ORACLE_DIVERGENCES ({reason}) but now \
                 matches the oracle -- drop the entry",
                stage.id()
            );
            continue;
        }
        if !same {
            divergences.push(format!(
                "{}:\n  rust: {mine:?}\n  cxx : {theirs:?}",
                stage.id()
            ));
        }
    }
    assert!(
        divergences.is_empty(),
        "validate_singleton_plan diverged from the C++ oracle on {} of {} stages:\n{}",
        divergences.len(),
        stages.len(),
        divergences.join("\n")
    );
}

/// What `validate_singleton_plan` decided, in the oracle's vocabulary.
#[derive(Debug, PartialEq, Eq)]
struct Verdict {
    ok: bool,
    error: String,
    operations: Vec<String>,
}

impl Verdict {
    fn of(stage: &CorpusStage) -> Self {
        match validate_singleton_plan(&stage.plan) {
            Ok(operations) => Self {
                ok: true,
                error: String::new(),
                operations: operations
                    .iter()
                    .map(|meta| {
                        format!(
                            "node={} result_base={} tag=0x{:02x} results={} chan={} args={}",
                            meta.node,
                            meta.result_base,
                            meta.op.tag,
                            meta.op.results,
                            meta.op.chan,
                            meta.op.args.len()
                        )
                    })
                    .collect(),
            },
            Err(error) => Self {
                ok: false,
                error,
                operations: Vec::new(),
            },
        }
    }

    fn parse(case: &str) -> Self {
        let mut verdict = Self {
            ok: false,
            error: String::new(),
            operations: Vec::new(),
        };
        for line in case.lines() {
            if let Some(value) = line.strip_prefix("ok: ") {
                verdict.ok = value == "true";
            } else if let Some(value) = line.strip_prefix("error: ") {
                verdict.error = value.to_string();
            } else if let Some(value) = line.strip_prefix("op: ") {
                verdict.operations.push(value.to_string());
            }
        }
        verdict
    }
}

/// The lines the oracle recorded between `=== <id>` and `=== end`.
fn oracle_case(dump: &str, id: &str) -> String {
    let header = format!("=== {id}\n");
    let start = dump
        .find(&header)
        .unwrap_or_else(|| panic!("the oracle dump has no case `{id}`"))
        + header.len();
    let end = dump[start..]
        .find("=== end\n")
        .expect("every oracle case is terminated");
    dump[start..start + end].to_string()
}
