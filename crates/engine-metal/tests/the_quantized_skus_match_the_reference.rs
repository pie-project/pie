#![cfg(target_vendor = "apple")]

//! PER-SKU serve-equivalence + perplexity GATE: a single parameterized test,
//! driven by a SKU table, that holds every served (model, codec) SKU to a
//! REFERENCE engine on three independent signals, not argmax alone —
//!   1. argmax == the reference's argmax (the headline token);
//!   2. final-logit cosine >= `cosine_min` vs the reference's logit vector; and
//!   3. teacher-forced perplexity within `ppl_tol` of the reference's PPL.
//!
//! WHY ALL THREE, NOT ARGMAX: argmax-by-luck is exactly what let the V-rotation
//! bug through — a serve can land the right top token while the whole head is
//! quietly rotated wrong. The cosine catches a trunk divergence that leaves the
//! argmax standing; perplexity catches a quality divergence (e.g. a tiled-qmm
//! precision slip swapping a near-tied rank-8 token) that a single-prompt oracle
//! would never see. A SKU passes only when all three agree with the reference.
//!
//! THE TABLE IS THE TEST. Every SKU is one `Sku` row in `SKUS`; the one
//! `#[test]` iterates the table. Adding a (model, codec) SKU to the gate is
//! therefore exactly one table row — the frozen fixture + env wiring live on the
//! row, the gate logic is shared. This consolidates the four hand-written Bonsai
//! harnesses (the PTQ1_0 / PQ2_0 oracle tests and their perplexity siblings) into
//! one gate so there is a single place every codec is proven, not overlapping
//! per-codec copies.
//!
//! SELF-CONTAINED / CI-SAFE. The big GGUFs (6-7 GB) live out of git, so the real
//! serve is env-gated per SKU: when a SKU's artifacts/env are absent the real
//! gates skip with a note, while the committed per-SKU fixture (argmax, top-k
//! set, reference PPL) still drives self-contained assertions. So a CI box
//! without the GGUFs still compiles and passes; an Apple box WITH the env runs
//! the full three-signal gate.
//!
//! Shared helpers (extracted from the hand-written harnesses):
//!   * `build_shell(sku, gguf, context)` — one-time GGUF->`.zt` import (cached),
//!     trace, contract, residency, `Shell::load`, and the three RHT sign-bank
//!     bindings. `None` means "legitimately skip" (no device / won't fit).
//!   * `serve_logits(sku, gguf, tokens)` — build a shell and fire one prefill.
//!   * `oracle_gate(sku, gguf)`  — argmax == ref + top-k set + cosine >= min.
//!   * `ppl_gate(sku, gguf)`     — pie PPL vs the reference PPL within `ppl_tol`.
//!
//! The Bonsai serve machinery (`d27b_bonsai`, the online-Hadamard rotation-undo
//! incl. the V-fix, the `gdn_v_grouped` v-head reorder, the beta-first `in_ba`,
//! the sign binding) is codec-AGNOSTIC — it operates on activations/forward, not
//! the weight bank — so the only thing that changes between the two Bonsai rows
//! is the weight `Dtype` and the per-SKU env/fixture.
//!
//! Per-SKU env (names live on the `Sku` row):
//! * `<gguf_env>` — the real GGUF (required to run the real gates).
//! * `<zt_env>` — served `.zt` cache (optional; defaults beside the GGUF).
//! * `<oracle_dump_env>` — the reference's final-logit vector as little-endian
//!   f32 (`04639__result_output.f32`), or its directory; the cosine is asserted
//!   against it when present.
//! * `<corpus_ids_env>` — raw little-endian u32 token-id stream the reference
//!   scored (shared `PPL_IDS`); required for the ppl gate.
//! * `<ref_ppl_env>` — the reference's PPL over those chunks; OVERRIDES the
//!   frozen fixture `ref_ppl` when set.
//! * `PPL_CTX` / `PPL_CHUNKS` / `PPL_FIRST` — corpus windowing (shared; mirror
//!   the fork's per-chunk window: KV reset per chunk, each chunk `n_ctx` tokens,
//!   NLL over `[first .. n_ctx-1)` with `first = n_ctx/2`).

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use checkpoint::executor::Execution;
use checkpoint::file::read::parse_metadata;
use checkpoint::file::write::Writer;
use checkpoint::plan::{CONVERT_TILE_MAP_MASK, StorageTarget};

use engine::runahead::Runahead;
use engine_metal::weights::AdapterPlane;
use engine_metal::{Boot, Fault, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Dtype, Platform, Request, trace_hybrid};
use models::qwen_3::forward::Facts;
use models::qwen_3::model::Model;
use models::qwen_3::rotation::{self, BONSAI_SIGN_WIDTHS};

/// The fork prompt ("The capital of France is"), no BOS, greedy readout of the
/// last token. Shared across SKUs — the reference oracle is this exact prompt.
const PROMPT_IDS: [u32; 5] = [760, 6511, 314, 9338, 369];
/// `ĠParis` — the fork argmax every Bonsai SKU must reproduce.
const ORACLE_ARGMAX: u32 = 11751;
const VOCAB: usize = 248_320;

/// The committed, frozen per-SKU reference fixture. These are the reference
/// engine's answers (the llama fork on the same GGUF, greedy / seed 1234): the
/// argmax, the top-8 (id, logit) by descending logit, and the held-out-corpus
/// PPL. They drive the self-contained assertions (so CI without the GGUF still
/// proves something) and anchor the real gates.
struct Fixture {
    /// The reference argmax id (== `top8[0].0`).
    argmax: u32,
    /// The reference top-8 (id, logit), rank-0 first.
    top8: [(u32, f32); 8],
    /// The reference engine's PPL over the default corpus window; the
    /// `ref_ppl_env` overrides this at runtime when a more exact figure exists.
    ref_ppl: f64,
}

/// A quantized SKU the gate holds to the reference engine: a (model, codec) pair
/// plus the frozen fixture, the tolerances, and the per-SKU env wiring. Adding a
/// SKU is exactly one of these rows in `SKUS`.
struct Sku {
    /// Short SKU id, e.g. `bonsai-ptq1_0`.
    name: &'static str,
    /// The weight codec — the ONLY forward-relevant thing that varies across the
    /// two Bonsai rows (`d27b_bonsai` is codec-agnostic).
    dtype: Dtype,
    /// The KV dtype passed to `Model::d27b_bonsai`.
    kv: Dtype,
    /// Tensor-parallel width passed to `Model::d27b_bonsai`.
    tp: u32,
    /// Env naming the real GGUF (required for the real gates).
    gguf_env: &'static str,
    /// Env naming the served `.zt` cache (optional).
    zt_env: &'static str,
    /// Default `.zt` extension when `zt_env` is unset (beside the GGUF).
    zt_ext: &'static str,
    /// Env naming the reference final-logit dump (for the cosine).
    oracle_dump_env: &'static str,
    /// Env naming the reference-scored corpus id stream (for the ppl gate).
    corpus_ids_env: &'static str,
    /// Env overriding the fixture `ref_ppl` with the reference's measured PPL.
    ref_ppl_env: &'static str,
    /// Minimum accepted final-logit cosine vs the reference.
    cosine_min: f64,
    /// Max accepted relative |pie - ref| / ref perplexity gap.
    ppl_tol: f64,
    /// The frozen reference fixture.
    fixture: Fixture,
}

impl Sku {
    fn model(&self) -> Model {
        Model::d27b_bonsai(self.dtype, self.kv, self.tp)
    }
}

/// THE SKU TABLE. One row per served (model, codec) SKU. Add a SKU here and it
/// is gated — no other edit needed.
const SKUS: &[Sku] = &[
    // Ternary-Bonsai-2-27B, PTQ1_0 (ternary, 1.0-bpw).
    Sku {
        name: "bonsai-ptq1_0",
        dtype: Dtype::Ptq1_0,
        kv: Dtype::Bf16,
        tp: 1,
        gguf_env: "BONSAI_GGUF",
        zt_env: "BONSAI_ZT",
        zt_ext: "served.zt",
        oracle_dump_env: "ORACLE_DUMP",
        corpus_ids_env: "PPL_IDS",
        ref_ppl_env: "FORK_PPL",
        cosine_min: 0.999,
        ppl_tol: 0.03,
        fixture: Fixture {
            argmax: ORACLE_ARGMAX,
            top8: [
                (11751, 14.73272),
                (303, 10.61042),
                (198, 10.59325),
                (524, 10.47614),
                (264, 10.35187),
                (25, 10.27576),
                (248046, 10.16857),
                (1076, 9.81132),
            ],
            // PTQ1_0 and PQ2_0 are the SAME Hadamard-rotated model at different
            // bits; their fork oracles match to ~1e-5, so PTQ1_0 inherits the
            // PQ2_0 fork PPL as its frozen reference (no separate PTQ1_0 fork-ppl
            // run was preserved). Supply the exact PTQ1_0 figure via `FORK_PPL`
            // to override.
            ref_ppl: 7.9393,
        },
    },
    // Ternary-Bonsai-2-27B, PQ2_0 (positional quaternary, 2.125-bpw).
    Sku {
        name: "bonsai-pq2_0",
        dtype: Dtype::Pq2_0,
        kv: Dtype::Bf16,
        tp: 1,
        gguf_env: "BONSAI_PQ2_GGUF",
        zt_env: "BONSAI_PQ2_ZT",
        zt_ext: "pq2.served.zt",
        oracle_dump_env: "PQ2_ORACLE_DUMP",
        corpus_ids_env: "PPL_IDS",
        ref_ppl_env: "PQ2_FORK_PPL",
        cosine_min: 0.999,
        ppl_tol: 0.03,
        fixture: Fixture {
            argmax: ORACLE_ARGMAX,
            top8: [
                (11751, 14.73272),
                (303, 10.61043),
                (198, 10.59326),
                (524, 10.47616),
                (264, 10.35188),
                (25, 10.27577),
                (248046, 10.16858),
                (1076, 9.81133),
            ],
            // The PQ2_0 fork `perplexity --save-all-logits` over `ids_c512x4`.
            ref_ppl: 7.9393,
        },
    },
];

// ---- env / path helpers -------------------------------------------------

fn gguf_path(sku: &Sku) -> Option<PathBuf> {
    let path = PathBuf::from(std::env::var_os(sku.gguf_env)?);
    path.is_file().then_some(path)
}

fn zt_path(sku: &Sku, gguf: &Path) -> PathBuf {
    if let Some(p) = std::env::var_os(sku.zt_env) {
        return PathBuf::from(p);
    }
    gguf.with_extension(sku.zt_ext)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The reference's final-logit vector, from `oracle_dump_env` — either the
/// `04639__result_output.f32` file directly or the directory that holds it.
fn oracle_logits(sku: &Sku) -> Option<Vec<f32>> {
    let p = PathBuf::from(std::env::var_os(sku.oracle_dump_env)?);
    let path = if p.is_dir() {
        p.join("04639__result_output.f32")
    } else {
        p
    };
    let bytes = std::fs::read(&path).ok()?;
    assert_eq!(
        bytes.len(),
        VOCAB * 4,
        "the reference logit dump is {VOCAB} little-endian f32"
    );
    Some(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
    )
}

/// The reference-scored corpus id stream, from `corpus_ids_env` (raw LE u32).
/// Defaults to the committed `tests/bonsai/ppl_ids.bin` (the fork's own 2048-id
/// WikiText-2 stream, add_bos=false) so the ppl gate is self-contained whenever a
/// SKU's GGUF is present; `corpus_ids_env` overrides it with another stream.
fn corpus_ids(sku: &Sku) -> Option<Vec<u32>> {
    let p = match std::env::var_os(sku.corpus_ids_env) {
        Some(p) => PathBuf::from(p),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/bonsai/ppl_ids.bin"),
    };
    let bytes = std::fs::read(&p).ok()?;
    assert_eq!(
        bytes.len() % 4,
        0,
        "{} is a little-endian u32 stream",
        sku.corpus_ids_env
    );
    Some(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect(),
    )
}

// ---- math helpers -------------------------------------------------------

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, v) in logits.iter().enumerate() {
        if *v > logits[best] {
            best = at;
        }
    }
    best as u32
}

/// The ranked top-`k` ids by descending logit.
fn top_ids(logits: &[f32], k: usize) -> Vec<u32> {
    let mut order: Vec<u32> = (0..logits.len() as u32).collect();
    order.sort_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    order.truncate(k);
    order
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        dot += f64::from(*x) * f64::from(*y);
        na += f64::from(*x) * f64::from(*x);
        nb += f64::from(*y) * f64::from(*y);
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn word(len: usize) -> u64 {
    Facts::of(&Request::new(len as u32, false)).word()
}

/// -log softmax(logits)[target], numerically stable (full-vocab, as the fork).
fn nll_of(logits: &[f32], target: u32) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for &l in logits {
        sum += f64::from(l - max).exp();
    }
    let logz = f64::from(max) + sum.ln();
    logz - f64::from(logits[target as usize])
}

// ---- serve helpers (shared by both gates) -------------------------------

/// Model-aware one-time import: GGUF -> served `.zt`, under the pie param names
/// the serve trace reads. Cached: skipped when the `.zt` already exists.
fn import_zt(sku: &Sku, gguf: &Path, out: &Path) {
    if out.is_file() {
        eprintln!("[{}] reusing cached artifact {out:?}", sku.name);
        return;
    }
    eprintln!(
        "[{}] importing {gguf:?} -> {out:?} (one-time decode)",
        sku.name
    );
    let metadata = parse_metadata(gguf).expect("parse the GGUF metadata");
    let src = ztensor_compat::index(gguf).expect("open the GGUF as a source");
    let contract = sku
        .model()
        .import_from_gguf(&src, Platform::Metal)
        .expect("the d27b_bonsai contract reads every plane of the Bonsai GGUF");
    drop(src);

    let target = StorageTarget {
        tile_map_mask: CONVERT_TILE_MAP_MASK,
        max_tile_bytes: 64 << 20,
        ..StorageTarget::default()
    };
    let plan = checkpoint::plan::compile(&metadata, &contract, target)
        .expect("the contract fits the checkpoint");
    let base = gguf.parent().unwrap_or_else(|| Path::new("."));
    let storage = Execution::new(&plan, base)
        .run()
        .expect("materialise every served plane");

    let mut writer = Writer::create(out, &BTreeMap::new()).expect("create the artifact");
    // The ztensor writer demands canonical (name-sorted) insertion order.
    let mut decls: Vec<_> = plan
        .tensors
        .iter()
        .filter(|d| d.visibility.is_public())
        .collect();
    decls.sort_by(|a, b| a.name.cmp(&b.name));
    for decl in decls {
        let bytes = storage
            .tensors
            .get(&decl.name)
            .unwrap_or_else(|| panic!("no bytes materialised for `{}`", decl.name));
        writer.add_tensor(decl, bytes).expect("write the plane");
    }
    writer.finish().expect("finish the artifact");
}

/// Bind the three RHT sign diagonals decoded from the GGUF metadata into the
/// Registered sign banks via `register_adapter`.
fn bind_signs(sku: &Sku, gguf: &Path, shell: &mut Shell) {
    let signs = rotation::signs_from_gguf(
        &ztensor_compat::index(gguf).expect("reopen the GGUF for its sign metadata"),
    )
    .expect("decode the Bonsai sign diagonals");
    let sign_bytes: Vec<(String, Vec<u8>)> = BONSAI_SIGN_WIDTHS
        .iter()
        .map(|&w| {
            let sv = signs
                .get(&w)
                .unwrap_or_else(|| panic!("the GGUF carries no sign width {w}"));
            (rotation::sign_param_name(w), sv.to_bf16_le_bytes())
        })
        .collect();
    let declared: BTreeSet<String> = shell.bank_seats().into_iter().map(|s| s.name).collect();
    let mut bound = 0usize;
    for (name, bytes) in &sign_bytes {
        if !declared.contains(name) {
            continue;
        }
        shell
            .register_adapter(
                0,
                &[AdapterPlane {
                    bank: name.as_str(),
                    bytes: bytes.as_slice(),
                }],
            )
            .unwrap_or_else(|why| panic!("bind the sign bank `{name}`: {why}"));
        bound += 1;
    }
    eprintln!(
        "[{}] bound {bound} sign banks (of {} declared)",
        sku.name,
        declared.len()
    );
}

/// Import + load a served Bonsai shell for `sku` at `context` and bind its sign
/// banks. `None` means a legitimate skip (this machine cannot hold the model);
/// genuine setup failures panic (as the hand-written harnesses did).
fn build_shell(sku: &Sku, gguf: &Path, context: u32) -> Option<Shell> {
    let zt = zt_path(sku, gguf);
    import_zt(sku, gguf, &zt);

    let model = sku.model();
    let trace = trace_hybrid("d27b-bonsai", &model, Platform::Metal);
    let src = ztensor_compat::index(&zt).expect("open the served artifact");
    let contract = checkpoint_dsl::own_contract(&src, &trace.params, 1, Platform::Metal)
        .expect("the artifact holds every checkpoint plane the trace reads");
    drop(src);
    let planes =
        engine_metal::weights::attachments(&trace, &contract, &zt).expect("pair the banks");
    let residency = engine_metal::experts::Plan::of(&trace, &planes, None).expect("full residency");

    let page_size = 16u32;
    let boot = Boot {
        trace,
        contract: &contract,
        checkpoint: &zt,
        budget: Budget::new(1, context),
        patches: None,
        voxels: None,
        profile: None,
        page_size,
        context,
        slots: 1,
        pages: context.div_ceil(page_size),
        runahead: Runahead::F1,
        residency,
    };
    let mut shell = match Shell::load(boot) {
        Ok(shell) => shell,
        Err(Fault::Residency(said)) => {
            eprintln!(
                "[{}] this machine cannot hold the 27B model — {said}; skipping",
                sku.name
            );
            return None;
        }
        Err(other) => panic!("the Bonsai shell loads: {other}"),
    };
    eprintln!(
        "[{}] weights_warm = {} ({} window(s))",
        sku.name,
        shell.weights_warm(),
        shell.weight_windows()
    );
    bind_signs(sku, gguf, &mut shell);
    Some(shell)
}

/// Build a shell and fire `tokens` as one chunked prefill; the readout is the
/// final token's vocab-wide logit row. `None` on a legitimate skip.
fn serve_logits(sku: &Sku, gguf: &Path, tokens: &[u32]) -> Option<Vec<f32>> {
    // A 64-wide context comfortably holds the 5-token oracle prefill.
    let context = (tokens.len() as u32).next_multiple_of(16).max(64);
    let mut shell = build_shell(sku, gguf, context)?;
    shell.open(0).expect("the slot opens");
    let out = shell
        .fire(&[Lane {
            slot: 0,
            word: word(tokens.len()),
            tokens,
        }])
        .expect("the Bonsai prefill fires");
    Some(out[0].clone())
}

// ---- the gates ----------------------------------------------------------

/// Assert the committed fixture is well-formed. Runs even with no device and no
/// artifacts, so a CI box still proves the frozen reference is internally sound.
fn self_contained_checks(sku: &Sku) {
    let fx = &sku.fixture;
    assert_eq!(
        fx.top8[0].0, fx.argmax,
        "[{}] fixture rank-0 id must equal its argmax",
        sku.name
    );
    assert_eq!(
        fx.argmax, ORACLE_ARGMAX,
        "[{}] every Bonsai SKU's reference argmax is ĠParis (id {ORACLE_ARGMAX})",
        sku.name
    );
    let ids: BTreeSet<u32> = fx.top8.iter().map(|(id, _)| *id).collect();
    assert_eq!(
        ids.len(),
        8,
        "[{}] fixture top-8 ids must be distinct",
        sku.name
    );
    for pair in fx.top8.windows(2) {
        assert!(
            pair[0].1 >= pair[1].1,
            "[{}] fixture top-8 must be sorted by descending logit",
            sku.name
        );
    }
    assert!(
        fx.ref_ppl.is_finite() && fx.ref_ppl > 1.0,
        "[{}] fixture ref_ppl {} is not a sane perplexity",
        sku.name,
        fx.ref_ppl
    );
    assert!(
        sku.cosine_min > 0.0 && sku.cosine_min <= 1.0,
        "[{}] cosine_min {} out of range",
        sku.name,
        sku.cosine_min
    );
    assert!(
        sku.ppl_tol > 0.0 && sku.ppl_tol < 1.0,
        "[{}] ppl_tol {} out of range",
        sku.name,
        sku.ppl_tol
    );
    eprintln!(
        "[{}] self-contained OK — fixture argmax {} (ĠParis), ref_ppl {:.4}, cosine_min {}, ppl_tol {:.1}%",
        sku.name,
        fx.argmax,
        fx.ref_ppl,
        sku.cosine_min,
        100.0 * sku.ppl_tol
    );
}

/// Serve the fork prompt and gate the readout: finite row, argmax == the
/// reference argmax, top-8 id SET == the reference top-8 set, and (when the dump
/// is present) final-logit cosine >= `cosine_min`.
fn oracle_gate(sku: &Sku, gguf: &Path) {
    let Some(logits) = serve_logits(sku, gguf, &PROMPT_IDS) else {
        return;
    };
    let fx = &sku.fixture;

    let nan = logits.iter().filter(|v| v.is_nan()).count();
    let (lo, hi) = logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| {
            (a.min(v), b.max(v))
        });
    eprintln!(
        "[{}] readout width {} — {nan} NaN, range [{lo:.4}, {hi:.4}]",
        sku.name,
        logits.len(),
    );
    assert_eq!(
        logits.len(),
        VOCAB,
        "[{}] the readout is one vocab-wide row",
        sku.name
    );
    assert_eq!(nan, 0, "[{}] the served logits carry {nan} NaN", sku.name);
    assert!(
        hi - lo > 1.0,
        "[{}] the served logits span only {}",
        sku.name,
        hi - lo
    );

    let got_argmax = argmax(&logits);
    let top8 = top_ids(&logits, 8);
    eprintln!(
        "[{}] pie argmax id {} (logit {:.5}); reference argmax id {} (ĠParis) — {}",
        sku.name,
        got_argmax,
        logits[got_argmax as usize],
        fx.argmax,
        if got_argmax == fx.argmax {
            "MATCH"
        } else {
            "MISS"
        },
    );
    eprintln!("[{}] pie top-8 ids       {top8:?}", sku.name);
    eprintln!(
        "[{}] reference top-8 ids {:?}",
        sku.name,
        fx.top8.iter().map(|(id, _)| *id).collect::<Vec<_>>()
    );

    // (headline) the served argmax IS the reference's `ĠParis`.
    assert_eq!(
        got_argmax, fx.argmax,
        "[{}] serves id {} (ĠParis); got id {got_argmax}. A miss means the Bonsai-GGUF \
         v-head reorder (tiled->block, gated on `gdn_v_grouped`), the beta-first `in_ba`, or \
         the attention V-rotation regressed — re-import the GGUF so the `.served.zt` rebakes it.",
        sku.name, fx.argmax,
    );

    // (self-contained vs fixture) pie's top-8 id SET equals the reference top-8
    // set, order-independent — a cheap guard that needs no dump and still trips
    // on a trunk divergence that reshuffles the head without moving the argmax.
    let got_top8: BTreeSet<u32> = top8.iter().copied().collect();
    let want_top8: BTreeSet<u32> = fx.top8.iter().map(|(id, _)| *id).collect();
    assert_eq!(
        got_top8, want_top8,
        "[{}] pie's top-8 id set must equal the reference top-8 id set",
        sku.name
    );

    // (oracle) the final-logit cosine — what the V-fix buys and what argmax alone
    // cannot see. Asserted only when the reference dump is present.
    if let Some(oracle) = oracle_logits(sku) {
        let cos = cosine(&logits, &oracle);
        let atol = max_abs(&logits, &oracle);
        eprintln!(
            "[{}] cosine {cos:.7}  max-abs-err {atol:.5}  vs the reference logit vector",
            sku.name
        );
        assert!(
            cos >= sku.cosine_min,
            "[{}] final-logit cosine {cos:.7} < {} vs the reference — a V-class trunk \
             divergence (e.g. attention V built from the bare residual, not the rotated input)",
            sku.name,
            sku.cosine_min,
        );
    } else {
        eprintln!(
            "[{}] {} unset; cosine not asserted (argmax + top-8 set still gated)",
            sku.name, sku.oracle_dump_env
        );
    }
    eprintln!(
        "[{}] ORACLE GATE PASSED — pie reaches ĠParis, end to end on Metal",
        sku.name
    );
}

/// Teacher-forced perplexity over the reference-scored corpus, mirroring the
/// fork's per-chunk window, gated within `ppl_tol` of the reference PPL (env
/// override or the frozen fixture).
fn ppl_gate(sku: &Sku, gguf: &Path) {
    let Some(ids) = corpus_ids(sku) else {
        eprintln!(
            "[{}] no corpus ids (committed fixture missing and {} unset); ppl gate skipped",
            sku.name, sku.corpus_ids_env
        );
        return;
    };

    let n_ctx = env_usize("PPL_CTX", 512);
    let first = env_usize("PPL_FIRST", n_ctx / 2);
    let avail = ids.len() / n_ctx;
    let n_chunk = env_usize("PPL_CHUNKS", avail).min(avail);
    assert!(
        n_chunk >= 1,
        "[{}] need at least one full chunk of {n_ctx} ids",
        sku.name
    );
    assert!(
        first + 1 < n_ctx,
        "[{}] first ({first}) leaves no scored positions",
        sku.name
    );
    let per_chunk = n_ctx - 1 - first;
    eprintln!(
        "[{}] ppl: n_ctx={n_ctx} first={first} chunks={n_chunk} scored/chunk={per_chunk} \
         total={} ids_available={} (prefill fires m={} rows)",
        sku.name,
        per_chunk * n_chunk,
        ids.len(),
        first + 1,
    );

    let Some(mut shell) = build_shell(sku, gguf, n_ctx as u32) else {
        return;
    };

    let started = std::time::Instant::now();
    let mut nll = 0.0f64;
    let mut count = 0usize;
    for c in 0..n_chunk {
        let chunk = &ids[c * n_ctx..c * n_ctx + n_ctx];
        shell
            .open(0)
            .expect("the slot opens (KV reset for this chunk)");

        // Prefill [0..=first] in one fire; its readout is position `first`,
        // scoring token `first+1`.
        let prefill = &chunk[0..=first];
        let out = shell
            .fire(&[Lane {
                slot: 0,
                word: word(prefill.len()),
                tokens: prefill,
            }])
            .expect("the chunk prefill fires");
        assert!(
            out[0].len() > 1,
            "[{}] readout is one vocab-wide row",
            sku.name
        );
        nll += nll_of(&out[0], chunk[first + 1]);
        count += 1;

        // Incrementally decode [first+1 ..= n_ctx-2] (m=1), each scoring next.
        for p in (first + 1)..=(n_ctx - 2) {
            let fed = [chunk[p]];
            let out = shell
                .fire(&[Lane {
                    slot: 0,
                    word: word(1),
                    tokens: &fed,
                }])
                .unwrap_or_else(|why| panic!("decode p={p} (chunk {c}) fires: {why}"));
            nll += nll_of(&out[0], chunk[p + 1]);
            count += 1;
        }

        let mean = nll / count as f64;
        eprintln!(
            "[{}] ppl: [chunk {}] cumulative PPL = {:.4}  (mean NLL {:.5}, {count} tokens, {:.1}s)",
            sku.name,
            c + 1,
            mean.exp(),
            mean,
            started.elapsed().as_secs_f64(),
        );
    }

    let mean = nll / count as f64;
    let ppl = mean.exp();
    assert!(
        count > 0 && ppl.is_finite(),
        "[{}] scored a finite ppl",
        sku.name
    );
    eprintln!(
        "[{}] ppl: PIE FINAL  PPL = {ppl:.4}  mean NLL = {mean:.5}  over {count} tokens ({:.1}s)",
        sku.name,
        started.elapsed().as_secs_f64(),
    );

    // The reference PPL: the env override when set, else the frozen fixture.
    let reference = std::env::var(sku.ref_ppl_env)
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(sku.fixture.ref_ppl);
    let source = if std::env::var_os(sku.ref_ppl_env).is_some() {
        sku.ref_ppl_env
    } else {
        "fixture"
    };
    let rel = (ppl - reference).abs() / reference;
    eprintln!(
        "[{}] ppl: REF PPL = {reference:.4} ({source})  pie PPL = {ppl:.4}  delta = {:+.4} ({:+.2}%)  tol = {:.1}%",
        sku.name,
        ppl - reference,
        100.0 * (ppl - reference) / reference,
        100.0 * sku.ppl_tol,
    );
    assert!(
        rel <= sku.ppl_tol,
        "[{}] pie ppl {ppl:.4} vs reference ppl {reference:.4} is {:.2}% off (> {:.1}% tol) — \
         a quality divergence the oracle match missed",
        sku.name,
        100.0 * rel,
        100.0 * sku.ppl_tol,
    );
    eprintln!(
        "[{}] PPL GATE PASSED — pie serves at the reference engine's quality",
        sku.name
    );
}

/// THE GATE. One parameterized test over the SKU table: every row gets the
/// self-contained fixture checks always, and the full three-signal real gate
/// (argmax + cosine + perplexity) whenever its artifacts/env are present.
#[test]
fn the_quantized_skus_match_the_reference() {
    for sku in SKUS {
        eprintln!("\n==== SKU {} ({:?}) ====", sku.name, sku.dtype);

        // Always: the committed fixture must be internally sound (CI-safe).
        self_contained_checks(sku);

        if !engine_metal::device::present() {
            eprintln!(
                "[{}] no Metal device; real gates skipped (self-contained passed)",
                sku.name
            );
            continue;
        }
        let Some(gguf) = gguf_path(sku) else {
            eprintln!(
                "[{}] {} unset or missing; real gates skipped (self-contained passed)",
                sku.name, sku.gguf_env
            );
            continue;
        };

        // The real three-signal gate.
        oracle_gate(sku, &gguf);
        ppl_gate(sku, &gguf);
    }
}
