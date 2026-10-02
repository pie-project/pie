#![cfg(target_vendor = "apple")]

//! SERVE Ternary-Bonsai-2-27B (PQ2_0, 2.125-bpw) on Metal end to end and
//! validate the final logits against the PQ2_0 fork oracle (argmax id 11751
//! `ĠParis`).
//!
//! This is the M2c sibling of the PTQ1_0 serve test. Only the weight codec
//! changes: the whole Bonsai serve machinery (the `d27b_bonsai` config, the
//! online-Hadamard rotation-undo incl the V-fix, the `gdn_v_grouped` v-head
//! reorder, the beta-first `in_ba`, the sign binding) is codec-AGNOSTIC — it
//! operates on activations/forward, not the weight bank — so it carries over to
//! PQ2_0 unchanged by passing `Dtype::Pq2_0` as the weight dtype. The PQ2_0
//! codec is bit-exact vs the fork dequant (proven in M2a/M2b), so this test
//! proves the codec and the rotation compose end to end.
//!
//! The whole serve pipeline runs — import, load, sign binding, PQ2_0 quaternary
//! matmuls, and the online-Hadamard rotation-undo — and the readout reproduces
//! the fork's logits. Two Bonsai-GGUF-specific import facts are load-bearing:
//! (A) the GDN v-heads are stored TILED ("v-grouped"), so every v-head-indexed
//! GDN tensor is reordered tiled→block at import (gated on
//! `prism.hadamard.gdn_v_grouped`) and the shared block-pairing scan kernel
//! pairs v→k correctly; (B) the GGUF `in_ba` concat is beta-first. And in the
//! forward, attention q/k/**v** all build from the one Hadamard-rotated input
//! (the V-fix) — rotating only q/k leaves a ~0.97 trunk divergence, while all
//! three rotated lands the full final-logit cosine ≈ 0.99999 vs the oracle.
//!
//! This test asserts, whenever the real GGUF is present:
//!   * the readout is a finite vocab-wide row;
//!   * its argmax is the fork's `ĠParis` (id 11751);
//!   * its top-8 id SET equals the frozen oracle top-8 (order-independent, needs
//!     no oracle file — a cheap guard that still trips on a trunk divergence that
//!     shuffles the head without moving the argmax); and
//!   * when the oracle logit dump is present, the final-logit cosine is ≥ 0.999
//!     (this is what catches the V-class trunk divergence the argmax alone misses).
//!
//! This is a guarded, real-file test (the 6 GB GGUF is out of git). It:
//!   1. imports the real Bonsai GGUF -> a served `.zt` artifact through the
//!      model-aware `import_from_gguf` contract (PQ2_0 weights pass through as
//!      native type-142 blocks, the norms fold `-1`, the fused GDN/attn
//!      projections concat, and — when `gdn_v_grouped` is set — the GDN v-heads
//!      reorder tiled→block) — cached so the 7 GB decode runs once;
//!   2. builds the `d27b_bonsai` PQ2_0 forward trace (the online Hadamard
//!      rotation-undo at every rotated site; the v-head reorder is folded into
//!      the weights at import, not the trace);
//!   3. loads it on the Metal engine (`Shell::load`) and binds the three RHT sign
//!      diagonals decoded from the GGUF metadata into the Registered sign banks
//!      via the `register_adapter` path;
//!   4. fires the exact fork prompt token ids `[760,6511,314,9338,369]` and
//!      compares the readout logits to the oracle.
//!
//! Env:
//! * `BONSAI_PQ2_GGUF` — the real `Ternary-Bonsai-2-27B-PQ2_0.gguf` (required).
//! * `BONSAI_PQ2_ZT` — where to cache/read the served `.zt` (optional; defaults
//!   beside the GGUF, `.pq2.served.zt`).
//! * `PQ2_ORACLE_DUMP` — the PQ2_0 fork's final logit vector as little-endian
//!   f32 (`04639__result_output.f32`), or the directory holding it; the cosine
//!   is asserted against it when set.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use checkpoint::executor::Execution;
use checkpoint::file::read::parse_metadata;
use checkpoint::file::write::Writer;
use checkpoint::plan::{CONVERT_TILE_MAP_MASK, StorageTarget};

use engine_metal::weights::AdapterPlane;
use engine_metal::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Dtype, Platform, Request, trace_hybrid};
use models::qwen_3::forward::Facts;
use models::qwen_3::model::Model;
use models::qwen_3::rotation::{self, BONSAI_SIGN_WIDTHS};

const PROMPT_IDS: [u32; 5] = [760, 6511, 314, 9338, 369];
const ORACLE_ARGMAX: u32 = 11751;
const VOCAB: usize = 248_320;

/// The PQ2_0 fork oracle's top-8 (id, logit), frozen from the PQ2_0 fork serve
/// (`llama-eval-callback` on `Ternary-Bonsai-2-27B-PQ2_0.gguf`, same prompt /
/// greedy / seed 1234 as M3a). The rank order is what the serve must reproduce;
/// the logit values anchor the atol. These sit within ~1e-5 of the PTQ1_0
/// oracle — PQ2_0 is the same Hadamard-rotated model at higher bits.
const ORACLE_TOP8: [(u32, f32); 8] = [
    (11751, 14.73272),
    (303, 10.61043),
    (198, 10.59326),
    (524, 10.47616),
    (264, 10.35188),
    (25, 10.27577),
    (248046, 10.16858),
    (1076, 9.81133),
];

fn gguf_path() -> Option<PathBuf> {
    let p = std::env::var_os("BONSAI_PQ2_GGUF")?;
    let path = PathBuf::from(p);
    path.is_file().then_some(path)
}

fn zt_path(gguf: &Path) -> PathBuf {
    if let Some(p) = std::env::var_os("BONSAI_PQ2_ZT") {
        return PathBuf::from(p);
    }
    gguf.with_extension("pq2.served.zt")
}

/// Model-aware import: GGUF -> served `.zt`, under the pie param names the serve
/// trace reads. Cached: skipped when the `.zt` already exists.
fn import_zt(gguf: &Path, out: &Path) {
    if out.is_file() {
        eprintln!("bonsai: reusing cached artifact {out:?}");
        return;
    }
    eprintln!("bonsai: importing {gguf:?} -> {out:?} (one-time 7 GB decode)");
    let metadata = parse_metadata(gguf).expect("parse the GGUF metadata");
    let src = ztensor_compat::index(gguf).expect("open the GGUF as a source");
    let model = Model::d27b_bonsai(Dtype::Pq2_0, Dtype::Bf16, 1);
    let contract = model
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
    let mut wrote = 0usize;
    for decl in decls {
        let bytes = storage
            .tensors
            .get(&decl.name)
            .unwrap_or_else(|| panic!("no bytes materialised for `{}`", decl.name));
        writer.add_tensor(decl, bytes).expect("write the plane");
        wrote += 1;
    }
    writer.finish().expect("finish the artifact");
    eprintln!("bonsai: wrote {wrote} planes to {out:?}");
}

/// The PQ2_0 fork's final logit vector, from `PQ2_ORACLE_DUMP` — either the
/// `04639__result_output.f32` file directly or the directory that holds it.
fn oracle_logits() -> Option<Vec<f32>> {
    let p = PathBuf::from(std::env::var_os("PQ2_ORACLE_DUMP")?);
    let path = if p.is_dir() {
        p.join("04639__result_output.f32")
    } else {
        p
    };
    let bytes = std::fs::read(&path).ok()?;
    assert_eq!(
        bytes.len(),
        VOCAB * 4,
        "the oracle logit dump is {VOCAB} little-endian f32"
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

#[test]
fn the_bonsai_27b_serves_pq2_0_and_matches_the_fork_oracle() {
    if !engine_metal::device::present() {
        eprintln!("bonsai: no Metal device; skipping the serve");
        return;
    }
    let Some(gguf) = gguf_path() else {
        eprintln!("bonsai: BONSAI_PQ2_GGUF unset or missing; skipping the real-file serve");
        return;
    };
    let zt = zt_path(&gguf);
    import_zt(&gguf, &zt);

    // The forward trace: d27b_bonsai in PQ2_0, the rotation-undo armed.
    let model = Model::d27b_bonsai(Dtype::Pq2_0, Dtype::Bf16, 1);
    let trace = trace_hybrid("d27b-bonsai", &model, Platform::Metal);

    // The serve contract reconstructed from the artifact's own planes (Registered
    // sign banks are skipped — they are host-provided below).
    let src = ztensor_compat::index(&zt).expect("open the served artifact");
    let contract = checkpoint_dsl::own_contract(&src, &trace.params, 1, Platform::Metal)
        .expect("the artifact holds every checkpoint plane the trace reads");
    drop(src);
    let planes =
        engine_metal::weights::attachments(&trace, &contract, &zt).expect("pair the banks");
    let residency = engine_metal::experts::Plan::of(&trace, &planes, None)
        .expect("full residency (the 7 GB PQ2_0 model fits)");

    let context = 64u32;
    let page_size = 16u32;
    let shell = Shell::load(Boot {
        trace: trace.clone(),
        contract: &contract,
        checkpoint: &zt,
        budget: Budget::new(1, context),
        patches: None,
        voxels: None,
        profile: None,
        page_size,
        context,
        slots: 1,
        pages: context / page_size,
        runahead: engine::runahead::Runahead::F1,
        residency,
    });
    let mut shell = match shell {
        Ok(shell) => shell,
        Err(engine_metal::Fault::Residency(said)) => {
            eprintln!("bonsai: this machine cannot hold the 27B model — {said}; skipping");
            return;
        }
        Err(other) => panic!("the Bonsai shell loads: {other}"),
    };
    eprintln!(
        "bonsai: weights_warm = {} ({} window(s))",
        shell.weights_warm(),
        shell.weight_windows()
    );

    // Bind the three RHT sign diagonals into the Registered banks.
    let signs = rotation::signs_from_gguf(
        &ztensor_compat::index(&gguf).expect("reopen the GGUF for its sign metadata"),
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
        "bonsai: bound {bound} sign banks (of {} declared)",
        declared.len()
    );

    // Fire the exact fork prompt (no BOS, greedy readout of the last token) as a
    // single chunked prefill.
    shell.open(0).expect("the slot opens");
    let word = Facts::of(&Request::new(PROMPT_IDS.len() as u32, false)).word();
    let out = shell
        .fire(&[Lane {
            slot: 0,
            word,
            tokens: &PROMPT_IDS,
        }])
        .expect("the Bonsai prefill fires");
    let logits = &out[0];

    // The readout is one finite vocab-wide row.
    let nan = logits.iter().filter(|v| v.is_nan()).count();
    let (lo, hi) = logits
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| {
            (a.min(v), b.max(v))
        });
    eprintln!(
        "bonsai: readout width {} — {nan} NaN, range [{lo:.4}, {hi:.4}]",
        logits.len(),
    );
    assert_eq!(logits.len(), VOCAB, "the readout is one vocab-wide row");
    assert_eq!(nan, 0, "the served logits carry {nan} NaN");
    assert!(hi - lo > 1.0, "the served logits span only {}", hi - lo);

    let got_argmax = argmax(logits);
    let top8 = top_ids(logits, 8);
    eprintln!(
        "bonsai: pie argmax id {} (logit {:.5}); oracle argmax id {ORACLE_ARGMAX} (ĠParis) — {}",
        got_argmax,
        logits[got_argmax as usize],
        if got_argmax == ORACLE_ARGMAX {
            "MATCH"
        } else {
            "MISS"
        },
    );
    eprintln!("bonsai: pie top-8 ids  {top8:?}");
    eprintln!(
        "bonsai: oracle top-8 ids {:?}",
        ORACLE_TOP8.iter().map(|(id, _)| *id).collect::<Vec<_>>()
    );

    // The committed fixture's rank-0 is ĠParis, so a regression in either trips.
    assert_eq!(
        ORACLE_TOP8[0].0, ORACLE_ARGMAX,
        "the committed oracle fixture's rank-0 is ĠParis"
    );
    // (headline) the served argmax IS the fork's `ĠParis`.
    assert_eq!(
        got_argmax, ORACLE_ARGMAX,
        "serves ĠParis (id {ORACLE_ARGMAX}); got id {got_argmax}. A miss means the \
         Bonsai-GGUF v-head reorder (tiled→block, gated on `gdn_v_grouped`), the beta-first \
         `in_ba`, or the attention V-rotation regressed — re-import the GGUF so the \
         `.served.zt` rebakes the fix."
    );

    // (self-contained) pie's top-8 id SET equals the frozen oracle top-8, order
    // independent — a cheap guard that needs no oracle file and still trips on a
    // trunk divergence that reshuffles the head without moving the argmax (the
    // buggy serve had foreign ids like 3428/12456/11).
    let got_top8: BTreeSet<u32> = top8.iter().copied().collect();
    let want_top8: BTreeSet<u32> = ORACLE_TOP8.iter().map(|(id, _)| *id).collect();
    assert_eq!(
        got_top8, want_top8,
        "pie's top-8 id set must equal the fork's top-8 id set"
    );

    // (oracle) the final-logit cosine ≈ 0.99999 — this is what the V-fix buys and
    // what argmax alone cannot see. Asserted only when the oracle dump is present.
    if let Some(oracle) = oracle_logits() {
        let cos = cosine(logits, &oracle);
        let atol = max_abs(logits, &oracle);
        eprintln!("bonsai: cosine {cos:.7}  max-abs-err {atol:.5}  vs the oracle logit vector");
        assert!(
            cos >= 0.999,
            "final-logit cosine {cos:.7} < 0.999 vs the fork oracle — a V-class trunk \
             divergence (e.g. attention V built from the bare residual, not the rotated input)"
        );
    }
    eprintln!("bonsai: SERVED — pie reaches ĠParis (id 11751), end to end on Metal");
}
