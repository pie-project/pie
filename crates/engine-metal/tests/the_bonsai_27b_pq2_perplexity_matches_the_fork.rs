#![cfg(target_vendor = "apple")]

//! REAL quality eval of pie-served Ternary-Bonsai-2-27B (PQ2_0, 2.125-bpw):
//! token-level perplexity (mean NLL) over a held-out corpus, compared to the
//! fork — the PQ2_0 sibling of `the_bonsai_27b_perplexity_matches_the_fork`.
//!
//! The one weight codec changes to `Dtype::Pq2_0`; everything else (the
//! `d27b_bonsai` config, the online-Hadamard rotation-undo + V-fix, the sign
//! binding, the fork-exact per-chunk windowing) carries over unchanged — those
//! operate on activations/forward, not the weight bank.
//!
//! WHY THIS TEST EXISTS BEYOND THE 5-TOKEN ORACLE: the PQ2_0 prefill routes its
//! m>=`PQ2_0_QMM_MIN_BATCH` projections through the GEMM-tiled qmm. That tiled
//! kernel is argmax-correct but NOT bit-identical to the qmv it replaces — it
//! can swap a near-tied rank-8 token. This test teacher-forces per-token NLL
//! over hundreds of corpus positions, exercising the tiled qmm on every chunk
//! prefill (m = PPL_FIRST+1 rows), and asks whether the tiled path still serves
//! at the fork engine's quality. A tiny gap is the expected serve tolerance; a
//! larger gap is a tiled-qmm precision finding the oracle argmax would miss.
//!
//! Apples-to-apples by construction (identical to the PTQ1_0 ppl test):
//!   * the corpus is tokenized ONCE, by the fork; we parse its exact id stream
//!     (`PPL_IDS`, raw little-endian u32) and feed the SAME ids to pie.
//!   * we mirror the fork's windowing: KV reset per chunk, each chunk `n_ctx`
//!     tokens, NLL over positions `[first .. n_ctx-1)` with `first = n_ctx/2`;
//!     `count = n_ctx - 1 - first` per chunk. add_bos=false (no per-chunk BOS).
//!
//! The chunk prefill fires positions `[0..=first]` in ONE batched fire (m =
//! `first+1` rows, e.g. 257 at the default ctx) — that is the m>=16 batch that
//! drives the tiled qmm (`kernels_metal::linear::quant::PQ2_0_QMM_MIN_BATCH`).
//! The incremental decode steps are m=1 (qmv). The within-pie control that turns
//! the tiled kernel OFF (raising that threshold so the prefill also routes to
//! qmv) isolates the tiled kernel's own ppl effect.
//!
//! Env:
//! * `BONSAI_PQ2_GGUF` — the real PQ2_0 GGUF (required): import + sign metadata.
//! * `BONSAI_PQ2_ZT`   — served `.zt` cache (optional; defaults `.pq2.served.zt`).
//! * `PPL_IDS`         — raw little-endian u32 token-id stream (required).
//! * `PPL_CTX`         — chunk length / context (default 512).
//! * `PPL_CHUNKS`      — number of chunks to score (default: all ids / ctx).
//! * `PPL_FIRST`       — first scored position within a chunk (default ctx/2).
//! * `FORK_PPL`        — the fork's PPL over these chunks (optional; asserted).
//! * `PPL_TOL`         — allowed relative |pie-fork|/fork (default 0.03 = 3%).

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

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn read_ids() -> Option<Vec<u32>> {
    let p = PathBuf::from(std::env::var_os("PPL_IDS")?);
    let bytes = std::fs::read(&p).ok()?;
    assert_eq!(bytes.len() % 4, 0, "PPL_IDS is a little-endian u32 stream");
    Some(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect(),
    )
}

/// The one-time GGUF -> served `.zt` import (identical to the PQ2_0 oracle test).
fn import_zt(gguf: &Path, out: &Path) {
    if out.is_file() {
        eprintln!("ppl: reusing cached artifact {out:?}");
        return;
    }
    eprintln!("ppl: importing {gguf:?} -> {out:?} (one-time 7 GB decode)");
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

#[test]
fn the_bonsai_27b_pq2_perplexity_matches_the_fork() {
    if !engine_metal::device::present() {
        eprintln!("ppl: no Metal device; skipping");
        return;
    }
    let Some(gguf) = gguf_path() else {
        eprintln!("ppl: BONSAI_PQ2_GGUF unset or missing; skipping the real-file eval");
        return;
    };
    let Some(ids) = read_ids() else {
        eprintln!("ppl: PPL_IDS unset or missing; skipping (nothing to score)");
        return;
    };

    let n_ctx = env_usize("PPL_CTX", 512);
    let first = env_usize("PPL_FIRST", n_ctx / 2);
    let avail = ids.len() / n_ctx;
    let n_chunk = env_usize("PPL_CHUNKS", avail).min(avail);
    assert!(n_chunk >= 1, "need at least one full chunk of {n_ctx} ids");
    assert!(
        first + 1 < n_ctx,
        "first ({first}) leaves no scored positions"
    );
    let per_chunk = n_ctx - 1 - first;
    eprintln!(
        "ppl: n_ctx={n_ctx} first={first} chunks={n_chunk} scored/chunk={per_chunk} \
         total={} ids_available={} (prefill fires m={} rows → tiled qmm when m>=PQ2_0_QMM_MIN_BATCH)",
        per_chunk * n_chunk,
        ids.len(),
        first + 1,
    );

    let zt = zt_path(&gguf);
    import_zt(&gguf, &zt);

    // Serve setup — identical to the PQ2_0 oracle test (Pq2_0 weight codec).
    let model = Model::d27b_bonsai(Dtype::Pq2_0, Dtype::Bf16, 1);
    let trace = trace_hybrid("d27b-bonsai", &model, Platform::Metal);
    let src = ztensor_compat::index(&zt).expect("open the served artifact");
    let contract = checkpoint_dsl::own_contract(&src, &trace.params, 1, Platform::Metal)
        .expect("the artifact holds every checkpoint plane the trace reads");
    drop(src);
    let planes =
        engine_metal::weights::attachments(&trace, &contract, &zt).expect("pair the banks");
    let residency = engine_metal::experts::Plan::of(&trace, &planes, None)
        .expect("full residency (the 7 GB PQ2_0 model fits)");

    let context = n_ctx as u32;
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
        pages: context.div_ceil(page_size),
        runahead: engine::runahead::Runahead::F1,
        residency,
    });
    let mut shell = match shell {
        Ok(shell) => shell,
        Err(engine_metal::Fault::Residency(said)) => {
            eprintln!("ppl: this machine cannot hold the 27B model — {said}; skipping");
            return;
        }
        Err(other) => panic!("the Bonsai shell loads: {other}"),
    };

    // Bind the three RHT sign diagonals (identical to the oracle test).
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
    }

    // Teacher-forced NLL, mirroring the fork's per-chunk window.
    let started = std::time::Instant::now();
    let mut nll = 0.0f64;
    let mut count = 0usize;
    for c in 0..n_chunk {
        let chunk = &ids[c * n_ctx..c * n_ctx + n_ctx];
        shell
            .open(0)
            .expect("the slot opens (KV reset for this chunk)");

        // Prefill positions [0..=first] in one fire (m = first+1 rows → tiled
        // qmm when m>=PQ2_0_QMM_MIN_BATCH); its readout is position `first`,
        // scoring token at `first+1`.
        let prefill = &chunk[0..=first];
        let out = shell
            .fire(&[Lane {
                slot: 0,
                word: word(prefill.len()),
                tokens: prefill,
            }])
            .expect("the chunk prefill fires");
        assert!(out[0].len() > 1, "readout is one vocab-wide row");
        nll += nll_of(&out[0], chunk[first + 1]);
        count += 1;

        // Incrementally decode positions [first+1 ..= n_ctx-2] (m=1, qmv).
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
            "ppl: [chunk {}] cumulative PPL = {:.4}  (mean NLL {:.5}, {count} tokens, {:.1}s)",
            c + 1,
            mean.exp(),
            mean,
            started.elapsed().as_secs_f64(),
        );
    }

    let mean = nll / count as f64;
    let ppl = mean.exp();
    eprintln!(
        "ppl: PIE FINAL  PPL = {ppl:.4}  mean NLL = {mean:.5}  over {count} tokens \
         ({:.1}s wall)",
        started.elapsed().as_secs_f64(),
    );

    if let Ok(fork) = std::env::var("FORK_PPL") {
        let fork: f64 = fork.parse().expect("FORK_PPL is a number");
        let tol: f64 = std::env::var("PPL_TOL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.03);
        let rel = (ppl - fork).abs() / fork;
        eprintln!(
            "ppl: FORK PPL = {fork:.4}  pie PPL = {ppl:.4}  delta = {:+.4} ({:+.2}%)  tol = {:.1}%",
            ppl - fork,
            100.0 * (ppl - fork) / fork,
            100.0 * tol,
        );
        assert!(
            rel <= tol,
            "pie ppl {ppl:.4} vs fork ppl {fork:.4} is {:.2}% off (> {:.1}% tol) — \
             a quality divergence the oracle match missed",
            100.0 * rel,
            100.0 * tol,
        );
    }
    assert!(count > 0 && ppl.is_finite(), "scored a finite ppl");
}
