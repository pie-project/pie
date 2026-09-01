//! **FIRST LIGHT FOR THE QWEN4 2-BIT SKU — THREE WALLS DOWN.**
//!
//! This file was written as `two_bit_moe_first_light.rs`'s twin one family
//! over: load `qwen38-flash-mlxu2-kv-bf16` through the catalog row, prefill,
//! decode, assert finite/spread/deterministic and hold the per-tensor affine
//! census against the artifact's own header. It could not do that, and the two
//! reasons it could not were recorded here as asserts rather than as prose, so
//! that the day either was lifted the assert would go RED and whoever lifted it
//! would have to say so here. **Both have been lifted, this is that saying —
//! and behind them was a third wall neither assert could have named.**
//!
//! # Wall one: this plane served no n-gram hasher — IT DOES NOW
//!
//! A qwen4's PLE gathers a hashed tri-gram memory at one early layer, and the
//! hash is `attention.ple_ngram_ids{,_chunked}`; its local mix is an
//! `attention.ssm_causal_conv1d` DILATED by `ngram_size`. `engine-metal`'s
//! `dispatch/attn.rs` answered `Unsupported` for the hasher's two arms by
//! NAME, and refused the convolution by PARAMETER at `dilation: 2..` while
//! serving it at 1 — and both applied equally to `qwen38-flash-mlxu4-kv-bf16`
//! and `qwen38-flash-bf16-kv-bf16`, which have shipped in the catalog all
//! along.
//!
//! Both doors are open. `attn/ple.metal` ports `ple.cuh` organ for organ —
//! the seed-derived odd multipliers, the xor fold over the window newest
//! first, the per-head modulus and offset, and the eos-segmentation rule that
//! masks every id behind a nearer eos — and `attn/ssm_causal_conv1d.metal`
//! takes its dilation whole, keeping `(conv_width − 1)·dilation + 1` rows of
//! history and striding its taps over them. The one thing that had to move
//! was WHERE THE HASH CONSTANTS LIVE: the CUDA entry hands them across the
//! launch ABI by value and this plane's `ArgValue` has no by-value blob seat,
//! so `engine_metal::scratch` lays a `u64` plane per distinct hashing and
//! writes it once at load.
//!
//! Both are measured on the card by
//! `engine-metal/tests/ple_conv_on_device.rs` — the hasher EXACTLY against
//! `kernels_metal::attn::ple::reference` (a hash off by one is a different
//! embedding row, so there is no band it could be inside), the convolution at
//! half a bf16 quantum against a host fp32 reference at dilation 1 and 3, with
//! the two answers held apart so a kernel that took the argument and did not
//! spend it would be caught. The arithmetic itself is pinned deviceless to
//! hand-computed integers in `kernels-metal`'s `--lib` suite.
//!
//! # Wall two: the miniature's n-gram table — RE-CARVED, AND IT AGREES NOW
//!
//! `Sawfwair/Qwen3.8-Flash-Next-MLX-Mixed-2bit`'s `mini-l4-e16-p8` used to
//! ship its PLE table as EIGHT of the SHIPPED model's 128 stored shards, kept
//! verbatim: 8 × 2 500 012 = 20 000 096 rows, which is not even a multiple of
//! the config's own `make_ngram_vocab_size_divisible_by: 128`, beside sixteen
//! published head offsets running to 300 001 275 — into a row space sixteen
//! times taller than the table under them. The load refused, by name and by
//! two integers, and this file asserted that it refused there and nowhere
//! earlier.
//!
//! It was the ARTIFACT that was wrong, and the artifact is what moved.
//! `benches/shrink_checkpoint.py`'s `PleCarve` re-cuts the table BY HEAD:
//! miniature head `h` takes the first of its own sixteen primes past
//! `ngram_vocab_size_base: 1250000` worth of rows out of head `h`'s segment
//! of the original — real rows of the real table at the head they belong to —
//! and the sixteen segments are concatenated and re-chopped into eight equal
//! shards of 2 500 192. The two published buffers beside it are rewritten to
//! the miniature's own primes and offsets, and `layer_multipliers` is left
//! alone because a shrink touches neither the seed nor the vocabulary.
//!
//! So the table is 20 001 536 rows, which is what `ngram_vocab_size_base:
//! 1250000` derives and what `models::qwen_4::model::Model::flash_mini`
//! declares; the last hashed row any head can name is 20 001 533; and config,
//! tensors and text all say one thing. `model/tests/the_qwen4_text_reads_the_
//! two_bit_miniature.rs` holds that agreement against the bytes.
//!
//! # Wall three, which nobody had written down
//!
//! With the table re-carved the shell LOADED — 1.4 s, 4.49 GiB of weights —
//! and the first fire refused. Wall one had opened two doors and there were
//! seven more behind them, none of which any census could see, because a
//! **bake is a compile and not a fire**: `what_this_plane_serves` was calling
//! every `qwen38-flash-*` row "clears the bake" while a prefill of one could
//! not reach its second layer. In the order they were met:
//!
//! 1. `elementwise.rmsnorm_grouped_plus_one` — the hyper-connection norm,
//!    whose gain bank spans the WIDE row (one plane per stream) where the
//!    per-head norm beside it shares one. Spent four times a layer.
//! 2. `elementwise.rmsnorm_gated` at `sigmoid` — refused by ENUM ARM and not
//!    by name, which is why it was on no list. The shader was already
//!    templated on the choice; only the instantiation and the entry's
//!    argument were missing, and qwen4's GatedDeltaNet is the sigmoid one.
//! 3. `elementwise.silu_scaled` — the shared expert's gate.
//! 4. `elementwise.hc_mix` and `elementwise.hc_inject` — the stream collapse
//!    and the write-back. The same residual algebra as dsv4's sinkhorn
//!    family with a different gate, so `elemwise/hc.metal` grew a second
//!    flavour rather than a second file.
//! 5. `elementwise.ple_gate` — the PLE's key·query gate.
//! 6. `layout.embed_concat` — sixteen hashed gathers per token, concatenated.
//!    This one needed NO new shader: an id and a slice are the same address at
//!    a different stride, so the head axis folds into the row axis in the grid
//!    and both existing embed points serve it.
//! 7. **And one that was not this plane's at all.** `models::qwen_4::forward`
//!    picked its routed-expert entry off an ALLOW-list of quantized dtypes —
//!    `Mxfp4 | MlxU4 | MlxU8` — so `MlxU2G128`, which carries its group in its
//!    name, fell through to the DENSE select and resolved a three-plane bank
//!    as one handle. `deepseek_v4::forward` had the test the right way round
//!    (the DENSE forms are the list) and this row now spells it the same.
//!
//! All seven are measured. `engine-metal/tests/qwen4_gated_residual_on_device.rs`
//! holds the five new arithmetic points against
//! `kernels_metal::elemwise::hc::reference` in bf16 quanta — and holds each one
//! APART from the plausible wrong port beside it, because a shared-plane norm,
//! a fan divided outside the sigmoid and an unsigned damping all land finite,
//! spread numbers — and holds the gather EXACTLY, since a gather off by one
//! head is a different row of a twenty-million-row table.
//!
//! # What first light claims
//!
//! There is **no external reference for this miniature** — four layers of
//! forty-eight and sixteen experts of five hundred and twelve. So this is not
//! token parity and does not pretend to be. It is four claims, each of which a
//! wrong load fails:
//!
//! 1. **it loads** — the import contract fits the real bytes and every plane
//!    seats on the device;
//! 2. **it fires** — a prefill and [`STEPS`] decodes complete, with the hashed
//!    n-gram gather and its dilated mix dispatching on the PLE layer of every
//!    fire;
//! 3. **the numbers are numbers** — finite, spread, and the same twice;
//! 4. **the census is truthful** — every affine bank the loaded PLAN carries
//!    is at the `(bits, group)` the file wrote it at, the re-carved n-gram
//!    table's row count included.
//!
//! Claim 4 is the one that separates a 2-bit load from a 2-bit-shaped one: a
//! plane dequantized at group 64 where the file wrote 32 lands the right
//! spread around the wrong centre, with no NaN to notice it by.
//!
//! ```text
//! cargo test -p engine-metal --release --test qwen4_two_bit_first_light -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use engine_metal::{Boot, Lane, Shell};
use kernels_metal::{RecurrentPool, Tensor};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};
use model_ir::Dtype;

/// The catalog row this file is written over.
const SKU: &str = "qwen38-flash-mlxu2-kv-bf16";

/// The repository the miniature lives in.
const REPO: &str = "models--Sawfwair--Qwen3.8-Flash-Next-MLX-Mixed-2bit";

/// How many decode fires follow the prefill.
///
/// Enough that the decode class is entered, re-entered, and entered again with
/// a longer cache than the prefill left. It also matters here for a reason the
/// dsv4 twin does not have: the PLE keeps an `ngram − 1` id history and a
/// dilated convolution history ACROSS fires, so a step that reads the wrong
/// row of either is only visible once several steps have run.
const STEPS: usize = 8;

/// The prompt. Nothing rides on the words — there is no reference continuation
/// for a four-layer sixteen-expert cut of a forty-eight-layer model. It is a
/// token sequence long enough to prefill a real rectangle and no more.
const PROMPT: &str = "The capital of France is the city of";

/// One shell at a time per process: these hold the whole weight table resident
/// and the measurements are only readable one at a time.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The snapshot: the checkpoint AND the tokenizer beside it.
/// `PIE_QWEN4_U2_SNAPSHOT` overrides where it is looked for.
fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_QWEN4_U2_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let usable = |path: &Path| {
        path.join("tokenizer.json").exists() && path.join("model.safetensors.index.json").exists()
    };
    // The suite runs as root over tailscale ssh, so `HOME` is not always the
    // owner's — the cache is named explicitly beside it.
    let homes = [
        std::env::var("HOME").unwrap_or_default(),
        "/Users/ingim".to_string(),
    ];
    homes.iter().find_map(|home| {
        let snapshots = Path::new(home)
            .join(".cache/huggingface/hub")
            .join(REPO)
            .join("snapshots");
        let mut found: Vec<PathBuf> = std::fs::read_dir(snapshots)
            .ok()?
            .filter_map(|entry| Some(entry.ok()?.path()))
            .filter(|path| usable(path))
            .collect();
        found.sort();
        found.into_iter().next()
    })
}

fn shards(snapshot: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(snapshot)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            name.ends_with(".safetensors").then_some(path)
        })
        .collect();
    found.sort();
    found
}

/// Every shard's header, joined: `name -> shape`. The miniature spans two
/// files and the n-gram table spans the seam between them, so one file's
/// header is not the artifact.
fn header(snapshot: &Path) -> Option<BTreeMap<String, Vec<u64>>> {
    let mut all = BTreeMap::new();
    for path in shards(snapshot) {
        let bytes = std::fs::read(&path).ok()?;
        let len = u64::from_le_bytes(bytes.get(..8)?.try_into().ok()?) as usize;
        let parsed: serde_json::Value = serde_json::from_slice(bytes.get(8..8 + len)?).ok()?;
        for (name, meta) in parsed.as_object()?.iter() {
            let Some(shape) = meta.get("shape").and_then(serde_json::Value::as_array) else {
                continue;
            };
            all.insert(
                name.clone(),
                shape.iter().filter_map(serde_json::Value::as_u64).collect(),
            );
        }
    }
    (!all.is_empty()).then_some(all)
}

/// **WALL ONE, ASKED OF THE PLAN AND THIS PLANE'S OWN ENTRIES — AND ANSWERED.**
///
/// No device is needed, and that is still the point. The question this test
/// used to ask was whether the trace names ops `engine-metal` refuses; it asked
/// the trace for its two hasher nodes and for the DILATION of its convolution
/// nodes, because a refusal by parameter is not a name and no census list can
/// carry it.
///
/// It asks the same two questions of the plan — the shape claim has not
/// changed and a PLE that moved should still be caught — and then asks the
/// entries themselves, through a recording sink, whether they still refuse.
/// They do not, and both spend the numbers they took:
/// `attention.ssm_causal_conv1d` marshals its dilation as a stated `int`
/// rather than dropping it, and the hasher marshals the shape its constants
/// plane describes.
///
/// **AND IT IS STILL TWO-DIRECTIONAL.** The day either entry starts refusing
/// again — or takes the dilation and does not state it — this goes red with
/// the entry's own sentence on it.
#[test]
fn this_plane_serves_the_qwen4_ngram_hasher_and_the_dilated_mix() {
    use model_ir::Attention;

    let trace = models::trace_of(SKU).expect("the catalog ships the qwen4 2-bit SKU");
    let trace = trace(Platform::Metal);

    let mut hasher = 0usize;
    let mut dilated = Vec::new();
    for node in &trace.nodes {
        match &node.op {
            model_ir::Operation::Attention(
                Attention::PleNgramIds { .. } | Attention::PleNgramIdsChunked { .. },
            ) => hasher += 1,
            model_ir::Operation::Attention(
                Attention::SsmCausalConv1d { dilation, .. }
                | Attention::SsmCausalConv1dChunked { dilation, .. },
            ) if *dilation >= 2 => dilated.push(*dilation),
            _ => {}
        }
    }

    assert_eq!(
        hasher, 2,
        "a qwen4 plan names the n-gram hasher twice — the prefill arm and the \
         chunked decode arm. If this count is now zero the PLE moved."
    );
    assert_eq!(
        dilated,
        vec![3, 3],
        "and the PLE's local mix is a causal conv DILATED by `ngram_size` — the \
         two nodes are the same op's prefill and decode arms"
    );

    // ── the entries, through a sink that records instead of encoding ─────
    let sink = Recorder::default();
    kernels_metal::attn::ssm::causal_conv1d(
        &sink,
        Tensor::new(1, 4, CHANNELS, Dtype::Bf16),
        Tensor::new(2, CHANNELS, TAPS, Dtype::Bf16),
        &recurrent(3, HISTORY * CHANNELS, Dtype::F32),
        TAPS,
        DILATION,
        Tensor::new(5, 4, CHANNELS, Dtype::Bf16),
    )
    .expect("the convolution serves dilation three — it answered `Unsupported` before this lane");
    let (fire, args) = sink.only();
    assert_eq!(fire.entrypoint, "causal_conv1d_bfloat16");
    assert!(
        args.contains(&kernels_metal::ArgValue::I32(DILATION as i32)),
        "the convolution took a dilation of {DILATION} and did not state it to the shader — \
         which is the one way a port of this can look right and be wrong: {args:?}"
    );

    let sink = Recorder::default();
    kernels_metal::attn::ple::ngram_ids(
        &sink,
        Tensor::new(1, 4, 1, Dtype::I32),
        &recurrent(3, NGRAM - 1, Dtype::I32),
        Tensor::new(4, 1, NGRAM + 2 * HEADS, Dtype::U64),
        EOS,
        &MULTS,
        &PRIMES,
        &OFFSETS,
        HEADS / (NGRAM - 1),
        Tensor::new(5, 4, HEADS, Dtype::I32),
    )
    .expect("the hasher serves — it answered `Unsupported` by name before this lane");
    assert_eq!(sink.only().0.entrypoint, "ple_ngram_ids_update");

    eprintln!(
        "wall one is down: {SKU} names {hasher} n-gram-hasher node(s) and {} dilated \
         conv node(s), and this plane serves all of them",
        dilated.len(),
    );
}

// The one hashing this file needs: `Qwen3.8-Flash-Next`'s own constants cut to
// four heads, the same set `kernels-metal`'s `--lib` pins and
// `ple_conv_on_device.rs` measure. Nothing here reads a table, so the numbers
// only have to be a legal shape — but they are the real ones anyway, because a
// fixture that could not occur proves less.
const MULTS: [u64; 3] = [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071];

const PRIMES: [u64; 4] = [20_000_003, 20_000_023, 20_000_033, 20_000_047];

const OFFSETS: [u64; 4] = [0, 20_000_003, 40_000_026, 60_000_059];

const EOS: u32 = 248_044;

const NGRAM: u32 = 3;

const HEADS: u32 = 4;

const CHANNELS: u32 = 8;

const TAPS: u32 = 4;

/// qwen4's PLE mixes at `ngram_size`, and this is the number that used to be
/// a refusal.
const DILATION: u32 = 3;

/// `(TAPS − 1) · DILATION + 1` — the rows of history a dilated conv keeps.
const HISTORY: u32 = (TAPS - 1) * DILATION + 1;

/// A recurrent pool over made-up handles: nothing resolves them, because
/// [`Recorder`] never encodes.
fn recurrent(handle: u32, width: u32, dtype: Dtype) -> RecurrentPool {
    let bank = Tensor::new(handle, 2, width, dtype);
    RecurrentPool {
        state: bank,
        slots: Tensor::new(handle + 100, 1, 4, Dtype::U32),
        conv_state: bank,
        new_conv_state: bank,
    }
}

/// A sink that writes the launch down instead of encoding it —
/// `kernels_metal::probe::Probe`'s shape, rebuilt here because that one is
/// crate-private and this file is one crate over.
#[derive(Default)]
struct Recorder(std::cell::RefCell<Vec<(kernels_metal::Fire, Vec<kernels_metal::ArgValue>)>>);

impl Recorder {
    fn only(&self) -> (kernels_metal::Fire, Vec<kernels_metal::ArgValue>) {
        let fires = self.0.borrow();
        assert_eq!(fires.len(), 1, "expected exactly one launch");
        fires[0].clone()
    }
}

impl kernels_metal::Encode for Recorder {
    fn fire(
        &self,
        fire: kernels_metal::Fire,
        args: &[kernels_metal::ArgValue],
    ) -> Result<(), kernels_metal::Error> {
        self.0.borrow_mut().push((fire, args.to_vec()));
        Ok(())
    }

    fn absent(&self) -> Result<kernels_metal::ArgValue, kernels_metal::Error> {
        Ok(kernels_metal::ArgValue::Buffer(u32::MAX))
    }
}

/// The `(bits, group)` one bank of a plan spends, read off the PLAN'S OWN
/// planes: the code width from the bank's dtype, and the group from the ratio
/// between the bank's contracted axis and the same axis of the `.scales`
/// companion it interned beside itself.
///
/// **DERIVED AND NOT TABULATED.** A `match` from `Dtype` to a pair, written
/// here, would be the claim written twice.
fn plan_point(params: &[model_ir::Param], bank: &str) -> (u64, u64) {
    let plane = |name: &str| -> &model_ir::Param {
        params
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("the plan interns `{name}`"))
    };
    let codes = plane(bank);
    let scales = plane(&format!("{bank}.scales"));
    let k = |p: &model_ir::Param| -> u64 {
        *p.shape
            .last()
            .unwrap_or_else(|| panic!("`{}` is a bank and has a contracted axis", p.name))
    };
    let (wide, narrow) = (k(codes), k(scales));
    assert!(
        narrow > 0 && wide % narrow == 0,
        "`{bank}` contracts over {wide} and its scales over {narrow}, which is no \
         whole number of groups"
    );
    (codes.dtype.bits(), wide / narrow)
}

/// **THE WIRED FOOTPRINT, AS THE KERNEL REPORTS IT**, in bytes.
///
/// Read off `vm_stat` rather than off the shell: what the shell can tell us is
/// how many bytes it asked for, and the question a 4.5 GiB artifact on a
/// 32 GiB box raises is how many the KERNEL then wired.
fn wired() -> Option<u64> {
    let out = std::process::Command::new("vm_stat").output().ok()?;
    let text = String::from_utf8(out.stdout).ok()?;
    let mut page = 4096u64;
    let mut pages = None;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("Mach Virtual Memory Statistics: (page size of ") {
            if let Some(n) = rest.split_whitespace().next() {
                page = n.parse().unwrap_or(page);
            }
        }
        if let Some(rest) = line.strip_prefix("Pages wired down:") {
            pages = rest.trim().trim_end_matches('.').parse::<u64>().ok();
        }
    }
    Some(pages? * page)
}

/// Greedy: the highest logit.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

/// The lane word this model's own `Classify` computes.
fn word(query_len: u32) -> u64 {
    models::qwen_4::forward::Facts::of(&Request::new(query_len, false)).word()
}

fn finite(logits: &[f32], what: &str) {
    assert!(!logits.is_empty(), "{what} produced no logits at all");
    let bad = logits.iter().position(|value| !value.is_finite());
    assert!(
        bad.is_none(),
        "{what} logit {} is {}, and a single NaN means the whole row is noise",
        bad.unwrap_or(0),
        logits[bad.unwrap_or(0)],
    );
    let spread = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - logits.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        spread > 1e-3,
        "{what} logits span {spread}, which is a rectangle nothing wrote"
    );
}

/// One prefill and `STEPS` decodes in one slot, greedy throughout.
fn run(shell: &mut Shell, slot: u32, prompt: &[u32]) -> (Vec<u32>, Vec<f64>) {
    shell.open(slot).expect("the slot opens");

    let prefill = shell
        .fire(&[Lane {
            slot,
            word: word(prompt.len() as u32),
            tokens: prompt,
        }])
        .expect("the prefill fires");
    assert_eq!(prefill.len(), 1, "one lane in, one row of logits out");
    finite(&prefill[0], "prefill");

    let mut produced = vec![argmax(&prefill[0])];
    let mut millis = Vec::with_capacity(STEPS);
    for step in 0..STEPS {
        let fed = [*produced.last().expect("a step feeds the last token back")];
        let at = Instant::now();
        let decode = shell
            .fire(&[Lane {
                slot,
                word: word(1),
                tokens: &fed,
            }])
            .unwrap_or_else(|why| panic!("decode step {step} fires: {why}"));
        millis.push(at.elapsed().as_secs_f64() * 1000.0);
        finite(&decode[0], &format!("decode step {step}"));
        produced.push(argmax(&decode[0]));
    }
    (produced, millis)
}

/// Everything the fires below share: a loaded 2-bit shell and its vocabulary,
/// or `None` and a sentence saying which precondition was missing.
fn ready(what: &str) -> Option<(Shell, tokenizer::Tokenizer)> {
    if !engine_metal::device::present() {
        eprintln!("skipping {what}: this machine publishes no Metal device");
        return None;
    }
    let Some(checkpoint) = snapshot() else {
        eprintln!(
            "skipping {what}: no {REPO} snapshot with a tokenizer beside it under \
             $HOME/.cache/huggingface/hub — name one in PIE_QWEN4_U2_SNAPSHOT"
        );
        return None;
    };
    let files = shards(&checkpoint);
    assert!(!files.is_empty(), "the snapshot holds tensor shards");

    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");

    let trace = models::trace_of(SKU).expect("the catalog ships the qwen4 2-bit SKU");
    let trace = trace(Platform::Metal);
    let source = ztensor_compat::index_all(&files).expect("the shards open as one source");
    let contract =
        models::import_of(SKU).expect("the catalog ships an import for the SKU")(&source)
            .expect("the 2-bit row's import contract fits the miniature");
    drop(source);

    let before = wired();
    let booted = Instant::now();
    let shell = Shell::load(Boot {
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        // §M-4c, as `serve_smoke` states it: an unstamped snapshot proceeds,
        // and the deployment's facts are stated honestly all the same.
        tp_size: 1,
        precision: models::precision_of(SKU)
            .expect("the catalog states this row's precision")
            .to_string(),
        // Conservative: a 4.5 GiB miniature on a 32 GiB box, four lanes and a
        // short context. The claim is that it loads and fires, not that it
        // scales.
        budget: Budget::new(4, 512),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
    })
    .expect("the qwen4 2-bit shell loads — wall two was the artifact's carve, and it is re-carved");
    let wall = booted.elapsed().as_secs_f64();
    let after = wired();

    let (weights, arena, pools, inputs) = shell.footprint();
    eprintln!(
        "loaded {SKU} on {} in {wall:.1}s — weights {:.2} GiB, arena {:.1} MiB, \
         pools {:.1} MiB, inputs {:.1} MiB",
        shell.device_name(),
        weights as f64 / (1 << 30) as f64,
        arena as f64 / (1 << 20) as f64,
        pools as f64 / (1 << 20) as f64,
        inputs as f64 / (1 << 20) as f64,
    );
    match (before, after) {
        (Some(before), Some(after)) => eprintln!(
            "wired down: {:.2} GiB -> {:.2} GiB, delta {:+.2} GiB",
            before as f64 / (1 << 30) as f64,
            after as f64 / (1 << 30) as f64,
            (after as f64 - before as f64) / (1 << 30) as f64,
        ),
        _ => eprintln!("wired down: not readable on this box"),
    }
    Some((shell, tokenizer))
}

/// **THE CLAIM.** The qwen4 2-bit miniature — hashed tri-gram memory, dilated
/// mix, GatedDeltaNet, sixteen routed experts and a four-stream gated residual
/// — prefilled and decoded on an Apple GPU, returning numbers.
#[test]
fn the_qwen4_two_bit_miniature_prefills_and_decodes() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the qwen4 2-bit first light") else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    assert!(
        !prompt.is_empty(),
        "the prompt encodes to no tokens, and a prefill of nothing proves nothing"
    );
    let (produced, millis) = run(&mut shell, 0, &prompt);
    assert_eq!(
        produced.len(),
        STEPS + 1,
        "one token off the prefill and one off each decode"
    );
    let warm = &millis[millis.len() / 2..];
    eprintln!(
        "{PROMPT:?} ({} tokens) -> {:?}\n  = {:?}\n  {:.2} ms/fire warm over {STEPS} decodes, \
         {} shader points compiled",
        prompt.len(),
        produced,
        tokenizer.decode(&produced, false),
        warm.iter().sum::<f64>() / warm.len() as f64,
        shell.compiled(),
    );
    // **NOT A PARITY BAR, AND NOT A COHERENCE ONE EITHER.** See the module
    // note: this snapshot is four layers of forty-eight and sixteen experts of
    // five hundred and twelve, and its n-gram table is a sixteenth of the
    // shipped one re-hashed into its own primes. It is not a language model
    // and it does not answer like one.
    //
    // What is asserted is what a wrong load fails: every fire returned finite,
    // spread logits (`run` checks each step), the steps happened, and the two
    // gates below hold.
}

/// **DETERMINISM.** Two identical runs in two slots produce identical tokens.
///
/// A dequantization that reads uninitialized bytes, a routed scratch aliased
/// across two fires, or a PLE id history that leaks between slots are all
/// invisible in one run and all visible here — and the last of those is this
/// family's own: the hasher and its dilated convolution both carry per-slot
/// state across fires, which nothing else in this tree does.
#[test]
fn the_same_prompt_twice_is_the_same_tokens() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the qwen4 2-bit determinism gate") else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    let (first, _) = run(&mut shell, 0, &prompt);
    let (again, _) = run(&mut shell, 1, &prompt);
    assert_eq!(
        first, again,
        "the same prompt decoded twice produced different tokens, and nothing in \
         this path is sampled"
    );
}

/// **THE AFFINE CENSUS, HELD AGAINST THE BYTES — TABLE INCLUDED.**
///
/// Every affine bank the LOADED PLAN carries, at the `(bits, group)` its stored
/// triplet was written at. Read off `trace.params` rather than off the model
/// text, because the plan is what the shell seats: a text that says the right
/// thing and a trace that interns something else is exactly the failure this
/// asks about. And held against the artifact's own header rather than a table
/// typed here, because a table typed here would be the same claim twice.
///
/// **AND THE N-GRAM TABLE IS IN IT NOW.** That row is the one wall two was.
/// The plan declares `ple.table` at `padded_vocab` rows and the file stores it
/// as eight shards; before the re-carve those two numbers were 20 001 536 and
/// 20 000 096 and the load stopped there. They are the same number now, and
/// this is the assert that would go red if the artifact were ever rebuilt by a
/// slicer again.
///
/// This one needs no device — it is the plan and the file — so it runs wherever
/// the snapshot is.
#[test]
fn the_plans_affine_banks_are_the_points_the_file_wrote() {
    let Some(checkpoint) = snapshot() else {
        eprintln!("not asked: no {REPO} snapshot under $HOME/.cache/huggingface/hub");
        return;
    };
    let Some(head) = header(&checkpoint) else {
        eprintln!("not asked: {checkpoint:?} holds no readable safetensors header");
        return;
    };

    let trace = models::trace_of(SKU).expect("the catalog ships the qwen4 2-bit SKU");
    let trace = trace(Platform::Metal);
    let last = |name: &str| -> u64 {
        *head
            .get(name)
            .unwrap_or_else(|| panic!("`{name}` is in the miniature's header"))
            .last()
            .unwrap_or_else(|| panic!("`{name}` has no last axis"))
    };

    // A stored triplet's own point, checked against the plan's: `words` u32
    // code words carry `words · 32/bits` codes, and `groups` scale rows cover
    // `groups · group` of them. Same rectangle, two ways of saying it.
    let mut faults = Vec::new();
    let mut agree = |what: &str, (bits, group): (u64, u64), words: u64, groups: u64| {
        if words * (32 / bits) != groups * group {
            faults.push(format!(
                "{what}: the plan says ({bits}, {group}) and the file stores {words} \
                 code words against {groups} scale groups"
            ));
        }
    };

    // ── the sixteen routed experts, on each of the four layers ──────────
    let mut routed = 0usize;
    for l in 0..4u32 {
        let stem = format!("language_model.model.layers.{l}.mlp.switch_mlp");
        // The plan FUSES gate and up into one bank; the file writes the two
        // halves separately at the same point, which is what makes the fusion
        // legal (`model/tests/..._miniature.rs` holds the halves together).
        let fused = plan_point(&trace.params, &format!("layer.{l}.experts_gate_up"));
        for half in ["gate_proj", "up_proj"] {
            agree(
                &format!("layer {l} `{half}`"),
                fused,
                last(&format!("{stem}.{half}.weight")),
                last(&format!("{stem}.{half}.scales")),
            );
            routed += 1;
        }
        let down = plan_point(&trace.params, &format!("layer.{l}.experts_down"));
        agree(
            &format!("layer {l} `down_proj`"),
            down,
            last(&format!("{stem}.down_proj.weight")),
            last(&format!("{stem}.down_proj.scales")),
        );
        routed += 1;
        assert_eq!(
            (fused, down),
            ((2, 128), (2, 128)),
            "the miniature's routed bank is two-bit at group 128, UNIFORM across \
             the three projections and every layer — which is the thing that lets \
             this family fuse the gate and up halves at all"
        );
    }

    // ── the hashed n-gram table: the rectangle wall two stood on ────────
    let stem = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding";
    let table = plan_point(&trace.params, "ple.table");
    assert_eq!(
        table,
        (4, 32),
        "the n-gram table is the one plane this artifact writes at four bits and \
         group 32"
    );
    agree(
        "`ple.table`",
        table,
        last(&format!("{stem}.shard_0.weight")),
        last(&format!("{stem}.shard_0.scales")),
    );

    let rows = |name: &str| -> u64 {
        head.get(name)
            .and_then(|shape| shape.first().copied())
            .unwrap_or_else(|| panic!("`{name}` is in the miniature's header"))
    };
    let shard_rows: Vec<u64> = (0..8).map(|i| rows(&format!("{stem}.shard_{i}.weight"))).collect();
    let stored: u64 = shard_rows.iter().sum();
    let declared = trace
        .params
        .iter()
        .find(|p| p.name == "ple.table")
        .expect("the plan interns the n-gram table")
        .shape[0];
    assert_eq!(
        shard_rows,
        vec![shard_rows[0]; 8],
        "eight n-gram shards of one size, which is how `split_ngram_parts` cuts \
         a padded row space"
    );
    assert_eq!(
        stored, declared,
        "**THIS IS WALL TWO.** The plan declares {declared} n-gram rows and the \
         eight shards hold {stored}. They were 20 001 536 and 20 000 096 until the \
         artifact was re-carved by head (`benches/shrink_checkpoint.py`'s \
         `PleCarve`); if they have parted again the miniature was rebuilt by a \
         slicer and the load will refuse at `ple.table.scales`."
    );
    assert_eq!(
        stored % 128,
        0,
        "and the row count is a multiple of `make_ngram_vocab_size_divisible_by`, \
         which every padded table is and no verbatim slice of one was"
    );

    assert!(
        faults.is_empty(),
        "the plan and the artifact disagree on {} affine plane(s):\n{}\n",
        faults.len(),
        faults.join("\n"),
    );
    eprintln!(
        "the plan's {routed} routed planes and its {stored}-row n-gram table agree \
         with the artifact, wall two included"
    );
}
