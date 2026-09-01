//! **FIRST LIGHT FOR THE 4-BIT PLANE.** The whole MLX affine-U4 path — the
//! `MlxAffineU4` checkpoint contract, the `WeightRow::Planes` seating, the
//! affine qmm/qmv dispatch ladder and `embed_gather_mb_4bit` — was built
//! against a written description of what an MLX checkpoint holds. This file
//! is the first time any of it reads one.
//!
//! It is `serve_smoke`'s three claims over a 4-bit artifact instead of a
//! bf16 one:
//!
//! 1. **finite** — no NaN, no infinity, and not a rectangle nothing wrote.
//! 2. **deterministic** — two identical runs produce identical tokens.
//! 3. **coherent** — the continuation of [`PROMPT`] begins with " Paris".
//!
//! The third is the one that separates a 4-bit load from a 4-bit-shaped
//! one. Every stage below it can bind, dispatch and return finite numbers
//! off codes read at the wrong nibble, scales attached to the wrong rows or
//! biases dropped entirely; only the answer says the dequantization is the
//! checkpoint's own.
//!
//! # The checkpoint
//!
//! ```text
//! mlx_lm.convert --hf-path Qwen/Qwen3.5-0.8B -q --q-bits 4 --q-group-size 64 \
//!     --mlx-path ~/.cache/huggingface/hub/\
//! models--mlx-community--Qwen3.5-0.8B-4bit/snapshots/local
//! ```
//!
//! One command, no post-processing: that conversion IS what `mlx-community`
//! publishes for this family, and reading it unedited is the point — the
//! bugs below were both in the gap between what `mlx_lm` writes and what
//! this tree was told it writes. `PIE_U4_SNAPSHOT` overrides where the
//! snapshot is looked for; absent that, any `mlx-community` Qwen3.5 4-bit
//! directory in the hugging face cache.
//!
//! # What first light found
//!
//! Neither bug was in a kernel or in the loader. Every one of the 514
//! planes this contract lands arrives byte-identical to the checkpoint, and
//! every affine point — qmv and qmm, at every shape this stack asks for,
//! the 248320-wide tied head included — computes what a host reference
//! computes. Both bugs were in the gap between what `mlx_lm` WRITES and
//! what this tree had been told it writes:
//!
//! 1. **The trunk prefix.** `mlx_lm` re-roots the tower as
//!    `language_model.model.*`; the contract named transformers'
//!    `model.language_model.*`, so `import` refused at the embedding and no
//!    MLX file had ever satisfied this SKU. `models::qwen_3::import`'s
//!    `Layout`.
//! 2. **The folded rmsnorm one.** `mlx_lm`'s `sanitize` adds `1.0` to every
//!    plain-RMSNorm weight so a kernel without the constant reproduces
//!    `Qwen3_5RMSNorm`'s `x * (1 + w)` — and writes the shifted values into
//!    the checkpoint. This engine HAS that constant, so it added a second
//!    one and computed `x * (2 + w)` at every norm in the stack: finite,
//!    deterministic, and nonsense. `Layout::folds_the_norm_one` takes it
//!    back out. Saying so needed `Expr::Bias`, which no device tile-map mask
//!    admitted although the host executor has always run it —
//!    `METAL_TILE_MAP_MASK` carries the bit now.
//!
//! # The parity bar
//!
//! `mlx_lm.generate` over the same snapshot and the same prompt, greedy, is
//! the reference this path is measured against, and the two now agree TOKEN
//! FOR TOKEN over nine greedy steps. It is PRINTED and not asserted,
//! because a test that shells out to a python interpreter fails for reasons
//! that are not about this engine; what is asserted in-tree is the ` Paris`
//! the reference also begins with, and determinism.
//!
//! # And what the batched ladder found
//!
//! First light's prompt is eight tokens, two clear of `qmm_min_batch`, so it
//! reaches `linear::quant`'s batched ladder at its narrowest rung and nowhere
//! else — one row tile of one 16-row block, per projection. The three
//! tests after the first two carry prompts chosen for their ROW COUNTS rather
//! than their words, and between them they fire every rung and every arm that
//! ladder has: the pre-cast pair at all three row blocks, the plain point and
//! the plain stamped point (in a process of their own, since the tuning table
//! freezes once — see [`RESPAWN_KEY`]), and a pad of twenty-six and then
//! forty-eight rows past the end of a prompt.
//!
//! `affine_floor` is the same ladder's floor with no checkpoint in it: every
//! point fired by name against a host reference, so a rung this file's prompts
//! stop reaching does not go quietly unmeasured.
//!
//! **THEIR PARITY IS ASSERTED AND NOT PRINTED**, which the prompts are what
//! makes legal: a counted list decides each of its greedy steps by more than
//! two logits, against a residual of 0.103. See [`PADDED_TOKENS`].
//!
//! Every one of those arms computed `mlx_lm`'s own tokens the first time it
//! was fired at a checkpoint. What the exercise moved was not a kernel but
//! `crate::scratch`'s aliasing argument, which rested on a shader fact that
//! is no longer true — the composed `fp16_precast_splitk` points ARE stamped,
//! and what keeps the two roles off each other's bytes is the ladder's own
//! `return`.
//!
//! # THE PLANE-ORDER HOLE, AND HOW IT CLOSED
//!
//! This suite was RED for one wave and nothing in it was adjusted to make it
//! green again, which is the whole reason the account is kept. The five arms
//! failed because `qwen35-d0.8b-mlxu4-kv-bf16`'s dense projections were
//! declared in a plane order this shell has no reader for:
//!
//! 1. **The flip.** `models::qwen_3::model`'s `Model::new` turned every dense
//!    projection of a `*-mlxu4-*` qwen_3 row from `U4g64` into `U4g64tiled` —
//!    MLX affine codes in m16n8k16 fragment order, which
//!    `kernels_cuda::linear::tiled` reads 2.5-6x faster. The predicate was the
//!    declared weight width and nothing else, because `Model` is constructed
//!    as an argument to `model_dsl::trace_hybrid` and so had no `Platform` to
//!    ask — and the SKU it flipped is this one.
//! 2. **The refusal on a raw snapshot** (§M-3 / §J4b). A repack is paid once
//!    per weight, so a SERVING plan may not carry one:
//!    `checkpoint::plan::passes::validate`'s `validate_target_support`
//!    refuses it by name and points at `pie model import`.
//!    `METAL_TILE_MAP_MASK` is `CAST | SCALE | DECODE | BIAS` — no
//!    `TILE_MAP_REPACK` — so loading the `mlx_lm.convert` output said
//!    *"this load would relayout a weight plane on the way in
//!    (Some(TiledAffineU4Weight))"*.
//! 3. **And the import answered it for CUDA, not here.** `pie model import`
//!    compiled the same contract under `CONVERT_TILE_MAP_MASK`, ran the
//!    repack host-side and wrote the relaid plane under the WEIGHT's name,
//!    which `qwen_3::import`'s `read_own` arm binds with no transform — so
//!    the artifact LOADED on this shell and did not answer. Measured, once:
//!
//!    ```text
//!    ✓ imported local — 411MiB in 713ms  (378 planes repacked)
//!    loaded qwen35-d0.8b-mlxu4-kv-bf16 on Apple M1 Max in 0.1s — weights 0.41 GiB
//!    "The capital of France is the city of" -> "一时的وات**!.energyамет安全问题standenaland!."
//!    ```
//!
//!    Finite, deterministic, and nonsense — `kernels_metal::linear::quant`'s
//!    qmm and qmv arms index an affine bank ROW-MAJOR, and neither has a
//!    fragment-order twin.
//!
//! **WHAT CLOSED IT** (§J4c): the flip at (1) learned the platform, exactly
//! as the paragraph that stood here asked it to, and NOT by growing a
//! load-time relayout — serving does not convert. `model_dsl::place` resolves
//! a placed dtype against the setup the declaration is read for
//! (`model_ir::Platform::placement`), and both readings of a family's text go
//! through one: the trace through `catalog!`, the load contract through
//! `runtime::engine::load` — and through `ready` below, which is a test that
//! builds its own. So this shell's trace of this row declares the canonical
//! `U4g64` its qmm and qmv arms already read.
//!
//! **AND THE DOOR THAT REOPENED IS THE RAW ONE.** With no repack in the plan,
//! (2) has nothing to refuse: the `mlx_lm.convert` snapshot serves AS STORED,
//! with no `pie model import` step at all, which is how this suite ran before
//! the flip and how it runs now. The parity bars below are the ones that were
//! written against `mlx_lm` before any of this and were never touched.
//!
//! # Gating
//!
//! Apple-only at compile time. The plane order is asserted FIRST and
//! unconditionally — it is a fact about the trace, so a box with no device
//! and no snapshot still checks it rather than skipping into a misleading
//! green. Past it, the suite SKIPS at run time naming which precondition was
//! missing — the device, the 4-bit snapshot, or the tokenizer beside it.
//!
//! ```text
//! cargo test -p engine-metal --release --test four_bit_first_light -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use engine_metal::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

/// The catalog row this smoke serves. `mlxu4` weights, bf16 kv — the 4-bit
/// half of the same 0.8B `serve_smoke` proves in bf16, so the two differ in
/// exactly one thing.
const SKU: &str = "qwen35-d0.8b-mlxu4-kv-bf16";

/// The prompt, and the reason it is this one rather than `serve_smoke`'s
/// bare "The capital of France is".
///
/// **A GATE HAS TO CLEAR THE NOISE FLOOR IT IS MEASURED AGAINST.** Two
/// independent bf16 implementations of one 24-layer stack do not produce
/// identical logits, and they do not have to: this shell and `mlx_lm` agree
/// on this checkpoint to a correlation of 0.9992, with a residual of 0.103
/// rms against logits whose own rms is 4.13 — bf16 accumulation order over
/// twenty-four layers, and nothing else.
///
/// On the bare prompt, this 4-bit checkpoint's top two logits are ` in` and
/// ` Paris` **EXACTLY TIED**, at 14.375 apiece in `mlx_lm`'s own arithmetic.
/// The two implementations split that tie in opposite directions, and an
/// assertion on the argmax would be reading which way the rounding fell:
/// green would not mean right and red would not mean wrong. Uniform 4-bit
/// at group 64 is simply a lot to ask of a 0.8B, and this is what that
/// looks like from underneath.
///
/// Four words more and the same fact is decided by 2.06 logits — twenty
/// times the residual — and every later step of the continuation below by
/// 0.25 or more. That is a claim about the model, not about the summation
/// order.
const PROMPT: &str = "The capital of France is the city of";

/// What a correct 4-bit load produces here — the same word the bf16 load
/// produces, which is the claim: quantization to four bits does not change
/// the capital of France.
const EXPECTED: &str = " Paris";

/// **THE PARITY BAR, RECORDED ONCE.** Nine greedy tokens from `mlx_lm` over
/// this snapshot and this prompt, no chat template, `temp 0`:
///
/// ```text
/// [11751, 13, 198, 760, 6511, 314, 9338, 369, 279]
/// ```
///
/// Printed against what the shell says rather than asserted, because a test
/// that shells out to a python interpreter fails for reasons that are not
/// about this engine — and the interpreter is not this repository's to keep
/// installed. What IS asserted is the ` Paris` this reference also begins
/// with, and determinism.
const MLX_REFERENCE: &str = " Paris.\nThe capital of France is the";

/// How many decode fires follow the prefill.
const STEPS: usize = 8;

/// **THE PADDED PREFILL, AND WHY THE ROW COUNT IS THE POINT.**
///
/// [`PROMPT`] encodes to eight tokens, over `qmm_min_batch` — so the
/// first-light prompt does reach the batched ladder, and since the 8 rung
/// landed it sits exactly on the narrowest step: `bm_rung` answers 8,
/// `mb_block` pads nothing, and every projection launches ONE row tile whose
/// every row the fire wrote. Everything the ladder does above that — a
/// second row tile, the wider rungs, the column tile the second tile
/// selects — is what the wider prompts below exist for.
///
/// Twenty tokens is where the pad and the tiling do real work: `bm_rung`
/// answers 16, `mb_block` pads 20 up to 32, and the launch is TWO row tiles
/// of which twelve rows hold nothing.
const PADDED_PROMPT: &str = "1, 2, 3, 4, 5, 6, 7,";

/// The rows [`PADDED_PROMPT`] encodes to.
///
/// **ASSERTED, BECAUSE THE ROW COUNT IS THE WHOLE SELECTION.** Which rung a
/// prompt lands on, how deep the pad is and how many tiles launch are all
/// functions of this number and of nothing else in the test; a tokenizer that
/// answered 15 or 33 here would move the fire to another rung and the file
/// would go on claiming it had covered this one.
const PADDED_ROWS: usize = 20;

/// **THE RAGGED PREFILL.** Thirty-eight tokens: `bm_rung` answers 32, so
/// `mb_block` pads to 64 and the launch is two 32-row tiles with TWENTY-SIX
/// rows past the end of the prompt.
///
/// That tail is the claim `mb_block` makes and this prompt is where it is
/// tested — a GEMM row's output depends only on its own input row, so the
/// product of twenty-six rows of whatever the activation slot happened to
/// hold cannot reach a row the fire reads. It lands in the result slot all
/// the same, at rows the next op does not walk. If that were wrong the tail
/// would reach the real rows through the projection that follows, and the
/// tokens below would not be `mlx_lm`'s.
const RAGGED_PROMPT: &str = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,";

/// The rows [`RAGGED_PROMPT`] encodes to — see [`PADDED_ROWS`].
const RAGGED_ROWS: usize = 38;

/// **THE PARITY BAR FOR THE BATCHED PROMPTS, RECORDED ONCE AND ASSERTED.**
///
/// Unlike [`MLX_REFERENCE`] these are asserted rather than printed, and the
/// prompts are what makes that legal. The residual between this shell and
/// `mlx_lm` on this checkpoint is 0.103 rms against logits whose own rms is
/// 4.13, so a greedy step is only decidable when its top two logits are
/// further apart than that. A counted list is: the narrowest margin over the
/// nine steps below is 2.75 logits on [`PADDED_PROMPT`], 3.0 on
/// [`RAGGED_PROMPT`] and 2.375 on [`WIDE_PROMPT`] — twenty-five times the
/// noise floor, measured in `mlx_lm`'s own arithmetic at every step. Nothing
/// here is deciding a tie.
///
/// Derived ONCE, from the venv beside this repository, and pinned:
///
/// ```text
/// mlx_lm.generate --model ~/.cache/huggingface/hub/\
/// models--mlx-community--Qwen3.5-0.8B-4bit/snapshots/local \
///     --prompt "1, 2, 3, 4, 5, 6, 7," --temp 0 --max-tokens 9 \
///     --ignore-chat-template
/// ```
///
/// which says `8, 9, 10` — the nine ids below. The same command over
/// [`RAGGED_PROMPT`] says `13, 14, ` and over [`WIDE_PROMPT`] `31 32 33`.
/// **THE TEST DOES NOT RUN THAT COMMAND**, for [`MLX_REFERENCE`]'s reason: a
/// test that shells out to a python interpreter fails for reasons that are
/// not about this engine.
const PADDED_TOKENS: &[u32] = &[220, 23, 11, 220, 24, 11, 220, 16, 15];

/// [`PADDED_TOKENS`] for [`RAGGED_PROMPT`] — `13, 14, `.
const RAGGED_TOKENS: &[u32] = &[220, 16, 18, 11, 220, 16, 19, 11, 220];

/// **THE WIDEST RUNG.** Eighty tokens: `bm_rung` answers 64, `mb_block` pads
/// to 128, and the launch is two 64-row tiles with FORTY-EIGHT rows past the
/// end of the prompt.
///
/// The third rung is a separate case from [`RAGGED_PROMPT`]'s second and not
/// a longer version of it: 64 is where a row block stops fitting four
/// threadgroups to a core, it is the last rung `BM_RUNGS` holds, and it is
/// the one whose pad can be wider than the prompt that asked for it.
const WIDE_PROMPT: &str =
    "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30";

/// The rows [`WIDE_PROMPT`] encodes to — see [`PADDED_ROWS`].
const WIDE_ROWS: usize = 80;

/// [`PADDED_TOKENS`] for [`WIDE_PROMPT`] — `31 32 33`, at a narrowest margin
/// of 2.375 logits.
const WIDE_TOKENS: &[u32] = &[220, 18, 16, 220, 18, 17, 220, 18, 18];

/// **THE ONE KNOB THAT CANNOT BE TURNED TWICE IN ONE PROCESS**, and the
/// re-exec that answers it.
///
/// `kernels_metal::tuning` folds the device row and the boot document's
/// `[metal.tuning]` table at the FIRST `current()` and freezes the answer
/// from then on — deliberately, because a table that could move under a fire
/// in flight would let two dispatches of one step disagree about which kernel
/// they are. So a file cannot hold one test at `fp16_qmm = true` and another
/// at `false`: whichever ran first would decide both, and `cargo test` does
/// not say which that is.
///
/// The arm that needs the other answer therefore runs in its own process —
/// this same test binary, re-entered with `--exact` at the one test, which
/// reads this key, opens a boot document that turns the FP16 path off, and
/// only then loads a shell. The parent is a launcher and asserts the exit
/// status. **The key is read by the TEST HARNESS and never by the shell**:
/// what reaches `kernels_metal::tuning` is the boot document, which is the
/// only channel there is (art. 9).
const RESPAWN_KEY: &str = "PIE_U4_FP16_OFF";

/// The boot document the re-entered process opens. One key, and the reason
/// it is worth a process: with the FP16 matrix path off, `fp16_gemm_format`
/// answers `false` for this bank, the pre-cast rung declines, and the two
/// arm BELOW it — the plain stamped point — becomes reachable. It used to
/// reach the split-K pair as well; see
/// [`the_plain_stamped_point_matches_the_same_tokens_with_fp16_off`] for
/// where that arm went.
const FP16_OFF: &[u8] = b"[metal.tuning]\nfp16_qmm = false\n";

/// A contract lookup the re-entered process never reaches: [`FP16_OFF`] is
/// opened for its `[metal.tuning]` table alone, and the shell below is loaded
/// through [`Shell::load`] as every other test here loads one.
fn no_door(
    _trace: &model_ir::Trace,
    _path: &Path,
) -> Result<checkpoint::contract::ModelContract, String> {
    Err("this door never loads".to_string())
}

/// One shell at a time per process: these hold the whole weight table
/// resident and the measurements are only readable one at a time.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The 4-bit snapshot: the checkpoint AND the tokenizer that goes with it.
fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_U4_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let usable = |path: &Path| path.join("tokenizer.json").exists() && container(path).is_some();

    // Otherwise any `mlx-community` 4-bit snapshot of this family in the
    // cache — which is also where a local `mlx_lm.convert` output belongs,
    // under `models--mlx-community--Qwen3.5-0.8B-4bit/snapshots/local`,
    // because a conversion of `Qwen/Qwen3.5-0.8B` at these settings IS what
    // that repository publishes. The suite runs as root over tailscale ssh,
    // so `HOME` is not the owner's — the cache is named explicitly beside
    // it.
    let homes = [
        std::env::var("HOME").unwrap_or_default(),
        "/Users/ingim".to_string(),
    ];
    homes.iter().find_map(|home| {
        let hub = Path::new(home).join(".cache/huggingface/hub");
        let mut repos: Vec<PathBuf> = std::fs::read_dir(&hub)
            .ok()?
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                let name = path.file_name()?.to_str()?.to_string();
                (name.starts_with("models--mlx-community--Qwen3.5") && name.ends_with("-4bit"))
                    .then_some(path)
            })
            .collect();
        repos.sort();
        repos.into_iter().find_map(|repo| {
            std::fs::read_dir(repo.join("snapshots"))
                .ok()?
                .filter_map(|entry| Some(entry.ok()?.path()))
                .find(|path| usable(path))
        })
    })
}

/// The container the contract is checked against — one file of the
/// snapshot, whichever one holds the tensors.
fn container(snapshot: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(snapshot)
        .ok()?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            (name.ends_with(".safetensors") || name.ends_with(".zt")).then_some(path)
        })
        .collect();
    found.sort();
    found.into_iter().next()
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

/// The lane word the model's own `Classify` computes.
fn word(query_len: u32) -> u64 {
    models::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
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
        finite(&decode[0], "decode");
        produced.push(argmax(&decode[0]));
    }
    (produced, millis)
}

/// **THE PLANE ORDER, ASSERTED BEFORE ANY PRECONDITION IS LOOKED FOR.**
///
/// The module header argues it; this is the one line of it that runs. A plane
/// order is a fact about the TRACE — no device, no snapshot and no checkpoint
/// enters into it — so this states itself on any Apple box, and a machine
/// missing the 4-bit snapshot checks it instead of skipping into a green that
/// means nothing.
///
/// **AND IT IS A PANIC RATHER THAN A SKIP.** A skip would say "this box could
/// not check", which is false: the box checked. It said so for a wave, while
/// the projection flip took the tiled order for every platform, and it went
/// quiet ON ITS OWN when §J4c gave the flip a platform to ask — which is the
/// whole reason it was spelled as a property of the trace rather than as an
/// `#[ignore]`. It stays for what it would catch next: this shell still has
/// no fragment-order reader, so a text that reaches past `model_dsl::place`
/// for one lands here rather than on the parity bars below.
fn serves_the_order_it_declares() {
    let trace = models::trace_of(SKU).expect("the catalog ships the 4-bit SKU")(Platform::Metal);
    let tiled: Vec<&str> = trace
        .params
        .iter()
        .filter(|param| param.dtype == model_dsl::Dtype::U4g64tiled)
        .map(|param| param.name.as_str())
        .collect();
    assert!(
        tiled.is_empty(),
        "{SKU} declares {} plane(s) as U4g64tiled — MLX affine codes in m16n8k16 fragment \
         order, which this shell has no reader for (first: {:?}). `kernels_metal::linear::quant` \
         indexes an affine bank row-major, so neither road serves: a raw MLX snapshot is \
         refused by the load plan (a serving plan does not repack) and a repacked artifact \
         loads and answers nonsense (see the module header for the measurement). A text \
         reaches that order only by asking for it — `model_dsl::place` answers this platform \
         with the canonical `U4g64` — so a declaration that states it outright, or a placed \
         variant with no `Platform::placement` row, is what put it here",
        tiled.len(),
        tiled.first().unwrap_or(&""),
    );
}

/// Everything the tests below share: a loaded 4-bit shell and its
/// vocabulary, or `None` and a sentence saying which precondition was
/// missing.
fn ready(what: &str) -> Option<(Shell, tokenizer::Tokenizer)> {
    serves_the_order_it_declares();
    if !engine_metal::device::present() {
        eprintln!("skipping {what}: this machine publishes no Metal device");
        return None;
    }
    let Some(checkpoint) = snapshot() else {
        eprintln!(
            "skipping {what}: no MLX 4-bit Qwen3.5 snapshot found — convert one with \
             `mlx_lm.convert --hf-path Qwen/Qwen3.5-0.8B -q --q-bits 4 --q-group-size 64` \
             and name it in PIE_U4_SNAPSHOT"
        );
        return None;
    };
    let Some(container) = container(&checkpoint) else {
        eprintln!("skipping {what}: {checkpoint:?} holds no tensor container");
        return None;
    };
    if !checkpoint.join("tokenizer.json").exists() {
        eprintln!("skipping {what}: {checkpoint:?} ships no tokenizer beside its tensors");
        return None;
    }

    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");

    let trace = models::trace_of(SKU).expect("the catalog ships the 4-bit SKU");
    let trace = trace(Platform::Metal);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    // **READ FOR THIS SHELL** (§J4c). A load contract is `fn(&Source)` and a
    // family's text may state a `Dtype` PLACEMENT, so the setup the contract
    // is read under is what decides whether a projection is claimed in
    // fragment order or row-major — and it has to be the setup the trace
    // above was taken for, or the two describe different planes. This is
    // `runtime::engine::load::contract_for`'s `trace.platform` wrap, in a
    // test that builds its own contract.
    let contract = models::placing_for(Platform::Metal, || {
        models::import_of(SKU).expect("the catalog ships an import for the SKU")(&source)
    })
    .expect("the 4-bit SKU's import contract fits a real MLX 4-bit checkpoint");
    drop(source);

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
        budget: Budget::new(4, 256),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
    })
    .expect("the 4-bit shell loads");
    let (weights, arena, pools, inputs) = shell.footprint();
    eprintln!(
        "loaded {SKU} on {} in {:.1}s — weights {:.2} GiB, arena {:.1} MiB, pools {:.1} MiB, \
         inputs {:.1} MiB",
        shell.device_name(),
        booted.elapsed().as_secs_f64(),
        weights as f64 / (1 << 30) as f64,
        arena as f64 / (1 << 20) as f64,
        pools as f64 / (1 << 20) as f64,
        inputs as f64 / (1 << 20) as f64,
    );
    Some((shell, tokenizer))
}

/// **THE CLAIM.** A real MLX affine-U4 checkpoint, prefilled and decoded on
/// an Apple GPU, says the true thing.
#[test]
fn a_real_four_bit_checkpoint_prefills_decodes_and_says_something_true() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the 4-bit first light") else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    let (produced, millis) = run(&mut shell, 0, &prompt);
    let text = tokenizer.decode(&produced, false);
    let warm = &millis[millis.len() / 2..];
    eprintln!(
        "{PROMPT:?} -> {text:?}  |  {:.2} ms/fire warm, {} shader points compiled",
        warm.iter().sum::<f64>() / warm.len() as f64,
        shell.compiled()
    );
    // The parity bar, printed rather than asserted — see MLX_REFERENCE.
    eprintln!("mlx_lm.generate said {MLX_REFERENCE:?}");
    eprintln!(
        "parity: {}",
        if text.starts_with(MLX_REFERENCE) || MLX_REFERENCE.starts_with(&text) {
            "token for token with mlx_lm over the shorter of the two"
        } else {
            "DIVERGES from mlx_lm — read the two strings above"
        }
    );
    assert!(
        text.starts_with(EXPECTED),
        "the 4-bit continuation is {text:?}, and a correct load begins it {EXPECTED:?}"
    );
}

/// The 4-bit half of `serve_smoke`'s determinism gate. A dequantization
/// that read an unsynchronized staging buffer, or a scales plane bound to a
/// rectangle two arms both write, shows up here and nowhere else.
#[test]
fn two_identical_four_bit_runs_produce_identical_tokens() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the 4-bit determinism gate") else {
        return;
    };
    let prompt = tokenizer.encode(PROMPT);
    let (first, _) = run(&mut shell, 0, &prompt);
    let (second, _) = run(&mut shell, 0, &prompt);
    eprintln!("first {first:?}\nsecond {second:?}");
    assert_eq!(
        first, second,
        "two identical 4-bit fires answered differently, which means two arms wrote the same rows"
    );
}

/// One pinned prompt, prefilled and decoded in one slot, against the tokens
/// `mlx_lm` produced for it.
///
/// The row count is asserted FIRST and separately: every arm below is
/// selected by it, so a prompt that encoded to some other length would run a
/// different rung and this file would go on saying it had covered the one it
/// names.
fn pinned(
    shell: &mut Shell,
    tokenizer: &tokenizer::Tokenizer,
    what: &str,
    prompt: &str,
    rows: usize,
    want: &[u32],
) {
    let encoded = tokenizer.encode(prompt);
    assert_eq!(
        encoded.len(),
        rows,
        "{what}: {prompt:?} encodes to {} tokens and the rung this arm exercises is chosen by \
         its {rows}",
        encoded.len(),
    );
    let (produced, millis) = run(shell, 0, &encoded);
    let warm = &millis[millis.len() / 2..];
    eprintln!(
        "{what}: {rows} rows -> {:?} ({:.2} ms/fire warm)\n  got  {produced:?}\n  want {want:?}",
        tokenizer.decode(&produced, false),
        warm.iter().sum::<f64>() / warm.len() as f64,
    );
    assert_eq!(
        produced, want,
        "{what}: token-for-token with mlx_lm is the gold assertion, and this arm did not hold it"
    );
}

/// Re-enter this same test binary at one test, with [`RESPAWN_KEY`] set — see
/// the key for why the arm needs a process of its own.
///
/// **THE NAME IS A LITERAL AND THE COUNT IS WHAT CHECKS IT.** `libtest` exits
/// zero for a filter that matched nothing, so an exit status alone would
/// report a renamed test as a passing one. The child's own tally is read
/// instead, and it has to say that exactly one test ran.
fn respawn(test: &str) {
    let exe = std::env::current_exe().expect("the test binary knows where it is");
    let out = std::process::Command::new(&exe)
        .args(["--exact", test, "--nocapture", "--test-threads=1"])
        .env(RESPAWN_KEY, "1")
        .output()
        .unwrap_or_else(|why| panic!("re-entering {exe:?} at {test}: {why}"));
    let said = String::from_utf8_lossy(&out.stdout).into_owned()
        + &String::from_utf8_lossy(&out.stderr);
    eprint!("{said}");
    assert!(
        said.contains("1 passed"),
        "re-entering at `{test}` ran no such test — the literal above and the function name \
         below it have to be the same string"
    );
    assert!(out.status.success(), "the re-entered arm failed; its output is above");
}

/// **RUNGS 1 AND 2 AGAINST A CHECKPOINT: the row pad, two row tiles, and the
/// FP16 pre-cast pair.** Twenty rows padded to thirty-two, `bm = 16`, the
/// `cast_qmm_input` staging dispatch and the `fp16_precast` GEMM at every
/// dense projection and at the 248320-wide head.
#[test]
fn the_precast_qmm_pair_matches_mlx_lm_token_for_token_over_a_padded_prefill() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the pre-cast arm") else {
        return;
    };
    pinned(
        &mut shell,
        &tokenizer,
        "the pre-cast pair",
        PADDED_PROMPT,
        PADDED_ROWS,
        PADDED_TOKENS,
    );
}

/// **RUNG 1 AT THE TWO WIDER RUNGS, AND THE TAILS THEY LEAVE.** Thirty-eight
/// rows padded to sixty-four at `bm = 32`, then eighty padded to a hundred
/// and twenty-eight at `bm = 64` — twenty-six and forty-eight rows of launch
/// past the end of the prompt, whose product lands in the result slot and
/// must reach no row the fire reads. See [`RAGGED_PROMPT`] and
/// [`WIDE_PROMPT`].
#[test]
fn a_ragged_prefill_pads_to_the_wider_rungs_and_its_tail_reaches_no_real_row() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the ragged rungs") else {
        return;
    };
    pinned(
        &mut shell,
        &tokenizer,
        "the second rung",
        RAGGED_PROMPT,
        RAGGED_ROWS,
        RAGGED_TOKENS,
    );
    pinned(
        &mut shell,
        &tokenizer,
        "the third rung",
        WIDE_PROMPT,
        WIDE_ROWS,
        WIDE_TOKENS,
    );
}

/// **THE PLAIN STAMPED POINT**, over the same three prompts and against the
/// same pinned tokens — which is the claim, since a plain point whose column
/// tile read the wrong scales would answer something else. `fp16_qmm` off is
/// what reaches it: with the FP16 matrix path on, every dense projection of a
/// 4-bit/group-64 checkpoint takes the pre-cast arm above and this one never
/// runs.
///
/// **IT USED TO REACH THE SPLIT-K PAIR TOO, AND THERE IS NO SPLIT ANY MORE.**
/// `linear::quant::act_x_wt` took the split whenever the pre-cast arm
/// declined, which with `fp16_qmm` off was every projection narrow enough;
/// it is deleted, because a partitioned contraction is not the order the
/// unsplit tile walks and its depth was a function of the FIRE's row count.
/// `linear::quant::split_k` carries the argument and
/// `affine_floor`'s fingerprint matrix carries the measurement. The three
/// prompts and the pinned tokens stay exactly as they were — and that they
/// still pass is worth something on its own: the split and the plain point
/// were landing the same TOKENS while landing different BITS, which is the
/// whole reason a greedy gate is not an invariance gate.
///
/// Runs in a process of its own: see [`RESPAWN_KEY`].
#[test]
fn the_plain_stamped_point_matches_the_same_tokens_with_fp16_off() {
    // **BEFORE THE RESPAWN AND NOT INSIDE IT.** The hole is the parent's to
    // report: a child that failed on it would come back as "re-entering ran
    // no such test", which names the wrong thing entirely.
    serves_the_order_it_declares();
    if std::env::var_os(RESPAWN_KEY).is_none() {
        if !engine_metal::device::present() {
            eprintln!("skipping the plain point: this machine publishes no Metal device");
            return;
        }
        return respawn("the_plain_stamped_point_matches_the_same_tokens_with_fp16_off");
    }
    // The re-entered process. The table is laid down before anything has
    // asked for it, which is the entire reason this is a process and not a
    // branch.
    engine_metal::open(FP16_OFF, no_door).expect("the boot document opens");
    assert!(
        !kernels_metal::tuning::current().fp16_qmm,
        "the boot document did not reach the tuning table, and the arm under test is the \
         one that is only reachable when it does"
    );
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the plain arm") else {
        return;
    };
    pinned(
        &mut shell,
        &tokenizer,
        "the plain point",
        PADDED_PROMPT,
        PADDED_ROWS,
        PADDED_TOKENS,
    );
    pinned(
        &mut shell,
        &tokenizer,
        "the plain point at the second rung",
        RAGGED_PROMPT,
        RAGGED_ROWS,
        RAGGED_TOKENS,
    );
    pinned(
        &mut shell,
        &tokenizer,
        "the plain point at the third rung",
        WIDE_PROMPT,
        WIDE_ROWS,
        WIDE_TOKENS,
    );
}
