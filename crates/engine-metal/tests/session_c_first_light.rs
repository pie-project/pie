//! **FIRST LIGHT FOR THE THREE NORTH-STAR MODELS.** `four_bit_first_light`
//! is this file's precedent and its idiom: the same three claims — finite,
//! deterministic, coherent — over a real `mlx-community` checkpoint read
//! unedited off the hugging face cache. What changes is the scale and the
//! number of families. The 0.8B that proved the affine-U4 plane is a
//! vehicle; these are the models the engine is FOR.
//!
//! | SKU | checkpoint | what had never been read |
//! |---|---|---|
//! | `qwen36-27b-mlxu4-kv-bf16` | `mlx-community/Qwen3.6-27B-4bit` | sixty-four layers, forty-eight of them gated-delta |
//! | `gptoss-20b-*` | `mlx-community/gpt-oss-20b-MXFP4-Q4` | the MXFP4 expert banks, and the sinks |
//! | `gemma4-31b-mlxu4-kv-bf16` | `mlx-community/gemma-4-31b-it-4bit` | two attention shapes in one stack |
//!
//! # The audit that comes before the run
//!
//! Every one of these checkpoints is written by `mlx_lm.convert`, and
//! `mlx_lm` does not write what transformers wrote. `four_bit_first_light`
//! found two such gaps on the 0.8B — a re-rooted trunk prefix and a constant
//! folded into every rmsnorm — and the rule it left behind is that the
//! family's `sanitize` is read BEFORE the first load. The three audits are
//! recorded at [`QWEN`], [`GPT_OSS`] and [`GEMMA`], each against the
//! `mlx_lm` source that produced the file.
//!
//! # Gating
//!
//! Apple-only at compile time, and every arm SKIPS at run time naming which
//! precondition was missing — the device, the snapshot, the tokenizer, or a
//! catalog row that does not exist yet. `PIE_QWEN36_SNAPSHOT`,
//! `PIE_GPTOSS_SNAPSHOT` and `PIE_GEMMA4_SNAPSHOT` override where each is
//! looked for.
//!
//! # One model at a time
//!
//! Fifteen to seventeen gibibytes of weights on a thirty-two gibibyte box:
//! [`ONE_AT_A_TIME`] serializes the arms and every arm drops its shell before
//! it returns, because two of these resident at once is a swap storm and not
//! a measurement.
//!
//! ```text
//! cargo test -p engine-metal --release --test session_c_first_light -- --nocapture --test-threads=1
//! ```

#![cfg(target_vendor = "apple")]

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use engine_metal::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

/// One prefill and this many greedy decodes.
const STEPS: usize = 8;

/// The prompt, and it is `four_bit_first_light`'s for that file's reason:
/// four words past "The capital of France is" the answer stops being a tie
/// between ` in` and ` Paris` and starts being a claim about the model.
const PROMPT: &str = "The capital of France is the city of";

/// What a correct load of any of the three says next.
const EXPECTED: &str = " Paris";

/// **A FAMILY'S FIRST LIGHT, AND THE AUDIT THAT PRECEDES IT.**
struct Family {
    /// The catalog row.
    sku: &'static str,
    /// The environment variable that overrides the snapshot directory.
    env: &'static str,
    /// The `models--*` directory in the hugging face cache, exactly.
    repo: &'static str,
    /// The lane word this family's own `Classify` computes.
    word: fn(u32) -> u64,
    /// **THE TOKEN THIS FAMILY WILL NOT DECODE WITHOUT**, or `None` for a
    /// family whose raw prompt is a whole input.
    ///
    /// A `tokenizer.json` encodes text; whether a sequence begins with a
    /// beginning is the MODEL's convention, and the two families here split on
    /// it. Qwen's reference — `mlx_lm.generate --ignore-chat-template` — feeds
    /// the bare encoding and this shell matches it token for token, so `None`
    /// is the whole truth there.
    ///
    /// **GEMMA IS THE OTHER CASE, AND IT IS NOT A NEAR MISS.** Measured on
    /// `mlx-community/gemma-4-31b-it-4bit`, in `mlx_lm`'s own arithmetic and
    /// with no engine involved: fed the eight bare tokens of [`PROMPT`],
    /// greedy decoding produces `759` nine times — `" la la la la la la la la
    /// la"`. Prepend `2` and the same nine steps produce `" Paris. Paris is
    /// known as the \"City"`. One token in front of the prompt is the
    /// difference between a degenerate loop and the answer, because every
    /// sequence this model was trained on began with one and a first position
    /// that holds a content token is a position the stack has never seen.
    ///
    /// This mattered to the audit and not only to the assertion: the first
    /// gemma fire this engine ever made said `" la la la ..."`, which reads
    /// exactly like a broken 4-bit load — and it was `mlx_lm` saying the SAME
    /// nine tokens off the same file that identified the shell as right and
    /// the prompt as wrong. A reference is what tells those two apart.
    bos: Option<u32>,
    /// `mlx_lm.generate` over this snapshot and [`PROMPT`], greedy, no chat
    /// template — or `None` while it has not been recorded. Printed and not
    /// asserted, for `four_bit_first_light`'s reason: a test that shells out
    /// to a python interpreter fails for reasons that are not about this
    /// engine.
    mlx: Option<&'static str>,
}

/// **QWEN3.6-27B — `mlx-community/Qwen3.6-27B-4bit`.**
///
/// # The sanitize audit
///
/// `config.json` says `model_type: qwen3_5`, so the file that wrote this
/// checkpoint is `mlx_lm/models/qwen3_5.py` — the same module that wrote the
/// 0.8B, and the two audits are therefore the same audit. Both of that file's
/// transformations apply:
///
/// 1. **The re-rooted trunk.** `Qwen3_5ForConditionalGeneration.sanitize`
///    rewrites `model.language_model.*` to `language_model.model.*` and
///    hoists the readout to `language_model.lm_head.*`. Confirmed against the
///    index: every one of this checkpoint's 2180 tensors is spelled that way.
///    `qwen_3::import`'s `Layout::Mlx` already says so.
///
/// 2. **The folded rmsnorm one, WHICH IS CONDITIONAL AND HAD TO BE
///    CHECKED.** `Qwen3_5Model.sanitize` adds `1.0` to every plain-RMSNorm
///    plane — but only `if should_shift_norm_weights`, which is
///    `has_mtp_weights or has_unsanitized_conv1d`, both facts about the
///    SOURCE checkpoint rather than about the family. `Layout::folds_the_norm_one`
///    answers `true` for every MLX file unconditionally, so a conversion the
///    predicate had declined would be read with a one taken out that was
///    never put in.
///
///    **THE MLX FILE ANSWERS THE QUESTION ABOUT ITSELF, AND `conv1d` IS THE
///    WITNESS.** The same `sanitize` that shifts the norms also moves the
///    depthwise convolution's axes, `v.moveaxis(2, 1)`, under the same
///    predicate — and transformers ships `conv1d.weight` as `[C, 1, K]`, so
///    `shape[-1] != 1` is exactly `has_unsanitized_conv1d`. A converted file
///    whose `conv1d.weight` ends in `1` is a file the predicate fired on;
///    one that still ends in `K` is a file it declined, and would also be
///    a file `sanitize` re-shifts on every load. Measured: this checkpoint's
///    `linear_attn.conv1d.weight` is `[10240, 4, 1]` — the move ran, so the
///    shift ran, and `folds_the_norm_one` is right about this file. (The
///    0.8B's is `[6144, 4, 1]`, the same answer, which is why that file's
///    first light held.)
///
/// 3. **No MTP planes**, because `sanitize`'s first act is to drop every key
///    holding `mtp.`. The catalog's `qwen36-27b-mlxu4-kv-bf16` is built on
///    `Model::d27b_undrafted` for this reason and the index agrees: no
///    `mtp.*` name is in the file.
///
/// 4. **mrope is not a text-decode fact.** `rope_parameters` carries
///    `mrope_section: [11, 11, 10]` and `mrope_interleaved: true`, and the
///    0.8B — which achieved exact `mlx_lm` parity — carries the identical
///    triple. A text token's three position components are equal, so the
///    interleave is the identity over this file's prompts.
///
/// 5. **`vision_tower.*` is present and is not read.** 72 tensors of a
///    27-block vision encoder that this text SKU declares nothing for; an
///    import states what it needs and ignores the rest.
///
/// The dims check out against `Model::d27b_dims` leaf for leaf: hidden 5120,
/// 64 layers at `full_attention_interval` 4, 24 q heads over 4 kv, head_dim
/// 256 at `partial_rotary_factor` 0.25 (= rotary_dim 64), theta 1e7, 16 key
/// and 48 value linear heads at dim 128, conv kernel 4, intermediate 17408,
/// vocab 248320, untied.
const QWEN: Family = Family {
    sku: "qwen36-27b-mlxu4-kv-bf16",
    env: "PIE_QWEN36_SNAPSHOT",
    repo: "models--mlx-community--Qwen3.6-27B-4bit",
    word: |query_len| {
        model::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
    },
    bos: None,
    // Nine greedy tokens, `[11751, 13, 198, 760, 6511, 314, 9338, 369, 279]`,
    // from `mlx_lm.generate --temp 0 --max-tokens 9 --ignore-chat-template`
    // over this snapshot and this prompt. The shell matched all nine on the
    // first fire it ever made at this checkpoint.
    mlx: Some(" Paris.\nThe capital of France is the"),
};

/// **GPT-OSS-20B — `mlx-community/gpt-oss-20b-MXFP4-Q4`.**
///
/// # The sanitize audit
///
/// **THE TRUNK DOES NOT MOVE.** gpt-oss is a plain `GptOssForCausalLM` with no
/// multimodal wrapper, so `GptOssModel.sanitize` re-roots nothing and the
/// index agrees: `model.layers.{l}.*`, `model.embed_tokens.*`, `lm_head.*`,
/// `model.norm.weight`, exactly where transformers put them. **No norm is
/// folded** either — `mlx_lm`'s gpt-oss uses the stock `nn.RMSNorm` and
/// `sanitize` touches no norm plane.
///
/// What it DOES do is repack the MoE, three ways at once — the fused
/// `gate_up_proj` split in two by row parity, the OCP `_blocks` bytes viewed
/// as `u32` words, and `_bias` split to match. `gpt_oss::import`'s `Layout`
/// carries all three, and its doc quotes the `sanitize` that is the reason for
/// each.
///
/// # What first light found, and it was a catalog gap
///
/// **THIS CHECKPOINT IS QUANTIZED IN TWO SCHEMES AND THE CATALOG NAMED ONE.**
/// `gptoss-20b-bf16-mxfp4-kv-bf16` says bf16 dense weights beside mxfp4
/// experts, which is `openai/gpt-oss-20b` exactly. The MLX conversion is
/// bf16 in neither place: `config.json`'s `quantization` map lists 98 tensors
/// at `{mode: affine, bits: 4, group_size: 64}` — every `q/k/v/o_proj`,
/// `model.embed_tokens` and `lm_head` — beside the global
/// `{mode: mxfp4, bits: 4, group_size: 32}` for the experts, and 24 more at
/// **8 bits** for the router gates. Three representations in one file.
///
/// So the row this arm serves is a NEW one,
/// `gptoss-20b-mlxu4-mxfp4-kv-bf16`, and landing it took the tree's first
/// eight-bit weight: see `dtype::Dtype::MlxU8`, whose doc carries the
/// `quant_predicate` that is the reason a router gate is coarser than the
/// stack around it. Nothing below the model text moved — the affine points
/// were already stamped at both widths and already read their `bits` off the
/// plane's own spec.
///
/// # The parity bar, and why it is PRINTED
///
/// `mlx_lm` gives TWO answers to this prompt, and the disagreement is its own:
///
/// ```text
/// generate_step, cached : [12650, 3692, 279, 2167, 1309, 316, 8420, 290, 21872]
/// full forward per step : [12650, 13, 623, 9029, 328, 10128, 382, 290, 5030]
/// ```
///
/// **THE SECOND IS THIS SHELL'S, TOKEN FOR TOKEN, ALL NINE.** The two mlx
/// paths part at step one and the reason is measurable in mlx's own
/// arithmetic: the top two logits there are `'.'` at 15.6875 and `'."'` at
/// **15.625**, a margin of 0.0625 — a bf16 rounding artifact, against a
/// two-implementation noise floor this family of checkpoints measures at
/// 0.103 rms. Every other step of the nine is decided by 0.94 to 5.13.
///
/// A step whose margin is 0.0625 is a step no assertion may read: green would
/// not mean right and red would not mean wrong, and mlx demonstrates it by
/// answering both ways itself. So the record is printed, [`EXPECTED`] is what
/// is asserted, and the tie is written down here rather than hidden in a pin.
const GPT_OSS: Family = Family {
    sku: "gptoss-20b-mlxu4-mxfp4-kv-bf16",
    env: "PIE_GPTOSS_SNAPSHOT",
    repo: "models--mlx-community--gpt-oss-20b-MXFP4-Q4",
    word: |query_len| {
        model::gpt_oss::forward::Facts::of(&Request::new(query_len, false)).word()
    },
    bos: None,
    // The full-forward path of the two above — see the type doc. Printed, not
    // asserted: step one is a 0.0625-logit tie.
    mlx: Some(" Paris. The capital of France is the city"),
};

/// **GEMMA-4-31B — `mlx-community/gemma-4-31b-it-4bit`.**
///
/// # The sanitize audit
///
/// 1. **The re-rooted trunk**, qwen's bug in gemma's spelling.
///    `Gemma4Model.sanitize` strips the leading `model.` and re-inserts it one
///    level down, so transformers' `model.language_model.layers.*` is written
///    `language_model.model.layers.*`. `gemma_4::import`'s new `Layout` says
///    so; before it, no `mlx_lm` output could satisfy this SKU.
///
/// 2. **GEMMA 4 DOES NOT FOLD THE RMSNORM ONE, AND GEMMA 3 DID.** The most
///    load-bearing finding of the three audits is an ABSENCE. `gemma3_text.py`
///    carries its own `RMSNorm` computing `rms_norm(x, 1.0 + weight)` — the
///    gemma convention through three generations, and the thing this file's
///    author would have wired a `Bias(-1.0)` for by analogy. `gemma4_text.py`
///    uses the stock `nn.RMSNorm`, which has no constant, and neither
///    `sanitize` touches a norm. The checkpoint says the same thing louder:
///    `layers.0.input_layernorm.weight` means +4.88 and reaches 444.0, and
///    `self_attn.q_norm.weight` is the constant 1.0234 across all 256 entries
///    — multiplicative scales, not offsets from one. `gemma_4::forward`
///    already spells every one `ops::elemwise::rmsnorm`, so the right fix was
///    to change nothing. See `gemma_4::import::Layout`.
///
/// 3. **`vision_tower.*` and `embed_vision.*` are present and unread** — 137
///    tensors of a 27-block encoder this text SKU declares nothing for.
///
/// # What first light found
///
/// Three bugs, all in the model text and none in a kernel:
///
/// 1. **The trunk prefix** above.
///
/// 2. **`attention_k_eq_v`: ten layers with no `v_proj`.** The 31B publishes
///    50 value projections for 60 layers, and the ten missing ones are exactly
///    the `full_attention` entries of `layer_types`. See
///    `gemma_4::import`'s `AttnBanks::Owned` arm — the value leg reads the
///    KEY's triplet, which is what `mlx_lm` computes (`values = keys`) and what
///    `qkv_unfused` was already shaped for.
///
/// 3. **Sixty `layer_scalar` planes read by nobody.** This text had the
///    per-layer scalar only under the PLE relay, which `b31` does not have —
///    and the scalars are not ones: 0.0894, 0.0654, ..., 0.0364, a factor of
///    twenty-seven across the stack. See `gemma_4::model::Layer::scalar`.
///
/// And one dim that was simply wrong: `sliding_window` is 1024 in this
/// checkpoint and this text said 512.
///
/// # The prompt is half the answer here
///
/// See [`Family::bos`]. Gemma's first fire said `" la la la la ..."` and so
/// did `mlx_lm` off the same file; one prepended `<bos>` is what turns both
/// into the answer.
const GEMMA: Family = Family {
    sku: "gemma4-31b-mlxu4-kv-bf16",
    env: "PIE_GEMMA4_SNAPSHOT",
    repo: "models--mlx-community--gemma-4-31b-it-4bit",
    word: |query_len| {
        model::gemma_4::forward::Facts::of(&Request::new(query_len, false)).word()
    },
    // `<bos>`, id 2 — see `Family::bos`, which this family is the reason for.
    bos: Some(2),
    // Nine greedy tokens from `mlx_lm` over this snapshot, this prompt and
    // the same prepended `<bos>`:
    // `[9079, 236761, 9079, 563, 3224, 618, 506, 623, 17698]`.
    mlx: Some(" Paris. Paris is known as the \"City"),
};

/// One shell at a time per process — see the module doc.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Family {
    /// The snapshot directory: the named override, or this family's own
    /// repository in the cache.
    fn snapshot(&self) -> Option<PathBuf> {
        if let Ok(stated) = std::env::var(self.env) {
            let path = PathBuf::from(stated);
            return path.is_dir().then_some(path);
        }
        let homes = [
            std::env::var("HOME").unwrap_or_default(),
            "/Users/ingim".to_string(),
        ];
        homes.iter().find_map(|home| {
            let snaps = Path::new(home)
                .join(".cache/huggingface/hub")
                .join(self.repo)
                .join("snapshots");
            let mut found: Vec<PathBuf> = std::fs::read_dir(&snaps)
                .ok()?
                .filter_map(|entry| Some(entry.ok()?.path()))
                .filter(|path| !containers(path).is_empty())
                .collect();
            found.sort();
            found.into_iter().next()
        })
    }
}

/// **EVERY CONTAINER OF THE SNAPSHOT, SORTED — AND THE PLURAL IS THE POINT.**
///
/// `four_bit_first_light` opens one file because the 0.8B is one file. All
/// three of these are sharded: three shards for the 27B and the 20B, four for
/// the 31B, and a contract built over shard one alone refuses at the first
/// tensor that lives in shard two. Sorted, because `index_all` numbers its
/// stores by position and a `read_dir` order is the filesystem's.
fn containers(snapshot: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(snapshot)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            (name.ends_with(".safetensors") || name.ends_with(".zt")).then_some(path)
        })
        .collect();
    found.sort();
    found
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

/// One prefill and [`STEPS`] decodes in one slot, greedy throughout.
fn run(shell: &mut Shell, family: &Family, slot: u32, prompt: &[u32]) -> (Vec<u32>, Vec<f64>) {
    shell.open(slot).expect("the slot opens");

    let prefill = shell
        .fire(&[Lane {
            slot,
            word: (family.word)(prompt.len() as u32),
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
                word: (family.word)(1),
                tokens: &fed,
            }])
            .unwrap_or_else(|why| panic!("decode step {step} fires: {why}"));
        millis.push(at.elapsed().as_secs_f64() * 1000.0);
        finite(&decode[0], "decode");
        produced.push(argmax(&decode[0]));
    }
    (produced, millis)
}

/// Everything an arm needs, or `None` and a sentence naming the missing
/// precondition.
///
/// **THE BUDGETS ARE SMALL ON PURPOSE.** One lane and 128 rows against a
/// 256-token context and a single slot: these are 15-17 GiB of weights on a
/// 32 GiB box, and the arena is the one term of the footprint a test gets to
/// choose. Nothing below asks for more than eight rows.
fn ready(family: &Family) -> Option<(Shell, tokenizer::Tokenizer)> {
    let sku = family.sku;
    if !engine_metal::device::present() {
        eprintln!("skipping {sku}: this machine publishes no Metal device");
        return None;
    }
    let Some(trace) = model::trace_of(sku) else {
        eprintln!("skipping {sku}: the catalog ships no row by that name");
        return None;
    };
    let Some(import) = model::import_of(sku) else {
        eprintln!("skipping {sku}: the catalog ships no import for that row");
        return None;
    };
    let Some(snapshot) = family.snapshot() else {
        eprintln!(
            "skipping {sku}: no snapshot of {} in the hugging face cache — name one in {}",
            family.repo, family.env
        );
        return None;
    };
    if !snapshot.join("tokenizer.json").exists() {
        eprintln!("skipping {sku}: {snapshot:?} ships no tokenizer beside its tensors");
        return None;
    }
    let files = containers(&snapshot);
    eprintln!("{sku}: reading {} container(s) under {snapshot:?}", files.len());

    let tokenizer = tokenizer::Tokenizer::from_file(&snapshot.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");

    let source = ztensor_compat::index_all(&files).expect("the checkpoint's shards open as one");
    let contract = import(&source).unwrap_or_else(|why| {
        panic!("{sku}'s import contract does not fit {snapshot:?}: {why}")
    });
    drop(source);

    let booted = Instant::now();
    let shell = Shell::load(Boot {
        trace: trace(Platform::Metal),
        contract: &contract,
        checkpoint: &snapshot,
        budget: Budget::new(1, 128),
        profile: None,
        page_size: 16,
        context: 256,
        slots: 1,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
    })
    .unwrap_or_else(|why| panic!("{sku} loads: {why}"));
    let (weights, arena, pools, inputs) = shell.footprint();
    eprintln!(
        "loaded {sku} on {} in {:.1}s — weights {:.2} GiB, arena {:.1} MiB, pools {:.1} MiB, \
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

/// **THE CLAIM, ONE FAMILY AT A TIME.** Prefill, decode, and say the true
/// thing — then say it identically a second time.
fn first_light(family: &Family) {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready(family) else {
        return;
    };
    let sku = family.sku;
    // The family's beginning, then the prompt's own tokens — see
    // [`Family::bos`].
    let mut prompt = Vec::new();
    prompt.extend(family.bos);
    prompt.extend(tokenizer.encode(PROMPT));
    eprintln!(
        "{sku}: {PROMPT:?} encodes to {} tokens{}",
        prompt.len(),
        match family.bos {
            Some(bos) => format!(", the first of which is the prepended bos {bos}"),
            None => ", with no prepended beginning".to_string(),
        },
    );

    let (produced, millis) = run(&mut shell, family, 0, &prompt);
    let text = tokenizer.decode(&produced, false);
    let warm = &millis[millis.len() / 2..];
    eprintln!(
        "{sku}: -> {text:?}\n  tokens {produced:?}\n  {:.2} ms/fire warm, {} shader points compiled",
        warm.iter().sum::<f64>() / warm.len() as f64,
        shell.compiled(),
    );

    match family.mlx {
        Some(reference) => eprintln!(
            "{sku}: mlx_lm said {reference:?} — parity: {}",
            if text.starts_with(reference) || reference.starts_with(&text) {
                "token for token over the shorter of the two"
            } else {
                "DIVERGES, read the two strings above"
            }
        ),
        None => eprintln!("{sku}: no mlx_lm reference recorded yet — the line above is the record"),
    }

    // Determinism, in the same loaded shell: a dequantization that read an
    // unsynchronized staging buffer, or a scales plane bound to a rectangle
    // two arms both write, shows up here and nowhere else.
    let (again, _) = run(&mut shell, family, 0, &prompt);
    assert_eq!(
        produced, again,
        "{sku}: two identical fires answered differently, which means two arms wrote the same rows"
    );

    assert!(
        text.starts_with(EXPECTED),
        "{sku}: the continuation is {text:?}, and a correct load begins it {EXPECTED:?}"
    );
}

/// **QWEN3.6-27B, FIRST LIGHT.** Sixty-four layers, forty-eight of them
/// gated-delta and sixteen full attention, every matmul bank affine-U4 at
/// group 64. See [`QWEN`] for the audit that comes before the run.
#[test]
fn qwen36_27b_four_bit_prefills_decodes_and_says_something_true() {
    first_light(&QWEN);
}

/// **GPT-OSS-20B, FIRST LIGHT.** Twenty-four layers alternating sliding and
/// full attention, 32 experts routed four ways, learned attention sinks. See
/// [`GPT_OSS`].
#[test]
fn gpt_oss_20b_mxfp4_prefills_decodes_and_says_something_true() {
    first_light(&GPT_OSS);
}

/// **GEMMA-4-31B, FIRST LIGHT.** Sixty layers over TWO attention shapes —
/// 512-wide global every sixth layer, 256-wide sliding between them — with a
/// softcapped readout. See [`GEMMA`].
#[test]
fn gemma4_31b_four_bit_prefills_decodes_and_says_something_true() {
    first_light(&GEMMA);
}
