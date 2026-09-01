//! **FIRST LIGHT FOR THE 2-BIT MoE PLANE.** The whole `MlxU2G{32,64,128}`
//! path — the two-bit affine checkpoint contract, the split expert pair and
//! the `linear.mlp_swiglu_clamp_split` combine that pair needs, the routed
//! qmv points at group 32 and 64, and every dsv4-flash organ around them —
//! was built against headers and a written description. This file is the
//! first time any of it reads a checkpoint.
//!
//! # The checkpoint
//!
//! `mlx-community/DeepSeek-V4-Flash-2bit-DQ`, the `mini-l5-e16` snapshot: the
//! real DeepSeek-V4-Flash geometry (hidden 4096, 64 MLA heads of width 512,
//! `moe_intermediate_size` 2048, 129 280 tokens) over the FIVE renumbered
//! layers 0, 1, 2, 3, 42 and SIXTEEN routed experts. 1.57 GiB of a
//! per-tensor "DQ" quantization: a default of `(4 bits, group 64)` and
//! seventy-one overrides, of which the routed experts are all 2-bit — and of
//! which ONE is the landmine this whole lane exists for.
//!
//! **THE LANDMINE.** The routed `gate_proj` is grouped by 32 on layers 0-3 and
//! by 64 on the last, while `up_proj` and `down_proj` are grouped by 64
//! throughout. A `Weight` carries ONE `Dtype` and a `Dtype` is where an affine
//! group is written down, so the fused `[experts, 2·inter, hidden]` expert
//! bank this family declared could not state four of these five layers at all:
//! 2048 rows of 128 groups do not join 2048 rows of 64 into a rectangle at any
//! axis. `models::deepseek_v4::model::GateUp::Split` is the answer — two banks,
//! each at its own point — and this file is where it meets bytes.
//!
//! # What first light found
//!
//! Nothing in the 2-bit path. Every fault between "the contract fits" and
//! "the logits are numbers" was in dsv4's own never-fired declarations and in
//! this shell's paged store, and each had stood green in the bake census
//! because a bake is a compile and not a fire:
//!
//! 1. **the shared kv plane** — `Pools::reserve` cut every kv row into a key
//!    half and a value half, and the MLA latent is ONE plane both readers
//!    address (`store::split`);
//! 2. **the factless rows** — it also demanded a paged launch's restatement
//!    for every row, and the latent, index and pool rows have none by
//!    construction (`model_exec::store::kv::SpaceFacts` says so outright);
//! 3. **`request_of_token`** — fire-wide and bound into no space's seat, so
//!    the first plan to read it (dsv4's pooled reader) met an unbound seat;
//! 4. **the cache row's width** — declared `heads · head_dim` where the
//!    appender writes `kv_down`'s own 512-wide latent, a row SIXTY-FOUR times
//!    too wide, with `kv_heads` 64 to match it;
//! 5. **two pooling ratios, one state plane** — dsv4 carries ratio 4 and
//!    ratio 128, the reservation lays one plane at the widest pitch, and the
//!    gather addressed it at its own;
//! 6. **the sink's dtype** — declared f32 to match the checkpoint, while
//!    `attention.sink` templates the plane on the ACTIVATION. An f32 plane at
//!    a `bfloat*` seat is what every NaN in this file's history was.
//!
//! # What woke since
//!
//! The first run of this file fired with FOUR of this family's organs
//! interned — their checkpoint planes read into the trace and no node firing
//! one. All four fire now, and
//! [`the_fired_plan_names_the_organs_that_woke`] is the gate that says so by
//! op name, because every other claim in this file was already true while
//! they slept:
//!
//! 1. **the hash gate** — layers 0-2 route by `ffn.gate.tid2eid`
//!    (`linear.moe_hash_route`), not by a softmax over a router they do not
//!    read;
//! 2. **the learned compressor** — `wkv`/`wgate` project the rolling state
//!    (`attention.pool_state_write`), `ape` folds into the gate logits, and
//!    the compressor's norm closes the pooled entry; the gather used to pool
//!    zeros;
//! 3. **the dynamic hc mix** — `{attn,ffn}_hc.fn` projects the mix row
//!    (`elementwise.hc_project`) the Sinkhorn gate splits, where the gate
//!    used to read the leading floats of the normed buffer;
//! 4. **the NSA lightning indexer** — the ratio-4 layers rank their own
//!    compressed rows (`attention.index_topk`, keyed off entries their OWN
//!    compressor pooled) and read the chosen ones
//!    (`attention.pool_lse_selected`), where the compressed branch used to
//!    read every visible row.
//!
//! The sliding-window branch (`attention.prefill_lse` at `window`) and the
//! merge that folds it against the compressed one were already live.
//!
//! # What first light claims
//!
//! There is **no external reference for this miniature**. It is five of a
//! forty-three-layer model's layers with the other thirty-eight deleted and
//! sixteen of its two hundred and fifty-six experts kept; `mlx_lm` will not
//! generate from it and no published continuation exists to compare against.
//! So first light here is NOT token parity, and this file does not pretend
//! otherwise. It is four claims, each of which a wrong load fails:
//!
//! 1. **it loads** — the import contract fits the real bytes, and every plane
//!    of it seats on the device;
//! 2. **it fires** — a prefill and [`STEPS`] decodes complete, with the split
//!    expert pair and its combine dispatching on every layer of every fire;
//! 3. **the numbers are numbers** — finite, and a spread no rectangle nobody
//!    wrote could have;
//! 4. **the census is truthful** — every routed plane the loaded PLAN carries
//!    is at the `(bits, group)` the file wrote its half at, the layer-4
//!    landmine included.
//!
//! Claim 4 is the one that separates a 2-bit load from a 2-bit-shaped one at
//! this bar. Everything above it can bind, dispatch and return finite numbers
//! off codes read at the wrong group — a plane dequantized at group 64 where
//! the file wrote 32 lands the right spread around the wrong centre, with no
//! NaN to notice it by — so the census is asserted against the artifact's own
//! header and not against a table.
//!
//! # Gating
//!
//! Apple-only at compile time, and SKIPS at run time naming which
//! precondition was missing — the device, the snapshot, or the tokenizer
//! beside it.
//!
//! ```text
//! cargo test -p engine-metal --release --test two_bit_moe_first_light -- --nocapture
//! ```

#![cfg(target_vendor = "apple")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use engine_metal::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

/// The catalog row this first light serves.
const SKU: &str = "dsv4-flash-mlxu2-kv-bf16";

/// The repository the snapshot lives in.
const REPO: &str = "models--mlx-community--DeepSeek-V4-Flash-2bit-DQ";

/// How many decode fires follow the prefill.
///
/// Enough that the decode class is entered, re-entered, and entered again
/// with a longer cache than the prefill left: a routed path that reads its
/// scratch off the wrong fire, or a kv append that lands one row short,
/// shows up as a step that is not finite or not the same twice, and neither
/// is visible in one fire.
const STEPS: usize = 8;

/// The prompt. Nothing rides on the words — see the module note: there is no
/// reference continuation for a five-layer sixteen-expert cut of a 43-layer
/// model, so this is a token sequence long enough to prefill a real
/// rectangle and no more.
const PROMPT: &str = "The capital of France is the city of";

/// One shell at a time per process: these hold the whole weight table
/// resident and the measurements are only readable one at a time.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The snapshot: the checkpoint AND the tokenizer that goes with it.
/// `PIE_U2_SNAPSHOT` overrides where it is looked for.
fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_U2_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let usable = |path: &Path| path.join("tokenizer.json").exists() && container(path).is_some();
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

/// The container the contract is checked against — one file of the snapshot,
/// whichever one holds the tensors.
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

/// One safetensors header, as `name -> value`.
fn header(path: &Path) -> Option<serde_json::Map<String, serde_json::Value>> {
    let bytes = std::fs::read(path).ok()?;
    let len = u64::from_le_bytes(bytes.get(..8)?.try_into().ok()?) as usize;
    let parsed: serde_json::Value = serde_json::from_slice(bytes.get(8..8 + len)?).ok()?;
    parsed.as_object().cloned()
}

/// The `(bits, group)` one bank of a plan spends, read off the PLAN'S OWN
/// planes: the code width from the bank's dtype, and the group from the ratio
/// between the bank's contracted axis and the same axis of the `.scales`
/// companion it interned beside itself.
///
/// **DERIVED AND NOT TABULATED.** A `match` from `Dtype` to a pair, written
/// here, would be the claim written twice — the test would pass whenever this
/// file and `checkpoint_dsl` agreed, whatever the plan did. This reads the two
/// rectangles the plan actually declares and divides them.
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
/// how many bytes it asked for, and the question a 1.57 GiB artifact on a
/// 32 GiB box raises is how many the KERNEL then wired. The two are different
/// numbers and only one of them is a fact about the machine. `None` where
/// `vm_stat` is not readable or its format is not the one parsed here — a
/// missing number is printed as missing, never as zero.
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
    models::deepseek_v4::forward::Facts::of(&Request::new(query_len, false)).word()
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

/// Everything the tests below share: a loaded 2-bit shell and its vocabulary,
/// or `None` and a sentence saying which precondition was missing.
fn ready(what: &str) -> Option<(Shell, tokenizer::Tokenizer)> {
    if !engine_metal::device::present() {
        eprintln!("skipping {what}: this machine publishes no Metal device");
        return None;
    }
    let Some(checkpoint) = snapshot() else {
        eprintln!(
            "skipping {what}: no {REPO} snapshot with a tokenizer beside it under \
             $HOME/.cache/huggingface/hub — name one in PIE_U2_SNAPSHOT"
        );
        return None;
    };
    let Some(container) = container(&checkpoint) else {
        eprintln!("skipping {what}: {checkpoint:?} holds no tensor container");
        return None;
    };

    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");

    let trace = models::trace_of(SKU).expect("the catalog ships the 2-bit SKU");
    let trace = trace(Platform::Metal);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract =
        models::import_of(SKU).expect("the catalog ships an import for the SKU")(&source)
            .expect("the 2-bit SKU's import contract fits the real DQ checkpoint");
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
        // Conservative: a miniature on a 32 GiB box, four lanes and a short
        // context. The claim is that it loads and fires, not that it scales.
        budget: Budget::new(4, 512),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
    })
    .expect("the 2-bit shell loads");
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

/// **THE CLAIM.** A real 2-bit MLX DQ checkpoint with an unfused expert pair,
/// prefilled and decoded on an Apple GPU, returns numbers.
#[test]
fn a_real_two_bit_moe_checkpoint_prefills_and_decodes() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the 2-bit first light") else {
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
    // note: this snapshot is five layers of forty-three and sixteen experts of
    // two hundred and fifty-six — it is not a language model and it does not
    // answer like one. As recorded, the continuation is one token then ONE
    // TOKEN REPEATED, which is what a tower with thirty-eight of its layers
    // deleted answers to anything.
    //
    // **THE ORGANS MOVED IT AND DID NOT FIX IT**, which is the honest
    // reading. The first recorded run of this file — with the hash gate, the
    // learned compressor and the dynamic hc mix all interned — repeated token
    // 26158 from the prefill onward. With those three fired it repeats 10177
    // off 1718, so the three are in the ARITHMETIC and not merely in the
    // plan; and it still repeats, because seven eighths of the tower is
    // missing and no organ fixes that.
    //
    // **AND THE FOURTH ORGAN MOVED NOTHING HERE, WHICH IS THE ANSWER IT
    // OWES.** The NSA fine branch fires — the plan gate below counts two
    // rankings and two selected readers — and the continuation is
    // TOKEN-FOR-TOKEN what it was with the branch asleep. That is not the
    // branch failing to reach the arithmetic; it is the branch's own
    // identity. This miniature's sequences are far inside the trained
    // `index_topk` of 512, so every row's visible compressed count fits its
    // budget, `index_topk_paged` publishes the identity `0..nvis-1`, and a
    // selected reader over the identity walks the dense reader's cells in
    // the dense reader's order. `nsa_selected_on_device`'s gate (b) measures
    // that equality on the card directly. The day a sequence here exceeds
    // 2048 tokens, THIS is the line that changes.
    //
    // **THE COMPRESSED ROPE MOVED AND THIS DID NOT, AND THAT IS WEAK
    // EVIDENCE, NOT STRONG.** The pooled entries used to rope at the token
    // their window CLOSED on and now rope at the compressed row's own
    // position (`arange(0, cutoff, ratio)`, `pool_on_device` gate (f)) — a
    // `ratio - 1` correction on every compressed key on both the attention
    // and the indexer branch. The continuation is byte-identical across it.
    // That is what this geometry predicts rather than a parity claim: a
    // sixteen-position run closes TWO ratio-4 boundaries and no ratio-128
    // one, so four compressed keys moved their angle inside a merge whose
    // other branch is a 128-wide window over every token — and the tower
    // answering this is five layers of forty-three, saturated on one repeated
    // id. A row long enough to make the compressed branch carry real mass is
    // where this claim would have teeth, and this file is not it.
    //
    // What is asserted is what a wrong load fails: every fire returned finite,
    // spread logits (`run` checks each step), the steps happened, and the
    // three gates below hold — including
    // [`the_fired_plan_names_the_organs_that_woke`], which is the one that
    // separates this run from one where the organs quietly stayed asleep. The
    // day a full-depth 2-bit artifact is on the box, THIS is the file that
    // grows a parity bar.
}

/// **DETERMINISM.** Two identical runs in two slots produce identical tokens.
///
/// A dequantization that reads uninitialized bytes, a routed scratch aliased
/// across two fires, or a sort whose ties fall by whatever the arena held are
/// all invisible in one run and all visible here.
#[test]
fn the_same_prompt_twice_is_the_same_tokens() {
    let _serial = serialized();
    let Some((mut shell, tokenizer)) = ready("the 2-bit determinism gate") else {
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

/// **THE ORGANS ARE IN THE PLAN, AND THIS FILE WOULD PASS WITHOUT THEM.**
///
/// Every claim above — it loads, it fires, the numbers are numbers, the same
/// prompt twice is the same tokens — was already true when four of this
/// family's organs were INTERNED: their checkpoint planes were read into the
/// trace and no node fired one. A first light that silently skipped them
/// would look exactly like this one. So the plan is asked, by op name, which
/// organs it actually names.
///
/// Needs no device and no snapshot: it is the trace and nothing else.
#[test]
fn the_fired_plan_names_the_organs_that_woke() {
    use model_ir::Operands;

    let trace = models::trace_of(SKU).expect("the catalog ships the 2-bit SKU");
    let trace = trace(Platform::Metal);
    let count = |op: &str| trace.nodes.iter().filter(|n| n.op.name() == op).count();

    // **THE HASH GATE.** `num_hash_layers` is three and the mini snapshot has
    // five layers, so three of them route by `ffn.gate.tid2eid` and two by the
    // `noaux_tc` correction bias. The softmax stand-in those three used while
    // no lookup op existed must be gone: it is what "interned" looked like
    // from the outside.
    assert_eq!(
        count("linear.moe_hash_route"),
        3,
        "the first three layers route by the token-id table"
    );
    assert_eq!(
        count("linear.moe_topk_sqrt_softplus"),
        2,
        "the two later layers score the router with the correction bias"
    );
    assert_eq!(
        count("linear.moe_topk_softmax"),
        0,
        "a plain softmax top-k is the stand-in the hash layers used before \
         `linear.moe_hash_route` landed, and nothing in this family wants one"
    );

    // **THE LEARNED COMPRESSOR — AND THERE ARE FIVE OF THEM, NOT THREE.**
    // Three of the five layers pool (ratios 4, 128, 4), and every one of them
    // WRITES the rolling state its gather reads. The two ratio-4 layers carry
    // a SECOND compressor beside it: the indexer's own
    // (`indexer.compressor.*`, `2 x index_head_dim` of window state into a
    // 128-wide key), which is the same organ at a narrower width and is where
    // this family's index keys come from.
    let compressors = count("attention.pool_gather");
    assert_eq!(
        compressors, 5,
        "three attention compressors and the two indexers' own"
    );
    assert_eq!(
        count("attention.pool_state_write"),
        compressors,
        "every compressor writes the state its own gather pools out of"
    );
    assert_eq!(
        count("attention.pool_kv_append"),
        compressors,
        "every compressor lands its pooled entry at the boundary cell"
    );

    // **THE DYNAMIC hc MIX.** Two sublayers a layer, five layers, and each
    // projects its `{attn,ffn}_hc.fn` plane into the mix row the gate splits.
    let gates = count("elementwise.hc_gates");
    assert_eq!(gates, 10, "two gated sublayers on each of the five layers");
    assert_eq!(
        count("elementwise.hc_project"),
        gates,
        "every gate is fed a projected mix row and not the normed buffer it \
         used to read the leading floats of"
    );

    // **THE NSA INDEXER'S FINE BRANCH FIRES**, and this assert is the line
    // that changed. The two blockers this file used to name are answered in
    // `models::deepseek_v4::forward::indexer`: the keys are the indexer's OWN
    // compressor's pooled entries — one per ratio-4 block, not one per token
    // — and the reader the selection feeds is the selected twin of the POOLED
    // reader, because the compressed branch is the only key set that grows
    // with the context and so the only one a top-k can cap.
    // Three layers pool (ratios 4, 128, 4); the two ratio-4 ones are indexed.
    let pooled = 3;
    let indexed = 2;
    assert_eq!(
        count("attention.index_topk"),
        indexed,
        "the two ratio-4 layers rank their own compressed rows"
    );
    assert_eq!(
        count("attention.prefill_lse"),
        5,
        "the sliding-window branch fires on every layer"
    );
    assert_eq!(
        count("attention.pool_lse_selected"),
        indexed,
        "an indexed layer reads the compressed rows its indexer chose"
    );
    assert_eq!(
        count("attention.pool_lse"),
        pooled - indexed,
        "the one un-indexed pooled layer (ratio 128) still reads its \
         compressed rows densely — `S / 128` needs no capping, which is why \
         that layer carries no indexer"
    );
    assert_eq!(
        count("attention.index_kv_append"),
        0,
        "this family's index keys are pooled, not per-token: `pool_kv_append` \
         lands them and glm_5's per-token appender is no part of it"
    );

    eprintln!(
        "the plan names: {} hash routes, {} compressor state writes, {} mix \
         projections, {} selected pooled readers, {} dense pooled readers, {} \
         windowed readers, {} indexer rankings",
        count("linear.moe_hash_route"),
        count("attention.pool_state_write"),
        count("elementwise.hc_project"),
        count("attention.pool_lse_selected"),
        count("attention.pool_lse"),
        count("attention.prefill_lse"),
        count("attention.index_topk"),
    );
}

/// **THE CENSUS, HELD AGAINST THE BYTES.**
///
/// Every routed expert plane the LOADED PLAN carries, at the `(bits, group)`
/// its stored triplet was written at — the layer-4 landmine included, where
/// the gate half moves from group 32 to group 64 and nothing else does.
///
/// Read off `trace.params` rather than off the model text, because the plan is
/// what the shell seats: a text that says the right thing and a trace that
/// interns something else is exactly the failure this asks about. And held
/// against the artifact's own header rather than a table, because a table
/// typed here would be the same claim written twice.
///
/// This one needs no device — it is the plan and the file — so it runs
/// wherever the snapshot is.
#[test]
fn the_plans_routed_planes_are_the_points_the_file_wrote() {
    let Some(checkpoint) = snapshot() else {
        eprintln!("not asked: no {REPO} snapshot under $HOME/.cache/huggingface/hub");
        return;
    };
    let Some(container) = container(&checkpoint) else {
        eprintln!("not asked: {checkpoint:?} holds no tensor container");
        return;
    };
    let Some(head) = header(&container) else {
        eprintln!("not asked: {container:?} is not a readable safetensors header");
        return;
    };

    let trace = models::trace_of(SKU).expect("the catalog ships the 2-bit SKU");
    let trace = trace(Platform::Metal);

    // The plan's own routed banks, by layer: `layer.<l>.experts_<half>`.
    let mut plan: BTreeMap<(u32, &str), (u64, u64)> = BTreeMap::new();
    for param in &trace.params {
        let Some(rest) = param.name.strip_prefix("layer.") else {
            continue;
        };
        let Some((l, tail)) = rest.split_once('.') else {
            continue;
        };
        let half = match tail {
            "experts_gate" => "gate_proj",
            "experts_up" => "up_proj",
            "experts_down" => "down_proj",
            "experts_gate_up" => panic!(
                "`{}` is a FUSED expert bank, and this artifact's gate and up halves \
                 have no joint rectangle — the 2-bit row must declare the pair",
                param.name
            ),
            _ => continue,
        };
        let l: u32 = l.parse().expect("a layer number");
        plan.insert((l, half), plan_point(&trace.params, &param.name));
    }

    let stored = |l: u32, half: &str| -> (u64, u64) {
        let last = |suffix: &str| -> u64 {
            let name = format!("model.layers.{l}.ffn.switch_mlp.{half}{suffix}");
            head.get(&name)
                .and_then(|t| t.get("shape"))
                .and_then(serde_json::Value::as_array)
                .and_then(|s| s.last())
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_else(|| panic!("`{name}` is in the header"))
        };
        // The codes are u32 words; the `.scales` last extent is the group
        // count. Two equations, two unknowns: `words · 32/bits == groups ·
        // group` and the row width is the same either way, so the pair the
        // file spends is recovered by matching the plan's own bits.
        (last(".weight"), last(".scales"))
    };

    let mut faults = Vec::new();
    let mut seen = 0usize;
    for ((l, half), (bits, group)) in &plan {
        seen += 1;
        let (words, groups) = stored(*l, half);
        let codes = words * (32 / bits);
        if codes != groups * group {
            faults.push(format!(
                "layer {l} `{half}`: the plan says ({bits}, {group}), which reads \
                 {words} words as {} groups, and the file ships {groups}",
                codes / group,
            ));
        }
    }
    assert!(
        faults.is_empty(),
        "the plan and the artifact disagree on {} routed plane(s):\n{}\n",
        faults.len(),
        faults.join("\n"),
    );
    assert_eq!(seen, 15, "five layers of three routed projections each");

    // And the landmine, named: the gate half is the only one that moves, and
    // it moves on exactly one layer.
    let gate: Vec<(u64, u64)> = (0..5).map(|l| plan[&(l, "gate_proj")]).collect();
    assert_eq!(
        gate,
        vec![(2, 32), (2, 32), (2, 32), (2, 32), (2, 64)],
        "the routed gate_proj groups by 32 on layers 0-3 and by 64 on the LAST \
         (original layer 42, renumbered to 4) — a PER-LAYER exception, and the \
         one a text with a single dtype per bank would have read wrong"
    );
    for l in 0..5 {
        assert_eq!(plan[&(l, "up_proj")], (2, 64), "up_proj is group 64 throughout");
        assert_eq!(
            plan[&(l, "down_proj")],
            (2, 64),
            "down_proj is group 64 throughout"
        );
    }
    eprintln!("the plan's fifteen routed planes agree with the artifact, landmine included");
}
