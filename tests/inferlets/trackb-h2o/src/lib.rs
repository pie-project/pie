//! H2O — Heavy-Hitter Oracle (Zhang et al., 2023).
//!
//! <https://arxiv.org/abs/2306.14048>
//!
//! H2O and TOVA read the *same* quantity — the attention distribution the
//! decode kernel just computed — and differ in exactly one line: TOVA ranks by
//! the current step's attention and throws the previous steps away, H2O
//! accumulates. A position is a "heavy hitter" if it has drawn a large share of
//! attention *over the whole generation so far*, which is a much more stable
//! statistic than any single step's.
//!
//! That one-line difference is the whole reason this inferlet exists as a
//! separate program rather than a flag: it is the sharpest available test that
//! the score tap is genuinely stateful across fires. `tova-attention` carries
//! nothing between fires at all; this one accumulates, so its score mass must
//! grow linearly in the number of fires. If the loop-carried channel were
//! silently re-seeded, or the tap re-read stale device memory, the mass would
//! be flat and the test would say so.
//!
//! ## Unlike `tova-attention`, this one ENFORCES — and pays for it
//!
//! `tova-attention` is an observer: it computes a keep-set and the model
//! attends over everything anyway, which is what makes it a clean parity test
//! against `naive-baseline`. That is not a policy, though — it is the cost of a
//! policy with none of the benefit. H2O here closes the loop through the
//! **attention mask**, the door `.wiki/alto/attn-score.md` §4 names: "evict"
//! executes as CUSTOM MASK updates (the masked arm), with a page drop only when
//! a page's tokens are all dead. `snapkv-eviction` and
//! `sliding-window-attention` ride the same descriptor port; nothing here is a
//! second-party intrinsic or engine surface this program invented.
//!
//! **THE MASK AND THE CAPTURE CANNOT SHARE A FIRE, AND THAT IS THE ONE THING
//! THIS PROGRAM HAS TO WORK AROUND.** `crates/models/src/qwen_3/forward.rs`
//! orders the `masked` window ahead of the `captures_scores` one in the
//! attention split, as a correctness ruling: "a lane that asked for both takes
//! the masked arm and keeps no scores — the arm that could honor both does not
//! exist in the vocabulary (`Attention::Masked` exports no lse)". SnapKV never
//! meets this, because it observes at the prefill epilogue and enforces over
//! the decode. TOVA never meets it, because it does not enforce at all. H2O is
//! the one policy in Track B that wants to observe AND enforce on the same
//! fire, every fire, and it cannot.
//!
//! So the generation is cut in two. `observe_fires` unmasked capturing decode
//! fires accumulate the cumulative heavy-hitter statistic — the carry that is
//! H2O's entire difference from TOVA, and what `mass_trace`'s staircase
//! measures. The keep-set is then chosen from that statistic, and the remaining
//! `enforce_tokens` are decoded MASKED behind it. Deviation (4) below states
//! what that costs against the paper.
//!
//! **The page-drop route was checked and is not available to a guest.** Naming
//! fewer pages in the page list makes the KV cache shorter without making the
//! SEQUENCE shorter: the cuda shell derives a lane's positions as
//! `held .. held + rows` and refuses an explicit list by name (`Unsupported`,
//! verb `explicit lane positions`), and `WorkingSet::discard` says outright
//! that "suffix indexes shift down". A kept key carries the RoPE it was written
//! with, so re-indexing the cache without re-indexing the query is wrong by
//! however much was evicted — which is StreamingLLM's technique, not H2O's. A
//! page drop is therefore only honest when a page is *entirely* dead at the end
//! of the address space, which a heavy-hitter set never guarantees.
//!
//! **Eviction is page-granular, and that is not a shortcut.** The KV cache is
//! paged; a position-granular mask can stop attention from *reading* a position
//! but cannot hand its memory back, so the thing H2O exists to do — run in less
//! memory — is only expressible at page granularity. The program therefore
//! folds its `[kv_max]` position scores into `[p_max]` page masses (a page's
//! mass is the sum of its positions', which is the natural collapse for a
//! quantity that is already a probability mass) and ranks pages.
//!
//! ## The cold start, and why the seed is a descending ramp
//!
//! The first decode fire has to choose a keep-set having observed nothing:
//! this program taps only its decode fires — H2O's statistic is the decode
//! history, and reading the prompt's own end-of-prompt window is SnapKV's
//! policy, not this one — so the accumulator is still at its seed. An all-zero
//! seed would make every page tie and the selection arbitrary — evicting real
//! context on the strength of no evidence.
//!
//! The seed is instead a small descending ramp over positions, so with no
//! attention observed yet the ranking degenerates to "keep the earliest
//! positions", and the keep row independently admits every position written
//! after the selection. That is precisely the Λ-shape (attention sink + local
//! window) that StreamingLLM shows is the right prior, and that H2O's own
//! keep-sets converge to. The ramp is scaled far below one layer-fire of
//! attention mass, so a single observation dominates it; it breaks ties and
//! nothing more.
//!
//! Since the selection now happens after the observation phase rather than at
//! the first fire, the ramp decides nothing on any run with `observe_fires > 0`
//! — it is kept because `enforce_tokens == max_tokens - 1` is a legal ask, and
//! because a tie among never-attended pages should still go somewhere
//! principled.
//!
//! Slots past the live prefix get the *smallest* prior, which matters: the
//! backend consults the mask only for pages the request actually has, but a
//! ranking that put unused pages on top would evict live ones to make room.
//!
//! ## What the backend does under `intrinsics::attn_score`
//!
//! **The graph wrote it; the epilogue reads it** (`.wiki/alto/attn-score.md`
//! §4). The attention capture arm accumulates per-key mass into an arena
//! rectangle as it runs, and the epilogue is handed the whole thing at once as
//! a device tensor:
//!
//! ```text
//! intrinsics::attn_score(planes) -> [planes, intrinsics::attn_score_kv_max()]
//! ```
//!
//! one row per (exported attention layer, query head), **layer-major,
//! head-minor** (`layer * heads + head`), so a program that declares fewer
//! planes than the load exports reads a prefix of the layers rather than a
//! stripe of the heads. There is no per-layer stage and no host in the loop;
//! `Stage::OnAttn` no longer admits the intrinsic at all. The scores are the
//! softmax probabilities the attention kernel itself computed — captured as it
//! ran, never recomputed — so they cannot drift from the attention the model
//! performed. Model-gated on `has_attn_score`.
//!
//! The observation window is the backend's statute — the last
//! `min(32, qo_len)` query rows of the request. H2O taps 1-row decode fires,
//! where the window is one row and the quantity is exactly the current query's
//! distribution, which is what the paper defines.
//!
//! ## Deviations from the paper, as built
//!
//! 1. **Heads are folded by the PROGRAM.** H2O ranks per head; one page list
//!    per request means a per-head keep-set has no representable consumer. The
//!    rectangle is per-head because observability wants it that way (§4), so
//!    this program takes the mean over heads itself, in-graph at the epilogue.
//!    `quest-attention` and `tova-attention` document the identical collapse.
//! 2. **Layers are folded by the program.** Same reason: one page list per
//!    request is one keep-set per request. Summing layers is monotone-
//!    equivalent to averaging them, so the ranking does not depend on how many
//!    layers were declared.
//! 3. **Pages, not positions** — see above. This is a property of a paged KV
//!    cache, not of H2O.
//! 4. **The keep-set is frozen at the observe/enforce boundary, not re-ranked
//!    every step.** The paper re-ranks after every generated token. Here a fire
//!    either observes or enforces, never both (the masked/capture ruling
//!    above), so the statistic is accumulated over the first half of the
//!    generation and the second half is decoded behind the set it chose. What
//!    survives exactly is the thing that makes H2O *H2O*: the ranking is the
//!    CUMULATIVE attention over the whole history so far, not the latest
//!    step's — that is the carry `mass_trace` measures, and it is what
//!    `tova-attention` deliberately does not do. What is lost is the per-step
//!    re-ranking. §4 blesses a one-step-old addend ("a one-step-old addend
//!    changes nothing"); this is a longer lag than that, and it is a mechanism
//!    limit rather than a modelling choice, so it is named here rather than
//!    hidden in the statistic. `enforce_tokens: 1` recovers §4's exact lag at
//!    the cost of enforcing over a single token.
//!
//! Both folds are ONE in-graph reduction, on the device (§4: "reduction stays
//! on device — only decisions cross to the host"):
//!
//! ```text
//! fire_row = reduce_sum(transpose(rect)) / heads
//! ```
//!
//! `transpose` puts the planes on the last axis so `reduce_sum` — which
//! reduces the LAST axis — sums down them; `/ heads` turns that plane-sum into
//! (mean over heads, then sum over layers), because
//! `Σ_l (1/H) Σ_h row = (1/H) Σ_planes row`. One fire therefore contributes
//! mass exactly `layers`, which is the staircase step `mass_trace` asserts.
//!
//! **The layer count is declared, not counted.** The per-layer accumulator and
//! the device layer counter both existed to fold a per-layer tap across the
//! layer loop; there is no layer loop left. `layers_per_fire` is
//! `planes / heads`, read off the shape the program declared, and
//! `layers_observed` is that times the number of fires drained.

use inferlet::eta::attention::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_prompt")]
    prompt: String,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_seed")]
    seed: u32,
    /// KV pages H2O keeps. Positions written after the keep-set was chosen are
    /// kept on top of this, so the effective local window is never zero.
    #[serde(default = "default_page_budget")]
    page_budget: u32,
    /// How many of the generated tokens are decoded UNDER the keep-set, as
    /// opposed to observed. A masked lane keeps no scores (see the header), so
    /// the generation splits: `max_tokens - 1 - enforce_tokens` unmasked
    /// capturing fires accumulate the heavy-hitter statistic, then this many
    /// masked fires are served the set it chose.
    ///
    /// The default is half the decode budget: enough observation for the
    /// cumulative statistic to mean something, enough enforcement for its
    /// effect on the text to be visible. `0` runs the program as a pure
    /// observer (`tova-attention`'s shape); the whole budget runs it on the
    /// seed ramp alone, which is StreamingLLM and not H2O.
    #[serde(default)]
    enforce_tokens: Option<usize>,
    /// Drain the per-step score row to the host. The tests assert on it, so it
    /// defaults on; a benchmark must turn it off, because
    /// a `[kv_max]` f32 readback per decode step is tens of kilobytes and a
    /// round-trip that the POLICY never performs. Timing it measures the
    /// harness. See §14.2 of the design document.
    #[serde(default = "default_report")]
    report: bool,
    /// Pins the reserved page count so it does not move with `max_tokens`.
    /// `p_max` sets the width of the page row folded every observing fire and
    /// the width of the keep row every enforcing fire carries, so a benchmark
    /// that differences two `max_tokens` would otherwise be comparing two
    /// different per-step workloads and would charge the policy for the
    /// difference. Setting this to the larger of the two endpoints
    /// makes the per-step cost identical. Defaults to `max_tokens`.
    #[serde(default)]
    reserve_tokens: Option<usize>,
    /// Attention layers the load EXPORTS a score plane for. Declared, not
    /// derived: the plane count is not in the model profile and the SDK has no
    /// host call for it, so the program states it and the backend refuses a
    /// claim larger than the load exports (the `hidden(width)` deviation —
    /// a declared ceiling, checked by name).
    ///
    /// The default is `Qwen/Qwen3.5-0.8B`: `Model::d0_8b` in
    /// `crates/models/src/qwen_3/model.rs` is `layers: 24, attn_every: 4`, and
    /// the SKU is hybrid — `attn_at(l) = l % attn_every == attn_every - 1`
    /// puts an attention mixer on 6 of the 24 layers and a GDN mixer on the
    /// other 18. Only the attention layers export a plane.
    #[serde(default = "default_layers")]
    layers: u32,
    /// Query heads per exported attention layer. `Model::d0_8b`'s
    /// `q_heads: 8`, at `tp == 1`.
    #[serde(default = "default_heads")]
    heads: u32,
    /// Prefill chunk width, clamped to the engine's `max_embed_length()`.
    /// Defaults to that limit, i.e. the fewest chunks the engine allows.
    /// Forcing it down runs the multi-chunk path on a short prompt, which is
    /// the only way to test chunk equivalence without a 16K-token prompt.
    #[serde(default)]
    prefill_chunk: Option<u32>,
}

fn default_prompt() -> String {
    "The capital of France is".to_string()
}
fn default_temperature() -> f32 {
    1.0
}
fn default_max_tokens() -> usize {
    32
}
fn default_seed() -> u32 {
    0x70Au32
}
fn default_page_budget() -> u32 {
    4
}
fn default_report() -> bool {
    true
}
/// qwen35-d0.8b: 24 layers, `attn_every: 4` → 6 exported attention layers.
fn default_layers() -> u32 {
    6
}
/// qwen35-d0.8b: `q_heads: 8` at `tp == 1`.
fn default_heads() -> u32 {
    8
}

/// Largest entry of the cold-start ramp (see the module docs). Four orders of
/// magnitude below the 1.0 a single layer-fire contributes, so one observation
/// dominates it.
const SEED_SCALE: f32 = 1e-4;

/// What a slot may carry and still be explained by the seed alone. The ramp is
/// bounded by `SEED_SCALE`; anything above this is attention mass.
const SEED_CEILING: f32 = SEED_SCALE * 1.001;

#[derive(Serialize)]
struct Output {
    sampler: &'static str,
    text: String,
    count: usize,
    /// The program's own KV window into the published score row: the exact
    /// page geometry it reserved, `max_pages * page_size`. The ROW's width is
    /// not the program's to declare any more — it is the published constant
    /// `intrinsics::attn_score_kv_max()` — so this is the prefix of that row
    /// the program reads, and the program refuses up front if its geometry
    /// would outgrow the published one.
    kv_max: u32,
    /// Tokens a KV page holds on the serving engine — what `page_mass` is
    /// binned by, so a reader can place a page in tokens.
    page_size: u32,
    /// Live KV positions at the last observed step.
    kv_len: u32,
    page_budget: u32,
    /// Pages the request actually had at the last observed fire.
    live_pages: u32,
    /// Layers folded into the accumulator, CUMULATIVELY across every drained
    /// fire -- H2O never resets this, for the same reason it never re-seeds its
    /// accumulator. `layers_per_fire` times the number of fires drained.
    layers_observed: u32,
    /// Attention layers per fire. DERIVED FROM THE DECLARED SHAPE
    /// (`planes / heads`), not recovered as the step of a device counter:
    /// there is no per-layer tap left to count, and the rectangle arrives
    /// whole. This is the expected step of `mass_trace`.
    layers_per_fire: u32,
    /// Slots inside the live prefix that carry a finite, non-negative score.
    live_scored: usize,
    /// Slots past the live prefix carrying more than the seed ramp could
    /// account for. TOVA can assert the tail is exactly zero; H2O cannot,
    /// because its seed is a non-zero ramp that is never re-seeded away. What
    /// must still hold is that no ATTENTION mass ever lands out there -- a
    /// position that does not exist cannot have been attended to -- so the bar
    /// is the seed ceiling rather than zero.
    tail_polluted: usize,
    /// The largest value past the live prefix, for the same check by eye.
    tail_max: f32,
    /// One past the highest slot carrying any attention mass, i.e. the live KV
    /// length as the ENGINE saw it. Reported next to `kv_len` (the length the
    /// program believes it declared) because a disagreement between the two is
    /// the whole failure mode: it means the row describes different positions
    /// than the program thinks it does.
    observed_live: usize,
    /// `(declared kv_len, observed live length)` for every drained fire.
    trace: Vec<(u32, usize)>,
    scores_nan: usize,
    /// `Σ score` over the live prefix. Each layer of each fire contributes one
    /// distribution, so this must come out at `layers_observed` -- which for
    /// H2O is the CUMULATIVE layer count, not the per-fire one. A flat mass
    /// across fires is the signature of an accumulator that is being re-seeded
    /// behind the program's back.
    score_mass: f32,
    /// `score_mass` after each drained fire. Must be strictly increasing, by
    /// one fire's worth of layers each time. This is the observable that
    /// separates H2O from TOVA.
    mass_trace: Vec<f32>,
    /// H2O's page-level keep-set: the `page_budget` highest-mass live pages,
    /// ranked off the device's own page fold of the cumulative accumulator.
    /// This is the set the enforcement arm's mask was built from.
    kept_pages: Vec<u32>,
    /// The page H2O would evict next (lowest mass).
    evicted_first: Option<u32>,
    /// Fires that OBSERVED (unmasked, capturing) and tokens that were decoded
    /// under the keep-set (masked, silent). Their sum plus the prefill's one
    /// token is `count`.
    observe_fires: usize,
    enforce_tokens: usize,
    /// Pages live when the keep-set was chosen, and how many of them the served
    /// mask lets through. The all-keep arm reports the two equal.
    selected_pages: u32,
    served_pages: u32,
    /// KV positions the served mask admits of everything live when the set was
    /// chosen, against the number an unmasked decode attends over. THE MASK IS
    /// THE MEASUREMENT: a count of set bits in the row the fire carried.
    served_kv: u32,
    full_kv: u32,
    /// Per-page mass at the last fire, live pages only.
    page_mass: Vec<String>,
    score_head: Vec<String>,
}

fn step(logits: Tensor, temperature: f32, rng_state: &Tensor) -> Tensor {
    let scaled = if temperature == 1.0 {
        logits
    } else {
        &logits / temperature
    };
    gumbel_max(scaled, rng_state)
}

#[inferlet::main]
async fn main(input: Input) -> Result<Output> {
    if !input.temperature.is_finite() || input.temperature <= 0.0 {
        return Err("temperature must be finite and greater than 0".into());
    }
    if input.page_budget == 0 {
        return Err("page_budget must be at least 1".into());
    }
    if input.layers == 0 || input.heads == 0 {
        return Err("layers and heads must both be at least 1".into());
    }
    // The declared rectangle. `planes = exported attention layers * query
    // heads`, layer-major; the backend refuses a claim larger than the load
    // exports, by name.
    let layers = input.layers;
    let heads = input.heads;
    let Some(planes) = layers.checked_mul(heads) else {
        return Err("layers * heads overflows the plane count".into());
    };
    let heads_f = heads as f32;
    let max_tokens = input.max_tokens;
    let temperature = input.temperature;
    let ws = WorkingSet::new();
    let page_size = kv_page_size();

    if max_tokens == 0 {
        return Err("max_tokens must be at least 1".into());
    }

    // The model's opening (`<bos>` where it has one) before the raw text —
    // the opening `naive-baseline` puts there, which is what makes this
    // program's all-keep arm comparable to it, and what puts the attention
    // sink these policies rank at position 0.
    let mut prompt = inferlet::chat::prefix();
    prompt.extend(model::encode(&input.prompt));
    if prompt.is_empty() {
        prompt.push(0);
    }
    let n = prompt.len() as u32;
    let reserve = input.reserve_tokens.unwrap_or(max_tokens).max(max_tokens);
    let max_pages = (n + reserve as u32 + 1).div_ceil(page_size).max(1);
    // The program cannot know the runtime KV length, so — exactly like Quest's
    // `p_max` — it declares a static ceiling and the backend refuses (rather
    // than truncates) a request that outgrows it. Sizing it off `max_pages`
    // rather than `n + max_tokens` keeps it an exact multiple of the page
    // geometry the engine derives its own length from.
    let kv_max = max_pages * page_size;
    // The score row's width is the backend's, not the program's: a slab pitch
    // cannot be a per-program number, so `attn_score_kv_max()` publishes the
    // one that was carved and the program reads a prefix of it. Refusing here
    // is the honest failure — a truncated read would produce a plausible
    // ranking over positions that are not the ones it names.
    if kv_max > intrinsics::attn_score_kv_max() {
        return Err(format!(
            "prompt + reserve needs {kv_max} KV slots, past the published \
             attn_score ceiling of {}",
            intrinsics::attn_score_kv_max()
        )
        .into());
    }
    let p_max = max_pages;
    let page_budget = input.page_budget.min(p_max);
    let report = input.report;
    ws.reserve(max_pages).context("reserve KV")?;

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    // ── PREFILL (chunked, C-wide): no TOVA tap. ──
    //
    // TOVA is defined on the decoding step: it ranks by the attention of the
    // most recent query token, and during prefill "most recent" is still
    // moving. The paper applies it from the first generated token onward.
    //
    // Split into `ceil(n / C)` chunks, `C = max_embed_length()`. A one-shot
    // fire cannot exceed the engine's per-launch token capacity, which capped
    // this policy at 8192 prompt tokens; chunk `i` attends over the whole
    // prefix written so far and writes only its own tokens, so the
    // concatenation equals the one-shot fire (section 17).
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    // The split is `prefill_chunks` (SDK), which spreads the remainder over the
    // FIRST chunks so the last one is never a sliver. Every inferlet in this
    // tree uses it, which is what makes their chunk boundaries identical: a
    // text difference above the ceiling is then a policy difference, not a
    // difference in the attention tile decomposition (§11.4).
    let spans = prefill_chunks(n, input.prefill_chunk);
    let pipe = Pipeline::new();

    let mut g0 = 0i32;
    for &(base, end) in &spans {
        let len = end - base;

        let toks_p = Channel::from(&prompt_i32[base as usize..end as usize]).named("toks_p");
        let embed_indptr_p = Channel::from([0u32, len]).named("embed_indptr_p");
        let positions_p = Channel::from_iter(base..end).named("positions_p");
        let pages_p = Channel::from_iter(0..max_pages).named("pages_p");
        let page_indptr_p = Channel::from([0u32, end.div_ceil(page_size)]).named("page_indptr_p");
        let w_slot_p = Channel::from_iter((base..end).map(|p| p / page_size)).named("w_slot_p");
        let w_off_p = Channel::from_iter((base..end).map(|p| p % page_size)).named("w_off_p");
        let kv_len_p = Channel::from([end]).named("kv_len_p");
        let rng_p = Channel::from([input.seed, 0]).named("rng_p");
        let tok_out_p = Channel::new([1], dtype::i32).named("tok_out_p");

        let fwd_p = ForwardPass::new();
        fwd_p.embed(&toks_p, &embed_indptr_p)?;
        fwd_p.attention(
            &ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: &kv_len_p,
                pages: &pages_p,
                page_indptr: &page_indptr_p,
                w_slot: &w_slot_p,
                w_off: &w_off_p,
                positions: &positions_p,
                mask: None,
            },
        )?;
        fwd_p.epilogue(move || {
            let r = rng_p.take();
            let logits = intrinsics::logits();
            let token = step(logits, temperature, &r);
            tok_out_p.put(&token);
            rng_p.put(&(&r + iota(2)));
        });

        fwd_p
            .submit(&pipe)
            .with_context(|| format!("prefill submit @{base}"))?;

        g0 = tok_out_p
            .take_host::<i32>()
            .await
            .with_context(|| format!("@{base}"))?;
    }
    generated.push(g0 as u32);

    let mut last_scores: Vec<f32> = Vec::new();
    let mut last_page_mass: Vec<f32> = Vec::new();
    // Declared, not counted: `planes / heads` is the layer count, so the only
    // thing the host has to keep is how many fires it drained.
    let mut layers_observed = 0u32;
    let layers_per_fire = layers;
    let mut last_kv_len = n + 1;
    let mut trace: Vec<(u32, usize)> = Vec::new();
    let mut mass_trace: Vec<f32> = Vec::new();

    // ── THE SPLIT: OBSERVE, THEN ENFORCE. ──
    //
    // A lane that brings a mask takes the masked arm and KEEPS NO SCORES --
    // `Attention::Masked` exports no lse, and `crates/models/src/qwen_3/
    // forward.rs` orders `masked` ahead of `captures_scores` in the split for
    // that reason, as a correctness ruling. So no single fire can both observe
    // and enforce, and H2O -- alone in Track B -- wants both on the same fire.
    // The generation is therefore cut in two: `observe_fires` unmasked
    // capturing fires that accumulate the cumulative heavy-hitter statistic,
    // then `enforce_tokens` masked fires served the keep-set that statistic
    // chose. See the module header's deviation (4) for what that costs.
    let decode_budget = max_tokens - generated.len();
    let enforce_tokens = input
        .enforce_tokens
        .unwrap_or(decode_budget / 2)
        .min(decode_budget);
    let observe_fires = decode_budget - enforce_tokens;

    // ── OBSERVATION LOOP (1-wide, run-ahead), with the score tap. ──
    if observe_fires > 0 {
        let tok_in = Channel::from([g0]).named("tok_in");
        // Same salt as `naive-baseline`. The coherence test asks whether an
        // all-keep page mask changes what the model produces, and that question
        // is only answerable if every other input is held fixed -- including
        // the Gumbel stream. A different salt makes the two runs disagree for
        // reasons that have nothing to do with attention.
        let rng = Channel::from([input.seed ^ 0x5bd1, 0]).named("rng");
        let tok_out = Channel::new([1], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("tok_out");
        let lane1 = Channel::from([0u32, 1u32]).named("embed_indptr");
        let positions = Channel::from([n]).named("positions");
        let pages = Channel::from_iter(0..max_pages).named("pages");
        let page_indptr = Channel::from([0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from([n / page_size]).named("w_slot");
        let w_off = Channel::from([n % page_size]).named("w_off");
        let kv_len = Channel::from([n + 1]).named("kv_len");

        // The heavy-hitter accumulator. It carries ACROSS fires and is never
        // re-seeded: H2O's statistic is cumulative over the whole generation,
        // and that carry is the entire difference between this and
        // `tova-attention`, which carries nothing at all. (What is gone is the
        // per-LAYER fold this channel also used to perform — the rectangle now
        // arrives whole and that fold is an in-graph reduction.)
        //
        // Contract #1: a loop-carried channel must be `take`n before it is
        // `put`, or the dummy run that infers the geometry sees a port that is
        // written but never read.
        //
        // The seed is a descending ramp, not zeros -- see the module docs. Its
        // largest entry is `SEED_SCALE`, four orders of magnitude below the
        // 1.0 a single layer-fire contributes, so it decides ties on the first
        // fire and is irrelevant afterwards.
        let seed: Vec<f32> = (0..kv_max)
            .map(|i| SEED_SCALE * (kv_max - i) as f32 / kv_max as f32)
            .collect();
        let acc = Channel::from(seed).named("h2o_acc");

        // Host drain. Absent entirely when `report` is off -- not merely
        // undrained -- so the epilogue never writes it and the geometry pass
        // never sees a port. `acc` stays unconditional: its fold feeds
        // `page_mass_epi`, which IS the policy.
        let scores_out = report.then(|| {
            Channel::new([kv_max], dtype::f32)
                .capacity(channel_capacity() as u32)
                .named("h2o_scores")
        });

        // The page fold, drained EVERY fire. It is the policy's own decision
        // row -- the accumulator folded from positions into pages -- and it is
        // `[p_max]` f32, tens of BYTES rather than the tens of kilobytes
        // `report` gates, so it crosses unconditionally: the keep-set the
        // enforcement arm serves is chosen from the last one drained.
        let page_mass_out = Channel::new([p_max], dtype::f32)
            .capacity(channel_capacity() as u32)
            .named("h2o_page_mass");

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &lane1)?;
        fwd.attention(
            &ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: (n / page_size)..,
                kv_len: &kv_len,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &positions,
                mask: None,
            },
        )?;

        fwd.epilogue(move || {
            let length = kv_len.take();
            let r = rng.take();
            let logits = intrinsics::logits();
            let token = step(logits, temperature, &r);

            let next_length = &length + 1u32;
            let page_count = next_length.div_ceil(page_size);

            tok_in.put(&token);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(&length / page_size);
            w_off.put(&length % page_size);
            page_indptr.put(indptr(1, &page_count));
            tok_out.put(&token);
            rng.put(&(&r + iota(2)));

            // ── THE TAP. Once per fire, at the epilogue, over the whole
            //    rectangle: `[planes, attn_score_kv_max()]`, layer-major.
            //
            //    `transpose` puts the planes on the last axis so `reduce_sum`
            //    — which reduces the LAST axis — sums down them; `/ heads`
            //    turns that plane-sum into (mean over heads, then sum over
            //    layers), since `Σ_l (1/H) Σ_h row = (1/H) Σ_planes row`. This
            //    fire therefore contributes mass exactly `layers`, which is the
            //    staircase step `mass_trace` asserts.
            //
            //    The `gather` narrows the published width to the program's own
            //    page geometry — `kv_max = max_pages * page_size` — so the
            //    accumulator, the page fold and the drained row all describe
            //    the prefix this program reserved and nothing past it.
            //
            //    Contract #1: `acc` is `take`n before it is `put`, or the dummy
            //    run that infers the geometry sees a port that is written but
            //    never read.
            let rect = intrinsics::attn_score(planes);
            let fire_row = &reduce_sum(&transpose(&rect)) / heads_f;
            let prev = acc.take();
            // CARRY IT FORWARD. This is the one line that separates H2O from
            // TOVA: `tova-attention` publishes this fire's row and keeps
            // nothing; H2O adds it to every fire before it, which is what makes
            // `score_mass == layers_observed` stay true as both grow.
            let folded = &prev + &gather(&fire_row, iota(kv_max));
            if let Some(c) = scores_out.as_ref() {
                c.put(&folded);
            }
            acc.put(&folded);

            // Fold positions into pages for the keep-set. `reduce_sum` reduces
            // the last axis per row, and `kv_max = p_max * page_size` exactly
            // (it was derived from the page geometry), so the reshape is a
            // reinterpretation rather than a resize. Summing is the right
            // collapse for a probability mass: a page's share of the attention
            // is the sum of its positions' shares.
            page_mass_out.put(&reduce_sum(&reshape(&folded, [p_max, page_size])));
        });

        run_ahead(&pipe, &fwd, observe_fires, async || {
            let t = tok_out
                .take_host::<i32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            last_page_mass = page_mass_out
                .take_host::<Vec<f32>>()
                .await
                .with_context(|| format!("page mass @{}", generated.len()))?;
            if let Some(sc) = scores_out.as_ref() {
                last_scores = sc
                    .take_host::<Vec<f32>>()
                    .await
                    .with_context(|| format!("@{}", generated.len()))?;
                // The fire that produced this row had `n + generated.len()` KV
                // positions live: the prompt plus every token committed before it.
                last_kv_len = n + generated.len() as u32;
                // One more fire folded in, `layers_per_fire` layers each. The
                // counter is cumulative for the same reason `acc` is.
                layers_observed += layers_per_fire;
                mass_trace.push(last_scores.iter().filter(|s| s.is_finite()).sum());
                trace.push((
                    last_kv_len,
                    last_scores
                        .iter()
                        .rposition(|s| *s != 0.0)
                        .map_or(0, |i| i + 1),
                ));
            }
            generated.push(t as u32);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    }
    // The observation stream will accept no more submissions. `run_ahead`
    // already said so when it spent its budget ("call close right after the
    // last submit"); this covers `observe_fires == 0`, where it never ran.
    pipe.close();

    // ── THE KEEP-SET: H2O's heavy hitters, chosen once the observation phase
    //    is over, off the device's own page fold of the cumulative
    //    accumulator. Ranking on the host is the same departure
    //    `snapkv-eviction` makes and for the same reason: the set is reported
    //    so the harness can check it, and the device-side top-k a serving
    //    program would use is one `rank_le` away.
    //
    //    Ties go to the EARLIER page, which is the seed ramp's own order (the
    //    ramp is inside `last_page_mass`, folded in with everything else) and
    //    the StreamingLLM prior.
    let sel_len = n + generated.len() as u32 - 1;
    let sel_pages = sel_len.div_ceil(page_size).min(p_max);
    let mut sel_order: Vec<u32> = (0..sel_pages).collect();
    let mass_of = |p: u32| last_page_mass.get(p as usize).copied().unwrap_or(0.0);
    sel_order.sort_by(|&a, &b| {
        mass_of(b)
            .partial_cmp(&mass_of(a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let evicted_first = sel_order.last().copied();
    let mut kept: Vec<u32> = sel_order
        .into_iter()
        .take(page_budget as usize)
        .collect();
    kept.sort_unstable();

    // One bit per KV slot the pool can address. A position inside a kept page
    // survives; a position past `sel_len` is kept unconditionally, because
    // those are the tokens the enforcement arm is about to write and a policy
    // that evicted its own output would be evicting the future. That second
    // clause IS H2O's local window -- it is what a backend force-keep of the
    // request's last page would provide, stated by the guest and at position
    // rather than page granularity.
    let keep_set: std::collections::BTreeSet<u32> = kept.iter().copied().collect();
    let keep_row: Vec<bool> = (0..kv_max)
        .map(|j| j >= sel_len || keep_set.contains(&(j / page_size)))
        .collect();
    let served_kv = keep_row[..(sel_len as usize).min(keep_row.len())]
        .iter()
        .filter(|k| **k)
        .count() as u32;

    // ── ENFORCEMENT ARM (1-wide, run-ahead), MASKED and therefore silent. ──
    //
    // A SECOND PIPELINE, and the host barrier is what makes that sound. The
    // two arms are sequential, not concurrent, and `Pipeline`'s own rule is
    // that sequential phases share one — but `run_ahead` ends its stream when
    // it spends its budget (it must: a lane that will not submit again is
    // holding the fleet's frame seal for nothing), so the observation loop's
    // pipeline is closed by the time the keep-set exists. The ordering the
    // shared pipeline would have provided is provided instead by the drain
    // above: every observation fire has been taken, so every KV write this arm
    // reads has settled. `prefix-tree-kv-cache` composes its leaves the same
    // way, for the same reason.
    if enforce_tokens > 0 && generated.len() < max_tokens {
        let pipe = Pipeline::new();
        let last = *generated.last().expect("prefill emitted a token") as i32;
        let at = sel_len;
        let tok_in = Channel::from([last]).named("e_tok_in");
        // The Gumbel stream picks up exactly where the observation loop left
        // it: each fire advanced `[s, k]` to `[s, k + 1]` (`r + iota(2)`), so
        // after `observe_fires` fires the state is `[s, observe_fires]`. Said
        // host-side because the two arms are separate ETA programs; getting
        // this wrong would make the coherence comparison against
        // `naive-baseline` measure the sampler instead of the mask.
        let rng = Channel::from([input.seed ^ 0x5bd1, observe_fires as u32]).named("e_rng");
        let tok_out = Channel::new([1], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("e_tok_out");
        let lane1 = Channel::from([0u32, 1u32]).named("e_embed_indptr");
        let positions = Channel::from([at]).named("e_positions");
        let pages = Channel::from_iter(0..max_pages).named("e_pages");
        // Republished every fire though its value never moves: a channel-bound
        // dense `AttnMask` declines the decode-envelope class and lands this
        // trace in the pool-owned device-geometry class, which requires every
        // geometry port's channel to be written by a stage
        // (`lease::detect_pooled_device_geometry`).
        let pages_src = Channel::from_iter(0..max_pages).named("e_pages_src");
        let page_indptr = Channel::from([0u32, (at + 1).div_ceil(page_size)]).named("e_page_indptr");
        let w_slot = Channel::from([at / page_size]).named("e_w_slot");
        let w_off = Channel::from([at % page_size]).named("e_w_off");
        let kv_len = Channel::from([at + 1]).named("e_kv_len");

        // `keep` is the frozen heavy-hitter set; `mask` is what the fire
        // carries -- the keep-set ANDed with THIS step's causal row, rebuilt on
        // the device every step because the causal half moves and the keep half
        // does not. ANDed and not substituted: a bound mask REPLACES the
        // derived causal bound (`Port::AttnMask`), so a row carrying only the
        // keep-set would let the query attend to slots that do not exist yet.
        let keep = Channel::from_shaped([1, kv_max], keep_row.clone()).named("e_keep");
        let mask = Channel::from_shaped(
            [1, kv_max],
            keep_row
                .iter()
                .enumerate()
                .map(|(j, k)| *k && (j as u32) <= at)
                .collect::<Vec<bool>>(),
        )
        .named("e_mask");

        let fwd = ForwardPass::new();
        fwd.embed(&tok_in, &lane1)?;
        fwd.attention(
            &ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: (n / page_size)..,
                kv_len: &kv_len,
                pages: &pages,
                page_indptr: &page_indptr,
                w_slot: &w_slot,
                w_off: &w_off,
                positions: &positions,
                mask: Some(&mask),
            },
        )?;
        fwd.epilogue(move || {
            let length = kv_len.take();
            let r = rng.take();
            let logits = intrinsics::logits();
            let token = step(logits, temperature, &r);

            let next_length = &length + 1u32;
            let page_count = next_length.div_ceil(page_size);

            tok_in.put(&token);
            kv_len.put(&next_length);
            positions.put(&length);
            w_slot.put(&length / page_size);
            w_off.put(&length % page_size);
            page_indptr.put(indptr(1, &page_count));
            tok_out.put(&token);
            rng.put(&(&r + iota(2)));

            // The next fire's served row: `length` is that fire's own query
            // position, so `causal_mask` cuts exactly the slots that exist for
            // it and the frozen keep-set cuts the ones H2O dropped.
            //
            // Contract #1: `keep` is `take`n before it is `put`.
            let k = keep.take();
            mask.put(&and(&causal_mask(&length, kv_max), &k));
            keep.put(&k);

            let ids = pages_src.take();
            pages.put(&ids);
            pages_src.put(&ids);
        });

        run_ahead(&pipe, &fwd, enforce_tokens, async || {
            let t = tok_out
                .take_host::<i32>()
                .await
                .with_context(|| format!("enforce @{}", generated.len()))?;
            generated.push(t as u32);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    }

    let live = (last_kv_len as usize).min(last_scores.len());
    let live_pages = last_kv_len.div_ceil(page_size).min(p_max);

    // The host-side mirror of the fold the device did: the drained position
    // scores collapsed into page masses. Recomputed rather than taken on trust
    // from `last_page_mass`, so a disagreement between the two folds is visible
    // in `Output` instead of silently deciding the keep-set.
    //
    // `live_pages` is derived from `kv_len`, which is known whether or not the
    // scores were drained, so it can outrun `last_scores` when `report` is off
    // and the row was never fetched. Clamp BOTH ends against the row actually
    // in hand rather than only `hi`: a page whose whole span sits past the end
    // must yield an empty slice, not a backwards one.
    let masses: Vec<f32> = (0..live_pages as usize)
        .map(|p| {
            let lo = (p * page_size as usize).min(last_scores.len());
            let hi = ((p + 1) * page_size as usize).min(last_scores.len());
            last_scores[lo..hi].iter().sum()
        })
        .collect();
    Ok(Output {
        sampler: "trackb-h2o",
        text: model::decode(&generated)?,
        count: generated.len(),
        kv_max,
        page_size,
        kv_len: last_kv_len,
        page_budget,
        live_pages,
        layers_observed,
        layers_per_fire,
        live_scored: last_scores[..live]
            .iter()
            .filter(|s| s.is_finite() && **s >= 0.0)
            .count(),
        tail_polluted: last_scores[live..]
            .iter()
            .filter(|s| **s > SEED_CEILING)
            .count(),
        tail_max: last_scores[live..].iter().copied().fold(0.0f32, f32::max),
        observed_live: last_scores
            .iter()
            .rposition(|s| *s != 0.0)
            .map_or(0, |i| i + 1),
        scores_nan: last_scores.iter().filter(|s| s.is_nan()).count(),
        score_mass: last_scores[..live].iter().sum(),
        mass_trace,
        observe_fires,
        enforce_tokens,
        selected_pages: sel_pages,
        served_pages: kept.len() as u32,
        served_kv,
        full_kv: sel_len,
        kept_pages: kept,
        evicted_first,
        page_mass: masses.iter().map(|m| format!("{m:.5}")).collect(),
        trace,
        score_head: last_scores[..live.min(8)]
            .iter()
            .map(|s| format!("{s:.5}"))
            .collect(),
    })
}
