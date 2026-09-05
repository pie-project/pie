//! SnapKV — prompt compression from the observation window (Li et al., 2024).
//!
//! <https://arxiv.org/abs/2404.14469>
//!
//! ## What makes SnapKV structurally different from H2O and TOVA
//!
//! H2O and TOVA are *decode-time* policies: they watch the attention of each
//! generated token and continuously re-rank the cache. SnapKV is a *prefill-time*
//! policy. It looks at the attention that the **tail of the prompt** paid to the
//! rest of the prompt, selects a keep-set once, and then holds it for the whole
//! generation. Its premise is that the last few prompt tokens are the best
//! available proxy for what the not-yet-generated continuation will want.
//!
//! That premise makes SnapKV the only member of Track B that reads the score
//! rectangle at a **prefill** epilogue rather than a decode one.
//!
//! ## The observation window is the BACKEND's, and SnapKV no longer carries one
//!
//! The capture arm folds the last `min(32, qo_len)` query rows of each request
//! into the row it publishes — 32 is the backend's statute, and it is SnapKV's
//! own width (`PIE_ATTN_SCORE_WINDOW` in the C++ lineage). So the row this
//! program reads at the prefill epilogue ALREADY IS the end-of-prompt
//! observation: there is no window channel to declare, no window fold to write,
//! and nothing for a guest to get wrong about which rows were averaged. TOVA
//! and H2O run 1-row decode fires, where that same statute degenerates to a
//! window of one — the current query's distribution, which is what those two
//! papers define. One backend rule serves all three.
//!
//! This program is still the end-to-end test of a path nothing else exercises:
//! a multi-row capture, its causal-aware normalisation (a prefill row must not
//! be scaled by query rows the causal mask discarded), and the layer/head fold
//! over a rectangle whose live prefix is the whole prompt rather than one
//! decode step's cache.
//!
//! ## What it enforces, and how the budget is spent
//!
//! One selection, made after prefill, then fixed. Two groups of positions are
//! kept:
//!
//!  1. Every position in the top `page_budget` **prompt** pages by observed
//!     attention mass.
//!  2. **Every position written after prefill**, unconditionally.
//!
//! (2) is not a hedge, it is SnapKV: the paper compresses the prompt and then
//! lets the cache grow normally from there. Evicting freshly generated tokens
//! would be a different (and worse) algorithm.
//!
//! ## The door the eviction goes through, and why it is the mask
//!
//! `.wiki/alto/attn-score.md` §4 already answered this, and the answer is the
//! first of the two honest deltas it records against the papers: **"evict"
//! executes as CUSTOM MASK updates (the masked arm), with a page drop only
//! when a page's tokens are all dead.** So the decode fire binds an `AttnMask`
//! — the descriptor port `snapkv-eviction` and `sliding-window-attention`
//! already ride — and the keep-set is a row of bits ANDed into the causal row.
//! Nothing here is a second-party intrinsic; nothing here is engine surface
//! this program invented.
//!
//! **AND THE MASK IS THE DOOR FOR A REASON, NOT FOR CONVENIENCE.** The other
//! route — naming fewer pages in the page list — makes the KV cache shorter
//! without making the SEQUENCE shorter, and from that instant the cache index
//! and the true position are two different numbers. A kept key carries the
//! RoPE it was written with, at its true position, so the query must carry its
//! true position too or every relative distance in the attention is wrong by
//! however much was evicted. The cuda shell derives a lane's positions as
//! `held .. held + rows` and refuses an explicit list by name (`Unsupported`,
//! verb `explicit lane positions`), and `WorkingSet::discard` says outright
//! that "suffix indexes shift down" — so a page-list cut would have to
//! re-index the query, which is StreamingLLM's technique and NOT SnapKV's.
//!
//! Under the mask the geometry does not move at all: the same page list, the
//! same `kv_len`, the same positions, and the only thing that changes is which
//! keys the softmax is allowed to see. Quality semantics exact; memory savings
//! quantized to page granularity and taken later, which is precisely the delta
//! §4 wrote down.
//!
//! **A masked lane keeps no scores**, and that costs this program nothing:
//! SnapKV observes at the PREFILL epilogue (unmasked, capturing) and enforces
//! over the decode (masked, not capturing), so the two never want the same
//! fire. That is the structural reason SnapKV is the member of Track B that
//! rides this door without a compromise — see `trackb-h2o` for the one that
//! does not.
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
//! `Stage::OnAttn` no longer admits the intrinsic at all. Model-gated on
//! `has_attn_score`.
//!
//! Both folds SnapKV needs are ONE in-graph reduction at the epilogue, on the
//! device (§4: "reduction stays on device — only decisions cross to the host"):
//!
//! ```text
//! folded = reduce_sum(transpose(rect)) / heads
//! ```
//!
//! `transpose` puts the planes on the last axis so `reduce_sum` — which
//! reduces the LAST axis — sums down them; `/ heads` turns that plane-sum into
//! (mean over heads, then sum over layers), because
//! `Σ_l (1/H) Σ_h row = (1/H) Σ_planes row`. The result is a row of mass
//! exactly `layers`, which is what `score_mass` checks. **The layer count is
//! declared, not counted**: the per-layer accumulator and the device layer
//! counter both existed to fold a per-layer tap across the layer loop, and
//! there is no layer loop left. `layers_observed` is `planes / heads`.
//!
//! ## Deviations from the paper, as built
//!
//! 1. **Pooling is page-granular and non-overlapping.** SnapKV max-pools the
//!    position scores with kernel 7, stride 1 before top-k, so that a selected
//!    position drags its neighbours in and the kept set is contiguous rather
//!    than speckled. Here the pool is the page fold: kernel = stride =
//!    `page_size`. That is not an approximation of SnapKV's pooling, it is the
//!    *enforceable* version of it — a paged KV cache only returns memory a page
//!    at a time, so a sub-page keep-set cannot free anything. The clustering
//!    effect SnapKV's pooling exists to produce is what page granularity gives
//!    for free.
//! 2. **Heads are folded by the PROGRAM.** SnapKV selects per head; one page
//!    list per request means a per-head keep-set has no representable consumer.
//!    The rectangle is per-head because observability wants it that way (§4),
//!    so this program takes the mean itself, in-graph. `quest-attention`,
//!    `tova-attention` and `trackb-h2o` document the identical collapse.
//! 3. **Layers are folded by the program.** Same reason. Summing layers is
//!    monotone-equivalent to averaging, so the ranking does not depend on how
//!    many layers were declared.
//!
//! ## Why the page mass takes a host round trip
//!
//! The selection is made in the prefill program and enforced in the decode
//! program, and those are two separately submitted ETA programs. The
//! established idiom for carrying a value between them is a host drain and a
//! fresh channel — `attention-sink` carries its first token that way. The
//! *fold* still happens on device, at the prefill epilogue, from a device
//! tensor; only the one-time hand-off and the rank are on the host, which is
//! also what makes the selection independently checkable in `Output`. §4 asks
//! that "only decisions cross to the host"; a `[p_max]` row crossing once per
//! request, never per step, is the smallest hand-off this two-program shape
//! admits, and `snapkv-eviction` makes the same trade for the same reason.

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
    /// Prompt KV pages SnapKV keeps. Positions written after prefill are kept
    /// on top of this, unconditionally — including the tail of the last prompt
    /// page, which is where the decode writes.
    #[serde(default = "default_page_budget")]
    page_budget: u32,
    /// Pins the reserved page count so it does not move with `max_tokens`.
    /// `p_max` sets the width of the score row folded at the prefill epilogue
    /// and the width of the keep row every decode fire carries, so a benchmark
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
    0x5A4Bu32
}
fn default_page_budget() -> u32 {
    4
}
/// qwen35-d0.8b: 24 layers, `attn_every: 4` → 6 exported attention layers.
fn default_layers() -> u32 {
    6
}
/// qwen35-d0.8b: `q_heads: 8` at `tp == 1`.
fn default_heads() -> u32 {
    8
}

/// Largest entry of the tie-break ramp given to prompt pages before anything has
/// been observed. Only reachable if the capture returns an all-zero row, which
/// would itself be a bug -- but a tie among all pages would then evict real
/// context arbitrarily, so the ramp degrades that to "keep the earliest pages",
/// the StreamingLLM prior.
const SEED_SCALE: f32 = 1e-4;

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
    /// Prompt length in tokens, i.e. the KV length the observed fire had.
    prompt_len: u32,
    page_budget: u32,
    /// Pages the prompt occupies. Only these compete for the budget.
    prompt_pages: u32,
    /// How many fires the prefill was split into, and how many tokens the
    /// FINAL (observed) one carried. Reported because the backend's window is
    /// the last `min(32, qo_len)` rows *of that fire*: if `prefill_final` drops
    /// below 32 the observation narrows with it, and a narrow window still
    /// produces a plausible-looking ranking. With even chunking
    /// `prefill_final == prompt_len / prefill_chunks`, the largest a last chunk
    /// can be.
    prefill_chunks: u32,
    prefill_final: u32,
    /// KV page size, reported so a consumer can convert the backend's
    /// observation window (a token count, `min(32, qo_len)`) into the page span
    /// it covers.
    page_size: u32,
    /// Attention layers folded into the observed row. DERIVED FROM THE
    /// DECLARED SHAPE (`planes / heads`), not counted by a device channel:
    /// there is no per-layer tap left to count, and the rectangle arrives
    /// whole. It is what `score_mass` must equal.
    layers_observed: u32,
    /// Slots inside the prompt that carry a finite, non-negative score.
    live_scored: usize,
    /// Slots past the prompt carrying anything at all. Unlike H2O there is no
    /// seed ramp in the captured row, so this must be exactly zero: the prefill
    /// capture writes one entry per LIVE kv position and nothing else.
    tail_nonzero: usize,
    /// One past the highest slot carrying attention mass, i.e. the prompt
    /// length as the ENGINE saw it. Reported next to `prompt_len` because a
    /// disagreement between the two means the row describes different positions
    /// than the program thinks it does.
    observed_live: usize,
    scores_nan: usize,
    /// `Σ score` over the prompt. The mean over heads is one distribution per
    /// layer -- the backend already averaged the observation window's rows, and
    /// nothing averages over layers -- so the layer-sum must come out at
    /// `layers_observed`.
    score_mass: f32,
    /// SnapKV's keep-set among the PROMPT pages: the `page_budget` highest-mass
    /// ones, ranked off the device's own page fold. This is the set the served
    /// mask was built from, not a second opinion about it.
    kept_pages: Vec<u32>,
    /// The prompt page SnapKV would drop first (lowest mass).
    evicted_first: Option<u32>,
    /// Pages the prompt occupies that the served mask lets through. The
    /// all-keep arm reports this equal to `prompt_pages`; **this is the shrink**
    /// and it is a property of the row that flew, not of the policy that
    /// planned it.
    served_pages: u32,
    /// Prompt KV positions the served mask admits, against the number an
    /// unmasked decode attends over. THE MASK IS THE MEASUREMENT: a count of
    /// set bits in the row the fire actually carried.
    served_kv: u32,
    full_kv: u32,
    /// Per-page mass over the prompt.
    page_mass: Vec<String>,
    /// The mass of the page holding the END of the prompt, as a fraction of the
    /// total. SnapKV's whole premise is that the observation window attends
    /// heavily to nearby positions, so this should be a large share -- and if it
    /// were not, the capture would not be describing the window it claims to.
    tail_page_share: f32,
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
    // Like Quest's `p_max` and H2O's, the program declares a static ceiling for
    // the score row; the backend refuses (rather than truncates) a request that
    // outgrows it. Sized off `max_pages` so it is an exact multiple of the page
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
    let prompt_pages = n.div_ceil(page_size).min(p_max);
    let page_budget = input.page_budget.min(prompt_pages);
    ws.reserve(max_pages).context("reserve KV")?;

    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

    // ── PREFILL (chunked), WITH the tap on the FINAL chunk. ──
    //
    // This is the inversion that defines SnapKV: every other Track B policy
    // taps the decode fire and leaves prefill alone. Here prefill is the only
    // fire that is observed at all -- which is exactly what makes chunking it
    // delicate, and why the tap is attached where it is.
    //
    // The capture folds the last `min(32, qo_len)` query rows OF THE FIRE into
    // the row it publishes. For a one-shot prefill that is the last 32 tokens
    // of the prompt, which is SnapKV's definition -- so the program declares no
    // window of its own and does no window fold: the published row already IS
    // the end-of-prompt observation. Under chunking only the FINAL chunk's
    // window is the prompt's tail; an earlier chunk's window is a well-defined
    // observation of an earlier position, but it is not the quantity SnapKV
    // selects on. So the tap goes on the final chunk alone, and the earlier
    // chunks run the plain prefill -- which also means they do not pay the
    // capture variant's cost.
    //
    // The split is `prefill_chunks`, whose docs explain why it spreads the
    // remainder over the FIRST chunks: a "C tokens at a time until the
    // remainder" split can leave a final chunk of one token, and a final chunk
    // shorter than `window` silently truncates SnapKV's observation to whatever
    // was left over. That produces a plausible ranking, which is worse than an
    // implausible one. Using the shared helper also guarantees this inferlet
    // picks the SAME chunk boundaries as the baseline it is compared against,
    // so a text difference above the ceiling means a policy difference and not
    // a difference in the attention tile decomposition (§11.4).
    let prompt_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    let spans = prefill_chunks(n, input.prefill_chunk);
    let k = spans.len() as u32;
    let pipe = Pipeline::new();

    // Non-final chunks: plain prefill, no tap, sampled token discarded.
    for &(base, end) in &spans[..spans.len() - 1] {
        let chunk = end - base;
        let toks_c = Channel::from(&prompt_i32[base as usize..end as usize]).named("toks_c");
        let embed_indptr_c = Channel::from([0u32, chunk]).named("embed_indptr_c");
        let positions_c = Channel::from_iter(base..end).named("positions_c");
        let pages_c = Channel::from_iter(0..max_pages).named("pages_c");
        let page_indptr_c = Channel::from([0u32, end.div_ceil(page_size)]).named("page_indptr_c");
        let w_slot_c = Channel::from_iter((base..end).map(|p| p / page_size)).named("w_slot_c");
        let w_off_c = Channel::from_iter((base..end).map(|p| p % page_size)).named("w_off_c");
        let kv_len_c = Channel::from([end]).named("kv_len_c");
        let rng_c = Channel::from([input.seed ^ 0x5bd1, 0]).named("rng_c");
        let tok_out_c = Channel::new([1], dtype::i32).named("tok_out_c");

        let fwd_c = ForwardPass::new();
        fwd_c.embed(&toks_c, &embed_indptr_c)?;
        fwd_c.attention(
            &ws,
            KvGeometry {
                readable_pages: ..,
                writable_pages: ..,
                kv_len: &kv_len_c,
                pages: &pages_c,
                page_indptr: &page_indptr_c,
                w_slot: &w_slot_c,
                w_off: &w_off_c,
                positions: &positions_c,
                mask: None,
            },
        )?;
        fwd_c.epilogue(move || {
            let r = rng_c.take();
            let logits = intrinsics::logits();
            let token = step(logits, temperature, &r);
            tok_out_c.put(&token);
            rng_c.put(&(&r + iota(2)));
        });
        fwd_c
            .submit(&pipe)
            .with_context(|| format!("prefill chunk submit @{base}"))?;
        tok_out_c
            .take_host::<Vec<i32>>()
            .await
            .with_context(|| format!("prefill chunk take @{base}"))?;
    }

    // ── FINAL CHUNK `[base, n)`: the observed one. ──
    let base = spans[spans.len() - 1].0;
    let tail = n - base;
    let toks_p = Channel::from(&prompt_i32[base as usize..n as usize]).named("toks_p");
    let embed_indptr_p = Channel::from([0u32, tail]).named("embed_indptr_p");
    let positions_p = Channel::from_iter(base..n).named("positions_p");
    let pages_p = Channel::from_iter(0..max_pages).named("pages_p");
    let page_indptr_p = Channel::from([0u32, n.div_ceil(page_size)]).named("page_indptr_p");
    let w_slot_p = Channel::from_iter((base..n).map(|p| p / page_size)).named("w_slot_p");
    let w_off_p = Channel::from_iter((base..n).map(|p| p % page_size)).named("w_off_p");
    let kv_len_p = Channel::from([n]).named("kv_len_p");
    // Same salt as `naive-baseline`. The coherence test asks whether a keep-set
    // that keeps everything changes what the model produces, and that question
    // is only answerable if every other input is held fixed -- including the
    // Gumbel stream.
    let rng_p = Channel::from([input.seed ^ 0x5bd1, 0]).named("rng_p");
    let tok_out_p = Channel::new([1], dtype::i32).named("tok_out_p");

    // NO ACCUMULATOR, AND NO LAYER COUNTER. Both were device-carried channels
    // that existed only to fold a per-layer tap across the layer loop, and
    // there is no layer loop: the epilogue is handed every exported layer's
    // rows at once, and SnapKV observes exactly one fire anyway. The layer
    // count that used to ride `snapkv_layers` back to the host is
    // `planes / heads`, a number the program declared.
    let scores_out = Channel::new([kv_max], dtype::f32).named("snapkv_scores");
    // The device's own fold, drained alongside the position row so the host can
    // check the two against each other rather than trusting either.
    let page_mass_out = Channel::new([p_max], dtype::f32).named("snapkv_page_mass");

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

        // ── THE TAP. Once, at this prefill fire's epilogue, over the whole
        //    rectangle: `[planes, attn_score_kv_max()]`, layer-major. Because
        //    this is a prefill fire the backend folded the last
        //    `min(32, prefill_final)` query rows into every row of it, so what
        //    arrives here is already SnapKV's end-of-prompt observation.
        //
        //    `transpose` puts the planes on the last axis so `reduce_sum` --
        //    which reduces the LAST axis -- sums down them; `/ heads` turns
        //    that plane-sum into (mean over heads, then sum over layers), since
        //    `Σ_l (1/H) Σ_h row = (1/H) Σ_planes row`. The result is a row of
        //    mass exactly `layers`.
        //
        //    The `gather` narrows the published width to the program's own page
        //    geometry -- `kv_max = max_pages * page_size` -- so the drained row
        //    and the page fold describe the prefix this program reserved and
        //    nothing past it.
        let rect = intrinsics::attn_score(planes);
        let folded = gather(&(&reduce_sum(&transpose(&rect)) / heads_f), iota(kv_max));
        scores_out.put(&folded);

        // Fold positions into pages. `reduce_sum` reduces the last axis per row
        // and `kv_max = p_max * page_size` exactly (it was derived from the page
        // geometry), so the reshape is a reinterpretation rather than a resize.
        // Summing is the right collapse for a probability mass: a page's share
        // of the attention is the sum of its positions' shares.
        page_mass_out.put(&reduce_sum(&reshape(&folded, [p_max, page_size])));
    });

    fwd_p
        .submit(&pipe)
        .with_context(|| format!("prefill submit @{base}"))?;

    let g0 = tok_out_p.take_host::<i32>().await?;
    generated.push(g0 as u32);

    let prefill_scores = scores_out.take_host::<Vec<f32>>().await?;
    // Declared, not counted: `planes / heads`.
    let layers_observed = layers;
    let device_page_mass = page_mass_out.take_host::<Vec<f32>>().await?;

    // ── THE CUT, CHOSEN ONCE, AT THE PREFILL EPILOGUE. ──
    //
    // SnapKV selects when prefill ends and holds the selection for the whole
    // generation, so this runs here and never again. It is host arithmetic
    // over the DEVICE's own page fold -- the fold stayed on device (§4), and
    // the only thing that crossed is the decision.
    //
    // The ramp is the tie-break and nothing else: two pages at equal observed
    // mass go to the earlier one, which is the StreamingLLM prior, and it sits
    // four orders of magnitude below the mass a single layer contributes.
    let ranked = |p: u32| -> f32 {
        device_page_mass.get(p as usize).copied().unwrap_or(0.0)
            + SEED_SCALE * (prompt_pages - p) as f32 / prompt_pages.max(1) as f32
    };
    let mut order: Vec<u32> = (0..prompt_pages).collect();
    order.sort_by(|&a, &b| {
        ranked(b)
            .partial_cmp(&ranked(a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let evicted_first = order.last().copied();
    let mut kept: Vec<u32> = order.into_iter().take(page_budget as usize).collect();
    kept.sort_unstable();

    // ── THE KEEP ROW: one bit per KV slot the pool can address, `true` where
    //    the softmax may look.
    //
    //    Positions inside a kept prompt page survive; positions inside a
    //    dropped one do not; **every position at or past the prompt is kept
    //    unconditionally**, and that is SnapKV rather than a hedge -- the paper
    //    compresses the PROMPT and then lets the cache grow normally from
    //    there. Evicting freshly generated tokens would be a different (and
    //    worse) algorithm. It used to be spelled as a mass far above anything a
    //    prompt page could reach plus a widened rank predicate; a row of bits
    //    says it directly.
    let keep_set: std::collections::BTreeSet<u32> = kept.iter().copied().collect();
    let keep_row: Vec<bool> = (0..kv_max)
        .map(|j| j >= n || keep_set.contains(&(j / page_size)))
        .collect();
    // What the served row admits OF THE PROMPT, counted from the bits the fire
    // actually carries rather than multiplied out of the page count -- so the
    // number in `Output` is a measurement of the mask, not a restatement of the
    // policy that built it.
    let served_kv = keep_row[..n as usize].iter().filter(|k| **k).count() as u32;

    // ── DECODE LOOP (1-wide, run-ahead), enforcing the fixed keep-set. ──
    if generated.len() < max_tokens {
        let tok_in = Channel::from([g0]).named("tok_in");
        let rng = Channel::from([input.seed ^ 0x5bd1, 2]).named("rng");
        let tok_out = Channel::new([1], dtype::i32)
            .capacity(channel_capacity() as u32)
            .named("tok_out");
        let lane1 = Channel::from([0u32, 1u32]).named("embed_indptr");
        let positions = Channel::from([n]).named("positions");
        let pages = Channel::from_iter(0..max_pages).named("pages");
        // **THE PAGE LIST IS RE-PUT EVERY FIRE, AND THAT IS THE MASK'S DOING.**
        // A channel-bound dense `AttnMask` declines the decode-envelope class
        // and lands the trace in the pool-owned device-geometry class
        // (`lease::detect_pooled_device_geometry`), which requires EVERY
        // geometry port's channel to be republished by a stage -- `pages`
        // included, though its value never changes. It is sourced from a
        // second channel rather than taken from `pages` itself for the reason
        // `sliding-window-attention` does the same: `pages` is a port channel
        // whose committed cell the fire reads, and this one only ever writes
        // it.
        let pages_src = Channel::from_iter(0..max_pages).named("pages_src");
        let page_indptr = Channel::from([0u32, (n + 1).div_ceil(page_size)]).named("page_indptr");
        let w_slot = Channel::from([n / page_size]).named("w_slot");
        let w_off = Channel::from([n % page_size]).named("w_off");
        let kv_len = Channel::from([n + 1]).named("kv_len");

        // ── THE ENFORCEMENT, AS TWO CHANNELS. `keep` is the frozen keep-set,
        //    loop-carried and never rewritten -- SnapKV selects once. `mask` is
        //    what the fire actually carries: the keep-set ANDed with THIS
        //    step's causal row, rebuilt on the device every step because the
        //    causal half moves and the keep half does not.
        //
        //    **ANDed, not substituted.** A bound mask REPLACES the derived
        //    causal bound (`Port::AttnMask`'s own doc), so a row that carried
        //    only the keep-set would let the query attend to slots that do not
        //    exist yet. `sliding-window-attention` evolves its row for the same
        //    reason and by the same means.
        let keep = Channel::from_shaped([1, kv_max], keep_row.clone()).named("snapkv_keep");
        let mask = Channel::from_shaped(
            [1, kv_max],
            keep_row
                .iter()
                .enumerate()
                .map(|(j, k)| *k && (j as u32) <= n)
                .collect::<Vec<bool>>(),
        )
        .named("snapkv_mask");

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

            // The next fire's served row. `length` is that fire's own query
            // position (it is what `positions` was just given), so
            // `causal_mask` cuts exactly the slots that exist for it, and the
            // frozen keep-set cuts the ones SnapKV dropped.
            //
            // Contract #1: `keep` is `take`n before it is `put`, or the dummy
            // run that infers the geometry sees a port written but never read.
            let k = keep.take();
            mask.put(&and(&causal_mask(&length, kv_max), &k));
            keep.put(&k);

            let ids = pages_src.take();
            pages.put(&ids);
            pages_src.put(&ids);
        });

        let budget_n = max_tokens - 1;
        run_ahead(&pipe, &fwd, budget_n as usize, async || {
            let t = tok_out
                .take_host::<i32>()
                .await
                .with_context(|| format!("@{}", generated.len()))?;
            generated.push(t as u32);
            Ok(ControlFlow::Continue(()))
        })
        .await?;
    }
    pipe.close();

    let live = (n as usize).min(prefill_scores.len());

    // Host-side mirror of the fold the device just did. Recomputed rather than
    // taken on trust, so a disagreement between the two is visible in `Output`
    // instead of silently deciding the keep-set.
    let masses: Vec<f32> = (0..prompt_pages as usize)
        .map(|p| {
            let lo = p * page_size as usize;
            let hi = ((p + 1) * page_size as usize).min(prefill_scores.len());
            if lo >= prefill_scores.len() {
                0.0
            } else {
                prefill_scores[lo..hi].iter().sum()
            }
        })
        .collect();
    let total_mass: f32 = masses.iter().sum();
    let tail_page_share = if total_mass > 0.0 && prompt_pages > 0 {
        masses[(prompt_pages - 1) as usize] / total_mass
    } else {
        0.0
    };

    Ok(Output {
        sampler: "trackb-snapkv",
        text: model::decode(&generated)?,
        count: generated.len(),
        kv_max,
        prompt_len: n,
        page_budget,
        prompt_pages,
        prefill_chunks: k,
        prefill_final: tail,
        page_size,
        layers_observed,
        live_scored: prefill_scores[..live]
            .iter()
            .filter(|s| s.is_finite() && **s >= 0.0)
            .count(),
        tail_nonzero: prefill_scores[live..].iter().filter(|s| **s != 0.0).count(),
        observed_live: prefill_scores
            .iter()
            .rposition(|s| *s != 0.0)
            .map_or(0, |i| i + 1),
        scores_nan: prefill_scores.iter().filter(|s| s.is_nan()).count(),
        score_mass: prefill_scores[..live].iter().sum(),
        served_pages: kept.len() as u32,
        served_kv,
        full_kv: n,
        kept_pages: kept,
        evicted_first,
        page_mass: masses.iter().map(|m| format!("{m:.5}")).collect(),
        tail_page_share,
        score_head: prefill_scores[..live.min(8)]
            .iter()
            .map(|s| format!("{s:.5}"))
            .collect(),
    })
}
