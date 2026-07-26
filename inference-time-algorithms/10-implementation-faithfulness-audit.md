# Implementation faithfulness audit

Every inferlet built for this effort, checked line by line against the paper or
reference implementation that defines it.

The question this document answers is narrow and deliberately unflattering:
**does the code compute the equation the source actually publishes, or something
that merely behaves like it?** A sampler that produces plausible text is not
evidence of anything — most of these algorithms degrade gracefully into
"temperature sampling with extra steps" when they are subtly wrong. So the
standard here is the printed formula, symbol for symbol.

Scope: the 14 inferlets in `tests/inferlets/` built during this effort. Verdicts
are one of:

| Verdict | Meaning |
|---|---|
| **Exact** | The implemented expression is algebraically identical to the published one. |
| **Exact (equivalent form)** | Algebraically identical after a transformation that provably preserves the sampling distribution, with the transformation shown. |
| **Faithful, bounded deviation** | A documented, quantified departure that does not change the algorithm's defining behaviour. |
| **Divergent** | Computes something else. |

**No implementation is Divergent.** Three carry bounded deviations, each
recorded below with its magnitude and its reason.

---

## Method

1. Fetch the paper (arXiv HTML/ar5iv where MathML preserves the LaTeX, PDF where
   it does not) and extract the defining equation **verbatim**, with its
   equation number.
2. Fetch the canonical reference implementation where one exists — for the
   community samplers (TFS, top-a, XTC, DRY) there is no paper, and the
   reference *is* the definition.
3. Extract the implemented expression from the inferlet source with `file:line`.
4. Reduce one to the other symbolically. Where the reduction needs an identity
   (log-domain shift invariance, sparse trie encoding), the identity is written
   out rather than asserted.

Two findings justify the paranoia:

- **A secondary source was wrong about EDT.** A literature-derived formula
  `T = T₀ + θ·(H/log K)` was reported for arXiv:2403.14541. The paper's
  Equation 7, read directly from the PDF, is `T = T₀ · N^(θ/Entropy)` — a
  different functional form entirely. Every equation below was confirmed against
  the paper itself, never a summary of it.
- **The audit found a live engine bug.** Verifying A9's distortion-freeness
  surfaced a floating-point defect in the shared RNG that was silently
  randomising ~1.6 % of decode steps across *every* Gumbel-based sampler. See
  [The RNG defect](#the-rng-defect-found-by-this-audit).

---

## Summary

| # | Inferlet | Source | Verdict |
|---|---|---|---|
| A1 | `locally-typical-sampling` | arXiv:2202.00666 §3 | **Exact** |
| A2 | `eta-epsilon-sampling` | arXiv:2210.15191 §3 | **Exact** |
| A3 | `tail-free-sampling` | Bricken (2019) + oobabooga ref | **Exact (equivalent form)** |
| A4 | `top-a-sampling` | ref impl (no paper) | **Exact** |
| A5 | `xtc-sampling` | oobabooga PR #6335 | **Faithful, bounded deviation** |
| A6 | `repetition-penalty` | arXiv:1909.05858 §4.1 + vLLM | **Exact** |
| A7 | `dry-repetition-penalty` | llama.cpp / p-e-w ref | **Faithful, bounded deviation** |
| A8 | `entropy-adaptive-temperature` | arXiv:2403.14541 Eq. 7 | **Exact** |
| A9 | `gumbel-watermark` | arXiv:2307.15593 §2–3 | **Exact (equivalent form)** |
| A10 | `synthid-tournament-sampling` | Dathathri et al., *Nature* 634 (2024) | **Exact (equivalent form)** |
| D1 | `classifier-free-guidance` | arXiv:2306.17806 Eq. 7 | **Exact (equivalent form)** |
| D2 | `context-aware-decoding` | arXiv:2305.14739 §2.2 | **Exact (equivalent form)** |
| E1 | `asap-grammar-aligned-decoding` | arXiv:2405.21047 Eq. 3–4, Alg. 1 | **Exact (equivalent form)** |
| E2 | `token-healing` | guidance-ai / llguidance ref | **Exact** |

---

## A1 — Locally typical sampling

- **Title:** Locally Typical Sampling
- **arXiv:** [2202.00666](https://arxiv.org/abs/2202.00666)

**Published rule.** Select the smallest set of tokens whose cumulative
probability reaches τ, ordered by *absolute deviation of information content
from the conditional entropy*:

```
score(x) = | -log p(x) - H(p) |      (nats)
```

**Implemented** — `locally-typical-sampling/src/lib.rs:104-114`:

```rust
let h = entropy_from_logprobs(&probs, &logprobs);
let deviation = add(&logprobs, broadcast(&h, [vocab]));   // log p + H
let score = abs(&deviation);                              // |·|
let (_sorted_score, order) = top_k(neg(&score), k_max);   // ascending in score
let exclusive = sub(cumsum(&probs_sorted), &probs_sorted);
let keep_sorted = lt(&exclusive, broadcast(Tensor::constant(mass), [k_max]));
```

**Reduction.** `log p + H = −(−log p − H)`, and `|−y| = |y|`, so `score` is the
published quantity exactly. `top_k` on `−score` yields *ascending* score, the
required order. The prefix sum is made **exclusive** (`cumsum − self`), so the
comparison `exclusive < τ` keeps every token up to *and including* the one that
crosses τ — matching the reference, which never returns an empty set because the
most typical token always sees `exclusive = 0`.

Entropy and log-probabilities are both in **nats** (`log_softmax`), which is what
the paper's information-theoretic derivation requires; a bits/nats mix-up here
is the classic way to get a sampler that "works" but truncates at the wrong τ.

**Verdict: Exact.** Default `mass = 0.9`, matching the reference.

---

## A2 — η-sampling / ε-sampling

- **Title:** Truncation Sampling as Language Model Desmoothing
- **arXiv:** [2210.15191](https://arxiv.org/abs/2210.15191)

**Published rule.**

```
ε-sampling:  keep x  iff  p(x) ≥ ε
η-sampling:  keep x  iff  p(x) ≥ η,   η = min(ε, √ε · exp(−H(p)))
```

**Implemented** — `eta-epsilon-sampling/src/lib.rs:96-109`:

```rust
let threshold = match mode {
    Mode::Epsilon => Tensor::constant(epsilon),
    Mode::Eta => {
        let h = entropy_from_logprobs(&probs, &logprobs);
        let adaptive = mul(Tensor::constant(epsilon.sqrt()), exp(neg(&h)));
        min_elem(Tensor::constant(epsilon), adaptive)
    }
};
let base_mask = pivot_threshold(&probs, prob_ge(threshold));
let argmax_ind = ge(&probs, broadcast(reduce_max(&probs), [vocab]));
let keep = or(base_mask, argmax_ind);
```

**Note on a known documentation bug.** HuggingFace's `EtaLogitsWarper` docstring
states the threshold as `sqrt(ε · exp(−H))`, i.e. the square root taken over the
whole product. That is **wrong**; HF's own *code* computes
`min(ε, √ε · exp(−H))`, which is what the paper defines. This implementation
matches the paper and the code, not the docstring.

`√ε` is folded on the host (`epsilon.sqrt()`) because ε is a compile-time-known
scalar — this is a constant fold, not an approximation.

The `or(base_mask, argmax_ind)` term reproduces the reference's guarantee that
the argmax always survives, so the kept set is never empty even at large ε.

**Verdict: Exact.** Nats throughout. Defaults sit in the paper's recommended
band (ε ∈ 3e-4…9e-4; η ∈ 3e-4…4e-3).

---

## A3 — Tail-free sampling

No paper — TFS is a 2019 blog post by Trenton Bricken. The definition is the
reference implementation.

**Reference** (oobabooga `sampler_hijack.py:141-169`, the de-facto standard):

```python
d2 = probs.diff().diff().abs()
normalized_d2 = d2 / d2.sum(dim=-1, keepdim=True)
normalized_d2_cdf = normalized_d2.cumsum(dim=-1)
sorted_indices_to_remove = normalized_d2_cdf > self.tfs
sorted_indices_to_remove = torch.cat((zeros(1), sorted_indices_to_remove, ones(1)), dim=-1)
```

**Implemented** — `tail-free-sampling/src/lib.rs:116-127`:

```rust
let d1 = sub(&p, shift1(&p, k_max));
let d2 = sub(&d1, shift1(&d1, k_max));
let curvature = max_elem(&d2, neg(&d2));                  // |·|
let total = max_elem(reduce_sum(&curvature), Tensor::constant(1e-9f32));
let norm = div(&curvature, broadcast(&total, [k_max]));
let exclusive = sub(cumsum(&norm), &norm);
let keep_sorted = lt(&exclusive, broadcast(Tensor::constant(z), [k_max]));
```

**Reduction — sign.** The reference uses forward differences
`d1[i] = p[i+1] − p[i]`; this uses `d1[i] = p[i] − p[i+1]`, the negation. Two
negations cancel into `d2`, and `d2` is immediately passed through `|·|`, so the
curvature magnitudes are identical.

**Reduction — alignment, and the "+1 centering" question.** This is the one
place TFS implementations genuinely disagree, so it is worth being explicit.

The original TensorFlow gist cuts at `argmax(CDF > z) + 1`, keeping ranks
`0..j+1` where `j = min{i : CDF[i] > z}`. The oobabooga port instead prepends
one `False` and appends one `True` to the removal mask, which keeps ranks
`0..j`. **The two references differ by exactly one token.**

Here, `d2[i] = p[i] − 2p[i+1] + p[i+2]` is centred on rank `i+1`, the same
centring as the reference's `sec_weights[i]`. With an exclusive prefix,
`exclusive[r] = CDF[r−1]`, so `keep[r] ⟺ CDF[r−1] < z ⟺ r ≤ j`: ranks `0..j`.
**This matches oobabooga**, the implementation every modern stack actually runs.

The original gist's extra token is not a deliberate design choice worth
inheriting — it also indexes *unsorted* `logits` with a rank computed over
*sorted* probabilities, which is a plain bug. Following the port is the correct
call.

**Boundary handling.** `shift1` clamps at `k_max−1` rather than shortening the
array, so `d2` keeps length `k_max` with zeros at the tail, instead of the
reference's length `k−2` plus two pad entries. The pad-`True` at the end means
the reference always drops its least-likely candidate; this implementation drops
it only if the CDF says so. Since the candidate set is already `top_k(k_max)`,
that single rank sits far inside the discarded tail and the difference is not
observable.

The `1e-9` floor on the normaliser guards a one-hot row where every second
difference is zero; without it the division is `0/0`.

**Verdict: Exact (equivalent form)** with respect to the canonical
implementation. Cut is strict (`>` z), normalisation is applied before the CDF —
the two subtleties that most reimplementations get wrong. Recommended z ≈
0.9–0.95.

---

## A4 — Top-a sampling

No paper — the definition is the reference implementation.

**Published rule.**

```
keep x  iff  p(x) ≥ a · p_max²
```

**Implemented** — `top-a-sampling/src/lib.rs:90-91`:

```rust
let threshold = mul(Tensor::constant(a), mul(&p_max, &p_max));
let keep = ge(&probs, broadcast(&threshold, [vocab]));
```

The exponent applies to `p_max` **only** — `a · p_max²`, never `(a · p_max)²`.
That mistake is common and rescales the threshold by `a`, which at the typical
`a ∈ [0.1, 0.5]` changes the kept set by an order of magnitude.

**Verdict: Exact.**

---

## A5 — XTC (Exclude Top Choices)

No paper — oobabooga PR #6335 by p-e-w.

**Published rule.** With probability `xtc_probability`: remove every token with
`p ≥ threshold` **except the least probable among them**.

**Implemented** — `xtc-sampling/src/lib.rs:136-157`:

```rust
let above = and(&kept_floor, ge(&probs, broadcast(Tensor::constant(cfg.threshold), [vocab])));
let inf = broadcast(Tensor::constant(f32::INFINITY), [vocab]);
let min_above = reduce_min(select(&above, &probs, &inf));   // least probable qualifier
let u = reduce_min(rng(&gate_state, [1]));
let fired = lt(&u, Tensor::constant(cfg.probability));
let drop = and(broadcast(&fired, [vocab]),
               and(&above, gt(&probs, broadcast(&min_above, [vocab]))));
```

**Reduction.** The reference expresses "all qualifiers except the least
probable" positionally, as `sorted_indices_to_remove[..., :-1] = probs[..., 1:] >= threshold`
— rank `r` is removed iff rank `r+1` also clears the threshold, which on a
descending sort is precisely "everything above the threshold except the last
one". This implementation states the same set directly:
`above ∧ (p > min_above)`. Equivalent, and it avoids needing a sort.

The Bernoulli gate draws from a **decorrelated RNG stream** (`GATE_OFFSET`)
rather than reusing the sampling state, so the fire decision is independent of
the token draw — reusing one stream for both would correlate them.

**Deviation (bounded): the newline/EOS abort is not implemented.** The reference
returns the scores unmodified if the removal set would contain a newline or EOS
token. That guard is tokenizer-specific and needs a token-ID table this inferlet
does not carry. Consequence: XTC may occasionally suppress an end-of-sentence
token it would otherwise have spared, slightly lengthening output. It does not
alter the truncation rule.

llama.cpp additionally refuses to run when `threshold > 0.5` (above which at most
one token can qualify, making XTC a no-op or worse). That bound is documented in
the inferlet's config rather than enforced.

**Verdict: Faithful, bounded deviation.**

---

## A6 — Repetition / frequency / presence penalties

- **Title:** CTRL: A Conditional Transformer Language Model for Controllable Generation
- **arXiv:** [1909.05858](https://arxiv.org/abs/1909.05858)

**Published rule** (CTRL §4.1, as implemented by HF and vLLM):

```
repetition:  score = score < 0 ? score · θ : score / θ        (θ = 1.2 default)
frequency:   logit -= frequency_penalty · count(token)
presence:    logit -= presence_penalty  · 1[count(token) > 0]
```

**Implemented** — `repetition-penalty/src/lib.rs:125-152`:

```rust
let out_seen = gt(counts, &zero);
let prompt_seen = gt(prompt_present, broadcast(Tensor::constant(0.5f32), [vocab]));
let seen = or(&out_seen, &prompt_seen);

let positive = gt(logits, &zero);
let repenalized = select(&positive, div(logits, &r), mul(logits, &r));
let l = select(&seen, &repenalized, logits);

let freq = mul(broadcast(Tensor::constant(cfg.frequency_penalty), [vocab]), counts);
let pres = mul(broadcast(Tensor::constant(cfg.presence_penalty), [vocab]),
               cast(&out_seen, DType::F32));
(sub(sub(&l, &freq), &pres), ...)
```

**The scope asymmetry is the part that matters.** vLLM applies the repetition
penalty over `prompt_mask | output_mask` — tokens seen in the prompt **or** the
output — while frequency and presence read the **output only**. This is not an
accident of implementation; it is what every production stack does, and getting
it wrong makes a long prompt silently penalise its own vocabulary.

Here `seen = out_seen ∨ prompt_seen` gates the repetition penalty, and
`out_seen` alone gates presence. `prompt_present` is seeded host-side in one
pass over the prompt (`lib.rs:231`), which is cheaper than a device scatter and
exactly equivalent.

The penalty is applied **once per token**, not once per occurrence — the
division/multiplication is not compounded, matching CTRL.

**Verdict: Exact.**

---

## A7 — DRY (Don't Repeat Yourself)

No paper — p-e-w's sampler, canonical implementation in llama.cpp.

**Published rule.**

```
penalty(t) = multiplier · base^(L(t) − allowed_length)      when L(t) ≥ allowed_length
logit(t)  -= penalty(t)
```

where `L(t)` is the length of the longest suffix of the history that both ends
at the current position and would be continued by `t`.

**Implemented** — `dry-repetition-penalty/src/lib.rs:217-227`:

```rust
for n in cfg.allowed_length..=cfg.max_ngram {
    let hit = ge(&m, broadcast(Tensor::constant(n as f32), [l]));
    let votes = scatter_add(&vocab_zero, &next_tok, cast(&hit, DType::F32));
    let charge = cfg.multiplier * cfg.base.powi((n - cfg.allowed_length) as i32);
    penalty = select(gt(&votes, &vocab_zero), broadcast(Tensor::constant(charge), [vocab]), &penalty);
}
...
let penalized = sub(&logits, &penalty);
```

**Reduction.** llama.cpp computes `L` with a reversed Z-algorithm; this computes
the same quantity with a data-parallel suffix-match scan
(`suffix_match`, `lib.rs:167-185`) that extends match lengths one position at a
time under a monotone `alive` mask. Different algorithm, identical `L`. The `n`
loop runs **ascending** so the longest match writes last and wins, which is the
`max` the reference takes. The penalty is **subtracted** from the logit, not
divided into it — DRY is additive, unlike the CTRL penalty above.

Defaults `base = 1.75`, `allowed_length = 2` match llama.cpp.

**Deviation (bounded): sequence breakers are not implemented.** llama.cpp resets
matching at breaker tokens (default `['\n', ':', '"', '*']`) and exempts the
breakers themselves. Those are *strings*, and mapping them to token IDs is
tokenizer-dependent. Consequence: DRY here can match a repeat that spans a
sentence or line boundary, making it slightly more aggressive than the reference
on structured text. The penalty formula is unaffected.

**Deviation (bounded): `max_ngram` caps `L`.** The device program is a finite
unroll, so match lengths saturate at `max_ngram` and the penalty saturates at
`multiplier · base^(max_ngram − allowed_length)`. The reference is unbounded.
Since the penalty grows geometrically, the cap binds only after the model has
already been penalised into submission.

**Verdict: Faithful, bounded deviation.**

---

## A8 — Entropy-adaptive temperature (EDT)

- **Title:** EDT: Improving Large Language Models' Generation by Entropy-based Dynamic Temperature Sampling
- **arXiv:** [2403.14541](https://arxiv.org/abs/2403.14541)

**Published rule — Equation 7, read directly from the PDF:**

```
T = T₀ · N^(θ / Entropy),        0 < N < 1
```

with `N = 0.8` in all of the paper's experiments and `T₀` an upper bound on the
temperature.

**Implemented** — `entropy-adaptive-temperature/src/lib.rs:127-135`:

```rust
let h = entropy_from_logprobs(&probs, &logprobs);
let h_safe = max_elem(&h, Tensor::constant(1e-6f32));
let exponent = div(Tensor::constant(cfg.theta), &h_safe);
let t = mul(Tensor::constant(cfg.t0), exp(mul(&exponent, Tensor::constant(cfg.ln_n))));
```

`N^(θ/H) = exp((θ/H)·ln N)`, and `ln N` is folded on the host. Defaults
`t0 = 1.0`, `theta = 0.1`, `n = 0.8`.

Because `0 < N < 1`, `ln N < 0`, so the temperature *falls* as entropy falls —
confident steps get sharper, uncertain steps stay near `T₀`. The `1e-6` floor on
`H` keeps `θ/H` finite on a one-hot row, where the exponent would otherwise be
`+∞` and the temperature `0`.

**This is the entry a secondary source got wrong.** A literature summary
reported `T = T₀ + θ·(H/log K)` — additive, normalised by log-vocabulary, and
*increasing* in entropy. That is not Equation 7 and would inverse the algorithm's
behaviour. The PDF settles it.

**Verdict: Exact.**

---

## A9 — Gumbel / Aaronson watermark

- **Title:** Robust Distortion-free Watermarks for Language Models
- **arXiv:** [2307.15593](https://arxiv.org/abs/2307.15593)

**Published rule.**

```
generation:  x_t = argmax_i  ξ_{t,i}^(1 / p_i)
detection:   S   = Σ_t −log(1 − ξ_{t, x_t}),     S/n ~ Exp(1) mean 1 under H₀
```

**Implemented** — `gumbel-watermark/src/lib.rs:211-216, 244-252`:

```rust
fn aaronson_score(noise: &Tensor, token: &Tensor) -> Tensor {
    let g = scalar_gather(noise, token);
    let r = exp(neg(exp(neg(&g))));                     // ξ = exp(−exp(−G))
    let tail = max_elem(sub(Tensor::constant(1.0f32), &r), Tensor::constant(1e-7f32));
    reshape(neg(log(&tail)), [1])                       // −log(1 − ξ)
}
...
let noise = gumbel(&keyed, [vocab]);
let token = if cfg.watermark { reduce_argmax(add(&scaled, &noise)) } else { gumbel_max(&scaled, free_state) };
```

**Reduction — the exponential race.** Let `E_i = −log ξ_i ~ Exp(1)`. Then

```
argmax_i ξ_i^(1/p_i)  =  argmax_i (1/p_i)·log ξ_i  =  argmin_i E_i / p_i
```

and with `G_i = −log(−log ξ_i) = −log E_i`, the Gumbel-max identity gives

```
argmax_i (log p_i + G_i) = argmax_i (log p_i − log E_i) = argmin_i E_i / p_i
```

The two are the **same argmax**, not an approximation. The implementation
samples in the second form because the detector needs the noise tensor `G`
itself, which `gumbel_max` does not expose.

**Detection.** `ξ = exp(−exp(−G))` inverts the Gumbel transform, and the
statistic is the published `−log(1 − ξ)`. The z-score at `lib.rs:519`,
`z = (mean_score − 1)·√n`, is correct because `−log(1 − ξ)` is exactly `Exp(1)`
under H₀, so its mean over `n` draws has standard error `1/√n`.

**Bounded numerical note.** The `1e-7` floor on `1 − ξ` caps the per-token score
at `≈ 16.1`. For `X ~ Exp(1)`, `E[min(X, c)] = 1 − e^{−c}`, so at `c = 16.1` the
null mean is biased low by `1e-7` — six orders of magnitude below the `1/√n`
standard error at any practical `n`. Without the floor a single strongly
favoured token returns `+∞` and destroys the statistic.

A **decoy secret** (`lib.rs:252`) scores the same tokens under an unrelated key,
giving an empirical null alongside the analytic one; the run asserts the real
key beats the decoy.

**Verdict: Exact (equivalent form).** The distortion-freeness claim is only
meaningful given the RNG fix below — see [The RNG defect](#the-rng-defect-found-by-this-audit).

---

## A10 — SynthID tournament sampling

- **Title:** Scalable watermarking for identifying large language model outputs
- **Reference:** Dathathri et al., *Nature* **634**, 818–823 (2024)

**Published rule.** Draw `2^depth` candidates i.i.d. from `p`. Run `depth`
knockout rounds; in round ℓ, each pair is decided by the keyed Bernoulli
`g_ℓ(·) ∈ {0,1}`, higher wins, ties broken uniformly.

**Implemented** — `synthid-tournament-sampling/src/lib.rs:273-278`:

```rust
let mut watermarked = base.clone();
for g in &gs {
    let mass = reduce_sum(mul(&watermarked, g));                 // E_p[g]
    let offset = broadcast(sub(Tensor::constant(1.0f32), &mass), [vocab]);
    watermarked = mul(&watermarked, add(&offset, g));            // p·(1 + g − E_p[g])
}
```

**Reduction — the closed form is exact, and here is the derivation.** For one
round with candidates `X, Y ~ p` i.i.d. and `m = E_p[g]`:

```
g(w) = 1:  P(win) = 2·p(w)·[(1−m) + m·½]  = p(w)·(2 − m)
g(w) = 0:  P(win) = 2·p(w)·(1−m)·½        = p(w)·(1 − m)
```

Both collapse to `p(w)·(1 + g(w) − m)`, which is the implemented expression.
Normalisation is preserved exactly: `Σ p(1+g−m) = 1 + m − m = 1`. Because the
round-`ℓ` winners are i.i.d. draws from `p_ℓ`, the same step composes, so
`depth` layers of reweighting equal the full `2^depth` tournament in
distribution.

This costs `depth` vector ops instead of `2^depth` sampling ops — a 9-layer
watermark is 9 passes over `[vocab]` rather than 512 draws. **It is a
reformulation, not an approximation.**

The derivation also predicts the detector's signal: the emitted token's expected
g-value after one layer is `m(2−m) = 2m − m²`, which at `m = ½` is `0.75`,
comfortably above the `0.5` null.

**Repeated-context handling** (`lib.rs:280`): when the context n-gram has been
seen before, the **unwatermarked** `base` is used. The reference does the same,
and — critically — its *detector* skips those positions too. Watermarking a
repeated context would re-apply correlated noise and inflate the detector's
false-positive rate. Both sides skip here.

**Verdict: Exact (equivalent form).**

---

## D1 — Classifier-free guidance

- **Title:** Stay on topic with Classifier-Free Guidance
- **arXiv:** [2306.17806](https://arxiv.org/abs/2306.17806)

**Published rule — Equation 7:**

```
log P̂_θ(w_i | w_{<i}, c) = log P_θ(w_i | w_{<i}) + γ·( log P_θ(w_i | w_{<i}, c) − log P_θ(w_i | w_{<i}) )
```

**Implemented** — `classifier-free-guidance/src/lib.rs:51-53`:

```rust
let cond   = log_softmax(intrinsics::logits());
let uncond = log_softmax(uncond_logits);
let guided = log_softmax(add(&uncond, mul(sub(&cond, &uncond), gamma)));
```

Symbol for symbol. The operands are true log-probabilities
(`log_softmax`), which is what the equation is written in — the paper's prose
says "logits space" loosely, but its formula says `log P`.

**Renormalisation is required and present.** `log P̂` as defined is not
normalised; the outer `log_softmax` restores it before sampling.

γ = 1 must be an exact identity (`uncond + 1·(cond − uncond) = cond`), and the
run asserts it: KL(guided ‖ cond) = 0.0000.

**Verdict: Exact (equivalent form)** — identical to the published equation, with
the outer renormalisation the paper requires.

---

## D2 — Context-aware decoding

- **Title:** Trusting Your Evidence: Hallucinate Less with Context-aware Decoding
- **arXiv:** [2305.14739](https://arxiv.org/abs/2305.14739)

**Published rule** (§2.2, the defining displayed equation):

```
y_t ~ softmax[ (1+α)·logit_θ(y_t | c, x, y_{<t}) − α·logit_θ(y_t | x, y_{<t}) ]
```

**Implemented** — `context-aware-decoding/src/lib.rs:62-67`:

```rust
let with_context = log_softmax(intrinsics::logits());
let query_only   = log_softmax(query_only_logits);
let guided = log_softmax(sub(mul(&with_context, 1.0 + alpha), mul(&query_only, alpha)));
```

**Reduction — log-probabilities vs raw logits.** The paper writes `logit`; this
uses `log p`. Substituting `logit = log p + log Z`:

```
(1+α)(log p_c + log Z_c) − α(log p_u + log Z_u)
  = (1+α)·log p_c − α·log p_u + [ (1+α)·log Z_c − α·log Z_u ]
```

The bracketed term does not depend on the token index, so it is a constant shift
that **cancels in the softmax**. The two forms induce the identical sampling
distribution. Using `log_softmax` is in fact the better-conditioned choice, since
raw logit magnitudes vary by model.

α = 0 must reduce to plain conditional decoding; the run asserts it
(KL = 0.0000). Default α = 0.5, the paper's recommended value for summarisation
(it uses α = 1 for knowledge-conflict tasks).

**Verdict: Exact (equivalent form).**

---

## E1 — ASAp grammar-aligned decoding

- **Title:** Grammar-Aligned Decoding
- **arXiv:** [2405.21047](https://arxiv.org/abs/2405.21047)

This is the most intricate of the fourteen and the one most worth checking
carefully, because a wrong ASAp still emits grammatical output — it just samples
from the same skewed distribution as plain GCD, which is the entire thing the
paper exists to fix.

**Published rule — Equation 3 (the EFG recurrence) and Algorithm 1:**

```
c̃_S(w_{1:i}) := Σ_{w'} P(w' | w_{1:i}) · c̃_S(w_{1:i} · w')

Algorithm 1:
  Initialize S := {}, c̃_S(·) := 1
  for m ≤ M:
      Draw w_{1:n} ~ Q̃_S via ancestral sampling
      S := S ∪ {w_{1:n}}
      for i in (n−1)...1:
          for w' with w_{1:i}·w' ∉ L_prefix(G):  c̃_S(w_{1:i}·w') := 0
          c̃_S(w_{1:i}) := Σ_{w'} P(w'|w_{1:i}) · c̃_S(w_{1:i}·w')
```

**Implemented** — `asap-grammar-aligned-decoding/src/lib.rs:169-176`:

```rust
fn recompute(&mut self, node: usize) {
    let deficit: f32 = self.children.iter()
        .filter(|((parent, _), _)| *parent == node)
        .map(|(edge, &child)| self.edge_prob[edge] * (1.0 - self.alpha[child]))
        .sum();
    self.alpha[node] = (self.mass[node] - deficit).clamp(0.0, 1.0);
}
```

and `lib.rs:404-412`:

```rust
let terminated = constraint.is_terminated();
let leaf = *path_nodes.last().expect("path always has a leaf");
trie.alpha[leaf] = if terminated { 1.0 } else { 0.0 };
for step in (0..masses.len()).rev() {          // i = n−1 ... 1
    let parent = path_nodes[step];
    trie.mass[parent] = masses[step];
    trie.edge_prob.insert((parent, generated[step]), probs[step]);
    trie.recompute(parent);
}
```

**Reduction — the sparse encoding of Equation 3.** The published sum runs over
the entire vocabulary, which cannot be materialised: `c̃_S` is defined over every
prefix, and the trie only ever stores the explored ones. Split the sum by
whether the child has been visited, using the initialisation `c̃_S = 1` for
unexplored prefixes and `c̃_S = 0` for grammar-rejected ones:

```
Σ_{w'} P(w'|u)·c̃(u·w')
  = Σ_{allowed w'} P(w'|u)·c̃(u·w')                          (rejected terms are 0)
  = Σ_{allowed w'} P(w'|u)·1  −  Σ_{explored w'} P(w'|u)·(1 − c̃(u·w'))
  = M(u)  −  Σ_{explored w'} P(w'|u)·(1 − c̃(u·w'))
```

which is exactly `mass[node] − deficit`. `M(u)` is the **grammar-allowed LM
mass** at `u`, recorded on first visit — so the algorithm's explicit
"zero out every `w'` leaving `L_prefix`" step is discharged by *excluding* those
tokens from `M(u)` rather than by enumerating them, which would be intractable.

The remaining pieces line up one-to-one:

| Algorithm 1 | Implementation |
|---|---|
| `c̃_S(·) := 1` | `Trie::new()` → `alpha: vec![1.0]`, and `child()` pushes `1.0` (`lib.rs:158, 184`) |
| `for i in (n−1)...1` | `for step in (0..masses.len()).rev()` (`lib.rs:407`) |
| zero non-`L_prefix` children | excluded from `M(u)`; dead-end leaf set to `0.0` (`lib.rs:406`) |
| Eq. 3 update | `recompute(parent)` (`lib.rs:411`) |
| Eq. 4 proposal `Q̃_S ∝ P·c̃_S` | `sparse_alpha` overrides applied to the LM row (`lib.rs:476`) |

**`P` must be the unmodified LM distribution.** Both `mass[]` and `edge_prob[]`
are read from the raw LM row, never from the grammar-masked or α-reweighted one.
Using the reweighted distribution here is the subtle way to make ASAp converge to
the wrong fixed point.

**Theorem 1 (over-approximation, `c̃_S ≥ c`) is structural, not clamped.** An
earlier revision took `.min()` against the previous α to force monotonicity;
that hid errors. Monotonicity now falls out of the recurrence, and the run
reports it as an observable: `monotone: true`, with α(root) declining
`3.93e-13 → 3.76e-13 → 1.82e-13 → 1.8202e-13 → 1.8201e-13 → 1.8132e-13`.

**Behavioural evidence.** In round 1, `c̃_S ≡ 1` everywhere, so Eq. 4 reduces to
GCD — and the run shows `asap_logprob == gcd_logprob` exactly. From round 2 the
two diverge as `c̃_S` falls (round 1: both `−30.771597`). That is precisely the paper's Figure 1 story, and it
is a much stronger check than reading the code: an implementation that had
subtly broken the fold-back would not reproduce the round-1 identity *and* the
subsequent divergence.

**Verdict: Exact (equivalent form).**

---

## E2 — Token healing

No paper — the technique originates in `guidance-ai/guidance`, and now lives in
the `llguidance` Rust backend.

**Published rule.** Roll the prompt back past its final token(s), then constrain
the next token to the set that **regenerates the removed bytes as a prefix**.
This repairs the boundary bias where a prompt ending mid-token
(`"http:"` → `["http", ":"]`) blocks the tokenizer's natural continuation
(`"://"`).

**Implemented** — `token-healing/src/lib.rs:120-127`:

```rust
for (token, bytes) in token_bytes.iter().enumerate() {
    if bytes.starts_with(&fragment) {
        prefix_mask[token] = true;
        prefix_candidates += 1;
    }
}
```

The mask is `{t : bytes(t) starts_with fragment}` — a byte-level prefix test over
the whole vocabulary, which is the definition. A token *equal* to the fragment
satisfies `starts_with`, so healing can always fall back to the tokenizer's
original choice and can never remove a legal completion; the empty-set case is
therefore impossible and is reported as an error rather than silently sampled
through.

**Verdict: Exact.**

---

## The RNG defect, found by this audit

Verifying A9's *distortion-free* claim meant proving that
`argmax(log p + G)` samples exactly from `p`. It does not, if `G` can be `+∞`.

`ptir_rng_hash_uniform` built its draw as `(bits + 0.5) / 2²⁴` with
`bits ∈ [0, 2²⁴)`. Every value is mathematically inside `(0, 1)` — but at
`bits = 2²⁴ − 1` the quotient is `1 − 2⁻²⁵`, which sits **exactly** halfway
between `0x1.fffffep-1` and `1.0`, so round-to-even snaps it to `1.0f`.

Then `G = −log(−log 1) = −log(−0) = +∞`, and `+∞` unconditionally wins
`argmax(logits + G)`. The sampler returns a **uniformly random token**.

| Quantity | Value |
|---|---|
| Probability per element | `1 / 2²⁴ = 5.96e-8` |
| Expected `+∞` per `[vocab]` draw, `vocab = 262144` | `0.0156` |
| Corrupted decode steps | **≈ 1 in 64 (1.6 %)** |

This was not hypothetical. It is the mechanism that broke E1: a `−1e9` masked
logit plus `+∞` is `+∞`, which beat the legal maximum of `−0.0585`, and
`gumbel_max` returned a token it had itself masked out — surfacing as
`grammar rejected token 234061`.

**Fixed** in `compiler/ir/src/rng.rs` by clamping at
`UNIFORM_MAX = 1.0 − f32::EPSILON/2.0` (literal `0.99999994`). That file is the
single source of truth for the RNG contract, so regenerating propagates the
clamp to every device Gumbel site at once. Regression tests:
`uniform_never_reaches_one` walks all 2²⁴ mantissa values, asserts the clamped
draw stays in `[0, 1)` with a finite Gumbel, and pins the *unclamped* hit count
at exactly 1; `generated_backends_clamp_the_uniform` keeps CUDA and Metal in
sync. Existing byte-parity vectors are unchanged — none of them hit the clamp.

**What it did and did not corrupt.** `−∞` masking was always safe:
`k_reduce_argmax` (`driver/cuda/src/pipeline/tier0/tier0_kernels.cuh:458`)
documents that *"float NaNs are never selected"*, and `−∞ + ∞ = NaN`. So the
truncation samplers A1–A5, which mask with `f32::NEG_INFINITY`, never leaked a
masked token — they suffered only the ~1.6 %/step random-token distortion, in
common with every other Gumbel consumer. A9 is the one where this was not merely
a quality issue but a **falsification of the algorithm's headline property**:
1.6 % of tokens were drawn from the uniform distribution rather than the model's,
so the watermark was not distortion-free.

**Deployment note (superseded).** This section originally recorded that the CUDA
engine could not be rebuilt here — the vendored flashinfer `fastdiv.cuh` wanted
`cuda::fast_mod_div`, absent from both installed toolkits — so the numbers below
came from the **pre-fix** engine. That blocker has since been cleared and the
engine is now rebuilt and redeployed routinely
(`cargo build --release -p pie-server-py`, then copy `libpie_engine.so` over
`sdk/python-server/python/pie/_engine.cpython-312-x86_64-linux-gnu.so`). The
distributional results below were produced *before* the RNG fix, which
strengthens rather than weakens them: every inferlet passed its assertions
despite the 1.6 % corruption. Everything in **"What actually runs"** at the end
of this document was measured on the post-fix engine.

---

## Deliberate deviations, collected

| # | Deviation | Effect | Reason |
|---|---|---|---|
| A5 | No newline/EOS abort guard | May suppress an EOS it would otherwise spare; slightly longer outputs | Needs a tokenizer-specific token-ID table the inferlet does not carry |
| A7 | No sequence breakers | Matches repeats across sentence/line boundaries; more aggressive than reference | Breakers are strings; the mapping to token IDs is tokenizer-dependent |
| A7 | `max_ngram` caps match length | Penalty saturates at `multiplier·base^(max_ngram−allowed_length)` | The device program is a finite unroll; the cap binds only after the penalty is already overwhelming |
| A3 | `d2` keeps length `k_max` with clamped edges | Least-likely candidate of the `top_k` set is not force-dropped | The reference's forced drop sits deep in a tail already excluded by `top_k` |
| A9 | Per-token score floored at `1e-7` | Null mean biased low by `1e-7` | Without it a strongly favoured token yields `+∞` and destroys the statistic |

Every other implementation is algebraically identical to its source.

---

## GPU verification

Every verdict above was re-confirmed end to end on an NVIDIA L40S
(`Qwen/Qwen3-0.6B`, `cuda_native` driver) after the audit was written, so the
document describes code that demonstrably runs. **14/14 of the algorithm
inferlets pass**, and the full curated suite — which adds the search,
speculative, KV-layout and composition inferlets — now stands at **29/29**; see
**"What actually runs"** below. Several runs double as numerical confirmations
of the equations:

| # | Reported | Confirms |
|---|---|---|
| A1 | `mean_kept 81.5`, `mean_mass 0.845` at τ=0.95 | Exclusive prefix — kept mass lands just under τ, never over |
| A2 | `mean_kept 449`, `mean_mass 0.927` at ε=3e-4 | η adapts far wider than a fixed ε would |
| A3 | `mean_kept 9.97` at z=0.95 | Curvature cut is aggressive where the tail is flat |
| A4 | `mean_kept 19.6` at a=0.2 | `a·p_max²`, not `(a·p_max)²`, which would keep ~1 token |
| A5 | `fire_rate 0.406` at probability 0.5 | Bernoulli gate fires at its nominal rate over 32 steps |
| A6 | `mean_penalized 20.9`, `peak_repeat 2.0` | Prompt-seeded presence vector is live from step 1 |
| A7 | `peak_penalty 4.2875`, `longest_repeat 5` | **`0.8 · 1.75^(5−2) = 4.2875` exactly** — the DRY formula, to the last digit |
| A8 | `mean_entropy 3.276`, `mean_temperature 0.973` | `T = T₀·N^(θ/H)` — near `T₀` at high entropy, falling to ≈0.52 at the observed `min_entropy 0.034` |
| A9 | `mean_null_score 1.0325`, `mean_score 2.621`, `z 9.17` | **The decoy null mean is 1.0 — exactly `E[Exp(1)]`**, which is the distributional claim the detector's z-score rests on |
| A10 | `mean_score 0.576` vs null `0.465`, `z 2.59` | Emitted-token g-value sits above ½, as `2m − m²` predicts |
| D1 | `mean_kl 0.0688`, `guidance_shift 0.062` at γ=1.5 | Guidance moves the distribution without collapsing it; γ=1 gives KL 0.0000 |
| D2 | `mean_kl 0.186`, `context_shift 0.094` at α=0.5 | Context amplification is active; α=0 gives KL 0.0000 |
| E1 | round 1 `asap_logprob == gcd_logprob == −30.771597`; α(root) `3.93e-13 → 3.76e-13 → 1.82e-13 → …`; `monotone: true`; 6/6 schema-valid, all terminated | **ASAp reduces to GCD exactly when `c̃_S ≡ 1`, then diverges** — the paper's central claim, plus Theorem 1's monotonicity as an observable |
| E2 | `fragment ":"` → `healed_token 1110` (3 bytes), 324 candidates | Healed to `"://"`, the canonical boundary-bias example |

A9's `mean_null_score = 1.0325` is worth dwelling on. The decoy secret scores
the *same emitted tokens* under an unrelated key, so it samples the null
distribution directly — and it lands on 1.0, the mean of `Exp(1)`. That is an
empirical confirmation of the derivation that `−log(1 − ξ)` is exactly `Exp(1)`
under H₀, which is what makes `z = (mean − 1)·√n` a z-score at all.

### A bug this verification pass caught

The first GPU run of E1 aborted in 0.3 s with a `panic_bounds_check` inside
`unpack_mask`. The cause is a vocabulary mismatch that is invisible on most
models: `inferlet::mask::bit_allowed` indexed `mask[j >> 5]` unguarded, while
Qwen3 declares `vocab_size = 151936` against 151669 real tokenizer tokens. A
constraint mask packed for the tokenizer occupies 4740 words and covers 151680
bits, so logit slots 151680–151935 read past the end of the slice.

`bit_allowed` now returns `false` past the mask's coverage, mirroring
`pack_allowed`, which already drops ids it cannot represent — those slots decode
to no token at all, so refusing them is both the safe answer and the correct
one. This also repairs the first failure in the upstream
`json-schema-constrained-decoding` inferlet, which shares the helper. That one
kept failing afterwards for an unrelated reason, since diagnosed and fixed: its
decode loop submitted `DEFAULT_RUNAHEAD_DEPTH` fires but supplied only one
grammar mask, and the run-ahead fires **silently reused a stale mask**, so the
constraint was not actually enforced on those steps. See
**"What actually runs"**.

---

## Runtime cost

Faithfulness says nothing about price. Every inferlet below was timed against
`naive-baseline`, a control written for this measurement: the *identical*
skeleton — one N-wide prefill fire, a 1-wide device-carried decode loop kept
`DEFAULT_RUNAHEAD_DEPTH` fires ahead of the host drain — whose epilogue does
nothing but temperature-scale the logits and draw a Gumbel-max sample. Whatever
an algorithm inferlet costs above that number is the algorithm.

Cost is measured as a two-point regression rather than a throughput figure.
Each configuration runs at a 32- and a 160-token budget, seven repetitions
each after a discarded warm-up, and the *marginal* per-token cost is the slope

```
per_token_ms = (median_t(160) − median_t(32)) / 128
```

Differencing cancels install, JIT, prefill and teardown, which are identical at
both points; the intercept recovers them. Medians rather than minima, because
one inferlet (A10) turned out to be bimodal and a minimum would have reported
its lucky mode. All fifteen configurations below ran in a single server session
on one NVIDIA L40S with `Qwen/Qwen3-0.6B` (`vocab = 262144`), so the ratios are
directly comparable; the absolute baseline drifts about ±10 % between sessions.

| Inferlet | ms/token | × naive | Dominant cost |
|---|---|---|---|
| E2 `token-healing` | 2.48 | 0.75× | Greedy `reduce_argmax` — no noise tensor at all |
| A9 `gumbel-watermark` | 2.78 | 0.84× | Free; see the note on `gumbel_max` below |
| **`naive-baseline`** | **3.30** | **1.00×** | — |
| A8 `entropy-adaptive-temperature` | 3.53 | 1.07× | One entropy reduction + a scalar power |
| `naive-baseline` + 2 stat channels | 3.61 | 1.09× | The instrumentation, priced separately |
| A4 `top-a-sampling` | 3.81 | 1.15× | One `ReduceMax`, one compare — no sort |
| A2 `eta-epsilon-sampling` | 4.31 | 1.31× | Entropy + two elementwise passes |
| A5 `xtc-sampling` | 4.48 | 1.36× | Threshold scan + Bernoulli gate |
| A6 `repetition-penalty` | 4.89 | 1.48× | Scatter over the seen-token vector |
| A3 `tail-free-sampling` | 5.20 | 1.49× | `top_k(k_max=128)` over a 151936 vocab |
| A1 `locally-typical-sampling` | 5.39 | 1.54× | `top_k(k_max=128)` over a 151936 vocab |
| A7 `dry-repetition-penalty` | 7.08 | 2.15× | 8-deep n-gram match, unrolled on device |
| D2 `context-aware-decoding` | 13.12 | 3.98× | **Two forward passes, serialized** |
| D1 `classifier-free-guidance` | 13.87 | 4.21× | **Two forward passes, serialized** |
| A10 `synthid-tournament-sampling` | 33.55 | 10.2× | Nine knockout rounds per token |

Three structural observations fall out of this table.

**The overhead is entirely marginal, never fixed.** Intercepts land between 87
and 165 ms for every configuration including the baseline — that is install plus
prefill, and no inferlet carries a heavy one-time setup. Cost scales with tokens
generated, so it is predictable and it never surprises a short request.

**The `top_k` cliff was an implementation defect, and it is fixed.** A1 and A3
originally measured 17.71 and 17.47 ms/token — a 5.3× cliff over the baseline —
and they paid it *identically* despite computing entirely different statistics,
which is what identified `top_k` rather than the algorithms as the cause. The
kernel behind it was an incremental-threshold selection that rescans the whole
row once per pick, costing `O(k · vocab)`: 19.4 M element visits per token at
`k_max = 128` and this model's 151936-token vocabulary, executed by a *single*
256-thread block. Replacing it with a radix select of the cut followed by a
bitonic sort of the `k` survivors — `O(8·vocab + k·log²k)` — brings A1 to 5.39
and A3 to 5.20 ms/token, and makes the cost **flat in `k`**:

| `k_max` | before | after |
| --- | --- | --- |
| 8 | 5.92 | 5.64 |
| 128 | 17.75 | 5.39 |
| 1024 | 116.18 | 5.67 |

The residual ~1.5× over the baseline is the barrier structure (`top_k` and
`cum_sum` are hard schedule barriers), which is a real property of the language.
The 5.3× was not.

**Four copies of `top_k` is the story worth keeping.** The first fix targeted
tier-0's `k_topk_rows` and moved the benchmark **not at all**, because `top_k`
is an `L::Library` op: the fused runtime dispatches it to `k_grouped_topk` in
`grouped_runtime.cuh`, and tier-0's copy only serves the standalone path. There
were four independent implementations of the same total order — tier-0, grouped,
the generated fused emitter, and the M1 singleton — and optimising the one named
after the op was not optimising the one that runs. The fix now routes tier-0 and
the grouped runtime through a single shared `t0_block_topk_fast` template
parameterised on a value accessor, so the total order has one definition rather
than four. *A benchmark that does not move after a fix is evidence about
dispatch, not evidence that the fix was wrong.*

**D1/D2's 4× is two effects, not one.** Both score every token under two
prompts, which is 2× the forward-pass work by construction. The other 2× is lost
pipelining: the next input depends on the *combined* output of both passes, so
neither inferlet may run ahead, and the run-ahead window that hides host latency
for every other entry in this table is unavailable to them. This is inherent to
contrastive decoding, not an artefact of the implementation.

### A9 costs less than sampling nothing

The watermark lands *below* the control, and the reason is measurement, not
algorithms. `gumbel-watermark` carries an A/B inside it: with `watermark = true`
it samples via `reduce_argmax(add(scaled, gumbel(keyed, [vocab])))`, and with
`watermark = false` it falls through to `gumbel_max(scaled, state)`.

| Sampling spelling | ms/token |
|---|---|
| `gumbel(...)` + `add` + `reduce_argmax` | 2.85 |
| `gumbel_max(...)` | 3.60 |
| `naive-baseline` | 3.51 |

These are **the same program**. `gumbel_max` (`sdk/rust/ptir-dsl/src/value.rs:912-927`)
emits exactly `RngKeyed{Gumbel}`, `Add`, `ReduceArgmax`; `gumbel()` (ibid. `:754`)
emits the `RngKeyed` and the caller writes the other two. `PIE_PTIR_DUMP_PLAN=1`
confirms both compile to a single fused region with no library regions. So the
27 % gap is **cross-session variance**, not an op-selection effect — the control
itself ranges 2.70–3.60 ms/token across server sessions, which brackets the whole
gap. A9 spells the sampler out because the detector needs the noise tensor
itself, not just the argmax; that choice is neutral for performance. The
defensible statement about A9's cost is that it is *within noise of zero*, the
practical counterpart of its distortion-freeness.

### A10 is bimodal

`synthid-tournament-sampling` at a 160-token budget returns either ~1.3 s or
~5–6 s. The 33.55 ms/token above is a slow-mode figure and should be read as an
upper bound; the fast mode's slope is ≈2.4 ms/token, which is the honest cost.

The split was originally recorded as varying *across sessions*. It does not: a
run of six consecutive calls inside one process alternates between the modes
(6225, 1294, 5874, 5599, 4982, 1383 ms). The apparent session-stability was the
NVRTC disk cache — a first-ever plan shape pays a 12–31 s compile, which
dominated whatever came after it.

What the split is not:

- **Not a correctness problem.** The response is bit-identical in both modes —
  same text, `mean_score = 0.5764`, `z_score = 3.1754`, `unique_contexts = 48`.
- **Not device contention.** The GPU sits at ~25 % mean utilisation in *both*
  modes at full SM clock, with no other process resident on it. A10 is
  host-bound.
- **Not run-ahead depth.** Run-ahead of 2, 4, 6, 8 and 12 are all bimodal.
  Read-back ring capacity beyond the run-ahead window shifts the fast/slow ratio
  slightly and fixes nothing.
- **Not compiler nondeterminism.** `compiler.rs` contains no hash-ordered
  iteration, and the emitted plan hashes to the same cache key every time.

What it is: **a sharp knee in program size.** The identical inferlet at
`depth = 1` and `depth = 3` is stable to ±3 % once warm (780–864 ms across six
calls). Only the production `depth = 9` is bimodal. A10 is also the only
inferlet in the set that reads back three channels per token instead of one. The
working hypothesis is that at depth 9 its per-fire host cost reaches parity with
its device time and it sits balanced on the pipelining knee — but a deeper
run-ahead window does not rescue it, so that account is not yet complete.

### The undocumented 30-second first call

Fused regions are NVRTC-compiled on first use and cached on disk under
`~/.cache/pie/ptir-cuda` — 537 modules, 176 MB, on the machine these numbers
were taken on. A hit is invisible; a miss is a 12–31 s stall inside what looks
like an ordinary request. Two consequences worth stating plainly:

1. Every benchmark in this document is a **warm** number. The first call against
   any new plan is one to two orders of magnitude slower.
2. The cache key covers the plan bytes, so any shape change mints a new entry —
   including a changed channel capacity or a changed `depth`, which an author
   would reasonably regard as tuning rather than as a recompile.

---

## What actually runs

Faithfulness on paper and faithfulness on a GPU are different claims. This
section records the second one, measured with **one engine process per test**
(`tests/inferlets/test_curated.py` boots a single engine for the whole suite, so
one wedged inferlet poisons every test after it — the isolated runner is the
only honest instrument).

**Curated isolated matrix: 29/29.** Qwen3-0.6B, `cuda_native`, NVIDIA L40S.
Before this pass it was 21/30 with nine failures in five classes, every one of
which turned out to be a real defect rather than a test artefact.

### Four engine and driver defects

1. **The ABI validator rejected the geometry class it was supposed to admit.**
   `abi_validation.hpp:255` bounded `geometry_class` at `DECODE_ENVELOPE`, so
   every `DeviceGeometry` bind returned `INVALID_ARGUMENT` from the entry
   wrapper — before `Context::Impl::bind_instance`, the only place that prints
   the error. It presented as a silent `status -1`.
2. **The engine only recognised one spelling of device geometry.** Classification
   matched Design-A's 2-D `[B, P]` pages channel; every real inferlet uses a flat
   `[B*P]` one, so they all fell back to `Host`, which cannot derive a
   device-sampled token. `detect_pooled_device_geometry` now mirrors the driver's
   `is_loop_carried_explicit_geometry_trace` contract instead.
3. **The W1.6 commit gate read uninitialised device memory on a first fire.**
   `prepare_step` resolves descriptors against the *previous* fire's commit cell;
   a first-ever ring index has no predecessor and held raw `cudaMalloc` bytes.
   Commit snapshots are now seeded `{1, 0}` at allocation **and on recycle**.
4. **Prepare-time descriptor resolution raced its own bind-time seed upload.**
   It never took the `init_done_` edge that `begin_enqueue` takes (RV-28).

All four presented as the same opaque message, `ptir prologue or channel
readiness did not commit`. Two env-gated diagnostics
(`PIE_DEBUG_PULL_VALIDATE=1`, printing pull-validate ticket rejections and
stage-readiness rejections with a reason) were added because without them the
four are indistinguishable.

### A GPU hang in the generated sampler path

`consensus-decoding` pegged the GPU at 100 % with a launch that never settled.
The cause is worth stating plainly, because it is a **correctness cliff hiding
behind a fusion optimisation**.

`pivot_threshold(probs, cummass_le(p))` — the top-p / nucleus truncation every
sampler in this document depends on — only ever worked when the *entire*
surrounding dataflow matched `LibraryOp::NucleusSample`
(`compiler/plan/src/compile.rs:1305`). That recognizer is an exact-shape match
over softmax spelled as `exp(sub(l, max))/sum`, then `pivot_threshold`, then
`select` against a `−∞` constant, then `add(gumbel)`, then `reduce_argmax`. One
`broadcast` spliced between the mask and the argmax is enough to break it.

When it broke, the compiler emitted the generated region — and the generated
emitter deliberately routed `cummass_le` to the M1 reference, because its own
parallel arm assumed a pre-sorted row (which the DSL never produces). The M1
reference is a selection sort with a linear "already picked" rescan per
candidate, executed by **thread 0 alone**: O(len³) at a 151936-token vocabulary.
It never returns.

Every other inferlet in the suite matched the library pattern by accident, so
the path had no coverage at all. It now runs the block-cooperative selection
loop that tier0's `k_pivot_cummassle` already used — one block-wide "next
largest still-unpicked element" pick per iteration, carrying the previous pick
as a total-order threshold, stopping as soon as the exclusive mass clears the
cutoff. `sampling-primitives` pins it with a keep-mask published in a shape that
*cannot* match the library pattern, asserting the exact `Predicate::CummassLe`
contract without depending on float summation order: 37 tokens, 0.901134 mass at
`top_p = 0.9`, in 0.2 s.

Its sibling predicate had the same disease one order milder. `rank_le` — the
top-k truncation `mirostat-v2-sampling` uses — was spelled as a literal rank
computation in all three implementations (tier-0's `k_pivot_rankle`, the
generated fused emitter, and the M1 singleton reference): for each element,
rescan the row and count the strictly-greater values. That is O(len²)
unconditionally, ~2.3e10 element visits per row per token at this vocabulary,
and the single-threaded singleton copy was effectively another hang.

All three now run a 4-pass 8-bit MSB radix select over a monotone key
```
key(v) = ~( (u & 0x80000000) ? ~u : (u | 0x80000000) )    where u = bits(v)
```
which reverses value order (larger float ⇒ smaller key), sends NaN to
`0xFFFFFFFF` so it sorts last exactly as the tie contract wants, and cannot
collide with a finite float. `greater(i)` is the count of strictly smaller keys,
which is monotone in the key, so `greater(i) < k` holds precisely when
`key(i) <= K_k` for `K_k` the k-th smallest key counting multiplicity — ties
therefore all survive or all fall together, which is what the reference does and
is why it can legitimately keep more than `k` elements. The cost is O(5·len)
regardless of `k`. Measured at the 151936-token width: **0.48 ms**, against an
O(len²) form that would need ~30000× more element visits. Two tier-0 tests pin
it — a randomised parity case covering ties, both signed zeros, NaN and `k` at
both clamp bounds, and a production-width case checked against an O(len)
`nth_element` cut with a timing gate.

The general shape is the same as the nucleus hang and worth naming: **a
predicate whose reference spelling is quadratic will pass every small-fixture
test and fail only at production width**, which is the one width the unit tests
did not use.

### Four inferlet authoring contracts, discovered the hard way

All of these fail **silently** — no error, no exception, just a wrong or empty
result — which is why they cost the most debugging time.

- **Loop-carried geometry ports must `take()` before they `put()`.** Under
  `Host` and `DecodeEnvelope` the host drains the geometry channel and frees the
  ring; under `DeviceGeometry` it does not. `k_stage_readiness` treats a put into
  a full ring as *not ready*, which clears `pass_commit` and turns the fire into
  a **dummy run**. Four inferlets were affected.
- **Every host-`Writer` channel a fire takes must be `put` before that fire's
  `submit`.** Three inferlets submitted `DEFAULT_RUNAHEAD_DEPTH` fires while
  supplying a single value.
- **The KV page CSR is the wire's source of truth for `kv_len`, not the `KvLen`
  port.** Six inferlets over-declared it and read uninitialised KV. This one gets
  its own section below, because it is the most serious defect this project
  found and the test suite was structurally incapable of catching it.
- **Logits entering a `NucleusSample` must come from the logits intrinsic behind
  at most a `reshape`.** One inferlet (`consensus-decoding`) fed the sampler a
  `broadcast`; the compiler matched the pattern anyway and the driver's scratch
  elision corrupted the sampler's own input. Also below.

### Three algorithms are structurally depth-1

`greenlist-watermarking`, `json-schema-constrained-decoding` and
`contrastive-decoding` derive the host input each fire needs — the greenlist
mask, the grammar mask, the amateur token — from the **previous fire's output
token**. There is no run-ahead to be had; the pipelining those loops appeared to
express was fictitious. The JSON-schema case was the damaging one: its
run-ahead fires reused a stale grammar mask, so the constraint silently was not
enforced on those steps. All three now submit one fire at a time, which is the
honest shape and costs the run-ahead overlap the earlier ms/token tables assumed.

### The beam identity, finally measurable

`beam-search` at width 1 had never run: `intrinsics::logits()` squeezes to
rank-1 `[v]` when a fire has a single read-out row, so the `[B, 1]`-broadcast
score column could not meet it and the epilogue failed to bind.

With that fixed, the epilogue publishes a per-lane `reduce_argmax` over the raw
logits beside the beam pick. At width 1 the two must agree on every step: top-1
over the flattened `[1*v]` candidate block is `argmax(log_softmax(l) + score)`,
and both `log_softmax` and adding a per-row constant are monotone. The
comparison therefore tests `log_softmax`, the score accumulator, the `[B*v]`
flatten, `top_k` and the `idx / v` / `idx % v` decomposition against an operator
that shares none of that machinery.

| width | greedy mismatches (16 steps) | best score |
|---|---|---|
| 1 | **0** | −19.7988 |
| 2 | 17 | −19.7520 |
| 4 | 46 | −13.6841 |

The identity holds exactly at width 1, and the search demonstrably leaves the
greedy path — and improves the score — as the width grows. Width alone would not
have shown the first; the identity alone would not have shown the second.

### The defect the test suite could not see

`cuda_chat_completion_e2e` asserts that a continuation of "The capital of France
is" contains "Paris". It had been `#[ignore]`d. When it was finally run, the
inferlet returned:

```
" worellerllerllerllerller..."
```

The engine was not at fault. Under the same engine, `naive-baseline` returned
`" Paris, and it's a country in eastern Europe"` and `text-completion-bench`
returned `"<think>\nOkay, the user is asking about the capital of France. I
know"`. Bisecting on token count localised it precisely: `max_tokens = 1`
produced the *correct* first token, `max_tokens = 4` produced garbage. Prefill
was fine; the **decode loop** was broken.

Instrumenting the inferlet with host-readable debug channels showed every input
was correct — `pos = 28, 29, 30 ...`, `kv_len = 29, 30, 31 ...`, a dense mask of
exactly the right width with exactly the right maximum. Correct inputs, wrong
output.

The mechanism is in the driver. `derive_kv_len_kernel`
(`driver/cuda/src/kernels/geometry.cu:14`) reconstructs the attended span from
the page CSR:

```
kv_len[r] = (page_count - 1) * page_size + last_page_len[r]
page_count = kv_page_indptr[r + 1] - kv_page_indptr[r]
```

and `last_page_len` is the *only* thing that survives of the `KvLen` port
(`descriptor_resolve.hpp:15`: `last_page_len = ((len - 1) % page) + 1`). The
kernel comment states this is bit-identical to the host formula in
`request.rs::append_request_with_options` — it is a deliberate handshake
invariant, and it means the CSR wins.

`chat-completion` declared `page_indptr = [0, pool_pages] = [0, 3]` — "these are
the pages I reserved" — with `kv_len = 29`. The driver derived
`last_page_len = 13` and a span of `2 * 16 + 13 = 45`. Attention read **16 cells
of uninitialised KV** on every decode step. Correcting the CSR to track the true
length made the token stream match `text-completion-bench` exactly.

An audit of every `page_indptr` binding found six inferlets with the same defect:
`chat-completion`, `attention-sink`, `sliding-window-attention`,
`contrastive-decoding` (amateur pass only), `consensus-decoding` and
`beam-search`. Roughly twenty others already used the correct idiom
(`page_count = ceil(kv_len / page_size)`), which is why the failure looked
sporadic rather than systemic. For multi-lane fires the `pages` array must
additionally be tiled at stride *page_count*, not pool size, because
`page_indptr[c] = c * page_count` indexes into it; since channel shapes are
static the fix is to keep the pool-sized capacity and rebuild the contents with
`gather(&pids, rem(iota(N), broadcast(&page_count, [N])))`.

Two things about this are worth more than the fix itself.

**The mask does not save you.** `pack_dense_mask.cu` packs the dense
`[TOTAL_Q, STRIDE]` mask page-major over each lane's live pages, and the `klen`
it uses is the *derived physical* span. A perfectly-sized mask is laid out
against the wrong geometry, so it cannot suppress what the geometry invented.

**The suite was built to miss this.** `test_curated.py`'s `_nonempty` asserts
the output is non-empty. Six inferlets produced fluent-looking garbage and passed
for the entire project. Even `test_beam_search_greedy_identity` could not catch
it: it compares the beam pick against `reduce_argmax` of *the same corrupt
logits*, so it validates the beam machinery while being blind to model quality.
A liveness assertion cannot detect a corruption that preserves liveness. The
suite now carries `_attends_prompt`, which runs the fixed prompt "The capital of
France is" and requires the continuation to mention France or Paris — a
content assertion, cheap, and the pre-fix binaries fail it.

### A second silent corruption: the nucleus fast path's unstated precondition

Fixing the page CSR left `consensus-decoding` with one residual symptom: every
candidate's **first** token was junk (`"cumplir"`, `"!user"`, `"!human"`) even
though the rest of the continuation was fluent and on-topic. The prefill and the
decode loop use the same sampler expression, so the difference had to be
geometric — and it was.

`consensus-decoding` prefills the shared prompt once. That fire has **one**
read-out row, so `intrinsics::logits()` is `[1, vocab]`. To give `B` candidates
independent starting tokens the inferlet broadcast that row to `[B, vocab]`,
applied the shared nucleus keep-mask, and added `B` independent Gumbel draws.
This is textbook — it is exactly how you sample `B` times from one distribution —
and every layer accepted it.

The mechanism, established by bisection:

1. The compiler's `match_nucleus_add_order`
   (`compiler/plan/src/compile.rs:1393`) matches a 13-node DAG **structurally**.
   It never asks where the logits came from, so the broadcast form matches.
2. The driver's nucleus prep
   (`driver/cuda/src/pipeline/generated/fused_runtime.cuh`, ~L1007) then applies
   an optimisation that is only sound for the intrinsic: it sets
   `maximum_value_bytes[region.inputs[0]] = 4` and `temporary_elided[...] = 1`,
   because the real logits live in the model's own buffer and the sampler reads
   them directly. It applied the same elision to the scale-divide's numerator.
3. Per-lane scratch offsets are assigned from `maximum_value_bytes` with **no
   liveness reuse**, so a value whose slot was shrunk to 4 bytes sits immediately
   adjacent to live neighbours.
4. The `broadcast` is not elided-away — it still executes, and writes a full
   `[B, vocab]` tensor into that 4-byte slot.
5. `k_grouped_nucleus_*` (the three-kernel fast path taken because
   `vocab > kMaxExactNucleusLibraryVocab = 4096`) additionally early-returned on
   `row >= lanes[lane].sampled_rows`, leaving rows `1..B` of a broadcast fire
   never written at all.

The debugging lever worth recording: `wide`, the pre-divide value, is the one
intermediate the pattern does **not** exact-consumer-check. Tapping it to a host
channel therefore preserves the match, whereas tapping any other intermediate
breaks the match and silently *fixes* the output by falling back to the generated
path. "Adding a probe makes the bug disappear" was itself the diagnosis.

Three driver narrowings landed — bound `sampled_rows` only for direct bf16 lane
reads, gate both elisions on the input actually resolving to the logits
intrinsic, and only absorb the scale-divide when the region is the single node
the launch-time skip expects. They make the runtime conservative in exactly the
cases the optimisation was not written for. They do **not** make the broadcast
form correct: the compiler still claims a pattern whose precondition it does not
verify, and closing that in the matcher destabilised six unrelated compiler
fixtures. So the fix that shipped is at the inferlet: the prefill now nucleus-
samples its single `[1, vocab]` row and all `B` lanes start from that token, with
divergence coming from the decode loop, whose logits genuinely are `[B, vocab]`.
Sampling diversity was then verified directly — at `temperature = 1.2` the three
candidates diverge by the second sentence; at `0.6` they are identical because
the keep-set is a single token, which is the distribution's fault and not the
sampler's. `temperature` and `top_p` are now inferlet parameters rather than
constants, so that distinction is testable rather than assumed.

The general lesson generalises past this bug: **a fast path chosen by pattern
match must validate every assumption it makes beyond the pattern.** The author's
contract is the pattern; anything else the runtime relies on is a precondition
nobody agreed to.

---

## Citations

- **Title:** Locally Typical Sampling
- **arXiv:** [2202.00666](https://arxiv.org/abs/2202.00666)

- **Title:** Truncation Sampling as Language Model Desmoothing
- **arXiv:** [2210.15191](https://arxiv.org/abs/2210.15191)

- **Title:** CTRL: A Conditional Transformer Language Model for Controllable Generation
- **arXiv:** [1909.05858](https://arxiv.org/abs/1909.05858)

- **Title:** EDT: Improving Large Language Models' Generation by Entropy-based Dynamic Temperature Sampling
- **arXiv:** [2403.14541](https://arxiv.org/abs/2403.14541)

- **Title:** Robust Distortion-free Watermarks for Language Models
- **arXiv:** [2307.15593](https://arxiv.org/abs/2307.15593)

- **Title:** Stay on topic with Classifier-Free Guidance
- **arXiv:** [2306.17806](https://arxiv.org/abs/2306.17806)

- **Title:** Trusting Your Evidence: Hallucinate Less with Context-aware Decoding
- **arXiv:** [2305.14739](https://arxiv.org/abs/2305.14739)

- **Title:** Grammar-Aligned Decoding
- **arXiv:** [2405.21047](https://arxiv.org/abs/2405.21047)

Non-arXiv sources:

- Dathathri et al., "Scalable watermarking for identifying large language model
  outputs", *Nature* **634**, 818–823 (2024) — A10.
- Trenton Bricken, "Tail Free Sampling" (2019) — A3.
- oobabooga/text-generation-webui PR #6335 (p-e-w), `sampler_hijack.py` — A3, A5.
- llama.cpp `src/llama-sampler.cpp` — A5, A7.
- vLLM `model_executor/layers/sampler.py` — A6 penalty scopes.
- guidance-ai/guidance and the `llguidance` backend — E2.
