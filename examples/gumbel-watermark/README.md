# gumbel-watermark

Context-keyed Gumbel-max watermarking (Aaronson): each token is the argmax of
`log p / T + g`, where the Gumbel noise `g` is keyed by a secret and a hash of
the last `context_width` tokens, so a detector holding the secret can
recompute it. The run detects its own text under the secret and under a decoy
secret (the empirical null) and reports a z-score over the distinct contexts.

## The self-check

The watermark registered when the keyed z reaches 4 over at least 100
distinct contexts. Fewer contexts are too few to judge: the run neither
passes nor fails. The default length, 256 tokens, leaves enough contexts on
the shipped models.

The keyed sampler picks each token from its context's noise, so on a peaked
model the same context draws the same token and the text can loop: past some
step it makes no new contexts however long it runs (Qwen3.5 stops at 86
distinct contexts at 256, 512 and 1024 tokens). The check detects that from
the token stream, when the contexts settle into a repeating period, and the
verdict says so:

> too few contexts to judge: the keyed output cycled after N distinct
> contexts (period P) — deterministic Gumbel repeats on peaked models; try a
> different prompt or secret
