# text-completion-spec

Plain text completion that exercises the speculative-decoding interface.

The SDK's `ctx.generate(sampler).next()` defaults to `Speculation::system()`
(== `Speculation::Default`), which sets `output_speculative_tokens(true)` on
every forward pass and feeds back any drafts the driver returns. With the
`pie_driver_sglang` driver and `spec_ngram_enabled = true`, the driver
maintains an n-gram trie and proposes linear continuations after the trie
warms up; this inferlet's runtime drops as the acceptance rate climbs.

The inferlet emits one structured stat line on stdout when it finishes:

```
SPEC_STATS prompt_tokens=N generated_tokens=M elapsed_ms=T tokens_per_sec=R
```

`tests/inferlets/test_text_completion_spec.py` parses that line to compare
NGRAM-on vs NGRAM-off throughput.
