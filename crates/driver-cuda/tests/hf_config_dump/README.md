# 58 real `config.json` files, and what they have been the oracle for

*(C++ oracle retired 2026-08-04; the Rust normalizer it checked retired with
the catalog refactor. The corpus outlived both.)*

`pie.model/1` moved HuggingFace `config.json` normalization from serve time to
import time, and from C++ to Rust. `crates/driver-cuda/csrc/src/model/config.cpp`
was what that Rust had to agree with: 855 lines, 25 `model_type` conditionals,
219 output fields, and — as its own comment admitted — a pile of rules that
existed only because "until we add per-arch metadata, derive it here."

Two implementations of the same normalization, in two languages, that must
agree by coincidence is exactly the skew an artifact exists to kill. So the
port was checked against the original **field for field on real configs**, and
then the original was deleted.

## The per-arch metadata arrived, and the normalizer went with it

That comment was the whole design, stated as a regret. `model::catalog` is the
per-arch metadata: one `const` row per model, stating the numbers a normalizer
used to derive, matched to a checkpoint by its TENSORS rather than by a
`model_type` string. `crates/model/src/{config,descriptor,facts,deployment_cuda}.rs`
are deleted — 3,595 lines — and so is `golden/`, which recorded a deleted
function's output for a deleted reader.

## What this directory is for now

`corpus/` — 58 `config.json` files, 31 real and 27 synthetic — is the oracle
for **`crates/model/tests/catalog_differential.rs`**, and it is a strictly
better one than `golden/` ever was.

`golden/` recorded what `config.cpp` *did*, including where that was wrong
(`attention_has_sinks` set for `deepseek_v4` and then unconditionally
overwritten, so DeepSeek-V4 had no sinks — the port asserted it reproduced the
bug). Agreement with it proved two programs read a file alike, never that
either read it right.

`corpus/` is what the **checkpoints themselves say**. A catalog row transcribes
numbers from a published model into Rust `const`s, and the failure that
introduces is a typo — a quietly wrong model, which is worse than a loud
refusal. Two things stop it, and they are complementary:

* `catalog_differential.rs` compares every number a corpus config states
  against the row that claims it. That catches a digit the *publisher*
  contradicts.
* `manifest.rs` matching compares the row's implied tensor shapes against the
  checkpoint's actual ones at load. That catches a digit a *tensor*
  contradicts.

Between them there is nowhere for a mistyped digit to hide.

## Regenerating the corpus

`generate.py` pulls real configs; `synthesize.py` writes the 27 synthetic ones
that exercise branches no cached model happens to contain; `hfconfig.py` is
shared by both. Adding a real checkpoint here means either transcribing a
catalog row for it or naming it in `catalog_differential.rs`'s `not_served`
list with a sentence saying why — the test fails on an unexplained one, which
is the closed-set discipline the catalog is built on.
