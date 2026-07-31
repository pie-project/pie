"""The research tree: the measurement suite, the verifications, the workloads.

The library is `engrain` and the engine is `engrain._engine`, re-exported
as `engrain.device`. Nothing here ships in the wheel; it is what the numbers
are produced with and what the device is checked against.

    engrain_lab.verify              every device check against the reference matcher
    engrain_lab.rigor               latency, soundness, overlap, serving, cost, fill
    engrain_lab.crossover           where a captured step overtakes a host matcher
    engrain_lab.vllm_smoke          end to end through vLLM, optionally verifying
    engrain_lab.generate_instances  stage one of the corpus: the JSON a model emits
    engrain_lab.replay_tokenizer    stage two: that text through a real tokenizer
    engrain_lab.residency           what one compiled schema costs to keep resident
    engrain_lab.token_groups        how far a real vocabulary collapses into groups

The pure-Python prototypes that came first - a canonical LR(1) compiler, a
byte-DFA schema compiler, the first Triton kernels and two samplers - were
deleted once the Rust front end and `engrain._engine` had superseded them
and nothing in any path imported them. `docs/prototype-history.md` is the
record of what they were for and what they measured; git holds the code.
"""
