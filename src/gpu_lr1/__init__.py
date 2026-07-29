"""The research tree: the measurement suite, the verifications, the workloads.

The library is `gpugrammar` and the engine is `gpugrammar._engine`, re-exported
as `gpugrammar.device`. Nothing here ships in the wheel; it is what the numbers
are produced with and what the device is checked against.

    gpu_lr1.verify              every device check against the reference matcher
    gpu_lr1.rigor               latency, soundness, overlap, serving, cost, fill
    gpu_lr1.crossover           where a captured step overtakes a host matcher
    gpu_lr1.vllm_smoke          end to end through vLLM, optionally verifying
    gpu_lr1.generate_instances  stage one of the corpus: the JSON a model emits
    gpu_lr1.replay_tokenizer    stage two: that text through a real tokenizer
    gpu_lr1.residency           what one compiled schema costs to keep resident
    gpu_lr1.token_groups        how far a real vocabulary collapses into groups

The pure-Python prototypes that came first - a canonical LR(1) compiler, a
byte-DFA schema compiler, the first Triton kernels and two samplers - were
deleted once the Rust front end and `gpugrammar._engine` had superseded them
and nothing in any path imported them. `docs/prototype-history.md` is the
record of what they were for and what they measured; git holds the code.
"""
