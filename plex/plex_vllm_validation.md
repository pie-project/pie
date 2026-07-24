# PLEX v0.6 / vLLM validation

PLEX v0.6 is live on the vLLM V1 scheduler at
`ingim/vllm@9540799585d27e8e8c1c732438315f22e6e96b5d`. The matching Pie runtime is
`pie-project/pie@eba63e00a5034695a4d2b13f33f6bfe872ac0491`; Wasmtime fuel was
removed in `c95e23cb1eeaaf5e8e15e7f53c1ab03a89a96d15`.

The scheduler hot path never invokes Wasm, parses JSON, accesses policy state,
or calls Python policy callbacks. It reads a cached immutable hint and
immediately uses native scheduling when no valid hint is available.

## Validated environment

| Component | Version |
|---|---|
| GPU | NVIDIA A100 80GB PCIe |
| Driver | 580.159.03 |
| Python | 3.12.13 |
| PyTorch / CUDA | 2.11.0+cu130 / 13.0 |
| FlashInfer | 0.6.15.post1 |
| vLLM package | 0.1.dev19007+g910cc8543 |
| `pie-plex` | 0.6.0 |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |

## Adapter boundary

| PLEX operation | vLLM behavior |
|---|---|
| `schedule` | One-shot singleton selection hints and token budgets, revalidated by native feasibility checks |
| `cache` | Ordered request-level virtual reclaim decisions, published only under KV pressure |
| `feedback` | Acked/coalesced progress, selection/cache enactment, preemption, completion, abort, and cleanup |
| `admit` | Not attached in-engine |
| `route` | Not attached in-engine |

Snapshots are coalesced to a 25 ms publication window and plans expire after
250 ms. Request membership changes invalidate older hints. Queue saturation,
stale or malformed outcomes, unsupported operations, and policy fallback all
select native behavior.

Actions and `schedule.atomic-enqueue@1` are not advertised. Multi-request
selection units are rejected because vLLM cannot guarantee all-or-none
allocation. The cache seam represents one virtual reclaim object per request;
it is not block/page-level cache admission.

`fields.body` is a host-protected mirror of engine inputs. Other request fields,
shared state, and scratch remain policy-private and do not directly mutate a
vLLM `Request`. Public PLEX metadata is not an authenticated identity source;
production ingress must derive principals and group authorization.

## Correctness and performance

The full PLEX v0.6 release gate passed, including Rust/Python SDK tests, policy
package rebuilds, 31-policy fixtures, replay, mechanics, layering, and
performance budgets. vLLM passed 14 scheduler PLEX tests, 3 OpenAI protocol
tests, the argument/config test, and Ruff checks.

For batch 16 with 256 generated tokens per request over five timed repeats:

| Mode | Median throughput |
|---|---:|
| Native | 6899.44 tok/s |
| PLEX VTC | 6891.87 tok/s |
| Delta | -0.110% |

The first-repeat outputs were token-for-token identical across all 4096 output
tokens (`sha256:e4366b992a6add6e5c8afa8fe7155ac70ea58c426bcd6cb34cdec31feeabc27a`).
The PLEX worker completed all 208 submissions with zero queue drops, zero
fallbacks, and zero unavailable outcomes. It observed 103 successful schedule
outcomes, 100 successful feedback outcomes, 1394 fully enacted selections, and
16 partially enacted selections.

Logical continuation generation `0 -> 1` produced 32 tokens in each generation
with zero fallback and zero drops. After every live run, `nvidia-smi` reported
no remaining compute process.

## Policy evidence

Nine representative policies attached to the real GPU engine:

| Policy | Live evidence |
|---|---|
| Coordinated | 10 schedule successes, 9 feedback successes, 42 full and 8 partial enactments |
| Agentix | Schedule and feedback success, no fallback/drop |
| VTC | 10 schedule successes, 8 feedback successes, 42 full and 8 partial enactments |
| FairServe | Schedule and feedback success, no fallback/drop |
| Helium | Schedule success; feedback unavailable by manifest, cleanup still committed |
| Continuum | Schedule/cache/feedback attachment, no fallback/drop |
| KVFlow | Schedule success; feedback unavailable by manifest |
| PEEK | Schedule success; feedback unavailable by manifest |
| RAGCache | 69 cache successes and 5 enacted request-level reclaims under a 13 MiB KV pool |

Unavailable counts for Helium, KVFlow, PEEK, and RAGCache are expected because
their manifests do not implement the corresponding feedback or schedule
operation. They are not fallback failures.

## Remaining boundary

This evidence does not claim in-engine admission or routing, page/object-level
cache admission, action enactment, multi-request atomic scheduling, distributed
policy state, or authentication of caller-supplied PLEX metadata. The complete
machine-readable result is in `plex_vllm_validation.json`.
