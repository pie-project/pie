# new-memory -> main: PR readiness (priority #1, user 2026-11-10)

Branch today: 432 commits ahead of main, 782 files, +86k / -13k. Main merged in regularly (no drift).

## A. North star items still open (must land before the PR)
| # | Item | Owner | State |
|---|---|---|---|
| A1 | CUDA quantized expert banks: NVFP4, affine u4/u2, block FP8, W4A8/W4A4, biased selects (MXFP4 done M67) | L40S | in progress |
| A2 | CUDA held load with a pool too small: illegal address at boot instead of the clean refusal (IT r9) | L40S | assigned |
| A3 | Runtime boot fit counts only committed segments (Metal) | RT3 | done M73 (Metal device run pending) |
| A4 | ROCm quantized expert banks (after A1) | AMD | queued |
| A5 | ROCm generated region kernels: flip + delete hand rules (b8/b64 in-process numbers pending) | AMD | in progress |
| A6 | XLA: Mosaic page copy / flash / kv append on the pool; speed vs main; MoE result varies with pool chunk count (CUDA plugin) | TPU | in progress |
| A7 | XLA weight streaming + snapshots (sc/xla2, partial) | TPU | queued |
| A8 | Cerebras wafer `Spatial` engine still states degenerate facts (owner's unmerged branch) | Cerebras owner | blocked |
| A9 | dsv41 two-window mini cannot fire on sm_89/sm_120 (fused.lse_sink kernel gap) -> two-window CUDA unverified | L40S | queued |

## B. Decide before the PR (user)
- B1: ROCm anchored (GEMM-epilogue) synth and other pre-existing parity gaps: block the PR, or follow-up?
- B2: Recordable rotor (budgeted loads decode ~10x slower): block, or follow-up?
- B3: `new-memory/` notes folder: ship as docs (e.g. docs/memory/) trimmed to SPEC + CONTRACT + BACKENDS, or drop.
- B4: Verification sweep before the PR: allow one broader run per backend (all goldens + catalog chunk tests + IT
  e2e scenarios), as an exception to targeted-tests-only.

## C. PR mechanics
- C1: Delete remaining degenerate-only scaffolding wherever every engine of that path states real facts (runtime
  keeps degenerate only for engines that still need it, e.g. Spatial).
- C2: Final main merge, full build matrix (cuda, rocm, metal, vulkan, wgpu, xla, cerebras; featureless too).
- C3: PR description: design summary, contract, per-backend matrix, verification evidence, behaviour changes
  (device_weight_budget semantics, host tier lazily pinned, held weights, etc.), known follow-ups (FOLLOWUPS.md).
- C4: No AI co-author lines anywhere in commits / PR body.
