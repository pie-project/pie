# new-memory -> main: PR readiness (priority #1, user 2026-11-10)

Branch today: 432 commits ahead of main, 782 files, +86k / -13k. Main merged in regularly (no drift).

## A. North star items still open (must land before the PR)
| # | Item | Owner | State |
|---|---|---|---|
| A1 | CUDA quantized expert banks: NVFP4, affine u4/u2, block FP8, W4A8/W4A4, biased selects (MXFP4 done M67) | L40S | in progress |
| A2 | CUDA held load with a pool too small: illegal address at boot (IT r9) | L40S | done M75 |
| A3 | Runtime boot fit counts only committed segments (Metal) | RT3 | done M73 (Metal device run pending) |
| A4 | ROCm quantized expert banks (after A1) | AMD | queued |
| A5 | ROCm generated region kernels: flip + delete hand rules (b8/b64 in-process numbers pending) | AMD | in progress |
| A6 | XLA: Mosaic page copy / flash / kv append on the pool; speed vs main; MoE result varies with pool chunk count (CUDA plugin) | TPU | in progress |
| A7 | XLA weight streaming + snapshots (sc/xla2, partial) | TPU | queued |
| A8 | Cerebras wafer `Spatial` engine still states degenerate facts (owner's unmerged branch) | Cerebras owner | blocked |
| A10 | Delete device/host weight budgets, dense rotor, dense host tier; max_state_slots | L40S, Metal, VK5 | done M76 (dead legacy tier cleanup left) |
| A11 | Eager path wrong tokens; nothing arms at small budgets | EAGER | done M82 (ROCm verify pending) |
| A12 | Named snapshots on the shared pool (take_host None); host-full resume hang + shortfall miscount; keep-handles key must cover the record | RT agent | in progress |
| A13 | Record rs-buffered fires (MTP / spec decode on hybrids walk eagerly; same on main) — graph capture rule | L40S | after perf fixes |
| A9 | dsv41 two-window mini cannot fire on sm_89/sm_120 (fused.lse_sink kernel gap) -> two-window CUDA unverified | L40S | queued |

## B. Decide before the PR (user)
- B1: DECIDED: ROCm anchored (GEMM-epilogue) synth is IN the PR (AMD, A4 of their synth plan).
- B2: DECIDED: dense weights always live on the device. device_weight_budget / host_weight_budget, the dense rotor
  and the dense host weight tier are DELETED (A10, L40S); the recordable-rotor task is moot.
- B3: DECIDED: all `new-memory/` docs are excluded from the PR (delete the folder from the PR branch at the end).
- B4: DECIDED: one broad verification sweep per backend right before the PR (all goldens + catalog chunk tests + IT
  e2e scenarios) is allowed.

## P. Performance gate (user: no regression vs main, any backend)
Each backend shows decode (c1 / c8 / c64 / c256), prefill (512 / 2048) and boot vs main on a quiet device, in-process
medians. Known gaps to close:
- XLA/TPU v6e (M77): prefill 512 -19%, 2048 -12%; qwen35 c64 -13%; MoE c1 -9%, c256 -12%, prefill -10%.
- ROCm synth: flip only at parity (in progress).
- CUDA (L40S, server-level, medians of 3, all models arm fully): regressions to fix, in order:
  (1) Qwen3-8B prefill 2048 145 -> 492 ms (3.4x, rising per request); (2) gpt-oss-20b prefill 2048 -23%;
  (3) gemma-4-E2B prefill 512 -15%; (4) c256 -4..-6% on Qwen3.5-0.8B, gemma, Qwen3-8B (c64 -1..-3%);
  (5) Qwen3-8B warm boot +5%. At parity: Qwen3.5-9B (all), c1/c8 everywhere. gpt-oss c64/c256 +24/+28% (faster).
- ROCm: default config arms 0 bodies on new-memory (flat range takes the spare) -> speed table waits for the fix.
- Vulkan/wgpu, Metal: full comparison vs main not yet run.

## C. PR mechanics
- C1: Delete remaining degenerate-only scaffolding wherever every engine of that path states real facts (runtime
  keeps degenerate only for engines that still need it, e.g. Spatial).
- C2: Final main merge, full build matrix (cuda, rocm, metal, vulkan, wgpu, xla, cerebras; featureless too).
- C3: PR description: design summary, contract, per-backend matrix, verification evidence, behaviour changes
  (device_weight_budget semantics, host tier lazily pinned, held weights, etc.), known follow-ups (FOLLOWUPS.md).
- C4: No AI co-author lines anywhere in commits / PR body.
