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
| A8 | Cerebras: main's new wafer-program backend (#1435, merged M87) states degenerate facts; pool / tiers / record copies / weight streaming need its owner + hardware | — | OUT OF SCOPE for this PR (user); listed as a known limitation |
| A10 | Delete device/host weight budgets, dense rotor, dense host tier; max_state_slots | L40S, Metal, VK5 | done M76 (dead legacy tier cleanup left) |
| A11 | Eager path wrong tokens; nothing arms at small budgets | EAGER | done M82 (ROCm verify pending) |
| A12 | Shared-pool snapshots; host-full hang + miscount; keep-handles key covers the record | RT | done M84 (other engines: A15) |
| A13 | Record rs-buffered fires (MTP / spec decode on hybrids walk eagerly; same on main) — main bug, fix in this PR (user 2026-11-11) | RSREC agent | done M107 (Mamba2 extended path still eager: Nemotron) |
| A14 | Wire the rung shrink (SPEC 6) | RT | done M92 (CUDA) |
| A21 | Activation rungs on Metal / Vulkan / wgpu / XLA (only CUDA states rungs; the walk is a no-op elsewhere) | backend owners | not applicable on XLA, Metal, Vulkan, wgpu (activation is a buffer of its own outside the pool; no rungs stated, walk no-op); CUDA done M92 |
| A15 | read_tier_slots / write_tier_slots (snapshots on shared pools) on Metal, Vulkan, wgpu, XLA (CUDA in sc/rt4; Cerebras out of scope) | Metal agent, VKP, TPU | done (Vulkan+wgpu M94, XLA M95, Metal M100) |
| A16 | Held weights flip to streaming on ROCm 30B MoE (body reservation) | EAGER | done M89 + M93 flat-fit; MI300X verified 2026-11-11 (30B holds weights, 106728 cache pages, 89/89 bodies armed on 4 models, 9 ROCm goldens equal); arm-after-residency (eager-fix-3) next |
| A17 | Guest panic under admission waits: bench barrier bug fixed (sc/sdk-wakeup) | SDK agent | done; A19 left |
| A19 | A restartable process whose own demand exceeds the pool restarts forever (should fail with the starvation error); bind hangs waiting for state slots at 96 MiB | RT7 | done M91 + M93 (rt7b: churn closed) |
| A18 | Idle cache demoted only when PLEX keys it above the floor | RT6 | done M90 |
| A20 | Window-only models never reuse a prefix (no auto checkpoint; main reused full window pages) -> auto checkpoints at page-aligned fire ends holding window slot refs | RT8 | done M99 (gpt-oss p2048 parity with main) |
| A22 | Metal perf vs main (M1 Max): ACCEPTED by user 2026-11-11 — c1 Qwen3-0.6B +0.3%, gemma -0.3%, Qwen3.5 -0.5% (c8 +1.1%); prefill and boot faster on all. (Metal serving encodes live on main too; ICB recording not required — user 2026-11-11) | Metal agent (sc/metal-gate) | in progress |
| A23 | Layer-major recurrent state (one fixed shape per recurrent layer; a layer's lanes contiguous) on every backend — user decision 2026-11-12 (see FOLLOWUPS F4 for numbers) | LM agent (core) + backend owners | in progress |
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
- CUDA (L40S, tip + pool-fit fix, server-level medians): remaining gaps: gemma-4-E2B c64 -3.9%, c256 -5.9%,
  warm boot +13% (154 vs 85 compositions armed); Qwen3.5-0.8B c64 -2.0%, c256 -3.4%; Qwen3-8B c256 -1.4%;
  gpt-oss prefill 2048 (A20, window checkpoint). Parity or faster elsewhere (gpt-oss c64/c256 +25/+30%; Qwen3.5
  prefill and boot faster; Qwen3-8B end RSS 3.6 vs 2.5 GB after the keyed-demote gate).
- ROCm (MI300X, M93 vs main f0364015, 89/89 armed): Qwen3-8B / Qwen3.5-9B decode+prefill at parity or better;
  regressions: boot +4.2 s (8B), +9.6 s (30B), +6.1 s (gemma); gemma decode c1 -4.2%, c8 -5.5%, c64 -3.1% (likely A3
  region flip vs main's hand fusions); 30B c8 -4%. 30B c256 stalls at ~512 tok/s on main too (pre-existing, noted).
  AMD fixing: EF3 flat over-count check -> gemma per-rule A/B -> boot breakdown -> 30B c8.
- Vulkan/wgpu, Metal: full comparison vs main not yet run.

## C. PR mechanics
- C1: Delete remaining degenerate-only scaffolding wherever every engine of that path states real facts (runtime
  keeps degenerate only for engines that still need it, e.g. Spatial).
- C2: Final main merge, full build matrix (cuda, rocm, metal, vulkan, wgpu, xla, cerebras; featureless too).
- C3: PR description: design summary, contract, per-backend matrix, verification evidence, behaviour changes
  (device_weight_budget semantics, host tier lazily pinned, held weights, etc.), known follow-ups (FOLLOWUPS.md).
- C4: No AI co-author lines anywhere in commits / PR body.
