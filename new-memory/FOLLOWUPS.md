# Follow-ups (deferred by the user until after the new-memory PR)

## F1. Weights as a holder; PLEX decides weight retention
Today `store/weights.rs` decides weights by a fixed rule (shed every weight to the floor before tier-pressure; LRU
by use counts inside the cache), and PLEX's retention table keys cached KV only. Make the model's weight atoms a
holder beside working sets / prefix entries / snapshots: the governor decides how much, PLEX decides what stays
(candidates include weight atoms, valued by `weight_uses` and the cost of a miss), the pool moves.
- Engines switch the held fast path both ways on their own (all atoms resident for a while -> located tables / fast
  arms; first eviction -> lazy miss copy from the file mapping, slow arms), with the re-arm cost known to PLEX.
- Snapshots become disk-tier holders. Dense weights always stay on the device (user decision; budgets deleted, A10).
- Removes: budget_device_pool, budget_disk_snapshots, weights_streaming.

## F2. Config surface rename (do together with F1, only for the keys that remain)
Agreed shape: `[engine.memory]` with prefix keys — budget_device / budget_host / budget_disk, disk_path,
disk_persistent, governor_backoff / governor_patience / governor_release_idle (replacing memory_tolerance profiles:
dedicated = 1.0 / never / false, aggressive = 0.9 / 4 / true, balanced = 0.8 / 2 / true, polite = 0.7 / 1 / true);
`[engine]` common max_batch_tokens / max_batch_sequences / max_context. Delete max_state_slots (still accepted by six
backend option structs and named in fit.rs messages), kv_page_size, xla seat_tokens. Old keys refused with the new
name.

## F3. Async admission in the guest API
The KV admission wait sits inside the synchronous `submit` WIT call, so a waiting lane blocks every lane of its guest
and admitted lanes can't drain. Make `submit` async (or add an admission future) across every SDK. Until then,
inferlets that batch should cap their in-flight lanes (text-completion-bench: batch_concurrency).

## F4 — Layer-major recurrent state — DECIDED 2026-11-12: in this PR, all backends (moved to PR-READINESS A23)
Per-lane state records put a layer's lane states one record apart, interleaved with other layers / conv rows. In
the partly-L2-resident regime (sm_120, 64 lanes, ~64 MB working set) any interleaving loses 12-16% on the GDN step
vs main's gapless per-layer bank; no stride / offset skew recovers it (sweep: 528 KiB..11 MiB strides all 69-75 us
vs bank 62.7). Serving cost: Qwen3.5 c64 gdn_step +7%, conv1d +6-12% per launch (~1.6% of fire GPU time) on CUDA;
XLA qwen35 d0.8b c64 -3.4% (gated_delta 61 -> 112 us, same cause per TPU). Fix: layer-major fixed shapes (each
recurrent layer's pieces from a shape of their own so a layer's lane slots are contiguous) — a chunk / record
structure change across poem_compiler, runtime and every backend.
