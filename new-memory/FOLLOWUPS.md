# Follow-ups (deferred by the user until after the new-memory PR)

## F1. Weights as a holder; PLEX decides weight retention
Today `store/weights.rs` decides weights by a fixed rule (shed every weight to the floor before tier-pressure; LRU
by use counts inside the cache), and PLEX's retention table keys cached KV only. Make the model's weight atoms a
holder beside working sets / prefix entries / snapshots: the governor decides how much, PLEX decides what stays
(candidates include weight atoms, valued by `weight_uses` and the cost of a miss), the pool moves.
- Engines switch the held fast path both ways on their own (all atoms resident for a while -> located tables / fast
  arms; first eviction -> lazy miss copy from the file mapping, slow arms), with the re-arm cost known to PLEX.
- Snapshots become disk-tier holders; dense planes eventually become weight shapes (needs the recordable rotor).
- Removes: budget_device_pool, budget_disk_snapshots, weights_streaming, budget_host_weights, and in the long run
  budget_device_weights.

## F2. Config surface rename (do together with F1, only for the keys that remain)
Agreed shape: `[engine.memory]` with prefix keys — budget_device / budget_host / budget_disk, disk_path,
disk_persistent, governor_backoff / governor_patience / governor_release_idle (replacing memory_tolerance profiles:
dedicated = 1.0 / never / false, aggressive = 0.9 / 4 / true, balanced = 0.8 / 2 / true, polite = 0.7 / 1 / true);
`[engine]` common max_batch_tokens / max_batch_sequences / max_context. Delete max_state_slots (still accepted by six
backend option structs and named in fit.rs messages), kv_page_size, xla seat_tokens. Old keys refused with the new
name.
